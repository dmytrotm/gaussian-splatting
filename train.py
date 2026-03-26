#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, entropy_loss
from utils.cauchy import CauchyActivation, BoundedCauchyActivation, cauchy_loss, scheduled_cauchy_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import MiniCam
from utils.general_utils import safe_state, get_expon_lr_func
from utils.graphics_utils import getProjectionMatrix
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from densification import get_strategy
from utils.metrics_tracker import MetricsTracker
from utils.regularization import opacity_reg_loss, scale_reg_loss
from utils.early_stopping import EarlyStopping
from scene.camera_opt import CameraOptModule

try:
    from lpipsPyTorch import lpips as compute_lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    import viser
    import nerfview
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, densification_strategy_name="default", viewer_port=0):  # MODIFIED

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    strategy = get_strategy(densification_strategy_name)  # MODIFIED: instantiate strategy
    tracker = MetricsTracker(output_dir=dataset.model_path)  # MODIFIED: metrics tracking

    # MODIFIED: early stopping
    early_stopper = EarlyStopping(
        patience=opt.early_stopping_patience,
        min_delta=opt.early_stopping_min_delta,
        output_dir=dataset.model_path,
    )

    # MODIFIED: web viewer — direct viser integration (like gsplat reference)
    _viewer = None
    if viewer_port > 0:
        if not VISER_AVAILABLE:
            print("[WARNING] viser/nerfview not installed — web viewer disabled.")
            print("  Install with: pip install viser nerfview")
        else:
            print(f"Starting web viewer at http://localhost:{viewer_port}")
            _viser_server = viser.ViserServer(port=viewer_port, verbose=False)

            @torch.no_grad()
            def _viewer_render_fn(camera_state: nerfview.CameraState, img_wh):
                w, h = img_wh
                fovy = camera_state.fov
                fovx = 2 * math.atan(math.tan(fovy / 2) * (w / h))
                w2c = np.linalg.inv(camera_state.c2w)
                wvt = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()
                proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
                full = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
                cam = MiniCam(w, h, fovy, fovx, 0.01, 100.0, wvt, full)
                try:
                    img = render(cam, gaussians, pipe, background,
                                 separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                except Exception as e:
                    print(f"Viewer render error: {e}")
                    return np.zeros((h, w, 3), dtype=np.float32)

            _viewer = nerfview.Viewer(
                server=_viser_server,
                render_fn=_viewer_render_fn,
                mode="rendering",
            )
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Pose noise: create a frozen perturbation module (applied once, not learned)
    pose_perturb = None
    if opt.pose_noise > 0.0:
        n_cams = len(viewpoint_stack)
        pose_perturb = CameraOptModule(n_cams).cuda()
        pose_perturb.random_init(std=opt.pose_noise)
        pose_perturb.requires_grad_(False)  # frozen — noise only, not optimized
        print(f"[INFO] Pose noise injected: std={opt.pose_noise}, {n_cams} cameras")

    # Cauchy activation (learnable color mapping)
    color_act = None
    color_act_optimizer = None
    if opt.cauchy_activation:
        if opt.cauchy_act_bounded:
            color_act = BoundedCauchyActivation(channels=3).cuda()
            print("[INFO] Bounded Cauchy activation — μ∈[0.3,0.7], γ∈[0.05,0.3]")
        else:
            color_act = CauchyActivation(channels=3).cuda()
            print("[INFO] Cauchy activation enabled — learnable color mapping")
        # Frozen initially if freeze_until > 0 (Strategy 1)
        act_lr = 1e-3
        if opt.cauchy_act_freeze_until > 0:
            for p in color_act.parameters():
                p.requires_grad = False
            print(f"[INFO] Activation params frozen until iter {opt.cauchy_act_freeze_until}")
        color_act_optimizer = torch.optim.Adam(color_act.parameters(), lr=act_lr)

    if opt.cauchy_loss:
        if opt.cauchy_scale_schedule:
            print(f"[INFO] Cauchy loss with scheduled scale (1.0 → 0.1 over {opt.iterations} iters)")
        else:
            print("[INFO] Cauchy loss enabled — robust Lorentzian loss (scale=0.1)")
    if opt.cauchy_grad_aware_densify:
        print("[INFO] Gradient-aware densification — auto-scaling threshold for Cauchy loss")
    if opt.max_gaussians > 0:
        print(f"[INFO] Gaussian count budget — max {opt.max_gaussians:,} points")

    # Gradient accumulation
    grad_accum_steps = max(1, opt.grad_accum_steps)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # Legacy TCP viewer loop (only when web viewer is NOT active)
        if _viewer is None:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, color_activation=color_act)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # Random patch cropping (when --patch_size > 0)
        if opt.patch_size > 0:
            h, w = image.shape[1], image.shape[2]
            ps = opt.patch_size
            if h > ps and w > ps:
                y0 = torch.randint(0, h - ps, (1,)).item()
                x0 = torch.randint(0, w - ps, (1,)).item()
                image = image[:, y0:y0+ps, x0:x0+ps]
                gt_image = gt_image[:, y0:y0+ps, x0:x0+ps]

        if opt.cauchy_loss:
            if opt.cauchy_scale_schedule:
                Ll1 = scheduled_cauchy_loss(image, gt_image, iteration, opt.iterations)
            else:
                Ll1 = cauchy_loss(image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Entropy loss (3D fixed: visibility mask + weight cap)
        if opt.entropy_reg:
            ent_loss = entropy_loss(
                gaussians._opacity, iteration, loss.item(),
                visibility_filter=visibility_filter,
                densify_until_iter=opt.densify_until_iter,
            )
            if isinstance(ent_loss, torch.Tensor):
                loss = loss + ent_loss

        # Opacity regularization
        if opt.opacity_reg > 0.0:
            loss = loss + opt.opacity_reg * opacity_reg_loss(gaussians)

        # Scale regularization
        if opt.scale_reg > 0.0:
            loss = loss + opt.scale_reg * scale_reg_loss(gaussians)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Scale loss for gradient accumulation
        (loss / grad_accum_steps).backward()

        iter_end.record()

        with torch.no_grad():
            # Per-iteration timing and VRAM tracking
            torch.cuda.synchronize()
            elapsed_ms = iter_start.elapsed_time(iter_end)
            tracker.log_iter_time(iteration, elapsed_ms)
            if iteration % 100 == 0:
                tracker.log_gpu_memory(iteration)
            if iteration % 1000 == 0:
                tracker.log_num_gaussians(iteration, gaussians.get_xyz.shape[0])

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                vram_gb = tracker.get_latest_vram_gb()
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Depth": f"{ema_Ll1depth_for_log:.{7}f}",
                    "VRAM": f"{vram_gb:.2f}GB",
                    "ms/it": f"{elapsed_ms:.1f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            test_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed_ms, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, tracker)

            # Early stopping check
            if test_psnr is not None and early_stopper.check(iteration, test_psnr):
                print(f"\n[ITER {iteration}] Early stopping triggered — saving final checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_early_stop_" + str(iteration) + ".pth")
                scene.save(iteration)
                progress_bar.close()
                break
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification  # MODIFIED: delegate to strategy
            # Strategy 3: gradient-aware densification threshold
            if opt.cauchy_grad_aware_densify and opt.cauchy_loss:
                _saved_thresh = opt.densify_grad_threshold
                opt.densify_grad_threshold = opt.densify_grad_threshold * 2.0
            strategy.step(gaussians, scene, iteration, visibility_filter, viewspace_point_tensor, radii, opt, dataset)
            if opt.cauchy_grad_aware_densify and opt.cauchy_loss:
                opt.densify_grad_threshold = _saved_thresh

            # Strategy 4: Gaussian count budget
            if opt.max_gaussians > 0 and gaussians.get_xyz.shape[0] > opt.max_gaussians:
                opacities = gaussians.get_opacity.squeeze()
                keep_mask = opacities > opacities.quantile(0.1)  # Prune bottom 10%
                gaussians.prune_points(~keep_mask)
                torch.cuda.empty_cache()

            # Optimizer step (with gradient accumulation support)
            if iteration < opt.iterations and iteration % grad_accum_steps == 0:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if color_act_optimizer is not None:
                    # Strategy 1: unfreeze after freeze_until
                    if (opt.cauchy_act_freeze_until > 0
                            and iteration == opt.cauchy_act_freeze_until
                            and color_act is not None):
                        for p in color_act.parameters():
                            p.requires_grad = True
                        # Switch to low LR after unfreezing
                        for pg in color_act_optimizer.param_groups:
                            pg['lr'] = 1e-5
                        print(f"\n[ITER {iteration}] Unfreezing Cauchy activation params (lr=1e-5)")
                    color_act_optimizer.step()
                    color_act_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                strategy.post_step(gaussians, iteration, opt)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # MODIFIED: save metrics and generate plots at end of training
    tracker.save_json()
    tracker.plot_all()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, tracker=None):
    """Run evaluation and log metrics. Returns test PSNR if evaluated, else None."""
    test_psnr_value = None

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if LPIPS_AVAILABLE:
                        lpips_test += compute_lpips(image.unsqueeze(0), gt_image.unsqueeze(0), net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                if LPIPS_AVAILABLE:
                    lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}{}".format(
                    iteration, config['name'], l1_test, psnr_test,
                    f" LPIPS {lpips_test:.6f}" if LPIPS_AVAILABLE else ""))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if LPIPS_AVAILABLE:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                if tracker is not None and config['name'] == 'test':
                    metrics = dict(psnr=psnr_test.item(), l1=l1_test.item())
                    if LPIPS_AVAILABLE:
                        metrics['lpips'] = lpips_test.item()
                    tracker.update(iteration=iteration, **metrics)
                    test_psnr_value = psnr_test.item()

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr_value

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--viewer_port', type=int, default=0,
                        help="Port for the viser web viewer (0 = disabled, e.g. 8080)")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--densification_strategy', type=str, default='default', choices=['default', 'mcmc'],
                        help='Densification strategy: default (Inria) or mcmc')
    parser.add_argument("--scenes", nargs="+", type=str, default=[],
                        help="Multiple source paths for batch scene processing")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start legacy TCP viewer (only when web viewer is not requested)
    if not args.disable_viewer and args.viewer_port == 0:
        network_gui.init(args.ip, args.port)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Batch scene processing: if --scenes is provided, iterate over each
    if args.scenes:
        for i, scene_path in enumerate(args.scenes):
            print(f"\n{'=' * 60}")
            print(f"  Batch scene {i+1}/{len(args.scenes)}: {scene_path}")
            print(f"{'=' * 60}")
            args.source_path = os.path.abspath(scene_path)
            # Auto-generate model path per scene
            scene_name = os.path.basename(scene_path.rstrip('/'))
            args.model_path = os.path.join("./output/", scene_name)
            dataset_args = lp.extract(args)
            training(
                dataset_args, op.extract(args), pp.extract(args),
                args.test_iterations, args.save_iterations,
                args.checkpoint_iterations, args.start_checkpoint,
                args.debug_from, args.densification_strategy,
                viewer_port=args.viewer_port,
            )
            print(f"\n  Scene {scene_name} complete.\n")
    else:
        # Single scene (default behavior)
        print("Optimizing " + args.model_path)
        training(
            lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations,
            args.checkpoint_iterations, args.start_checkpoint,
            args.debug_from, args.densification_strategy,
            viewer_port=args.viewer_port,
        )

    # All done
    print("\nTraining complete.")
