"""
Standalone 3D Gaussian Splatting Web Viewer

An interactive web-based viewer for trained 3DGS models.
Uses viser + nerfview to serve a browser-accessible viewer.

Usage:
    # View a trained model
    python -m gaussian_renderer.viewer -m output/your_experiment

    # Specify iteration
    python -m gaussian_renderer.viewer -m output/your_experiment --iteration 7000

    # Custom port
    python -m gaussian_renderer.viewer -m output/your_experiment --port 8080

Controls:
    Left-click + drag:  Rotate camera
    Right-click + drag: Pan camera
    Scroll:             Zoom in/out
"""

import math
import os
import sys
import time
import torch
import numpy as np

from argparse import ArgumentParser

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.graphics_utils import getProjectionMatrix
from scene.cameras import MiniCam

try:
    import viser
    import nerfview
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def run_viewer(dataset, pipeline, iteration, port, quiet):
    """Load a trained model and serve an interactive web viewer."""

    if not VISER_AVAILABLE:
        print("ERROR: viser and nerfview are required for the web viewer.")
        print("Install them with: pip install viser nerfview")
        sys.exit(1)

    # --- Load the trained model ---
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        loaded_iter = scene.loaded_iter

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    num_gaussians = gaussians.get_xyz.shape[0]
    sh_degree = gaussians.active_sh_degree

    print(f"\n{'=' * 60}")
    print(f"  Loaded model: {dataset.model_path}")
    print(f"  Iteration:    {loaded_iter}")
    print(f"  Gaussians:    {num_gaussians:,}")
    print(f"  SH degree:    {sh_degree}")
    print(f"{'=' * 60}")

    # --- Viewer render callback ---
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh):
        width, height = img_wh
        c2w = camera_state.c2w  # (4, 4) numpy, OpenCV convention
        fovy = camera_state.fov

        fovx = 2 * math.atan(math.tan(fovy / 2) * (width / height))
        w2c = np.linalg.inv(c2w)
        world_view_transform = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()

        projection_matrix = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy
        ).transpose(0, 1).cuda()

        full_proj_transform = (
            world_view_transform.unsqueeze(0)
            .bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)

        cam = MiniCam(width, height, fovy, fovx, 0.01, 100.0,
                      world_view_transform, full_proj_transform)

        try:
            render_pkg = render(cam, gaussians, pipeline, background,
                                separate_sh=SPARSE_ADAM_AVAILABLE)
            image = render_pkg["render"]  # (3, H, W) in [0, 1]
            image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        except Exception as e:
            print(f"Render error: {e}")
            image = np.zeros((height, width, 3), dtype=np.float32)

        return image

    # --- Start the viewer server ---
    server = viser.ViserServer(port=port, verbose=False)
    viewer = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )

    print(f"\n  Viewer running at: http://localhost:{port}")
    print(f"\n  Controls:")
    print(f"    Left-click + drag:  Rotate camera")
    print(f"    Right-click + drag: Pan camera")
    print(f"    Scroll:             Zoom in/out")
    print(f"\n  Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


if __name__ == "__main__":
    parser = ArgumentParser(description="3DGS Web Viewer")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Iteration to load (-1 = latest)")
    parser.add_argument("--port", default=8080, type=int,
                        help="Port for the web viewer (default: 8080)")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print(f"Starting viewer for: {args.model_path}")

    run_viewer(
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        iteration=args.iteration,
        port=args.port,
        quiet=args.quiet,
    )
