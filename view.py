import torch
import math
import numpy as np
from utils.system_utils import searchForMaxIteration
import os
import time
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix
import viser
import nerfview
from utils.orientation_utils import compute_up_vector_from_cameras



try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False
from typing import Tuple

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def start_viewer(dataset: ModelParams, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        # Find the latest iteration
        point_cloud_dir = os.path.join(dataset.model_path, "point_cloud")
        if not os.path.exists(point_cloud_dir):
            print(f"No point_cloud directory found in {dataset.model_path}")
            return
            
        loaded_iter = searchForMaxIteration(point_cloud_dir)
        ply_path = os.path.join(point_cloud_dir, f"iteration_{loaded_iter}", "point_cloud.ply")
        if not os.path.exists(ply_path):
            print(f"Point cloud file not found: {ply_path}")
            return
            
        print(f"Loading trained model at iteration {loaded_iter} from {ply_path}")
        gaussians.load_ply(ply_path, False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        server = viser.ViserServer(port=8081, verbose=True)
        server.gui.configure_theme(control_layout="collapsible")
        try:
            up_vec = compute_up_vector_from_cameras(dataset.model_path)
            server.scene.set_up_direction(up_vec)
            print(f"Set Viser up direction from cameras to {up_vec}")
        except Exception as e:
            print(f"Failed to set up direction from density: {e}")
            server.scene.set_up_direction("-y")
        
        @torch.no_grad()
        def render_fn(camera_state: nerfview.CameraState, img_wh):
            if hasattr(img_wh, 'viewer_width'):
                w, h = img_wh.viewer_width, img_wh.viewer_height
            else:
                w, h = img_wh
            fovy = camera_state.fov
            fovx = 2 * math.atan(math.tan(fovy / 2) * (w / h))
            w2c = np.linalg.inv(camera_state.c2w)
            wvt = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()
            proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            full = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            cam = MiniCam(w, h, fovy, fovx, 0.01, 100.0, wvt, full)
            
            try:
                # Use a dark blue-grey bg so it's not pure black like a crash
                viewer_bg = torch.tensor([0.1, 0.15, 0.2], dtype=torch.float32, device="cuda")
                img = render(cam, gaussians, pipeline, viewer_bg, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                return img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_img = np.zeros((h, w, 3), dtype=np.float32)
                err_img[:, :, 0] = 1.0
                return err_img

        viewer = nerfview.Viewer(
            server=server,
            render_fn=render_fn,
            mode="rendering",
        )
        print("Viewer running on port 8081. Press Ctrl+C to quit.")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    parser = ArgumentParser(description="Viewer script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_combined_args(parser)
    safe_state(False)
    start_viewer(model.extract(args), pipeline.extract(args))
