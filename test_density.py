import sys
sys.path.append("/workspace/gaussian-splatting")
from utils.orientation_utils import compute_up_vector_from_density
from scene.gaussian_model import GaussianModel
import torch

gaussians = GaussianModel(3)
gaussians.load_ply("output/MipNeRF_360_2/point_cloud/iteration_7000/point_cloud.ply", False)
up = compute_up_vector_from_density(gaussians.get_xyz)
print(f"Computed UP vector: {up}")
