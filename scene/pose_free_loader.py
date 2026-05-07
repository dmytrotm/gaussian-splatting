#
# Pose-Free Scene Loader
#
# Reads images from a folder WITHOUT any COLMAP/SfM data.
# Initializes cameras on a Fibonacci hemisphere looking at the origin.
# Used when --pose_free is enabled.
#

import os
import sys
import math
import numpy as np
from PIL import Image
from pathlib import Path

from scene.dataset_readers import CameraInfo, SceneInfo, getNerfppNorm, storePly, fetchPly
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB
from utils.graphics_utils import focal2fov


def fibonacci_hemisphere(n, radius=4.0):
    """Generate n points evenly distributed on the upper hemisphere.
    
    Uses the Fibonacci spiral method for near-uniform distribution.
    Returns (n, 3) array of 3D positions.
    """
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2
    
    for i in range(n):
        # theta: azimuthal angle (0 to 2*pi)
        theta = 2 * math.pi * i / golden_ratio
        # phi: polar angle — only upper hemisphere (0 to pi/2)
        # Use a linear spacing in cos(phi) for uniform area coverage
        phi = math.acos(1 - (i + 0.5) / n)
        # Clamp to upper hemisphere
        phi = min(phi, math.pi * 0.85)  # allow slightly below equator
        
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.cos(phi)  # y-up
        z = radius * math.sin(phi) * math.sin(theta)
        points.append([x, y, z])
    
    return np.array(points, dtype=np.float64)


def look_at(eye, target, up=np.array([0.0, 1.0, 0.0])):
    """Compute rotation matrix R and translation T for a camera at `eye`
    looking at `target`.
    
    Returns R, T in COLMAP convention:
        R: (3, 3) rotation matrix (transposed — world-to-camera rotation)
        T: (3,)   translation vector (world-to-camera)
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    
    # Forward direction (camera looks along -z in OpenGL, but COLMAP uses +z)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Right direction
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        # up and forward are parallel — use a different up vector
        up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # True up direction
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    
    # World-to-camera rotation matrix (COLMAP convention)
    # COLMAP: x-right, y-down, z-forward
    R_w2c = np.zeros((3, 3), dtype=np.float64)
    R_w2c[0, :] = right
    R_w2c[1, :] = -true_up  # y-down in COLMAP
    R_w2c[2, :] = forward
    
    # Translation: T = -R @ eye
    T = -R_w2c @ eye
    
    # COLMAP stores R transposed (for historical reasons)
    R_stored = R_w2c.T
    
    return R_stored, T


def readPoseFreeSceneInfo(path, images, eval, default_fov=60.0, 
                           num_random_points=100_000, llffhold=8):
    """Load a scene from images only — no COLMAP required.
    
    Args:
        path: Path to scene directory (must contain images/ subfolder)
        images: Name of images subfolder (default: "images")
        eval: If True, hold out every llffhold-th image for testing
        default_fov: Default field of view in degrees
        num_random_points: Number of random points to initialize
        llffhold: Hold-out frequency for test split
    
    Returns:
        SceneInfo with hemisphere-initialized cameras and random point cloud
    """
    reading_dir = "images" if images is None else images
    images_folder = os.path.join(path, reading_dir)
    
    if not os.path.exists(images_folder):
        print(f"ERROR: Images folder not found at {images_folder}")
        sys.exit(1)
    
    # Collect all image files
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = sorted([
        f for f in os.listdir(images_folder)
        if Path(f).suffix.lower() in supported_extensions
    ])
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {images_folder}")
        sys.exit(1)
    
    print(f"[Pose-Free] Found {len(image_files)} images in {images_folder}")
    
    # Get image dimensions from the first image
    sample_img = Image.open(os.path.join(images_folder, image_files[0]))
    width, height = sample_img.size
    
    # Compute FoV from default
    fov_rad = default_fov * math.pi / 180.0
    FovX = fov_rad
    FovY = focal2fov(fov2focal_simple(FovX, width), height)
    
    print(f"[Pose-Free] Image size: {width}x{height}, FoV: {default_fov}°")
    
    # Initialize cameras on hemisphere
    n_cameras = len(image_files)
    positions = fibonacci_hemisphere(n_cameras, radius=4.0)
    target = np.array([0.0, 0.0, 0.0])
    
    # Determine test split
    if eval:
        test_indices = set(range(0, n_cameras, llffhold))
        print(f"[Pose-Free] Eval mode: {len(test_indices)} test cameras (every {llffhold}-th)")
    else:
        test_indices = set()
    
    # Build camera infos
    cam_infos = []
    for idx, img_name in enumerate(image_files):
        sys.stdout.write(f'\rInitializing camera {idx+1}/{n_cameras}')
        sys.stdout.flush()
        
        R, T = look_at(eye=positions[idx], target=target)
        
        image_path = os.path.join(images_folder, img_name)
        
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params=None,
            image_path=image_path,
            image_name=img_name,
            depth_path="",
            width=width,
            height=height,
            is_test=(idx in test_indices),
        )
        cam_infos.append(cam_info)
    
    sys.stdout.write('\n')
    
    # Split train/test
    train_cam_infos = [c for c in cam_infos if not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    
    print(f"[Pose-Free] Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")
    
    # Compute normalization from hemisphere cameras
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # Generate random point cloud (always regenerate to match requested count)
    ply_path = os.path.join(path, "points3d_posefree.ply")
    print(f"[Pose-Free] Generating random point cloud ({num_random_points} points)...")
    # Points within the scene bounds (smaller than camera radius)
    xyz = np.random.uniform(-1.5, 1.5, (num_random_points, 3))
    shs = np.random.random((num_random_points, 3)) / 255.0
    storePly(ply_path, xyz, (SH2RGB(shs) * 255).astype(np.uint8))

    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
    )
    
    return scene_info


def fov2focal_simple(fov, pixels):
    """Convert field of view (radians) to focal length (pixels)."""
    return pixels / (2.0 * math.tan(fov / 2.0))
