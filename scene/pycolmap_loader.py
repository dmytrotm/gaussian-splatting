"""
pycolmap-based Data Loader — alternative COLMAP parser

Uses ``pycolmap`` Python bindings to read COLMAP reconstruction files
instead of the existing custom binary/text parsers.

Advantages:
- Cleaner API, fewer edge cases
- Native support for all COLMAP camera models (including fisheye)
- Returns normalized camera intrinsics

Usage:
    from scene.pycolmap_loader import load_colmap_scene

    cameras, images, points3D = load_colmap_scene("data/sparse/0")

Requirements:
    pip install pycolmap

Note: This is an OPTIONAL alternative. The existing ``dataset_readers.py``
parsers remain the default. Use ``--data_loader pycolmap`` to enable.
"""

import os
import numpy as np

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False


def load_colmap_scene(sparse_dir: str):
    """Load a COLMAP reconstruction using pycolmap.

    Args:
        sparse_dir: Path to sparse reconstruction directory (e.g., "sparse/0").

    Returns:
        Tuple of (cameras_dict, images_dict, points3D_array):
            - cameras_dict: {camera_id: dict with model, width, height, params}
            - images_dict: {image_id: dict with name, qvec, tvec, camera_id, c2w}
            - points3D_array: (N, 3) numpy array of 3D point positions
    """
    if not PYCOLMAP_AVAILABLE:
        raise ImportError(
            "pycolmap is required for this loader. Install with: pip install pycolmap"
        )

    reconstruction = pycolmap.Reconstruction(sparse_dir)

    # Parse cameras
    cameras_dict = {}
    for camera_id, camera in reconstruction.cameras.items():
        cam_info = {
            "model": camera.model_name,
            "width": camera.width,
            "height": camera.height,
            "params": np.array(camera.params),
            "focal_x": camera.focal_length_x,
            "focal_y": camera.focal_length_y,
            "cx": camera.principal_point_x,
            "cy": camera.principal_point_y,
        }

        # Classify camera type for the rasterizer
        model = camera.model_name
        if model in ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"):
            cam_info["type"] = "perspective"
        elif model in ("OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            cam_info["type"] = "fisheye"
        else:
            cam_info["type"] = "perspective"  # fallback

        cameras_dict[camera_id] = cam_info

    # Parse images
    images_dict = {}
    for image_id, image in reconstruction.images.items():
        # Build world-to-camera (w2c) and camera-to-world (c2w) matrices
        R = image.rotmat()
        t = image.tvec
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)

        images_dict[image_id] = {
            "name": image.name,
            "qvec": image.qvec,
            "tvec": image.tvec,
            "camera_id": image.camera_id,
            "w2c": w2c.astype(np.float32),
            "c2w": c2w.astype(np.float32),
        }

    # Parse 3D points
    points_list = []
    colors_list = []
    for point_id, point in reconstruction.points3D.items():
        points_list.append(point.xyz)
        colors_list.append(point.color)

    points3D = np.array(points_list, dtype=np.float32) if points_list else np.zeros((0, 3), dtype=np.float32)
    colors3D = np.array(colors_list, dtype=np.uint8) if colors_list else np.zeros((0, 3), dtype=np.uint8)

    return cameras_dict, images_dict, points3D, colors3D


def get_fisheye_distortion_params(camera_info: dict) -> dict:
    """Extract fisheye distortion parameters from a pycolmap camera.

    Args:
        camera_info: Camera dict from ``load_colmap_scene``.

    Returns:
        Dict with keys: k1, k2, k3, k4 (fisheye distortion coefficients).
        Returns empty dict for non-fisheye cameras.
    """
    if camera_info["type"] != "fisheye":
        return {}

    params = camera_info["params"]
    model = camera_info["model"]

    if model == "OPENCV_FISHEYE":
        # params: fx, fy, cx, cy, k1, k2, k3, k4
        return {"k1": params[4], "k2": params[5], "k3": params[6], "k4": params[7]}
    elif model == "SIMPLE_RADIAL_FISHEYE":
        # params: f, cx, cy, k
        return {"k1": params[3], "k2": 0.0, "k3": 0.0, "k4": 0.0}
    elif model == "RADIAL_FISHEYE":
        # params: f, cx, cy, k1, k2
        return {"k1": params[3], "k2": params[4], "k3": 0.0, "k4": 0.0}
    return {}
