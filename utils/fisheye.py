"""
Fisheye Camera Undistortion Utility

Undistorts fisheye images to perspective projection before passing
them to the Gaussian Splatting rasterizer (which only supports
perspective cameras natively).

Uses the equidistant fisheye model (OPENCV_FISHEYE) for undistortion.

Usage:
    from utils.fisheye import undistort_fisheye, compute_undistort_maps

    # Precompute maps (once per camera)
    maps = compute_undistort_maps(K_fisheye, dist_coeffs, (w, h), K_new)

    # Undistort each frame
    undistorted = undistort_fisheye(image, maps)
"""

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def compute_undistort_maps(
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple,
    K_new: np.ndarray = None,
    balance: float = 0.0,
) -> tuple:
    """Precompute undistortion maps for a fisheye camera.

    Args:
        K: (3, 3) camera intrinsic matrix.
        dist_coeffs: (4,) fisheye distortion coefficients [k1, k2, k3, k4].
        image_size: (width, height) of the input images.
        K_new: Optional new intrinsic matrix for the undistorted output.
                If None, uses ``cv2.fisheye.estimateNewCameraMatrixForUndistortRectify``.
        balance: Balance between keeping all pixels (1.0) and
                 minimizing black borders (0.0). Default: 0.0.

    Returns:
        Tuple of (map1, map2) for use with ``cv2.remap``.
    """
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is required for fisheye undistortion")

    w, h = image_size
    dist = np.array(dist_coeffs, dtype=np.float64).reshape(4, 1)

    if K_new is None:
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K.astype(np.float64), dist, (w, h), np.eye(3), balance=balance
        )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K.astype(np.float64), dist, np.eye(3), K_new.astype(np.float64),
        (w, h), cv2.CV_32FC1
    )

    return map1, map2, K_new


def undistort_fisheye(image: np.ndarray, maps: tuple) -> np.ndarray:
    """Apply precomputed undistortion maps to a fisheye image.

    Args:
        image: (H, W, C) input fisheye image.
        maps: Tuple of (map1, map2) from ``compute_undistort_maps()``.

    Returns:
        (H, W, C) undistorted perspective image.
    """
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python is required for fisheye undistortion")

    map1, map2 = maps[0], maps[1]
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def make_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix.

    Args:
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point coordinates.

    Returns:
        (3, 3) numpy array.
    """
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
