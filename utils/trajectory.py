"""
Camera trajectory generation for smooth video rendering.

Ported from gsplat reference (complete_trainer.py L247-311).

Usage:
    from utils.trajectory import generate_interpolated_path, generate_ellipse_path

    # Spline interpolation between keyframes
    new_c2ws = generate_interpolated_path(c2w_array, n_interp=120)

    # Elliptical orbit
    new_c2ws = generate_ellipse_path(c2w_array, n_frames=240)
"""

import numpy as np

try:
    import scipy.interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _normalize(x: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return x / (np.linalg.norm(x) + 1e-12)


def _viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct a camera-to-world matrix from look direction, up, and position."""
    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def _focus_point(poses: np.ndarray) -> np.ndarray:
    """Find the point nearest to all camera focal axes."""
    directions = poses[:, :3, 2:3]
    origins = poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def _average_pose(poses: np.ndarray) -> np.ndarray:
    """Compute the average camera pose."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    return _viewmatrix(z_axis, up, position)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
) -> np.ndarray:
    """Create a smooth spline path between input keyframe camera poses.

    Args:
        poses: (N, 3, 4) or (N, 4, 4) array of camera-to-world matrices.
        n_interp: Number of interpolated frames per segment.
        spline_degree: B-spline degree (default: 5).
        smoothness: Spline smoothing factor (default: 0.03).
        rot_weight: Weight for rotation vs position in the path (default: 0.1).

    Returns:
        (M, 3, 4) array of interpolated camera poses.

    Raises:
        ImportError: If scipy is not installed.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for trajectory generation. "
            "Install with: pip install scipy"
        )

    poses = poses[:, :3, :4]  # Ensure (N, 3, 4)

    def _poses_to_points(p, dist):
        pos = p[:, :3, -1]
        lookat = p[:, :3, -1] - dist * p[:, :3, 2]
        up = p[:, :3, -1] + dist * p[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def _points_to_poses(points):
        return np.array([_viewmatrix(p - l, u - p, p) for p, l, u in points])

    def _interp(points, n, k, s):
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = _poses_to_points(poses, dist=rot_weight)
    new_points = _interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return _points_to_poses(new_points)


def generate_ellipse_path(
    poses: np.ndarray,
    n_frames: int = 240,
    const_speed: bool = True,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical orbit camera path around the scene.

    Args:
        poses: (N, 3, 4) or (N, 4, 4) array of camera-to-world matrices.
        n_frames: Number of frames in the orbit (default: 240).
        const_speed: If True, parameterize by arc length for constant speed.
        z_variation: Amplitude of vertical oscillation (default: 0.0).
        z_phase: Phase offset for vertical oscillation (default: 0.0).

    Returns:
        (n_frames, 3, 4) array of orbit camera poses.
    """
    poses = poses[:, :3, :4]
    center = _focus_point(poses)
    avg_pose = _average_pose(poses)

    # Compute radii from camera positions
    cam_positions = poses[:, :3, 3]
    offsets = cam_positions - center
    up = avg_pose[:3, 1]
    right = avg_pose[:3, 0]

    # Project onto right and up axes for ellipse radii
    r_right = np.abs(offsets @ right).mean()
    r_up_horiz = np.abs(offsets @ np.cross(up, avg_pose[:3, 2])).mean()
    radius_x = max(r_right, r_up_horiz)
    radius_y = min(r_right, r_up_horiz) * 0.8  # Slightly flatten

    t = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

    if const_speed:
        # Approximate arc-length parameterization
        t_dense = np.linspace(0, 2 * np.pi, n_frames * 100)
        dx = -radius_x * np.sin(t_dense)
        dy = radius_y * np.cos(t_dense)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.cumsum(ds)
        s = s / s[-1] * 2 * np.pi
        t = np.interp(np.linspace(0, 2 * np.pi, n_frames, endpoint=False), s, t_dense)

    render_poses = []
    for angle in t:
        # Position on ellipse
        x = radius_x * np.cos(angle)
        y = radius_y * np.sin(angle)
        z = z_variation * np.sin(angle + z_phase)

        position = center + x * right + y * np.cross(up, avg_pose[:3, 2]) + z * up
        lookdir = _normalize(center - position)
        pose = _viewmatrix(lookdir, up, position)
        render_poses.append(pose)

    return np.stack(render_poses, axis=0)
