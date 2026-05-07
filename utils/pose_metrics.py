#
# Pose Error Metrics
#
# Computes angular and translational error between predicted and ground-truth
# camera poses. Used to evaluate pose refinement quality.
#

import torch
import math
import numpy as np


def angular_error_rotation(R_pred, R_gt):
    """Compute rotation error in degrees between two rotation matrices.
    
    Args:
        R_pred: (3, 3) predicted rotation matrix (numpy or torch)
        R_gt: (3, 3) ground truth rotation matrix (numpy or torch)
    
    Returns:
        Angular error in degrees (float)
    """
    if isinstance(R_pred, torch.Tensor):
        R_pred = R_pred.detach().cpu().numpy()
    if isinstance(R_gt, torch.Tensor):
        R_gt = R_gt.detach().cpu().numpy()
    
    R_rel = R_pred @ R_gt.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle_rad))


def translation_error(t_pred, t_gt):
    """Compute L2 translation error.
    
    Args:
        t_pred: (3,) predicted translation (numpy or torch)
        t_gt: (3,) ground truth translation (numpy or torch)
    
    Returns:
        L2 distance (float)
    """
    if isinstance(t_pred, torch.Tensor):
        t_pred = t_pred.detach().cpu().numpy()
    if isinstance(t_gt, torch.Tensor):
        t_gt = t_gt.detach().cpu().numpy()
    
    return float(np.linalg.norm(t_pred - t_gt))


def compute_pose_errors(cameras_pred, cameras_gt):
    """Compute mean angular and translational errors across all cameras.
    
    Args:
        cameras_pred: list of Camera objects (with corrected poses)
        cameras_gt: list of Camera objects (ground truth COLMAP poses)
    
    Returns:
        dict with 'mean_angular_error_deg', 'mean_translation_error',
              'per_camera_angular', 'per_camera_translation'
    """
    angular_errors = []
    translation_errors = []
    
    for cam_p, cam_g in zip(cameras_pred, cameras_gt):
        # Extract R, T
        R_p = cam_p.R if isinstance(cam_p.R, np.ndarray) else cam_p.R.detach().cpu().numpy()
        R_g = cam_g.R if isinstance(cam_g.R, np.ndarray) else cam_g.R.detach().cpu().numpy()
        T_p = cam_p.T if isinstance(cam_p.T, np.ndarray) else cam_p.T.detach().cpu().numpy()
        T_g = cam_g.T if isinstance(cam_g.T, np.ndarray) else cam_g.T.detach().cpu().numpy()
        
        angular_errors.append(angular_error_rotation(R_p, R_g))
        translation_errors.append(translation_error(T_p, T_g))
    
    return {
        'mean_angular_error_deg': float(np.mean(angular_errors)),
        'mean_translation_error': float(np.mean(translation_errors)),
        'per_camera_angular': angular_errors,
        'per_camera_translation': translation_errors,
    }


def extract_pose_from_viewmatrix(world_view_transform):
    """Extract R, T from a 4x4 world_view_transform tensor.
    
    The world_view_transform is stored transposed (column-major for OpenGL).
    
    Args:
        world_view_transform: (4, 4) tensor
        
    Returns:
        R: (3, 3) numpy rotation matrix (stored transposed, COLMAP convention)
        T: (3,) numpy translation vector
    """
    if isinstance(world_view_transform, torch.Tensor):
        wvt = world_view_transform.detach().cpu().numpy()
    else:
        wvt = world_view_transform
    
    # The transform is stored transposed
    w2c = wvt.T
    R_w2c = w2c[:3, :3]
    T = w2c[:3, 3]
    
    # COLMAP stores R transposed
    R_stored = R_w2c.T
    
    return R_stored, T
