import os
import json
import numpy as np

def compute_up_vector_from_cameras(model_path: str) -> np.ndarray:
    """
    Computes the UP vector by finding the mean of the -Y axis (up in image space)
    for all cameras in cameras.json. This robustly works for front-facing or 360 scenes.
    """
    cameras_json = os.path.join(model_path, "cameras.json")
    if not os.path.exists(cameras_json):
        # Fallback to -Y if no cameras.json
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
        
    try:
        with open(cameras_json, 'r') as f:
            cameras = json.load(f)
            
        up_vectors = []
        for cam in cameras:
            # rot is C2W rotation matrix (columns are local X, Y, Z axes in world space)
            rot = np.array(cam["rotation"])
            # The local Y axis of the camera points DOWN in the image.
            # So the local -Y axis points UP in the image.
            # We take the 2nd column of rot (index 1), which is the Y axis, and negate it.
            # Note: numpy arrays from json are row-major, so rot[:, 1] is the 2nd column.
            up_vector = -rot[:, 1]
            up_vectors.append(up_vector)
            
        mean_up = np.mean(up_vectors, axis=0)
        norm = np.linalg.norm(mean_up)
        if norm > 1e-6:
            return (mean_up / norm).astype(np.float32)
            
    except Exception as e:
        print(f"Failed to compute up vector from cameras: {e}")
    
    return np.array([0.0, -1.0, 0.0], dtype=np.float32)

def compute_up_vector_from_density(xyz):
    # Dummy to not break anything if it's still imported elsewhere
    return np.array([0.0, -1.0, 0.0], dtype=np.float32)

