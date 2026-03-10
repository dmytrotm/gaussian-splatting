import torch
import traceback
import math
import numpy as np
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix
import threading

try:
    import viser
    import nerfview
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False

host = "0.0.0.0"
port = 6009

conn = None # Set to True when Viser is ready
_server = None
_viewer = None

# We use threading events to coordinate the async nerfview with the sync train.py
_request_event = threading.Event()
_response_event = threading.Event()

_current_cam_state = None
_img_wh = None
_image_result = None

def init(wish_host, wish_port):
    global host, port, conn, _server, _viewer
    host = "0.0.0.0"
    port = wish_port
    
    if not VISER_AVAILABLE:
        print("Viser not available, network viewing disabled.")
        return

    print(f"Starting Viser web viewer on http://localhost:{port}")
    _server = viser.ViserServer(host=host, port=port, verbose=False)
    
    def render_fn(camera_state: nerfview.CameraState, img_wh):
        global _current_cam_state, _img_wh, _image_result
        
        # Pass request to train.py
        _current_cam_state = camera_state
        _img_wh = img_wh
        _request_event.set()
        
        # Wait for train.py to call send()
        _response_event.wait()
        _response_event.clear()
        
        if _image_result is not None:
            return _image_result
        else:
            return np.zeros((img_wh[1], img_wh[0], 3), dtype=np.uint8)

    _viewer = nerfview.Viewer(
        server=_server,
        render_fn=render_fn,
        mode="rendering",
    )
    conn = True # Signals train.py to enter the network loop

def try_connect():
    pass # Managed by Viser now

def receive():
    global _request_event
    
    # Non-blocking check for a render request
    has_request = _request_event.wait(timeout=0.001)
    if not has_request:
        # No request from viewer, continue training
        # do_training=True means train.py breaks out of the GUI loop and trains
        return None, True, False, False, True, 1.0

    _request_event.clear()
    
    try:
        width, height = _img_wh
        fovy = _current_cam_state.fov
        fovx = 2 * math.atan(math.tan(fovy / 2) * (width / height))
        znear = 0.01
        zfar = 100.0
        
        # Viser provides c2w in OpenCV convention (+x right, +y down, +z forward)
        c2w = _current_cam_state.c2w
        w2c = np.linalg.inv(c2w)
        
        # 3DGS expects column-major matrices, so we transpose
        world_view_transform = torch.tensor(w2c).float().transpose(0, 1).cuda()
        
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        
        # do_training=True ensures training isn't paused
        do_training = True
        do_shs_python = False
        do_rot_scale_python = False
        keep_alive = True
        scaling_modifier = 1.0
        
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
        
    except Exception as e:
        traceback.print_exc()
        # On error, release the lock and continue
        _image_result = None
        _response_event.set()
        return None, True, False, False, True, 1.0

def send(message_bytes, verify):
    global _image_result, _response_event
    if message_bytes is not None:
        # message_bytes is memoryview of shape (H, W, 3) in uint8
        _image_result = np.array(message_bytes)
    else:
        _image_result = None
        
    # Signal Viser that the image is ready
    _response_event.set()
