"""
Gaussian Splatting Web API Server
Single-file aiohttp server with WebSocket support.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add repo root to sys.path so splatting imports work
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aiohttp import web
import aiohttp_cors

# ---------------------------------------------------------------------------
# ENV config
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", os.path.join(REPO_ROOT, "data")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", os.path.join(REPO_ROOT, "output")))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")
VIEWER_PORT = int(os.environ.get("VIEWER_PORT", "8081"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Job system
# ---------------------------------------------------------------------------
JOBS: dict[str, dict] = {}
RUNS: dict[str, dict] = {}
WS_CLIENTS: set[web.WebSocketResponse] = set()
executor = ThreadPoolExecutor(max_workers=4)

def make_job(job_type: str, meta: dict | None = None) -> dict:
    job = {
        "id": str(uuid.uuid4()),
        "type": job_type,
        "status": "queued",
        "meta": meta or {},
        "created_at": time.time(),
    }
    JOBS[job["id"]] = job
    return job


# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------
async def ws_broadcast(msg: dict):
    dead = set()
    payload = json.dumps(msg)
    for ws in WS_CLIENTS:
        try:
            await ws.send_str(payload)
        except Exception:
            dead.add(ws)
    WS_CLIENTS.difference_update(dead)


def broadcast_from_thread(loop: asyncio.AbstractEventLoop, msg: dict):
    asyncio.run_coroutine_threadsafe(ws_broadcast(msg), loop)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
CANONICAL_DATASETS = {
    "mipnerf360": {
        "name": "MipNeRF 360",
        "scenes": ["garden", "bicycle", "stump", "room", "counter", "kitchen", "bonsai"],
    },
    "mipnerf360_extra": {
        "name": "MipNeRF 360 (Extra)",
        "scenes": ["flowers", "treehill"],
    },
    "bilarf_data": {
        "name": "BiLARF",
        "scenes": ["bilarf"],
    },
    "zipnerf": {
        "name": "Zip-NeRF",
        "scenes": ["berlin", "london", "nyc", "alameda"],
    },
}

DATASET_DIR_MAP = {
    "mipnerf360": "360_v2",
    "mipnerf360_extra": "360_v2",
    "bilarf_data": "bilarf",
    "zipnerf": "zipnerf",
}


def scan_datasets():
    results = []
    for ds_id, info in CANONICAL_DATASETS.items():
        dir_name = DATASET_DIR_MAP.get(ds_id, ds_id)
        ds_path = DATA_DIR / dir_name
        available = ds_path.exists()
        scenes = []
        if available:
            for child in sorted(ds_path.iterdir()):
                if child.is_dir():
                    has_sparse = (child / "sparse").exists()
                    has_images = (child / "images").exists()
                    if has_sparse or has_images:
                        scenes.append(child.name)
        if not scenes:
            scenes = info["scenes"]
        results.append({
            "id": ds_id,
            "name": info["name"],
            "type": "canonical",
            "available": available,
            "scenes": scenes,
        })
    # uploads
    uploads_dir = DATA_DIR / "uploads"
    if uploads_dir.exists():
        for child in sorted(uploads_dir.iterdir()):
            if child.is_dir():
                has_sparse = (child / "sparse").exists()
                has_images = (child / "images").exists()
                if has_sparse or has_images:
                    results.append({
                        "id": f"upload_{child.name}",
                        "name": child.name,
                        "type": "upload",
                        "available": True,
                        "scenes": [child.name],
                    })
    return results


# ---------------------------------------------------------------------------
# Blocking workers (run in executor)
# ---------------------------------------------------------------------------
def worker_download(job_id: str, dataset_id: str, loop: asyncio.AbstractEventLoop):
    job = JOBS[job_id]
    job["status"] = "running"
    broadcast_from_thread(loop, {
        "type": "job_progress", "job_id": job_id, "jobType": "download",
        "step": "download", "message": f"Downloading {dataset_id}...", "percent": 0,
    })
    try:
        from download_dataset import dataset_download as dd
        dd(dataset=dataset_id, save_dir=DATA_DIR)
        job["status"] = "complete"
        broadcast_from_thread(loop, {
            "type": "job_complete", "job_id": job_id, "jobType": "download",
        })
    except Exception as e:
        job["status"] = "error"
        job["meta"]["error"] = str(e)
        broadcast_from_thread(loop, {
            "type": "job_error", "job_id": job_id, "jobType": "download", "error": str(e),
        })


def worker_colmap(job_id: str, upload_dir: Path, loop: asyncio.AbstractEventLoop):
    job = JOBS[job_id]
    job["status"] = "running"

    steps = [
        ("extract_frames", "Extracting frames from video"),
        ("feature_extraction", "Running COLMAP feature extraction"),
        ("feature_matching", "Running COLMAP feature matching"),
        ("reconstruction", "Running COLMAP sparse reconstruction"),
        ("convert_model", "Converting model to TXT format"),
    ]

    def emit(step: str, msg: str, pct: int | None = None):
        broadcast_from_thread(loop, {
            "type": "job_progress", "job_id": job_id, "jobType": "colmap",
            "step": step, "message": msg, "percent": pct,
        })

    try:
        # Find input video
        video_file = None
        for f in upload_dir.iterdir():
            if f.name.startswith("input_video"):
                video_file = f
                break
        if not video_file:
            raise FileNotFoundError("No input video found")

        images_dir = upload_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # 1. Extract frames
        emit("extract_frames", "Extracting frames with ffmpeg...", 0)
        cmd = [
            "ffmpeg", "-i", str(video_file),
            "-qscale:v", "1", "-qmin", "1",
            str(images_dir / "%04d.jpg"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr[:500]}")
        emit("extract_frames", "Frames extracted", 20)

        # 2. Feature extraction
        emit("feature_extraction", "Extracting features...", 20)
        db_path = upload_dir / "database.db"
        cmd = [
            COLMAP_BIN, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "0",
        ]
        import os
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"Feature extraction failed: {r.stderr[:500]}")
        emit("feature_extraction", "Features extracted", 40)

        # 3. Feature matching
        emit("feature_matching", "Matching features...", 40)
        cmd = [
            COLMAP_BIN, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", "0",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"Feature matching failed: {r.stderr[:500]}")
        emit("feature_matching", "Features matched", 60)

        # 4. Sparse reconstruction
        emit("reconstruction", "Building sparse model...", 60)
        sparse_dir = upload_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        cmd = [
            COLMAP_BIN, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"Mapper failed: {r.stderr[:500]}")
        emit("reconstruction", "Sparse reconstruction done", 80)

        # 5. Convert model
        emit("convert_model", "Converting model...", 80)
        model_dir = sparse_dir / "0"
        if not model_dir.exists():
            # Some COLMAP versions output directly to sparse/
            model_dir = sparse_dir
        cmd = [
            COLMAP_BIN, "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(model_dir),
            "--output_type", "TXT",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Model conversion failed: {r.stderr[:500]}")
        emit("convert_model", "Model converted", 100)

        job["status"] = "complete"
        broadcast_from_thread(loop, {
            "type": "job_complete", "job_id": job_id, "jobType": "colmap",
        })
    except Exception as e:
        job["status"] = "error"
        job["meta"]["error"] = str(e)
        broadcast_from_thread(loop, {
            "type": "job_error", "job_id": job_id, "jobType": "colmap",
            "error": str(e), "step": job.get("meta", {}).get("current_step", ""),
        })


def worker_train(run_id: str, params: dict, loop: asyncio.AbstractEventLoop):
    """Run training in a thread, parsing stdout for progress."""
    run = RUNS[run_id]
    run["status"] = "running"

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        model_path = str(OUTPUT_DIR / run_id)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(os.path.join(model_path, "previews"), exist_ok=True)

        # Build command
        cmd = [
            sys.executable, os.path.join(REPO_ROOT, "train.py"),
            "-s", params["source_path"],
            "-m", model_path,
            "--iterations", str(params.get("iterations", 30000)),
            "--sh_degree", str(params.get("sh_degree", 3)),
            "--densification_strategy", params.get("densification_strategy", "default"),
            "--viewer_port", str(VIEWER_PORT),
            "--test_iterations", "7000", "30000",
            "--save_iterations", "7000", "30000",
        ]

        if params.get("optimizer_type") and params["optimizer_type"] != "default":
            cmd.extend(["--optimizer_type", params["optimizer_type"]])
        if params.get("white_background"):
            cmd.append("--white_background")
        if params.get("cauchy_activation"):
            cmd.append("--cauchy_activation")
        if params.get("cauchy_loss"):
            cmd.append("--cauchy_loss")
        if params.get("entropy_reg"):
            cmd.append("--entropy_reg")

        total_iters = int(params.get("iterations", 30000))

        # Run as subprocess to capture stdout
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=REPO_ROOT,
        )

        iteration_re = re.compile(
            r"Training progress.*?(\d+)%\|.*?(\d+)/(\d+).*?Loss[=:]\s*([\d.]+)"
        )
        loss_re = re.compile(r"Loss[=:]\s*([\d.eE\-\+]+)")
        vram_re = re.compile(r"VRAM[=:]\s*([\d.]+)")
        iter_re = re.compile(r"(\d+)%\|")
        saving_re = re.compile(r"\[ITER (\d+)\] Saving Gaussians")
        eval_psnr_re = re.compile(r"\[ITER (\d+)\] Evaluating test.*?PSNR\s+tensor\(([\d.]+)")

        last_broadcast = 0
        current_iter = 0
        last_lines = []

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            last_lines.append(line)
            if len(last_lines) > 15:
                last_lines.pop(0)

            # Parse tqdm-style progress
            # Try to find iteration count from progress bar
            pct_match = iter_re.search(line)
            loss_match = loss_re.search(line)
            vram_match = vram_re.search(line)

            if pct_match:
                pct = int(pct_match.group(1))
                current_iter = int(pct * total_iters / 100)

            # Check for saving checkpoints (trigger preview)
            save_match = saving_re.search(line)
            if save_match:
                save_iter = int(save_match.group(1))
                # Check for rendered preview
                preview_candidates = [
                    os.path.join(model_path, "test", f"ours_{save_iter}", "renders", "00000.png"),
                ]
                for pc in preview_candidates:
                    if os.path.exists(pc):
                        broadcast_from_thread(loop, {
                            "type": "run_preview",
                            "run_id": run_id,
                            "iter": save_iter,
                            "image_url": f"/renders/{run_id}/test/ours_{save_iter}/renders/00000.png",
                        })
                        break

            # Check for PSNR evaluation
            psnr_match = eval_psnr_re.search(line)
            if psnr_match:
                run["metrics"]["psnr"] = float(psnr_match.group(2))

            # Throttle broadcast to ~2/sec
            now = time.time()
            if now - last_broadcast > 0.5 and (loss_match or pct_match):
                msg = {
                    "type": "run_progress",
                    "run_id": run_id,
                    "iteration": current_iter,
                    "total": total_iters,
                    "loss": float(loss_match.group(1)) if loss_match else None,
                    "vram_mb": float(vram_match.group(1)) * 1024 if vram_match else None,
                }
                if run["metrics"].get("psnr"):
                    msg["psnr"] = run["metrics"]["psnr"]
                broadcast_from_thread(loop, msg)
                last_broadcast = now

        proc.wait()

        if proc.returncode != 0:
            error_msg = "\\n".join(last_lines)
            raise RuntimeError(f"Training process exited with code {proc.returncode}\\n{error_msg}")

        # Load final metrics if available
        metrics_file = os.path.join(model_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                run["metrics"].update(json.load(f))

        run["status"] = "complete"
        broadcast_from_thread(loop, {
            "type": "run_complete", "run_id": run_id,
            "metrics": run["metrics"],
        })

    except Exception as e:
        run["status"] = "error"
        run["error"] = str(e)
        broadcast_from_thread(loop, {
            "type": "run_error", "run_id": run_id, "error": str(e),
        })


# ---------------------------------------------------------------------------
# HTTP Handlers
# ---------------------------------------------------------------------------
async def handle_health(request: web.Request):
    gpu = False
    gpu_name = None
    try:
        import torch
        gpu = torch.cuda.is_available()
        if gpu:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return web.json_response({"ok": True, "gpu": gpu, "gpu_name": gpu_name})


async def handle_datasets(request: web.Request):
    datasets = await asyncio.get_event_loop().run_in_executor(executor, scan_datasets)
    return web.json_response(datasets)


async def handle_download(request: web.Request):
    body = await request.json()
    ds_id = body.get("id")
    if ds_id not in CANONICAL_DATASETS:
        return web.json_response({"error": "Unknown dataset"}, status=400)
    job = make_job("download", {"dataset_id": ds_id})
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, worker_download, job["id"], ds_id, loop)
    return web.json_response({"job_id": job["id"]})


async def handle_upload(request: web.Request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        return web.json_response({"error": "No file uploaded"}, status=400)

    upload_id = str(uuid.uuid4())[:8]
    upload_dir = DATA_DIR / "uploads" / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = field.filename or "video.mp4"
    ext = Path(filename).suffix or ".mp4"
    dest = upload_dir / f"input_video{ext}"

    with open(dest, "wb") as f:
        while True:
            chunk = await field.read_chunk(8192)
            if not chunk:
                break
            f.write(chunk)

    job = make_job("colmap", {"upload_id": upload_id, "upload_dir": str(upload_dir)})
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, worker_colmap, job["id"], upload_dir, loop)
    return web.json_response({"job_id": job["id"], "upload_id": upload_id})


async def handle_create_run(request: web.Request):
    body = await request.json()

    try:
        import torch
        if not torch.cuda.is_available():
            return web.json_response({"error": "CUDA not available"}, status=503)
    except ImportError:
        return web.json_response({"error": "PyTorch not installed"}, status=503)

    run_id = str(uuid.uuid4())[:8]
    source_path = body.get("source_path", "")
    scene = body.get("scene", "")

    # Resolve source path
    if not os.path.isabs(source_path):
        source_path = str(DATA_DIR / source_path)
    if scene:
        source_path = os.path.join(source_path, scene)

    run = {
        "id": run_id,
        "status": "queued",
        "params": body,
        "source_path": source_path,
        "model_path": str(OUTPUT_DIR / run_id),
        "created_at": time.time(),
        "metrics": {},
        "viewer_url": f"http://{HOST}:{VIEWER_PORT}" if VIEWER_PORT > 0 else None,
    }
    RUNS[run_id] = run

    params = dict(body)
    params["source_path"] = source_path

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, worker_train, run_id, params, loop)

    return web.json_response({
        "run_id": run_id,
        "viewer_url": run["viewer_url"],
    })


async def handle_list_runs(request: web.Request):
    results = []
    # In-memory runs
    for run in RUNS.values():
        results.append({
            "id": run["id"],
            "status": run["status"],
            "params": run.get("params", {}),
            "metrics": run.get("metrics", {}),
            "created_at": run.get("created_at"),
            "viewer_url": run.get("viewer_url"),
        })
    # Also scan OUTPUT_DIR for runs not in memory
    if OUTPUT_DIR.exists():
        for child in OUTPUT_DIR.iterdir():
            if child.is_dir() and child.name not in RUNS:
                cfg_file = child / "cfg_args"
                metrics_file = child / "metrics.json"
                run_info = {
                    "id": child.name,
                    "status": "complete",
                    "params": {},
                    "metrics": {},
                    "created_at": child.stat().st_mtime,
                }
                if cfg_file.exists():
                    try:
                        run_info["params"]["cfg_args"] = cfg_file.read_text()
                    except Exception:
                        pass
                if metrics_file.exists():
                    try:
                        run_info["metrics"] = json.loads(metrics_file.read_text())
                    except Exception:
                        pass
                results.append(run_info)
    results.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return web.json_response(results)


async def handle_get_run(request: web.Request):
    run_id = request.match_info["id"]
    run = RUNS.get(run_id)
    if run:
        previews = []
        preview_dir = OUTPUT_DIR / run_id / "previews"
        if preview_dir.exists():
            for f in sorted(preview_dir.iterdir()):
                previews.append(f"/renders/{run_id}/previews/{f.name}")
        return web.json_response({
            "id": run["id"],
            "status": run["status"],
            "params": run.get("params", {}),
            "metrics": run.get("metrics", {}),
            "previews": previews,
            "viewer_url": run.get("viewer_url"),
            "error": run.get("error"),
        })
    # Check disk
    run_dir = OUTPUT_DIR / run_id
    if run_dir.exists():
        return web.json_response({
            "id": run_id,
            "status": "complete",
            "params": {},
            "metrics": {},
            "previews": [],
        })
    return web.json_response({"error": "Run not found"}, status=404)


async def handle_run_renders(request: web.Request):
    run_id = request.match_info["id"]
    run_dir = OUTPUT_DIR / run_id
    renders = []
    if run_dir.exists():
        for root, dirs, files in os.walk(run_dir):
            for f in sorted(files):
                if f.endswith((".png", ".jpg", ".jpeg")):
                    rel = os.path.relpath(os.path.join(root, f), OUTPUT_DIR)
                    renders.append(f"/renders/{rel}")
    return web.json_response({"renders": renders})


async def handle_get_job(request: web.Request):
    job_id = request.match_info["id"]
    job = JOBS.get(job_id)
    if not job:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response(job)


async def handle_ws(request: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    WS_CLIENTS.add(ws)
    try:
        async for msg in ws:
            pass  # Server doesn't need client messages
    finally:
        WS_CLIENTS.discard(ws)
    return ws


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
def create_app():
    app = web.Application(client_max_size=1024 * 1024 * 500)  # 500MB upload

    # Routes
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/datasets", handle_datasets)
    app.router.add_post("/api/datasets/download", handle_download)
    app.router.add_post("/api/upload", handle_upload)
    app.router.add_post("/api/runs", handle_create_run)
    app.router.add_get("/api/runs", handle_list_runs)
    app.router.add_get("/api/runs/{id}", handle_get_run)
    app.router.add_get("/api/runs/{id}/renders", handle_run_renders)
    app.router.add_get("/api/jobs/{id}", handle_get_job)
    app.router.add_get("/ws", handle_ws)

    # Static file serving for renders
    if OUTPUT_DIR.exists():
        app.router.add_static("/renders/", path=str(OUTPUT_DIR), name="renders")

    # CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*",
        )
    })
    for route in list(app.router.routes()):
        try:
            cors.add(route)
        except ValueError:
            pass  # Skip WebSocket and static routes

    return app


if __name__ == "__main__":
    print(f"Starting Gaussian Splatting API server on {HOST}:{PORT}")
    print(f"  DATA_DIR:    {DATA_DIR}")
    print(f"  OUTPUT_DIR:  {OUTPUT_DIR}")
    print(f"  VIEWER_PORT: {VIEWER_PORT}")
    app = create_app()
    web.run_app(app, host=HOST, port=PORT)
