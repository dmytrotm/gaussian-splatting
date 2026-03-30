"""
Video-to-Images Extraction with Quality Filtering

Extracts frames from a video at a specified FPS, then applies two filters:

1. **Tenengrad sharpness** (Sobel-gradient energy): rejects motion-blurred
   or out-of-focus frames whose sharpness falls below a percentile threshold.
2. **SSIM deduplication**: compares consecutive surviving frames via
   Structural Similarity and drops near-duplicates that exceed a similarity
   ceiling, retaining only visually informative keyframes.

The combination produces a compact, high-quality frame set suitable for
COLMAP SfM and subsequent 3DGS training.

Usage (standalone):
    python -m utils.video2imgs \\
        --video input.mp4 \\
        --output data/my_scene/input \\
        --fps 3 \\
        --sharpness_percentile 20 \\
        --ssim_threshold 0.92

Usage (from code):
    from utils.video2imgs import extract_frames
    n = extract_frames("vid.mp4", "out/", fps=3,
                       sharpness_percentile=20, ssim_threshold=0.92)
"""

import argparse
import os
import sys
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Tenengrad sharpness (Sobel gradient energy)
# ──────────────────────────────────────────────────────────────

def tenengrad(image: np.ndarray, ksize: int = 3) -> float:
    """Compute Tenengrad focus measure via Sobel gradient magnitude.

    Tenengrad is defined as the mean squared gradient magnitude:

        T = (1/N) Σ (Gx² + Gy²)

    where Gx, Gy are horizontal / vertical Sobel responses computed
    on the grayscale image.  Higher values indicate sharper focus.

    Args:
        image: BGR or grayscale uint8 image.
        ksize: Sobel kernel size (3, 5, or 7).

    Returns:
        Scalar sharpness score (float).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.mean(gx ** 2 + gy ** 2))


# ──────────────────────────────────────────────────────────────
# SSIM (simplified structural similarity)
# ──────────────────────────────────────────────────────────────

def compute_ssim(img_a: np.ndarray, img_b: np.ndarray,
                 downsample: int = 4) -> float:
    """Compute mean SSIM between two images (fast, downsampled).

    Uses the Wang et al. (2004) formulation with default constants
    C1 = (0.01 × 255)² and C2 = (0.03 × 255)² on a Gaussian-weighted
    11×11 window, computed on the luminance channel only.

    Args:
        img_a: BGR or grayscale uint8 image.
        img_b: BGR or grayscale uint8 image (same shape as img_a).
        downsample: Factor to downsample before computation (speed).

    Returns:
        SSIM in [−1, 1] (typically [0.5, 1.0] for natural images).
    """
    def _to_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    a = _to_gray(img_a).astype(np.float64)
    b = _to_gray(img_b).astype(np.float64)

    # Downsample for speed
    if downsample > 1:
        a = cv2.resize(a, (a.shape[1] // downsample, a.shape[0] // downsample),
                       interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (b.shape[1] // downsample, b.shape[0] // downsample),
                       interpolation=cv2.INTER_AREA)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    win_size = 11

    mu_a = cv2.GaussianBlur(a, (win_size, win_size), 1.5)
    mu_b = cv2.GaussianBlur(b, (win_size, win_size), 1.5)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a ** 2, (win_size, win_size), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b ** 2, (win_size, win_size), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (win_size, win_size), 1.5) - mu_ab

    ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2))

    return float(np.mean(ssim_map))


# ──────────────────────────────────────────────────────────────
# Main extraction pipeline
# ──────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 2.0,
    sharpness_percentile: float = 15.0,
    ssim_threshold: float = 0.93,
) -> int:
    """Extract frames from video with Tenengrad + SSIM quality filtering.

    Pipeline:
        1. Decode video at the target FPS (temporal subsampling).
        2. Score every extracted frame with Tenengrad sharpness.
        3. Reject the bottom ``sharpness_percentile`` % of frames.
        4. Walk the remaining frames sequentially and drop any frame
           whose SSIM with the previous kept frame exceeds
           ``ssim_threshold`` (near-duplicate suppression).
        5. Save surviving frames as JPEG to ``output_dir``.

    Args:
        video_path: Path to input video file.
        output_dir: Directory to save filtered frames.
        fps: Target extraction rate (frames per second).
        sharpness_percentile: Bottom percentile of Tenengrad scores to
            reject (0 = keep all, 30 = reject blurriest 30%).
        ssim_threshold: Maximum allowed SSIM between consecutive kept
            frames. Pairs above this are considered duplicates.
            Range guidance: 0.90–0.95 for moderate dedup, 0.85 aggressive.

    Returns:
        Number of frames saved.
    """
    if not CV2_AVAILABLE:
        print("ERROR: opencv-python is required. Install with: pip install opencv-python")
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {video_fps:.1f}, Frames: {total_frames}, "
          f"Duration: {duration:.1f}s")
    print(f"  Target extraction FPS: {fps}")

    # ── Step 1: Extract raw frames at target FPS ─────────────
    frame_interval = max(1, int(video_fps / fps))
    raw_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            raw_frames.append(frame)
        frame_idx += 1

    cap.release()
    print(f"\n  Phase 1 — Extracted {len(raw_frames)} raw frames "
          f"(every {frame_interval}th)")

    if len(raw_frames) == 0:
        print("  ERROR: No frames extracted.")
        return 0

    # ── Step 2: Tenengrad sharpness scoring ──────────────────
    print("  Phase 2 — Computing Tenengrad sharpness scores...")
    scores = [tenengrad(f) for f in raw_frames]
    scores_arr = np.array(scores)

    threshold = np.percentile(scores_arr, sharpness_percentile)
    sharp_mask = scores_arr >= threshold
    sharp_indices = np.where(sharp_mask)[0]

    n_rejected_blur = len(raw_frames) - len(sharp_indices)
    print(f"    Sharpness range: [{scores_arr.min():.0f}, "
          f"{scores_arr.max():.0f}], threshold (p{sharpness_percentile:.0f}): "
          f"{threshold:.0f}")
    print(f"    Rejected {n_rejected_blur} blurry frames, "
          f"{len(sharp_indices)} remain")

    # ── Step 3: SSIM deduplication ───────────────────────────
    print(f"  Phase 3 — SSIM deduplication (threshold={ssim_threshold})...")
    kept_indices = [sharp_indices[0]]

    for i in range(1, len(sharp_indices)):
        idx = sharp_indices[i]
        prev_idx = kept_indices[-1]
        ssim_val = compute_ssim(raw_frames[prev_idx], raw_frames[idx])
        if ssim_val < ssim_threshold:
            kept_indices.append(idx)

    n_rejected_dup = len(sharp_indices) - len(kept_indices)
    print(f"    Rejected {n_rejected_dup} near-duplicate frames, "
          f"{len(kept_indices)} unique frames remain")

    # ── Step 4: Save surviving frames ────────────────────────
    print(f"  Phase 4 — Saving {len(kept_indices)} frames to {output_dir}/")
    for save_idx, orig_idx in enumerate(kept_indices):
        filename = os.path.join(output_dir, f"frame_{save_idx:05d}.jpg")
        cv2.imwrite(filename, raw_frames[orig_idx], [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Summary
    print(f"\n  ── Summary ──────────────────────────────────")
    print(f"  Raw extracted:    {len(raw_frames):>5}")
    print(f"  Rejected (blur):  {n_rejected_blur:>5}  "
          f"(bottom {sharpness_percentile:.0f}% Tenengrad)")
    print(f"  Rejected (dupes): {n_rejected_dup:>5}  "
          f"(SSIM > {ssim_threshold})")
    print(f"  Final kept:       {len(kept_indices):>5}  "
          f"({100*len(kept_indices)/len(raw_frames):.0f}% of raw)")
    print(f"  Saved to: {output_dir}/")

    return len(kept_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from video with sharpness + dedup filtering")
    parser.add_argument("--video", required=True,
                        help="Path to input video file")
    parser.add_argument("--output", required=True,
                        help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Target extraction rate (default: 2.0)")
    parser.add_argument("--sharpness_percentile", type=float, default=15.0,
                        help="Reject bottom N%% by Tenengrad sharpness (default: 15)")
    parser.add_argument("--ssim_threshold", type=float, default=0.93,
                        help="SSIM ceiling for duplicate rejection (default: 0.93)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps,
                   args.sharpness_percentile, args.ssim_threshold)
