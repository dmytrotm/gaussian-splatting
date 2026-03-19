"""
Video-to-Images Extraction Utility

Extract frames from a video file at a specified FPS for use as
3DGS training data.

Usage:
    python -m utils.video2imgs --video input.mp4 --output data/images/ --fps 2

Ported from gsplat reference (video2imgs.py).
"""

import argparse
import os
import sys

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def extract_frames(video_path: str, output_dir: str, fps: float = 2.0) -> int:
    """Extract frames from a video file at the specified FPS.

    Args:
        video_path: Path to input video file.
        output_dir: Directory to save extracted frames.
        fps: Frames per second to extract (default: 2.0).

    Returns:
        Number of frames extracted.
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
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {video_fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")
    print(f"  Extracting at {fps} FPS -> ~{int(duration * fps)} frames")

    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))

    extracted = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{extracted:05d}.jpg")
            cv2.imwrite(filename, frame)
            extracted += 1

        frame_idx += 1

    cap.release()
    print(f"  Extracted {extracted} frames to {output_dir}")
    return extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract (default: 2.0)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps)
