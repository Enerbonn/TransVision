# Adapted from FrameRead; modified in2026 for TransVision. See NOTICE and LICENSE.
"""
frame_extractor.py
Extracts scene-change keyframes directly from a video file utilizing OpenCV.
Automatically falls back to strict CPU bounds vs accelerated GPU batch.
"""

import cv2
import os
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass
from src.utils.hardware import HardwareConfig
from src.config import HIST_THRESHOLD, SSIM_THRESHOLD, MIN_FRAME_INTERVAL
from src.utils.logger import log


@dataclass
class ExtractedFrame:
    path: str
    timestamp_str: str
    frame_idx: int


def extract_frames(
    video_path: str,
    output_folder: str,
    hardware: HardwareConfig,
    observer=None,
    content_roi=None,
    start_frame=0,
    end_frame=None,
) -> list[ExtractedFrame]:
    """
    Extract keyframes on structural / scene changes and save them to output_folder.

    Scene-change comparisons are performed on a small 320x180 copy of each frame
    to reduce CPU usage. Original-resolution frames are still saved when a
    scene change is detected.
    """
    mode_str = "GPU-BATCH" if hardware.device == "cuda" else "CPU-SEQ"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 error opening video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = total_frames if end_frame is None else min(total_frames, end_frame)
    if not 0 <= start_frame < stop:
        cap.release()
        raise ValueError("Invalid frame range.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    log("FRAMES", "Starting keyframe extraction...")
    log("FRAMES", f"Mode: {mode_str} | Video: {width}x{height} @ {fps:.1f}fps")
    log(
        "FRAMES",
        f"Thresholds → hist={HIST_THRESHOLD}, ssim={SSIM_THRESHOLD}, min_interval={MIN_FRAME_INTERVAL} frames",
    )

    # Small resolution used ONLY for scene-change comparisons.
    # Original frames are still saved when a scene change is detected.
    compare_width = 320
    compare_height = 180

    def content(frame):
        if content_roi is None:
            return frame
        x, y, w, h = content_roi
        return frame[y : y + h, x : x + w]

    start_time = time.time()
    timings = {
        "decode_s": 0.0,
        "resize_histogram_s": 0.0,
        "ssim_s": 0.0,
        "candidate_write_s": 0.0,
        "compared_frames": 0,
        "ssim_identical_skips": 0,
    }
    last_heartbeat = time.perf_counter()

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        log("FRAMES", "Warning: Could not read even the first frame of the video!")
        return []
    if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1 != start_frame:
        cap.release()
        raise RuntimeError(
            "Video decoder could not seek to the selected starting frame."
        )

    # Resize first frame for comparisons.
    prev_small = cv2.resize(
        content(prev_frame),
        (compare_width, compare_height),
        interpolation=cv2.INTER_AREA,
    )

    # Initialize color histograms using the small frame.
    prev_hist = []
    for channel in cv2.split(prev_small):
        h = cv2.calcHist([channel], [0], None, [256], [0, 256])
        h = cv2.normalize(h, h).flatten()
        prev_hist.append(h)

    # Initialize grayscale comparison frame.
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)

    frame_idx = start_frame
    saved_count = 0
    extracted_records = []

    def format_ts(f_idx: int, fps_val: float) -> str:
        s = f_idx / fps_val
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    # Save frame 0 at its original resolution.
    ts_0 = format_ts(frame_idx, fps)
    filename0 = os.path.join(output_folder, f"keyframe_{saved_count:04d}.jpg")
    cv2.imwrite(filename0, prev_frame)
    extracted_records.append(ExtractedFrame(filename0, ts_0, frame_idx))

    log(
        "FRAMES",
        f"Processing frame {start_frame} → saved keyframe_{saved_count:04d} (range opening)",
    )

    saved_count += 1
    last_save_idx = start_frame

    while frame_idx + 1 < stop:
        tick = time.perf_counter()
        ret, frame = cap.read()
        timings["decode_s"] += time.perf_counter() - tick
        if not ret:
            break

        frame_idx += 1
        if observer and time.perf_counter() - last_heartbeat >= 5:
            observer(
                "progress", frame_idx=frame_idx, total_frames=total_frames, **timings
            )
            last_heartbeat = time.perf_counter()

        if frame_idx - last_save_idx < MIN_FRAME_INTERVAL:
            continue

        # Resize ONLY for scene-change comparison.
        tick = time.perf_counter()
        small_frame = cv2.resize(
            content(frame),
            (compare_width, compare_height),
            interpolation=cv2.INTER_AREA,
        )

        # Color Histogram check on the small frame.
        hist_diff_max = 0

        for i, channel in enumerate(cv2.split(small_frame)):
            h = cv2.calcHist([channel], [0], None, [256], [0, 256])
            h = cv2.normalize(h, h).flatten()

            diff = cv2.compareHist(prev_hist[i], h, cv2.HISTCMP_CHISQR)

            hist_diff_max = max(hist_diff_max, diff)

        # SSIM comparison on the small grayscale frame.
        curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        timings["resize_histogram_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        if np.array_equal(prev_gray, curr_gray):
            ssim_value = 1.0
            timings["ssim_identical_skips"] += 1
        else:
            ssim_value = ssim(
                prev_gray, curr_gray, data_range=curr_gray.max() - curr_gray.min()
            )
        timings["ssim_s"] += time.perf_counter() - tick
        timings["compared_frames"] += 1

        should_save = False
        reason = ""

        if hist_diff_max > HIST_THRESHOLD:
            should_save = True
            reason += f"hist={hist_diff_max:.3f} "

        if ssim_value < SSIM_THRESHOLD:
            should_save = True
            reason += f"ssim={ssim_value:.3f}"

        if should_save:
            # IMPORTANT:
            # Save the ORIGINAL full-resolution frame, not small_frame.
            filename = os.path.join(output_folder, f"keyframe_{saved_count:04d}.jpg")

            tick = time.perf_counter()
            cv2.imwrite(filename, frame)
            timings["candidate_write_s"] += time.perf_counter() - tick
            if observer:
                observer(
                    "candidate",
                    frame_idx=frame_idx,
                    hist_difference=hist_diff_max,
                    ssim=float(ssim_value),
                    reason=reason,
                )

            ts_f = format_ts(frame_idx, fps)

            extracted_records.append(ExtractedFrame(filename, ts_f, frame_idx))

            log(
                "FRAMES",
                f"Frame {frame_idx} → SCENE CHANGE "
                f"[{reason.strip()}] → "
                f"saved keyframe_{saved_count:04d}",
            )

            # Update comparison reference using the small frame.
            prev_hist = []

            for channel in cv2.split(small_frame):
                h = cv2.calcHist([channel], [0], None, [256], [0, 256])

                h = cv2.normalize(h, h).flatten()

                prev_hist.append(h)

            prev_gray = curr_gray

            last_save_idx = frame_idx
            saved_count += 1

    cap.release()
    if frame_idx != stop - 1:
        raise RuntimeError(
            f"Video decoding stopped at frame {frame_idx}, before selected end {stop - 1}."
        )
    if observer:
        observer(
            "complete",
            frame_idx=frame_idx,
            total_frames=total_frames,
            processed_frames=frame_idx - start_frame + 1,
            start_frame=start_frame,
            end_frame_exclusive=stop,
            candidates=saved_count,
            **timings,
        )

    elapsed = time.time() - start_time

    log(
        "FRAMES",
        f"✓ Extraction complete — {saved_count} keyframes "
        f"from {frame_idx - start_frame + 1} processed frames ({elapsed:.1f}s)",
    )

    log("FRAMES", f"Frames saved → {output_folder}")

    return extracted_records
