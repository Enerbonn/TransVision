"""
region_activity.py
Layout-independent jitter classifier — new stage that runs AFTER
extract_frames() (frame_extractor.py). Does NOT replace it.

KEY IDEA: we don't know in advance where a webcam or participant-thumbnail
box will be on screen. But whatever they are, they move in almost every
transition between keyframes, for the ENTIRE video, regardless of what's
happening in the lecture. Real content changes (writing, erasing, slide
swaps) are the opposite: they show up in only a handful of transitions,
clustered around specific moments.

So instead of hardcoding a screen region, this computes a per-cell
"activity ratio" across ALL keyframe-to-keyframe transitions in the video:
cells active in a high fraction of transitions get flagged "ambient" and
excluded from the meaningful-change decision for every transition. Cells
active only occasionally are real content.

This is a whole-sequence (batch) pass, not per-frame online logic, because
"chronically active" is only meaningful relative to the full video.

Input: the List[ExtractedFrame] that frame_extractor.extract_frames() already
produces (your existing fast baseline, unchanged).
Output: the same list, each frame carrying two new attributes:
    - content_change_frac: float, non-ambient changed-cell fraction vs. the
      previous keyframe
    - is_meaningful: bool
plus the learned ambient_mask (for logging/debugging).
"""

import cv2
import numpy as np

from src.config import (
    AMBIENT_GRID_SIZE,
    AMBIENT_DIFF_THRESHOLD,
    AMBIENT_CELL_ACTIVE_FRAC,
    AMBIENT_RATIO,
    MIN_CONTENT_AREA_FRAC,
)
from src.utils.logger import log
from src.video.frame_extractor import ExtractedFrame


def _cell_activity_map(gray_a, gray_b, grid_size):
    """Returns a (grid_size, grid_size) bool array: which cells changed."""
    diff = cv2.absdiff(gray_a, gray_b)
    changed = diff > AMBIENT_DIFF_THRESHOLD
    h, w = changed.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    activity = np.zeros((grid_size, grid_size), dtype=bool)
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = gy * cell_h, (gy + 1) * cell_h if gy < grid_size - 1 else h
            x0, x1 = gx * cell_w, (gx + 1) * cell_w if gx < grid_size - 1 else w
            cell = changed[y0:y1, x0:x1]
            if cell.size and cell.mean() > AMBIENT_CELL_ACTIVE_FRAC:
                activity[gy, gx] = True
    return activity


def classify_frames(
    frames: list[ExtractedFrame], resize_to=(320, 180), content_roi=None
):
    """
    frames: List[ExtractedFrame] from extract_frames(), sorted by frame_idx.

    Returns (classified, ambient_mask) where `classified` is a list of dicts:
        {"frame": ExtractedFrame, "content_change_frac": float, "is_meaningful": bool}
    """
    log(
        "REVIEW",
        f"Classifying {len(frames)} keyframes for ambient vs. meaningful change...",
    )

    n = len(frames)
    grid = AMBIENT_GRID_SIZE
    previous = None
    activity_count = np.zeros((grid, grid), dtype=int)
    per_transition_activity = []
    for fr in frames:
        img = cv2.imread(fr.path)
        if img is None:
            raise RuntimeError(f"Could not read candidate frame: {fr.path}")
        if content_roi is not None:
            x, y, w, h = content_roi
            img = img[y : y + h, x : x + w]
        small = cv2.resize(img, resize_to)
        current = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            act = _cell_activity_map(previous, current, grid)
            per_transition_activity.append(np.packbits(act))
            activity_count += act.astype(int)
        previous = current

    if n < 2:
        classified = [
            {"frame": fr, "content_change_frac": 0.0, "is_meaningful": False}
            for fr in frames
        ]
        return classified, np.zeros((grid, grid), dtype=bool)

    ambient_mask = (activity_count / (n - 1)) >= AMBIENT_RATIO
    ambient_cells = int(ambient_mask.sum())
    log(
        "REVIEW",
        f"Learned ambient mask: {ambient_cells}/{grid * grid} cells flagged as chronic motion (webcam/thumbnail-like)",
    )

    total_cells = grid * grid
    classified = [
        {"frame": frames[0], "content_change_frac": 0.0, "is_meaningful": False}
    ]
    for i in range(1, n):
        act = (
            np.unpackbits(per_transition_activity[i - 1], count=grid * grid)
            .reshape(grid, grid)
            .astype(bool)
        )
        non_ambient_active = act & (~ambient_mask)
        frac = non_ambient_active.sum() / total_cells
        is_meaningful = bool(frac >= MIN_CONTENT_AREA_FRAC)
        classified.append(
            {
                "frame": frames[i],
                "content_change_frac": round(float(frac), 4),
                "is_meaningful": is_meaningful,
            }
        )

    meaningful_count = sum(1 for c in classified if c["is_meaningful"])
    log(
        "REVIEW",
        f"{meaningful_count}/{len(classified)} keyframes classified as meaningful content change",
    )

    return classified, ambient_mask
