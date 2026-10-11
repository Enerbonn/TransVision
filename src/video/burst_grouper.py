"""
burst_grouper.py
New stage that runs AFTER region_activity.classify_frames(). Groups
temporally-close "meaningful" keyframes into a single activity burst (e.g.
one whole writing-then-pause round) and keeps only the LAST frame of each
burst as the checkpoint — representing "writing/erasing finished" for that
round.

This is deliberately NOT the same as the earlier failed persistence
experiment, which required the WHOLE VIDEO to go quiet before checkpointing
(collapsed 62 real candidates down to 3-7 and lost real events). Here,
stability is only required *within* a local burst window (BURST_GAP_S), so
multiple separate writing/erasing rounds each still get their own checkpoint.

A candidate whose content_change_frac is far above the recent baseline (a
MAJOR_CHANGE_MULT multiple of the median meaningful size) is treated as an
immediate "slide change" checkpoint on its own, regardless of burst timing.

Needs fps to convert ExtractedFrame.frame_idx into seconds (timestamp_str is
whole-second precision only, which is too coarse for burst-gap comparisons).
"""

import statistics

from src.config import BURST_GAP_S, MAJOR_CHANGE_MULT
from src.utils.logger import log


def group_bursts(classified: list[dict], fps: float, burst_gap_s: float = BURST_GAP_S):
    """
    classified: output of region_activity.classify_frames() (list of
        {"frame": ExtractedFrame, "content_change_frac": float, "is_meaningful": bool})
    fps: source video's frames-per-second, used to convert frame_idx -> seconds.

    Returns a list of checkpoint dicts:
        {"frame": ExtractedFrame, "timestamp_s": float, "content_change_frac": float,
         "trigger": "burst_end" | "slide_change", "burst_size": int, "burst_start_s": float}
    """
    meaningful = [c for c in classified if c["is_meaningful"]]
    for c in meaningful:
        c["timestamp_s"] = c["frame"].frame_idx / fps

    log(
        "REVIEW",
        f"Grouping {len(meaningful)} meaningful keyframes into bursts (gap<= {burst_gap_s}s)...",
    )

    if not meaningful:
        return []

    fracs = [c["content_change_frac"] for c in meaningful]
    median_frac = statistics.median(fracs) if fracs else 0.0
    major_threshold = median_frac * MAJOR_CHANGE_MULT

    checkpoints = []
    current_burst = []

    def flush_burst(burst):
        last = burst[-1]
        checkpoints.append(
            {
                **last,
                "trigger": "burst_end",
                "burst_size": len(burst),
                "burst_start_s": burst[0]["timestamp_s"],
            }
        )

    for c in meaningful:
        is_major = major_threshold > 0 and c["content_change_frac"] >= major_threshold

        if is_major:
            if current_burst:
                flush_burst(current_burst)
            checkpoints.append(
                {
                    **c,
                    "trigger": "slide_change",
                    "burst_size": 1,
                    "burst_start_s": c["timestamp_s"],
                }
            )
            current_burst = []
            continue

        if not current_burst:
            current_burst = [c]
            continue

        prev = current_burst[-1]
        gap = c["timestamp_s"] - prev["timestamp_s"]

        if gap <= burst_gap_s:
            current_burst.append(c)
        else:
            flush_burst(current_burst)
            current_burst = [c]

    if current_burst:
        flush_burst(current_burst)

    log(
        "REVIEW",
        f"Burst grouping complete: {len(meaningful)} meaningful keyframes -> {len(checkpoints)} checkpoints",
    )
    return checkpoints
