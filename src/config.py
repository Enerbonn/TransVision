# Adapted from FrameRead; modified in2026 for TransVision. See NOTICE and LICENSE.
"""
config.py
Tunable constants for the FrameRead v2 manual-review pipeline.

Everything that used to live here for the retired Qwen2-VL / Ollama
auto-summary pipeline (VISION_MODEL, SUMMARY_MODEL, OLLAMA_HOST,
OLLAMA_STARTUP_WAIT, FRAME_ANALYSIS_PROMPT, SUMMARY_SYSTEM_PROMPT,
SUMMARY_USER_TEMPLATE, QA_SYSTEM_PROMPT, PRINT_SUMMARY, TEMP_DIR_PREFIX)
has been removed. See docs/legacy/ if you ever need that pipeline back.
"""

# ── Frame Extraction (src/video/frame_extractor.py) ──
HIST_THRESHOLD = 0.28  # χ² histogram distance. Lower = more sensitive.
SSIM_THRESHOLD = 0.89  # Structural similarity. Lower = more keyframes.
MIN_FRAME_INTERVAL = 8  # Minimum frames between two saved keyframes.

# Post-processing thresholds; the frame-extraction baseline above is unchanged.
AMBIENT_GRID_SIZE = 24
AMBIENT_DIFF_THRESHOLD = 12
AMBIENT_CELL_ACTIVE_FRAC = 0.08
AMBIENT_RATIO = 0.35
MIN_CONTENT_AREA_FRAC = 0.003
BURST_GAP_S = 6.0
MAJOR_CHANGE_MULT = 2.5
REVIEW_THUMB_MAX_DIM = 240

# NOTE: region_activity.py (GRID_SIZE, DIFF_THRESHOLD, CELL_ACTIVE_FRAC,
# AMBIENT_RATIO, MIN_CONTENT_AREA_FRAC) and burst_grouper.py (BURST_GAP_S,
# MAJOR_CHANGE_MULT) still keep their own tunables as module-level
# constants rather than here. Migrating them into this file would match
# the project's original "no magic numbers outside config.py" rule, but
# wasn't done in this pass to keep this cleanup's diff small and reversible.
