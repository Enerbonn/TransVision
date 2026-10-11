"""Recover outgoing content at major slide boundaries, using the chosen ROI."""

import cv2
import numpy as np

from src.content import crop_image


def final_frames(session, markers, cues=None, report=None, recorder=None):
    cues = sorted(cues if cues is not None else markers, key=lambda m: m["frame_idx"])
    result = [dict(m) for m in markers]
    existing = {m["frame_idx"] for m in markers}
    cap = cv2.VideoCapture(session["video"])
    if not cap.isOpened():
        raise ValueError("Cannot open source video.")
    roi = session.get("content_roi", [0, 0, session["width"], session["height"]])

    def read(index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = cap.read()
        if not ok:
            raise ValueError(f"Cannot read boundary frame {index}.")
        if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1 != index:
            raise ValueError("Boundary decoder frame mismatch.")
        return cv2.cvtColor(
            cv2.resize(crop_image(img, roi), (320, 180)), cv2.COLOR_BGR2GRAY
        )

    def distance(a, b):
        return float(np.mean(cv2.absdiff(a, b) > 24))

    try:
        for i, cue in enumerate(cues):
            end = cue["frame_idx"]
            if cue.get("kind") != "slide_change" or end < 1 or not i:
                continue
            start = cues[i - 1]["frame_idx"]
            if start >= end:
                continue
            old, new = read(start), read(end)
            # Ignore small changes such as handwriting or webcam-only movement.
            if distance(old, new) < 0.06:
                if recorder:
                    recorder.event(
                        "boundary_rejected",
                        frame_idx=end,
                        reason="small_content_change",
                    )
                continue
            prior = read(end - 1)
            if distance(prior, old) < distance(prior, new):
                last = end - 1
            else:
                # Detection can lag the actual cut. Locate the old/new boundary.
                lo, hi = start, end
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    image = read(mid)
                    if distance(image, old) < distance(image, new):
                        lo = mid
                    else:
                        hi = mid
                last = lo
            if last not in existing:
                result.append(
                    {
                        "id": f"pre-{last}",
                        "frame_idx": last,
                        "timestamp_s": last / session["fps"],
                        "kind": "pre_change",
                        "included": True,
                        "ai": False,
                        "before_frame_idx": end,
                    }
                )
                existing.add(last)
            if recorder:
                recorder.event(
                    "boundary_capture",
                    frame_idx=last,
                    timestamp_s=last / session["fps"],
                    before_frame_idx=end,
                )
            if report:
                report(
                    f"Finding completed outgoing frames: {i + 1}/{len(cues)}",
                    progress=round(100 * (i + 1) / len(cues)),
                )
    finally:
        cap.release()
    return sorted(result, key=lambda m: m["frame_idx"])
