"""Content-area selection and conservative, recoverable visual duplicate checks."""

import cv2
import numpy as np


def validate_roi(roi, width, height):
    if roi is None:
        return [0, 0, width, height]
    if (
        not isinstance(roi, list)
        or len(roi) != 4
        or any(isinstance(v, bool) or not isinstance(v, int) for v in roi)
    ):
        raise ValueError(
            "Content area must be four pixel integers: x, y, width, height."
        )
    x, y, w, h = roi
    if x < 0 or y < 0 or w < 30 or h < 30 or x + w > width or y + h > height:
        raise ValueError(
            "Content area must stay inside the video and be at least 30 × 30 pixels."
        )
    return roi


def suggest_roi(image):
    """Largest flat light background, with its heading included. Otherwise full frame.

    This is only a suggestion, visible and editable in the UI; no fixed screen position.
    """
    height, width = image.shape[:2]
    small = cv2.resize(image, (max(1, width // 2), max(1, height // 2)))
    bins = (small // 32).astype(np.int32)
    codes = bins[:, :, 0] + 8 * bins[:, :, 1] + 64 * bins[:, :, 2]
    counts = np.bincount(codes.ravel(), minlength=512)
    candidates = []
    for code in np.argsort(counts)[-12:]:
        color = np.array([code % 8, (code // 8) % 8, code // 64])
        if color.mean() < 4:
            continue
        mask = (codes == code).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= 0.25 * small.shape[0] * small.shape[1] and w > h:
                candidates.append(
                    (
                        cv2.contourArea(contour),
                        [
                            x * 2,
                            y * 2,
                            min(w * 2, width - x * 2),
                            min(h * 2, height - y * 2),
                        ],
                    )
                )
    if not candidates:
        return [0, 0, width, height]
    _, rect = max(candidates, key=lambda v: v[0])
    x, y, w, h = rect
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    columns = gray[y : y + h].mean(axis=0)
    threshold = max(80, min(160, float(np.median(columns[x : x + w])) * 0.65))
    left = x
    right = x + w
    while left > 0 and columns[left - 1] > threshold:
        left -= 1
    while right < width and columns[right] > threshold:
        right += 1
    x, w = left, right - left
    # Include a dark title bar/footer above/below the light slide canvas, while
    # excluding black video padding. Look only inside the inferred content columns.
    rows = gray[:, x + w // 4 : x + 3 * w // 4].mean(axis=1) > 22
    top = y
    while top > 0 and rows[top - 1]:
        top -= 1
    bottom = y + h
    while bottom < height and rows[bottom]:
        bottom += 1
    return [x, top, w, bottom - top]


def crop_image(image, roi):
    x, y, w, h = validate_roi(roi, image.shape[1], image.shape[0])
    return image[y : y + h, x : x + w]


def comparison_image(image, roi):
    crop = crop_image(image, roi)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Preserve source resolution: a small new formula must not disappear in a tiny thumbnail.
    return cv2.GaussianBlur(gray, (3, 3), 0)


def compare_content(left, right):
    if left.shape != right.shape:
        return {
            "duplicate": False,
            "changed_pixels": left.size,
            "largest_change": left.size,
        }
    diff = cv2.absdiff(left, right)
    changed = (diff > 20).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(changed)
    largest = int(max(stats[1:, cv2.CC_STAT_AREA], default=0))
    count = int(changed.sum())
    # Compression speckles are disconnected; actual strokes are connected.
    # Exclusion is recoverable in the editor. Manual/AI marks are never auto-excluded.
    duplicate = (
        largest < 32
        and count < max(100, 0.001 * left.size)
        and float(diff.mean()) < 2.5
    )
    return {
        "duplicate": duplicate,
        "changed_pixels": count,
        "largest_change": largest,
        "mean_difference": round(float(diff.mean()), 4),
    }


def optimize_markers(session, markers, roi, report=None):
    cap = cv2.VideoCapture(session["video"])
    if not cap.isOpened():
        raise ValueError("Cannot open the source video.")
    kept = []
    updated = []
    suppressed = 0
    try:
        for i, original in enumerate(sorted(markers, key=lambda m: m["frame_idx"])):
            mark = dict(original)
            # Restore only earlier automatic duplicate exclusions for re-comparison.
            if mark.get("duplicate_of") is not None and not mark.get("user_override"):
                mark["included"] = True
            mark.pop("duplicate_of", None)
            mark.pop("similarity", None)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mark["frame_idx"])
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Cannot read frame {mark['frame_idx']}.")
            current = comparison_image(image, roi)
            comparison = (
                compare_content(kept[-1][1], current) if kept else {"duplicate": False}
            )
            if (
                comparison["duplicate"]
                and mark["kind"] != "manual"
                and not mark["ai"]
                and not mark.get("user_override")
            ):
                mark["duplicate_of"] = kept[-1][0]
                mark["similarity"] = comparison
                if mark["included"]:
                    mark["included"] = False
                    suppressed += 1
            elif mark["included"]:
                kept = [(mark["frame_idx"], current)]
            updated.append(mark)
            if report:
                report(
                    f"Checking content changes: {i + 1}/{len(markers)}",
                    progress=round((i + 1) * 100 / max(1, len(markers))),
                )
    finally:
        cap.release()
    return updated, {
        "excluded_similar": suppressed,
        "included": sum(m["included"] for m in updated),
        "content_roi": roi,
    }
