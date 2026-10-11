"""Reusable visual analysis and recoverable export detail levels. No OCR needed."""

from pathlib import Path
import copy
import math
import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / "models/face_detection_yunet_2023mar.onnx"


def session_sensitivity(session):
    """Read the unified control, or translate the old two-control review."""
    if "sensitivity" in session:
        return validate_sensitivity(session["sensitivity"])
    level = session.get("detail_level", "completed")
    old = validate_sensitivity(session.get("detail_sensitivity", 50))
    if level == "completed":
        return 0
    threshold = (0.012 * 0.12 ** (old / 100) if level == "major"
                 else 0.003 * 0.1 ** (old / 100))
    middle = 0.012 * 0.12 ** 0.5
    value = (1 + 49 * math.log(threshold / 0.12) / math.log(middle / 0.12)
             if threshold >= middle
             else 50 + 50 * math.log(threshold / middle) / math.log(0.0003 / middle))
    return max(1, min(100, round(value)))


def validate_sensitivity(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value) or not 0 <= value <= 100:
        raise ValueError("Sensitivity must be a whole number between 0 and 100.")
    return int(value)


def sensitivity_threshold(value):
    value = validate_sensitivity(value)
    middle = 0.012 * 0.12 ** 0.5
    if value <= 50:
        return 0.12 * (middle / 0.12) ** ((value - 1) / 49)
    return middle * (0.0003 / middle) ** ((value - 50) / 50)


def apply_sensitivity(markers, sensitivity=50, hide_camera=True, motion_mode="summary"):
    """Monotonic detail selection over cached candidates, without video decoding."""
    sensitivity = validate_sensitivity(sensitivity)
    from src.motion import effective_analysis, validate_motion_mode
    validate_motion_mode(motion_mode)
    result = copy.deepcopy(markers)
    groups = {}
    for mark in result:
        mark.pop("detail_excluded", None)
        # Protected manual/AI marks need no analysis to retain the user's choice.
        if mark.get("user_override") or mark.get("kind") == "manual" or mark.get("ai"):
            continue
        analysis = effective_analysis(mark, motion_mode) if mark.get("visual_analysis") else None
        if not analysis:
            raise ValueError("Analyze this review first to preview sensitivity.")
        mark["included"] = False
        if analysis.get("suppressed"):
            mark["detail_excluded"] = analysis.get("reason", "repeated content")
            continue
        if hide_camera and analysis.get("camera"):
            mark["detail_excluded"] = "camera"
            continue
        groups.setdefault(analysis["page_group"], []).append(mark)
    threshold = sensitivity_threshold(sensitivity)
    for group in groups.values():
        group.sort(key=lambda m: m["frame_idx"])
        eligible = [m for m in group if not effective_analysis(m, motion_mode).get("noise")]
        # A trailing cursor-only/no-change candidate must not replace useful content.
        final = (eligible or group)[-1]
        keep = {final["frame_idx"]}
        keep.update(m["frame_idx"] for m in group if effective_analysis(m, motion_mode).get("force_keep"))
        if sensitivity:
            keep.update(m["frame_idx"] for m in eligible
                        if sensitivity == 100 or effective_analysis(m, motion_mode).get("changed_fraction", 0) >= threshold)
        for mark in group:
            mark["included"] = mark["frame_idx"] in keep
            if not mark["included"]:
                mark["detail_excluded"] = (effective_analysis(mark, motion_mode).get("reason", "pointer/no change") if effective_analysis(mark, motion_mode).get("noise")
                                           else "earlier writing step")
    return result, {
        "included": sum(bool(m.get("included", True)) for m in result),
        "total": len(result),
        "camera_excluded": sum(m.get("detail_excluded") == "camera" for m in result),
        "sensitivity": sensitivity, "hide_camera": hide_camera,
    }


class CameraFilter:
    def __init__(self):
        self.detector = (
            cv2.FaceDetectorYN.create(str(MODEL), "", (320, 192), 0.85, 0.3, 100)
            if MODEL.is_file()
            else None
        )
        self.reference = None

    def check(self, image):
        if self.detector is None:
            return False
        _, faces = self.detector.detect(cv2.resize(image, (320, 192)))
        # A small presenter tile must not hide a slide. Only a large foreground face.
        face = faces is not None and any(
            w * h / (320 * 192) > 0.065 and w / 320 > 0.18
            for x, y, w, h, *rest in faces
        )
        gray = cv2.cvtColor(cv2.resize(image, (128, 72)), cv2.COLOR_BGR2GRAY)
        # Keep a recognized camera scene when a face looks down or becomes occluded.
        continuity = False
        if self.reference is not None:
            delta = cv2.absdiff(gray, self.reference)
            continuity = float(delta.mean()) < 14 and float((delta < 20).mean()) > 0.55
        if face:
            self.reference = gray
        return bool(face or continuity)


def thumbnail(image):
    return cv2.GaussianBlur(
        cv2.cvtColor(
            cv2.resize(image, (480, 270), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2GRAY,
        ),
        (3, 3),
        0,
    )


def difference(a, b):
    changed = (cv2.absdiff(a, b) > 25).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(changed)
    significant = stats[1:][stats[1:, cv2.CC_STAT_AREA] >= 6]
    total = int(sum(significant[:, cv2.CC_STAT_AREA])) if len(significant) else 0
    compact = bool(
        len(significant) <= 4
        and all(
            max(row[cv2.CC_STAT_WIDTH], row[cv2.CC_STAT_HEIGHT]) < 24
            for row in significant
        )
    )
    fraction = total / a.size
    # Compact, isolated pointer movement is noise. Larger/new equation content remains.
    spread = 0
    if len(significant) > 1:
        centers = significant[:, :2] + significant[:, 2:4] / 2
        spread = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    noise = (
        fraction < 0.0002 and not len(significant)
        or compact
        and len(significant) > 1
        and spread > 30
        and fraction < 0.002
    )
    old = a < 170
    new = b < 170
    removed = float((old & ~new).sum() / max(1, old.sum()))
    page_change = not noise and (fraction > 0.12 or removed > 0.28 and fraction > 0.008)
    return {
        "changed_fraction": round(fraction, 5),
        "removed_ink": round(removed, 5),
        "noise": noise,
        "page_change": page_change,
    }


def apply_detail(markers, detail="completed", sensitivity=50, hide_camera=True):
    if detail not in ("completed", "major", "steps"):
        raise ValueError("Choose completed pages, major steps, or every step.")
    sensitivity = int(sensitivity)
    if not 0 <= sensitivity <= 100:
        raise ValueError("Sensitivity must be between 0 and 100.")
    result = copy.deepcopy(markers)
    groups = {}
    for mark in result:
        if "visual_analysis" not in mark:
            raise ValueError(
                "Analyze this review first to enable export detail controls."
            )
        a = mark["visual_analysis"]
        mark.pop("detail_excluded", None)
        if mark.get("user_override") or mark.get("kind") == "manual" or mark.get("ai"):
            continue
        mark["included"] = False
        if hide_camera and a.get("camera"):
            mark["detail_excluded"] = "camera"
            continue
        groups.setdefault(a["page_group"], []).append(mark)
    threshold = 0.012 * (0.12 ** (sensitivity / 100))
    for group in groups.values():
        eligible = [m for m in group if not m["visual_analysis"].get("noise")]
        if not eligible:
            eligible = group[:1]
        keep = {group[-1]["frame_idx"]} if group else set()
        if detail == "steps":
            keep.update(
                m["frame_idx"]
                for m in eligible
                if m["visual_analysis"].get("changed_fraction", 0)
                >= 0.003 * (0.1 ** (sensitivity / 100))
            )
        elif detail == "major":
            keep.update(
                m["frame_idx"]
                for m in eligible
                if m["visual_analysis"].get("changed_fraction", 0) >= threshold
            )
        for mark in group:
            mark["included"] = mark["frame_idx"] in keep
            if not mark["included"]:
                mark["detail_excluded"] = (
                    "pointer/no change"
                    if mark["visual_analysis"].get("noise")
                    else "earlier writing step"
                )
    return result, {
        "included": sum(m.get("included", True) for m in result),
        "total": len(result),
        "camera_excluded": sum(m.get("detail_excluded") == "camera" for m in result),
        "detail": detail,
        "sensitivity": sensitivity,
        "hide_camera": hide_camera,
    }


def analyze_selection(
    session,
    markers,
    roi,
    report=None,
    recorder=None,
    analysis_paths=None,
    analysis_cropped=False,
    camera_overrides=None,
):
    from src.content import crop_image

    camera = CameraFilter()
    cap = cv2.VideoCapture(session["video"])
    result = []
    previous = None
    from src.motion import GrayCache
    motion_images = GrayCache()
    motion_circles = []
    group = -1
    was_camera = False
    exported = {
        e["frame_idx"]: Path(session["result"]["folder"]) / e["filename"]
        for e in (session.get("result") or {}).get("entries", [])
    }
    try:
        for i, original in enumerate(sorted(markers, key=lambda m: m["frame_idx"])):
            mark = dict(original)
            path = exported.get(mark["frame_idx"])
            image = None
            if analysis_paths and mark["frame_idx"] in analysis_paths:
                full = cv2.imread(str(analysis_paths[mark["frame_idx"]]))
                image = (
                    (full if analysis_cropped else crop_image(full, roi))
                    if full is not None
                    else None
                )
            if image is None and path and path.is_file() and session.get("result"):
                entry = next(
                    e
                    for e in session["result"]["entries"]
                    if e["frame_idx"] == mark["frame_idx"]
                )
                if (
                    entry.get("export_cropped")
                    and entry.get("source_content_roi") == roi
                ):
                    image = cv2.imread(str(path))
                elif not entry.get("export_cropped"):
                    full = cv2.imread(str(path))
                    image = crop_image(full, roi) if full is not None else None
            if image is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, mark["frame_idx"])
                ok, full = cap.read()
                if (
                    not ok
                    or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                    != mark["frame_idx"]
                ):
                    raise ValueError("Cannot read exact analysis frame.")
                image = crop_image(full, roi)
            is_camera = (
                camera_overrides[mark["frame_idx"]]
                if camera_overrides is not None
                and mark["frame_idx"] in camera_overrides
                else camera.check(image)
            )
            current = thumbnail(image)
            motion_images.append(current)
            from src.motion import circle_candidates
            motion_circles.append(np.packbits(circle_candidates(image)))
            d = (
                difference(previous, current)
                if previous is not None
                else {
                    "noise": False,
                    "page_change": True,
                    "changed_fraction": 1,
                    "removed_ink": 0,
                }
            )
            if d["page_change"] or is_camera != was_camera:
                group += 1
            mark["visual_analysis"] = {
                **d,
                "camera": bool(is_camera),
                "page_group": group,
            }
            result.append(mark)
            previous = current
            was_camera = is_camera
            if recorder:
                recorder.event(
                    "selection_analyzed",
                    frame_idx=mark["frame_idx"],
                    **mark["visual_analysis"],
                )
            if report and (i % 25 == 0 or i == len(markers) - 1):
                report(
                    f"Analyzing export detail: {i + 1}/{len(markers)}",
                    progress=round((i + 1) * 100 / max(1, len(markers))),
                    completed=i + 1,
                    total=len(markers),
                )
    finally:
        cap.release()
    from src.motion import annotate_motion
    return annotate_motion(result, motion_images, session["fps"], report, motion_circles)
