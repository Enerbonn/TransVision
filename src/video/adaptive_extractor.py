"""Time-based coarse sampling, local transition refinement, and camera suppression."""

import time
from pathlib import Path

import cv2

from src.cancellation import check_cancel
from src.content import crop_image
from src.selection import CameraFilter, difference, thumbnail
from src.video.frame_extractor import ExtractedFrame
from src.motion import ScanMotionGuard


def extract_adaptive(
    video,
    folder,
    roi,
    start,
    stop,
    fps,
    observer=None,
    sample_seconds=0.5,
    hide_camera=True,
):
    if not 0.125 <= sample_seconds <= 2:
        raise ValueError("Scan interval must be 0.125 to 2 seconds.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    refiner = cv2.VideoCapture(str(video))
    camera = CameraFilter()
    motion_guard = ScanMotionGuard()
    records = []
    seen = set()
    step = max(1, round(fps * sample_seconds))
    half = max(1, step // 2)
    timings = {
        "decode_s": 0.0,
        "comparison_s": 0.0,
        "refinement_s": 0.0,
        "candidate_write_s": 0.0,
        "camera_s": 0.0,
        "compared_frames": 0,
        "refined_frames": 0,
        "camera_samples": 0,
        "decoded_frames": 0,
    }
    last_progress = time.perf_counter()
    prev = None
    anchor = None
    prev_idx = start
    prev_camera = False
    idx = start - 1

    def notify(kind, **data):
        if observer:
            observer(kind, **data)

    def save(index, image, kind, **extra):
        if index in seen:
            return
        seen.add(index)
        path = folder / f"keyframe_{len(records):04d}.jpg"
        t = time.perf_counter()
        # Candidates are analysis proxies. Final selected PNGs always use source pixels.
        proxy = cv2.resize(
            crop_image(image, roi), (640, 360), interpolation=cv2.INTER_LINEAR
        )
        if not cv2.imwrite(str(path), proxy, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError("Could not save candidate.")
        timings["candidate_write_s"] += time.perf_counter() - t
        rec = ExtractedFrame(str(path), str(index / fps), index)
        rec.kind = kind
        rec.extra = extra
        records.append(rec)
        notify("candidate", frame_idx=index, reason=kind, **extra)

    def read(index):
        t = time.perf_counter()
        refiner.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = refiner.read()
        timings["refinement_s"] += time.perf_counter() - t
        timings["refined_frames"] += 1
        if not ok or int(round(refiner.get(cv2.CAP_PROP_POS_FRAMES))) - 1 != index:
            raise RuntimeError("Refinement decoder frame mismatch.")
        return image

    try:
        check_cancel()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        # Midpoint probes bound brief-slide misses to half the chosen scan interval.
        while idx + 1 < stop:
            check_cancel()
            idx += 1
            t = time.perf_counter()
            ok = cap.grab()
            timings["decode_s"] += time.perf_counter() - t
            timings["decoded_frames"] += 1
            if not ok:
                raise RuntimeError(f"Decoding stopped at frame {idx}.")
            if idx != start and idx != stop - 1 and (idx - start) % half:
                continue
            t = time.perf_counter()
            ok, image = cap.retrieve()
            timings["decode_s"] += time.perf_counter() - t
            if not ok:
                raise RuntimeError(f"Cannot retrieve sampled frame {idx}.")
            content = crop_image(image, roi)
            t = time.perf_counter()
            is_camera = camera.check(content) if hide_camera else False
            timings["camera_s"] += time.perf_counter() - t
            if is_camera:
                timings["camera_samples"] += 1
            t = time.perf_counter()
            current = thumbnail(content)
            d = (
                difference(prev, current)
                if prev is not None
                else {"noise": False, "page_change": True, "changed_fraction": 1}
            )
            timings["comparison_s"] += time.perf_counter() - t
            timings["compared_frames"] += 1
            if not is_camera:
                d = motion_guard.check(current, d)
            if prev is None:
                save(idx, image, "opening", camera=bool(is_camera))
                anchor = current
            elif is_camera != prev_camera:
                save(prev_idx, read(prev_idx), "pre_change", camera=bool(prev_camera))
                save(
                    idx,
                    image,
                    "camera" if is_camera else "slide_change",
                    camera=bool(is_camera),
                )
                anchor = current
            elif not is_camera and (
                not d["noise"]
                or anchor is not None
                and not difference(anchor, current)["noise"]
            ):
                if d["page_change"]:
                    lo, hi = prev_idx, idx
                    # Refine only abrupt, one-boundary intervals. Preserve both sides.
                    while hi - lo > 1:
                        check_cancel()
                        mid = (lo + hi) // 2
                        m = thumbnail(crop_image(read(mid), roi))
                        if (
                            abs(m.astype("float32") - prev).mean()
                            <= abs(m.astype("float32") - current).mean()
                        ):
                            lo = mid
                        else:
                            hi = mid
                    save(lo, read(lo), "pre_change", camera=False)
                    save(hi, read(hi), "slide_change", camera=False)
                else:
                    save(idx, image, "burst_end", camera=False)
                anchor = current
            if idx == stop - 1 and not is_camera:
                save(idx, image, "pre_change", camera=False)
            prev = current
            prev_idx = idx
            prev_camera = is_camera
            if time.perf_counter() - last_progress >= 5:
                notify("progress", frame_idx=idx, total_frames=stop, **timings)
                last_progress = time.perf_counter()
    finally:
        cap.release()
        refiner.release()
    notify(
        "complete",
        frame_idx=stop - 1,
        total_frames=stop,
        processed_frames=stop - start,
        start_frame=start,
        end_frame_exclusive=stop,
        candidates=len(records),
        sample_seconds=sample_seconds,
        **timings,
    )
    return sorted(records, key=lambda r: r.frame_idx)
