"""Persistent, sequential local video jobs. No LLM or paid API calls."""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil

from src.cancellation import JobCancelled, cancellable, check_cancel
from src.content import suggest_roi, validate_roi
from src.pdf_export import export_pdf
from src.pdf_parts import split_pdf
from src.pipeline import detect, finalize, prepare_video, save_json
from src.selection import CameraFilter, apply_sensitivity, validate_sensitivity


def stamp():
    return datetime.now(timezone.utc).isoformat()


def settings(value):
    task = value.get("task", "pdf")
    if task not in ("pdf", "ocr", "both"):
        raise ValueError("Choose PDF only, OCR only, or both.")
    reference = float(value.get("crop_reference_seconds", 5))
    interval = float(value.get("sample_seconds", 0.5))
    if not math.isfinite(reference) or reference < 0:
        raise ValueError("Crop reference time must be zero or greater.")
    if not math.isfinite(interval) or not 0.125 <= interval <= 2:
        raise ValueError("Scan interval must be 0.125 to 2 seconds.")
    profile = value.get("ocr_profile", "prose")
    if profile not in ("auto", "prose", "code", "math"):
        raise ValueError("Choose a valid OCR profile.")
    max_pages = value.get("max_pdf_pages", 100)
    if (
        isinstance(max_pages, bool)
        or not isinstance(max_pages, int)
        or not 1 <= max_pages <= 100
    ):
        raise ValueError("PDF parts must contain between 1 and 100 pages.")
    return {
        "task": task,
        "sensitivity": validate_sensitivity(value.get("sensitivity", 50)),
        "automatic_export": bool(value.get("automatic_export", True)),
        "pdf_timestamps": bool(value.get("pdf_timestamps", True)),
        "hide_camera": bool(value.get("hide_camera", True)),
        "slides_only": bool(value.get("slides_only", True)),
        "crop_reference_seconds": reference,
        "sample_seconds": interval,
        "ocr_profile": profile,
        "max_pdf_pages": max_pages,
    }


def presentation_content(image, face_detector=None):
    """Conservative light-slide check for optional unattended slide-only export."""
    small = cv2.resize(image, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    colors = (small // 32).astype(np.int32)
    codes = colors[:, :, 0] + 8 * colors[:, :, 1] + 64 * colors[:, :, 2]
    flat = np.bincount(codes.ravel(), minlength=512).max() / codes.size
    if not ((gray > 170).mean() > 0.45 and flat > 0.35):
        return False
    if face_detector is not None:
        _, faces = face_detector.detect(cv2.resize(image, (320, 192)))
        # The crop excludes presenter tiles on genuine slides. Faces inside it
        # usually mean the meeting switched to a participant/camera grid.
        if faces is not None and len(faces):
            return False
    return True


class FrameDecodingError(ValueError):
    """Reference frames could not be decoded, rather than lacking slide content."""


class NoSlideDetectedError(ValueError):
    """Decoded reference frames did not pass the visual presentation checks."""


def crop_reference(video, metadata, seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Crop reference time must be zero or greater.")
    fps, count = metadata["fps"], metadata["frame_count"]
    duration = min(metadata.get("duration", count / fps), count / fps)
    candidates = []
    seen = set()
    for offset in (0, 5, 30, 120, 300):
        value = seconds + offset
        if value >= duration:
            continue
        index = round(value * fps)
        if not 0 <= index < count or index in seen:
            continue
        seen.add(index)
        candidates.append(index)
    if not candidates:
        raise NoSlideDetectedError(
            "No crop-reference timestamps fall within this video. Choose an earlier reference time."
        )
    cap = cv2.VideoCapture(str(video))
    decoded = 0
    failed = 0
    try:
        if not cap.isOpened():
            raise FrameDecodingError(
                "Cannot open the video for automatic crop detection. Check its path and codec."
            )
        for index in candidates:
            check_cancel()
            try:
                sought = cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read() if sought else (False, None)
                exact = ok and int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1 == index
            except cv2.error:
                ok, image, exact = False, None, False
            if not ok or image is None or not image.size or not exact:
                failed += 1
                continue
            decoded += 1
            roi = validate_roi(
                suggest_roi(image), metadata["width"], metadata["height"]
            )
            x, y, w, h = roi
            # A valid slide can already fill the entire frame. Full frame is not
            # evidence of detector failure; evaluate its visual content too.
            if presentation_content(image[y : y + h, x : x + w]):
                return roi, index / fps
        if not decoded:
            raise FrameDecodingError(
                f"Could not decode any of the {failed} crop-reference frames. Check the video codec or retry."
            )
        detail = f" {failed} reference frame(s) could not be decoded." if failed else ""
        raise NoSlideDetectedError(
            "No slide detected in the decoded crop-reference frames. Select a crop manually or change the reference time."
            + detail
        )
    finally:
        cap.release()


class BatchQueue:
    def __init__(self, state_file, output_root):
        self.path = Path(state_file).resolve()
        self.output_root = Path(output_root).resolve()
        self.lock_path = self.path.with_suffix(".worker.json")
        self.pause_path = self.path.with_suffix(".pause")
        self.stop_path = self.path.with_suffix(".stop")
        self.guard = threading.RLock()
        self.thread = None

    def running(self):
        try:
            lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
            process = psutil.Process(lock["pid"])
            return (
                process.is_running()
                and abs(process.create_time() - lock["created"]) < 1
            )
        except (OSError, ValueError, KeyError, psutil.Error):
            return False

    def snapshot(self):
        with self.guard:
            if self.path.is_file():
                result = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                result = {"settings": settings({}), "jobs": []}
            result["running"] = self.running()
            result["pause_requested"] = self.pause_path.exists()
            result["stop_requested"] = self.stop_path.exists()
            return result

    def write(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.path, state)

    def enqueue(self, inputs, options):
        with self.guard:
            if self.running():
                raise ValueError(
                    "Pause or finish the current batch before adding another batch."
                )
            config = settings(options)
            if not isinstance(inputs, list) or not 1 <= len(inputs) <= 100:
                raise ValueError("Add between 1 and 100 video paths.")
            jobs = []
            for entry in inputs:
                entry = {"video": entry} if isinstance(entry, str) else entry
                path = Path(str(entry["video"]).strip().strip('"')).resolve()
                if not path.is_file():
                    raise ValueError(f"Video not found: {path.name}")
                if any(j["video"] == str(path) for j in jobs):
                    continue
                jobs.append(
                    {
                        "id": uuid.uuid4().hex,
                        "video": str(path),
                        "name": path.name,
                        "status": "queued",
                        "progress": 0,
                        "message": "Queued",
                        "added_at": stamp(),
                        "completed_at": None,
                        "content_roi": entry.get("content_roi"),
                        "crop_source": "manual"
                        if entry.get("content_roi") is not None
                        else "automatic",
                        "requested_reference_seconds": entry.get(
                            "reference_seconds", config["crop_reference_seconds"]
                        ),
                        "crop_reference_seconds": entry.get(
                            "reference_seconds", config["crop_reference_seconds"]
                        ),
                        "settings": copy.deepcopy(config),
                        "outputs": [],
                        "error": None,
                    }
                )
            state = self.snapshot()
            state = {
                "created_at": state.get("created_at", stamp()),
                "settings": config,
                "jobs": state["jobs"] + jobs,
                "output_root": state.get("output_root", str(self.output_root)),
            }
            self.write(state)
            return self.snapshot()

    def update(self, job_id, **changes):
        with self.guard:
            state = self.snapshot()
            job = next(j for j in state["jobs"] if j["id"] == job_id)
            job.update(changes)
            self.write(state)

    def pause(self):
        self.pause_path.parent.mkdir(parents=True, exist_ok=True)
        self.pause_path.write_text("Pause after the current file.\n", encoding="utf-8")

    def stop(self):
        self.stop_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_path.write_text(
            "Stop current job and leave pending jobs queued.\n", encoding="utf-8"
        )

    def start(self):
        with self.guard:
            if self.running() or self.thread and self.thread.is_alive():
                raise ValueError("This batch is already running.")
            self.stop_path.unlink(missing_ok=True)
            self.thread = threading.Thread(
                target=lambda: self.run(clear_stop=False), daemon=True
            )
            self.thread.start()

    def run(self, clear_stop=True):
        if clear_stop:
            self.stop_path.unlink(missing_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.running():
            raise ValueError("This batch is already running in another process.")
        # A stale lock is preserved, rather than mistaken for a running worker.
        if self.lock_path.exists():
            self.lock_path.replace(
                self.lock_path.with_name(
                    self.lock_path.name + ".previous-" + uuid.uuid4().hex[:8]
                )
            )
        with self.lock_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {"pid": os.getpid(), "created": psutil.Process().create_time()}, stream
            )
        if self.pause_path.exists():
            self.pause_path.replace(
                self.pause_path.with_name(
                    self.pause_path.name + ".previous-" + uuid.uuid4().hex[:8]
                )
            )
        awake = None
        if os.name == "nt":
            import ctypes

            awake = ctypes.windll.kernel32.SetThreadExecutionState
            awake(
                0x80000001
            )  # Keep the machine awake while working; display may sleep.
        try:
            state = self.snapshot()
            self.output_root = Path(
                state.get("output_root", self.output_root)
            ).resolve()
            for item in state["jobs"]:
                if item["status"] in ("completed", "needs_review", "failed"):
                    continue
                if self.pause_path.exists() or self.stop_path.exists():
                    break
                with cancellable(self.stop_path.exists):
                    self.process(item, item.get("settings", state["settings"]))
        finally:
            if awake:
                awake(0x80000000)
            self.lock_path.replace(
                self.lock_path.with_name(
                    self.lock_path.name + ".finished-" + uuid.uuid4().hex[:8]
                )
            )

    def process(self, job, config):
        job_id = job["id"]

        def report(message, progress=0, **details):
            check_cancel()
            self.update(
                job_id,
                message=message,
                progress=progress,
                stage=details.get("stage"),
                eta_seconds=details.get("eta_seconds"),
                updated_at=stamp(),
            )

        try:
            check_cancel()
            self.update(
                job_id,
                status="scanning",
                error=None,
                error_kind=None,
                message="Preparing slide crop",
            )
            metadata = prepare_video(job["video"])
            initial_reference = job.get(
                "requested_reference_seconds", job["crop_reference_seconds"]
            )
            manual_crop = (
                job.get("content_roi") is not None
                and job.get("crop_source") != "automatic"
            )
            if manual_crop:
                roi = validate_roi(
                    job["content_roi"], metadata["width"], metadata["height"]
                )
                reference = initial_reference
            else:
                roi, reference = crop_reference(
                    job["video"], metadata, initial_reference
                )
            self.update(
                job_id,
                content_roi=roi,
                crop_reference_seconds=reference,
                requested_reference_seconds=initial_reference,
                crop_source="manual" if manual_crop else "automatic",
            )
            # Resume a completed scan if an export was interrupted.
            path = job.get("session_path")
            if path and Path(path).is_file():
                session = json.loads(Path(path).read_text(encoding="utf-8"))
            else:
                session = detect(
                    job["video"],
                    self.output_root,
                    report,
                    content_roi=roi,
                    adaptive=True,
                    sample_seconds=config["sample_seconds"],
                    hide_camera=config["hide_camera"],
                )
                path = str(Path(session["folder"]) / "session.json")
                self.update(job_id, session_path=path)
            session.update(
                sensitivity=config["sensitivity"],
                pdf_timestamps=config["pdf_timestamps"],
                ocr_profile=config["ocr_profile"],
                export_cropped=True,
            )
            markers, stats = apply_sensitivity(
                session["markers"], config["sensitivity"], config["hide_camera"]
            )
            non_slide = 0
            if config["slides_only"]:
                face_detector = CameraFilter().detector
                for mark in markers:
                    source = session.get("candidate_paths", {}).get(
                        str(mark["frame_idx"])
                    )
                    if source:
                        image = cv2.imread(source)
                        if image is not None and not presentation_content(
                            image, face_detector
                        ):
                            mark.update(
                                included=False,
                                user_override=True,
                                detail_excluded="no slide content",
                            )
                            non_slide += 1
            session.update(markers=markers, optimization=stats, batch_job=job_id)
            save_json(Path(session["folder"]) / "session.json", session)
            self.update(
                job_id,
                selected_frames=sum(m["included"] for m in markers),
                available_frames=len(markers),
                non_slide_excluded=non_slide,
            )
            if not config["automatic_export"]:
                self.update(
                    job_id,
                    status="needs_review",
                    progress=100,
                    message="Ready for your review; nothing exported",
                )
                return
            self.update(job_id, status="exporting", progress=0)
            self.output_root.mkdir(parents=True, exist_ok=True)
            outputs = []
            if config["task"] in ("pdf", "both"):
                result = export_pdf(session, markers, report)
                session["pdf_result"] = result
                stem = Path(job["video"]).stem
                if (self.output_root / (stem + ".pdf")).exists() or (
                    self.output_root / (stem + "_01.pdf")
                ).exists():
                    stem += "-" + job_id[:8]
                parts = split_pdf(
                    Path(result["folder"]) / result["filename"],
                    self.output_root,
                    stem,
                    config.get("max_pdf_pages", 100),
                )
                outputs.extend(part["file"] for part in parts)
            if config["task"] in ("ocr", "both"):
                result = finalize(session, markers, report)
                session["result"] = result
                outputs.append(str(Path(result["folder"]) / "visual_transcript.txt"))
            save_json(Path(session["folder"]) / "session.json", session)
            self.update(
                job_id,
                status="completed",
                progress=100,
                message="Export complete",
                outputs=outputs,
                completed_at=stamp(),
            )
        except JobCancelled:
            self.update(
                job_id,
                status="queued",
                progress=0,
                error=None,
                message="Stopped; ready to resume",
                completed_at=None,
            )
        except Exception as exc:
            self.update(
                job_id,
                status="failed",
                error=str(exc),
                error_kind=type(exc).__name__,
                message=str(exc),
                completed_at=stamp(),
            )

    def retry(self):
        if self.running():
            raise ValueError(
                "Wait for the current batch to finish before retrying failures."
            )
        state = self.snapshot()
        for job in state["jobs"]:
            if job["status"] == "failed":
                job.update(
                    status="queued",
                    progress=0,
                    error=None,
                    error_kind=None,
                    completed_at=None,
                )
        self.write(state)

    def review_exported(self, session):
        """Reflect manual review exports in the queue without rerunning the job."""
        job_id = session.get("batch_job")
        if not job_id:
            return
        state = self.snapshot()
        job = next((j for j in state["jobs"] if j["id"] == job_id), None)
        if not job:
            return
        config = job.get("settings", state["settings"])
        outputs = []
        if session.get("pdf_result"):
            pdf = session["pdf_result"]
            if pdf.get("parts"):
                outputs.extend(part["file"] for part in pdf["parts"])
            else:
                outputs.append(str(Path(pdf["folder"]) / pdf["filename"]))
        if session.get("result"):
            outputs.append(
                str(Path(session["result"]["folder"]) / "visual_transcript.txt")
            )
        complete = (
            config["task"] == "pdf"
            and bool(session.get("pdf_result"))
            or config["task"] == "ocr"
            and bool(session.get("result"))
            or config["task"] == "both"
            and bool(session.get("pdf_result"))
            and bool(session.get("result"))
        )
        self.update(
            job_id,
            outputs=outputs,
            status="completed" if complete else "needs_review",
            completed_at=stamp() if complete else None,
            message="Reviewed export complete"
            if complete
            else "One export ready; remaining export needs review",
        )
