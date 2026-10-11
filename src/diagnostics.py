"""Local, append-only measurements for real lecture trials. Never uploads data."""

import json
import os
import platform
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import psutil


class TestRecorder:
    def __init__(self, root, context):
        self.folder = Path(root) / (
            datetime.now().strftime("test-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        )
        self.folder.mkdir(parents=True)
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.counts = Counter()
        self.stages = []
        self.quality = Counter()
        self.detector = {}
        self.peak_rss = 0
        self.context = context
        self.event(
            "run_started",
            context=context,
            machine={
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "ram_bytes": psutil.virtual_memory().total,
            },
        )
        self.write_summary()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def event(self, event_type, **data):
        with self.lock:
            self.counts[event_type] += 1
            if event_type == "frame_ocr":
                self.quality["ocr_frames"] += 1
                self.quality["needs_image_review"] += bool(data.get("needs_review"))
                self.quality["uncertain_lines"] += data.get("uncertain_lines", 0)
                self.quality["repeated_text_blocks_skipped"] += not data.get(
                    "transcript_included", True
                )
            if event_type == "detector_complete":
                self.detector = data
            row = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(time.monotonic() - self.started, 3),
                "event": event_type,
                **data,
            }
            with (self.folder / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            try:
                processes = [process] + process.children(recursive=True)
                rss = cpu = 0
                for item in processes:
                    try:
                        rss += item.memory_info().rss
                        times = item.cpu_times()
                        cpu += times.user + times.system
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_rss = max(self.peak_rss, rss)
                self.event(
                    "resource_sample",
                    rss_bytes=rss,
                    cpu_seconds=round(cpu, 3),
                    system_available_bytes=psutil.virtual_memory().available,
                    disk_free_bytes=psutil.disk_usage(str(self.folder)).free,
                )
                self.write_summary()
            except Exception as exc:
                self.event("sampling_error", error=str(exc))
            self.stop.wait(5)

    def stage(self, name, seconds, **data):
        with self.lock:
            self.stages.append({"name": name, "seconds": round(seconds, 3), **data})
            self.event("stage_complete", **self.stages[-1])
            self.write_summary()

    def write_summary(self, status="recording"):
        with self.lock:
            summary = {
                "status": status,
                "context": self.context,
                "elapsed_s": round(time.monotonic() - self.started, 3),
                "peak_process_tree_rss_bytes": self.peak_rss,
                "event_counts": dict(self.counts),
                "stages": self.stages,
            }
            summary.update(quality=dict(self.quality), detector=self.detector)
            temporary = self.folder / "summary.tmp"
            temporary.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            temporary.replace(self.folder / "summary.json")
            lines = [
                "# TransVision test report",
                f"Status: {status}",
                f"Elapsed: {summary['elapsed_s']:.1f} seconds",
                f"Peak FrameRead + child process RAM: {self.peak_rss / 1024**2:.1f} MiB",
                "",
                "## Stage timings",
            ]
            lines += [f"- {s['name']}: {s['seconds']:.3f}s" for s in self.stages]
            measured = [
                s
                for s in self.stages
                if s["name"] not in ("detecting", "processing", "optimizing")
            ]
            lines += ["", "## Findings"]
            if measured:
                slowest = max(measured, key=lambda s: s["seconds"])
                lines.append(
                    f"Largest measured stage: {slowest['name']} ({slowest['seconds']:.3f}s)."
                )
            if self.detector:
                lines.append(
                    "Detector breakdown: "
                    + ", ".join(
                        f"{k}: {v:.3f}s"
                        for k, v in self.detector.items()
                        if k.endswith("_s")
                    )
                )
            lines += [
                f"{key.replace('_', ' ')}: {value}"
                for key, value in self.quality.items()
            ]
            lines += [
                f"Outgoing boundary captures: {self.counts['boundary_capture']}",
                f"Review edits: {self.counts['review_edit']}; observations: {self.counts['user_observation']}; job errors: {self.counts['job_error']}",
            ]
            lines += [
                "",
                "## Evidence",
                "events.jsonl contains timestamped resource samples, candidate decisions,",
                "selected frame extraction/OCR durations, review edits, UI actions/errors, and your notes.",
                "OCR confidence is a model score, not proof of correct math. Review images flagged OCR CHECK.",
                "A recording status after a crash means the run did not close cleanly.",
                "Use source timestamps in notes to identify missed content or duplicates.",
            ]
            (self.folder / "REPORT.md").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

    def close(self, status="closed"):
        self.stop.set()
        self.thread.join(timeout=2)
        self.event("run_closed", status=status)
        self.write_summary(status)
