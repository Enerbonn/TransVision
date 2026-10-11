"""Loopback-only editor server. Run `python -m src.app`, or double-click Start TransVision.cmd."""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2

from src.batch import BatchQueue
from src.boundaries import final_frames
from src.cancellation import JobCancelled, cancellable, check_cancel
from src.content import optimize_markers, suggest_roi, validate_roi
from src.diagnostics import TestRecorder
from src.export_location import ExportLocation
from src.motion import VERSION as MOTION_VERSION, validate_motion_mode
from src.ocr import available_engines
from src.pdf_export import export_pdf_parts
from src.pipeline import (
    detect,
    finalize,
    frame_range,
    prepare_video,
    save_json,
    validate_markers,
)
from src.selection import (
    analyze_selection,
    apply_sensitivity,
    session_sensitivity,
    validate_sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]


class Application:
    def __init__(self, out_dir, session_path=None, batch_path=None, preferences_path=None):
        self.out_dir = Path(out_dir).resolve()
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.worker = None
        self.clients = {}
        self.auto_close = False
        self.clients_seen = False
        self.last_client_at = time.monotonic()
        self.preview_lock = threading.Lock()
        self.preview_cache = OrderedDict()
        self.preview_capture = None
        self.preview_video = None
        self.recorder = None
        self.batch = BatchQueue(
            batch_path or ROOT / ".local/batch-queue.json", self.out_dir
        )
        self.export_location = ExportLocation(
            preferences_path or self.batch.path.parent / "export-preferences.json",
            self.out_dir,
        )
        self.state = {
            "phase": "idle",
            "message": "Choose a video to begin.",
            "progress": 0,
            "session": None,
            "error": None,
        }
        if session_path:
            session = json.loads(Path(session_path).read_text(encoding="utf-8"))
            if "content_roi" not in session:
                cap = cv2.VideoCapture(session["video"])
                try:
                    ok, image = cap.read()
                    session["content_roi"] = (
                        suggest_roi(image)
                        if ok
                        else [0, 0, session["width"], session["height"]]
                    )
                finally:
                    cap.release()
            session.setdefault("ocr_engine", available_engines()[0])
            session.setdefault("suggested_roi", session["content_roi"].copy())
            session.setdefault("dedupe_ocr", True)
            session.setdefault("export_cropped", False)
            session.setdefault("revision", 0)
            session["sensitivity"] = session_sensitivity(session)
            self.state.update(
                phase="review", session=session, message="Saved review restored."
            )
        self.state["ocr_engines"] = available_engines()
        self.base_url = ""
        if session_path:
            self.start_recording({"resumed_session": str(session_path)})

    def start_recording(self, context):
        if self.recorder:
            self.recorder.close("continued_in_new_run")
        self.recorder = TestRecorder(ROOT / "test_runs", context)
        self.state["test_folder"] = str(self.recorder.folder)
        if self.state.get("session"):
            self.state["session"]["test_folder"] = str(self.recorder.folder)

    def record(self, kind, **data):
        if self.recorder:
            self.recorder.event(kind, **data)

    def suggest_crop(self, frame_idx, folder=None):
        with self.lock:
            session = self.state.get("session")
            if not session or folder and folder != session["folder"]:
                raise ValueError("Choose the current video first.")
            if (
                isinstance(frame_idx, bool)
                or not isinstance(frame_idx, int)
                or not 0 <= frame_idx < session["frame_count"]
            ):
                raise ValueError("Select a valid video frame.")
            video = session["video"]
            fps = session["fps"]
        cap = cv2.VideoCapture(video)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1 != frame_idx:
                raise ValueError("Could not read the selected crop-reference frame.")
            roi = suggest_roi(image)
        finally:
            cap.release()
        self.record(
            "crop_suggested",
            frame_idx=frame_idx,
            timestamp_s=frame_idx / fps,
            content_roi=roi,
        )
        return {
            "content_roi": roi,
            "frame_idx": frame_idx,
            "timestamp_s": frame_idx / fps,
        }

    def prepare(self, video):
        with self.lock:
            if self.batch.running():
                raise ValueError(
                    "Pause or finish the batch before processing another video."
                )
            if self.state["phase"] in (
                "detecting",
                "processing",
                "optimizing",
                "pdf_export",
            ):
                raise ValueError("A job is already running.")
        prepared = prepare_video(str(video).strip().strip('"'))
        check_cancel()
        with self.lock:
            self.start_recording(
                {"video": prepared["video"], "setup_before_detection": True}
            )
            prepared.update(
                folder="setup-" + secrets.token_hex(8),
                markers=[],
                revision=0,
                result=None,
                preflight=True,
                trim_start_s=0,
                trim_end_s=prepared["duration"],
                ocr_engine=available_engines()[0],
                ocr_profile="prose",
                export_cropped=True,
                sample_seconds=0.5,
                hide_camera=True,
                sensitivity=50,
            )
            self.state.update(
                phase="setup",
                session=prepared,
                error=None,
                progress=0,
                message="Confirm the crop and trim, then start detection.",
            )
            self.record("video_prepared", **prepared)
            return prepared

    def report(self, message, progress=0, **details):
        check_cancel()
        print(message, flush=True)
        with self.lock:
            self.state.update(
                message=message, progress=progress, updated_at=time.time(), **details
            )

    def snapshot(self, compact=False, known_folder=None, known_revision=None):
        with self.lock:
            current = self.state.get("session")
            if (
                compact
                and current
                and current["folder"] == known_folder
                and str(current.get("revision", 0)) == known_revision
            ):
                return copy.deepcopy(
                    {
                        **{k: v for k, v in self.state.items() if k != "session"},
                        "session": None,
                    }
                )
            if compact and self.state["session"]:
                state = {k: v for k, v in self.state.items() if k != "session"}
                session = {
                    k: v for k, v in self.state["session"].items() if k != "result"
                }
                result = self.state["session"].get("result")
                if result:
                    result = {
                        **{k: v for k, v in result.items() if k != "entries"},
                        "entries": [
                            {
                                k: v
                                for k, v in entry.items()
                                if k not in ("text", "raw_ocr_text", "ocr_lines")
                            }
                            for entry in result["entries"]
                        ],
                    }
                session["result"] = result
                state["session"] = session
                return copy.deepcopy(state)
            return copy.deepcopy(self.state)

    def media_session(self):
        with self.lock:
            session = self.state["session"]
            if not session:
                return None
            return {
                "video": session["video"],
                "folder": session["folder"],
                "frame_count": session["frame_count"],
                "pdf_result": session.get("pdf_result"),
                "result": {"folder": session["result"]["folder"]}
                if session.get("result")
                else None,
            }

    def preview_image(self, video, idx, width):
        # Reuse one decoder and cache rendered previews. Opening a decoder for every
        # thumbnail was slow and spawned excessive codec threads on Windows.
        started = time.perf_counter()
        key = (video, idx, width)
        with self.preview_lock:
            if key in self.preview_cache:
                self.preview_cache.move_to_end(key)
                return self.preview_cache[key]
            if self.preview_video != video:
                if self.preview_capture:
                    self.preview_capture.release()
                self.preview_capture = cv2.VideoCapture(video)
                self.preview_video = video
            self.preview_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.preview_capture.read()
            if not ok:
                raise ValueError("Could not read preview frame.")
            frame = cv2.resize(
                frame, (width, max(1, round(frame.shape[0] * width / frame.shape[1])))
            )
            ok, data = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("Could not render frame.")
            result = data.tobytes()
            self.preview_cache[key] = result
            while len(self.preview_cache) > 256:
                self.preview_cache.popitem(last=False)
            self.record(
                "preview_decoded",
                frame_idx=idx,
                width=width,
                seconds=round(time.perf_counter() - started, 4),
            )
            return result

    def write_review(self):
        with self.lock:
            session = self.state["session"]
            if session and not session.get("preflight"):
                save_json(Path(session["folder"]) / "session.json", session)
                (Path(session["folder"]) / "review_ui.html").write_text(
                    self.html(), encoding="utf-8"
                )

    def html(self):
        return (
            (ROOT / "web/editor.html")
            .read_text(encoding="utf-8")
            .replace("__SERVER__", self.base_url)
            .replace("__TOKEN__", self.token)
            .replace("__MOTION_VERSION__", str(MOTION_VERSION))
        )

    def stop(self):
        with self.lock:
            if self.worker and self.worker.is_alive():
                self.cancel_event.set()
                self.state.update(
                    stopping=True, message="Stopping at the next safe checkpoint…"
                )
                return {"ok": True, "stopping": True}
            return {"ok": True, "stopping": False}

    def reset(self):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("Stop the current job before choosing another video.")
            self.state.update(
                phase="idle",
                session=None,
                error=None,
                progress=0,
                message="Choose a video to begin.",
                stopping=False,
                export_folder=None,
                frames_ready=False,
            )
        return {"ok": True}

    def window_ping(self, client, closed=False):
        if not isinstance(client, str) or not 1 <= len(client) <= 100:
            raise ValueError("Invalid window identity.")
        with self.lock:
            if closed:
                self.clients.pop(client, None)
            else:
                self.clients[client] = time.monotonic()
                self.clients_seen = True
            self.last_client_at = time.monotonic()
        return {"ok": True}

    def windows_closed(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            # A throttled/background tab must not cancel an unattended job.
            working = self.worker and self.worker.is_alive() or self.batch.running()
            if not working:
                self.clients = {
                    key: seen for key, seen in self.clients.items() if now - seen < 90
                }
            return (
                self.auto_close
                and self.clients_seen
                and not self.clients
                and now - self.last_client_at > 4
            )

    def launch(self, phase, work):
        with self.lock:
            if self.batch.running():
                raise ValueError(
                    "Pause or finish the batch before starting an editor job."
                )
            if self.state["phase"] in (
                "preparing",
                "detecting",
                "processing",
                "optimizing",
                "pdf_export",
            ):
                raise ValueError("A job is already running. Wait for it to finish.")
            self.cancel_event.clear()
            self.state.update(
                stopping=False,
                phase=phase,
                error=None,
                progress=0,
                eta_seconds=None,
                stage=phase,
                started_at=time.time(),
                updated_at=time.time(),
                export_folder=None,
                frames_ready=False,
            )

        def runner():
            started = time.perf_counter()
            app_recorder = self.recorder
            self.record("job_started", phase=phase)
            try:
                with cancellable(self.cancel_event.is_set):
                    result = work()
                    check_cancel()
                with self.lock:
                    if phase in ("preparing", "detecting"):
                        self.state["session"] = result
                    elif phase == "optimizing":
                        self.state["session"].update(result)
                        self.state["session"]["revision"] = (
                            self.state["session"].get("revision", 0) + 1
                        )
                    elif phase == "pdf_export":
                        self.state["session"]["pdf_result"] = result
                        self.state["session"]["revision"] = (
                            self.state["session"].get("revision", 0) + 1
                        )
                    else:
                        self.state["session"]["result"] = result
                        self.state["session"]["revision"] = (
                            self.state["session"].get("revision", 0) + 1
                        )
                    self.state["phase"] = (
                        "setup"
                        if phase == "preparing"
                        else "review"
                        if phase in ("detecting", "optimizing")
                        else "done"
                    )
                self.write_review()
                if phase in ("pdf_export", "processing"):
                    self.batch.review_exported(self.state["session"])
                if app_recorder:
                    app_recorder.stage(phase, time.perf_counter() - started)
            except JobCancelled:
                with self.lock:
                    current = self.state.get("session")
                    self.state.update(
                        phase="setup"
                        if current and current.get("preflight")
                        else "review"
                        if current
                        else "idle",
                        error=None,
                        message="Stopped. Your review and completed exports are preserved.",
                        progress=0,
                        eta_seconds=None,
                        frames_ready=False,
                    )
                self.record("job_stopped", phase=phase)
            except Exception as exc:
                traceback.print_exc()
                self.record(
                    "job_error",
                    phase=phase,
                    error=str(exc),
                    seconds=round(time.perf_counter() - started, 3),
                )
                with self.lock:
                    current = self.state.get("session")
                    self.state.update(
                        phase="setup"
                        if current and current.get("preflight")
                        else "review"
                        if current
                        else "idle",
                        error=str(exc),
                        message="Could not finish. See the error below and retry.",
                    )

            finally:
                with self.lock:
                    self.state["stopping"] = False

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            route = urlparse(self.path).path
            if route not in ("/api/status", "/api/frame", "/video", "/api/window"):
                # Query strings contain ephemeral local session tokens; never log them.
                print(f"{self.command} {route}", flush=True)

        def authorized_host(self):
            return self.headers.get("Host") == urlparse(app.base_url).netloc

        def safety_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "frame-ancestors 'none'; base-uri 'none'; object-src 'none'",
            )

        def send_bytes(self, body, content_type="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Cache-Control",
                "private, max-age=3600" if content_type == "image/jpeg" else "no-store",
            )
            self.safety_headers()
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # A seek/reload may cancel an in-flight preview.

        def respond(self, value, status=200):
            self.send_bytes(
                json.dumps(value, ensure_ascii=False).encode(), status=status
            )

        def serve_file(self, path, ranged=False):
            path = Path(path)
            if not path.is_file():
                return self.respond({"error": "File not found."}, 404)
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range") if ranged else None
            if range_header:
                try:
                    unit, value = range_header.split("=", 1)
                    left, right = value.split("-", 1)
                    if unit != "bytes" or "," in value:
                        raise ValueError()
                    if not left:
                        start = max(0, size - int(right))
                    else:
                        start = int(left)
                        end = min(size - 1, int(right)) if right else size - 1
                    if start < 0 or start >= size or end < start:
                        raise ValueError()
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if range_header else 200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(str(path))[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(end - start + 1))
            self.safety_headers()
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(262144, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # Browsers cancel video ranges when seeking.

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self.authorized_host():
                return self.respond({"error": "Invalid host."}, 403)
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)
            if route == "/":
                return self.send_bytes(app.html().encode(), "text/html; charset=utf-8")
            if route in (
                "/editor.js", "/editor.css", "/selection.js", "/batch.js",
                "/export-location.js",
            ):
                return self.serve_file(ROOT / "web" / route[1:])
            if route == "/app-preview.png":
                return self.serve_file(ROOT / "web/assets/app-preview.png")
            if route == "/logo.png":
                return self.serve_file(ROOT / "web/assets/logo.png")
            if route == "/manual":
                return self.send_bytes(
                    (ROOT / "docs" / "MANUAL.md").read_bytes(),
                    "text/plain; charset=utf-8",
                )
            if route == "/favicon.ico":
                return self.send_bytes(b"", status=204)
            token = (
                self.headers.get("X-TransVision-Token") or query.get("token", [""])[0]
            )
            if not secrets.compare_digest(token, app.token):
                return self.respond(
                    {"error": "Local session expired. Reopen the terminal URL."}, 403
                )
            if route == "/api/batch":
                return self.respond(app.batch.snapshot())
            if route == "/api/export-settings":
                return self.respond(app.export_location.snapshot())
            if route == "/api/batch/file":
                try:
                    job = next(
                        j
                        for j in app.batch.snapshot()["jobs"]
                        if j["id"] == query.get("id", [""])[0]
                    )
                    index = int(query.get("index", ["0"])[0])
                    if not 0 <= index < len(job["outputs"]):
                        raise ValueError("Output not found.")
                    return self.serve_file(Path(job["outputs"][index]))
                except (StopIteration, ValueError):
                    return self.respond({"error": "Output not found."}, 404)
            if route == "/api/status":
                return self.respond(
                    app.snapshot(
                        compact=True,
                        known_folder=query.get("folder", [None])[0],
                        known_revision=query.get("revision", [None])[0],
                    )
                )
            session = app.media_session()
            if not session:
                return self.respond({"error": "Select a video first."}, 400)
            if route == "/video":
                return self.serve_file(session["video"], ranged=True)
            if route == "/api/frame":
                try:
                    if query.get("video") and query["video"][0] != session["folder"]:
                        return self.respond(
                            {"error": "This preview belongs to a previous video."}, 409
                        )
                    idx = max(
                        0,
                        min(
                            session["frame_count"] - 1,
                            int(query.get("frame", ["0"])[0]),
                        ),
                    )
                    w = min(1280, max(120, int(query.get("width", ["240"])[0])))
                    return self.send_bytes(
                        app.preview_image(session["video"], idx, w), "image/jpeg"
                    )
                except ValueError as exc:
                    return self.respond({"error": str(exc)}, 400)
            if route.startswith("/output/") and session.get("result"):
                folder = Path(session["result"]["folder"]).resolve()
                target = (folder / unquote(route[len("/output/") :])).resolve()
                if not target.is_relative_to(folder):
                    return self.respond({"error": "Invalid file path."}, 403)
                return self.serve_file(target)
            if route.startswith("/pdf/") and session.get("pdf_result"):
                folder = Path(session["pdf_result"]["folder"]).resolve()
                target = (folder / unquote(route[len("/pdf/") :])).resolve()
                if not target.is_relative_to(folder):
                    return self.respond({"error": "Invalid PDF path."}, 403)
                return self.serve_file(target)
            return self.respond({"error": "Not found."}, 404)

        def do_POST(self):
            if (
                not self.authorized_host()
                or self.headers.get("Origin") not in (None, app.base_url)
                or not secrets.compare_digest(
                    self.headers.get("X-TransVision-Token", ""), app.token
                )
            ):
                return self.respond({"error": "Invalid local session."}, 403)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("Invalid request size.")
                data = json.loads(self.rfile.read(length))
                route = urlparse(self.path).path
                if route == "/api/export-settings":
                    with app.lock:
                        result = app.export_location.save(
                            data.get("folder"), data.get("mode")
                        )
                    return self.respond(result)
                if route == "/api/export-folder/browse":
                    import tkinter as tk
                    from tkinter import filedialog

                    picker = tk.Tk()
                    picker.withdraw()
                    picker.attributes("-topmost", True)
                    try:
                        path = filedialog.askdirectory(
                            title="Choose an export folder",
                            initialdir=app.export_location.snapshot()["folder"],
                        )
                    finally:
                        picker.destroy()
                    return self.respond({"path": path})
                if route == "/api/stop":
                    return self.respond(app.stop())
                if route == "/api/reset":
                    return self.respond(app.reset())
                if route == "/api/window":
                    return self.respond(
                        app.window_ping(data.get("client"), data.get("closed", False))
                    )
                if route == "/api/batch/stop":
                    app.batch.stop()
                    return self.respond({"ok": True})
                if route == "/api/suggest-crop":
                    return self.respond(
                        app.suggest_crop(data.get("frame_idx"), data.get("folder"))
                    )
                if route == "/api/batch/enqueue":
                    return self.respond(
                        app.batch.enqueue(data.get("videos"), data.get("settings", {}))
                    )
                if route == "/api/batch/start":
                    with app.lock:
                        if app.state["phase"] in (
                            "preparing",
                            "detecting",
                            "processing",
                            "optimizing",
                            "pdf_export",
                        ):
                            raise ValueError(
                                "Wait for the current editor job to finish."
                            )
                        destination = app.export_location.resolve(data.get("export_root"))
                        app.batch.set_output_root(destination)
                        app.batch.start()
                    return self.respond({"ok": True})
                if route == "/api/batch/pause":
                    app.batch.pause()
                    return self.respond({"ok": True})
                if route == "/api/batch/retry":
                    app.batch.retry()
                    return self.respond({"ok": True})
                if route == "/api/batch/review":
                    if app.batch.running() or app.state["phase"] in (
                        "preparing",
                        "detecting",
                        "processing",
                        "optimizing",
                        "pdf_export",
                    ):
                        raise ValueError(
                            "Pause or finish processing before opening a batch review."
                        )
                    job = next(
                        (
                            j
                            for j in app.batch.snapshot()["jobs"]
                            if j["id"] == data.get("id")
                        ),
                        None,
                    )
                    if not job or not job.get("session_path"):
                        raise ValueError("No saved review for this file yet.")
                    restored = json.loads(
                        Path(job["session_path"]).read_text(encoding="utf-8")
                    )
                    with app.lock:
                        app.state.update(
                            session=restored,
                            phase="review",
                            error=None,
                            progress=100,
                            message="Batch review opened.",
                        )
                    return self.respond({"ok": True})
                if route == "/api/batch/open-folder":
                    state = app.batch.snapshot()
                    target = Path(state.get("output_root", app.out_dir))
                    target.mkdir(parents=True, exist_ok=True)
                    os.startfile(str(target))
                    return self.respond({"ok": True})
                if route == "/api/browse":
                    import tkinter as tk
                    from tkinter import filedialog

                    picker = tk.Tk()
                    picker.withdraw()
                    picker.attributes("-topmost", True)
                    try:
                        path = filedialog.askopenfilename(
                            title="Choose a video",
                            filetypes=[
                                ("Video", "*.mp4 *.mkv *.mov *.avi *.webm"),
                                ("All files", "*.*"),
                            ],
                        )
                    finally:
                        picker.destroy()
                    return self.respond({"path": path})
                if route == "/api/prepare":
                    app.launch("preparing", lambda: app.prepare(data.get("video", "")))
                    return self.respond({"ok": True})
                if route == "/api/setup":
                    with app.lock:
                        prepared = app.state["session"]
                        if (
                            app.state["phase"] != "setup"
                            or not prepared
                            or not prepared.get("preflight")
                        ):
                            raise ValueError("Choose a video for setup first.")
                        roi = validate_roi(
                            data.get("content_roi", prepared["content_roi"]),
                            prepared["width"],
                            prepared["height"],
                        )
                        first, stop = frame_range(
                            prepared,
                            data.get("start_s", prepared["trim_start_s"]),
                            data.get("end_s", prepared["trim_end_s"]),
                        )
                        settings = {
                            key: data[key]
                            for key in (
                                "ocr_engine",
                                "ocr_profile",
                                "dedupe_ocr",
                                "export_cropped",
                                "ocr_second_pass",
                                "sample_seconds",
                                "hide_camera",
                            )
                            if key in data
                        }
                        if (
                            not 0.125
                            <= float(
                                settings.get(
                                    "sample_seconds",
                                    prepared.get("sample_seconds", 0.5),
                                )
                            )
                            <= 2
                        ):
                            raise ValueError(
                                "Scan interval must be 0.125 to 2 seconds."
                            )
                        if (
                            settings.get("ocr_engine", prepared.get("ocr_engine"))
                            not in available_engines()
                        ):
                            raise ValueError("Choose an installed OCR engine.")
                        if settings.get(
                            "ocr_profile", prepared.get("ocr_profile", "auto")
                        ) not in ("auto", "prose", "code", "math"):
                            raise ValueError("Choose an OCR content profile.")
                        prepared.update(
                            content_roi=roi,
                            trim_start_s=first / prepared["fps"],
                            trim_end_s=stop / prepared["fps"],
                            revision=prepared["revision"] + 1,
                            **settings,
                        )
                        app.record(
                            "setup_changed",
                            content_roi=roi,
                            start_frame=first,
                            end_frame_exclusive=stop,
                        )
                    return self.respond(
                        {
                            "ok": True,
                            "revision": prepared["revision"],
                            "trim_start_s": prepared["trim_start_s"],
                            "trim_end_s": prepared["trim_end_s"],
                        }
                    )
                if route == "/api/start":
                    if app.state["phase"] in (
                        "preparing",
                        "detecting",
                        "processing",
                        "optimizing",
                        "pdf_export",
                    ):
                        raise ValueError("A job is already running.")
                    prepared = copy.deepcopy(app.state["session"])
                    if not prepared or not prepared.get("preflight"):
                        raise ValueError(
                            "Preview the video and confirm its crop and trim first."
                        )

                    def detection():
                        result = detect(
                            prepared["video"],
                            app.out_dir,
                            app.report,
                            app.recorder,
                            content_roi=prepared["content_roi"],
                            start_s=prepared["trim_start_s"],
                            end_s=prepared["trim_end_s"],
                            suggested_roi=prepared["suggested_roi"],
                            adaptive=True,
                            sample_seconds=float(prepared.get("sample_seconds", 0.5)),
                            hide_camera=prepared.get("hide_camera", True),
                        )
                        for key in (
                            "ocr_engine",
                            "ocr_profile",
                            "dedupe_ocr",
                            "export_cropped",
                            "ocr_second_pass",
                        ):
                            if key in prepared:
                                result[key] = prepared[key]
                        return result

                    app.launch("detecting", detection)
                    return self.respond({"ok": True})
                if route == "/api/test-note":
                    note = str(data.get("note", "")).strip()[:4000]
                    if not note:
                        raise ValueError("Enter a test observation.")
                    app.record(
                        "user_observation",
                        note=note,
                        frame_idx=data.get("frame_idx"),
                        timestamp_s=data.get("timestamp_s"),
                    )
                    if app.recorder:
                        app.recorder.write_summary()
                    return self.respond({"ok": True})
                if route == "/api/ui-events":
                    events = data.get("events", [])
                    if not isinstance(events, list) or len(events) > 100:
                        raise ValueError("Invalid events.")
                    for event in events:
                        if isinstance(event, dict):
                            app.record(
                                "ui_event",
                                details={
                                    k: v
                                    for k, v in event.items()
                                    if k
                                    in (
                                        "action",
                                        "frame_idx",
                                        "timestamp_s",
                                        "video_name",
                                        "seconds",
                                        "message",
                                        "zoom",
                                        "rows",
                                    )
                                },
                            )
                    return self.respond({"ok": True})
                if route == "/api/open-test":
                    if not app.recorder:
                        raise ValueError("No test recording yet. Select a video first.")
                    os.startfile(str(app.recorder.folder))
                    return self.respond({"ok": True})
                if route in (
                    "/api/save",
                    "/api/finalize",
                    "/api/optimize",
                    "/api/final-frames",
                    "/api/pdf",
                    "/api/detail",
                ):
                    with app.lock:
                        if app.state["phase"] in (
                            "preparing",
                            "detecting",
                            "processing",
                            "optimizing",
                            "pdf_export",
                        ):
                            raise ValueError("Wait for the current job to finish.")
                        session = app.state["session"]
                        if not session:
                            raise ValueError("Choose a video first.")
                        if session.get("preflight"):
                            raise ValueError("Finish setup and detect frames first.")
                        markers = validate_markers(data.get("markers"), session)
                        roi = validate_roi(
                            data.get("content_roi", session.get("content_roi")),
                            session["width"],
                            session["height"],
                        )
                        engine = data.get(
                            "ocr_engine",
                            session.get("ocr_engine", available_engines()[0]),
                        )
                        if engine not in available_engines():
                            raise ValueError("Choose an installed OCR engine.")
                        profile = data.get(
                            "ocr_profile", session.get("ocr_profile", "auto")
                        )
                        if profile not in ("auto", "prose", "code", "math"):
                            raise ValueError("Choose an OCR content profile.")
                        sensitivity = validate_sensitivity(
                            data.get("sensitivity", session_sensitivity(session))
                        )
                        motion_mode = validate_motion_mode(
                            data.get("motion_mode", session.get("motion_mode", "summary"))
                        )
                        if route in ("/api/finalize", "/api/pdf") and not any(
                            m["included"] for m in markers
                        ):
                            raise ValueError("Include at least one frame before Run.")
                        old = {m["frame_idx"]: m for m in session["markers"]}
                        new = {m["frame_idx"]: m for m in markers}
                        for idx in old.keys() | new.keys():
                            fields = (
                                "frame_idx",
                                "kind",
                                "included",
                                "ai",
                                "user_override",
                            )
                            before = (
                                {
                                    k: (
                                        bool(old[idx].get(k))
                                        if k == "user_override"
                                        else old[idx].get(k)
                                    )
                                    for k in fields
                                }
                                if idx in old
                                else None
                            )
                            after = (
                                {
                                    k: (
                                        bool(new[idx].get(k))
                                        if k == "user_override"
                                        else new[idx].get(k)
                                    )
                                    for k in fields
                                }
                                if idx in new
                                else None
                            )
                            if before != after:
                                app.record(
                                    "review_edit",
                                    frame_idx=idx,
                                    timestamp_s=idx / session["fps"],
                                    before=before,
                                    after=after,
                                    source_route=route,
                                )
                        session["markers"] = markers
                        session.update(
                            content_roi=roi,
                            ocr_engine=engine,
                            dedupe_ocr=bool(
                                data.get("dedupe_ocr", session.get("dedupe_ocr", True))
                            ),
                            export_cropped=bool(
                                data.get(
                                    "export_cropped",
                                    session.get("export_cropped", False),
                                )
                            ),
                            ocr_second_pass=bool(
                                data.get(
                                    "ocr_second_pass",
                                    session.get("ocr_second_pass", False),
                                )
                            ),
                        )
                        session["pdf_timestamps"] = bool(
                            data.get(
                                "pdf_timestamps", session.get("pdf_timestamps", True)
                            )
                        )
                        session["ocr_profile"] = profile
                        session.update(
                            sensitivity=sensitivity,
                            motion_mode=motion_mode,
                            hide_camera=bool(
                                data.get(
                                    "hide_camera", session.get("hide_camera", True)
                                )
                            ),
                        )
                        session.pop("detail_level", None)
                        session.pop("detail_sensitivity", None)
                        app.write_review()
                        snapshot = copy.deepcopy(session)
                        if route in ("/api/finalize", "/api/pdf"):
                            snapshot["export_root"] = str(
                                app.export_location.resolve(data.get("export_root"))
                            )
                        if route == "/api/detail":

                            def detail():
                                analyzed = markers
                                if snapshot.get("motion_analysis_version") != MOTION_VERSION or snapshot.get("selection_analysis_roi") != roi or any(
                                    "visual_analysis" not in m
                                    and not (
                                        m.get("user_override")
                                        or m.get("kind") == "manual"
                                        or m.get("ai")
                                    )
                                    for m in markers
                                ):
                                    same_cache = (
                                        snapshot.get("candidate_cache_roi") == roi
                                    )
                                    paths = (
                                        {
                                            int(k): v
                                            for k, v in snapshot.get(
                                                "candidate_paths", {}
                                            ).items()
                                        }
                                        if same_cache
                                        else None
                                    )
                                    overrides = (
                                        {
                                            int(k): v
                                            for k, v in snapshot.get(
                                                "candidate_camera", {}
                                            ).items()
                                        }
                                        if same_cache
                                        else None
                                    )
                                    analyzed = analyze_selection(
                                        snapshot,
                                        markers,
                                        roi,
                                        app.report,
                                        app.recorder,
                                        analysis_paths=paths,
                                        analysis_cropped=bool(
                                            same_cache
                                            and snapshot.get("candidate_cache_cropped")
                                        ),
                                        camera_overrides=overrides,
                                    )
                                updated, stats = apply_sensitivity(
                                    analyzed,
                                    sensitivity,
                                    snapshot["hide_camera"],
                                    snapshot["motion_mode"],
                                )
                                app.record("sensitivity_changed", **stats)
                                app.report(
                                    f"{stats['included']} images / PDF pages selected. Review before export.",
                                    progress=100,
                                )
                                return {
                                    "markers": updated,
                                    "optimization": stats,
                                    "selection_analysis_roi": roi,
                                    "analysis_pending": False,
                                    "motion_analysis_version": MOTION_VERSION,
                                }

                            app.launch("optimizing", detail)
                        elif route == "/api/finalize":
                            app.launch(
                                "processing",
                                lambda: finalize(
                                    snapshot, markers, app.report, app.recorder
                                ),
                            )
                        elif route == "/api/pdf":
                            app.launch(
                                "pdf_export",
                                lambda: export_pdf_parts(
                                    snapshot, markers, app.report, app.recorder
                                ),
                            )
                        elif route == "/api/final-frames":

                            def boundaries():
                                path = (
                                    Path(snapshot["folder"]) / "detected_markers.json"
                                )
                                cues = (
                                    json.loads(path.read_text(encoding="utf-8"))
                                    if path.is_file()
                                    else markers
                                )
                                updated = final_frames(
                                    snapshot, markers, cues, app.report, app.recorder
                                )
                                app.report(
                                    f"Added {len(updated) - len(markers)} outgoing frames. Review before exporting.",
                                    progress=100,
                                )
                                return {"markers": updated}

                            app.launch("optimizing", boundaries)
                        elif route == "/api/optimize":

                            def optimize():
                                updated, stats = optimize_markers(
                                    snapshot, markers, roi, app.report
                                )
                                app.report(
                                    f"{stats['excluded_similar']} similar automatic marks excluded; {stats['included']} included.",
                                    progress=100,
                                )
                                return {"markers": updated, "optimization": stats}

                            app.launch("optimizing", optimize)
                    return self.respond({"ok": True})
                if route == "/api/open-output":
                    session = app.snapshot()["session"]
                    current = app.snapshot()
                    target = Path(
                        current["export_folder"]
                        if current.get("frames_ready") and current.get("export_folder")
                        else session["result"]["folder"]
                        if session and session.get("result")
                        else session["folder"]
                        if session
                        else app.out_dir
                    )
                    os.startfile(str(target))
                    return self.respond({"ok": True})
                return self.respond({"error": "Not found."}, 404)
            except (
                ValueError,
                TypeError,
                KeyError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                return self.respond({"error": str(exc)}, 400)

    return Handler


def main():
    # Windows redirected consoles may default to cp1252; baseline logs contain Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="TransVision local visual transcript editor"
    )
    parser.add_argument(
        "--video", help="Preview this video for crop and trim setup before detection"
    )
    parser.add_argument(
        "--session", help="Resume an output/<video>/session.json review"
    )
    parser.add_argument("--out-dir", default=str(ROOT / "output"))
    parser.add_argument(
        "--port", type=int, default=0, help="Default: choose a free local port"
    )
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--app-window",
        action="store_true",
        help="Open a clean browser app window without tabs",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep server running after its tabs close",
    )
    parser.add_argument(
        "--close-with-tabs",
        action="store_true",
        help="Enable tab-close shutdown when using --no-open",
    )
    args = parser.parse_args()
    app = Application(args.out_dir, args.session)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(app))
    app.base_url = f"http://127.0.0.1:{server.server_port}"
    app.auto_close = not args.keep_alive and (not args.no_open or args.close_with_tabs)
    if args.session:
        app.write_review()
    print(
        f"TransVision editor: {app.base_url}\nKeep this terminal open. Ctrl+C stops TransVision.\nOutput: {app.out_dir}",
        flush=True,
    )
    if args.video:
        app.prepare(args.video)
    if not args.no_open:
        browsers = [
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Google/Chrome/Application/chrome.exe",
        ]
        executable = next((p for p in browsers if p.is_file()), None)
        if args.app_window and executable:
            subprocess.Popen(
                [str(executable), "--app=" + app.base_url],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            webbrowser.open(app.base_url)

    def watch_windows():
        while True:
            time.sleep(1)
            if app.windows_closed():
                print("TransVision window closed. Stopping local workers…", flush=True)
                app.stop()
                app.batch.stop()
                if app.worker:
                    app.worker.join()
                if app.batch.thread:
                    app.batch.thread.join()
                server.shutdown()
                return

    threading.Thread(target=watch_windows, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(
            "\nTransVision stopped. Saved reviews and completed exports remain on disk."
        )
    finally:
        if app.recorder:
            app.recorder.close(
                "interrupted"
                if app.state["phase"]
                in ("preparing", "detecting", "processing", "optimizing", "pdf_export")
                else "closed"
            )
        server.server_close()
        with app.preview_lock:
            if app.preview_capture:
                app.preview_capture.release()


if __name__ == "__main__":
    main()
