"""Lossless presentation pages, with optional selectable source-time captions."""

import io
import time
from pathlib import Path

import cv2
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.cancellation import JobCancelled, check_cancel
from src.content import crop_image, validate_roi


def export_pdf_parts(session, markers, report, recorder=None):
    """Interactive export: bounded video-named parts, with the master preserved."""
    from src.pdf_parts import split_pdf

    result = export_pdf(session, markers, report, recorder)
    original = result["filename"]
    try:
        check_cancel()
        parts = split_pdf(
            Path(result["folder"]) / original,
            result["folder"],
            Path(session["video"]).stem,
            session.get("max_pdf_pages", 100),
        )
        check_cancel()
    except JobCancelled:
        (Path(result["folder"]) / "INCOMPLETE.txt").write_text(
            "Stopped while splitting PDF parts. Retry export.\n", encoding="utf-8"
        )
        raise
    return {
        **result,
        "master_filename": original,
        "filename": Path(parts[0]["file"]).name,
        "parts": parts,
    }


def export_pdf(session, markers, report, recorder=None):
    from src.pipeline import (
        frame_filename,
        save_json,
        time_label,
        unique_folder,
        validate_markers,
    )

    selected = [m for m in validate_markers(markers, session) if m["included"]]
    if not selected:
        raise ValueError("Include at least one frame before exporting PDF.")
    folder = unique_folder(
        Path(session["folder"]) / "exports", Path(session["video"]).stem + "_pdf"
    )
    path = folder / "presentation_frames.pdf"
    temporary = folder / "presentation_frames.partial.pdf"
    roi = validate_roi(session.get("content_roi"), session["width"], session["height"])
    timestamps = session.get("pdf_timestamps", True)
    document = canvas.Canvas(str(temporary), pageCompression=1)
    name = session.get("video_name", Path(session["video"]).name)
    document.setTitle(name + " - presentation frames")
    document.setAuthor("TransVision")
    document.setSubject(
        "Selected presentation frames from "
        + name
        + ". Match video timestamps to the audio transcript."
        if timestamps
        else "Selected presentation frames from " + name
    )
    cap = cv2.VideoCapture(session["video"])
    pages = []
    started = time.perf_counter()
    if not cap.isOpened():
        raise ValueError("Cannot open source video.")
    try:
        for i, mark in enumerate(selected):
            check_cancel()
            tick = time.perf_counter()
            index = mark["frame_idx"]
            ts = index / session["fps"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1 != index:
                raise ValueError(f"Cannot read exact frame {index}.")
            image = crop_image(image, roi)
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("Cannot encode PDF frame.")
            if mark["ai"]:
                ai_folder = folder / "AI-marked"
                ai_folder.mkdir(exist_ok=True)
                (ai_folder / frame_filename(ts, True)).write_bytes(encoded.tobytes())
            w, h = image.shape[1] * 0.75, image.shape[0] * 0.75
            header = 34 if timestamps else 0
            document.setPageSize((w + 24, h + 24 + header))
            document.drawImage(
                ImageReader(io.BytesIO(encoded.tobytes())), 12, 12, width=w, height=h
            )
            if timestamps:
                document.setFont("Helvetica", 9)
                caption = "Source video: " + name
                while document.stringWidth(caption) > w and len(caption) > 20:
                    check_cancel()
                    caption = caption[:-4] + "..."
                document.drawString(12, h + header + 13, caption)
                document.setFont("Helvetica-Bold", 10)
                document.drawString(
                    12,
                    h + 25,
                    f"Video timestamp: {time_label(ts)}   |   Page {i + 1} of {len(selected)}"
                    + ("   |   AI" if mark["ai"] else ""),
                )
            document.bookmarkPage(f"frame-{index}")
            document.addOutlineEntry(
                time_label(ts) if timestamps else f"Page {i + 1}",
                f"frame-{index}",
                level=0,
            )
            document.showPage()
            page = {
                "page": i + 1,
                "frame_idx": index,
                "timestamp": time_label(ts),
                "timestamp_s": ts,
                "content_roi": roi,
                "ai": mark["ai"],
            }
            pages.append(page)
            if recorder:
                recorder.event(
                    "pdf_page_exported",
                    **page,
                    seconds=round(time.perf_counter() - tick, 4),
                )
            elapsed = time.perf_counter() - started
            report(
                f"Creating cropped PDF page {i + 1}/{len(selected)}",
                progress=round(95 * (i + 1) / len(selected)),
                stage="pdf",
                eta_seconds=elapsed / (i + 1) * (len(selected) - i - 1),
                export_folder=str(folder),
                frames_ready=False,
            )
        check_cancel()
        document.save()
        check_cancel()
        temporary.replace(path)
        save_json(
            folder / "pages.json",
            {
                "video_name": name,
                "fps": session["fps"],
                "timestamps_on_pages": timestamps,
                "pages": pages,
            },
        )
        elapsed = round(time.perf_counter() - started, 3)
        if recorder:
            recorder.stage(
                "cropped_pdf_export",
                elapsed,
                pages=len(pages),
                timestamps_on_pages=timestamps,
                folder=str(folder),
            )
        report(
            f"PDF ready: {len(pages)} cropped pages.",
            progress=100,
            stage="done",
            eta_seconds=0,
            export_folder=str(folder),
            frames_ready=True,
        )
        return {
            "folder": str(folder),
            "filename": path.name,
            "pages": len(pages),
            "timestamps": timestamps,
            "elapsed": elapsed,
        }
    except Exception as exc:
        (folder / "INCOMPLETE.txt").write_text(str(exc), encoding="utf-8")
        raise
    finally:
        cap.release()
