"""Lightweight CPU OCR with a local Tesseract fallback. No cloud processing."""

import importlib.util
import threading
import re
import time
import cv2
import numpy as np
import pytesseract

_rapid = None
_symbols = None
_lock = threading.Lock()
_local = threading.local()


def available_engines():
    return (
        ["rapidocr", "tesseract"]
        if importlib.util.find_spec("rapidocr")
        and importlib.util.find_spec("onnxruntime")
        else ["tesseract"]
    )


def tesseract_lines(image, psm=11, code=False):
    """One TSV pass, retaining spatial lines and code indentation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(
        gray,
        lang="eng",
        config=f"--psm {psm} -c preserve_interword_spaces=1",
        output_type=pytesseract.Output.DICT,
        timeout=120,
    )
    groups = {}
    for i, word in enumerate(data["text"]):
        if word.strip():
            groups.setdefault(
                (data["block_num"][i], data["par_num"][i], data["line_num"][i]), []
            ).append(i)
    widths = [
        data["width"][i] / max(1, len(data["text"][i]))
        for ids in groups.values()
        for i in ids
    ]
    char_width = float(np.median(widths)) if widths else 10
    base = min((data["left"][ids[0]] for ids in groups.values()), default=0)
    result = []
    for ids in groups.values():
        ids.sort(key=lambda i: data["left"][i])
        left = min(data["left"][i] for i in ids)
        top = min(data["top"][i] for i in ids)
        right = max(data["left"][i] + data["width"][i] for i in ids)
        bottom = max(data["top"][i] + data["height"][i] for i in ids)
        text = " " * max(0, round((left - base) / char_width)) if code else ""
        previous = None
        for i in ids:
            if previous is not None:
                gap = data["left"][i] - (
                    data["left"][previous] + data["width"][previous]
                )
                text += " " * max(1, round(gap / char_width)) if code else " "
            text += data["text"][i]
            previous = i
        scores = [float(data["conf"][i]) for i in ids if float(data["conf"][i]) >= 0]
        result.append(
            {
                "text": text,
                "score": float(np.mean(scores)) / 100 if scores else 0,
                "box": [
                    [left / scale, top / scale],
                    [right / scale, top / scale],
                    [right / scale, bottom / scale],
                    [left / scale, bottom / scale],
                ],
                "recognizer": "tesseract-eng",
            }
        )
    if code:
        result.sort(key=lambda row: row["box"][0][1])
        indents = [len(row["text"]) - len(row["text"].lstrip()) for row in result]
        quantum = min((value for value in indents if value >= 2), default=1)
        for row, indent in zip(result, indents):
            snapped = round(indent / quantum) * quantum
            if abs(snapped - indent) <= 1:
                row["text"] = " " * snapped + row["text"].lstrip()
    return result


def read_image(image, engine="rapidocr", profile="auto", metrics=None):
    if profile == "code":
        engine = "tesseract"
    if engine == "rapidocr":
        global _rapid
        from rapidocr import RapidOCR, OCRVersion, LangRec, ModelType
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        cuda = "CUDAExecutionProvider" in providers
        dml = not cuda and "DmlExecutionProvider" in providers
        with _lock:
            _rapid = getattr(_local, "rapid", None)
            if _rapid is None:
                _rapid = RapidOCR(
                    params={
                        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                        "Det.ocr_version": OCRVersion.PPOCRV4,
                        "Det.model_type": ModelType.MOBILE,
                        "Det.limit_type": "max",
                        "Det.limit_side_len": 1280,
                        "Global.use_cls": False,
                        "Rec.ocr_version": OCRVersion.PPOCRV5,
                        "Rec.lang_type": LangRec.EN,
                        "Rec.model_type": ModelType.MOBILE,
                        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                        "EngineConfig.onnxruntime.use_cuda": cuda,
                        "EngineConfig.onnxruntime.use_dml": dml,
                        "Global.log_level": "warning",
                    }
                )
                _local.rapid = _rapid
            rapid = _rapid
        tick = time.perf_counter()
        result = rapid(image)
        if metrics is not None:
            metrics.update(
                primary_seconds=round(time.perf_counter() - tick, 4),
                requested_backend="CUDA" if cuda else "DirectML" if dml else "CPU",
                detector="PP-OCRv4 mobile / max1280",
                orientation="upright",
            )
        if result.txts is None or len(result.txts) == 0 or result.scores is None:
            return "", 0, []
        # Preserve punctuation and Unicode symbols rather than normalizing them away.
        # RapidOCR detects individual lines, including much of the handwritten content.
        boxes = result.boxes.tolist()
        lines = [
            {"text": str(text), "score": float(score), "box": box}
            for text, score, box in zip(result.txts, result.scores, boxes)
        ]
        math_like = bool(
            re.search(
                r"\b(?:Theorem|Norms|Vectors)\b|[√∥≥≤]|·|,,|=\s*\.",
                "\n".join(r["text"] for r in lines),
            )
        )
        if profile == "math" or profile == "auto" and math_like:
            global _symbols
            # A cached second recognizer sees the same line crops, without running
            # another detector. It preserves more math operators than English OCR.
            with _lock:
                _symbols = getattr(_local, "symbols", None)
                if _symbols is None:
                    _symbols = RapidOCR(
                        params={
                            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                            "EngineConfig.onnxruntime.use_cuda": cuda,
                            "EngineConfig.onnxruntime.use_dml": dml,
                            "Global.log_level": "warning",
                        }
                    )
                    _local.symbols = _symbols
                symbol_model = _symbols
            tick = time.perf_counter()
            symbol_result = symbol_model.recognize_txt(
                symbol_model.crop_text_regions(
                    image, np.asarray(result.boxes, dtype=np.float32)
                )
            )
            if metrics is not None:
                metrics["symbol_seconds"] = round(time.perf_counter() - tick, 4)
            for row, candidate, score in zip(
                lines, symbol_result.txts, symbol_result.scores
            ):
                row["raw_text"] = row["text"]
                row["symbol_text"] = candidate
                row["symbol_score"] = float(score)
                symbol_like = bool(re.search(r"[√∥≥≤∈ⁿ·×\u4e00-\u9fff]", candidate))
                missing = bool(re.search(r"=\s*(?:\.|$)", row["text"]))
                if float(score) > 0.65 and (symbol_like or missing):
                    row["text"] = re.sub(
                        r"[\u4e00-\u9fff]+", "[unclear symbol]", candidate
                    )
                    row["symbol_used"] = symbol_like
            try:
                tick = time.perf_counter()
                candidates = tesseract_lines(image, 11)
                # Enclosed theorem/example panels read better as a text block than
                # as sparse rows mixed with neighboring handwriting.
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                mask = (gray < 135).astype(np.uint8) * 255
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                )
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                panels = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    if (
                        w > 120
                        and h > 65
                        and 0.02 * gray.size < area < 0.65 * gray.size
                        and area > 0.75 * w * h
                    ):
                        panels.append((x, y, w, h))
                for x, y, w, h in sorted(
                    panels, key=lambda p: p[2] * p[3], reverse=True
                )[:2]:
                    for row in tesseract_lines(image[y : y + h, x : x + w], 6):
                        row["box"] = [[px + x, py + y] for px, py in row["box"]]
                        row["panel_pass"] = True
                        candidates.append(row)
                for row in lines:
                    points = np.asarray(row["box"])
                    x, y = points.min(axis=0)
                    right, bottom = points.max(axis=0)
                    matches = []
                    for candidate in candidates:
                        box = np.asarray(candidate["box"])
                        cx, cy = box.min(axis=0)
                        cr, cb = box.max(axis=0)
                        overlap_y = max(0, min(bottom, cb) - max(y, cy)) / max(
                            1, min(bottom - y, cb - cy)
                        )
                        overlap_x = max(0, min(right, cr) - max(x, cx)) / max(
                            1, min(right - x, cr - cx)
                        )
                        coverage = max(0, min(right, cr) - max(x, cx)) / max(
                            1, max(right - x, cr - cx)
                        )
                        if (
                            overlap_y > 0.55
                            and overlap_x > 0.65
                            and coverage > 0.4
                            and candidate["score"] > 0.35
                        ):
                            matches.append(candidate)
                    if matches:
                        best = max(
                            matches,
                            key=lambda r: (
                                bool(r.get("panel_pass")),
                                len(re.findall(r"[A-Za-z0-9]", r["text"])),
                            ),
                        )
                        raw = row["text"]
                        row.setdefault("raw_text", raw)
                        row["alternative_text"] = best["text"]
                        if not row.get("symbol_used") and len(
                            re.findall(r"[A-Za-z0-9]", best["text"])
                        ) > len(re.findall(r"[A-Za-z0-9]", raw)):
                            row["text"] = best["text"]
                            row["secondary_used"] = True
                if metrics is not None:
                    metrics["tesseract_recovery_seconds"] = round(
                        time.perf_counter() - tick, 4
                    )
            except (
                pytesseract.TesseractError,
                pytesseract.TesseractNotFoundError,
                RuntimeError,
            ):
                pass
        return (
            "\n".join(row["text"] for row in lines),
            round(float(np.mean(result.scores)) * 100, 1),
            lines,
        )
    lines = tesseract_lines(image, 6 if profile == "code" else 11, profile == "code")
    return (
        "\n".join(row["text"] for row in lines),
        round(float(np.mean([r["score"] for r in lines])) * 100, 1) if lines else 0,
        lines,
    )


def readable_text(text, lines, image, second_pass=False, profile="auto"):
    """Recover omitted English tokens without pretending OCR is reliable mathematics."""
    output = []
    uncertain = 0
    for line in lines or [{"text": value, "score": 1} for value in text.splitlines()]:
        value = line["text"]
        box = line.get("box")
        # Repeated slide footers/page numbers distract from lecture content.
        if (
            box
            and min(p[1] for p in box) > image.shape[0] * 0.91
            and (re.search(r"©|\b\d+\s*/\s*\d+\b", value))
        ):
            continue
        if profile == "code":
            output.append(value.rstrip())
            uncertain += line.get("score", 1) < 0.65
            continue
        # Broken dot products, missing vector variables and non-English accents
        # are retained in raw OCR JSON; the readable transcript labels uncertainty.
        broken = (
            bool(re.search(r"[\u4e00-\u9fff]|,,|=\s*(?:\.|$)", value))
            or line.get("score", 1) < 0.75
        )
        suspect = (
            bool(re.search(r"[√∥≥≤²³ⁿ]|·|\bTheorem\b|\[unclear symbol\]", value))
            or broken
            or line.get("secondary_used", False)
            or line.get("symbol_used", False)
            or profile == "math"
        )
        if suspect:
            box = line.get("box")
            if box and second_pass:
                points = np.asarray(box)
                x, y = points.min(axis=0).astype(int)
                right, bottom = points.max(axis=0).astype(int)
                patch = image[
                    max(0, y - 5) : min(image.shape[0], bottom + 5),
                    max(0, x - 5) : min(image.shape[1], right + 5),
                ]
                try:
                    candidate = pytesseract.image_to_string(
                        cv2.resize(patch, None, fx=2, fy=2),
                        lang="eng",
                        config="--psm 7",
                        timeout=20,
                    ).strip()
                    if candidate and len(re.findall(r"[A-Za-z0-9]", candidate)) > len(
                        re.findall(r"[A-Za-z0-9]", value)
                    ):
                        value = candidate
                except (
                    pytesseract.TesseractError,
                    pytesseract.TesseractNotFoundError,
                    RuntimeError,
                ):
                    pass
            # Broken recognition is not useful lecture text. Preserve its full raw
            # evidence in selected_frames.json instead of filling the transcript with it.
            # Keep useful partial prose/tokens; do not repeat warnings after every row.
            value = re.sub(r"[\u4e00-\u9fff]+", "[unclear symbol]", value)
            if line.get("score", 1) < 0.35 and len(re.findall(r"[A-Za-z]", value)) < 3:
                value = "[Unreadable visual text]"
            uncertain += 1
        if value == "[Unreadable visual text]" and output and output[-1] == value:
            continue
        output.append(value)
    return "\n".join(output), uncertain
