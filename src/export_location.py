"""Private, persistent export preferences shared by the editor and batch UI."""

import json
import tempfile
from pathlib import Path

from src.pipeline import save_json


def writable_folder(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Choose an export folder.")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError("Use a full export-folder path.")
    path = path.resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
    except OSError as exc:
        raise ValueError("Cannot write to this export folder. Choose another folder.") from exc
    return path


class ExportLocation:
    def __init__(self, file, default):
        self.file = Path(file)
        self.default = str(Path(default).resolve())

    def snapshot(self):
        if self.file.is_file():
            value = json.loads(self.file.read_text(encoding="utf-8"))
            return {"folder": value["folder"], "mode": value["mode"], "configured": True}
        return {"folder": self.default, "mode": "saved", "configured": False}

    def save(self, folder, mode):
        if mode not in ("saved", "ask"):
            raise ValueError("Choose Remember folder or Ask each time.")
        path = writable_folder(folder)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.file, {"folder": str(path), "mode": mode})
        return self.snapshot()

    def resolve(self, requested=None):
        state = self.snapshot()
        if requested:
            return writable_folder(requested)
        if not state["configured"] or state["mode"] == "ask":
            raise ValueError("Choose an export folder before exporting.")
        return writable_folder(state["folder"])
