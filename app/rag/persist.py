"""Tiny JSON-file persistence for state shared between processes.

Most stores in this project are in-memory by design (offline-first). A few must
be visible to more than one process — e.g. API keys created in the web UI must
be usable by the separate MCP server process — so they are mirrored to a JSON
file under ``settings.data_dir``. Writes are atomic (temp file + rename).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.config import settings


def data_file(name: str) -> Path:
    return Path(settings.data_dir) / name


def load_json(name: str, default: Any) -> Any:
    path = data_file(name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(name: str, data: Any) -> None:
    path = data_file(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
