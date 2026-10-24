"""Durable append-only evidence and atomic update-boundary manifests."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".commit-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Recorder:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "episodes.jsonl"
        self.lock = threading.Lock()

    def append(self, kind: str, value: Any) -> None:
        line = json.dumps({"kind": kind, "data": value}, sort_keys=True, allow_nan=False)
        with self.lock, self.path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
