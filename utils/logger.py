"""Minimal timestamped terminal and file logger."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(self, path: str | Path) -> None:
        self.stream = Path(path).open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")

    def close(self) -> None:
        self.stream.close()
