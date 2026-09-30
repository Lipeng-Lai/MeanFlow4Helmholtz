"""Atomic numbered checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
import re
import torch


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def latest_checkpoint(directory: str | Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(r"model_(?:epoch|step)_(\d+)\.pth")
    for path in Path(directory).glob("model_*.pth"):
        match = pattern.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]
