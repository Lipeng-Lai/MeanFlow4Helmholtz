"""Configuration and project-relative path helpers."""

from __future__ import annotations

from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_dir(config: dict) -> Path:
    return resolve_path(config["experiment"]["output_dir"])


def save_config(path: str | Path, config: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
