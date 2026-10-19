"""Memory-mapped reader for paired factorized Helmholtz forward data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class HelmholtzNpyDataset:
    """Lazily reconstruct FNO forward samples from factorized arrays."""

    def __init__(self, split_root: str | Path) -> None:
        self.split_root = Path(split_root)
        self.velocity_path = self.split_root / "data" / "velocity.npy"
        self.background_path = self.split_root / "data" / "background.npy"
        self.label_path = self.split_root / "label.npy"
        self.models_path = self.split_root / "models.csv"
        self._velocity = self._background = self._label = None
        self._open()
        self._validate()
        self.model_count = int(self._velocity.shape[0])
        self.frequency_count = int(self._background.shape[0])
        self.source_count = int(self._background.shape[1])
        self.metadata = self._read_metadata()

    def _open(self) -> None:
        self._velocity = np.load(self.velocity_path, mmap_mode="r")
        self._background = np.load(self.background_path, mmap_mode="r")
        self._label = np.load(self.label_path, mmap_mode="r")

    def _validate(self) -> None:
        velocity, background, label = self._velocity, self._background, self._label
        if velocity.ndim != 4 or velocity.shape[1] != 1 or velocity.shape[0] % 2:
            raise ValueError(f"invalid velocity shape {velocity.shape}")
        if background.ndim != 5 or background.shape[2] != 2:
            raise ValueError(f"invalid background shape {background.shape}")
        expected_label = (
            velocity.shape[0], background.shape[0], background.shape[1], 2,
            velocity.shape[-2], velocity.shape[-1],
        )
        if label.shape != expected_label:
            raise ValueError(f"label shape {label.shape} != {expected_label}")

    def _read_metadata(self) -> list[dict[str, str]]:
        if not self.models_path.is_file():
            raise FileNotFoundError(f"paired model metadata is missing: {self.models_path}")
        with self.models_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != self.model_count:
            raise ValueError(f"models.csv has {len(rows)} rows, expected {self.model_count}")
        for index, row in enumerate(rows):
            expected_variant = "original" if index % 2 == 0 else "smoothed"
            if int(row["model_index"]) != index or int(row["pair_index"]) != index // 2:
                raise ValueError("models.csv does not follow interleaved pair ordering")
            if row["variant"] != expected_variant:
                raise ValueError(f"model {index} variant is {row['variant']}, expected {expected_variant}")
        return rows

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for name in ("_velocity", "_background", "_label"):
            state[name] = None
        return state

    def __len__(self) -> int:
        return self.model_count * self.frequency_count * self.source_count

    def sample_indices(self, index: int) -> tuple[int, int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        model_index, remainder = divmod(index, self.frequency_count * self.source_count)
        frequency_index, source_index = divmod(remainder, self.source_count)
        return model_index, frequency_index, source_index

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.bool_ | int]:
        if self._velocity is None:
            self._open()
        model_index, frequency_index, source_index = self.sample_indices(index)
        condition = np.concatenate(
            (self._background[frequency_index, source_index], self._velocity[model_index]), axis=0
        )
        target = np.array(self._label[model_index, frequency_index, source_index], copy=True)
        return {
            "condition": condition,
            "target": target,
            "model_index": model_index,
            "frequency_index": frequency_index,
            "source_index": source_index,
        }

