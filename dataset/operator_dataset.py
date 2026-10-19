"""PyTorch-facing paired Helmholtz dataset and streaming normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .helmholtz_dataset import HelmholtzNpyDataset


class HelmholtzOperatorDataset:
    def __init__(self, split_root: str | Path, frequencies_hz: Sequence[float]) -> None:
        self.base = HelmholtzNpyDataset(split_root)
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=np.float32)
        if len(self.frequencies_hz) != self.base.frequency_count:
            raise ValueError(
                f"configured {len(self.frequencies_hz)} frequencies but data has "
                f"{self.base.frequency_count}"
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        sample["frequency_hz"] = self.frequencies_hz[int(sample["frequency_index"])]
        return sample


def _channel_moments(array: np.ndarray, channel_axis: int, chunk_models: int):
    channels = int(array.shape[channel_axis])
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    count = 0
    for start in range(0, len(array), chunk_models):
        block = np.asarray(array[start : start + chunk_models])
        values = np.moveaxis(block, channel_axis, 0).reshape(channels, -1).astype(np.float64)
        total += values.sum(axis=1)
        total_sq += np.square(values).sum(axis=1)
        count += values.shape[1]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    return mean, np.sqrt(variance)


def compute_normalization_stats(split_root: str | Path, chunk_models=16, epsilon=1.0e-6):
    split_root = Path(split_root)
    velocity = np.load(split_root / "data" / "velocity.npy", mmap_mode="r")
    background = np.load(split_root / "data" / "background.npy", mmap_mode="r")
    target = np.load(split_root / "label.npy", mmap_mode="r")
    velocity_mean, velocity_std = _channel_moments(velocity, 1, chunk_models)
    background_mean, background_std = _channel_moments(background, 2, chunk_models)
    target_mean, target_std = _channel_moments(target, 3, chunk_models)
    return {
        "condition_mean": np.concatenate((background_mean, velocity_mean)).astype(np.float32),
        "condition_std": np.maximum(
            np.concatenate((background_std, velocity_std)), epsilon
        ).astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_std": np.maximum(target_std, epsilon).astype(np.float32),
    }


def load_or_compute_normalization(split_root, stats_path, recompute, chunk_models, epsilon):
    stats_path = Path(stats_path)
    if stats_path.exists() and not recompute:
        with np.load(stats_path) as saved:
            return {key: saved[key] for key in saved.files}
    stats = compute_normalization_stats(split_root, chunk_models, epsilon)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, **stats)
    return stats
