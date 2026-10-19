"""Training-set channel normalization in physical units."""

from __future__ import annotations

import numpy as np
import torch
from dataset.operator_dataset import load_or_compute_normalization
from .config import resolve_path


class PhysicalNormalizer:
    def __init__(self, stats: dict, device: torch.device) -> None:
        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(stats[name], device=device).float()[None, :, None, None]

        self.condition_mean = tensor("condition_mean")
        self.condition_std = tensor("condition_std")
        self.target_mean = tensor("target_mean")
        self.target_std = tensor("target_std")
        self.state_mean = tensor("state_mean") if "state_mean" in stats else self.target_mean
        self.state_std = tensor("state_std") if "state_std" in stats else self.target_std

    def encode_condition(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.condition_mean) / self.condition_std

    def encode_target(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.target_mean) / self.target_std

    def decode_target(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.target_std + self.target_mean

    def encode_state(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.state_mean) / self.state_std

    def decode_state(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.state_std + self.state_mean

    def encode_velocity(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.condition_mean[:, 2:3]) / self.condition_std[:, 2:3]


def normalization_stats(config: dict, stats_path, recompute: bool = False) -> dict[str, np.ndarray]:
    data = config["data"]
    return load_or_compute_normalization(
        resolve_path(data["train_root"]),
        stats_path,
        recompute,
        int(data["normalization_chunk_models"]),
        float(data["normalization_epsilon"]),
    )

