"""Dataset and DataLoader construction."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dataset.prodiff_dataset import HelmholtzDiffusionDataset
from .config import resolve_path


def build_dataset(config: dict, split: str):
    root = resolve_path(config["data"][f"{split}_root"])
    return HelmholtzDiffusionDataset(root, config["data"]["frequencies_hz"])


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
