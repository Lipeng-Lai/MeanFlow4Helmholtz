"""Per-sample wavefield and PDE metrics with grouped aggregation."""

from __future__ import annotations

from collections import defaultdict
import torch

from .physics import residual_mse_per_sample


def per_sample_metrics(
    prediction_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    frequency_hz: torch.Tensor,
    physics: dict,
) -> dict[str, torch.Tensor]:
    error = prediction - target
    return {
        "mse": (prediction_normalized - target_normalized).square().mean((1, 2, 3)),
        "relative_l2": (
            error.square().sum((1, 2, 3))
            / target.square().sum((1, 2, 3)).clamp_min(1.0e-12)
        ).sqrt(),
        "pde_loss": residual_mse_per_sample(
            prediction, condition, frequency_hz, physics
        ),
    }


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, values: dict[str, torch.Tensor]) -> None:
        batch = len(next(iter(values.values())))
        self.count += batch
        for name, value in values.items():
            self.sums[name] += value.detach().double().sum().item()

    def compute(self) -> dict[str, float]:
        return {name: total / self.count for name, total in self.sums.items()}


def format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(
        f"{name}={format_scalar(value)}" for name, value in metrics.items()
    )


def format_scalar(value: float, precision: int = 8) -> str:
    """Format a scalar without scientific notation."""
    return f"{float(value):.{precision}f}"
