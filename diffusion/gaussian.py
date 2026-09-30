"""Cosine Gaussian diffusion with direct clean-sample prediction."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    times = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    alpha_bar = torch.cos((times + offset) / (1.0 + offset) * math.pi / 2.0).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999).float()


class GaussianDiffusion:
    def __init__(self, steps: int, schedule: str = "cosine") -> None:
        if schedule != "cosine":
            raise ValueError(f"unsupported noise schedule: {schedule}")
        self.steps = int(steps)
        self.betas = cosine_beta_schedule(self.steps)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)

    @classmethod
    def from_config(cls, config: dict) -> "GaussianDiffusion":
        return cls(int(config["steps"]), str(config["schedule"]))

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return values.to(timesteps.device)[timesteps].to(x.dtype).view(-1, 1, 1, 1)

    def q_sample(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        noise = torch.randn_like(clean) if noise is None else noise
        alpha_bar = self._extract(self.alphas_cumprod, timesteps, clean)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def training_loss(
        self,
        model,
        clean: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        noisy = self.q_sample(clean, timesteps)
        prediction = model(noisy, timesteps, condition)
        return F.mse_loss(prediction, clean)
