"""Exponential moving average of model parameters and buffers."""

from __future__ import annotations

from copy import deepcopy
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                value.lerp_(source[name], 1.0 - self.decay)
            else:
                value.copy_(source[name])

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)
