"""Two-dimensional Fourier neural operator for Helmholtz wavefields."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Real-input spectral convolution with independent positive/negative modes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_height: int,
        modes_width: int,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_height = int(modes_height)
        self.modes_width = int(modes_width)
        self.rank = int(rank) if rank is not None else None
        if min(self.modes_height, self.modes_width) < 1:
            raise ValueError("FNO mode counts must be positive")
        if self.rank is not None and self.rank < 1:
            raise ValueError("FNO spectral rank must be positive")

        scale = 1.0 / math.sqrt(self.in_channels * self.out_channels)
        if self.rank is None:
            shape = (
                self.in_channels,
                self.out_channels,
                self.modes_height,
                self.modes_width,
            )
            self.positive_real = nn.Parameter(scale * torch.randn(shape))
            self.positive_imag = nn.Parameter(scale * torch.randn(shape))
            self.negative_real = nn.Parameter(scale * torch.randn(shape))
            self.negative_imag = nn.Parameter(scale * torch.randn(shape))
        else:
            left_shape = (
                self.in_channels,
                self.rank,
                self.modes_height,
                self.modes_width,
            )
            right_shape = (
                self.rank,
                self.out_channels,
                self.modes_height,
                self.modes_width,
            )
            factor_scale = math.sqrt(scale)
            for branch in ("positive", "negative"):
                for side, shape in (("left", left_shape), ("right", right_shape)):
                    setattr(
                        self,
                        f"{branch}_{side}_real",
                        nn.Parameter(factor_scale * torch.randn(shape)),
                    )
                    setattr(
                        self,
                        f"{branch}_{side}_imag",
                        nn.Parameter(factor_scale * torch.randn(shape)),
                    )

    @staticmethod
    def multiply(inputs: torch.Tensor, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        weights = torch.complex(real, imag)
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    @staticmethod
    def multiply_factorized(
        inputs: torch.Tensor,
        left_real: torch.Tensor,
        left_imag: torch.Tensor,
        right_real: torch.Tensor,
        right_imag: torch.Tensor,
    ) -> torch.Tensor:
        left = torch.complex(left_real, left_imag)
        right = torch.complex(right_real, right_imag)
        hidden = torch.einsum("bixy,irxy->brxy", inputs, left)
        return torch.einsum("brxy,roxy->boxy", hidden, right)

    def branch(self, inputs: torch.Tensor, name: str, mh: int, mw: int) -> torch.Tensor:
        if self.rank is None:
            return self.multiply(
                inputs,
                getattr(self, f"{name}_real")[:, :, :mh, :mw],
                getattr(self, f"{name}_imag")[:, :, :mh, :mw],
            )
        return self.multiply_factorized(
            inputs,
            getattr(self, f"{name}_left_real")[:, :, :mh, :mw],
            getattr(self, f"{name}_left_imag")[:, :, :mh, :mw],
            getattr(self, f"{name}_right_real")[:, :, :mh, :mw],
            getattr(self, f"{name}_right_imag")[:, :, :mh, :mw],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = inputs.shape
        modes_height = min(self.modes_height, height // 2)
        modes_width = min(self.modes_width, width // 2 + 1)
        transformed = torch.fft.rfft2(inputs.float(), norm="ortho")
        output = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            device=inputs.device,
            dtype=transformed.dtype,
        )
        output[:, :, :modes_height, :modes_width] = self.branch(
            transformed[:, :, :modes_height, :modes_width],
            "positive",
            modes_height,
            modes_width,
        )
        output[:, :, -modes_height:, :modes_width] = self.branch(
            transformed[:, :, -modes_height:, :modes_width],
            "negative",
            modes_height,
            modes_width,
        )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho").to(inputs.dtype)


class ChannelMLP(nn.Module):
    def __init__(self, channels: int, ratio: float) -> None:
        super().__init__()
        hidden = max(channels, int(round(channels * ratio)))
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class FNOBlock(nn.Module):
    def __init__(
        self,
        width: int,
        modes_height: int,
        modes_width: int,
        mlp_ratio: float,
        spectral_rank: int | None,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(
            width, width, modes_height, modes_width, rank=spectral_rank
        )
        self.local = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)
        self.mlp = ChannelMLP(width, mlp_ratio)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mixed = self.norm(self.spectral(inputs) + self.local(inputs))
        return inputs + self.mlp(F.gelu(mixed))


class FourierNeuralOperator2d(nn.Module):
    """Map normalized ``(Re(U0), Im(U0), velocity)`` to ``(Re(dU), Im(dU))``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        modes_height: int,
        modes_width: int,
        num_layers: int,
        mlp_ratio: float = 2.0,
        spectral_rank: int | None = None,
        padding: int = 8,
        use_coordinates: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.padding = int(padding)
        self.use_coordinates = bool(use_coordinates)
        if self.width < 1 or int(num_layers) < 1:
            raise ValueError("FNO width and num_layers must be positive")
        if self.padding < 0:
            raise ValueError("FNO padding cannot be negative")

        lifted_channels = self.in_channels + (2 if self.use_coordinates else 0)
        self.lift = nn.Conv2d(lifted_channels, self.width, 1)
        self.blocks = nn.ModuleList(
            [
                FNOBlock(
                    self.width,
                    int(modes_height),
                    int(modes_width),
                    float(mlp_ratio),
                    spectral_rank,
                )
                for _ in range(int(num_layers))
            ]
        )
        projection_width = self.width * 2
        self.projection = nn.Sequential(
            nn.Conv2d(self.width, projection_width, 1),
            nn.GELU(),
            nn.Conv2d(projection_width, self.out_channels, 1),
        )

    @classmethod
    def from_config(cls, config: dict) -> "FourierNeuralOperator2d":
        return cls(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            width=int(config["width"]),
            modes_height=int(config["modes_height"]),
            modes_width=int(config["modes_width"]),
            num_layers=int(config["num_layers"]),
            mlp_ratio=float(config.get("mlp_ratio", 2.0)),
            spectral_rank=(
                int(config["spectral_rank"])
                if config.get("spectral_rank") is not None
                else None
            ),
            padding=int(config.get("padding", 8)),
            use_coordinates=bool(config.get("use_coordinates", True)),
        )

    @staticmethod
    def coordinate_grid(inputs: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = inputs.shape
        y = torch.linspace(0.0, 1.0, height, device=inputs.device, dtype=inputs.dtype)
        x = torch.linspace(0.0, 1.0, width, device=inputs.device, dtype=inputs.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((grid_y, grid_x), dim=0)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 4 or condition.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [batch,{self.in_channels},height,width], got {tuple(condition.shape)}"
            )
        original_height, original_width = condition.shape[-2:]
        if self.use_coordinates:
            condition = torch.cat((condition, self.coordinate_grid(condition)), dim=1)
        hidden = self.lift(condition)
        if self.padding:
            hidden = F.pad(hidden, (0, self.padding, 0, self.padding))
        for block in self.blocks:
            hidden = block(hidden)
        if self.padding:
            hidden = hidden[..., :original_height, :original_width]
        return self.projection(hidden)
