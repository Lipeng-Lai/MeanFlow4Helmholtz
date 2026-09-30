"""Time-conditioned U-Net for seismic scattered-wavefield diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, channels: int) -> torch.Tensor:
    half = channels // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    if channels % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def normalization(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, dropout: float):
        super().__init__()
        self.in_layers = nn.Sequential(
            normalization(in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.time(time).to(h.dtype)[..., None, None]
        return self.skip(x) + self.out_layers(h)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        head_channels = channels // self.num_heads
        q = q.reshape(batch, self.num_heads, head_channels, -1).transpose(-1, -2)
        k = k.reshape(batch, self.num_heads, head_channels, -1).transpose(-1, -2)
        v = v.reshape(batch, self.num_heads, head_channels, -1).transpose(-1, -2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(-1, -2).reshape(batch, channels, height * width)
        return x + self.proj(h).reshape(batch, channels, height, width)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class ConditionedSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x, time) if isinstance(layer, ResBlock) else layer(x)
        return x


class ConditionalUNet(nn.Module):
    """Concatenate state and physical condition with one or two time inputs."""

    def __init__(
        self,
        noisy_channels: int,
        condition_channels: int,
        out_channels: int,
        base_channels: int,
        channel_multipliers: tuple[int, ...],
        num_res_blocks: int,
        attention_downsample_factors: tuple[int, ...],
        num_heads: int,
        dropout: float,
        time_inputs: int = 1,
        auxiliary_velocity_head: bool = False,
    ) -> None:
        super().__init__()
        self.noisy_channels = noisy_channels
        self.condition_channels = condition_channels
        self.time_inputs = int(time_inputs)
        self.auxiliary_velocity_head = bool(auxiliary_velocity_head)
        if self.time_inputs not in {1, 2}:
            raise ValueError("ConditionalUNet time_inputs must be 1 or 2")
        self.downsample_factor = 2 ** (len(channel_multipliers) - 1)
        time_channels = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels * self.time_inputs, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        channels = base_channels
        self.input_conv = nn.Conv2d(noisy_channels + condition_channels, channels, 3, padding=1)
        skip_channels = [channels]
        self.down = nn.ModuleList()
        factor = 1
        for level, multiplier in enumerate(channel_multipliers):
            level_channels = base_channels * multiplier
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [ResBlock(channels, level_channels, time_channels, dropout)]
                channels = level_channels
                if factor in attention_downsample_factors:
                    layers.append(AttentionBlock(channels, num_heads))
                self.down.append(ConditionedSequential(*layers))
                skip_channels.append(channels)
            if level != len(channel_multipliers) - 1:
                self.down.append(ConditionedSequential(Downsample(channels)))
                skip_channels.append(channels)
                factor *= 2

        self.middle = ConditionedSequential(
            ResBlock(channels, channels, time_channels, dropout),
            AttentionBlock(channels, num_heads),
            ResBlock(channels, channels, time_channels, dropout),
        )

        self.up = nn.ModuleList()
        for level, multiplier in reversed(list(enumerate(channel_multipliers))):
            level_channels = base_channels * multiplier
            for block_index in range(num_res_blocks + 1):
                skip = skip_channels.pop()
                layers = [ResBlock(channels + skip, level_channels, time_channels, dropout)]
                channels = level_channels
                if factor in attention_downsample_factors:
                    layers.append(AttentionBlock(channels, num_heads))
                if level > 0 and block_index == num_res_blocks:
                    layers.append(Upsample(channels))
                    factor //= 2
                self.up.append(ConditionedSequential(*layers))
        assert not skip_channels
        self.output = nn.Sequential(
            normalization(channels), nn.SiLU(), nn.Conv2d(channels, out_channels, 3, padding=1)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.velocity_output = None
        if self.auxiliary_velocity_head:
            self.velocity_output = nn.Sequential(
                normalization(channels),
                nn.SiLU(),
                nn.Conv2d(channels, out_channels, 3, padding=1),
            )
            nn.init.zeros_(self.velocity_output[-1].weight)
            nn.init.zeros_(self.velocity_output[-1].bias)

    @classmethod
    def from_config(cls, config: dict) -> "ConditionalUNet":
        return cls(
            noisy_channels=int(config["noisy_channels"]),
            condition_channels=int(config["condition_channels"]),
            out_channels=int(config["out_channels"]),
            base_channels=int(config["base_channels"]),
            channel_multipliers=tuple(map(int, config["channel_multipliers"])),
            num_res_blocks=int(config["num_res_blocks"]),
            attention_downsample_factors=tuple(map(int, config["attention_downsample_factors"])),
            num_heads=int(config["num_heads"]),
            dropout=float(config["dropout"]),
            time_inputs=int(config.get("time_inputs", 1)),
            auxiliary_velocity_head=bool(
                config.get("auxiliary_velocity_head", False)
            ),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        end_time: torch.Tensor | None = None,
        return_velocity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        original_height, original_width = noisy.shape[-2:]
        pad_height = (-original_height) % self.downsample_factor
        pad_width = (-original_width) % self.downsample_factor
        if pad_height or pad_width:
            noisy = F.pad(noisy, (0, pad_width, 0, pad_height))
            condition = F.pad(condition, (0, pad_width, 0, pad_height))
        time_embedding = [
            timestep_embedding(timesteps, self.time_mlp[0].in_features // self.time_inputs)
        ]
        if self.time_inputs == 2:
            if end_time is None:
                raise ValueError("MeanFlow U-Net requires end_time")
            time_embedding.append(
                timestep_embedding(
                    end_time, self.time_mlp[0].in_features // self.time_inputs
                )
            )
        time = self.time_mlp(torch.cat(time_embedding, dim=1))
        h = self.input_conv(torch.cat((noisy, condition), dim=1))
        skips = [h]
        for module in self.down:
            h = module(h, time)
            skips.append(h)
        h = self.middle(h, time)
        for module in self.up:
            skip = skips.pop()
            h = module(torch.cat((h, skip), dim=1), time)
        output = self.output(h)[..., :original_height, :original_width]
        if not return_velocity:
            return output
        if self.velocity_output is None:
            raise ValueError(
                "return_velocity=True requires model.auxiliary_velocity_head=true"
            )
        velocity = self.velocity_output(h)[..., :original_height, :original_width]
        return output, velocity
