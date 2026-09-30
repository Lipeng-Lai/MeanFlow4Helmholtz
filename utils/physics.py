"""Differentiable scattered-field Helmholtz residuals."""

from __future__ import annotations

import math
import torch


def nine_point_laplacian(field: torch.Tensor, spacing: float) -> torch.Tensor:
    axial = (
        field[..., :-2, 1:-1]
        + field[..., 2:, 1:-1]
        + field[..., 1:-1, :-2]
        + field[..., 1:-1, 2:]
    )
    diagonal = (
        field[..., :-2, :-2]
        + field[..., :-2, 2:]
        + field[..., 2:, :-2]
        + field[..., 2:, 2:]
    )
    return (4.0 * axial + diagonal - 20.0 * field[..., 1:-1, 1:-1]) / (
        6.0 * spacing**2
    )


def helmholtz_forcing(
    condition: torch.Tensor, frequency_hz: torch.Tensor, physics: dict
) -> torch.Tensor:
    velocity = condition[:, 2:3]
    background = condition[:, :2]
    omega_sq = (2.0 * math.pi * frequency_hz.float()).square()[:, None, None, None]
    contrast = velocity.reciprocal().square() - float(
        physics["background_velocity_km_s"]
    ) ** -2
    return (omega_sq * contrast * background)[..., 1:-1, 1:-1]


def helmholtz_residual(
    scattered: torch.Tensor,
    condition: torch.Tensor,
    frequency_hz: torch.Tensor,
    physics: dict,
) -> torch.Tensor:
    spacing_z, spacing_x = map(float, physics["spacing_km"])
    if spacing_z != spacing_x:
        raise ValueError("the isotropic nine-point stencil requires dx=dz")
    scattered = scattered.float()
    condition = condition.float()
    velocity = condition[:, 2:3]
    omega_sq = (2.0 * math.pi * frequency_hz.float()).square()[:, None, None, None]
    operator = nine_point_laplacian(scattered, spacing_x)
    operator = operator + (
        omega_sq * velocity.reciprocal().square() * scattered
    )[..., 1:-1, 1:-1]
    return operator + helmholtz_forcing(condition, frequency_hz, physics)


def residual_mse_per_sample(
    scattered: torch.Tensor,
    condition: torch.Tensor,
    frequency_hz: torch.Tensor,
    physics: dict,
) -> torch.Tensor:
    return helmholtz_residual(scattered, condition, frequency_hz, physics).square().mean((1, 2, 3))


def coordinate_laplacian(
    field: torch.Tensor,
    coordinates: torch.Tensor,
    domain_size: tuple[float, float],
) -> torch.Tensor:
    """Differentiate pointwise decoder values with respect to normalized coordinates."""
    channels = []
    for channel in field.unbind(1):
        gradient = torch.autograd.grad(
            channel.sum(), coordinates, create_graph=True, retain_graph=True
        )[0]
        second_z = torch.autograd.grad(
            gradient[..., 0].sum(), coordinates, create_graph=True, retain_graph=True
        )[0][..., 0]
        second_x = torch.autograd.grad(
            gradient[..., 1].sum(), coordinates, create_graph=True, retain_graph=True
        )[0][..., 1]
        channels.append(
            second_z / domain_size[0] ** 2 + second_x / domain_size[1] ** 2
        )
    return torch.stack(channels, dim=1)


def continuous_helmholtz_residual(
    scattered: torch.Tensor,
    coordinates: torch.Tensor,
    condition: torch.Tensor,
    frequency_hz: torch.Tensor,
    physics: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous-coordinate Helmholtz residual and forcing at query points."""
    spacing_z, spacing_x = map(float, physics["spacing_km"])
    height, width = map(int, physics["grid_size"])
    domain_size = ((height - 1) * spacing_z, (width - 1) * spacing_x)
    laplacian = coordinate_laplacian(scattered, coordinates, domain_size)
    velocity = condition[:, 2:3]
    background = condition[:, :2]
    omega_sq = (2.0 * math.pi * frequency_hz.float()).square()[:, None, None]
    contrast = velocity.reciprocal().square() - float(
        physics["background_velocity_km_s"]
    ) ** -2
    forcing = omega_sq * contrast * background
    operator = laplacian + omega_sq * velocity.reciprocal().square() * scattered
    return operator + forcing, forcing


def sample_at_coordinates(
    field: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    """Bilinearly sample BCHW fields at normalized (z, x) coordinates."""
    grid = coordinates[..., (1, 0)].mul(2.0).sub(1.0).unsqueeze(2)
    sampled = torch.nn.functional.grid_sample(
        field, grid, mode="bilinear", align_corners=True
    )
    return sampled.squeeze(-1)
