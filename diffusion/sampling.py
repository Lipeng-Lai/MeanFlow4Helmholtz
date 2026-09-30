"""DDIM sampling with optional Helmholtz PDE guidance."""

from __future__ import annotations

import torch

from utils.physics import residual_mse_per_sample


def _guided_clean_prediction(
    prediction_normalized: torch.Tensor,
    condition_physical: torch.Tensor,
    frequency_hz: torch.Tensor,
    normalizer,
    physics: dict,
    guidance: dict,
) -> torch.Tensor:
    """Apply a physical-scale gradient correction to a DDIM state."""
    if not bool(guidance.get("enabled", False)):
        return prediction_normalized
    with torch.enable_grad():
        prediction = normalizer.decode_target(prediction_normalized.detach().float())
        prediction.requires_grad_(True)
        loss = residual_mse_per_sample(
            prediction, condition_physical.float(), frequency_hz, physics
        ).sum()
        gradient = torch.autograd.grad(loss, prediction)[0]
        if guidance.get("normalize_gradient", "linf") == "linf":
            scale = gradient.abs().flatten(1).amax(1).clamp_min(1.0e-12)
            gradient = gradient / scale[:, None, None, None]
        corrected = prediction - float(guidance["scale"]) * gradient
    return normalizer.encode_target(corrected).to(prediction_normalized.dtype).detach()


def ddim_sample(model, diffusion, condition_normalized: torch.Tensor,
                noise: torch.Tensor, sampling_steps: int, *, eta: float = 0.0,
                clip_denoised: bool = False,
                condition_physical: torch.Tensor | None = None,
                frequency_hz: torch.Tensor | None = None,
                normalizer=None,
                physics: dict | None = None,
                guidance: dict | None = None) -> tuple[torch.Tensor, int]:
    sampling_steps = int(sampling_steps)
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")

    timesteps = torch.linspace(
        diffusion.steps - 1,
        0,
        sampling_steps,
        device=noise.device,
    ).round().long()
    current = noise
    guidance = guidance or {"enabled": False}
    for index, timestep in enumerate(timesteps):
        t = torch.full(
            (len(current),),
            int(timestep),
            device=current.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            clean = model(current, t, condition_normalized)
        if clip_denoised:
            clean = clean.clamp(-1.0, 1.0)

        if index == len(timesteps) - 1:
            current = clean
        else:
            previous = timesteps[index + 1]
            alpha = diffusion.alphas_cumprod[int(timestep)].to(
                current.device, current.dtype
            )
            alpha_previous = diffusion.alphas_cumprod[int(previous)].to(
                current.device, current.dtype
            )
            epsilon = (current - alpha.sqrt() * clean) / (
                1.0 - alpha
            ).sqrt().clamp_min(1.0e-12)
            sigma = float(eta) * (
                (1.0 - alpha_previous)
                / (1.0 - alpha)
                * (1.0 - alpha / alpha_previous)
            ).clamp_min(0.0).sqrt()
            direction = (
                1.0 - alpha_previous - sigma.square()
            ).clamp_min(0.0).sqrt() * epsilon
            current = alpha_previous.sqrt() * clean + direction
            if eta:
                current = current + sigma * torch.randn_like(current)

        if guidance.get("enabled", False):
            if (
                condition_physical is None
                or frequency_hz is None
                or normalizer is None
                or physics is None
            ):
                raise ValueError(
                    "PDE guidance requires condition_physical, frequency_hz, "
                    "normalizer, and physics"
                )
            current = _guided_clean_prediction(
                current,
                condition_physical,
                frequency_hz,
                normalizer,
                physics,
                guidance,
            )
    return current, len(timesteps)
