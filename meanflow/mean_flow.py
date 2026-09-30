"""MeanFlow training and finite-interval sampling utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import jvp

from utils.physics import residual_mse_per_sample

def _require_finite(name: str, value: torch.Tensor) -> None:
    """Fail fast with a useful location when MeanFlow becomes non-finite."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"MeanFlow produced non-finite {name}")


def sample_meanflow_times(
    batch_size: int,
    device: torch.device,
    *,
    sampler: str = "logit_normal",
    mu: float = -0.4,
    sigma: float = 1.0,
    ratio_r_not_equal_t: float = 0.25,
    endpoint_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ordered MeanFlow intervals, optionally including full endpoints.

    ``ratio_r_not_equal_t`` is the total fraction of finite-interval samples.
    ``endpoint_ratio`` reserves part of that fraction for the exact one-step
    inference interval ``(r, t) = (0, 1)``.
    """
    sampler = str(sampler).lower()
    if sampler == "uniform":
        samples = torch.rand(batch_size, 2, device=device)
    elif sampler == "logit_normal":
        samples = torch.sigmoid(
            torch.randn(batch_size, 2, device=device) * float(sigma) + float(mu)
        )
    else:
        raise ValueError(f"unsupported MeanFlow time sampler: {sampler}")
    if not 0.0 <= float(ratio_r_not_equal_t) <= 1.0:
        raise ValueError("ratio_r_not_equal_t must be in [0, 1]")
    if not 0.0 <= float(endpoint_ratio) <= float(ratio_r_not_equal_t):
        raise ValueError("endpoint_ratio must be in [0, ratio_r_not_equal_t]")

    ordered, _ = samples.sort(dim=1)
    r, t = ordered[:, 0], ordered[:, 1]
    endpoint_mask = torch.rand(batch_size, device=device) < float(endpoint_ratio)
    remaining_probability = 1.0 - float(endpoint_ratio)
    if remaining_probability > 0.0:
        random_interval_probability = (
            float(ratio_r_not_equal_t) - float(endpoint_ratio)
        ) / remaining_probability
    else:
        random_interval_probability = 0.0
    random_interval_mask = (
        ~endpoint_mask
        & (torch.rand(batch_size, device=device) < random_interval_probability)
    )
    equal_mask = ~(endpoint_mask | random_interval_mask)
    r = torch.where(equal_mask, t, r)
    r = torch.where(endpoint_mask, torch.zeros_like(r), r)
    t = torch.where(endpoint_mask, torch.ones_like(t), t)
    return r, t


def _model_call(
    model,
    state: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor,
    time_scale: float,
    *,
    return_velocity: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Call a MeanFlow model using the official ``(t, t-r)`` conditioning."""
    kwargs = {"end_time": (t - r) * float(time_scale)}
    if return_velocity:
        kwargs["return_velocity"] = True
    return model(state, t * float(time_scale), condition, **kwargs)


def _adaptive_error_loss(
    error: torch.Tensor,
    *,
    weighting: str,
    adaptive_p: float,
    adaptive_eps: float,
) -> torch.Tensor:
    """Apply the MeanFlow adaptive metric to a per-pixel regression error."""
    per_sample_loss = error.float().square().flatten(1).sum(1)
    _require_finite("per-sample loss", per_sample_loss)
    weighting = str(weighting).lower()
    if weighting == "adaptive":
        weights = (per_sample_loss.detach() + float(adaptive_eps)).pow(
            -float(adaptive_p)
        )
        return (weights * per_sample_loss).mean()
    if weighting == "uniform":
        return per_sample_loss.mean()
    raise ValueError(f"unsupported MeanFlow weighting: {weighting}")


def _guided_state(
    state: torch.Tensor,
    condition_physical: torch.Tensor | None,
    frequency_hz: torch.Tensor | None,
    normalizer,
    physics: dict | None,
    guidance: dict | None,
) -> torch.Tensor:
    """Apply a physical-scale Helmholtz correction after each flow step."""
    guidance = guidance or {"enabled": False}
    if not bool(guidance.get("enabled", False)):
        return state
    if (
        condition_physical is None
        or frequency_hz is None
        or normalizer is None
        or physics is None
    ):
        raise ValueError(
            "PDE guidance requires physical condition, frequency, normalizer, and physics"
        )

    with torch.enable_grad():
        physical = normalizer.decode_target(state.detach().float())
        physical.requires_grad_(True)
        loss = residual_mse_per_sample(
            physical, condition_physical.float(), frequency_hz, physics
        ).sum()
        gradient = torch.autograd.grad(loss, physical)[0]
        if guidance.get("normalize_gradient", "linf") == "linf":
            scale = gradient.abs().flatten(1).amax(1).clamp_min(1.0e-12)
            gradient = gradient / scale[:, None, None, None]
        corrected = physical - float(guidance["scale"]) * gradient
    return normalizer.encode_target(corrected).to(state.dtype).detach()


def meanflow_loss(
    model,
    clean: torch.Tensor,
    condition: torch.Tensor,
    *,
    time_scale: float,
    time_sampler: str = "logit_normal",
    time_mu: float = -0.4,
    time_sigma: float = 1.0,
    ratio_r_not_equal_t: float = 0.25,
    endpoint_ratio: float = 0.0,
    weighting: str = "adaptive",
    adaptive_p: float = 1.0,
    adaptive_eps: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the MeanFlow Identity loss and an instantaneous-velocity metric."""
    batch_size = len(clean)
    r, t = sample_meanflow_times(
        batch_size,
        clean.device,
        sampler=time_sampler,
        mu=time_mu,
        sigma=time_sigma,
        ratio_r_not_equal_t=ratio_r_not_equal_t,
        endpoint_ratio=endpoint_ratio,
    )
    noise = torch.randn_like(clean)
    weight = t.to(clean.dtype).view(-1, 1, 1, 1)
    state = (1.0 - weight) * clean + weight * noise
    velocity = noise - clean
    zeros = torch.zeros_like(r)
    ones = torch.ones_like(t)

    prediction = _model_call(
        model, state, r, t, condition, time_scale
    )

    def call(state_value: torch.Tensor, r_value: torch.Tensor, t_value: torch.Tensor):
        return _model_call(
            model,
            state_value,
            r_value,
            t_value,
            condition.float(),
            time_scale,
        )

    # The JVP is part of a stopped target.  Keeping it out of the parameter
    # graph avoids unnecessary higher-order state and follows the reference
    # PyTorch implementation's numerically stable FP32 path.
    with torch.no_grad():
        _, du_dt = jvp(
            call,
            (state.float(), r.float(), t.float()),
            (velocity.float(), zeros.float(), ones.float()),
        )
    _require_finite("model prediction", prediction)
    _require_finite("JVP time derivative", du_dt)
    target = velocity.float() - (t - r).float().view(-1, 1, 1, 1) * du_dt
    _require_finite("MeanFlow target", target)
    error = prediction.float() - target.detach().float()
    loss = _adaptive_error_loss(
        error,
        weighting=weighting,
        adaptive_p=adaptive_p,
        adaptive_eps=adaptive_eps,
    )
    _require_finite("loss", loss)
    velocity_mse = F.mse_loss(prediction.float(), velocity.float())
    _require_finite("instantaneous velocity MSE", velocity_mse)
    return loss, velocity_mse.detach()


@torch.no_grad()
def meanflow_sample(
    model,
    noise: torch.Tensor,
    condition: torch.Tensor,
    sampling_steps: int,
    *,
    time_scale: float,
    condition_physical: torch.Tensor | None = None,
    frequency_hz: torch.Tensor | None = None,
    normalizer=None,
    physics: dict | None = None,
    guidance: dict | None = None,
) -> tuple[torch.Tensor, int]:
    """Sample with MeanFlow finite-interval updates, defaulting to one NFE."""
    sampling_steps = int(sampling_steps)
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")

    current = noise.clone()
    time_steps = torch.linspace(1.0, 0.0, sampling_steps + 1, device=noise.device)
    for index in range(sampling_steps):
        t_value = time_steps[index]
        r_value = time_steps[index + 1]
        t = t_value.expand(len(current))
        r = r_value.expand(len(current))
        average_velocity = _model_call(
            model, current, r, t, condition, time_scale
        )
        current = current - (t - r).to(current.dtype).view(-1, 1, 1, 1) * average_velocity
        current = _guided_state(
            current,
            condition_physical,
            frequency_hz,
            normalizer,
            physics,
            guidance,
        )
    return current, sampling_steps
