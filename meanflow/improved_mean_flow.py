"""Improved MeanFlow (iMF) training utilities.

The iMF objective rewrites MeanFlow as a velocity regression problem.  Its JVP
tangent can use either the parameter-free boundary prediction
``u_theta(z_t, t, t)`` or a separately supervised auxiliary marginal-velocity
head, while retaining the original one-step MeanFlow sampler.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import jvp

from .mean_flow import (
    _adaptive_error_loss,
    _model_call,
    _require_finite,
    meanflow_sample,
    sample_meanflow_times,
)


def improved_meanflow_loss(
    model,
    clean: torch.Tensor,
    condition: torch.Tensor,
    *,
    time_scale: float,
    time_sampler: str = "logit_normal",
    time_mu: float = -0.4,
    time_sigma: float = 1.0,
    ratio_r_not_equal_t: float = 0.5,
    endpoint_ratio: float = 0.0,
    weighting: str = "adaptive",
    adaptive_p: float = 1.0,
    adaptive_eps: float = 0.01,
    velocity_parameterization: str = "boundary",
    velocity_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the iMF objective and detached compound/velocity metrics.

    ``velocity_parameterization='auxiliary_head'`` follows the stronger iMF
    implementation: a separately supervised marginal-velocity prediction is
    used as the JVP tangent.  ``'boundary'`` retains the parameter-free
    ``u(z_t, t, t)`` reuse variant.
    """
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
    time_weight = t.to(clean.dtype).view(-1, 1, 1, 1)
    state = (1.0 - time_weight) * clean + time_weight * noise
    velocity_target = noise - clean
    zeros = torch.zeros_like(r)
    ones = torch.ones_like(t)

    velocity_parameterization = str(velocity_parameterization).lower()
    if velocity_parameterization == "auxiliary_head":
        average_velocity, velocity_prediction = _model_call(
            model,
            state,
            r,
            t,
            condition,
            time_scale,
            return_velocity=True,
        )
    elif velocity_parameterization == "boundary":
        average_velocity = _model_call(
            model, state, r, t, condition, time_scale
        )
        velocity_prediction = _model_call(
            model, state, t, t, condition, time_scale
        )
    else:
        raise ValueError(
            "velocity_parameterization must be 'auxiliary_head' or 'boundary'"
        )
    _require_finite("average velocity", average_velocity)
    _require_finite("marginal velocity prediction", velocity_prediction)

    def model_call(
        state_value: torch.Tensor,
        r_value: torch.Tensor,
        t_value: torch.Tensor,
    ) -> torch.Tensor:
        return _model_call(
            model,
            state_value,
            r_value,
            t_value,
            condition.float(),
            time_scale,
        )

    with torch.no_grad():
        _, du_dt = jvp(
            model_call,
            (state.float(), r.float(), t.float()),
            (velocity_prediction.detach().float(), zeros.float(), ones.float()),
        )
    _require_finite("JVP time derivative", du_dt)

    compound_velocity = average_velocity + (
        (t - r).to(du_dt.dtype).view(-1, 1, 1, 1) * du_dt
    )
    _require_finite("compound velocity", compound_velocity)

    compound_error = compound_velocity.float() - velocity_target.float()
    compound_loss = _adaptive_error_loss(
        compound_error,
        weighting=weighting,
        adaptive_p=adaptive_p,
        adaptive_eps=adaptive_eps,
    )
    velocity_error = velocity_prediction.float() - velocity_target.float()
    velocity_loss = _adaptive_error_loss(
        velocity_error,
        weighting=weighting,
        adaptive_p=adaptive_p,
        adaptive_eps=adaptive_eps,
    )
    loss = compound_loss + float(velocity_loss_weight) * velocity_loss
    _require_finite("loss", loss)

    compound_mse = F.mse_loss(compound_velocity.float(), velocity_target.float())
    velocity_mse = F.mse_loss(velocity_prediction.float(), velocity_target.float())
    _require_finite("compound velocity MSE", compound_mse)
    _require_finite("marginal velocity MSE", velocity_mse)
    return loss, compound_mse.detach(), velocity_mse.detach()


def improved_meanflow_sample(*args, **kwargs):
    """Use the original finite-interval MeanFlow sampler for iMF inference."""
    return meanflow_sample(*args, **kwargs)


__all__ = ["improved_meanflow_loss", "improved_meanflow_sample"]
