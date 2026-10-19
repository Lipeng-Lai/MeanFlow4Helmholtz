"""Fixed validation probes for velocity derivatives; never a training objective."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import splu

from dataset.generate.helmholtz import (
    background_green_wavefield,
    build_helmholtz_matrix,
    sponge_profile,
)
from .config import load_config


def numerical_directional_derivative(velocity, direction, frequency, positions,
                                     spacing, background_velocity, solver,
                                     singularity_radius):
    """Solve the exact tangent of the discrete scattered-wavefield equation."""
    absorbing = solver["absorbing_layer"]
    padding = int(absorbing.get("padding_cells", 0))
    velocity = np.asarray(velocity, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if padding:
        velocity = np.pad(velocity, padding, mode="edge")
        direction = np.pad(direction, padding, mode="edge")
    positions = [(z + padding, x + padding) for z, x in positions]
    sign = float(absorbing.get("complex_sign", -1.0))
    matrix = build_helmholtz_matrix(
        velocity, spacing, frequency, str(solver["stencil"]), padding,
        float(absorbing["strength"]), float(absorbing["power"]), sign,
    )
    factorization = splu(matrix)
    background = np.stack([
        background_green_wavefield(
            velocity.shape, spacing, position, frequency, background_velocity,
            singularity_radius,
        ) for position in positions
    ])
    omega_squared = (2.0 * np.pi * frequency) ** 2
    rhs = -omega_squared * (velocity ** -2 - background_velocity ** -2) * background
    scattered = factorization.solve(rhs.reshape(len(positions), -1).T).T.reshape(rhs.shape)
    stretch = 1.0 + 1j * sign * sponge_profile(
        velocity.shape, padding, float(absorbing["strength"]), float(absorbing["power"])
    )
    tangent_rhs = (
        2.0 * omega_squared * direction / velocity ** 3
        * (background + stretch ** 2 * scattered)
    )
    tangent_matrix = factorization.solve(tangent_rhs.reshape(len(positions), -1).T)
    residual = matrix @ tangent_matrix - tangent_rhs.reshape(len(positions), -1).T
    relative_residual = np.linalg.norm(residual, axis=0) / np.maximum(
        np.linalg.norm(tangent_rhs.reshape(len(positions), -1).T, axis=0),
        np.finfo(np.float64).eps,
    )
    if np.max(relative_residual) > float(solver.get("relative_residual_tolerance", 1.0e-8)):
        raise RuntimeError("numerical sensitivity solve failed the residual tolerance")
    tangent = tangent_matrix.T.reshape(rhs.shape)
    if padding:
        tangent = tangent[:, padding:-padding, padding:-padding]
    return np.stack((tangent.real, tangent.imag), axis=1).astype(np.float32)


class SensitivityMonitor:
    """Mean relative squared error of J(v)d on fixed held-out probes.

    Both derivatives are in physical units, with the background held fixed.
    Numerical references use the exact discrete tangent; FNO uses autograd JVP.
    This samples directional derivatives, not the complete Jacobian.
    """

    def __init__(self, dataset, config, normalizer, device, output,
                 predictor: Callable | None = None):
        settings = config["sensitivity_monitor"]
        generation = load_config(settings["solver_config"])
        base = dataset.base
        if list(base._velocity.shape[-2:]) != list(generation["grid"]["shape"]):
            raise ValueError("sensitivity solver grid does not match validation data")
        if not np.allclose(config["physics"]["spacing_km"], generation["grid"]["spacing_km"]):
            raise ValueError("sensitivity solver spacing does not match training physics")
        if not np.isclose(config["physics"]["background_velocity_km_s"], generation["physics"]["background_velocity_km_s"]):
            raise ValueError("sensitivity background velocity does not match training physics")
        if not np.allclose(dataset.frequencies_hz, generation["frequencies_hz"]):
            raise ValueError("sensitivity solver frequencies do not match validation data")
        rng = np.random.default_rng(int(settings["seed"]))
        conditions, directions, references, keys = [], [], [], []
        source_indices = [int(i) for i in settings["source_indices"]]
        positions = [(int(generation["source"]["z_index"]),
                      int(generation["source"]["x_indices"][i])) for i in source_indices]
        for model_index in settings["model_indices"]:
            model_index = int(model_index)
            velocity = np.asarray(base._velocity[model_index, 0], dtype=np.float64)
            for direction_index in range(int(settings["directions_per_model"])):
                direction = gaussian_filter(
                    rng.standard_normal(velocity.shape),
                    float(settings["direction_smoothing_sigma"]), mode="reflect",
                )
                direction /= np.sqrt(np.mean(direction ** 2))
                for frequency_index in settings["frequency_indices"]:
                    frequency_index = int(frequency_index)
                    reference = numerical_directional_derivative(
                        velocity, direction, float(dataset.frequencies_hz[frequency_index]),
                        positions, generation["grid"]["spacing_km"],
                        float(generation["physics"]["background_velocity_km_s"]),
                        generation["solver"],
                        float(generation["source"]["singularity_radius_cells"]),
                    )
                    for source_offset, source_index in enumerate(source_indices):
                        condition = np.concatenate((base._background[frequency_index, source_index],
                                                    velocity[None].astype(np.float32)))
                        tangent = np.zeros_like(condition)
                        tangent[2] = direction
                        conditions.append(condition)
                        directions.append(tangent)
                        references.append(reference[source_offset])
                        keys.append((model_index, frequency_index, source_index, direction_index))
        if not conditions:
            raise ValueError("sensitivity_monitor requires at least one probe")
        arrays = dict(condition=np.stack(conditions), direction=np.stack(directions),
                      reference=np.stack(references), keys=np.asarray(keys))
        np.savez_compressed(output / "sensitivity_probes.npz", **arrays)
        self.conditions = torch.as_tensor(arrays["condition"], device=device)
        self.directions = torch.as_tensor(arrays["direction"], device=device)
        self.references = torch.as_tensor(arrays["reference"], device=device)
        self.normalizer = normalizer
        # ``predictor`` is an optional inference adapter for models whose
        # public forward signature is not ``model(condition)``.  It is used
        # by the one-step diffusion GNO with fixed noise and timestep.  The
        # returned sensitivity value remains diagnostic only.
        self.predictor = predictor
        self.count = len(conditions)

    def evaluate(self, model):
        was_training = model.training
        model.eval()
        errors = []
        try:
            with torch.enable_grad():
                for index in range(self.count):
                    def forward(condition):
                        if self.predictor is not None:
                            return self.predictor(model, condition, index)
                        return self.normalizer.decode_target(
                            model(self.normalizer.encode_condition(condition))
                        )
                    _, derivative = torch.autograd.functional.jvp(
                        forward, self.conditions[index:index + 1],
                        self.directions[index:index + 1], create_graph=False,
                    )
                    reference = self.references[index:index + 1]
                    error = (derivative - reference).square().sum() / reference.square().sum().clamp_min(1.0e-20)
                    errors.append(float(error.detach()))
        finally:
            model.train(was_training)
        value = float(np.mean(errors))
        if not np.isfinite(value):
            raise FloatingPointError("non-finite sensitivity loss")
        return value
