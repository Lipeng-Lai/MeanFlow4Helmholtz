"""Numerical building blocks for scattered Helmholtz wavefield data."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import hankel2


def _validate_grid(
    shape: Sequence[int], spacing_km: Sequence[float]
) -> tuple[tuple[int, int], tuple[float, float]]:
    if len(shape) != 2 or len(spacing_km) != 2:
        raise ValueError("shape and spacing_km must each have two entries")
    nz, nx = (int(shape[0]), int(shape[1]))
    dz, dx = (float(spacing_km[0]), float(spacing_km[1]))
    if nz < 3 or nx < 3 or dz <= 0.0 or dx <= 0.0:
        raise ValueError(f"invalid grid: shape={shape}, spacing_km={spacing_km}")
    return (nz, nx), (dz, dx)


@lru_cache(maxsize=8)
def _laplacian_five_point(
    nz: int, nx: int, dz: float, dx: float
) -> sparse.csc_matrix:
    sx = sparse.diags(
        [np.ones(nx - 1), np.ones(nx - 1)], [-1, 1], shape=(nx, nx), format="csc"
    )
    sz = sparse.diags(
        [np.ones(nz - 1), np.ones(nz - 1)], [-1, 1], shape=(nz, nz), format="csc"
    )
    ix = sparse.eye(nx, format="csc")
    iz = sparse.eye(nz, format="csc")
    horizontal = sparse.kron(iz, sx, format="csc") / dx**2
    vertical = sparse.kron(sz, ix, format="csc") / dz**2
    center = -2.0 * (dx**-2 + dz**-2) * sparse.eye(nz * nx, format="csc")
    return horizontal + vertical + center


@lru_cache(maxsize=8)
def _laplacian_nine_point(nz: int, nx: int, spacing: float) -> sparse.csc_matrix:
    """Return the standard isotropic nine-point (Mehrstellen) Laplacian."""
    sx = sparse.diags(
        [np.ones(nx - 1), np.ones(nx - 1)], [-1, 1], shape=(nx, nx), format="csc"
    )
    sz = sparse.diags(
        [np.ones(nz - 1), np.ones(nz - 1)], [-1, 1], shape=(nz, nz), format="csc"
    )
    ix = sparse.eye(nx, format="csc")
    iz = sparse.eye(nz, format="csc")
    axial = sparse.kron(iz, sx, format="csc") + sparse.kron(sz, ix, format="csc")
    diagonal = sparse.kron(sz, sx, format="csc")
    center = sparse.eye(nz * nx, format="csc")
    return (4.0 * axial + diagonal - 20.0 * center) / (6.0 * spacing**2)


def finite_difference_laplacian(
    shape: Sequence[int], spacing_km: Sequence[float], stencil: str
) -> sparse.csc_matrix:
    (nz, nx), (dz, dx) = _validate_grid(shape, spacing_km)
    if stencil == "five_point":
        return _laplacian_five_point(nz, nx, dz, dx)
    if stencil == "nine_point_isotropic":
        if not np.isclose(dz, dx):
            raise ValueError("nine_point_isotropic requires equal x/z grid spacing")
        return _laplacian_nine_point(nz, nx, dx)
    raise ValueError(f"unsupported finite-difference stencil: {stencil}")


def sponge_profile(
    shape: Sequence[int], width_cells: int, strength: float, power: float
) -> np.ndarray:
    """Dimensionless polynomial damping profile, zero outside the boundary band."""
    nz, nx = int(shape[0]), int(shape[1])
    width_cells = int(width_cells)
    if width_cells < 0 or width_cells * 2 >= min(nz, nx):
        raise ValueError(f"invalid absorbing width {width_cells} for shape {(nz, nx)}")
    if width_cells == 0:
        return np.zeros((nz, nx), dtype=np.float64)

    iz, ix = np.indices((nz, nx))
    distance = np.minimum.reduce((iz, ix, nz - 1 - iz, nx - 1 - ix))
    normalized = np.clip((width_cells - distance) / width_cells, 0.0, 1.0)
    return float(strength) * normalized ** float(power)


def background_green_wavefield(
    shape: Sequence[int],
    spacing_km: Sequence[float],
    source_rc: Sequence[int],
    frequency_hz: float,
    background_velocity_km_s: float,
    singularity_radius_cells: float = 0.5,
) -> np.ndarray:
    """Analytic 2-D constant-medium Green wavefield used as FNO input."""
    (nz, nx), (dz, dx) = _validate_grid(shape, spacing_km)
    row, col = int(source_rc[0]), int(source_rc[1])
    if not (0 <= row < nz and 0 <= col < nx):
        raise ValueError(f"source {source_rc} lies outside grid {(nz, nx)}")
    if frequency_hz <= 0.0 or background_velocity_km_s <= 0.0:
        raise ValueError("frequency and background velocity must be positive")

    z = (np.arange(nz, dtype=np.float64) - row) * dz
    x = (np.arange(nx, dtype=np.float64) - col) * dx
    radius = np.hypot(z[:, None], x[None, :])
    radius_floor = float(singularity_radius_cells) * min(dz, dx)
    radius = np.maximum(radius, radius_floor)
    wavenumber = 2.0 * np.pi * float(frequency_hz) / float(background_velocity_km_s)
    return 0.25j * hankel2(0, wavenumber * radius)


def build_helmholtz_matrix(
    velocity_km_s: np.ndarray,
    spacing_km: Sequence[float],
    frequency_hz: float,
    stencil: str,
    absorbing_width_cells: int,
    absorbing_strength: float,
    absorbing_power: float,
    absorbing_complex_sign: float,
) -> sparse.csc_matrix:
    velocity = np.asarray(velocity_km_s, dtype=np.float64)
    if velocity.ndim != 2 or not np.all(np.isfinite(velocity)) or np.any(velocity <= 0.0):
        raise ValueError("velocity must be a finite, positive 2-D array")

    laplacian = finite_difference_laplacian(velocity.shape, spacing_km, stencil)
    damping = sponge_profile(
        velocity.shape, absorbing_width_cells, absorbing_strength, absorbing_power
    )
    omega = 2.0 * np.pi * float(frequency_hz)
    stretch = 1.0 + 1j * float(absorbing_complex_sign) * damping
    wavenumber_squared = (omega / velocity) ** 2 * stretch**2
    return laplacian.astype(np.complex128) + sparse.diags(
        wavenumber_squared.ravel(), format="csc"
    )


def solve_scattered_wavefields(
    velocity_km_s: np.ndarray,
    source_positions_rc: Iterable[Sequence[int]],
    frequency_hz: float,
    background_velocity_km_s: float,
    spacing_km: Sequence[float],
    solver_config: dict,
    singularity_radius_cells: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve all sources for one velocity/frequency using one sparse LU factorization.

    Returns background fields, scattered fields, and one relative linear-system
    residual per source.  Arrays have shape ``[n_sources, nz, nx]``.
    """
    physical_velocity = np.asarray(velocity_km_s, dtype=np.float64)
    positions = [tuple(map(int, position)) for position in source_positions_rc]
    if not positions:
        raise ValueError("at least one source position is required")
    if physical_velocity.ndim != 2:
        raise ValueError("velocity must be a 2-D array")
    nz, nx = physical_velocity.shape
    if any(not (0 <= row < nz and 0 <= col < nx) for row, col in positions):
        raise ValueError(f"source positions must lie inside physical grid {(nz, nx)}")

    absorbing_config = solver_config["absorbing_layer"]
    padding = int(absorbing_config.get("padding_cells", 0))
    if padding < 0:
        raise ValueError("absorbing padding must be non-negative")
    if padding:
        velocity = np.pad(physical_velocity, padding, mode="edge")
        solve_positions = [(row + padding, col + padding) for row, col in positions]
    else:
        velocity = physical_velocity
        solve_positions = positions

    matrix = build_helmholtz_matrix(
        velocity,
        spacing_km=spacing_km,
        frequency_hz=frequency_hz,
        stencil=str(solver_config["stencil"]),
        absorbing_width_cells=padding,
        absorbing_strength=float(absorbing_config["strength"]),
        absorbing_power=float(absorbing_config["power"]),
        absorbing_complex_sign=float(
            absorbing_config.get("complex_sign", -1.0)
        ),
    )
    use_gpu = str(solver_config.get("device", "cpu")).lower().startswith("cuda")
    if use_gpu:
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpsparse
            from cupyx.scipy.sparse.linalg import splu as cpsplu
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "GPU numerical Helmholtz requires cupy-cuda12x in the active environment"
            ) from exc
        matrix_gpu = cpsparse.csc_matrix(
            (cp.asarray(matrix.data), cp.asarray(matrix.indices), cp.asarray(matrix.indptr)),
            shape=matrix.shape,
        )
        factorization = cpsplu(matrix_gpu)
    else:
        factorization = splu(matrix)

    background = np.stack(
        [
            background_green_wavefield(
                velocity.shape,
                spacing_km,
                position,
                frequency_hz,
                background_velocity_km_s,
                singularity_radius_cells,
            )
            for position in solve_positions
        ],
        axis=0,
    )
    omega = 2.0 * np.pi * float(frequency_hz)
    contrast = velocity**-2 - float(background_velocity_km_s) ** -2
    right_hand_side = -(omega**2) * contrast[None, :, :] * background
    rhs_matrix = right_hand_side.reshape(len(positions), -1).T
    if use_gpu:
        rhs_gpu = cp.asarray(rhs_matrix)
        solution_gpu = factorization.solve(rhs_gpu)
        scattered = cp.asnumpy(solution_gpu.T.reshape(background.shape))
        residual_matrix = matrix_gpu @ solution_gpu - rhs_gpu
        numerator = cp.asnumpy(cp.linalg.norm(residual_matrix, axis=0))
        denominator = cp.asnumpy(cp.linalg.norm(rhs_gpu, axis=0)) + np.finfo(np.float64).eps
    else:
        solution_matrix = factorization.solve(rhs_matrix)
        scattered = solution_matrix.T.reshape(background.shape)
        residual_matrix = matrix @ solution_matrix - rhs_matrix
        numerator = np.linalg.norm(residual_matrix, axis=0)
        denominator = np.linalg.norm(rhs_matrix, axis=0) + np.finfo(np.float64).eps
    if padding:
        core = (slice(None), slice(padding, -padding), slice(padding, -padding))
        background = background[core]
        scattered = scattered[core]
    return background, scattered, numerator / denominator


