"""Plot Gaussian noise and clean/noisy scattered wavefields.

The training labels store the complex scattered wavefield as two channels:
channel 0 is the real part and channel 1 is the imaginary part.  This script
uses one training example, adds reproducible Gaussian noise to the clean
scattered field, and writes six axis-free PNG files:

    gaussian_noise_{real,imag}.png
    clean_scattered_{real,imag}.png
    noisy_scattered_{real,imag}.png

Example
-------
    python paper_plot/plot_method/plot_method.py

The sample and noise strength can be changed from the command line without
changing the plotting code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset" / "train"
OUTPUT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_INDEX = 0
DEFAULT_FREQUENCY_INDEX = 4  # 16 Hz in configs/generate_helmholtz.yaml
DEFAULT_SOURCE_INDEX = 2
DEFAULT_SEED = 20260930
DEFAULT_NOISE_STD_RATIO = 0.50
FIELD_CMAP = "jet"
NOISE_CMAP = "jet"
DEFAULT_FIGSIZE = (3.0, 3.0)
SAVE_DPI = 1200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-index", type=int, default=DEFAULT_MODEL_INDEX)
    parser.add_argument(
        "--frequency-index",
        type=int,
        default=DEFAULT_FREQUENCY_INDEX,
        help="frequency index in the training array (0: 8 Hz, ..., 4: 16 Hz)",
    )
    parser.add_argument("--source-index", type=int, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--noise-std-ratio",
        type=float,
        default=DEFAULT_NOISE_STD_RATIO,
        help="Gaussian-noise std divided by the clean-field std",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="directory for the six PNG files",
    )
    return parser.parse_args()


def load_clean_field(
    model_index: int, frequency_index: int, source_index: int
) -> np.ndarray:
    """Load one ``(real, imaginary, z, x)`` scattered-field sample."""
    label_path = DATASET_ROOT / "label.npy"
    labels = np.load(label_path, mmap_mode="r")
    if not 0 <= model_index < labels.shape[0]:
        raise IndexError(f"model index {model_index} is outside [0, {labels.shape[0]})")
    if not 0 <= frequency_index < labels.shape[1]:
        raise IndexError(
            f"frequency index {frequency_index} is outside [0, {labels.shape[1]})"
        )
    if not 0 <= source_index < labels.shape[2]:
        raise IndexError(
            f"source index {source_index} is outside [0, {labels.shape[2]})"
        )
    return np.asarray(
        labels[model_index, frequency_index, source_index], dtype=np.float32
    )


def make_noise(
    clean_field: np.ndarray, seed: int, noise_std_ratio: float
) -> np.ndarray:
    if noise_std_ratio < 0:
        raise ValueError("--noise-std-ratio must be non-negative")
    rng = np.random.default_rng(seed)
    noise = np.empty_like(clean_field, dtype=np.float32)
    for component_index in range(clean_field.shape[0]):
        component_std = float(clean_field[component_index].std())
        noise[component_index] = rng.normal(
            loc=0.0,
            scale=noise_std_ratio * component_std,
            size=clean_field[component_index].shape,
        ).astype(np.float32)
    return noise


def robust_symmetric_limit(*fields: np.ndarray) -> float:
    """Return a shared symmetric limit while ignoring rare extreme pixels."""
    values = np.concatenate([np.abs(field).ravel() for field in fields])
    limit = float(np.percentile(values, 99.5))
    return max(limit, np.finfo(np.float32).eps)


def save_field(
    field: np.ndarray,
    output_path: Path,
    vlim: float,
    cmap: str,
    interpolation: str = "bilinear",
    border: bool = False,
) -> None:
    """Save an axis-free raster at exactly the requested export DPI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=DEFAULT_FIGSIZE, frameon=False)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(
        field,
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        origin="upper",
        interpolation=interpolation,
        resample=True,
        aspect="equal",
    )
    axis.set_axis_off()
    if border:
        axis.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=axis.transAxes,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
                clip_on=False,
                zorder=10,
            )
        )
    figure.savefig(
        output_path,
        dpi=SAVE_DPI,
        format="png",
        facecolor="white",
        edgecolor="none",
        pad_inches=0,
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    clean = load_clean_field(
        args.model_index, args.frequency_index, args.source_index
    )
    if clean.shape[0] != 2:
        raise ValueError(f"expected two field components, got shape {clean.shape}")

    noise = make_noise(clean, args.seed, args.noise_std_ratio)
    noisy = clean + noise
    args.output_dir.mkdir(parents=True, exist_ok=True)

    component_names = ("real", "imag")
    for component_index, component_name in enumerate(component_names):
        clean_component = clean[component_index]
        noise_component = noise[component_index]
        noisy_component = noisy[component_index]

        # The clean/noisy pair uses one scale for a fair visual comparison.
        field_limit = robust_symmetric_limit(clean_component, noisy_component)
        noise_limit = robust_symmetric_limit(noise_component)

        save_field(
            noise_component,
            args.output_dir / f"gaussian_noise_{component_name}.png",
            noise_limit,
            NOISE_CMAP,
            "nearest",
            True,
        )
        save_field(
            clean_component,
            args.output_dir / f"clean_scattered_{component_name}.png",
            field_limit,
            FIELD_CMAP,
            border=True,
        )
        save_field(
            noisy_component,
            args.output_dir / f"noisy_scattered_{component_name}.png",
            field_limit,
            FIELD_CMAP,
        )

    print(f"Saved six PNG files to {args.output_dir}")
    print(
        "sample="
        f"(model={args.model_index}, frequency={args.frequency_index}, "
        f"source={args.source_index}), seed={args.seed}, "
        f"noise_std_ratio={args.noise_std_ratio}"
    )


if __name__ == "__main__":
    main()
