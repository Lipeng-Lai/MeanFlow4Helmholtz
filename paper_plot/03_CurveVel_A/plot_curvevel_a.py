"""Plot the selected CurveVel-A comparison sample as a 2-by-4 figure.

The selected validation example is the requested sample from
``test_openfwi/gno_ema999_vs_meanflow_ema999_samples.csv``:

    sample_index=2306, model_index=115, frequency_index=1, source_index=2

The field panels use the real component by default.  Pass ``--component
imag`` to render the corresponding imaginary component instead.  The
    sampling-step panel evaluates the current sample without PDE guidance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "curvevel_a_comparison"
VELOCITY_MODEL_TITLE = "CurveVel-A velocity model"
DATASET_SPLIT = "val"
SOURCE_VELOCITY_PATH: Path | None = None
PREDICTION_STEPS = 1
sys.path.insert(0, str(ROOT))

from diffusion import GaussianDiffusion  # noqa: E402
from diffusion.sampling import ddim_sample  # noqa: E402
from meanflow import improved_meanflow_sample  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.normalization import PhysicalNormalizer  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_INDEX = 2306
MODEL_INDEX = 115
FREQUENCY_INDEX = 1
SOURCE_INDEX = 2
FREQUENCIES_HZ = [8, 10, 12, 14, 16]
SOURCE_COUNT = 4
COMPONENT_NAMES = ("Real", "Imaginary")

DX_KM = 0.01
DZ_KM = 0.01
SOURCE_Z_INDEX = 2
SOURCE_X_INDICES = [21, 31, 41, 51]

BACKGROUND_VMIN = -0.25
BACKGROUND_VMAX = 0.25
PREDICTION_VMIN = -0.20
PREDICTION_VMAX = 0.20
ERROR_VMIN = -0.02
ERROR_VMAX = 0.02
VELOCITY_VMIN = 1500.0
VELOCITY_VMAX = 4500.0

SAMPLING_STEPS = [1, 2, 5, 10]
METHOD_COLORS = {"diffusion": "#0072B2", "meanflow": "#D55E00"}
METHOD_DISPLAY_NAMES = {
    "diffusion": "Diffusion",
    "meanflow": "Mean Flows",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("real", "imag"),
        default="real",
        help="field component shown in the wavefield panels (default: real)",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "savefig.dpi": 600,
        }
    )


def load_checkpoint(checkpoint_name: str):
    checkpoint_dir = ROOT / "checkpoint" / checkpoint_name
    config = load_config(checkpoint_dir / "config.yaml")
    state = torch.load(
        checkpoint_dir / "model_best.pth",
        map_location=DEVICE,
        weights_only=False,
    )
    model = build_model(config["model"]).to(DEVICE).float()
    model.load_state_dict(state.get("ema", state["model"]))
    model.eval()
    normalizer = PhysicalNormalizer(state["normalization"], DEVICE)
    diffusion = (
        GaussianDiffusion.from_config(config["diffusion"])
        if config["method"]["type"] == "gno"
        else None
    )
    return config, model, normalizer, diffusion


def load_sample() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor
]:
    computed_sample_index = (
        MODEL_INDEX * (len(FREQUENCIES_HZ) * SOURCE_COUNT)
        + FREQUENCY_INDEX * SOURCE_COUNT
        + SOURCE_INDEX
    )
    if computed_sample_index != SAMPLE_INDEX:
        raise ValueError("the configured sample indices do not match SAMPLE_INDEX")

    split_root = ROOT / "dataset" / DATASET_SPLIT
    velocity_all = np.load(split_root / "data" / "velocity.npy", mmap_mode="r")
    background_all = np.load(split_root / "data" / "background.npy", mmap_mode="r")
    label_all = np.load(split_root / "label.npy", mmap_mode="r")

    if SOURCE_VELOCITY_PATH is None:
        velocity = np.asarray(velocity_all[MODEL_INDEX, 0], dtype=np.float32)
    else:
        velocity = np.asarray(
            np.load(SOURCE_VELOCITY_PATH, mmap_mode="r"), dtype=np.float32
        ) * 0.001
    background = np.asarray(
        background_all[FREQUENCY_INDEX, SOURCE_INDEX], dtype=np.float32
    )
    target = np.asarray(
        label_all[MODEL_INDEX, FREQUENCY_INDEX, SOURCE_INDEX], dtype=np.float32
    )
    condition = np.concatenate((background, velocity[None]), axis=0).astype(np.float32)

    noise_generator = torch.Generator().manual_seed(42 + SAMPLE_INDEX)
    noise = torch.randn((2, *velocity.shape), generator=noise_generator)
    return velocity, background, target, torch.from_numpy(condition), noise


@torch.no_grad()
def predict(
    loaded,
    condition: torch.Tensor,
    noise: torch.Tensor,
    sampling_steps: int,
) -> np.ndarray:
    config, model, normalizer, diffusion = loaded
    condition_device = condition.unsqueeze(0).to(DEVICE)
    noise_device = noise.unsqueeze(0).to(DEVICE)
    condition_normalized = normalizer.encode_condition(condition_device)

    # No physical/PDE guidance is passed to either sampler.
    if config["method"]["type"] == "gno":
        prediction_normalized, _ = ddim_sample(
            model=model,
            diffusion=diffusion,
            condition_normalized=condition_normalized,
            noise=noise_device,
            sampling_steps=sampling_steps,
            eta=0.0,
            clip_denoised=bool(config["diffusion"]["clip_denoised"]),
        )
    else:
        prediction_normalized, _ = improved_meanflow_sample(
            model=model,
            noise=noise_device,
            condition=condition_normalized,
            sampling_steps=sampling_steps,
            time_scale=float(config["flow"]["time_scale"]),
        )
    return normalizer.decode_target(prediction_normalized.float())[0].cpu().numpy()


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1.0e-12)
    )


def current_sampling_accuracy(
    diffusion_loaded,
    meanflow_loaded,
    condition: torch.Tensor,
    noise: torch.Tensor,
    target: np.ndarray,
) -> dict[str, dict[int, float]]:
    """Evaluate current-sample accuracy for every displayed step."""
    values: dict[str, dict[int, float]] = {"diffusion": {}, "meanflow": {}}
    for steps in SAMPLING_STEPS:
        diffusion_prediction = predict(
            diffusion_loaded, condition, noise.clone(), sampling_steps=steps
        )
        meanflow_prediction = predict(
            meanflow_loaded, condition, noise.clone(), sampling_steps=steps
        )
        values["diffusion"][steps] = relative_l2(diffusion_prediction, target)
        values["meanflow"][steps] = relative_l2(meanflow_prediction, target)
    return values


def plot_field(
    ax: plt.Axes,
    field: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
) -> None:
    height, width = field.shape
    extent = [0.0, width * DX_KM, height * DZ_KM, 0.0]
    image = ax.imshow(
        field,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title, pad=4, fontsize=13)
    ax.set_xlabel("x (km)", labelpad=2)
    ax.set_ylabel("z (km)", labelpad=2)
    ax.tick_params(direction="out", length=3, width=0.7)
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.025, fraction=0.046)
    colorbar.set_label(colorbar_label, labelpad=4)
    colorbar.ax.tick_params(labelsize=7, width=0.6, length=2)


def plot_sampling_accuracy(ax: plt.Axes, values: dict[str, dict[int, float]]) -> None:
    for method, method_values in values.items():
        steps = sorted(method_values)
        ax.plot(
            steps,
            [method_values[step] for step in steps],
            color=METHOD_COLORS[method],
            linewidth=2.0 if len(steps) > 1 else 0.0,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=METHOD_COLORS[method],
            markeredgewidth=1.2,
            label=METHOD_DISPLAY_NAMES[method],
            zorder=3,
        )
    ax.set_xlabel("Sampling steps")
    ax.set_ylabel(r"Relative $\mathit{L}_{2}$ Error")
    ax.set_title("Sampling-Step Comparison", pad=4, fontsize=13)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(SAMPLING_STEPS)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.30, zorder=0)
    ax.legend(frameon=False, loc="upper right", fontsize=13)
    ax.set_box_aspect(1)
    ax.tick_params(direction="out", length=3, width=0.7)


def add_source_marker(ax: plt.Axes) -> None:
    ax.plot(
        SOURCE_X_INDICES[SOURCE_INDEX] * DX_KM,
        SOURCE_Z_INDEX * DZ_KM,
        marker="*",
        markersize=24,
        markerfacecolor="red",
        markeredgecolor="black",
        markeredgewidth=0.6,
        zorder=5,
    )


def main() -> None:
    args = parse_args()
    configure_style()
    component_index = 0 if args.component == "real" else 1
    component_name = COMPONENT_NAMES[component_index]

    velocity, background, target, condition, noise = load_sample()
    diffusion = load_checkpoint("gno_ema999")
    meanflow = load_checkpoint("meanflow_improved_ema999")
    diffusion_prediction = predict(
        diffusion, condition, noise.clone(), sampling_steps=PREDICTION_STEPS
    )
    meanflow_prediction = predict(
        meanflow, condition, noise.clone(), sampling_steps=PREDICTION_STEPS
    )
    sampling_accuracy = current_sampling_accuracy(
        diffusion, meanflow, condition, noise, target
    )

    step_word = "step" if PREDICTION_STEPS == 1 else "steps"

    target_field = target[component_index]
    fields = (
        (
            background[component_index],
            f"Background wavefield ({component_name})",
            BACKGROUND_VMIN,
            BACKGROUND_VMAX,
            "Amplitude",
        ),
        (target_field, "Ground Truth", PREDICTION_VMIN, PREDICTION_VMAX, "Amplitude"),
        (
            diffusion_prediction[component_index],
            f"Diffusion Prediction ({PREDICTION_STEPS} {step_word})",
            PREDICTION_VMIN,
            PREDICTION_VMAX,
            "Amplitude",
        ),
        (
            meanflow_prediction[component_index],
            f"Mean Flows Prediction ({PREDICTION_STEPS} {step_word})",
            PREDICTION_VMIN,
            PREDICTION_VMAX,
            "Amplitude",
        ),
        (
            velocity * 1000.0,
            VELOCITY_MODEL_TITLE,
            VELOCITY_VMIN,
            VELOCITY_VMAX,
            "Velocity (m/s)",
        ),
        None,
        (
            target_field - diffusion_prediction[component_index],
            "Ground Truth - Diffusion Prediction",
            ERROR_VMIN,
            ERROR_VMAX,
            "Error",
        ),
        (
            target_field - meanflow_prediction[component_index],
            "Ground Truth - Mean Flows Prediction",
            ERROR_VMIN,
            ERROR_VMAX,
            "Error",
        ),
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18.0, 8.3),
        constrained_layout=True,
    )
    for index, (ax, field_spec) in enumerate(zip(axes.flat, fields)):
        if field_spec is None:
            plot_sampling_accuracy(ax, sampling_accuracy)
        else:
            plot_field(ax, *field_spec)
            if index == 4:
                add_source_marker(ax)
        ax.text(
            -0.16,
            1.04,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    stem = OUTPUT_DIR / OUTPUT_STEM
    fig.savefig(stem.with_suffix(".pdf"), dpi=1200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"device: {DEVICE}")
    print(
        f"sample={SAMPLE_INDEX}, model={MODEL_INDEX}, "
        f"frequency={FREQUENCIES_HZ[FREQUENCY_INDEX]} Hz, source={SOURCE_INDEX}, "
        f"component={component_name}"
    )
    print(f"Diffusion Relative L2: {relative_l2(diffusion_prediction, target):.8f}")
    print(f"Mean Flows Relative L2:  {relative_l2(meanflow_prediction, target):.8f}")
    for method, values in sampling_accuracy.items():
        print(f"{method} sampling-step Relative L2: {values}")
    print(f"saved: {stem.with_suffix('.pdf')}")
    print(f"saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
