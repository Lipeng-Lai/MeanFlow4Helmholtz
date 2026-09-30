"""Plot OOD-frequency sampling accuracy for the Overthrust test model.

The figure evaluates the third source of
``dataset/test/velocity_models_70x70/overthrust.npy`` at every frequency from
8 Hz through 17 Hz.  Odd frequencies are not stored in ``dataset/test`` and
are therefore generated with the repository Helmholtz solver.  Each panel
shows relative L2 error versus sampling steps for Diffusion and Mean Flows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = OUTPUT_DIR / "sampling_curves_overthrust"
VELOCITY_PATH = ROOT / "dataset" / "test" / "velocity_models_70x70" / "overthrust.npy"
GENERATION_CONFIG_PATH = ROOT / "configs" / "generate_helmholtz.yaml"

FREQUENCIES_HZ = tuple(range(8, 18))
SAMPLING_STEPS = (1, 2, 5, 10)
PANEL_FREQUENCY_INDICES = (0, 2, 4, 6, 8, 1, 3, 5, 7, 9)
SOURCE_INDEX = 2  # third source, using zero-based indexing
SOURCE_X_INDICES = (21, 31, 41, 51)
SOURCE_Z_INDEX = 2

METHODS = ("diffusion", "meanflow")
CHECKPOINTS = {
    "diffusion": "gno_ema999",
    "meanflow": "meanflow_improved_ema999",
}
METHOD_COLORS = {"diffusion": "#0072B2", "meanflow": "#D55E00"}
METHOD_DISPLAY_NAMES = {"diffusion": "Diffusion", "meanflow": "Mean Flows"}

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(ROOT))

from dataset.generate.helmholtz import solve_scattered_wavefields  # noqa: E402
from diffusion import GaussianDiffusion  # noqa: E402
from diffusion.sampling import ddim_sample  # noqa: E402
from meanflow import improved_meanflow_sample  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.normalization import PhysicalNormalizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=str(DEVICE),
        help="torch device, for example cuda or cpu (default: auto-detected)",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=OUTPUT_STEM,
        help="output path without the .png/.pdf suffix",
    )
    return parser.parse_args()


def load_generation_config() -> dict:
    with GENERATION_CONFIG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_cases() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Build conditions and numerical targets for all ten OOD frequencies."""
    generation_config = load_generation_config()
    velocity_m_s = np.asarray(np.load(VELOCITY_PATH), dtype=np.float64)
    velocity_km_s = velocity_m_s * 0.001

    source_position = (
        SOURCE_Z_INDEX,
        SOURCE_X_INDICES[SOURCE_INDEX],
    )
    grid_spacing = tuple(generation_config["grid"]["spacing_km"])
    background_velocity = float(
        generation_config["physics"]["background_velocity_km_s"]
    )
    solver_config = generation_config["solver"]
    singularity_radius = float(
        generation_config["source"].get("singularity_radius_cells", 0.5)
    )

    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for frequency_hz in FREQUENCIES_HZ:
        background, scattered, residuals = solve_scattered_wavefields(
            velocity_km_s=velocity_km_s,
            source_positions_rc=[source_position],
            frequency_hz=float(frequency_hz),
            background_velocity_km_s=background_velocity,
            spacing_km=grid_spacing,
            solver_config=solver_config,
            singularity_radius_cells=singularity_radius,
        )
        maximum_residual = float(np.max(residuals))
        tolerance = float(solver_config["relative_residual_tolerance"])
        if maximum_residual > tolerance:
            raise RuntimeError(
                f"Helmholtz residual at {frequency_hz} Hz is "
                f"{maximum_residual:.3e}, above {tolerance:.3e}"
            )

        conditions.append(
            np.concatenate(
                (
                    background[0].real[None],
                    background[0].imag[None],
                    velocity_km_s[None],
                ),
                axis=0,
            ).astype(np.float32)
        )
        targets.append(
            np.stack((scattered[0].real, scattered[0].imag), axis=0).astype(
                np.float32
            )
        )
        print(
            f"generated {frequency_hz} Hz, "
            f"relative residual={maximum_residual:.3e}"
        )

    condition_tensor = torch.from_numpy(np.stack(conditions))
    target_array = np.stack(targets)
    return condition_tensor, target_array, velocity_km_s


def load_checkpoint(checkpoint_name: str, device: torch.device):
    checkpoint_dir = ROOT / "checkpoint" / checkpoint_name
    config = load_config(checkpoint_dir / "config.yaml")
    state = torch.load(
        checkpoint_dir / "model_best.pth",
        map_location=device,
        weights_only=False,
    )
    model = build_model(config["model"]).to(device).float()
    model.load_state_dict(state.get("ema", state["model"]))
    model.eval()
    normalizer = PhysicalNormalizer(state["normalization"], device)
    diffusion = (
        GaussianDiffusion.from_config(config["diffusion"])
        if config["method"]["type"] == "gno"
        else None
    )
    return config, model, normalizer, diffusion


@torch.no_grad()
def predict_all_steps(
    loaded,
    conditions: torch.Tensor,
    noises: torch.Tensor,
    device: torch.device,
) -> dict[int, np.ndarray]:
    config, model, normalizer, diffusion = loaded
    condition_normalized = normalizer.encode_condition(conditions.to(device))
    predictions: dict[int, np.ndarray] = {}
    for sampling_steps in SAMPLING_STEPS:
        if config["method"]["type"] == "gno":
            prediction_normalized, _ = ddim_sample(
                model=model,
                diffusion=diffusion,
                condition_normalized=condition_normalized,
                noise=noises.to(device),
                sampling_steps=sampling_steps,
                eta=0.0,
                clip_denoised=bool(config["diffusion"]["clip_denoised"]),
            )
        else:
            prediction_normalized, _ = improved_meanflow_sample(
                model=model,
                noise=noises.to(device),
                condition=condition_normalized,
                sampling_steps=sampling_steps,
                time_scale=float(config["flow"]["time_scale"]),
            )
        predictions[sampling_steps] = (
            normalizer.decode_target(prediction_normalized.float())
            .cpu()
            .numpy()
        )
    return predictions


def relative_l2_per_frequency(
    predictions: dict[int, np.ndarray], targets: np.ndarray
) -> dict[int, np.ndarray]:
    target_norm = np.linalg.norm(targets.reshape(len(targets), -1), axis=1)
    target_norm = np.maximum(target_norm, 1.0e-12)
    return {
        steps: np.linalg.norm(
            (prediction - targets).reshape(len(targets), -1), axis=1
        )
        / target_norm
        for steps, prediction in predictions.items()
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "savefig.dpi": 1200,
        }
    )


def plot_accuracy(values: dict[str, dict[int, np.ndarray]], output_stem: Path) -> None:
    configure_style()
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(20.0, 7.5),
        sharex=True,
        constrained_layout=False,
    )

    for panel_index, frequency_index in enumerate(PANEL_FREQUENCY_INDICES):
        row, column = divmod(panel_index, 5)
        axis = axes[row, column]
        frequency_hz = FREQUENCIES_HZ[frequency_index]
        for method in METHODS:
            axis.plot(
                SAMPLING_STEPS,
                [
                    values[method][steps][frequency_index]
                    for steps in SAMPLING_STEPS
                ],
                color=METHOD_COLORS[method],
                linewidth=2.0,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=METHOD_COLORS[method],
                markeredgewidth=1.0,
                label=METHOD_DISPLAY_NAMES[method],
            )
        axis.set_title(f"{frequency_hz} Hz", pad=5)
        axis.set_xticks(SAMPLING_STEPS)
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.30, zorder=0)
        axis.set_axisbelow(True)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        maximum = max(
            float(values[method][steps][frequency_index])
            for method in METHODS
            for steps in SAMPLING_STEPS
        )
        axis.set_ylim(0.0, maximum * 1.18)
        axis.tick_params(direction="out", length=3, width=0.7)
        if row == 1:
            axis.set_xlabel("Sampling steps")
        else:
            axis.tick_params(labelbottom=False)
        if column == 0:
            axis.set_ylabel(r"Relative $L_2$ Error")

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=3.0,
            label=METHOD_DISPLAY_NAMES[method],
        )
        for method in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.942, 0.50),
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.91,
        bottom=0.095,
        top=0.94,
        wspace=0.30,
        hspace=0.52,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=1200, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), dpi=1200, bbox_inches="tight")
    plt.close(figure)
    print(f"saved: {output_stem.with_suffix('.png')}")
    print(f"saved: {output_stem.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    print(f"device: {device}")
    print(
        f"velocity={VELOCITY_PATH}, source={SOURCE_INDEX + 1} "
        f"(zero-based index {SOURCE_INDEX})"
    )

    conditions, targets, _ = build_cases()
    noise_generator = torch.Generator().manual_seed(SEED)
    noises = torch.randn(
        (len(FREQUENCIES_HZ), 2, conditions.shape[-2], conditions.shape[-1]),
        generator=noise_generator,
    )

    values: dict[str, dict[int, np.ndarray]] = {}
    for method in METHODS:
        loaded = load_checkpoint(CHECKPOINTS[method], device)
        predictions = predict_all_steps(loaded, conditions, noises, device)
        values[method] = relative_l2_per_frequency(predictions, targets)
        for steps in SAMPLING_STEPS:
            print(
                f"{METHOD_DISPLAY_NAMES[method]}, {steps} step(s): "
                + ", ".join(
                    f"{frequency} Hz={error:.6f}"
                    for frequency, error in zip(
                        FREQUENCIES_HZ, values[method][steps]
                    )
                )
            )
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_accuracy(values, args.output_stem)


if __name__ == "__main__":
    main()
