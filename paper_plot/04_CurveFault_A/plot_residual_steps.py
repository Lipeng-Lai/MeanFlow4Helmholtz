"""Plot CurveFault-A residuals for diffusion and Mean Flows at multiple steps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "curvefault_a_residual_steps"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import plot_curvefault_a as WRAPPER  # noqa: E402


# The wrapper applies the CurveFault-A sample configuration to the shared plotter.
BASE = WRAPPER.PLOT
METHODS = ("diffusion", "meanflow")
CHECKPOINTS = {
    "diffusion": "gno_ema999",
    "meanflow": "meanflow_improved_ema999",
}
DISPLAY_NAMES = {"diffusion": "Diffusion", "meanflow": "Mean Flows"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("real", "imag"),
        default="real",
        help="wavefield component used for the residuals (default: real)",
    )
    return parser.parse_args()


def _predict_residuals(
    condition: torch.Tensor,
    noise: torch.Tensor,
    target: np.ndarray,
    component_index: int,
) -> dict[str, dict[int, np.ndarray]]:
    residuals: dict[str, dict[int, np.ndarray]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        loaded = BASE.load_checkpoint(CHECKPOINTS[method])
        for steps in BASE.SAMPLING_STEPS:
            prediction = BASE.predict(
                loaded,
                condition,
                noise.clone(),
                sampling_steps=steps,
            )
            residuals[method][steps] = (
                target[component_index] - prediction[component_index]
            )
            print(
                f"{method} {steps} step(s): "
                f"relative L2={BASE.relative_l2(prediction, target):.8f}"
            )
        del loaded
        if BASE.DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return residuals


def _plot_residuals(
    residuals: dict[str, dict[int, np.ndarray]],
    component_name: str,
) -> Path:
    BASE.configure_style()
    fig, axes = plt.subplots(
        2,
        len(BASE.SAMPLING_STEPS),
        figsize=(18.0, 8.3),
        constrained_layout=True,
    )

    images = []
    for row, method in enumerate(METHODS):
        for column, steps in enumerate(BASE.SAMPLING_STEPS):
            ax = axes[row, column]
            height, width = residuals[method][steps].shape
            image = ax.imshow(
                residuals[method][steps],
                cmap="jet",
                vmin=BASE.ERROR_VMIN,
                vmax=BASE.ERROR_VMAX,
                extent=[
                    0.0,
                    width * BASE.DX_KM,
                    height * BASE.DZ_KM,
                    0.0,
                ],
                aspect="equal",
                interpolation="nearest",
            )
            images.append(image)
            step_word = "step" if steps == 1 else "steps"
            ax.set_title(
                f"{DISPLAY_NAMES[method]} {steps} {step_word}",
                pad=4,
                fontsize=13,
            )
            ax.set_xlabel("x (km)", labelpad=2)
            ax.set_ylabel("z (km)", labelpad=2)
            ax.tick_params(direction="out", length=3, width=0.7)

    colorbar = fig.colorbar(
        images[0],
        ax=axes.ravel().tolist(),
        pad=0.025,
        fraction=0.046,
    )
    colorbar.set_label("Error", labelpad=4)
    colorbar.ax.tick_params(labelsize=7, width=0.6, length=2)

    stem = OUTPUT_DIR / OUTPUT_STEM
    fig.savefig(stem.with_suffix(".pdf"), dpi=1200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"component: {component_name}")
    print(f"saved: {stem.with_suffix('.pdf')}")
    print(f"saved: {stem.with_suffix('.png')}")
    return stem


def main() -> None:
    args = parse_args()
    component_index = 0 if args.component == "real" else 1
    component_name = BASE.COMPONENT_NAMES[component_index]
    _, _, target, condition, noise = BASE.load_sample()
    residuals = _predict_residuals(
        condition,
        noise,
        target,
        component_index,
    )
    _plot_residuals(residuals, component_name)


if __name__ == "__main__":
    main()
