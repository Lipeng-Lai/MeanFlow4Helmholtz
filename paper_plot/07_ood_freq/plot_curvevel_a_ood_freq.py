"""Plot sampling accuracy from 8 Hz through 17 Hz.

By default the sample/source match ``paper_plot/03_CurveVel_A``. The
FlatFault-A, Marmousi and Overthrust entry scripts reuse the selections in
their respective comparison plots. Stored fields are matched by physical
frequency; missing frequencies are generated with the Helmholtz solver.
The result is a 2-by-5 sampling-accuracy figure with vertical legend bars
matching image.png. Frequencies 8--12 Hz occupy the top row and 13--17 Hz
the bottom row. FNO is a direct predictor, shown only as a green dot at one
step. Its green matches paper_plot/02_loss.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT_DIR = ROOT / "paper_plot" / "03_CurveVel_A"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import plot_curvevel_a as PLOT  # noqa: E402
from dataset.generate.helmholtz import solve_scattered_wavefields  # noqa: E402


FREQUENCIES_HZ = tuple(range(8, 18))
PANEL_FREQUENCY_INDICES = tuple(range(len(FREQUENCIES_HZ)))
SAMPLING_STEPS = tuple(PLOT.SAMPLING_STEPS)
MODEL_INDEX = PLOT.MODEL_INDEX
SOURCE_INDEX = PLOT.SOURCE_INDEX
SOURCE_X_INDICES = tuple(PLOT.SOURCE_X_INDICES)
SOURCE_Z_INDEX = PLOT.SOURCE_Z_INDEX
OUTPUT_STEM = HERE / "sampling_curves_curvevel_a"
DATA_ROOT = ROOT / "dataset" / PLOT.DATASET_SPLIT
CASE_NAME = "CurveVel-A"
CASE_SCRIPTS = {
    "curvevel_a": SCRIPT_DIR / "plot_curvevel_a.py",
    "flatfault_a": ROOT / "paper_plot" / "04_FlatFault_A" / "plot_curvefault_a.py",
    "marmousi": ROOT / "paper_plot" / "05_Marmousi" / "plot_marmousi.py",
    "overthrust": ROOT / "paper_plot" / "06_Overthrust" / "plot_overthrust.py",
}
GENERATION_CONFIG_PATH = ROOT / "configs" / "generate_dataset.yaml"
METHODS = ("diffusion", "meanflow", "fno")
CHECKPOINTS = {
    "diffusion": "gno_ema999",
    "meanflow": "meanflow_improved_ema999",
    "fno": "fno",
}
METHOD_COLORS = {"diffusion": "#0072B2", "meanflow": "#D55E00", "fno": "#009E73"}
METHOD_DISPLAY_NAMES = {"diffusion": "Diffusion", "meanflow": "Mean Flows", "fno": "FNO"}


def configure_case(case: str) -> None:
    """Reuse the model and source selected in the corresponding comparison plot."""
    global PLOT, MODEL_INDEX, SOURCE_INDEX, SOURCE_X_INDICES, SOURCE_Z_INDEX
    global SAMPLING_STEPS, OUTPUT_STEM, DATA_ROOT, CASE_NAME
    script = CASE_SCRIPTS[case]
    spec = importlib.util.spec_from_file_location(f"{case}_comparison_config", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load comparison configuration: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PLOT = module if case == "curvevel_a" else module.PLOT
    MODEL_INDEX = PLOT.MODEL_INDEX
    SOURCE_INDEX = PLOT.SOURCE_INDEX
    SOURCE_X_INDICES = tuple(PLOT.SOURCE_X_INDICES)
    SOURCE_Z_INDEX = PLOT.SOURCE_Z_INDEX
    SAMPLING_STEPS = tuple(PLOT.SAMPLING_STEPS)
    DATA_ROOT = ROOT / "dataset" / PLOT.DATASET_SPLIT
    CASE_NAME = PLOT.VELOCITY_MODEL_TITLE.removesuffix(" velocity model")
    OUTPUT_STEM = HERE / f"sampling_curves_{case}"


def load_cases() -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    velocity_all = np.load(DATA_ROOT / "data" / "velocity.npy", mmap_mode="r")
    background_all = np.load(DATA_ROOT / "data" / "background.npy", mmap_mode="r")
    label_all = np.load(DATA_ROOT / "label.npy", mmap_mode="r")
    if background_all.shape[1] != len(SOURCE_X_INDICES):
        raise ValueError("stored source count does not match the configuration")
    if PLOT.SOURCE_VELOCITY_PATH is None:
        velocity_km_s = np.asarray(velocity_all[MODEL_INDEX, 0], dtype=np.float32)
    else:
        velocity_km_s = np.asarray(
            np.load(PLOT.SOURCE_VELOCITY_PATH, mmap_mode="r"), dtype=np.float32
        ) * 0.001

    with GENERATION_CONFIG_PATH.open(encoding="utf-8") as stream:
        generation_config = yaml.safe_load(stream)
    grid_spacing = tuple(generation_config["grid"]["spacing_km"])
    background_velocity = float(
        generation_config["physics"]["background_velocity_km_s"]
    )
    solver_config = generation_config["solver"]
    singularity_radius = float(
        generation_config["source"].get("singularity_radius_cells", 0.5)
    )
    source_position = (SOURCE_Z_INDEX, SOURCE_X_INDICES[SOURCE_INDEX])

    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    stored_index = PLOT.stored_frequency_indices(background_all, generation_config)
    for frequency_hz in FREQUENCIES_HZ:
        if frequency_hz in stored_index:
            frequency_index = stored_index[frequency_hz]
            background = np.asarray(
                background_all[frequency_index, SOURCE_INDEX], dtype=np.float32
            )
            target = np.asarray(
                label_all[MODEL_INDEX, frequency_index, SOURCE_INDEX],
                dtype=np.float32,
            )
            residual_text = f"stored {PLOT.DATASET_SPLIT} field"
        else:
            background_complex, scattered, residuals = solve_scattered_wavefields(
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
            if not np.isfinite(residuals).all() or maximum_residual > tolerance:
                raise RuntimeError(
                    f"Helmholtz residual at {frequency_hz} Hz is "
                    f"{maximum_residual:.3e}, above {tolerance:.3e}"
                )
            background = np.stack(
                (background_complex[0].real, background_complex[0].imag), axis=0
            ).astype(np.float32)
            target = np.stack(
                (scattered[0].real, scattered[0].imag), axis=0
            ).astype(np.float32)
            residual_text = f"relative residual={maximum_residual:.3e}"

        conditions.append(
            np.concatenate((background, velocity_km_s[None]), axis=0).astype(
                np.float32
            )
        )
        targets.append(target)
        print(f"{CASE_NAME} {frequency_hz} Hz: {residual_text}")

    noise = torch.randn(
        (len(FREQUENCIES_HZ), 2, *velocity_km_s.shape),
        generator=torch.Generator().manual_seed(42),
    )
    return torch.from_numpy(np.stack(conditions)), np.stack(targets), noise


@torch.no_grad()
def predict_fno(loaded, condition: torch.Tensor) -> np.ndarray:
    config, model, normalizer, _ = loaded
    if config["method"]["type"] != "fno":
        raise ValueError("the FNO baseline requires an FNO checkpoint")
    encoded = normalizer.encode_condition(condition.unsqueeze(0).to(PLOT.DEVICE))
    prediction = model(encoded)
    return normalizer.decode_target(prediction.float())[0].cpu().numpy()


def plot_accuracy(
    values: dict[str, dict[int, np.ndarray]], output_stem: Path
) -> None:
    PLOT.configure_style()
    for method in METHODS:
        for steps in SAMPLING_STEPS:
            errors = np.asarray(values[method][steps])
            if errors.shape != (len(FREQUENCIES_HZ),) or not np.isfinite(errors).all():
                raise ValueError(f"invalid accuracy values for {method}, {steps} steps")
    # Match the available row height to the column width; explicitly keep
    # every plotting box square independently of its data limits.
    figure = plt.figure(figsize=(20.0, 7.05))
    grid = figure.add_gridspec(
        2, 6,
        width_ratios=[1, 1, 1, 1, 1, 0.55],
        left=0.045, right=0.99, bottom=0.095, top=0.94,
        wspace=0.32, hspace=0.25,
    )
    axes = grid[:, :5].subgridspec(2, 5, wspace=0.32, hspace=0.25).subplots(sharex=True)
    for panel_index, frequency_index in enumerate(PANEL_FREQUENCY_INDICES):
        row, column = divmod(panel_index, 5)
        axis = axes[row, column]
        axis.set_box_aspect(1)
        frequency_hz = FREQUENCIES_HZ[frequency_index]
        for method in METHODS:
            if method == "fno":
                axis.plot(
                    [1], [values[method][1][frequency_index]],
                    linestyle="none", marker="o", markersize=5.5,
                    color=METHOD_COLORS[method],
                    markerfacecolor=METHOD_COLORS[method],
                    markeredgecolor=METHOD_COLORS[method],
                    label=METHOD_DISPLAY_NAMES[method], zorder=4,
                )
                continue
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
        axis.set_xlim(min(SAMPLING_STEPS) - 0.5, max(SAMPLING_STEPS) + 0.5)
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.30, zorder=0)
        axis.set_axisbelow(True)
        axis.yaxis.set_major_locator(
            MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
        )
        maximum = max(
            float(values[method][steps][frequency_index])
            for method in METHODS
            for steps in SAMPLING_STEPS
        )
        axis.set_ylim(0.0, max(maximum * 1.18, 1e-12))
        axis.tick_params(direction="out", length=3, width=0.7)
        if row == 1:
            axis.set_xlabel("Sampling steps")
        else:
            axis.tick_params(labelbottom=False)
        if column == 0:
            axis.set_ylabel(r"Relative $L_2$ Error")

    legend_axis = figure.add_subplot(grid[:, 5])
    legend_position = legend_axis.get_position()
    legend_axis.set_position([
        axes[0, -1].get_position().x1 + 0.015,
        legend_position.y0, legend_position.width, legend_position.height,
    ])
    legend_axis.set_axis_off()
    for method, center_y in zip(METHODS, (0.80, 0.50, 0.20)):
        if method == "fno":
            legend_axis.plot(
                [0.08], [center_y], transform=legend_axis.transAxes,
                linestyle="none", marker="o", markersize=8,
                color=METHOD_COLORS[method], clip_on=False,
            )
        else:
            legend_axis.plot(
                [0.08, 0.08], [center_y - 0.13, center_y + 0.13],
                transform=legend_axis.transAxes,
                color=METHOD_COLORS[method], linewidth=7,
                solid_capstyle="butt", clip_on=False,
            )
        legend_axis.text(
            0.32, center_y, METHOD_DISPLAY_NAMES[method],
            transform=legend_axis.transAxes, va="center", ha="left", fontsize=14,
        )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), dpi=1200, bbox_inches="tight")
    plt.close(figure)
    print(f"saved: {output_stem.with_suffix('.png')}")
    print(f"saved: {output_stem.with_suffix('.pdf')}")


def save_accuracy(values: dict[str, dict[int, np.ndarray]], output_stem: Path) -> None:
    """Save the physical real/imag relative L2 errors used in the figure."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    path = output_stem.with_suffix(".csv")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "model_index", "source_index", "source_x", "source_z", "frequency_hz",
            "sampling_steps", "diffusion_relative_l2", "meanflow_relative_l2", "fno_relative_l2",
        ])
        for index, frequency in enumerate(FREQUENCIES_HZ):
            for steps in SAMPLING_STEPS:
                writer.writerow([
                    MODEL_INDEX, SOURCE_INDEX, SOURCE_X_INDICES[SOURCE_INDEX], SOURCE_Z_INDEX,
                    frequency, steps, *(float(values[method][steps][index]) for method in METHODS),
                ])
    print(f"saved: {path}")


def main(case: str = "curvevel_a") -> None:
    configure_case(case)
    conditions, targets, noises = load_cases()
    values: dict[str, dict[int, np.ndarray]] = {}
    for method in METHODS:
        loaded = PLOT.load_checkpoint(CHECKPOINTS[method])
        values[method] = {}
        if method == "fno":
            errors = np.asarray([
                PLOT.relative_l2(predict_fno(loaded, condition), target)
                for condition, target in zip(conditions, targets)
            ])
            values[method] = {steps: errors.copy() for steps in SAMPLING_STEPS}
        else:
            for steps in SAMPLING_STEPS:
                errors = []
                for condition, noise, target in zip(conditions, noises, targets):
                    prediction = PLOT.predict(
                        loaded, condition, noise.clone(), sampling_steps=steps
                    )
                    errors.append(PLOT.relative_l2(prediction, target))
                values[method][steps] = np.asarray(errors)
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
        if PLOT.DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    save_accuracy(values, OUTPUT_STEM)
    plot_accuracy(values, OUTPUT_STEM)


if __name__ == "__main__":
    main()
