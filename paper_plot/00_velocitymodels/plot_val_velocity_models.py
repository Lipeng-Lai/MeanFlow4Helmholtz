"""Plot the original validation velocity models in two 5-by-8 figures.

The generated validation arrays contain two variants for each base model:
the original model and a smoothed model.  This figure set shows the 80
independent validation models (the original variants), grouped into the two
families CurveVel-A and FlatFault-A.  Each family contains 40 models from its
two configured subgroups and is rendered as a 5-by-8 grid.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset" / "val"
OUTPUT_DIR = Path(__file__).resolve().parent
MODELS_CSV = DATASET_ROOT / "models.csv"
VELOCITY_PATH = DATASET_ROOT / "data" / "velocity.npy"

FAMILY_GROUPS = {
    "CurveVel-A": ("curvevel_a_1", "curvevel_a_2"),
    "FlatFault-A": ("flatfault_a_1", "flatfault_a_2"),
}
VARIANT = "original"
ROWS = 5
COLUMNS = 8
DX_KM = 0.01
DZ_KM = 0.01
VELOCITY_VMIN = 1500.0
VELOCITY_VMAX = 4500.0
PDF_DPI = 600
PNG_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for generated PNG/PDF figures",
    )
    return parser.parse_args()


def load_metadata() -> list[dict[str, str]]:
    with MODELS_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"model_index", "pair_index", "variant", "model_group", "source_model_index"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{MODELS_CSV} is missing required metadata columns")
    return rows


def select_family_rows(rows: list[dict[str, str]], groups: tuple[str, ...]) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["variant"] == VARIANT and row["model_group"] in groups
    ]
    selected.sort(key=lambda row: (groups.index(row["model_group"]), int(row["pair_index"])))
    if len(selected) != ROWS * COLUMNS:
        raise ValueError(
            f"expected {ROWS * COLUMNS} {VARIANT} models for {groups}, found {len(selected)}"
        )
    return selected


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "savefig.dpi": 600,
        }
    )


def plot_family(
    family: str,
    rows: list[dict[str, str]],
    velocity: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(
        ROWS,
        COLUMNS,
        figsize=(16.0, 10.5),
    )
    axes_flat = axes.ravel()
    images = []
    for axis, row in zip(axes_flat, rows):
        model_index = int(row["model_index"])
        model = np.asarray(velocity[model_index, 0], dtype=np.float32) * 1000.0
        height, width = model.shape
        image = axis.imshow(
            model,
            cmap="jet",
            vmin=VELOCITY_VMIN,
            vmax=VELOCITY_VMAX,
            extent=[0.0, width * DX_KM, height * DZ_KM, 0.0],
            aspect="equal",
            interpolation="nearest",
        )
        images.append(image)
        axis.set_title(
            f"{row['model_group']} | val {model_index}\nsource {row['source_model_index']}",
            pad=2,
        )
        axis.set_xlabel("x (km)", labelpad=1)
        axis.set_ylabel("z (km)", labelpad=1)
        axis.tick_params(direction="out", length=2, width=0.5)

    figure.subplots_adjust(
        left=0.035,
        right=0.91,
        bottom=0.045,
        top=0.925,
        wspace=0.32,
        hspace=0.62,
    )
    colorbar_axis = figure.add_axes([0.925, 0.12, 0.014, 0.75])
    colorbar = figure.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label("Velocity (m/s)", labelpad=4)
    colorbar.ax.tick_params(labelsize=7, width=0.6, length=2)
    figure.suptitle(
        f"{family} validation velocity models ({VARIANT}, n={len(rows)})",
        fontsize=14,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{family.lower().replace('-', '_')}_val_models"
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    # Keep the canvas at a fixed layout.  Re-running a tight bounding-box
    # layout during high-resolution PNG rendering can crash older Matplotlib
    # builds when a figure contains this many axes.
    figure.savefig(pdf_path, dpi=PDF_DPI)
    figure.savefig(png_path, dpi=PNG_DPI)
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    configure_style()
    metadata = load_metadata()
    velocity = np.load(VELOCITY_PATH, mmap_mode="r")
    if velocity.ndim != 4 or velocity.shape[1] != 1:
        raise ValueError(f"expected velocity array [N, 1, nz, nx], got {velocity.shape}")
    if len(metadata) != velocity.shape[0]:
        raise ValueError(
            f"metadata contains {len(metadata)} rows but velocity array contains {velocity.shape[0]} models"
        )

    for family, groups in FAMILY_GROUPS.items():
        rows = select_family_rows(metadata, groups)
        pdf_path, png_path = plot_family(family, rows, velocity, args.output_dir)
        print(f"{family}: {len(rows)} models, saved {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
