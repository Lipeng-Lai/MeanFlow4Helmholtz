from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent

METHODS = {
    "diffusion": {
        "log_path": ROOT / "checkpoint/gno_ema999/runlog.txt",
        "color": "#0072B2",
    },
    "meanflow": {
        "log_path": ROOT / "checkpoint/meanflow_improved_ema999/runlog.txt",
        "color": "#D55E00",
    },
}

DISPLAY_NAMES = {
    "diffusion": "Diffusion",
    "meanflow": "Mean Flows",
}

VALIDATION_PATTERN = re.compile(
    r"step=(?P<step>\d+)\s+"
    r"val mse=(?P<val_mse>[0-9.eE+-]+),\s+"
    r"relative_l2=(?P<relative_l2>[0-9.eE+-]+)"
)


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    """Read validation metrics from a training run log."""
    if not path.is_file():
        raise FileNotFoundError(f"Training log not found: {path}")

    rows: dict[int, dict[str, float]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = VALIDATION_PATTERN.search(line)
        if match:
            step = int(match.group("step"))
            rows[step] = {
                "val_mse": float(match.group("val_mse")),
                "relative_l2": float(match.group("relative_l2")),
            }

    if not rows:
        raise ValueError(f"No validation metrics found in: {path}")
    return rows


def main() -> None:
    curves = {}
    for name, spec in METHODS.items():
        rows = parse_log(spec["log_path"])
        curves[name] = rows
        print(
            f"{name}: {spec['log_path'].name}, "
            f"steps={len(rows)}, range={min(rows)}-{max(rows)}"
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    panels = (
        ("val_mse", "Validation MSE"),
        ("relative_l2", r"Relative $\mathit{L}_{2}$ Error"),
    )

    for ax, (metric, ylabel) in zip(axes, panels):
        for name, spec in METHODS.items():
            rows = curves[name]
            steps = sorted(rows)
            ax.plot(
                steps,
                [rows[step][metric] for step in steps],
                color=spec["color"],
                linewidth=2.0,
                label=DISPLAY_NAMES[name],
            )

        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        if metric == "val_mse":
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.30)
        ax.legend(frameon=False, loc="best")

    for index, ax in enumerate(axes):
        ax.text(
            -0.12,
            1.06,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )

    fig.tight_layout(w_pad=2.2)
    output_stem = OUTPUT_DIR / "loss_curves"
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=1200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_stem.with_suffix(".csv")
    common_steps = sorted(set.intersection(*(set(rows) for rows in curves.values())))
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "diffusion_val_mse",
                "diffusion_relative_l2",
                "meanflow_val_mse",
                "meanflow_relative_l2",
            ]
        )
        for step in common_steps:
            writer.writerow(
                [
                    step,
                    curves["diffusion"][step]["val_mse"],
                    curves["diffusion"][step]["relative_l2"],
                    curves["meanflow"][step]["val_mse"],
                    curves["meanflow"][step]["relative_l2"],
                ]
            )

    print(f"saved: {output_stem.with_suffix('.pdf')}")
    print(f"saved: {output_stem.with_suffix('.png')}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
