"""Benchmark inference steps and parallel 3--16 Hz Mean Flows generation.

Panel (a) times a ten-model batch at 12 Hz. Panel (b) batches frequencies for each velocity model in a single
Mean Flows call and compares with sequential CPU Helmholtz solves at one and four threads.
All reported timings are medians of ten timed runs after three warm-up runs.
FNO is a direct operator and appears only at one step in panel (a).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_info, threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = OUTPUT_DIR / "multifrequency_efficiency"
DETAILS_CSV = OUTPUT_DIR / "multifrequency_efficiency.csv"
SAMPLING_CSV = OUTPUT_DIR / "sampling_step_efficiency.csv"
RAW_CSV = OUTPUT_DIR / "efficiency_timing_runs.csv"
METADATA_PATH = OUTPUT_DIR / "efficiency_benchmark_metadata.json"
DATA_CONFIG = "configs/gno_ema999.yaml"

METHODS = ("fno", "diffusion", "meanflow")
FREQUENCIES_HZ = tuple(range(3, 17))
FREQUENCY_COUNTS = tuple(range(1, len(FREQUENCIES_HZ) + 1))
SAMPLING_STEPS = (1, 2, 4, 8)
FIXED_SAMPLING_FREQUENCY_HZ = 12
MODEL_INDICES = tuple(range(10))
SOURCE_INDEX = 0
MULTIFREQUENCY_SAMPLING_STEPS = 1
WARMUP_RUNS = 3
TIMED_RUNS = 10
CPU_THREADS = 4
NUMERICAL_THREADS = (1, 4)
TIMING_RECORDS: list[dict] = []

COLORS = {
    "diffusion": "#0072B2",
    "meanflow": "#D55E00",
    "numerical": "#009E73",
    "fno": "#009E73",
    "numerical_4threads": "#0072B2",
    "speedup": "#D62728",
    "speedup_4threads": "#CC79A7",
}
DISPLAY_NAMES = {
    "fno": "FNO",
    "diffusion": "Diffusion",
    "meanflow": "Mean Flows",
}


def _load_benchmark_module():
    script = OUTPUT_DIR / "plot_efficency.py"
    spec = importlib.util.spec_from_file_location("multifrequency_benchmark", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load runtime benchmark: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()
sys.path.insert(0, str(ROOT))

from dataset.generate.helmholtz import (  # noqa: E402
    background_green_wavefield,
    solve_scattered_wavefields,
)
from utils.config import load_config, resolve_path  # noqa: E402


def _load_cases(
    device: torch.device,
) -> tuple[list[list[dict]], dict, tuple[int, int]]:
    data_config = load_config(DATA_CONFIG)
    split_root = resolve_path(data_config["data"]["val_root"])
    velocity_all = np.load(split_root / "data" / "velocity.npy", mmap_mode="r")

    with (ROOT / "configs" / "generate_dataset.yaml").open(
        encoding="utf-8"
    ) as file:
        generation_config = yaml.safe_load(file)

    spacing_km = tuple(generation_config["grid"]["spacing_km"])
    background_velocity = float(
        generation_config["physics"]["background_velocity_km_s"]
    )
    source_config = generation_config["source"]
    source_position = (
        int(source_config["z_index"]),
        int(source_config["x_indices"][SOURCE_INDEX]),
    )
    singularity_radius = float(
        source_config.get("singularity_radius_cells", 0.5)
    )

    cases_by_model: list[list[dict]] = []
    for model_index in MODEL_INDICES:
        velocity = np.asarray(velocity_all[model_index, 0], dtype=np.float64)
        model_cases = []
        for frequency_index, frequency_hz in enumerate(FREQUENCIES_HZ):
            background = background_green_wavefield(
                velocity.shape,
                spacing_km,
                source_position,
                float(frequency_hz),
                background_velocity,
                singularity_radius,
            )
            condition = np.stack(
                (background.real, background.imag, velocity), axis=0
            ).astype(np.float32)
            noise_generator = torch.Generator().manual_seed(
                42
                + model_index * len(FREQUENCIES_HZ)
                + frequency_index
            )
            noise = torch.randn(
                (2, *velocity.shape), generator=noise_generator
            ).to(device)
            model_cases.append(
                {
                    "frequency_hz": float(frequency_hz),
                    "velocity": velocity,
                    "condition": torch.from_numpy(condition).to(device),
                    "noise": noise,
                }
            )
        cases_by_model.append(model_cases)
    return cases_by_model, generation_config, source_position


def _prepare_learned_batches(
    loaded: tuple,
    cases_by_model: list[list[dict]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    _, _, normalizer, _ = loaded
    batches = []
    for model_cases in cases_by_model:
        condition = torch.stack([case["condition"] for case in model_cases])
        noise = torch.stack([case["noise"] for case in model_cases])
        batches.append((normalizer.encode_condition(condition), noise))
    return batches


def _measure_learned_counts(
    loaded: tuple,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    method: str,
    device: torch.device,
    warmup_runs: int,
    timed_runs: int,
) -> dict[int, float]:
    config, model, _, diffusion = loaded
    model_count = len(batches)
    results: dict[int, float] = {}

    def run_count(frequency_count: int) -> None:
        for condition, noise in batches:
            # The leading dimension contains all requested frequencies for
            # ONE model. Do not loop over individual frequencies here.
            BENCHMARK._sample(
                method,
                model,
                diffusion,
                condition[:frequency_count],
                noise[:frequency_count],
                config,
                MULTIFREQUENCY_SAMPLING_STEPS,
            )

    for frequency_count in FREQUENCY_COUNTS:
        for _ in range(warmup_runs):
            run_count(frequency_count)
        BENCHMARK._synchronize(device)

        measurements = []
        for repeat in range(timed_runs):
            BENCHMARK._synchronize(device)
            started = time.perf_counter()
            run_count(frequency_count)
            BENCHMARK._synchronize(device)
            measurements.append(
                (time.perf_counter() - started) / model_count
            )
            TIMING_RECORDS.append({
                "benchmark": "parallel_frequencies", "method": method,
                "sampling_steps": MULTIFREQUENCY_SAMPLING_STEPS,
                "frequency_count": frequency_count, "frequency_hz": "",
                "repeat": repeat, "seconds": measurements[-1], "unit": "s/model",
            })
        results[frequency_count] = statistics.median(measurements)
        print(
            f"{method}, frequencies={frequency_count}: "
            f"median={results[frequency_count]:.6f} s/model"
        )
    return results


def _measure_sampling_steps(
    loaded: tuple,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    method: str,
    device: torch.device,
    warmup_runs: int,
    timed_runs: int,
) -> dict[int, float]:
    """Measure a ten-model batch at 12 Hz for panel (a)."""
    config, model, _, diffusion = loaded
    frequency_index = FREQUENCIES_HZ.index(FIXED_SAMPLING_FREQUENCY_HZ)
    condition = torch.stack(
        [model_condition[frequency_index] for model_condition, _ in batches]
    )
    noise = torch.stack(
        [model_noise[frequency_index] for _, model_noise in batches]
    )
    results: dict[int, float] = {}

    steps = (1,) if method == "fno" else SAMPLING_STEPS
    for sampling_steps in steps:
        for _ in range(warmup_runs):
            BENCHMARK._sample(
                method,
                model,
                diffusion,
                condition,
                noise,
                config,
                sampling_steps,
            )
        BENCHMARK._synchronize(device)

        measurements = []
        for repeat in range(timed_runs):
            BENCHMARK._synchronize(device)
            started = time.perf_counter()
            BENCHMARK._sample(
                method,
                model,
                diffusion,
                condition,
                noise,
                config,
                sampling_steps,
            )
            BENCHMARK._synchronize(device)
            measurements.append(time.perf_counter() - started)
            TIMING_RECORDS.append({
                "benchmark": "sampling_steps", "method": method,
                "sampling_steps": sampling_steps, "frequency_count": 1,
                "frequency_hz": FIXED_SAMPLING_FREQUENCY_HZ,
                "repeat": repeat, "seconds": measurements[-1], "unit": "s/10-model batch",
            })
        results[sampling_steps] = statistics.median(measurements)
        print(
            f"{method}, sampling_steps={sampling_steps}: "
            f"median={results[sampling_steps] * 1000.0:.3f} ms/10-model batch"
        )
    return results


def _solve_numerical_case(
    case: dict,
    generation_config: dict,
    source_position: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solver_config = {**generation_config["solver"], "device": "cpu"}
    result = solve_scattered_wavefields(
        velocity_km_s=case["velocity"],
        source_positions_rc=[source_position],
        frequency_hz=case["frequency_hz"],
        background_velocity_km_s=float(
            generation_config["physics"]["background_velocity_km_s"]
        ),
        spacing_km=tuple(generation_config["grid"]["spacing_km"]),
        solver_config=solver_config,
        singularity_radius_cells=float(
            generation_config["source"].get("singularity_radius_cells", 0.5)
        ),
    )
    residuals = result[2]
    tolerance = float(solver_config["relative_residual_tolerance"])
    if not all(np.isfinite(array).all() for array in result):
        raise RuntimeError(f"CPU numerical solver returned non-finite values")
    if float(np.max(residuals)) > tolerance:
        raise RuntimeError(
            f"CPU numerical residual {np.max(residuals):.3e} "
            f"exceeds tolerance {tolerance:.3e}"
        )
    return result


def _measure_numerical_frequencies(
    cases_by_model: list[list[dict]],
    generation_config: dict,
    source_position: tuple[int, int],
    warmup_runs: int,
    timed_runs: int,
    threads: int,
) -> tuple[dict[float, float], dict[int, float]]:
    model_count = len(cases_by_model)
    per_frequency: dict[float, float] = {}
    frequency_measurements: list[list[float]] = []

    def solve_frequency(frequency_index: int) -> None:
        for model_cases in cases_by_model:
            case = model_cases[frequency_index]
            _solve_numerical_case(case, generation_config, source_position)

    for frequency_index, frequency_hz in enumerate(FREQUENCIES_HZ):
        for _ in range(warmup_runs):
            solve_frequency(frequency_index)

        measurements = []
        for repeat in range(timed_runs):
            started = time.perf_counter()
            solve_frequency(frequency_index)
            measurements.append(
                (time.perf_counter() - started) / model_count
            )
            TIMING_RECORDS.append({
                "benchmark": "numerical_frequency", "method": f"numerical_{threads}threads",
                "sampling_steps": "", "frequency_count": 1,
                "frequency_hz": frequency_hz, "repeat": repeat,
                "seconds": measurements[-1], "unit": "s/model",
            })
        frequency_measurements.append(measurements)
        per_frequency[float(frequency_hz)] = statistics.median(measurements)
        print(
            f"numerical ({threads} threads), frequency={frequency_hz} Hz: "
            f"median={per_frequency[float(frequency_hz)]:.6f} s/model"
        )

    cumulative: dict[int, float] = {}
    # Sum each repetition's frequency times before taking the median, so
    # cumulative timings also represent medians of the timed frequency sets.
    for count in FREQUENCY_COUNTS:
        cumulative[count] = statistics.median(
            sum(values[repeat] for values in frequency_measurements[:count])
            for repeat in range(timed_runs)
        )
    return per_frequency, cumulative


def _write_csv(
    meanflow: dict[int, float],
    numerical: dict[int, dict[int, float]],
    sampling: dict[str, dict[int, float]],
) -> None:
    with DETAILS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frequency_count",
                "frequency_range_hz",
                "meanflow_time_s",
                "numerical_1thread_time_s",
                "numerical_4threads_time_s",
                "meanflow_speedup_vs_1thread",
                "meanflow_speedup_vs_4threads",
            ]
        )
        for count in FREQUENCY_COUNTS:
            writer.writerow(
                [
                    count,
                    f"{FREQUENCIES_HZ[0]}-{FREQUENCIES_HZ[count - 1]}",
                    meanflow[count],
                    numerical[1][count],
                    numerical[4][count],
                    numerical[1][count] / meanflow[count],
                    numerical[4][count] / meanflow[count],
                ]
            )
    print(f"saved: {DETAILS_CSV}")

    with SAMPLING_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["sampling_steps", "fno_time_ms", "diffusion_time_ms", "meanflow_time_ms"]
        )
        for sampling_steps in SAMPLING_STEPS:
            writer.writerow(
                [
                    sampling_steps,
                    sampling["fno"][1] * 1000.0 if sampling_steps == 1 else "",
                    sampling["diffusion"][sampling_steps] * 1000.0,
                    sampling["meanflow"][sampling_steps] * 1000.0,
                ]
            )
    print(f"saved: {SAMPLING_CSV}")

    with RAW_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(TIMING_RECORDS[0]))
        writer.writeheader()
        writer.writerows(TIMING_RECORDS)
    print(f"saved: {RAW_CSV}")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
        }
    )


def _plot(
    meanflow: dict[int, float],
    numerical: dict[int, dict[int, float]],
    sampling: dict[str, dict[int, float]],
) -> None:
    _configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    positions = np.arange(len(SAMPLING_STEPS))
    bar_width = 0.25
    maximum = max(
        value * 1000.0
        for method in METHODS
        for value in sampling[method].values()
    )
    for method in METHODS:
        plotted_steps = (1,) if method == "fno" else SAMPLING_STEPS
        values = [sampling[method][steps] * 1000.0 for steps in plotted_steps]
        offset = (METHODS.index(method) - (len(METHODS) - 1) / 2) * bar_width
        bar_positions = np.array([SAMPLING_STEPS.index(steps) for steps in plotted_steps]) + offset
        bars = axes[0].bar(
            bar_positions,
            values,
            width=bar_width,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.8,
            label=DISPLAY_NAMES[method],
            zorder=3,
        )
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + maximum * 0.015,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[0].set_xlabel("Sampling steps")
    axes[0].set_ylabel("Runtime (ms)")
    axes[0].set_xticks(
        positions,
        [
            f"{steps} step" if steps == 1 else f"{steps} steps"
            for steps in SAMPLING_STEPS
        ],
    )
    axes[0].set_ylim(0, maximum * 1.20)
    axes[0].legend(frameon=False, loc="upper left")

    counts = list(FREQUENCY_COUNTS)

    meanflow_values = [meanflow[count] for count in counts]
    speed_ax = axes[1].twinx()
    meanflow_line = axes[1].plot(
        counts,
        meanflow_values,
        color=COLORS["meanflow"],
        linewidth=2.0,
        marker="o",
        markersize=4,
        label=DISPLAY_NAMES["meanflow"],
    )[0]
    numerical_lines = []
    speedup_lines = []
    for threads, color, speed_color, marker, dash in (
        (1, COLORS["numerical"], COLORS["speedup"], "o", "--"),
        (4, COLORS["numerical_4threads"], COLORS["speedup_4threads"], "s", "-."),
    ):
        thread_label = "1 thread" if threads == 1 else "4 threads"
        numerical_lines.append(axes[1].plot(
            counts,
            [numerical[threads][count] for count in counts],
            color=color, linewidth=2.0, marker=marker, markersize=4,
            label=f"Numerical Simulation ({thread_label})",
        )[0])
        speedup_lines.append(speed_ax.plot(
            counts,
            [numerical[threads][count] / meanflow[count] for count in counts],
            color=speed_color, linewidth=1.6, linestyle=dash,
            label=f"Speed Up ({thread_label})",
        )[0])
    axes[1].set_xlabel("Number of frequencies")
    axes[1].set_ylabel("Elapsed time (s)")
    axes[1].set_xticks(counts)
    speed_ax.set_ylabel("Speed Up ratio")
    axes[1].legend(
        handles=[meanflow_line, *numerical_lines, *speedup_lines],
        frameon=False,
        loc="upper left",
    )

    for panel_index, ax in enumerate(axes):
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.30, zorder=0)
        ax.text(
            -0.12,
            1.04,
            f"({chr(ord('a') + panel_index)})",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=1200, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"saved: {OUTPUT_STEM.with_suffix('.png')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda; use cpu explicitly for diagnostics only).",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=TIMED_RUNS)
    parser.add_argument(
        "--threads",
        type=int,
        default=CPU_THREADS,
        help="Torch CPU thread limit; numerical baselines always use 1 and 4 threads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA benchmarking was requested, but this Python environment has "
            "no CUDA-enabled PyTorch. Use the project's forward_test environment "
            "(PyTorch 2.8+cu128) or pass --device cpu only for diagnostics."
        )
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    torch.set_num_threads(args.threads)
    TIMING_RECORDS.clear()
    print(f"frequency range: {FREQUENCIES_HZ[0]}--{FREQUENCIES_HZ[-1]} Hz, batched per model")
    print(f"warmup={args.warmup}, runs={args.runs}, CPU threads={args.threads}")
    BENCHMARK.seed_everything(42)

    cases_by_model, generation_config, source_position = _load_cases(device)
    sampling: dict[str, dict[int, float]] = {}
    meanflow: dict[int, float] | None = None
    for method in METHODS:
        loaded = BENCHMARK._load_experiment(method, device)
        batches = _prepare_learned_batches(loaded, cases_by_model)
        sampling[method] = _measure_sampling_steps(
            loaded, batches, method, device, args.warmup, args.runs
        )
        if method == "meanflow":
            meanflow = _measure_learned_counts(
                loaded,
                batches,
                method,
                device,
                args.warmup,
                args.runs,
            )
        del loaded, batches
        if device.type == "cuda":
            torch.cuda.empty_cache()

    numerical: dict[int, dict[int, float]] = {}
    numerical_threadpools = {}
    for threads in NUMERICAL_THREADS:
        with threadpool_limits(limits=threads):
            numerical_threadpools[str(threads)] = threadpool_info()
            _, numerical[threads] = _measure_numerical_frequencies(
                cases_by_model,
                generation_config,
                source_position,
                args.warmup,
                args.runs,
                threads,
            )
    if meanflow is None:
        raise RuntimeError("Mean Flows timings were not collected")
    _write_csv(meanflow, numerical, sampling)
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "python_executable": sys.executable,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu": cpu_model, "torch_cpu_threads": args.threads,
        "numerical_cpu_threads": NUMERICAL_THREADS,
        "numerical_threadpools": numerical_threadpools,
        "warmup_runs": args.warmup, "timed_runs": args.runs,
        "aggregation": "median of repeated synchronized wall-clock measurements",
        "model_indices": MODEL_INDICES, "source_index": SOURCE_INDEX,
        "source_position_zx": source_position, "dataset_split": "val",
        "frequencies_hz": FREQUENCIES_HZ, "sampling_steps": SAMPLING_STEPS,
        "panel_a": {"frequency_hz": FIXED_SAMPLING_FREQUENCY_HZ, "batch_size": len(MODEL_INDICES), "methods": list(METHODS), "fno_steps": [1]},
        "panel_b": {"meanflow_steps": MULTIFREQUENCY_SAMPLING_STEPS,
                    "numerical_cpu_threads": NUMERICAL_THREADS,
                    "speedup": "each numerical baseline median divided by Mean Flows median",
                    "batch_axis": "frequencies for one velocity model; one sampler call per model and count",
                    "numerical": "sequential independent CPU frequency solves; sum frequency times within each repetition, then take median",
                    "time_unit": "seconds per velocity model"},
        "learned_timing_scope": "inference on pre-normalized device-resident inputs; excludes checkpoint loading, input generation, transfers, encoding and output decoding",
        "numerical_timing_scope": "CPU solver including field/operator assembly, factorization, solve, finite-output and residual checks",
        "checkpoint_configs": {name: str(path) for name, path in BENCHMARK.CHECKPOINT_CONFIGS.items()},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {METADATA_PATH}")
    _plot(meanflow, numerical, sampling)


if __name__ == "__main__":
    main()
