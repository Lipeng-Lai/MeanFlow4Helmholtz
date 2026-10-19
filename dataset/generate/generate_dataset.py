"""Generate paired full-waveform Helmholtz data for FNO training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

try:
    from .helmholtz import (
        background_green_wavefield,
        solve_scattered_wavefields,
    )
except ImportError:
    from helmholtz import (
        background_green_wavefield,
        solve_scattered_wavefields,
    )


_WORKER: dict = {}


def project_root_from_config(path: Path) -> Path:
    return path.resolve().parent.parent


def resolve_project_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = ("frequencies_hz", "grid", "physics", "source", "solver", "smoothing", "dataset")
    if not isinstance(config, dict) or any(key not in config for key in required):
        raise ValueError(f"{path} is missing one of {required}")
    return config


def fixed_source_positions(shape: Sequence[int], config: dict) -> list[tuple[int, int]]:
    if config.get("placement") != "fixed":
        raise ValueError("source.placement must be fixed")
    nz, nx = map(int, shape)
    z = int(config["z_index"])
    xs = [int(value) for value in config["x_indices"]]
    positions = [(z, x) for x in xs]
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("source positions must be non-empty and unique")
    if any(not (0 <= row < nz and 0 <= col < nx) for row, col in positions):
        raise ValueError(f"source positions {positions} exceed grid {(nz, nx)}")
    return positions


def validate_velocity_file(path: Path, shape: Sequence[int]) -> np.ndarray:
    models = np.load(path, mmap_mode="r")
    original_shape = tuple(models.shape)
    if models.ndim == 4 and models.shape[1] == 1:
        models = models[:, 0]
    if models.ndim != 3 or tuple(models.shape[1:]) != tuple(shape):
        raise ValueError(f"{path}: expected [N,1,{shape[0]},{shape[1]}], got {original_shape}")
    if not np.all(np.isfinite(models)) or np.any(models <= 0.0):
        raise ValueError(f"{path} contains invalid velocities")
    return models


def array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def select_base_models(group_files: dict[str, Path], shape: Sequence[int], config: dict) -> dict[str, list[dict]]:
    dataset = config["dataset"]
    smoothing = config["smoothing"]
    train_count = int(dataset["train_models_per_group"])
    val_count = int(dataset["val_models_per_group"])
    split_seed = int(dataset["split_seed"])
    sigma_min, sigma_max = map(float, smoothing["sigma_range"])
    if int(smoothing.get("variants_per_model", 1)) != 1 or not 0.0 <= sigma_min <= sigma_max:
        raise ValueError("this generator requires one smoothed variant and a valid sigma_range")

    selections: dict[str, list[dict]] = {"train": [], "val": []}
    arrays: dict[str, np.ndarray] = {}
    for group_offset, (group, path) in enumerate(group_files.items()):
        models = validate_velocity_file(path, shape)
        arrays[group] = models
        if len(models) != train_count + val_count:
            raise ValueError(
                f"{group} has {len(models)} models; split requests {train_count + val_count}"
            )
        permutation = np.random.default_rng(split_seed + group_offset).permutation(len(models))
        for split, indices in {
            "train": permutation[:train_count],
            "val": permutation[train_count : train_count + val_count],
        }.items():
            for index in indices:
                sigma_seed = int(smoothing["seed"]) + group_offset * 100_000 + int(index)
                sigma = float(np.random.default_rng(sigma_seed).uniform(sigma_min, sigma_max))
                selections[split].append(
                    {
                        "model_group": group,
                        "velocity_path": str(path.resolve()),
                        "source_model_index": int(index),
                        "velocity_sha256": array_hash(models[int(index)]),
                        "smoothing_sigma": sigma,
                    }
                )
    for offset, split in enumerate(("train", "val")):
        order = np.random.default_rng(split_seed + 10_000 + offset).permutation(len(selections[split]))
        selections[split] = [selections[split][int(index)] for index in order]
    train_hashes = {item["velocity_sha256"] for item in selections["train"]}
    val_hashes = {item["velocity_sha256"] for item in selections["val"]}
    overlap = train_hashes & val_hashes
    if overlap:
        raise ValueError(f"train and val contain {len(overlap)} duplicate base models")
    return selections


def storage_bytes(base_count: int, frequencies: int, sources: int, shape: Sequence[int], dtype) -> int:
    points = int(np.prod(shape))
    scalars = (
        2 * base_count * points
        + frequencies * sources * 2 * points
        + 2 * base_count * frequencies * sources * 2 * points
    )
    return scalars * np.dtype(dtype).itemsize


def _paths(split_root: Path) -> dict[str, Path]:
    final = {
        "velocity": split_root / "data" / "velocity.npy",
        "background": split_root / "data" / "background.npy",
        "label": split_root / "label.npy",
        "models": split_root / "models.csv",
    }
    result = {f"final_{key}": value for key, value in final.items()}
    for key, value in final.items():
        suffix = ".incomplete.csv" if value.suffix == ".csv" else ".incomplete.npy"
        result[f"temporary_{key}"] = value.with_suffix(suffix)
    result["progress"] = split_root / "generation_progress.npy"
    return result


def _open_or_create_array(path: Path, shape: tuple[int, ...], dtype, resume: bool):
    if resume and path.is_file():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"resume array {path} has {array.shape}/{array.dtype}, expected {shape}/{dtype}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def prepare_split_arrays(split_root: Path, base_count: int, frequency_count: int, source_count: int,
                         shape: Sequence[int], dtype, resume: bool, overwrite: bool):
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "data").mkdir(parents=True, exist_ok=True)
    paths = _paths(split_root)
    final_existing = [value for key, value in paths.items() if key.startswith("final_") and value.exists()]
    if final_existing and not overwrite:
        raise FileExistsError("completed output exists; use --overwrite: " + ", ".join(map(str, final_existing)))
    if overwrite:
        for value in paths.values():
            if value.is_file():
                value.unlink()
    nz, nx = map(int, shape)
    progress_existed = paths["progress"].is_file()
    arrays = {
        "velocity": _open_or_create_array(
            paths["temporary_velocity"], (2 * base_count, 1, nz, nx), dtype, resume
        ),
        "background": _open_or_create_array(
            paths["temporary_background"], (frequency_count, source_count, 2, nz, nx), dtype, resume
        ),
        "label": _open_or_create_array(
            paths["temporary_label"],
            (2 * base_count, frequency_count, source_count, 2, nz, nx), dtype, resume,
        ),
        "progress": _open_or_create_array(paths["progress"], (base_count,), np.bool_, resume),
    }
    if not resume or not progress_existed:
        arrays["progress"][:] = False
        arrays["progress"].flush()
    return paths, arrays


def generate_background(shape, spacing, frequencies, positions, config, dtype) -> np.ndarray:
    padding = int(config["solver"]["absorbing_layer"].get("padding_cells", 0))
    padded_shape = (int(shape[0]) + 2 * padding, int(shape[1]) + 2 * padding)
    padded_positions = [(row + padding, col + padding) for row, col in positions]
    core = (slice(padding, -padding), slice(padding, -padding)) if padding else (slice(None), slice(None))
    result = np.empty((len(frequencies), len(positions), 2, *shape), dtype=dtype)
    for frequency_index, frequency in enumerate(frequencies):
        fields = np.stack(
            [
                background_green_wavefield(
                    padded_shape,
                    spacing,
                    position,
                    frequency,
                    float(config["physics"]["background_velocity_km_s"]),
                    float(config["source"].get("singularity_radius_cells", 0.5)),
                )[core]
                for position in padded_positions
            ]
        )
        result[frequency_index] = np.stack((fields.real, fields.imag), axis=1).astype(dtype)
    return result


def _worker_init(payload: dict) -> None:
    global _WORKER
    _WORKER = dict(payload)
    _WORKER["velocity"] = np.load(payload["velocity_array"], mmap_mode="r+")
    _WORKER["label"] = np.load(payload["label_array"], mmap_mode="r+")
    _WORKER["progress"] = np.load(payload["progress_array"], mmap_mode="r+")
    _WORKER["source_arrays"] = {}


def _worker_generate(task: tuple[int, dict]) -> dict:
    pair_index, record = task
    if bool(_WORKER["progress"][pair_index]):
        return {"pair_index": pair_index, "skipped": True, "maximum_residual": 0.0, "seconds": 0.0}
    started = time.perf_counter()
    path = record["velocity_path"]
    if path not in _WORKER["source_arrays"]:
        models = np.load(path, mmap_mode="r")
        _WORKER["source_arrays"][path] = models[:, 0] if models.ndim == 4 else models
    source = _WORKER["source_arrays"][path]
    original = np.asarray(source[int(record["source_model_index"])], dtype=np.float64) * float(
        _WORKER["velocity_scale"]
    )
    smoothed = gaussian_filter(
        original,
        sigma=float(record["smoothing_sigma"]),
        mode=str(_WORKER["smoothing_mode"]),
    )
    original_index, smoothed_index = 2 * pair_index, 2 * pair_index + 1
    _WORKER["velocity"][original_index, 0] = original.astype(_WORKER["dtype"])
    _WORKER["velocity"][smoothed_index, 0] = smoothed.astype(_WORKER["dtype"])
    maximum_residual = 0.0
    for frequency_index, frequency in enumerate(_WORKER["frequencies"]):
        _, original_field, residual_original = solve_scattered_wavefields(
            original,
            _WORKER["positions"],
            frequency_hz=float(frequency),
            background_velocity_km_s=float(_WORKER["background_velocity"]),
            spacing_km=_WORKER["spacing"],
            solver_config=_WORKER["solver"],
            singularity_radius_cells=float(_WORKER["singularity_radius"]),
        )
        _, smoothed_field, residual_smoothed = solve_scattered_wavefields(
            smoothed,
            _WORKER["positions"],
            frequency_hz=float(frequency),
            background_velocity_km_s=float(_WORKER["background_velocity"]),
            spacing_km=_WORKER["spacing"],
            solver_config=_WORKER["solver"],
            singularity_radius_cells=float(_WORKER["singularity_radius"]),
        )
        residuals = np.concatenate((residual_original, residual_smoothed))
        maximum_residual = max(maximum_residual, float(np.max(residuals)))
        _WORKER["label"][original_index, frequency_index] = np.stack(
            (original_field.real, original_field.imag), axis=1
        ).astype(_WORKER["dtype"])
        _WORKER["label"][smoothed_index, frequency_index] = np.stack(
            (smoothed_field.real, smoothed_field.imag), axis=1
        ).astype(_WORKER["dtype"])
    if maximum_residual > float(_WORKER["residual_tolerance"]):
        raise RuntimeError(
            f"pair {pair_index}: residual {maximum_residual:.3e} exceeds "
            f"{float(_WORKER['residual_tolerance']):.3e}"
        )
    for name in ("velocity", "label"):
        _WORKER[name].flush()
    _WORKER["progress"][pair_index] = True
    _WORKER["progress"].flush()
    return {
        "pair_index": pair_index,
        "skipped": False,
        "maximum_residual": maximum_residual,
        "seconds": time.perf_counter() - started,
    }


def write_models_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "model_index", "pair_index", "variant", "model_group",
                "source_model_index", "smoothing_sigma", "velocity_sha256",
            ),
        )
        writer.writeheader()
        for pair_index, record in enumerate(records):
            for variant_offset, variant in enumerate(("original", "smoothed")):
                writer.writerow(
                    {
                        "model_index": 2 * pair_index + variant_offset,
                        "pair_index": pair_index,
                        "variant": variant,
                        "model_group": record["model_group"],
                        "source_model_index": record["source_model_index"],
                        "smoothing_sigma": "0" if variant == "original" else f"{record['smoothing_sigma']:.10g}",
                        "velocity_sha256": record["velocity_sha256"],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/generate_dataset.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frequency", type=float, nargs="+")
    parser.add_argument("--split", action="append", choices=("train", "val"))
    parser.add_argument("--max-velocity-models", type=int, help="maximum base pairs per split")
    parser.add_argument("--max-shots", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)
    shape = tuple(map(int, config["grid"]["shape"]))
    spacing = tuple(map(float, config["grid"]["spacing_km"]))
    frequencies = [float(value) for value in (args.frequency or config["frequencies_hz"])]
    positions = fixed_source_positions(shape, config["source"])
    if args.max_shots is not None:
        positions = positions[: int(args.max_shots)]
    if not frequencies or not positions:
        raise ValueError("at least one frequency and source are required")
    split_names = args.split or ["train", "val"]
    output_root = args.output_root or resolve_project_path(root, config["dataset"]["output_root"])
    group_files = {
        group: resolve_project_path(root, path)
        for group, path in config["dataset"]["model_groups"].items()
    }
    selections = select_base_models(group_files, shape, config)
    if args.max_velocity_models is not None:
        if args.max_velocity_models <= 0:
            raise ValueError("max-velocity-models must be positive")
        selections = {name: values[: args.max_velocity_models] for name, values in selections.items()}
    dtype = np.dtype(config["dataset"].get("dtype", "float32"))
    plan = {
        "output_root": str(output_root.resolve()),
        "frequencies_hz": frequencies,
        "grid_shape": list(shape),
        "spacing_km": list(spacing),
        "source_positions_zx": [list(value) for value in positions],
        "base_models": {split: len(selections[split]) for split in split_names},
        "forward_models": {split: 2 * len(selections[split]) for split in split_names},
        "samples": {
            split: 2 * len(selections[split]) * len(frequencies) * len(positions)
            for split in split_names
        },
        "estimated_storage_gb": {
            split: storage_bytes(len(selections[split]), len(frequencies), len(positions), shape, dtype) / 1.0e9
            for split in split_names
        },
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    workers = int(args.workers or config["dataset"].get("workers", 1))
    resume = not args.no_resume and bool(config["dataset"].get("resume", True))
    overwrite = args.overwrite or bool(config["dataset"].get("overwrite", False))
    background = generate_background(shape, spacing, frequencies, positions, config, dtype)
    summaries = {}
    for split in split_names:
        records = selections[split]
        split_root = output_root / split
        paths, arrays = prepare_split_arrays(
            split_root, len(records), len(frequencies), len(positions), shape, dtype, resume, overwrite
        )
        arrays["background"][:] = background
        arrays["background"].flush()
        pending = [(index, record) for index, record in enumerate(records) if not bool(arrays["progress"][index])]
        del arrays
        payload = {
            "velocity_array": str(paths["temporary_velocity"]),
            "label_array": str(paths["temporary_label"]),
            "progress_array": str(paths["progress"]),
            "frequencies": frequencies,
            "positions": positions,
            "spacing": spacing,
            "solver": config["solver"],
            "background_velocity": float(config["physics"]["background_velocity_km_s"]),
            "singularity_radius": float(config["source"].get("singularity_radius_cells", 0.5)),
            "velocity_scale": float(config["dataset"].get("velocity_scale_to_km_s", 1.0)),
            "smoothing_mode": str(config["smoothing"].get("mode", "reflect")),
            "dtype": dtype,
            "residual_tolerance": float(config["solver"].get("relative_residual_tolerance", 1.0e-8)),
        }
        maximum_residual = 0.0
        completed = len(records) - len(pending)
        if pending:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max(1, workers), mp_context=context, initializer=_worker_init, initargs=(payload,)
            ) as executor:
                futures = {executor.submit(_worker_generate, task): task[0] for task in pending}
                for future in as_completed(futures):
                    result = future.result()
                    completed += 1
                    maximum_residual = max(maximum_residual, float(result["maximum_residual"]))
                    print(
                        f"generated {split}: pair {completed}/{len(records)} "
                        f"(index={result['pair_index']}, {result['seconds']:.2f}s)",
                        flush=True,
                    )
        progress = np.load(paths["progress"], mmap_mode="r")
        if not bool(np.all(progress)):
            raise RuntimeError(f"{split}: only {int(progress.sum())}/{len(progress)} pairs completed")
        del progress
        write_models_csv(paths["temporary_models"], records)
        for key in ("velocity", "background", "label", "models"):
            paths[f"temporary_{key}"].replace(paths[f"final_{key}"])
        paths["progress"].unlink(missing_ok=True)
        summaries[split] = {
            "base_models": len(records),
            "forward_models": 2 * len(records),
            "samples": 2 * len(records) * len(frequencies) * len(positions),
            "maximum_linear_relative_residual_in_run": maximum_residual,
            "array_shapes": {
                "data/velocity.npy": [2 * len(records), 1, *shape],
                "data/background.npy": [len(frequencies), len(positions), 2, *shape],
                "label.npy": [2 * len(records), len(frequencies), len(positions), 2, *shape],
            },
        }

    manifest = {
        "plan": plan,
        "split": {
            "seed": int(config["dataset"]["split_seed"]),
            "train_models_per_group": int(config["dataset"]["train_models_per_group"]),
            "val_models_per_group": int(config["dataset"]["val_models_per_group"]),
            "train_validation_velocity_overlap": 0,
            "model_groups": list(group_files),
        },
        "smoothing": config["smoothing"],
        "summary": summaries,
    }
    with (output_root / "generation_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    # Prevent BLAS oversubscription when users invoke the module directly.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
