"""Shared one-step evaluation for GNO and Improved MeanFlow."""

from __future__ import annotations

from collections import defaultdict
import time

import torch

from diffusion.sampling import ddim_sample
from meanflow import improved_meanflow_sample
from .data import batch_to_device
from .metrics import MetricAccumulator, per_sample_metrics


def fixed_noise(batch: dict, shape: tuple[int, ...], seed: int, frequency_count: int,
                source_count: int, device: torch.device) -> torch.Tensor:
    flat_indices = (
        batch["model_index"].cpu().long() * frequency_count * source_count
        + batch["frequency_index"].cpu().long() * source_count
        + batch["source_index"].cpu().long()
    )
    samples = []
    for index in flat_indices.tolist():
        generator = torch.Generator().manual_seed(seed + index)
        samples.append(torch.randn(shape, generator=generator))
    return torch.stack(samples).to(device=device, dtype=torch.float32)


@torch.no_grad()
def evaluate_batches(model, loader, normalizer, device: torch.device, config: dict,
                     diffusion=None, sampling_steps: int | None = None,
                     grouped: bool = False, max_batches: int | None = None):
    method = config["method"]["type"]
    if method not in {"gno", "meanflow_improved"}:
        raise ValueError(f"unsupported evaluation method: {method}")
    model.eval()
    overall = MetricAccumulator()
    groups: dict[tuple[str, int], MetricAccumulator] = defaultdict(MetricAccumulator)
    steps = int(sampling_steps or config["sampling"]["steps"])
    if steps != 1:
        raise ValueError("this repository only supports one-step sampling")

    for batch_index, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch_to_device(cpu_batch, device)
        condition = batch["condition"].float()
        target = batch["target"].float()
        condition_normalized = normalizer.encode_condition(condition)
        target_normalized = normalizer.encode_target(target)
        noise = fixed_noise(
            cpu_batch,
            (2, condition.shape[-2], condition.shape[-1]),
            int(config["experiment"]["seed"]),
            len(config["data"]["frequencies_hz"]),
            int(config["data"]["source_count"]),
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if method == "gno":
            prediction_normalized, _ = ddim_sample(
                model, diffusion, condition_normalized, noise, 1,
                eta=0.0,
                clip_denoised=bool(config["diffusion"]["clip_denoised"]),
            )
        else:
            prediction_normalized, _ = improved_meanflow_sample(
                model, noise, condition_normalized, 1,
                time_scale=float(config["flow"]["time_scale"]),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) / len(target)
        prediction = normalizer.decode_target(prediction_normalized.float())
        values = per_sample_metrics(
            prediction_normalized.float(), target_normalized,
            prediction, target, condition, batch["frequency_hz"], config["physics"],
        )
        values["nfe"] = torch.ones(len(target), device=device)
        values["backward_evals"] = torch.zeros(len(target), device=device)
        values["seconds_per_sample"] = torch.full((len(target),), elapsed, device=device)
        overall.update(values)
        if grouped:
            for sample_index in range(len(target)):
                sample_values = {
                    name: value[sample_index:sample_index + 1]
                    for name, value in values.items()
                }
                for name in ("frequency_index", "source_index", "model_index"):
                    groups[(name.removesuffix("_index"), int(batch[name][sample_index]))].update(sample_values)
    return overall.compute(), {key: accumulator.compute() for key, accumulator in groups.items()}
