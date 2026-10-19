"""Evaluation helpers for the direct FNO Helmholtz surrogate."""

from __future__ import annotations

from collections import defaultdict
import time

import torch

from .data import batch_to_device
from .metrics import MetricAccumulator, per_sample_metrics


@torch.no_grad()
def evaluate_fno_batches(
    model,
    loader,
    normalizer,
    device: torch.device,
    config: dict,
    grouped: bool = False,
    max_batches: int | None = None,
):
    model.eval()
    overall = MetricAccumulator()
    groups: dict[tuple[str, int], MetricAccumulator] = defaultdict(MetricAccumulator)
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch_to_device(cpu_batch, device)
        condition = batch["condition"].float()
        target = batch["target"].float()
        condition_normalized = normalizer.encode_condition(condition)
        target_normalized = normalizer.encode_target(target)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction_normalized = model(condition_normalized)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) / len(target)
        prediction = normalizer.decode_target(prediction_normalized.float())
        values = per_sample_metrics(
            prediction_normalized.float(),
            target_normalized,
            prediction,
            target,
            condition,
            batch["frequency_hz"],
            config["physics"],
        )
        values["nfe"] = torch.ones(len(target), device=device)
        values["backward_evals"] = torch.zeros(len(target), device=device)
        values["seconds_per_sample"] = torch.full((len(target),), elapsed, device=device)
        overall.update(values)
        if grouped:
            for sample_index in range(len(target)):
                sample_values = {
                    name: value[sample_index : sample_index + 1]
                    for name, value in values.items()
                }
                for name in ("frequency_index", "source_index", "model_index"):
                    key = (name.removesuffix("_index"), int(batch[name][sample_index]))
                    groups[key].update(sample_values)
    return overall.compute(), {key: accumulator.compute() for key, accumulator in groups.items()}
