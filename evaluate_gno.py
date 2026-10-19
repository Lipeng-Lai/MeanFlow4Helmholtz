"""Evaluate a trained one-step conditional diffusion GNO checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from diffusion import GaussianDiffusion
from models import build_model
from train_gno import fixed_noise
from utils.config import load_config, output_dir, resolve_path
from utils.data import batch_to_device, build_dataset, make_loader
from utils.logger import Logger
from utils.metrics import MetricAccumulator, format_metrics, per_sample_metrics
from utils.normalization import PhysicalNormalizer
from utils.runtime import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gno_ema999.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if str(config["method"]["type"]).lower() != "gno":
        raise ValueError("evaluate_gno.py requires method.type=gno")
    seed_everything(int(config["experiment"]["seed"]))
    device = select_device()
    directory = output_dir(config)
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else directory / "model_best.pth"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    normalizer = PhysicalNormalizer(state["normalization"], device)
    model = build_model(config["model"]).to(device).float()
    model.load_state_dict(state.get("ema", state["model"]))
    diffusion = GaussianDiffusion.from_config(config["diffusion"])
    dataset = build_dataset(config, args.split)
    loader = make_loader(
        dataset, int(config["train"]["val_batch_size"]), False,
        int(config["data"]["num_workers"]),
    )
    overall = MetricAccumulator()
    groups: dict[tuple[str, int], MetricAccumulator] = defaultdict(MetricAccumulator)
    frequency_count = len(config["data"]["frequencies_hz"])
    source_count = int(config["data"]["source_count"])
    for batch_index, cpu_batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        batch = batch_to_device(cpu_batch, device)
        condition = batch["condition"].float()
        target = batch["target"].float()
        noise = fixed_noise(
            cpu_batch, (2, condition.shape[-2], condition.shape[-1]),
            int(config["experiment"]["seed"]), frequency_count, source_count, device,
        )
        timesteps = torch.full(
            (len(target),), diffusion.steps - 1, device=device, dtype=torch.long
        )
        with torch.no_grad():
            prediction_normalized = model(
                noise, timesteps, normalizer.encode_condition(condition)
            )
            prediction = normalizer.decode_target(prediction_normalized.float())
        values = per_sample_metrics(
            prediction_normalized.float(), normalizer.encode_target(target),
            prediction, target, condition, batch["frequency_hz"], config["physics"],
        )
        values["nfe"] = torch.ones(len(target), device=device)
        values["backward_evals"] = torch.zeros(len(target), device=device)
        overall.update(values)
        for sample_index in range(len(target)):
            sample_values = {
                name: value[sample_index : sample_index + 1]
                for name, value in values.items()
            }
            for name in ("frequency_index", "source_index", "model_index"):
                key = (name.removesuffix("_index"), int(batch[name][sample_index]))
                groups[key].update(sample_values)

    logger = Logger(directory / "runlog.txt")
    logger.log(f"{args.split}: {format_metrics(overall.compute())}")
    frequencies = config["data"]["frequencies_hz"]
    for (kind, index), metrics in sorted(groups.items()):
        label = f"{frequencies[index]}Hz" if kind == "frequency" else f"{kind}_{index}"
        logger.log(f"{args.split} {label}: {format_metrics(metrics.compute())}")
    logger.close()


if __name__ == "__main__":
    main()
