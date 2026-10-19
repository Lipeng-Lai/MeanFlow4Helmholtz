"""Evaluate a trained FNO Helmholtz surrogate."""

from __future__ import annotations

import argparse

import torch

from models import build_model
from utils.config import load_config, output_dir, resolve_path
from utils.data import build_dataset, make_loader
from utils.fno_evaluation import evaluate_fno_batches
from utils.logger import Logger
from utils.metrics import format_metrics
from utils.normalization import PhysicalNormalizer
from utils.runtime import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fno.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if str(config["method"]["type"]).lower() != "fno":
        raise ValueError("evaluate_fno.py requires method.type=fno")
    seed_everything(int(config["experiment"]["seed"]))
    device = select_device()
    directory = output_dir(config)
    checkpoint_path = (
        resolve_path(args.checkpoint) if args.checkpoint else directory / "model_best.pth"
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    normalizer = PhysicalNormalizer(state["normalization"], device)
    model = build_model(config["model"]).to(device).float()
    model.load_state_dict(state.get("ema", state["model"]))
    dataset = build_dataset(config, args.split)
    loader = make_loader(
        dataset,
        int(config["train"]["val_batch_size"]),
        False,
        int(config["data"]["num_workers"]),
    )
    overall, groups = evaluate_fno_batches(
        model, loader, normalizer, device, config, grouped=True
    )
    logger = Logger(directory / "runlog.txt")
    logger.log(f"{args.split}: {format_metrics(overall)}")
    frequencies = config["data"]["frequencies_hz"]
    for (kind, index), metrics in sorted(groups.items()):
        label = f"{frequencies[index]}Hz" if kind == "frequency" else f"{kind}_{index}"
        logger.log(f"{args.split} {label}: {format_metrics(metrics)}")
    logger.close()


if __name__ == "__main__":
    main()
