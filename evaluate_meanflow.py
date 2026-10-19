"""Evaluate a trained one-step GNO or Improved MeanFlow checkpoint."""

from __future__ import annotations

import argparse

import torch

from diffusion import GaussianDiffusion
from models import build_model
from utils.config import load_config, output_dir, resolve_path
from utils.data import build_dataset, make_loader
from utils.evaluation import evaluate_batches
from utils.logger import Logger
from utils.metrics import format_metrics
from utils.normalization import PhysicalNormalizer
from utils.runtime import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/meanflow_improved_ema999.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    method = config["method"]["type"]
    if method not in {"gno", "meanflow_improved"}:
        raise ValueError(f"unsupported method: {method}")
    seed_everything(int(config["experiment"]["seed"]))
    device = select_device(config["experiment"].get("device", "auto"))
    directory = output_dir(config)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else directory / "model_best.pth"
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    normalizer = PhysicalNormalizer(state["normalization"], device)
    model = build_model(config["model"]).to(device).float()
    model.load_state_dict(state.get("ema", state["model"]))
    dataset = build_dataset(config, args.split)
    loader = make_loader(dataset, int(config["train"]["val_batch_size"]), False, int(config["data"]["num_workers"]))
    diffusion = GaussianDiffusion.from_config(config["diffusion"]) if method == "gno" else None
    overall, groups = evaluate_batches(
        model, loader, normalizer, device, config, diffusion=diffusion, grouped=True
    )
    logger = Logger(directory / "runlog.txt")
    logger.log(f"{args.split}: {format_metrics(overall)}")
    frequencies = config["data"]["frequencies_hz"]
    model_names = config["data"].get("test_model_names", [])
    for (kind, index), metrics in sorted(groups.items()):
        if kind == "frequency":
            label = f"{frequencies[index]}Hz"
        elif kind == "model" and index < len(model_names):
            label = model_names[index]
        else:
            label = f"{kind}_{index}"
        logger.log(f"{args.split} {label}: {format_metrics(metrics)}")
    logger.close()


if __name__ == "__main__":
    main()
