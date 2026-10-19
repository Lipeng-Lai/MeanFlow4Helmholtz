"""Train a direct Fourier neural operator for Helmholtz wavefields."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from models import build_model
from utils.checkpoint import latest_checkpoint, save_checkpoint
from utils.config import load_config, output_dir, resolve_path, save_config
from utils.data import batch_to_device, build_dataset, make_loader
from utils.ema import EMA
from utils.fno_evaluation import evaluate_fno_batches
from utils.logger import Logger
from utils.metrics import format_metrics, format_scalar
from utils.normalization import PhysicalNormalizer, normalization_stats
from utils.runtime import seed_everything, select_device
from utils.sensitivity import SensitivityMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fno.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def optimizer_and_scheduler(model: torch.nn.Module, config: dict, duration: int):
    settings = config["train"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        betas=tuple(map(float, settings["betas"])),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, duration),
        eta_min=float(config["train"]["scheduler"]["eta_min"]),
    )
    return optimizer, scheduler


def checkpoint_payload(config, stats, model, optimizer, scheduler, best, step, ema):
    return {
        "format_version": 3,
        "method": "fno",
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "normalization": stats,
        "config": config,
        "step": int(step),
        "best_relative_l2": float(best),
    }


def train(config: dict, resume: bool = True) -> None:
    if str(config["method"]["type"]).lower() != "fno":
        raise ValueError("train_fno.py requires method.type=fno")
    device = select_device()
    seed_everything(int(config["experiment"]["seed"]))
    directory = output_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    save_config(directory / "config.yaml", config)
    logger = Logger(directory / "runlog.txt")

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    state = None
    if resume:
        checkpoint = latest_checkpoint(directory)
        if checkpoint is None and (directory / "model_best.pth").exists():
            checkpoint = directory / "model_best.pth"
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=device, weights_only=False)

    stats = state["normalization"] if state is not None else normalization_stats(
        config, directory / "normalization.npz"
    )
    normalizer = PhysicalNormalizer(stats, device)
    model = build_model(config["model"]).to(device).float()
    max_steps = int(config["train"]["max_steps"])
    optimizer, scheduler = optimizer_and_scheduler(model, config, max_steps)
    ema = EMA(model, float(config["train"]["ema_decay"]))
    start_step, best = 1, float("inf")
    if state is not None:
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"]) + 1
        best = float(state["best_relative_l2"])

    workers = int(config["data"]["num_workers"])
    train_loader = make_loader(
        train_dataset, int(config["train"]["batch_size"]), True, workers
    )
    val_loader = make_loader(
        val_dataset, int(config["train"]["val_batch_size"]), False, workers
    )
    iterator = iter(train_loader)
    accumulation = int(config["train"].get("gradient_accumulation", 1))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    estimated_checkpoint_mib = parameter_count * 4 * 4 / 1024**2
    logger.log(
        f"method=fno device={device} parameters={parameter_count:,} "
        f"estimated_full_checkpoint_mib={estimated_checkpoint_mib:.1f} "
        f"state={'resume' if state is not None else 'scratch'}"
    )

    sensitivity = SensitivityMonitor(val_dataset, config, normalizer, device, directory)
    logger.log(
        f"step={start_step - 1} sensitivity_loss={format_scalar(sensitivity.evaluate(model))} "
        f"sensitivity_probes={sensitivity.count} objective=data_mse_only"
    )

    try:
        for step in range(start_step, max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(accumulation):
                try:
                    cpu_batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    cpu_batch = next(iterator)
                batch = batch_to_device(cpu_batch, device)
                condition = normalizer.encode_condition(batch["condition"].float())
                target = normalizer.encode_target(batch["target"].float())
                forward_loss = F.mse_loss(model(condition), target)
                (forward_loss / accumulation).backward()
                loss_sum += float(forward_loss.detach())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["train"]["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("FNO training produced non-finite gradients")
            optimizer.step()
            scheduler.step()
            ema.update(model)

            if step % int(config["train"]["log_every"]) == 0 or step == max_steps:
                logger.log(
                    f"step={step}/{max_steps} mse={format_scalar(loss_sum / accumulation)} "
                    f"sensitivity_loss={format_scalar(sensitivity.evaluate(model))} "
                    f"lr={format_scalar(scheduler.get_last_lr()[0])}"
                )
            improved = False
            if step % int(config["train"]["validate_every"]) == 0 or step == max_steps:
                validation, _ = evaluate_fno_batches(
                    ema.model,
                    val_loader,
                    normalizer,
                    device,
                    config,
                    max_batches=config["validation"].get("max_batches"),
                )
                improved = validation["relative_l2"] < best
                best = min(best, validation["relative_l2"])
                logger.log(
                    f"step={step} val {format_metrics(validation)} "
                    f"best_relative_l2={format_scalar(best)}"
                )
            if improved or step % int(config["train"]["save_every"]) == 0 or step == max_steps:
                payload = checkpoint_payload(
                    config, stats, model, optimizer, scheduler, best, step, ema
                )
                if improved:
                    save_checkpoint(directory / "model_best.pth", payload)
                if step % int(config["train"]["save_every"]) == 0 or step == max_steps:
                    save_checkpoint(directory / f"model_step_{step}.pth", payload)
    finally:
        logger.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        config["train"]["max_steps"] = int(args.max_steps)
        config["train"]["validate_every"] = min(
            int(config["train"]["validate_every"]), int(args.max_steps)
        )
        config["train"]["save_every"] = min(
            int(config["train"]["save_every"]), int(args.max_steps)
        )
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = str(args.output_dir)
    train(config, resume=not args.no_resume)


if __name__ == "__main__":
    main()
