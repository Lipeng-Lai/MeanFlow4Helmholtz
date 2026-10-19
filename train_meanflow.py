"""Train the GNO or Improved MeanFlow Helmholtz model in float32."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion import GaussianDiffusion
from meanflow import improved_meanflow_loss
from models import build_model
from utils.checkpoint import latest_checkpoint, save_checkpoint
from utils.config import load_config, output_dir, save_config
from utils.data import batch_to_device, build_dataset, make_loader
from utils.ema import EMA
from utils.evaluation import evaluate_batches
from utils.logger import Logger
from utils.metrics import format_metrics, format_scalar
from utils.normalization import PhysicalNormalizer, normalization_stats
from utils.runtime import seed_everything, select_device
from utils.sensitivity import SensitivityMonitor


def meanflow_predictor(model, condition: torch.Tensor, probe_index: int,
                       normalizer: PhysicalNormalizer, time_scale: float,
                       seed: int) -> torch.Tensor:
    """Differentiable deterministic one-step MeanFlow map for JVP probes."""
    shape = (condition.shape[0], 2, condition.shape[-2], condition.shape[-1])
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(probe_index))
    noise = torch.randn(shape, generator=generator).to(condition.device, condition.dtype)
    time = torch.ones(condition.shape[0], device=condition.device, dtype=condition.dtype)
    average_velocity = model(
        noise,
        time * float(time_scale),
        normalizer.encode_condition(condition),
        end_time=time * float(time_scale),
    )
    return normalizer.decode_target(noise - average_velocity)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def optimizer_and_scheduler(model: torch.nn.Module, config: dict, duration: int):
    settings = config["train"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        betas=tuple(map(float, settings["betas"])),
    )
    scheduler_config = config["train"]["scheduler"]
    scheduler_type = str(scheduler_config.get("type", "cosine")).lower()
    if scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=duration, eta_min=float(scheduler_config["eta_min"])
        )
    else:
        raise ValueError(f"unsupported scheduler type: {scheduler_type}")
    return optimizer, scheduler


def load_resume(config: dict, device: torch.device):
    directory = output_dir(config)
    path = latest_checkpoint(directory)
    if path is None:
        best_path = directory / "model_best.pth"
        path = best_path if best_path.exists() else None
    return None if path is None else torch.load(path, map_location=device, weights_only=False)


def prepare(config: dict, device: torch.device):
    seed_everything(int(config["experiment"]["seed"]))
    directory = output_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    save_config(directory / "config.yaml", config)
    logger = Logger(directory / "runlog.txt")
    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    state = load_resume(config, device)
    if state is not None:
        stats = state["normalization"]
    else:
        stats = normalization_stats(config, directory / "normalization.npz")
    normalizer = PhysicalNormalizer(stats, device)
    return directory, logger, train_dataset, val_dataset, state, stats, normalizer


def common_payload(config: dict, stats: dict, model: torch.nn.Module,
                   optimizer, scheduler, best: float, step: int, ema: EMA) -> dict:
    return {
        "format_version": 2,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "normalization": stats,
        "config": config,
        "step": step,
        "best_relative_l2": float(best),
    }


def make_loaders(train_dataset, val_dataset, config: dict):
    workers = int(config["data"]["num_workers"])
    return (
        make_loader(train_dataset, int(config["train"]["batch_size"]), True, workers),
        make_loader(val_dataset, int(config["train"]["val_batch_size"]), False, workers),
    )


def train_gno(config: dict) -> None:
    device = select_device(config["experiment"].get("device", "auto"))
    directory, logger, train_dataset, val_dataset, state, stats, normalizer = prepare(config, device)
    model = build_model(config["model"]).to(device).float()
    diffusion = GaussianDiffusion.from_config(config["diffusion"])
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

    train_loader, val_loader = make_loaders(train_dataset, val_dataset, config)
    iterator = iter(train_loader)
    accumulation = int(config["train"]["gradient_accumulation"])
    logger.log(f"method=gno device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    for step in range(start_step, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = batch_to_device(batch, device)
            condition = normalizer.encode_condition(batch["condition"].float())
            target = normalizer.encode_target(batch["target"].float())
            timesteps = torch.randint(diffusion.steps, (len(target),), device=device)
            loss = diffusion.training_loss(model, target, condition, timesteps) / accumulation
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["gradient_clip_norm"]))
        optimizer.step()
        scheduler.step()
        ema.update(model)
        if step % int(config["train"]["log_every"]) == 0:
            logger.log(
                f"step={step}/{max_steps} diffusion_mse={format_scalar(total_loss)} "
                f"lr={format_scalar(scheduler.get_last_lr()[0])}"
            )
        improved = False
        if step % int(config["train"]["validate_every"]) == 0 or step == max_steps:
            validation, _ = evaluate_batches(
                ema.model, val_loader, normalizer, device, config, diffusion=diffusion,
                max_batches=config["validation"].get("max_batches"),
            )
            improved = validation["relative_l2"] < best
            best = min(best, validation["relative_l2"])
            logger.log(f"step={step} val {format_metrics(validation)} best_relative_l2={format_scalar(best)}")
        if improved or step % int(config["train"]["save_every"]) == 0 or step == max_steps:
            payload = common_payload(config, stats, model, optimizer, scheduler, best, step, ema)
            if improved:
                save_checkpoint(directory / "model_best.pth", payload)
            if step % int(config["train"]["save_every"]) == 0 or step == max_steps:
                save_checkpoint(directory / f"model_step_{step}.pth", payload)
    logger.close()


def train_meanflow_improved(config: dict) -> None:
    device = select_device(config["experiment"].get("device", "auto"))
    directory, logger, train_dataset, val_dataset, state, stats, normalizer = prepare(config, device)
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

    train_loader, val_loader = make_loaders(train_dataset, val_dataset, config)
    iterator = iter(train_loader)
    accumulation = int(config["train"]["gradient_accumulation"])
    mf_config = config["meanflow_improved"]
    logger.log(
        f"method=meanflow_improved device={device} "
        f"parameters={sum(p.numel() for p in model.parameters()):,} "
        f"target=scattered terminal=gaussian sampling_steps={config['sampling']['steps']} "
        f"parameterization={mf_config['velocity_parameterization']}"
    )
    sensitivity = SensitivityMonitor(
        val_dataset, config, normalizer, device, directory,
        predictor=lambda current_model, condition, probe_index: meanflow_predictor(
            current_model, condition, probe_index, normalizer,
            float(config["flow"]["time_scale"]), int(config["experiment"]["seed"]),
        ),
    )
    logger.log(
        f"step={start_step - 1} sensitivity_loss={format_scalar(sensitivity.evaluate(model))} "
        f"sensitivity_probes={sensitivity.count} objective=meanflow_training_only"
    )
    for step in range(start_step, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_compound = total_velocity = 0.0
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = batch_to_device(batch, device)
            condition = normalizer.encode_condition(batch["condition"].float())
            target = normalizer.encode_target(batch["target"].float())
            loss, compound_mse, velocity_mse = improved_meanflow_loss(
                model, target, condition,
                time_scale=float(config["flow"]["time_scale"]),
                time_sampler=mf_config["time_sampler"],
                time_mu=float(mf_config["time_mu"]),
                time_sigma=float(mf_config["time_sigma"]),
                ratio_r_not_equal_t=float(mf_config["ratio_r_not_equal_t"]),
                endpoint_ratio=float(mf_config.get("endpoint_ratio", 0.0)),
                weighting=mf_config["weighting"],
                adaptive_p=float(mf_config["adaptive_p"]),
                adaptive_eps=float(mf_config["adaptive_eps"]),
                velocity_parameterization=mf_config["velocity_parameterization"],
                velocity_loss_weight=float(mf_config.get("velocity_loss_weight", 1.0)),
            )
            (loss / accumulation).backward()
            total_loss += loss.item()
            total_compound += compound_mse.item()
            total_velocity += velocity_mse.item()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["gradient_clip_norm"]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Improved MeanFlow produced non-finite gradients")
        optimizer.step()
        scheduler.step()
        ema.update(model)
        if step % int(config["train"]["log_every"]) == 0:
            sensitivity_value = sensitivity.evaluate(model)
            logger.log(
                f"step={step}/{max_steps} improved_meanflow_loss={format_scalar(total_loss)} "
                f"compound_velocity_mse={format_scalar(total_compound / accumulation)} "
                f"marginal_velocity_mse={format_scalar(total_velocity / accumulation)} "
                f"sensitivity_loss={format_scalar(sensitivity_value)} "
                f"lr={format_scalar(scheduler.get_last_lr()[0])}"
            )
        improved = False
        if step % int(config["train"]["validate_every"]) == 0 or step == max_steps:
            validation, _ = evaluate_batches(
                ema.model, val_loader, normalizer, device, config,
                max_batches=config["validation"].get("max_batches"),
            )
            improved = validation["relative_l2"] < best
            best = min(best, validation["relative_l2"])
            logger.log(f"step={step} val {format_metrics(validation)} best_relative_l2={format_scalar(best)}")
        if improved or step % int(config["train"]["save_every"]) == 0 or step == max_steps:
            payload = common_payload(config, stats, model, optimizer, scheduler, best, step, ema)
            if improved:
                save_checkpoint(directory / "model_best.pth", payload)
            if step % int(config["train"]["save_every"]) == 0 or step == max_steps:
                save_checkpoint(directory / f"model_step_{step}.pth", payload)
    logger.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    method = config["method"]["type"]
    if method == "gno":
        train_gno(config)
    elif method == "meanflow_improved":
        train_meanflow_improved(config)
    else:
        raise ValueError(f"unsupported method: {method}")


if __name__ == "__main__":
    main()
