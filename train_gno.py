"""Train a one-step conditional diffusion U-Net (GNO) for Helmholtz fields.

The training target and physical normalization are intentionally the same as
``train_fno.py``.  The model learns a clean scattered field from a noisy state
at a random diffusion time; inference uses deterministic one-step DDIM with a
fixed noise tensor.  Sensitivity probes are evaluated for diagnostics only and
never contribute to the diffusion loss or optimizer gradients.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion import GaussianDiffusion
from diffusion.sampling import ddim_sample
from models import build_model
from utils.checkpoint import latest_checkpoint, save_checkpoint
from utils.config import load_config, output_dir, resolve_path, save_config
from utils.data import batch_to_device, build_dataset, make_loader
from utils.ema import EMA
from utils.logger import Logger
from utils.metrics import MetricAccumulator, format_metrics, format_scalar, per_sample_metrics
from utils.normalization import PhysicalNormalizer, normalization_stats
from utils.runtime import seed_everything, select_device
from utils.sensitivity import SensitivityMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gno_ema999.yaml")
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
    scheduler_config = config["train"]["scheduler"]
    scheduler_type = str(scheduler_config.get("type", "cosine")).lower()
    if scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, duration), eta_min=float(scheduler_config["eta_min"])
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


def fixed_noise(batch: dict, shape: tuple[int, ...], seed: int,
                frequency_count: int, source_count: int, device: torch.device) -> torch.Tensor:
    """Return reproducible per-sample noise for one-step evaluation/JVPs."""
    flat_indices = (
        batch["model_index"].detach().cpu().long() * frequency_count * source_count
        + batch["frequency_index"].detach().cpu().long() * source_count
        + batch["source_index"].detach().cpu().long()
    )
    samples = []
    for index in flat_indices.tolist():
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(index))
        samples.append(torch.randn(shape, generator=generator))
    return torch.stack(samples).to(device=device, dtype=torch.float32)


def gno_predictor(model, condition: torch.Tensor, probe_index: int,
                  normalizer: PhysicalNormalizer, diffusion: GaussianDiffusion,
                  seed: int) -> torch.Tensor:
    """Deterministic one-step GNO map used by the diagnostic sensitivity JVP."""
    del probe_index
    shape = (condition.shape[0], 2, condition.shape[-2], condition.shape[-1])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(shape, generator=generator).to(condition.device, condition.dtype)
    timesteps = torch.full(
        (condition.shape[0],), diffusion.steps - 1,
        device=condition.device, dtype=torch.long,
    )
    prediction = model(noise, timesteps, normalizer.encode_condition(condition))
    return normalizer.decode_target(prediction)


@torch.no_grad()
def evaluate_gno_batches(model, diffusion, loader, normalizer, device, config,
                         grouped: bool = False, max_batches: int | None = None):
    model.eval()
    overall = MetricAccumulator()
    frequency_count = len(config["data"]["frequencies_hz"])
    source_count = int(config["data"]["source_count"])
    seed = int(config["experiment"]["seed"])
    for batch_index, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch_to_device(cpu_batch, device)
        condition = batch["condition"].float()
        target = batch["target"].float()
        noise = fixed_noise(
            cpu_batch, (2, condition.shape[-2], condition.shape[-1]), seed,
            frequency_count, source_count, device,
        )
        condition_normalized = normalizer.encode_condition(condition)
        target_normalized = normalizer.encode_target(target)
        timesteps = torch.full(
            (len(target),), diffusion.steps - 1, device=device, dtype=torch.long
        )
        prediction_normalized = model(noise, timesteps, condition_normalized)
        prediction = normalizer.decode_target(prediction_normalized.float())
        values = per_sample_metrics(
            prediction_normalized.float(), target_normalized,
            prediction, target, condition, batch["frequency_hz"], config["physics"],
        )
        values["nfe"] = torch.ones(len(target), device=device)
        values["backward_evals"] = torch.zeros(len(target), device=device)
        overall.update(values)
        if grouped:
            # Keep the same grouped API as evaluate_fno.py without requiring
            # a second aggregation implementation for this training entrypoint.
            pass
    return overall.compute()


def checkpoint_payload(config, stats, model, optimizer, scheduler, best, step, ema):
    return {
        "format_version": 1,
        "method": "gno",
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
    if str(config["method"]["type"]).lower() != "gno":
        raise ValueError("train_gno.py requires method.type=gno")
    device = select_device()
    seed_everything(int(config["experiment"]["seed"]))
    directory = output_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    save_config(directory / "config.yaml", config)
    logger = Logger(directory / "runlog.txt")

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    state = load_resume(config, device) if resume else None
    stats = state["normalization"] if state is not None else normalization_stats(
        config, directory / "normalization.npz"
    )
    normalizer = PhysicalNormalizer(stats, device)
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

    train_loader = make_loader(
        train_dataset, int(config["train"]["batch_size"]), True,
        int(config["data"]["num_workers"]),
    )
    val_loader = make_loader(
        val_dataset, int(config["train"]["val_batch_size"]), False,
        int(config["data"]["num_workers"]),
    )
    iterator = iter(train_loader)
    accumulation = int(config["train"].get("gradient_accumulation", 1))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.log(
        f"method=gno device={device} parameters={parameter_count:,} "
        f"ema_decay={float(config['train']['ema_decay']):g} "
        "objective=diffusion_mse_only sensitivity=diagnostic_only"
    )

    sensitivity = SensitivityMonitor(
        val_dataset, config, normalizer, device, directory,
        predictor=lambda current_model, condition, probe_index: gno_predictor(
            current_model, condition, probe_index, normalizer, diffusion,
            int(config["experiment"]["seed"]) + int(probe_index),
        ),
    )
    logger.log(
        f"step={start_step - 1} sensitivity_loss={format_scalar(sensitivity.evaluate(model))} "
        f"sensitivity_probes={sensitivity.count} objective=diffusion_mse_only"
    )

    try:
        for step in range(start_step, max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(accumulation):
                try:
                    cpu_batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    cpu_batch = next(iterator)
                batch = batch_to_device(cpu_batch, device)
                condition = normalizer.encode_condition(batch["condition"].float())
                target = normalizer.encode_target(batch["target"].float())
                timesteps = torch.randint(diffusion.steps, (len(target),), device=device)
                loss = diffusion.training_loss(model, target, condition, timesteps) / accumulation
                loss.backward()
                total_loss += float(loss.detach())
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["train"]["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("GNO training produced non-finite gradients")
            optimizer.step()
            scheduler.step()
            ema.update(model)

            if step % int(config["train"]["log_every"]) == 0 or step == max_steps:
                sensitivity_value = sensitivity.evaluate(model)
                logger.log(
                    f"step={step}/{max_steps} diffusion_mse={format_scalar(total_loss)} "
                    f"sensitivity_loss={format_scalar(sensitivity_value)} "
                    f"lr={format_scalar(scheduler.get_last_lr()[0])} "
                    "sensitivity_objective=false"
                )
            improved = False
            if step % int(config["train"]["validate_every"]) == 0 or step == max_steps:
                validation = evaluate_gno_batches(
                    ema.model, diffusion, val_loader, normalizer, device, config,
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
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = str(args.output_dir)
    train(config, resume=not args.no_resume)


if __name__ == "__main__":
    main()
