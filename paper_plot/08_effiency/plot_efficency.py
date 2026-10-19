"""Runtime helpers shared by the multi-frequency efficiency benchmark.

The file name is kept for compatibility with the existing plotting script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion import GaussianDiffusion  # noqa: E402
from diffusion.sampling import ddim_sample  # noqa: E402
from meanflow import improved_meanflow_sample  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import load_config, output_dir  # noqa: E402
from utils.normalization import PhysicalNormalizer  # noqa: E402
from utils.runtime import seed_everything  # noqa: E402


CHECKPOINT_CONFIGS = {
    "fno": ROOT / "checkpoint" / "fno" / "config.yaml",
    "diffusion": ROOT / "checkpoint" / "gno_ema999" / "config.yaml",
    "meanflow": ROOT / "checkpoint" / "meanflow_improved_ema999" / "config.yaml",
}


def _synchronize(device: torch.device) -> None:
    """Synchronize only asynchronous CUDA work; CPU has no barrier."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_experiment(method: str, device: torch.device) -> tuple:
    """Load a checkpoint and its inference helpers for ``method``."""
    try:
        config_path = CHECKPOINT_CONFIGS[method]
    except KeyError as error:
        raise ValueError(f"unsupported benchmark method: {method}") from error
    config = load_config(config_path)
    checkpoint_path = output_dir(config) / "model_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"benchmark checkpoint not found: {checkpoint_path}. "
            "Train the corresponding experiment before running the figure."
        )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    normalizer = PhysicalNormalizer(state["normalization"], device)
    model = build_model(config["model"]).to(device).float()
    model.load_state_dict(state["ema"] if "ema" in state else state["model"])
    model.eval()
    diffusion = (
        GaussianDiffusion.from_config(config["diffusion"])
        if method == "diffusion"
        else None
    )
    return config, model, normalizer, diffusion


@torch.no_grad()
def _sample(
    method: str,
    model,
    diffusion,
    condition: torch.Tensor,
    noise: torch.Tensor,
    config: dict,
    sampling_steps: int,
) -> torch.Tensor:
    """Run one benchmark sample without moving tensors between devices."""
    if method == "fno":
        if sampling_steps != 1:
            raise ValueError("FNO uses a single direct forward pass")
        return model(condition)
    if method == "diffusion":
        prediction, _ = ddim_sample(
            model,
            diffusion,
            condition,
            noise,
            int(sampling_steps),
            eta=0.0,
            clip_denoised=bool(config["diffusion"].get("clip_denoised", False)),
        )
        return prediction
    if method == "meanflow":
        prediction, _ = improved_meanflow_sample(
            model,
            noise,
            condition,
            int(sampling_steps),
            time_scale=float(config["flow"]["time_scale"]),
        )
        return prediction
    raise ValueError(f"unsupported benchmark method: {method}")
