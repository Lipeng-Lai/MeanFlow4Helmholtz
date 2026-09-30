"""Model factory for the conditional Helmholtz U-Net."""

from .unet import ConditionalUNet


def build_model(config: dict):
    backbone = str(config["backbone"]).lower()
    if backbone == "unet":
        return ConditionalUNet.from_config(config)
    raise ValueError(f"unknown backbone: {backbone}")


__all__ = ["ConditionalUNet", "build_model"]
