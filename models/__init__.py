"""Model factory for the Helmholtz Fourier neural operator."""

from .fno import FourierNeuralOperator2d
from .unet import ConditionalUNet


def build_model(config: dict):
    backbone = str(config["backbone"]).lower()
    if backbone == "fno":
        return FourierNeuralOperator2d.from_config(config)
    if backbone == "unet":
        return ConditionalUNet.from_config(config)
    raise ValueError(f"unknown backbone: {backbone}")


__all__ = ["ConditionalUNet", "FourierNeuralOperator2d", "build_model"]
