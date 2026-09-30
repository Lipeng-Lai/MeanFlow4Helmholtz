"""Plot Marmousi diffusion/Mean Flows residuals at multiple sampling steps."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED_SCRIPT = HERE.parent / "03_CurveVel_A" / "plot_residual_steps.py"
SPEC = importlib.util.spec_from_file_location("marmousi_residual_plot", SHARED_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load shared plotting script: {SHARED_SCRIPT}")

PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)

# Test-set model 0 is Marmousi.  Keep the same 16 Hz/source-2 sample as the
# existing Marmousi comparison figure.
PLOT.BASE.SAMPLE_INDEX = 18
PLOT.BASE.MODEL_INDEX = 0
PLOT.BASE.FREQUENCY_INDEX = 4
PLOT.BASE.SOURCE_INDEX = 2
PLOT.BASE.DATASET_SPLIT = "test"
PLOT.BASE.SOURCE_VELOCITY_PATH = (
    PLOT.BASE.ROOT / "dataset" / "test" / "velocity_models_70x70" / "marmousi.npy"
)
PLOT.OUTPUT_DIR = HERE
PLOT.OUTPUT_STEM = "marmousi_residual_steps"


if __name__ == "__main__":
    PLOT.main()
