"""Plot Overthrust diffusion/Mean Flows residuals at multiple sampling steps."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED_SCRIPT = HERE.parent / "03_CurveVel_A" / "plot_residual_steps.py"
SPEC = importlib.util.spec_from_file_location("overthrust_residual_plot", SHARED_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load shared plotting script: {SHARED_SCRIPT}")

PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)

# Test-set model 3 is overthrust.  Keep the same 16 Hz/source-1 sample as the
# existing Overthrust comparison figure.
PLOT.BASE.SAMPLE_INDEX = 221
PLOT.BASE.MODEL_INDEX = 3
PLOT.BASE.FREQUENCY_INDEX = 13
PLOT.BASE.SOURCE_INDEX = 1
PLOT.BASE.DATASET_SPLIT = "test"
PLOT.BASE.FREQUENCIES_HZ = list(range(3, 17))
PLOT.BASE.SOURCE_VELOCITY_PATH = (
    PLOT.BASE.ROOT / "dataset" / "velocity_models_70x70" / "overthrust.npy"
)
PLOT.OUTPUT_DIR = HERE
PLOT.OUTPUT_STEM = "overthrust_residual_steps"


if __name__ == "__main__":
    PLOT.main()
