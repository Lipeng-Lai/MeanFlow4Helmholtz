"""Plot the Overthrust test sample using the Marmousi layout."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED_SCRIPT = HERE.parent / "03_CurveVel_A" / "plot_curvevel_a.py"
SPEC = importlib.util.spec_from_file_location("overthrust_plot", SHARED_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load shared plotting script: {SHARED_SCRIPT}")

PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)

# Test-set model 3 is Overthrust, at 16 Hz with source (x, z) = (31, 2).
# Inherit the shared one-step prediction and 1/2/4/8-step comparison settings.
PLOT.SAMPLE_INDEX = 221
PLOT.MODEL_INDEX = 3
PLOT.FREQUENCY_INDEX = 13
PLOT.SOURCE_INDEX = 1
PLOT.DATASET_SPLIT = "test"
# Use 3--16 Hz; the shared loader maps stored fields by physical frequency
# and solves frequencies that are absent from the test arrays.
PLOT.FREQUENCIES_HZ = list(range(3, 17))
PLOT.SOURCE_VELOCITY_PATH = (
    PLOT.ROOT / "dataset" / "velocity_models_70x70" / "overthrust.npy"
)
PLOT.OUTPUT_DIR = HERE
PLOT.OUTPUT_STEM = "overthrust_comparison"
PLOT.VELOCITY_MODEL_TITLE = "Overthrust velocity model"


if __name__ == "__main__":
    PLOT.main()
