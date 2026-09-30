"""Plot Marmousi test sample using the shared 2-by-4 layout."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED_SCRIPT = HERE.parent / "03_CurveVel_A" / "plot_curvevel_a.py"
SPEC = importlib.util.spec_from_file_location("marmousi_plot", SHARED_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load shared plotting script: {SHARED_SCRIPT}")

PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)

# Test-set model 0 is Marmousi.  Use 16 Hz and source 2, matching the
# frequency/source choice used by the preceding comparison figures.
PLOT.SAMPLE_INDEX = 18
PLOT.MODEL_INDEX = 0
PLOT.FREQUENCY_INDEX = 4
PLOT.SOURCE_INDEX = 2
PLOT.DATASET_SPLIT = "test"
PLOT.SOURCE_VELOCITY_PATH = (
    PLOT.ROOT / "dataset" / "test" / "velocity_models_70x70" / "marmousi.npy"
)
PLOT.OUTPUT_DIR = HERE
PLOT.OUTPUT_STEM = "marmousi_comparison"
PLOT.VELOCITY_MODEL_TITLE = "Marmousi velocity model"
PLOT.PREDICTION_STEPS = 2


if __name__ == "__main__":
    PLOT.main()
