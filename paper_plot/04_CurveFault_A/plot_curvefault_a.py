"""Plot sample 2240 using the shared CurveFault-A comparison layout."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED_SCRIPT = HERE.parent / "03_CurveVel_A" / "plot_curvevel_a.py"
SPEC = importlib.util.spec_from_file_location("curvevel_a_plot", SHARED_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load shared plotting script: {SHARED_SCRIPT}")

PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)

PLOT.SAMPLE_INDEX = 2240
PLOT.MODEL_INDEX = 112
PLOT.FREQUENCY_INDEX = 0
PLOT.SOURCE_INDEX = 0
PLOT.OUTPUT_DIR = HERE
PLOT.OUTPUT_STEM = "curvefault_a_comparison"
PLOT.VELOCITY_MODEL_TITLE = "CurveFault-A velocity model"


if __name__ == "__main__":
    PLOT.main()
