"""Plot a validation FlatFault-A sample using the shared comparison layout."""

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

# Use the original flatfault_a_2 source model 153 (validation model 40),
# at 14 Hz with source position (x, z) = (41, 2).
PLOT.SAMPLE_INDEX = 2286
PLOT.MODEL_INDEX = 40
PLOT.FREQUENCY_INDEX = 11
PLOT.SOURCE_INDEX = 2
PLOT.OUTPUT_DIR = HERE
PLOT.OUTPUT_STEM = "curvefault_a_comparison"
PLOT.VELOCITY_MODEL_TITLE = "FlatFault-A velocity model"


if __name__ == "__main__":
    PLOT.main()
