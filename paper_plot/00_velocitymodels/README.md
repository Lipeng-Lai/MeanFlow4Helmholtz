# Validation velocity-model figures

`plot_val_velocity_models.py` reads `dataset/val/models.csv` and plots the 80
independent validation models (`variant=original`). It creates two figures:

- CurveVel-A: `curvevel_a_1` and `curvevel_a_2`, 40 models in a 5×8 grid.
- FlatFault-A: `flatfault_a_1` and `flatfault_a_2`, 40 models in a 5×8 grid.

The smoothed rows are augmentation variants of the same 80 base models and are
therefore excluded from these overview figures.
