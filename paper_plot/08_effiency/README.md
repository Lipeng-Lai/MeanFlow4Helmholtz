# Efficiency benchmark

Run in the `forward_test` environment:

```bash
conda activate forward_test
MPLBACKEND=Agg python paper_plot/08_effiency/plot_multifrequency_efficiency.py
```

The default benchmark uses validation velocity models 0–9, source index 0
at grid indices `(x, z) = (21, 2)`, three warm-up runs, and ten timed runs.
Reported times are **medians**. Each timed neural measurement is bracketed
by CUDA synchronization. The Torch CPU thread limit is four. Numerical baselines explicitly limit
BLAS/OpenMP libraries to one and four threads, respectively. These limits
do not restrict the number of CUDA threads used by neural inference. Options: `--warmup`, `--runs`, `--threads`,
and `--device`.

- Panel (a) measures a batch of ten velocity models at 12 Hz. It shows FNO,
  Diffusion, and Mean Flows. FNO uses one direct forward pass and has a green
  bar only at one step; Diffusion and Mean Flows use 1, 2, 4, and 8 steps.
- Panel (b) shows Mean Flows, two Numerical Simulation baselines (one and four
  CPU threads), and two matching Speed Up curves for frequency sets `{3}`,
  `{3, 4}`, …, `{3, …, 16}` Hz (14 frequencies in total).
  **For each velocity model, all
  requested frequencies form the leading batch dimension of one Mean Flows
  sampler call.** Models are processed sequentially and each repetition's
  runtime is divided by the ten-model count before taking the median.
  Numerical Simulation uses the CPU solver for each model/frequency
  independently. For each repetition index, its per-frequency times are
  summed over the requested set, then the median of the ten sums is taken.
  Each Speed Up curve is its CPU numerical baseline median time divided
  by Mean Flows median time.
  The horizontal label is `Number of frequencies`, with ticks 1–14.

Learned-method timings include inference on normalized, device-resident
inputs. Checkpoint loading, analytical background generation, transfers,
normalization, and output decoding are outside the timing interval. The
numerical reference includes operator/field assembly, sparse factorization,
solution, and finite-output and residual checks. These boundaries describe
inference throughput with prepared neural inputs rather than a complete
application pipeline. Numerical outputs must be finite and satisfy the
configured relative residual tolerance (`1e-8`). The benchmark has no CuPy
or GPU numerical solver dependency.

Outputs:

- `multifrequency_efficiency.pdf` / `.png`: the two-panel figure.
- `sampling_step_efficiency.csv`: panel (a), milliseconds per ten-model batch;
  FNO entries above one step are empty because it is a direct operator.
- `multifrequency_efficiency.csv`: panel (b), Mean Flows and both CPU baselines in seconds
  per model, plus each baseline's speed-up ratio. Column names explicitly
  identify the one-thread and four-thread references.
- `efficiency_timing_runs.csv`: all timed repetitions with explicit units.
- `efficiency_benchmark_metadata.json`: hardware, environment, checkpoint
  configurations, timing scope, median aggregation, and batching protocol.

Raw numerical repetitions use method names `numerical_1threads` and
`numerical_4threads`. Metadata records the actual threadpool limits for
each baseline. The numerical solves still process models and frequencies
sequentially; the thread count limits the libraries within each solve.
