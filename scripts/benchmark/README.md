# Pipeline benchmark

`benchmark.py` measures the performance of a complete NeuROM **simulation** —
the full pipeline of the 1D clamped-beam problem (build connectivity, fields,
mesh, mapping, physics, loss and FEM model, then train with LBFGS). It runs the
whole simulation many times and reports the **mean** and **standard deviation**
of the wall-clock time, split into a *setup* and a *training* phase plus the
*total*.

The problem is self-contained (no external mesh file needed) and scales with
`--nodes`, so it can be used to track performance over time or compare changes.

## Usage

```bash
# Default: 10 measured runs (after 2 warmup runs), 100 nodes, 3 epochs
uv run python scripts/benchmark/benchmark.py

# Larger problem, more repetitions, and save the raw numbers to JSON
uv run python scripts/benchmark/benchmark.py --nodes 400 --repeats 30 --json results.json

# Different quadrature rule
uv run python scripts/benchmark/benchmark.py --quadrature two-points
```

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `-r`, `--repeats` | `10` | Number of measured runs used for the statistics. |
| `-w`, `--warmup` | `2` | Unmeasured warmup runs (absorb one-time costs; excluded from stats). |
| `-n`, `--nodes` | `100` | Number of mesh nodes (problem size). |
| `-e`, `--epochs` | `3` | Number of training epochs per simulation. |
| `-q`, `--quadrature` | `midpoint` | Quadrature rule: `midpoint` or `two-points`. |
| `--device` | `cpu` | Torch device, e.g. `cpu` or `cuda` (CUDA timings are synchronised). |
| `--seed` | `0` | RNG seed, held constant so every run does identical work. |
| `--threads` | (torch default) | Override the CPU thread count for more reproducible timing. |
| `--json` | (none) | Path to write the config, summary stats and per-run timings as JSON. |

## Example output

```
NeuROM pipeline benchmark
  device=cpu dtype=torch.float32 nodes=100 epochs=3 quadrature=midpoint
  repeats=10 (after 2 warmup run(s))

phase      mean [ms]    std [ms]    min [ms]    max [ms]
--------------------------------------------------------
setup          0.73        0.15        0.61        0.96
train        125.22        0.37      124.91      125.82
total        125.95        0.38      125.52      126.57
```

## Notes

- For the most stable numbers, pin the thread count (`--threads 1`) and increase
  `--repeats`. Wall-clock timing is sensitive to other load on the machine.
- The seed is reset before every run, so the work performed is identical across
  repetitions; the spread therefore reflects machine/runtime variability rather
  than differences in the computation.
