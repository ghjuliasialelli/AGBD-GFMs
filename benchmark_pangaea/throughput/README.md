# Throughput benchmark (Pareto figure)

Measures inference throughput (samples/s), latency, and peak GPU memory for each of the
11 benchmarked encoders on AGBD-Lite, and renders the RMSE-vs-throughput Pareto plot.

| file | role |
|---|---|
| `throughput.py` | the benchmark itself; one encoder per invocation |
| `gen.py` | prints the 11 `throughput.py` commands (regenerates `run_all.sh`) |
| `run_all.sh` | the 11 commands, as run for the paper |
| `plot.py` | renders the Pareto figure; measured numbers are inlined at the top |
| `pareto_color.png` | the figure as published |

## Running it

`throughput.py` imports `pangaea.*` and its Hydra `config_path="configs"` resolves
relative to the script's own location, so it only runs from the root of a
`pangaea-bench2.0` checkout. It lives here rather than in the fork because it is our
analysis code, not part of the benchmark framework — but it must be executed there:

```bash
# from a pangaea-bench2.0 checkout at the pinned SHA (see ../README.md)
cp /path/to/AGBD-GFMs/benchmark_pangaea/throughput/throughput.py .
cp /path/to/AGBD-GFMs/benchmark_pangaea/throughput/run_all.sh .
bash run_all.sh
```

Copy the printed throughput values into the `models` dict at the top of `plot.py`, then:

```bash
python plot.py --mode color   # writes pareto_color.png
```

`plot.py` has no dependency on the fork and runs anywhere.

## Notes

- Requires a GPU; the script exits if none is found, since CPU numbers are meaningless here.
- `--dummy` benchmarks on random tensors, skipping dataset loading — useful to isolate
  model cost from I/O.
- Real-data mode overlaps loading with compute via a prefetch thread, so the reported
  figure is model-bound rather than loader-bound.
- The published numbers used `batch_size=32`, `--warmup 20 --iterations 100`.
