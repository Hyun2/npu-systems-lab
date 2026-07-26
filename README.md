# NPU Systems Lab

Hands-on projects across the NPU software stack: model optimization,
quantization, multi-backend inference, runtime integration, and profiling.

Each numbered directory is a self-contained project with its own README,
code, and measured results. Nothing here is a tutorial reimplementation --
the code is glue and instrumentation around production tools, and the
output is measurements plus the reasoning behind them.

## Projects

| # | Project | Focus | Status |
|---|---|---|---|
| 01 | [Multi-Backend Parity Harness](01-multi-backend-parity/) | Numerical parity of one model across PyTorch, vLLM, llama.cpp and OpenVINO, under four quantization tiers | In progress |

## Measurement environment

Every number in this repository is produced on one fixed machine, so
results stay comparable across projects.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12GB, Compute Capability 8.6 (Ampere) |
| CPU | AMD Ryzen 7 5800X, 8C/16T |
| RAM | 46GB |
| Storage | NVMe SSD |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8, driver 580.159.03, CUDA 12.0 toolkit |

When a project needs hardware this machine does not have -- FP8 requires
Ada (8.9) or newer, for example -- that is stated in the project README
rather than quietly worked around.

## Conventions

- Each project directory carries its own README with the question it
  answers, the method, and the results table.
- Results are committed. Model weights and virtual environments are not.
- Where a measurement contradicts an initial assumption, both the
  assumption and the correction are kept in the write-up. The correction
  is usually the interesting part.
