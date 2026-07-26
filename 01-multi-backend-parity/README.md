# 01. Multi-Backend Parity Harness

**Status: in progress.** Step 0 of 8 complete -- backends surveyed, third
backend chosen, environments built and verified against the GPU. No parity
measurements yet.

| Environment | Verified | Result |
|---|---|---|
| PyTorch | `torch.cuda.is_available()` plus a real bf16 matmul | torch 2.13.0+cu130, transformers 5.14.1, RTX 3060 cap 8.6, 11.52 GiB free |
| vLLM | `nvidia-smi` inside the container | GPU visible via CDI (`--device nvidia.com/gpu=all`, not `--gpus`) |
| llama.cpp | `llama-cli --list-devices` | `CUDA0: NVIDIA GeForce RTX 3060 (11909 MiB)`, built for arch 86 only |
| OpenVINO | not yet | built at step 6 |

## The question

Two inference backends load the same model and return different logits.
Is that difference acceptable numerical noise, or a bug in the port?

Answering it by eye does not scale. This project builds a harness that
answers it with a number and a stated threshold, then uses that harness to
compare four quantization tiers across four backends.

## Approach

```
PyTorch reference  ->  backend adapters  ->  quantization tiers  ->  logit parity + accuracy + profiling
```

The controlling idea is to establish an **FP16-to-FP16 baseline first**.
Before any quantization is applied, PyTorch and vLLM are compared running
the same weights at the same precision. Whatever difference shows up there
is pure implementation difference -- kernel selection, reduction order,
attention backend. Every later measurement is read against that floor,
so quantization loss is never confused with porting noise.

Self-written code is deliberately limited to glue and instrumentation:
the adapter interface, the comparator, a checkpoint inspector, and the
report generator. Quantization, serving and evaluation are done with the
tools the industry actually uses.

## Target model

`google/gemma-4-E2B-it`, text-only configuration. Apache 2.0, released
2026-03-31.

The "E" in E2B means *effective* parameters. The checkpoint holds 5.1B
parameters (10.25GB in bf16); Per-Layer Embeddings keep the
compute-active count near 2.3B. Text-only loading skips the vision
(~150M) and audio (~300M) encoders, landing around 9.3GB -- which is what
makes this fit a 12GB card at all.

Secondary model for cross-checking harness generality:
`meta-llama/Llama-3.2-1B-Instruct`.

## Backends

Surveyed 2026-07-26 against vendor documentation and source.

| Backend | Status | Evidence |
|---|---|---|
| PyTorch / Transformers | Reference | `Gemma4ForConditionalGeneration` |
| vLLM | Supported | Listed in supported models with model-specific notes; merged PRs from 2026-04-03, three days after release |
| llama.cpp | Supported | Google publishes official QAT GGUF for E2B |
| OpenVINO | Supported, adopted as third backend | OpenVINO org publishes an E2B IR, split into per-layer-embedding / language-model / vision files |
| ONNX Runtime | Deferred | Runtime registers `gemma4` and `gemma4_text`, but the official `builder.py` has no Gemma4 branch. Conversion path unproven |
| ExecuTorch | Deferred | No Gemma reference in official LLM docs or examples |

The third-backend choice was reversed during the survey. ONNX Runtime was
the original first pick for being framework-independent; it lost to
OpenVINO because runtime support and conversion support turned out to be
different things. Both deferred backends remain on the extension list with
explicit time boxes.

## Quantization tiers

Four tiers of increasing sophistication, so the accuracy-versus-effort
curve is visible rather than asserted.

| Tier | Method | Artifact |
|---|---|---|
| 1 | Round-to-nearest | llama.cpp `Q4_0` |
| 2 | k-quant | llama.cpp `Q4_K_M` |
| 3 | Calibration-based | AWQ (group sizes 128 / 64 / 32), GPTQ |
| 4 | Quantization-aware training | `google/gemma-4-E2B-it-qat-w4a16-ct`, `...-qat-q4_0-gguf` |

### An open question found during the survey

Google ships the same QAT model in two formats, both labelled 4-bit:

| Artifact | Size | Reduction from bf16 (10.25GB) |
|---|---|---|
| `qat-w4a16-ct` (compressed-tensors) | 8.35GB | 19 percent |
| `qat-q4_0-gguf` (llama.cpp) | 3.35GB + 0.99GB mmproj | 67 percent |

A 2.5x gap between two artifacts of the same model at the same nominal bit
width. The working hypothesis is that the compressed-tensors build leaves
the Per-Layer Embedding tables unquantized -- OpenVINO's IR keeps those in
a separate 2.35GB file even at int4, which is suggestive.

Confirming or refuting this is the first job of the checkpoint inspector,
because it is this project's central claim in miniature: *the same "4-bit"
label can mean different things in different backends.*

## Layout

```
01-multi-backend-parity/
  harness/
    adapters/    one adapter per backend, behind a single interface
    compare.py   logit comparator
    inspect.py   checkpoint anatomy -- what did the tool actually store
    report.py    results table generation
  configs/
  results/
```

## Reproducing

Instructions land here once step 1 produces something runnable.
