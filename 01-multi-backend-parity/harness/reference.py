"""Step 2 -- produce the reference logits every later backend is judged against.

Run on the GPU machine:

    python -m harness.reference --model google/gemma-4-E2B-it --precision bf16

One model load does three jobs, because the load is the expensive part and
the card holds exactly one model at a time:

1. records what actually got loaded (class, dtype, attention kernel) and how
   much device memory it really costs -- step 1's plan carried a *calculated*
   9.3GB for the text-only tower, never a measured one;
2. writes fp32 logits for the fixed prompt set through the versioned cache;
3. dumps the parameter layout, so step 4 can predict which tensors a
   quantiser will leave alone.

Prompts are tokenised once, here, with the reference tokeniser. Adapters
receive integers (D1) and the token ids travel into each sidecar, so a later
session can prove two backends saw the same input.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .adapters.pytorch import PyTorchAdapter
from .cache import save_logits, slug
from .inspect import group_parameters
from .prompts import PROMPTS

_GIB = 1024 ** 3


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.results) / "reference" / slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    with PyTorchAdapter() as adapter:
        t0 = time.perf_counter()
        adapter.load(
            args.model,
            args.precision,
            device=args.device,
            attn_implementation=args.attn,
        )
        load_seconds = time.perf_counter() - t0

        meta = adapter.meta()
        _report_load(meta, load_seconds)

        structure = _structure(adapter, meta)
        _report_structure(structure)

        runs = _run_prompts(adapter, args, meta)

    report = {
        "meta": meta,
        "load_seconds": round(load_seconds, 2),
        "structure": structure,
        "prompts": runs,
    }
    path = out_dir / f"{slug(args.precision)}-structure.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwrote {path}")


# -- the three jobs --------------------------------------------------------


def _structure(adapter: PyTorchAdapter, meta: dict[str, Any]) -> dict[str, Any]:
    """Parameter layout and attention schedule, as loaded."""
    model = adapter._model  # noqa: SLF001 -- inspection is this module's job
    config = model.config

    named = [
        {
            "name": n,
            "shape": tuple(p.shape),
            "dtype": str(p.dtype),
            "params": p.numel(),
            "bytes": p.numel() * p.element_size(),
        }
        for n, p in model.named_parameters()
    ]
    groups = group_parameters((p["name"], p["params"]) for p in named)

    layer_types = list(getattr(config, "layer_types", []) or [])
    return {
        "config": {
            k: getattr(config, k, None)
            for k in (
                "model_type",
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "sliding_window",
                "num_kv_shared_layers",
                "rope_theta",
            )
        },
        "attention_schedule": {
            "layer_types": layer_types,
            "counts": {t: layer_types.count(t) for t in sorted(set(layer_types))},
            "full_attention_at": [i for i, t in enumerate(layer_types) if t != "sliding_attention"],
            "last_layer": layer_types[-1] if layer_types else None,
        },
        "parameter_groups": groups,
        "total_params": sum(p["params"] for p in named),
        "total_bytes": sum(p["bytes"] for p in named),
        "parameters": named,
    }


def _run_prompts(adapter: PyTorchAdapter, args: argparse.Namespace, meta: dict[str, Any]) -> list[dict[str, Any]]:
    tokenizer = adapter.tokenizer
    runs: list[dict[str, Any]] = []

    print(f"\n{'prompt':<16} {'tokens':>7} {'latency_s':>10} {'peak_alloc_GiB':>15}")
    for prompt in PROMPTS:
        token_ids = tokenizer(prompt.text)["input_ids"]

        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        t0 = time.perf_counter()
        logits = adapter.logits(token_ids)
        seconds = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0

        run = {
            "prompt_id": prompt.id,
            "category": prompt.category,
            "n_tokens": len(token_ids),
            "latency_seconds": round(seconds, 4),
            "peak_allocated_bytes": peak,
            "vocab_size": int(logits.shape[0]),
        }
        runs.append(run)
        print(f"{prompt.id:<16} {len(token_ids):>7} {seconds:>10.4f} {peak / _GIB:>15.3f}")

        save_logits(
            args.results,
            args.model,
            adapter.name,
            args.precision,
            prompt.id,
            logits,
            {**meta, "run": run},
            token_ids=list(token_ids),
        )

    return runs


# -- reporting -------------------------------------------------------------


def _report_load(meta: dict[str, Any], load_seconds: float) -> None:
    vram = meta["vram"]
    print(f"loaded in {load_seconds:.1f}s")
    print(f"  class      {meta['model_class']}  (config {meta['config_class']}, model_type {meta['model_type']})")
    print(f"  dtype      {meta['effective_dtype']}")
    print(f"  attention  requested {meta['requested']['attn_implementation']!r}"
          f" -> effective {meta['effective_attn_implementation']!r}")
    print(f"  params     {meta['param_count'] / 1e9:.3f}B")
    print(f"  VRAM       device {vram['device_used_bytes'] / _GIB:.3f} GiB used"
          f"  |  torch allocated {vram['torch_allocated_bytes'] / _GIB:.3f} GiB"
          f"  |  free {vram['device_free_after_load_bytes'] / _GIB:.3f} GiB"
          f" of {vram['device_total_bytes'] / _GIB:.3f} GiB")


def _report_structure(structure: dict[str, Any]) -> None:
    total = structure["total_params"]
    print("\nparameter groups")
    for group, counts in structure["parameter_groups"].items():
        share = 100.0 * counts["params"] / total if total else 0.0
        print(f"  {group:<22} {counts['params'] / 1e6:>10.1f}M  {share:>5.1f}%  ({counts['tensors']} tensors)")
    print(f"  {'total':<22} {total / 1e6:>10.1f}M")

    sched = structure["attention_schedule"]
    if sched["layer_types"]:
        print(f"\nattention  {sched['counts']}"
              f"\n  full-attention layers: {sched['full_attention_at']}"
              f"\n  last layer: {sched['last_layer']}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-4-E2B-it")
    p.add_argument("--precision", default="bf16", choices=("bf16", "fp16"))
    p.add_argument("--results", default="results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn", default="eager", help="attention implementation to request")
    return p.parse_args()


if __name__ == "__main__":
    main()
