"""On-disk logit cache.

Only one backend fits in VRAM at a time, so backends cannot be compared
live. Each writes its logits to disk and the comparison happens afterwards
against the files. That makes this module load-bearing: if the cache mixes
results, the comparison is wrong in a way that looks like a real finding.

The guard is `SEMANTICS_VERSION` in the path. Bump it when the meaning of a
stored array changes -- a different position being read, a different dtype,
a different determinism guarantee. Do not bump it for ordinary edits: every
bump invalidates results that took GPU hours to produce.

Layout:

    results/logits/v1/<model>/<backend>/<precision>/<prompt_id>.npy
                                                   /<prompt_id>.meta.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SEMANTICS_VERSION", "logit_path", "save_logits", "load_logits", "have_logits", "slug"]

# Bump only when the semantics of a stored logit array change.
SEMANTICS_VERSION = 1

_SAFE = str.maketrans({"/": "__", " ": "_", ":": "-"})


def slug(s: str) -> str:
    """Make a model or backend id safe to use as one path segment."""
    return s.translate(_SAFE)


def logit_path(
    root: str | Path,
    model: str,
    backend: str,
    precision: str,
    prompt_id: str,
) -> Path:
    """Path for one (model, backend, precision, prompt) logit array."""
    return (
        Path(root)
        / "logits"
        / f"v{SEMANTICS_VERSION}"
        / slug(model)
        / slug(backend)
        / slug(precision)
        / f"{slug(prompt_id)}.npy"
    )


def save_logits(
    root: str | Path,
    model: str,
    backend: str,
    precision: str,
    prompt_id: str,
    logits: np.ndarray,
    meta: dict[str, Any],
    *,
    token_ids: list[int] | None = None,
) -> Path:
    """Write logits plus a sidecar describing what produced them.

    The sidecar carries the adapter's `meta()` verbatim and the token ids the
    logits were computed from. Storing the token ids is what lets a later
    session prove two backends were fed the same input, rather than assume it.
    """
    path = logit_path(root, model, backend, precision, prompt_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(logits)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D logits, got shape {arr.shape}")
    np.save(path, arr)

    sidecar = {
        "model": model,
        "backend": backend,
        "precision": precision,
        "prompt_id": prompt_id,
        "vocab_size": int(arr.shape[0]),
        "dtype": str(arr.dtype),
        "semantics_version": SEMANTICS_VERSION,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "token_ids": list(token_ids) if token_ids is not None else None,
        "backend_meta": meta,
    }
    path.with_suffix(".meta.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def have_logits(root: str | Path, model: str, backend: str, precision: str, prompt_id: str) -> bool:
    return logit_path(root, model, backend, precision, prompt_id).exists()


def load_logits(
    root: str | Path,
    model: str,
    backend: str,
    precision: str,
    prompt_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read back logits and their sidecar."""
    path = logit_path(root, model, backend, precision, prompt_id)
    if not path.exists():
        raise FileNotFoundError(f"no cached logits at {path}")
    arr = np.load(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return arr, meta
