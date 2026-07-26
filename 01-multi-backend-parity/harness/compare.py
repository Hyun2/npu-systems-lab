"""Logit comparison.

This module reports numbers and does not judge them. Thresholds live in
configs and are applied downstream by the report step, so that a
disappointing result cannot be fixed by quietly widening a tolerance in the
comparator.

Two families of metric are reported side by side, and the distinction
matters when reading results:

- Raw-logit metrics (`max_abs_diff`, `mean_abs_diff`, `cosine`) see the
  actual numbers a backend produced. They move if one backend offsets every
  logit by a constant.
- Distribution metrics (`kl`, `top1_match`, `top5_overlap`) see only what the
  model would do next. Softmax is shift-invariant, so a constant offset is
  invisible to them.

When the two families disagree -- large `max_abs_diff` but near-zero `kl` --
the backends have not diverged in substance, they differ by an offset.
`logit_shift` is reported to make that case immediately readable rather than
something to be inferred.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compare", "softmax", "kl_divergence"]


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D logit vector."""
    z = np.asarray(x, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) in nats, for probability vectors.

    Terms where p == 0 contribute nothing (0 log 0 is taken as 0). A point
    where q == 0 while p > 0 is genuinely infinite divergence: the target
    assigns zero probability to something the reference expects. That is
    reported as inf rather than smoothed away, because in this project it
    means a backend underflowed and that is a finding, not noise.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    support = p > 0
    if np.any(q[support] == 0.0):
        return float("inf")
    return float(np.sum(p[support] * np.log(p[support] / q[support])))


def compare(ref: np.ndarray, tgt: np.ndarray, *, topk: int = 5) -> dict[str, float | bool | int]:
    """Compare a target backend's logits against the reference's.

    Both arrays are next-token logits over the same vocabulary, as returned
    by `BackendAdapter.logits`. Computation is in float64 regardless of the
    input dtype, so that the comparison itself does not contribute error to
    the thing being measured.
    """
    ref = np.asarray(ref, dtype=np.float64).ravel()
    tgt = np.asarray(tgt, dtype=np.float64).ravel()

    if ref.shape != tgt.shape:
        raise ValueError(f"vocabulary size differs: {ref.shape} vs {tgt.shape}")
    if ref.size == 0:
        raise ValueError("empty logit vector")

    diff = ref - tgt
    ref_norm = np.linalg.norm(ref)
    tgt_norm = np.linalg.norm(tgt)

    p = softmax(ref)
    q = softmax(tgt)

    ref_top = set(np.argsort(ref)[-topk:].tolist())
    tgt_top = set(np.argsort(tgt)[-topk:].tolist())

    return {
        # raw-logit family: shift-sensitive
        "max_abs_diff": float(np.abs(diff).max()),
        "mean_abs_diff": float(np.abs(diff).mean()),
        "logit_shift": float(diff.mean()),
        "cosine": float(ref @ tgt / (ref_norm * tgt_norm)) if ref_norm and tgt_norm else float("nan"),
        # distribution family: shift-invariant
        "kl": kl_divergence(p, q),
        "top1_match": bool(int(ref.argmax()) == int(tgt.argmax())),
        f"top{topk}_overlap": int(len(ref_top & tgt_top)),
    }
