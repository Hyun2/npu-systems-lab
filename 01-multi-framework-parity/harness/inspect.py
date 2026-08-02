"""Checkpoint dissection -- what a tool actually stored, not what it claims.

Right now this holds one function, used in step 2 to answer "where do this
model's parameters live". It is the same question step 4 asks of a quantised
checkpoint ("which tensors did the quantiser skip"), which is why it lives
here rather than in the step-2 runner.

The grouping is deliberately dumb: substring matching over parameter names,
with an `other` bucket that catches whatever the patterns miss. Buckets that
silently drop parameters would answer the question wrongly and look right,
so the totals are made to reconcile instead.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["PARAM_GROUPS", "group_parameters"]

# Ordered: the first matching pattern wins. Per-layer embeddings must be
# tested before plain embeddings, since their names contain both.
PARAM_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("per_layer_embeddings", ("per_layer",)),
    ("embeddings", ("embed", "lm_head", "wte")),
    ("attention", ("self_attn", "attn")),
    ("mlp", ("mlp", "feed_forward", "ffn")),
    ("norm", ("norm", "layernorm")),
)

_OTHER = "other"


def group_parameters(named_counts: Iterable[tuple[str, int]]) -> dict[str, dict[str, int]]:
    """Total parameters per group, keyed by group name.

    Takes (name, numel) pairs -- the caller's `model.named_parameters()` with
    shapes already reduced -- so this stays importable without torch.

    Every group is present in the result even when empty, and the group
    totals always sum to the input total. `other` being large is a signal
    that `PARAM_GROUPS` no longer describes this architecture.
    """
    groups = {name: {"params": 0, "tensors": 0} for name, _ in PARAM_GROUPS}
    groups[_OTHER] = {"params": 0, "tensors": 0}

    for name, numel in named_counts:
        lowered = name.lower()
        bucket = _OTHER
        for group, patterns in PARAM_GROUPS:
            if any(p in lowered for p in patterns):
                bucket = group
                break
        groups[bucket]["params"] += numel
        groups[bucket]["tensors"] += 1

    return groups
