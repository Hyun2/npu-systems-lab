"""Self-check for the harness skeleton.

Run:  python -m tests.test_harness      (from 01-multi-framework-parity/)

Covers the three pieces that carry real logic. Everything else in the
skeleton is declaration.

The shift case is the one worth reading. It encodes the reason the
comparator reports two families of metric: a framework that offsets every
logit by a constant produces a large max_abs_diff while behaving
identically. If that assertion ever fails, the two families have stopped
being distinguishable and the comparator can no longer tell "different
numbers" from "different behaviour".
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.adapters.base import FrameworkAdapter, NonDeterministicFramework  # noqa: E402
from harness.cache import load_logits, logit_path, save_logits  # noqa: E402
from harness.compare import compare, kl_divergence, softmax  # noqa: E402
from harness.inspect import group_parameters  # noqa: E402
from harness.prompts import PROMPTS, build_long_prompt, by_id  # noqa: E402


def test_identical_logits() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=512)
    m = compare(x, x.copy())

    assert m["max_abs_diff"] == 0.0
    assert m["mean_abs_diff"] == 0.0
    assert abs(m["cosine"] - 1.0) < 1e-12
    assert m["kl"] == 0.0
    assert m["top1_match"] is True
    assert m["top5_overlap"] == 5


def test_constant_shift_is_invisible_to_the_distribution() -> None:
    """A uniform offset changes the logits but not the model's behaviour."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=512)
    shift = 3.7
    m = compare(x, x + shift)

    # raw-logit family sees it
    assert abs(m["max_abs_diff"] - shift) < 1e-9
    assert abs(m["logit_shift"] + shift) < 1e-9  # ref - tgt == -shift

    # distribution family does not
    assert m["kl"] < 1e-12
    assert m["top1_match"] is True
    assert m["top5_overlap"] == 5


def test_real_difference_is_caught() -> None:
    x = np.array([0.1, 5.0, 0.2, 0.3])
    y = np.array([9.0, 5.0, 0.2, 0.3])
    m = compare(x, y, topk=2)

    assert m["top1_match"] is False
    assert m["kl"] > 0.1


def test_kl_is_asymmetric_and_finite_on_overlap() -> None:
    """KL is directional: KL(P||Q) is not KL(Q||P).

    That matters for how results are read. `compare(ref, tgt)` measures
    KL(reference || target) -- how surprised the reference would be by the
    target's distribution. Swapping the arguments gives a different number
    for the same pair of frameworks, so the direction has to stay fixed across
    the whole project or the values are not comparable to each other.

    Note the pair below is deliberately lopsided. A mirrored pair such as
    softmax([2,1,0]) against softmax([0,1,2]) yields *exactly* equal
    divergence in both directions, because relabelling the categories maps
    one onto the other. Asymmetry is a property of KL in general, not of
    every pair, and an earlier version of this test asserted it using such a
    mirrored pair and failed.
    """
    p = softmax(np.array([5.0, 0.0, 0.0]))
    q = softmax(np.array([0.0, 0.0, 1.0]))

    assert kl_divergence(p, q) > 0
    assert abs(kl_divergence(p, q) - kl_divergence(q, p)) > 1.0
    assert kl_divergence(p, p) == 0.0

    # a mirrored pair really is symmetric -- documented so the property above
    # is not mistaken for something that holds universally
    m1 = softmax(np.array([2.0, 1.0, 0.0]))
    m2 = softmax(np.array([0.0, 1.0, 2.0]))
    assert abs(kl_divergence(m1, m2) - kl_divergence(m2, m1)) < 1e-12


def test_mismatched_vocab_is_an_error() -> None:
    try:
        compare(np.zeros(4), np.zeros(5))
    except ValueError:
        return
    raise AssertionError("comparing different vocab sizes must raise")


def test_cache_roundtrip_and_versioned_path() -> None:
    rng = np.random.default_rng(2)
    arr = rng.normal(size=64).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        p = save_logits(
            tmp, "google/gemma-4-E2B-it", "pytorch", "bf16", "short-factual",
            arr, {"framework_version": "torch 2.13.0+cu130"}, token_ids=[1, 2, 3],
        )
        assert p.exists()

        # the semantics version must be in the path, or stale results from a
        # different logit definition would silently mix with fresh ones
        assert "/v1/" in p.as_posix()
        # the model id contains a slash; it must not create a nested directory
        assert "gemma-4-E2B-it" in p.as_posix()
        assert p.as_posix().count("/logits/") == 1

        back, meta = load_logits(tmp, "google/gemma-4-E2B-it", "pytorch", "bf16", "short-factual")
        assert np.array_equal(arr, back)
        assert meta["token_ids"] == [1, 2, 3]
        assert meta["framework_meta"]["framework_version"] == "torch 2.13.0+cu130"
        assert meta["vocab_size"] == 64


def test_pre_rename_sidecar_is_rejected() -> None:
    """A v1 sidecar can predate the backend -> framework rename.

    Both spellings claim semantics_version 1 and share the v1 path, so the
    version guard cannot separate them. Loading must fail where the file is
    named, not with a KeyError in whatever reads the metadata later.

    The two keys were renamed by hand, so a half-converted file is a real
    shape, not a hypothetical one. Each is checked on its own, and the last
    shape is a copy rather than a rename -- it keeps the modern key, which a
    guard that pairs the keys up would wrongly accept.
    """
    rng = np.random.default_rng(3)
    arr = rng.normal(size=8).astype(np.float32)
    shapes = {
        "both keys stale": lambda m: m.update(
            backend=m.pop("framework"), backend_meta=m.pop("framework_meta")
        ),
        "only the top-level key stale": lambda m: m.update(backend=m.pop("framework")),
        "only the nested key stale": lambda m: m.update(backend_meta=m.pop("framework_meta")),
        "old and new spellings coexist": lambda m: m.update(backend_meta=m["framework_meta"]),
    }

    for label, mangle in shapes.items():
        with tempfile.TemporaryDirectory() as tmp:
            p = save_logits(tmp, "m", "pytorch", "bf16", "short-factual", arr, {"v": 1})
            meta_path = p.with_suffix(".meta.json")
            stale = json.loads(meta_path.read_text(encoding="utf-8"))
            mangle(stale)
            meta_path.write_text(json.dumps(stale), encoding="utf-8")

            try:
                load_logits(tmp, "m", "pytorch", "bf16", "short-factual")
            except ValueError as err:
                # The message has to name the offending key, or a caller
                # cannot tell which sidecar to fix.
                assert "backend" in str(err), f"{label}: message names no key"
                continue
            raise AssertionError(f"a pre-rename sidecar loaded silently: {label}")


class _FlakyAdapter(FrameworkAdapter):
    """Returns a different value on the second call. Stands in for a framework
    with sampling left on, or a nondeterministic kernel."""

    def __init__(self, *, flaky: bool) -> None:
        self.flaky = flaky
        self.calls = 0
        self.unloaded = False

    def _load(self, model_path: str, precision: str, **kwargs: object) -> None:
        self.calls = 0

    def unload(self) -> None:
        self.unloaded = True

    def logits(self, token_ids) -> np.ndarray:  # type: ignore[no-untyped-def]
        self.calls += 1
        base = np.arange(8, dtype=np.float64)
        return base + (0.001 * self.calls if self.flaky else 0.0)

    def generate(self, token_ids, max_tokens: int) -> str:  # type: ignore[no-untyped-def]
        return ""

    def meta(self) -> dict:
        return {"framework": "fake"}


def test_determinism_check_rejects_a_flaky_framework() -> None:
    try:
        _FlakyAdapter(flaky=True).load("dummy", "bf16")
    except NonDeterministicFramework:
        return
    raise AssertionError("load() must reject a framework that cannot reproduce itself")


def test_determinism_check_passes_a_stable_framework() -> None:
    _FlakyAdapter(flaky=False).load("dummy", "bf16")


def test_unknown_precision_is_rejected() -> None:
    try:
        _FlakyAdapter(flaky=False).load("dummy", "int8-ish")
    except ValueError:
        return
    raise AssertionError("unknown precision tags must be rejected at load time")


def test_context_manager_unloads() -> None:
    a = _FlakyAdapter(flaky=False)
    with a:
        a.load("dummy", "bf16")
    assert a.unloaded, "leaving the with-block must free VRAM"


def test_prompt_set_shape() -> None:
    cats = {p.category for p in PROMPTS}
    assert {"short", "long", "multi"} <= cats
    assert len({p.id for p in PROMPTS}) == len(PROMPTS), "prompt ids must be unique"
    assert by_id("short-factual").text.startswith("The capital")

    # long prompt must clear the 512-token sliding window by a wide margin;
    # word count is a conservative proxy for token count
    assert len(build_long_prompt().split()) > 1500


def test_parameter_groups_reconcile() -> None:
    named = [
        ("model.embed_tokens.weight", 400),
        ("model.embed_tokens_per_layer.weight", 250),
        ("model.layers.0.self_attn.q_proj.weight", 30),
        ("model.layers.0.mlp.up_proj.weight", 60),
        ("model.layers.0.input_layernorm.weight", 4),
        ("model.some_future_thing.weight", 7),
    ]
    groups = group_parameters(named)

    # Per-layer embeddings must not be swallowed by the plain embedding
    # bucket -- separating them is the whole point of the step-4 hypothesis
    # that one quantiser skipped them and the other did not.
    assert groups["per_layer_embeddings"]["params"] == 250
    assert groups["embeddings"]["params"] == 400

    # Nothing may vanish: an unrecognised name lands in `other`, and the
    # group totals still add up to the input total.
    assert groups["other"]["params"] == 7
    assert sum(g["params"] for g in groups.values()) == sum(n for _, n in named)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
