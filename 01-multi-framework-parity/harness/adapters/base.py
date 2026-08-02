"""The contract every framework must satisfy.

Four frameworks (PyTorch, vLLM, llama.cpp, OpenVINO) are driven through one
interface so that a difference in their output can be attributed to the
framework rather than to how each was called.

Two properties of this contract are load-bearing and easy to lose:

1. `logits` takes token ids, never a string. If each framework tokenised its
   own prompt, the measured difference would mix kernel behaviour with
   tokeniser behaviour, and the project could not tell them apart. Tokenise
   once, upstream, and feed every framework the same integers.

2. Loading is scoped. The reference model is 10.25GB against 11.9GB of
   VRAM, so two frameworks cannot be resident at once. Adapters are context
   managers and free their memory on exit -- forgetting to unload is not a
   slow leak here, it is an out-of-memory error on the next framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

# Precision tags used across the project. The string is passed to the
# adapter, which maps it onto whatever its framework actually calls the thing.
PRECISIONS = (
    "bf16",
    "fp16",
    "awq-int4",
    "gptq-int4",
    "q4_0",
    "q4_k_m",
    "qat-w4a16",
)

# A short, model-agnostic token sequence used only to check that a freshly
# loaded framework returns the same numbers twice. Low token ids exist in every
# vocabulary we use (smallest is Llama 3.2 at 128,256), and the content is
# irrelevant -- reproducibility is the only thing being measured.
_DETERMINISM_PROBE: tuple[int, ...] = tuple(range(1, 17))


class NonDeterministicFramework(RuntimeError):
    """Raised when a framework returns different logits for the same input.

    Measuring parity against a framework that cannot reproduce itself is
    meaningless: any difference found later could be the framework disagreeing
    with itself rather than with another framework.
    """


class FrameworkAdapter(ABC):
    """Base class for framework adapters.

    Subclasses implement the `_`-prefixed methods. The public `load` wraps
    `_load` so the determinism check cannot be skipped by forgetting to call
    it.
    """

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "FrameworkAdapter":
        return self

    def __exit__(self, *exc: object) -> bool:
        self.unload()
        return False

    def load(
        self,
        model_path: str,
        precision: str,
        *,
        check_determinism: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load a model, then verify the framework reproduces itself.

        Set `check_determinism=False` only when the cost of one extra forward
        pass matters and determinism has already been established for this
        (framework, precision) pair in the same session.
        """
        if precision not in PRECISIONS:
            raise ValueError(
                f"unknown precision {precision!r}; expected one of {PRECISIONS}"
            )
        self._load(model_path, precision, **kwargs)
        if check_determinism:
            self.assert_deterministic()

    @abstractmethod
    def _load(self, model_path: str, precision: str, **kwargs: Any) -> None:
        """Bring the model into memory. Called by `load`."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model and its device memory.

        Must be safe to call twice, and safe to call when nothing is loaded.
        """

    # -- measurement -------------------------------------------------------

    @abstractmethod
    def logits(self, token_ids: Sequence[int]) -> np.ndarray:
        """Next-token logits at the final position. shape=(vocab_size,).

        Must be deterministic: greedy, no sampling, thinking mode off, batch
        size 1. Returns raw logits, not probabilities -- softmax discards the
        scale differences this project is looking for.
        """

    @abstractmethod
    def generate(self, token_ids: Sequence[int], max_tokens: int) -> str:
        """Decode a continuation. For eyeballing output, not for measurement."""

    @abstractmethod
    def meta(self) -> dict[str, Any]:
        """What actually ran, recorded verbatim.

        Framework name and version, the kernel or op set actually selected,
        artifact size, memory used. Requesting `awq` from vLLM may get you
        `awq_marlin`; without this the resulting numbers cannot be explained.

        Store raw strings rather than normalising them. Normalisation now
        discards detail that turns out to matter later, and it cannot be
        recovered.
        """

    # -- self-check --------------------------------------------------------

    def assert_deterministic(self, token_ids: Sequence[int] | None = None) -> None:
        """Fail unless two identical calls produce byte-identical logits.

        Byte comparison rather than `==` so that NaN, which never equals
        itself, does not read as nondeterminism -- a framework emitting NaN is
        broken, but it is a different fault and should surface as such.
        """
        probe = tuple(token_ids) if token_ids is not None else _DETERMINISM_PROBE
        first = self.logits(probe)
        second = self.logits(probe)

        if first.shape != second.shape or first.dtype != second.dtype:
            raise NonDeterministicFramework(
                f"{type(self).__name__}: shape/dtype changed between identical "
                f"calls ({first.shape}/{first.dtype} then {second.shape}/{second.dtype})"
            )
        if first.tobytes() != second.tobytes():
            delta = float(np.abs(first.astype(np.float64) - second.astype(np.float64)).max())
            raise NonDeterministicFramework(
                f"{type(self).__name__}: identical input produced different logits "
                f"(max delta {delta:.3e}). Check greedy decoding, thinking mode, "
                f"seed, and batch size before measuring anything."
            )
