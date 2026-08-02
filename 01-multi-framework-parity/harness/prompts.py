"""The fixed prompt set.

Three categories, chosen because each exercises a different code path rather
than because three is a nice number:

- `short`  -- a few tokens. Nothing exotic runs; if frameworks disagree here,
              the disagreement is in the basic forward pass.
- `long`   -- over 2K tokens. Gemma 4 E2B uses a 512-token sliding window on
              most layers, so a long prompt crosses the boundary between
              windowed and global attention. Frameworks implement that
              boundary differently and a short prompt never reaches it.
- `multi`  -- non-Latin scripts. The model covers 140+ languages and the
              tokeniser takes a different path for multi-byte text.

Prompts are text here and are tokenised once, upstream, by the reference
tokeniser. Adapters receive the resulting integers. See `adapters/base.py`
for why.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Prompt", "PROMPTS", "by_id", "build_long_prompt"]


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    text: str


# ponytail: the long prompt is assembled from a fixed sentence pool rather
# than pasted as a 2K-token literal. Repetition is not ideal -- a natural
# passage would exercise the attention window slightly differently -- but it
# keeps the file readable and is reproducible byte-for-byte, which is the
# property that actually matters for parity. Swap in a real passage if the
# long-context results ever look suspiciously clean.
_LONG_POOL = (
    "A compiler lowers a high level description into machine specific code.",
    "Quantisation trades numerical precision for memory bandwidth and speed.",
    "An attention layer compares every position against every other position.",
    "Sliding window attention bounds that comparison to a fixed neighbourhood.",
    "A kernel is the concrete implementation of one operation on one device.",
    "Reduction order changes floating point results without changing the maths.",
    "Calibration data determines where a quantiser places its scale factors.",
    "Memory bandwidth, not arithmetic throughput, limits most inference work.",
)


def build_long_prompt(target_words: int = 1600) -> str:
    """Assemble a deterministic long passage.

    Word count is a proxy for token count; 1600 words lands comfortably past
    2K tokens for every tokeniser in this project. The exact length does not
    need to be pinned -- crossing the 512-token window boundary by a wide
    margin is the requirement.
    """
    parts: list[str] = []
    words = 0
    i = 0
    while words < target_words:
        sentence = _LONG_POOL[i % len(_LONG_POOL)]
        parts.append(sentence)
        words += len(sentence.split())
        i += 1
    parts.append("Summarising the discussion above, the single most important point is that")
    return " ".join(parts)


PROMPTS: tuple[Prompt, ...] = (
    Prompt("short-factual", "short", "The capital of France is"),
    Prompt("short-arith", "short", "Two plus two equals"),
    Prompt("long-technical", "long", build_long_prompt()),
    Prompt("multi-ko", "multi", "대한민국의 수도는"),
    Prompt("multi-ja", "multi", "日本の首都は"),
)


def by_id(prompt_id: str) -> Prompt:
    for p in PROMPTS:
        if p.id == prompt_id:
            return p
    raise KeyError(f"unknown prompt id {prompt_id!r}; have {[p.id for p in PROMPTS]}")
