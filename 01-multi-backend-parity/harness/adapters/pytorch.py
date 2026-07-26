"""The PyTorch reference adapter.

Every other backend is measured against this one, so its job is not to be
fast but to be explainable: plain eager attention, no cache, no sampling,
fp32 on the way out.

The one non-obvious thing here is `text_config_of`. `google/gemma-4-E2B-it`
is a multimodal repository -- its top-level `model_type` is `gemma4`, which
`AutoModelForCausalLM` resolves to `Gemma4ForConditionalGeneration`, vision
and audio encoders included. The text tower is described by the nested
`text_config` (`model_type: gemma4_text`), which resolves to
`Gemma4ForCausalLM`. Handing that sub-config to the Auto class is what
selects the text-only tower. Checkpoints without a `text_config` -- Llama
3.2, the control model -- pass through untouched, so the adapter stays
model-agnostic.

Verified against transformers 5.14.1:

    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["gemma4"]      = Gemma4ForConditionalGeneration
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["gemma4_text"] = Gemma4ForCausalLM

Picking the class is only half of it. The checkpoint stores the text tower
under `model.language_model.*`, while `Gemma4ForCausalLM` expects
`model.*`, and transformers 5.14.1 ships no conversion mapping for
`gemma4_text` -- `conversion_mapping.py` has entries for `gemma4_unified`
and for `qwen3_5_text` (`PrefixChange(prefix_to_remove="language_model")`)
but none for this one. Without `key_mapping` the load "succeeds" with every
language-model tensor reported UNEXPECTED and all 24 of the model's own
parameter groups randomly initialised. It runs. It produces logits. They
are noise. Hence `_assert_fully_loaded`: a reference that silently
random-initialises is worse than one that fails.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .base import BackendAdapter

__all__ = ["PyTorchAdapter", "text_config_of"]

# The two precisions this backend serves. Quantised tags in `PRECISIONS`
# belong to other backends; asking for one here is a mistake worth failing on.
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


# Checkpoint keys for the text tower of a multimodal repository. The name
# follows the library's own composition -- `Gemma4Model.language_model` --
# rather than anything specific to Gemma.
_TEXT_KEY_MAPPING = {r"^model\.language_model\.": "model."}


def text_config_of(config: Any) -> Any:
    """The language-model config of a possibly multimodal checkpoint."""
    return getattr(config, "text_config", None) or config


class PyTorchAdapter(BackendAdapter):
    """Reference logits from transformers.

    Not registered in `adapters/__init__.py` on purpose: importing it pulls in
    torch, and the edit machine has no CUDA build. Import it by module path
    from code that runs on the GPU machine.
    """

    name = "pytorch"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = "cuda"
        self._requested: dict[str, Any] = {}
        self._vram: dict[str, int] = {}
        self._loading_info: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------

    def _load(
        self,
        model_path: str,
        precision: str,
        *,
        device: str = "cuda",
        attn_implementation: str = "eager",
        **kwargs: Any,
    ) -> None:
        """Load the text tower.

        `attn_implementation` defaults to eager rather than to whatever the
        library would pick. A reference should compute attention the way the
        formula reads; sdpa and flash kernels reorder the reduction, and this
        project exists to measure exactly that kind of difference in the
        backends being compared -- not to smuggle it into the baseline.
        """
        if precision not in _DTYPES:
            raise ValueError(
                f"{type(self).__name__} serves {sorted(_DTYPES)}, not {precision!r}"
            )

        self._device = device
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        free_before, total = _cuda_mem(device)

        full_config = AutoConfig.from_pretrained(model_path)
        config = text_config_of(full_config)
        is_multimodal = config is not full_config

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            dtype=_DTYPES[precision],
            device_map=device,
            attn_implementation=attn_implementation,
            key_mapping=dict(_TEXT_KEY_MAPPING) if is_multimodal else None,
            output_loading_info=True,
            **kwargs,
        )
        self._loading_info = {k: len(v) for k, v in loading_info.items()}
        _assert_fully_loaded(model_path, loading_info)
        self._model.eval()

        free_after, _ = _cuda_mem(device)
        self._requested = {
            "precision": precision,
            "dtype": str(_DTYPES[precision]),
            "attn_implementation": attn_implementation,
            "model_path": model_path,
            "device": device,
        }
        self._vram = {
            # Allocator view: tensors only. Excludes the CUDA context, so it
            # understates what the card actually has to give up.
            "torch_allocated_bytes": torch.cuda.memory_allocated(device) if device.startswith("cuda") else 0,
            "torch_reserved_bytes": torch.cuda.memory_reserved(device) if device.startswith("cuda") else 0,
            # Driver view: everything this process holds, context included.
            # This is the number that decides whether the model fits.
            "device_used_bytes": free_before - free_after,
            "device_total_bytes": total,
            "device_free_after_load_bytes": free_after,
        }

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- measurement -------------------------------------------------------

    @torch.no_grad()
    def logits(self, token_ids: Sequence[int]) -> np.ndarray:
        self._require_loaded()
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self._model.device)
        # use_cache=False: nothing here is incremental, and a cache is one
        # more place for a backend to differ from itself between calls.
        out = self._model(input_ids=ids, use_cache=False)
        return out.logits[0, -1].float().cpu().numpy()

    @torch.no_grad()
    def generate(self, token_ids: Sequence[int], max_tokens: int) -> str:
        self._require_loaded()
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self._model.device)
        out = self._model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        return self._tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def meta(self) -> dict[str, Any]:
        self._require_loaded()
        config = self._model.config
        return {
            "backend": self.name,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(self._device) if self._device.startswith("cuda") else "cpu",
            # What was asked for, and what the library actually selected. D5:
            # keep both verbatim. Requesting eager does not guarantee eager.
            "requested": dict(self._requested),
            "model_class": type(self._model).__name__,
            "config_class": type(config).__name__,
            "model_type": config.model_type,
            "effective_attn_implementation": getattr(config, "_attn_implementation", None),
            "effective_dtype": str(self._model.dtype),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "param_count": sum(p.numel() for p in self._model.parameters()),
            # Key counts, not the keys themselves: a large `unexpected_keys`
            # is the encoders being left behind, which is what we asked for.
            "loading_info_counts": dict(self._loading_info),
            "vram": dict(self._vram),
        }

    # -- internals ---------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """The reference tokeniser. Callers tokenise once with this and feed
        the same integers to every backend (D1)."""
        self._require_loaded()
        return self._tokenizer

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("no model loaded; call load() inside a with-block")


def _assert_fully_loaded(model_path: str, loading_info: dict[str, list[str]]) -> None:
    """Refuse a model whose weights did not all come from the checkpoint.

    Unexpected keys are fine and expected here -- the vision and audio towers
    of a multimodal repository have no home in a text-only model, and leaving
    them behind is the point. Missing keys are not fine: transformers fills
    them with fresh random values and carries on.
    """
    missing = loading_info.get("missing_keys") or []
    if missing:
        raise RuntimeError(
            f"{len(missing)} parameters of {model_path} were randomly initialised "
            f"rather than loaded, e.g. {missing[:5]}. Reference logits from such a "
            "model are noise. Check that the checkpoint's key prefix matches the "
            "model class, and extend the adapter's key mapping if it does not."
        )


def _cuda_mem(device: str) -> tuple[int, int]:
    """(free, total) device memory in bytes, as the driver reports it."""
    if not device.startswith("cuda"):
        return (0, 0)
    return torch.cuda.mem_get_info(device)
