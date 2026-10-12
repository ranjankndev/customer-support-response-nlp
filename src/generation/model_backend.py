"""
model_backend.py — one place to load a model and generate (plan §5.2)
=====================================================================
Two backends, same interface:

  unsloth : Kaggle T4. The T4 has no native bf16 and Gemma overflows in plain
            fp16, so Unsloth's patched fp16 path is the only safe one there.
  hf      : plain transformers on a bf16 GPU (e.g. a 4090). On a GPU without
            bf16 it falls back to 4-bit with float32 compute, never plain fp16.

Gemma-3 checkpoints hand back a *processor*, not a tokenizer, so everything that
pads, decodes or counts tokens goes through `text_tokenizer(tok)`.

The generation path fixes three bugs the old generator had:
  F3  no silent `truncation=True` at 2048 — the prompt budget is enforced by
      `utils.fit_prompt` before we get here, and anything still too long is
      logged with its index instead of being cut in the middle
  F4  `add_special_tokens=False`, because the chat template already starts with
      <bos>; tokenizing it again gave Gemma two of them
  F5  no hard-coded repetition_penalty=1.3 / no_repeat_ngram_size=4; both come
      from the config and default to off
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

log = logging.getLogger("baseline")

BACKENDS = ("hf", "unsloth")


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def text_tokenizer(tok):
    """The text tokenizer behind a Gemma-3 processor (or the tokenizer itself)."""
    return getattr(tok, "tokenizer", tok)


def make_chat_fn(tok) -> Callable[[List[str]], List[str]]:
    """
    Returns chat_fn(prompts) -> templated strings.

    Gemma-3 expects content as a list of typed parts; Gemma-2 expects a plain
    string. Try the typed form first and fall back, per prompt, so one function
    serves both checkpoints.
    """
    def chat_fn(prompts: List[str]) -> List[str]:
        out = []
        for prompt in prompts:
            templated = None
            for content in ([{"type": "text", "text": prompt}], prompt):
                try:
                    templated = tok.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False, add_generation_prompt=True)
                    break
                except Exception:
                    continue
            # A pre-trained model has no chat template: the raw prompt is right.
            out.append(templated if templated is not None else prompt)
        return out

    return chat_fn


def count_tokens_fn(tok) -> Callable[[str], int]:
    """count_tokens(text) -> length in tokens, for the prompt budget (§5.3)."""
    text_tok = text_tokenizer(tok)

    def count_tokens(text: str) -> int:
        return len(text_tok(text, add_special_tokens=False)["input_ids"])

    return count_tokens


# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_unsloth(model_id: str, use_4bit: bool, adapter_path: str,
                  max_seq_len: int, token: Optional[str]):
    from unsloth import FastModel

    # An adapter directory from `step3_raft train` carries its base model in
    # adapter_config.json, so Unsloth can load the pair in one call. PeftModel
    # stays as the fallback for an adapter it cannot resolve.
    source = adapter_path or model_id
    try:
        model, tok = FastModel.from_pretrained(
            source, max_seq_length=max_seq_len,
            load_in_4bit=bool(use_4bit), token=token)
    except Exception as exc:
        if not adapter_path:
            raise
        log.warning(f"[Backend] Unsloth could not load the adapter directly "
                    f"({exc}); falling back to base + PeftModel")
        from peft import PeftModel

        model, tok = FastModel.from_pretrained(
            model_id, max_seq_length=max_seq_len,
            load_in_4bit=bool(use_4bit), token=token)
        model = PeftModel.from_pretrained(model, adapter_path)

    FastModel.for_inference(model)
    return tok, model


def _load_hf(model_id: str, use_4bit: bool, adapter_path: str,
             attn_impl: str, torch_dtype: str, token: Optional[str]):
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                              AutoTokenizer, BitsAndBytesConfig)

    config = AutoConfig.from_pretrained(model_id, token=token)
    multimodal = getattr(config, "vision_config", None) is not None
    if multimodal:
        from transformers import Gemma3ForConditionalGeneration as ModelCls
    else:
        ModelCls = AutoModelForCausalLM

    kwargs = {"token": token, "device_map": "auto"}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    if use_4bit:
        # Gemma-2/3 overflow in fp16: use bf16 compute where the GPU has it and
        # float32 elsewhere. The vision tower and lm_head stay unquantised.
        compute_dtype = (torch.bfloat16 if torch.cuda.is_available()
                         and torch.cuda.is_bf16_supported() else torch.float32)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector",
                                   "lm_head"],
        )
    else:
        kwargs["torch_dtype"] = {"bfloat16": torch.bfloat16,
                                 "float16": torch.float16,
                                 "float32": torch.float32}.get(torch_dtype,
                                                               torch.bfloat16)

    model = ModelCls.from_pretrained(model_id, **kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    if multimodal:
        tok = AutoProcessor.from_pretrained(model_id, token=token)
    else:
        tok = AutoTokenizer.from_pretrained(model_id, token=token)
    return tok, model


def load_generator(model_id: str, backend: str = "hf", use_4bit: bool = True,
                   adapter_path: str = "", max_seq_len: int = 4096,
                   attn_impl: str = "sdpa", torch_dtype: str = "bfloat16",
                   token: Optional[str] = None
                   ) -> Tuple[object, object, Callable[[List[str]], List[str]]]:
    """
    Load a generator and return (tok, model, chat_fn).

    chat_fn(prompts) applies the model's chat template and returns the templated
    strings; `generate()` below consumes exactly those.
    """
    backend = (backend or "hf").lower()
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

    log.info(f"[Backend] {backend} | {model_id} | 4bit={use_4bit}"
             + (f" | adapter={adapter_path}" if adapter_path else ""))

    if backend == "unsloth":
        tok, model = _load_unsloth(model_id, use_4bit, adapter_path,
                                   max_seq_len, token)
    else:
        tok, model = _load_hf(model_id, use_4bit, adapter_path, attn_impl,
                              torch_dtype, token)

    text_tok = text_tokenizer(tok)
    if getattr(text_tok, "pad_token", None) is None:
        text_tok.pad_token = text_tok.eos_token

    model.eval()
    log.info("[Backend] Ready")
    return tok, model, make_chat_fn(tok)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def eos_token_ids(tok) -> List[int]:
    """EOS plus Gemma's <end_of_turn>, so a reply stops at the end of its turn."""
    text_tok = text_tokenizer(tok)
    ids = [text_tok.eos_token_id]
    try:
        end_of_turn = text_tok.convert_tokens_to_ids("<end_of_turn>")
    except Exception:
        end_of_turn = None
    if (isinstance(end_of_turn, int)
            and end_of_turn not in (None, text_tok.unk_token_id)
            and end_of_turn not in ids):
        ids.append(end_of_turn)
    return [i for i in ids if isinstance(i, int)]


def generate(prompts: List[str], tok, model, chat_fn=None,
             max_new_tokens: int = 200, do_sample: bool = False,
             repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
             max_input_tokens: int = 0, row_ids: Optional[List] = None
             ) -> List[str]:
    """
    Generate one reply per prompt. The prompts are expected to be within budget
    already (utils.fit_prompt); an over-long one is reported, never truncated.
    """
    import torch

    if not prompts:
        return []
    chat_fn = chat_fn or make_chat_fn(tok)
    templated = chat_fn(list(prompts))

    text_tok = text_tokenizer(tok)
    original_padding_side = text_tok.padding_side
    text_tok.padding_side = "left"
    try:
        # add_special_tokens=False: the chat template already carries <bos> (F4)
        enc = text_tok(templated, return_tensors="pt", padding=True,
                       add_special_tokens=False)
    finally:
        text_tok.padding_side = original_padding_side

    if max_input_tokens:
        limit = max_input_tokens + 64
        for position, ids in enumerate(enc["input_ids"]):
            length = int((ids != text_tok.pad_token_id).sum())
            if length > limit:
                row = (row_ids[position] if row_ids and position < len(row_ids)
                       else position)
                log.warning(f"[Backend] Prompt for row {row} is {length} tokens, "
                            f"over max_input_tokens+64 ({limit})")

    enc = enc.to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                  pad_token_id=text_tok.pad_token_id,
                  eos_token_id=eos_token_ids(tok),
                  repetition_penalty=repetition_penalty)
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    with torch.no_grad():
        out = model.generate(**enc, **kwargs)

    prompt_len = enc["input_ids"].shape[1]
    return [text_tok.decode(sequence[prompt_len:], skip_special_tokens=True).strip()
            for sequence in out]
