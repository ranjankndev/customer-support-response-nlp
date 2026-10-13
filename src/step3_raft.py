"""
step3_raft.py — Aspect-RAFT: build the dataset, then fine-tune (plan §7)
========================================================================
RAFT = retrieval-augmented fine-tuning. The model is trained on the prompt it
will actually see at inference — same builder, same aspect checklist, same
language line, same prompt budget — so it learns to use retrieved context
instead of being surprised by it.

Two deliberate kinds of noise:
  distractor_prob   a share of examples get random context only, so the model
                    learns to answer from the ticket when retrieval fails
  use_aspects_prob  a share are built without the aspect checklist, so one
                    adapter serves both S4 (with aspects) and S5 (without)

Leakage rules asserted here (plan §7.4): only the train split is used, a
training ticket never retrieves itself, and near-duplicate hits are dropped.

USAGE
    python -m src.step3_raft build --config config/config_raft.yaml \
        --overrides config/kaggle.yaml [--limit 20]
    python -m src.step3_raft train --config config/config_raft.yaml \
        --overrides config/kaggle.yaml [--limit 64 --max_steps 5]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SEED_DEFAULT = 42
JSONL_FIELDS = ("idx", "language", "prompt", "answer", "distractor",
                "with_aspects")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RaftConfig:
    base_model       : str = "unsloth/gemma-3-4b-it"
    retrieval_config : str = "config/inference_retrieval_config.yaml"
    generation_config: str = "config/generation_config.yaml"
    train_csv        : str = "data/minimal/raft_train_3000_with_aspects.csv"
    raft_jsonl       : str = "output/raft/raft_train.jsonl"
    top_k            : int = 3
    distractor_prob  : float = 0.2
    use_aspects_prob : float = 0.8
    near_dup_cos     : float = 0.97
    max_seq_len      : int = 2048
    seed             : int = SEED_DEFAULT
    lora             : Dict = field(default_factory=lambda: {"r": 16,
                                                             "alpha": 16,
                                                             "dropout": 0.0})
    lr               : float = 2.0e-4
    epochs           : int = 1
    batch            : int = 2
    grad_accum       : int = 8
    val_frac         : float = 0.05
    save_steps       : int = 50
    eval_steps       : int = 50
    logging_steps    : int = 10
    warmup_ratio     : float = 0.03
    out_dir          : str = "output/raft/gemma3-4b-aspect-raft"
    hf_repo          : str = ""
    # samples that must never appear in training (plan §7.4)
    exclude_csvs     : List[str] = field(default_factory=list)


def load_raft_config(path: str,
                     overrides_path: Optional[str] = None) -> RaftConfig:
    """Load config_raft.yaml, merging the `raft` section of an overrides file."""
    from src.utils import load_yaml_with_overrides

    raw = load_yaml_with_overrides(path, overrides_path, section="raft")
    known = RaftConfig.__dataclass_fields__
    config = RaftConfig(**{k: v for k, v in raw.items() if k in known})

    if overrides_path:
        import yaml

        data = yaml.safe_load(Path(overrides_path).read_text(
            encoding="utf-8")) or {}
        section = data.get("raft") or {}
        # config/kaggle.yaml speaks in n_train; the build takes that many rows
        if "n_train" in section:
            config.n_train = int(section["n_train"])     # type: ignore[attr-defined]
        paths = data.get("paths") or {}
        work_dir = str(paths.get("work_dir", "") or "").strip()
        if work_dir:
            from pathlib import PurePosixPath

            base = PurePosixPath(work_dir)
            config.raft_jsonl = str(base / "output" / "raft" / "raft_train.jsonl")
            config.out_dir = str(base / "output" / "raft"
                                 / "gemma3-4b-aspect-raft")
    return config


# ─────────────────────────────────────────────────────────────────────────────
# BUILD  (plan §7.2)
# ─────────────────────────────────────────────────────────────────────────────

def metadata_lookup(metadata: List[Dict]) -> Dict[Tuple[str, str, str], str]:
    """
    (subject, body[:200], language) -> ticket_id, built once.

    A training ticket is in the train index, so it would retrieve itself as the
    perfect "past ticket" and the model would learn to copy the answer out of
    the context. The row's own id has to be excluded, and the CSV carries no
    index position, so it is matched on its text.
    """
    lookup = {}
    for record in metadata:
        key = (str(record.get("subject", "")).strip(),
               str(record.get("body", ""))[:200].strip(),
               str(record.get("language", "")).strip())
        lookup.setdefault(key, str(record.get("ticket_id", "")))
    return lookup


def own_ticket_ids(lookup: Dict[Tuple[str, str, str], str], subject: str,
                   body: str, language: str) -> set:
    key = (str(subject or "").strip(), str(body or "")[:200].strip(),
           str(language or "").strip())
    found = lookup.get(key)
    return {found} if found else set()


def drop_self_and_near_duplicates(hits: List[Dict], answer: str,
                                  near_dup_cos: float, top_k: int
                                  ) -> List[Dict]:
    """
    Remove hits that are the training row in disguise: a dense score at or above
    near_dup_cos, or a response identical to the gold answer. Without this the
    target text sits in the prompt and the loss rewards copying.
    """
    kept = []
    gold = (answer or "").strip()
    for hit in hits:
        dense = hit.get("dense")
        if isinstance(dense, (int, float)) and float(dense) >= near_dup_cos:
            continue
        if gold and str(hit.get("response", "") or "").strip() == gold:
            continue
        kept.append(hit)
        if len(kept) >= top_k:
            break
    return kept


def random_distractors(metadata: List[Dict], rng: random.Random, top_k: int,
                       avoid_intent: str = "") -> List[Dict]:
    """
    `top_k` random documents, preferring a different intent so the context is
    plausibly formatted but genuinely unhelpful.
    """
    if not metadata:
        return []
    wanted = str(avoid_intent or "").strip().lower()

    def different(record: Dict) -> bool:
        if not wanted:
            return True
        aspects = record.get("aspects") or {}
        for key in ("intent", "queue"):
            value = str(aspects.get(key, "") or "").strip().lower()
            if value:
                return value != wanted
        return True

    pool = [r for r in metadata if different(r)] or list(metadata)
    chosen = rng.sample(pool, k=min(top_k, len(pool)))
    return [{"subject": r.get("subject", ""), "body": r.get("body", ""),
             "response": r.get("response", ""),
             "language": r.get("language", ""),
             "aspects": r.get("aspects", {}), "dense": 0.0,
             "ticket_id": str(r.get("ticket_id", ""))} for r in chosen]


def assert_no_evaluation_rows(df: pd.DataFrame, exclude_csvs: List[str]) -> None:
    """plan §7.4: the test and validate samples are never trained on."""
    if "idx" not in df.columns:
        return
    train_ids = set(df["idx"].astype(str))
    for csv_path in exclude_csvs or []:
        path = Path(csv_path)
        if not path.is_file():
            print(f"[RAFT] WARNING: exclude list not found: {path}")
            continue
        other = pd.read_csv(path, dtype=str).fillna("")
        column = "idx" if "idx" in other.columns else None
        if column is None:
            continue
        overlap = train_ids & set(other[column].astype(str))
        if overlap:
            raise AssertionError(
                f"{len(overlap)} rows of {path} are in the RAFT training set "
                f"(e.g. {sorted(overlap)[:5]}): training must use the train "
                f"split only."
            )
        print(f"[RAFT] No overlap with {path}")


def build_examples(df: pd.DataFrame, retriever, count_tokens,
                   config: RaftConfig, aspect_columns: List[str]
                   ) -> List[Dict]:
    """One JSONL record per row with a non-empty gold answer."""
    from src.utils import fit_prompt

    rng = random.Random(config.seed)
    lookup = metadata_lookup(retriever.metadata)
    budget = max(config.max_seq_len - 256, 256)
    examples = []
    skipped = 0

    for position, row in df.iterrows():
        answer = str(row.get("answer", "") or "").strip()
        if not answer:
            skipped += 1
            continue

        subject = str(row.get("subject", "") or "")
        body = str(row.get("body", "") or "")
        language = str(row.get("language", "en") or "en")
        spans = {column: str(row.get(column, "") or "")
                 for column in aspect_columns if column in df.columns}
        spans = {k: v for k, v in spans.items()
                 if v and v.lower() not in ("none", "nan")}

        with_aspects = rng.random() < config.use_aspects_prob
        distractor = rng.random() < config.distractor_prob
        prompt_spans = spans if (with_aspects and spans) else None

        if distractor:
            hits = random_distractors(retriever.metadata, rng, config.top_k,
                                      spans.get("intent", ""))
        else:
            raw_hits = retriever.retrieve(
                subject=subject, body=body, language=language,
                aspect_spans=prompt_spans, top_k=config.top_k + 3,
                exclude_ids=own_ticket_ids(lookup, subject, body, language))
            hits = drop_self_and_near_duplicates(raw_hits, answer,
                                                 config.near_dup_cos,
                                                 config.top_k)

        prompt, _ = fit_prompt(
            lambda docs: retriever.build_prompt(subject, body, docs,
                                                prompt_spans, language),
            hits, count_tokens, budget)

        examples.append({"idx": str(row.get("idx", position)),
                         "language": language.lower()[:2],
                         "prompt": prompt,
                         "answer": answer,
                         "distractor": bool(distractor),
                         "with_aspects": bool(prompt_spans)})

    if skipped:
        print(f"[RAFT] Skipped {skipped} rows without a gold answer")
    return examples


def example_stats(examples: List[Dict], count_tokens=None) -> Dict:
    """Counts, language split and prompt-length percentiles."""
    import numpy as np

    total = len(examples)
    stats = {
        "n": total,
        "en": sum(1 for e in examples if e["language"] == "en"),
        "de": sum(1 for e in examples if e["language"] == "de"),
        "distractor_share": round(
            sum(1 for e in examples if e["distractor"]) / total, 4) if total else 0.0,
        "with_aspects_share": round(
            sum(1 for e in examples if e["with_aspects"]) / total, 4) if total else 0.0,
    }
    if total and count_tokens is not None:
        lengths = [count_tokens(e["prompt"]) for e in examples]
        stats["prompt_tokens_p50"] = int(np.percentile(lengths, 50))
        stats["prompt_tokens_p95"] = int(np.percentile(lengths, 95))
    return stats


def write_jsonl(examples: List[Dict], path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            stream.write(json.dumps({k: example[k] for k in JSONL_FIELDS},
                                    ensure_ascii=False) + "\n")
    print(f"[RAFT] {len(examples)} examples -> {out}")
    return str(out)


def read_jsonl(path: str) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cmd_build(config: RaftConfig, limit: Optional[int] = None) -> str:
    from src.generation.model_backend import count_tokens_fn
    from src.minimal_run import ASPECT_COLUMNS
    from src.retrieval.inference_retriever import load_retriever_config
    from src.retrieval.retriever import TicketRetriever
    from transformers import AutoTokenizer

    df = pd.read_csv(config.train_csv, dtype=str, low_memory=False).fillna("")
    assert_no_evaluation_rows(df, config.exclude_csvs)
    n_train = int(getattr(config, "n_train", 0) or 0)
    if limit:
        df = df.head(limit)
    elif n_train and n_train < len(df):
        df = df.head(n_train)
    print(f"[RAFT] Building from {len(df)} rows of {config.train_csv}")

    retriever = TicketRetriever(load_retriever_config(config.retrieval_config))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    count_tokens = count_tokens_fn(tokenizer)

    examples = build_examples(df, retriever, count_tokens, config,
                              ASPECT_COLUMNS)
    stats = example_stats(examples, count_tokens)
    print(f"[RAFT] {stats}")
    return write_jsonl(examples, config.raft_jsonl)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN  (plan §7.3)
# ─────────────────────────────────────────────────────────────────────────────

def strip_leading_bos(text: str) -> str:
    """The trainer adds <bos> itself; a second one shifts every position."""
    return text[len("<bos>"):] if text.startswith("<bos>") else text


def chat_text(tokenizer, prompt: str, answer: str) -> str:
    """One training string: the inference prompt plus the gold reply."""
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        text = tokenizer.apply_chat_template(
            [{"role": m["role"], "content": [{"type": "text",
                                              "text": m["content"]}]}
             for m in messages], tokenize=False)
    return strip_leading_bos(text)


def split_examples(examples: List[Dict], val_frac: float, seed: int
                   ) -> Tuple[List[Dict], List[Dict]]:
    """Shuffled train/val split; val_frac 0 means no eval set."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * max(val_frac, 0.0)))
    if val_frac > 0 and len(shuffled) > 1:
        n_val = max(1, min(n_val, len(shuffled) - 1))
    return shuffled[n_val:], shuffled[:n_val]


def latest_checkpoint(out_dir: str) -> Optional[str]:
    """Newest checkpoint-N, so a timed-out Kaggle session can resume (plan §11)."""
    directory = Path(out_dir)
    if not directory.is_dir():
        return None
    checkpoints = [p for p in directory.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    newest = max(checkpoints,
                 key=lambda p: int(p.name.split("-")[-1])
                 if p.name.split("-")[-1].isdigit() else -1)
    return str(newest)


def sft_config_kwargs(config: RaftConfig, has_eval: bool,
                      max_steps: Optional[int] = None) -> Dict:
    """
    SFTConfig arguments. TRL renamed a few of these between versions, so
    `train` filters them against the installed signature rather than pinning a
    version (plan §7.3).
    """
    kwargs = {
        "dataset_text_field": "text",
        "per_device_train_batch_size": config.batch,
        "gradient_accumulation_steps": config.grad_accum,
        "num_train_epochs": config.epochs,
        "learning_rate": config.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": 2,
        "output_dir": config.out_dir,
        "report_to": "none",
        "seed": config.seed,
        "max_seq_length": config.max_seq_len,
    }
    if has_eval:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = config.eval_steps
    if max_steps:
        kwargs["max_steps"] = int(max_steps)
    return kwargs


def filter_kwargs(target, kwargs: Dict) -> Dict:
    """
    Keep the arguments `target` actually accepts, renaming the ones TRL moved
    (max_seq_length -> max_length). An unknown argument is dropped with a note,
    which is what lets this run against whatever TRL Kaggle has installed.
    """
    import inspect

    try:
        accepted = set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return kwargs
    renames = {"max_seq_length": "max_length"}
    out = {}
    for key, value in kwargs.items():
        if key in accepted:
            out[key] = value
            continue
        alias = renames.get(key)
        if alias and alias in accepted:
            out[alias] = value
            continue
        print(f"[RAFT] {getattr(target, '__name__', target)} does not accept "
              f"{key}; dropped")
    return out


def cmd_train(config: RaftConfig, limit: Optional[int] = None,
              max_steps: Optional[int] = None, push: bool = True) -> str:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    from src.utils import get_hf_token

    examples = read_jsonl(config.raft_jsonl)
    if limit:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"No examples in {config.raft_jsonl}; run `build` first")
    print(f"[RAFT] Training on {len(examples)} examples")

    token = get_hf_token() or None
    model, tokenizer = FastModel.from_pretrained(
        config.base_model, max_seq_length=config.max_seq_len,
        load_in_4bit=True, token=token)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # text only: the tickets have no images
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=int(config.lora.get("r", 16)),
        lora_alpha=int(config.lora.get("alpha", 16)),
        lora_dropout=float(config.lora.get("dropout", 0.0)),
        bias="none", random_state=config.seed)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    trainable = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    vision = [name for name in trainable if "vision" in name.lower()]
    print(f"[RAFT] Trainable tensors: {len(trainable)}")
    if vision:
        raise AssertionError(f"{len(vision)} vision parameters are trainable "
                             f"(e.g. {vision[:3]}); the language model only.")

    train_examples, val_examples = split_examples(examples, config.val_frac,
                                                  config.seed)
    def to_dataset(records):
        return Dataset.from_list([
            {"text": chat_text(tokenizer, r["prompt"], r["answer"])}
            for r in records])

    train_dataset = to_dataset(train_examples)
    val_dataset = to_dataset(val_examples) if val_examples else None
    print(f"[RAFT] train={len(train_dataset)} "
          f"val={len(val_dataset) if val_dataset else 0}")

    args = SFTConfig(**filter_kwargs(
        SFTConfig, sft_config_kwargs(config, bool(val_dataset), max_steps)))
    trainer_kwargs = filter_kwargs(SFTTrainer, {
        "model": model, "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "train_dataset": train_dataset, "eval_dataset": val_dataset,
        "args": args})
    # tokenizer= and processing_class= are the same argument in different TRL
    # versions; keep whichever this one accepts.
    if "tokenizer" in trainer_kwargs and "processing_class" in trainer_kwargs:
        trainer_kwargs.pop("processing_class")
    trainer = SFTTrainer(**trainer_kwargs)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n")

    resume = latest_checkpoint(config.out_dir)
    if resume:
        print(f"[RAFT] Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    Path(config.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.out_dir)
    tokenizer.save_pretrained(config.out_dir)
    print(f"[RAFT] Adapter saved -> {config.out_dir}")

    # Sanity check: one EN, one DE and one distractor prompt (plan §7.3)
    sanity_check(model, tokenizer, examples, config)

    if push and config.hf_repo and token:
        print(f"[RAFT] Pushing to {config.hf_repo}")
        model.push_to_hub(config.hf_repo, token=token)
        tokenizer.push_to_hub(config.hf_repo, token=token)
    elif push and config.hf_repo:
        print("[RAFT] No HF token in this environment — push the adapter "
              "locally after `kaggle_job.py pull`")
    return config.out_dir


def pick_sanity_examples(examples: List[Dict]) -> List[Dict]:
    """One EN, one DE and one distractor example, as far as they exist."""
    picked = []
    for predicate in (lambda e: e["language"] == "en" and not e["distractor"],
                      lambda e: e["language"] == "de" and not e["distractor"],
                      lambda e: e["distractor"]):
        match = next((e for e in examples if predicate(e)), None)
        if match is not None and match not in picked:
            picked.append(match)
    return picked


def sanity_check(model, tokenizer, examples: List[Dict],
                 config: RaftConfig) -> List[str]:
    from src.generation.model_backend import generate

    picked = pick_sanity_examples(examples)
    if not picked:
        return []
    replies = generate([e["prompt"] for e in picked], tokenizer, model,
                       max_new_tokens=120)
    for example, reply in zip(picked, replies):
        print(f"\n[RAFT] --- {example['language']} "
              f"distractor={example['distractor']} "
              f"with_aspects={example['with_aspects']} ---")
        print(f"[RAFT] gold : {example['answer'][:200]}")
        print(f"[RAFT] model: {reply[:200]}")
    return replies


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="config/config_raft.yaml")
        sub.add_argument("--overrides", default=None,
                         help="e.g. config/kaggle.yaml")
        sub.add_argument("--limit", type=int, default=None,
                         help="use only the first N rows/examples")
        if name == "train":
            sub.add_argument("--max_steps", type=int, default=None,
                             help="stop after N optimiser steps (smoke test)")
            sub.add_argument("--no_push", action="store_true",
                             help="do not push the adapter to HF")
    args = parser.parse_args(argv)

    try:
        config = load_raft_config(args.config, args.overrides)
        if args.command == "build":
            cmd_build(config, args.limit)
        else:
            cmd_train(config, args.limit, args.max_steps,
                      push=not args.no_push)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
