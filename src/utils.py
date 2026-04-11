"""
utils.py — Shared DataLoader, ContextBuilder, Generator, Metrics, Pipeline
"""

from __future__ import annotations

import re, json, logging, sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from baseline.baseline_schema import EvalConfig
    from baseline.baseline_loader import load_eval_config
    _BASELINE_AVAILABLE = True
except ImportError:
    _BASELINE_AVAILABLE = False
    EvalConfig = Any  # type: ignore[misc, assignment]

    def load_eval_config(path: str) -> Any:
        raise ImportError(
            "baseline package missing: add ./baseline/ with baseline_schema.py and "
            "baseline_loader.py, or avoid EvaluationPipeline / load_eval_config."
        )

try:
    from system_profiler import Profiler
except ImportError:

    class Profiler:
        def __enter__(self):
            self.result = self._Result()
            return self

        def __exit__(self, *args):
            return None

        class _Result:
            wall_time_sec = 0.0
            gpu_name = "n/a"
            peak_vram_gb = 0.0

            def summary(self) -> str:
                return "(system_profiler not installed)"

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
# removed: bert_score replaced with direct transformers implementation
from sklearn.metrics import accuracy_score, f1_score

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-8s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('baseline')


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    def __init__(self, config):
        self.cfg = config

    def load(self) -> Dict:
        log.info(f"[Data] Loading {self.cfg.tickets_csv}")
        all_cols = list(dict.fromkeys(
            self.cfg.input_cols + self.cfg.ground_truth_cols
        ))
        df = pd.read_csv(self.cfg.tickets_csv, usecols=all_cols, dtype=str).fillna("")

        if self.cfg.test_size:
            df = df.sample(n=min(self.cfg.test_size, len(df)),
                           random_state=self.cfg.random_seed).reset_index(drop=True)

        log.info(
            f"[Data] {len(df)} rows | "
            f"languages: {df[self.cfg.col_language].value_counts().to_dict()}"
        )
        return {
            "input_df": df[self.cfg.input_cols].copy(),
            "gt_df"   : df[self.cfg.ground_truth_cols].copy(),
            "lang_col": df[self.cfg.col_language].copy(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class ContextBuilder:
    def __init__(self, data_cfg, context_cfg):
        self.data_cfg    = data_cfg
        self.context_cfg = context_cfg

    def build(self, row: pd.Series) -> str:
        lines = []
        for col in self.data_cfg.input_cols:
            label     = self.context_cfg.col_labels.get(col, col.replace("_", " ").title())
            value     = str(row.get(col, "")).strip()
            max_chars = self.context_cfg.col_max_chars.get(col)
            if max_chars:
                value = value[:max_chars]
            lines.append(f"{label:<22}: {value}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATOR  (batched + optional 4-bit quantization)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineGenerator:
    _SEQ2SEQ = ("t5", "flan", "bart", "pegasus", "mt5")

    def __init__(self, model_cfg, data_cfg, context_cfg):
        self.model_cfg   = model_cfg
        self.ctx_builder = ContextBuilder(data_cfg, context_cfg)
        self.model       = None
        self.tokenizer   = None
        self._seq2seq    = False

    def load(self):
        log.info(f"[Generator] Loading  : {self.model_cfg.model_id}")
        log.info(f"[Generator] 4-bit    : {self.model_cfg.use_4bit}")
        log.info(f"[Generator] Batch    : {self.model_cfg.batch_size}")

        self._seq2seq = any(
            self.model_cfg.model_id.lower().startswith(p) for p in self._SEQ2SEQ
        )
        Cls = AutoModelForSeq2SeqLM if self._seq2seq else AutoModelForCausalLM

        # ── quantization config ───────────────────────────────────────────────
        if self.model_cfg.use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit               = True,
                bnb_4bit_compute_dtype     = torch.float16,
                bnb_4bit_use_double_quant  = True,   # nested quant saves ~0.4 GB extra
                bnb_4bit_quant_type        = "nf4",  # nf4 > fp4 for LLMs
            )
            self.model = Cls.from_pretrained(
                self.model_cfg.model_id,
                quantization_config = bnb_cfg,
                device_map          = "auto",
            )
        else:
            dtype_map = {"float16": torch.float16,
                         "bfloat16": torch.bfloat16,
                         "float32": torch.float32}
            dtype     = dtype_map.get(self.model_cfg.torch_dtype, torch.float16)
            self.model = Cls.from_pretrained(
                self.model_cfg.model_id,
                torch_dtype = dtype,
                device_map  = "auto" if self.model_cfg.device == "cuda" else None,
            )
            if self.model_cfg.device != "cuda":
                self.model = self.model.to(self.model_cfg.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_id)

        # pad token needed for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        log.info(f"[Generator] Ready | seq2seq={self._seq2seq}")

    def _build_prompts(self, rows: List[pd.Series]) -> List[str]:
        """
        Build prompts for each row.

        Instruct/causal models (Gemma, Llama etc.) — use apply_chat_template()
        so the model receives the exact format it was trained on:
          Gemma : <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model
          Llama : [INST] ... [/INST]

        Seq2seq models (T5, Flan-T5) — no chat template, plain text only.
        """
        prompts = []
        for row in rows:
            text = self.model_cfg.prompt_template.format(
                context=self.ctx_builder.build(row)
            )
            if self._seq2seq:
                # T5 / Flan-T5 — plain text input, no chat template
                prompts.append(text)
            else:
                # instruct models — wrap in chat template the model was trained on
                messages  = [{"role": "user", "content": text}]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize              = False,
                    add_generation_prompt = True,   # adds model-turn opener token
                )
                prompts.append(formatted)
        return prompts

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Run one batch of prompts through the model."""
        inputs = self.tokenizer(
            prompts,
            return_tensors   = "pt",
            truncation       = True,
            padding          = True,
            max_length       = 512,
        ).to(self.model_cfg.device)

        gen_kwargs = dict(
            max_new_tokens = self.model_cfg.max_new_tokens,
            do_sample      = self.model_cfg.do_sample,
        )
        if self.model_cfg.do_sample:
            gen_kwargs.update(
                temperature = self.model_cfg.temperature,
                top_p       = self.model_cfg.top_p,
            )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # causal models: strip the prompt from the output
        if not self._seq2seq:
            decoded = [
                d[len(p):].strip() for d, p in zip(decoded, prompts)
            ]
        return decoded

    @staticmethod
    def _parse(raw: str) -> Dict[str, str]:
        try:
            start = raw.find('{')
            end   = raw.rfind('}') + 1
            if start >= 0 and end > start:
                obj = json.loads(raw[start:end])
                return {
                    "response_pred": str(obj.get("response", "")),
                    "queue_pred"   : str(obj.get("queue",    "General Inquiry")),
                    "priority_pred": str(obj.get("priority", "medium")).lower(),
                }
        except Exception:
            pass
        resp  = re.search(r'"response"\s*:\s*"([^"]+)"', raw)
        queue = re.search(r'"queue"\s*:\s*"([^"]+)"',    raw)
        pri   = re.search(r'"priority"\s*:\s*"([^"]+)"', raw)
        return {
            "response_pred": resp.group(1)        if resp  else raw[:200],
            "queue_pred"   : queue.group(1)       if queue else "General Inquiry",
            "priority_pred": pri.group(1).lower() if pri   else "medium",
        }

    def generate_all(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            self.load()

        rows_list = [row for _, row in input_df.iterrows()]
        batch_sz  = self.model_cfg.batch_size
        results   = []
        total     = len(rows_list)

        for start in range(0, total, batch_sz):
            batch_rows    = rows_list[start: start + batch_sz]
            prompts       = self._build_prompts(batch_rows)
            raw_outputs   = self._generate_batch(prompts)
            results.extend(self._parse(r) for r in raw_outputs)

            done = min(start + batch_sz, total)
            log.info(f"[Generator] {done}/{total}")

        return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:
    def __init__(self, config):
        self.cfg = config

    def rouge(self, hyps, refs):
        scorer = rouge_scorer.RougeScorer(
            self.cfg.rouge_types, use_stemmer=self.cfg.rouge_use_stemmer)
        acc = {t: [] for t in self.cfg.rouge_types}
        for h, r in zip(hyps, refs):
            s = scorer.score(r, h)
            for t in self.cfg.rouge_types:
                acc[t].append(s[t].fmeasure)
        return {t: round(float(np.mean(v)), 4) for t, v in acc.items()}

    def bleu(self, hyps, refs):
        b = BLEU(tokenize=self.cfg.bleu_tokenizer)
        r = b.corpus_score(hyps, [refs])
        return {"bleu1": round(r.precisions[0] / 100, 4),
                "bleu4": round(r.precisions[3] / 100, 4)}

    def bertscore(self, hyps, refs):
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F

        log.info(f"[Metrics] BERTScore using {self.cfg.bertscore_model} ...")
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.bertscore_model)
        model     = AutoModel.from_pretrained(
                        self.cfg.bertscore_model
                    ).to(self.cfg.bertscore_device)
        model.eval()

        def encode(texts):
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=128, return_tensors="pt"
                           ).to(self.cfg.bertscore_device)
            with torch.no_grad():
                out = model(**enc)
            # use CLS token (index 0) — avoids mean pooling collapse
            return out.last_hidden_state[:, 0, :]

        batch = self.cfg.bertscore_batch
        f1s   = []
        for i in range(0, len(hyps), batch):
            h = encode(hyps[i:i+batch])
            r = encode(refs[i:i+batch])
            h = F.normalize(h, dim=-1)
            r = F.normalize(r, dim=-1)
            f1s.extend((h * r).sum(-1).cpu().tolist())

        del model
        torch.cuda.empty_cache()
        return {"bertscore_f1": round(float(sum(f1s)/len(f1s)), 4)}

    def classification(self, y_true, y_pred, label):
        return {
            f"{label}_accuracy"    : round(accuracy_score(y_true, y_pred), 4),
            f"{label}_f1_macro"    : round(f1_score(y_true, y_pred,
                                            average='macro',    zero_division=0), 4),
            f"{label}_f1_weighted" : round(f1_score(y_true, y_pred,
                                            average='weighted', zero_division=0), 4),
            f"{label}_correct"     : sum(t == p for t, p in zip(y_true, y_pred)),
            f"{label}_total"       : len(y_true),
        }

    def compute_all(self, pred_df, gt_df, lang_series) -> Dict:
        results = {}
        hyps = pred_df["response_pred"].fillna("").tolist()
        # use whatever the first ground truth column is (answer or response)
        ref_col = gt_df.columns[0]
        refs = gt_df[ref_col].fillna("").tolist()

        log.info("[Metrics] ROUGE / BLEU / BERTScore ...")
        results.update(self.rouge(hyps, refs))
        results.update(self.bleu(hyps, refs))
        results.update(self.bertscore(hyps, refs))

        for lang in lang_series.unique():
            mask  = (lang_series == lang).values
            h_sub = [h for h, m in zip(hyps, mask) if m]
            r_sub = [r for r, m in zip(refs, mask) if m]
            if len(h_sub) < 2:
                continue
            for k, v in {**self.rouge(h_sub, r_sub),
                         **self.bleu(h_sub, r_sub),
                         **self.bertscore(h_sub, r_sub)}.items():
                results[f"{k}_{lang}"] = v

        log.info("[Metrics] Queue + Priority ...")
        results.update(self.classification(
            gt_df["queue"].str.strip().tolist(),
            pred_df["queue_pred"].str.strip().tolist(), "queue"
        ))
        results.update(self.classification(
            gt_df["priority"].str.lower().str.strip().tolist(),
            pred_df["priority_pred"].str.lower().str.strip().tolist(), "priority"
        ))
        return results

    @staticmethod
    def print_table(label, metrics):
        print(f"\n{'═'*65}\n  {label}\n{'═'*65}")
        print("  RESPONSE QUALITY")
        for k in ["rouge1","rouge2","rougeL","bleu1","bleu4","bertscore_f1"]:
            if k in metrics:
                print(f"    {k:<22}: {metrics[k]}")
        print("  QUEUE CLASSIFICATION")
        for k in ["queue_accuracy","queue_f1_macro","queue_correct","queue_total"]:
            if k in metrics:
                print(f"    {k:<25}: {metrics[k]}")
        print("  PRIORITY CLASSIFICATION")
        for k in ["priority_accuracy","priority_f1_macro","priority_correct","priority_total"]:
            if k in metrics:
                print(f"    {k:<25}: {metrics[k]}")
        langs = sorted(set(k.split("_")[-1] for k in metrics if k.startswith("rouge1_")))
        for lang in langs:
            print(f"  LANGUAGE: {lang.upper()}")
            for base in ["rouge1","rouge2","rougeL","bleu1","bleu4","bertscore_f1"]:
                key = f"{base}_{lang}"
                if key in metrics:
                    print(f"    {key:<28}: {metrics[key]}")
        print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationPipeline:
    def __init__(self, config: EvalConfig):
        if not _BASELINE_AVAILABLE:
            raise ImportError(
                "EvaluationPipeline requires the baseline package "
                "(baseline_schema, baseline_loader)."
            )
        self.cfg        = config
        self.loader     = DataLoader(config.data)
        self.generator  = BaselineGenerator(config.model, config.data, config.context)
        self.calculator = MetricsCalculator(config.metric)
        Path(config.output.results_dir).mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        log.info(f"[Pipeline] {self.cfg.system_label}")

        with Profiler() as prof:
            data    = self.loader.load()
            pred_df = self.generator.generate_all(data["input_df"])

            # save predictions immediately after generation
            # so they are not lost if metrics fail
            if self.cfg.output.save_hypotheses:
                safe = self.cfg.system_label.replace(" ","_").replace("/","-")
                out  = Path(self.cfg.output.results_dir) / safe
                out.mkdir(parents=True, exist_ok=True)
                pd.concat([data["input_df"], data["gt_df"], pred_df],
                          axis=1).to_csv(out / "predictions.csv", index=False)
                log.info(f"[Pipeline] Predictions saved → {out}/predictions.csv")

            metrics = self.calculator.compute_all(
                pred_df, data["gt_df"], data["lang_col"]
            )

        metrics["wall_time_sec"] = prof.result.wall_time_sec
        metrics["gpu"]           = prof.result.gpu_name
        metrics["peak_vram_gb"]  = prof.result.peak_vram_gb

        if self.cfg.output.save_hypotheses:
            safe = self.cfg.system_label.replace(" ","_").replace("/","-")
            out  = Path(self.cfg.output.results_dir) / safe
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"system": self.cfg.system_label,
                           **metrics}]).to_csv(out / "metrics.csv", index=False)
            log.info(f"[Pipeline] Saved → {out}/")

        if self.cfg.output.verbose:
            self.calculator.print_table(self.cfg.system_label, metrics)
            print(prof.result.summary())

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. HF TOKEN
# ─────────────────────────────────────────────────────────────────────────────

def get_hf_token() -> str:
    """
    Returns HuggingFace token from environment or .env file.
    Used by any component that loads gated models (Gemma, Llama).
    """
    import os
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return token


# ─────────────────────────────────────────────────────────────────────────────
# 7. FILE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_or_excel(path: str, **kwargs) -> pd.DataFrame:
    """
    Load CSV or Excel file. Used by all phases that read ticket data.
    """
    path = str(path)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, dtype=str, low_memory=False, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 8. CHECKPOINTER
# ─────────────────────────────────────────────────────────────────────────────

class Checkpointer:
    """
    Save/resume progress for long generation runs.
    Survives Kaggle/Vast session interruptions.

    Usage:
        ckpt = Checkpointer("run_rag")
        done = ckpt.load()          # list of already-completed dicts
        ckpt.save_if_due(rows, n)   # save every N rows
        ckpt.clear()                # remove on clean finish
    """

    def __init__(self, label: str, every: int = 50):
        self.path  = Path(f".ckpt_{label}.csv")
        self.every = every

    def load(self) -> List[dict]:
        if self.path.exists():
            df = pd.read_csv(self.path, dtype=str).fillna("")
            log.info(f"[Checkpoint] Resumed {len(df)} rows from {self.path}")
            return df.to_dict("records")
        return []

    def save(self, rows: List[dict]):
        pd.DataFrame(rows).to_csv(self.path, index=False)

    def save_if_due(self, rows: List[dict], n_done: int):
        if (n_done + 1) % self.every == 0:
            self.save(rows)
            log.info(f"[Checkpoint] Saved {len(rows)} rows → {self.path}")

    def clear(self):
        if self.path.exists():
            self.path.unlink()
            log.info(f"[Checkpoint] Cleared {self.path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(results: Dict[str, pd.DataFrame],
                   id_col: str = "idx",
                   ref_col: str = "answer") -> pd.DataFrame:
    """
    Build a side-by-side comparison CSV from multiple model result DataFrames.
    results: {"baseline": df_baseline, "rag": df_rag, ...}
    """
    first_df  = list(results.values())[0]
    base_cols = [c for c in [id_col, "language", "subject", ref_col]
                 if c in first_df.columns]
    comp = first_df[base_cols].copy()

    for mode, df in results.items():
        resp_col = f"{mode}_response"
        len_col  = f"{mode}_length"
        cols     = [id_col] + [c for c in [resp_col, len_col] if c in df.columns]
        comp     = comp.merge(df[cols], on=id_col, how="left")

    return comp
