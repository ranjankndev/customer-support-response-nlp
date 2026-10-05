# AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)

> **Audience:** the implementing model (e.g. Sonnet). Read this whole file before editing anything.
> **Root:** repo `C:\devspectforge` (GitHub `ranjankndev/customer-support-response-nlp`). All paths are relative to it.
> **Execution model:** Claude Code builds, tests and runs everything (Kaggle via API); see §K and `CLAUDE.md`.
> **Goal:** upgrade the system (Aspect v2, multilingual Gemma-3, hybrid retrieval, Aspect-RAFT, CRAG router) and produce a
> **final results table with confidence intervals** from a **minimal run** that fits on **Kaggle T4×2 in about 6–8 GPU hours**.
> **Scope rule:** only what is in this file. No full-test-set runs, no extra baselines, no 12B model, no agents.
> Keep the folder structure; new files are listed in §2.

---

## 0. Why: problems found in the current code

| # | Problem | Where | Fix (section) |
|---|---|---|---|
| F1 | Generator `gemma-2-2b-it` is English-centric | `config/generation_config.yaml` | Gemma-3-4B-it (§5) |
| F2 | **No prompt states the reply language**, so DE tickets often get English replies | all prompt builders | language line (§5.4) |
| F3 | **Right-side truncation at 2048 tokens** cuts off the ticket and `Support reply:` | `generate_batch` | prompt budget (§5.3) |
| F4 | **Double `<bos>`** (chat template + tokenizer both add it) | `generate_batch` | `add_special_tokens=False` (§5.2) |
| F5 | Hard-coded `repetition_penalty=1.3`, `no_repeat_ngram_size=4` on an IT model | `generate_batch` | config, IT defaults off (§5.2) |
| F6 | **ROUGE drops ä/ö/ü/ß** and uses the English stemmer, so DE ROUGE is wrong | `src/utils.py` | Unicode tokenizer (§8) |
| F7 | `bertscore_f1` is actually CLS-cosine; `bleu1/4` are n-gram precisions | `src/utils.py` | honest names + real metrics (§8) |
| F8 | VADER sentiment and English-only severity keywords applied to DE | generator | §5.5 |
| F9 | Dense-only E5 retrieval; DE gets no reranking | `retriever.py` | hybrid + multilingual rerank (§6) |
| F10 | Generator never trained on the RAG task | — | Aspect-RAFT (§7) |
| F11 | **Aspects are weak** (the project's namesake): `intent` collapses to `general_inquiry`, `prob_sub`==`cause`, spans are pleasantries, `entities` = title-case subject words, `ticket_type_nli` has 2 values, DE mojibake | `aspect_extractor.py` outputs | Aspect v2 (§A) |
| F12 | Aspects barely reach retrieval: EN drops the aspect score after the cross-encoder, DE ignores aspect query | `retriever.py` | §6.3 + §A |
| F13 | **No metric measures what the name promises** (aspect-complete replies) | — | Aspect Coverage metrics (§A.4) |

---

## 1. What the minimal run produces

**Fixed evaluation sample:** 500 test tickets (250 EN + 250 DE), stratified, seed 42. **Every system uses the same 500 rows.**

**6 systems** (all use hybrid retrieval + bge-reranker + the bug fixes):

| ID | System | Generator | Aspects | Adapter | Router |
|---|---|---|---|---|---|
| S1 | Old generator, fixed pipeline | gemma-2-2b-it | ✓ | – | – |
| S2 | Multilingual generator | gemma-3-4b-it | ✓ | – | – |
| S3 | S2 + CRAG router | gemma-3-4b-it | ✓ | – | ✓ |
| S4 | **Aspect-RAFT (main)** | gemma-3-4b-it + LoRA | ✓ | ✓ | – |
| S5 | RAFT without aspects | gemma-3-4b-it + LoRA | – | ✓ | – |
| S6 | Gemma-3 RAG without aspects | gemma-3-4b-it | – | – | – |

Comparisons: S1→S2 = effect of the multilingual model; S2→S3 = router; S2→S4 = RAFT; S6→S2 = aspects (zero-shot); S5→S4 = aspects under RAFT.

**Plus a retrieval-only table (no generation, cheap):** E5-dense vs BGE-M3-dense vs BGE-M3-hybrid vs hybrid+rerank on 500 validate tickets.

**Metrics** (overall / EN / DE): ROUGE-1/2/L (Unicode), sacreBLEU, BERTScore-F1 (xlm-roberta-large), `lang_match`, mean length, **ACS / AAS aspect coverage (§A.4)**.
**Aspect quality table:** accuracy / macro-F1 of the aspect heads vs gold `queue/type/priority` on val (§A.1).
**Statistics:** paired bootstrap (1000 resamples) 95% CI for each metric, and the CI of the **difference** vs S2 for S3/S4/S5 and S1.

**Deliverables:** `output/report/results.md`, `results_table.csv`, `retrieval_table.csv`, `aspect_table.csv`, `results_chart.png`, and an updated README.

---

## 2. Files: new and changed

**New**
| File | Purpose |
|---|---|
| `src/minimal_run.py` | `sample` (build the fixed 500/500 samples), `crag_tune`, `report` (metrics + bootstrap + chart) |
| `src/step3_raft.py` | `build` (RAFT JSONL) and `train` (Unsloth QLoRA) |
| `src/generation/model_backend.py` | one place to load models + generate: `unsloth` (Kaggle T4) and `hf` (bf16 GPUs) |
| `src/retrieval/eval_retrieval.py` | retrieval-only proxy metrics |
| `config/config_raft.yaml` | RAFT settings |
| `config/kaggle.yaml` | path + batch overrides for Kaggle (merged on top of the other configs) |
| `notebooks/kaggle_01_prep_train.ipynb` | Session 1 |
| `notebooks/kaggle_02_generate_eval.ipynb` | Session 2 |
| `requirements-kaggle.txt` | Kaggle installs |

**Changed:** `src/generation/response_generator_v2.py`, `src/retrieval/retriever.py`, `src/retrieval/indexer.py`,
`src/retrieval/inference_retriever.py`, `src/utils.py`, `config/generation_config.yaml`,
`config/retrieval_config.yaml`, `config/inference_retrieval_config.yaml`, `README.md`.

**Not touched:** `aspect_extractor.py` (except the config path override), `step1_silver_labels.py`, `step2_sft.py`, `baseline_t5.py`, `preprocessing.py`.

---

## 3. Kaggle constraints the code must respect

| Constraint | Consequence for code |
|---|---|
| Use **GPU T4 ×2** (not P100: bitsandbytes/Unsloth don't support it well) | 16 GB per GPU; run two jobs in parallel with `CUDA_VISIBLE_DEVICES=0/1` |
| **T4 has no native bf16**, and Gemma-2/3 overflow in plain fp16 | Kaggle backend = **Unsloth** (handles Gemma fp16 on T4). Never load Gemma in plain fp16 with the `hf` backend; on non-bf16 GPUs the `hf` backend must use 4-bit with `bnb_4bit_compute_dtype=torch.float32` |
| 12 h session limit, about 30 GPU h/week | two sessions (§9); every long step checkpoints and is resumable |
| `/kaggle/input` is read-only; `/kaggle/working` (20 GB) is saved only when the session ends via "Save Version" | all outputs/checkpoints go to `/kaggle/working/Code/...`; the adapter is also pushed to HF |
| Internet + HF token | notebook needs Internet ON; token via Kaggle **Secrets** (`HF_TOKEN`), read by the existing `get_hf_token()`. Never hard-code it |
| Gemma licences | accept the Gemma-2 and Gemma-3 terms on HF with the account that owns the token |

**Inputs to upload as a private Kaggle dataset `aspectforge-data`:**
- `data/processed/cs-support-multi-{train,validate,test}.csv`
- `output/aspect/aspects_train.csv`, **if you have it** (from the earlier project). If it's missing, the pipeline still works: see §6.2 and §7.1.

**Code on Kaggle:** push the changed repo to GitHub and `git clone` it in the notebook (preferred), or upload the `Code` folder as a second Kaggle dataset and copy it to `/kaggle/working/Code`.

---

## 4. `config/kaggle.yaml` (overrides) and config merging

Add a tiny helper in `src/utils.py`:
```python
def load_yaml_with_overrides(path, overrides_path=None, section=None):
    """Load YAML; if overrides_path is given, deep-merge overrides[section] on top."""
```
Every CLI entry point used in the minimal run (`response_generator_v2`, `indexer`, `aspect_extractor`, `step3_raft`, `minimal_run`, `eval_retrieval`) gets an optional `--overrides config/kaggle.yaml`.

```yaml
# config/kaggle.yaml
paths:
  data_dir   : /kaggle/input/aspectforge-data
  work_dir   : /kaggle/working/Code
generation:
  backend              : unsloth
  generation_batch_size: 8
  max_input_tokens     : 2560
  max_new_tokens       : 200
  output_dir           : /kaggle/working/Code/output/generation
  skip_metrics         : true          # metrics computed once by minimal_run report
retrieval:
  device     : cuda
  batch_size : 32
raft:
  n_train    : 3000
  max_seq_len: 2048
  batch      : 2
  grad_accum : 8
```
(Paths in the base configs stay relative so local runs still work.)

---

## 5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)

### 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)
```yaml
model_id            : "google/gemma-3-4b-it"
backend             : "hf"            # hf | unsloth   (kaggle.yaml sets unsloth)
use_4bit            : true
attn_implementation : "sdpa"          # hf backend only; "eager" for gemma-2
adapter_path        : ""              # RAFT adapter: local dir or HF repo id
repetition_penalty  : 1.0
no_repeat_ngram_size: 0
max_input_tokens    : 3072
ctx_doc_max_chars   : 1200
reply_language_instruction: true
crag_threshold      : 0.30            # set by minimal_run crag_tune
sample_csv          : ""              # fixed 500-row sample; overrides --input when set
skip_metrics        : false
```
CLI additions: `--model_id`, `--adapter_path`, `--overrides`, `--crag_threshold`, `--sample_csv`. Each one overrides the YAML value, so one config file can serve S1–S5.

### 5.2 `src/generation/model_backend.py`
```python
def load_generator(model_id, backend, use_4bit, adapter_path="", max_seq_len=4096,
                   attn_impl="sdpa", token=None):
    """Returns (tok, model, chat_fn).  chat_fn(prompts:list[str]) -> list[str] of templated strings."""

def generate(prompts, tok, model, chat_fn, max_new_tokens, do_sample=False,
             repetition_penalty=1.0, no_repeat_ngram_size=0) -> list[str]:
```

**`unsloth` backend (Kaggle):**
```python
from unsloth import FastModel
model, tok = FastModel.from_pretrained(model_id, max_seq_length=max_seq_len,
                                       load_in_4bit=True, token=token)
if adapter_path:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
FastModel.for_inference(model)
```
- Gemma-3 checkpoints return a **processor**, not a plain tokenizer. Use `text_tok = getattr(tok, "tokenizer", tok)` for padding, decoding and token counting. For the chat template, call `tok.apply_chat_template([{"role":"user","content":[{"type":"text","text":p}]}], tokenize=False, add_generation_prompt=True)`; if that raises, fall back to `content=p` (a string). **Verify with a 2-prompt smoke test cell** (§9) before any long job.
- Gemma-2 (`unsloth/gemma-2-2b-it` or `google/gemma-2-2b-it`) loads through the same `FastModel` call and returns a plain tokenizer. The `getattr` handles both cases.
- Loading with an adapter: if `adapter_path` is a directory saved by `step3_raft train` (§7.3), it's simplest to load it directly with `FastModel.from_pretrained(adapter_path, ...)`. Unsloth resolves the base model from `adapter_config.json`. Use that path; keep `PeftModel` as a fallback.

**`hf` backend (local bf16 GPU, e.g. 4090):** `AutoConfig` → if `vision_config` is present, use `Gemma3ForConditionalGeneration`, else `AutoModelForCausalLM`. Use 4-bit NF4 with `llm_int8_skip_modules=["vision_tower","multi_modal_projector","lm_head"]` and compute dtype `bfloat16` if `torch.cuda.is_bf16_supported()` else `float32`. Load the adapter via `PeftModel`. Needs `transformers>=4.50`.

**`generate()` (shared by both backends):**
```python
text_tok.padding_side = "left"
enc = text_tok(templated, return_tensors="pt", padding=True,
               add_special_tokens=False).to(model.device)          # F4: no 2nd <bos>
eos = [text_tok.eos_token_id]
eot = text_tok.convert_tokens_to_ids("<end_of_turn>")
if isinstance(eot, int) and eot not in (None, text_tok.unk_token_id): eos.append(eot)
kw = dict(max_new_tokens=..., do_sample=..., pad_token_id=text_tok.pad_token_id,
          eos_token_id=eos, repetition_penalty=repetition_penalty)
if no_repeat_ngram_size > 0: kw["no_repeat_ngram_size"] = no_repeat_ngram_size
out = model.generate(**enc, **kw)
# decode only new tokens: out[:, enc["input_ids"].shape[1]:]
```
No `truncation=` (F3). If any input is longer than `max_input_tokens + 64`, log a warning with its idx.

In `response_generator_v2.py`: `load_model()` and `generate_batch()` become thin wrappers around these two functions. Remove the hard-coded penalties (F5). Leave `load_extra_model`/`generate_batch_extra` (T5/gemma_pt) unchanged; they are not used in the minimal run.

### 5.3 Prompt budget (F3)
Put `fit_prompt` in `src/utils.py` (it's also used by RAFT build, so the training and inference prompts are identical):
```python
def fit_prompt(build_fn, retrieved, count_tokens, max_tokens):
    docs = list(retrieved)
    while True:
        prompt = build_fn(docs)
        if count_tokens(prompt) + 16 <= max_tokens or not docs:
            return prompt, len(docs)
        docs = docs[:-1]          # drop the lowest-ranked doc
```
Also cap each retrieved response at `ctx_doc_max_chars` inside `build_prompt` (add `ctx_doc_max_chars: int = 0` to `RetrievalConfig`; `_make_retriever` sets it from the generation config with `dataclasses.replace`). Store `n_ctx_used` in the output rows.

### 5.4 Language line (F2)
In `src/retrieval/retriever.py`:
```python
LANG_NAMES = {"en": "English", "de": "German"}
def reply_language_line(language):
    name = LANG_NAMES.get((language or "en").lower().strip()[:2], "the customer's language")
    return f"Write the reply in {name}, the same language as the customer's message."
```
Add a `language="en"` parameter to `TicketRetriever.build_prompt` and `build_aspect_only_prompt`, and put the line **directly before `Support reply:`** (only when `reply_language_instruction` is on). Pass `lang` from the runners. (`build_baseline_prompt` and `build_guided_rag_prompt` can get the same parameter, but they aren't used in the minimal run.)

### 5.5 Sentiment/severity for DE (F8)
In `extract_sentiment_severity`: if the language is not `en`, set sentiment to `"n/a"` and compound to 0, and omit the Sentiment line from prompts. Copy the German keywords from `src/aspect/aspect_extractor.py` (`_CRITICAL/_HIGH/_MEDIUM`, around lines 653–668) into the generator's severity lists. (S1–S5 use `run_rag_aspect`, which doesn't inject sentiment, so this mainly keeps the code correct.)

### 5.6 Runners and modes for S1–S5
- `run_rag_aspect(df, ..., label="rag_aspect", use_aspects=True, crag=False)`:
  - `label` sets the checkpoint name and the `{label}_prompt/_response/_length` columns.
  - `use_aspects=False` → pass `aspect_spans=None` to both `retrieve` and `build_prompt` (S5).
  - Always store `top_ce_score` (= `retrieved[0]["ce_score"]` or empty) and `n_ctx_used`.
  - `crag=True` → router (§7.5).
- `--mode` choices: add `raft_aspect` (S4) and `raft_rag` (S5). S1 and S2 are `rag_aspect` with different `--model_id`; S3 is `rag_aspect_crag`. Running a `raft_*` mode requires a non-empty `adapter_path` (raise `ValueError` otherwise). Non-raft modes must **ignore** `adapter_path`.
- **Use `--label` (new CLI flag) to name outputs**, e.g. `--label s1_gemma2`. The output file is `{output_dir}/{label}.csv`. The minimal-run report finds systems by these names.
- Make sure `Checkpointer` writes into `output_dir` (today it writes `.ckpt_*.csv` into the current directory). Add a `dir` parameter.
- Aspects: the minimal run passes `--sample_csv` files that **already contain aspect columns** (§6.1), so the generator must not trigger `enrich_dataframe_with_aspects`. It already skips when `_aspect_columns_ready(df)` is true; confirm that path works and doesn't write `*_with_aspects*` files.

---

## 6. Data, aspects and retrieval

### 6.1 `src/minimal_run.py sample`
```
python -m src.minimal_run sample --overrides config/kaggle.yaml \
   --test_n 500 --val_n 500 --raft_n 3000 --seed 42
```
- From the test CSV: stratified by `language`, 250 EN + 250 DE → `data/minimal/test_500.csv`. Keep `idx`.
- From the validate CSV: 500 stratified rows → `data/minimal/val_500.csv` (used for retrieval eval and CRAG tuning; 150 of them are used for CRAG).
- From the train CSV: 3000 stratified rows → `data/minimal/raft_train_3000.csv`.
- If a sample file already exists, reuse it (never resample silently; print its hash).

**Aspects for the samples** (Qwen2.5-1.5B LoRA, existing extractor; fine on T4 in fp16):
```
python -m src.aspect.aspect_extractor --input data/minimal/test_500.csv  --out data/minimal/test_500_aspects.csv  --adapter ranjan56cse/cs-support-labels --adapter_subfolder sft_checkpoint
python -m src.aspect.aspect_extractor --input data/minimal/val_500.csv   --out data/minimal/val_500_aspects.csv   ...
python -m src.aspect.aspect_extractor --input data/minimal/raft_train_3000.csv --out data/minimal/raft_train_3000_aspects.csv ...
```
Output must keep `idx`, `language`, `subject`, `body`, `answer` plus the 8 aspect columns. If the extractor drops `body`/`answer`, merge them back on `idx` inside `minimal_run sample --attach_aspects` (add that small subcommand).

### 6.2 Index (`indexer.py`)
Config fields to add (to `IndexerConfig`, `RetrievalConfig`, and both retrieval YAMLs; defaults = old behaviour):
```yaml
model_name            : "BAAI/bge-m3"
query_prefix          : ""          # E5: "query: "
passage_prefix        : ""          # E5: "passage: "
retrieval_mode        : "hybrid"    # dense | hybrid
bm25_path             : "data/index/bm25"
rrf_k                 : 60
n_candidates          : 50
use_cross_encoder     : true
cross_encoder_model   : "BAAI/bge-reranker-v2-m3"
rerank_languages      : ["en", "de"]
rerank_doc_field      : "body"      # body | response
aspect_query_languages: ["en"]
aspect_rerank_weight  : 0.1
metadata_body_max_chars: 1000
```
Changes:
- Use `passage_prefix` from the config (remove the hard-coded `"passage: "`); set `model.max_seq_length = max_length`.
- Store `body[:metadata_body_max_chars]` in metadata.
- **Aspects optional:** if `aspects_csv` is missing or empty, index the train CSV directly with `aspects = {}` (log a warning). The aspect bonus is then simply 0 for those hits.
- **Exclude the evaluation rows**: the index is built from train only (already true). Assert that no `idx` from `test_500` or `val_500` is in the index.
- Hybrid: `bm25s.tokenize(texts, stopwords=None)` → `bm25s.BM25().index(...)` → `.save(bm25_path)`, with the same row order as FAISS.
- Index type stays `faiss.IndexFlatIP`. Use **`faiss-cpu`** on Kaggle (fast enough for this corpus).
- For the retrieval table, also build an **E5 dense** index into `data/index_e5/` (same script, different config values passed via `--overrides`). If time is short, skip it and drop the E5 row.

### 6.3 Retriever (`retriever.py`)
- `STEmbedder.encode`: use `query_prefix`/`passage_prefix` from the config.
- `get_cross_encoder(name, max_length=512)` as a singleton keyed by name; use `CrossEncoder(name, max_length=512)`. On T4, pass `model_kwargs={"torch_dtype": torch.float16}` if supported, else leave the default.
- Load BM25 in `__init__` if hybrid.
- `rrf_fuse(rank_lists, k)` helper.
- `retrieve(..., exclude_ids: set = None)`:
  1. Dense query (aspect-weighted only if `lang in aspect_query_languages` and spans exist; otherwise subject+body).
  2. FAISS top `n_candidates`; if hybrid, BM25 top `n_candidates`.
  3. Drop `exclude_ids`; apply `filter_by_language` if set.
  4. RRF → candidates (keep `dense`, add `rrf`).
  5. If the reranker is on and `lang in rerank_languages`: CE score on (query `subject\nbody`, doc `subject\nbody` or `subject response`), `batch_size=64`. Then `final = ce_score + alpha * aspect_score(spans, hit_aspects)` (aspect term only when spans exist).
  6. Else `final = dense + alpha * aspect_score`.
  7. Return the top `top_k` with `ce_score` kept.

### 6.4 `src/retrieval/eval_retrieval.py`
For each row of `val_500_aspects.csv`: retrieve top-3 and record
- `top1_rougeL`: Unicode ROUGE-L between the top-1 retrieved **response** and the gold `answer`;
- `top1_cos`: cosine (BGE-M3 embeddings) between the same pair;
- `intent@3`: share of top-3 hits whose stored `aspects.intent` equals the query's `intent` (skip rows without intent or when the index has no aspects).

Configs to compare: `e5_dense`, `m3_dense`, `m3_hybrid`, `m3_hybrid_rerank` (switched via `--overrides` / CLI flags). Output: `output/report/retrieval_table.csv` (mean + bootstrap CI, EN/DE split).

---

## 7. Aspect-RAFT (`src/step3_raft.py`, `config/config_raft.yaml`)

### 7.1 Config
```yaml
base_model        : "unsloth/gemma-3-4b-it"     # same weights as google/gemma-3-4b-it, Unsloth-packaged
retrieval_config  : "config/inference_retrieval_config.yaml"
generation_config : "config/generation_config.yaml"   # ctx_doc_max_chars, max_input_tokens, language line
train_csv         : "data/minimal/raft_train_3000_aspects.csv"
raft_jsonl        : "output/raft/raft_train.jsonl"
top_k             : 3
distractor_prob   : 0.2       # RAFT: share of examples with random-only context
use_aspects_prob  : 0.8       # 20% trained without aspect slots → one adapter serves S4 and S5
near_dup_cos      : 0.97
max_seq_len       : 2048
lora: {r: 16, alpha: 16, dropout: 0.0}
lr                : 2.0e-4
epochs            : 1
batch             : 2
grad_accum        : 8
val_frac          : 0.05
out_dir           : "output/raft/gemma3-4b-aspect-raft"
hf_repo           : ""        # e.g. ranjan56cse/aspectforge-raft-gemma3 (strongly recommended on Kaggle)
```

### 7.2 `build`
```
python -m src.step3_raft build --config config/config_raft.yaml --overrides config/kaggle.yaml
```
For each row (gold `answer` must be non-empty):
1. Find the row's own id in the index: match on (`subject`, `body[:200]`, `language`) against metadata, built once as a dict. Put it in `exclude_ids` (the training ticket is in the train index).
2. `hits = retriever.retrieve(subject, body, lang, spans, top_k=top_k+3, exclude_ids=...)`. Then drop hits with `dense >= near_dup_cos` or `response.strip() == answer.strip()`, and keep `top_k`.
3. With `distractor_prob`: replace the hits with `top_k` random metadata records (different `intent` when available). Use `random.Random(seed)`.
4. With `1 - use_aspects_prob`: `spans = None`.
5. `prompt, n = fit_prompt(lambda d: retriever.build_prompt(subject, body, d, spans, language=lang), hits, count_tokens, max_seq_len - 256)`. **This is the same builder and settings as inference.**
6. Write `{"idx","language","prompt","answer","distractor","with_aspects"}` to JSONL.

Print the stats: count, EN/DE split, distractor share, with-aspects share, prompt token p50/p95.

### 7.3 `train` (Unsloth QLoRA; follows Unsloth's Gemma-3 fine-tuning recipe)
```python
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
model, tok = FastModel.from_pretrained(cfg.base_model, max_seq_length=cfg.max_seq_len,
                                       load_in_4bit=True, token=hf_token)
model = FastModel.get_peft_model(model,
        finetune_vision_layers=False, finetune_language_layers=True,   # language model only
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=42)
tok = get_chat_template(tok, chat_template="gemma-3")
# dataset: text = tok.apply_chat_template([{"role":"user","content":prompt},
#                                          {"role":"assistant","content":answer}], tokenize=False)
#          strip a leading "<bos>" from text (the trainer adds it).
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=train_ds, eval_dataset=val_ds,
        args=SFTConfig(dataset_text_field="text", per_device_train_batch_size=2,
                       gradient_accumulation_steps=8, num_train_epochs=1, learning_rate=2e-4,
                       lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
                       eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50,
                       save_total_limit=2, output_dir=cfg.out_dir, report_to="none", seed=42,
                       max_seq_length=cfg.max_seq_len))
trainer = train_on_responses_only(trainer,
        instruction_part="<start_of_turn>user\n", response_part="<start_of_turn>model\n")
trainer.train(resume_from_checkpoint=<latest checkpoint in out_dir or None>)
model.save_pretrained(cfg.out_dir); tok.save_pretrained(cfg.out_dir)
if cfg.hf_repo: model.push_to_hub(cfg.hf_repo, token=hf_token); tok.push_to_hub(cfg.hf_repo, token=hf_token)
```
Notes:
- Unsloth and TRL argument names change between versions. **Check against the installed versions**, e.g. `SFTConfig(max_seq_length=...)` vs `max_length`, and `tokenizer=` vs `processing_class=`. Fix any mismatch rather than downgrading blindly.
- Print the trainable parameters; confirm that no `vision` parameters are trainable.
- Sanity check after training: generate for 3 JSONL prompts (1 EN, 1 DE, 1 distractor) and print them.
- Expected time on a T4: 3000 examples × about 1.3k tokens ≈ 4M tokens → **about 60–90 min**.
- Gemma-2 (S1) is not fine-tuned.

### 7.4 Leakage rules (must hold, add asserts)
- The test and validate samples are never used for training.
- `crag_threshold` is tuned on validate only.
- A training ticket is excluded from its own retrieved context, along with near-duplicates.

### 7.5 CRAG router (S3)
In `run_rag_aspect(..., crag=True)`:
```python
top = retrieved[0].get("ce_score") if retrieved else None
if top is None or top < config.crag_threshold:
    prompt = retriever.build_prompt(subject, body, [], spans, language=lang)   # same template, empty context
    route = "no_context"
else:
    prompt, n_used = fit_prompt(...); route = "rag"
```
Store `route`, `top_ce_score`. Log the share of each route.

**Cheap tuning with no threshold sweep re-runs (`minimal_run crag_tune`):** on 150 rows of `val_500_aspects.csv`, generate **once with context and once without** (300 generations with S2's model). For each τ in {0.05, 0.10, …, 0.60}, choose per row the "rag" output if `top_ce_score >= τ`, else the "no_context" output. Compute ROUGE-L for each τ and pick the best. Write `crag_threshold` to `output/report/crag_tuning.csv`. S3 on test then needs only the `no_context` generations for rows below τ; the rest are copied from S2's outputs by `idx`. Implement this reuse (it saves about half of S3's cost); fall back to full generation if S2's file is missing.

---

## 8. Metrics and report (`src/utils.py`, `src/minimal_run.py report`)

### 8.1 `MetricsCalculator` fixes
```python
class _UnicodeTokenizer:
    _re = re.compile(r"\w+", re.UNICODE)
    def tokenize(self, text): return self._re.findall((text or "").lower())
```
- `rouge`: `RougeScorer(types, use_stemmer=False, tokenizer=_UnicodeTokenizer())`.
- `bleu`: add `"bleu": round(r.score, 2)` (real sacreBLEU); keep the precisions as `bleu_p1`/`bleu_p4`.
- Rename the CLS-cosine output `bertscore_f1` → `cls_cosine_mbert`.
- New `bertscore_true(hyps, refs, model_type="xlm-roberta-large", batch_size=16)` via `bert_score.score` → `bertscore_f1`. Return **per-row** F1 too (needed for the bootstrap).
- New `lang_match(hyps, langs)` with `langdetect` (`DetectorFactory.seed = 0`); per-row 0/1.
- **Every metric needs a per-row version** (ROUGE and BERTScore are naturally per row; BLEU is corpus-level, so bootstrap it by recomputing corpus BLEU on each resample).

### 8.2 `python -m src.minimal_run report --systems s1_gemma2 s2_gemma3 s3_gemma3_crag s4_raft_aspect s5_raft_noaspect`
1. Load `{output_dir}/{label}.csv` for each system. Assert all have the **same 500 idx**; align by idx.
2. Per-row metrics for each system → `output/report/per_row_metrics.csv`.
3. Means overall/EN/DE + bootstrap 95% CI (1000 resamples of row indices, seed 42; the **same resample indices for all systems** = paired).
4. Paired differences vs S2 (and S1 vs S2) with CI; mark **significant** if the CI excludes 0.
5. Write `results_table.csv` and `results.md` (a markdown table formatted as `mean [low, high]`, plus one-line takeaways generated from the numbers, plus the retrieval table, CRAG route share, and RAFT train/eval loss from the trainer log).
6. `results_chart.png`: grouped bars (ROUGE-L, BERTScore-F1, lang_match) per system with CI error bars, split by EN/DE. Use matplotlib.
7. Add 3 qualitative examples (1 EN, 2 DE): ticket, gold, S2 output, S4 output.

---

## 9. Kaggle notebooks

Both notebooks start with the same setup cells:
```python
# 1. GPU check
!nvidia-smi
# 2. code
!git clone https://github.com/ranjankndev/customer-support-response-nlp.git /kaggle/working/Code   # or copy from /kaggle/input
%cd /kaggle/working/Code
# 3. installs (Unsloth first; let it pick compatible transformers/trl/peft)
!pip install -q unsloth
!pip install -q -r requirements-kaggle.txt
# 4. secrets
from kaggle_secrets import UserSecretsClient
import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
# 5. data link
!mkdir -p data/processed output/aspect && cp /kaggle/input/aspectforge-data/*.csv data/processed/
```
`requirements-kaggle.txt`: `sentence-transformers`, `faiss-cpu`, `bm25s`, `rouge-score`, `sacrebleu`, `bert-score`, `langdetect`, `pyyaml`, `pandas`, `matplotlib`, `vaderSentiment`. Do **not** list `torch`, `transformers`, `trl` or `peft` here (Unsloth manages them).

**Smoke cell (run before anything long, in both notebooks):** load `unsloth/gemma-3-4b-it`, generate for one EN and one DE prompt with the language line. Check that the output is non-empty, the DE reply is German, and it stops at `<end_of_turn>`. Repeat with `unsloth/gemma-2-2b-it`.

### Notebook 1: `kaggle_01_prep_train.ipynb` (about 3–3.5 h)
| Step | Command | Est. T4 time |
|---|---|---|
| 1 | `minimal_run sample` | 1 min |
| 2 | aspect extraction for test_500, val_500, raft_train_3000 (3 runs; **GPU 0 and GPU 1 in parallel**) | 25–40 min |
| 3 | build BGE-M3 hybrid index (+ optional E5 index on the other GPU in parallel) | 20–35 min |
| 4 | `eval_retrieval` (4 configs, 500 val) | 10–15 min |
| 5 | `step3_raft build` | 15–20 min |
| 6 | `step3_raft train` (pushes the adapter to HF) | 60–90 min |
| 7 | **Save Version** (keeps `/kaggle/working` as output) | — |

### Notebook 2: `kaggle_02_generate_eval.ipynb` (about 2.5–3.5 h)
Attach Notebook 1's output as input; copy `data/minimal`, `data/index*`, `output/` into the working dir. Set `adapter_path` to the HF repo id (or the copied dir).

| Step | GPU 0 | GPU 1 |
|---|---|---|
| A | S1: `--mode rag_aspect --model_id unsloth/gemma-2-2b-it --label s1_gemma2` | S4: `--mode raft_aspect --adapter_path <repo> --label s4_raft_aspect` |
| B | S2: `--mode rag_aspect --model_id unsloth/gemma-3-4b-it --label s2_gemma3` | S5: `--mode raft_rag --adapter_path <repo> --label s5_raft_noaspect` |
| C | `minimal_run crag_tune` (val 150 × 2) | — |
| D | S3: `--mode rag_aspect_crag --crag_threshold <τ> --label s3_gemma3_crag` (reuses S2) | — |
| E | `minimal_run report` | — |

Launch pattern for parallel jobs (both write logs to `logs/`):
```python
!CUDA_VISIBLE_DEVICES=0 nohup python -m src.generation.response_generator_v2 --config config/generation_config.yaml --overrides config/kaggle.yaml --sample_csv data/minimal/test_500_aspects.csv ... > logs/s1.log 2>&1 &
```
Then a polling cell (`!tail -n 3 logs/*.log`). Every runner checkpoints every 50 rows, so a restart resumes.

Estimated generation cost: 500 rows × 4 systems + about 150 (S3 fallback rows) + 300 (tuning) ≈ 2,450 generations at batch 8 → **about 1.5–2.5 h wall-clock across both GPUs**.

**Total ≈ 6–7 GPU-session hours out of the ~30 h weekly quota.**

---

## A. Aspect v2: make the "Aspect" in AspectForge measurable and strong

### A.0 Evidence (from `docs/samples/aspects_SAMPLE.csv`, 11 test tickets)
| Aspect | Observed problem |
|---|---|
| `intent` | 6/11 = `general_inquiry` (label collapse); the dataset already has a gold `queue` (~10 classes) that is a much better intent signal |
| `ticket_type_nli` | only 2 values (`complaint`, `product_support`); the dataset has gold `type` = Incident / Request / Problem / Change |
| `urgency` | rule-based, disagrees with gold `priority` on most rows (e.g. "critical" vs priority "low") |
| `prob_sub` / `cause` | often **identical** strings; sometimes a pleasantry ("Your timely assistance is greatly appreciated…") |
| `action_taken` | often equals `cause` or a pleasantry; rarely an actual action the customer tried |
| `entities` | mostly title-case words of the subject ("Detected, Unauthorized, Access, …") |
| DE text | mojibake (`�ber`) reaches the aspects, so German aspects are degraded |

Today aspects also barely influence retrieval (EN drops the aspect score after the cross-encoder; DE ignores the aspect query), and **no metric checks whether a reply covers the aspects**, which is the project's whole claim ("complete" responses). §A fixes all three with CPU-only work.

### A.1 Supervised aspect heads (replace the SFT labels at inference; CPU, minutes)
New file `src/aspect/aspect_heads.py` with `train` / `predict` / `eval` subcommands.
- Targets from gold columns of the **train split**: `intent ← queue`, `ticket_type ← type`, `urgency ← priority`. (Check the processed CSVs keep `type`, `queue`, `priority`; `src/preprocessing.py` only adds `idx`, so they should.)
- Features: BGE-M3 embeddings of `subject + "\n\n" + body`. The indexer already embeds every train ticket, so it must also save the matrix to `data/index/dense.npy` (row order = metadata order). Queries are embedded with the same model and prefix.
- Model: `sklearn.linear_model.LogisticRegression(max_iter=2000, class_weight="balanced")`, one per head, saved to `data/index/aspect_heads.joblib`.
- `eval` on `val_500`: accuracy + macro-F1 per head, overall / EN / DE, vs a majority-class baseline. Also report the old SFT label distribution (share of `general_inquiry`, number of distinct types) to document the collapse → `output/report/aspect_table.csv`.
- Predictions are model outputs, not gold, so there's no leakage. Gold labels are used only for training and evaluation.
- `aspect_extractor.py`: add `--labels heads|sft` (default `heads`). With `heads`, the Qwen SFT model is **not loaded**: aspect extraction becomes CPU-only, which saves GPU time on Kaggle. `sft` keeps the legacy path.

### A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU)
1. `ftfy.fix_text` on subject/body **before** extraction (and on outputs). Fixes DE mojibake.
2. Pleasantry filter (EN+DE regex: thank, appreciate, look forward, regards, best wishes, danke, vielen dank, freundliche grüße, ich freue mich, mit freundlichen): a matching span is replaced by the next-ranked candidate sentence, or `none`.
3. De-duplication: if Jaccard(`cause`, `prob_sub`) > 0.8 → `cause = none`; same for `action_taken` vs `prob_sub`/`cause`.
4. `action_taken` must contain an action cue (EN: tried, attempted, restarted, rebooted, reinstalled, cleared, updated, reset, checked, contacted, followed; DE: versucht, neu gestartet, neu installiert, aktualisiert, zurückgesetzt, überprüft, geprüft, kontaktiert), otherwise `none`.
5. `extract_entities`: keep the domain patterns (step 1); keep single capitalised words only if they occur in the **body** at least twice, or contain a digit/camelCase. This drops subject title-case noise.

Unit tests in `tests/test_aspects.py` using the sample rows: a pleasantry span is replaced, a duplicate cause becomes `none`, and `Klage Ã¼ber` becomes `Klage über`.

### A.3 Aspects in retrieval and prompt
- Index metadata: store **gold** `queue/type/priority` for train docs (past resolved tickets legitimately carry this metadata). Queries use **predicted** heads.
- `aspect_score`: intent(queue) match 0.4, type match 0.2, `prob_sub` token overlap 0.3 (Unicode tokenizer), cause overlap 0.1. Applied **after** reranking for EN and DE (§6.3).
- **Aspect checklist prompt** replaces the current `Aspects : k: v | …` dump, in `build_prompt` (so RAFT learns it) and in the aspect-only path:
  ```
  Your reply must:
  1. Solve the problem: {prob_sub}
  2. Address the likely cause: {cause}                       (omit line if none)
  3. Acknowledge what the customer already tried: {action_taken}; do not suggest it again   (omit if none)
  4. Tone for urgency "{urgency}": high/critical -> "prioritise; give a concrete next step and timeline", else "friendly and concise"
  Category: {intent} / {ticket_type}. Mention these products if relevant: {entities}   (omit if empty)
  ```

### A.4 Aspect-centric metrics (computed in `minimal_run report`, with bootstrap CIs like all other metrics)
- **ACS: Aspect Coverage Score.** For each reply and each non-`none` span aspect `a ∈ {prob_sub, cause}`: `max_i cos(BGE-M3(a), BGE-M3(sentence_i))`, averaged over aspects, then over rows. Report the **gold answers' ACS** as a reference ceiling. Sentence split: the existing `split_sentences` helper in `aspect_extractor.py`.
- **AAS: Action Acknowledgement Score.** Same computation for `action_taken`, only on rows where it is not `none`.
- **Aspect table** in `results.md`: ACS / AAS per system (EN/DE), plus the A.1 head accuracy table.
- Optional (only if an `ANTHROPIC_API_KEY` is set locally; never on Kaggle): an LLM-judge "completeness" score (0–2 per aspect) on 100 rows × 6 systems with a Claude Haiku model, run locally on the downloaded CSVs.

### A.5 Ablation for the aspect claim
- S4 vs S5 (RAFT with vs without aspects) is already in §1.
- **S6 = Gemma-3 RAG without aspects (zero-shot)**: `--mode rag_aspect --no_aspects --label s6_gemma3_noaspect`. This costs about 25 min across both GPUs; S2 vs S6 shows the aspect effect without fine-tuning.
- `results.md` gets an "Aspect contribution" paragraph generated from S2−S6 and S4−S5 with CIs, covering ROUGE-L, BERTScore and ACS.

---

## K. Automation: Claude Code runs everything, the human approves

### K.1 Division of labour (target: human ≈ 5–10 %)
| Human (one-time or review only) | Claude Code (everything else) |
|---|---|
| Kaggle: phone-verify the account (needed for GPU + Internet) | write code, unit tests and docs; keep `PROGRESS.md` current |
| Accept a Gemma licence only if a `google/` model is used (`unsloth/gemma-3-4b-it` is ungated) | push to GitHub on the work branch, open the PR |
| Approve permission prompts / keep auto mode on | push and run Kaggle kernels via the Kaggle CLI, poll status, download outputs and logs |
| Review the PR and the final `results.md` | diagnose failures from logs, fix, re-run (max 3 attempts per job, then stop and ask) |
| Merge the PR | push the RAFT adapter to HF from the local machine, update README/report figures |

### K.2 Kaggle via API (no notebook clicking)
New `scripts/kaggle_job.py`:
```
python scripts/kaggle_job.py push   --job prep_train|generate_eval|hello [--commit <sha>]
python scripts/kaggle_job.py status --job ...
python scripts/kaggle_job.py wait   --job ... --poll 300        # blocks, prints status changes
python scripts/kaggle_job.py pull   --job ...  --out runs/<job>/<timestamp>/
```
- Each job is a folder `kaggle/<job>/` holding `kernel-metadata.json` + `run.py`. Script kernels are easier to generate and diff than notebooks; the notebooks in §9 become optional human-readable mirrors.
- `kernel-metadata.json`: `"kernel_type": "script"`, `"enable_gpu": true`, `"enable_internet": true`, `"is_private": true`, `dataset_sources: ["tobiasbueck/multilingual-customer-support-tickets"]` (raw data straight from Kaggle, nothing to upload). For `generate_eval`, also add `kernel_sources: ["ranjankumarnayak/aspectforge-prep-train"]` to read job 1's outputs. For T4×2 set `"machine_shape": "NvidiaTeslaT4"` in the metadata (per the Kaggle CLI `docs/kernels_metadata.md`; the same value works with `kaggle kernels push --accelerator NvidiaTeslaT4`). Kaggle CLI 2.2.4 is installed locally and authenticated as user `ranjankumarnayak`, so kernel ids are `ranjankumarnayak/aspectforge-<job>`.
- `run.py`: `git clone --depth 1 -b <branch>` this repo → checkout the exact commit sha baked in at push time → `pip install unsloth` + `requirements-kaggle.txt` → run the pipeline steps as subprocesses with logs in `/kaggle/working/logs/` → write `/kaggle/working/STATUS.json` (`{"step": ..., "ok": bool, "error": ...}`) after every step, so a failed run can be diagnosed from `pull` alone.
- Raw data: `run.py` regenerates the splits from the Kaggle dataset with `src/preprocessing.py` (seed 42). Determine which raw CSV reproduces the report's 28,237 EN+DE rows and record it in `config/config_preprocess.yaml`.
- **No secrets on Kaggle:** all models come from ungated repos (`unsloth/gemma-3-4b-it`, `unsloth/gemma-2-2b-it`, `BAAI/*`, `ranjan56cse/cs-support-labels`). The RAFT adapter is saved as kernel output. After `pull`, Claude pushes it to HF **locally**, and job 2 reads it from job 1's kernel output via `kernel_sources`.
- Long waits: run `scripts/kaggle_job.py wait` in a background shell (or `/loop` at a 10–20 min cadence), never foreground polling.

### K.3 Local CPU test harness (catch bugs before spending GPU)
- `tests/` with pytest; fixtures are the 10-record sample CSVs in `docs/samples/` (committed via a `.gitignore` exception).
- `tests/test_metrics.py`, `test_aspects.py`, `test_prompt.py` (language line, checklist, budget), `test_retriever_fusion.py` (RRF on toy ranks), `test_report.py` (bootstrap on fake CSVs).
- `scripts/smoke_local.py`: runs the generator with a tiny random CPU model on 4 rows end-to-end, exercising the plumbing without a GPU.
- Rule: **no Kaggle push unless `pytest -q` passes.**

### K.4 Session protocol (also in `CLAUDE.md`)
1. Read `PROGRESS.md`, then the relevant `UPGRADE_PLAN.md` section; use `graphify query` for code navigation.
2. Work on branch `v3-minimal-run`; small commits; run `graphify update .` after code changes.
3. After each §10 step: tests green → commit → push → tick the box in `PROGRESS.md` with the date and commit sha.
4. Stop and ask the human only for credentials or licences, any paid resource, deleting data, merging to `main`, or after 3 failed attempts at the same job.

---

## 10. Implementation order and acceptance tests (do in this order; commit after each)

0. **Automation scaffold** (§K): `scripts/kaggle_job.py`, `kaggle/<job>/` skeletons, `tests/` + fixtures, `PROGRESS.md`. *Test:* `pytest -q` runs; `kaggle_job.py push --job hello` runs a 1-minute GPU kernel that prints `nvidia-smi` and comes back with `pull`.
1. **Config plumbing**: `load_yaml_with_overrides`, `kaggle.yaml`, the new dataclass fields. *Test:* old configs still load; `--overrides` changes `output_dir`.
2. **Metrics** (§8.1). *Test:* `rouge(["Größe ändern"],["Größe ändern"])` → rougeL 1.0; `lang_match(["Hello there, thanks","Vielen Dank für Ihre Nachricht"],["en","de"])` → 1.0.
2b. **Aspect v2** (§A.1–A.3): `aspect_heads.py`, `postprocess_spans`, entity fix, checklist prompt. *Test:* `tests/test_aspects.py`; heads trained on a CPU subset beat the majority baseline on macro-F1.
3. **model_backend + generator fixes** (§5.2–5.5). *Test (local CPU is not enough; do it in the Kaggle smoke cell):* one `<bos>` (print the first 3 ids); DE prompt → German reply; a prompt with 5 long hits stays ≤ `max_input_tokens` and still ends with `Support reply:`.
4. **Runners/modes/labels/Checkpointer dir** (§5.6). *Test:* `--mode rag_aspect --sample_csv ... --n 4` writes `{label}.csv` with `top_ce_score`, `n_ctx_used`.
5. **Retrieval** (§6.2–6.3). *Test:* `inference_retriever --n 20` runs; DE rows have `ce_score`; `exclude_ids` removes the given id.
6. **minimal_run sample / attach_aspects** (§6.1). *Test:* 250/250 split; rerun reuses the file (same hash).
7. **eval_retrieval** (§6.4).
8. **step3_raft build + train** (§7). *Test:* build with `--limit 20`, inspect 3 lines; train with `--limit 64 --max_steps 5` on Kaggle.
9. **CRAG** (§7.5) + `crag_tune`.
10. **report** (§8.2 + §A.4 ACS/AAS + aspect table). *Test:* run on fake CSVs (2 systems × 20 rows) generated by a tiny test helper, and check that the CI columns exist.
11. **Notebooks** (§9) + `requirements-kaggle.txt`.
12. **README**: embed the new architecture figure `docs/architecture_v3.png` (source: `docs/architecture_v3.svg`; it replaces Figure 2 of the old report), replace the pipeline section with "Minimal run on Kaggle", a model table (Gemma-3-4B-it, BGE-M3, bge-reranker-v2-m3), the new modes, the metric definitions (including what changed and why), the correct indexer command (`--config` only), and a link to `output/report/results.md`.

Steps 0–2b, 4–7 and 10 can be written and unit-tested without a GPU (fixtures in `docs/samples/`). Steps 3, 8 and 9 need the Kaggle smoke runs.

---

## 11. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Unsloth install conflicts with Kaggle's preinstalled packages | restart the kernel after `pip install unsloth`; if it still fails, use `backend: hf` with 4-bit + float32 compute (slower, about 1.5× the generation time) |
| Gemma-3 processor vs tokenizer issues in batching | use `getattr(tok, "tokenizer", tok)` for padding/decoding; string-content fallback in the chat template |
| T4 OOM | `generation_batch_size: 4`, `max_input_tokens: 2048`; training `batch: 1, grad_accum: 16` |
| Session times out mid-training | `save_steps=50` + `resume_from_checkpoint`; adapter pushed to HF at the end |
| `aspects_train.csv` not available | the index works without aspects (§6.2); RAFT uses the sample's own extracted aspects; note in results.md that the aspect rerank bonus was off |
| DE `lang_match` < 0.95 | add `"Antworte auf Deutsch."` as the last line before `Support reply:` for DE prompts and rerun the smoke test |
| Differences not significant at n=500 | report them honestly as "not significant"; the CI table is the point of the minimal run |
