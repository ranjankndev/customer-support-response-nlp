# AspectForge — Multilingual Aspect-Aware Customer Support Response Generation

Aspect-aware RAG system for bilingual (EN/DE) customer support tickets.  
Decomposes each ticket into 8 structured aspect fields and uses them to guide retrieval, reranking, and response generation.

---

## Project Structure

```
customer-support-response-nlp/
├── src/
│   ├── retrieval/                        # Retrieval logic
│   │   ├── retriever.py                  # Core retrieval and reranking
│   │   └── inference_retriever.py        # Inference-time retrieval wrapper
│   ├── generation/                       # Response generation
│   │   └── response_generator_v2.py      # Main generation script (4 modes)
│   ├── baseline/                         # Baseline models
│   │   └── baseline_t5.py               # Flan-T5-base zero-shot baseline
│   ├── aspect/                           # Aspect extraction inference
│   │   └── aspect_extractor.py           # Runs SFT adapter at inference time
│   ├── step1_silver_labels.py            # LLM-as-Judge silver label generation
│   ├── step2_sft.py                      # SFT training (Qwen2.5-1.5B + LoRA)
│   ├── utils.py                          # Shared utilities
│   └── __init__.py
├── config/
│   ├── generation_config.yaml            # Generator settings
│   └── inference_retrieval_config.yaml   # Retrieval settings
├── data/
│   ├── processed/                        # Cleaned ticket CSVs
│   └── index/                            # FAISS index and metadata
├── output/
│   ├── generation/                       # Generated response CSVs
│   └── aspect/                           # Extracted aspect CSVs
├── eval_results/                         # Evaluation metric outputs
├── logs/                                 # nohup and training logs
├── preprocessing.py                      # PII removal and data cleaning
└── requirements.txt
```

---

## Setup

```bash
pip install torch transformers sentence-transformers faiss-gpu \
            bitsandbytes peft accelerate rouge-score bert-score \
            pandas numpy pyyaml
```

Log in to HuggingFace (required for gated models):
```bash
huggingface-cli login
```

---

## Pipeline — Run Order

### Step 1 — Preprocess raw tickets
```bash
python preprocessing.py \
  --input  data/raw/tickets.csv \
  --output data/processed/cs-support-multi-train.csv
```
Removes PII (names, phone numbers, emails) and replaces them with
placeholders (`<name>`, `<tel_num>`). Detects language (EN/DE).

---

### Step 2 — Generate silver labels (LLM-as-Judge)
```bash
python -m src.step1_silver_labels \
  --input  data/processed/cs-support-multi-train.csv \
  --output output/aspect/silver_labels.csv
```
A large teacher model annotates each ticket with aspect fields.
These silver labels are used to train the SFT aspect extraction model.

---

### Step 3 — Build FAISS index
```bash
python -m src.retrieval.indexer \
  --input    data/processed/cs-support-multi-train.csv \
  --index    data/index/faiss_index.bin \
  --metadata data/index/metadata.json
```
Encodes all ticket-response pairs with `intfloat/multilingual-e5-large`
and stores them in a flat FAISS inner-product index.

---

### Step 3b — Train SFT aspect extraction model
```bash
python -m src.step2_sft \
  --input  output/aspect/silver_labels.csv \
  --output output/aspect/sft_checkpoint
```
Fine-tunes Qwen2.5-1.5B with LoRA adapters on the silver labels produced
in Step 2. Learns to classify `intent` and `ticket_type_nli` for each ticket.
The trained adapter is saved locally and can be pushed to HuggingFace Hub.

---

### Step 4 — Extract aspects for test set
```bash
python -m src.aspect.aspect_extractor \
  --input   data/processed/cs-support-multi-test.csv \
  --adapter ranjan56cse/cs-support-labels \
  --adapter_subfolder sft_checkpoint \
  --out     output/aspect/aspect_test_v2.csv
```
Runs the full four-stage extraction pipeline on the test set:
sentence splitting → role ranking → entity extraction → SFT classification.
Produces the aspects file needed for `--mode aspect` and `--mode rag_aspect`.
Output: `output/aspect/aspect_test_v2.csv`

---

### Step 5 — Generate responses

**Flan-T5-base baseline:**
```bash
python -m src.baseline.baseline_t5 \
  --config config/generation_config.yaml \
  --input  data/processed/cs-support-multi-test.csv \
  --start_idx 0 --end_idx 22462
```

**RAG only:**
```bash
python -m src.generation.response_generator_v2 \
  --config config/generation_config.yaml \
  --mode   rag \
  --input  data/processed/cs-support-multi-test.csv \
  --start_idx 0 --end_idx 22462
```

**Aspect only:**
```bash
python -m src.generation.response_generator_v2 \
  --config       config/generation_config.yaml \
  --mode         aspect \
  --input        data/processed/cs-support-multi-test.csv \
  --aspects_file output/aspect/aspect_test_v2.csv \
  --start_idx 0 --end_idx 22462
```

**RAG + Aspects (full system):**
```bash
python -m src.generation.response_generator_v2 \
  --config       config/generation_config.yaml \
  --mode         rag_aspect \
  --input        data/processed/cs-support-multi-test.csv \
  --aspects_file output/aspect/aspect_test_v2.csv \
  --start_idx 0 --end_idx 22462
```

Add `nohup ... > logs/run.log 2>&1 &` to run in background.

---

## File Descriptions

### `preprocessor.py`
Cleans raw ticket CSV files. Removes PII and standardises text encoding.
- **Input:** raw CSV with `subject`, `body`, `language` columns
- **Output:** cleaned CSV with `<name>`, `<tel_num>` placeholders

### `indexer.py`
Builds the FAISS retrieval index from the training set.
- **Input:** processed train CSV
- **Output:** `data/index/faiss_index.bin`, `data/index/metadata.json`
- **Model:** `intfloat/multilingual-e5-large`

### `retriever.py`
Core retrieval logic used at inference time. Not run directly.

Key functions:
- `aspect_query_vector()` — builds weighted query embedding from aspect fields
- `aspect_score()` — computes structured aspect overlap for DE reranking
- `cross_encoder_rerank()` — applies CE reranking for EN tickets
- `TicketRetriever.retrieve()` — main entry point, language-conditional

Language-conditional strategy:
- **EN:** aspect-weighted query → FAISS → cross-encoder reranking
- **DE:** dense query (subject+body) → FAISS → aspect score bonus (α=0.1)

### `inference_retriever.py`
Thin wrapper around `retriever.py` for standalone inference use.
- **CLI:** `--config`, `--input`, `--out`, `--n`
- **Functions:** `retrieve_similar()` for single tickets, `run_file()` for batch

### `step1_silver_labels.py`
LLM-as-Judge distillation. A large teacher model annotates each ticket
with aspect fields to produce silver labels for SFT training.
- **Input:** processed ticket CSV
- **Output:** CSV with 5 annotated fields: `prob_sub`, `prob_statement`,
  `cause`, `intent`, `ticket_type_nli`

### `step2_sft.py`
Supervised fine-tuning of the aspect extraction model. Trains
Qwen2.5-1.5B with LoRA adapters on the silver labels from Step 1.
The student model learns to classify `intent` and `ticket_type_nli`
at a fraction of the teacher model's inference cost.
- **Input:** silver labels CSV from `step1_silver_labels.py`
- **Output:** LoRA adapter checkpoint (local dir or HuggingFace Hub)
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Default adapter hub:** `ranjan56cse/cs-support-labels`

### `aspect_extractor.py`
Runs inference using the trained SFT LoRA adapter. Extracts all
five SFT-classified fields per ticket and combines them with the
rule-based fields (entities, action_taken, urgency) to produce the
full eight-aspect output used by the generation pipeline.
- **Input:** processed ticket CSV
- **Output:** CSV with 8 aspect fields per ticket
- **CLI:** `--input`, `--adapter`, `--adapter_subfolder`, `--out`,
  `--config`, `--batch`, `--n`
  
  
  
  

```bash
# Using HuggingFace adapter (recommended)
python -m src.aspect.aspect_extractor \
  --input             data/processed/cs-support-multi-test.csv \
  --adapter           ranjan56cse/cs-support-labels \
  --adapter_subfolder sft_checkpoint \
  --out               output/aspect/aspect_test_v2.csv

# Using local adapter
python -m src.aspect.aspect_extractor \
  --input   data/processed/cs-support-multi-test.csv \
  --adapter ./sft_checkpoint \
  --out     output/aspect/aspect_test_v2.csv
```

Output columns: `idx`, `language`, `subject`, `prob_sub`,
`prob_statement`, `cause`, `intent`, `ticket_type_nli`,
`entities`, `action_taken`, `urgency`

### `response_generator_v2.py`
Main generation script. Runs all ablation configurations.
- **Model:** `google/gemma-2-2b-it` (4-bit NF4, bfloat16 compute)
- **Modes:** `baseline`, `rag`, `aspect`, `rag_aspect`, `all`
- **Output:** `output/generation/` — one CSV per mode with responses and metrics
- **Checkpointing:** resumes from `.ckpt_<mode>.csv` if interrupted

**CLI parameters:**

| Parameter | Required | Description |
|---|---|---|
| `--config` | Yes | Path to `generation_config.yaml` |
| `--mode` | Yes | `baseline` / `rag` / `aspect` / `rag_aspect` / `all` |
| `--input` | Yes | Input ticket CSV |
| `--aspects_file` | For aspect/rag_aspect | Extracted aspects CSV |
| `--start_idx` | No | First row index (inclusive) |
| `--end_idx` | No | Last row index (inclusive) |

**Run commands — one mode at a time (recommended):**

Clear checkpoints before each run:
```bash
rm -f .ckpt_baseline.csv .ckpt_rag_only.csv       .ckpt_aspect_only.csv .ckpt_rag_aspect.csv
rm -f output/generation/cs-support-multi-test_with_aspects*.csv
```

**Baseline (zero-shot, no RAG, no aspects):**
```bash
nohup python -m src.generation.response_generator_v2   --config config/generation_config.yaml   --mode   baseline   --input  data/processed/cs-support-multi-test.csv   --start_idx 0 --end_idx 2462   > "logs/rg_v2_baseline_$(date +%Y%m%d_%H%M).log" 2>&1 &
echo "PID: $!"
```

**RAG only:**
```bash
nohup python -m src.generation.response_generator_v2   --config config/generation_config.yaml   --mode   rag   --input  data/processed/cs-support-multi-test.csv   --start_idx 0 --end_idx 2462   > "logs/rg_v2_rag_$(date +%Y%m%d_%H%M).log" 2>&1 &
echo "PID: $!"
```

**Aspect only:**
```bash
nohup python -m src.generation.response_generator_v2   --config       config/generation_config.yaml   --mode         aspect   --input        data/processed/cs-support-multi-test.csv   --aspects_file output/aspect/aspect_test_v2.csv   --start_idx 0 --end_idx 2462   > "logs/rg_v2_aspect_$(date +%Y%m%d_%H%M).log" 2>&1 &
echo "PID: $!"
```

**RAG + Aspects (full system):**
```bash
nohup python -m src.generation.response_generator_v2   --config       config/generation_config.yaml   --mode         rag_aspect   --input        data/processed/cs-support-multi-test.csv   --aspects_file output/aspect/aspect_test_v2.csv   --start_idx 0 --end_idx 2462   > "logs/rg_v2_rag_aspect_$(date +%Y%m%d_%H%M).log" 2>&1 &
echo "PID: $!"
```

**Check if a job is running:**
```bash
ps aux | grep "response_generator\|baseline_t5" | grep -v grep
```

**Monitor log in real time:**
```bash
tail -f logs/rg_v2_*.log
```

**Check for errors or completion:**
```bash
grep "Done\|ERROR\|Traceback" logs/rg_v2_*.log
```

### `baseline_t5.py`
Flan-T5-base zero-shot seq2seq baseline.
- **Model:** `google/flan-t5-base`
- **Output:** `output/generation/*_t5_base.csv`
- No retrieval, no aspects — pure zero-shot generation

---

## Configuration

### `config/generation_config.yaml`
```yaml
model_id              : "google/gemma-2-2b-it"
use_4bit              : true
max_new_tokens        : 256
generation_batch_size : 24
output_dir            : "output/generation"
checkpoint_every      : 50
bertscore_model       : "bert-base-multilingual-cased"
```

### `config/inference_retrieval_config.yaml`
```yaml
model_name            : intfloat/multilingual-e5-large
top_k                 : 5
use_cross_encoder     : true
cross_encoder_model   : cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
aspect_rerank_weight  : 0.1
filter_by_language    : false
```

---

## Output Files

| File | Description |
|---|---|
| `*_baseline.csv` | Zero-shot Gemma responses |
| `*_rag_only.csv` | RAG-only responses |
| `*_aspect_only.csv` | Aspect-only responses |
| `*_rag_aspect.csv` | RAG + aspect responses |
| `*_t5_base.csv` | Flan-T5-base baseline responses |
| `*_metrics.csv` | ROUGE / BLEU / BERTScore per system |

All output CSVs include: `idx`, `language`, `subject`, `body`,
`answer` (gold), `generated_answer`, prompt, response, and length columns.

---

## Hardware

Tested on NVIDIA RTX 4090 (24GB VRAM).  



