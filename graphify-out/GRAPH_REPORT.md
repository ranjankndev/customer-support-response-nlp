# Graph Report - aspectforge  (2026-10-05)

## Corpus Check
- 27 files · ~43,998 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 465 nodes · 849 edges · 28 communities (18 shown, 10 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f0585f5b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- GenerationConfig
- inference_retriever.py
- MetricsCalculator
- utils.py
- run_baseline
- aspect_extractor.py
- baseline_t5.py
- DataPreprocessor
- File Descriptions
- AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)
- kaggle_job.py
- AspectForge v3 — Progress
- Profiler
- response_generator_v2.py
- .claude/CLAUDE.md
- indexer.py
- _resolve_aspect_extractor_kwargs
- step2_sft.py
- run_rag_only
- Checkpointer
- run
- run_rag_asp_guided

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 19 edges
2. `run()` - 18 edges
3. `Checkpointer` - 18 edges
4. `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` - 15 edges
5. `DataPreprocessor` - 14 edges
6. `MetricsCalculator` - 14 edges
7. `process_row()` - 11 edges
8. `generate_batch()` - 11 edges
9. `run_rag_aspect()` - 11 edges
10. `run_rag_asp_guided()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `K.2 Kaggle via API (no notebook clicking)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `A.3 Aspects in retrieval and prompt` --references--> `aspect_score()`  [INFERRED]
  UPGRADE_PLAN.md → src/retrieval/retriever.py
- `GPU work = Kaggle via API (plan §K)` --references--> `pull()`  [INFERRED]
  CLAUDE.md → scripts/kaggle_job.py
- `10. Implementation order and acceptance tests (do in this order; commit after each)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py

## Import Cycles
- None detected.

## Communities (28 total, 10 thin omitted)

### Community 0 - "GenerationConfig"
Cohesion: 0.17
Nodes (10): _apply_chat_template(), generate_batch(), generate_one(), GenerationConfig, load_generation_config(), _make_retriever(), resolve_data_path(), run_rag_aspect() (+2 more)

### Community 1 - "inference_retriever.py"
Cohesion: 0.09
Nodes (15): `inference_retriever.py`, `retriever.py`, get_retriever(), load_retriever_config(), main(), RetrievalConfig, retrieve_similar(), run_file() (+7 more)

### Community 2 - "MetricsCalculator"
Cohesion: 0.26
Nodes (4): MetricsCalculator, 8.1 `MetricsCalculator` fixes, 8.2 `python -m src.minimal_run report --systems s1_gemma2 s2_gemma3 s3_gemma3_crag s4_raft_aspect s5_raft_noaspect`, 8. Metrics and report (`src/utils.py`, `src/minimal_run.py report`)

### Community 3 - "utils.py"
Cohesion: 0.06
Nodes (7): strip_email_filler(), track_run(), DataLoader, EvaluationPipeline, load_eval_config(), Profiler, _Result

### Community 4 - "run_baseline"
Cohesion: 0.22
Nodes (4): build_baseline_prompt(), build_guided_rag_prompt(), run_baseline(), 5.4 Language line (F2)

### Community 5 - "aspect_extractor.py"
Cohesion: 0.08
Nodes (27): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_entities(), extract_labels_batch(), get_content_sentences() (+19 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.15
Nodes (10): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+2 more)

### Community 7 - "DataPreprocessor"
Cohesion: 0.10
Nodes (8): Ask the human only for, AspectForge — Claude Code instructions, graphify, Start of every session, Useful facts, Working rules, DataPreprocessor, fix_encoding()

### Community 8 - "File Descriptions"
Cohesion: 0.08
Nodes (23): `aspect_extractor.py`, AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, `baseline_t5.py`, `config/generation_config.yaml`, `config/inference_retrieval_config.yaml`, Configuration, File Descriptions, Hardware (+15 more)

### Community 9 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.06
Nodes (27): BaselineGenerator, compare_models(), ContextBuilder, 0. Why: problems found in the current code, 11. Risks and fallbacks, 1. What the minimal run produces, 2. Files: new and changed, 4. `config/kaggle.yaml` (overrides) and config merging (+19 more)

### Community 10 - "kaggle_job.py"
Cohesion: 0.12
Nodes (18): GPU work = Kaggle via API (plan §K), log(), main(), update_status(), kaggle_prefix(), main(), pull(), push() (+10 more)

### Community 11 - "AspectForge v3 — Progress"
Cohesion: 0.33
Nodes (5): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup

### Community 13 - "response_generator_v2.py"
Cohesion: 0.15
Nodes (8): generate_batch_extra(), _is_pretrained(), load_extra_model(), load_model(), run_extra_baseline(), get_hf_token(), 3. Kaggle constraints the code must respect, 5.2 `src/generation/model_backend.py`

### Community 21 - "indexer.py"
Cohesion: 0.08
Nodes (9): load_config(), main(), reorder_columns(), StratifiedSplitter, CorpusIndexer, IndexerConfig, load_config(), main() (+1 more)

### Community 23 - "step2_sft.py"
Cohesion: 0.07
Nodes (15): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch() (+7 more)

### Community 25 - "run_rag_only"
Cohesion: 0.20
Nodes (7): extract_sentiment_severity(), inject_sentiment_severity(), run_rag_only(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.5 Sentiment/severity for DE (F8), 5.6 Runners and modes for S1–S5, 5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)

### Community 27 - "run"
Cohesion: 0.21
Nodes (6): _aspect_columns_ready(), compute_metrics(), enrich_dataframe_with_aspects(), maybe_merge_aspects_file(), print_metrics(), run()

### Community 29 - "run_rag_asp_guided"
Cohesion: 0.18
Nodes (6): _aspect_row(), build_aspect_only_prompt(), build_baseline_prompt_pt(), _get_spans(), run_aspect_only(), run_rag_asp_guided()

## Knowledge Gaps
- **47 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `graphify` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 199 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `File Descriptions` connect `File Descriptions` to `inference_retriever.py`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _47 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `inference_retriever.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09269162210338681 - nodes in this community are weakly interconnected._
- **Why does `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` connect `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` to `MetricsCalculator`, `aspect_extractor.py`, `kaggle_job.py`, `response_generator_v2.py`, `run_rag_only`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Should `utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06456456456456457 - nodes in this community are weakly interconnected._
- **Why does `DataPreprocessor` connect `DataPreprocessor` to `utils.py`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Should `aspect_extractor.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07505285412262157 - nodes in this community are weakly interconnected._