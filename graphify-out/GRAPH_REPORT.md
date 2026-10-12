# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 28 files · ~45,711 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 495 nodes · 912 edges · 25 communities (14 shown, 11 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 41 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `98d4421e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)
- retriever.py
- MetricsCalculator
- AspectForge — Claude Code instructions
- aspect_extractor.py
- baseline_t5.py
- preprocessing.py
- File Descriptions
- utils.py
- kaggle_job.py
- test_config.py
- Profiler
- response_generator_v2.py
- .claude/CLAUDE.md
- indexer.py
- step2_sft.py

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 23 edges
2. `run()` - 18 edges
3. `Checkpointer` - 18 edges
4. `load_yaml_with_overrides()` - 16 edges
5. `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` - 15 edges
6. `DataPreprocessor` - 14 edges
7. `MetricsCalculator` - 14 edges
8. `process_row()` - 11 edges
9. `generate_batch()` - 11 edges
10. `run_rag_aspect()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `GPU work = Kaggle via API (plan §K)` --references--> `pull()`  [INFERRED]
  CLAUDE.md → scripts/kaggle_job.py
- `K.2 Kaggle via API (no notebook clicking)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `3. Kaggle constraints the code must respect` --references--> `get_hf_token()`  [INFERRED]
  UPGRADE_PLAN.md → src/utils.py
- `10. Implementation order and acceptance tests (do in this order; commit after each)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py

## Import Cycles
- None detected.

## Communities (25 total, 11 thin omitted)

### Community 0 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.11
Nodes (18): 0. Why: problems found in the current code, 10. Implementation order and acceptance tests (do in this order; commit after each), 11. Risks and fallbacks, 1. What the minimal run produces, 3. Kaggle constraints the code must respect, 4. `config/kaggle.yaml` (overrides) and config merging, 6.1 `src/minimal_run.py sample`, 6.4 `src/retrieval/eval_retrieval.py` (+10 more)

### Community 1 - "retriever.py"
Cohesion: 0.11
Nodes (9): `retriever.py`, aspect_query_vector(), aspect_score(), cross_encoder_rerank(), get_cross_encoder(), STEmbedder, TicketRetriever, 6.3 Retriever (`retriever.py`) (+1 more)

### Community 2 - "MetricsCalculator"
Cohesion: 0.26
Nodes (4): MetricsCalculator, 8.1 `MetricsCalculator` fixes, 8.2 `python -m src.minimal_run report --systems s1_gemma2 s2_gemma3 s3_gemma3_crag s4_raft_aspect s5_raft_noaspect`, 8. Metrics and report (`src/utils.py`, `src/minimal_run.py report`)

### Community 4 - "AspectForge — Claude Code instructions"
Cohesion: 0.25
Nodes (7): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules

### Community 5 - "aspect_extractor.py"
Cohesion: 0.08
Nodes (27): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_entities(), extract_labels_batch(), get_content_sentences() (+19 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.15
Nodes (10): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+2 more)

### Community 7 - "preprocessing.py"
Cohesion: 0.08
Nodes (4): DataPreprocessor, fix_encoding(), strip_email_filler(), track_run()

### Community 8 - "File Descriptions"
Cohesion: 0.08
Nodes (23): `aspect_extractor.py`, AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, `baseline_t5.py`, `config/generation_config.yaml`, `config/inference_retrieval_config.yaml`, Configuration, File Descriptions, Hardware (+15 more)

### Community 9 - "utils.py"
Cohesion: 0.06
Nodes (15): BaselineGenerator, compare_models(), ContextBuilder, DataLoader, EvaluationPipeline, load_eval_config(), Profiler, _Result (+7 more)

### Community 10 - "kaggle_job.py"
Cohesion: 0.14
Nodes (16): log(), main(), update_status(), kaggle_prefix(), main(), pull(), push(), read_job() (+8 more)

### Community 11 - "test_config.py"
Cohesion: 0.09
Nodes (23): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup, apply_overrides(), load_generation_config(), deep_merge() (+15 more)

### Community 13 - "response_generator_v2.py"
Cohesion: 0.05
Nodes (42): _apply_chat_template(), _aspect_columns_ready(), _aspect_row(), build_aspect_only_prompt(), build_baseline_prompt(), build_baseline_prompt_pt(), build_guided_rag_prompt(), compute_metrics() (+34 more)

### Community 21 - "indexer.py"
Cohesion: 0.06
Nodes (18): `inference_retriever.py`, load_config(), main(), reorder_columns(), StratifiedSplitter, CorpusIndexer, IndexerConfig, load_config() (+10 more)

### Community 23 - "step2_sft.py"
Cohesion: 0.07
Nodes (15): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch() (+7 more)

## Knowledge Gaps
- **46 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `graphify` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 208 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `File Descriptions` connect `File Descriptions` to `retriever.py`, `indexer.py`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._
- **Why does `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` connect `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` to `utils.py`, `MetricsCalculator`, `aspect_extractor.py`, `response_generator_v2.py`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `load_yaml_with_overrides()` (e.g. with `§10 steps` and `10. Implementation order and acceptance tests (do in this order; commit after each)`) actually correct?**
  _`load_yaml_with_overrides()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `retriever.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1076923076923077 - nodes in this community are weakly interconnected._