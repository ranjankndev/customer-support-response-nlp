# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 29 files · ~46,879 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 535 nodes · 974 edges · 31 communities (20 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 49 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0ad427c1`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)
- inference_retriever.py
- test_metrics.py
- response_generator_v2.py
- aspect_extractor.py
- baseline_t5.py
- preprocessing.py
- File Descriptions
- utils.py
- kaggle_job.py
- test_config.py
- Profiler
- run_extra_baseline
- .claude/CLAUDE.md
- indexer.py
- step2_sft.py
- run_rag_aspect
- run_rag_asp_guided
- GenerationConfig
- Checkpointer
- DataFrame
- run

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 23 edges
2. `MetricsCalculator` - 20 edges
3. `run()` - 18 edges
4. `Checkpointer` - 18 edges
5. `load_yaml_with_overrides()` - 16 edges
6. `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` - 15 edges
7. `DataPreprocessor` - 14 edges
8. `process_row()` - 11 edges
9. `generate_batch()` - 11 edges
10. `run_rag_aspect()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `GPU work = Kaggle via API (plan §K)` --references--> `pull()`  [INFERRED]
  CLAUDE.md → scripts/kaggle_job.py
- `K.2 Kaggle via API (no notebook clicking)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `3. Kaggle constraints the code must respect` --references--> `get_hf_token()`  [INFERRED]
  UPGRADE_PLAN.md → src/utils.py
- `10. Implementation order and acceptance tests (do in this order; commit after each)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py

## Import Cycles
- None detected.

## Communities (31 total, 11 thin omitted)

### Community 0 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.11
Nodes (17): 0. Why: problems found in the current code, 10. Implementation order and acceptance tests (do in this order; commit after each), 11. Risks and fallbacks, 1. What the minimal run produces, 3. Kaggle constraints the code must respect, 4. `config/kaggle.yaml` (overrides) and config merging, 8.2 `python -m src.minimal_run report --systems s1_gemma2 s2_gemma3 s3_gemma3_crag s4_raft_aspect s5_raft_noaspect`, 8. Metrics and report (`src/utils.py`, `src/minimal_run.py report`) (+9 more)

### Community 1 - "inference_retriever.py"
Cohesion: 0.07
Nodes (20): `inference_retriever.py`, `retriever.py`, get_retriever(), load_retriever_config(), main(), RetrievalConfig, retrieve_similar(), run_file() (+12 more)

### Community 2 - "test_metrics.py"
Cohesion: 0.06
Nodes (15): MetricsCalculator, _UnicodeTokenizer, calc(), _FakeTensor, MetCfg, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1() (+7 more)

### Community 4 - "response_generator_v2.py"
Cohesion: 0.20
Nodes (4): _apply_chat_template(), _is_pretrained(), load_model(), get_hf_token()

### Community 5 - "aspect_extractor.py"
Cohesion: 0.08
Nodes (26): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_entities(), extract_labels_batch(), get_content_sentences() (+18 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.16
Nodes (9): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+1 more)

### Community 7 - "preprocessing.py"
Cohesion: 0.06
Nodes (11): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules, DataPreprocessor (+3 more)

### Community 8 - "File Descriptions"
Cohesion: 0.08
Nodes (23): `aspect_extractor.py`, AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, `baseline_t5.py`, `config/generation_config.yaml`, `config/inference_retrieval_config.yaml`, Configuration, File Descriptions, Hardware (+15 more)

### Community 9 - "utils.py"
Cohesion: 0.06
Nodes (14): BaselineGenerator, ContextBuilder, DataLoader, EvaluationPipeline, load_eval_config(), Profiler, _Result, 2. Files: new and changed (+6 more)

### Community 10 - "kaggle_job.py"
Cohesion: 0.14
Nodes (15): log(), main(), update_status(), kaggle_prefix(), main(), pull(), push(), read_job() (+7 more)

### Community 11 - "test_config.py"
Cohesion: 0.08
Nodes (26): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup, apply_overrides(), load_generation_config(), deep_merge() (+18 more)

### Community 13 - "run_extra_baseline"
Cohesion: 0.16
Nodes (7): build_baseline_prompt(), build_baseline_prompt_pt(), generate_batch_extra(), _get_spans(), load_extra_model(), run_extra_baseline(), 5.2 `src/generation/model_backend.py`

### Community 21 - "indexer.py"
Cohesion: 0.08
Nodes (10): load_config(), main(), reorder_columns(), StratifiedSplitter, CorpusIndexer, IndexerConfig, load_config(), _load_yaml() (+2 more)

### Community 23 - "step2_sft.py"
Cohesion: 0.07
Nodes (15): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch() (+7 more)

### Community 25 - "run_rag_aspect"
Cohesion: 0.20
Nodes (6): _aspect_row(), build_aspect_only_prompt(), extract_sentiment_severity(), run_aspect_only(), run_rag_aspect(), 5.5 Sentiment/severity for DE (F8)

### Community 26 - "run_rag_asp_guided"
Cohesion: 0.20
Nodes (8): build_guided_rag_prompt(), _make_retriever(), run_rag_asp_guided(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.3 Prompt budget (F3), 5.4 Language line (F2), 5.6 Runners and modes for S1–S5, 5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)

### Community 27 - "GenerationConfig"
Cohesion: 0.25
Nodes (7): generate_batch(), generate_one(), GenerationConfig, resolve_data_path(), run_baseline(), run_rag_guided(), test_sample_csv_overrides_the_input_argument()

### Community 28 - "Checkpointer"
Cohesion: 0.20
Nodes (3): inject_sentiment_severity(), run_rag_only(), Checkpointer

### Community 29 - "DataFrame"
Cohesion: 0.22
Nodes (5): load_config(), _aspect_columns_ready(), compute_metrics(), enrich_dataframe_with_aspects(), _resolve_aspect_extractor_kwargs()

### Community 30 - "run"
Cohesion: 0.24
Nodes (5): maybe_merge_aspects_file(), print_metrics(), run(), compare_models(), load_csv_or_excel()

## Knowledge Gaps
- **44 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `graphify` (+39 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 227 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MetricsCalculator` connect `test_metrics.py` to `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)`, `response_generator_v2.py`, `baseline_t5.py`, `utils.py`, `DataFrame`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _44 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._
- **Why does `File Descriptions` connect `File Descriptions` to `inference_retriever.py`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MetricsCalculator` (e.g. with `calc()` and `test_cls_cosine_is_no_longer_called_bertscore()`) actually correct?**
  _`MetricsCalculator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `inference_retriever.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06565656565656566 - nodes in this community are weakly interconnected._