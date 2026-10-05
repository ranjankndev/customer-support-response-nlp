# Graph Report - aspectforge  (2026-10-05)

## Corpus Check
- 23 files · ~42,797 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: .csv 4, (none) 1, .pptx 1)

## Summary
- 434 nodes · 763 edges · 21 communities (12 shown, 9 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 35 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c9e0dd73`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- response_generator_v2.py
- retriever.py
- ProgressCallback
- utils.py
- indexer.py
- aspect_extractor.py
- baseline_t5.py
- preprocessing.py
- File Descriptions
- AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)
- BaselineGenerator
- AspectForge v3 — Progress
- Profiler
- AspectForge — Claude Code instructions
- .claude/CLAUDE.md

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
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `5.6 Runners and modes for S1–S5` --references--> `enrich_dataframe_with_aspects()`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `5.3 Prompt budget (F3)` --references--> `_make_retriever()`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `A.3 Aspects in retrieval and prompt` --references--> `aspect_score()`  [INFERRED]
  UPGRADE_PLAN.md → src/retrieval/retriever.py
- `3. Kaggle constraints the code must respect` --references--> `get_hf_token()`  [INFERRED]
  UPGRADE_PLAN.md → src/utils.py

## Import Cycles
- None detected.

## Communities (21 total, 9 thin omitted)

### Community 0 - "response_generator_v2.py"
Cohesion: 0.05
Nodes (42): load_config(), _apply_chat_template(), _aspect_columns_ready(), _aspect_row(), build_aspect_only_prompt(), build_baseline_prompt(), build_baseline_prompt_pt(), build_guided_rag_prompt() (+34 more)

### Community 1 - "retriever.py"
Cohesion: 0.10
Nodes (10): `retriever.py`, aspect_query_vector(), aspect_score(), cross_encoder_rerank(), get_cross_encoder(), STEmbedder, TicketRetriever, 5.6 Runners and modes for S1–S5 (+2 more)

### Community 3 - "utils.py"
Cohesion: 0.08
Nodes (7): DataLoader, EvaluationPipeline, load_eval_config(), MetricsCalculator, Profiler, _Result, 8.1 `MetricsCalculator` fixes

### Community 4 - "indexer.py"
Cohesion: 0.06
Nodes (18): `inference_retriever.py`, load_config(), main(), reorder_columns(), StratifiedSplitter, CorpusIndexer, IndexerConfig, load_config() (+10 more)

### Community 5 - "aspect_extractor.py"
Cohesion: 0.05
Nodes (32): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_labels_batch(), get_content_sentences(), get_gpu_profile() (+24 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.15
Nodes (10): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+2 more)

### Community 7 - "preprocessing.py"
Cohesion: 0.08
Nodes (4): DataPreprocessor, fix_encoding(), strip_email_filler(), track_run()

### Community 8 - "File Descriptions"
Cohesion: 0.08
Nodes (23): `aspect_extractor.py`, AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, `baseline_t5.py`, `config/generation_config.yaml`, `config/inference_retrieval_config.yaml`, Configuration, File Descriptions, Hardware (+15 more)

### Community 9 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.07
Nodes (27): extract_entities(), split_sentences(), 0. Why: problems found in the current code, 10. Implementation order and acceptance tests (do in this order; commit after each), 11. Risks and fallbacks, 1. What the minimal run produces, 4. `config/kaggle.yaml` (overrides) and config merging, 6.1 `src/minimal_run.py sample` (+19 more)

### Community 10 - "BaselineGenerator"
Cohesion: 0.11
Nodes (10): BaselineGenerator, compare_models(), ContextBuilder, 2. Files: new and changed, 7.1 Config, 7.2 `build`, 7.3 `train` (Unsloth QLoRA; follows Unsloth's Gemma-3 fine-tuning recipe), 7.4 Leakage rules (must hold, add asserts) (+2 more)

### Community 11 - "AspectForge v3 — Progress"
Cohesion: 0.33
Nodes (5): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup

### Community 13 - "AspectForge — Claude Code instructions"
Cohesion: 0.25
Nodes (7): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules

## Knowledge Gaps
- **50 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `GPU work = Kaggle via API (plan §K)`, `Ask the human only for` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 197 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` connect `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` to `response_generator_v2.py`, `BaselineGenerator`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _50 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `response_generator_v2.py` be split into smaller, more focused modules?**
  _Cohesion score 0.050837496326770495 - nodes in this community are weakly interconnected._
- **Why does `File Descriptions` connect `File Descriptions` to `retriever.py`, `indexer.py`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Should `retriever.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09788359788359788 - nodes in this community are weakly interconnected._
- **Should `utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08266129032258064 - nodes in this community are weakly interconnected._
- **Should `indexer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06352941176470588 - nodes in this community are weakly interconnected._