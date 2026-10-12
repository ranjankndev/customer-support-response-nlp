# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 31 files · ~51,690 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 634 nodes · 1217 edges · 47 communities (33 shown, 14 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 52 edges (avg confidence: 0.94)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f80ca780`
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
- AspectForge — Multilingual Aspect-Aware Customer Support Response Generation
- utils.py
- kaggle_job.py
- test_config.py
- run
- test_aspects.py
- .claude/CLAUDE.md
- indexer.py
- step2_sft.py
- DataFrame
- build_aspect_only_prompt
- generate_batch
- Checkpointer
- GenerationConfig
- aspect_heads.py
- evaluate_heads
- Any
- step1_silver_labels.py
- .retrieve
- process_row
- retriever.py
- is_pleasantry
- AspectForge — Claude Code instructions
- File Descriptions
- 6. Data, aspects and retrieval
- Pipeline — Run Order
- ProgressCallback
- predict_head_labels
- fix_encoding
- run_rag_guided
- test_sft_label_collapse_is_documented_from_the_sample

## God Nodes (most connected - your core abstractions)
1. `postprocess_spans()` - 23 edges
2. `GenerationConfig` - 23 edges
3. `MetricsCalculator` - 20 edges
4. `run()` - 18 edges
5. `load_yaml_with_overrides()` - 18 edges
6. `Checkpointer` - 18 edges
7. `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` - 15 edges
8. `DataPreprocessor` - 14 edges
9. `process_row()` - 13 edges
10. `build_aspect_checklist()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `GPU work = Kaggle via API (plan §K)` --references--> `pull()`  [INFERRED]
  CLAUDE.md → scripts/kaggle_job.py
- `K.2 Kaggle via API (no notebook clicking)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `10. Implementation order and acceptance tests (do in this order; commit after each)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `10. Implementation order and acceptance tests (do in this order; commit after each)` --references--> `postprocess_spans()`  [INFERRED]
  UPGRADE_PLAN.md → src/aspect/aspect_extractor.py

## Import Cycles
- None detected.

## Communities (47 total, 14 thin omitted)

### Community 0 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.06
Nodes (30): extract_entities(), test_a_word_repeated_in_the_body_is_kept(), test_product_patterns_still_win(), test_title_case_subject_words_no_longer_become_entities(), 0. Why: problems found in the current code, 10. Implementation order and acceptance tests (do in this order; commit after each), 11. Risks and fallbacks, 1. What the minimal run produces (+22 more)

### Community 1 - "inference_retriever.py"
Cohesion: 0.20
Nodes (8): `inference_retriever.py`, get_retriever(), load_retriever_config(), main(), RetrievalConfig, retrieve_similar(), run_file(), 5.3 Prompt budget (F3)

### Community 2 - "test_metrics.py"
Cohesion: 0.06
Nodes (17): MetricsCalculator, _UnicodeTokenizer, calc(), _FakeTensor, MetCfg, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1() (+9 more)

### Community 4 - "response_generator_v2.py"
Cohesion: 0.13
Nodes (10): build_baseline_prompt(), build_baseline_prompt_pt(), generate_batch_extra(), _is_pretrained(), load_extra_model(), load_model(), run_extra_baseline(), get_hf_token() (+2 more)

### Community 5 - "aspect_extractor.py"
Cohesion: 0.17
Nodes (10): get_content_sentences(), get_gpu_profile(), get_model(), get_ranker(), is_pure_greeting(), run(), split_sentences(), strip_greeting_prefix() (+2 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.15
Nodes (10): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+2 more)

### Community 7 - "preprocessing.py"
Cohesion: 0.08
Nodes (4): DataPreprocessor, fix_encoding(), strip_email_filler(), track_run()

### Community 8 - "AspectForge — Multilingual Aspect-Aware Customer Support Response Generation"
Cohesion: 0.22
Nodes (8): AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, `config/generation_config.yaml`, `config/inference_retrieval_config.yaml`, Configuration, Hardware, Output Files, Project Structure, Setup

### Community 9 - "utils.py"
Cohesion: 0.07
Nodes (11): _aspect_value(), BaselineGenerator, compare_models(), ContextBuilder, DataLoader, EvaluationPipeline, load_eval_config(), Profiler (+3 more)

### Community 10 - "kaggle_job.py"
Cohesion: 0.14
Nodes (16): log(), main(), update_status(), kaggle_prefix(), main(), pull(), push(), read_job() (+8 more)

### Community 11 - "test_config.py"
Cohesion: 0.09
Nodes (21): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup, apply_overrides(), load_generation_config(), deep_merge() (+13 more)

### Community 12 - "run"
Cohesion: 0.22
Nodes (5): maybe_merge_aspects_file(), print_metrics(), Profiler, _Result, run()

### Community 13 - "test_aspects.py"
Cohesion: 0.11
Nodes (22): jaccard(), postprocess_spans(), build_aspect_checklist(), _stub_retriever(), test_a_cause_identical_to_the_problem_becomes_none(), test_a_distinct_cause_survives(), test_a_pleasantry_span_becomes_none_without_a_candidate(), test_a_pleasantry_span_is_replaced_by_the_next_ranked_sentence() (+14 more)

### Community 21 - "indexer.py"
Cohesion: 0.08
Nodes (11): load_config(), main(), reorder_columns(), StratifiedSplitter, CorpusIndexer, IndexerConfig, load_config(), _load_yaml() (+3 more)

### Community 23 - "step2_sft.py"
Cohesion: 0.12
Nodes (6): build_input_prompt(), build_target_output(), get_gpu_profile(), _push_checkpoint_to_hf(), run(), SilverDataset

### Community 25 - "DataFrame"
Cohesion: 0.18
Nodes (8): _aspect_columns_ready(), _aspect_row(), enrich_dataframe_with_aspects(), _get_spans(), _make_retriever(), run_aspect_only(), run_rag_asp_guided(), run_rag_aspect()

### Community 26 - "build_aspect_only_prompt"
Cohesion: 0.20
Nodes (7): build_aspect_only_prompt(), extract_sentiment_severity(), test_aspect_only_prompt_uses_the_checklist_too(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.4 Language line (F2), 5.5 Sentiment/severity for DE (F8), 5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)

### Community 27 - "generate_batch"
Cohesion: 0.25
Nodes (4): _apply_chat_template(), generate_batch(), generate_one(), run_baseline()

### Community 28 - "Checkpointer"
Cohesion: 0.20
Nodes (3): inject_sentiment_severity(), run_rag_only(), Checkpointer

### Community 29 - "GenerationConfig"
Cohesion: 0.22
Nodes (7): load_config(), compute_metrics(), GenerationConfig, _resolve_aspect_extractor_kwargs(), resolve_data_path(), test_generation_config_defaults_match_the_plan(), test_sample_csv_overrides_the_input_argument()

### Community 30 - "aspect_heads.py"
Cohesion: 0.20
Nodes (13): align_targets(), _cmd_eval(), _cmd_predict(), _cmd_train(), embed_texts(), HeadsConfig, load_bundle(), load_dense() (+5 more)

### Community 31 - "evaluate_heads"
Cohesion: 0.20
Nodes (9): evaluate_heads(), predict_from_features(), score_head(), train_heads(), _separable_features(), test_heads_beat_the_majority_baseline_on_macro_f1(), test_heads_predict_labels_for_new_rows(), test_predict_rejects_embeddings_of_the_wrong_width() (+1 more)

### Community 32 - "Any"
Cohesion: 0.18
Nodes (6): _build_chat_text(), build_sft_prompt(), _decode_generated(), extract_labels_batch(), _next_candidate(), _parse_labels()

### Community 33 - "step1_silver_labels.py"
Cohesion: 0.27
Nodes (8): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch()

### Community 34 - ".retrieve"
Cohesion: 0.24
Nodes (6): `retriever.py`, aspect_query_vector(), aspect_score(), cross_encoder_rerank(), 5.6 Runners and modes for S1–S5, A.3 Aspects in retrieval and prompt

### Community 35 - "process_row"
Cohesion: 0.20
Nodes (6): build_prob_statement(), encode_anchors(), pick_best_span(), process_row(), rank_sentences(), score_urgency()

### Community 36 - "retriever.py"
Cohesion: 0.24
Nodes (3): get_cross_encoder(), STEmbedder, TicketRetriever

### Community 37 - "is_pleasantry"
Cohesion: 0.22
Nodes (6): has_action_cue(), is_pleasantry(), test_pleasantries_are_recognised(), test_problem_sentences_are_not_pleasantries(), test_process_row_cleans_the_spans_of_a_real_sample_row(), test_real_actions_keep_their_cue()

### Community 38 - "AspectForge — Claude Code instructions"
Cohesion: 0.25
Nodes (7): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules

### Community 39 - "File Descriptions"
Cohesion: 0.25
Nodes (8): `aspect_extractor.py`, `baseline_t5.py`, File Descriptions, `indexer.py`, `preprocessor.py`, `response_generator_v2.py`, `step1_silver_labels.py`, `step2_sft.py`

### Community 40 - "6. Data, aspects and retrieval"
Cohesion: 0.25
Nodes (5): 6.1 `src/minimal_run.py sample`, 6.2 Index (`indexer.py`), 6.3 Retriever (`retriever.py`), 6.4 `src/retrieval/eval_retrieval.py`, 6. Data, aspects and retrieval

### Community 41 - "Pipeline — Run Order"
Cohesion: 0.29
Nodes (7): Pipeline — Run Order, Step 1 — Preprocess raw tickets, Step 2 — Generate silver labels (LLM-as-Judge), Step 3 — Build FAISS index, Step 3b — Train SFT aspect extraction model, Step 4 — Extract aspects for test set, Step 5 — Generate responses

### Community 43 - "predict_head_labels"
Cohesion: 0.33
Nodes (3): predict_head_labels(), embedding_text(), test_embedding_text_matches_what_the_indexer_embeds()

## Knowledge Gaps
- **45 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `Useful facts` (+40 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 259 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `File Descriptions` connect `File Descriptions` to `AspectForge — Multilingual Aspect-Aware Customer Support Response Generation`, `inference_retriever.py`, `.retrieve`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `postprocess_spans()` (e.g. with `10. Implementation order and acceptance tests (do in this order; commit after each)` and `A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU)`) actually correct?**
  _`postprocess_spans()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _45 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._
- **Why does `MetricsCalculator` connect `test_metrics.py` to `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)`, `response_generator_v2.py`, `baseline_t5.py`, `utils.py`, `GenerationConfig`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_metrics.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05612244897959184 - nodes in this community are weakly interconnected._