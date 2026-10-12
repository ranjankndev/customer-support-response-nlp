# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 37 files · ~61,376 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 863 nodes · 1736 edges · 58 communities (33 shown, 25 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 67 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e1a19c0d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)
- inference_retriever.py
- test_metrics.py
- run_extra_baseline
- aspect_extractor.py
- baseline_t5.py
- preprocessing.py
- File Descriptions
- BaselineGenerator
- kaggle_job.py
- test_config.py
- Profiler
- test_aspects.py
- .claude/CLAUDE.md
- indexer.py
- step2_sft.py
- test_retriever_fusion.py
- GenerationConfig
- run
- Checkpointer
- response_generator_v2.py
- aspect_heads.py
- extract_entities
- model_backend.py
- test_minimal_run.py
- retriever.py
- test_prompt.py
- reply_language_line
- is_pleasantry
- CorpusIndexer
- build_aspect_only_prompt
- _FakeProcessor
- evaluate_heads
- utils.py
- count_tokens_fn
- fix_encoding
- .__init__
- aspect_score
- load_model
- rrf_fuse
- _StubEmbedder
- test_the_embedder_uses_the_configured_prefixes
- build_prompt
- compare_models
- test_a_non_raft_mode_ignores_a_configured_adapter
- test_a_sample_with_aspect_columns_is_not_re_extracted
- test_sft_label_collapse_is_documented_from_the_sample
- index_dir
- sample_df

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 40 edges
2. `postprocess_spans()` - 23 edges
3. `run_rag_aspect()` - 23 edges
4. `Checkpointer` - 23 edges
5. `load_yaml_with_overrides()` - 22 edges
6. `_split()` - 22 edges
7. `run()` - 21 edges
8. `MetricsCalculator` - 20 edges
9. `_retriever()` - 17 edges
10. `TicketRetriever` - 16 edges

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

## Communities (58 total, 25 thin omitted)

### Community 0 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.07
Nodes (27): 0. Why: problems found in the current code, 10. Implementation order and acceptance tests (do in this order; commit after each), 11. Risks and fallbacks, 1. What the minimal run produces, 2. Files: new and changed, 4. `config/kaggle.yaml` (overrides) and config merging, 7.1 Config, 7.2 `build` (+19 more)

### Community 1 - "inference_retriever.py"
Cohesion: 0.11
Nodes (13): `inference_retriever.py`, get_retriever(), load_retriever_config(), main(), RetrievalConfig, retrieve_similar(), run_file(), 5.3 Prompt budget (F3) (+5 more)

### Community 2 - "test_metrics.py"
Cohesion: 0.05
Nodes (19): MetricsCalculator, _UnicodeTokenizer, calc(), _FakeTensor, MetCfg, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1() (+11 more)

### Community 4 - "run_extra_baseline"
Cohesion: 0.20
Nodes (8): generate_batch_extra(), load_extra_model(), run_extra_baseline(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.2 `src/generation/model_backend.py`, 5.5 Sentiment/severity for DE (F8), 5.6 Runners and modes for S1–S5, 5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)

### Community 5 - "aspect_extractor.py"
Cohesion: 0.08
Nodes (23): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_labels_batch(), get_content_sentences(), get_gpu_profile() (+15 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.18
Nodes (9): compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base(), T5Config (+1 more)

### Community 7 - "preprocessing.py"
Cohesion: 0.09
Nodes (4): DataPreprocessor, fix_encoding(), strip_email_filler(), track_run()

### Community 8 - "File Descriptions"
Cohesion: 0.08
Nodes (23): `aspect_extractor.py`, AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, `baseline_t5.py`, `config/generation_config.yaml`, `config/inference_retrieval_config.yaml`, Configuration, File Descriptions, Hardware (+15 more)

### Community 10 - "kaggle_job.py"
Cohesion: 0.10
Nodes (23): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules, log() (+15 more)

### Community 11 - "test_config.py"
Cohesion: 0.06
Nodes (27): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup, apply_overrides(), load_generation_config(), load_config() (+19 more)

### Community 13 - "test_aspects.py"
Cohesion: 0.09
Nodes (25): has_action_cue(), jaccard(), postprocess_spans(), build_aspect_checklist(), _stub_retriever(), test_a_cause_identical_to_the_problem_becomes_none(), test_a_distinct_cause_survives(), test_a_pleasantry_span_becomes_none_without_a_candidate() (+17 more)

### Community 21 - "indexer.py"
Cohesion: 0.15
Nodes (8): IndexerConfig, load_config(), _load_yaml(), main(), STEmbedder, test_records_without_aspects_still_carry_the_gold_columns(), test_the_indexer_refuses_an_index_containing_evaluation_rows(), test_the_indexer_stores_body_and_the_gold_columns()

### Community 23 - "step2_sft.py"
Cohesion: 0.07
Nodes (15): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch() (+7 more)

### Community 25 - "test_retriever_fusion.py"
Cohesion: 0.15
Nodes (17): _retriever(), StubConfig, _StubCrossEncoder, test_aspect_query_languages_decides_which_queries_are_aspect_weighted(), test_dense_retrieval_returns_the_matching_document(), test_exclude_ids_accepts_non_string_ids(), test_exclude_ids_removes_the_given_ticket(), test_filter_by_language_keeps_only_the_query_language() (+9 more)

### Community 26 - "GenerationConfig"
Cohesion: 0.18
Nodes (18): GenerationConfig, resolve_data_path(), run_rag_aspect(), test_sample_csv_overrides_the_input_argument(), _FakeRetriever, _patch_runner(), test_a_raft_mode_without_an_adapter_is_refused(), test_a_resumed_run_skips_the_rows_already_done() (+10 more)

### Community 27 - "run"
Cohesion: 0.17
Nodes (8): _aspect_columns_ready(), compute_metrics(), enrich_dataframe_with_aspects(), maybe_merge_aspects_file(), print_metrics(), _resolve_aspect_extractor_kwargs(), run(), run_rag_only()

### Community 28 - "Checkpointer"
Cohesion: 0.20
Nodes (4): Checkpointer, test_checkpointer_defaults_to_the_current_directory(), test_checkpointer_round_trips_rows(), test_checkpointer_writes_into_the_given_directory()

### Community 29 - "response_generator_v2.py"
Cohesion: 0.09
Nodes (15): _apply_chat_template(), _aspect_row(), build_baseline_prompt(), build_baseline_prompt_pt(), build_guided_rag_prompt(), generate_batch(), generate_one(), _get_spans() (+7 more)

### Community 30 - "aspect_heads.py"
Cohesion: 0.15
Nodes (16): predict_head_labels(), align_targets(), _cmd_eval(), _cmd_predict(), _cmd_train(), embed_texts(), embedding_text(), HeadsConfig (+8 more)

### Community 31 - "extract_entities"
Cohesion: 0.33
Nodes (4): extract_entities(), test_a_word_repeated_in_the_body_is_kept(), test_product_patterns_still_win(), test_title_case_subject_words_no_longer_become_entities()

### Community 32 - "model_backend.py"
Cohesion: 0.12
Nodes (13): eos_token_ids(), generate(), load_generator(), _load_hf(), _load_unsloth(), make_chat_fn(), chat_fn(), text_tokenizer() (+5 more)

### Community 33 - "test_minimal_run.py"
Cohesion: 0.10
Nodes (35): attach_aspects(), build_sample(), cmd_attach_aspects(), cmd_sample(), file_hash(), load_minimal_config(), main(), MinimalRunConfig (+27 more)

### Community 34 - "retriever.py"
Cohesion: 0.12
Nodes (7): `retriever.py`, aspect_query_vector(), cross_encoder_rerank(), get_cross_encoder(), _legacy_cross_encoder_cache(), STEmbedder, TicketRetriever

### Community 35 - "test_prompt.py"
Cohesion: 0.14
Nodes (15): inject_sentiment_severity(), fit_prompt(), _stub_retriever(), test_a_german_ticket_gets_a_german_instruction_last(), test_ctx_doc_max_chars_caps_each_retrieved_response(), test_fit_prompt_accepts_no_retrieved_docs(), test_fit_prompt_drops_the_lowest_ranked_docs_until_it_fits(), test_fit_prompt_keeps_every_doc_when_the_budget_allows() (+7 more)

### Community 36 - "reply_language_line"
Cohesion: 0.33
Nodes (3): reply_language_line(), test_language_line_falls_back_for_an_unknown_code(), test_language_line_names_the_ticket_language()

### Community 37 - "is_pleasantry"
Cohesion: 0.50
Nodes (3): is_pleasantry(), test_pleasantries_are_recognised(), test_problem_sentences_are_not_pleasantries()

### Community 39 - "build_aspect_only_prompt"
Cohesion: 0.20
Nodes (7): build_aspect_only_prompt(), extract_sentiment_severity(), test_aspect_only_prompt_uses_the_checklist_too(), test_english_tickets_still_get_a_sentiment(), test_german_severity_keywords_are_recognised(), test_sentiment_is_not_invented_for_a_german_ticket(), test_the_aspect_only_prompt_gets_the_language_line()

### Community 40 - "_FakeProcessor"
Cohesion: 0.22
Nodes (4): _FakeProcessor, _FakeTextTokenizer, test_chat_fn_uses_typed_content_for_a_gemma_3_processor(), test_text_tokenizer_reaches_through_a_processor()

### Community 41 - "evaluate_heads"
Cohesion: 0.20
Nodes (9): evaluate_heads(), predict_from_features(), score_head(), train_heads(), _separable_features(), test_heads_beat_the_majority_baseline_on_macro_f1(), test_heads_predict_labels_for_new_rows(), test_predict_rejects_embeddings_of_the_wrong_width() (+1 more)

### Community 42 - "utils.py"
Cohesion: 0.17
Nodes (4): _aspect_value(), get_hf_token(), load_eval_config(), 3. Kaggle constraints the code must respect

### Community 45 - ".__init__"
Cohesion: 0.18
Nodes (4): DataLoader, EvaluationPipeline, Profiler, _Result

### Community 46 - "aspect_score"
Cohesion: 0.20
Nodes (5): aspect_score(), test_aspect_score_ignores_none_values(), test_aspect_score_matches_predicted_heads_against_gold_metadata(), test_aspect_score_uses_unicode_tokens_for_overlap(), test_aspect_score_weights_intent_type_and_overlap()

### Community 48 - "rrf_fuse"
Cohesion: 0.33
Nodes (5): rrf_fuse(), test_rrf_k_damps_the_head_of_each_list(), test_rrf_of_nothing_is_empty(), test_rrf_rewards_a_document_both_lists_agree_on(), test_rrf_scores_one_list_by_reciprocal_rank()

## Knowledge Gaps
- **46 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `Useful facts` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 346 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_yaml_with_overrides()` connect `test_config.py` to `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)`, `test_minimal_run.py`, `inference_retriever.py`, `utils.py`, `indexer.py`, `response_generator_v2.py`, `aspect_heads.py`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` be split into smaller, more focused modules?**
  _Cohesion score 0.06666666666666667 - nodes in this community are weakly interconnected._
- **Why does `File Descriptions` connect `File Descriptions` to `inference_retriever.py`, `retriever.py`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `postprocess_spans()` (e.g. with `10. Implementation order and acceptance tests (do in this order; commit after each)` and `A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU)`) actually correct?**
  _`postprocess_spans()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `inference_retriever.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10869565217391304 - nodes in this community are weakly interconnected._