# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 42 files · ~68,815 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 1060 nodes · 2182 edges · 64 communities (45 shown, 19 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 78 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c8dc0e1d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)
- typing
- test_metrics.py
- build_aspect_only_prompt
- aspect_extractor.py
- baseline_t5.py
- preprocessing.py
- File Descriptions
- BaselineGenerator
- kaggle_job.py
- test_config.py
- test_crag.py
- test_aspects.py
- .claude/CLAUDE.md
- indexer.py
- step2_sft.py
- test_retriever_fusion.py
- GenerationConfig
- response_generator_v2.py
- Checkpointer
- generate_batch
- aspect_heads.py
- test_raft.py
- generate
- pandas
- TicketRetriever
- test_prompt.py
- test_eval_retrieval.py
- inference_retriever.py
- CorpusIndexer
- train_heads
- _FakeProcessor
- build_examples
- utils.py
- model_backend.py
- 6. Data, aspects and retrieval
- .__init__
- extract_entities
- get_hf_token
- fit_prompt
- reply_language_line
- Profiler
- STEmbedder
- test_a_non_raft_mode_ignores_a_configured_adapter
- test_a_sample_with_aspect_columns_is_not_re_extracted
- step3_raft.py
- sample_df
- chat_text
- random_distractors
- eos_token_ids
- _make_retriever
- metadata_lookup
- assert_no_evaluation_rows
- load_raft_config
- test_unknown_trl_arguments_are_dropped_and_renamed

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 40 edges
2. `run_rag_aspect()` - 24 edges
3. `MetricsCalculator` - 24 edges
4. `load_yaml_with_overrides()` - 24 edges
5. `postprocess_spans()` - 23 edges
6. `Checkpointer` - 23 edges
7. `_split()` - 22 edges
8. `run()` - 21 edges
9. `TicketRetriever` - 20 edges
10. `build_examples()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `K.2 Kaggle via API (no notebook clicking)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `3. Kaggle constraints the code must respect` --references--> `get_hf_token()`  [INFERRED]
  UPGRADE_PLAN.md → src/utils.py
- `GPU work = Kaggle via API (plan §K)` --references--> `pull()`  [INFERRED]
  CLAUDE.md → scripts/kaggle_job.py
- `10. Implementation order and acceptance tests (do in this order; commit after each)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py

## Import Cycles
- None detected.

## Communities (64 total, 19 thin omitted)

### Community 0 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.08
Nodes (24): 0. Why: problems found in the current code, 10. Implementation order and acceptance tests (do in this order; commit after each), 11. Risks and fallbacks, 1. What the minimal run produces, 2. Files: new and changed, 3. Kaggle constraints the code must respect, 4. `config/kaggle.yaml` (overrides) and config merging, 7.1 Config (+16 more)

### Community 1 - "typing"
Cohesion: 0.19
Nodes (4): load_config(), main(), reorder_columns(), StratifiedSplitter

### Community 2 - "test_metrics.py"
Cohesion: 0.06
Nodes (15): MetricsCalculator, _UnicodeTokenizer, calc(), _FakeTensor, MetCfg, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1() (+7 more)

### Community 4 - "build_aspect_only_prompt"
Cohesion: 0.16
Nodes (8): build_aspect_only_prompt(), build_baseline_prompt(), build_baseline_prompt_pt(), generate_batch_extra(), load_extra_model(), run_extra_baseline(), 5.2 `src/generation/model_backend.py`, 5.4 Language line (F2)

### Community 5 - "aspect_extractor.py"
Cohesion: 0.08
Nodes (24): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_labels_batch(), fix_encoding(), get_content_sentences() (+16 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.16
Nodes (9): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+1 more)

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
Cohesion: 0.08
Nodes (24): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup, apply_overrides(), load_generation_config(), deep_merge() (+16 more)

### Community 12 - "test_crag.py"
Cohesion: 0.07
Nodes (30): crag_route(), tune_crag_threshold(), write_crag_threshold(), _FakeRetriever, no_context_frame(), paired_rows(), reuse_s2_outputs_from(), _run() (+22 more)

### Community 13 - "test_aspects.py"
Cohesion: 0.08
Nodes (29): has_action_cue(), is_pleasantry(), jaccard(), postprocess_spans(), build_aspect_checklist(), _stub_retriever(), test_a_cause_identical_to_the_problem_becomes_none(), test_a_distinct_cause_survives() (+21 more)

### Community 21 - "indexer.py"
Cohesion: 0.24
Nodes (4): load_config(), _load_yaml(), main(), test_kaggle_yaml_has_every_section_the_plan_lists()

### Community 23 - "step2_sft.py"
Cohesion: 0.07
Nodes (15): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch() (+7 more)

### Community 25 - "test_retriever_fusion.py"
Cohesion: 0.05
Nodes (36): aspect_score(), rrf_fuse(), index_dir(), _retriever(), StubConfig, _StubCrossEncoder, _StubEmbedder, test_aspect_query_languages_decides_which_queries_are_aspect_weighted() (+28 more)

### Community 26 - "GenerationConfig"
Cohesion: 0.17
Nodes (18): GenerationConfig, resolve_data_path(), run_rag_aspect(), test_sample_csv_overrides_the_input_argument(), _FakeRetriever, _patch_runner(), test_a_raft_mode_without_an_adapter_is_refused(), test_a_resumed_run_skips_the_rows_already_done() (+10 more)

### Community 27 - "response_generator_v2.py"
Cohesion: 0.10
Nodes (13): load_config(), _apply_chat_template(), _aspect_columns_ready(), compute_metrics(), enrich_dataframe_with_aspects(), _is_pretrained(), maybe_merge_aspects_file(), print_metrics() (+5 more)

### Community 28 - "Checkpointer"
Cohesion: 0.20
Nodes (4): Checkpointer, test_checkpointer_defaults_to_the_current_directory(), test_checkpointer_round_trips_rows(), test_checkpointer_writes_into_the_given_directory()

### Community 29 - "generate_batch"
Cohesion: 0.16
Nodes (7): _aspect_row(), generate_batch(), generate_one(), _get_spans(), run_aspect_only(), run_baseline(), run_rag_asp_guided()

### Community 30 - "aspect_heads.py"
Cohesion: 0.12
Nodes (20): predict_head_labels(), align_targets(), _cmd_eval(), _cmd_predict(), _cmd_train(), embed_texts(), embedding_text(), evaluate_heads() (+12 more)

### Community 31 - "test_raft.py"
Cohesion: 0.15
Nodes (15): drop_self_and_near_duplicates(), example_stats(), pick_sanity_examples(), split_examples(), test_a_hit_whose_response_is_the_gold_answer_is_dropped(), test_a_hit_without_a_dense_score_survives(), test_a_near_duplicate_hit_is_dropped(), test_a_tiny_dataset_still_keeps_one_validation_example() (+7 more)

### Community 32 - "generate"
Cohesion: 0.22
Nodes (6): generate(), make_chat_fn(), chat_fn(), test_chat_fn_falls_back_to_string_content(), test_chat_fn_returns_the_raw_prompt_for_a_pretrained_model(), test_chat_fn_uses_typed_content_for_a_gemma_3_processor()

### Community 33 - "pandas"
Cohesion: 0.08
Nodes (41): attach_aspects(), build_sample(), cmd_attach_aspects(), cmd_crag_tune(), cmd_sample(), file_hash(), load_minimal_config(), main() (+33 more)

### Community 34 - "TicketRetriever"
Cohesion: 0.12
Nodes (7): `retriever.py`, aspect_query_vector(), cross_encoder_rerank(), get_cross_encoder(), _legacy_cross_encoder_cache(), STEmbedder, TicketRetriever

### Community 35 - "test_prompt.py"
Cohesion: 0.15
Nodes (14): extract_sentiment_severity(), inject_sentiment_severity(), _stub_retriever(), test_a_german_ticket_gets_a_german_instruction_last(), test_ctx_doc_max_chars_caps_each_retrieved_response(), test_english_tickets_still_get_a_sentiment(), test_german_severity_keywords_are_recognised(), test_sentiment_is_not_invented_for_a_german_ticket() (+6 more)

### Community 36 - "test_eval_retrieval.py"
Cohesion: 0.08
Nodes (30): bootstrap_mean(), cosine(), evaluate_rows(), _hit_intent(), intent_at_k(), main(), rouge_l(), _RougeCfg (+22 more)

### Community 37 - "inference_retriever.py"
Cohesion: 0.22
Nodes (7): `inference_retriever.py`, get_retriever(), load_retriever_config(), main(), RetrievalConfig, retrieve_similar(), run_file()

### Community 38 - "CorpusIndexer"
Cohesion: 0.14
Nodes (5): CorpusIndexer, IndexerConfig, test_records_without_aspects_still_carry_the_gold_columns(), test_the_indexer_refuses_an_index_containing_evaluation_rows(), test_the_indexer_stores_body_and_the_gold_columns()

### Community 39 - "train_heads"
Cohesion: 0.31
Nodes (6): train_heads(), _separable_features(), test_heads_beat_the_majority_baseline_on_macro_f1(), test_heads_predict_labels_for_new_rows(), test_predict_rejects_embeddings_of_the_wrong_width(), test_train_heads_needs_the_gold_column()

### Community 40 - "_FakeProcessor"
Cohesion: 0.25
Nodes (3): _FakeProcessor, _FakeTextTokenizer, test_text_tokenizer_reaches_through_a_processor()

### Community 41 - "build_examples"
Cohesion: 0.21
Nodes (12): build_examples(), RaftConfig, _StubRetriever, test_build_asks_for_extra_candidates_to_survive_the_filter(), test_build_excludes_each_rows_own_ticket(), test_build_is_reproducible_for_a_seed(), test_build_skips_rows_without_a_gold_answer(), test_build_writes_one_example_per_row() (+4 more)

### Community 42 - "utils.py"
Cohesion: 0.15
Nodes (4): _aspect_value(), compare_models(), load_csv_or_excel(), load_eval_config()

### Community 43 - "model_backend.py"
Cohesion: 0.19
Nodes (7): count_tokens_fn(), load_generator(), _load_hf(), _load_unsloth(), text_tokenizer(), test_an_unknown_backend_is_rejected(), test_count_tokens_does_not_add_special_tokens()

### Community 44 - "6. Data, aspects and retrieval"
Cohesion: 0.25
Nodes (5): 6.1 `src/minimal_run.py sample`, 6.2 Index (`indexer.py`), 6.3 Retriever (`retriever.py`), 6.4 `src/retrieval/eval_retrieval.py`, 6. Data, aspects and retrieval

### Community 46 - "extract_entities"
Cohesion: 0.33
Nodes (4): extract_entities(), test_a_word_repeated_in_the_body_is_kept(), test_product_patterns_still_win(), test_title_case_subject_words_no_longer_become_entities()

### Community 47 - "get_hf_token"
Cohesion: 0.33
Nodes (3): load_model(), get_hf_token(), test_load_model_rejects_a_seq2seq_model()

### Community 48 - "fit_prompt"
Cohesion: 0.23
Nodes (7): fit_prompt(), test_fit_prompt_accepts_no_retrieved_docs(), test_fit_prompt_drops_the_lowest_ranked_docs_until_it_fits(), test_fit_prompt_keeps_every_doc_when_the_budget_allows(), test_fit_prompt_returns_the_bare_prompt_when_nothing_fits(), test_the_real_prompt_stays_within_budget_with_five_long_hits(), _words()

### Community 49 - "reply_language_line"
Cohesion: 0.33
Nodes (3): reply_language_line(), test_language_line_falls_back_for_an_unknown_code(), test_language_line_names_the_ticket_language()

### Community 55 - "step3_raft.py"
Cohesion: 0.19
Nodes (12): cmd_build(), cmd_train(), filter_kwargs(), latest_checkpoint(), main(), read_jsonl(), sanity_check(), sft_config_kwargs() (+4 more)

### Community 60 - "chat_text"
Cohesion: 0.17
Nodes (7): chat_text(), to_dataset(), strip_leading_bos(), _StubTokenizer, test_a_leading_bos_is_stripped(), test_chat_text_contains_the_prompt_and_the_answer_without_a_second_bos(), test_chat_text_falls_back_to_typed_content()

### Community 61 - "random_distractors"
Cohesion: 0.29
Nodes (4): random_distractors(), test_distractors_fall_back_to_the_whole_pool(), test_distractors_of_an_empty_index_are_empty(), test_distractors_prefer_a_different_intent()

### Community 62 - "eos_token_ids"
Cohesion: 0.40
Nodes (3): eos_token_ids(), test_eos_includes_gemma_end_of_turn(), test_eos_skips_an_unknown_end_of_turn_token()

### Community 63 - "_make_retriever"
Cohesion: 0.18
Nodes (7): build_guided_rag_prompt(), _make_retriever(), run_rag_guided(), run_rag_only(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.3 Prompt budget (F3), 5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)

### Community 64 - "metadata_lookup"
Cohesion: 0.33
Nodes (4): metadata_lookup(), own_ticket_ids(), test_a_training_row_finds_its_own_ticket_id(), test_the_lookup_keeps_the_first_id_for_duplicate_text()

### Community 66 - "load_raft_config"
Cohesion: 0.50
Nodes (3): load_raft_config(), test_the_raft_config_keeps_relative_paths_locally(), test_the_raft_config_loads_and_the_kaggle_overrides_apply()

## Knowledge Gaps
- **46 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `Useful facts` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 407 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MetricsCalculator` connect `test_metrics.py` to `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)`, `pandas`, `test_eval_retrieval.py`, `baseline_t5.py`, `utils.py`, `.__init__`, `response_generator_v2.py`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` be split into smaller, more focused modules?**
  _Cohesion score 0.07692307692307693 - nodes in this community are weakly interconnected._
- **Why does `TicketRetriever` connect `TicketRetriever` to `test_prompt.py`, `test_eval_retrieval.py`, `inference_retriever.py`, `test_aspects.py`, `reply_language_line`, `step3_raft.py`, `test_retriever_fusion.py`, `response_generator_v2.py`, `_make_retriever`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MetricsCalculator` (e.g. with `calc()` and `test_cls_cosine_is_no_longer_called_bertscore()`) actually correct?**
  _`MetricsCalculator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_metrics.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05893719806763285 - nodes in this community are weakly interconnected._