# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 43 files · ~72,564 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 1143 nodes · 2396 edges · 85 communities (56 shown, 29 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 78 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bb05d8a5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)
- pathlib
- test_metrics.py
- response_generator_v2.py
- aspect_extractor.py
- baseline_t5.py
- preprocessing.py
- inference_retriever.py
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
- run
- Checkpointer
- generate_batch
- pandas
- test_raft.py
- test_chat_fn_falls_back_to_string_content
- test_minimal_run.py
- TicketRetriever
- test_prompt.py
- test_eval_retrieval.py
- minimal_run.py
- CorpusIndexer
- evaluate_heads
- _FakeProcessor
- build_examples
- utils.py
- model_backend.py
- test_report.py
- .__init__
- extract_entities
- load_model
- fit_prompt
- build_aspect_only_prompt
- load_systems
- DataFrame
- MetricsCalculator
- test_a_non_raft_mode_ignores_a_configured_adapter
- test_a_sample_with_aspect_columns_is_not_re_extracted
- step3_raft.py
- Any
- sample_df
- step1_silver_labels.py
- aspect_score
- chat_text
- random_distractors
- test_eos_skips_an_unknown_end_of_turn_token
- _make_retriever
- metadata_lookup
- assert_no_evaluation_rows
- _system_frame
- test_unknown_trl_arguments_are_dropped_and_renamed
- _FakeTensor
- aspect_table
- get_content_sentences
- ProgressCallback
- A. Aspect v2: make the "Aspect" in AspectForge measurable and strong
- AspectForge v3 — Progress
- predict_head_labels
- extract_sentiment_severity
- rrf_fuse
- is_pleasantry
- Profiler
- fix_encoding
- _StubEmbedder
- test_the_embedder_uses_the_configured_prefixes
- test_sft_label_collapse_is_documented_from_the_sample
- test_count_tokens_does_not_add_special_tokens
- index_dir

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 40 edges
2. `MetricsCalculator` - 25 edges
3. `run_rag_aspect()` - 24 edges
4. `load_yaml_with_overrides()` - 24 edges
5. `postprocess_spans()` - 23 edges
6. `Checkpointer` - 23 edges
7. `_split()` - 22 edges
8. `run()` - 21 edges
9. `load_systems()` - 20 edges
10. `TicketRetriever` - 20 edges

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

## Communities (85 total, 29 thin omitted)

### Community 0 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.06
Nodes (27): 0. Why: problems found in the current code, 11. Risks and fallbacks, 1. What the minimal run produces, 2. Files: new and changed, 4. `config/kaggle.yaml` (overrides) and config merging, 6.1 `src/minimal_run.py sample`, 6.2 Index (`indexer.py`), 6.3 Retriever (`retriever.py`) (+19 more)

### Community 1 - "pathlib"
Cohesion: 0.19
Nodes (4): load_config(), main(), reorder_columns(), StratifiedSplitter

### Community 2 - "test_metrics.py"
Cohesion: 0.09
Nodes (8): _UnicodeTokenizer, calc(), MetCfg, test_cls_cosine_is_no_longer_called_bertscore(), test_lang_match_is_one_when_every_reply_uses_the_ticket_language(), test_rouge_scores_identical_german_as_one(), test_rouge_without_the_unicode_tokenizer_loses_german(), test_unicode_tokenizer_keeps_umlauts_and_lowercases()

### Community 4 - "response_generator_v2.py"
Cohesion: 0.13
Nodes (8): _apply_chat_template(), build_baseline_prompt(), build_baseline_prompt_pt(), generate_batch_extra(), _is_pretrained(), load_extra_model(), run_extra_baseline(), 5.2 `src/generation/model_backend.py`

### Community 5 - "aspect_extractor.py"
Cohesion: 0.15
Nodes (12): build_prob_statement(), encode_anchors(), get_gpu_profile(), get_model(), get_ranker(), load_config(), pick_best_span(), process_row() (+4 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.13
Nodes (12): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+4 more)

### Community 7 - "preprocessing.py"
Cohesion: 0.09
Nodes (4): DataPreprocessor, fix_encoding(), strip_email_filler(), track_run()

### Community 8 - "inference_retriever.py"
Cohesion: 0.06
Nodes (30): `aspect_extractor.py`, AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, `baseline_t5.py`, `config/generation_config.yaml`, `config/inference_retrieval_config.yaml`, Configuration, File Descriptions, Hardware (+22 more)

### Community 10 - "kaggle_job.py"
Cohesion: 0.11
Nodes (21): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules, log() (+13 more)

### Community 11 - "test_config.py"
Cohesion: 0.11
Nodes (18): apply_overrides(), load_generation_config(), deep_merge(), load_yaml_with_overrides(), test_aspect_extractor_apply_overrides_is_a_no_op_without_a_section(), test_aspect_extractor_apply_overrides_merges_its_section(), test_deep_merge_recurses_and_leaves_the_base_untouched(), test_deep_merge_replaces_lists_wholesale() (+10 more)

### Community 12 - "test_crag.py"
Cohesion: 0.07
Nodes (30): crag_route(), tune_crag_threshold(), write_crag_threshold(), _FakeRetriever, no_context_frame(), paired_rows(), reuse_s2_outputs_from(), _run() (+22 more)

### Community 13 - "test_aspects.py"
Cohesion: 0.09
Nodes (25): has_action_cue(), jaccard(), postprocess_spans(), build_aspect_checklist(), _stub_retriever(), test_a_cause_identical_to_the_problem_becomes_none(), test_a_distinct_cause_survives(), test_a_pleasantry_span_becomes_none_without_a_candidate() (+17 more)

### Community 21 - "indexer.py"
Cohesion: 0.16
Nodes (8): IndexerConfig, load_config(), _load_yaml(), main(), STEmbedder, test_records_without_aspects_still_carry_the_gold_columns(), test_the_indexer_refuses_an_index_containing_evaluation_rows(), test_the_indexer_stores_body_and_the_gold_columns()

### Community 23 - "step2_sft.py"
Cohesion: 0.12
Nodes (6): build_input_prompt(), build_target_output(), get_gpu_profile(), _push_checkpoint_to_hf(), run(), SilverDataset

### Community 25 - "test_retriever_fusion.py"
Cohesion: 0.15
Nodes (17): _retriever(), StubConfig, _StubCrossEncoder, test_aspect_query_languages_decides_which_queries_are_aspect_weighted(), test_dense_retrieval_returns_the_matching_document(), test_exclude_ids_accepts_non_string_ids(), test_exclude_ids_removes_the_given_ticket(), test_filter_by_language_keeps_only_the_query_language() (+9 more)

### Community 26 - "GenerationConfig"
Cohesion: 0.17
Nodes (18): GenerationConfig, resolve_data_path(), run_rag_aspect(), test_sample_csv_overrides_the_input_argument(), _FakeRetriever, _patch_runner(), test_a_raft_mode_without_an_adapter_is_refused(), test_a_resumed_run_skips_the_rows_already_done() (+10 more)

### Community 27 - "run"
Cohesion: 0.17
Nodes (8): _aspect_columns_ready(), compute_metrics(), enrich_dataframe_with_aspects(), maybe_merge_aspects_file(), print_metrics(), _resolve_aspect_extractor_kwargs(), run(), run_rag_only()

### Community 28 - "Checkpointer"
Cohesion: 0.20
Nodes (4): Checkpointer, test_checkpointer_defaults_to_the_current_directory(), test_checkpointer_round_trips_rows(), test_checkpointer_writes_into_the_given_directory()

### Community 29 - "generate_batch"
Cohesion: 0.16
Nodes (7): _aspect_row(), generate_batch(), generate_one(), _get_spans(), run_aspect_only(), run_baseline(), run_rag_asp_guided()

### Community 30 - "pandas"
Cohesion: 0.20
Nodes (13): align_targets(), _cmd_eval(), _cmd_predict(), _cmd_train(), embed_texts(), HeadsConfig, load_bundle(), load_dense() (+5 more)

### Community 31 - "test_raft.py"
Cohesion: 0.11
Nodes (19): drop_self_and_near_duplicates(), example_stats(), load_raft_config(), pick_sanity_examples(), split_examples(), test_a_hit_whose_response_is_the_gold_answer_is_dropped(), test_a_hit_without_a_dense_score_survives(), test_a_near_duplicate_hit_is_dropped() (+11 more)

### Community 33 - "test_minimal_run.py"
Cohesion: 0.09
Nodes (38): attach_aspects(), build_sample(), cmd_attach_aspects(), cmd_sample(), file_hash(), load_minimal_config(), MinimalRunConfig, sample_specs() (+30 more)

### Community 34 - "TicketRetriever"
Cohesion: 0.12
Nodes (6): `retriever.py`, aspect_query_vector(), cross_encoder_rerank(), get_cross_encoder(), _legacy_cross_encoder_cache(), TicketRetriever

### Community 35 - "test_prompt.py"
Cohesion: 0.19
Nodes (10): inject_sentiment_severity(), _stub_retriever(), test_a_german_ticket_gets_a_german_instruction_last(), test_an_unknown_backend_is_rejected(), test_chat_fn_returns_the_raw_prompt_for_a_pretrained_model(), test_ctx_doc_max_chars_caps_each_retrieved_response(), test_the_language_line_can_be_switched_off(), test_the_legacy_slot_prompt_also_gets_the_language_line() (+2 more)

### Community 36 - "test_eval_retrieval.py"
Cohesion: 0.08
Nodes (30): bootstrap_mean(), cosine(), evaluate_rows(), _hit_intent(), intent_at_k(), main(), rouge_l(), _RougeCfg (+22 more)

### Community 37 - "minimal_run.py"
Cohesion: 0.13
Nodes (14): bootstrap_table(), cmd_crag_tune(), cmd_report(), main(), _metrics_calculator(), _resample_indices(), _response_label(), results_chart() (+6 more)

### Community 39 - "evaluate_heads"
Cohesion: 0.20
Nodes (9): evaluate_heads(), predict_from_features(), score_head(), train_heads(), _separable_features(), test_heads_beat_the_majority_baseline_on_macro_f1(), test_heads_predict_labels_for_new_rows(), test_predict_rejects_embeddings_of_the_wrong_width() (+1 more)

### Community 40 - "_FakeProcessor"
Cohesion: 0.20
Nodes (5): _FakeProcessor, _FakeTextTokenizer, test_chat_fn_uses_typed_content_for_a_gemma_3_processor(), test_eos_includes_gemma_end_of_turn(), test_text_tokenizer_reaches_through_a_processor()

### Community 41 - "build_examples"
Cohesion: 0.21
Nodes (12): build_examples(), RaftConfig, _StubRetriever, test_build_asks_for_extra_candidates_to_survive_the_filter(), test_build_excludes_each_rows_own_ticket(), test_build_is_reproducible_for_a_seed(), test_build_skips_rows_without_a_gold_answer(), test_build_writes_one_example_per_row() (+4 more)

### Community 42 - "utils.py"
Cohesion: 0.17
Nodes (3): _aspect_value(), compare_models(), load_eval_config()

### Community 43 - "model_backend.py"
Cohesion: 0.16
Nodes (9): count_tokens_fn(), eos_token_ids(), generate(), load_generator(), _load_hf(), _load_unsloth(), make_chat_fn(), chat_fn() (+1 more)

### Community 44 - "test_report.py"
Cohesion: 0.16
Nodes (11): aspect_coverage(), split_sentences_for_acs(), _StubEmbedder, test_aas_only_looks_at_action_taken(), test_acs_averages_over_the_aspects_of_a_row(), test_acs_drops_when_the_reply_ignores_the_aspect(), test_acs_is_none_when_there_is_no_aspect_to_cover(), test_acs_is_one_when_a_sentence_states_the_aspect() (+3 more)

### Community 45 - ".__init__"
Cohesion: 0.18
Nodes (4): DataLoader, EvaluationPipeline, Profiler, _Result

### Community 46 - "extract_entities"
Cohesion: 0.33
Nodes (4): extract_entities(), test_a_word_repeated_in_the_body_is_kept(), test_product_patterns_still_win(), test_title_case_subject_words_no_longer_become_entities()

### Community 48 - "fit_prompt"
Cohesion: 0.23
Nodes (7): fit_prompt(), test_fit_prompt_accepts_no_retrieved_docs(), test_fit_prompt_drops_the_lowest_ranked_docs_until_it_fits(), test_fit_prompt_keeps_every_doc_when_the_budget_allows(), test_fit_prompt_returns_the_bare_prompt_when_nothing_fits(), test_the_real_prompt_stays_within_budget_with_five_long_hits(), _words()

### Community 49 - "build_aspect_only_prompt"
Cohesion: 0.20
Nodes (6): build_aspect_only_prompt(), reply_language_line(), test_aspect_only_prompt_uses_the_checklist_too(), test_language_line_falls_back_for_an_unknown_code(), test_language_line_names_the_ticket_language(), test_the_aspect_only_prompt_gets_the_language_line()

### Community 50 - "load_systems"
Cohesion: 0.16
Nodes (11): load_systems(), per_row_metrics(), _StubCalc, table(), test_a_missing_system_file_says_which(), test_a_system_compared_with_itself_has_no_difference(), test_per_row_metrics_can_skip_bertscore_and_the_aspect_scores(), test_results_markdown_has_every_section() (+3 more)

### Community 51 - "DataFrame"
Cohesion: 0.15
Nodes (11): aspect_contribution(), build_results_markdown(), qualitative_examples(), route_shares(), system_reply(), takeaway_lines(), test_examples_cover_one_en_and_two_de(), test_route_shares_are_empty_without_crag() (+3 more)

### Community 55 - "step3_raft.py"
Cohesion: 0.19
Nodes (12): cmd_build(), cmd_train(), filter_kwargs(), latest_checkpoint(), main(), read_jsonl(), sanity_check(), sft_config_kwargs() (+4 more)

### Community 56 - "Any"
Cohesion: 0.18
Nodes (6): _build_chat_text(), build_sft_prompt(), _decode_generated(), extract_labels_batch(), _next_candidate(), _parse_labels()

### Community 58 - "step1_silver_labels.py"
Cohesion: 0.27
Nodes (8): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch()

### Community 59 - "aspect_score"
Cohesion: 0.20
Nodes (6): aspect_score(), field(), test_aspect_score_ignores_none_values(), test_aspect_score_matches_predicted_heads_against_gold_metadata(), test_aspect_score_uses_unicode_tokens_for_overlap(), test_aspect_score_weights_intent_type_and_overlap()

### Community 60 - "chat_text"
Cohesion: 0.17
Nodes (7): chat_text(), to_dataset(), strip_leading_bos(), _StubTokenizer, test_a_leading_bos_is_stripped(), test_chat_text_contains_the_prompt_and_the_answer_without_a_second_bos(), test_chat_text_falls_back_to_typed_content()

### Community 61 - "random_distractors"
Cohesion: 0.29
Nodes (4): random_distractors(), test_distractors_fall_back_to_the_whole_pool(), test_distractors_of_an_empty_index_are_empty(), test_distractors_prefer_a_different_intent()

### Community 63 - "_make_retriever"
Cohesion: 0.18
Nodes (9): build_guided_rag_prompt(), _make_retriever(), run_rag_guided(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.3 Prompt budget (F3), 5.4 Language line (F2), 5.5 Sentiment/severity for DE (F8), 5.6 Runners and modes for S1–S5 (+1 more)

### Community 64 - "metadata_lookup"
Cohesion: 0.33
Nodes (4): metadata_lookup(), own_ticket_ids(), test_a_training_row_finds_its_own_ticket_id(), test_the_lookup_keeps_the_first_id_for_duplicate_text()

### Community 66 - "_system_frame"
Cohesion: 0.20
Nodes (6): output_dir(), _system_frame(), test_a_system_with_different_rows_is_refused(), test_duplicate_rows_are_refused(), test_route_shares_are_reported_when_crag_ran(), 10. Implementation order and acceptance tests (do in this order; commit after each)

### Community 68 - "_FakeTensor"
Cohesion: 0.28
Nodes (5): _FakeTensor, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1(), fake_score()

### Community 69 - "aspect_table"
Cohesion: 0.29
Nodes (6): aspect_table(), format_interval(), markdown_table(), test_the_aspect_table_lists_acs_and_aas(), test_the_aspect_table_says_when_it_has_no_scores(), test_the_table_is_formatted_as_mean_with_an_interval()

### Community 70 - "get_content_sentences"
Cohesion: 0.29
Nodes (5): get_content_sentences(), is_pure_greeting(), split_sentences(), strip_greeting_prefix(), A.4 Aspect-centric metrics (computed in `minimal_run report`, with bootstrap CIs like all other metrics)

### Community 72 - "A. Aspect v2: make the "Aspect" in AspectForge measurable and strong"
Cohesion: 0.29
Nodes (6): A.0 Evidence (from `docs/samples/aspects_SAMPLE.csv`, 11 test tickets), A.1 Supervised aspect heads (replace the SFT labels at inference; CPU, minutes), A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU), A.3 Aspects in retrieval and prompt, A.5 Ablation for the aspect claim, A. Aspect v2: make the "Aspect" in AspectForge measurable and strong

### Community 73 - "AspectForge v3 — Progress"
Cohesion: 0.33
Nodes (5): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup

### Community 74 - "predict_head_labels"
Cohesion: 0.33
Nodes (3): predict_head_labels(), embedding_text(), test_embedding_text_matches_what_the_indexer_embeds()

### Community 75 - "extract_sentiment_severity"
Cohesion: 0.33
Nodes (4): extract_sentiment_severity(), test_english_tickets_still_get_a_sentiment(), test_german_severity_keywords_are_recognised(), test_sentiment_is_not_invented_for_a_german_ticket()

### Community 76 - "rrf_fuse"
Cohesion: 0.33
Nodes (5): rrf_fuse(), test_rrf_k_damps_the_head_of_each_list(), test_rrf_of_nothing_is_empty(), test_rrf_rewards_a_document_both_lists_agree_on(), test_rrf_scores_one_list_by_reciprocal_rank()

### Community 77 - "is_pleasantry"
Cohesion: 0.50
Nodes (3): is_pleasantry(), test_pleasantries_are_recognised(), test_problem_sentences_are_not_pleasantries()

## Knowledge Gaps
- **48 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `Useful facts` (+43 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 438 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **29 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MetricsCalculator` connect `MetricsCalculator` to `test_metrics.py`, `response_generator_v2.py`, `minimal_run.py`, `baseline_t5.py`, `test_eval_retrieval.py`, `utils.py`, `.__init__`, `run`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _48 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` be split into smaller, more focused modules?**
  _Cohesion score 0.06451612903225806 - nodes in this community are weakly interconnected._
- **Why does `TicketRetriever` connect `TicketRetriever` to `test_prompt.py`, `response_generator_v2.py`, `test_eval_retrieval.py`, `inference_retriever.py`, `test_aspects.py`, `build_aspect_only_prompt`, `step3_raft.py`, `test_retriever_fusion.py`, `_make_retriever`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MetricsCalculator` (e.g. with `calc()` and `test_cls_cosine_is_no_longer_called_bertscore()`) actually correct?**
  _`MetricsCalculator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_metrics.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08666666666666667 - nodes in this community are weakly interconnected._