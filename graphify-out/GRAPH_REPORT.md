# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 47 files · ~76,171 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 1197 nodes · 2559 edges · 84 communities (57 shown, 27 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 104 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8f0f16fb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- prep_train/run.py
- pathlib
- test_metrics.py
- 5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)
- aspect_extractor.py
- baseline_t5.py
- preprocessing.py
- AspectForge — Multilingual Aspect-Aware Customer Support Response Generation
- BaselineGenerator
- kaggle_job.py
- test_config.py
- test_crag.py
- test_aspects.py
- .claude/CLAUDE.md
- generate_eval/run.py
- step2_sft.py
- test_retriever_fusion.py
- GenerationConfig
- response_generator_v2.py
- Checkpointer
- test_report.py
- aspect_heads.py
- test_raft.py
- generate
- pandas
- TicketRetriever
- test_prompt.py
- test_eval_retrieval.py
- MetricsCalculator
- argparse
- evaluate_heads
- _FakeProcessor
- build_examples
- utils.py
- model_backend.py
- minimal_run.py
- load_raft_config
- A. Aspect v2: make the "Aspect" in AspectForge measurable and strong
- load_model
- fit_prompt
- reply_language_line
- per_row_metrics
- load_systems
- Any
- test_a_non_raft_mode_ignores_a_configured_adapter
- test_a_sample_with_aspect_columns_is_not_re_extracted
- step3_raft.py
- .__init__
- sample_df
- step1_silver_labels.py
- aspect_score
- chat_text
- random_distractors
- eos_token_ids
- generate_batch
- metadata_lookup
- assert_no_evaluation_rows
- _FakeTensor
- test_unknown_trl_arguments_are_dropped_and_renamed
- cmd_report
- get_content_sentences
- ProgressCallback
- table
- sft_label_collapse
- fix_encoding
- get_hf_token
- build_aspect_only_prompt
- rrf_fuse
- is_pleasantry
- Profiler
- calc
- _StubEmbedder
- test_the_embedder_uses_the_configured_prefixes
- test_cls_cosine_is_no_longer_called_bertscore
- index_dir

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 40 edges
2. `MetricsCalculator` - 25 edges
3. `postprocess_spans()` - 24 edges
4. `run_rag_aspect()` - 24 edges
5. `load_yaml_with_overrides()` - 24 edges
6. `Checkpointer` - 23 edges
7. `_split()` - 22 edges
8. `run()` - 21 edges
9. `load_systems()` - 20 edges
10. `TicketRetriever` - 20 edges

## Surprising Connections (you probably didn't know these)
- `Minimal run on Kaggle` --references--> `pull()`  [INFERRED]
  README.md → scripts/kaggle_job.py
- `K.2 Kaggle via API (no notebook clicking)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `Aspects` --references--> `postprocess_spans()`  [INFERRED]
  README.md → src/aspect/aspect_extractor.py
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `A.3 Aspects in retrieval and prompt` --references--> `aspect_score()`  [INFERRED]
  UPGRADE_PLAN.md → src/retrieval/retriever.py

## Import Cycles
- None detected.

## Communities (84 total, 27 thin omitted)

### Community 0 - "prep_train/run.py"
Cohesion: 0.09
Nodes (33): aspects(), checkout(), collect_outputs(), eval_retrieval(), find_raw_csv(), gpu_check(), index(), install() (+25 more)

### Community 1 - "pathlib"
Cohesion: 0.20
Nodes (4): load_config(), main(), reorder_columns(), StratifiedSplitter

### Community 2 - "test_metrics.py"
Cohesion: 0.11
Nodes (5): _UnicodeTokenizer, test_lang_match_is_one_when_every_reply_uses_the_ticket_language(), test_rouge_scores_identical_german_as_one(), test_rouge_without_the_unicode_tokenizer_loses_german(), test_unicode_tokenizer_keeps_umlauts_and_lowercases()

### Community 4 - "5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)"
Cohesion: 0.13
Nodes (10): build_baseline_prompt(), build_baseline_prompt_pt(), build_guided_rag_prompt(), generate_batch_extra(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.2 `src/generation/model_backend.py`, 5.3 Prompt budget (F3), 5.4 Language line (F2) (+2 more)

### Community 5 - "aspect_extractor.py"
Cohesion: 0.15
Nodes (12): build_prob_statement(), encode_anchors(), get_gpu_profile(), get_model(), get_ranker(), load_config(), pick_best_span(), process_row() (+4 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.15
Nodes (10): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+2 more)

### Community 7 - "preprocessing.py"
Cohesion: 0.09
Nodes (4): DataPreprocessor, fix_encoding(), strip_email_filler(), track_run()

### Community 8 - "AspectForge — Multilingual Aspect-Aware Customer Support Response Generation"
Cohesion: 0.17
Nodes (11): AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, Aspects, Configuration, Generation modes, Hardware, Minimal run on Kaggle, Models, Output files (+3 more)

### Community 10 - "kaggle_job.py"
Cohesion: 0.07
Nodes (30): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules, log() (+22 more)

### Community 11 - "test_config.py"
Cohesion: 0.06
Nodes (31): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup, apply_overrides(), load_generation_config(), get_retriever() (+23 more)

### Community 12 - "test_crag.py"
Cohesion: 0.07
Nodes (31): crag_route(), cmd_crag_tune(), tune_crag_threshold(), write_crag_threshold(), _FakeRetriever, no_context_frame(), paired_rows(), reuse_s2_outputs_from() (+23 more)

### Community 13 - "test_aspects.py"
Cohesion: 0.09
Nodes (24): extract_entities(), postprocess_spans(), build_aspect_checklist(), _stub_retriever(), test_a_cause_identical_to_the_problem_becomes_none(), test_a_distinct_cause_survives(), test_a_pleasantry_span_becomes_none_without_a_candidate(), test_a_pleasantry_span_is_replaced_by_the_next_ranked_sentence() (+16 more)

### Community 21 - "generate_eval/run.py"
Cohesion: 0.14
Nodes (25): checkout(), collect_outputs(), copy_job1_outputs(), crag_threshold(), generate(), gpu_check(), install(), log() (+17 more)

### Community 23 - "step2_sft.py"
Cohesion: 0.12
Nodes (6): build_input_prompt(), build_target_output(), get_gpu_profile(), _push_checkpoint_to_hf(), run(), SilverDataset

### Community 25 - "test_retriever_fusion.py"
Cohesion: 0.15
Nodes (17): _retriever(), StubConfig, _StubCrossEncoder, test_aspect_query_languages_decides_which_queries_are_aspect_weighted(), test_dense_retrieval_returns_the_matching_document(), test_exclude_ids_accepts_non_string_ids(), test_exclude_ids_removes_the_given_ticket(), test_filter_by_language_keeps_only_the_query_language() (+9 more)

### Community 26 - "GenerationConfig"
Cohesion: 0.17
Nodes (18): GenerationConfig, resolve_data_path(), run_rag_aspect(), test_sample_csv_overrides_the_input_argument(), _FakeRetriever, _patch_runner(), test_a_raft_mode_without_an_adapter_is_refused(), test_a_resumed_run_skips_the_rows_already_done() (+10 more)

### Community 27 - "response_generator_v2.py"
Cohesion: 0.10
Nodes (14): _apply_chat_template(), _aspect_columns_ready(), _aspect_row(), compute_metrics(), enrich_dataframe_with_aspects(), _get_spans(), _is_pretrained(), maybe_merge_aspects_file() (+6 more)

### Community 28 - "Checkpointer"
Cohesion: 0.20
Nodes (4): Checkpointer, test_checkpointer_defaults_to_the_current_directory(), test_checkpointer_round_trips_rows(), test_checkpointer_writes_into_the_given_directory()

### Community 29 - "test_report.py"
Cohesion: 0.16
Nodes (10): aspect_coverage(), split_sentences_for_acs(), _StubEmbedder, test_aas_only_looks_at_action_taken(), test_acs_averages_over_the_aspects_of_a_row(), test_acs_drops_when_the_reply_ignores_the_aspect(), test_acs_is_none_when_there_is_no_aspect_to_cover(), test_acs_is_one_when_a_sentence_states_the_aspect() (+2 more)

### Community 30 - "aspect_heads.py"
Cohesion: 0.17
Nodes (15): predict_head_labels(), align_targets(), _cmd_eval(), _cmd_predict(), _cmd_train(), embed_texts(), embedding_text(), HeadsConfig (+7 more)

### Community 31 - "test_raft.py"
Cohesion: 0.15
Nodes (15): drop_self_and_near_duplicates(), example_stats(), pick_sanity_examples(), split_examples(), test_a_hit_whose_response_is_the_gold_answer_is_dropped(), test_a_hit_without_a_dense_score_survives(), test_a_near_duplicate_hit_is_dropped(), test_a_tiny_dataset_still_keeps_one_validation_example() (+7 more)

### Community 32 - "generate"
Cohesion: 0.22
Nodes (6): generate(), make_chat_fn(), chat_fn(), test_chat_fn_falls_back_to_string_content(), test_chat_fn_returns_the_raw_prompt_for_a_pretrained_model(), test_chat_fn_uses_typed_content_for_a_gemma_3_processor()

### Community 33 - "pandas"
Cohesion: 0.08
Nodes (38): attach_aspects(), build_sample(), cmd_attach_aspects(), cmd_sample(), file_hash(), load_minimal_config(), main(), MinimalRunConfig (+30 more)

### Community 34 - "TicketRetriever"
Cohesion: 0.12
Nodes (5): aspect_query_vector(), cross_encoder_rerank(), get_cross_encoder(), _legacy_cross_encoder_cache(), TicketRetriever

### Community 35 - "test_prompt.py"
Cohesion: 0.26
Nodes (8): inject_sentiment_severity(), _stub_retriever(), test_a_german_ticket_gets_a_german_instruction_last(), test_ctx_doc_max_chars_caps_each_retrieved_response(), test_the_language_line_can_be_switched_off(), test_the_legacy_slot_prompt_also_gets_the_language_line(), test_the_sentiment_line_is_omitted_when_there_is_no_sentiment(), test_zero_ctx_doc_max_chars_keeps_the_full_response()

### Community 36 - "test_eval_retrieval.py"
Cohesion: 0.08
Nodes (30): bootstrap_mean(), cosine(), evaluate_rows(), _hit_intent(), intent_at_k(), main(), rouge_l(), _RougeCfg (+22 more)

### Community 37 - "MetricsCalculator"
Cohesion: 0.14
Nodes (5): Metrics, MetricsCalculator, 8.1 `MetricsCalculator` fixes, 8.2 `python -m src.minimal_run report --systems s1_gemma2 s2_gemma3 s3_gemma3_crag s4_raft_aspect s5_raft_noaspect`, 8. Metrics and report (`src/utils.py`, `src/minimal_run.py report`)

### Community 38 - "argparse"
Cohesion: 0.07
Nodes (14): CorpusIndexer, IndexerConfig, load_config(), _load_yaml(), main(), STEmbedder, test_records_without_aspects_still_carry_the_gold_columns(), test_the_indexer_refuses_an_index_containing_evaluation_rows() (+6 more)

### Community 39 - "evaluate_heads"
Cohesion: 0.20
Nodes (9): evaluate_heads(), predict_from_features(), score_head(), train_heads(), _separable_features(), test_heads_beat_the_majority_baseline_on_macro_f1(), test_heads_predict_labels_for_new_rows(), test_predict_rejects_embeddings_of_the_wrong_width() (+1 more)

### Community 40 - "_FakeProcessor"
Cohesion: 0.25
Nodes (3): _FakeProcessor, _FakeTextTokenizer, test_text_tokenizer_reaches_through_a_processor()

### Community 41 - "build_examples"
Cohesion: 0.21
Nodes (12): build_examples(), RaftConfig, _StubRetriever, test_build_asks_for_extra_candidates_to_survive_the_filter(), test_build_excludes_each_rows_own_ticket(), test_build_is_reproducible_for_a_seed(), test_build_skips_rows_without_a_gold_answer(), test_build_writes_one_example_per_row() (+4 more)

### Community 42 - "utils.py"
Cohesion: 0.15
Nodes (3): _aspect_value(), compare_models(), load_eval_config()

### Community 43 - "model_backend.py"
Cohesion: 0.17
Nodes (7): count_tokens_fn(), load_generator(), _load_hf(), _load_unsloth(), text_tokenizer(), test_an_unknown_backend_is_rejected(), test_count_tokens_does_not_add_special_tokens()

### Community 44 - "minimal_run.py"
Cohesion: 0.12
Nodes (17): aspect_contribution(), aspect_table(), build_results_markdown(), format_interval(), markdown_table(), qualitative_examples(), _rouge_l_rows(), _RougeCfg (+9 more)

### Community 45 - "load_raft_config"
Cohesion: 0.50
Nodes (3): load_raft_config(), test_the_raft_config_keeps_relative_paths_locally(), test_the_raft_config_loads_and_the_kaggle_overrides_apply()

### Community 46 - "A. Aspect v2: make the "Aspect" in AspectForge measurable and strong"
Cohesion: 0.29
Nodes (6): A.0 Evidence (from `docs/samples/aspects_SAMPLE.csv`, 11 test tickets), A.1 Supervised aspect heads (replace the SFT labels at inference; CPU, minutes), A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU), A.3 Aspects in retrieval and prompt, A.5 Ablation for the aspect claim, A. Aspect v2: make the "Aspect" in AspectForge measurable and strong

### Community 48 - "fit_prompt"
Cohesion: 0.23
Nodes (7): fit_prompt(), test_fit_prompt_accepts_no_retrieved_docs(), test_fit_prompt_drops_the_lowest_ranked_docs_until_it_fits(), test_fit_prompt_keeps_every_doc_when_the_budget_allows(), test_fit_prompt_returns_the_bare_prompt_when_nothing_fits(), test_the_real_prompt_stays_within_budget_with_five_long_hits(), _words()

### Community 49 - "reply_language_line"
Cohesion: 0.33
Nodes (3): reply_language_line(), test_language_line_falls_back_for_an_unknown_code(), test_language_line_names_the_ticket_language()

### Community 50 - "per_row_metrics"
Cohesion: 0.16
Nodes (9): bootstrap_table(), per_row_metrics(), _resample_indices(), _StubCalc, test_a_system_compared_with_itself_has_no_difference(), test_per_row_metrics_can_skip_bertscore_and_the_aspect_scores(), test_results_markdown_has_every_section(), test_the_bootstrap_is_reproducible() (+1 more)

### Community 51 - "load_systems"
Cohesion: 0.16
Nodes (10): load_systems(), route_shares(), _system_frame(), test_a_missing_system_file_says_which(), test_a_system_with_different_rows_is_refused(), test_duplicate_rows_are_refused(), test_per_row_metrics_cover_every_system_and_row(), test_route_shares_are_empty_without_crag() (+2 more)

### Community 52 - "Any"
Cohesion: 0.18
Nodes (6): _build_chat_text(), build_sft_prompt(), _decode_generated(), extract_labels_batch(), _next_candidate(), _parse_labels()

### Community 55 - "step3_raft.py"
Cohesion: 0.19
Nodes (12): cmd_build(), cmd_train(), filter_kwargs(), latest_checkpoint(), main(), read_jsonl(), sanity_check(), sft_config_kwargs() (+4 more)

### Community 56 - ".__init__"
Cohesion: 0.18
Nodes (4): DataLoader, EvaluationPipeline, Profiler, _Result

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

### Community 62 - "eos_token_ids"
Cohesion: 0.40
Nodes (3): eos_token_ids(), test_eos_includes_gemma_end_of_turn(), test_eos_skips_an_unknown_end_of_turn_token()

### Community 63 - "generate_batch"
Cohesion: 0.18
Nodes (6): generate_batch(), generate_one(), _make_retriever(), run_baseline(), run_rag_guided(), run_rag_only()

### Community 64 - "metadata_lookup"
Cohesion: 0.33
Nodes (4): metadata_lookup(), own_ticket_ids(), test_a_training_row_finds_its_own_ticket_id(), test_the_lookup_keeps_the_first_id_for_duplicate_text()

### Community 66 - "_FakeTensor"
Cohesion: 0.28
Nodes (5): _FakeTensor, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1(), fake_score()

### Community 68 - "cmd_report"
Cohesion: 0.25
Nodes (6): cmd_report(), _metrics_calculator(), results_chart(), STEmbedder, test_the_report_folds_in_the_retrieval_and_head_tables(), test_the_report_writes_every_artefact()

### Community 69 - "get_content_sentences"
Cohesion: 0.29
Nodes (5): get_content_sentences(), is_pure_greeting(), split_sentences(), strip_greeting_prefix(), A.4 Aspect-centric metrics (computed in `minimal_run report`, with bootstrap CIs like all other metrics)

### Community 71 - "table"
Cohesion: 0.33
Nodes (4): output_dir(), table(), 10. Implementation order and acceptance tests (do in this order; commit after each), 5.6 Runners and modes for S1–S5

### Community 75 - "build_aspect_only_prompt"
Cohesion: 0.20
Nodes (7): build_aspect_only_prompt(), extract_sentiment_severity(), test_aspect_only_prompt_uses_the_checklist_too(), test_english_tickets_still_get_a_sentiment(), test_german_severity_keywords_are_recognised(), test_sentiment_is_not_invented_for_a_german_ticket(), test_the_aspect_only_prompt_gets_the_language_line()

### Community 76 - "rrf_fuse"
Cohesion: 0.33
Nodes (5): rrf_fuse(), test_rrf_k_damps_the_head_of_each_list(), test_rrf_of_nothing_is_empty(), test_rrf_rewards_a_document_both_lists_agree_on(), test_rrf_scores_one_list_by_reciprocal_rank()

### Community 77 - "is_pleasantry"
Cohesion: 0.18
Nodes (8): has_action_cue(), is_pleasantry(), jaccard(), test_jaccard_measures_token_overlap_with_umlauts(), test_pleasantries_are_recognised(), test_problem_sentences_are_not_pleasantries(), test_process_row_cleans_the_spans_of_a_real_sample_row(), test_real_actions_keep_their_cue()

## Knowledge Gaps
- **32 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `Useful facts` (+27 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 441 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_yaml_with_overrides()` connect `test_config.py` to `pandas`, `argparse`, `table`, `utils.py`, `minimal_run.py`, `load_raft_config`, `step3_raft.py`, `response_generator_v2.py`, `aspect_heads.py`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _32 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `prep_train/run.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09191583610188261 - nodes in this community are weakly interconnected._
- **Why does `TicketRetriever` connect `TicketRetriever` to `test_prompt.py`, `test_eval_retrieval.py`, `test_config.py`, `test_aspects.py`, `reply_language_line`, `step3_raft.py`, `test_retriever_fusion.py`, `response_generator_v2.py`, `generate_batch`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MetricsCalculator` (e.g. with `calc()` and `test_cls_cosine_is_no_longer_called_bertscore()`) actually correct?**
  _`MetricsCalculator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_metrics.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._