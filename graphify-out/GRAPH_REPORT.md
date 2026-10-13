# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 47 files · ~75,483 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 1210 nodes · 2574 edges · 73 communities (47 shown, 26 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 106 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7ed8c415`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- generate_eval/run.py
- pathlib
- test_metrics.py
- run_extra_baseline
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
- argparse
- step2_sft.py
- test_retriever_fusion.py
- GenerationConfig
- response_generator_v2.py
- Checkpointer
- inference_retriever.py
- aspect_heads.py
- test_raft.py
- test_chat_fn_falls_back_to_string_content
- pandas
- TicketRetriever
- test_prompt.py
- test_eval_retrieval.py
- 6. Data, aspects and retrieval
- CorpusIndexer
- train_heads
- _FakeProcessor
- build_examples
- utils.py
- model_backend.py
- test_report.py
- load_raft_config
- extract_entities
- load_model
- fit_prompt
- build_aspect_only_prompt
- _apply_chat_template
- _is_pretrained
- test_a_non_raft_mode_ignores_a_configured_adapter
- test_a_sample_with_aspect_columns_is_not_re_extracted
- step3_raft.py
- sample_df
- aspect_score
- chat_text
- random_distractors
- test_eos_skips_an_unknown_end_of_turn_token
- _make_retriever
- metadata_lookup
- assert_no_evaluation_rows
- test_unknown_trl_arguments_are_dropped_and_renamed
- _StubCrossEncoder
- extract_sentiment_severity
- rrf_fuse
- is_pleasantry
- Profiler
- _StubEmbedder
- test_the_embedder_uses_the_configured_prefixes
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
- `3. Kaggle constraints the code must respect` --references--> `get_hf_token()`  [INFERRED]
  UPGRADE_PLAN.md → src/utils.py
- `§10 steps` --references--> `eval_retrieval()`  [INFERRED]
  PROGRESS.md → kaggle/prep_train/run.py

## Import Cycles
- None detected.

## Communities (73 total, 26 thin omitted)

### Community 0 - "generate_eval/run.py"
Cohesion: 0.05
Nodes (61): checkout(), collect_outputs(), copy_job1_outputs(), crag_threshold(), generate(), gpu_check(), install(), log() (+53 more)

### Community 1 - "pathlib"
Cohesion: 0.19
Nodes (4): load_config(), main(), reorder_columns(), StratifiedSplitter

### Community 2 - "test_metrics.py"
Cohesion: 0.05
Nodes (17): MetricsCalculator, _UnicodeTokenizer, calc(), _FakeTensor, MetCfg, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1() (+9 more)

### Community 4 - "run_extra_baseline"
Cohesion: 0.15
Nodes (9): build_baseline_prompt(), build_baseline_prompt_pt(), generate_batch_extra(), run_extra_baseline(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.2 `src/generation/model_backend.py`, 5.5 Sentiment/severity for DE (F8), 5.6 Runners and modes for S1–S5 (+1 more)

### Community 5 - "aspect_extractor.py"
Cohesion: 0.10
Nodes (20): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_labels_batch(), get_content_sentences(), get_gpu_profile() (+12 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.12
Nodes (12): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+4 more)

### Community 7 - "preprocessing.py"
Cohesion: 0.09
Nodes (4): DataPreprocessor, fix_encoding(), strip_email_filler(), track_run()

### Community 8 - "File Descriptions"
Cohesion: 0.08
Nodes (23): `aspect_extractor.py`, AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, `baseline_t5.py`, `config/generation_config.yaml`, `config/inference_retrieval_config.yaml`, Configuration, File Descriptions, Hardware (+15 more)

### Community 10 - "kaggle_job.py"
Cohesion: 0.08
Nodes (27): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules, kaggle_prefix() (+19 more)

### Community 11 - "test_config.py"
Cohesion: 0.08
Nodes (25): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup, apply_overrides(), load_generation_config(), deep_merge() (+17 more)

### Community 12 - "test_crag.py"
Cohesion: 0.07
Nodes (31): crag_route(), cmd_crag_tune(), tune_crag_threshold(), write_crag_threshold(), _FakeRetriever, no_context_frame(), paired_rows(), reuse_s2_outputs_from() (+23 more)

### Community 13 - "test_aspects.py"
Cohesion: 0.09
Nodes (26): fix_encoding(), has_action_cue(), jaccard(), postprocess_spans(), build_aspect_checklist(), _stub_retriever(), test_a_cause_identical_to_the_problem_becomes_none(), test_a_distinct_cause_survives() (+18 more)

### Community 21 - "argparse"
Cohesion: 0.33
Nodes (3): load_config(), _load_yaml(), main()

### Community 23 - "step2_sft.py"
Cohesion: 0.07
Nodes (15): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch() (+7 more)

### Community 25 - "test_retriever_fusion.py"
Cohesion: 0.20
Nodes (14): _retriever(), StubConfig, test_aspect_query_languages_decides_which_queries_are_aspect_weighted(), test_dense_retrieval_returns_the_matching_document(), test_exclude_ids_accepts_non_string_ids(), test_exclude_ids_removes_the_given_ticket(), test_filter_by_language_keeps_only_the_query_language(), test_hybrid_falls_back_to_dense_without_a_bm25_index() (+6 more)

### Community 26 - "GenerationConfig"
Cohesion: 0.17
Nodes (18): GenerationConfig, resolve_data_path(), run_rag_aspect(), test_sample_csv_overrides_the_input_argument(), _FakeRetriever, _patch_runner(), test_a_raft_mode_without_an_adapter_is_refused(), test_a_resumed_run_skips_the_rows_already_done() (+10 more)

### Community 27 - "response_generator_v2.py"
Cohesion: 0.11
Nodes (15): _aspect_columns_ready(), _aspect_row(), compute_metrics(), enrich_dataframe_with_aspects(), generate_batch(), generate_one(), _get_spans(), maybe_merge_aspects_file() (+7 more)

### Community 28 - "Checkpointer"
Cohesion: 0.20
Nodes (4): Checkpointer, test_checkpointer_defaults_to_the_current_directory(), test_checkpointer_round_trips_rows(), test_checkpointer_writes_into_the_given_directory()

### Community 29 - "inference_retriever.py"
Cohesion: 0.23
Nodes (5): `inference_retriever.py`, get_retriever(), main(), retrieve_similar(), run_file()

### Community 30 - "aspect_heads.py"
Cohesion: 0.11
Nodes (20): predict_head_labels(), align_targets(), _cmd_eval(), _cmd_predict(), _cmd_train(), embed_texts(), embedding_text(), evaluate_heads() (+12 more)

### Community 31 - "test_raft.py"
Cohesion: 0.15
Nodes (15): drop_self_and_near_duplicates(), example_stats(), pick_sanity_examples(), split_examples(), test_a_hit_whose_response_is_the_gold_answer_is_dropped(), test_a_hit_without_a_dense_score_survives(), test_a_near_duplicate_hit_is_dropped(), test_a_tiny_dataset_still_keeps_one_validation_example() (+7 more)

### Community 33 - "pandas"
Cohesion: 0.08
Nodes (38): attach_aspects(), build_sample(), cmd_attach_aspects(), cmd_sample(), file_hash(), load_minimal_config(), main(), MinimalRunConfig (+30 more)

### Community 34 - "TicketRetriever"
Cohesion: 0.12
Nodes (6): `retriever.py`, aspect_query_vector(), cross_encoder_rerank(), get_cross_encoder(), _legacy_cross_encoder_cache(), TicketRetriever

### Community 35 - "test_prompt.py"
Cohesion: 0.19
Nodes (10): inject_sentiment_severity(), _stub_retriever(), test_a_german_ticket_gets_a_german_instruction_last(), test_an_unknown_backend_is_rejected(), test_chat_fn_returns_the_raw_prompt_for_a_pretrained_model(), test_ctx_doc_max_chars_caps_each_retrieved_response(), test_the_language_line_can_be_switched_off(), test_the_legacy_slot_prompt_also_gets_the_language_line() (+2 more)

### Community 36 - "test_eval_retrieval.py"
Cohesion: 0.07
Nodes (31): bootstrap_mean(), cosine(), evaluate_rows(), _hit_intent(), intent_at_k(), main(), rouge_l(), _RougeCfg (+23 more)

### Community 37 - "6. Data, aspects and retrieval"
Cohesion: 0.20
Nodes (7): RetrievalConfig, 5.3 Prompt budget (F3), 6.1 `src/minimal_run.py sample`, 6.2 Index (`indexer.py`), 6.3 Retriever (`retriever.py`), 6.4 `src/retrieval/eval_retrieval.py`, 6. Data, aspects and retrieval

### Community 38 - "CorpusIndexer"
Cohesion: 0.12
Nodes (6): CorpusIndexer, IndexerConfig, STEmbedder, test_records_without_aspects_still_carry_the_gold_columns(), test_the_indexer_refuses_an_index_containing_evaluation_rows(), test_the_indexer_stores_body_and_the_gold_columns()

### Community 39 - "train_heads"
Cohesion: 0.27
Nodes (7): predict_from_features(), train_heads(), _separable_features(), test_heads_beat_the_majority_baseline_on_macro_f1(), test_heads_predict_labels_for_new_rows(), test_predict_rejects_embeddings_of_the_wrong_width(), test_train_heads_needs_the_gold_column()

### Community 40 - "_FakeProcessor"
Cohesion: 0.20
Nodes (5): _FakeProcessor, _FakeTextTokenizer, test_chat_fn_uses_typed_content_for_a_gemma_3_processor(), test_eos_includes_gemma_end_of_turn(), test_text_tokenizer_reaches_through_a_processor()

### Community 41 - "build_examples"
Cohesion: 0.21
Nodes (12): build_examples(), RaftConfig, _StubRetriever, test_build_asks_for_extra_candidates_to_survive_the_filter(), test_build_excludes_each_rows_own_ticket(), test_build_is_reproducible_for_a_seed(), test_build_skips_rows_without_a_gold_answer(), test_build_writes_one_example_per_row() (+4 more)

### Community 42 - "utils.py"
Cohesion: 0.09
Nodes (7): _aspect_value(), compare_models(), DataLoader, EvaluationPipeline, load_eval_config(), Profiler, _Result

### Community 43 - "model_backend.py"
Cohesion: 0.16
Nodes (9): count_tokens_fn(), eos_token_ids(), generate(), load_generator(), _load_hf(), _load_unsloth(), make_chat_fn(), chat_fn() (+1 more)

### Community 44 - "test_report.py"
Cohesion: 0.05
Nodes (55): aspect_contribution(), aspect_coverage(), aspect_table(), bootstrap_table(), build_results_markdown(), cmd_report(), format_interval(), load_systems() (+47 more)

### Community 45 - "load_raft_config"
Cohesion: 0.50
Nodes (3): load_raft_config(), test_the_raft_config_keeps_relative_paths_locally(), test_the_raft_config_loads_and_the_kaggle_overrides_apply()

### Community 46 - "extract_entities"
Cohesion: 0.14
Nodes (11): extract_entities(), split_sentences(), test_a_word_repeated_in_the_body_is_kept(), test_product_patterns_still_win(), test_title_case_subject_words_no_longer_become_entities(), A.0 Evidence (from `docs/samples/aspects_SAMPLE.csv`, 11 test tickets), A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU), A.3 Aspects in retrieval and prompt (+3 more)

### Community 48 - "fit_prompt"
Cohesion: 0.23
Nodes (7): fit_prompt(), test_fit_prompt_accepts_no_retrieved_docs(), test_fit_prompt_drops_the_lowest_ranked_docs_until_it_fits(), test_fit_prompt_keeps_every_doc_when_the_budget_allows(), test_fit_prompt_returns_the_bare_prompt_when_nothing_fits(), test_the_real_prompt_stays_within_budget_with_five_long_hits(), _words()

### Community 49 - "build_aspect_only_prompt"
Cohesion: 0.20
Nodes (6): build_aspect_only_prompt(), reply_language_line(), test_aspect_only_prompt_uses_the_checklist_too(), test_language_line_falls_back_for_an_unknown_code(), test_language_line_names_the_ticket_language(), test_the_aspect_only_prompt_gets_the_language_line()

### Community 55 - "step3_raft.py"
Cohesion: 0.19
Nodes (12): cmd_build(), cmd_train(), filter_kwargs(), latest_checkpoint(), main(), read_jsonl(), sanity_check(), sft_config_kwargs() (+4 more)

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
Cohesion: 0.29
Nodes (4): build_guided_rag_prompt(), _make_retriever(), run_rag_guided(), 5.4 Language line (F2)

### Community 64 - "metadata_lookup"
Cohesion: 0.33
Nodes (4): metadata_lookup(), own_ticket_ids(), test_a_training_row_finds_its_own_ticket_id(), test_the_lookup_keeps_the_first_id_for_duplicate_text()

### Community 72 - "_StubCrossEncoder"
Cohesion: 0.22
Nodes (4): _StubCrossEncoder, test_german_rows_are_reranked_too(), test_the_aspect_bonus_is_added_on_top_of_the_rerank_score(), A.1 Supervised aspect heads (replace the SFT labels at inference; CPU, minutes)

### Community 75 - "extract_sentiment_severity"
Cohesion: 0.33
Nodes (4): extract_sentiment_severity(), test_english_tickets_still_get_a_sentiment(), test_german_severity_keywords_are_recognised(), test_sentiment_is_not_invented_for_a_german_ticket()

### Community 76 - "rrf_fuse"
Cohesion: 0.33
Nodes (5): rrf_fuse(), test_rrf_k_damps_the_head_of_each_list(), test_rrf_of_nothing_is_empty(), test_rrf_rewards_a_document_both_lists_agree_on(), test_rrf_scores_one_list_by_reciprocal_rank()

### Community 77 - "is_pleasantry"
Cohesion: 0.33
Nodes (4): is_pleasantry(), _next_candidate(), test_pleasantries_are_recognised(), test_problem_sentences_are_not_pleasantries()

## Knowledge Gaps
- **43 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `Useful facts` (+38 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 452 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_yaml_with_overrides()` connect `test_config.py` to `pandas`, `test_eval_retrieval.py`, `utils.py`, `test_report.py`, `load_raft_config`, `argparse`, `step3_raft.py`, `response_generator_v2.py`, `inference_retriever.py`, `aspect_heads.py`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _43 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `generate_eval/run.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05134825014343087 - nodes in this community are weakly interconnected._
- **Why does `TicketRetriever` connect `TicketRetriever` to `test_prompt.py`, `test_eval_retrieval.py`, `test_aspects.py`, `build_aspect_only_prompt`, `step3_raft.py`, `test_retriever_fusion.py`, `response_generator_v2.py`, `inference_retriever.py`, `_make_retriever`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MetricsCalculator` (e.g. with `calc()` and `test_cls_cosine_is_no_longer_called_bertscore()`) actually correct?**
  _`MetricsCalculator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_metrics.py` be split into smaller, more focused modules?**
  _Cohesion score 0.052244897959183675 - nodes in this community are weakly interconnected._