# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 34 files · ~55,811 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 741 nodes · 1455 edges · 52 communities (35 shown, 17 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 62 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e5f43d88`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)
- retriever.py
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
- run_rag_asp_guided
- GenerationConfig
- generate_batch
- Checkpointer
- response_generator_v2.py
- aspect_heads.py
- extract_entities
- model_backend.py
- utils.py
- .__init__
- fit_prompt
- test_prompt.py
- is_pleasantry
- generate
- build_aspect_only_prompt
- _FakeProcessor
- typing
- jaccard
- eos_token_ids
- fix_encoding
- get_content_sentences
- AspectForge v3 — Progress
- load_model
- deep_merge
- test_a_non_raft_mode_ignores_a_configured_adapter
- test_a_sample_with_aspect_columns_is_not_re_extracted
- sample_df

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 40 edges
2. `postprocess_spans()` - 23 edges
3. `run_rag_aspect()` - 23 edges
4. `Checkpointer` - 23 edges
5. `run()` - 21 edges
6. `MetricsCalculator` - 20 edges
7. `load_yaml_with_overrides()` - 18 edges
8. `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` - 15 edges
9. `DataPreprocessor` - 14 edges
10. `_FakeRetriever` - 14 edges

## Surprising Connections (you probably didn't know these)
- `K.2 Kaggle via API (no notebook clicking)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `A.3 Aspects in retrieval and prompt` --references--> `aspect_score()`  [INFERRED]
  UPGRADE_PLAN.md → src/retrieval/retriever.py
- `§10 steps` --references--> `load_yaml_with_overrides()`  [INFERRED]
  PROGRESS.md → src/utils.py
- `GPU work = Kaggle via API (plan §K)` --references--> `pull()`  [INFERRED]
  CLAUDE.md → scripts/kaggle_job.py

## Import Cycles
- None detected.

## Communities (52 total, 17 thin omitted)

### Community 0 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.08
Nodes (23): 0. Why: problems found in the current code, 10. Implementation order and acceptance tests (do in this order; commit after each), 11. Risks and fallbacks, 1. What the minimal run produces, 2. Files: new and changed, 4. `config/kaggle.yaml` (overrides) and config merging, 7.1 Config, 7.2 `build` (+15 more)

### Community 1 - "retriever.py"
Cohesion: 0.07
Nodes (20): `inference_retriever.py`, `retriever.py`, get_retriever(), load_retriever_config(), main(), RetrievalConfig, retrieve_similar(), run_file() (+12 more)

### Community 2 - "test_metrics.py"
Cohesion: 0.06
Nodes (15): MetricsCalculator, _UnicodeTokenizer, calc(), _FakeTensor, MetCfg, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1() (+7 more)

### Community 4 - "run_extra_baseline"
Cohesion: 0.11
Nodes (13): build_baseline_prompt(), build_baseline_prompt_pt(), build_guided_rag_prompt(), generate_batch_extra(), load_extra_model(), run_baseline(), run_extra_baseline(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`) (+5 more)

### Community 5 - "aspect_extractor.py"
Cohesion: 0.11
Nodes (16): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_labels_batch(), get_gpu_profile(), get_model() (+8 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.13
Nodes (11): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+3 more)

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
Cohesion: 0.12
Nodes (16): apply_overrides(), load_generation_config(), load_yaml_with_overrides(), test_aspect_extractor_apply_overrides_is_a_no_op_without_a_section(), test_aspect_extractor_apply_overrides_merges_its_section(), test_empty_overrides_argument_is_a_no_op(), test_existing_configs_still_load_without_overrides(), test_generation_config_defaults_match_the_plan() (+8 more)

### Community 13 - "test_aspects.py"
Cohesion: 0.10
Nodes (22): postprocess_spans(), build_aspect_checklist(), _stub_retriever(), test_a_cause_identical_to_the_problem_becomes_none(), test_a_distinct_cause_survives(), test_a_pleasantry_span_becomes_none_without_a_candidate(), test_a_pleasantry_span_is_replaced_by_the_next_ranked_sentence(), test_an_action_duplicating_the_cause_becomes_none() (+14 more)

### Community 21 - "indexer.py"
Cohesion: 0.13
Nodes (6): CorpusIndexer, IndexerConfig, load_config(), _load_yaml(), main(), STEmbedder

### Community 23 - "step2_sft.py"
Cohesion: 0.07
Nodes (15): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch() (+7 more)

### Community 25 - "run_rag_asp_guided"
Cohesion: 0.29
Nodes (4): _aspect_row(), _get_spans(), run_aspect_only(), run_rag_asp_guided()

### Community 26 - "GenerationConfig"
Cohesion: 0.20
Nodes (16): GenerationConfig, run_rag_aspect(), _FakeRetriever, _patch_runner(), test_a_raft_mode_without_an_adapter_is_refused(), test_a_resumed_run_skips_the_rows_already_done(), test_crag_is_not_available_before_step_9(), test_every_row_records_top_ce_score_and_n_ctx_used() (+8 more)

### Community 27 - "generate_batch"
Cohesion: 0.22
Nodes (5): generate_batch(), generate_one(), _make_retriever(), run_rag_guided(), run_rag_only()

### Community 28 - "Checkpointer"
Cohesion: 0.20
Nodes (4): Checkpointer, test_checkpointer_defaults_to_the_current_directory(), test_checkpointer_round_trips_rows(), test_checkpointer_writes_into_the_given_directory()

### Community 29 - "response_generator_v2.py"
Cohesion: 0.10
Nodes (13): _apply_chat_template(), _aspect_columns_ready(), compute_metrics(), enrich_dataframe_with_aspects(), _is_pretrained(), maybe_merge_aspects_file(), print_metrics(), _resolve_aspect_extractor_kwargs() (+5 more)

### Community 30 - "aspect_heads.py"
Cohesion: 0.11
Nodes (24): predict_head_labels(), align_targets(), _cmd_eval(), _cmd_predict(), _cmd_train(), embed_texts(), embedding_text(), evaluate_heads() (+16 more)

### Community 31 - "extract_entities"
Cohesion: 0.17
Nodes (10): extract_entities(), test_a_word_repeated_in_the_body_is_kept(), test_product_patterns_still_win(), test_title_case_subject_words_no_longer_become_entities(), A.0 Evidence (from `docs/samples/aspects_SAMPLE.csv`, 11 test tickets), A.1 Supervised aspect heads (replace the SFT labels at inference; CPU, minutes), A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU), A.3 Aspects in retrieval and prompt (+2 more)

### Community 32 - "model_backend.py"
Cohesion: 0.19
Nodes (7): count_tokens_fn(), load_generator(), _load_hf(), _load_unsloth(), text_tokenizer(), test_an_unknown_backend_is_rejected(), test_count_tokens_does_not_add_special_tokens()

### Community 33 - "utils.py"
Cohesion: 0.18
Nodes (3): _aspect_value(), load_eval_config(), test_kaggle_yaml_has_every_section_the_plan_lists()

### Community 34 - ".__init__"
Cohesion: 0.18
Nodes (4): DataLoader, EvaluationPipeline, Profiler, _Result

### Community 35 - "fit_prompt"
Cohesion: 0.23
Nodes (7): fit_prompt(), test_fit_prompt_accepts_no_retrieved_docs(), test_fit_prompt_drops_the_lowest_ranked_docs_until_it_fits(), test_fit_prompt_keeps_every_doc_when_the_budget_allows(), test_fit_prompt_returns_the_bare_prompt_when_nothing_fits(), test_the_real_prompt_stays_within_budget_with_five_long_hits(), _words()

### Community 36 - "test_prompt.py"
Cohesion: 0.21
Nodes (11): inject_sentiment_severity(), reply_language_line(), _stub_retriever(), test_a_german_ticket_gets_a_german_instruction_last(), test_ctx_doc_max_chars_caps_each_retrieved_response(), test_language_line_falls_back_for_an_unknown_code(), test_language_line_names_the_ticket_language(), test_the_language_line_can_be_switched_off() (+3 more)

### Community 37 - "is_pleasantry"
Cohesion: 0.33
Nodes (4): is_pleasantry(), _next_candidate(), test_pleasantries_are_recognised(), test_problem_sentences_are_not_pleasantries()

### Community 38 - "generate"
Cohesion: 0.25
Nodes (5): generate(), make_chat_fn(), chat_fn(), test_chat_fn_falls_back_to_string_content(), test_chat_fn_returns_the_raw_prompt_for_a_pretrained_model()

### Community 39 - "build_aspect_only_prompt"
Cohesion: 0.20
Nodes (7): build_aspect_only_prompt(), extract_sentiment_severity(), test_aspect_only_prompt_uses_the_checklist_too(), test_english_tickets_still_get_a_sentiment(), test_german_severity_keywords_are_recognised(), test_sentiment_is_not_invented_for_a_german_ticket(), test_the_aspect_only_prompt_gets_the_language_line()

### Community 40 - "_FakeProcessor"
Cohesion: 0.22
Nodes (4): _FakeProcessor, _FakeTextTokenizer, test_chat_fn_uses_typed_content_for_a_gemma_3_processor(), test_text_tokenizer_reaches_through_a_processor()

### Community 41 - "typing"
Cohesion: 0.19
Nodes (4): load_config(), main(), reorder_columns(), StratifiedSplitter

### Community 42 - "jaccard"
Cohesion: 0.22
Nodes (6): has_action_cue(), jaccard(), _tokens(), test_jaccard_measures_token_overlap_with_umlauts(), test_process_row_cleans_the_spans_of_a_real_sample_row(), test_real_actions_keep_their_cue()

### Community 43 - "eos_token_ids"
Cohesion: 0.40
Nodes (3): eos_token_ids(), test_eos_includes_gemma_end_of_turn(), test_eos_skips_an_unknown_end_of_turn_token()

### Community 45 - "get_content_sentences"
Cohesion: 0.29
Nodes (5): get_content_sentences(), is_pure_greeting(), split_sentences(), strip_greeting_prefix(), A.4 Aspect-centric metrics (computed in `minimal_run report`, with bootstrap CIs like all other metrics)

### Community 46 - "AspectForge v3 — Progress"
Cohesion: 0.33
Nodes (5): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup

### Community 48 - "deep_merge"
Cohesion: 0.50
Nodes (3): deep_merge(), test_deep_merge_recurses_and_leaves_the_base_untouched(), test_deep_merge_replaces_lists_wholesale()

## Knowledge Gaps
- **47 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `Useful facts` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 303 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `File Descriptions` connect `File Descriptions` to `retriever.py`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _47 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._
- **Why does `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` connect `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` to `retriever.py`, `run_extra_baseline`, `baseline_t5.py`, `extract_entities`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `postprocess_spans()` (e.g. with `10. Implementation order and acceptance tests (do in this order; commit after each)` and `A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU)`) actually correct?**
  _`postprocess_spans()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `retriever.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06866002214839424 - nodes in this community are weakly interconnected._