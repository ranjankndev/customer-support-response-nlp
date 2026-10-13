# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 39 files · ~62,986 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 911 nodes · 1846 edges · 63 communities (40 shown, 23 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 67 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4753fa08`
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
- test_eval_retrieval.py
- is_pleasantry
- CorpusIndexer
- build_aspect_only_prompt
- text_tokenizer
- step1_silver_labels.py
- utils.py
- count_tokens_fn
- fix_encoding
- .__init__
- aspect_score
- load_model
- fit_prompt
- _StubEmbedder
- test_the_embedder_uses_the_configured_prefixes
- build_prompt
- compare_models
- test_a_non_raft_mode_ignores_a_configured_adapter
- test_a_sample_with_aspect_columns_is_not_re_extracted
- json
- A. Aspect v2: make the "Aspect" in AspectForge measurable and strong
- sample_df
- jaccard
- 6. Data, aspects and retrieval
- AspectForge — Claude Code instructions
- ProgressCallback
- eos_token_ids

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 40 edges
2. `postprocess_spans()` - 23 edges
3. `run_rag_aspect()` - 23 edges
4. `Checkpointer` - 23 edges
5. `MetricsCalculator` - 22 edges
6. `load_yaml_with_overrides()` - 22 edges
7. `_split()` - 22 edges
8. `run()` - 21 edges
9. `TicketRetriever` - 18 edges
10. `_retriever()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `GPU work = Kaggle via API (plan §K)` --references--> `pull()`  [INFERRED]
  CLAUDE.md → scripts/kaggle_job.py
- `K.2 Kaggle via API (no notebook clicking)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `A.3 Aspects in retrieval and prompt` --references--> `aspect_score()`  [INFERRED]
  UPGRADE_PLAN.md → src/retrieval/retriever.py
- `10. Implementation order and acceptance tests (do in this order; commit after each)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py

## Import Cycles
- None detected.

## Communities (63 total, 23 thin omitted)

### Community 0 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.09
Nodes (21): 0. Why: problems found in the current code, 10. Implementation order and acceptance tests (do in this order; commit after each), 11. Risks and fallbacks, 1. What the minimal run produces, 2. Files: new and changed, 4. `config/kaggle.yaml` (overrides) and config merging, 7.1 Config, 7.2 `build` (+13 more)

### Community 1 - "inference_retriever.py"
Cohesion: 0.09
Nodes (14): `inference_retriever.py`, load_config(), main(), reorder_columns(), StratifiedSplitter, get_retriever(), load_retriever_config(), main() (+6 more)

### Community 2 - "test_metrics.py"
Cohesion: 0.05
Nodes (19): MetricsCalculator, _UnicodeTokenizer, calc(), _FakeTensor, MetCfg, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1() (+11 more)

### Community 4 - "run_extra_baseline"
Cohesion: 0.20
Nodes (8): generate_batch_extra(), load_extra_model(), run_extra_baseline(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.2 `src/generation/model_backend.py`, 5.5 Sentiment/severity for DE (F8), 5.6 Runners and modes for S1–S5, 5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)

### Community 5 - "aspect_extractor.py"
Cohesion: 0.10
Nodes (19): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_labels_batch(), get_content_sentences(), get_gpu_profile() (+11 more)

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
Cohesion: 0.19
Nodes (13): kaggle_prefix(), main(), pull(), push(), read_job(), resolve_commit(), run(), stage_job() (+5 more)

### Community 11 - "test_config.py"
Cohesion: 0.09
Nodes (22): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup, apply_overrides(), load_generation_config(), deep_merge() (+14 more)

### Community 13 - "test_aspects.py"
Cohesion: 0.10
Nodes (22): postprocess_spans(), build_aspect_checklist(), _stub_retriever(), test_a_cause_identical_to_the_problem_becomes_none(), test_a_distinct_cause_survives(), test_a_pleasantry_span_becomes_none_without_a_candidate(), test_a_pleasantry_span_is_replaced_by_the_next_ranked_sentence(), test_an_action_duplicating_the_cause_becomes_none() (+14 more)

### Community 21 - "indexer.py"
Cohesion: 0.28
Nodes (3): load_config(), _load_yaml(), main()

### Community 23 - "step2_sft.py"
Cohesion: 0.12
Nodes (6): build_input_prompt(), build_target_output(), get_gpu_profile(), _push_checkpoint_to_hf(), run(), SilverDataset

### Community 25 - "test_retriever_fusion.py"
Cohesion: 0.13
Nodes (18): index_dir(), _retriever(), StubConfig, _StubCrossEncoder, test_aspect_query_languages_decides_which_queries_are_aspect_weighted(), test_dense_retrieval_returns_the_matching_document(), test_exclude_ids_accepts_non_string_ids(), test_exclude_ids_removes_the_given_ticket() (+10 more)

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
Cohesion: 0.10
Nodes (24): predict_head_labels(), align_targets(), _cmd_eval(), _cmd_predict(), _cmd_train(), embed_texts(), embedding_text(), evaluate_heads() (+16 more)

### Community 31 - "extract_entities"
Cohesion: 0.33
Nodes (4): extract_entities(), test_a_word_repeated_in_the_body_is_kept(), test_product_patterns_still_win(), test_title_case_subject_words_no_longer_become_entities()

### Community 32 - "model_backend.py"
Cohesion: 0.15
Nodes (9): generate(), load_generator(), _load_hf(), _load_unsloth(), make_chat_fn(), chat_fn(), test_an_unknown_backend_is_rejected(), test_chat_fn_falls_back_to_string_content() (+1 more)

### Community 33 - "test_minimal_run.py"
Cohesion: 0.10
Nodes (35): attach_aspects(), build_sample(), cmd_attach_aspects(), cmd_sample(), file_hash(), load_minimal_config(), main(), MinimalRunConfig (+27 more)

### Community 34 - "retriever.py"
Cohesion: 0.09
Nodes (11): `retriever.py`, aspect_query_vector(), cross_encoder_rerank(), get_cross_encoder(), _legacy_cross_encoder_cache(), rrf_fuse(), TicketRetriever, test_rrf_k_damps_the_head_of_each_list() (+3 more)

### Community 35 - "test_prompt.py"
Cohesion: 0.21
Nodes (11): inject_sentiment_severity(), reply_language_line(), _stub_retriever(), test_a_german_ticket_gets_a_german_instruction_last(), test_ctx_doc_max_chars_caps_each_retrieved_response(), test_language_line_falls_back_for_an_unknown_code(), test_language_line_names_the_ticket_language(), test_the_language_line_can_be_switched_off() (+3 more)

### Community 36 - "test_eval_retrieval.py"
Cohesion: 0.08
Nodes (30): bootstrap_mean(), cosine(), evaluate_rows(), _hit_intent(), intent_at_k(), main(), rouge_l(), _RougeCfg (+22 more)

### Community 37 - "is_pleasantry"
Cohesion: 0.33
Nodes (4): is_pleasantry(), _next_candidate(), test_pleasantries_are_recognised(), test_problem_sentences_are_not_pleasantries()

### Community 38 - "CorpusIndexer"
Cohesion: 0.12
Nodes (6): CorpusIndexer, IndexerConfig, STEmbedder, test_records_without_aspects_still_carry_the_gold_columns(), test_the_indexer_refuses_an_index_containing_evaluation_rows(), test_the_indexer_stores_body_and_the_gold_columns()

### Community 39 - "build_aspect_only_prompt"
Cohesion: 0.20
Nodes (7): build_aspect_only_prompt(), extract_sentiment_severity(), test_aspect_only_prompt_uses_the_checklist_too(), test_english_tickets_still_get_a_sentiment(), test_german_severity_keywords_are_recognised(), test_sentiment_is_not_invented_for_a_german_ticket(), test_the_aspect_only_prompt_gets_the_language_line()

### Community 40 - "text_tokenizer"
Cohesion: 0.18
Nodes (5): text_tokenizer(), _FakeProcessor, _FakeTextTokenizer, test_chat_fn_uses_typed_content_for_a_gemma_3_processor(), test_text_tokenizer_reaches_through_a_processor()

### Community 41 - "step1_silver_labels.py"
Cohesion: 0.27
Nodes (8): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch()

### Community 42 - "utils.py"
Cohesion: 0.17
Nodes (4): _aspect_value(), get_hf_token(), load_eval_config(), 3. Kaggle constraints the code must respect

### Community 45 - ".__init__"
Cohesion: 0.18
Nodes (4): DataLoader, EvaluationPipeline, Profiler, _Result

### Community 46 - "aspect_score"
Cohesion: 0.20
Nodes (5): aspect_score(), test_aspect_score_ignores_none_values(), test_aspect_score_matches_predicted_heads_against_gold_metadata(), test_aspect_score_uses_unicode_tokens_for_overlap(), test_aspect_score_weights_intent_type_and_overlap()

### Community 48 - "fit_prompt"
Cohesion: 0.23
Nodes (7): fit_prompt(), test_fit_prompt_accepts_no_retrieved_docs(), test_fit_prompt_drops_the_lowest_ranked_docs_until_it_fits(), test_fit_prompt_keeps_every_doc_when_the_budget_allows(), test_fit_prompt_returns_the_bare_prompt_when_nothing_fits(), test_the_real_prompt_stays_within_budget_with_five_long_hits(), _words()

### Community 55 - "json"
Cohesion: 0.31
Nodes (3): log(), main(), update_status()

### Community 56 - "A. Aspect v2: make the "Aspect" in AspectForge measurable and strong"
Cohesion: 0.20
Nodes (8): split_sentences(), A.0 Evidence (from `docs/samples/aspects_SAMPLE.csv`, 11 test tickets), A.1 Supervised aspect heads (replace the SFT labels at inference; CPU, minutes), A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU), A.3 Aspects in retrieval and prompt, A.4 Aspect-centric metrics (computed in `minimal_run report`, with bootstrap CIs like all other metrics), A.5 Ablation for the aspect claim, A. Aspect v2: make the "Aspect" in AspectForge measurable and strong

### Community 58 - "jaccard"
Cohesion: 0.22
Nodes (6): has_action_cue(), jaccard(), _tokens(), test_jaccard_measures_token_overlap_with_umlauts(), test_process_row_cleans_the_spans_of_a_real_sample_row(), test_real_actions_keep_their_cue()

### Community 59 - "6. Data, aspects and retrieval"
Cohesion: 0.22
Nodes (5): STEmbedder, 6.1 `src/minimal_run.py sample`, 6.3 Retriever (`retriever.py`), 6.4 `src/retrieval/eval_retrieval.py`, 6. Data, aspects and retrieval

### Community 60 - "AspectForge — Claude Code instructions"
Cohesion: 0.25
Nodes (7): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules

### Community 62 - "eos_token_ids"
Cohesion: 0.40
Nodes (3): eos_token_ids(), test_eos_includes_gemma_end_of_turn(), test_eos_skips_an_unknown_end_of_turn_token()

## Knowledge Gaps
- **46 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `Useful facts` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 360 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_yaml_with_overrides()` connect `test_config.py` to `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)`, `test_minimal_run.py`, `inference_retriever.py`, `utils.py`, `indexer.py`, `response_generator_v2.py`, `aspect_heads.py`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` be split into smaller, more focused modules?**
  _Cohesion score 0.08695652173913043 - nodes in this community are weakly interconnected._
- **Why does `MetricsCalculator` connect `test_metrics.py` to `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)`, `test_eval_retrieval.py`, `baseline_t5.py`, `utils.py`, `.__init__`, `run`, `response_generator_v2.py`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `postprocess_spans()` (e.g. with `10. Implementation order and acceptance tests (do in this order; commit after each)` and `A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU)`) actually correct?**
  _`postprocess_spans()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `inference_retriever.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08870967741935484 - nodes in this community are weakly interconnected._