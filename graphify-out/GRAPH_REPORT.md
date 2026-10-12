# Graph Report - aspectforge  (2026-10-13)

## Corpus Check
- 35 files · ~59,200 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .csv 4, (none) 1)

## Summary
- 810 nodes · 1582 edges · 47 communities (32 shown, 15 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 66 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b058a6fc`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)
- TicketRetriever
- test_metrics.py
- run_extra_baseline
- aspect_extractor.py
- baseline_t5.py
- preprocessing.py
- File Descriptions
- utils.py
- kaggle_job.py
- test_config.py
- Profiler
- test_aspects.py
- .claude/CLAUDE.md
- indexer.py
- step2_sft.py
- test_retriever_fusion.py
- GenerationConfig
- generate_batch
- Checkpointer
- response_generator_v2.py
- aspect_heads.py
- extract_entities
- model_backend.py
- MetricsCalculator
- _make_retriever
- fit_prompt
- test_prompt.py
- is_pleasantry
- make_chat_fn
- build_aspect_only_prompt
- text_tokenizer
- _FakeTensor
- jaccard
- count_tokens_fn
- fix_encoding
- calc
- load_model

## God Nodes (most connected - your core abstractions)
1. `GenerationConfig` - 40 edges
2. `postprocess_spans()` - 23 edges
3. `run_rag_aspect()` - 23 edges
4. `Checkpointer` - 23 edges
5. `run()` - 21 edges
6. `MetricsCalculator` - 20 edges
7. `load_yaml_with_overrides()` - 20 edges
8. `_retriever()` - 17 edges
9. `TicketRetriever` - 16 edges
10. `CorpusIndexer` - 15 edges

## Surprising Connections (you probably didn't know these)
- `K.2 Kaggle via API (no notebook clicking)` --references--> `pull()`  [INFERRED]
  UPGRADE_PLAN.md → scripts/kaggle_job.py
- `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)` --references--> `GenerationConfig`  [INFERRED]
  UPGRADE_PLAN.md → src/generation/response_generator_v2.py
- `A.3 Aspects in retrieval and prompt` --references--> `aspect_score()`  [INFERRED]
  UPGRADE_PLAN.md → src/retrieval/retriever.py
- `3. Kaggle constraints the code must respect` --references--> `get_hf_token()`  [INFERRED]
  UPGRADE_PLAN.md → src/utils.py
- `GPU work = Kaggle via API (plan §K)` --references--> `pull()`  [INFERRED]
  CLAUDE.md → scripts/kaggle_job.py

## Import Cycles
- None detected.

## Communities (47 total, 15 thin omitted)

### Community 0 - "AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)"
Cohesion: 0.08
Nodes (22): 0. Why: problems found in the current code, 10. Implementation order and acceptance tests (do in this order; commit after each), 11. Risks and fallbacks, 1. What the minimal run produces, 2. Files: new and changed, 3. Kaggle constraints the code must respect, 4. `config/kaggle.yaml` (overrides) and config merging, 7.1 Config (+14 more)

### Community 1 - "TicketRetriever"
Cohesion: 0.07
Nodes (15): `inference_retriever.py`, get_retriever(), load_retriever_config(), main(), RetrievalConfig, retrieve_similar(), run_file(), get_cross_encoder() (+7 more)

### Community 2 - "test_metrics.py"
Cohesion: 0.10
Nodes (6): _UnicodeTokenizer, test_cls_cosine_is_no_longer_called_bertscore(), test_lang_match_is_one_when_every_reply_uses_the_ticket_language(), test_rouge_scores_identical_german_as_one(), test_rouge_without_the_unicode_tokenizer_loses_german(), test_unicode_tokenizer_keeps_umlauts_and_lowercases()

### Community 4 - "run_extra_baseline"
Cohesion: 0.14
Nodes (8): generate(), build_baseline_prompt(), build_baseline_prompt_pt(), generate_batch_extra(), load_extra_model(), run_extra_baseline(), get_hf_token(), 5.2 `src/generation/model_backend.py`

### Community 5 - "aspect_extractor.py"
Cohesion: 0.11
Nodes (18): _build_chat_text(), build_prob_statement(), build_sft_prompt(), _decode_generated(), encode_anchors(), extract_labels_batch(), get_content_sentences(), get_gpu_profile() (+10 more)

### Community 6 - "baseline_t5.py"
Cohesion: 0.15
Nodes (9): build_prompt(), compute_metrics(), generate_batch(), load_model(), load_t5_config(), print_metrics(), run(), run_t5_base() (+1 more)

### Community 7 - "preprocessing.py"
Cohesion: 0.06
Nodes (8): DataPreprocessor, fix_encoding(), strip_email_filler(), track_run(), load_config(), main(), reorder_columns(), StratifiedSplitter

### Community 8 - "File Descriptions"
Cohesion: 0.08
Nodes (23): `aspect_extractor.py`, AspectForge — Multilingual Aspect-Aware Customer Support Response Generation, `baseline_t5.py`, `config/generation_config.yaml`, `config/inference_retrieval_config.yaml`, Configuration, File Descriptions, Hardware (+15 more)

### Community 9 - "utils.py"
Cohesion: 0.08
Nodes (8): _aspect_value(), BaselineGenerator, ContextBuilder, DataLoader, EvaluationPipeline, load_eval_config(), Profiler, _Result

### Community 10 - "kaggle_job.py"
Cohesion: 0.10
Nodes (22): Ask the human only for, AspectForge — Claude Code instructions, GPU work = Kaggle via API (plan §K), graphify, Start of every session, Useful facts, Working rules, log() (+14 more)

### Community 11 - "test_config.py"
Cohesion: 0.07
Nodes (28): §10 steps, AspectForge v3 — Progress, Kaggle runs, Open issues / decisions, Setup, apply_overrides(), load_generation_config(), resolve_data_path() (+20 more)

### Community 13 - "test_aspects.py"
Cohesion: 0.11
Nodes (21): postprocess_spans(), build_aspect_checklist(), _stub_retriever(), test_a_cause_identical_to_the_problem_becomes_none(), test_a_distinct_cause_survives(), test_a_pleasantry_span_becomes_none_without_a_candidate(), test_a_pleasantry_span_is_replaced_by_the_next_ranked_sentence(), test_an_action_duplicating_the_cause_becomes_none() (+13 more)

### Community 21 - "indexer.py"
Cohesion: 0.09
Nodes (9): CorpusIndexer, IndexerConfig, load_config(), _load_yaml(), main(), STEmbedder, test_records_without_aspects_still_carry_the_gold_columns(), test_the_indexer_refuses_an_index_containing_evaluation_rows() (+1 more)

### Community 23 - "step2_sft.py"
Cohesion: 0.07
Nodes (15): auto_save_to_dataset(), build_judge_prompt(), get_gpu_profile(), load_judge(), parse_judge_output(), _read_jsonl(), run(), run_judge_batch() (+7 more)

### Community 25 - "test_retriever_fusion.py"
Cohesion: 0.05
Nodes (32): aspect_score(), _legacy_cross_encoder_cache(), rrf_fuse(), index_dir(), _retriever(), StubConfig, _StubCrossEncoder, _StubEmbedder (+24 more)

### Community 26 - "GenerationConfig"
Cohesion: 0.13
Nodes (19): GenerationConfig, run_rag_aspect(), _FakeRetriever, _patch_runner(), sample_df(), test_a_non_raft_mode_ignores_a_configured_adapter(), test_a_raft_mode_without_an_adapter_is_refused(), test_a_resumed_run_skips_the_rows_already_done() (+11 more)

### Community 27 - "generate_batch"
Cohesion: 0.25
Nodes (4): generate_batch(), generate_one(), run_baseline(), run_rag_only()

### Community 28 - "Checkpointer"
Cohesion: 0.20
Nodes (4): Checkpointer, test_checkpointer_defaults_to_the_current_directory(), test_checkpointer_round_trips_rows(), test_checkpointer_writes_into_the_given_directory()

### Community 29 - "response_generator_v2.py"
Cohesion: 0.09
Nodes (16): load_config(), _apply_chat_template(), _aspect_columns_ready(), _aspect_row(), compute_metrics(), enrich_dataframe_with_aspects(), _get_spans(), _is_pretrained() (+8 more)

### Community 30 - "aspect_heads.py"
Cohesion: 0.08
Nodes (28): `retriever.py`, predict_head_labels(), align_targets(), _cmd_eval(), _cmd_predict(), _cmd_train(), embed_texts(), embedding_text() (+20 more)

### Community 31 - "extract_entities"
Cohesion: 0.14
Nodes (11): extract_entities(), split_sentences(), test_a_word_repeated_in_the_body_is_kept(), test_product_patterns_still_win(), test_title_case_subject_words_no_longer_become_entities(), A.0 Evidence (from `docs/samples/aspects_SAMPLE.csv`, 11 test tickets), A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU), A.3 Aspects in retrieval and prompt (+3 more)

### Community 32 - "model_backend.py"
Cohesion: 0.38
Nodes (4): load_generator(), _load_hf(), _load_unsloth(), test_an_unknown_backend_is_rejected()

### Community 33 - "MetricsCalculator"
Cohesion: 0.19
Nodes (4): MetricsCalculator, 8.1 `MetricsCalculator` fixes, 8.2 `python -m src.minimal_run report --systems s1_gemma2 s2_gemma3 s3_gemma3_crag s4_raft_aspect s5_raft_noaspect`, 8. Metrics and report (`src/utils.py`, `src/minimal_run.py report`)

### Community 34 - "_make_retriever"
Cohesion: 0.20
Nodes (8): build_guided_rag_prompt(), _make_retriever(), run_rag_guided(), 5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`), 5.3 Prompt budget (F3), 5.4 Language line (F2), 5.6 Runners and modes for S1–S5, 5. Generator changes (`response_generator_v2.py` + new `model_backend.py`)

### Community 35 - "fit_prompt"
Cohesion: 0.23
Nodes (7): fit_prompt(), test_fit_prompt_accepts_no_retrieved_docs(), test_fit_prompt_drops_the_lowest_ranked_docs_until_it_fits(), test_fit_prompt_keeps_every_doc_when_the_budget_allows(), test_fit_prompt_returns_the_bare_prompt_when_nothing_fits(), test_the_real_prompt_stays_within_budget_with_five_long_hits(), _words()

### Community 36 - "test_prompt.py"
Cohesion: 0.21
Nodes (11): inject_sentiment_severity(), reply_language_line(), _stub_retriever(), test_a_german_ticket_gets_a_german_instruction_last(), test_ctx_doc_max_chars_caps_each_retrieved_response(), test_language_line_falls_back_for_an_unknown_code(), test_language_line_names_the_ticket_language(), test_the_language_line_can_be_switched_off() (+3 more)

### Community 37 - "is_pleasantry"
Cohesion: 0.33
Nodes (4): is_pleasantry(), _next_candidate(), test_pleasantries_are_recognised(), test_problem_sentences_are_not_pleasantries()

### Community 38 - "make_chat_fn"
Cohesion: 0.25
Nodes (5): make_chat_fn(), chat_fn(), test_chat_fn_falls_back_to_string_content(), test_chat_fn_returns_the_raw_prompt_for_a_pretrained_model(), test_chat_fn_uses_typed_content_for_a_gemma_3_processor()

### Community 39 - "build_aspect_only_prompt"
Cohesion: 0.18
Nodes (8): build_aspect_only_prompt(), extract_sentiment_severity(), test_aspect_only_prompt_uses_the_checklist_too(), test_english_tickets_still_get_a_sentiment(), test_german_severity_keywords_are_recognised(), test_sentiment_is_not_invented_for_a_german_ticket(), test_the_aspect_only_prompt_gets_the_language_line(), 5.5 Sentiment/severity for DE (F8)

### Community 40 - "text_tokenizer"
Cohesion: 0.14
Nodes (7): eos_token_ids(), text_tokenizer(), _FakeProcessor, _FakeTextTokenizer, test_eos_includes_gemma_end_of_turn(), test_eos_skips_an_unknown_end_of_turn_token(), test_text_tokenizer_reaches_through_a_processor()

### Community 41 - "_FakeTensor"
Cohesion: 0.28
Nodes (5): _FakeTensor, test_bertscore_true_honours_explicit_arguments(), fake_score(), test_bertscore_true_uses_bert_score_and_returns_per_row_f1(), fake_score()

### Community 42 - "jaccard"
Cohesion: 0.22
Nodes (6): has_action_cue(), jaccard(), _tokens(), test_jaccard_measures_token_overlap_with_umlauts(), test_process_row_cleans_the_spans_of_a_real_sample_row(), test_real_actions_keep_their_cue()

## Knowledge Gaps
- **46 isolated node(s):** `graphify`, `Start of every session`, `Working rules`, `Ask the human only for`, `Useful facts` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 331 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `File Descriptions` connect `File Descriptions` to `TicketRetriever`, `aspect_heads.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `GenerationConfig` (e.g. with `test_generation_config_exposes_the_new_fields()` and `5.1 New config keys (`GenerationConfig` dataclass + `generation_config.yaml`)`) actually correct?**
  _`GenerationConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Start of every session`, `Working rules` to the rest of the system?**
  _46 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AspectForge v3 — Minimal Run on Kaggle (end-to-end plan)` be split into smaller, more focused modules?**
  _Cohesion score 0.08333333333333333 - nodes in this community are weakly interconnected._
- **Why does `TicketRetriever` connect `TicketRetriever` to `_make_retriever`, `test_prompt.py`, `test_aspects.py`, `test_retriever_fusion.py`, `response_generator_v2.py`, `aspect_heads.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `postprocess_spans()` (e.g. with `10. Implementation order and acceptance tests (do in this order; commit after each)` and `A.2 Span clean-up (`postprocess_spans()` in `aspect_extractor.py`, CPU)`) actually correct?**
  _`postprocess_spans()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `TicketRetriever` be split into smaller, more focused modules?**
  _Cohesion score 0.07394957983193277 - nodes in this community are weakly interconnected._