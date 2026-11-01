# AspectForge v3 — Progress

Spec: `UPGRADE_PLAN.md` (step order in §10). Branch: `v3-minimal-run`.
Legend: `[x]` done (date, sha) · `[~]` in progress · `[ ]` todo

## Setup
- [x] 2026-10-05 Repo moved out of OneDrive to `C:\dev\aspectforge`, synced with GitHub `main`
- [x] 2026-10-05 graphify installed (project scope) and code graph built
- [x] 2026-10-05 Plan extended: Aspect v2 (§A) + automation (§K); `CLAUDE.md` + this file added
- [x] 2026-10-05 Kaggle CLI upgraded to 2.2.4 (auth OK, user `ranjankumarnayak`); T4×2 = `machine_shape: NvidiaTeslaT4`
- [x] 2026-10-05 Work branch `v3-minimal-run` created and pushed.
- [x] 2026-10-13 §10 steps 0-12 implemented on CPU; PR #1 open. **Next agent: fix Kaggle auth, then run the four pending GPU acceptance checks (steps 0, 3, 8, 11) and the two jobs.**

## §10 steps
- [~] 0. Automation scaffold: `scripts/kaggle_job.py`, `kaggle/hello`, `tests/` + fixtures — code done 2026-10-13 (46d5405), `pytest -q` green; the `hello` GPU acceptance run is **blocked on Kaggle auth** (see open issues)
- [x] 2026-10-13 (b062c62) 1. Config plumbing (`load_yaml_with_overrides`, `config/kaggle.yaml`, new dataclass fields) — also made `src/utils.py` import lazily (1.8 s instead of 68 s); local test deps installed: `rouge-score`, `sacrebleu`, `faiss-cpu`, `peft`
- [x] 2026-10-13 (186c194) 2. Metrics (Unicode ROUGE, sacreBLEU, true BERTScore, lang_match, per-row) — `bertscore_f1` now means real BERTScore; the old CLS cosine is `cls_cosine_mbert`; BLEU precisions renamed `bleu_p1`/`bleu_p4`
- [x] 2026-10-13 (cbe206a) 2b. Aspect v2 (heads, span clean-up, entity fix, checklist prompt) — heads are unit-tested on synthetic clusters; the real macro-F1 vs majority run needs the train split (blocked with Kaggle auth). `aspect_extractor --labels heads` is now the default and loads no GPU model
- [~] 3. model_backend + generator fixes (Gemma-3, double BOS, budget, language line) — code done 2026-10-13 (884c7d1), `tests/test_prompt.py` green. The GPU acceptance checks (one `<bos>`, DE prompt -> German reply, 5 long hits within budget) need the Kaggle smoke cell: **blocked on Kaggle auth**
- [x] 2026-10-13 (a3b5137) 4. Runners / modes / labels / Checkpointer dir — `--mode rag_aspect --label s2_gemma3` writes `{output_dir}/s2_gemma3.csv` with `top_ce_score` and `n_ctx_used`; `tests/test_runners.py` covers it with a fake retriever
- [x] 2026-10-13 (2d3f55a) 5. Hybrid retrieval + multilingual reranker — DE rows now get a `ce_score` (bge-reranker-v2-m3), the aspect bonus is applied after reranking for both languages, `exclude_ids` works; `tests/test_retriever_fusion.py` builds a 3-doc FAISS+BM25 index. The `inference_retriever --n 20` run on real data needs the index (blocked with Kaggle auth / no local data)
- [x] 2026-10-13 (08cc0c2) 6. `minimal_run sample` / `attach_aspects` — 250/250 split and sample reuse (same hash) covered by `tests/test_minimal_run.py`; the real samples need the processed splits, which are not on this machine
- [x] 2026-10-13 (b04fde1) 7. `eval_retrieval` — `top1_rougeL`, `top1_cos`, `intent@3` with bootstrap CIs per language, written to `output/report/retrieval_table.csv` (one row set per config label). Needs the built index to run on real data
- [~] 8. `step3_raft build` + `train` — code done 2026-10-13 (8d34db4), `tests/test_raft.py` covers the build, the leakage guards and the TRL-argument shim. `build --limit 20` needs the index and `train --limit 64 --max_steps 5` needs Kaggle: **both blocked on Kaggle auth**
- [x] 2026-10-13 (b7a6f55) 9. CRAG router + `crag_tune` — `--mode rag_aspect_crag` stores `route`; `crag_tune` replays the router over tau 0.05-0.60 from paired generations (no sweep re-runs) and `crag_reuse` builds S3 from S2's outputs. `tests/test_crag.py` covers all three
- [x] 2026-10-13 (5b9fccb) 10. `minimal_run report` (+ ACS/AAS, aspect table) — the plan's acceptance test (2 fake systems x 20 rows, CI columns present) is in `tests/test_report.py`; writes `per_row_metrics.csv`, `results_table.csv`, `results.md`, `results_chart.png`
- [~] 11. Kaggle jobs `prep_train`, `generate_eval` (+ optional notebook mirrors) — scripts + `requirements-kaggle.txt` done 2026-10-13 (2c92185), validated by `tests/test_kaggle_job.py` (17 tests). **Neither job has been pushed: blocked on Kaggle auth.** The notebook mirrors (plan §9) are optional and not written
- [x] 2026-10-13 (7d38c2d) 12. README + figures; **PR #1 open** <https://github.com/ranjankndev/customer-support-response-nlp/pull/1> — the human reviews and merges

## Kaggle runs
| Date | Job | Commit | Result | Notes |
|---|---|---|---|---|
| 2026-10-31 | hello | 1194a2b | OK (2x T4, exact-sha checkout) | attempt 1; `wait` crashed on cp1252 and titles mis-slugged; both fixed |
| 2026-10-31 | prep_train | 8fc4a40 | ERROR at splits | attempt 1; smoke OK (Gemma-3 + Gemma-2: 1 bos, German reply, budget); public dataset mount had no CSV |
| 2026-11-01 | prep_train | 7284a07 | ERROR at sample | attempt 2; raw = private dataset `ranjankumarnayak/aspectforge-cs-support` (36,159 rows); splits OK 25,319/5,420/5,420; leakage guard: per-split idx renumbering collided -> fixed in 3fb543b (`split.renumber_idx`) |
| 2026-11-01 | prep_train | 3fb543b | ERROR at aspects | attempt 3; sample OK (no idx overlap). HF `ranjan56cse/cs-support-labels` is private -> 401 on Kaggle. Human chose: copy adapter to private Kaggle dataset `ranjankumarnayak/aspectforge-aspect-sft` |
| 2026-11-01 | prep_train | 8975945 | ERROR at aspects | attempt 4; adapter dataset found at /kaggle/input/aspectforge-aspect-sft but the extractor still appended subfolder `sft_checkpoint` |
| 2026-11-01 | prep_train | 484dafc | ERROR at eval_retrieval | attempt 5; aspects OK (51 min, 2 GPUs), index OK (25,319 docs), aspect heads: intent 7 values (top share 0.41). Reranker OOM: 14.5 GB in use, 3.4 GB torch -> TF/JAX preallocation suspected |
| 2026-11-01 | prep_train | ec2bb69 | ERROR at raft_train | attempt 6; cache restore OK, eval_retrieval OK (val top1_rougeL EN 0.539 / DE 0.221, top1_cos 0.857; intent@3 0.000 = metric bug), raft_build OK (3000 ex., distractor 0.20, aspects 0.81, prompt p95 710 tok). TRL bf16 default refused on T4 |
| 2026-11-01 | prep_train | 894d9b2 | ERROR at raft_train | attempt 7; cached RAFT examples OK; intent@3 = 0.551 [0.525, 0.581] (EN 0.551 / DE 0.552). Unsloth loaded bf16 weights under the fp16 grad scaler |
| 2026-11-01 | prep_train | (next) | running | attempt 8; Unsloth dtype follows precision (fp16 on T4) |

## Open issues / decisions
- RESOLVED 2026-11-01: raw CSV = the human's `data/cs-support-multi.csv` (25,072 EN / 11,087 DE), uploaded as private dataset `ranjankumarnayak/aspectforge-cs-support`. `data/cs-support-multi_old_discard.csv.csv` is not used. Row counts differ from the old report's 28,237, so v3 numbers are a fresh run.
- 2026-10-13 HF login fixed: local `hf auth whoami` = `ranjankn`. Upload the RAFT adapter to `ranjankn/aspectforge-raft-gemma3`. `ranjan56cse/cs-support-labels` (aspect SFT) stays read-only/public.
- Human one-time: Kaggle phone verification (GPU + Internet).
- **2026-10-31 RESOLVED — Kaggle auth.** `kaggle auth login` (OAuth) wrote `~/.kaggle/credentials.json`; the stale legacy key was renamed to `~/.kaggle/kaggle.json.old` because the CLI preferred it over OAuth. `kaggle kernels list --mine` works. Steps 0/3/8/11 are unblocked.
- **2026-10-13 BLOCKER — Kaggle API auth rejected.** `~/.kaggle/kaggle.json` (user `ranjankumarnayak`, `auth_method: LEGACY_API_KEY`) is read by the CLI, but every authenticated call (`kernels list --mine`, `kernels push`) returns the login prompt, so the legacy key is expired/revoked. Human fix: run `kaggle auth login` (OAuth) or generate a new token at <https://www.kaggle.com/settings/api> and replace `~/.kaggle/kaggle.json`. Until then step 0's GPU acceptance test and §10 steps 3/8/9/11 cannot run; CPU-only steps continue.
