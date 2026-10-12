# AspectForge v3 — Progress

Spec: `UPGRADE_PLAN.md` (step order in §10). Branch: `v3-minimal-run`.
Legend: `[x]` done (date, sha) · `[~]` in progress · `[ ]` todo

## Setup
- [x] 2026-10-05 Repo moved out of OneDrive to `C:\dev\aspectforge`, synced with GitHub `main`
- [x] 2026-10-05 graphify installed (project scope) and code graph built
- [x] 2026-10-05 Plan extended: Aspect v2 (§A) + automation (§K); `CLAUDE.md` + this file added
- [x] 2026-10-05 Kaggle CLI upgraded to 2.2.4 (auth OK, user `ranjankumarnayak`); T4×2 = `machine_shape: NvidiaTeslaT4`
- [x] 2026-10-05 Work branch `v3-minimal-run` created and pushed. **Next agent starts at §10 step 0.**

## §10 steps
- [~] 0. Automation scaffold: `scripts/kaggle_job.py`, `kaggle/hello`, `tests/` + fixtures — code done 2026-10-13 (46d5405), `pytest -q` green; the `hello` GPU acceptance run is **blocked on Kaggle auth** (see open issues)
- [x] 2026-10-13 (b062c62) 1. Config plumbing (`load_yaml_with_overrides`, `config/kaggle.yaml`, new dataclass fields) — also made `src/utils.py` import lazily (1.8 s instead of 68 s); local test deps installed: `rouge-score`, `sacrebleu`, `faiss-cpu`, `peft`
- [x] 2026-10-13 (186c194) 2. Metrics (Unicode ROUGE, sacreBLEU, true BERTScore, lang_match, per-row) — `bertscore_f1` now means real BERTScore; the old CLS cosine is `cls_cosine_mbert`; BLEU precisions renamed `bleu_p1`/`bleu_p4`
- [x] 2026-10-13 (cbe206a) 2b. Aspect v2 (heads, span clean-up, entity fix, checklist prompt) — heads are unit-tested on synthetic clusters; the real macro-F1 vs majority run needs the train split (blocked with Kaggle auth). `aspect_extractor --labels heads` is now the default and loads no GPU model
- [~] 3. model_backend + generator fixes (Gemma-3, double BOS, budget, language line) — code done 2026-10-13 (884c7d1), `tests/test_prompt.py` green. The GPU acceptance checks (one `<bos>`, DE prompt -> German reply, 5 long hits within budget) need the Kaggle smoke cell: **blocked on Kaggle auth**
- [x] 2026-10-13 (a3b5137) 4. Runners / modes / labels / Checkpointer dir — `--mode rag_aspect --label s2_gemma3` writes `{output_dir}/s2_gemma3.csv` with `top_ce_score` and `n_ctx_used`; `tests/test_runners.py` covers it with a fake retriever
- [x] 2026-10-13 (2d3f55a) 5. Hybrid retrieval + multilingual reranker — DE rows now get a `ce_score` (bge-reranker-v2-m3), the aspect bonus is applied after reranking for both languages, `exclude_ids` works; `tests/test_retriever_fusion.py` builds a 3-doc FAISS+BM25 index. The `inference_retriever --n 20` run on real data needs the index (blocked with Kaggle auth / no local data)
- [x] 2026-10-13 (08cc0c2) 6. `minimal_run sample` / `attach_aspects` — 250/250 split and sample reuse (same hash) covered by `tests/test_minimal_run.py`; the real samples need the processed splits, which are not on this machine
- [ ] 7. `eval_retrieval`
- [ ] 8. `step3_raft build` + `train`
- [ ] 9. CRAG router + `crag_tune`
- [ ] 10. `minimal_run report` (+ ACS/AAS, aspect table)
- [ ] 11. Kaggle jobs `prep_train`, `generate_eval` (+ optional notebook mirrors)
- [ ] 12. README + figures; open PR

## Kaggle runs
| Date | Job | Commit | Result | Notes |
|---|---|---|---|---|

## Open issues / decisions
- Which raw CSV of the Kaggle dataset reproduces the report's 28,237 EN+DE rows? (resolve in step 11)
- 2026-10-13 HF login fixed: local `hf auth whoami` = `ranjankn`. Upload the RAFT adapter to `ranjankn/aspectforge-raft-gemma3`. `ranjan56cse/cs-support-labels` (aspect SFT) stays read-only/public.
- Human one-time: Kaggle phone verification (GPU + Internet).
- **2026-10-13 BLOCKER — Kaggle API auth rejected.** `~/.kaggle/kaggle.json` (user `ranjankumarnayak`, `auth_method: LEGACY_API_KEY`) is read by the CLI, but every authenticated call (`kernels list --mine`, `kernels push`) returns the login prompt, so the legacy key is expired/revoked. Human fix: run `kaggle auth login` (OAuth) or generate a new token at <https://www.kaggle.com/settings/api> and replace `~/.kaggle/kaggle.json`. Until then step 0's GPU acceptance test and §10 steps 3/8/9/11 cannot run; CPU-only steps continue.
