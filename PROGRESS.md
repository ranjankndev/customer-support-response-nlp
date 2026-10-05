# AspectForge v3 — Progress

Spec: `UPGRADE_PLAN.md` (step order in §10). Branch: `v3-minimal-run`.
Legend: `[x]` done (date, sha) · `[~]` in progress · `[ ]` todo

## Setup
- [x] 2026-10-05 Repo moved out of OneDrive to `C:\dev\aspectforge`, synced with GitHub `main`
- [x] 2026-10-05 graphify installed (project scope) and code graph built
- [x] 2026-10-05 Plan extended: Aspect v2 (§A) + automation (§K); `CLAUDE.md` + this file added

## §10 steps
- [ ] 0. Automation scaffold: `scripts/kaggle_job.py`, `kaggle/hello`, `tests/` + fixtures
- [ ] 1. Config plumbing (`load_yaml_with_overrides`, `config/kaggle.yaml`, new dataclass fields)
- [ ] 2. Metrics (Unicode ROUGE, sacreBLEU, true BERTScore, lang_match, per-row)
- [ ] 2b. Aspect v2 (heads, span clean-up, entity fix, checklist prompt)
- [ ] 3. model_backend + generator fixes (Gemma-3, double BOS, budget, language line)
- [ ] 4. Runners / modes / labels / Checkpointer dir
- [ ] 5. Hybrid retrieval + multilingual reranker
- [ ] 6. `minimal_run sample` / `attach_aspects`
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
- HF API checks from this machine returned "Invalid username or password" for some public repos; check the local HF login before any local HF upload.
- Human one-time: Kaggle phone verification (GPU + Internet).
