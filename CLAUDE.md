# AspectForge — Claude Code instructions

Aspect-aware RAG for bilingual (EN/DE) customer-support reply generation (IIIT Hyderabad INLP project).
v3 adds Aspect v2, multilingual Gemma-3, hybrid retrieval, Aspect-RAFT and a CRAG router, evaluated by a
**minimal run on Kaggle T4×2**.

## Start of every session
1. Read `PROGRESS.md` (what is done, what is next, open issues).
2. Read the `UPGRADE_PLAN.md` section for the next unchecked step (§10 lists the order). The plan is the spec; don't add scope.
3. Navigate code with graphify (section below) before grepping.

## Working rules
- Branch: `v3-minimal-run`. Never commit to `main` directly; the human merges the PR.
- One §10 step at a time: implement → `pytest -q` green → `graphify update .` → commit → push → tick the box in `PROGRESS.md` with the date and short sha.
- Match the existing code style (banner comments `# ── X ──`, dataclass configs, YAML-driven). Keep the folder structure; only add the files listed in plan §2/§A/§K.
- Every new behaviour is config-driven; the old behaviour must stay reachable via config.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## GPU work = Kaggle via API (plan §K)
- `python scripts/kaggle_job.py push|status|wait|pull --job <name>`. Jobs live in `kaggle/<job>/`.
- **Never push a Kaggle job unless `pytest -q` passes.** Push the commit first; `run.py` checks out the exact sha.
- Wait in the background (`wait` in a background shell or `/loop` at 10–20 min), never foreground polling.
- After `pull`, read `STATUS.json` and `logs/` before anything else. Max 3 attempts per job, then stop and report.
- No secrets on Kaggle. Models must come from ungated repos (`unsloth/gemma-3-4b-it`, `unsloth/gemma-2-2b-it`, `BAAI/*`, `ranjan56cse/cs-support-labels`). HF pushes happen locally.

## Ask the human only for
credentials or licences, anything that costs money (vast.ai via `C:\dev\gpu-shuttle` is a paid fallback: ask first), deleting data, merging to `main`, or after 3 failed attempts.

## Useful facts
- Raw data: Kaggle dataset `tobiasbueck/multilingual-customer-support-tickets`; splits come from `src/preprocessing.py` (seed 42, stratified by language). Root `preprocessing.py` is the older cleaning/PII pipeline (has `fix_encoding`, entity regexes).
- Test fixtures: `docs/samples/*.csv` (10 real rows: raw, aspects, two system outputs).
- Aspect SFT adapter: HF `ranjan56cse/cs-support-labels` (subfolder `sft_checkpoint`).
- Old report and slides: `docs/report/`. New architecture figure: `docs/architecture_v3.{svg,png}`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
