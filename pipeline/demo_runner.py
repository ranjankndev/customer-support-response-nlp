"""
demo_runner.py
─────────────────────────────────────────────────────────────────────────────
Self-contained Kaggle notebook runner for the 50-ticket aspect-injection demo.
Copy each CELL block into a separate Kaggle notebook cell.

Run order: Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 (view results)
─────────────────────────────────────────────────────────────────────────────
"""

# ═════════════════════════════════════════════════════════════════════════════
# CELL 1 — pull repo + clear module cache
# ═════════════════════════════════════════════════════════════════════════════
CELL_1 = '''
from kaggle_secrets import UserSecretsClient
import subprocess, sys, os

token    = UserSecretsClient().get_secret("GITHUB_TOKEN")
REPO_DIR = "/kaggle/working/repo"

if os.path.exists(REPO_DIR):
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
else:
    subprocess.run([
        "git", "clone", "--depth=1", "-b", "main",
        f"https://{token}@github.com/ranjan56cse/customer-support-response-nlp.git",
        REPO_DIR
    ], check=True)

# clear cached modules so edits to pipeline files are picked up
for mod in list(sys.modules.keys()):
    if any(m in mod for m in ["rag", "instruct_generator", "bm25_tokenizer",
                               "nlp_enricher", "aspect_pipeline"]):
        del sys.modules[mod]

sys.path.insert(0, f"{REPO_DIR}/pipeline")
print("✓ Repo pulled, path set")
'''

# ═════════════════════════════════════════════════════════════════════════════
# CELL 2 — install dependencies
# ═════════════════════════════════════════════════════════════════════════════
CELL_2 = '''
import subprocess, sys

pkgs = [
    "vaderSentiment",
    "rank-bm25",
    "rouge-score",
    "bert-score",
    "ftfy",
    "faiss-gpu",          # or faiss-cpu if no GPU
    "accelerate>=0.26",
]

for pkg in pkgs:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

# transformers >= 4.37 required for Qwen2.5
import transformers
print(f"transformers version: {transformers.__version__}")
# if < 4.37 uncomment below:
# subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.37"], check=True)

print("✓ Dependencies installed")
'''

# ═════════════════════════════════════════════════════════════════════════════
# CELL 3 — paths
# ═════════════════════════════════════════════════════════════════════════════
CELL_3 = '''
import os

# ── data paths ────────────────────────────────────────────────────────────────
KB_PATH      = "/kaggle/input/datasets/ranjankumarnayak/cs-dataset/customer_support_28k_fixed.csv"
ASPECTS_PATH = "/kaggle/input/datasets/ranjankumarnayak/cs-aspects-28k/aspects_customer_support_28k_fixed.csv"
OUT_DIR      = "/kaggle/working"

# ── model choice ──────────────────────────────────────────────────────────────
# Option A — 1.5B (faster, ~3 GB VRAM, good for demo)
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Option B — 3B (better quality, ~6 GB VRAM, recommended for final run)
# GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ── demo size ─────────────────────────────────────────────────────────────────
N_DEMO = 50

for p in [KB_PATH, ASPECTS_PATH]:
    assert os.path.exists(p), f"Missing: {p}"

print(f"✓ Paths verified | Model: {GEN_MODEL} | n={N_DEMO}")
'''

# ═════════════════════════════════════════════════════════════════════════════
# CELL 4 — run demo
# ═════════════════════════════════════════════════════════════════════════════
CELL_4 = '''
from instruct_generator import demo_run

results_df = demo_run(
    aspects_path = ASPECTS_PATH,
    kb_path      = KB_PATH,
    n            = N_DEMO,
    model_name   = GEN_MODEL,
    out_dir      = OUT_DIR,
)
'''

# ═════════════════════════════════════════════════════════════════════════════
# CELL 5 — inspect results inline
# ═════════════════════════════════════════════════════════════════════════════
CELL_5 = '''
import pandas as pd

results_df = pd.read_csv(f"{OUT_DIR}/demo_{N_DEMO}_results.csv")

print("\\n=== OVERALL METRICS ===")
print(results_df[["section_coverage","entity_coverage","tone_match","overall_score"]].describe().round(3))

print("\\n=== TOP 5 (best overall score) ===")
top5 = results_df.nlargest(5, "overall_score")[
    ["ticket_idx","prob_sub","overall_score","section_coverage","response"]
]
for _, r in top5.iterrows():
    print(f"  [{int(r.ticket_idx):03d}] score={r.overall_score:.3f}  prob={r.prob_sub[:50]}")
    print(f"        {r.response[:120]}...")
    print()

print("\\n=== BOTTOM 5 (worst overall score) ===")
bot5 = results_df.nsmallest(5, "overall_score")[
    ["ticket_idx","prob_sub","overall_score","missing_entities","response"]
]
for _, r in bot5.iterrows():
    print(f"  [{int(r.ticket_idx):03d}] score={r.overall_score:.3f}  missing={r.missing_entities}")
    print(f"        prob={r.prob_sub[:50]}")
    print(f"        {r.response[:120]}...")
    print()

print("\\n=== FULL REPORT PATH ===")
print(f"  {OUT_DIR}/demo_{N_DEMO}_report.txt")
print("  (download from Kaggle output or open in notebook with:)")
print(f"  !head -120 {OUT_DIR}/demo_{N_DEMO}_report.txt")
'''

# ═════════════════════════════════════════════════════════════════════════════
# CELL 6 — side-by-side comparison vs T5 (optional)
# ═════════════════════════════════════════════════════════════════════════════
CELL_6 = '''
"""
Optional: run the old T5 response_generator on the same 50 tickets
and compare metrics side-by-side.
"""
import pandas as pd
import sys

sys.path.insert(0, "/kaggle/working/repo/pipeline")
from response_generator import run as t5_run

t5_df = t5_run(
    aspects_path = ASPECTS_PATH,
    kb_path      = KB_PATH,
    mode         = "rag",
    n            = N_DEMO,
)
t5_metrics = t5_df[["rouge1","rouge2","rougeL","bertscore","entity_coverage"]].mean()

instruct_df = pd.read_csv(f"{OUT_DIR}/demo_{N_DEMO}_results.csv")
instruct_metrics = instruct_df[["section_coverage","entity_coverage","tone_match","overall_score"]].mean()

print("\\n=== T5-large (RAG+Aspect, zero-shot) ===")
print(t5_metrics.round(3))

print("\\n=== Qwen2.5-Instruct (aspect-injection) ===")
print(instruct_metrics.round(3))
'''


# ─────────────────────────────────────────────────────────────────────────────
# When run directly, print all cells for easy copy-paste
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cells = {
        "CELL 1 — Pull repo": CELL_1,
        "CELL 2 — Install deps": CELL_2,
        "CELL 3 — Set paths": CELL_3,
        "CELL 4 — Run demo": CELL_4,
        "CELL 5 — Inspect results": CELL_5,
        "CELL 6 — Compare vs T5 (optional)": CELL_6,
    }
    for title, code in cells.items():
        print(f"\n{'#'*72}")
        print(f"# {title}")
        print(f"{'#'*72}")
        print(code)
