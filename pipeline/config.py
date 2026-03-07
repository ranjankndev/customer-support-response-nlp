"""
Configuration — single source of truth.
Change DATASET_FILENAME to switch between 3k / 28k / any other file.
"""

import os
import sys
import torch
from pathlib import Path

# ============================================================================
# ✅ CHANGE ONLY THIS LINE TO SWITCH DATASETS
# ============================================================================
DATASET_FILENAME = "customer_support_3k.xlsx"   # ← swap to 28k file here

# ============================================================================
# HuggingFace Hub — set your repo names here
# ============================================================================
HF_USERNAME            = "ranjankumarnayak"
HF_ASPECT_MODEL_REPO   = f"{HF_USERNAME}/customer-support-aspect-extractor-mt5"
HF_T5_MODEL_REPO       = f"{HF_USERNAME}/customer-support-t5-generator"

# ============================================================================
# ENV DETECTION
# ============================================================================

def _find_in_kaggle(slug: str) -> Path:
    kaggle_input = Path("/kaggle/input")
    if (kaggle_input / "datasets").exists():
        for p in (kaggle_input / "datasets").rglob(slug):
            return p if p.is_dir() else p.parent
    for p in kaggle_input.glob(slug):
        return p if p.is_dir() else p.parent
    raise FileNotFoundError(f"'{slug}' not found under /kaggle/input")


def _setup_code_path(slug: str):
    try:
        p = _find_in_kaggle(slug)
        if str(p) not in sys.path:
            sys.path.append(str(p))
    except FileNotFoundError:
        pass   # running locally — no action needed


IS_KAGGLE = os.path.exists("/kaggle")

# ============================================================================
# PATHS
# ============================================================================

if IS_KAGGLE:
    BASE_DIR  = Path("/kaggle/working")
    try:
        DATA_DIR = _find_in_kaggle("customer-support-3k")
    except FileNotFoundError:
        DATA_DIR = BASE_DIR / "data"
    _setup_code_path("inlp-project-code")
else:
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data"

# Dataset path — driven by DATASET_FILENAME above
DATASET_PATH       = DATA_DIR / DATASET_FILENAME

PROCESSED_DATA_DIR = BASE_DIR / "processed"
RESULTS_DIR        = BASE_DIR / "results"
MODELS_DIR         = BASE_DIR / "models"
ASPECT_MODEL_DIR   = MODELS_DIR / "aspect_extractor"
T5_MODEL_DIR       = MODELS_DIR / "t5_generator"

for d in [PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR,
          ASPECT_MODEL_DIR, T5_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD YAML CONFIG
# ============================================================================
try:
    from config_loader import load_config
    baseline_config = load_config(Path(__file__).parent / 'baseline.yaml')
except Exception:
    baseline_config = {}

# ============================================================================
# DATA SPLITS
# ============================================================================
TRAIN_SPLIT = baseline_config.get('data', {}).get('train_split', 0.8)
VAL_SPLIT   = baseline_config.get('data', {}).get('val_split',   0.1)
TEST_SPLIT  = baseline_config.get('data', {}).get('test_split',  0.1)
RANDOM_SEED = baseline_config.get('data', {}).get('random_seed', 42)

# ============================================================================
# MODEL NAMES
# ============================================================================
SENTENCE_BERT_MODEL  = baseline_config.get('model', {}).get(
    'sentence_bert', 'paraphrase-multilingual-mpnet-base-v2')
T5_MODEL             = 'google/flan-t5-base'   # better than t5-small for generation
ASPECT_EXTRACTOR_MODEL = 'google/mt5-small'    # multilingual — EN + DE

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
USE_GPU = torch.cuda.is_available()

BATCH_SIZE                = 8  if USE_GPU else 2
GRADIENT_ACCUMULATION_STEPS = 2 if USE_GPU else 8
MAX_INPUT_LENGTH          = 512
MAX_TARGET_LENGTH         = 256
LEARNING_RATE             = 5e-5
NUM_EPOCHS                = 10
WARMUP_STEPS              = 100
WEIGHT_DECAY              = 0.01

# Aspect extractor specific
ASPECT_EXTRACTOR_EPOCHS   = 5
ASPECT_EXTRACTOR_LR       = 3e-4
ASPECT_EXTRACTOR_BATCH    = 8 if USE_GPU else 4

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
NUM_BEAMS          = 4
TOP_K              = 50
TOP_P              = 0.95
TEMPERATURE        = 0.8
REPETITION_PENALTY = 1.2

# ============================================================================
# RAG
# ============================================================================
RAG_TOP_K              = baseline_config.get('retrieval', {}).get('top_k', 5)
RAG_SIMILARITY_THRESHOLD = 0.7

# ============================================================================
# EVALUATION FLAGS
# ============================================================================
COMPUTE_BERTSCORE       = True
COMPUTE_ROUGE           = True
COMPUTE_BLEU            = True
ASPECT_COVERAGE_ENABLED = True

# ============================================================================
# ASPECT KEYWORDS (fallback if extractor not trained yet)
# ============================================================================
ASPECT_KEYWORDS = {
    'problem':    ['problem','issue','error','crash','fail','not working','outage',
                   'Problem','Fehler','Absturz','Ausfall','Störung'],
    'cause':      ['because','due to','caused by','reason','after','since',
                   'weil','aufgrund','verursacht','infolge','nach'],
    'solution':   ['fix','resolve','solution','repair','solve','recommend','suggest',
                   'please try','restart','update','beheben','Lösung','empfehlen'],
    'prevention': ['prevent','avoid','ensure','future','monitor',
                   'verhindern','vermeiden','künftig','sicherstellen']
}

# ============================================================================
# METADATA / TAG FIELDS — auto-detected from dataset, these are defaults
# ============================================================================
METADATA_FIELDS = ['type', 'queue', 'priority', 'language']
TAG_FIELDS      = ['tag_1','tag_2','tag_3','tag_4','tag_5',
                   'tag_6','tag_7','tag_8']   # dataset has up to tag_8

# ============================================================================
# PII PATTERNS
# ============================================================================
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
    'url':   r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
}

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FILE  = RESULTS_DIR / 'training.log'

print(f"[Config] Dataset  : {DATASET_PATH}")
print(f"[Config] Device   : {'GPU' if USE_GPU else 'CPU'} | batch={BATCH_SIZE}")
print(f"[Config] Env      : {'Kaggle' if IS_KAGGLE else 'Local'}")
