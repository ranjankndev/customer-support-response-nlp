"""
preprocessing.py — Ticket Ingestion, Cleaning and Train/Val/Test Preparation
=============================================================================
INPUT  : Raw CSV or XLSX support ticket file
         Required columns : subject, body, answer
         Optional columns : language, priority, queue, type, version, tag_1..tag_8

OUTPUT : Cleaned DataFrame with additional columns:
           body_clean      — body with email filler content removed (used by VADER)
           body_entities   — regex-extracted versions, error codes, product names
           input_text      — formatted T5 training prompt
           target_text     — T5 training target (= answer column)
         Saved splits      — processed_train.csv / val.csv / test.csv / full.csv

WHAT THIS FILE DOES (in order):
  1. Load CSV/XLSX and auto-detect double-encoded German text (UTF-8 mojibake)
  2. Fix encoding in-place: 'fÃ¼r' → 'für' via codepoint map (no external lib needed)
  3. Strip email filler content from body — greetings, hope-lines, sign-offs
     (prevents "I hope this finds you well" from skewing VADER sentiment scores)
  4. Mask PII: emails, phone numbers, URLs, dataset placeholder tokens
  5. Extract domain entities via regex: version numbers, HTTP error codes, SaaS tools
  6. Build input_text / target_text fields for T5 fine-tuning
  7. Stratified 80/10/10 train/val/test split by language

INTERNAL IMPORTS : config.py, run_tracker.py, logger.py  (all in same folder)
EXTERNAL IMPORTS : pandas, numpy, scikit-learn  (standard ML stack, no spaCy needed)
                   ftfy (optional — fallback for edge-case encoding only)
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent))
try:
    from logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger(name)
log = get_logger('preprocessing')

try:
    from run_tracker import track_run
except ImportError:
    def track_run(**_): pass

try:
    from config import (DATASET_PATH, PROCESSED_DATA_DIR, RANDOM_SEED,
                        PII_PATTERNS, METADATA_FIELDS)
except ImportError:
    DATASET_PATH       = 'customer_support_28k_fixed.csv'
    PROCESSED_DATA_DIR = './processed'
    RANDOM_SEED        = 42
    PII_PATTERNS       = {
        'email': r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b',
        'phone': r'\b(?:\+?\d[\d\s\-().]{7,}\d)\b',
        'url':   r'https?://\S+|www\.\S+',
    }
    METADATA_FIELDS = ['type', 'queue', 'priority', 'language', 'version']


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENCODING FIX
#    German text double-encoded as UTF-8-read-as-Latin-1 is the primary
#    data quality issue in this corpus.
#
#    WHY codepoint map instead of encode/decode roundtrip:
#      Old: x.encode('latin-1').decode('utf-8')
#           → raises UnicodeEncodeError on any char outside latin-1 range
#      New: direct U+00C3 U+00BC → ü substitution
#           → deterministic, never raises, handles all German umlauts correctly
# ─────────────────────────────────────────────────────────────────────────────

_MOJIBAKE_MAP = {
    '\u00c3\u00bc': 'ü',  '\u00c3\u00b6': 'ö',
    '\u00c3\u00a4': 'ä',  '\u00c3\u009f': 'ß',
    '\u00c3\u009c': 'Ü',  '\u00c3\u0096': 'Ö',
    '\u00c3\u0084': 'Ä',  '\u00c3\u00a9': 'é',
    '\u00c3\u00a8': 'è',
}
_MOJIBAKE_RE = re.compile(
    '|'.join(re.escape(k) for k in sorted(_MOJIBAKE_MAP, key=len, reverse=True))
)

def fix_encoding(text) -> str:
    """Fix double-encoded UTF-8 mojibake.  'fÃ¼r' → 'für'"""
    if not isinstance(text, str):
        return str(text) if text is not None else ''
    fixed = _MOJIBAKE_RE.sub(lambda m: _MOJIBAKE_MAP[m.group(0)], text)
    if fixed != text:
        return fixed
    try:                            # ftfy handles remaining edge cases
        import ftfy
        return ftfy.fix_text(text)
    except ImportError:
        pass
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMAIL FILLER CONTENT REMOVAL
#    Support emails contain openers and closers that carry zero ticket
#    information but heavily distort VADER sentiment scores.
#
#    Example problem:
#      "I hope this message finds you well. Our entire payment service is DOWN."
#      VADER compound on full text: +0.45  (Positive — WRONG)
#      VADER compound after strip:  -0.82  (Frustrated/Urgent — CORRECT)
#
#    Patterns cover EN and DE greetings, hope-lines, and sign-off phrases.
#    Stripping is position-aware — only removes from the opening of the body.
# ─────────────────────────────────────────────────────────────────────────────

_OPENER_EN = re.compile(
    r'^(?:dear|hello|hi|good\s+(?:morning|afternoon|evening))'
    r'\s*[\w\s,.\-]{0,35}[,.]?\s*',
    re.I | re.MULTILINE
)
_OPENER_DE = re.compile(
    r'^(?:sehr\s+geehrte[rs]?|guten\s+(?:morgen|tag|abend)'
    r'|hallo|liebe[rs]?)\s*[\w\s,.\-]{0,35}[,.]?\s*',
    re.I | re.MULTILINE
)
_FILLER_EN = re.compile(
    r'(?:^|\.\s+)(?:'
    r'I\s+hope\s+(?:this|you|my)[^.]*\.|'
    r'I\s+am\s+(?:writing|reaching\s+out)\s+to\s+(?:inform|request|inquire|report)[^.]*\.|'
    r'(?:Thank|Thanks)\s+you\s+for\s+(?:your\s+)?(?:time|support|assistance|help)[^.]*\.|'
    r'I\s+look\s+forward\s+to[^.]*\.|'
    r'Please\s+do\s+not\s+hesitate\s+to[^.]*\.|'
    r'Feel\s+free\s+to\s+contact[^.]*\.)',
    re.I
)
_FILLER_DE = re.compile(
    r'(?:^|\.\s+)(?:'
    r'Ich\s+hoffe[^.]*\.|'
    r'Ich\s+(?:schreibe|wende\s+mich)[^.]*\.|'
    r'(?:Vielen\s+Dank|Danke)[^.]*\.|'
    r'Mit\s+(?:freundlichen|besten)\s+Grüßen[^.]*\.)',
    re.I
)

def strip_email_filler(text: str, lang: str = 'en') -> str:
    """
    Remove opening greetings and content-free filler sentences from email body.
    Returns substantive content only.  Safe to call on already-clean text.
    """
    if not text or not text.strip():
        return ''
    lang  = (lang or 'en')[:2].lower()
    clean = text.strip()
    if lang == 'de':
        clean = _OPENER_DE.sub('', clean).strip()
        clean = _FILLER_DE.sub('. ', clean)
    else:
        clean = _OPENER_EN.sub('', clean).strip()
        clean = _FILLER_EN.sub('. ', clean)
    clean = re.sub(r'\.\s*\.', '.', clean)
    clean = re.sub(r'^[-,.:;\s]+', '', re.sub(r'\s+', ' ', clean)).strip()
    return clean if len(clean) > 10 else text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. DOMAIN ENTITY PATTERNS
#    Regex-only extraction — no spaCy dependency here.
#    spaCy entity ruler (TOOL, VERSION, ERROR labels) lives in nlp_enricher.py
#    and is used at retrieval time by rag.py, not at preprocessing time.
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_TOOLS = (
    'MySQL','PostgreSQL','MongoDB','Redis','Kafka','Docker','Kubernetes',
    'AWS','Azure','GCP','Windows','Linux','MacOS','Android','iOS',
    'Chrome','Firefox','Safari','ClickUp','Smartsheet','Airtable','Zapier',
    'Alteryx','Elasticsearch','RapidMiner','DataRobot','SendGrid','Bitbucket',
    'DocuSign','Magento','Shopware','HubSpot','Salesforce','ServiceNow',
    'Zendesk','Freshdesk','Intercom','Jira','Confluence','ActiveCampaign',
    'Databricks','Snowflake','Tableau','Stripe','Twilio','Cloudflare',
    'Datadog','PagerDuty',
)
_TOOLS_RE    = re.compile(r'\b(' + '|'.join(re.escape(t) for t in _DOMAIN_TOOLS) + r')\b', re.I)
_DE_TERMS_RE = re.compile(
    r'\b(Patientendaten|Sicherheitsma[sß]nahmen|Sicherheitsprotokolle|'
    r'SaaS-Plattform|Datenschutzrichtlinien|Systemintegration)\b', re.I
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN PREPROCESSOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessor:
    """
    End-to-end preprocessing for customer support tickets.

    Typical usage:
        prep   = DataPreprocessor('customer_support_28k_fixed.csv')
        splits = prep.save_processed_data()          # → ./processed/

    Or step-by-step:
        df     = prep.load_data()
        df     = prep.preprocess(df)
        splits = prep.prepare_for_training(df)

    Inference (single ticket):
        result = prep.preprocess_inference_input(subject, body, language)
    """

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or DATASET_PATH
        self.df = None

    # ── Load ─────────────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        path = str(self.dataset_path)
        log.info(f'Loading data from {path}...')
        self.df = pd.read_excel(path) if path.endswith(('.xlsx', '.xls')) \
                  else pd.read_csv(path)

        for col in ['subject', 'body', 'answer']:
            if col not in self.df.columns:
                continue
            n_garbled = sum(1 for t in self.df[col].fillna('') if '\u00c3' in str(t))
            if n_garbled > 0:
                log.warning(f'[Encoding] {n_garbled} garbled rows in "{col}" — fixing')
                self.df[col] = self.df[col].fillna('').apply(
                    lambda x: fix_encoding(x) if '\u00c3' in str(x) else str(x)
                )
        log.info(f'Loaded {len(self.df)} tickets | columns: {self.df.columns.tolist()}')
        return self.df

    # ── Clean helpers ─────────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ''
        text = ' '.join(str(text).split())
        return re.sub(r'[^\w\s.,!?;:\-äöüÄÖÜß]', '', text).strip()

    def _mask_pii(self, text: str) -> str:
        if pd.isna(text):
            return ''
        text = re.sub(PII_PATTERNS['email'], '<email>', text)
        text = re.sub(PII_PATTERNS['phone'], '<phone>', text)
        text = re.sub(PII_PATTERNS['url'],   '<url>',   text)
        text = re.sub(r'\btel_num\b', '<phone>',   text, flags=re.I)
        text = re.sub(r'\bacc_num\b', '<account>', text, flags=re.I)
        return text

    def _extract_entities(self, text: str) -> Dict:
        return {
            'versions':    list(set(re.findall(r'\bv?\d+\.\d+(?:\.\d+)?\b', text, re.I))),
            'error_codes': list(set(
                re.findall(r'\b(?:error|err|code|errno)[:\s#]*(\w+)\b', text, re.I)
                + re.findall(r'\b[A-Z]{2,}_[A-Z_]{2,}\b', text)
                + re.findall(r'\bHTTP\s*[45]\d{2}\b', text, re.I)
            )),
            'products':    list(set(p.lower() for p in _TOOLS_RE.findall(text))),
            'de_domain':   list(set(_DE_TERMS_RE.findall(text))),
        }

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = (self.load_data() if self.df is None else self.df).copy()
        else:
            df = df.copy()

        log.info('Cleaning text...')
        for col in ['subject', 'body', 'answer']:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_text)

        log.info('Masking PII...')
        for col in ['body', 'answer']:
            if col in df.columns:
                df[col] = df[col].apply(self._mask_pii)

        log.info('Stripping email filler content...')
        if 'body' in df.columns:
            lang_series = df['language'] if 'language' in df.columns \
                          else pd.Series(['en'] * len(df))
            df['body_clean'] = [
                strip_email_filler(body, lang)
                for body, lang in zip(df['body'], lang_series)
            ]

        log.info('Extracting entities...')
        if 'body' in df.columns:
            df['body_entities'] = df['body'].apply(lambda t: str(self._extract_entities(t)))

        for col in [c for c in df.columns if c.startswith('tag_')]:
            df[col] = df[col].fillna('')

        df = df[(df['body'].str.len() > 0) & (df['answer'].str.len() > 0)]
        log.info(f'After preprocessing: {len(df)} tickets')
        return df

    # ── Training preparation ──────────────────────────────────────────────────

    def _metadata_string(self, row: pd.Series) -> str:
        parts = []
        for field in METADATA_FIELDS:
            if field in row and pd.notna(row[field]) and row[field] != '':
                parts.append(f"{field.capitalize()}: {row[field]}")
        tags = [str(row[t]) for t in row.index
                if t.startswith('tag_') and pd.notna(row[t]) and str(row[t]) not in ('', 'nan')]
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        return ' | '.join(parts)

    def prepare_for_training(self, df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        if df is None:
            df = self.preprocess()

        df['input_text'] = df.apply(
            lambda r: (f"Generate customer support response:\n"
                       f"Subject: {r['subject']}\n"
                       f"Complaint: {r['body']}\n"
                       f"{self._metadata_string(r)}"),
            axis=1
        )
        df['target_text'] = df['answer']

        stratify = df['language'] if 'language' in df.columns else None
        train_df, temp_df = train_test_split(
            df, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
        )
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)
        log.info(f'Split → train:{len(train_df)} | val:{len(val_df)} | test:{len(test_df)}')
        return {'train': train_df, 'val': val_df, 'test': test_df, 'full': df}

    def preprocess_inference_input(self, subject: str, body: str,
                                   language: str = 'en') -> Dict:
        """Apply identical cleaning to a single ticket at inference time."""
        subject    = fix_encoding(str(subject or ''))
        body       = fix_encoding(str(body    or ''))
        clean_subj = self._clean_text(subject)
        clean_body = self._mask_pii(self._clean_text(body))
        return {
            'subject':    clean_subj,
            'body':       clean_body,
            'body_clean': strip_email_filler(clean_body, language),
            'entities':   self._extract_entities(clean_body),
        }

    def save_processed_data(self, output_dir: str = None) -> Dict[str, pd.DataFrame]:
        output_dir = Path(output_dir or PROCESSED_DATA_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        splits = self.prepare_for_training()
        splits['full'].to_csv(output_dir / 'processed_full.csv', index=False)
        for name, sdf in splits.items():
            if name != 'full':
                sdf.to_csv(output_dir / f'processed_{name}.csv', index=False)
        log.info(f'Saved processed data → {output_dir}')
        track_run(stage='preprocessing', metrics={
            'total_samples': len(splits['full']),
            'train_samples': len(splits['train']),
            'val_samples':   len(splits['val']),
            'test_samples':  len(splits['test']),
        })
        return splits


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data',   required=True, help='Path to CSV or XLSX ticket file')
    p.add_argument('--output', default=None,  help='Output directory (default: ./processed)')
    args = p.parse_args()
    splits = DataPreprocessor(args.data).save_processed_data(args.output)
    print(f"Done. train={len(splits['train'])} | val={len(splits['val'])} | test={len(splits['test'])}")
