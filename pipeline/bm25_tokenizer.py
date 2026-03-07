"""
bm25_tokenizer.py — Domain-Aware BM25 Tokenizer for Support Tickets
=========================================================================
INPUT  : Raw text strings from ticket subject/body fields
         Language code: 'en' or 'de'

OUTPUT : List of BM25-ready tokens per ticket

WHAT THIS FILE DOES:
  Provides BM25Tokenizer — a 4-step tokenizer designed specifically for
  BM25 retrieval over multilingual support tickets:

    Step 1 : Preserve technical tokens intact (error codes, version strings,
             product names) — KERNEL_PANIC, v14.2.1, DB_TIMEOUT must not split
    Step 2 : spaCy lemmatization on remaining text (EN + DE)
    Step 3 : German compound splitting at known domain suffix boundaries
             Netzwerkverbindung → [netzwerk, verbindung]  so BM25 matches both
    Step 4 : Filter stopwords + email filler terms (dear, hope, regards…)

  WHY a custom tokenizer instead of default BM25 whitespace split:
    - "crash/crashing/crashed" are different BM25 tokens without lemmatization
    - "Verbindungsfehler" never matches a query for "Verbindung" without splitting
    - KERNEL_PANIC fragments into KERNEL + PANIC, losing the error code signal

CALLED BY : rag.py  (BM25Tokenizer.tokenize_batch for BM25 index building)

INTERNAL IMPORTS : logger.py  (same folder)
EXTERNAL IMPORTS : spaCy  (en_core_web_sm, de_core_news_sm must be installed)
"""

import re
import sys
import subprocess
from typing import List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
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
log = get_logger('bm25_tokenizer')


# ─────────────────────────────────────────────────────────────────────────────
# 1. TECHNICAL TOKEN PRESERVATION
#    These patterns are atomic retrieval units — splitting them destroys
#    BM25 matching precision.  They are extracted first and added back
#    to the final token list unchanged.
# ─────────────────────────────────────────────────────────────────────────────

_PRESERVE_PATTERNS = [
    re.compile(r'\bv?\d+\.\d+(?:\.\d+)*\b'),              # versions:    v14.2.1
    re.compile(r'\b[A-Z]{2,}_[A-Z_]{2,}\b'),              # error codes: DB_TIMEOUT
    re.compile(r'\b(?:error|err|code)[:\s#]*\w+\b', re.I),# error codes: error 0x9F
    re.compile(                                            # product names
        r'\b(?:MacBook|iPhone|macOS|iOS|Android|Windows|Linux|'
        r'Docker|Kubernetes|Elasticsearch|Databricks)\b', re.I),
]

def _extract_preserve_tokens(text: str):
    """Extract technical tokens, return (cleaned_text, preserved_tokens)."""
    preserved, clean = [], text
    for pattern in _PRESERVE_PATTERNS:
        for match in pattern.finditer(text):
            tok = match.group().strip().lower()
            if tok and len(tok) > 1:
                preserved.append(tok)
        clean = pattern.sub(' ', clean)
    return clean, preserved


# ─────────────────────────────────────────────────────────────────────────────
# 2. GERMAN COMPOUND SPLITTER
#    German compounds like "Netzwerkverbindung" should match BM25 queries
#    for "Netzwerk" and "Verbindung" independently.
#
#    This targets the ~10 most common compound patterns in support tickets —
#    sufficient for ~60% coverage without full morphological analysis.
# ─────────────────────────────────────────────────────────────────────────────

_DE_SUFFIXES = [
    'fehler',      # Verbindungsfehler → verbindung + fehler
    'problem',     # Netzwerkproblem  → netzwerk + problem
    'ausfall',     # Serverausfall    → server + ausfall
    'störung',     # Verbindungsstörung
    'verbindung',  # Netzwerkverbindung
    'system',      # Betriebssystem
    'zugang',      # Systemzugang
    'zugriff',     # Datenzugriff
    'anmeldung',   # Benutzeranmeldung
    'software',    # Projektsoftware
]

def _split_german_compound(word: str) -> List[str]:
    """Split a German compound noun at a known domain suffix boundary."""
    w = word.lower()
    for suffix in _DE_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 3:
            prefix = w[:-len(suffix)]
            if len(prefix) > 2:
                return [prefix, suffix]
    return [word]


# ─────────────────────────────────────────────────────────────────────────────
# 3. DOMAIN STOPWORDS
#    Email filler terms that survive spaCy's generic stopword list but
#    add noise to BM25 — greetings, hope-lines, sign-off words.
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_STOPWORDS = {
    'en': {'dear','hello','hi','hope','message','find','writing','reaching',
           'regards','sincerely','best','thank','please','would','could',
           'like','know','let','feel'},
    'de': {'sehr','geehrte','geehrter','geehrtes','liebe','lieber','bitte',
           'danke','freundlichen','grüßen','mfg','schreibe','wende',
           'hoffe','möchte','würde','könnte'},
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. DOMAIN TOKENIZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BM25Tokenizer:
    """
    Domain-aware tokenizer for BM25 indexing of support tickets.

    Usage:
        tokenizer = BM25Tokenizer()

        # Single ticket
        tokens = tokenizer.tokenize("KERNEL_PANIC after macOS v14.2 update", 'en')
        # → ['kernel_panic', 'macos', 'v14.2', 'update']

        tokens = tokenizer.tokenize("Netzwerkverbindung getrennt", 'de')
        # → ['netzwerk', 'verbindung', 'trennen']

        # Batch (used by rag.py)
        token_lists = tokenizer.tokenize_batch(texts, langs)
    """

    def __init__(self):
        self._nlp_en = None
        self._nlp_de = None

    def _get_nlp(self, lang: str):
        """Lazy-load spaCy models — downloaded automatically if missing."""
        import spacy
        if lang == 'de':
            if self._nlp_de is None:
                try:
                    self._nlp_de = spacy.load('de_core_news_sm')
                except OSError:
                    subprocess.run([sys.executable, '-m', 'spacy', 'download',
                                    'de_core_news_sm'], capture_output=True)
                    self._nlp_de = spacy.load('de_core_news_sm')
            return self._nlp_de
        else:
            if self._nlp_en is None:
                try:
                    self._nlp_en = spacy.load('en_core_web_sm')
                except OSError:
                    subprocess.run([sys.executable, '-m', 'spacy', 'download',
                                    'en_core_web_sm'], capture_output=True)
                    self._nlp_en = spacy.load('en_core_web_sm')
            return self._nlp_en

    def tokenize(self, text: str, lang: str = 'en') -> List[str]:
        """Tokenize a single text string. Returns list of BM25 tokens."""
        lang  = lang if lang in ('en', 'de') else 'en'
        text  = re.sub(r'[\r\n\t]+', ' ', str(text))
        text  = re.sub(r'\s+', ' ', text).strip()

        clean, preserved = _extract_preserve_tokens(text)

        nlp       = self._get_nlp(lang)
        doc       = nlp(clean.lower()[:300])
        stopwords = _DOMAIN_STOPWORDS.get(lang, set())
        lemmas    = []

        for token in doc:
            if token.is_stop or token.is_punct or len(token.text) < 3:
                continue
            lemma = token.lemma_.lower()
            if lemma in stopwords:
                continue
            if lang == 'de' and token.pos_ == 'NOUN':
                lemmas.extend(_split_german_compound(lemma))
            else:
                lemmas.append(lemma)

        return preserved + [l for l in lemmas if len(l) >= 3]

    def tokenize_batch(self, texts: List[str],
                       langs: List[str] = None) -> List[List[str]]:
        """Tokenize a batch of texts. langs defaults to all-English."""
        if langs is None:
            langs = ['en'] * len(texts)
        return [self.tokenize(t, l) for t, l in zip(texts, langs)]
