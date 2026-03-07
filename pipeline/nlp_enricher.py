"""
nlp_enricher.py — spaCy NER, POS Filtering and BM25 Query Enrichment
======================================================================
INPUT  : spaCy language model (en_core_web_sm or de_core_news_sm)
         Raw text strings from ticket subject/body fields

OUTPUT : Enriched text, filtered query tokens, domain entity labels

WHAT THIS FILE DOES:
  1. build_entity_ruler()    — adds TOOL / VERSION / ERROR patterns to spaCy NER
                               so "ClickUp", "v14.2", "DB_TIMEOUT" are recognized
                               as named entities (not generic tokens or PERSON)
  2. pos_filter_phrases()    — filters KeyBERT keyphrases by POS composition;
                               removes verb-dominated phrases, keeps noun-heavy ones
  3. extract_noun_chunks()   — fallback problem-phrase extractor using spaCy
                               noun chunks scored by failure adjectives + position
  4. wsd_filter_query()      — WSD-style POS filter for BM25 query tokens;
                               removes "issue" when used as VERB, keeps as NOUN

CALLED BY:
  aspect_extractor_v3.py → build_entity_ruler()
  rag.py                 → wsd_filter_query()
  (pos_filter_phrases and extract_noun_chunks used by aspect_extractor v1/v2)

INTERNAL IMPORTS : logger.py  (same folder)
EXTERNAL IMPORTS : spaCy  (en_core_web_sm, de_core_news_sm must be installed)
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
log = get_logger('nlp_enricher')


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN TOOL LIST
# Same set as preprocessing.py — kept in sync manually.
# spaCy's default en_core_web_sm tags "ClickUp" as PERSON, "Elasticsearch"
# as ORG with low confidence.  Entity ruler overrides these with TOOL label.
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_TOOLS = [
    'MySQL','PostgreSQL','MongoDB','Redis','Kafka','Docker','Kubernetes',
    'AWS','Azure','GCP','Windows','Linux','MacOS','Android','iOS',
    'Chrome','Firefox','Safari','ClickUp','Smartsheet','Airtable','Zapier',
    'Alteryx','Elasticsearch','RapidMiner','DataRobot','SendGrid','Bitbucket',
    'DocuSign','Magento','Shopware','HubSpot','Salesforce','ServiceNow',
    'Zendesk','Freshdesk','Intercom','Jira','Confluence','ActiveCampaign',
    'Databricks','Snowflake','Tableau','Stripe','Twilio','Cloudflare',
    'Datadog','PagerDuty',
]

DE_DOMAIN_TERMS = [
    'Patientendaten','Sicherheitsmaßnahmen','Sicherheitsprotokolle',
    'SaaS-Plattform','Datenschutzrichtlinien','Systemintegration',
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOMAIN ENTITY RULER
# ─────────────────────────────────────────────────────────────────────────────

def build_entity_ruler(nlp, overwrite_ents: bool = False):
    """
    Add domain entity ruler to a spaCy pipeline.
    Call once after loading the language model — safe to call multiple times
    (skips silently if ruler already present).

    Adds three label types:
      TOOL    — SaaS platforms, databases, cloud services (case-insensitive)
      VERSION — version strings: v2.3, 8.0.1, v12
      ERROR   — error codes: DB_TIMEOUT, HTTP 500, ERR_404

    Args:
        nlp            : spaCy language model
        overwrite_ents : if True, ruler overrides default NER labels

    Returns: modified nlp pipeline
    """
    if 'entity_ruler' in nlp.pipe_names:
        return nlp

    ruler = nlp.add_pipe(
        'entity_ruler',
        before='ner',
        config={'overwrite_ents': overwrite_ents}
    )

    patterns = []

    for tool in DOMAIN_TOOLS:
        patterns.append({'label': 'TOOL', 'pattern': [{'LOWER': tool.lower()}]})
        # Also match "MySQL 8.0", "Elasticsearch 7"
        patterns.append({'label': 'TOOL', 'pattern': [
            {'LOWER': tool.lower()},
            {'TEXT': {'REGEX': r'\d+(?:\.\d+)?'}, 'OP': '?'}
        ]})

    for term in DE_DOMAIN_TERMS:
        patterns.append({'label': 'DE_DOMAIN', 'pattern': [{'LOWER': term.lower()}]})

    patterns.append({'label': 'VERSION', 'pattern': [{'TEXT': {'REGEX': r'v?\d+\.\d+(?:\.\d+)?'}}]})
    patterns.append({'label': 'ERROR',   'pattern': [{'TEXT': {'REGEX': r'[A-Z]{2,}_[A-Z_]{2,}'}}]})
    patterns.append({'label': 'ERROR',   'pattern': [
        {'LOWER': {'IN': ['http', 'error', 'err']}},
        {'TEXT': {'REGEX': r'[45]\d{2}'}}
    ]})

    ruler.add_patterns(patterns)
    log.info(f'[EntityRuler] Added {len(patterns)} patterns '
             f'({len(DOMAIN_TOOLS)} tools, {len(DE_DOMAIN_TERMS)} DE terms)')
    return nlp


# ─────────────────────────────────────────────────────────────────────────────
# 2. POS-BASED PHRASE QUALITY FILTER  (used by aspect_extractor v1/v2)
#
# KeyBERT returns ANY keyphrase including verb-dominated phrases like
# "I am reaching out to report" — entirely verbs, useless as a problem span.
# This filter scores each phrase by POS composition and removes verb-heavy ones.
# ─────────────────────────────────────────────────────────────────────────────

_CONTENT_POS = {'NOUN', 'PROPN', 'ADJ', 'NUM'}
_NOISE_POS   = {'VERB', 'AUX', 'PRON', 'DET', 'CCONJ', 'SCONJ', 'PART'}

def pos_filter_phrases(nlp,
                       phrases: List[Tuple[str, float]],
                       min_content_ratio: float = 0.4) -> List[Tuple[str, float]]:
    """
    Filter KeyBERT phrases by POS content ratio.
    Removes verb-dominated phrases; boosts noun-heavy ones.

    Args:
        nlp               : spaCy language model
        phrases           : list of (phrase, score) tuples from KeyBERT
        min_content_ratio : minimum fraction of content-POS tokens required

    Returns: filtered and re-sorted list of (phrase, adjusted_score)
    """
    filtered = []
    for phrase, score in phrases:
        doc     = nlp(phrase)
        tokens  = [t for t in doc if not t.is_punct]
        if not tokens:
            continue
        content_ratio  = sum(1 for t in tokens if t.pos_ in _CONTENT_POS) / len(tokens)
        adjusted_score = score * (1.0 + 0.3 * content_ratio)
        if content_ratio >= min_content_ratio:
            filtered.append((phrase, adjusted_score))
    return sorted(filtered, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. NOUN CHUNK EXTRACTION — PROBLEM PHRASE FALLBACK  (used by aspect_extractor)
#
# When KeyBERT returns nothing useful, spaCy noun chunks are scored by:
#   - presence of failure adjectives (slow, failed, broken, offline…)
#   - overlap with subject tokens (subject anchors the problem domain)
#   - position in document (earlier = more likely to be the problem statement)
# ─────────────────────────────────────────────────────────────────────────────

_FAILURE_ADJS = {
    'en': {'slow','failed','broken','offline','missing','corrupt',
           'disconnected','unavailable','incorrect','invalid','critical'},
    'de': {'langsam','fehler','kaputt','offline','fehlend','beschädigt',
           'getrennt','falsch','ungültig','kritisch'},
}

def extract_noun_chunks(nlp,
                        text: str,
                        subject: str = '',
                        lang: str = 'en') -> Optional[str]:
    """
    Extract the best noun chunk as a problem-phrase fallback.
    Returns the highest-scoring chunk or None if nothing qualifies.
    """
    doc        = nlp(text[:400])
    fail_adjs  = _FAILURE_ADJS.get(lang, _FAILURE_ADJS['en'])
    subj_words = set(subject.lower().split())
    best_phrase, best_score = None, 0.0

    for i, chunk in enumerate(doc.noun_chunks):
        phrase = chunk.text.strip()
        if len(phrase) < 4 or len(phrase.split()) < 2:
            continue
        chunk_words = set(phrase.lower().split())
        score  = 0.0
        score += 1.5 if chunk_words & fail_adjs else 0.0
        score += len(chunk_words & subj_words) * 0.5
        score += max(0, (5 - i) * 0.1)
        score += 0.3 if any(t.pos_ == 'ADJ' for t in chunk) else 0.0
        if score > best_score:
            best_score, best_phrase = score, phrase

    return best_phrase[:120] if best_phrase and best_score > 0.3 else None


# ─────────────────────────────────────────────────────────────────────────────
# 4. WSD-STYLE QUERY FILTER FOR BM25  (used by rag.py)
#
# The word "issue" used as a VERB (to issue a refund) should be dropped
# from a BM25 query.  The same word used as a NOUN (a login issue) should
# be kept.  Standard stopword lists don't distinguish these — POS tagging does.
# ─────────────────────────────────────────────────────────────────────────────

_NOUN_ONLY_TERMS = {
    'issue','fix','error','fault','crash','drop','run',
    'load','charge','update','hang','fail','break','freeze',
}

def wsd_filter_query(nlp, text: str) -> str:
    """
    POS-based query filter for BM25 tokens.
    Removes ambiguous words when used as verbs; keeps them as nouns.
    Returns space-joined filtered lemmas for BM25 tokenization.
    """
    doc    = nlp(text[:300])
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lemma = token.lemma_.lower()
        pos   = token.pos_
        if lemma in _NOUN_ONLY_TERMS:
            if pos in ('NOUN', 'PROPN'):
                tokens.append(lemma)
            continue
        if pos in _CONTENT_POS:
            tokens.append(lemma)
    return ' '.join(tokens)
