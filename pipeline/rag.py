"""
rag.py — Hybrid Aspect-Aware RAG for Customer Support
======================================================

RETRIEVAL PIPELINE:
  1. build_index()  — SBERT -> FAISS + BM25, per language
  2. retrieve()     — BM25 + FAISS -> RRF -> aspect+entity rerank
  3. build_prompt() — structured aspect-slot prompt for T5

NOVEL CONTRIBUTIONS (our own logic, clearly marked [OUR]):
  [A] aspect_query_vector()  weighted encoding of extracted spans
  [B] rrf_fusion()           Reciprocal Rank Fusion of BM25+FAISS
  [C] aspect_score()         cosine sim between query aspects and answer
  [D] build_prompt()         aspect-slot structured T5 prompt template
  [E] entity_overlap()       Jaccard over version/error/product entities

LIBRARY CALLS (clearly marked [LIB]):
  [LIB] SentenceTransformer.encode()  dense vector encoding
  [LIB] BM25Okapi.get_scores()        sparse keyword scoring
  [LIB] faiss.IndexFlatIP             ANN search index
  [LIB] CrossEncoder.predict()        cross-encoder reranking

CHANGES vs previous version:
  - SBERT loading: uses finetuned_sbert/ if available, falls back to pretrained
  - extract_entities(): expanded product list from OOV analysis (18 new tools)
  - No other changes — RRF, aspect scoring, FAISS, BM25 all unchanged
"""

import re
import sys
import numpy as np
import pandas as pd
import faiss                                          # [LIB] ANN search
import torch
from pathlib import Path
from typing import Dict, List
from rank_bm25 import BM25Okapi                       # [LIB] sparse retrieval
from sentence_transformers import SentenceTransformer # [LIB] dense encoding
from sentence_transformers import CrossEncoder        # [LIB] reranking

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

log = get_logger('rag')
try:
    from config import METADATA_FIELDS
except ImportError:
    METADATA_FIELDS = ['language', 'ticket_type', 'queue', 'priority']

BODY_LEN   = 800   # max chars from ticket body used in retrieval
ANSWER_LEN = 800   # max chars from answer used in reranking + prompt

# Domain tokenizer for BM25 — lemmatization + compound split [OUR]
try:
    from bm25_tokenizer import BM25Tokenizer
    _DOMAIN_TOKENIZER = BM25Tokenizer()
except ImportError:
    _DOMAIN_TOKENIZER = None

# WSD-style query filter for BM25 — POS disambiguation [OUR C]
try:
    from nlp_enricher import wsd_filter_query, severity_score
    from aspect_extractor import ModelRegistry
    _NLP_ENRICHER_AVAILABLE = True
except ImportError:
    _NLP_ENRICHER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SBERT MODEL LOADER — fine-tuned first, pretrained fallback
#
# After sbert_comparison.py finishes, finetuned_sbert/ exists and is used.
# Before that (or if fine-tuning was skipped), falls back to pretrained.
# ─────────────────────────────────────────────────────────────────────────────

_PRETRAINED_SBERT = 'paraphrase-multilingual-mpnet-base-v2'
_FINETUNED_DIRS = [
    Path(__file__).parent / 'finetuned_sbert',                           # local
    Path('/kaggle/input/datasets/ranjankumarnayak/finetuned-sbert-customer-support/finetuned_sbert'),  # Kaggle
]
def _resolve_sbert_model(config: dict = None) -> str:
    for d in _FINETUNED_DIRS:
        if d.exists() and any(d.iterdir()):
            log.info(f'[RAG] Loading fine-tuned SBERT from {d}')
            return str(d)
    fallback = (config or {}).get('model', {}).get(
        'sentence_bert', _PRETRAINED_SBERT
    )
    log.info(f'[RAG] Fine-tuned SBERT not found -- using pretrained: {fallback}')
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY PATTERNS — updated from OOV analysis
#
# Original only caught: MySQL, PostgreSQL, Redis, Kafka, Docker, Kubernetes,
#   Windows, Linux, MacOS, Android, iOS, Chrome, Firefox, Safari
#
# Added from OOV analysis (high-frequency tools missing from GloVe vocab):
#   ClickUp, Smartsheet, Airtable, Zapier, Alteryx, Elasticsearch,
#   RapidMiner, DataRobot, SendGrid, Bitbucket, DocuSign, Magento,
#   Shopware, HubSpot, Salesforce, ServiceNow, Zendesk, Databricks
#
# Kept in sync with preprocessing.py _DOMAIN_TOOLS list.
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_PRODUCTS = (
    # Original
    'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Kafka',
    'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP',
    'Windows', 'Linux', 'MacOS', 'Android', 'iOS',
    'Chrome', 'Firefox', 'Safari',
    # Added from OOV analysis
    'ClickUp', 'Smartsheet', 'Airtable', 'Zapier', 'Alteryx',
    'Elasticsearch', 'RapidMiner', 'DataRobot', 'SendGrid',
    'Bitbucket', 'DocuSign', 'Magento', 'Shopware',
    'HubSpot', 'Salesforce', 'ServiceNow', 'Zendesk',
    'Freshdesk', 'Intercom', 'Jira', 'Confluence',
    'ActiveCampaign', 'Databricks', 'Snowflake', 'Tableau',
    'Stripe', 'Twilio', 'Cloudflare', 'Datadog', 'PagerDuty',
)
_PRODUCTS_RE = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in _ENTITY_PRODUCTS) + r')\b', re.I
)


# ─────────────────────────────────────────────────────────────────────────────
# [OUR A] ASPECT-WEIGHTED QUERY ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def aspect_query_vector(sbert,
                        subject: str,
                        spans: Dict[str, str]) -> np.ndarray:
    """[OUR A] Weighted combination of aspect span embeddings as query."""
    parts, weights = [], []

    if spans.get('prob_sub', 'none') != 'none':
        parts.append(spans['prob_sub']);       weights.append(2.0)
    if spans.get('prob_statement', 'none') != 'none':
        parts.append(spans['prob_statement']); weights.append(1.5)
    if spans.get('cause', 'none') != 'none':
        parts.append(spans['cause']);          weights.append(1.0)

    parts.append(subject or 'support ticket')
    weights.append(1.0)

    vecs = sbert.encode(parts, convert_to_numpy=True)         # [LIB]
    vec  = np.average(vecs, axis=0,
                      weights=np.array(weights[:len(vecs)], dtype='float32'))
    vec  = vec.astype('float32')
    vec /= (np.linalg.norm(vec) + 1e-9)
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# [OUR B] RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fusion(dense_idx: List[int],
               sparse_idx: List[int],
               k: int = 60) -> List[int]:
    """[OUR B] Merge BM25 and FAISS ranked lists via RRF."""
    scores: Dict[int, float] = {}
    for rank, i in enumerate(dense_idx):
        scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    for rank, i in enumerate(sparse_idx):
        scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# [OUR C] ASPECT SIMILARITY RERANKING SCORE
# ─────────────────────────────────────────────────────────────────────────────

def aspect_score(sbert,
                 spans: Dict[str, str],
                 answer: str) -> float:
    """[OUR C] Cosine similarity between query aspects and retrieved answer."""
    asp_text = ' '.join(
        v for k, v in spans.items()
        if k in ('prob_sub', 'prob_statement', 'cause') and v and v != 'none'
    ).strip()
    if not asp_text or not answer.strip():
        return 0.0
    vecs = sbert.encode([asp_text, answer[:ANSWER_LEN]],   # [LIB]
                        convert_to_numpy=True)
    return float(max(0.0,
        np.dot(vecs[0], vecs[1]) /
        (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]) + 1e-9)
    ))


# ─────────────────────────────────────────────────────────────────────────────
# [OUR E] ENTITY EXTRACTION AND OVERLAP
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    [OUR E] Extract version numbers, error codes, product names.
    Product list updated from OOV analysis — now catches 28 additional
    SaaS tools that were missing from the original regex.
    """
    return {
        'versions': re.findall(r'\bv?\d+\.\d+(?:\.\d+)?\b', text, re.I),
        'error_codes': (
            re.findall(r'\b(?:error|err|code)[:\s#]*(\w+)\b', text, re.I)
            + re.findall(r'\b[A-Z]{2,}_[A-Z_]{2,}\b', text)
            + re.findall(r'\bHTTP\s*[45]\d{2}\b', text, re.I)
        ),
        'products': list(set(
            p.lower() for p in _PRODUCTS_RE.findall(text)
        )),
    }


def entity_overlap(q: Dict, kb: Dict) -> float:
    """[OUR E] Jaccard similarity across all entity types."""
    s1 = {v for vals in q.values()  for v in vals}
    s2 = {v for vals in kb.values() for v in vals}
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


# ─────────────────────────────────────────────────────────────────────────────
# RAG SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class RAGSystem:
    """
    Hybrid Aspect-Aware RAG.
    Index  : per-language FAISS + BM25
    Query  : [OUR A] aspect vector -> [OUR B] RRF -> [OUR C] aspect rerank
    Prompt : [OUR D] structured aspect-slot template
    """

    RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    def __init__(self, config: dict = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info(f'[RAG] device={self.device}')

        if config is None:
            config = {}   # use defaults — top_k=5, no reranker

        # Fine-tuned SBERT if available, pretrained fallback
        sbert_name    = _resolve_sbert_model(config)
        self.sbert    = SentenceTransformer(sbert_name,       # [LIB]
                                            device=self.device)
        self.top_k        = config.get('retrieval', {}).get('top_k', 5)
        self.use_reranker = config.get('retrieval', {}).get('reranker', False)

        self.faiss_index: Dict[str, faiss.Index]  = {}
        self.bm25_index:  Dict[str, BM25Okapi]    = {}
        self.kb_df:       Dict[str, pd.DataFrame] = {}
        self.reranker     = None

    # ── KB TEXT ───────────────────────────────────────────────────────────────

    def _kb_text(self, row: pd.Series) -> str:
        """[OUR] Rich KB text: subject + body + tag columns (flat, for BM25)."""
        tags = ' '.join(
            str(row[c]) for c in row.index
            if c.startswith('tag_') and pd.notna(row[c])
            and str(row[c]) != 'nan'
        )
        return f"{row.get('subject','')} {row.get('body','')} {tags}".strip()

    def _kb_field_vecs(self, row: pd.Series, sbert) -> np.ndarray:
        """
        [OUR] Field-weighted KB vector: subject x2, tags x1.5, body x0.5.
        Encodes each field separately so subject dominates retrieval signal.
        """
        parts, weights = [], []

        subj = str(row.get('subject', '')).strip()
        body = str(row.get('body', ''))[:BODY_LEN].strip()
        tags = [str(row[c]) for c in row.index
                if c.startswith('tag_') and pd.notna(row[c])
                and str(row[c]) not in ('nan', '')]

        if subj:  parts.append(subj);           weights.append(2.0)
        if tags:  parts.append(' '.join(tags));  weights.append(1.5)
        if body:  parts.append(body);            weights.append(0.5)
        if not parts:
            parts, weights = ['support ticket'], [1.0]

        vecs = sbert.encode(parts, convert_to_numpy=True,   # [LIB]
                             show_progress_bar=False)
        w    = np.array(weights[:len(vecs)], dtype='float32')
        vec  = np.average(vecs, axis=0, weights=w).astype('float32')
        vec /= (np.linalg.norm(vec) + 1e-9)
        return vec

    # ── BUILD INDEX ───────────────────────────────────────────────────────────

    def build_index(self, kb_df: pd.DataFrame):
        """
        Build per-language FAISS (dense) + BM25 (sparse) indices.
        Uses fine-tuned SBERT embeddings if finetuned_sbert/ exists.
        """
        for lang, group in kb_df.groupby('language'):
            lang  = str(lang).lower()[:2]
            lang  = lang if lang in ('en', 'de') else 'en'
            group = group.reset_index(drop=True)
            texts = [self._kb_text(r) for _, r in group.iterrows()]

            # [OUR] Field-weighted KB encoding
            vecs = np.vstack([
                self._kb_field_vecs(r, self.sbert)
                for _, r in group.iterrows()
            ]).astype('float32')

            # [LIB] FAISS flat inner-product index
            idx = faiss.IndexFlatIP(vecs.shape[1])    # [LIB]
            idx.add(vecs)                             # [LIB]
            self.faiss_index[lang] = idx

            # [OUR] Domain tokenizer for BM25
            if _DOMAIN_TOKENIZER is not None:
                bm25_corpus = _DOMAIN_TOKENIZER.tokenize_batch(
                    texts, langs=[lang] * len(texts)
                )
            else:
                bm25_corpus = [t.lower().split() for t in texts]

            # [LIB] BM25 sparse index
            self.bm25_index[lang] = BM25Okapi(        # [LIB]
                bm25_corpus,
                k1=1.2,
                b=0.5,
            )
            self.kb_df[lang] = group
            log.info(f'[RAG] {lang.upper()} — {len(group)} tickets indexed')

        if self.use_reranker:
            self.reranker = CrossEncoder(self.RERANK_MODEL)  # [LIB]
            log.info(f'[RAG] Reranker loaded: {self.RERANK_MODEL}')

    # ── RETRIEVE ──────────────────────────────────────────────────────────────

    def retrieve(self,
                 query: str,
                 aspect_spans: Dict[str, str] = None,
                 subject: str = '',
                 lang: str = 'en',
                 top_k: int = None) -> List[Dict]:
        """
        Hybrid retrieval:
          [OUR A] Build aspect-weighted query vector
          [LIB]   FAISS ANN -> dense candidates
          [LIB]   BM25 -> sparse candidates
          [OUR B] RRF merge
          [OUR C] aspect_score + [OUR E] entity_overlap -> combined rerank
          [LIB]   CrossEncoder optional final rerank
        """
        top_k  = top_k or self.top_k
        lang   = lang if lang in self.faiss_index else 'en'
        spans  = aspect_spans or {}
        kb     = self.kb_df[lang]
        n_cand = top_k * 4

        # [OUR A] or fallback plain encoding
        if spans:
            qvec = aspect_query_vector(self.sbert, subject, spans)
        else:
            qvec = self.sbert.encode(                 # [LIB]
                [query], convert_to_numpy=True, device=self.device
            )[0].astype('float32')
            qvec /= (np.linalg.norm(qvec) + 1e-9)

        # [LIB] FAISS dense retrieval
        _, d_idx = self.faiss_index[lang].search(     # [LIB]
            qvec.reshape(1, -1), n_cand
        )
        dense = [int(i) for i in d_idx[0] if i >= 0]

        # [OUR C] WSD-filtered query tokens for BM25
        if _NLP_ENRICHER_AVAILABLE:
            nlp_model   = ModelRegistry.get_spacy(lang)
            bm25_tokens = wsd_filter_query(nlp_model, query).split()
        else:
            bm25_tokens = query.lower().split()

        # [LIB] BM25 sparse retrieval
        bm25_raw = self.bm25_index[lang].get_scores(  # [LIB]
            bm25_tokens
        )
        bm25_max = max(bm25_raw) + 1e-9
        sparse   = list(np.argsort(bm25_raw)[::-1][:n_cand])

        # [OUR B] RRF fusion
        merged = rrf_fusion(dense, sparse)

        # Query tag signals for tag-match scoring
        _qtags = set()
        if 'tags' in spans:
            _qtags = set(str(spans['tags']).lower().split())

        # [OUR C + E] Rerank by aspect similarity + entity overlap
        q_ents  = extract_entities(query)
        results = []

        for idx in merged[:n_cand]:
            row    = kb.iloc[idx]
            answer = str(row.get('answer', ''))

            a_sc  = aspect_score(self.sbert, spans, answer)
            e_sc  = entity_overlap(
                q_ents, extract_entities(str(row.get('body', '')))
            )
            kb_tags = set(
                str(kb.iloc[idx][c]).lower()
                for c in kb.columns if c.startswith('tag_')
                and pd.notna(kb.iloc[idx][c])
                and str(kb.iloc[idx][c]) not in ('nan', '')
            )
            tag_sc = (len(_qtags & kb_tags) / max(len(_qtags | kb_tags), 1)
                      if (_qtags or kb_tags) else 0.0)

            final = (0.40 * a_sc
                   + 0.25 * float(bm25_raw[idx] / bm25_max)
                   + 0.20 * e_sc
                   + 0.15 * tag_sc)

            results.append({
                'idx':      idx,
                'subject':  row.get('subject', ''),
                'body':     str(row.get('body', ''))[:BODY_LEN],
                'answer':   answer,
                'language': row.get('language', lang),
                'a_score':  round(a_sc, 3),
                'e_score':  round(e_sc, 3),
                'final':    round(final, 3),
            })

        results.sort(key=lambda x: x['final'], reverse=True)

        # [LIB] Optional CrossEncoder rerank
        if self.reranker and len(results) > top_k:
            pairs  = [(query, r['answer'][:ANSWER_LEN]) for r in results[:top_k * 2]]
            scores = self.reranker.predict(pairs)     # [LIB]
            order  = np.argsort(scores)[::-1]
            results = [results[i] for i in order[:top_k]]

        return results[:top_k]

    # ── PROMPT BUILDING ───────────────────────────────────────────────────────

    def build_prompt(self,
                     query_row: pd.Series,
                     retrieved: List[Dict],
                     aspect_spans: Dict[str, str] = None) -> str:
        """
        [OUR D] Structured aspect-slot T5 prompt.
        Slot order: Aspects -> Entities -> Metadata -> RAG Context -> Raw Query
        """
        body    = str(query_row.get('body', ''))
        subject = str(query_row.get('subject', ''))
        spans   = aspect_spans or {}

        asp_str = ' | '.join(
            f"{k}: {v}" for k, v in spans.items() if v and v != 'none'
        ) or 'GENERAL'

        ents    = extract_entities(body)
        ent_str = ' | '.join(
            f"{k}: {', '.join(v[:2])}" for k, v in ents.items() if v
        ) or 'none'

        meta_parts = [
            f"{f}: {query_row[f]}" for f in METADATA_FIELDS
            if pd.notna(query_row.get(f)) and str(query_row.get(f, '')) not in ('', 'nan')
        ]
        tags = [
            str(query_row[c]) for c in query_row.index
            if c.startswith('tag_') and pd.notna(query_row[c])
            and str(query_row[c]) not in ('', 'nan')
        ]
        if tags:
            meta_parts.append(f"tags: {', '.join(tags[:3])}")
        meta_str = ' | '.join(meta_parts)

        ctx_str = ' '.join(
            f"[{i+1}] {r['subject']} -> {str(r['answer'])[:150]}"
            for i, r in enumerate(retrieved[:2])
        )

        # [OUR D] Level-1 aspect enforcement — explicit instruction per slot
        urgency  = spans.get('urgency_vibe', 'neutral')
        priority = spans.get('priority', 'medium')
        prob     = spans.get('prob_sub', '') or spans.get('prob_statement', '')
        cause    = spans.get('cause', '')

        enforce_str = (
            f"The response MUST directly address: {prob}. "
            + (f"Root cause: {cause}. " if cause and cause != 'none' else "")
            + f"Tone: {urgency}. Priority: {priority}."
        )

        return (
            f"Write a professional customer support response to the following ticket.\n"
            f"{enforce_str}\n\n"
            f"Aspects: {asp_str}\n"
            f"Entities: {ent_str}\n"
            f"Metadata: {meta_str}\n"
            f"Context: {ctx_str}\n"
            f"Subject: {subject}\n"
            f"Complaint: {body[:BODY_LEN]}\n"
            f"Response:"
        )

    def build_prompts_for_df(self,
                             df: pd.DataFrame,
                             aspect_extractor=None) -> List[str]:
        """Build RAG prompts for all rows in dataframe."""
        assert self.faiss_index, "Call build_index() first"
        prompts = []

        for _, row in df.iterrows():
            lang    = str(row.get('language', 'en')).lower()[:2]
            body    = str(row.get('body', ''))
            subject = str(row.get('subject', ''))
            spans   = {}

            if aspect_extractor is not None:
                spans = aspect_extractor.extract(
                    body, subject=subject, lang=lang
                )

            tag_vals = [
                str(row[c]) for c in row.index
                if c.startswith('tag_') and pd.notna(row[c])
                and str(row[c]) not in ('nan', '')
            ]
            if tag_vals:
                spans['tags'] = ' '.join(tag_vals)

            retrieved = self.retrieve(
                query        = f"{subject} {body}",
                aspect_spans = spans,
                subject      = subject,
                lang         = lang,
            )
            prompts.append(self.build_prompt(row, retrieved, spans))

        log.info(f'[RAG] Built {len(prompts)} prompts')
        return prompts
