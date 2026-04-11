"""
retriever.py — Aspect-Aware FAISS Retriever
============================================
INPUT  : new ticket aspects + subject + body  (from aspect_extractor.py)
OUTPUT : top-K past tickets with gold responses  (fed to generator)

NOVEL CONTRIBUTIONS:
  [A] aspect_query_vector() — weighted query embedding (body included)
  [B] cross_encoder_rerank() — multilingual cross-encoder reranking
  [C] build_prompt()         — structured aspect-slot prompt for generator

DESIGN DECISIONS:
  - Dense-only FAISS (cosine via normalized inner-product)
  - Single multilingual index (E5 multilingual)
  - Cross-encoder reranking: mmarco-mMiniLMv2-L12-H384-v1 (EN+DE)
  - Language-conditional: cross-encoder for EN, pure dense for DE
  - aspect_rerank_weight reduced to 0.1 (was 0.25)
  - Body included in aspect query vector (weight 1.0)
"""

import re
import json
import numpy as np
import faiss
from typing import Dict, List, Optional

_cross_encoder = None


def get_cross_encoder(model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
    """
    Singleton cross-encoder loader.
    mmarco-mMiniLMv2-L12-H384-v1 is trained on MS MARCO multilingual —
    works for both EN and DE without language-specific fine-tuning.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print(f"[Retriever] Loading cross-encoder: {model_name}")
        _cross_encoder = CrossEncoder(model_name)
        print("[Retriever] Cross-encoder ready.")
    return _cross_encoder

ANSWER_LEN = 0   # 0 = no truncation
BODY_LEN   = 0   # 0 = no truncation


# ─────────────────────────────────────────────────────────────────────────────
# [OUR A] ASPECT-WEIGHTED QUERY VECTOR
# ─────────────────────────────────────────────────────────────────────────────

def aspect_query_vector(embedder, subject: str, body: str,
                        spans: Dict[str, str]) -> np.ndarray:
    """
    Weighted combination of aspect span embeddings + full body as query vector.

    FIX: body is now included with weight 1.0.
    Previously body was dropped entirely when aspects were present — for DE
    tickets with noisy aspect extraction this caused FAISS to retrieve
    irrelevant results. Body grounds the query in the actual ticket content.

    Weight scheme:
      prob_sub      : 2.0  — core problem topic, highest signal
      prob_statement: 1.5  — full problem description
      cause         : 1.0  — root cause
      subject       : 1.0  — email subject line
      body          : 1.0  — full ticket body (NEW — prevents query drift)
    """
    parts, weights = [], []
    if spans.get('prob_sub', 'none') not in ('none', ''):
        parts.append(spans['prob_sub']);       weights.append(2.0)
    if spans.get('prob_statement', 'none') not in ('none', ''):
        parts.append(spans['prob_statement']); weights.append(1.5)
    if spans.get('cause', 'none') not in ('none', ''):
        parts.append(spans['cause']);          weights.append(1.0)
    parts.append(subject or 'support ticket')
    weights.append(1.0)
    # Body included — anchors query in full ticket content
    if body and body.strip():
        parts.append(body)
        weights.append(1.0)

    vecs = embedder.encode(parts, kind="query")
    vec  = np.average(vecs, axis=0,
                      weights=np.array(weights[:len(vecs)], dtype='float32'))
    vec  = vec.astype('float32')
    vec /= (np.linalg.norm(vec) + 1e-9)
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# [OUR B] ASPECT SCORE — reranking signal
# ─────────────────────────────────────────────────────────────────────────────

def aspect_score(query_spans: Dict[str, str],
                 stored_aspects: Dict[str, str]) -> float:
    """
    Structured aspect matching between query and stored aspects.
    Uses field-level comparison since aspects are stored as dicts.

    Scoring:
      - intent match        : highest weight (0.4) — same category of problem
      - ticket_type match   : medium weight  (0.2) — same ticket type
      - prob_sub overlap    : medium weight  (0.3) — similar problem subject
      - cause overlap       : low weight     (0.1) — similar root cause

    Stored aspects are dicts: {prob_sub, prob_statement, cause, intent, ticket_type_nli}
    This allows clean filtering like intent=="login_or_access" without string searching.
    """
    if not query_spans or not stored_aspects:
        return 0.0

    score = 0.0

    # exact intent match — most discriminative signal
    if (query_spans.get("intent","").lower() ==
            stored_aspects.get("intent","").lower()):
        score += 0.4

    # exact ticket_type match
    if (query_spans.get("ticket_type_nli","").lower() ==
            stored_aspects.get("ticket_type_nli","").lower()):
        score += 0.2

    # prob_sub word overlap
    q_words = set(query_spans.get("prob_sub","").lower().split())
    s_words = set(stored_aspects.get("prob_sub","").lower().split())
    if q_words and s_words:
        score += 0.3 * len(q_words & s_words) / len(q_words | s_words)

    # cause word overlap
    q_cause = set(query_spans.get("cause","").lower().split())
    s_cause = set(stored_aspects.get("cause","").lower().split())
    if q_cause and s_cause:
        score += 0.1 * len(q_cause & s_cause) / len(q_cause | s_cause)

    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# [OUR B2] CROSS-ENCODER RERANKING
# ─────────────────────────────────────────────────────────────────────────────

def cross_encoder_rerank(
    query_subject: str,
    query_body: str,
    candidates: List[Dict],
    ce_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
) -> List[Dict]:
    """
    Rerank FAISS candidates using a multilingual cross-encoder.

    mmarco-mMiniLMv2-L12-H384-v1:
      - Trained on MS MARCO multilingual (EN + DE + 24 other languages)
      - Joint encoding of [query, document] pair — much richer than cosine
      - Replaces the Jaccard-based aspect_score for EN tickets
      - ~120MB, loads once as singleton

    Query  : "{subject} {body}"
    Document: "{hit_subject} {hit_response}"

    Returns candidates list with 'ce_score' added and sorted by ce_score.
    """
    ce = get_cross_encoder(ce_model_name)
    query_text = f"{query_subject} {query_body}".strip()
    pairs = [
        (query_text, f"{r['subject']} {r['response']}")
        for r in candidates
    ]
    ce_scores = ce.predict(pairs)
    for r, sc in zip(candidates, ce_scores):
        r['ce_score'] = round(float(sc), 4)
    candidates.sort(key=lambda x: x['ce_score'], reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDER (SentenceTransformers / E5 multilingual)
# ─────────────────────────────────────────────────────────────────────────────

class STEmbedder:
    def __init__(self, config):
        from sentence_transformers import SentenceTransformer
        self.model  = SentenceTransformer(config.model_name, device=getattr(config, "device", "cuda"))
        self.config = config

    def encode(self, texts: List[str], kind: str = "passage") -> np.ndarray:
        """
        E5 convention:
          - documents: "passage: ..."
          - queries:   "query: ..."
        We always return L2-normalized float32 vectors (cosine similarity).
        """
        if kind not in ("query", "passage"):
            kind = "passage"
        prefix = "query: " if kind == "query" else "passage: "
        prefixed = [prefix + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=getattr(self.config, "batch_size", 64),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TICKET RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────

class TicketRetriever:
    def __init__(self, config):
        self.config   = config
        self.embedder = STEmbedder(config)
        self.index    = faiss.read_index(config.index_path)
        with open(config.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[Retriever] {self.index.ntotal} vectors | "
              f"{len(self.metadata)} records loaded")
        # Pre-load cross-encoder at init so first retrieve() call is not slow
        ce_model = getattr(config, "cross_encoder_model",
                           "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        if getattr(config, "use_cross_encoder", True):
            get_cross_encoder(ce_model)

    def retrieve(self,
                 subject      : str,
                 body         : str,
                 language     : str,
                 aspect_spans : Dict[str, str] = None,
                 top_k        : int = None) -> List[Dict]:
        """
        Retrieve top-K similar past tickets.

        subject      : email subject of new ticket
        body         : email body
        language     : "en" or "de"
        aspect_spans : {prob_sub, prob_statement, cause, intent, ticket_type_nli,
                        entities, action_taken, urgency}  -- from aspect_extractor.py

        FIXES applied:
          1. Body included in aspect_query_vector (weight 1.0) — stops query drift
          2. alpha reduced to 0.1 (was 0.25) — aspect bonus no longer flips ranking
          3. Cross-encoder reranking for EN (replaces aspect_score Jaccard)
          4. Pure dense retrieval for DE — avoids noisy DE aspect extraction
             pulling FAISS in the wrong direction
        """
        top_k  = top_k or self.config.top_k
        spans  = aspect_spans or {}
        # Retrieve more candidates to give reranker good material
        n_cand = top_k * 6

        body_clip = (body[:self.config.body_max_chars]
                     if getattr(self.config, "body_max_chars", 0) else body)

        # ── Query vector ──────────────────────────────────────────────────────
        # FIX 1: body now included in aspect_query_vector (weight 1.0).
        # FIX 4: For DE, always use pure dense query (subject+body) regardless
        #        of whether aspects are present — DE aspect extraction is noisier
        #        and the weighted aspect vector degrades retrieval quality.
        lang_lower = (language or "en").lower().strip()
        use_aspect_query = bool(spans) and lang_lower == "en"

        if use_aspect_query:
            qvec = aspect_query_vector(self.embedder, subject, body_clip, spans)
        else:
            qvec = self.embedder.encode(
                [f"{subject}\n\n{body_clip}"], kind="query")[0]

        # ── FAISS dense retrieval ─────────────────────────────────────────────
        qvec_norm = qvec.reshape(1, -1).copy()
        faiss.normalize_L2(qvec_norm)
        dense_scores, d_idx = self.index.search(qvec_norm, n_cand)
        dense_scores = dense_scores[0].tolist()

        candidates = []
        for i, s in zip(d_idx[0].tolist(), dense_scores):
            if i < 0:
                continue
            meta = self.metadata[i]
            if (getattr(self.config, "filter_by_language", False)
                    and meta["language"] != language):
                continue
            candidates.append({
                "idx"     : int(i),
                "subject" : meta["subject"],
                "response": meta.get("response", ""),
                "language": meta["language"],
                "aspects" : meta.get("aspects", {}),
                "dense"   : round(float(s), 4),
            })

        if not candidates:
            return []

        # ── Reranking — language-conditional ─────────────────────────────────
        # FIX 3: EN — cross-encoder reranking replaces Jaccard aspect_score.
        #         Cross-encoder jointly encodes [query, document] — far richer
        #         than word-overlap scoring and works well for EN.
        # FIX 4: DE — pure dense ranking, no reranking.
        #         Avoids noisy DE aspect spans degrading retrieval quality.
        use_ce = (getattr(self.config, "use_cross_encoder", True)
                  and lang_lower == "en")

        if use_ce:
            ce_model = getattr(self.config, "cross_encoder_model",
                               "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
            candidates = cross_encoder_rerank(
                subject, body_clip, candidates, ce_model)
            for r in candidates:
                r["final"] = r["ce_score"]
                r["a_score"] = 0.0
        else:
            # FIX 2: alpha reduced from 0.25 → 0.1 — aspect bonus no longer
            # flips top-5 ranking when dense scores are close.
            alpha = float(getattr(self.config, "aspect_rerank_weight", 0.1) or 0.1)
            for r in candidates:
                a_sc = (aspect_score(spans, r["aspects"])
                        if spans else 0.0)
                r["a_score"] = round(a_sc, 3)
                r["final"]   = round(r["dense"] + alpha * a_sc, 4)
            candidates.sort(key=lambda x: x["final"], reverse=True)

        return candidates[:top_k]

    def build_prompt(self, subject: str, body: str,
                     retrieved: List[Dict],
                     aspect_spans: Dict[str, str] = None) -> str:
        """
        [OUR C] Structured aspect-slot prompt for the generator.
        Slot order: Enforce → Aspects → Entities/ActionTaken/Urgency → Context → Ticket

        NEW fields integrated: entities, action_taken, urgency
        """
        spans   = aspect_spans or {}

        # Core aspect fields (original 5)
        core_asp = {k: v for k, v in spans.items()
                    if k in ('prob_sub','prob_statement','cause',
                              'intent','ticket_type_nli')
                    and v and v not in ('none', '')}
        asp_str = ' | '.join(f"{k}: {v}" for k, v in core_asp.items()) or 'GENERAL'

        # New fields
        entities    = spans.get('entities',    '') or ''
        action_taken= spans.get('action_taken','') or ''
        urgency     = spans.get('urgency',     '') or ''

        ctx_str = '\n'.join(
            f"[{i+1}] Subject: {r['subject']}\n     Response: {r['response']}"
            for i, r in enumerate(retrieved)
        )
        prob    = spans.get('prob_sub', '') or spans.get('prob_statement', '')
        cause   = spans.get('cause', '')
        enforce = (
            f"The response MUST directly address: {prob}. "
            + (f"Root cause: {cause}. " if cause and cause not in ('none','') else "")
        )
        prompt_body_max = int(getattr(self.config, "prompt_body_max_chars", BODY_LEN) or 0)
        body_for_prompt = body[:prompt_body_max] if prompt_body_max > 0 else body

        # Build optional new-field lines
        extra_lines = ''
        if entities:
            extra_lines += f"Entities : {entities}\n"
        if action_taken:
            extra_lines += f"ActionTaken: {action_taken}\n"
        if urgency:
            extra_lines += f"Urgency  : {urgency}\n"

        return (
            f"Write a concise customer support reply (under 150 words). "
            f"No markdown headers, bullet points, or invented examples.\n"
            f"{enforce}\n\n"
            f"Aspects : {asp_str}\n"
            f"{extra_lines}"
            f"Context : {ctx_str}\n"
            f"Subject : {subject}\n"
            f"Message : {body_for_prompt}\n"
            f"Support reply:"
        )
