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

_cross_encoders: Dict[str, object] = {}


def get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3",
                      max_length: int = 512):
    """
    Cross-encoder singleton, keyed by name so the retrieval table can compare
    rerankers in one process (plan §6.3).

    bge-reranker-v2-m3 is the multilingual default: unlike the mmarco MiniLM it
    was trained for EN *and* DE reranking, which is what lets DE tickets be
    reranked at all instead of falling back to pure dense scores. On a T4 it is
    loaded in fp16 when the installed sentence-transformers supports passing
    model_kwargs.
    """
    key = f"{model_name}@{max_length}"
    if key not in _cross_encoders:
        from sentence_transformers import CrossEncoder
        print(f"[Retriever] Loading cross-encoder: {model_name}")
        try:
            import torch

            fp16 = torch.cuda.is_available()
        except ImportError:
            fp16 = False
        encoder = None
        if fp16:
            try:
                import torch

                encoder = CrossEncoder(
                    model_name, max_length=max_length,
                    model_kwargs={"torch_dtype": torch.float16})
            except TypeError:
                encoder = None          # older sentence-transformers
        if encoder is None:
            encoder = CrossEncoder(model_name, max_length=max_length)
        _cross_encoders[key] = encoder
    return _cross_encoders[key]


def _legacy_cross_encoder_cache():
    """Kept so tests and callers can clear the singleton cache."""
    return _cross_encoders

ANSWER_LEN = 0   # 0 = no truncation
BODY_LEN   = 0   # 0 = no truncation

# ── Reply language (plan §5.4) ────────────────────────────────────────────────
# A German ticket with English retrieved context used to get an English reply:
# the prompt never said which language to answer in. The instruction goes last,
# directly before "Support reply:", where the model is least likely to lose it.
LANG_NAMES = {"en": "English", "de": "German"}


def reply_language_line(language: str) -> str:
    name = LANG_NAMES.get((language or "en").lower().strip()[:2],
                          "the customer's language")
    return f"Write the reply in {name}, the same language as the customer's message."


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

    def field(source: Dict[str, str], *keys: str) -> str:
        for key in keys:
            value = str(source.get(key, "") or "").strip().lower()
            if value and value != "none":
                return value
        return ""

    def tokens(text: str) -> set:
        # Unicode tokens, so German overlap is measured on real words and not on
        # umlaut-mangled fragments (plan A.3).
        return set(re.findall(r"\w+", text, re.UNICODE))

    score = 0.0

    # The query's intent/type come from the supervised heads; a train document's
    # come from its gold queue/type, which it legitimately carries (plan A.3).
    query_intent = field(query_spans, "intent")
    doc_intent   = field(stored_aspects, "intent", "queue")
    if query_intent and query_intent == doc_intent:
        score += 0.4

    query_type = field(query_spans, "ticket_type", "ticket_type_nli")
    doc_type   = field(stored_aspects, "ticket_type", "ticket_type_nli", "type")
    if query_type and query_type == doc_type:
        score += 0.2

    # prob_sub token overlap
    q_words = tokens(field(query_spans, "prob_sub"))
    s_words = tokens(field(stored_aspects, "prob_sub"))
    if q_words and s_words:
        score += 0.3 * len(q_words & s_words) / len(q_words | s_words)

    # cause token overlap
    q_cause = tokens(field(query_spans, "cause"))
    s_cause = tokens(field(stored_aspects, "cause"))
    if q_cause and s_cause:
        score += 0.1 * len(q_cause & s_cause) / len(q_cause | s_cause)

    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION  (plan §6.3)
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """
    Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank(d)).

    Dense cosine and BM25 scores are not on a comparable scale, so they are
    fused by rank, not by value. `k` damps the head of each list; 60 is the
    value from the original RRF paper and the one the config defaults to.

    Returns {doc_id: fused_score} for every id seen in any list.
    """
    fused: Dict[int, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# [OUR B2] CROSS-ENCODER RERANKING
# ─────────────────────────────────────────────────────────────────────────────

def cross_encoder_rerank(
    query_subject: str,
    query_body: str,
    candidates: List[Dict],
    ce_model_name: str = "BAAI/bge-reranker-v2-m3",
    doc_field: str = "body",
    batch_size: int = 64,
    max_length: int = 512,
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
    ce = get_cross_encoder(ce_model_name, max_length=max_length)
    query_text = f"{query_subject}\n{query_body}".strip()

    def document_text(record: Dict) -> str:
        # "body" compares ticket to ticket (what the index was built on);
        # "response" compares the ticket to the past answer (plan §6.3).
        if doc_field == "response":
            return f"{record.get('subject', '')} {record.get('response', '')}".strip()
        body = record.get("body") or record.get("response", "")
        return f"{record.get('subject', '')}\n{body}".strip()

    pairs = [(query_text, document_text(r)) for r in candidates]
    ce_scores = ce.predict(pairs, batch_size=batch_size)
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
        max_length = int(getattr(config, "max_length", 0) or 0)
        if max_length:
            self.model.max_seq_length = max_length

    def encode(self, texts: List[str], kind: str = "passage") -> np.ndarray:
        """
        Prefixes come from the config (plan §6.3): E5 wants "query: "/"passage: ",
        BGE-M3 wants none. They used to be hard-coded to the E5 convention, which
        silently prepended "passage: " to every BGE-M3 text.
        Always returns L2-normalized float32 vectors (cosine similarity).
        """
        if kind not in ("query", "passage"):
            kind = "passage"
        prefix = (getattr(self.config, "query_prefix", "") if kind == "query"
                  else getattr(self.config, "passage_prefix", "")) or ""
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

        # ── BM25 (hybrid mode, plan §6.3) ─────────────────────────────────────
        # Dense retrieval misses exact strings — error codes, product names,
        # invoice numbers — which is exactly what support tickets quote.
        self.bm25 = None
        self.bm25_mode = str(getattr(config, "retrieval_mode", "dense")
                             or "dense").lower()
        if self.bm25_mode == "hybrid":
            self.bm25 = self._load_bm25()

        # Pre-load cross-encoder at init so first retrieve() call is not slow
        if getattr(config, "use_cross_encoder", True):
            get_cross_encoder(
                getattr(config, "cross_encoder_model", "BAAI/bge-reranker-v2-m3"),
                int(getattr(config, "cross_encoder_max_length", 512) or 512))

    def _load_bm25(self):
        """Load the BM25 index the indexer saved in the same row order as FAISS."""
        path = str(getattr(self.config, "bm25_path", "") or "")
        if not path:
            print("[Retriever] retrieval_mode=hybrid but bm25_path is empty "
                  "— falling back to dense")
            self.bm25_mode = "dense"
            return None
        try:
            import bm25s
        except ImportError:
            print("[Retriever] bm25s is not installed — falling back to dense")
            self.bm25_mode = "dense"
            return None
        try:
            retriever = bm25s.BM25.load(path, mmap=False)
        except Exception as exc:
            print(f"[Retriever] Could not load BM25 from {path} ({exc}) "
                  f"— falling back to dense")
            self.bm25_mode = "dense"
            return None
        print(f"[Retriever] BM25 loaded from {path}")
        return retriever

    def _bm25_ranking(self, query: str, n_candidates: int) -> List[int]:
        """BM25 document ids, best first."""
        if self.bm25 is None:
            return []
        import bm25s

        tokens = bm25s.tokenize([query], stopwords=None, show_progress=False)
        count = min(n_candidates, len(self.metadata))
        if count <= 0:
            return []
        doc_ids, _ = self.bm25.retrieve(tokens, k=count, show_progress=False)
        return [int(i) for i in doc_ids[0]]

    def retrieve(self,
                 subject      : str,
                 body         : str,
                 language     : str,
                 aspect_spans : Dict[str, str] = None,
                 top_k        : int = None,
                 exclude_ids  : set = None) -> List[Dict]:
        """
        Retrieve top-K similar past tickets.

        subject      : email subject of new ticket
        body         : email body
        language     : "en" or "de"
        aspect_spans : {prob_sub, prob_statement, cause, intent, ticket_type_nli,
                        entities, action_taken, urgency}  -- from aspect_extractor.py

        exclude_ids  : ticket_ids never to return (RAFT leakage guard, §7.4)

        Pipeline (plan §6.3): dense (aspect-weighted where the aspects are
        trustworthy) + optional BM25, fused by RRF, then a multilingual
        cross-encoder, then the aspect bonus on top of whichever score ranked.

        What changed from v2, and why:
          - DE is no longer excluded from reranking. bge-reranker-v2-m3 handles
            German, so DE rows get a cross-encoder score instead of raw dense.
          - the aspect bonus is applied *after* reranking for both languages; it
            used to be dropped entirely for EN and ignored for DE queries.
          - BM25 joins the dense list, so quoted error codes and product names
            can be matched literally.
          - exclude_ids keeps a RAFT training row from retrieving itself.
        """
        top_k  = top_k or self.config.top_k
        spans  = aspect_spans or {}
        n_cand = int(getattr(self.config, "n_candidates", 0) or 0) or top_k * 6
        exclude = {str(i) for i in (exclude_ids or set())}

        body_clip = (body[:self.config.body_max_chars]
                     if getattr(self.config, "body_max_chars", 0) else body)
        lang_lower = (language or "en").lower().strip()[:2]

        # ── 1. query vector ───────────────────────────────────────────────────
        aspect_langs = [str(l).lower()[:2] for l in
                        (getattr(self.config, "aspect_query_languages", ["en"])
                         or [])]
        use_aspect_query = bool(spans) and lang_lower in aspect_langs
        if use_aspect_query:
            qvec = aspect_query_vector(self.embedder, subject, body_clip, spans)
        else:
            qvec = self.embedder.encode(
                [f"{subject}\n\n{body_clip}"], kind="query")[0]

        # ── 2. dense + BM25 candidates ────────────────────────────────────────
        qvec_norm = qvec.reshape(1, -1).copy()
        faiss.normalize_L2(qvec_norm)
        dense_scores, d_idx = self.index.search(qvec_norm, n_cand)
        dense_by_id = {int(i): float(s)
                       for i, s in zip(d_idx[0].tolist(), dense_scores[0].tolist())
                       if i >= 0}
        dense_ranking = [int(i) for i in d_idx[0].tolist() if i >= 0]

        rank_lists = [dense_ranking]
        bm25_ranking = []
        if self.bm25_mode == "hybrid":
            bm25_ranking = self._bm25_ranking(f"{subject}\n{body_clip}", n_cand)
            if bm25_ranking:
                rank_lists.append(bm25_ranking)

        # ── 3. filters ────────────────────────────────────────────────────────
        filter_by_language = bool(getattr(self.config, "filter_by_language", False))

        def keep(doc_id: int) -> bool:
            if doc_id < 0 or doc_id >= len(self.metadata):
                return False
            meta = self.metadata[doc_id]
            if exclude and str(meta.get("ticket_id", doc_id)) in exclude:
                return False
            if filter_by_language and meta.get("language") != language:
                return False
            return True

        rank_lists = [[i for i in ranked if keep(i)] for ranked in rank_lists]

        # ── 4. reciprocal rank fusion ─────────────────────────────────────────
        rrf_k = int(getattr(self.config, "rrf_k", 60) or 60)
        fused = rrf_fuse(rank_lists, rrf_k)
        if not fused:
            return []

        candidates = []
        for doc_id, rrf_score in sorted(fused.items(), key=lambda kv: -kv[1])[:n_cand]:
            meta = self.metadata[doc_id]
            candidates.append({
                "idx"       : int(doc_id),
                "ticket_id" : str(meta.get("ticket_id", doc_id)),
                "subject"   : meta.get("subject", ""),
                "body"      : meta.get("body", ""),
                "response"  : meta.get("response", ""),
                "language"  : meta.get("language", ""),
                "aspects"   : meta.get("aspects", {}),
                "dense"     : round(dense_by_id.get(doc_id, 0.0), 4),
                "rrf"       : round(float(rrf_score), 6),
            })

        # ── 5/6. rerank, then the aspect bonus on top ─────────────────────────
        rerank_langs = [str(l).lower()[:2] for l in
                        (getattr(self.config, "rerank_languages", ["en", "de"])
                         or [])]
        use_ce = (bool(getattr(self.config, "use_cross_encoder", True))
                  and lang_lower in rerank_langs)
        alpha = float(getattr(self.config, "aspect_rerank_weight", 0.1) or 0.0)

        if use_ce:
            candidates = cross_encoder_rerank(
                subject, body_clip, candidates,
                getattr(self.config, "cross_encoder_model",
                        "BAAI/bge-reranker-v2-m3"),
                doc_field=str(getattr(self.config, "rerank_doc_field", "body")),
                batch_size=int(getattr(self.config, "rerank_batch_size", 64) or 64),
                max_length=int(getattr(self.config, "cross_encoder_max_length",
                                       512) or 512))
            base_key = "ce_score"
        else:
            for record in candidates:
                record["ce_score"] = ""
            base_key = "dense"

        for record in candidates:
            a_score = aspect_score(spans, record["aspects"]) if spans else 0.0
            record["a_score"] = round(a_score, 3)
            base = record[base_key] if isinstance(record[base_key], (int, float)) else 0.0
            record["final"] = round(float(base) + alpha * a_score, 4)
        candidates.sort(key=lambda r: r["final"], reverse=True)

        # ── 7. top_k ──────────────────────────────────────────────────────────
        return candidates[:top_k]

    def build_prompt(self, subject: str, body: str,
                     retrieved: List[Dict],
                     aspect_spans: Dict[str, str] = None,
                     language: str = "en") -> str:
        """
        [OUR C] Structured aspect prompt for the generator.
        Slot order: instruction → aspect checklist → context → ticket → language

        The reply-language line comes last, directly before "Support reply:"
        (plan §5.4); `reply_language_instruction: false` in the config turns it
        off. Each retrieved response is capped at `ctx_doc_max_chars` so one long
        past answer cannot eat the whole prompt budget (§5.3).
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

        ctx_doc_max = int(getattr(self.config, "ctx_doc_max_chars", 0) or 0)

        def _ctx_response(record: Dict) -> str:
            response = str(record.get('response', '') or '')
            return response[:ctx_doc_max] if ctx_doc_max > 0 else response

        ctx_str = '\n'.join(
            f"[{i+1}] Subject: {r['subject']}\n     Response: {_ctx_response(r)}"
            for i, r in enumerate(retrieved)
        )

        want_language_line = bool(getattr(self.config,
                                          "reply_language_instruction", True))
        language_line = (f"{reply_language_line(language)}\n"
                         if want_language_line else "")

        prob    = spans.get('prob_sub', '') or spans.get('prob_statement', '')
        cause   = spans.get('cause', '')
        enforce = (
            f"The response MUST directly address: {prob}. "
            + (f"Root cause: {cause}. " if cause and cause not in ('none','') else "")
        )
        prompt_body_max = int(getattr(self.config, "prompt_body_max_chars", BODY_LEN) or 0)
        body_for_prompt = body[:prompt_body_max] if prompt_body_max > 0 else body

        # ── checklist style (plan A.3, default) ───────────────────────────────
        # The aspect dump tells the model what the ticket is about; the checklist
        # tells it what the reply has to do. Aspect-RAFT trains on this format,
        # so it has to be the same string at inference time.
        style = str(getattr(self.config, "aspect_prompt_style", "checklist")
                    or "checklist").lower()
        if style == "checklist":
            from src.utils import build_aspect_checklist

            checklist = build_aspect_checklist(spans)
            return (
                f"Write a concise customer support reply (under 150 words). "
                f"No markdown headers, bullet points, or invented examples.\n"
                + (f"{checklist}\n" if checklist else "")
                + f"\nContext : {ctx_str}\n"
                f"Subject : {subject}\n"
                f"Message : {body_for_prompt}\n"
                + language_line
                + f"Support reply:"
            )

        # ── legacy slot dump (aspect_prompt_style: slots) ─────────────────────
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
            + language_line
            + f"Support reply:"
        )
