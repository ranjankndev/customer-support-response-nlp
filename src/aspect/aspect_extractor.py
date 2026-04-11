"""
aspect_extractor.py
===============================================================================
Aspect Extractor with SFT Model + Sentence-Level Ranker + Entity Coverage

PIPELINE (per ticket):
  STAGE 1 - Sentence Splitter
      Split body into sentences. Handles EN+DE, HTML breaks, embedded
      newlines, greeting stripping, abbreviation protection.

  STAGE 2 - Sentence Role Ranker  (NEW)
      paraphrase-multilingual-MiniLM-L12-v2 scores every sentence against
      bilingual role anchor centroids. Positional prior added for sentences
      inside each role's search window (from 50-row data analysis):
        prob_sub     : S1..S3  -- greeting may push it to S2
        cause        : S1..S4  -- often embedded in S1 or separate S2/S3
        action_taken : S2..S6  -- never S1, usually middle sentences
        urgency      : S2..end
      For prob_sub: early sentences (S1-S3) with cos>=0.15 are forced into
      candidates even if a later sentence scores higher overall.

  STAGE 3 - Entity Extractor  (NEW)
      Extracts named entities from subject+body without any NER model:
        - Domain patterns: CamelCase, version numbers, ALL-CAPS acronyms,
          hyphenated product names (Mini-Beamer, WLAN-Router)
        - Single capitalised nouns filtered by EN+DE stopword list
        - DE noun heuristic: keep only words in subject OR appearing 2+ times
      Greeting words, pronouns, function words, false positives all stripped.

  STAGE 4 - SFT Model (Qwen2.5-1.5B-Instruct + LoRA adapter)
      Receives: subject + pre-ranked sentence spans + entity hints
      Extracts ONLY: intent, ticket_type_nli (classification labels)
      prob_sub / prob_statement / cause are taken from Stage 2 directly
      to avoid hallucinated spans from the generative model.

OUTPUT SCHEMA (extended vs original):
  prob_sub        -- best PROBLEM sentence from ranker (S1..S3)
  prob_statement  -- PROBLEM | CAUSE combined (ranker)
  cause           -- best CAUSE sentence from ranker (S1..S4)
  intent          -- one of 7 labels (SFT model)
  ticket_type_nli -- one of 4 labels (SFT model)
  entities        -- NEW: comma-sep named entities from body+subject
  action_taken    -- NEW: what customer already tried (ranker S2..S6)
  urgency         -- NEW: critical/high/medium/low (rule-based)
  idx             -- NEW: preserved from input for idx-based merge

BUG FIXES vs original:
  1. _decode_generated: now uses attention_mask[i].sum() per row instead of
     padded shape[1] -- fixes silent token drop in left-padded batches
  2. body[:600] hard truncation in build_prompt removed
  3. idx column preserved in output CSV
  4. temperature dead parameter removed

USAGE:
  python aspect_extractor.py --input data.csv \
      --adapter ranjan56cse/cs-support-labels \
      --adapter_subfolder sft_checkpoint

  python aspect_extractor.py --config config_asp_sft.yaml
"""

import os
import re
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

from typing import Optional, List, Dict, Any

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ── DEFAULTS ──────────────────────────────────────────────────────────────────
STUDENT_BASE_DEFAULT = os.environ.get('STUDENT_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')
ADAPTER_DEFAULT      = os.environ.get('SFT_ADAPTER',  'ranjan56cse/cs-support-labels')
ADAPTER_SUBF_DEFAULT = os.environ.get('SFT_SUBFOLDER','sft_checkpoint')

MAX_NEW_TOKENS_DEFAULT = 80     # only generating 2 labels now, not full spans
MAX_SEQ_LEN_DEFAULT    = 900
RANKER_MODEL           = 'paraphrase-multilingual-MiniLM-L12-v2'

INTENT_LABELS = [
    'performance_issue','login_or_access','billing_issue',
    'feature_request','service_outage','general_inquiry','data_or_integration',
]
TICKET_TYPE_LABELS = ['complaint','escalation','feedback','product_support']

SYSTEM_PROMPT = (
    "You are an expert support ticket analyst. Extract ONLY two fields: "
    "intent and ticket_type_nli. Respond with a single valid JSON object "
    "and nothing else. No markdown fences, no explanation, no preamble."
)


# ══════════════════════════════════════════════════════════════════════════════
# STOPWORDS / GREETING FILTERS
# ══════════════════════════════════════════════════════════════════════════════

# Strips greeting prefixes at START of sentence (greeting + content mixed).
GREETING_PREFIX_RE = re.compile(
    r'^('
    r'hello[\s,]*|hi[\s,]*|dear[\s,]*|'
    r'good\s+(morning|afternoon|evening)[\s,]*|'
    r'greetings[\s,]*|hey[\s,]*|'
    r'to\s+whom\s+it\s+may\s+concern[\s,]*|'
    r'customer\s+support[\s,]*|support\s+team[\s,]*|'
    r'sehr\s+geehrte[rns]?[\s,]*|liebe[rns]?[\s,]*|'
    r'guten\s+(morgen|tag|abend)[\s,]*|'
    r'hallo[\s,]*|kundensupport[\s,]*|'
    r'lieber\s+kundensupport[\s,]*|liebes\s+support[\s,]*'
    r')',
    re.IGNORECASE
)

# Pure greeting sentence -- entire sentence has no content, skip for aspects.
PURE_GREETING_RE = re.compile(
    r'^('
    r'(hello|hi|hey|dear|greetings)[^.!?]{0,40}[.!?]?\s*$|'
    r'thank\s+you[\s.,!]*$|thanks[\s.,!]*$|'
    r'best\s+regards[\s.,!]*$|sincerely[\s.,!]*$|'
    r'looking\s+forward\s+to[\s\S]{0,50}$|'
    r'i\s+look\s+forward[\s\S]{0,50}$|'
    r'your\s+(prompt\s+)?assistance[\s\S]{0,40}$|'
    r'we\s+(thank|appreciate)[\s\S]{0,40}$|'
    r'(sehr\s+geehrte[rns]?|liebe[rns]?|hallo)[^.!?]{0,40}[.!?]?\s*$|'
    r'danke[\s.,!]*$|vielen\s+dank[\s.,!]*$|'
    r'mit\s+freundlichen\s+gr[üu][sz]en[\s\S]{0,30}$|'
    r'ich\s+hoffe[\s,].{0,60}(gut|well)[\s.,!]*$|'
    r'ich\s+freue\s+mich[\s,].{0,80}$|'
    r'ich\s+danke\s+ihnen[\s\S]{0,60}$|'
    r'wir\s+(danken|freuen)[\s\S]{0,60}$'
    r')',
    re.IGNORECASE
)

# Stopword set for entity extraction -- EN + DE (lowercase for lookup).
ENTITY_STOPWORDS = {
    # EN pronouns / determiners
    'i','we','you','he','she','it','they','our','your','their','my','its',
    'this','that','these','those','who','which','what','where','when','how',
    # EN articles / prepositions
    'the','a','an','of','in','on','at','by','for','with','from','to','into',
    'through','during','before','after','above','below','between','among',
    # EN conjunctions / adverbs
    'and','or','but','if','as','so','yet','nor','not','no','yes','also',
    'just','already','still','yet','again','once','often','always','never',
    'very','too','quite','rather','more','most','less','least',
    # EN sentence starters that are capitalised but not entities
    'there','here','please','kindly','thank','additionally','furthermore',
    'however','therefore','moreover','although','despite','meanwhile','thus',
    'hence','currently','recently','immediately','overall','finally',
    'initially','subsequently',
    # EN polite phrases
    'hello','hi','dear','best','regards','sincerely','greetings',
    # EN verbs / auxiliaries
    'is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might',
    'shall','can','need','let','make','get','give','take','keep','help',
    # DE pronouns / determiners
    'ich','wir','sie','er','es','ihr','uns','euch','ihnen','sich',
    'der','die','das','ein','eine','einem','einer','eines','den','dem','des',
    # DE prepositions / conjunctions
    'und','oder','aber','wenn','als','ob','da','weil','dass','mit','von',
    'zu','in','an','auf','fur','durch','uber','unter','nach','vor','bei',
    'bis','seit','ohne','gegen','um','statt','trotz','wahrend',
    # DE auxiliaries
    'ist','sind','war','waren','wird','werden','wurde','wurden',
    'hat','haben','hatte','hatten','kann','konnen','konnte','konnten',
    'muss','mussen','soll','sollen','darf','durfen',
    # DE adverbs / filler
    'bitte','danke','leider','gerne','bald','immer','nie','oft',
    'mehr','weniger','alle','keine','kein','nicht','nur','auch',
    'schon','bereits','jedoch','daher','deshalb','damit','somit',
    'besonders','insbesondere','zudem','ausserdem','trotzdem',
    # DE formal pronoun forms (always capitalised in DE text, not entities)
    'ihnen','ihrem','ihrer','ihres','ihren','ihre',
    # DE polite starters
    'sehr','geehrte','liebe','hallo','kundensupport',
}

# Capitalised words that are sentence starters, not entities
ENTITY_FP_RE = re.compile(
    r'^(The|This|These|Those|There|Here|Dear|Hello|Please|Thank|Kindly|'
    r'Additionally|Furthermore|However|Therefore|Moreover|Although|Despite|'
    r'Meanwhile|Immediate|Currently|Recently|Already|After|Before|During|'
    r'Since|While|When|Where|What|Which|Who|How|Why|If|As|So|'
    r'We|Our|My|Your|Their|Its|'
    r'Sehr|Liebe|Bitte|Danke|Leider|Trotz|Obwohl|Wegen|Falls|Wenn|'
    r'Auch|Noch|Schon|Bereits|Jedoch|Daher|Deshalb|Damit|Somit|'
    r'Zudem|Ausserdem|Trotzdem|Besonders|Insbesondere)$',
    re.IGNORECASE
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read().strip()
    if not s:
        return {}
    if s.lstrip().startswith('{'):
        return json.loads(s)
    cfg: dict = {}
    for line in s.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v.lower() in ('true', 'false'):
            cfg[k] = (v.lower() == 'true')
            continue
        try:
            cfg[k] = float(v) if '.' in v else int(v)
            continue
        except Exception:
            cfg[k] = v
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# GPU PROFILE
# ══════════════════════════════════════════════════════════════════════════════

def get_gpu_profile() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")
    props   = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    dtype   = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    batch   = 12 if vram_gb >= 23 else 6
    print(f"  GPU  : {props.name}")
    print(f"  VRAM : {vram_gb:.1f} GB")
    print(f"  DType: {'bf16' if dtype == torch.bfloat16 else 'fp16'}")
    return {'dtype': dtype, 'vram_gb': vram_gb, 'batch': batch}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 -- SENTENCE SPLITTER
# ══════════════════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> List[str]:
    """
    Split body into sentences. Handles:
      - Standard EN/DE punctuation
      - HTML <br> tags and embedded newlines
      - Protects abbreviations: Mr. Dr. z.B. d.h. etc.
      - Strips markdown bold markers
    """
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\*\*[^*]+\*\*', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # Protect abbreviations from being split
    abbrev = r'\b(Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|bzw|inkl|ggf|z\.B|d\.h|u\.a)\.'
    text   = re.sub(abbrev, lambda m: m.group(0).replace('.', '<<<DOT>>>'), text)

    sents = re.split(r'(?<=[.!?])\s+(?=[A-Za-zÜÖÄ\[\*])', text)
    sents = [s.replace('<<<DOT>>>', '.').strip() for s in sents]
    return [s for s in sents if len(s.split()) >= 2]


def is_pure_greeting(sentence: str) -> bool:
    if len(sentence.split()) > 18:
        return False
    return bool(PURE_GREETING_RE.match(sentence.strip()))


def strip_greeting_prefix(sentence: str) -> str:
    cleaned = GREETING_PREFIX_RE.sub('', sentence).strip().lstrip(',').strip()
    if len(cleaned.split()) < 4:
        return sentence
    return cleaned[0].upper() + cleaned[1:] if cleaned else sentence


def get_content_sentences(body: str) -> List[Dict[str, Any]]:
    """
    Returns list of sentence dicts:
      sent_num     : 1-based position
      total_sents  : total count
      text         : original text
      cleaned_text : greeting-prefix-stripped (for ranker + entities)
      is_greeting  : True = pure greeting, skip for aspect search
      position_pct : 0.0 (first) .. 1.0 (last)
    """
    raw   = split_sentences(body)
    n     = len(raw)
    result = []
    for i, s in enumerate(raw):
        pure = is_pure_greeting(s)
        cleaned = strip_greeting_prefix(s) if not pure else s
        result.append({
            'sent_num'    : i + 1,
            'total_sents' : n,
            'text'        : s,
            'cleaned_text': cleaned,
            'is_greeting' : pure,
            'position_pct': i / max(n - 1, 1),
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 -- SENTENCE ROLE RANKER
# ══════════════════════════════════════════════════════════════════════════════

# Bilingual anchor phrases per role.
# MiniLM scores each sentence against the centroid of its role's anchors.
ROLE_ANCHORS: Dict[str, List[str]] = {

    # prob_sub: SHORT topic label -- what the ticket is ABOUT.
    # Anchors target sentences naming the problem CATEGORY or TOPIC.
    # Typically S1 or S2 after greeting strip.
    'prob_sub': [
        # EN -- short problem topic anchors
        "the main problem is",
        "we are experiencing an issue with",
        "I am reporting a problem",
        "there is an error in the system",
        "the system has failed to work",
        "I am contacting you to report",
        "we have encountered an issue",
        "I am writing to report a problem",
        "a crash occurred in our software",
        "we detected unauthorized access",
        "there was a security breach",
        # DE -- short problem topic anchors
        "das Hauptproblem ist",
        "wir haben ein Problem mit",
        "ich melde ein Problem",
        "es gibt einen Fehler",
        "das System funktioniert nicht mehr",
        "ich schreibe um ein Problem zu melden",
        "wir haben ein Problem festgestellt",
        "es kam zu einem unerwarteten Fehler",
        "unerlaubter Zugriff wurde festgestellt",
    ],

    # prob_statement: FULL detailed description of what is happening.
    # More expansive than prob_sub -- describes IMPACT, SCOPE, AFFECTED SYSTEM.
    # Independently scored with its own anchors and search window S1..S4.
    # NOT derived from prob_sub+cause -- it is extracted on its own.
    'prob_statement': [
        # EN -- full problem description anchors
        "the system is completely down and affecting all users",
        "we are unable to access our data and operations are halted",
        "the crash happened during a critical operation causing significant delays",
        "medical data has been compromised and patient records are at risk",
        "the security breach is affecting all hospital systems and data",
        "the software keeps crashing and the issue continues to persist",
        "our campaign metrics are declining significantly despite all adjustments",
        "data synchronization between systems is failing and blocking operations",
        "the integration is broken and the entire business process is blocked",
        "unauthorized access was detected in the medical records system",
        "the platform is experiencing severe performance degradation affecting workflows",
        "investment projections are inaccurate due to data inconsistencies",
        "the digital tools crashed and critical data access is disrupted",
        # DE -- full problem description anchors
        "das System ist vollstandig ausgefallen und betrifft alle Benutzer",
        "wir konnen nicht auf unsere Daten zugreifen und der Betrieb ist blockiert",
        "der Absturz geschah wahrend eines kritischen Vorgangs und verursacht Verzogerungen",
        "medizinische Daten wurden kompromittiert und Patientendaten sind gefahrdet",
        "die Sicherheitsverletzung betrifft alle Systeme des Krankenhauses",
        "die Software sturzt standig ab und das Problem besteht weiterhin fort",
        "unsere Kampagnenmetriken sinken erheblich trotz aller Anpassungen",
        "unerlaubter Zugriff auf das medizinische Datensystem wurde festgestellt",
        "die digitalen Werkzeuge sind abgestuerzt und der Datenzugriff ist gestort",
    ],

    'cause': [
        # EN
        "the root cause is",
        "this happened because",
        "the issue is due to",
        "triggered by a phishing attack",
        "caused by incompatible software updates",
        "this might be related to",
        "the probable cause is",
        "possibly due to a software bug",
        "the problem stems from",
        "this could be due to",
        # DE
        "die Ursache ist",
        "das Problem entstand durch",
        "ausgelost durch",
        "verursacht durch inkompatible Updates",
        "aufgrund von Softwarekonflikten",
        "konnte auf zuruckzufuhren sein",
        "wahrscheinlich aufgrund einer Serveruberlastung",
        "die mogliche Ursache ist",
    ],
    'action_taken': [
        # EN
        "we have already tried to fix it",
        "I have restarted the system",
        "we attempted to resolve by restarting",
        "despite our efforts to resolve",
        "we have updated the software but the problem persists",
        "I cleared the cache and restarted",
        "we ran antivirus scans",
        "passwords have been reset",
        "we applied software updates without success",
        # DE
        "wir haben bereits versucht",
        "ich habe das System neu gestartet",
        "wir haben versucht das Problem zu beheben",
        "trotz unserer Bemuhungen bleibt das Problem",
        "wir haben die Software aktualisiert",
        "ich habe den Cache geleert und neu gestartet",
        "Systemupdates wurden durchgefuhrt",
        "Passworter wurden zuruckgesetzt",
    ],
    'urgency': [
        "this is urgent and needs immediate attention",
        "we need immediate assistance now",
        "critical issue affecting business operations",
        "production system is down",
        "data loss is occurring",
        "dringend Unterstutzung benotigt",
        "kritisches Problem betrifft den Betrieb",
        "sofortige Hilfe benotigt",
        "Produktion ist ausgefallen",
    ],
}

# Search windows (1-based, inclusive).
# Based on 50-row data analysis:
#   prob_sub    : S1..S3  -- greeting may push to S2
#   cause       : S1..S4  -- can be embedded in S1 or separate S2/S3/S4
#   action_taken: S2..S6  -- never S1, usually middle sentences
#   urgency     : S2..end
ROLE_SEARCH_WINDOW = {
    'prob_sub'      : (1, 3),   # topic label -- first 3 sentences
    'prob_statement': (1, 4),   # full description -- S1..S4 (broader than prob_sub)
    'cause'         : (1, 4),   # cause -- S1..S4
    'action_taken'  : (2, 6),   # action taken -- S2..S6
    'urgency'       : (2, 99),  # urgency -- anywhere after S1
}
POSITION_WEIGHT = 0.12   # cosine bonus for in-window sentences

_ranker_model = None


def get_ranker():
    global _ranker_model
    if _ranker_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  [Ranker] Loading {RANKER_MODEL} ...")
        _ranker_model = SentenceTransformer(RANKER_MODEL)
        print("  [Ranker] Ready.")
    return _ranker_model


def encode_anchors(ranker) -> Dict[str, np.ndarray]:
    """Pre-encode role anchor centroids once per run."""
    encoded = {}
    for role, anchors in ROLE_ANCHORS.items():
        vecs     = ranker.encode(anchors, normalize_embeddings=True,
                                  show_progress_bar=False)
        centroid = np.mean(vecs, axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-9)
        encoded[role] = centroid.astype(np.float32)
    return encoded


def rank_sentences(
    sentences: List[Dict[str, Any]],
    anchor_vecs: Dict[str, np.ndarray],
    ranker,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Score each non-greeting sentence against each role anchor.

    Special handling for prob_sub:
      Early sentences S1-S3 with cosine >= 0.15 are forced into candidates
      even if a later sentence scores higher -- prob_sub lives early in data.

    Returns top-3 candidates per role sorted by final_score descending.
    """
    scoreable = [
        s for s in sentences
        if not s['is_greeting'] and len(s['cleaned_text'].split()) >= 3
    ]
    if not scoreable:
        return {role: [] for role in ROLE_ANCHORS}

    texts     = [s['cleaned_text'] for s in scoreable]
    sent_vecs = ranker.encode(texts, normalize_embeddings=True,
                               show_progress_bar=False)

    results: Dict[str, List[Dict[str, Any]]] = {}
    for role, anchor_vec in anchor_vecs.items():
        win_min, win_max = ROLE_SEARCH_WINDOW[role]
        scored = []
        for i, s in enumerate(scoreable):
            cos_sim   = float(np.dot(sent_vecs[i], anchor_vec))
            in_window = win_min <= s['sent_num'] <= win_max
            pos_bonus = POSITION_WEIGHT if in_window else 0.0
            scored.append({
                **s,
                'cos_sim'    : round(cos_sim, 4),
                'final_score': round(cos_sim + pos_bonus, 4),
            })

        if role == 'prob_sub':
            # prob_sub: force early sentences (S1-S3) into candidates -- topic
            # almost always stated at the start of the ticket.
            early = [s for s in scored
                     if 1 <= s['sent_num'] <= 3 and s['cos_sim'] >= 0.15]
            rest  = [s for s in scored if s not in early]
            early.sort(key=lambda x: x['final_score'], reverse=True)
            rest.sort(key=lambda x:  x['final_score'], reverse=True)
            merged = early[:2] + [r for r in rest if r not in early]
            results[role] = merged[:3]
        elif role == 'prob_statement':
            # prob_statement: prefer sentences with higher detail/impact signals.
            # S1-S4 are in-window; also prefer longer sentences (more detail).
            # Add a small length bonus (capped) to favour descriptive sentences.
            for s in scored:
                word_count = len(s['cleaned_text'].split())
                length_bonus = min(word_count / 100.0, 0.08)  # max +0.08
                s['final_score'] = round(s['final_score'] + length_bonus, 4)
            scored.sort(key=lambda x: x['final_score'], reverse=True)
            results[role] = scored[:3]
        else:
            scored.sort(key=lambda x: x['final_score'], reverse=True)
            results[role] = scored[:3]

    return results


def pick_best_span(ranked: List[Dict[str, Any]], min_words: int = 4) -> str:
    """Pick best span from top-3 ranked sentences, skipping closings."""
    closing_re = re.compile(
        r'^(thank|danke|vielen|bitte|please|regards|sincerely|'
        r'ich\s+freue|ich\s+danke|mit\s+freundlichen|'
        r'looking\s+forward|wir\s+freuen|best\s+regards)',
        re.IGNORECASE
    )
    for s in ranked:
        text = s['cleaned_text'].strip()
        if len(text.split()) < min_words:
            continue
        if closing_re.match(text):
            continue
        return text
    return ''


def build_prob_statement(prob_sub: str, cause: str) -> str:
    parts = []
    if prob_sub:
        parts.append(prob_sub)
    if cause and cause != prob_sub:
        parts.append(cause)
    return ' | '.join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 -- ENTITY EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

_DOMAIN_PATS = [
    # Product name + version: IBM SPSS 28, Qwen2.5, DaVinci Resolve 17
    re.compile(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z0-9]*)*\s+(?:v\.?\s*)?\d+(?:\.\d+)*\b'),
    # CamelCase: QuickBooks, ClickUp, ActiveCampaign, PyTorch
    re.compile(r'\b[A-Z][a-z]+[A-Z][a-zA-Z0-9]+\b'),
    # ALL-CAPS acronyms 2-6 chars: GDPR, HIPAA, IBM, SQL, WLAN
    re.compile(r'\b[A-Z]{2,6}\b'),
    # Hyphenated product names: Mini-Beamer, WLAN-Router, Smart-Tracker
    re.compile(r'\b[A-Z][a-zA-Z]+-[A-Z]?[a-zA-Z]+\b'),
]
_SINGLE_CAP_RE = re.compile(r'\b[A-Z][a-zA-Z]{3,}\b')
_DE_NOUN_RE    = re.compile(r'\b[A-Z][a-z]{4,}\b')


def extract_entities(subject: str, body: str, language: str = 'en') -> str:
    """
    Extract named entities (product/software/system names).
    No NER model required -- uses patterns + stopword filtering.

    For DE: keeps only capitalised nouns appearing in subject OR 2+ times
    in body to cut German grammatical capitalisation false positives.
    """
    text = GREETING_PREFIX_RE.sub(' ', f"{subject} {body}")
    entities: List[str] = []

    # 1. Domain patterns (highest precision)
    for pat in _DOMAIN_PATS:
        for m in pat.finditer(text):
            w = m.group(0).strip()
            if w.lower() not in ENTITY_STOPWORDS:
                entities.append(w)

    # 2. Single capitalised words
    for m in _SINGLE_CAP_RE.finditer(text):
        w = m.group(0).strip()
        if ENTITY_FP_RE.match(w) or w.lower() in ENTITY_STOPWORDS:
            continue
        if len(w) < 3:
            continue
        entities.append(w)

    # 3. DE noun heuristic
    if language == 'de':
        subj_words = set(subject.split())
        body_freq: Dict[str, int] = {}
        for m in _DE_NOUN_RE.finditer(body):
            w = m.group(0)
            body_freq[w] = body_freq.get(w, 0) + 1
        for w, freq in body_freq.items():
            if w.lower() in ENTITY_STOPWORDS or ENTITY_FP_RE.match(w):
                continue
            if w in subj_words or freq >= 2:
                entities.append(w)

    # Deduplicate
    seen: set = set()
    clean: List[str] = []
    for e in entities:
        key = e.lower().strip()
        if key in ENTITY_STOPWORDS or key in seen or len(key) < 2:
            continue
        if ENTITY_FP_RE.match(e):
            continue
        seen.add(key)
        clean.append(e)

    return ', '.join(clean[:10])


# ══════════════════════════════════════════════════════════════════════════════
# URGENCY SCORER -- rule-based
# ══════════════════════════════════════════════════════════════════════════════

_CRITICAL = [
    'down','outage','offline','unavailable','cannot access',"can't access",
    'cannot login',"can't login",'unable to login','locked out',
    'security breach','data loss','incident','urgent','asap','immediately',
    'dringend','sofort','systemausfall','datenverlust','sicherheitsverletzung',
    'unerlaubter zugriff',
]
_HIGH = [
    'error','failed','failure','not working','crash','crashing','timeout',
    'bug','broken','payment','refund','invoice','breach','persists',
    'fehler','absturz','funktioniert nicht','abgestuerzt','ausgefallen',
]
_MEDIUM = [
    'slow','latency','performance','delay','integration','api','sync',
    'langsam','verzogerung','synchronisation','optimierung',
]


def score_urgency(body: str) -> str:
    t = body.lower()
    if any(k in t for k in _CRITICAL): return 'critical'
    if any(k in t for k in _HIGH):     return 'high'
    if any(k in t for k in _MEDIUM):   return 'medium'
    return 'low'


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 -- SFT MODEL (classification only)
# ══════════════════════════════════════════════════════════════════════════════

_model     = None
_tokenizer = None


def get_model(student_base: str, adapter: str,
              adapter_subfolder: Optional[str], dtype: torch.dtype):
    global _model, _tokenizer
    if _model is None:
        print(f"  Base model : {student_base}")
        print(f"  SFT adapter: {adapter}" +
              (f" (subfolder={adapter_subfolder})" if adapter_subfolder else ""))
        _tokenizer = AutoTokenizer.from_pretrained(
            student_base, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = 'left'   # required for batched causal LM

        base = AutoModelForCausalLM.from_pretrained(
            student_base, device_map={'': 0},
            torch_dtype=dtype, trust_remote_code=True)
        peft_kw = {}
        if adapter_subfolder:
            peft_kw['subfolder'] = adapter_subfolder
        _model = PeftModel.from_pretrained(base, adapter, **peft_kw)
        _model.eval()
        try:
            _model = _model.to(dtype=dtype)
        except Exception:
            pass
        print(f"  Model ready. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return _tokenizer, _model


def build_sft_prompt(subject: str, prob_sub_sent: str, cause_sent: str,
                     action_sent: str, entities: str, lang: str) -> str:
    """Focused prompt for SFT -- classification only, uses ranked spans."""
    s = subject if subject and subject.lower() not in ('nan', 'none', '') \
        else '(none)'
    parts = []
    if prob_sub_sent:
        parts.append(f"Problem: {prob_sub_sent}")
    if cause_sent and cause_sent != prob_sub_sent:
        parts.append(f"Cause: {cause_sent}")
    if action_sent:
        parts.append(f"Action taken: {action_sent}")
    if entities:
        parts.append(f"Key entities: {entities}")
    context = '\n'.join(parts) or '(see subject)'
    return (
        f"Classify this support ticket. Extract ONLY intent and ticket_type_nli.\n\n"
        f"intent must be one of: performance_issue | login_or_access | billing_issue | "
        f"feature_request | service_outage | general_inquiry | data_or_integration\n\n"
        f"ticket_type_nli must be one of: complaint | escalation | feedback | product_support\n\n"
        f"Ticket language: {lang}\n"
        f"Subject: {s}\n"
        f"Key information:\n{context}\n\n"
        f"JSON:"
    )


def _parse_labels(raw: str) -> dict:
    raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
    m   = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        return {}
    s = re.sub(r',\s*([}\]])', r'\1', m.group(0)).replace("'", '"')
    try:
        obj = json.loads(s)
    except Exception:
        obj = {}
        for field in ('intent', 'ticket_type_nli'):
            m2 = re.search(rf'"{field}"\s*:\s*"([^"]*)"', s)
            if m2:
                obj[field] = m2.group(1).strip()

    if obj.get('intent') not in INTENT_LABELS:
        raw_i = str(obj.get('intent', '')).lower().replace(' ', '_')
        obj['intent'] = next(
            (l for l in INTENT_LABELS if l in raw_i or raw_i in l),
            'general_inquiry')
    if obj.get('ticket_type_nli') not in TICKET_TYPE_LABELS:
        raw_t = str(obj.get('ticket_type_nli', '')).lower()
        obj['ticket_type_nli'] = next(
            (l for l in TICKET_TYPE_LABELS if l in raw_t or raw_t in l),
            'product_support')
    return obj


def _build_chat_text(tok, r: Dict[str, Any]) -> str:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': build_sft_prompt(
            str(r.get('subject',       '') or ''),
            str(r.get('prob_sub_sent', '') or ''),
            str(r.get('cause_sent',    '') or ''),
            str(r.get('action_sent',   '') or ''),
            str(r.get('entities',      '') or ''),
            str(r.get('language',     'en') or 'en'),
        )},
    ]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def _decode_generated(tok, out: torch.Tensor,
                       attention_mask: torch.Tensor) -> List[Dict[str, Any]]:
    """
    BUG FIX: Use per-row actual prompt length from attention_mask[i].sum().
    Original used padded shape[1] for all rows -- wrong for left-padded batches.
    """
    results = []
    for i in range(out.shape[0]):
        actual_prompt_len = int(attention_mask[i].sum().item())
        gen_ids = out[i][actual_prompt_len:]
        raw     = tok.decode(gen_ids, skip_special_tokens=True).strip()
        parsed  = _parse_labels(raw)
        results.append({
            'intent'         : parsed.get('intent',          'general_inquiry'),
            'ticket_type_nli': parsed.get('ticket_type_nli', 'product_support'),
        })
    return results


@torch.inference_mode()
def extract_labels_batch(
    tok, model,
    batch_stage3: List[Dict[str, Any]],
    max_seq_len: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    """SFT model batch -- classification labels only."""
    texts  = [_build_chat_text(tok, r) for r in batch_stage3]
    inputs = tok(texts, return_tensors='pt', padding=True,
                 truncation=True, max_length=max_seq_len).to(model.device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id,
        )
    except Exception as e:
        if len(batch_stage3) <= 1:
            print(f"  WARNING: generate() failed -- defaults used: {e!r}")
            return [{'intent': 'general_inquiry',
                     'ticket_type_nli': 'product_support'}
                    for _ in batch_stage3]
        torch.cuda.empty_cache()
        merged = []
        for r in batch_stage3:
            merged.extend(extract_labels_batch(
                tok, model, [r], max_seq_len, max_new_tokens))
        return merged
    return _decode_generated(tok, out, inputs['attention_mask'])


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE PER ROW (Stages 1-3, CPU)
# ══════════════════════════════════════════════════════════════════════════════

def process_row(row: Dict[str, Any],
                anchor_vecs: Dict[str, np.ndarray],
                ranker) -> Dict[str, Any]:
    """Run Stages 1-3 for one row. Returns dict for Stage 4 batch."""
    subject  = str(row.get('subject',  '') or '')
    body     = str(row.get('body',     '') or '')
    language = str(row.get('language', 'en') or 'en')

    sentences = get_content_sentences(body)               # Stage 1
    ranked    = rank_sentences(sentences, anchor_vecs, ranker)  # Stage 2

    # Each field independently ranked -- prob_statement is NOT derived from others.
    # prob_sub      : short topic label (S1..S3, high-cosine early sentences)
    # prob_statement: full detailed problem description (S1..S4, length-boosted)
    # cause         : root cause sentence (S1..S4, cause-specific anchors)
    # action_taken  : what was already tried (S2..S6)
    prob_sub_sent       = pick_best_span(ranked.get('prob_sub',       []))
    prob_statement_sent = pick_best_span(ranked.get('prob_statement', []))
    cause_sent          = pick_best_span(ranked.get('cause',          []))
    action_sent         = pick_best_span(ranked.get('action_taken',   []))

    # prob_statement: use independently ranked sentence as primary.
    # Fall back to combining prob_sub + cause only if ranker found nothing.
    if not prob_statement_sent:
        prob_statement_sent = build_prob_statement(prob_sub_sent, cause_sent)

    entities = extract_entities(subject, body, language)  # Stage 3
    urgency  = score_urgency(body)

    return {
        # Stage 4 inputs
        'subject'       : subject,
        'prob_sub_sent' : prob_sub_sent,
        'cause_sent'    : cause_sent,
        'action_sent'   : action_sent,
        'entities'      : entities,
        'language'      : language,
        # Final output fields (Stages 1-3 complete)
        'prob_sub'      : prob_sub_sent,
        'prob_statement': prob_statement_sent,
        'cause'         : cause_sent,
        'action_taken'  : action_sent,
        'urgency'       : urgency,
        'entities'      : entities,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(input_path: str,
        out_path: Optional[str],
        n: Optional[int],
        student_base: str,
        adapter: str,
        adapter_subfolder: Optional[str],
        batch_size: int,
        max_seq_len: int,
        max_new_tokens: int):

    print("=" * 65)
    print("Aspect Extractor -- Ranker + Entity Coverage + SFT Labels")
    print("=" * 65)

    profile = get_gpu_profile()
    if not batch_size:
        batch_size = profile['batch']

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"\nLoading data: {input_path}")
    df = (pd.read_excel(input_path)
          if input_path.lower().endswith(('.xlsx', '.xls'))
          else pd.read_csv(input_path, low_memory=False))

    if n and n < len(df):
        df = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)
    total = len(df)
    print(f"  {total:,} records")

    missing = [c for c in ('subject', 'body') if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if 'language' not in df.columns:
        df['language'] = 'en'
    if 'idx' not in df.columns:           # preserve for merge safety
        df['idx'] = df.index

    print("\nLoading sentence ranker ...")
    ranker      = get_ranker()
    anchor_vecs = encode_anchors(ranker)
    print("  Anchor vectors encoded.")

    print("\nLoading SFT model ...")
    tok, model = get_model(student_base, adapter, adapter_subfolder,
                           profile['dtype'])

    rows_in  = df.to_dict(orient='records')
    records: List[Dict[str, Any]] = []
    t0 = time.time()

    for start in range(0, total, batch_size):
        batch_rows = rows_in[start:start + batch_size]

        # Stages 1-3: CPU, per row
        stage3_out = [process_row(r, anchor_vecs, ranker) for r in batch_rows]

        # Stage 4: GPU batched -- classification only
        label_out  = extract_labels_batch(tok, model, stage3_out,
                                           max_seq_len, max_new_tokens)

        for orig, s3, labels in zip(batch_rows, stage3_out, label_out):
            records.append({
                'idx'            : orig.get('idx', ''),
                'language'       : s3['language'],
                'subject'        : s3['subject'],
                'body'           : str(orig.get('body', '') or ''),
                # Core aspect fields (backward compatible with pipeline)
                'prob_sub'       : s3['prob_sub'],
                'prob_statement' : s3['prob_statement'],
                'cause'          : s3['cause'],
                'intent'         : labels['intent'],
                'ticket_type_nli': labels['ticket_type_nli'],
                # New fields
                'entities'       : s3['entities'],
                'action_taken'   : s3['action_taken'],
                'urgency'        : s3['urgency'],
            })

        done = min(start + batch_size, total)
        if done % max(batch_size * 5, 50) == 0 or done == total:
            el   = time.time() - t0
            mem  = torch.cuda.memory_allocated() / 1e9
            rate = done / max(el, 1e-6)
            print(f"  [{done:>6,}/{total:,}]  {rate:.2f} rec/s  GPU {mem:.1f} GB")

    out_df = pd.DataFrame(records)
    base   = re.sub(r'\.(csv|xlsx|xls)$', '', os.path.basename(input_path))
    if not out_path:
        out_path = f"aspects_{base}.csv"

    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}  ({len(out_df)} rows)")
    print(f"Output columns: {list(out_df.columns)}")
    return out_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Aspect Extractor -- Ranker + Entity + SFT labels')
    p.add_argument('--config',             default=None)
    p.add_argument('--input',              default=None)
    p.add_argument('--out',                default=None)
    p.add_argument('--n',                  type=int, default=None)
    p.add_argument('--student_base',       default=None)
    p.add_argument('--adapter',            default=None)
    p.add_argument('--adapter_subfolder',  default=None)
    p.add_argument('--batch',              type=int, default=0)
    p.add_argument('--max_seq_len',        type=int, default=0)
    p.add_argument('--max_new_tokens',     type=int, default=0)
    args = p.parse_args()

    cfg        = load_config(args.config)
    raw_in     = args.input or cfg.get('input')
    input_path = raw_in.strip() if isinstance(raw_in, str) else raw_in
    if not input_path:
        raise ValueError(
            "Missing input file.\n"
            "  Pass: --input /path/to/data.csv\n"
            f"  Config keys: {list(cfg.keys()) if cfg else '(empty)'}"
        )

    run(
        input_path        = input_path,
        out_path          = (args.out or cfg.get('out') or '').strip() or None,
        n                 = args.n if args.n is not None else cfg.get('n'),
        student_base      = (args.student_base
                             or cfg.get('student_base', STUDENT_BASE_DEFAULT)),
        adapter           = args.adapter or cfg.get('adapter', ADAPTER_DEFAULT),
        adapter_subfolder = (args.adapter_subfolder
                             or cfg.get('adapter_subfolder', ADAPTER_SUBF_DEFAULT)),
        batch_size        = int(args.batch or cfg.get('batch', 0) or 0),
        max_seq_len       = int(args.max_seq_len
                                or cfg.get('max_seq_len', MAX_SEQ_LEN_DEFAULT)),
        max_new_tokens    = int(args.max_new_tokens
                                or cfg.get('max_new_tokens', MAX_NEW_TOKENS_DEFAULT)),
    )
