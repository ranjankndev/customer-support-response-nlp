"""
corpus_builder.py  (v2 — with TicketCleaner boilerplate removal)
=================================================================

CLEANING PIPELINE (runs BEFORE tokenization):
  Uses statistical boilerplate discovery from ticket_cleaner.py logic,
  adapted to work without spaCy (pure regex + frequency analysis).

  Stage 1 — Structural regex stripping (language-aware):
    • Greeting lines:     "Dear Customer Support Team," -> removed
    • Filler sentences:   "I hope this message finds you well." -> removed
    • Sign-offs:          "Thank you for your assistance." -> removed
    • Excel \\n artifacts: literal backslash-n -> normalized to space

  Stage 2 — Data-driven boilerplate discovery (from ticket_cleaner.py logic):
    • Position-bias analysis: tokens appearing 2.5× more in first 15 words
    • Mutual Information filter: low-MI tokens add no category signal
    • Diversity filter: tokens appearing in >25% of tickets = noise

  Stage 3 — Tokenization stopword filter:
    • Domain-specific stopwords: "dear", "team", "hope", "regards"
    • Standard EN/DE function words

WHY THIS MATTERS FOR EMBEDDINGS:
  Without cleaning, "dear" (freq=113) and "team" (freq=110) become
  the most frequent tokens and embed near EVERY ticket type.
  After cleaning:
    "outage"  -> clusters near: offline, blocked, unavailable, down
    "billing" -> clusters near: payment, invoice, charge, refund
    Embedding space reflects PROBLEM semantics, not email conventions.

SENTENCE CONSTRUCTION (7 types per ticket):
  Tags co-occur with body start words via WINDOW_SIZE=3 context window
"""

import re
import math
import pandas as pd
from typing import List, Tuple, Dict
from collections import Counter, defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: STRUCTURAL BOILERPLATE REMOVAL (regex, language-aware)
# Logic adapted directly from ticket_cleaner.py GREETING_PATTERNS + FILLER_SENTENCES
# ─────────────────────────────────────────────────────────────────────────────

_GREETING_EN = re.compile(
    r'(?:dear\s+[\w\s,\.]{0,40}?[,\n]\s*'
    r'|hello\s*[\w\s]{0,20}?[,\n]\s*'
    r'|hi\s*[\w\s]{0,15}?[,\n]\s*)',
    re.I
)

_GREETING_DE = re.compile(
    r'(?:sehr\s+geehrte[rs]?\s+[\w\s\-,\.]{0,40}?[,\n]\s*'
    r'|guten\s+(?:morgen|tag|abend)\s*[\w\s]{0,15}?[,\n]\s*'
    r'|hallo\s*[\w\s]{0,15}?[,\n]\s*'
    r'|liebe[rs]?\s+[\w\s]{0,20}?[,\n]\s*)',
    re.I
)

# Filler sentences — same patterns as ticket_cleaner.FILLER_SENTENCES
_FILLER_EN = re.compile(
    r'I\s+hope\s+this\s+message\s+(?:finds|reaches)\s+you\s+well[\w\s,]*?[.!]\s*|'
    r'I\s+am\s+(?:writing|reaching\s+out)\s+to\s+(?:inform|request|inquire|clarify|report|bring)[^.!?]*[.!?]\s*|'
    r'(?:Thank|Thanks)\s+you\s+for\s+(?:your\s+)?(?:time|support|assistance|help|reading|reaching)[^.!?]*[.!?]\s*|'
    r'I\s+look\s+forward\s+to\s+hearing[^.!?]*[.!?]\s*|'
    r'Please\s+(?:do\s+not\s+hesitate|feel\s+free)\s+to[^.!?]*[.!?]\s*|'
    r'Best\s+regards[\w\s,]*?[.!]?\s*|'
    r'Kind\s+regards[\w\s,]*?[.!]?\s*|'
    r'Yours\s+(?:sincerely|faithfully)[\w\s,]*?[.!]?\s*',
    re.I | re.DOTALL
)

_FILLER_DE = re.compile(
    r'Ich\s+hoffe[^.!?]*[.!?]\s*|'
    r'Ich\s+(?:schreibe|wende\s+mich)[^.!?]*[.!?]\s*|'
    r'(?:Vielen\s+Dank|Danke)[^.!?]*[.!?]\s*|'
    r'Mit\s+(?:freundlichen|besten)\s+Grüßen[^.!?]*[.!?]?\s*|'
    r'Ich\s+freue\s+mich[^.!?]*[.!?]\s*',
    re.I
)


def strip_boilerplate(text: str, lang: str = 'en') -> str:
    """
    Remove greeting, filler and sign-off boilerplate from ticket body.
    Adapts ticket_cleaner.py TicketPreprocessor.strip() without spaCy.
    Returns cleaned substantive content only.
    """
    if not text:
        return ''
    # Fix Excel storage artifacts: literal \\n -> actual whitespace
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
    text = text.replace('\r', ' ')

    if lang == 'de':
        text = _GREETING_DE.sub('', text, count=1).strip()
        text = _FILLER_DE.sub('', text).strip()
    else:
        text = _GREETING_EN.sub('', text, count=1).strip()
        text = _FILLER_EN.sub('', text).strip()

    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,.\-:\s]+', '', text).strip()  # strip leading noise
    return text


# ─────────────────────────────────────────────────────────────────────────────
# GERMAN COMPOUND WORD SPLITTER
#
# German compounds are single tokens built from 2+ meaningful words:
#   "Netzwerkverbindung" = Netzwerk + Verbindung (network + connection)
#   "Serverausfall"      = Server   + Ausfall    (server  + outage)
#   "Verbindungsfehler"  = Verbindung + Fehler   (connection + error)
#
# WITHOUT splitting, Skip-Gram with window=3 sees:
#   context("netzwerkverbindung") = [die, ist, ausgefallen]
#   -> single OOV token, no embedding learned, never matches queries
#
# WITH splitting -> "netzwerk verbindung":
#   context("netzwerk")   = [die, verbindung, ist]   <- learns network associations
#   context("verbindung") = [netzwerk, ist, ausgefallen] <- learns connection context
#   -> both tokens get embeddings + correct domain neighbours
# ─────────────────────────────────────────────────────────────────────────────

# 22 common German IT-support compound suffixes + plural forms
# Ordered longest-first so "aktualisierung" matches before "ierung"
_DE_SUFFIXES = [
    'aktualisierungen', 'aktualisierung',
    'einstellungen', 'einstellung',
    'schnittstellen', 'schnittstelle',
    'verbindungen', 'verbindung',
    'verwaltungen', 'verwaltung',
    'anmeldungen', 'anmeldung',
    'protokolle', 'protokoll',
    'sicherheit', 'netzwerk',
    'ausfälle', 'ausfall',
    'störungen', 'störung',
    'probleme', 'problem',
    'zugänge', 'zugang',
    'zugriff', 'software',
    'dienste', 'dienst',
    'fehler', 'system',
    'portal', 'server',
    'konten', 'konto',
    'daten',
]

def split_german_compound(word: str) -> List[str]:
    """
    Split a German compound noun at a known IT-support suffix boundary.
    Handles the German binding-s ("Verbindungs-fehler" -> verbindung + fehler).
    Returns [word] unchanged if no suffix matches.

    "Netzwerkverbindung"     -> ["netzwerk",   "verbindung"]
    "Verbindungsfehler"      -> ["verbindung",  "fehler"]
    "Verbindungseinstellung" -> ["verbindung",  "einstellung"]  (strips binding-s)
    "Serverausfall"          -> ["server",      "ausfall"]
    "online"                 -> ["online"]  (no match)
    """
    w = word.lower()
    for suffix in _DE_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 3:
            prefix = w[: -len(suffix)]
            # Strip German binding-s (Genitiv-s): "verbindungs" -> "verbindung"
            if prefix.endswith('s') and len(prefix) > 4:
                prefix = prefix[:-1]
            if len(prefix) >= 3:
                return [prefix, suffix]
    return [word]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: DATA-DRIVEN BOILERPLATE DISCOVERY
# Adapted from ticket_cleaner.compute_position_bias + compute_mutual_information
# Returns a set of TOKEN strings to filter during tokenization
# ─────────────────────────────────────────────────────────────────────────────

def _discover_boilerplate_tokens(texts: List[str],
                                  tags: List[str],
                                  window: int = 15,
                                  position_thresh: float = 2.5,
                                  diversity_thresh: float = 0.25,
                                  mi_thresh: float = 0.02,
                                  min_freq: int = 5) -> set:
    """
    Adapted from ticket_cleaner.discover_boilerplate().
    Finds tokens that are:
      - 2.5× more frequent in first 15 body tokens than overall  (position bias)
      - Present in >25% of all tickets                           (diversity)
      - MI < 0.02 with ticket category tags                      (non-discriminating)

    These tokens contaminate embedding space because they co-occur with
    every ticket type rather than specific problem types.
    """
    n_docs       = len(texts)
    front_counts = Counter()
    full_counts  = Counter()
    doc_counts   = Counter()

    for text in texts:
        toks  = re.sub(r'[^\w\s]', ' ', str(text).lower()).split()
        front = toks[:window]
        front_counts.update(front)
        full_counts.update(toks)
        seen = set(toks[:window])
        for t in seen:
            doc_counts[t] += 1

    # MI calculation — same as ticket_cleaner.compute_mutual_information
    n_total    = n_docs
    phrase_tag = defaultdict(Counter)
    p_freq     = Counter()
    for text, tag in zip(texts, tags):
        toks = re.sub(r'[^\w\s]', ' ', str(text).lower()).split()[:window]
        seen = set()
        for t in toks:
            if t not in seen:
                phrase_tag[t][tag] += 1
                p_freq[t] += 1
                seen.add(t)

    tag_freq = Counter(tags)
    mi_scores = {}
    for phrase, p_count in p_freq.items():
        if p_count < min_freq:
            continue
        mi = 0.0
        p_x = p_count / n_total
        for tag, pt_count in phrase_tag[phrase].items():
            p_y  = tag_freq[tag] / n_total
            p_xy = pt_count / n_total
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
        mi_scores[phrase] = max(0.0, mi)

    boilerplate = set()
    for tok in front_counts:
        if full_counts.get(tok, 0) < min_freq:
            continue
        pos_bias  = front_counts[tok] / full_counts.get(tok, 1)
        diversity = doc_counts.get(tok, 0) / n_docs
        mi        = mi_scores.get(tok, 0.0)
        if pos_bias >= position_thresh and diversity >= diversity_thresh and mi <= mi_thresh:
            boilerplate.add(tok)

    return boilerplate


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: TOKENIZATION WITH DOMAIN STOPWORDS
# ─────────────────────────────────────────────────────────────────────────────

# Domain stopwords: email-convention words with no problem-signal value.
# "dear" appears in 113/120 bodies -> embeds near everything -> noise.
# This list extends ticket_cleaner.DOMAIN_STOPWORDS with support-specific terms.
SUPPORT_STOPWORDS_EN = {
    # Greeting / salutation
    'dear', 'hello', 'team', 'hope', 'regards', 'sincerely', 'faithfully',
    # Meta-communication (the act of writing, not the problem)
    'writing', 'reaching', 'inquire', 'clarify', 'report', 'inform',
    'kindly', 'appreciate', 'hesitate',
    # Generic function words
    'the', 'to', 'and', 'is', 'in', 'of', 'for', 'on', 'at', 'by',
    'it', 'be', 'as', 'an', 'this', 'that', 'are', 'was', 'were',
    'with', 'or', 'but', 'not', 'from', 'have', 'has', 'had',
    'will', 'would', 'could', 'should', 'may', 'can', 'do', 'did',
    'my', 'your', 'our', 'we', 'you', 'am', 'me', 'us', 'they',
    # Answer-side boilerplate (present in 'answer' field)
    'apologize', 'inconvenience', 'patience', 'soon', 'possible', 'actively',
}

SUPPORT_STOPWORDS_DE = {
    # Greeting / salutation
    'sehr', 'geehrte', 'geehrter', 'geehrtes', 'liebe', 'lieber',
    'guten', 'morgen', 'bitte', 'danke', 'freundlichen', 'grüßen',
    'mfg', 'schreibe', 'wende', 'hoffe', 'möchte', 'würde', 'könnte',
    # Pronouns + possessives (common in DE tickets, no problem signal)
    'ich', 'sie', 'wir', 'mein', 'meine', 'meinen', 'meinem', 'meiner',
    'ihr', 'ihre', 'ihren', 'ihrem', 'ihrer', 'unser', 'unsere',
    # Function words
    'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'ist', 'sind',
    'hat', 'haben', 'mit', 'von', 'für', 'auf', 'nach', 'bei', 'beim',
    'den', 'dem', 'des', 'zum', 'zur', 'aus', 'als', 'über', 'unter',
    'vor', 'an', 'im', 'am', 'zu', 'er', 'es', 'nicht', 'auch',
    'noch', 'aber', 'wenn', 'dass', 'sich', 'wie', 'was', 'einen',
    'diese', 'dieser', 'diesem', 'diesen', 'dieses',
    'seit', 'werden', 'wurde', 'worden', 'sein',
}

MIN_TOKEN_LEN = 3


def normalize(text: str) -> str:
    """
    Lowercase, fix Excel escape artifacts, strip punctuation.
    Hyphens in compound words like 'support-team' are split into
    separate tokens so each part gets filtered by stopwords independently.
    """
    text = str(text).lower().strip()
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
    text = re.sub(r'[^\w\s]', ' ', text)   # hyphens -> space (splits support-team)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str,
             lang: str = 'en',
             stopwords: set = None,
             learned_boilerplate: set = None) -> List[str]:
    """
    Language-aware tokenizer.

    EN: standard tokenization -> stopword filter
    DE: tokenization -> compound splitting -> stopword filter

    Mixed-language tickets (e.g. "Dear Support, mein Account ist offline"):
    Always apply BOTH EN+DE stopwords regardless of lang flag,
    because DE tickets commonly contain EN terms and vice versa.
    Code-switching is real in support tickets — "mein Account Portal
    ist offline" has both DE function words and EN domain words.
    """
    sw = stopwords or set()
    bp = learned_boilerplate or set()

    # Always include both EN+DE stopwords to handle code-switching
    combined_sw = sw | SUPPORT_STOPWORDS_EN | SUPPORT_STOPWORDS_DE

    tokens = normalize(text).split()
    result = []

    for tok in tokens:
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in combined_sw or tok in bp:
            continue

        if lang == 'de':
            # Attempt compound split — if split found, add both parts
            parts = split_german_compound(tok)
            for p in parts:
                if len(p) >= MIN_TOKEN_LEN and p not in combined_sw and p not in bp:
                    result.append(p)
        else:
            result.append(tok)

    return result


def tokenize_tag(tag: str) -> List[str]:
    """
    Tags are NOT stopword-filtered — category-signal tokens.
    Handles three tag formats found in the dataset:

    Single value  : 'Tech Support'   -> ['tech', 'support']
    Multi-value   : 'outage,billing' -> ['outage', 'billing']  (comma-separated)
    Hyphenated    : 'cloud-native'   -> ['cloud', 'native']
    Typos fixed   : 'feeedback'      -> 'feedback' (triple-e)
                    'documentatoin'  -> 'documentation'
                    'documentatio'   -> 'documentation'
    """
    # Fix known typos discovered in OOV analysis
    TYPO_MAP = {
        'feeedback':     'feedback',
        'documentatoin': 'documentation',
        'documentatio':  'documentation',
    }
    tag = TYPO_MAP.get(tag.strip().lower(), tag)

    # Replace commas with spaces to handle multi-value tags
    # 'crash,performance,outage' -> 'crash performance outage'
    tag = tag.replace(',', ' ')

    return [t for t in normalize(tag).split() if len(t) >= 2]


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(df: pd.DataFrame,
                 verbose: bool = True) -> Tuple[List[List[str]], dict]:
    """
    Build training corpus from all ticket fields with full 3-stage cleaning.

    Returns (sentences, stats).
    stats includes token reduction metrics so you can verify cleaning quality.
    """
    # ── Auto-fix encoding before anything else ────────────────────────────────
    # Same fix as multilingual_oov_check.py — handles mojibake in DE text
    _MOJIBAKE = {
        '\u00c3\u00bc': 'ü', '\u00c3\u00b6': 'ö', '\u00c3\u00a4': 'ä',
        '\u00c3\u009f': 'ß', '\u00c3\u009c': 'Ü', '\u00c3\u0096': 'Ö',
        '\u00c3\u0084': 'Ä',
    }
    _MRE = re.compile('|'.join(re.escape(k) for k in sorted(_MOJIBAKE, key=len, reverse=True)))
    def _fix(text):
        fixed = _MRE.sub(lambda m: _MOJIBAKE[m.group(0)], str(text))
        try:
            import ftfy; return ftfy.fix_text(fixed)
        except ImportError:
            return fixed

    garbled = sum(1 for t in df['body'].fillna('') if '\u00c3' in str(t))
    if garbled > 0:
        if verbose:
            print(f"[Corpus] Auto-fixing encoding in {garbled} rows...")
        for col in [c for c in ['body','subject','answer'] if c in df.columns]:
            df = df.copy()
            df[col] = df[col].fillna('').apply(
                lambda x: _fix(x) if '\u00c3' in str(x) else str(x)
            )

    tag_cols = [c for c in df.columns if c.startswith('tag_')]
    lang_col = 'language' if 'language' in df.columns else None

    # Stage 2: discover boilerplate from raw corpus BEFORE structural cleaning
    # (raw text used here to catch anything the regex might miss)
    raw_bodies = df['body'].fillna('').astype(str).tolist()
    tag_for_mi = []
    for _, row in df.iterrows():
        vals = [str(row[tc]) for tc in tag_cols
                if pd.notna(row.get(tc)) and str(row.get(tc)) not in ('nan', '')]
        tag_for_mi.append(vals[0] if vals else 'unknown')

    learned_bp = _discover_boilerplate_tokens(raw_bodies, tag_for_mi)
    if verbose:
        print(f"[Cleaner] Stage 2 learned boilerplate tokens ({len(learned_bp)}): "
              f"{sorted(learned_bp)[:20]}")

    sentences = []
    stats = {
        'tickets':        0,
        'sentences':      0,
        'tokens_raw':     0,
        'tokens_clean':   0,
    }

    for _, row in df.iterrows():
        stats['tickets'] += 1
        lang = 'en'
        if lang_col and pd.notna(row.get(lang_col)):
            lang = str(row[lang_col]).lower()[:2]
        sw = SUPPORT_STOPWORDS_DE if lang == 'de' else SUPPORT_STOPWORDS_EN

        raw_body = str(row.get('body', '') or '')
        stats['tokens_raw'] += len(raw_body.split())

        # Stage 1: structural boilerplate removal
        clean_body    = strip_boilerplate(raw_body, lang)
        clean_subject = strip_boilerplate(str(row.get('subject', '') or ''), lang)
        clean_answer  = strip_boilerplate(str(row.get('answer', '')  or ''), lang)

        # Stage 3: tokenize with lang + stopwords + stage-2 learned boilerplate
        subj_tok = tokenize(clean_subject, lang, sw, learned_bp)
        body_tok = tokenize(clean_body,    lang, sw, learned_bp)
        ans_tok  = tokenize(clean_answer,  lang, sw, learned_bp)
        stats['tokens_clean'] += len(body_tok)

        # Tags: no stopword filter (category-signal tokens)
        tag_tok = []
        for tc in tag_cols:
            val = row.get(tc)
            if pd.notna(val) and str(val).strip() not in ('', 'nan'):
                tag_tok.extend(tokenize_tag(str(val)))

        # Metadata: queue / priority / type
        meta_tok = []
        for field in ('queue', 'priority', 'type'):
            val = row.get(field)
            if pd.notna(val) and str(val).strip() not in ('', 'nan'):
                meta_tok.extend(tokenize(str(val), stopwords=set()))

        # ── 7 sentence types per ticket ───────────────────────────────────────
        if subj_tok:
            sentences.append(subj_tok)                     # 1: subject

        if body_tok:
            sentences.append(body_tok)                     # 2: body (clean)

        if ans_tok:
            sentences.append(ans_tok)                      # 3: answer (clean)

        if subj_tok and tag_tok:
            sentences.append(subj_tok + tag_tok)           # 4: subject + tags

        if tag_tok and body_tok:
            sentences.append(tag_tok + body_tok[:50])      # 5: tags + body start

        if meta_tok and subj_tok:
            sentences.append(meta_tok + subj_tok)          # 6: metadata + subject

        if tag_tok and ans_tok:
            sentences.append(tag_tok + ans_tok[:30])       # 7: tags + answer start

    stats['sentences']   = len(sentences)
    reduction            = (1 - stats['tokens_clean'] / max(stats['tokens_raw'], 1)) * 100
    if verbose:
        print(f"[Corpus] {stats['tickets']} tickets -> {stats['sentences']} sentences")
        print(f"[Corpus] Boilerplate reduction: {reduction:.1f}% "
              f"({stats['tokens_raw']:,} -> {stats['tokens_clean']:,} tokens)")

    return sentences, stats


def get_corpus_stats(sentences: List[List[str]]) -> dict:
    all_tokens = [t for s in sentences for t in s]
    counts     = Counter(all_tokens)
    return {
        'total_tokens':  len(all_tokens),
        'unique_tokens': len(counts),
        'top_30':        counts.most_common(30),
        'singletons':    sum(1 for c in counts.values() if c == 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run directly to build and save corpus
# python corpus_builder.py --data customer_support_28k_fixed.csv
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, pickle
    p = argparse.ArgumentParser(description='Build support ticket training corpus')
    p.add_argument('--data',   required=True,  help='Path to .xlsx or .csv')
    p.add_argument('--output', default='support_corpus.pkl',
                   help='Output pickle path (default: support_corpus.pkl)')
    args = p.parse_args()

    print(f"Loading {args.data}...")
    df = pd.read_excel(args.data) if args.data.endswith(('.xlsx','.xls')) \
        else pd.read_csv(args.data)
    print(f"Loaded {len(df)} tickets")

    sentences, stats = build_corpus(df, verbose=True)

    vocab = get_corpus_stats(sentences)
    print(f"\n=== CORPUS STATS ===")
    print(f"Sentences  : {stats['sentences']:,}")
    print(f"Tokens     : {vocab['total_tokens']:,}")
    print(f"Vocabulary : {vocab['unique_tokens']:,} unique tokens")
    print(f"Singletons : {vocab['singletons']:,} (appear only once)")
    print(f"\nTop 30 tokens (sanity check — should be domain words, not boilerplate):")
    for word, count in vocab['top_30']:
        print(f"  {word:20s}: {count:,}")

    with open(args.output, 'wb') as f:
        pickle.dump(sentences, f)
    print(f"\nCorpus saved -> {args.output}")
    print(f"Next step: python domain_skipgram.py --data {args.data}")
