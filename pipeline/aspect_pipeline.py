"""
aspect_pipeline_v4.py
═══════════════════════════════════════════════════════════════════════════════
6-Aspect Extraction Pipeline — Customer Support Tickets (EN + DE)

ASPECT   SOURCE          METHOD
───────────────────────────────────────────────────────────────────────────────
1  categorization   tag_1..tag_8      Your exact categorize_support_tickets()
2  prob_sub         subject+body      QA: roberta-base-squad2 (EN) / gelectra-base-germanquad (DE)
3  prob_statement   subject+body      QA: roberta-base-squad2 (EN) / gelectra-base-germanquad (DE)
4  cause            body              QA: roberta-base-squad2 (EN) / gelectra-base-germanquad (DE)
5  priority         priority field    Pass-through (normalise casing)
6  urgency_vibe     subject+body      Your exact get_vibe(text) via VADER
───────────────────────────────────────────────────────────────────────────────

RUN (Kaggle GPU — recommended):
  python aspect_pipeline_v4.py --input aspect_results_500.csv
  python aspect_pipeline_v4.py --input customer_support_28k_fixed.csv
  python aspect_pipeline_v4.py --input slected_record_cs.xlsx

RUN (local CPU — slow but works):
  pip install pandas openpyxl vaderSentiment transformers sentencepiece torch
  python aspect_pipeline_v4.py --input your_data.xlsx

After fine-tuning (step3):
  python aspect_pipeline_v4.py --input data.csv

OUTPUT:
  aspects_{name}.csv    machine-readable, use for RAG / LLM labelling
  aspects_{name}.xlsx   human-readable with column widths
"""

import re
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT   = 'aspect_results_500.csv'
QA_MODEL_EN     = 'deepset/roberta-base-squad2'          # F1 82.9 on SQuAD 2.0
QA_MODEL_DE     = 'deepset/gelectra-base-germanquad'      # F1 ~83+ on GermanQuAD
MAX_CONTEXT_LEN = 800    # chars fed to QA model for prob_sub / prob_statement
MAX_SUBJECT_LEN = 200    # chars fed to QA model for cause (subject only)

# QA confidence thresholds
THRESHOLDS = {
    'prob_sub':       0.05,   # component almost always present
    'prob_statement': 0.05,   # symptom always present
    'cause':          0.10,   # cause rarely stated in subject → higher bar
}

# Questions must exactly match step2_prepare_squad.py (fine-tuning alignment)
QUESTIONS = {
    'prob_sub': (
        "What specific system, component, service, product, or topic "
        "is affected or being asked about?"
    ),
    'prob_statement': (
        "What is the main problem, issue, symptom, error, or request "
        "being described?"
    ),
    'cause': (
        "What caused this problem or issue?"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# ASPECT 1: CATEGORIZATION
# YOUR EXACT categorize_support_tickets() function — zero changes
# ═══════════════════════════════════════════════════════════════════════════════

def categorize_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates tags and categorizes tickets into fixed buckets
    based on keyword priority logic.
    SOURCE: your exact specification — unchanged.
    """
    tag_cols = ['tag_1', 'tag_2', 'tag_3', 'tag_4',
                'tag_5', 'tag_6', 'tag_7', 'tag_8']

    def get_all_tags(row):
        tags = [str(row[col]).strip().lower()
                for col in tag_cols
                if col in row.index
                and pd.notnull(row[col])
                and str(row[col]).strip() != '']
        return ", ".join(tags)

    df['all_tags_combined'] = df.apply(get_all_tags, axis=1)

    buckets = {
        'security-issue':     ['security', 'breach', 'confidentiality',
                               'access control', 'unauthorized', 'login', 'password'],
        'outage':             ['outage', 'offline', 'down', 'unavailable',
                               'service disruption'],
        'network-issue':      ['network', 'connectivity', 'wifi', 'internet',
                               'bandwidth', 'connection'],
        'performance':        ['performance', 'slow', 'latency', 'speed',
                               'optimization', 'lag'],
        'bug-fix':            ['bug', 'software', 'fix', 'error', 'defect',
                               'crash', 'fail'],
        'incident-reporting': ['incident', 'disruption', 'issue', 'problem',
                               'recovery', 'critical'],
        'fraud':              ['fraud', 'scam', 'suspicious', 'identity theft'],
        'documentation':      ['documentation', 'guide', 'manual', 'how-to',
                               'instructions', 'faq'],
        'feedback':           ['feedback', 'suggestion', 'recommendation',
                               'improvement', 'feature request'],
    }

    priority_order = [
        'security-issue', 'outage', 'network-issue', 'performance',
        'bug-fix', 'fraud', 'incident-reporting', 'documentation', 'feedback',
    ]

    def map_to_bucket(tag_str):
        if not tag_str:
            return 'general-support'
        for bucket in priority_order:
            keywords = buckets[bucket]
            if any(kw in tag_str for kw in keywords):
                return bucket
        return 'general-support'

    df['categorization'] = df['all_tags_combined'].apply(map_to_bucket)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDERS (one for each QA source)
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_body(body: str) -> str:
    """Decode literal \\n stored in CSV/Excel back to real newlines."""
    return str(body or '').replace('\\n', '\n').replace('\\t', ' ')


def build_context_for_prob(subject: str, body: str,
                           tag_str: str, bucket: str) -> str:
    """
    Context for prob_sub and prob_statement.
    Sources: subject + body  (per spec)
    Prefix with tags + category so QA attention aligns to domain.

    Format: "Subject: {s}. Tags: {t}. Category: {c}. {body_text}"
    """
    body_clean = re.sub(r'\s+', ' ', _decode_body(body)).strip()
    subj = str(subject or '').strip()
    subj = '' if subj.lower() in ('nan', 'none', '') else subj

    parts = []
    if subj:     parts.append(f"Subject: {subj}.")
    if tag_str:  parts.append(f"Tags: {tag_str}.")
    if bucket:   parts.append(f"Category: {bucket}.")
    return (' '.join(parts) + ' ' + body_clean)[:MAX_CONTEXT_LEN]


def build_context_for_cause(subject: str, body: str) -> str:
    """
    Context for cause.
    Source: email BODY  (confirmed spec — cause is in the body text).

    Body contains sentences like:
      "this might be due to recent algorithm changes on social platforms"
      "Das Problem wird wahrscheinlich durch Serverüberlastung verursacht"
      "we believe the root cause might be a driver compatibility issue"
    QA returns 'none' when no causal language appears in body.

    Subject is prepended so QA has topic anchor, but the causal
    span itself is always extracted from body content.
    """
    body_clean = re.sub(r'\s+', ' ', _decode_body(body)).strip()
    subj = str(subject or '').strip()
    subj = '' if subj.lower() in ('nan', 'none', '') else subj
    prefix = f"Subject: {subj}. " if subj else ''
    return (prefix + body_clean)[:MAX_CONTEXT_LEN]


# ═══════════════════════════════════════════════════════════════════════════════
# ASPECTS 2, 3, 4: QA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_span(span: str) -> str:
    """Strip leading copulas and trim."""
    if not span:
        return 'none'
    span = re.sub(r'^(is|are|was|were|ist|sind|war|waren)\s+',
                  '', span, flags=re.IGNORECASE).strip()
    return span[:300] if len(span) > 3 else 'none'


def qa_extract(prob_ctx: str, cause_ctx: str, qa_pipe) -> dict:
    """
    Run three QA calls per ticket:
      prob_sub       — on subject+body context
      prob_statement — on subject+body context
      cause          — on subject+body context (cause lives in body)

    Returns dict with all three aspects.
    """
    result = {'prob_sub': 'none', 'prob_statement': 'none', 'cause': 'none'}

    if prob_ctx:
        for asp in ('prob_sub', 'prob_statement'):
            out = qa_pipe(
                question=QUESTIONS[asp],
                context=prob_ctx,
                handle_impossible_answer=True
            )
            if out['score'] >= THRESHOLDS[asp] and out['answer']:
                result[asp] = _clean_span(out['answer'].strip())

    if cause_ctx:
        out = qa_pipe(
            question=QUESTIONS['cause'],
            context=cause_ctx,
            handle_impossible_answer=True
        )
        if out['score'] >= THRESHOLDS['cause'] and out['answer']:
            result['cause'] = _clean_span(out['answer'].strip())

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ASPECT 5: PRIORITY
# Pass-through from existing field, normalise to High / Medium / Low
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_priority(val: str) -> str:
    v = str(val or '').strip().lower()
    if v in ('high', 'h', '1', 'critical', 'urgent', 'p1'):  return 'High'
    if v in ('medium', 'med', 'm', '2', 'normal', 'p2'):     return 'Medium'
    if v in ('low', 'l', '3', 'minor', 'p3', 'p4'):          return 'Low'
    return 'Medium'


# ═══════════════════════════════════════════════════════════════════════════════
# ASPECT 6: URGENCY VIBE
# YOUR EXACT get_vibe(text) function — zero changes to the function body.
# Applied to: subject + body combined into one text string.
# ═══════════════════════════════════════════════════════════════════════════════

_vader = SentimentIntensityAnalyzer()


def get_vibe(text: str) -> str:
    """
    YOUR EXACT function from the specification — unchanged.

    Thresholds:
      compound <= -0.5              → "Frustrated / Urgent"
      -0.5 < compound <= -0.05     → "Disappointed"
      -0.05 < compound < 0.05      → "Neutral / Professional"
      compound >= 0.05             → "Positive / Satisfied"
    """
    if not isinstance(text, str):
        return "Neutral"

    score    = _vader.polarity_scores(text)
    compound = score['compound']

    if compound <= -0.5:
        return "Frustrated / Urgent"
    elif -0.5 < compound <= -0.05:
        return "Disappointed"
    elif -0.05 < compound < 0.05:
        return "Neutral / Professional"
    else:
        return "Positive / Satisfied"


def get_vibe_for_ticket(subject: str, body: str) -> str:
    """
    Combine subject + body into one text string, then call get_vibe().
    Spec: "from subject, body" → both fields → single combined text.

    df['urgency_vibe'] = df.apply(
        lambda r: get_vibe_for_ticket(r['subject'], r['body']), axis=1)
    """
    subj = str(subject or '').strip()
    body_decoded = _decode_body(body)[:400]
    text = f"{subj} {body_decoded}".strip()
    return get_vibe(text)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADER  (handles CSV + XLSX + XLS)
# ═══════════════════════════════════════════════════════════════════════════════

def load_input(path: str) -> pd.DataFrame:
    if path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run(input_path: str) -> pd.DataFrame:
    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading: {input_path}")
    df_raw = load_input(input_path)
    n = len(df_raw)
    print(f"  {n:,} records | columns: {list(df_raw.columns)}\n")

    # ── Aspect 1: Categorization (DataFrame-level, exact spec function) ───────
    print("Aspect 1: Categorization (rule-based)...")
    df_raw = categorize_support_tickets(df_raw)
    print(f"  Done. Distribution: "
          f"{df_raw['categorization'].value_counts().to_dict()}\n")

    # ── Load dual QA models (language-routed) ────────────────────────────────
    print("Loading QA models (EN + DE)...")
    print("  (first run: ~450 MB each, ~60s total)")
    from transformers import pipeline as hf_pipeline
    _qa_en = hf_pipeline("question-answering", model=QA_MODEL_EN,
                          handle_impossible_answer=True)
    _qa_de = hf_pipeline("question-answering", model=QA_MODEL_DE,
                          handle_impossible_answer=True)
    def get_qa_pipe(lang: str):
        return _qa_de if lang.upper() == 'DE' else _qa_en
    print("  Models ready.\n")

    # ── Per-ticket extraction ─────────────────────────────────────────────────
    records = []
    t0 = time.time()

    for i, row in df_raw.iterrows():
        subject      = str(row.get('subject',  '') or '')
        body         = str(row.get('body',     '') or '')
        priority_raw = str(row.get('priority', 'medium') or 'medium')
        lang         = str(row.get('language', 'en') or 'en')
        t_type       = str(row.get('type',     '') or '')
        queue        = str(row.get('queue',    '') or '')
        tag_str      = str(row.get('all_tags_combined', '') or '')
        bucket       = str(row.get('categorization',   '') or '')

        # Aspect 2 + 3: prob_sub / prob_statement — QA on subject+body
        prob_ctx  = build_context_for_prob(subject, body, tag_str, bucket)

        # Aspect 4: cause — QA on subject+body (cause lives in body)
        cause_ctx = build_context_for_cause(subject, body)

        qa_out = qa_extract(prob_ctx, cause_ctx, get_qa_pipe(lang))

        # Aspect 5: priority — pass-through
        priority = normalise_priority(priority_raw)

        # Aspect 6: urgency_vibe — get_vibe(subject + body)
        urgency_vibe = get_vibe_for_ticket(subject, body)

        records.append({
            # Metadata
            'idx':           i,
            'language':      lang,
            'ticket_type':   t_type,
            'queue':         queue,
            'subject':       subject,
            # ── The 6 Aspects ─────────────────────────────────────────────
            'categorization':  bucket,
            'prob_sub':        qa_out['prob_sub'],
            'prob_statement':  qa_out['prob_statement'],
            'cause':           qa_out['cause'],
            'priority':        priority,
            'urgency_vibe':    urgency_vibe,
            # Supporting
            'all_tags_combined': tag_str,
        })

        if (i + 1) % 50 == 0:
            el   = time.time() - t0
            rem  = el / (i + 1) * (n - i - 1)
            rate = (i + 1) / el
            print(f"  [{i+1:>6,}/{n:,}]  {el/60:.1f}m elapsed  "
                  f"~{rem/60:.1f}m left  {rate:.1f} t/s")

    out_df = pd.DataFrame(records)

    # ── Save outputs ──────────────────────────────────────────────────────────
    base      = re.sub(r'\.(csv|xlsx|xls)$', '',
                       input_path.replace('\\', '/').split('/')[-1])
    csv_path  = f"aspects_{base}.csv"
    xlsx_path = f"aspects_{base}.xlsx"

    out_df.to_csv(csv_path, index=False)

    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            out_df.to_excel(writer, index=False, sheet_name='Aspects')
            ws = writer.sheets['Aspects']
            widths = {'A':6,'B':10,'C':14,'D':28,'E':38,
                      'F':22,'G':45,'H':65,'I':45,
                      'J':10,'K':24,'L':55}
            for col, w in widths.items():
                ws.column_dimensions[col].width = w
    except Exception as e:
        print(f"  (XLSX skipped: {e})")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═'*62}")
    print(f"  COMPLETE — {n:,} tickets in {elapsed:.0f}s "
          f"({n/elapsed:.1f} t/s)")
    print(f"{'═'*62}")

    print(f"\n  Aspect 1 — Categorization:")
    for cat, cnt in out_df['categorization'].value_counts().items():
        bar = '█' * max(1, int(cnt / n * 32))
        print(f"    {cat:22s} {cnt:6,}  {bar}")

    print(f"\n  Aspect 5 — Priority:")
    for p, cnt in out_df['priority'].value_counts().items():
        print(f"    {p:8s} {cnt:6,}  ({cnt/n*100:.0f}%)")

    print(f"\n  Aspect 6 — Urgency Vibe:")
    for v, cnt in out_df['urgency_vibe'].value_counts().items():
        print(f"    {v:25s} {cnt:6,}  ({cnt/n*100:.0f}%)")

    print(f"\n  Aspect 2 — prob_sub filled     : "
          f"{(out_df['prob_sub']!='none').sum():,}/{n:,}")
    print(f"  Aspect 3 — prob_statement filled: "
          f"{(out_df['prob_statement']!='none').sum():,}/{n:,}")
    print(f"  Aspect 4 — cause filled        : "
          f"{(out_df['cause']!='none').sum():,}/{n:,}  "
          f"(~{(out_df['cause']!='none').mean()*100:.0f}% — "
          f"cause appears in ~30% of ticket bodies)")

    print(f"\n  Saved → {csv_path}")
    print(f"  Saved → {xlsx_path}\n")

    return out_df


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='6-Aspect extraction pipeline (EN + DE support tickets)')
    parser.add_argument(
        '--input',  default=DEFAULT_INPUT,
        help='Input CSV or XLSX file')
    args = parser.parse_args()
    run(args.input)


# ═══════════════════════════════════════════════════════════════════════════════
# RAG INTEGRATION NOTES
# ═══════════════════════════════════════════════════════════════════════════════
"""
INDEXING — what to embed per ticket:
  text = f"Component: {prob_sub}. Problem: {prob_statement}. Cause: {cause}"
  Store all 6 aspects as metadata.

RETRIEVAL — at query time:
  1. Run this pipeline on the incoming ticket
  2. Build the same embed text
  3. Optionally pre-filter by categorization (outage → retrieve outage tickets)
  4. Retrieve top-K similar past tickets by embedding similarity

GENERATION — build LLM prompt:
  "New ticket: Component={prob_sub}, Problem={prob_statement},
   Cause={cause}, Priority={priority}, Vibe={urgency_vibe}.
   Similar past tickets: {retrieved}. Generate a response."
  Urgency vibe drives tone:
    Frustrated/Urgent → acknowledge impact in first sentence
    Neutral/Professional → standard informative response

FINE-TUNING LOOP:
  step1_llm_labeller.py  → label prob_sub/prob_statement/cause for 500 tickets
  step2_prepare_squad.py → SQuAD format, 3 QA pairs per ticket
  step3_finetune_qa.py   → fine-tune xlm-roberta on Kaggle GPU
  After fine-tuning, update QA_MODEL_EN / QA_MODEL_DE constants to finetuned_qa_en/ finetuned_qa_de/
"""
