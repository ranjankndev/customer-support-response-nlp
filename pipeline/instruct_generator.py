"""
instruct_generator.py
─────────────────────────────────────────────────────────────────────────────
Aspect-aware response generator using a causal instruct model (Qwen2.5-Instruct).

Key differences from response_generator.py (T5):
  ▸ Uses chat-template  system / user / assistant  format — not a flat string
  ▸ System prompt carries a STRUCTURED 4-SECTION DIRECTIVE derived from aspects
  ▸ Aspects are pre-processed into an action-plan before prompting
  ▸ Coverage check verifies all 4 sections actually appear in the output
  ▸ Level-2 retry fires when section coverage < 0.75

Supported models (swap via GEN_MODEL constant):
  "Qwen/Qwen2.5-1.5B-Instruct"   ← default, ~3 GB VRAM, good for demo
  "Qwen/Qwen2.5-3B-Instruct"     ← better quality, ~6 GB VRAM, final run
  "Qwen/Qwen2.5-7B-Instruct"     ← best quality, ~14 GB VRAM (T4 tight)

Usage (standalone):
  python instruct_generator.py \\
      --aspects /kaggle/input/cs-aspects-28k/aspects_customer_support_28k_fixed.csv \\
      --kb      /kaggle/input/datasets/.../customer_support_28k_fixed.csv \\
      --n       50 --demo

Usage (import):
  from instruct_generator import InstructGenerator, demo_run
  demo_run(aspects_path=..., kb_path=..., n=50)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import re, os, sys, logging, argparse, textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ── model config ──────────────────────────────────────────────────────────────
GEN_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"   # swap to 3B for final run
MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.3          # low = more deterministic / instruction-following
TOP_P           = 0.9
BODY_LEN        = 600          # max chars of ticket body passed to model
RETRIEVED_LEN   = 200          # max chars per retrieved answer shown to model

# ── aspect → tone mapping ─────────────────────────────────────────────────────
TONE_MAP = {
    "frustrated/urgent":   ("empathetic and urgent",
                            "Begin with a sincere apology. Acknowledge the business impact immediately."),
    "disappointed":        ("empathetic and reassuring",
                            "Acknowledge the frustration. Express commitment to resolve quickly."),
    "neutral/professional":("professional and clear",
                            "Be concise and solution-focused."),
    "positive/satisfied":  ("warm and helpful",
                            "Match the positive tone. Be friendly and thorough."),
}

def _tone_instructions(urgency_vibe: str) -> Tuple[str, str]:
    key = urgency_vibe.lower().strip() if urgency_vibe else ""
    for k, v in TONE_MAP.items():
        if k in key or key in k:
            return v
    return TONE_MAP["neutral/professional"]

# ── version / error regex (same as response_generator) ───────────────────────
_VERSION_RE = re.compile(r'\bv?\d+\.\d+(?:\.\d+)?\b', re.I)
_ERROR_RE   = re.compile(r'\b(?:error|err|exception|code|fault)[_\-\s]?\w+\b', re.I)

# ─────────────────────────────────────────────────────────────────────────────
# ASPECT SUMMARISER  — converts raw 6-aspect dict → action-plan dict
# This is the key piece missing from response_generator.py
# ─────────────────────────────────────────────────────────────────────────────

def build_aspect_summary(spans: Dict[str, str]) -> Dict[str, str]:
    """
    Turn the 6 raw aspect strings into a structured action-plan that
    the model can follow unambiguously.

    Returns a dict with keys:
      problem, cause, priority_label, tone_style, tone_instruction,
      action_plan  (human-readable ordered directive string)
    """
    prob_sub   = spans.get("prob_sub", "").strip()
    prob_stmt  = spans.get("prob_statement", "").strip()
    cause      = spans.get("cause", "").strip()
    priority   = spans.get("priority", "Medium").strip()
    urgency    = spans.get("urgency_vibe", "Neutral/Professional").strip()
    category   = spans.get("categorization", "").strip()

    # build PROBLEM line — prefer prob_sub + prob_statement together
    if prob_sub and prob_stmt:
        problem = f"{prob_sub} — {prob_stmt}"
    elif prob_sub:
        problem = prob_sub
    elif prob_stmt:
        problem = prob_stmt
    else:
        problem = "the reported issue"

    tone_style, tone_instruction = _tone_instructions(urgency)

    # build numbered action-plan
    steps = []
    steps.append(f"1. ACKNOWLEDGE  the specific problem: \"{problem}\"")
    if cause and cause.lower() not in ("none", "nan", ""):
        steps.append(f"2. EXPLAIN      the root cause: \"{cause}\"")
    else:
        steps.append("2. EXPLAIN      what you have investigated / are investigating")
    steps.append("3. RESOLVE      provide clear step-by-step fix or workaround")
    steps.append("4. PREVENT      close with one preventive tip or follow-up offer")

    action_plan = "\n".join(steps)

    return {
        "problem":           problem,
        "cause":             cause if cause and cause.lower() not in ("none","nan","") else "",
        "priority_label":    priority,
        "urgency_vibe":      urgency,
        "tone_style":        tone_style,
        "tone_instruction":  tone_instruction,
        "action_plan":       action_plan,
        "category":          category,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER  — chat-format system / user messages
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_messages(
    row: pd.Series,
    aspect_summary: Dict[str, str],
    retrieved: List[Dict],
    retry: bool = False,
) -> List[Dict]:
    """
    Build the system + user messages list for apply_chat_template().

    System prompt: carries the structured 4-section directive — aspects are DRIVERS here
    User prompt:   carries the actual ticket + retrieved resolution
    """
    s  = aspect_summary
    subject = str(row.get("subject", "")).strip()
    body    = str(row.get("body", ""))[:BODY_LEN].strip()

    # ── SYSTEM: aspect-driven directive ──────────────────────────────────────
    retry_emphasis = (
        "\n\nIMPORTANT: The previous attempt missed required sections. "
        "You MUST include all 4 numbered sections explicitly." if retry else ""
    )

    system_content = textwrap.dedent(f"""
        You are a senior customer support agent. Your task is to write a complete,
        professional support response that addresses every aspect of the customer's issue.

        TONE RULE: {s['tone_style']}
        {s['tone_instruction']}
        Priority: {s['priority_label']} — adjust urgency of resolution accordingly.

        YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE — no section may be skipped:

        {s['action_plan']}

        Key facts you MUST reference in the response:
        - Problem: {s['problem']}
        {"- Root cause: " + s['cause'] if s['cause'] else "- Root cause: under investigation"}
        {"- Category: " + s['category'] if s['category'] else ""}

        Do NOT write a generic response. Do NOT skip any numbered section.
        Do NOT start with "I" — start with an acknowledgement phrase.
        Keep the response between 80 and 200 words.{retry_emphasis}
    """).strip()

    # ── USER: ticket + retrieved context ─────────────────────────────────────
    ctx_parts = []
    for i, r in enumerate(retrieved[:2]):
        ans = str(r.get("answer", ""))[:RETRIEVED_LEN].strip()
        subj = str(r.get("subject", "")).strip()
        if ans:
            ctx_parts.append(f"[Resolution {i+1}] (re: {subj})\n{ans}")
    ctx_block = "\n\n".join(ctx_parts) if ctx_parts else "No similar resolution found."

    user_content = textwrap.dedent(f"""
        Customer ticket:
        Subject: {subject}
        Body: {body}

        Relevant past resolutions from knowledge base:
        {ctx_block}

        Write the structured support response now.
    """).strip()

    return [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": user_content},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# COVERAGE CHECKER — verifies all 4 sections + entity grounding
# ─────────────────────────────────────────────────────────────────────────────

def check_aspect_coverage(
    aspect_summary: Dict[str, str],
    response: str,
    spans: Dict[str, str],
) -> Dict:
    """
    Full coverage check across 4 dimensions:
      section_coverage : did the model produce all 4 structured sections?
      entity_coverage  : are key entities from prob_sub/prob_statement present?
      tone_match       : does tone align with urgency_vibe?
      overall_score    : weighted average
    """
    resp_lower = response.lower()

    # ── 1. Section coverage ───────────────────────────────────────────────────
    section_checks = {
        "acknowledge": any(w in resp_lower for w in [
            "apologize", "apology", "sorry", "understand", "acknowledge",
            "thank you for", "reaching out", "contacting"
        ]),
        "explain": any(w in resp_lower for w in [
            "cause", "reason", "due to", "because", "resulted", "this occurred",
            "investigating", "identified"
        ]),
        "resolve": any(w in resp_lower for w in [
            "please", "follow", "step", "click", "navigate", "try", "resolve",
            "fix", "solution", "workaround", "you can", "recommend"
        ]),
        "prevent": any(w in resp_lower for w in [
            "future", "prevent", "avoid", "tip", "suggest", "recommend",
            "ensure", "going forward", "if you", "feel free", "contact us"
        ]),
    }
    section_score = sum(section_checks.values()) / 4

    # ── 2. Entity coverage ────────────────────────────────────────────────────
    prob_sub  = spans.get("prob_sub", "")
    prob_stmt = spans.get("prob_statement", "")
    combined  = f"{prob_sub} {prob_stmt}"
    versions  = _VERSION_RE.findall(combined)
    errors    = _ERROR_RE.findall(combined)
    _SKIP = {'the','a','an','is','are','was','were','our','your','this','that',
             'issue','problem','error','bug','request','question','ticket','have','been'}
    core_nouns = [w for w in prob_sub.lower().split()[:6]
                  if len(w) > 3 and w not in _SKIP][:2]
    must_mention = list(set(versions + errors + core_nouns))
    if must_mention:
        missing   = [e for e in must_mention if e.lower() not in resp_lower]
        ent_score = 1.0 - len(missing) / len(must_mention)
    else:
        missing, ent_score = [], 1.0

    # ── 3. Tone match ─────────────────────────────────────────────────────────
    urgency = aspect_summary.get("urgency_vibe", "").lower()
    if "frustrated" in urgency or "urgent" in urgency:
        tone_ok = any(w in resp_lower for w in ["apologize","sorry","apology","inconvenience"])
    elif "disappointed" in urgency:
        tone_ok = any(w in resp_lower for w in ["understand","apologize","sorry","concern"])
    else:
        tone_ok = True   # neutral/positive — any professional response is fine
    tone_score = 1.0 if tone_ok else 0.0

    # ── overall weighted score ────────────────────────────────────────────────
    overall = round(0.50 * section_score + 0.35 * ent_score + 0.15 * tone_score, 3)

    return {
        "section_coverage":  round(section_score, 3),
        "sections_found":    section_checks,
        "entity_coverage":   round(ent_score, 3),
        "missing_entities":  ", ".join(missing) if missing else "",
        "tone_match":        tone_ok,
        "overall_score":     overall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCT GENERATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class InstructGenerator:

    def __init__(self, model_name: str = GEN_MODEL, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Loading instruct model: {model_name}  device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        log.info("Model loaded.")

    def _generate_from_messages(
        self, messages: List[Dict], retry: bool = False
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens   = MAX_NEW_TOKENS,
                temperature      = TEMPERATURE,
                top_p            = TOP_P,
                do_sample        = True,
                pad_token_id     = self.tokenizer.eos_token_id,
                repetition_penalty = 1.15,
            )
        # decode only newly generated tokens
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(
        self,
        row: pd.Series,
        spans: Dict[str, str],
        retrieved: List[Dict],
    ) -> Dict:
        """
        Full generation pipeline for one ticket:
          1. Build aspect summary (action plan)
          2. Build chat messages
          3. Generate response
          4. Check coverage
          5. If section_coverage < 0.75 → retry with harder prompt
        Returns dict with response + all coverage metrics + aspect_summary
        """
        aspect_summary = build_aspect_summary(spans)
        messages = build_chat_messages(row, aspect_summary, retrieved, retry=False)
        response = self._generate_from_messages(messages)
        coverage = check_aspect_coverage(aspect_summary, response, spans)

        # Level-2: retry if section coverage is poor
        if coverage["section_coverage"] < 0.75:
            log.debug("Coverage < 0.75, retrying with stronger prompt...")
            messages_retry = build_chat_messages(row, aspect_summary, retrieved, retry=True)
            response_retry = self._generate_from_messages(messages_retry, retry=True)
            coverage_retry = check_aspect_coverage(aspect_summary, response_retry, spans)
            if coverage_retry["overall_score"] > coverage["overall_score"]:
                response, coverage = response_retry, coverage_retry

        return {
            "response":         response,
            "aspect_summary":   aspect_summary,
            **coverage,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO RUN — 50 tickets with full per-ticket trace
# ─────────────────────────────────────────────────────────────────────────────

def _load_data(aspects_path: str, kb_path: str, n: int):
    aspects_df = pd.read_csv(aspects_path)
    kb_df      = pd.read_csv(kb_path)
    # use first n rows that have at least prob_sub or prob_statement
    mask = (
        aspects_df["prob_sub"].notna() | aspects_df["prob_statement"].notna()
    )
    sample = aspects_df[mask].head(n).reset_index(drop=True)
    log.info(f"Loaded {len(sample)} tickets for demo (from {len(aspects_df)} total)")
    return sample, kb_df


def _build_rag(kb_df: pd.DataFrame):
    """Lazy import so the file runs even if rag.py not in path."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from rag import RAGPipeline
        rag = RAGPipeline()
        rag.build_index(kb_df)
        log.info("RAG index built.")
        return rag
    except Exception as e:
        log.warning(f"Could not build RAG: {e}. Demo will run without retrieved context.")
        return None


def _format_ticket_trace(
    idx: int,
    row: pd.Series,
    spans: Dict[str, str],
    result: Dict,
) -> str:
    """Pretty-print a single ticket's full trace for the report."""
    cov = result
    as_ = result["aspect_summary"]
    secs = cov.get("sections_found", {})

    def tick(v): return "✓" if v else "✗"

    lines = [
        f"\n{'─'*72}",
        f"TICKET {idx+1:03d} | {str(row.get('language','?')).upper()} | "
        f"Priority: {spans.get('priority','?')} | "
        f"Category: {spans.get('categorization','?')}",
        f"{'─'*72}",
        "",
        "── ASPECTS EXTRACTED ──────────────────────────────────────────────────",
        f"  prob_sub       : {spans.get('prob_sub','-')}",
        f"  prob_statement : {spans.get('prob_statement','-')}",
        f"  cause          : {spans.get('cause','-') or '(not stated)'}",
        f"  priority       : {spans.get('priority','-')}",
        f"  urgency_vibe   : {spans.get('urgency_vibe','-')}",
        f"  categorization : {spans.get('categorization','-')}",
        "",
        "── ASPECT ACTION PLAN (driver sent to model) ──────────────────────────",
        f"  Problem  : {as_['problem']}",
        f"  Cause    : {as_['cause'] or '(under investigation)'}",
        f"  Tone     : {as_['tone_style']}  |  {as_['tone_instruction']}",
        f"  Sections:",
    ]
    for line in as_["action_plan"].split("\n"):
        lines.append(f"    {line}")

    lines += [
        "",
        "── GENERATED RESPONSE ─────────────────────────────────────────────────",
    ]
    for para in textwrap.wrap(result["response"], width=68):
        lines.append(f"  {para}")

    lines += [
        "",
        "── ASPECT COVERAGE ────────────────────────────────────────────────────",
        f"  {tick(secs.get('acknowledge'))} Acknowledge  "
        f"  {tick(secs.get('explain'))} Explain  "
        f"  {tick(secs.get('resolve'))} Resolve  "
        f"  {tick(secs.get('prevent'))} Prevent",
        f"  Section coverage : {cov['section_coverage']:.0%}  "
        f"| Entity coverage : {cov['entity_coverage']:.0%}  "
        f"| Tone match : {tick(cov['tone_match'])}",
        f"  Missing entities : {cov['missing_entities'] or 'none'}",
        f"  ► OVERALL SCORE  : {cov['overall_score']:.3f}",
    ]
    return "\n".join(lines)


def demo_run(
    aspects_path: str,
    kb_path: str,
    n: int = 50,
    model_name: str = GEN_MODEL,
    out_dir: str = "/kaggle/working",
):
    """
    Run the aspect-injection demo on n tickets.
    Saves:
      {out_dir}/demo_{n}_results.csv    — per-ticket metrics
      {out_dir}/demo_{n}_report.txt     — full human-readable trace
    """
    os.makedirs(out_dir, exist_ok=True)
    sample, kb_df = _load_data(aspects_path, kb_path, n)
    rag = _build_rag(kb_df)
    gen = InstructGenerator(model_name=model_name)

    ASPECT_COLS = ["prob_sub", "prob_statement", "cause",
                   "priority", "urgency_vibe", "categorization"]

    records   = []
    trace_lines = [
        "ASPECT-INJECTION DEMO REPORT",
        f"Model : {model_name}",
        f"Tickets: {n}",
        "=" * 72,
    ]

    for idx, row in sample.iterrows():
        spans = {
            k: str(row.get(k, "") or "").strip()
            for k in ASPECT_COLS
            if str(row.get(k, "") or "").strip() not in ("", "none", "nan")
        }

        # retrieve context
        if rag is not None:
            try:
                subject = str(row.get("subject", ""))
                body    = str(row.get("body", ""))
                lang    = str(row.get("language", "en")).lower()[:2]
                retrieved = rag.retrieve(
                    query        = f"{subject} {body}",
                    aspect_spans = spans,
                    subject      = subject,
                    lang         = lang,
                    top_k        = 3,
                )
            except Exception as e:
                log.warning(f"RAG retrieve failed for row {idx}: {e}")
                retrieved = []
        else:
            retrieved = []

        result = gen.generate(row=row, spans=spans, retrieved=retrieved)
        trace_lines.append(_format_ticket_trace(idx, row, spans, result))

        records.append({
            "ticket_idx":       idx,
            "language":         row.get("language", ""),
            "priority":         spans.get("priority", ""),
            "urgency_vibe":     spans.get("urgency_vibe", ""),
            "prob_sub":         spans.get("prob_sub", ""),
            "cause":            spans.get("cause", ""),
            "response":         result["response"],
            "section_coverage": result["section_coverage"],
            "entity_coverage":  result["entity_coverage"],
            "tone_match":       int(result["tone_match"]),
            "overall_score":    result["overall_score"],
            "missing_entities": result["missing_entities"],
        })

        # progress
        if (idx + 1) % 10 == 0:
            log.info(f"  {idx+1}/{n} done | "
                     f"avg overall={np.mean([r['overall_score'] for r in records]):.3f}")

    # ── summary stats ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    summary_lines = [
        "\n" + "=" * 72,
        "SUMMARY",
        "=" * 72,
        f"  Tickets processed       : {len(results_df)}",
        f"  Avg section coverage    : {results_df['section_coverage'].mean():.3f}",
        f"  Avg entity coverage     : {results_df['entity_coverage'].mean():.3f}",
        f"  Tone match rate         : {results_df['tone_match'].mean():.1%}",
        f"  Avg overall score       : {results_df['overall_score'].mean():.3f}",
        f"  Score >= 0.75 (good)    : {(results_df['overall_score'] >= 0.75).sum()} / {len(results_df)}",
        f"  Score <  0.50 (poor)    : {(results_df['overall_score'] <  0.50).sum()} / {len(results_df)}",
        "",
        "  Per-urgency breakdown:",
    ]
    for urg, grp in results_df.groupby("urgency_vibe"):
        summary_lines.append(
            f"    {urg:<30} n={len(grp):3d}  "
            f"overall={grp['overall_score'].mean():.3f}  "
            f"tone={grp['tone_match'].mean():.1%}"
        )
    trace_lines += summary_lines

    # ── save ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, f"demo_{n}_results.csv")
    txt_path = os.path.join(out_dir, f"demo_{n}_report.txt")

    results_df.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(trace_lines))

    log.info(f"Saved: {csv_path}")
    log.info(f"Saved: {txt_path}")
    print("\n".join(summary_lines))
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Instruct-model aspect-injection generator")
    p.add_argument("--aspects", required=True,
                   help="Path to aspects CSV (output of aspect_pipeline.py)")
    p.add_argument("--kb", required=True,
                   help="Path to full knowledge-base CSV (customer_support_28k_fixed.csv)")
    p.add_argument("--n", type=int, default=50,
                   help="Number of tickets to process (default 50)")
    p.add_argument("--model", default=GEN_MODEL,
                   help=f"Instruct model name (default: {GEN_MODEL})")
    p.add_argument("--out", default="/kaggle/working",
                   help="Output directory for CSV + report")
    p.add_argument("--demo", action="store_true",
                   help="Run demo mode (same as default, explicit flag for clarity)")
    args = p.parse_args()

    demo_run(
        aspects_path = args.aspects,
        kb_path      = args.kb,
        n            = args.n,
        model_name   = args.model,
        out_dir      = args.out,
    )
