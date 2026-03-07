"""
step1_llm_labeller.py — LLM-as-Labeller (updated for 6-aspect design)
Labels: prob_sub / prob_statement / cause  (verbatim spans from body)

RUN locally:  pip install anthropic pandas
              export ANTHROPIC_API_KEY=sk-ant-...
              python step1_llm_labeller.py
COST: ~$0.50  TIME: ~15 min
"""
import re, json, time
import anthropic
import pandas as pd

INPUT_CSV  = 'aspect_results_500.csv'
OUTPUT_CSV = 'llm_labels_500.csv'
MODEL      = 'claude-sonnet-4-20250514'
DELAY_SEC  = 1.0
MAX_RETRY  = 3

client = anthropic.Anthropic()

SYSTEM = """You extract verbatim spans from multilingual support tickets (EN/DE).
CRITICAL RULES:
1. Every value MUST be copied word-for-word from the BODY — no paraphrasing
2. If the aspect is absent, return "none"
3. Return ONLY valid JSON, no markdown fences"""

def make_prompt(subject, body, ticket_type, lang):
    return f"""Extract from this {lang.upper()} ticket (type={ticket_type}):

SUBJECT: {subject}
BODY: {body}

Extract as VERBATIM spans from BODY:

"prob_sub": The specific system, component, service, product, or topic affected.
  - Copy the technical noun phrase: e.g. "centralized account management portal"
  - For requests: the topic being asked about
  - Do NOT copy generic words like "problem" or "issue"

"prob_statement": The main symptom, impact, or request description.
  - A complete clause explaining what is wrong or wanted
  - e.g. "portal appears to be offline, blocking access to account settings"
  - For requests: e.g. "request detailed information about smart home integration"

"cause": The explicitly stated reason the problem occurred.
  - Extract from BODY — cause is in the ticket body text
  - Look for: because, due to, caused by, might be due to, aufgrund, durch, zurückzuführen
  - Return "none" if no cause is explicitly stated in the body

Return ONLY this JSON:
{{"prob_sub": "...", "prob_statement": "...", "cause": "..."}}"""

def verify(span, body):
    if not span or span == 'none':
        return True
    return span.lower().strip() in body.lower()

def label(subject, body, ticket_type, lang):
    for attempt in range(MAX_RETRY):
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=300, system=SYSTEM,
                messages=[{"role":"user",
                           "content": make_prompt(subject, body[:700],
                                                  ticket_type, lang)}]
            )
            raw = re.sub(r'```(?:json)?|```', '', r.content[0].text).strip()
            d   = json.loads(raw)
            out = {}
            for k in ('prob_sub', 'prob_statement', 'cause'):
                v = str(d.get(k, 'none') or 'none').strip()[:250]
                out[k] = v if (v == 'none' or verify(v, body)) else 'none'
            return out
        except json.JSONDecodeError:
            print(f"  JSON error attempt {attempt+1}")
        except Exception as e:
            print(f"  API error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return {'prob_sub':'none', 'prob_statement':'none', 'cause':'none'}

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Tickets: {len(df):,} | EN={(df['language']=='en').sum()} "
          f"DE={(df['language']=='de').sum()}\n")

    prob_subs, prob_stmts, causes = [], [], []
    t0 = time.time()

    for i, row in df.iterrows():
        subj  = str(row.get('subject','') or '')
        body  = str(row.get('body','') or '').replace('\\n','\n')
        lang  = str(row.get('language','en') or 'en')
        ttype = str(row.get('type','Incident') or 'Incident')

        if not body.strip() or len(body.split()) < 3:
            prob_subs.append('none')
            prob_stmts.append('none')
            causes.append('none')
            continue

        L = label(subj, body, ttype, lang)
        prob_subs.append(L['prob_sub'])
        prob_stmts.append(L['prob_statement'])
        causes.append(L['cause'])

        if (i+1) % 50 == 0:
            el  = time.time()-t0
            rem = el/(i+1)*(len(df)-i-1)
            valid = sum(1 for p in prob_subs if p != 'none')
            print(f"  [{i+1:3d}/500] {el/60:.1f}m | ~{rem/60:.1f}m left "
                  f"| valid prob_sub={valid}/{len(prob_subs)}")
        time.sleep(DELAY_SEC)

    df['prob_sub_llm']       = prob_subs
    df['prob_statement_llm'] = prob_stmts
    df['cause_llm']          = causes
    df['label_valid']        = df['prob_sub_llm'] != 'none'
    df.to_csv(OUTPUT_CSV, index=False)

    n = len(df)
    print(f"\n{'='*52}")
    print(f"  prob_sub filled      : {(df['prob_sub_llm']!='none').sum()}/{n}")
    print(f"  prob_statement filled: {(df['prob_statement_llm']!='none').sum()}/{n}")
    print(f"  cause filled         : {(df['cause_llm']!='none').sum()}/{n}")
    print(f"  Saved → {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
