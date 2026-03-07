"""
step2_prepare_squad.py — Convert LLM labels → SQuAD 2.0 format
Updated for 6-aspect design: prob_sub / prob_statement / cause

RUN locally (pure Python):
  python step2_prepare_squad.py

INPUT:  llm_labels_500.csv
OUTPUT: squad_train.json  (300 tickets, 3 QA pairs each = 900 QA pairs)
        squad_val.json    (100 tickets)
        squad_test.json   (100 tickets)
"""
import re, json, random
import pandas as pd

INPUT_CSV  = 'llm_labels_500.csv'
TRAIN_JSON = 'squad_train.json'
VAL_JSON   = 'squad_val.json'
TEST_JSON  = 'squad_test.json'
SEED       = 42
BODY_LEN   = 800   # max chars from body used in SQuAD context

# Must match exactly what aspect_pipeline_v4.py uses at inference time
QUESTIONS = {
    'prob_sub': (
        "What specific system, component, service, product, or topic "
        "is affected or being asked about?"
    ),
    'prob_statement': (
        "What is the main problem, issue, symptom, error, or request "
        "being described?"
    ),
    'cause': "What caused this problem or issue?",
}

TAG_COLS = [f'tag_{i}' for i in range(1,9)]

BUCKETS = {
    'security-issue':['security','breach','unauthorized','login','password'],
    'outage':['outage','offline','down','unavailable','disruption'],
    'network-issue':['network','connectivity','wifi','internet','connection'],
    'performance':['performance','slow','latency','speed','lag'],
    'bug-fix':['bug','error','defect','crash','fail'],
    'incident-reporting':['incident','issue','problem','recovery','critical'],
    'fraud':['fraud','scam','suspicious'],
    'billing':['billing','invoice','payment','refund','subscription'],
    'documentation':['documentation','guide','manual','faq'],
    'feedback':['feedback','suggestion','improvement','feature'],
}
PRIORITY_ORDER = ['security-issue','outage','network-issue','performance',
                  'bug-fix','fraud','incident-reporting','billing',
                  'documentation','feedback']

def get_tags(row):
    return ", ".join([
        str(row[c]).strip().lower() for c in TAG_COLS
        if c in row.index and pd.notnull(row[c])
        and str(row[c]).strip().lower() not in ('','nan')
    ])

def categorize(tag_str):
    if not tag_str: return 'general-support'
    for b in PRIORITY_ORDER:
        if any(k in tag_str for k in BUCKETS[b]): return b
    return 'general-support'

def build_context(subject, body, tag_str, bucket):
    body = str(body or '').replace('\\n','\n').replace('\\t',' ')
    body = re.sub(r'\s+', ' ', body).strip()
    subj = str(subject or '').strip()
    subj = '' if subj.lower() in ('nan','none','') else subj
    parts = []
    if subj: parts.append(f"Subject: {subj}.")
    if tag_str: parts.append(f"Tags: {tag_str}.")
    if bucket: parts.append(f"Category: {bucket}.")
    return (' '.join(parts) + ' ' + body)[:BODY_LEN]

def find_start(context, span):
    if not span or span == 'none': return -1
    idx = context.lower().find(span.lower())
    if idx >= 0: return idx
    anchor = ' '.join(span.split()[:4])
    return context.lower().find(anchor.lower())

def make_qa(ticket_id, aspect, context, span):
    impossible = (not span or span == 'none')
    if impossible:
        return {"id":f"t{ticket_id}_{aspect}", "question":QUESTIONS[aspect],
                "answers":[], "is_impossible":True}
    start = find_start(context, span)
    if start < 0:
        return {"id":f"t{ticket_id}_{aspect}", "question":QUESTIONS[aspect],
                "answers":[], "is_impossible":True}
    return {"id":f"t{ticket_id}_{aspect}", "question":QUESTIONS[aspect],
            "answers":[{"text":span, "answer_start":start}],
            "is_impossible":False}

def to_squad(rows):
    data, skipped = [], 0
    for r in rows:
        subj = str(r.get('subject','') or '')
        body = str(r.get('body','') or '')
        tags = get_tags(pd.Series(r))
        ctx  = build_context(subj, body, tags, categorize(tags))
        if not ctx.strip(): skipped += 1; continue

        ps  = str(r.get('prob_sub_llm','none') or 'none')
        pst = str(r.get('prob_statement_llm','none') or 'none')
        cau = str(r.get('cause_llm','none') or 'none')
        if ps == 'none': skipped += 1; continue

        data.append({
            "title": f"ticket_{r['_idx']}",
            "paragraphs": [{"context": ctx, "qas": [
                make_qa(r['_idx'], 'prob_sub',       ctx, ps),
                make_qa(r['_idx'], 'prob_statement',  ctx, pst),
                make_qa(r['_idx'], 'cause',           ctx, cau),
            ]}]
        })
    print(f"  → {len(data)} examples built, {skipped} skipped")
    return {"version":"v2.0", "data":data}

def main():
    df = pd.read_csv(INPUT_CSV)
    df['_idx'] = df.index
    valid = df[df['label_valid']==True].copy()
    print(f"Valid labelled: {len(valid)}/{len(df)}\n")

    random.seed(SEED)
    en = valid[valid['language']=='en'].to_dict('records')
    de = valid[valid['language']=='de'].to_dict('records')
    random.shuffle(en); random.shuffle(de)

    def split(lst, r1=0.60, r2=0.20):
        n=len(lst); a=int(n*r1); b=int(n*r2)
        return lst[:a], lst[a:a+b], lst[a+b:]

    en_tr,en_va,en_te = split(en)
    de_tr,de_va,de_te = split(de)
    train = en_tr+de_tr; random.shuffle(train)
    val   = en_va+de_va
    test  = en_te+de_te

    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")
    print(f"  EN: {len(en_tr)}/{len(en_va)}/{len(en_te)}")
    print(f"  DE: {len(de_tr)}/{len(de_va)}/{len(de_te)}\n")

    for path, rows, name in [(TRAIN_JSON,train,'Train'),
                              (VAL_JSON,val,'Val'),
                              (TEST_JSON,test,'Test')]:
        print(f"Building {name}...")
        sq = to_squad(rows)
        with open(path,'w',encoding='utf-8') as f:
            json.dump(sq, f, ensure_ascii=False, indent=2)
        n_ans = sum(1 for d in sq['data']
                    for p in d['paragraphs']
                    for qa in p['qas'] if not qa['is_impossible'])
        print(f"  → {path} | {len(sq['data'])} tickets | "
              f"{n_ans} answerable QA pairs\n")

    # Sanity check: verify answer_start offsets
    print("Sanity check (5 QA pairs from train):")
    with open(TRAIN_JSON) as f:
        td = json.load(f)
    checked = 0
    for item in td['data']:
        ctx = item['paragraphs'][0]['context']
        for qa in item['paragraphs'][0]['qas']:
            if not qa['is_impossible']:
                a   = qa['answers'][0]
                got = ctx[a['answer_start']:a['answer_start']+len(a['text'])]
                ok  = got.lower() == a['text'].lower()
                print(f"  {'OK' if ok else 'FAIL':4s} | {qa['id']:35s} "
                      f"| '{a['text'][:40]}'")
                checked += 1
                if checked >= 5: break
        if checked >= 5: break

if __name__ == '__main__':
    main()
