"""
aspect_evaluator.py — Aspect QA model evaluation — zero-shot (v3) vs fine-tuned (v4)
Updated for 6-aspect design: prob_sub / prob_statement / cause

RUN: Kaggle GPU (after step3)
  Upload: squad_test.json, finetuned_qa/, this file

OUTPUT: evaluation_results.json
        v3_vs_v4_comparison.csv
"""
import json, re, csv
from collections import Counter
from transformers import pipeline

BASE_MODEL = "deepset/xlm-roberta-base-squad2"
FT_MODEL   = "finetuned_qa"
TEST_JSON  = "squad_test.json"

QUESTIONS = {
    'prob_sub': (
        "What specific system, component, service, product, or topic "
        "is affected or being asked about?"),
    'prob_statement': (
        "What is the main problem, issue, symptom, error, or request "
        "being described?"),
    'cause': "What caused this problem or issue?",
}
THRESHOLDS = {'prob_sub':0.05, 'prob_statement':0.05, 'cause':0.12}

def normalise(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9äöüß\s]"," ", text)
    return text.split()

def f1(pred, gold):
    if gold == 'none' or not gold:
        return 1.0 if (pred == 'none' or not pred) else 0.0
    if pred == 'none' or not pred: return 0.0
    pc = Counter(normalise(pred)); gc = Counter(normalise(gold))
    common = sum((pc & gc).values())
    if common == 0: return 0.0
    prec = common/sum(pc.values()); rec = common/sum(gc.values())
    return 2*prec*rec/(prec+rec)

def load_test(path):
    with open(path) as f: raw = json.load(f)
    records = []
    for item in raw['data']:
        for para in item['paragraphs']:
            ctx = para['context']
            entry = {'context':ctx, 'id':item['title']}
            for qa in para['qas']:
                asp = qa['id'].split('_')[-1]
                if asp not in QUESTIONS: asp = qa['id'].split('_',1)[-1]
                gold = 'none' if qa['is_impossible'] else qa['answers'][0]['text']
                entry[f'{asp}_gold'] = gold
            records.append(entry)
    return records

def run_model(pipe, records):
    preds = []
    for r in records:
        pred = {}
        for asp, q in QUESTIONS.items():
            out = pipe(question=q, context=r['context'],
                       handle_impossible_answer=True)
            span = out['answer'].strip() if out['answer'] else ''
            pred[asp] = span if out['score'] >= THRESHOLDS[asp] else 'none'
        preds.append(pred)
    return preds

def score(records, preds):
    scores = {asp:[] for asp in QUESTIONS}
    for r, p in zip(records, preds):
        for asp in QUESTIONS:
            gold = r.get(f'{asp}_gold','none')
            scores[asp].append(f1(p.get(asp,'none'), gold))
    return {asp: round(sum(v)/len(v),4) for asp,v in scores.items()}

def main():
    records = load_test(TEST_JSON)
    print(f"Test tickets: {len(records)}\n")

    print("Running v3 zero-shot...")
    v3 = pipeline("question-answering", model=BASE_MODEL,
                  handle_impossible_answer=True)
    v3_preds  = run_model(v3, records)
    v3_scores = score(records, v3_preds)

    print("Running v4 fine-tuned...")
    v4 = pipeline("question-answering", model=FT_MODEL,
                  handle_impossible_answer=True)
    v4_preds  = run_model(v4, records)
    v4_scores = score(records, v4_preds)

    print(f"\n{'='*58}")
    print(f"  RESULTS — {len(records)} TEST TICKETS")
    print(f"{'='*58}")
    print(f"  {'Aspect':15s}  {'v3 zero-shot':>12s}  "
          f"{'v4 fine-tuned':>13s}  {'Delta':>8s}")
    print(f"  {'-'*54}")
    for asp in QUESTIONS:
        v3s = v3_scores[asp]; v4s = v4_scores[asp]
        d = v4s-v3s
        ar = '↑' if d>0.01 else ('↓' if d<-0.01 else '→')
        print(f"  {asp:15s}  {v3s:>12.3f}  {v4s:>13.3f}  "
              f"{ar} {d:+.3f}")
    ov3 = sum(v3_scores.values())/3
    ov4 = sum(v4_scores.values())/3
    print(f"  {'-'*54}")
    print(f"  {'OVERALL':15s}  {ov3:>12.3f}  {ov4:>13.3f}  "
          f"{'↑' if ov4>ov3 else '↓'} {ov4-ov3:+.3f}")
    print(f"{'='*58}")

    out = {'v3_zero_shot':v3_scores, 'v4_finetuned':v4_scores,
           'delta':{a:round(v4_scores[a]-v3_scores[a],4) for a in QUESTIONS},
           'n_test':len(records)}
    with open("evaluation_results.json","w") as f:
        json.dump(out, f, indent=2)

    with open("v3_vs_v4_comparison.csv","w",newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ticket_id','aspect','gold','v3_pred','v4_pred',
                    'v3_f1','v4_f1'])
        for r, v3p, v4p in zip(records, v3_preds, v4_preds):
            for asp in QUESTIONS:
                gold = r.get(f'{asp}_gold','none')
                w.writerow([r['id'], asp, gold,
                            v3p.get(asp,'none'), v4p.get(asp,'none'),
                            round(f1(v3p.get(asp,'none'),gold),3),
                            round(f1(v4p.get(asp,'none'),gold),3)])
    print("\nSaved → evaluation_results.json")
    print("Saved → v3_vs_v4_comparison.csv")

if __name__ == '__main__':
    main()
