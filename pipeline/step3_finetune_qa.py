"""
step3_finetune_qa.py — Fine-tune xlm-roberta-base-squad2 on domain tickets
Updated for 6-aspect design: prob_sub / prob_statement / cause

RUN: Kaggle GPU T4 x2 (~20 min)
  Upload: squad_train.json, squad_val.json, this file
  Accelerator: GPU T4 x2

INPUT:  squad_train.json (300 tickets × 3 QA pairs = 900 pairs)
        squad_val.json   (100 tickets × 3 = 300 pairs)
OUTPUT: finetuned_qa/    (model + tokenizer, ~450MB)
        training_log.json
"""
import json, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
                          TrainingArguments, Trainer, default_data_collator)

BASE_MODEL = "deepset/xlm-roberta-base-squad2"
TRAIN_JSON = "squad_train.json"
VAL_JSON   = "squad_val.json"
OUTPUT_DIR = "finetuned_qa"
MAX_LEN    = 384
STRIDE     = 128

def load_squad(path):
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    examples = []
    for item in raw['data']:
        for para in item['paragraphs']:
            ctx = para['context']
            for qa in para['qas']:
                examples.append({
                    'id': qa['id'], 'context': ctx,
                    'question': qa['question'],
                    'answers': {
                        'text':         [a['text'] for a in qa['answers']],
                        'answer_start': [a['answer_start'] for a in qa['answers']],
                    },
                    'is_impossible': qa['is_impossible'],
                })
    return examples

def preprocess(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    tokenized = tokenizer(
        questions, examples["context"],
        max_length=MAX_LEN, truncation="only_second",
        stride=STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    offset_map = tokenized.pop("offset_mapping")
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    answers    = examples["answers"]
    impossible = examples["is_impossible"]

    starts, ends = [], []
    for i, offsets in enumerate(offset_map):
        idx     = sample_map[i]
        imp     = impossible[idx]
        ans     = answers[idx]
        ids     = tokenized["input_ids"][i]
        cls_idx = ids.index(tokenizer.cls_token_id)

        if imp or len(ans["answer_start"]) == 0:
            starts.append(cls_idx); ends.append(cls_idx); continue

        s_char = ans["answer_start"][0]
        e_char = s_char + len(ans["text"][0])
        seq    = tokenized.sequence_ids(i)
        c0     = seq.index(1)
        c1     = len(seq) - seq[::-1].index(1) - 1

        if offsets[c0][0] > e_char or offsets[c1][1] < s_char:
            starts.append(cls_idx); ends.append(cls_idx); continue

        ts = c0
        while ts <= c1 and offsets[ts][0] <= s_char: ts += 1
        starts.append(ts - 1)
        te = c1
        while te >= c0 and offsets[te][1] >= e_char: te -= 1
        ends.append(te + 1)

    tokenized["start_positions"] = starts
    tokenized["end_positions"]   = ends
    return tokenized

def main():
    print(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL)

    train_raw = load_squad(TRAIN_JSON)
    val_raw   = load_squad(VAL_JSON)
    print(f"Train QA pairs: {len(train_raw)} | Val: {len(val_raw)}")
    print(f"  Train answerable: {sum(1 for e in train_raw if not e['is_impossible'])}")

    fn = lambda x: preprocess(x, tokenizer)
    train_tok = Dataset.from_list(train_raw).map(fn, batched=True,
                    remove_columns=Dataset.from_list(train_raw).column_names)
    val_tok   = Dataset.from_list(val_raw).map(fn, batched=True,
                    remove_columns=Dataset.from_list(val_raw).column_names)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        tokenizer=tokenizer, data_collator=default_data_collator,
    )

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"\nTraining on: {gpu}  (~20 min on T4)\n")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved → {OUTPUT_DIR}/")

    log = trainer.state.log_history
    with open("training_log.json","w") as f: json.dump(log, f, indent=2)

    print("\nTraining summary:")
    print(f"  {'Epoch':>6}  {'Val Loss':>10}")
    for e in log:
        if 'eval_loss' in e:
            print(f"  {e.get('epoch','?'):>6.1f}  {e['eval_loss']:>10.4f}")

    print("\nAfter training, run:")
    print("  python aspect_pipeline_v4.py --input your_data.csv "
          "--model finetuned_qa/")

if __name__ == '__main__':
    main()
