"""
step2_sft.py — RTX 4090 / H100 / T4 optimised
═══════════════════════════════════════════════════════════════════════════════
Phase 2 — Supervised Fine-Tuning (SFT) Warmup

PURPOSE
───────
Fine-tune Qwen2.5-1.5B-Instruct on silver labels from step1.
Teaches the model the JSON output schema and domain vocabulary
before GRPO refinement in step3.

RUN (RTX 4090 — 5k pilot):
  python step2_sft.py \
      --silver /workspace/outputs/silver_labels_full.jsonl \
      --out    /workspace/checkpoints/sft_checkpoint/ \
      --n      5000 \
      --epochs 3

RUN (H100 — full 44k):
  python step2_sft.py \
      --silver /workspace/outputs/silver_labels_full.jsonl \
      --out    /workspace/checkpoints/sft_checkpoint/ \
      --epochs 2

RUN (push to HuggingFace after training):
  HF_TOKEN=hf_xxx python step2_sft.py \
      --silver /workspace/outputs/silver_labels_full.jsonl \
      --out    /workspace/checkpoints/sft_checkpoint/ \
      --hf_repo ranjan56cse/cs-support-labels
"""

import os
import json
import time
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    TrainerCallback, TrainerState, TrainerControl,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ─── CONFIG ───────────────────────────────────────────────────────────────────
STUDENT_MODEL = os.environ.get('STUDENT_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')
MAX_SEQ_LEN   = 1024

# LoRA — fixed for this task, do not change without good reason
LORA_CONFIG = dict(
    r              = 16,         # rank: sweet spot for structured extraction on 5k records
    lora_alpha     = 32,         # always 2*r — do not change independently
    lora_dropout   = 0.05,       # light regularisation for 5k records
    bias           = 'none',
    task_type      = TaskType.CAUSAL_LM,
    target_modules = ['q_proj','v_proj','k_proj','o_proj','gate_proj','up_proj'],
)

# ─── GPU PROFILE ──────────────────────────────────────────────────────────────
def get_gpu_profile():
    """
    Detect GPU and return optimal training config.

    RTX 4090 (24GB): BF16 full, batch=2, grad_accum=8 → eff_batch=16
    H100/A100 (80GB): BF16 full, batch=4, grad_accum=4 → eff_batch=16
    T4 (15GB): 4-bit NF4, batch=2, grad_accum=8 → eff_batch=16

    4-bit during training adds ~15-20% overhead vs BF16 full.
    Only use it when VRAM < 20GB.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            'No GPU detected.\n'
            'Vast.ai: verify GPU is attached to instance.\n'
            'Kaggle: Settings → Accelerator → GPU T4 x1 → Save.'
        )

    props    = torch.cuda.get_device_properties(0)
    vram     = props.total_memory / 1e9
    use_bf16 = torch.cuda.is_bf16_supported()

    print(f"  GPU  : {props.name}")
    print(f"  VRAM : {vram:.1f} GB")
    print(f"  BF16 : {'YES' if use_bf16 else 'NO'}")

    if vram >= 20:
        # RTX 4090 (24GB) / A100 / H100 — BF16 full, no quant
        batch  = 4 if vram >= 40 else 2
        gacc   = 4 if vram >= 40 else 8
        profile = {
            'quant':      None,
            'dtype':      torch.bfloat16 if use_bf16 else torch.float16,
            'batch':      batch,
            'grad_accum': gacc,
            'bf16':       use_bf16,
            'fp16':       not use_bf16,
        }
        print(f"  Mode : BF16 full precision (no quantisation)")
        print(f"  Batch: {batch} × grad_accum {gacc} = effective {batch*gacc}")
    else:
        # T4 (15GB) — must use 4-bit
        profile = {
            'quant': BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_quant_type       = 'nf4',
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype    = torch.bfloat16 if use_bf16 else torch.float16,
            ),
            'dtype':      torch.bfloat16 if use_bf16 else torch.float16,
            'batch':      2,
            'grad_accum': 8,
            'bf16':       use_bf16,
            'fp16':       not use_bf16,
        }
        print(f"  Mode : 4-bit NF4 (VRAM < 20GB)")
        print(f"  Batch: 2 × grad_accum 8 = effective 16")

    return profile


# ─── PROMPTS ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert support ticket analyst. Your job is to extract "
    "structured information from customer support tickets with high precision. "
    "You MUST respond with a single valid JSON object and nothing else — "
    "no markdown fences, no explanation, no preamble."
)

def build_input_prompt(subject: str, body: str, lang: str) -> str:
    s = subject if subject and subject.lower() not in ('nan','none','') else '(none)'
    return (
        f"Extract 5 fields from this support ticket as a JSON object with keys: "
        f"prob_sub, prob_statement, cause, intent, ticket_type_nli.\n\n"
        f"intent: performance_issue|login_or_access|billing_issue|"
        f"feature_request|service_outage|general_inquiry|data_or_integration\n"
        f"ticket_type_nli: complaint|escalation|feedback|product_support\n\n"
        f"Language: {lang}\nSubject: {s}\nBody: {body[:600]}\n\nJSON:"
    )

def build_target_output(record: dict) -> str:
    return json.dumps({
        'prob_sub':        record.get('prob_sub',        ''),
        'prob_statement':  record.get('prob_statement',  ''),
        'cause':           record.get('cause',           ''),
        'intent':          record.get('intent',          'general_inquiry'),
        'ticket_type_nli': record.get('ticket_type_nli', 'product_support'),
    }, ensure_ascii=False)


# ─── DATASET ──────────────────────────────────────────────────────────────────
class SilverDataset(Dataset):
    """
    Converts silver_labels.jsonl into (input_ids, labels) pairs.
    Labels are masked on the prompt tokens — model only trains on JSON response.
    10% held out as validation split to monitor overfitting.
    """

    def __init__(self, jsonl_path: str, tokenizer,
                 max_len: int = MAX_SEQ_LEN,
                 n: int = None,
                 split: str = 'train'):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        records        = []

        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # skip corrupted lines

        # Subsample before split so both train/val come from same pool
        if n and n < len(records):
            random.seed(42)
            records = random.sample(records, n)

        # 90/10 train/val split with fixed seed
        random.seed(42)
        random.shuffle(records)
        split_idx = int(len(records) * 0.9)

        if split == 'train':
            self.records = records[:split_idx]
        else:
            self.records = records[split_idx:]

        print(f"  Dataset [{split}]: {len(self.records):,} records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        subj = str(rec.get('subject', '') or '')
        body = str(rec.get('body',    '') or '')
        lang = str(rec.get('language','en') or 'en')

        messages_full = [
            {'role': 'system',    'content': SYSTEM_PROMPT},
            {'role': 'user',      'content': build_input_prompt(subj, body, lang)},
            {'role': 'assistant', 'content': build_target_output(rec)},
        ]
        messages_prompt = messages_full[:-1]

        full_text   = self.tokenizer.apply_chat_template(
            messages_full,   tokenize=False, add_generation_prompt=False)
        prompt_text = self.tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True)

        full_ids   = self.tokenizer(
            full_text,   return_tensors='pt',
            truncation=True, max_length=self.max_len)['input_ids'][0]
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors='pt',
            truncation=True, max_length=self.max_len)['input_ids'][0]

        labels = full_ids.clone()
        labels[:len(prompt_ids)] = -100   # mask prompt — train on JSON answer only

        return {
            'input_ids':      full_ids,
            'labels':         labels,
            'attention_mask': torch.ones_like(full_ids),
        }


# ─── PROGRESS CALLBACK ────────────────────────────────────────────────────────
class ProgressCallback(TrainerCallback):
    """Richer progress: VRAM usage printed every logging_steps."""

    def on_log(self, args, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        mem   = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        # Safe format: loss may be float or not present
        loss  = logs.get('loss')
        eloss = logs.get('eval_loss')
        lr    = logs.get('learning_rate')
        step  = state.global_step
        total = state.max_steps

        parts = [f"  Step {step:>5}/{total}"]
        if loss  is not None: parts.append(f"loss={float(loss):.4f}")
        if eloss is not None: parts.append(f"eval_loss={float(eloss):.4f}")
        if lr    is not None: parts.append(f"lr={float(lr):.2e}")
        parts.append(f"GPU {mem:.1f}GB")
        print("  ".join(parts))

    def on_epoch_end(self, args, state: TrainerState,
                     control: TrainerControl, **kwargs):
        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"\n  ── Epoch {state.epoch:.0f} complete  "
              f"step={state.global_step}  GPU {mem:.1f}GB ──\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run(silver_path: str, out_dir: str,
        epochs: int = 3, lr: float = 2e-4,
        n: int = None,
        hf_repo: str = None,
        hf_token: str = None):

    t0 = time.time()

    print("=" * 60)
    print("STEP 2 — SFT Warmup")
    print("=" * 60)

    # ── GPU profile ───────────────────────────────────────────────────────────
    profile = get_gpu_profile()

    # ── Verify input ──────────────────────────────────────────────────────────
    if not os.path.exists(silver_path):
        raise FileNotFoundError(
            f"Silver labels not found: {silver_path}\n"
            f"Run step1_silver_labels.py first."
        )

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Student    : {STUDENT_MODEL}")
    print(f"  Silver     : {silver_path}")
    print(f"  Output     : {out_dir}")
    print(f"  Epochs     : {epochs}  LR: {lr}")
    if n:
        print(f"  Subsample  : {n:,} records (90% train / 10% val)")
    if hf_repo:
        print(f"  HF repo    : {hf_repo}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'   # right-padding required for causal LM training

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n  Loading model...")
    load_kwargs = dict(
        device_map        = 'auto',
        trust_remote_code = True,
        torch_dtype       = profile['dtype'],   # correct kwarg for from_pretrained
    )
    if profile['quant'] is not None:
        load_kwargs['quantization_config'] = profile['quant']

    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, **load_kwargs)

    # Disable KV cache — incompatible with gradient checkpointing
    # Without this the Trainer prints a warning and disables it anyway
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    # Order matters:
    #   1. gradient_checkpointing_enable()
    #   2. prepare_model_for_kbit_training() — only for 4-bit
    #   3. get_peft_model()
    model.gradient_checkpointing_enable()
    if profile['quant'] is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()

    used = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after model load: {used:.1f} GB")

    # ── Dataset — 90/10 train/val split ───────────────────────────────────────
    train_dataset = SilverDataset(silver_path, tok, n=n, split='train')
    val_dataset   = SilverDataset(silver_path, tok, n=n, split='val')

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = out_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = profile['batch'],
        gradient_accumulation_steps = profile['grad_accum'],
        learning_rate               = lr,
        lr_scheduler_type           = 'cosine',
        warmup_ratio                = 0.05,
        weight_decay                = 0.01,
        bf16                        = profile['bf16'],
        fp16                        = profile['fp16'],
        logging_steps               = 10,
        eval_strategy               = 'epoch',   # eval after each epoch
        save_strategy               = 'no',      # save once manually at end
        dataloader_num_workers      = 0,         # 0 avoids fork issues in Docker
        report_to                   = 'none',    # disable wandb/tensorboard
        remove_unused_columns       = False,
        ddp_find_unused_parameters  = False,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        callbacks     = [ProgressCallback()],
        data_collator = DataCollatorForSeq2Seq(
            tok, model=model,
            padding            = True,
            pad_to_multiple_of = 8,   # tensor core alignment
        ),
    )

    print(f"\n  Starting SFT training...")
    print(f"  Effective batch: {profile['batch']} × {profile['grad_accum']}"
          f" = {profile['batch']*profile['grad_accum']}")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE — {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"  Checkpoint saved: {out_dir}")
    print(f"  Files: {os.listdir(out_dir)}")
    print(f"{'='*60}")

    # ── Push to HuggingFace Hub ───────────────────────────────────────────────
    if hf_repo:
        _push_checkpoint_to_hf(out_dir, hf_repo, hf_token)

    return out_dir


def _push_checkpoint_to_hf(adapter_dir: str, repo_id: str,
                             token: str = None) -> None:
    """Push LoRA adapter to HuggingFace Hub after training."""
    try:
        from hf_hub import push_adapter, verify_token
        verify_token(token)
        push_adapter(adapter_dir, repo_id, prefix='sft_checkpoint', token=token)
        print(f"\n  SFT checkpoint → https://huggingface.co/datasets/{repo_id}")
        print(f"  Pull later: from hf_hub import pull_adapter")
        print(f"              pull_adapter('{repo_id}', 'sft_checkpoint')")
    except ImportError:
        print("\n  WARNING: hf_hub.py not found — skipping HF push")
        print("  Copy hf_hub.py to the same directory as step2_sft.py")
    except Exception as e:
        print(f"\n  WARNING: HF push failed — {e}")
        print(f"  Checkpoint is safe locally at: {adapter_dir}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 2 — SFT warmup (RTX 4090 / H100 / T4 optimised)')
    parser.add_argument('--silver',   required=True,
                        help='silver_labels.jsonl from step1')
    parser.add_argument('--out',      default='/workspace/checkpoints/sft_checkpoint/',
                        help='Output directory for LoRA adapter')
    parser.add_argument('--epochs',   type=int,   default=3,
                        help='Training epochs (default 3; use 2 for H100 full run)')
    parser.add_argument('--lr',       type=float, default=2e-4,
                        help='Learning rate (default 2e-4)')
    parser.add_argument('--n',        type=int,   default=None,
                        help='Subsample N records (default: all). Use 5000 for pilot.')
    parser.add_argument('--model',    default=None,
                        help='Override student model HF name')
    parser.add_argument('--hf_repo',  default=None,
                        help='HF repo to push checkpoint after training '
                             '(e.g. ranjan56cse/cs-support-labels)')
    parser.add_argument('--hf_token', default=None,
                        help='HF write token (default: HF_TOKEN env var)')
    args = parser.parse_args()

    if args.model:
        STUDENT_MODEL = args.model

    run(
        silver_path = args.silver,
        out_dir     = args.out,
        epochs      = args.epochs,
        lr          = args.lr,
        n           = args.n,
        hf_repo     = args.hf_repo,
        hf_token    = args.hf_token,
    )
