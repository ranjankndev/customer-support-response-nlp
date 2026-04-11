"""
step1_silver_labels.py  — H100 / A100 optimised
Phase 1 — Silver Label Generation via LLM-as-Judge

H100 / A100 run command (Vast.ai):
  pip install -q transformers accelerate flash-attn optimum
  python step1_silver_labels.py \
      --input  /workspace/processed_full.csv \
      --out    /workspace/silver_labels_full.jsonl \
      --batch  32

Kaggle T4 run command:
  pip install -q transformers peft bitsandbytes accelerate
  python step1_silver_labels.py \
      --input /kaggle/input/cs-support-ds/processed_full.csv \
      --out   /kaggle/working/silver_labels.jsonl \
      --batch 4

Split across 2 GPUs (2x RTX 4090 or 2x H100):
  CUDA_VISIBLE_DEVICES=0 python step1_silver_labels.py --input data.csv --split 0 2 --out out_gpu0.jsonl
  CUDA_VISIBLE_DEVICES=1 python step1_silver_labels.py --input data.csv --split 1 2 --out out_gpu1.jsonl
  cat out_gpu0.jsonl out_gpu1.jsonl > silver_labels_full.jsonl
"""

import os, re, json, time, argparse, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch

# ─── ENVIRONMENT DETECTION ────────────────────────────────────────────────────
IS_KAGGLE   = os.path.isdir('/kaggle/working')
IS_VAST     = os.path.isdir('/workspace')
WORKING_DIR = '/kaggle/working' if IS_KAGGLE else ('/workspace' if IS_VAST else '.')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
JUDGE_MODEL    = os.environ.get('JUDGE_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
MAX_NEW_TOKENS = 180    # JSON output for 5 fields = ~80-120 tokens; 180 is safe ceiling
                        # Was 400 — wasted ~33% throughput on padding tokens
TEMPERATURE    = 0.0    # 0.0 = fully greedy, matches do_sample=False in generate()
                        # TOP_P removed — irrelevant when do_sample=False
BATCH_SIZE     = 4      # auto-set from GPU profile in get_gpu_profile()
                        # user can override with --batch

# Auto batch size by VRAM — overridden by --batch if passed
_AUTO_BATCH = {
    80: 32,   # H100 / A100 80GB
    40: 16,   # A100 40GB
    24: 8,    # RTX 4090
    20: 8,    # RTX 3090
    15: 4,    # T4
}

INTENT_LABELS      = [
    'performance_issue','login_or_access','billing_issue',
    'feature_request','service_outage','general_inquiry','data_or_integration',
]
TICKET_TYPE_LABELS = ['escalation','complaint','feedback','product_support']

# ─── GPU DETECTION AND MODEL LOADING STRATEGY ─────────────────────────────────
def get_gpu_profile():
    """
    Detect GPU and return the optimal loading strategy.

    Key insight:
      T4  (15GB):  MUST use 4-bit quant — 7B BF16 = 14GB, leaves 1GB for KV cache
      A100 (40GB): SHOULD use BF16 full — 7B BF16 = 14GB, 26GB free for batch KV
      A100 (80GB): MUST  use BF16 full — 7B BF16 = 14GB, 66GB free
      H100 (80GB): MUST  use BF16 full + FA2 + compile
      RTX 4090 (24GB): BF16 is tight — use BF16 or 8-bit

    4-bit quantisation on H100/A100 is a net LOSS:
      - Saves VRAM we don't need
      - Adds dequantisation overhead on EVERY token generated
      - Disables torch.compile (incompatible with bitsandbytes on sm_90)
      - Prevents Flash Attention 2 integration
    """
    if not torch.cuda.is_available():
        return {'quant': None, 'compile': False, 'fa2': False,
                'dtype': torch.float32, 'vram': 0, 'name': 'CPU'}

    props     = torch.cuda.get_device_properties(0)
    vram_gb   = props.total_memory / 1e9
    name      = props.name
    sm        = props.major * 10 + props.minor  # compute capability e.g. 90 for H100

    print(f"  GPU  : {name}")
    print(f"  VRAM : {vram_gb:.1f} GB")
    print(f"  SM   : {sm} (compute capability {props.major}.{props.minor})")

    # 7B model in BF16 = ~14GB. Need headroom for KV cache and batch activations.
    use_bf16  = torch.cuda.is_bf16_supported()
    use_fa2   = sm >= 80          # FlashAttention-2: Ampere (A100) + Hopper (H100)
    use_compile = sm >= 80        # torch.compile works well from Ampere onwards

    if vram_gb >= 40:
        # A100 40GB / 80GB / H100 — load in BF16, no quantisation
        # Auto batch: pick from _AUTO_BATCH table
        auto_batch = next((v for k, v in sorted(_AUTO_BATCH.items(), reverse=True)
                           if vram_gb >= k), 4)
        profile = {
            'quant':      None,
            'compile':    use_compile,
            'fa2':        use_fa2,
            'dtype':      torch.bfloat16 if use_bf16 else torch.float16,
            'vram':       vram_gb,
            'name':       name,
            'auto_batch': auto_batch,
        }
        print(f"  Mode       : BF16 full precision (no quantisation)")
        print(f"  FA2        : {'YES' if use_fa2 else 'NO'}")
        print(f"  Compile    : {'YES — max-autotune (~3-5 min warmup)' if use_compile else 'NO'}")
        print(f"  Auto batch : {auto_batch}")

    elif vram_gb >= 20:
        # RTX 4090 24GB — BF16 is tight but works (14GB model + ~6GB KV batch)
        auto_batch = next((v for k, v in sorted(_AUTO_BATCH.items(), reverse=True)
                           if vram_gb >= k), 4)
        profile = {
            'quant':      None,
            'compile':    False,   # compile unstable with tight VRAM
            'fa2':        use_fa2,
            'dtype':      torch.bfloat16 if use_bf16 else torch.float16,
            'vram':       vram_gb,
            'name':       name,
            'auto_batch': auto_batch,
        }
        print(f"  Mode       : BF16 full precision (24GB — tight but works)")
        print(f"  FA2        : {'YES' if use_fa2 else 'NO'}")
        print(f"  Auto batch : {auto_batch}")

    else:
        # T4 15GB / smaller — MUST use 4-bit
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = 'nf4',
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype    = torch.bfloat16 if use_bf16 else torch.float16,
        )
        profile = {
            'quant':      quant,
            'compile':    False,
            'fa2':        False,
            'dtype':      torch.bfloat16 if use_bf16 else torch.float16,
            'vram':       vram_gb,
            'name':       name,
            'auto_batch': 4,
        }
        print(f"  Mode       : 4-bit NF4 (VRAM < 20GB — quantisation required)")
        print(f"  Auto batch : 4")

    if vram_gb < 8:
        raise RuntimeError(f"Only {vram_gb:.1f}GB VRAM — minimum 8GB needed for 7B 4-bit.")

    return profile

# ─── PROMPT ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert support ticket analyst. Extract structured information "
    "from support tickets. Respond with a single valid JSON object only — "
    "no markdown fences, no explanation, no preamble."
)

def build_judge_prompt(subject, body, lang):
    s = subject if subject and subject.lower() not in ('nan','none','') else '(none)'
    return f"""Extract 5 fields from this support ticket. Respond ONLY with a JSON object matching this exact schema — no extra keys, no markdown:

{{
  "prob_sub": "<The specific system, product, tool, or service this ticket is about. Always check the Subject first — if it names a system or service, use it (e.g. Subject 'Anfrage zu JIRA-Tools' → 'JIRA tools', Subject 'Datensicherheit im medizinischen Bereich' → 'medical data security systems', Subject 'Verbesserung der IFTTT-Integration' → 'IFTTT integration'). If subject is missing or generic, extract from the body. Empty string only if no specific system or service can be identified anywhere.>",
  "prob_statement": "<What exactly went wrong or is being requested, in one sentence. For problems: the symptom. For inquiries: the request. Never leave empty.>",
  "cause": "<Root cause if stated OR implied. Include indirect causes: 'through vulnerabilities in X', 'due to outdated software', 'manual workarounds are error-prone', 'strategy no longer meets needs', 'rising cyberattacks'. Empty string only if the ticket gives absolutely no reason or cause for the situation.>",
  "intent": "<exactly one of: performance_issue | login_or_access | billing_issue | feature_request | service_outage | general_inquiry | data_or_integration>",
  "ticket_type_nli": "<exactly one of: complaint | escalation | feedback | product_support>"
}}

INTENT:
  performance_issue   — system works but slow, laggy, crashing, or producing errors
  login_or_access     — cannot log in, account locked, password reset, permissions
  billing_issue       — wrong charges, unexpected invoice, payment failure, refund
  feature_request     — asking for new functionality or improvement to be built
  service_outage      — system completely unavailable, down, offline
  general_inquiry     — asking for information, pricing, guidance, capabilities, how-to
  data_or_integration — data sync failure, API error, integration issue, data loss

TICKET_TYPE:
  complaint       — reporting a negative experience or unresolved failure
  escalation      — urgent, high-impact, or previously unresolved issue
  feedback        — suggestion, improvement idea, or feature proposal
  product_support — question or request for help, guidance, or information

Ticket language: {lang}
Subject: {s}
Body: {body[:700]}

JSON:"""

# ─── JSON PARSER ──────────────────────────────────────────────────────────────
_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)

def parse_judge_output(raw):
    raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
    m   = _JSON_RE.search(raw)
    if not m:
        return {}
    s = re.sub(r',\s*([}\]])', r'\1', m.group(0)).replace("'", '"')
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        obj = {}
        for field in ('prob_sub','prob_statement','cause','intent','ticket_type_nli'):
            m2 = re.search(rf'"{field}"\s*:\s*"([^"]*)"', s)
            if m2:
                obj[field] = m2.group(1).strip()
    if obj.get('intent') not in INTENT_LABELS:
        ri = str(obj.get('intent','')).lower().replace(' ','_')
        obj['intent'] = next((l for l in INTENT_LABELS if l in ri or ri in l),
                             'general_inquiry')
    if obj.get('ticket_type_nli') not in TICKET_TYPE_LABELS:
        rt = str(obj.get('ticket_type_nli','')).lower()
        obj['ticket_type_nli'] = next((l for l in TICKET_TYPE_LABELS if l in rt or rt in l),
                                      'product_support')
    return obj

# ─── MODEL LOADER ─────────────────────────────────────────────────────────────
def load_judge(model_name: str, profile: dict):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n  Loading judge: {model_name}")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = 'left'   # required for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build kwargs for from_pretrained based on GPU profile
    load_kwargs = dict(
        trust_remote_code = True,
        torch_dtype       = profile['dtype'],
        device_map        = 'auto',
    )

    # Flash Attention 2 — verify it actually imports before enabling
    # Setting attn_implementation='flash_attention_2' without a working
    # flash_attn install causes a hard crash inside from_pretrained
    if profile['fa2'] and profile['quant'] is None:
        try:
            import flash_attn  # test import — will raise if broken
            load_kwargs['attn_implementation'] = 'flash_attention_2'
            print(f"  FlashAttention-2: enabled (v{flash_attn.__version__})")
        except (ImportError, Exception) as e:
            print(f"  FlashAttention-2: disabled ({e.__class__.__name__}) — using standard attention")
            # Do NOT set attn_implementation — let transformers use default

    # Quantisation — only for small VRAM GPUs
    if profile['quant'] is not None:
        load_kwargs['quantization_config'] = profile['quant']

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # torch.compile — H100/A100 only, disabled for 4-bit (incompatible)
    if profile['compile'] and profile['quant'] is None:
        print(f"\n  torch.compile: starting (first batch takes 3-5 min on H100 — do not interrupt)...")
        # max-autotune handles variable batch shapes (last batch < BATCH_SIZE)
        # reduce-overhead would recompile on every shape change — wrong for this use case
        model = torch.compile(model, mode='max-autotune')
        print(f"  torch.compile: ready")

    used = torch.cuda.memory_allocated() / 1e9
    total = profile['vram']
    print(f"  VRAM used: {used:.1f} / {total:.1f} GB  ({used/total*100:.0f}%)")
    return tok, model

# ─── BATCHED INFERENCE ────────────────────────────────────────────────────────
def run_judge_batch(tok, model, prompts: list) -> list:
    """
    Run inference on a batch of prompts simultaneously.

    H100 optimisations applied here:
    1. Greedy decoding (do_sample=False) — removes multinomial sampling overhead.
       At temperature=0.1 sampling is near-deterministic anyway. Greedy is
       both faster and fully deterministic.
    2. pad_to_multiple_of=64 — aligns tensor shapes to 64-byte boundaries,
       matching H100 tensor core requirements for maximum throughput.
    3. Prompt lengths tracked per-sequence so decode slicing is correct
       even with variable-length left-padded inputs.
    """
    texts = []
    for prompt in prompts:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': prompt},
        ]
        texts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    # Tokenize with left-padding + 64-byte alignment for tensor core efficiency
    inputs = tok(
        texts,
        return_tensors      = 'pt',
        truncation          = True,
        max_length          = 1024,
        padding             = True,
        pad_to_multiple_of  = 64,   # H100 tensor core alignment
    ).to(model.device)

    # Record per-sequence prompt lengths BEFORE padding alters shape
    # With left-padding, all sequences have the same padded input length.
    # We need the actual (non-padding) token count to slice new tokens correctly.
    attention_mask  = inputs['attention_mask']
    prompt_lengths  = attention_mask.sum(dim=1).tolist()  # real tokens per sequence

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens    = MAX_NEW_TOKENS,
            do_sample         = False,       # greedy — faster + deterministic for JSON
            use_cache         = True,        # KV cache — critical for autoregressive speed
            eos_token_id      = tok.eos_token_id,   # STOP at first EOS — don't pad to max_new_tokens
            pad_token_id      = tok.eos_token_id,
            repetition_penalty= 1.1,         # prevents JSON repetition loops at greedy temp
        )

    # Decode only newly generated tokens per sequence.
    # With left-padding: padded_len is uniform across the batch.
    # We slice from padded_len onward; skip_special_tokens strips EOS + any
    # trailing pad tokens cleanly.
    padded_len = inputs['input_ids'].shape[1]
    results = []
    for seq in out:
        new_tokens = seq[padded_len:]
        results.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
    return results

# ─── SAFE JSONL READER ────────────────────────────────────────────────────────
def _read_jsonl(path):
    records, bad = [], 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  WARNING: {bad} corrupted line(s) skipped in {path}")
    return pd.DataFrame(records)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run(input_path, n=None, out_path=None, csv_out_path=None, split_index=0, split_total=1):
    if out_path is None:
        out_path = os.path.join(WORKING_DIR, 'silver_labels.jsonl')
    if csv_out_path is None:
        # Default: create a sidecar CSV next to the JSONL.
        base = re.sub(r'\.jsonl$', '', out_path, flags=re.IGNORECASE)
        csv_out_path = base + "_aspects.csv"

    print("=" * 60)
    print("STEP 1 — Silver Label Generation")
    if split_total > 1:
        print(f"SPLIT {split_index+1} of {split_total} — GPU {split_index}")
    print("=" * 60)

    # ── GPU profile — auto-selects BF16 vs 4-bit based on VRAM ───────────────
    profile = get_gpu_profile()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading data: {input_path}")
    df = pd.read_excel(input_path) if input_path.lower().endswith(('.xlsx','.xls')) \
         else pd.read_csv(input_path, low_memory=False)

    if split_total > 1:
        df    = df.reset_index(drop=True)
        chunk = len(df) // split_total
        start = split_index * chunk
        end   = start + chunk if split_index < split_total - 1 else len(df)
        df    = df.iloc[start:end].reset_index(drop=True)
        print(f"  Split {split_index+1}/{split_total}: rows {start}–{end-1} ({len(df):,} records)")

    if n and n < len(df):
        df = df.sample(n, random_state=42).reset_index(drop=True)
    total = len(df)
    print(f"  Loaded: {total:,} tickets")

    # ── Resume: skip already-processed rows ───────────────────────────────────
    done_idxs = set()
    if os.path.exists(out_path):
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                try:
                    done_idxs.add(json.loads(line)['idx'])
                except Exception:
                    pass
        print(f"  Resume: {len(done_idxs):,} done, {total-len(done_idxs):,} remaining")
    else:
        print(f"  Output: {out_path}  (new file)")

    # ── Load model ────────────────────────────────────────────────────────────
    tok, model = load_judge(JUDGE_MODEL, profile)

    # Auto-set BATCH_SIZE from GPU profile unless user explicitly overrode it
    global BATCH_SIZE
    if BATCH_SIZE == 4:   # still at default — apply auto-detected value
        BATCH_SIZE = profile.get('auto_batch', 4)
        print(f"  Auto batch size: {BATCH_SIZE} (use --batch N to override)")

    # ── Build work queue ──────────────────────────────────────────────────────
    rows_to_process = []
    for i, row in df.iterrows():
        row_idx = int(row['idx']) if 'idx' in row.index and pd.notna(row['idx']) else i
        if row_idx not in done_idxs:
            rows_to_process.append((i, row_idx, row))

    print(f"\n  Batch size   : {BATCH_SIZE}")
    print(f"  Total batches: {(len(rows_to_process) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"  Processing   : {len(rows_to_process):,} records...\n")

    failures          = 0
    t0                = time.time()
    session_processed = 0
    last_print_time   = 0   # track time of last progress print

    # ── Warmup batch — burn in torch.compile + CUDA kernels ──────────────────
    # The first real batch after torch.compile triggers JIT compilation.
    # Running a silent warmup prevents the first 100-record window from showing
    # an artificially slow rate to the user.
    if rows_to_process and profile.get('compile'):
        print("  Warming up (burning in compiled kernels — ~60s)...")
        warmup_row = rows_to_process[0]
        _, _, row  = warmup_row
        subject    = str(row.get('subject','') or '')
        body       = str(row.get('body','')    or '')
        lang       = str(row.get('language','en') or 'en')
        try:
            run_judge_batch(tok, model, [build_judge_prompt(subject, body, lang)])
        except Exception:
            pass
        torch.cuda.synchronize()
        print("  Warmup done — timing starts now\n")
        t0 = time.time()   # reset timer AFTER warmup

    with open(out_path, 'a', encoding='utf-8') as out_f:
        for batch_start in range(0, len(rows_to_process), BATCH_SIZE):
            batch    = rows_to_process[batch_start : batch_start + BATCH_SIZE]
            prompts  = []
            row_idxs = []
            row_data = []

            for _, row_idx, row in batch:
                subject = str(row.get('subject','') or '')
                subject = '' if subject.lower() in ('nan','none') else subject
                body    = str(row.get('body','')    or '')
                lang    = str(row.get('language','en') or 'en')
                prompts.append(build_judge_prompt(subject, body, lang))
                row_idxs.append(row_idx)
                row_data.append((subject, body, lang))

            # Batch inference with fallback to single on OOM
            try:
                raw_outputs = run_judge_batch(tok, model, prompts)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on batch={len(prompts)} — falling back to single inference")
                torch.cuda.empty_cache()
                raw_outputs = []
                for prompt in prompts:
                    try:
                        raw_outputs.append(
                            run_judge_batch(tok, model, [prompt])[0])
                    except Exception as e:
                        raw_outputs.append(f"ERROR: {e}")
                        failures += 1
            except Exception as e:
                print(f"  Batch error: {e} — falling back to single inference")
                raw_outputs = [f"ERROR: {e}"] * len(prompts)
                failures   += len(prompts)

            # Write results
            for (subject, body, lang), row_idx, raw_out in zip(row_data, row_idxs, raw_outputs):
                parsed = parse_judge_output(raw_out) if not raw_out.startswith('ERROR') else {}
                if raw_out.startswith('ERROR'):
                    failures += 1

                out_f.write(json.dumps({
                    'idx':             row_idx,
                    'language':        lang,
                    'subject':         subject,
                    'body':            body,          # no truncation
                    'prob_sub':        parsed.get('prob_sub',        ''),
                    'prob_statement':  parsed.get('prob_statement',  ''),
                    'cause':           parsed.get('cause',           ''),
                    'intent':          parsed.get('intent',          'general_inquiry'),
                    'ticket_type_nli': parsed.get('ticket_type_nli', 'product_support'),
                    'judge_raw':       raw_out,       # no truncation
                }, ensure_ascii=False) + '\n')
                done_idxs.add(row_idx)

            out_f.flush()
            session_processed += len(batch)

            # Progress every 30 seconds — works for any batch size
            now = time.time()
            if now - last_print_time >= 30 or batch_start == 0:
                last_print_time = now
                el   = now - t0
                rate = session_processed / max(el, 1e-6)
                rem  = (total - len(done_idxs)) / max(rate, 1e-6)
                mem  = torch.cuda.memory_allocated() / 1e9
                print(f"  [{len(done_idxs):>6,}/{total:,}]  {el/60:.1f}m elapsed  "
                      f"~{rem/60:.1f}m left  {rate:.2f} rec/s  "
                      f"GPU {mem:.1f}GB  failures={failures}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE — {total:,} tickets in {elapsed:.0f}s ({total/max(elapsed,1):.2f} rec/s)")
    print(f"  Failures: {failures}/{total}")
    print(f"  Output  : {out_path}")

    df_out = _read_jsonl(out_path)
    n_out  = len(df_out)

    # Also write a compact CSV with the 5 aspects (+ subject/body/language/judge_raw).
    # This is useful for downstream steps that expect a CSV like aspect_extractor outputs.
    if n_out > 0:
        cols = [
            'idx', 'language', 'subject', 'body',
            'prob_sub', 'prob_statement', 'cause', 'intent', 'ticket_type_nli',
            'judge_raw',
        ]
        for c in cols:
            if c not in df_out.columns:
                df_out[c] = ''
        df_out[cols].to_csv(csv_out_path, index=False, encoding='utf-8')
        print(f"  Aspects CSV: {csv_out_path}")
    if n_out > 0:
        print(f"\n  prob_sub   filled : {(df_out['prob_sub']!='').sum()}/{n_out}  "
              f"({(df_out['prob_sub']!='').sum()/n_out*100:.0f}%)")
        print(f"  prob_stmt  filled : {(df_out['prob_statement']!='').sum()}/{n_out}  "
              f"({(df_out['prob_statement']!='').sum()/n_out*100:.0f}%)")
        print(f"  cause      filled : {(df_out['cause']!='').sum()}/{n_out}  "
              f"({(df_out['cause']!='').sum()/n_out*100:.0f}%)")
        print(f"\n  Intent distribution:")
        for v, c in df_out['intent'].value_counts().items():
            print(f"    {v:25s} {c:5d}  ({c/n_out*100:.0f}%)")
        print(f"\n  Ticket type distribution:")
        for v, c in df_out['ticket_type_nli'].value_counts().items():
            print(f"    {v:20s} {c:5d}  ({c/n_out*100:.0f}%)")

    # Auto-save only on Kaggle
    if IS_KAGGLE:
        auto_save_to_dataset(out_path, n_out)
    else:
        print(f"\n  Files saved locally to: {out_path}")
        print(f"  To download: scp root@<vast-ip>:<port>:{out_path} ./")

    return df_out


# ─── AUTO-SAVE TO KAGGLE DATASET (Kaggle only) ────────────────────────────────
def auto_save_to_dataset(jsonl_path, record_count):
    import subprocess, shutil
    from datetime import datetime

    if not os.path.exists(jsonl_path):
        print("  Auto-save skipped: file not found")
        return

    hhmm      = datetime.now().strftime('%H%M')
    base_name = f'silver_labels_{hhmm}.jsonl'
    size_mb   = os.path.getsize(jsonl_path) / 1e6

    print(f"\n{'='*60}")
    print(f"  AUTO-SAVING to Kaggle dataset...")
    print(f"  {jsonl_path}  ({size_mb:.1f} MB, {record_count:,} records)")
    print(f"  Saving as: {base_name}")

    stage = '/kaggle/working/dataset_upload/'
    os.makedirs(stage, exist_ok=True)
    shutil.copy(jsonl_path, stage + base_name)

    with open(stage + 'dataset-metadata.json', 'w') as f:
        json.dump({"title": "cs-support-ds",
                   "id":    "ranjankumarnayak/cs-support-ds",
                   "licenses": [{"name": "CC0-1.0"}]}, f)

    result = subprocess.run(
        ['kaggle', 'datasets', 'version', '-p', stage,
         '-m', f'{base_name} — {record_count} records', '--dir-mode', 'zip'],
        capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  SUCCESS → /kaggle/input/cs-support-ds/{base_name}")
    else:
        print(f"  FAILED: {result.stderr.strip()}")
        print(f"  Fallback: Save Version → Save and Run All → Output tab")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 1 — Silver label generation (H100/A100/T4 optimised)')
    parser.add_argument('--input',  required=True,
                        help='Input CSV or XLSX')
    parser.add_argument('--n',      type=int, default=None,
                        help='Sample N records (default: all)')
    parser.add_argument('--out',    default=None,
                        help='Output JSONL path')
    parser.add_argument('--csv_out', default=None,
                        help='Output CSV path with 5 aspects (default: <out>_aspects.csv)')
    parser.add_argument('--batch',  type=int, default=4,
                        help='Batch size: T4=4, RTX4090=8, A100/H100=32')
    parser.add_argument('--split',  type=int, nargs=2, default=None,
                        metavar=('INDEX', 'TOTAL'),
                        help='Multi-GPU split: --split 0 2 for GPU0, --split 1 2 for GPU1')
    parser.add_argument('--model',  default=None,
                        help='Override judge model (default: Qwen/Qwen2.5-7B-Instruct)')
    args = parser.parse_args()

    if args.model:
        JUDGE_MODEL = args.model
    BATCH_SIZE  = args.batch
    split_index = args.split[0] if args.split else 0
    split_total = args.split[1] if args.split else 1

    run(args.input, n=args.n, out_path=args.out, csv_out_path=args.csv_out,
        split_index=split_index, split_total=split_total)
