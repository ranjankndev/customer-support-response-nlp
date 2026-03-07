"""
domain_skipgram.py
==================
Domain-specific Skip-Gram Word2Vec for customer support tickets.

ADAPTED FROM: Assignment 2 Skip-Gram implementation.
CHANGES vs original:
  - Corpus: support ticket text + tags (not Brown corpus)
  - Sentence construction: 7-sentence strategy per ticket (corpus_builder.py)
  - Vocab: includes tag tokens, metadata tokens, multi-word tag pieces
  - Subsampling: tuned for domain (common words like "please", "dear" discarded)
  - Output: saves vocab + embeddings for downstream use in:
      * FAISS retrieval index (problem-solution embedding space)
      * BM25 query expansion (word2vec neighbors expand query)
      * TicketTypeClassifier features (mean embedding of ticket)
      * Aspect extractor (embedding similarity to find problem span)

CONFIG (word2vec_support_config.json):
  WINDOW_SIZE   : 3  (each word sees 3 left + 3 right neighbors)
  EMBEDDING_DIM : 100
  MIN_COUNT     : 2  (lower than Brown — domain vocab is smaller)
  NEG_SAMPLES   : 5
  EPOCHS        : 10
  BATCH_SIZE    : 512
  SUBSAMPLE_T   : 1e-4
  LR            : 0.001
"""

import os
import json
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

from corpus_builder import build_corpus, get_corpus_stats


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (mirrors your assignment config structure)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "WINDOW_SIZE":    3,
    "EMBEDDING_DIM":  100,
    "MIN_COUNT":      2,
    "NEG_SAMPLES":    5,
    "EPOCHS":         10,
    "BATCH_SIZE":     512,
    "SUBSAMPLE_T":    1e-4,
    "LR":             0.001,
    "LR_STEP":        3,
    "LR_GAMMA":       0.5,
    "RANDOM_SEED":    42,
    "OUTPUT_DIR":     "embeddings",
    "SAVE_FILENAME":  "support_skipgram.pt",
    "VOCAB_FILENAME": "support_vocab.pt",
}


def load_config(path: str = "word2vec_support_config.json") -> dict:
    if Path(path).exists():
        with open(path) as f:
            cfg = json.load(f)
        # Merge with defaults so missing keys don't break
        return {**DEFAULT_CONFIG, **cfg}
    return DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (unchanged from assignment — same SkipGram architecture)
# ─────────────────────────────────────────────────────────────────────────────

class SkipGramModel(nn.Module):
    """
    Two-embedding Skip-Gram with Negative Sampling.
    v_embeddings: target (center) word embeddings  -> used at inference
    u_embeddings: context word embeddings           -> used during training only
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.v_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.u_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        initrange = 0.5 / embed_dim
        self.v_embeddings.weight.data.uniform_(-initrange, initrange)
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data[0].fill_(0)
        self.u_embeddings.weight.data[0].fill_(0)

    def forward(self, target: torch.Tensor,
                context: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        v       = self.v_embeddings(target)
        u       = self.u_embeddings(context)
        n       = self.u_embeddings(negative)
        pos_loss = torch.log(torch.sigmoid(torch.sum(v * u, dim=1)) + 1e-10)
        neg_loss = torch.sum(
            torch.log(torch.sigmoid(-torch.bmm(n, v.unsqueeze(2)).squeeze(2)) + 1e-10),
            dim=1
        )
        return -torch.mean(pos_loss + neg_loss)


# ─────────────────────────────────────────────────────────────────────────────
# VOCAB BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(sentences: List[List[str]],
                min_count: int) -> Tuple[Dict[str, int], Counter, List[str]]:
    """
    Build vocabulary from sentences.
    Same structure as assignment: PAD=0, UNK=1, then sorted valid words.

    NOTE on min_count=2 for domain corpus:
      Brown corpus has 1M tokens -> min_count=5 makes sense.
      28k tickets corpus is ~2-3M tokens but domain vocabulary is small.
      min_count=2 keeps rare-but-important terms like specific error codes,
      product names, and tag values that appear in only a few tickets.
    """
    word_counts = Counter(w for s in sentences for w in s)
    valid_words = sorted(w for w, c in word_counts.items() if c >= min_count)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(valid_words)})
    print(f"[Vocab] Total tokens: {sum(word_counts.values()):,}")
    print(f"[Vocab] Unique words before filter: {len(word_counts):,}")
    print(f"[Vocab] Vocab size (min_count={min_count}): {len(vocab):,}")
    return vocab, word_counts, valid_words


# ─────────────────────────────────────────────────────────────────────────────
# NEGATIVE SAMPLING DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def build_neg_distribution(vocab: Dict[str, int],
                           word_counts: Counter,
                           valid_words: List[str]) -> torch.Tensor:
    """
    Unigram^0.75 distribution for negative sampling.
    Identical to assignment — smoothed to give rare words more chance
    of being selected as negatives (prevents common words dominating).
    """
    freq_tensor = torch.zeros(len(vocab))
    for w in valid_words:
        freq_tensor[vocab[w]] = word_counts[w]
    neg_dist  = freq_tensor ** 0.75
    neg_dist /= neg_dist.sum()
    return neg_dist


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PAIR GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_training_pairs(sentences: List[List[str]],
                         vocab: Dict[str, int],
                         valid_words: List[str],
                         word_counts: Counter,
                         window_size: int,
                         subsample_t: float) -> List[Tuple[int, int]]:
    """
    Generate (target, context) index pairs with subsampling.

    WINDOW_SIZE=3 means: for each word, look at 3 words left AND 3 right.
    This is why tags prepended to body sentences create associations:
      sentence: [outage, disruption, tech, server, went, offline, blocking]
      window=3 around 'server': [outage, disruption, tech] + [went, offline, blocking]
      -> 'server' learns to be close to 'outage' and 'disruption' tags

    Subsampling discards high-freq words probabilistically:
      P(discard) = 1 - sqrt(t / freq(w))
      Common words like "please", "dear", "the" get heavily subsampled.
      Domain terms like "outage", "api", "timeout" are rarely discarded.
    """
    total_words = sum(word_counts[w] for w in valid_words)
    word_freq   = {w: word_counts[w] / total_words for w in valid_words}
    data        = []
    print("Generating training pairs...")

    for sentence in sentences:
        ids = []
        for w in sentence:
            if w not in vocab or w in ("<PAD>", "<UNK>"):
                continue
            # Subsampling — same formula as assignment
            p_discard = max(0.0, 1 - math.sqrt(subsample_t / (word_freq.get(w, 1e-9) + 1e-9)))
            if random.random() >= p_discard:
                ids.append(vocab[w])

        for i, target_id in enumerate(ids):
            lo = max(0, i - window_size)
            hi = min(len(ids), i + window_size + 1)
            for j in range(lo, hi):
                if i != j:
                    data.append((target_id, ids[j]))

    print(f"[Pairs] Total training pairs: {len(data):,}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(model: SkipGramModel,
          data: List[Tuple[int, int]],
          neg_distribution: torch.Tensor,
          device: torch.device,
          config: dict) -> List[float]:
    """
    Training loop — same structure as assignment with added loss tracking.
    Returns per-epoch losses for plotting/comparison.
    """
    optimizer = optim.Adam(model.parameters(), lr=config["LR"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config["LR_STEP"], gamma=config["LR_GAMMA"]
    )
    neg_dist_device = neg_distribution.to(device)
    epoch_losses = []

    print(f"[Train] Device: {device} | Pairs: {len(data):,} | "
          f"Epochs: {config['EPOCHS']} | Batch: {config['BATCH_SIZE']}")

    for epoch in range(config["EPOCHS"]):
        random.shuffle(data)
        epoch_loss, num_batches = 0.0, 0

        for i in range(0, len(data), config["BATCH_SIZE"]):
            batch = data[i : i + config["BATCH_SIZE"]]
            if len(batch) < 2:
                continue

            targets   = torch.tensor([p[0] for p in batch], dtype=torch.long).to(device)
            contexts  = torch.tensor([p[1] for p in batch], dtype=torch.long).to(device)
            negatives = torch.multinomial(
                neg_dist_device,
                len(batch) * config["NEG_SAMPLES"],
                replacement=True
            ).view(len(batch), config["NEG_SAMPLES"])

            optimizer.zero_grad()
            loss = model(targets, contexts, negatives)
            loss.backward()
            optimizer.step()
            epoch_loss  += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{config['EPOCHS']} | "
              f"Loss: {avg_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()

    return epoch_losses


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def train_support_embeddings(data_path: str,
                              config_path: str = "word2vec_support_config.json") -> dict:
    """
    Full training pipeline.
    Call this from main.py or a notebook.

    Returns dict with model, vocab, and training stats for downstream use.
    """
    config = load_config(config_path)
    random.seed(config["RANDOM_SEED"])
    torch.manual_seed(config["RANDOM_SEED"])

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DOMAIN SKIP-GRAM — Customer Support Ticket Embeddings")
    print(f"{'='*60}")

    df = pd.read_excel(data_path) if data_path.endswith(('.xlsx', '.xls')) \
        else pd.read_csv(data_path)
    print(f"[Data] Loaded {len(df)} tickets")

    # ── Build corpus ──────────────────────────────────────────────────────────
    sentences, corpus_stats = build_corpus(df)
    vocab_stats = get_corpus_stats(sentences)
    print(f"[Corpus] Unique tokens: {vocab_stats['unique_tokens']:,}")
    print(f"[Corpus] Singletons: {vocab_stats['singletons']:,}")

    # ── Build vocab ───────────────────────────────────────────────────────────
    vocab, word_counts, valid_words = build_vocab(sentences, config["MIN_COUNT"])

    # ── Check tag tokens are in vocab ─────────────────────────────────────────
    tag_cols = [c for c in df.columns if c.startswith('tag_')]
    all_tags = set()
    for tc in tag_cols:
        for v in df[tc].dropna().unique():
            for tok in str(v).lower().split():
                all_tags.add(tok)
    tag_coverage = sum(1 for t in all_tags if t in vocab) / max(len(all_tags), 1)
    print(f"[Tags] {len(all_tags)} unique tag tokens | "
          f"Coverage in vocab: {tag_coverage:.1%}")

    # ── Negative sampling distribution ────────────────────────────────────────
    neg_distribution = build_neg_distribution(vocab, word_counts, valid_words)

    # ── Training pairs ────────────────────────────────────────────────────────
    data = build_training_pairs(
        sentences, vocab, valid_words, word_counts,
        config["WINDOW_SIZE"], config["SUBSAMPLE_T"]
    )

    # ── Train model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SkipGramModel(len(vocab), config["EMBEDDING_DIM"]).to(device)
    losses = train(model, data, neg_distribution, device, config)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    embed_path = os.path.join(config["OUTPUT_DIR"], config["SAVE_FILENAME"])
    vocab_path = os.path.join(config["OUTPUT_DIR"], config["VOCAB_FILENAME"])
    torch.save(model.v_embeddings.weight.data.cpu(), embed_path)
    torch.save(vocab, vocab_path)
    print(f"\n[Saved] Embeddings -> {embed_path}")
    print(f"[Saved] Vocab      -> {vocab_path}")

    return {
        "model":         model,
        "vocab":         vocab,
        "word_counts":   word_counts,
        "valid_words":   valid_words,
        "epoch_losses":  losses,
        "corpus_stats":  corpus_stats,
        "vocab_size":    len(vocab),
        "embed_path":    embed_path,
        "vocab_path":    vocab_path,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True, help="Path to .xlsx or .csv")
    p.add_argument("--config", default="word2vec_support_config.json")
    args = p.parse_args()
    train_support_embeddings(args.data, args.config)
