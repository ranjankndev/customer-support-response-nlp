# Customer Support NLP Pipeline

NL2SQL-adjacent research project — Aspect-Aware RAG for customer support ticket response generation.

## Pipeline
1. `preprocessing.py` — clean, encode-fix, PII mask, train/val/test split
2. `sbert_finetune.py` — fine-tune multilingual SBERT for ticket retrieval
3. `aspect_pipeline.py` — extract 6 aspects per ticket (categorization, prob_sub, prob_statement, cause, priority, urgency_vibe)
4. `rag.py` — hybrid BM25+FAISS retrieval with aspect-aware reranking
5. `response_generator.py` — T5 response generation, baseline vs RAG+aspect comparison

## Running on Kaggle
See `kaggle/` folder for notebook runners.
Data lives in Kaggle Dataset: `customer-support-28k` (private).