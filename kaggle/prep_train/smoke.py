"""Plan §10 step 3 acceptance on a GPU: one <bos>, DE prompt -> German reply,
5 long hits within budget. Run from the repo root: python kaggle/prep_train/smoke.py"""

import json
import sys
import types

sys.path.insert(0, ".")

from src.generation.model_backend import (count_tokens_fn, generate,  # noqa: E402
                                          load_generator, make_chat_fn,
                                          text_tokenizer)
from src.retrieval.retriever import TicketRetriever  # noqa: E402
from src.utils import fit_prompt  # noqa: E402

MODELS = ("unsloth/gemma-3-4b-it", "unsloth/gemma-2-2b-it")
BUDGET = 1024
GERMAN = (" der ", " die ", " das ", " und ", " ihre ", " wir ", " sie ", " ihnen ")


def stub_retriever() -> TicketRetriever:
    config = types.SimpleNamespace(aspect_prompt_style="checklist",
                                   prompt_body_max_chars=0, ctx_doc_max_chars=1200,
                                   reply_language_instruction=True)
    retriever = TicketRetriever.__new__(TicketRetriever)
    retriever.config = config
    return retriever


def check(model_id: str) -> dict:
    tok, model, chat_fn = load_generator(model_id, backend="hf", use_4bit=True)
    text_tok = text_tokenizer(tok)
    count = count_tokens_fn(tok)
    retriever = stub_retriever()
    spans = {"prob_sub": "Der Export schlaegt fehl", "urgency": "high"}
    hits = [{"subject": f"Ticket {i}", "response": "word " * 800} for i in range(5)]

    prompts, used = {}, {}
    for lang, body in (("en", "My CSV export fails with an error since yesterday."),
                       ("de", "Mein CSV-Export schlaegt seit gestern mit einem Fehler fehl.")):
        prompts[lang], used[lang] = fit_prompt(
            lambda docs, lang=lang, body=body: retriever.build_prompt(
                "Export", body, docs, spans, lang), hits, count, BUDGET)

    templated = make_chat_fn(tok)([prompts["de"]])[0]
    ids = text_tok(templated, add_special_tokens=False)["input_ids"][:3]
    n_bos = text_tok(templated, add_special_tokens=False)["input_ids"].count(
        text_tok.bos_token_id)
    replies = generate([prompts["en"], prompts["de"]], tok, model, chat_fn,
                       max_new_tokens=120, max_input_tokens=BUDGET)
    de_reply = f" {replies[1].lower()} "
    result = {
        "model": model_id, "first_3_ids": ids, "n_bos": n_bos,
        "prompt_tokens": {k: count(v) for k, v in prompts.items()},
        "docs_used": used, "budget": BUDGET,
        "ends_with_support_reply": all(p.rstrip().endswith("Support reply:")
                                       for p in prompts.values()),
        "en_reply": replies[0], "de_reply": replies[1],
        "de_is_german": sum(w in de_reply for w in GERMAN) >= 2,
    }
    result["ok"] = (result["n_bos"] == 1 and bool(replies[0].strip())
                    and result["de_is_german"] and result["ends_with_support_reply"]
                    and all(n + 16 <= BUDGET for n in result["prompt_tokens"].values())
                    and all(u < 5 for u in used.values()))
    del model
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    results = [check(m) for m in MODELS]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    with open("smoke_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    sys.exit(0 if all(r["ok"] for r in results) else 1)
