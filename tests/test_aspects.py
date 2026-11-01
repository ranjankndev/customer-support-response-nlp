"""Step 2b (plan §A.1–A.3): span clean-up, entity fix, supervised heads and the
aspect checklist prompt. Fixtures are the committed sample rows."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aspect.aspect_extractor import (NONE, extract_entities, fix_encoding,
                                         has_action_cue, is_pleasantry, jaccard,
                                         postprocess_spans)
from src.utils import build_aspect_checklist

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "docs" / "samples"


# ── A.2.1 encoding ───────────────────────────────────────────────────────────

def test_mojibake_is_repaired():
    """The plan's acceptance case: `Klage Ã¼ber` must become `Klage über`."""
    assert fix_encoding("Klage Ã¼ber") == "Klage über"
    assert fix_encoding("GrÃ¶Ãe Ã¤ndern").startswith("Gr")
    assert fix_encoding("") == ""
    assert fix_encoding(None) == ""


def test_postprocess_fixes_encoding_in_every_span():
    cleaned = postprocess_spans({
        "prob_sub": "Klage Ã¼ber das Zielgruppenmarketing",
        "cause": NONE,
        "action_taken": NONE,
    })

    assert cleaned["prob_sub"] == "Klage über das Zielgruppenmarketing"


# ── A.2.2 pleasantries ───────────────────────────────────────────────────────

@pytest.mark.parametrize("span", [
    "Your timely assistance is greatly appreciated, and I look forward to hearing back soon.",
    "Thank you for your help with this matter.",
    "Vielen Dank für Ihre Unterstützung.",
    "Mit freundlichen Grüßen, Anna",
    "Ich freue mich auf Ihre Rückmeldung.",
])
def test_pleasantries_are_recognised(span):
    assert is_pleasantry(span)


@pytest.mark.parametrize("span", [
    "The login page returns a 500 error after the update.",
    "Das System wurde heute Nacht nicht mehr erreichbar.",
    "",
])
def test_problem_sentences_are_not_pleasantries(span):
    assert not is_pleasantry(span)


def test_a_pleasantry_span_is_replaced_by_the_next_ranked_sentence():
    """Row 3 of the sample had prob_sub = "Your timely assistance is greatly
    appreciated ..." — a closing, not the problem."""
    ranked = {"prob_sub": [
        {"cleaned_text": "Your timely assistance is greatly appreciated."},
        {"cleaned_text": "I cannot export my Cubase project to WAV."},
    ]}

    cleaned = postprocess_spans(
        {"prob_sub": "Your timely assistance is greatly appreciated.",
         "cause": NONE, "action_taken": NONE}, ranked)

    assert cleaned["prob_sub"] == "I cannot export my Cubase project to WAV."


def test_a_pleasantry_span_becomes_none_without_a_candidate():
    cleaned = postprocess_spans({"prob_sub": "Thank you very much for your help.",
                                 "cause": NONE, "action_taken": NONE})

    assert cleaned["prob_sub"] == NONE


# ── A.2.3 de-duplication ─────────────────────────────────────────────────────

def test_jaccard_measures_token_overlap_with_umlauts():
    assert jaccard("Größe ändern", "Größe ändern") == 1.0
    assert jaccard("Größe ändern", "völlig anderes Thema") == 0.0
    assert jaccard("", "anything") == 0.0


def test_a_cause_identical_to_the_problem_becomes_none():
    """Sample row 0 had cause == prob_sub verbatim."""
    span = ("There was an unauthorized access attempt on the hospital's system, "
            "triggered by a phishing incident.")

    cleaned = postprocess_spans({"prob_sub": span, "cause": span,
                                 "action_taken": NONE})

    assert cleaned["prob_sub"] == span
    assert cleaned["cause"] == NONE


def test_a_distinct_cause_survives():
    cleaned = postprocess_spans({
        "prob_sub": "A customer reported a potential security breach.",
        "cause": "There is a possibility of a malware intrusion attempt.",
        "action_taken": NONE})

    assert cleaned["cause"] == "There is a possibility of a malware intrusion attempt."


def test_an_action_duplicating_the_cause_becomes_none():
    cause = ("Weitere Analyse erforderlich, um die Ursache zu identifizieren "
             "und Lösungen umzusetzen.")

    cleaned = postprocess_spans({"prob_sub": "Das Marketing wirkt unscheinbar.",
                                 "cause": cause, "action_taken": cause})

    assert cleaned["action_taken"] == NONE


# ── A.2.4 action cue ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("span", [
    "Attempts to restart systems and run antivirus scans have been unsuccessful.",
    "I have already reinstalled the client and cleared the cache.",
    "Ich habe das System neu gestartet und die Firewall-Einstellungen überprüft.",
    "Wir haben den Cache zurückgesetzt.",
])
def test_real_actions_keep_their_cue(span):
    assert has_action_cue(span)


def test_an_action_without_a_cue_becomes_none():
    cleaned = postprocess_spans({
        "prob_sub": "The reporting dashboard is empty.",
        "cause": NONE,
        "action_taken": "Additionally, could you provide information on compatibility?"})

    assert cleaned["action_taken"] == NONE


def test_an_action_with_a_cue_survives():
    action = "I restarted the service twice and checked the logs."

    cleaned = postprocess_spans({"prob_sub": "The service keeps failing.",
                                 "cause": NONE, "action_taken": action})

    assert cleaned["action_taken"] == action


def test_missing_spans_are_normalised_to_none():
    cleaned = postprocess_spans({"prob_sub": "", "cause": "", "action_taken": ""})

    assert cleaned["prob_sub"] == NONE
    assert cleaned["cause"] == NONE
    assert cleaned["action_taken"] == NONE
    assert cleaned["prob_statement"] == NONE


# ── A.2.5 entities ───────────────────────────────────────────────────────────

def test_title_case_subject_words_no_longer_become_entities():
    """Sample row 0 produced "Detected, Unauthorized, Access, Hospital, System,
    Following, Phishing, Incident, Security" — not one of them is a product."""
    subject = "Detected Unauthorized Access Following Phishing Incident"
    body = ("We noticed an access attempt last night. Please advise on the next "
            "steps for our team.")

    entities = extract_entities(subject, body, "en")

    for noise in ("Detected", "Unauthorized", "Following", "Phishing"):
        assert noise not in entities


def test_a_word_repeated_in_the_body_is_kept():
    subject = "Problem with Cubase"
    body = ("Cubase crashes on export. Cubase also freezes when loading a "
            "project. Please help with Cubase.")

    entities = extract_entities(subject, body, "en")

    assert "Cubase" in entities


def test_product_patterns_still_win():
    entities = extract_entities("IBM SPSS 28 licence question",
                                "Our QuickBooks sync to IBM SPSS 28 fails.", "en")

    assert "QuickBooks" in entities
    assert "IBM" in entities or "IBM SPSS 28" in entities


# ── A.3 checklist prompt ─────────────────────────────────────────────────────

def test_checklist_lists_what_the_reply_has_to_do():
    checklist = build_aspect_checklist({
        "prob_sub": "The login page returns a 500 error",
        "cause": "A failed cache migration",
        "action_taken": "Cleared the browser cache",
        "urgency": "high",
        "intent": "technical_support",
        "ticket_type": "Incident",
        "entities": "QuickBooks",
    })

    assert checklist.startswith("Your reply must:")
    assert "1. Solve the problem: The login page returns a 500 error" in checklist
    assert "2. Address the likely cause: A failed cache migration" in checklist
    assert "do not suggest it again" in checklist
    assert "prioritise; give a concrete next step and timeline" in checklist
    assert "Category: technical_support / Incident." in checklist
    assert "Mention these products if relevant: QuickBooks." in checklist
    # the old dump format is gone
    assert "prob_sub:" not in checklist


def test_checklist_omits_missing_aspects_and_renumbers():
    checklist = build_aspect_checklist({"prob_sub": "Invoice is wrong",
                                        "cause": NONE, "action_taken": NONE,
                                        "urgency": "low"})

    assert "1. Solve the problem: Invoice is wrong" in checklist
    assert '2. Tone for urgency "low": friendly and concise' in checklist
    assert "cause" not in checklist
    assert "3." not in checklist


def test_checklist_accepts_the_legacy_ticket_type_column():
    checklist = build_aspect_checklist({"prob_sub": "x", "ticket_type_nli": "complaint"})

    assert "Category: complaint." in checklist


def test_checklist_is_empty_when_no_aspect_survived():
    assert build_aspect_checklist({"prob_sub": NONE, "cause": NONE,
                                   "action_taken": NONE, "urgency": ""}) == ""
    assert build_aspect_checklist({}) == ""


def test_checklist_tone_depends_on_urgency():
    for urgency in ("high", "critical"):
        assert "prioritise" in build_aspect_checklist({"prob_sub": "x",
                                                       "urgency": urgency})
    for urgency in ("low", "medium"):
        assert "friendly and concise" in build_aspect_checklist({"prob_sub": "x",
                                                                 "urgency": urgency})


# ── A.1 supervised heads ─────────────────────────────────────────────────────

def test_head_targets_are_the_gold_columns():
    from src.aspect import aspect_heads

    assert aspect_heads.HEAD_TARGETS == {"intent": "queue",
                                         "ticket_type": "type",
                                         "urgency": "priority"}


def _separable_features(labels, dim=8, seed=0):
    """One cluster per class, so a linear head can learn it."""
    rng = np.random.default_rng(seed)
    classes = sorted(set(labels))
    centres = {c: rng.normal(size=dim) * 3 for c in classes}
    return np.vstack([centres[label] + rng.normal(scale=0.1, size=dim)
                      for label in labels]).astype(np.float32)


def test_heads_beat_the_majority_baseline_on_macro_f1():
    """The plan's acceptance test, on a CPU-sized synthetic subset: the heads must
    beat predicting the majority class."""
    from src.aspect.aspect_heads import evaluate_heads, train_heads

    queues = ["Technical Support", "Billing", "Product Support"] * 20
    types = ["Incident", "Request", "Problem"] * 20
    priorities = ["low", "medium", "high"] * 20
    targets = pd.DataFrame({"queue": queues, "type": types, "priority": priorities,
                            "language": (["en"] * 30 + ["de"] * 30)})
    features = _separable_features(queues)

    bundle = train_heads(features, targets)
    table = evaluate_heads(bundle, features, targets, targets["language"])

    overall = table[table["split"] == "all"].set_index("head")
    assert set(overall.index) == {"intent", "ticket_type", "urgency"}
    assert overall.loc["intent", "f1_macro"] > overall.loc["intent", "majority_f1_macro"]
    assert overall.loc["intent", "beats_majority"]
    assert set(table["split"]) == {"all", "en", "de"}


def test_heads_predict_labels_for_new_rows():
    from src.aspect.aspect_heads import predict_from_features, train_heads

    queues = ["Billing", "Technical Support"] * 25
    targets = pd.DataFrame({"queue": queues,
                            "type": ["Request", "Incident"] * 25,
                            "priority": ["low", "high"] * 25})
    features = _separable_features(queues)

    bundle = train_heads(features, targets)
    predicted = predict_from_features(bundle, features[:4])

    assert predicted["intent"] == queues[:4]
    assert len(predicted["urgency"]) == 4


def test_predict_rejects_embeddings_of_the_wrong_width():
    from src.aspect.aspect_heads import predict_from_features, train_heads

    targets = pd.DataFrame({"queue": ["a", "b"] * 10, "type": ["x", "y"] * 10,
                            "priority": ["low", "high"] * 10})
    bundle = train_heads(_separable_features(["a", "b"] * 10), targets)

    with pytest.raises(ValueError, match="dimensional"):
        predict_from_features(bundle, np.zeros((2, 3), dtype=np.float32))


def test_train_heads_needs_the_gold_column():
    from src.aspect.aspect_heads import train_heads

    targets = pd.DataFrame({"queue": ["a", "b"] * 10})

    with pytest.raises(ValueError, match="type"):
        train_heads(_separable_features(["a", "b"] * 10), targets)


def test_missing_heads_bundle_says_how_to_train_it(tmp_path):
    from src.aspect.aspect_heads import load_bundle

    with pytest.raises(FileNotFoundError, match="aspect_heads"):
        load_bundle(str(tmp_path / "nope.joblib"))


def test_sft_label_collapse_is_documented_from_the_sample():
    """A.0's evidence, turned into a number the report can cite."""
    from src.aspect.aspect_heads import sft_label_collapse

    table = sft_label_collapse(str(SAMPLES / "aspects_SAMPLE.csv")).set_index("label")

    assert table.loc["intent", "collapsed_share"] > 0.5      # general_inquiry
    assert table.loc["ticket_type_nli", "distinct_values"] <= 2


def test_embedding_text_matches_what_the_indexer_embeds():
    from src.aspect.aspect_heads import embedding_text

    assert embedding_text("Subj", "Body text") == "Subj\n\nBody text"
    assert embedding_text("Subj", "0123456789", body_max_chars=4) == "Subj\n\n0123"


# ── the extractor still runs end to end on the sample rows ───────────────────

def test_process_row_cleans_the_spans_of_a_real_sample_row(monkeypatch):
    """Stages 1-3 with a stub ranker: what matters is that postprocess_spans is
    applied, not how the sentences are ranked (that needs the MiniLM model)."""
    import src.aspect.aspect_extractor as extractor

    row = pd.read_csv(SAMPLES / "cs-support-multi-test_RAW_DATA.csv",
                      dtype=str).fillna("").iloc[0].to_dict()

    def fake_rank(sentences, anchor_vecs, ranker):
        ordered = [{"cleaned_text": s["cleaned_text"]} for s in sentences]
        return {field: ordered for field in
                ("prob_sub", "prob_statement", "cause", "action_taken")}

    monkeypatch.setattr(extractor, "rank_sentences", fake_rank)
    result = extractor.process_row(row, {}, object())

    assert result["prob_sub"]
    # every span is either real text or an explicit 'none'
    for field in ("prob_sub", "cause", "action_taken"):
        value = result[field]
        assert value == NONE or len(value.split()) >= 4
    # cause never just repeats the problem any more
    if result["cause"] != NONE:
        assert jaccard(result["cause"], result["prob_sub"]) <= 0.8
    if result["action_taken"] != NONE:
        assert has_action_cue(result["action_taken"])


# ── A.3 in the two prompt builders ───────────────────────────────────────────

def _stub_retriever(style):
    import types

    from src.retrieval.retriever import TicketRetriever

    retriever = TicketRetriever.__new__(TicketRetriever)
    retriever.config = types.SimpleNamespace(aspect_prompt_style=style,
                                             prompt_body_max_chars=0)
    return retriever


SPANS = {"prob_sub": "Login fails after the update",
         "cause": NONE,
         "action_taken": "I restarted the router",
         "urgency": "high",
         "intent": "technical_support",
         "ticket_type_nli": "Incident"}


def test_rag_aspect_prompt_uses_the_checklist_by_default():
    prompt = _stub_retriever("checklist").build_prompt(
        "Subj", "Body", [{"subject": "S1", "response": "R1"}], SPANS)

    assert "Your reply must:" in prompt
    assert "1. Solve the problem: Login fails after the update" in prompt
    assert "Aspects :" not in prompt
    # the RAG context and the ticket are still there
    assert "Context : [1] Subject: S1" in prompt
    assert prompt.rstrip().endswith("Support reply:")


def test_the_legacy_slot_dump_stays_reachable_via_config():
    prompt = _stub_retriever("slots").build_prompt(
        "Subj", "Body", [{"subject": "S1", "response": "R1"}], SPANS)

    assert "Aspects : prob_sub: Login fails after the update" in prompt
    assert "Your reply must:" not in prompt


def test_aspect_only_prompt_uses_the_checklist_too():
    from src.generation.response_generator_v2 import build_aspect_only_prompt

    row = pd.Series({"subject": "Subj", "body": "Body", "language": "en"})

    prompt = build_aspect_only_prompt(row, SPANS)

    assert "Your reply must:" in prompt
    assert "Aspects  :" not in prompt
    assert prompt.rstrip().endswith("Support reply:")
