"""Step 10 (plan §8.2 and §A.4): the report — per-row metrics, paired bootstrap
CIs, ACS/AAS, results.md and the chart. Built on fake system CSVs, as the plan's
acceptance test asks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.minimal_run import (BASELINE_SYSTEM, REPORT_METRICS, aspect_coverage,
                             aspect_table, bootstrap_table,
                             build_results_markdown, cmd_report,
                             format_interval, load_systems, markdown_table,
                             per_row_metrics, qualitative_examples,
                             route_shares, split_sentences_for_acs,
                             system_reply, takeaway_lines)


# ── fake system outputs ──────────────────────────────────────────────────────

def _system_frame(label: str, n: int = 20, quality: float = 1.0,
                  route: str = "") -> pd.DataFrame:
    """`quality` 1.0 copies the gold answer; 0.0 writes something unrelated."""
    rows = []
    for i in range(n):
        language = "en" if i % 2 == 0 else "de"
        gold = (f"please reset your password number {i}" if language == "en"
                else f"bitte setzen Sie Ihr Passwort zurück Nummer {i}")
        reply = gold if (i / n) < quality else (
            "völlig anderes Thema" if language == "de" else "unrelated topic")
        rows.append({
            "idx": str(i), "language": language,
            "subject": f"subject {i}", "body": f"body {i}", "answer": gold,
            f"{label}_response": reply, f"{label}_prompt": "p",
            f"{label}_length": len(reply.split()),
            "prob_sub": f"password reset {i}", "prob_statement": f"p {i}",
            "cause": "none", "action_taken": "none",
            "intent": "login_or_access", "ticket_type_nli": "Incident",
            "entities": "", "urgency": "low",
            "top_ce_score": f"{0.9 - 0.04 * i:.3f}",
            "route": route,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def output_dir(tmp_path):
    """Two systems over the same 20 rows: s2 is strong, s1 is weaker."""
    _system_frame("s2_gemma3", 20, quality=1.0).to_csv(
        tmp_path / "s2_gemma3.csv", index=False)
    _system_frame("s1_gemma2", 20, quality=0.3).to_csv(
        tmp_path / "s1_gemma2.csv", index=False)
    return tmp_path


class _StubEmbedder:
    """Embeds by token overlap, so ACS is predictable without a model."""

    VOCAB = ["password", "reset", "passwort", "zurück", "unrelated", "anderes",
             "invoice"]

    def encode(self, texts, kind="passage"):
        vectors = []
        for text in texts:
            tokens = set((text or "").lower().replace(".", "").split())
            vectors.append([1.0 if word in tokens else 0.0
                            for word in self.VOCAB])
        return np.asarray(vectors, dtype=np.float32)


class _StubCalc:
    """ROUGE and lang_match without loading rouge_score/langdetect."""

    def rouge(self, hypotheses, references, per_row=False):
        rows = []
        for hypothesis, reference in zip(hypotheses, references):
            h = set((hypothesis or "").lower().split())
            r = set((reference or "").lower().split())
            rows.append(len(h & r) / len(h | r) if (h or r) else 0.0)
        out = {name: float(np.mean(rows)) if rows else 0.0
               for name in ("rouge1", "rouge2", "rougeL")}
        if per_row:
            for name in ("rouge1", "rouge2", "rougeL"):
                out[f"{name}_rows"] = list(rows)
        return out

    def lang_match(self, hypotheses, languages, per_row=False):
        rows = [1 if (language or "") == "de" and "passwort" in (h or "").lower()
                else 1 if (language or "") == "en" and "password" in (h or "").lower()
                else 0 for h, language in zip(hypotheses, languages)]
        out = {"lang_match": float(np.mean(rows)) if rows else 0.0}
        if per_row:
            out["lang_match_rows"] = rows
        return out

    def bertscore_true(self, hypotheses, references, per_row=False):
        rows = [1.0 if (h or "").strip() == (r or "").strip() else 0.2
                for h, r in zip(hypotheses, references)]
        out = {"bertscore_f1": float(np.mean(rows)) if rows else 0.0}
        if per_row:
            out["bertscore_f1_rows"] = rows
        return out


# ── loading and alignment ────────────────────────────────────────────────────

def test_systems_are_loaded_and_aligned_on_the_same_rows(output_dir):
    frames = load_systems(str(output_dir), ["s1_gemma2", "s2_gemma3"])

    assert set(frames) == {"s1_gemma2", "s2_gemma3"}
    assert list(frames["s1_gemma2"]["idx"]) == list(frames["s2_gemma3"]["idx"])
    assert len(frames["s2_gemma3"]) == 20


def test_a_system_with_different_rows_is_refused(output_dir):
    """plan §8.2 step 1: the bootstrap is paired, so the samples must match."""
    extra = _system_frame("s9_other", 20)
    extra.loc[0, "idx"] = "999"
    extra.to_csv(output_dir / "s9_other.csv", index=False)

    with pytest.raises(AssertionError, match="same sample"):
        load_systems(str(output_dir), ["s2_gemma3", "s9_other"])


def test_a_missing_system_file_says_which(output_dir):
    with pytest.raises(FileNotFoundError, match="s7_missing"):
        load_systems(str(output_dir), ["s7_missing"])


def test_duplicate_rows_are_refused(output_dir):
    frame = _system_frame("s8_dup", 4)
    frame = pd.concat([frame, frame.head(1)])
    frame.to_csv(output_dir / "s8_dup.csv", index=False)

    with pytest.raises(ValueError, match="duplicate idx"):
        load_systems(str(output_dir), ["s8_dup"])


def test_the_reply_column_is_found_by_label(output_dir):
    frames = load_systems(str(output_dir), ["s2_gemma3"])

    replies = system_reply(frames["s2_gemma3"], "s2_gemma3")

    assert len(replies) == 20
    assert replies[0].startswith("please reset")


# ── ACS / AAS ────────────────────────────────────────────────────────────────

def test_sentence_split_falls_back_to_the_whole_reply():
    assert split_sentences_for_acs("") == []
    assert split_sentences_for_acs("short") == ["short"]
    assert len(split_sentences_for_acs("First one here. Second one here.")) == 2


def test_acs_is_one_when_a_sentence_states_the_aspect():
    scores = aspect_coverage(_StubEmbedder(),
                             ["We will reset your password now."],
                             [{"prob_sub": "password reset", "cause": "none"}],
                             ["prob_sub", "cause"])

    assert scores[0] == pytest.approx(1.0)


def test_acs_drops_when_the_reply_ignores_the_aspect():
    scores = aspect_coverage(_StubEmbedder(), ["unrelated invoice advice"],
                             [{"prob_sub": "password reset"}], ["prob_sub"])

    assert scores[0] == pytest.approx(0.0)


def test_acs_is_none_when_there_is_no_aspect_to_cover():
    """Such a row is excluded from the mean, not counted as zero."""
    scores = aspect_coverage(_StubEmbedder(), ["any reply"],
                             [{"prob_sub": "none", "cause": ""}],
                             ["prob_sub", "cause"])

    assert scores == [None]


def test_aas_only_looks_at_action_taken():
    scores = aspect_coverage(
        _StubEmbedder(), ["You already tried a password reset."],
        [{"action_taken": "password reset", "prob_sub": "invoice"}],
        ["action_taken"])

    assert scores[0] == pytest.approx(1.0)


def test_acs_averages_over_the_aspects_of_a_row():
    scores = aspect_coverage(
        _StubEmbedder(), ["We will reset your password."],
        [{"prob_sub": "password reset", "cause": "invoice"}],
        ["prob_sub", "cause"])

    # prob_sub covered (1.0), cause not (0.0)
    assert scores[0] == pytest.approx(0.5)


# ── per-row metrics ──────────────────────────────────────────────────────────

def test_per_row_metrics_cover_every_system_and_row(output_dir):
    frames = load_systems(str(output_dir), ["s1_gemma2", "s2_gemma3"])

    per_row = per_row_metrics(frames, _StubCalc(), _StubEmbedder())

    assert len(per_row) == 40
    assert set(per_row["system"]) == {"s1_gemma2", "s2_gemma3"}
    for column in ("rouge1", "rouge2", "rougeL", "lang_match", "bertscore_f1",
                   "acs", "aas", "language", "idx"):
        assert column in per_row.columns
    # the strong system scores above the weak one
    means = per_row.groupby("system")["rougeL"].mean()
    assert means["s2_gemma3"] > means["s1_gemma2"]


def test_per_row_metrics_can_skip_bertscore_and_the_aspect_scores(output_dir):
    frames = load_systems(str(output_dir), ["s2_gemma3"])

    per_row = per_row_metrics(frames, _StubCalc(), embedder=None,
                              with_bertscore=False)

    assert "bertscore_f1" not in per_row.columns
    assert per_row["acs"].isna().all()


# ── bootstrap ────────────────────────────────────────────────────────────────

@pytest.fixture
def table(output_dir):
    frames = load_systems(str(output_dir), ["s1_gemma2", "s2_gemma3"])
    per_row = per_row_metrics(frames, _StubCalc(), _StubEmbedder())
    return bootstrap_table(per_row)


def test_the_ci_columns_exist(table):
    """The plan's acceptance test."""
    for column in ("system", "split", "metric", "n", "mean", "ci_low",
                   "ci_high"):
        assert column in table.columns
    assert (table["ci_low"] <= table["mean"] + 1e-9).all()
    assert (table["mean"] <= table["ci_high"] + 1e-9).all()


def test_every_metric_is_reported_overall_and_per_language(table):
    assert set(table["split"]) == {"all", "en", "de"}
    reported = set(table["metric"])
    assert {"rouge1", "rougeL", "lang_match", "bertscore_f1", "acs"} <= reported
    assert reported <= set(REPORT_METRICS)


def test_differences_are_paired_against_the_baseline(table):
    assert BASELINE_SYSTEM == "s2_gemma3"
    baseline = table[table["system"] == "s2_gemma3"]
    other = table[table["system"] == "s1_gemma2"]

    assert baseline["diff"].isna().all()
    assert (other["vs_baseline"] == "s2_gemma3").all()
    for column in ("diff", "diff_ci_low", "diff_ci_high", "significant"):
        assert column in other.columns


def test_a_real_gap_is_called_significant(table):
    row = table[(table["system"] == "s1_gemma2") & (table["split"] == "all")
                & (table["metric"] == "rougeL")].iloc[0]

    assert row["diff"] < 0
    assert row["diff_ci_high"] < 0
    assert bool(row["significant"])


def test_a_system_compared_with_itself_has_no_difference(output_dir):
    frames = load_systems(str(output_dir), ["s2_gemma3"])
    per_row = per_row_metrics(frames, _StubCalc(), _StubEmbedder())

    table = bootstrap_table(per_row)

    assert set(table["system"]) == {"s2_gemma3"}
    assert "diff" not in table.columns or table["diff"].isna().all()


def test_the_bootstrap_is_reproducible(output_dir):
    frames = load_systems(str(output_dir), ["s1_gemma2", "s2_gemma3"])
    per_row = per_row_metrics(frames, _StubCalc(), _StubEmbedder())

    first = bootstrap_table(per_row)
    second = bootstrap_table(per_row)

    pd.testing.assert_frame_equal(first, second)


def test_the_same_resamples_are_used_for_both_systems(output_dir):
    """What makes a difference paired: identical rows on both sides."""
    frames = load_systems(str(output_dir), ["s1_gemma2", "s2_gemma3"])
    per_row = per_row_metrics(frames, _StubCalc(), _StubEmbedder())
    table = bootstrap_table(per_row)

    row = table[(table["system"] == "s1_gemma2") & (table["split"] == "all")
                & (table["metric"] == "rougeL")].iloc[0]
    baseline = table[(table["system"] == "s2_gemma3")
                     & (table["split"] == "all")
                     & (table["metric"] == "rougeL")].iloc[0]

    assert row["diff"] == pytest.approx(row["mean"] - baseline["mean"], abs=1e-4)


# ── markdown ─────────────────────────────────────────────────────────────────

def test_the_table_is_formatted_as_mean_with_an_interval(table):
    row = table.iloc[0]

    assert format_interval(row).startswith(f"{row['mean']:.4f} [")

    markdown = markdown_table(table, "all")
    assert markdown.startswith("| System |")
    assert "s2_gemma3" in markdown and "s1_gemma2" in markdown
    assert markdown.count("[") >= 2


def test_takeaways_are_generated_from_the_numbers(table):
    lines = takeaway_lines(table)

    assert lines
    assert any("significant" in line for line in lines)
    assert any("s1_gemma2" in line for line in lines)


def test_takeaways_say_so_when_nothing_is_significant():
    table = pd.DataFrame([{"system": "a", "split": "all", "metric": "rougeL",
                           "n": 20, "mean": 0.3, "ci_low": 0.2, "ci_high": 0.4,
                           "vs_baseline": "b", "diff": 0.01,
                           "diff_ci_low": -0.05, "diff_ci_high": 0.07,
                           "significant": False}])

    lines = takeaway_lines(table)

    assert len(lines) == 1
    assert "not significant" in lines[0] or "includes 0" in lines[0]


def test_the_aspect_table_lists_acs_and_aas(table):
    markdown = aspect_table(table)

    assert "ACS" in markdown and "AAS" in markdown
    assert "s2_gemma3" in markdown


def test_the_aspect_table_says_when_it_has_no_scores():
    table = pd.DataFrame([{"system": "a", "split": "all", "metric": "rougeL",
                           "n": 1, "mean": 0.1, "ci_low": 0.0, "ci_high": 0.2}])

    assert "no aspect scores" in aspect_table(table)


def test_route_shares_are_reported_when_crag_ran(output_dir):
    _system_frame("s3_gemma3_crag", 20, route="no_context").to_csv(
        output_dir / "s3_gemma3_crag.csv", index=False)
    frames = load_systems(str(output_dir), ["s3_gemma3_crag"])
    per_row = per_row_metrics(frames, _StubCalc(), None, with_bertscore=False)

    assert "no_context 100%" in route_shares(per_row)


def test_route_shares_are_empty_without_crag(output_dir):
    frames = load_systems(str(output_dir), ["s2_gemma3"])
    per_row = per_row_metrics(frames, _StubCalc(), None, with_bertscore=False)

    assert route_shares(per_row) == ""


def test_examples_cover_one_en_and_two_de(output_dir):
    frames = load_systems(str(output_dir), ["s1_gemma2", "s2_gemma3"])

    markdown = qualitative_examples(frames)

    assert markdown.count("#### Ticket") == 3
    assert "Gold answer:" in markdown
    assert "**s1_gemma2:**" in markdown and "**s2_gemma3:**" in markdown


def test_results_markdown_has_every_section(output_dir, table):
    frames = load_systems(str(output_dir), ["s1_gemma2", "s2_gemma3"])
    per_row = per_row_metrics(frames, _StubCalc(), _StubEmbedder())

    markdown = build_results_markdown(table, per_row, frames)

    for heading in ("# AspectForge v3", "## Overall", "## EN", "## DE",
                    "## Takeaways", "## Aspect coverage", "## Examples"):
        assert heading in markdown
    assert "paired resamples" in markdown


# ── end to end ───────────────────────────────────────────────────────────────

def test_the_report_writes_every_artefact(output_dir, tmp_path, monkeypatch):
    import src.minimal_run as module

    monkeypatch.setattr(module, "_metrics_calculator", lambda *a, **k: _StubCalc())
    report_dir = tmp_path / "report"

    written = cmd_report(str(output_dir), ["s1_gemma2", "s2_gemma3"],
                         str(report_dir), retrieval_config=None,
                         with_bertscore=True, with_aspects=False)

    assert Path(written["per_row_metrics"]).is_file()
    assert Path(written["results_table"]).is_file()
    assert Path(written["results_md"]).is_file()
    assert Path(written["results_chart"]).is_file()

    table = pd.read_csv(written["results_table"])
    assert {"mean", "ci_low", "ci_high"} <= set(table.columns)
    assert "## Takeaways" in Path(written["results_md"]).read_text(
        encoding="utf-8")


def test_the_report_folds_in_the_retrieval_and_head_tables(output_dir, tmp_path,
                                                           monkeypatch):
    import src.minimal_run as module

    monkeypatch.setattr(module, "_metrics_calculator", lambda *a, **k: _StubCalc())
    retrieval = tmp_path / "retrieval_table.csv"
    pd.DataFrame([{"config": "m3_hybrid", "split": "all",
                   "metric": "top1_rougeL", "mean": 0.4}]).to_csv(retrieval,
                                                                  index=False)
    heads = tmp_path / "aspect_table.csv"
    pd.DataFrame([{"head": "intent", "split": "all", "f1_macro": 0.7,
                   "majority_f1_macro": 0.2}]).to_csv(heads, index=False)
    log = tmp_path / "trainer.log"
    log.write_text("step 50 loss 1.23\n", encoding="utf-8")

    written = cmd_report(str(output_dir), ["s2_gemma3"], str(tmp_path / "r"),
                         retrieval_config=None, with_bertscore=False,
                         with_aspects=False,
                         retrieval_table_path=str(retrieval),
                         aspect_table_path=str(heads),
                         trainer_log_path=str(log))

    markdown = Path(written["results_md"]).read_text(encoding="utf-8")
    assert "## Retrieval" in markdown and "m3_hybrid" in markdown
    assert "## Aspect heads" in markdown and "intent" in markdown
    assert "## RAFT training" in markdown and "step 50 loss 1.23" in markdown
