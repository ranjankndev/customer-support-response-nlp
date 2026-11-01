"""Step 1 (plan §4): config merging and the new GenerationConfig fields."""

from pathlib import Path

import pytest
import yaml

from src.utils import deep_merge, load_yaml_with_overrides

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
GENERATION_CONFIG = CONFIG_DIR / "generation_config.yaml"
KAGGLE_OVERRIDES = CONFIG_DIR / "kaggle.yaml"


# ── deep_merge ───────────────────────────────────────────────────────────────

def test_deep_merge_recurses_and_leaves_the_base_untouched():
    base = {"paths": {"data_dir": "data", "work_dir": "."}, "top_k": 3}
    override = {"paths": {"data_dir": "/kaggle/input"}, "device": "cuda"}

    merged = deep_merge(base, override)

    assert merged == {
        "paths": {"data_dir": "/kaggle/input", "work_dir": "."},
        "top_k": 3,
        "device": "cuda",
    }
    assert base["paths"]["data_dir"] == "data"


def test_deep_merge_replaces_lists_wholesale():
    merged = deep_merge({"col_aspects": ["a", "b", "c"]}, {"col_aspects": ["x"]})

    assert merged["col_aspects"] == ["x"]


# ── load_yaml_with_overrides ─────────────────────────────────────────────────

@pytest.mark.parametrize("config_name", [
    "generation_config.yaml",
    "retrieval_config.yaml",
    "inference_retrieval_config.yaml",
    "config_preprocess.yaml",
])
def test_existing_configs_still_load_without_overrides(config_name):
    loaded = load_yaml_with_overrides(CONFIG_DIR / config_name)

    assert isinstance(loaded, dict) and loaded


def test_kaggle_overrides_change_the_generation_output_dir():
    base = load_yaml_with_overrides(GENERATION_CONFIG)
    merged = load_yaml_with_overrides(GENERATION_CONFIG, KAGGLE_OVERRIDES,
                                      section="generation")

    assert base["output_dir"] == "output/generation"
    assert merged["output_dir"] == "/kaggle/working/Code/output/generation"
    assert merged["backend"] == "unsloth"
    assert merged["generation_batch_size"] == 8
    assert merged["skip_metrics"] is True
    # untouched keys survive the merge
    assert merged["top_k"] == base["top_k"]


def test_overrides_section_is_scoped():
    merged = load_yaml_with_overrides(CONFIG_DIR / "retrieval_config.yaml",
                                      KAGGLE_OVERRIDES, section="retrieval")

    assert merged["batch_size"] == 32
    # the generation section must not leak into the retrieval config
    assert "skip_metrics" not in merged


def test_missing_overrides_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_yaml_with_overrides(GENERATION_CONFIG, CONFIG_DIR / "nope.yaml")


def test_empty_overrides_argument_is_a_no_op():
    assert (load_yaml_with_overrides(GENERATION_CONFIG, "")
            == load_yaml_with_overrides(GENERATION_CONFIG))


def test_non_mapping_section_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("generation: [1, 2]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_yaml_with_overrides(GENERATION_CONFIG, bad, section="generation")


# ── kaggle.yaml contents (plan §4) ───────────────────────────────────────────

def test_kaggle_yaml_has_every_section_the_plan_lists():
    data = yaml.safe_load(KAGGLE_OVERRIDES.read_text(encoding="utf-8"))

    assert set(data) == {"paths", "generation", "retrieval", "raft"}
    assert data["paths"]["data_dir"] == "/kaggle/input/aspectforge-data"
    assert data["paths"]["work_dir"] == "/kaggle/working/Code"
    assert data["raft"] == {"n_train": 3000, "max_seq_len": 2048,
                            "batch": 2, "grad_accum": 8}


def test_base_configs_keep_relative_paths_so_local_runs_work():
    for name in ("generation_config.yaml", "retrieval_config.yaml"):
        text = (CONFIG_DIR / name).read_text(encoding="utf-8")
        assert "/kaggle/" not in text, f"{name} must stay environment-agnostic"


# ── GenerationConfig ─────────────────────────────────────────────────────────

def test_generation_config_exposes_the_new_fields():
    from src.generation.response_generator_v2 import GenerationConfig

    fields = GenerationConfig.__dataclass_fields__
    for name in ("backend", "attn_implementation", "adapter_path",
                 "repetition_penalty", "no_repeat_ngram_size", "max_input_tokens",
                 "ctx_doc_max_chars", "reply_language_instruction",
                 "crag_threshold", "sample_csv", "skip_metrics"):
        assert name in fields, f"GenerationConfig is missing {name}"


def test_generation_config_defaults_match_the_plan():
    from src.generation.response_generator_v2 import GenerationConfig

    cfg = GenerationConfig()

    assert cfg.backend == "hf"
    assert cfg.attn_implementation == "sdpa"
    assert cfg.adapter_path == ""
    assert cfg.repetition_penalty == 1.0
    assert cfg.no_repeat_ngram_size == 0
    assert cfg.max_input_tokens == 3072
    assert cfg.ctx_doc_max_chars == 1200
    assert cfg.reply_language_instruction is True
    assert cfg.crag_threshold == 0.30
    assert cfg.sample_csv == ""
    assert cfg.skip_metrics is False


def test_load_generation_config_applies_the_kaggle_overrides():
    from src.generation.response_generator_v2 import load_generation_config

    base = load_generation_config(str(GENERATION_CONFIG))
    kaggle = load_generation_config(str(GENERATION_CONFIG), str(KAGGLE_OVERRIDES))

    assert base.output_dir == "output/generation"
    assert base.backend == "hf"
    assert kaggle.output_dir == "/kaggle/working/Code/output/generation"
    assert kaggle.backend == "unsloth"
    assert kaggle.max_input_tokens == 2560
    assert kaggle.skip_metrics is True


def test_sample_csv_overrides_the_input_argument():
    from src.generation.response_generator_v2 import (GenerationConfig,
                                                      resolve_data_path)

    class Args:
        input = "data/processed/cs-support-multi-test.csv"
        aspects = None

    cfg = GenerationConfig()
    assert resolve_data_path(Args(), cfg) == Args.input

    cfg.sample_csv = "output/minimal_run/sample_500.csv"
    assert resolve_data_path(Args(), cfg) == "output/minimal_run/sample_500.csv"


# ── other entry points accept --overrides ────────────────────────────────────

def test_indexer_load_config_accepts_overrides():
    import src.retrieval.indexer as indexer

    base = indexer.load_config(str(CONFIG_DIR / "retrieval_config.yaml"))
    kaggle = indexer.load_config(str(CONFIG_DIR / "retrieval_config.yaml"),
                                 str(KAGGLE_OVERRIDES))

    assert base.batch_size == 64
    assert kaggle.batch_size == 32
    assert kaggle.device == "cuda"
    assert kaggle.index_path == base.index_path


def test_aspect_extractor_apply_overrides_is_a_no_op_without_a_section():
    from src.aspect.aspect_extractor import apply_overrides

    cfg = {"input": "data/processed/cs-support-multi-train.csv"}

    assert apply_overrides(cfg, None) == cfg
    # config/kaggle.yaml has no `aspect` section, so the config is unchanged
    assert apply_overrides(cfg, str(KAGGLE_OVERRIDES)) == cfg
    with pytest.raises(FileNotFoundError):
        apply_overrides(cfg, str(CONFIG_DIR / "nope.yaml"))


def test_aspect_extractor_apply_overrides_merges_its_section(tmp_path):
    from src.aspect.aspect_extractor import apply_overrides

    overrides = tmp_path / "over.yaml"
    overrides.write_text("aspect:\n  batch: 16\n", encoding="utf-8")

    merged = apply_overrides({"batch": 4, "input": "x.csv"}, str(overrides))

    assert merged == {"batch": 16, "input": "x.csv"}
