"""CPU-only tests of E4: E3's first-stage logits remain exactly frozen."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from urban_vlm.cli import parser
from urban_vlm.data import load_config
from urban_vlm.hierarchical import run_hierarchical
from urban_vlm.operation_e4 import (
    load_operation_prompts, run_operation_e4, validate_e4_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
E3_BANK = ROOT / "configs" / "prompts_hierarchical.json"
E4_BANK = ROOT / "configs" / "prompts_operation_e4.json"


class E3Fake:
    device = "fake-cpu"
    commit = "fixed-hypothetical-commit"

    def predict_logits(self, image, texts):
        assert len(texts) == 20
        # commercial=4, noncommercial=0; active=3, unverifiable=1
        return [4.0] * 5 + [0.0] * 5 + [3.0] * 5 + [1.0] * 5


class E4Fake:
    device = "fake-cpu"
    commit = "fixed-hypothetical-commit"

    def predict_logits(self, image, texts):
        assert len(texts) == 10, "E4 must never reinfer E3 presence prompts"
        return [1.0] * 5 + [3.0] * 5


def prepare_data(tmp_path):
    cfg = load_config(ROOT / "configs" / "experiment.yaml")
    images = tmp_path / "images"
    images.mkdir()
    for k in range(1, 14):
        Image.new("RGB", (640, 640), (110, 132, 141)).save(images / f"fachada_{k:02d}.jpg")
    cfg["images_dir"] = str(images)
    return cfg, images


def get_e3_fixture(tmp_path):
    cfg, images = prepare_data(tmp_path)
    e3dir = tmp_path / "e3"
    run_hierarchical(cfg, E3_BANK, e3dir, region="both", backend=E3Fake())
    return cfg, images, e3dir


def records(filename):
    with Path(filename).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_bank_has_exact_10_texts():
    bank = load_operation_prompts(E4_BANK)
    assert len(bank["active"]) == 5 and len(bank["unverifiable"]) == 5
    with pytest.raises(ValueError):
        load_operation_prompts(ROOT / "configs" / "prompts_hierarchical.json")


def test_e4_freezes_stage1_and_reports_all_outcomes(tmp_path):
    cfg, _, e3dir = get_e3_fixture(tmp_path)
    preflight = validate_e4_inputs(cfg, e3dir, E3_BANK, E4_BANK, region="both")
    assert preflight["images"] == 13 and preflight["e3_records_available"] == 26
    out = tmp_path / "e4"
    result = run_operation_e4(cfg, e3dir, E3_BANK, E4_BANK, out,
                              region="both", backend=E4Fake())
    assert (result["predictions"], result["prompt_logits"], result["new_stage2_logits"]) == (26, 520, 260)
    new = records(out / "operation_e4_predictions.csv")
    prior = records(e3dir / "hierarchical_predictions.csv")
    old_index = {(r["file"], r["region"]): r for r in prior}
    assert set(row["predicted"] for row in new) == {"I"}
    assert all(row["presence_predicted"] == "commercial" for row in new)
    for row in new:
        old = old_index[row["file"], row["region"]]
        assert row["margin_presence"] == old["margin_presence"]
        assert row["score_presence_commercial"] == old["score_presence_commercial"]
        assert row["e3_predicted"] == old["predicted"]
    logits = records(out / "operation_e4_logits.csv")
    assert sum(row["source"] == "E3_cached_unchanged" for row in logits) == 260
    assert sum(row["source"] == "E4_new_inference" for row in logits) == 260
    paired = records(out / "paired_e3_e4.csv")
    assert len(paired) == 26
    assert (out / "metrics.json").is_file()
    assert (out / "E4_report.md").is_file()
    with pytest.raises(FileExistsError):
        run_operation_e4(cfg, e3dir, E3_BANK, E4_BANK, out, backend=E4Fake())


def test_e4_does_not_recompute_first_stage_in_smoke(tmp_path):
    cfg, _, e3dir = get_e3_fixture(tmp_path)
    out = tmp_path / "smoke"
    result = run_operation_e4(cfg, e3dir, E3_BANK, E4_BANK, out,
                              region="roi", limit=2, backend=E4Fake())
    assert (result["images"], result["predictions"], result["prompt_logits"]) == (2, 2, 40)


def test_e4_rejects_changed_image(tmp_path):
    cfg, images, e3dir = get_e3_fixture(tmp_path)
    Image.new("RGB", (640, 640), (90, 90, 90)).save(images / "fachada_03.jpg")
    with pytest.raises(ValueError, match="does not match"):
        validate_e4_inputs(cfg, e3dir, E3_BANK, E4_BANK, region="roi")


def test_e4_rejects_changed_prompts_and_wrong_commit(tmp_path):
    cfg, _, e3dir = get_e3_fixture(tmp_path)
    other = tmp_path / "edited_bank.json"
    data = json.loads(E3_BANK.read_text(encoding="utf-8"))
    data["presence"]["commercial"][0] += " modificada"
    other.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="E3 prompt bank changed"):
        validate_e4_inputs(cfg, e3dir, other, E4_BANK)
    fake = E4Fake()
    fake.commit = "a-different-model-revision"
    with pytest.raises(ValueError, match="Model revision mismatch"):
        run_operation_e4(cfg, e3dir, E3_BANK, E4_BANK, tmp_path / "rejected", backend=fake)


def test_cli_registers_e4_and_dryrun():
    a = parser().parse_args(["operation-e4", "--e3-dir", "runs/e3", "--dry-run"])
    assert a.command == "operation-e4" and a.e3_dir == "runs/e3"
    assert a.region == "roi" and a.aggregation == "mean"
