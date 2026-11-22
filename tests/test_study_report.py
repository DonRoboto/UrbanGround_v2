"""Offline regression tests against archived E0/E3/E4 developer-set outputs."""
from pathlib import Path
import csv
import json
import pytest

from urban_vlm.study_report import build_study_report
from urban_vlm.cli import parser

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "reference" / "historical_2026_11_21"
BASE = HIST / "baseline" / "predictions.csv"
E3 = HIST / "e3" / "hierarchical_predictions.csv"
E4 = HIST / "e4" / "paired_e3_e4.csv"
LOGITS = HIST / "e4" / "operation_e4_logits.csv"
ANN = ROOT / "data" / "annotations.csv"


def csv_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_historical_report_reconstructs_results(tmp_path):
    out = tmp_path / "report"
    result = build_study_report(BASE, ANN, out, e3=E3, e4_paired=E4, e4_logits=LOGITS)
    assert result["n_photographs"] == 13
    assert result["comparisons"] == 8
    metrics = {(x["method"], x["region"], x["prompt_bank"]): x for x in csv_rows(out / "metrics_all.csv")}
    assert int(metrics[("baseline", "roi", "generic")]["correct"]) == 10
    assert int(metrics[("E3", "roi", "hierarchical_evidence")]["correct"]) == 9
    assert int(metrics[("E4", "roi", "operation_e4")]["correct"]) == 9
    assert float(metrics[("E3", "roi", "hierarchical_evidence")]["macro_f1"]) > float(metrics[("baseline", "roi", "generic")]["macro_f1"])
    assert len(csv_rows(out / "operation_prompt_diagnostics.csv")) == 52
    stage1 = {x["region"]: x for x in csv_rows(out / "stage1_presence_metrics.csv")}
    assert int(stage1["roi"]["correct"]) == 13
    cases = {x["file"]: x for x in csv_rows(out / "cases_13.csv")}
    assert len(cases) == 13
    assert cases["fachada_07.jpg"]["e3_to_e4_change_roi"] == "corrected"
    assert cases["fachada_08.jpg"]["e3_to_e4_change_roi"] == "new_error"
    assert cases["fachada_11.jpg"]["human"] == "N"
    assert (out / "study_report.md").is_file()
    provenance = json.loads((out / "study_inputs.json").read_text())
    assert len(provenance["file_sha256"]) == 5


def test_report_from_baseline_only(tmp_path):
    report = tmp_path / "baseline"
    result = build_study_report(BASE, ANN, report)
    assert result["comparisons"] == 4
    assert not (report / "stage1_presence_metrics.csv").exists()
    assert len(csv_rows(report / "cases_13.csv")) == 13


def test_pairing_rejects_changed_presence(tmp_path):
    records = csv_rows(E4)
    records[0]["margin_presence"] = str(float(records[0]["margin_presence"]) + 1)
    modified = tmp_path / "tampered_pair.csv"
    with modified.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError, match="presence changed"):
        build_study_report(BASE, ANN, tmp_path / "out", e3=E3, e4_paired=modified)
    assert not (tmp_path / "out").exists()


def test_e4_without_e3_rejected(tmp_path):
    with pytest.raises(ValueError, match="requires an E3"):
        build_study_report(BASE, ANN, tmp_path / "out", e4_paired=E4)


def test_report_cannot_overwrite(tmp_path):
    out = tmp_path / "out"
    build_study_report(BASE, ANN, out)
    with pytest.raises(FileExistsError):
        build_study_report(BASE, ANN, out)


def test_cli_supports_report():
    args = parser().parse_args(["report", "--baseline", str(BASE), "--output", "somewhere"])
    assert args.command == "report"
    assert args.plots is False


def test_optional_report_figures(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "report_plots"
    res = build_study_report(BASE, ANN, out, e3=E3, e4_paired=E4, plots=True)
    assert (out / "accuracy_comparison.png").is_file()
    assert len(res["figures"]) == 4
