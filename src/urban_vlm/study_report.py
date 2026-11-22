"""Independent, offline E0–E4 evidence report. Never alters prompts or human labels.

Only observations actually present in the input files are reported. Human labels are
loaded from annotations.csv, not assumed from model prediction columns.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .data import CLASSES, EXPECTED_NAMES, file_sha256, read_annotations, write_csv
from .evaluation import evaluate_predictions

SUMMARY_FIELDS = ["method", "region", "prompt_bank", "n", "correct", "accuracy", "macro_f1", "recall_A", "recall_I", "recall_N"]
CASE_FIELDS = ["file", "human", "evidence", "baseline_full_generic", "baseline_roi_generic",
               "baseline_full_contextual", "baseline_roi_contextual", "e3_full", "e3_roi",
               "e4_full", "e4_roi", "e3_margin_presence_roi", "e3_margin_operation_roi",
               "e4_margin_operation_roi", "e3_to_e4_change_roi"]
PROMPT_FIELDS = ["file", "region", "hypothesis", "best_prompt_index", "best_prompt_text", "best_logit",
                 "mean_logit", "gap_best_minus_mean"]
STAGE1_FIELDS = ["region", "n", "correct", "accuracy", "human_commercial", "identified_commercial"]


def _rows(path: Path, columns: set[str]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        records = list(reader)
    if not records:
        raise ValueError(f"Empty input file: {path}")
    return records


def _index(rows: list[dict], columns: tuple[str, ...], description: str) -> dict[tuple, dict]:
    index = {}
    for row in rows:
        key = tuple(row[c] for c in columns)
        if key in index:
            raise ValueError(f"Duplicate {description}: {key}")
        index[key] = row
    return index


def _check_label(row: dict, annotations: dict, label_field: str, path_desc: str) -> None:
    name = row["file"]
    if name not in annotations or row[label_field] != annotations[name]["label"]:
        raise ValueError(f"Annotation conflict in {path_desc}: {name}")


def _stage_summary(method: str, rows: list[dict], annotations: dict) -> tuple[list[dict], dict]:
    for row in rows:
        _check_label(row, annotations, "label", method)
    result = evaluate_predictions(rows)
    return [{"method": method, **line} for line in result["summary"]], result["detailed"]


def _confusion_csv(target: Path, matrices: dict) -> None:
    records = []
    for key, detail in sorted(matrices.items()):
        method, region, bank = key
        for human in CLASSES:
            for predicted in CLASSES:
                records.append({"method": method, "region": region, "prompt_bank": bank,
                                "human": human, "predicted": predicted,
                                "count": detail["confusion"][human][predicted]})
    write_csv(target, records, ["method", "region", "prompt_bank", "human", "predicted", "count"])


def _e4_prompt_diagnostics(logit_path: Path) -> tuple[list[dict], dict]:
    rows = _rows(logit_path, {"file", "region", "stage", "hypothesis", "prompt_index", "prompt_text", "logit", "source"})
    groups = defaultdict(list)
    for row in rows:
        if row["stage"] != "operation" or row["source"] == "E3_cached_unchanged":
            continue
        groups[(row["file"], row["region"], row["hypothesis"])].append(row)
    output, mapped = [], {}
    for (name, region, hypothesis), values in sorted(groups.items()):
        if hypothesis not in ("active", "unverifiable") or len(values) != 5:
            raise ValueError(f"E4 operation bank incomplete: {name}/{region}/{hypothesis}")
        ids = {int(v["prompt_index"]) for v in values}
        if ids != {1, 2, 3, 4, 5}:
            raise ValueError(f"Missing E4 prompt indices: {name}/{region}/{hypothesis}")
        best = max(values, key=lambda z: float(z["logit"]))
        mean_logit = sum(float(v["logit"]) for v in values) / 5
        entry = {"file": name, "region": region, "hypothesis": hypothesis,
                 "best_prompt_index": int(best["prompt_index"]), "best_prompt_text": best["prompt_text"],
                 "best_logit": float(best["logit"]), "mean_logit": mean_logit,
                 "gap_best_minus_mean": float(best["logit"]) - mean_logit}
        output.append(entry)
        mapped[(name, region, hypothesis)] = entry
    if len(output) != 52:
        raise ValueError(f"E4 should have 52 operation hypothesis groups, got {len(output)}")
    return output, mapped


def _plot_figures(out: Path, summary: list[dict], confusion: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('For plots, install: pip install -e ".[report]"') from exc
    names = [f"{r['method']} / {r['region']} / {r['prompt_bank']}" for r in summary]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.48 * len(summary))))
    pos = list(range(len(names)))
    bars = ax.barh(pos, [float(x["accuracy"]) for x in summary])
    ax.set(yticks=pos, yticklabels=names, xlim=(0, 1), xlabel="Accuracy (development set)")
    ax.invert_yaxis()
    for rect, item in zip(bars, summary):
        ax.text(float(item['accuracy']) + .015, rect.get_y() + rect.get_height()/2,
                f"{item['correct']}/{item['n']}", va="center", fontsize=8)
    ax.set_xlim(0, 1.08)
    fig.tight_layout()
    img1 = out / "accuracy_comparison.png"
    fig.savefig(img1, dpi=160)
    plt.close(fig)

    selected = [("baseline", "roi", "generic"), ("E3", "roi", "hierarchical_evidence"),
                ("E4", "roi", "operation_e4")]
    saved = [img1.name]
    for key in selected:
        detail = confusion.get(key)
        if detail is None:
            continue
        matrix = [[detail["confusion"][label][pred] for pred in CLASSES] for label in CLASSES]
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.imshow(matrix, vmin=0, vmax=7)
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set(xlabel="Predicted", ylabel="Annotated", title=" / ".join(key))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=12,
                        color=("white" if matrix[i][j] >= 4 else "black"))
        fig.tight_layout()
        filename = f"confusion_{key[0].lower()}_roi.png"
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
        saved.append(filename)
    return saved


def build_study_report(
    baseline: str | Path,
    annotations: str | Path,
    output: str | Path,
    e3: str | Path | None = None,
    e4_paired: str | Path | None = None,
    e4_logits: str | Path | None = None,
    spatial_effects: str | Path | None = None,
    plots: bool = False,
) -> dict:
    """Build full 13-image tables and reports from E0–E4 saved predictions, no model inference.

    Each pair of provided E3/E4 files must cover both full and ROI for all 13 images.
    Runs are not assumed independent; this is a descriptive report of development data.
    """
    baseline, annotations, output = Path(baseline), Path(annotations), Path(output)
    data_inputs = {"baseline": baseline, "annotations": annotations}
    ann = read_annotations(annotations)
    if set(ann) != set(EXPECTED_NAMES):
        raise ValueError("Study report requires canonical 13 annotations")
    base = _rows(baseline, {"file", "region", "prompt_bank", "label", "predicted"})
    bidx = _index(base, ("file", "region", "prompt_bank"), "baseline")
    expected_base = {(name, region, bank) for name in EXPECTED_NAMES for region in ("full", "roi")
                     for bank in ("generic", "contextual")}
    if set(bidx) != expected_base:
        raise ValueError(f"Expected exactly 52 baseline decisions; missing={len(expected_base-set(bidx))}, extra={len(set(bidx)-expected_base)}")
    for row in base:
        _check_label(row, ann, "label", "baseline")
        if row["predicted"] not in CLASSES:
            raise ValueError("Unknown class in baseline")
    summaries, all_confusions = [], {}
    values, matrices = _stage_summary("baseline", base, ann)
    summaries.extend(values)
    all_confusions.update({("baseline", *key.split("|", 1)): val for key, val in matrices.items()})
    e3idx = {}
    stage1_summary = []
    if e3:
        e3 = Path(e3); data_inputs["E3"] = e3
        e3rows = _rows(e3, {"file", "region", "label", "predicted", "presence_predicted", "operation_predicted", "margin_presence", "margin_operation"})
        e3idx = _index(e3rows, ("file", "region"), "E3")
        expected = {(name, region) for name in EXPECTED_NAMES for region in ("full", "roi")}
        if set(e3idx) != expected:
            raise ValueError("E3 should have exactly 26 decisions (13 × 2)")
        e3as = []
        for region in ("full", "roi"):
            selected = [r for r in e3rows if r["region"] == region]
            n = len(selected)
            correct = sum((r["presence_predicted"] == "commercial") ==
                          (ann[r["file"]]["label"] in ("A", "I")) for r in selected)
            stage1_summary.append({"region": region, "n": n, "correct": correct,
                                   "accuracy": correct / n,
                                   "human_commercial": sum(ann[r["file"]]["label"] in ("A", "I") for r in selected),
                                   "identified_commercial": sum(r["presence_predicted"] == "commercial" for r in selected)})
        for row in e3rows:
            _check_label(row, ann, "label", "E3")
            if row["predicted"] not in CLASSES:
                raise ValueError("Unknown E3 class")
            e3as.append({**row, "prompt_bank": "hierarchical_evidence"})
        values, matrices = _stage_summary("E3", e3as, ann)
        summaries.extend(values)
        all_confusions.update({("E3", *key.split("|", 1)): val for key, val in matrices.items()})
    if e4_paired and not e3:
        raise ValueError("E4 paired file requires an E3 predictions file from the same run")
    e4idx = {}
    if e4_paired:
        e4_paired = Path(e4_paired); data_inputs["E4_pair"] = e4_paired
        e4rows = _rows(e4_paired, {"file", "region", "human", "e3_predicted", "e4_predicted", "margin_presence", "e3_margin_operation", "e4_margin_operation"})
        e4idx = _index(e4rows, ("file", "region"), "E4")
        if set(e4idx) != set(e3idx):
            raise ValueError("E4 comparisons and E3 predictions differ in image/region coverage")
        e4as = []
        for key, row in e4idx.items():
            _check_label(row, ann, "human", "E4")
            if row["e3_predicted"] != e3idx[key]["predicted"]:
                raise ValueError(f"E4 is not paired to this E3 prediction at {key}")
            if row["e4_predicted"] not in CLASSES:
                raise ValueError("Unknown E4 predicted class")
            if abs(float(row["margin_presence"]) - float(e3idx[key]["margin_presence"])) > 1e-4:
                raise ValueError(f"E3 presence changed in E4: {key}")
            e4as.append({"file":row["file"],"region":row["region"],"prompt_bank":"operation_e4",
                         "label":ann[row["file"]]["label"],"predicted":row["e4_predicted"]})
        values, matrices = _stage_summary("E4", e4as, ann)
        summaries.extend(values)
        all_confusions.update({("E4", *key.split("|", 1)): val for key, val in matrices.items()})
    prompt_entries, prompt_index = [], {}
    if e4_logits:
        if not e4idx:
            raise ValueError("E4 logits were supplied without E4 paired results")
        data_inputs["E4_logits"] = Path(e4_logits)
        prompt_entries, prompt_index = _e4_prompt_diagnostics(Path(e4_logits))
    spatial_rows = []
    if spatial_effects:
        spatial_effects = Path(spatial_effects)
        data_inputs["spatial_effects"] = spatial_effects
        spatial_rows = _rows(spatial_effects, {"file", "prompt_bank", "intervention", "predicted", "label_changed", "delta_A_minus_N"})
        if not set(x["file"] for x in spatial_rows) <= set(EXPECTED_NAMES):
            raise ValueError("Unknown file in spatial effects")

    # Refuse to overwrite a prior report (same protection as experiment runs).
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics_all.csv", summaries, SUMMARY_FIELDS)
    if stage1_summary:
        write_csv(output / "stage1_presence_metrics.csv", stage1_summary, STAGE1_FIELDS)
    _confusion_csv(output / "confusion_all.csv", all_confusions)
    case_rows = []
    for name in EXPECTED_NAMES:
        r = {"file": name, "human": ann[name]["label"], "evidence": ann[name]["evidence"]}
        for region in ("full", "roi"):
            for bank in ("generic", "contextual"):
                r[f"baseline_{region}_{bank}"] = bidx[(name, region, bank)]["predicted"]
            r[f"e3_{region}"] = e3idx[(name, region)]["predicted"] if e3idx else ""
            r[f"e4_{region}"] = e4idx[(name, region)]["e4_predicted"] if e4idx else ""
        r["e3_margin_presence_roi"] = e3idx[(name,"roi")]["margin_presence"] if e3idx else ""
        r["e3_margin_operation_roi"] = e3idx[(name,"roi")]["margin_operation"] if e3idx else ""
        r["e4_margin_operation_roi"] = e4idx[(name,"roi")]["e4_margin_operation"] if e4idx else ""
        if e4idx:
            old, new = r["e3_roi"], r["e4_roi"]
            if old == new: status = "unchanged_correct" if new == r["human"] else "unchanged_wrong"
            elif new == r["human"]: status = "corrected"
            elif old == r["human"]: status = "new_error"
            else: status = "changed_wrong"
            r["e3_to_e4_change_roi"] = status
        else:
            r["e3_to_e4_change_roi"] = ""
        case_rows.append(r)
    write_csv(output / "cases_13.csv", case_rows, CASE_FIELDS)
    if prompt_entries:
        write_csv(output / "operation_prompt_diagnostics.csv", prompt_entries, PROMPT_FIELDS)
    if spatial_rows:
        write_csv(output / "spatial_evidence.csv", spatial_rows,
                  list(spatial_rows[0]))
    sections = [
        "# UrbanGroundFloor — consolidated scientific report (E0–E4)", "",
        "**Status:** exploratory development-set analysis; not a held-out evaluation.",
        "**Question:** Is there visible evidence of active commerce at ground level on the target frontage?", "",
        "## Definitions and annotation protocol", "",
        "- **A:** observable business prepared/open for commerce (transaction not required).",
        "- **I:** commercial cues exist but operating state cannot be verified from this static image.",
        "- **N:** no visible commerce on the **target** frontage; cannot exclude hidden commerce.",
        "- For mixed-use buildings, inspect only the targeted ground floor; residences above do not imply N.",
        "- F02 and F13 are distinct photographs that may depict the same establishment; not independent shops.",
        "- Labels were confirmed after iterative model inspection, not via independent blinded annotation.", "",
        "## Quantitative results", "",
        "| Method | Region | Prompts | Correct | Accuracy | Macro-F1 | Recall A | Recall I | Recall N |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        sections.append(f"| {row['method']} | {row['region']} | {row['prompt_bank']} | {row['correct']}/{row['n']} | "
                        f"{row['accuracy']:.3f} | {row['macro_f1']:.3f} | {row['recall_A']:.3f} | {row['recall_I']:.3f} | {row['recall_N']:.3f} |")
    if stage1_summary:
        sections += ["", "## E3 stage 1 — detection of commercial premises", "",
                     "| Region | Correct | Accuracy | Human commercial images | Predicted commercial images |",
                     "|---|---:|---:|---:|---:|"]
        for s1 in stage1_summary:
            sections.append(f"| {s1['region']} | {s1['correct']}/{s1['n']} | {s1['accuracy']:.3f} | "
                            f"{s1['human_commercial']} | {s1['identified_commercial']} |")
        sections.append("")
    sections += ["", "## Per-photograph decisions (ROI)", "",
                 "| Photo | Human | Baseline generic | Baseline contextual | E3 | E4 | E3→E4 |",
                 "|---|:---:|:---:|:---:|:---:|:---:|---|"]
    for r in case_rows:
        sections.append(f"| {r['file']} | {r['human']} | {r['baseline_roi_generic']} | {r['baseline_roi_contextual']} | "
                        f"{r['e3_roi'] or '—'} | {r['e4_roi'] or '—'} | {r['e3_to_e4_change_roi'] or '—'} |")
    sections += ["", "## Case-by-case diagnostic record", "",
                 "**Interpretations below are hypotheses, not causal explanations of neural attention.**", ""]
    for r in case_rows:
        sections += [f"### {r['file']} — annotated {r['human']}", "",
                     f"**Human visual evidence:** {r['evidence']}", "",
                     f"- Baseline generic: full **{r['baseline_full_generic']}**, ROI **{r['baseline_roi_generic']}**.",
                     f"- Baseline contextual: full **{r['baseline_full_contextual']}**, ROI **{r['baseline_roi_contextual']}**."]
        if e3idx:
            sections.append(f"- E3 ROI: **{r['e3_roi']}**, presence margin {float(r['e3_margin_presence_roi']):+.3f}, "
                            f"operation margin {float(r['e3_margin_operation_roi']):+.3f}.")
        if e4idx:
            sections.append(f"- E4 ROI: **{r['e4_roi']}**, operation margin {float(r['e4_margin_operation_roi']):+.3f}; "
                            f"paired status: **{r['e3_to_e4_change_roi']}**.")
        if prompt_index:
            for hypothesis in ("active", "unverifiable"):
                p = prompt_index.get((r["file"], "roi", hypothesis))
                if p:
                    sections.append(f"- Highest E4 {hypothesis} prompt: #{p['best_prompt_index']} "
                                    f"(logit {p['best_logit']:+.3f}); mean {p['mean_logit']:+.3f}. "
                                    f"Text: *{p['best_prompt_text']}*")
        if r["human"] == "I" and r["baseline_roi_generic"] == "A":
            mechanism = "Presence of a commercial establishment may be conflated with direct evidence of operation."
        elif r["human"] == "I" and r["baseline_roi_generic"] == "N":
            mechanism = "Business signage or access may be too subtle for the current descriptions/visual scale."
        elif r["human"] == "A" and r["e4_roi"] == "I":
            mechanism = "Operation wording may favor uncertainty despite an open/prepared business; inspect individual prompts."
        elif r["baseline_full_generic"] != r["baseline_roi_generic"]:
            mechanism = "Framing changes decision; crop simultaneously removes surroundings and increases relative target scale."
        else:
            mechanism = "No demonstrated classification error in this condition; compare text similarities if needed."
        sections += [f"- **Mechanism hypothesis:** {mechanism}", ""]
    if spatial_rows:
        sections += ["## Spatial intervention evidence", "",
                     "ROI cropping changes context **and** relative object scale. Blur interventions are out-of-distribution transformations.", "",
                     "| Intervention | Bank | Images | Changed class vs original |",
                     "|---|---|---:|---:|"]
        counts = defaultdict(lambda: [0, 0])
        for row in spatial_rows:
            key=(row["intervention"], row["prompt_bank"])
            counts[key][0] += 1
            counts[key][1] += int(row["label_changed"])
        for (name,bank),(n, changed) in sorted(counts.items()):
            sections.append(f"| {name} | {bank} | {n} | {changed} |")
        sections.append("")
    sections += ["## Limits and interpretation", "",
        "- All 13 images were used during prompt/ROI development; accuracy and F1 are descriptive, not independent generalization estimates.",
        "- Only one human N case; any N recall estimate is extremely unstable.",
        "- An indeterminate image does **not** establish that a business is closed.",
        "- SigLIP 2 text-image logits are uncalibrated similarities, not probabilities or proofs of observed transactions.",
        "- F02 and F13 likely refer to the same business. Do not infer 13 independent establishments.",
        "- Changes after rewording the stage-2 prompts (E4) are exploratory and may overfit this study set.",
        "- No regional pretraining bias is causally established without an independent cross-region comparison.", "",
        "## Reproducibility", "",
        "See `study_inputs.json` for the SHA256 hashes of every file used in this report.",
        "For photo-based re-inference, use the original 640×640 images, frozen ROI boxes, original texts and checkpoint revision.", "",
    ]
    (output / "study_report.md").write_text("\n".join(sections), encoding="utf-8")
    metadata = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "analysis_type": "development set descriptive report; no tuning/held-out metrics",
                "n_photographs": 13,
                "file_sha256": {name: file_sha256(path) for name,path in data_inputs.items()},
                "data_files": {name: path.name for name,path in data_inputs.items()},
                "reported_methods": [f"{r['method']}:{r['region']}:{r['prompt_bank']}" for r in summaries]}
    figures = _plot_figures(output, summaries, all_confusions) if plots else []
    metadata["figures"] = figures
    (output / "study_inputs.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "n_photographs": 13, "comparisons": len(summaries),
            "figures": figures, "inputs": list(data_inputs)}
