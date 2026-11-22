"""Per-case interpretive report grounded in visual annotations, logits and interventions.

Descriptions are hypotheses compatible with observed scores, not proofs of causal
pixel attribution or assertions about shop opening hours.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from .data import CLASSES, read_annotations, write_csv
from .evaluation import load_predictions

FIELDS = [
    "file", "human_label", "evidence", "prompt_bank", "full_predicted", "roi_predicted",
    "full_correct", "roi_correct", "full_runnerup_margin", "roi_runnerup_margin",
    "full_score_A", "full_score_I", "full_score_N",
    "roi_score_A", "roi_score_I", "roi_score_N", "delta_roi_score_A",
    "delta_roi_score_I", "delta_roi_score_N", "full_top_prompt_for_human",
    "roi_top_prompt_for_human", "roi_top_prompt_predicted",
    "blur_outside_roi_prediction", "blur_inside_roi_prediction", "blur_upper_prediction",
    "mechanism_hypothesis_not_proof", "hierarchical_roi_prediction",
]


def csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def mechanism_hypothesis(human: str, full: str, roi: str) -> str:
    parts = []
    if full != human and roi == human:
        parts.append("The ROI corrects the full-scene mismatch; plausible context/scale sensitivity")
    if human == "I" and roi == "A":
        parts.append("Commercial-premises cues may be mistaken for evidence of current operation")
    elif human == "I" and roi == "N":
        parts.append("Small/occluded business cues or an unsuitable uncertainty prompt may lead to N")
    elif human == "A" and roi == "I":
        parts.append("Operating cues may be outweighed by generic uncertainty wording or prompt averaging")
    elif human == "A" and roi == "N":
        parts.append("Positive street-level cues may remain too weak for whole-image similarity")
    elif human == "N" and roi != "N":
        parts.append("Potential spillover from adjacent fronts or overly generic commercial evidence")
    if full != roi:
        parts.append("Prediction changes with spatial framing; not causal pixel attribution")
    if not parts:
        parts.append("No mismatch in the ROI condition; inspect class-specific logits if needed")
    return "; ".join(parts) + "."


def build_case_analysis(
    predictions_path: str | Path, logits_path: str | Path,
    annotations_path: str | Path, out_dir: str | Path,
    interventions_path: str | Path | None = None,
    hierarchical_path: str | Path | None = None,
) -> dict:
    prediction = load_predictions(predictions_path)
    anns = read_annotations(annotations_path)
    p_index = {(r["file"],r["region"],r["prompt_bank"]):r for r in prediction}
    logit_rows = csv_rows(logits_path)
    logit_index = defaultdict(list)
    for row in logit_rows:
        if not {"file","region","prompt_bank","class","prompt_index","logit"}.issubset(row):
            raise ValueError("Unexpected prompt_logits schema")
        logit_index[(row["file"],row["region"],row["prompt_bank"],row["class"])].append(row)
    interventions = {}
    if interventions_path:
        for r in load_predictions(interventions_path):
            key=(r["file"],r["region"],r["prompt_bank"])
            if key in interventions:
                raise ValueError(f"Duplicate intervention {key}")
            interventions[key]=r["predicted"]
    hierarchical = {}
    if hierarchical_path:
        for r in load_predictions(hierarchical_path):
            hierarchical[(r["file"],r["region"])]=r["predicted"]
    rows = []

    def best_prompt(name, region, bank, label):
        candidates=logit_index.get((name,region,bank,label), [])
        if not candidates:
            return "Not available"
        hit=max(candidates,key=lambda x:float(x["logit"]))
        return f'{hit.get("prompt_text", "")} [logit={float(hit["logit"]):.3f}]'

    banks=sorted({p["prompt_bank"] for p in prediction})
    for name, ann in sorted(anns.items()):
        for bank in banks:
            key_full=(name,"full",bank)
            key_roi=(name,"roi",bank)
            if key_full not in p_index or key_roi not in p_index:
                continue
            full, roi=p_index[key_full],p_index[key_roi]
            def number(row, field):
                try: return float(row[field])
                except (ValueError, KeyError, TypeError): return None
            row={
                "file":name, "human_label":ann["label"],"evidence":ann["evidence"],
                "prompt_bank":bank,"full_predicted":full["predicted"],"roi_predicted":roi["predicted"],
                "full_correct":int(full["predicted"]==ann["label"]),
                "roi_correct":int(roi["predicted"]==ann["label"]),
                "full_runnerup_margin":full.get("runner_up_margin", ""),
                "roi_runnerup_margin":roi.get("runner_up_margin", ""),
                "full_top_prompt_for_human":best_prompt(name,"full",bank,ann["label"]),
                "roi_top_prompt_for_human":best_prompt(name,"roi",bank,ann["label"]),
                "roi_top_prompt_predicted":best_prompt(name,"roi",bank,roi["predicted"]),
                "mechanism_hypothesis_not_proof":mechanism_hypothesis(ann["label"],full["predicted"],roi["predicted"]),
                "hierarchical_roi_prediction":hierarchical.get((name,"roi"),""),
            }
            for label in CLASSES:
                fvalue=number(full,f"score_{label}")
                rvalue=number(roi,f"score_{label}")
                row[f"full_score_{label}"]=fvalue if fvalue is not None else ""
                row[f"roi_score_{label}"]=rvalue if rvalue is not None else ""
                row[f"delta_roi_score_{label}"]=(rvalue-fvalue if rvalue is not None and fvalue is not None else "")
            for variant in ("blur_outside_roi", "blur_inside_roi", "blur_upper"):
                row[f"{variant}_prediction"]=interventions.get((name,variant,bank),"")
            rows.append(row)
    if not rows:
        raise ValueError("No paired full/ROI predictions found")
    target=Path(out_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite analysis folder: {target}")
    target.mkdir(parents=True,exist_ok=True)
    write_csv(target/"per_image_analysis.csv",rows,FIELDS)
    sections=[
        "# Ground-floor commerce: transparent per-case diagnosis", "",
        "**Important:** qualitative explanations below are hypotheses supported by annotations and score changes, not causal evidence.",
        "The 13 images were inspected during ROI/prompt design; there is no independent test set.",
        "Labels: A=visible evidence of activity; I=operation unverifiable; N=no commerce visible on TARGET frontage.",
        "F02 and F13 likely depict the same establishment.", "",
    ]
    for name in sorted({r["file"] for r in rows}):
        case=[r for r in rows if r["file"]==name]
        sections += [f"## {name} — human label {case[0]['human_label']}", "", f"**Visual evidence:** {case[0]['evidence']}", ""]
        for r in case:
            sections += [
                f"### {r['prompt_bank']}: full {r['full_predicted']} → ROI {r['roi_predicted']}",
                f"- **Hypothesis:** {r['mechanism_hypothesis_not_proof']}",
                f"- **Best ROI prompt for human label:** {r['roi_top_prompt_for_human']}",
                f"- **Best ROI prompt for predicted class:** {r['roi_top_prompt_predicted']}",
            ]
            if interventions_path:
                sections.append(
                    f"- **Spatial variants:** blur outside={r['blur_outside_roi_prediction'] or 'unavailable'}, "
                    f"blur ROI={r['blur_inside_roi_prediction'] or 'unavailable'}, "
                    f"blur upper={r['blur_upper_prediction'] or 'unavailable'}."
                )
            if hierarchical_path:
                sections.append(f"- **Two-stage exploratory result (ROI):** {r['hierarchical_roi_prediction'] or 'unavailable'}.")
            sections.append("")
    sections += ["## Interpretation restrictions", "",
                 "Differences in logits are uncalibrated compatibility scores, not probabilities or verified opening hours.",
                 "ROI cropping changes relative scale and context simultaneously; blurring is an OOD intervention.",
                 "Any better scores on these 13 development images are descriptive, not generalization claims."]
    (target/"case_report.md").write_text("\n".join(sections)+"\n",encoding="utf-8")
    return {"cases":len({r["file"] for r in rows}),"rows":len(rows),"output":str(target)}
