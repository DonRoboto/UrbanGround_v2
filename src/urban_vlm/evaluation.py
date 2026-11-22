"""Transparent three-way metrics and mechanism hypotheses, not causal claims."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from .data import CLASSES, read_annotations, write_csv


def evaluate_predictions(predictions: list[dict]) -> dict:
    if not predictions:
        raise ValueError("No predictions")
    grouped = {}
    for p in predictions:
        label, pred = p["label"], p["predicted"]
        if label not in CLASSES or pred not in CLASSES:
            raise ValueError(f"Invalid label/prediction: {label}/{pred}")
        key = (p["region"], p["prompt_bank"])
        grouped.setdefault(key, []).append(p)
    summary = []
    matrices = {}
    for (region, bank), group in sorted(grouped.items()):
        matrix = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
        for p in group:
            matrix[p["label"]][p["predicted"]] += 1
        total = len(group)
        per_class = {}
        for cls in CLASSES:
            tp = matrix[cls][cls]
            support = sum(matrix[cls].values())
            predicted_count = sum(matrix[t][cls] for t in CLASSES)
            precision = tp / predicted_count if predicted_count else 0.0
            recall = tp / support if support else 0.0
            f1 = 2*precision*recall/(precision+recall) if precision + recall else 0.0
            per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        summary.append({"region": region, "prompt_bank": bank, "n": total,
                        "correct": sum(matrix[c][c] for c in CLASSES),
                        "accuracy": sum(matrix[c][c] for c in CLASSES) / total,
                        "macro_f1": sum(per_class[c]["f1"] for c in CLASSES) / len(CLASSES),
                        **{f"recall_{c}": per_class[c]["recall"] for c in CLASSES}})
        matrices[f"{region}|{bank}"] = {"confusion": matrix, "per_class": per_class}
    return {"summary": summary, "detailed": matrices, "note": "Development set, 13 images; F02/F13 may be the same establishment. No generalization claim."}


def diagnose_disagreements(predictions: list[dict], annotations: dict) -> list[dict]:
    rows = []
    for p in predictions:
        if p["label"] == p["predicted"]:
            continue
        true, predicted = p["label"], p["predicted"]
        if true == "I" and predicted == "A":
            mechanism = "Possible conflation of commercial premises with visible operation"
        elif true == "I" and predicted == "N":
            mechanism = "Possible weak or occluded commercial cues; non-operation remains unobservable"
        elif true == "A" and predicted == "N":
            mechanism = "Possible dilution of small ground-floor cues by streets, homes or vehicles"
        elif true == "A" and predicted == "I":
            mechanism = "Prompt semantic overlap or averaging heterogeneous shop subtypes"
        elif true == "N" and predicted in ("I", "A"):
            mechanism = "Possible spillover from adjacent fronts or generic commercial cues"
        else:
            mechanism = "Inspect ROI, annotated front, and class-specific prompt logits"
        rows.append({"file": p["file"], "region": p["region"], "prompt_bank": p["prompt_bank"],
                     "human": true, "predicted": predicted, "evidence": annotations[p["file"]]["evidence"],
                     "hypothesis_not_proof": mechanism, "runner_up_margin": p.get("runner_up_margin", "")})
    return rows


def write_report(out_dir: str | Path, predictions: list[dict], annotations: dict, provenance: str = "new run") -> dict:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    obj = evaluate_predictions(predictions)
    diagnoses = diagnose_disagreements(predictions, annotations)
    (dest / "metrics.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(dest / "metrics.csv", obj["summary"], ["region", "prompt_bank", "n", "correct", "accuracy", "macro_f1", "recall_A", "recall_I", "recall_N"])
    write_csv(dest / "disagreements.csv", diagnoses, ["file","region","prompt_bank","human","predicted","evidence","hypothesis_not_proof","runner_up_margin"])
    parts = ["# Parte B — SigLIP 2: resultados " + provenance, "",
             "Clasificación exploratoria A (actividad visible), I (no verificable), N (sin comercio visible).", "",
             "| Region | Prompts | Correct | Accuracy | Macro-F1 | Recall A | Recall I | Recall N |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in obj["summary"]:
        parts.append(f'| {s["region"]} | {s["prompt_bank"]} | {s["correct"]}/{s["n"]} | {s["accuracy"]:.3f} | {s["macro_f1"]:.3f} | {s["recall_A"]:.3f} | {s["recall_I"]:.3f} | {s["recall_N"]:.3f} |')
    parts += ["", "## Desacuerdos (mecanismos propuestos, no causales)", ""]
    for d in diagnoses:
        parts += [f'**{d["file"]} · {d["region"]}/{d["prompt_bank"]}: {d["human"]} → {d["predicted"]}.** {d["evidence"]} Hipótesis: {d["hypothesis_not_proof"]}.', ""]
    parts += ["## Límites", "",
              "Las 13 fotografías se usaron durante el diseño de ROI y prompts: estos resultados no constituyen validación independiente.",
              "I no significa negocio cerrado; indica que la imagen no permite verificar su operación.",
              "F02 y F13 podrían mostrar el mismo negocio. La única clase N contiene una imagen. Los logits no son probabilidades calibradas.",
              "Un recorte cambia simultáneamente escala aparente y contexto; difuminar regiones no identifica mecanismos causales internos."]
    (dest / "report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    return obj


def load_predictions(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    needed = {"file", "region", "prompt_bank", "label", "predicted"}
    if not rows or not needed.issubset(rows[0]):
        raise ValueError(f"Prediction CSV must include {sorted(needed)}")
    if len({(p["file"],p["region"],p["prompt_bank"]) for p in rows}) != len(rows):
        raise ValueError("Duplicate file/region/bank prediction rows")
    return rows
