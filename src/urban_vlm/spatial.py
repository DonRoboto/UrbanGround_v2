"""Paired spatial sensitivity diagnostics, never pixel-level causal explanations."""
from __future__ import annotations
from .data import write_csv

EFFECT_FIELDS = [
    "file", "prompt_bank", "intervention", "human", "baseline_predicted", "predicted",
    "label_changed", "correct", "score_A", "score_I", "score_N",
    "delta_score_A", "delta_score_I", "delta_score_N",
    "delta_A_minus_N", "delta_A_minus_I", "delta_I_minus_N",
    "scale_or_canvas_changed", "interpretation",
]


def compute_spatial_effects(predictions: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in predictions:
        grouped.setdefault((row["file"], row["prompt_bank"]), []).append(row)
    rows = []
    for (name, bank), variants in sorted(grouped.items()):
        origs = [r for r in variants if r["region"] == "original"]
        if len(origs) != 1:
            raise ValueError(f"Need exactly one original condition: {name}/{bank}")
        baseline = origs[0]
        for row in variants:
            variant = row["region"]
            scores = {c: float(row[f"score_{c}"]) for c in "AIN"}
            original = {c: float(baseline[f"score_{c}"]) for c in "AIN"}
            delta = {c: scores[c]-original[c] for c in "AIN"}
            if variant == "roi_crop":
                comment = "Crop changes both relative target scale and surrounding context."
            elif variant == "blur_outside_roi":
                comment = "Full canvas and ROI scale retained; background appearance changed (OOD blur)."
            elif variant == "blur_inside_roi":
                comment = "Local ROI appearance altered, potentially including vehicles or walls."
            elif variant == "blur_upper":
                comment = "Upper image appearance altered; not a causal ablation."
            else:
                comment = "Unmodified reference for within-bank comparisons."
            rows.append({
                "file": name, "prompt_bank": bank, "intervention": variant,
                "human": row["label"], "baseline_predicted": baseline["predicted"],
                "predicted": row["predicted"], "label_changed": int(row["predicted"] != baseline["predicted"]),
                "correct": int(row["predicted"] == row["label"]),
                **{f"score_{c}":scores[c] for c in "AIN"},
                **{f"delta_score_{c}":delta[c] for c in "AIN"},
                "delta_A_minus_N": delta["A"]-delta["N"],
                "delta_A_minus_I": delta["A"]-delta["I"],
                "delta_I_minus_N": delta["I"]-delta["N"],
                "scale_or_canvas_changed": int(variant == "roi_crop"),
                "interpretation": comment,
            })
    return rows


def write_spatial_diagnostics(output, predictions: list[dict]) -> int:
    result = compute_spatial_effects(predictions)
    write_csv(output / "intervention_effects.csv", result, EFFECT_FIELDS)
    return len(result)
