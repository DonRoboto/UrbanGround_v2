"""Experimental two-stage classification using frozen SigLIP 2.

Stage 1: recognizable commercial premises? Stage 2: observable operation?
The scores within a stage are comparable, but NOT across stages. No thresholds
or prompts are tuned using the labels at prediction time. A static image cannot
establish whether a shop is open in the world at the time of analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .aggregation import aggregate
from .backend import SiglipBackend
from .data import (
    read_annotations, read_rois, read_image, crop_roi, validate_inputs,
    write_csv, file_sha256,
)
from .evaluation import write_report
from .pipeline import manifest

STAGES = {
    "presence": ("commercial", "noncommercial"),
    "operation": ("active", "unverifiable"),
}
SUPPORTED_AGGREGATION = ("mean", "median", "top2", "max", "logmeanexp")

PRED_FIELDS = [
    "file", "region", "prompt_bank", "label", "predicted", "correct",
    "presence_predicted", "operation_predicted",
    "score_presence_commercial", "score_presence_noncommercial",
    "score_operation_active", "score_operation_unverifiable",
    "margin_presence", "margin_operation", "aggregation", "runner_up_margin",
]
LOGIT_FIELDS = ["file", "region", "stage", "hypothesis", "prompt_index", "prompt_text", "logit"]


def load_hierarchical_prompts(path: str | Path) -> dict:
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(prompts, dict) or set(prompts) != set(STAGES):
        raise ValueError("Hierarchical prompts must define stages: presence, operation")
    for stage, groups in STAGES.items():
        if not isinstance(prompts[stage], dict) or set(prompts[stage]) != set(groups):
            raise ValueError(f"Stage {stage} must define {groups}")
        for name in groups:
            texts = prompts[stage][name]
            if not isinstance(texts, list) or len(texts) != 5 or any(
                not isinstance(t, str) or not t.strip() for t in texts
            ):
                raise ValueError(f"Stage {stage}/{name} needs exactly five non-empty texts")
    return prompts


def flatten_hierarchical(prompts: Mapping) -> tuple[list[str], list[tuple[str, str, int, str]]]:
    texts, descriptors = [], []
    for stage, groups in STAGES.items():
        for hypothesis in groups:
            for index, prompt in enumerate(prompts[stage][hypothesis], 1):
                texts.append(prompt)
                descriptors.append((stage, hypothesis, index, prompt))
    return texts, descriptors


def hierarchical_decision(logits: list[float], prompts: dict, aggregation: str = "mean") -> tuple[dict, list[dict]]:
    if aggregation not in SUPPORTED_AGGREGATION:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    texts, descriptors = flatten_hierarchical(prompts)
    if len(logits) != len(texts):
        raise ValueError(f"Expected {len(texts)} logits, got {len(logits)}")
    rows = []
    grouped = {(s, k): [] for s, keys in STAGES.items() for k in keys}
    for (stage, hypothesis, index, prompt), value in zip(descriptors, logits):
        val = float(value)
        grouped[stage, hypothesis].append(val)
        rows.append({"stage": stage, "hypothesis": hypothesis, "prompt_index": index,
                     "prompt_text": prompt, "logit": val})
    scores = {key: aggregate(vals, aggregation) for key, vals in grouped.items()}
    margin_presence = scores["presence", "commercial"] - scores["presence", "noncommercial"]
    margin_operation = scores["operation", "active"] - scores["operation", "unverifiable"]
    # Ties treated conservatively, not as evidence of active business.
    presence = "commercial" if margin_presence > 0 else "noncommercial"
    operation = "active" if margin_operation > 0 else "unverifiable"
    label = "N" if presence == "noncommercial" else ("A" if operation == "active" else "I")
    result = {
        "predicted": label, "presence_predicted": presence, "operation_predicted": operation,
        "score_presence_commercial": scores["presence", "commercial"],
        "score_presence_noncommercial": scores["presence", "noncommercial"],
        "score_operation_active": scores["operation", "active"],
        "score_operation_unverifiable": scores["operation", "unverifiable"],
        "margin_presence": margin_presence, "margin_operation": margin_operation,
        "aggregation": aggregation,
        "runner_up_margin": "",  # NO comparable three-way logit scores exist for this method.
    }
    return result, rows


def run_hierarchical(
    cfg: dict, prompts_file: str | Path, output_dir: str | Path,
    *, region: str = "roi", aggregation: str = "mean", device: str = "auto",
    limit: int | None = None, backend=None,
) -> dict:
    validate_inputs(cfg, require_images=True)
    prompt_file = Path(prompts_file).resolve()
    prompts = load_hierarchical_prompts(prompt_file)
    anns, rois = read_annotations(cfg["annotations"]), read_rois(cfg["rois"])
    names = sorted(anns)
    if limit is not None:
        if not 1 <= limit <= len(names):
            raise ValueError("limit must be between 1 and the number of images")
        names = names[:limit]
    if region not in ("roi", "full", "both"):
        raise ValueError("region must be roi, full or both")
    regions = ("full", "roi") if region == "both" else (region,)
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {target}")
    if backend is None:
        backend = SiglipBackend(cfg["model_id"], device, cfg.get("model_revision"), cfg.get("seed", 42))
    texts, _ = flatten_hierarchical(prompts)
    predictions, prompt_logits = [], []
    for index, name in enumerate(names, 1):
        img = read_image(Path(cfg["images_dir"]) / name)
        views = {"full": img, "roi": crop_roi(img, rois[name])}
        print(f"[hierarchical {index}/{len(names)}] {name}", flush=True)
        for view_name in regions:
            result, logits = hierarchical_decision(
                backend.predict_logits(views[view_name], texts), prompts, aggregation
            )
            predictions.append({
                "file": name, "region": view_name, "prompt_bank": "hierarchical_evidence",
                "label": anns[name]["label"], "correct": int(result["predicted"] == anns[name]["label"]),
                **result,
            })
            prompt_logits.extend({"file": name, "region": view_name, **row} for row in logits)
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "hierarchical_predictions.csv", predictions, PRED_FIELDS)
    write_csv(target / "hierarchical_logits.csv", prompt_logits, LOGIT_FIELDS)
    metadata = manifest(cfg, names, backend, len(predictions), limit is not None)
    metadata.update({
        "workflow": "hierarchical_experimental_zero_shot",
        "prompt_bank_file": str(prompt_file),
        "hierarchical_prompt_sha256": file_sha256(prompt_file),
        "aggregation": aggregation, "regions": list(regions),
        "stage1": "commercial presence vs noncommercial frontage",
        "stage2": "visibly active vs operation not verifiable",
        "unverifiable_not_equivalent_to_closed": True,
        "warning": "Exploratory design on the same 13 development images. Stage logits cannot be compared across stages."
    })
    (target / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_report(target, predictions, anns, provenance="EXPLORATORY two-stage zero-shot (not independent validation)")
    return {"images": len(names), "regions": list(regions), "predictions": len(predictions),
            "prompt_logits": len(prompt_logits), "output": str(target)}
