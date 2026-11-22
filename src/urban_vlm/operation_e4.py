"""E4: compare alternative operation descriptions while freezing E3's presence logits.

No retraining. The model is called ONLY with 10 new operation prompts.
The first-stage image-text scores (E3: commercial/noncommercial) are re-used
bit for bit from an archived E3 run, so changes in predictions can be attributed
operationally to the modified second-stage descriptions/aggregation.

All evaluation is exploratory on the same 13 development images.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .aggregation import aggregate
from .backend import SiglipBackend
from .data import (
    crop_roi, file_sha256, read_annotations, read_image, read_rois,
    validate_inputs, write_csv,
)
from .evaluation import evaluate_predictions, write_report
from .hierarchical import (
    LOGIT_FIELDS as E3_LOGIT_FIELDS,
    STAGES, SUPPORTED_AGGREGATION, load_hierarchical_prompts,
)
from .pipeline import manifest

NEW_BANK = "e4_operation_frozen_presence"
PRED_FIELDS = [
    "file", "region", "prompt_bank", "label", "predicted", "correct",
    "presence_predicted", "operation_predicted",
    "score_presence_commercial", "score_presence_noncommercial",
    "score_operation_active", "score_operation_unverifiable",
    "margin_presence", "margin_operation", "aggregation", "runner_up_margin",
    "e3_predicted", "e3_operation_predicted", "e3_correct", "changed_from_e3",
]
LOGIT_FIELDS = E3_LOGIT_FIELDS + ["source"]
COMPARE_FIELDS = [
    "file", "region", "human", "e3_predicted", "e4_predicted",
    "e3_correct", "e4_correct", "change", "evidence",
    "presence_predicted", "margin_presence",
    "e3_operation_predicted", "e4_operation_predicted",
    "e3_margin_operation", "e4_margin_operation", "changed_operation",
]


def _read_csv(path: Path, columns: set[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def load_operation_prompts(path: str | Path) -> dict[str, list[str]]:
    """Exactly two hypotheses of five descriptive texts each (only stage 2)."""
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(prompts, dict) or set(prompts) != {"active", "unverifiable"}:
        raise ValueError("E4 prompts must contain exactly 'active' and 'unverifiable'")
    for cls, texts in prompts.items():
        if (not isinstance(texts, list) or len(texts) != 5 or
            any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise ValueError(f"E4 {cls} requires exactly five nonempty prompts")
    return prompts


def _as_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite value in {label}")
    return number


def _get_e3_context(
    cfg: dict, e3_dir: Path, e3_prompts_file: Path,
    *, regions: tuple[str, ...], limit: int | None = None,
) -> dict:
    """Validate E3's sources/labels/logits before reusing first-stage scores.

    This deliberately refuses to reuse cached logits with modified photographs,
    annotations, ROIs, model ID or prompt bank.
    """
    validate_inputs(cfg, require_images=True)
    anns = read_annotations(cfg["annotations"])
    rois = read_rois(cfg["rois"])
    names = sorted(anns)
    if limit is not None:
        if not 1 <= limit <= len(names):
            raise ValueError("--limit must lie within the dataset size")
        names = names[:limit]

    if not e3_dir.is_dir():
        raise FileNotFoundError(f"E3 run directory does not exist: {e3_dir}")
    e3_manifest = json.loads((e3_dir / "manifest.json").read_text(encoding="utf-8"))
    if e3_manifest.get("workflow") != "hierarchical_experimental_zero_shot":
        raise ValueError("Reference run is not an E3 hierarchical run")
    if e3_manifest.get("model_id") != cfg["model_id"]:
        raise ValueError("E3 checkpoint model ID differs from experiment configuration")
    if e3_manifest.get("aggregation") != "mean":
        raise ValueError("For the controlled E4 comparison, E3 must have aggregation='mean'")
    if e3_manifest.get("hierarchical_prompt_sha256") != file_sha256(e3_prompts_file):
        raise ValueError("E3 prompt bank changed since the reference run")
    if e3_manifest.get("seed") != cfg.get("seed"):
        raise ValueError("E3 random seed differs from experiment configuration")
    for key, path in (("annotations.csv", cfg["annotations"]),
                      ("rois.csv", cfg["rois"])):
        if e3_manifest.get("config_and_label_hashes", {}).get(key) != file_sha256(path):
            raise ValueError(f"Cannot freeze E3: {key} changed since its execution")
    image_hashes = e3_manifest.get("image_hashes", {})
    for name in names:
        if name not in image_hashes:
            raise ValueError(f"E3 did not contain image {name}; run E3 on the same images first")
        if file_sha256(Path(cfg["images_dir"]) / name) != image_hashes[name]:
            raise ValueError(f"Image {name} does not match the E3 source file")

    old_bank = load_hierarchical_prompts(e3_prompts_file)
    e3_predictions = _read_csv(
        e3_dir / "hierarchical_predictions.csv",
        {"file", "region", "label", "predicted", "presence_predicted",
         "operation_predicted", "score_presence_commercial", "score_presence_noncommercial",
         "score_operation_active", "score_operation_unverifiable",
         "margin_presence", "margin_operation"},
    )
    pred_index = {}
    for row in e3_predictions:
        key = (row["file"], row["region"])
        if key in pred_index:
            raise ValueError(f"Duplicated E3 prediction: {key}")
        pred_index[key] = row

    cached = defaultdict(list)
    for row in _read_csv(
        e3_dir / "hierarchical_logits.csv",
        set(E3_LOGIT_FIELDS),
    ):
        key = (row["file"], row["region"], row["stage"], row["hypothesis"])
        cached[key].append(row)

    selected = {}
    for name in names:
        for region in regions:
            key = (name, region)
            if key not in pred_index:
                raise ValueError(f"E3 has no prediction for {name} / {region}")
            old = pred_index[key]
            if old["label"] != anns[name]["label"]:
                raise ValueError(f"Human label for {name} changed since E3")
            for stage, classes in STAGES.items():
                for cls in classes:
                    bucket = cached.get((name, region, stage, cls), [])
                    if len(bucket) != 5:
                        raise ValueError(f"E3 missing five logits for {name}/{region}/{stage}/{cls}")
                    by_index = {int(x["prompt_index"]): x for x in bucket}
                    if set(by_index) != set(range(1, 6)):
                        raise ValueError(f"E3 prompt indices invalid for {name}/{region}/{stage}/{cls}")
                    for idx, text in enumerate(old_bank[stage][cls], 1):
                        if by_index[idx]["prompt_text"] != text:
                            raise ValueError("E3 prompt text differs from the archived bank")
                    score = aggregate(
                        [_as_float(by_index[i]["logit"], "E3 logit") for i in range(1, 6)],
                        "mean",
                    )
                    if abs(score - _as_float(old[f"score_{stage}_{cls}"], "E3 score")) > 1e-5:
                        raise ValueError(f"Archived E3 class score does not match saved logits: {key}/{stage}/{cls}")
            for stage, first, second in (
                ("presence", "commercial", "noncommercial"),
                ("operation", "active", "unverifiable"),
            ):
                computed = (_as_float(old[f"score_{stage}_{first}"], "E3 score") -
                            _as_float(old[f"score_{stage}_{second}"], "E3 score"))
                if abs(computed - _as_float(old[f"margin_{stage}"], "E3 margin")) > 1e-5:
                    raise ValueError(f"E3 {stage} margin differs from its scores for {key}")
            expected_presence = ("commercial" if _as_float(old["margin_presence"], "presence") > 0
                                 else "noncommercial")
            expected_operation = ("active" if _as_float(old["margin_operation"], "operation") > 0
                                 else "unverifiable")
            expected_label = ("N" if expected_presence == "noncommercial" else
                              ("A" if expected_operation == "active" else "I"))
            if ((old["presence_predicted"], old["operation_predicted"], old["predicted"])
                != (expected_presence, expected_operation, expected_label)):
                raise ValueError(f"E3 decisions inconsistent with E3 margins: {key}")
            selected[key] = old
    return {
        "names": names, "regions": regions, "annotations": anns, "rois": rois,
        "e3_manifest": e3_manifest, "e3_predictions": selected, "e3_logits": cached,
        "e3_dir": e3_dir,
    }


def validate_e4_inputs(
    cfg: dict, e3_dir: str | Path, e3_prompts_file: str | Path,
    prompts_file: str | Path, *, region: str = "roi", limit: int | None = None,
) -> dict:
    """Perform all input checks with no inference or GPU activity."""
    if region not in ("roi", "full", "both"):
        raise ValueError("--region must be roi, full or both")
    regions = ("full", "roi") if region == "both" else (region,)
    prompts = load_operation_prompts(prompts_file)
    context = _get_e3_context(
        cfg, Path(e3_dir).resolve(), Path(e3_prompts_file).resolve(),
        regions=regions, limit=limit,
    )
    return {
        "images": len(context["names"]), "regions": list(regions),
        "e3_records_available": len(context["e3_predictions"]),
        "new_operation_prompts": sum(len(x) for x in prompts.values()),
        "first_stage": "E3 first-stage logits reused unchanged",
        "model": cfg["model_id"], "e3_dir": str(Path(e3_dir).resolve()),
    }


def run_operation_e4(
    cfg: dict, e3_dir: str | Path, e3_prompts_file: str | Path,
    prompts_file: str | Path, output_dir: str | Path,
    *, region: str = "roi", aggregation: str = "mean", device: str = "auto",
    limit: int | None = None, backend=None,
) -> dict:
    """Run E4 with unchanged E3 presence scores and new operation prompts.

    Only the two operation hypotheses are re-inferred; each image-region
    requires 10 new prompt logits. Report all changes, not only improvements.
    """
    if aggregation != "mean":
        raise ValueError("E4 fixes aggregation='mean' to isolate only the effect of stage-2 prompt wording")
    if region not in ("roi", "full", "both"):
        raise ValueError("--region must be roi, full or both")
    regions = ("full", "roi") if region == "both" else (region,)
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {target}")
    prompt_file = Path(prompts_file).resolve()
    e3_prompt_file = Path(e3_prompts_file).resolve()
    prompts = load_operation_prompts(prompt_file)
    context = _get_e3_context(cfg, Path(e3_dir).resolve(), e3_prompt_file,
                              regions=regions, limit=limit)
    requested = cfg.get("model_revision")
    if context["e3_manifest"].get("model_revision_requested") != requested:
        raise ValueError("E3 and E4 requested different model revisions")
    prior_commit = context["e3_manifest"].get("model_commit_hash_detected")
    if backend is None:
        # Prefer an immutable revision actually observed in the E3 manifest.
        backend = SiglipBackend(cfg["model_id"], device, requested or prior_commit,
                                cfg.get("seed", 42))
    new_commit = getattr(backend, "commit", None)
    if prior_commit and new_commit and prior_commit != new_commit:
        raise ValueError(f"Model revision mismatch: E3={prior_commit}, E4={new_commit}")

    texts = prompts["active"] + prompts["unverifiable"]
    predictions, logits, comparisons = [], [], []
    for index, name in enumerate(context["names"], start=1):
        image = read_image(Path(cfg["images_dir"]) / name)
        views = {"full": image, "roi": crop_roi(image, context["rois"][name])}
        print(f"[operation-e4 {index}/{len(context['names'])}] {name}", flush=True)
        for view in context["regions"]:
            key = (name, view)
            old = context["e3_predictions"][key]
            new_logits = backend.predict_logits(views[view], texts)
            if len(new_logits) != 10:
                raise ValueError(f"Expected 10 operation logits, got {len(new_logits)}")
            a_values = [_as_float(x, "new active") for x in new_logits[:5]]
            i_values = [_as_float(x, "new unverifiable") for x in new_logits[5:]]
            active = aggregate(a_values, "mean")
            unverifiable = aggregate(i_values, "mean")
            op_margin = active - unverifiable
            operation = "active" if op_margin > 0 else "unverifiable"
            presence = old["presence_predicted"]  # E3's frozen first-stage result
            result = "N" if presence == "noncommercial" else ("A" if operation == "active" else "I")
            truth = context["annotations"][name]["label"]
            predicted = {
                "file": name, "region": view, "prompt_bank": NEW_BANK,
                "label": truth, "predicted": result, "correct": int(result == truth),
                "presence_predicted": presence, "operation_predicted": operation,
                "score_presence_commercial": float(old["score_presence_commercial"]),
                "score_presence_noncommercial": float(old["score_presence_noncommercial"]),
                "score_operation_active": active, "score_operation_unverifiable": unverifiable,
                "margin_presence": float(old["margin_presence"]), "margin_operation": op_margin,
                "aggregation": "mean", "runner_up_margin": "",
                "e3_predicted": old["predicted"],
                "e3_operation_predicted": old["operation_predicted"],
                "e3_correct": int(old["predicted"] == truth),
                "changed_from_e3": int(old["predicted"] != result),
            }
            predictions.append(predicted)
            for cls in STAGES["presence"]:
                for old_row in sorted(context["e3_logits"][(name, view, "presence", cls)],
                                      key=lambda z: int(z["prompt_index"])):
                    logits.append({
                        "file": name, "region": view,
                        "stage": "presence", "hypothesis": cls,
                        "prompt_index": int(old_row["prompt_index"]),
                        "prompt_text": old_row["prompt_text"],
                        "logit": float(old_row["logit"]), "source": "E3_cached_unchanged",
                    })
            for cls, values in (("active", a_values), ("unverifiable", i_values)):
                for prompt_idx, (prompt, logit) in enumerate(zip(prompts[cls], values), 1):
                    logits.append({
                        "file": name, "region": view,
                        "stage": "operation", "hypothesis": cls,
                        "prompt_index": prompt_idx, "prompt_text": prompt,
                        "logit": logit, "source": "E4_new_inference",
                    })
            old_ok, new_ok = old["predicted"] == truth, result == truth
            change = ("corrected" if not old_ok and new_ok else
                      "regressed" if old_ok and not new_ok else
                      "unchanged_correct" if old_ok else "unchanged_wrong")
            comparisons.append({
                "file": name, "region": view, "human": truth,
                "e3_predicted": old["predicted"], "e4_predicted": result,
                "e3_correct": int(old_ok), "e4_correct": int(new_ok), "change": change,
                "evidence": context["annotations"][name]["evidence"],
                "presence_predicted": presence,
                "margin_presence": float(old["margin_presence"]),
                "e3_operation_predicted": old["operation_predicted"],
                "e4_operation_predicted": operation,
                "e3_margin_operation": float(old["margin_operation"]),
                "e4_margin_operation": op_margin,
                "changed_operation": int(operation != old["operation_predicted"]),
            })

    # Only touch the destination AFTER every image has been scored successfully.
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "operation_e4_predictions.csv", predictions, PRED_FIELDS)
    write_csv(target / "operation_e4_logits.csv", logits, LOGIT_FIELDS)
    write_csv(target / "paired_e3_e4.csv", comparisons, COMPARE_FIELDS)
    write_report(target, predictions, context["annotations"], provenance="E4 stage-2 contrastive prompts, E3 stage-1 frozen")
    e3_rows = [
        {"file": k[0], "region": k[1], "prompt_bank": "e3_hierarchical",
         "label": context["annotations"][k[0]]["label"], "predicted": row["predicted"]}
        for k, row in context["e3_predictions"].items()
    ]
    e3_metrics = evaluate_predictions(e3_rows)
    (target / "e3_reference_metrics.json").write_text(
        json.dumps(e3_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    change_counts = {label: sum(x["change"] == label for x in comparisons)
                     for label in ("corrected", "regressed", "unchanged_correct", "unchanged_wrong")}
    meta = manifest(cfg, context["names"], backend, len(predictions), limit is not None)
    meta.update({
        "workflow": "operation_e4_frozen_e3_presence",
        "source_e3_dir": str(Path(e3_dir).resolve()),
        "source_e3_manifest_sha256": file_sha256(Path(e3_dir) / "manifest.json"),
        "source_e3_predictions_sha256": file_sha256(Path(e3_dir) / "hierarchical_predictions.csv"),
        "source_e3_logits_sha256": file_sha256(Path(e3_dir) / "hierarchical_logits.csv"),
        "e3_hierarchical_prompts_sha256": file_sha256(e3_prompt_file),
        "e4_effective_model_revision": requested or prior_commit,
        "e4_operation_prompts_sha256": file_sha256(prompt_file),
        "new_prompts_file": str(prompt_file),
        "stage1_logits_reused_without_reinference": True,
        "stage2_logits_newly_inferred": True,
        "aggregation_fixed": "mean",
        "regions": list(regions), "n_prompt_logits": len(logits),
        "change_counts": change_counts,
        "paired_comparison_not_independent_validation": True,
        "note": "E4 is post-hoc exploratory design. Do not claim causal interpretation, calibration or generalization."
    })
    (target / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# E4 — Operation Evidence Refinement (desarrollo exploratorio)", "",
        "**Etapa de presencia:** los 10 logits de E3 se reutilizan sin cambios. Solo se evalúan 10 textos nuevos de operación por región.",
        "El diseño textual E4 es posterior al análisis de estas 13 imágenes; NO constituye evaluación independiente.",
        "**La etiqueta I significa operación no verificable, no cierre definitivo.**", "",
        "| Region | E3 correct | E4 correct | E3 macro-F1 | E4 macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    e4_metrics = evaluate_predictions(predictions)
    for view in regions:
        old = next(s for s in e3_metrics["summary"] if s["region"] == view)
        new = next(s for s in e4_metrics["summary"] if s["region"] == view)
        lines.append(f"| {view} | {old['correct']}/{old['n']} | {new['correct']}/{new['n']} | "
                     f"{old['macro_f1']:.3f} | {new['macro_f1']:.3f} |")
    lines += ["", "## Comparación pareada por fotografía", "",
              "| File | Region | Human | E3 | E4 | Cambio | Presencia E3 | E3 margen op. | E4 margen op. |",
              "|---|---|:---:|:---:|:---:|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['file']} | {row['region']} | {row['human']} | "
                     f"{row['e3_predicted']} | {row['e4_predicted']} | {row['change']} | "
                     f"{row['presence_predicted']} | {row['e3_margin_operation']:.3f} | "
                     f"{row['e4_margin_operation']:.3f} |")
    lines += ["", "## Límites", "",
              "El diseño E4 es post hoc. Las 13 fotografías no son muestras independientes del proceso de selección de prompts.",
              "Dos fotografías (F02/F13) pueden mostrar el mismo establecimiento.",
              "Los márgenes de etapa 2 se calculan frente a bancos textuales diferentes y no son probabilidades calibradas.",
              "Los cambios de decisión no identifican qué píxeles causaron el error del modelo."]
    (target / "E4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "images": len(context["names"]), "regions": list(regions),
        "predictions": len(predictions), "prompt_logits": len(logits),
        "new_stage2_logits": len(predictions) * 10,
        "frozen_stage1_logits": len(predictions) * 10,
        "change_counts": change_counts, "output": str(target),
    }
