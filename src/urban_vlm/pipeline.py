"""Experiment runner. Inference can be replaced with a fake backend in unit tests."""
from __future__ import annotations
import json
import platform
import sys
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from pathlib import Path
from .backend import SiglipBackend
from .data import (
    CLASSES, read_annotations, read_rois, read_prompt_bank,
    validate_inputs, read_image, crop_roi, file_sha256, write_csv,
)
from .scoring import calculate_scores
from .evaluation import write_report
from .perturbations import make_variants
from .spatial import write_spatial_diagnostics

PRED_FIELDS = ["file", "region", "prompt_bank", "label", "predicted", "score_A", "score_I", "score_N", "runner_up_margin", "correct"]
LOGIT_FIELDS = ["file", "region", "prompt_bank", "class", "prompt_index", "prompt_text", "logit"]


def score_one(backend, img, bank, file: str, region: str, bank_name: str, label: str):
    texts = [prompt for cls in CLASSES for prompt in bank[cls]]
    scores, predicted, margin, items = calculate_scores(backend.predict_logits(img, texts), bank)
    prediction = {"file": file, "region": region, "prompt_bank": bank_name, "label": label,
                  "predicted": predicted, **{f"score_{c}": scores[c] for c in CLASSES},
                  "runner_up_margin": margin, "correct": int(label == predicted)}
    logit_rows = [{"file": file, "region": region, "prompt_bank": bank_name, **item} for item in items]
    return prediction, logit_rows


def manifest(cfg, files, backend, rows: int, partial: bool):
    import PIL
    import yaml
    from . import __version__
    from .data import file_sha256
    data_files = [cfg["annotations"], cfg["rois"], cfg["config_path"], *cfg["prompts"].values()]
    def package_version(name):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None
    torch_runtime = getattr(backend, "torch", None)
    gpu_name = None
    if torch_runtime is not None and torch_runtime.cuda.is_available():
        gpu_name = torch_runtime.cuda.get_device_name(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": __version__, "model_id": cfg["model_id"],
        "model_revision_requested": cfg.get("model_revision"),
        "model_commit_hash_detected": getattr(backend, "commit", None),
        "device": getattr(backend, "device", "fake-test"),
        "seed": cfg.get("seed"),
        "python": sys.version.split()[0], "pillow": PIL.__version__,
        "pyyaml": yaml.__version__, "platform": platform.platform(),
        "runtime_packages": {name: package_version(name) for name in
            ("transformers", "torch", "tokenizers", "numpy", "safetensors")},
        "cuda_runtime": str(torch_runtime.version.cuda) if torch_runtime is not None else None,
        "gpu_name": gpu_name,
        "image_hashes": {name: file_sha256(Path(cfg["images_dir"]) / name) for name in files},
        "config_and_label_hashes": {str(Path(p).name): file_sha256(p) for p in data_files},
        "image_count": len(files), "prediction_count": rows, "partial_run": partial,
        "interpretation": "exploratory development set; no claims of regional bias or generalization"
    }


def run_experiment(cfg: dict, output_dir: str | Path, *, device: str = "auto", limit: int | None = None, backend=None):
    validate_inputs(cfg, require_images=True)
    anns, rois = read_annotations(cfg["annotations"]), read_rois(cfg["rois"])
    banks = {key: read_prompt_bank(path) for key, path in cfg["prompts"].items()}
    files = sorted(anns)
    if limit is not None:
        if not (1 <= limit <= len(files)):
            raise ValueError(f"--limit must be in 1..{len(files)}")
        files = files[:limit]
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {target}")
    if backend is None:
        backend = SiglipBackend(cfg["model_id"], device, cfg.get("model_revision"), cfg.get("seed", 42))
    predictions, logits = [], []
    for i, filename in enumerate(files, 1):
        image = read_image(Path(cfg["images_dir"]) / filename)
        views = {"full": image, "roi": crop_roi(image, rois[filename])}
        print(f"[{i}/{len(files)}] {filename}", flush=True)
        for region in cfg["regions"]:
            for bank_name, bank in banks.items():
                pred, detail = score_one(backend, views[region], bank, filename, region, bank_name, anns[filename]["label"])
                predictions.append(pred)
                logits.extend(detail)
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "predictions.csv", predictions, PRED_FIELDS)
    write_csv(target / "prompt_logits.csv", logits, LOGIT_FIELDS)
    (target / "manifest.json").write_text(json.dumps(manifest(cfg, files, backend, len(predictions), limit is not None), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(target, predictions, anns, provenance="PARTIAL SMOKE TEST" if limit is not None else "13-image exploratory experiment")
    return {"images": len(files), "predictions": len(predictions), "prompt_logits": len(logits), "output": str(target)}


def run_diagnostics(cfg: dict, output_dir: str | Path, *, device: str = "auto", files: list[str] | None = None, backend=None):
    validate_inputs(cfg, require_images=True)
    anns, rois = read_annotations(cfg["annotations"]), read_rois(cfg["rois"])
    names = sorted(anns) if not files else files
    if any(name not in anns for name in names):
        raise ValueError("Unknown image name; check annotations.csv")
    banks = {k: read_prompt_bank(p) for k, p in cfg["prompts"].items()}
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing diagnostic run: {target}")
    if backend is None:
        backend = SiglipBackend(cfg["model_id"], device, cfg.get("model_revision"), cfg.get("seed", 42))
    predictions, logits = [], []
    for i, name in enumerate(names, 1):
        img = read_image(Path(cfg["images_dir"]) / name)
        variants = make_variants(img, rois[name], cfg.get("blur_radius", 24.0))
        print(f"[diagnostics {i}/{len(names)}] {name}", flush=True)
        for bank_name, bank in banks.items():
            for variant_name, var_img in variants.items():
                pred, detail = score_one(backend, var_img, bank, name, variant_name, bank_name, anns[name]["label"])
                predictions.append(pred)
                logits.extend(detail)
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "interventions.csv", predictions, PRED_FIELDS)
    write_csv(target / "intervention_logits.csv", logits, LOGIT_FIELDS)
    effect_rows = write_spatial_diagnostics(target, predictions)
    (target / "manifest.json").write_text(json.dumps({**manifest(cfg,names,backend,len(predictions),files is not None),
        "WARNING": "Interventions mix appearance, scale and context; sensitivity does not establish model causality.",
        "variant_names": ["original","roi_crop","blur_upper","blur_inside_roi","blur_outside_roi"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"images":len(names),"predictions":len(predictions),"prompt_logits":len(logits),"effect_rows":effect_rows,"output":str(target)}
