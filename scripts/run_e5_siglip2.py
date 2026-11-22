#!/usr/bin/env python3
"""E5: paired SigLIP 2 comparison of full images and manual/OpenCV/OWLv2 ROIs.

Run from any working directory; script derives project root from its own location.
No modification of existing config, manual ROI or annotations. Refuses to overwrite runs.

Examples:
  python scripts/run_e5_siglip2.py --opencv-rois runs/E5_opencv_01/rois_auto.csv \\
      --owlv2-rois runs/E5_owlv2_01/rois_auto.csv --output runs/E5_siglip_01

  python scripts/run_e5_siglip2.py --opencv-rois runs/E5_opencv_01/rois_auto.csv \\
      --owlv2-rois runs/E5_owlv2_01/rois_auto.csv --output runs/E5_smoke_01 \\
      --limit 2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from urban_vlm.data import file_sha256, load_config, read_annotations, read_rois, validate_inputs  # noqa: E402
from urban_vlm.evaluation import evaluate_predictions  # noqa: E402

METHODS = ("manual", "opencv", "owlv2")
BANKS = ("generic", "contextual")
NAMES = tuple(f"fachada_{i:02d}.jpg" for i in range(1, 14))


def rooted(p: str | Path) -> Path:
    v = Path(p).expanduser()
    return v.resolve() if v.is_absolute() else (PROJECT_ROOT / v).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def roi_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def roi_preflight(cfg: dict, paths: dict[str, Path], *, limit: int | None = None) -> dict:
    """Validate every ROI, sizes and original image hashes; never uses reference ROIs to choose boxes."""
    names = set(read_annotations(cfg["annotations"]))
    if names != set(NAMES):
        raise ValueError("Expected 13 annotations with canonical names")
    validate_inputs(cfg, require_images=True)
    image_hashes = {n: file_sha256(Path(cfg["images_dir"]) / n) for n in sorted(names)}
    manual_hash = file_sha256(cfg["rois"])
    label_hash = file_sha256(cfg["annotations"])
    rois = {}
    for method, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {method} ROI file: {path}")
        rois[method] = read_rois(path)
        check_cfg = {**cfg, "rois": str(path)}
        validate_inputs(check_cfg, require_images=True)
        if set(rois[method]) != names:
            raise ValueError(f"{method}: ROI names differ from 13 annotations")
        if method == "manual":
            continue
        m = path.parent / "manifest.json"
        if not m.is_file():
            raise FileNotFoundError(f"{method}: missing E5 localization manifest: {m}")
        metadata = read_json(m)
        if metadata.get("method") != method:
            raise ValueError(f"{method}: manifest method is {metadata.get('method')!r}")
        if metadata.get("images_sha256") != image_hashes:
            raise ValueError(f"{method}: localization images differ from current data/images")
        if metadata.get("annotation_file_sha256") != label_hash:
            raise ValueError(f"{method}: labels changed since ROI localization")
        if metadata.get("reference_rois_sha256") not in (None, manual_hash):
            raise ValueError(f"{method}: manual ROI reference changed since localization")
    if limit is not None and not (1 <= limit <= 13):
        raise ValueError("--limit must be between 1 and 13")
    table = []
    for n in NAMES:
        row = {"file": n}
        for method in ("opencv", "owlv2"):
            row[f"iou_{method}"] = round(roi_iou(rois["manual"][n], rois[method][n]), 8)
        table.append(row)
    return {"names": list(NAMES), "roi_files": {k: str(v) for k, v in paths.items()},
            "roi_hashes": {k: file_sha256(v) for k, v in paths.items()},
            "image_hashes": image_hashes, "annotations_sha256": label_hash,
            "roi_iou": table, "limit": limit}


def read_run(run: Path, allowed: list[str] | None = None) -> tuple[dict, dict]:
    m = run / "manifest.json"
    preds = run / "predictions.csv"
    if not m.is_file() or not preds.is_file():
        raise FileNotFoundError(f"Missing predictions.csv or manifest.json in {run}")
    manifest = read_json(m)
    with preds.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    expected_names = set(allowed or NAMES)
    expected_keys = {(name, region, bank)
                     for name in expected_names for region in ("full", "roi") for bank in BANKS}
    keys = {(r["file"], r["region"], r["prompt_bank"]) for r in rows}
    if len(keys) != len(rows) or keys != expected_keys:
        missing = sorted(expected_keys - keys)[:6]
        extra = sorted(keys - expected_keys)[:6]
        raise ValueError(f"Unexpected prediction rows in {run}: missing={missing}, extra={extra}")
    return manifest, {(r["file"], r["region"], r["prompt_bank"]): r for r in rows}


def check_run_compatibility(manifests: dict, preflight: dict, cfg: dict, names: list[str]) -> None:
    current_hashes = {n: preflight["image_hashes"][n] for n in names}
    model_ids, commits = set(), set()
    for method, m in manifests.items():
        model_ids.add(m.get("model_id"))
        if m.get("image_hashes") != current_hashes:
            raise ValueError(f"{method}: run photos differ from current photos")
        if m.get("image_count") != len(names) or m.get("prediction_count") != 4 * len(names):
            raise ValueError(f"{method}: unexpected number of images/predictions")
        ch = m.get("config_and_label_hashes", {})
        if ch.get(Path(cfg["annotations"]).name) != preflight["annotations_sha256"]:
            raise ValueError(f"{method}: annotations do not match current labels")
        # All 3 modes use same basename rois.csv (manual) vs rois_auto.csv (auto).
        roi_name = Path(preflight["roi_files"][method]).name
        if ch.get(roi_name) != preflight["roi_hashes"][method]:
            raise ValueError(f"{method}: input ROI hash differs from the ROI file used for comparison")
        for bank_file in cfg["prompts"].values():
            if ch.get(Path(bank_file).name) != file_sha256(bank_file):
                raise ValueError(f"{method}: prompt bank changed")
        commit = m.get("model_commit_hash_detected")
        if not commit:
            raise ValueError(f"{method}: checkpoint commit not recorded; cannot guarantee same model")
        commits.add(commit)
    if len(model_ids) != 1 or len(commits) != 1:
        raise ValueError(f"Model/checkpoint mismatch between runs: {model_ids}, {commits}")


def compare_runs(cfg: dict, run_dirs: dict[str, Path], preflight: dict,
                 report_folder: Path, *, names: list[str] | None = None) -> dict:
    """Compare full-image control and paired ROI classifications; inspect full-image consistency."""
    names = names or list(NAMES)
    if report_folder.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_folder}")
    manifests, predictions = {}, {}
    for method in METHODS:
        manifests[method], predictions[method] = read_run(run_dirs[method], names)
    check_run_compatibility(manifests, preflight, cfg, names)
    anns = read_annotations(cfg["annotations"])
    comparison_rows, stats_rows, changes, changes_vs_full, warnings = [], [], [], [], []
    ious = {r["file"]: r for r in preflight["roi_iou"]}
    for bank in BANKS:
        # Full-image control: same photos and prompts, without any ROI.
        # Read it from the manual run; other runs' full-image predictions are
        # checked for consistency below, and any disagreement is reported.
        full_subset = []
        for name in names:
            r = predictions["manual"][name, "full", bank]
            actual = anns[name]["label"]
            if r["label"] != actual:
                raise ValueError(f"full/{name}: annotation label does not match source")
            full_subset.append({**r, "label": actual})
        full_stat = evaluate_predictions(full_subset)["summary"][0]
        stats_rows.append({"method": "full_image", "prompt_bank": bank, **full_stat})

        for method in METHODS:
            subset = []
            for name in names:
                r = predictions[method][name, "roi", bank]
                actual = anns[name]["label"]
                if r["label"] != actual:
                    raise ValueError(f"{method}/{name}: annotation label does not match source")
                subset.append({**r, "label": actual})
            s = evaluate_predictions(subset)["summary"][0]
            stats_rows.append({"method": method, "prompt_bank": bank, **s})
        for name in names:
            label = anns[name]["label"]
            m = predictions["manual"][name, "roi", bank]["predicted"]
            full_pred = predictions["manual"][name, "full", bank]["predicted"]
            row = {"file": name, "prompt_bank": bank, "label": label,
                   "full_prediction": full_pred,
                   "manual_prediction": m,
                   "opencv_prediction": predictions["opencv"][name, "roi", bank]["predicted"],
                   "owlv2_prediction": predictions["owlv2"][name, "roi", bank]["predicted"],
                   "iou_opencv": ious[name]["iou_opencv"],
                   "iou_owlv2": ious[name]["iou_owlv2"]}
            comparison_rows.append(row)
            for method in METHODS:
                pred_roi = row[f"{method}_prediction"] if method != "manual" else m
                status_full = ("unchanged_correct" if pred_roi == full_pred == label else
                               "unchanged_wrong" if pred_roi == full_pred else
                               "improved" if pred_roi == label else
                               "degraded" if full_pred == label else "different_wrong")
                changes_vs_full.append({"file": name, "prompt_bank": bank,
                                        "method": method, "label": label,
                                        "full_prediction": full_pred,
                                        "roi_prediction": pred_roi, "status": status_full})
            for method in ("opencv", "owlv2"):
                pred = row[f"{method}_prediction"]
                status = ("unchanged_correct" if pred == m == label else
                          "unchanged_wrong" if pred == m else
                          "improved" if pred == label else
                          "degraded" if m == label else "different_wrong")
                changes.append({"file": name, "prompt_bank": bank,
                                "method": method, "label": label,
                                "manual_prediction": m, "auto_prediction": pred,
                                "status": status, "iou_vs_manual": ious[name][f"iou_{method}"]})
    for bank in BANKS:
        for name in names:
            base = predictions["manual"][name, "full", bank]["predicted"]
            for method in ("opencv", "owlv2"):
                alt = predictions[method][name, "full", bank]["predicted"]
                if base != alt:
                    warnings.append(f"Full-image prediction differs: {name}, {bank}, manual={base}, {method}={alt}")
    mean_iou = {m: statistics.mean(r[f"iou_{m}"] for r in preflight["roi_iou"] if r["file"] in names)
                for m in ("opencv", "owlv2")}
    report_folder.mkdir(parents=True)
    csv_write(report_folder / "metrics_summary.csv", stats_rows,
              ["method", "region", "prompt_bank", "n", "correct", "accuracy", "macro_f1", "recall_A", "recall_I", "recall_N"])
    csv_write(report_folder / "paired_predictions.csv", comparison_rows,
              ["file", "prompt_bank", "label", "full_prediction", "manual_prediction", "opencv_prediction", "owlv2_prediction", "iou_opencv", "iou_owlv2"])
    csv_write(report_folder / "changes_vs_manual.csv", changes,
              ["file", "prompt_bank", "method", "label", "manual_prediction", "auto_prediction", "status", "iou_vs_manual"])
    csv_write(report_folder / "changes_vs_full.csv", changes_vs_full,
              ["file", "prompt_bank", "method", "label", "full_prediction", "roi_prediction", "status"])
    csv_write(report_folder / "roi_iou.csv", [r for r in preflight["roi_iou"] if r["file"] in names],
              ["file", "iou_opencv", "iou_owlv2"])
    summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "partial_smoke_test": len(names) != 13, "n_images": len(names),
               "checkpoint_commit": manifests["manual"]["model_commit_hash_detected"],
               "mean_roi_iou": mean_iou, "results": stats_rows,
               "warnings": warnings,
               "inputs": {"roi_sha256": preflight["roi_hashes"],
                          "annotation_sha256": preflight["annotations_sha256"],
                          "runs": {m: str(run_dirs[m]) for m in METHODS}}}
    (report_folder / "comparison_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# E5 — SigLIP 2: full image versus three ROI strategies", "",
             f"Images: **{len(names)}**; checkpoint commit: `{summary['checkpoint_commit']}`.", "",
             "| Image region | Prompt bank | Correct | Accuracy | Macro-F1 | Recall A | Recall I | Recall N |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in stats_rows:
        lines.append(f"| {r['method']} | {r['prompt_bank']} | {r['correct']}/{r['n']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['recall_A']:.3f} | {r['recall_I']:.3f} | {r['recall_N']:.3f} |")
    lines += ["", "**Control:** `full_image` means no ROI/cropping; the other three rows use the same SigLIP 2 weights and prompts with different ROI selection strategies.",
              "", f"Mean box IoU against manual: OpenCV **{mean_iou['opencv']:.3f}**, OWLv2 **{mean_iou['owlv2']:.3f}**.",
              "", "## Interpretation and limitations", "",
              "- These 13 images/labels were used for exploration; this is NOT an independent generalization estimate.",
              "- Box IoU measures overlap with the manual localization, not business detection accuracy.",
              "- OWLv2 uses commercially related text queries; OpenCV uses visual structure. Neither is guaranteed to locate the target frontage.",
              "- The negative class contains one photograph, and F02/F13 likely depict the same establishment.",
              "- The class I denotes operation NOT VERIFIABLE, not closed business.",
              "- Read paired_predictions.csv, changes_vs_manual.csv and changes_vs_full.csv to audit every changed decision.", ""]
    if warnings:
        lines += ["## Full-image control warnings", ""] + ["- " + w for w in warnings] + [""]
    if len(names) != 13:
        lines += ["**PARTIAL SMOKE TEST: do not report these metrics as study findings.**", ""]
    (report_folder / "E5_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_inference(config: Path, rois: Path, dest: Path, *, device: str, limit: int | None):
    env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "urban_vlm", "run", "--config", str(config),
           "--rois-file", str(rois), "--device", device, "--output", str(dest)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    print("Executing:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiment_pinned.yaml",
                   help="Pinned SigLIP 2 checkpoint recommended; can use configs/experiment.yaml")
    p.add_argument("--opencv-rois", default="runs/E5_opencv_01/rois_auto.csv")
    p.add_argument("--owlv2-rois", default="runs/E5_owlv2_01/rois_auto.csv")
    p.add_argument("--output", default="runs/E5_siglip2_01")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test N images only (do NOT cite scores as final results)")
    p.add_argument("--dry-run", action="store_true", help="Check inputs and manifests without running SigLIP 2")
    p.add_argument("--compare-only", action="store_true", help="Compare existing runs without inference")
    p.add_argument("--manual-run", default=None, help="Reuse a prior manual-ROI run with manifest (optional)")
    p.add_argument("--opencv-run", default=None, help="Required with --compare-only")
    p.add_argument("--owlv2-run", default=None, help="Required with --compare-only")
    args = p.parse_args(argv)

    config = rooted(args.config)
    cfg = load_config(config)
    roi_paths = {"manual": Path(cfg["rois"]), "opencv": rooted(args.opencv_rois), "owlv2": rooted(args.owlv2_rois)}
    checked = roi_preflight(cfg, roi_paths, limit=args.limit)
    target = rooted(args.output)
    if target.exists():
        raise FileExistsError(f"Output directory already exists: {target}. Choose a new --output")
    print("ROI preflight OK — 13 boxes per method, image/annotation hashes agree.")
    print("Model:", cfg["model_id"], "revision:", cfg.get("model_revision"))
    if args.dry_run:
        print("DRY RUN: no inference and no file creation")
        return 0
    names = list(NAMES if args.limit is None else NAMES[:args.limit])
    if args.compare_only:
        if not (args.manual_run and args.opencv_run and args.owlv2_run):
            raise ValueError("--compare-only requires --manual-run, --opencv-run, --owlv2-run")
        dirs = {"manual": rooted(args.manual_run), "opencv": rooted(args.opencv_run), "owlv2": rooted(args.owlv2_run)}
        compare_runs(cfg, dirs, checked, target / "comparison", names=names)
        print("Report:", target / "comparison" / "E5_comparison_report.md")
        return 0
    target.mkdir(parents=True)
    dirs = {"manual": rooted(args.manual_run) if args.manual_run else target / "manual",
            "opencv": target / "opencv", "owlv2": target / "owlv2"}
    if not args.manual_run:
        run_inference(config, roi_paths["manual"], dirs["manual"], device=args.device, limit=args.limit)
    for method in ("opencv", "owlv2"):
        run_inference(config, roi_paths[method], dirs[method], device=args.device, limit=args.limit)
    result = compare_runs(cfg, dirs, checked, target / "comparison", names=names)
    print("\nComplete. Report:", target / "comparison" / "E5_comparison_report.md")
    print("Mean ROI IoU:", result["mean_roi_iou"])
    print("These numbers describe the 13-image DEVELOPMENT SET; no generalization claim.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        print("ERROR:", e, file=sys.stderr)
        raise SystemExit(2)
