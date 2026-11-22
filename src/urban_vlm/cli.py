"""CLI: inspect reproducibility, infer, diagnose, evaluate without changing labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from .data import load_config, validate_inputs, read_annotations
from .pipeline import run_experiment, run_diagnostics
from .previews import create_roi_preview
from .evaluation import load_predictions, write_report


def parser():
    p=argparse.ArgumentParser(prog="urban-vlm", description="13-facade ground-floor commerce with SigLIP 2")
    sub=p.add_subparsers(dest="command", required=True)
    for name in ("validate", "preview-rois", "run", "diagnose", "hierarchical", "operation-e4", "auto-rois"):
        s=sub.add_parser(name)
        s.add_argument("--config", default="configs/experiment.yaml")
        s.add_argument("--rois-file", default=None,
                       help="Use an alternative ROI CSV (auto-rois output). Original data/rois.csv is never overwritten.")
        if name in ("preview-rois", "run", "diagnose", "hierarchical", "operation-e4"):
            s.add_argument("--output", default=None)
        if name in ("run", "diagnose", "hierarchical", "operation-e4"):
            s.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
        if name == "run":
            s.add_argument("--limit", type=int, help="Smoke test first N images only; not an evaluation")
        if name == "diagnose":
            s.add_argument("--files", nargs="+", help="Optional image filenames e.g. fachada_05.jpg")
        if name == "hierarchical":
            s.add_argument("--prompts", default="configs/prompts_hierarchical.json")
            s.add_argument("--region", choices=["full", "roi", "both"], default="roi")
            s.add_argument("--aggregation", choices=["mean","median","top2","max","logmeanexp"], default="mean")
            s.add_argument("--limit", type=int, help="Quick smoke test, not a 13-image evaluation")
        if name == "auto-rois":
            s.add_argument("--output", required=True, help="New output directory for rois_auto.csv and QA sheet")
            s.add_argument("--method", choices=["opencv", "owlv2"], default="opencv")
            s.add_argument("--center-prior", type=float, default=0.50,
                           help="Normalized x of the target frontage, when known, default 0.5")
            s.add_argument("--min-width-ratio", type=float, default=0.28)
            s.add_argument("--owl-threshold", type=float, default=0.12)
            s.add_argument("--owl-model", default="google/owlv2-base-patch16-ensemble")
            s.add_argument("--owl-revision", default=None, help="Fixed Hugging Face commit for reproducible OWLv2 ROIs")
            s.add_argument("--owl-device", choices=["auto", "cuda", "cpu"], default="auto")
            s.add_argument("--files", nargs="+", default=None,
                           help="Optional filenames for a small test (13 is default)")
            s.add_argument("--no-manual-comparison", action="store_true",
                           help="Do not calculate IoU with existing manually drawn ROIs")
        if name == "operation-e4":
            s.add_argument("--e3-dir", required=True,
                           help="Existing E3 run containing hierarchical_predictions.csv, hierarchical_logits.csv and manifest.json")
            s.add_argument("--e3-prompts", default="configs/prompts_hierarchical.json",
                           help="Original E3 prompts (presence and operation), unchanged")
            s.add_argument("--prompts", default="configs/prompts_operation_e4.json",
                           help="E4 stage-2 prompts, five texts per hypothesis")
            s.add_argument("--region", choices=["full", "roi", "both"], default="roi")
            s.add_argument("--aggregation", choices=["mean"], default="mean",
                           help="E4 fixes mean aggregation to isolate new operation wording")
            s.add_argument("--limit", type=int, help="Smoke test first N images from E3")
            s.add_argument("--dry-run", action="store_true",
                           help="Validate E3 hashes/inputs without loading SigLIP 2")
    c=sub.add_parser("compare", help="Compare a newly run CSV against historical decisions")
    c.add_argument("--reference", default="reference/prior_exploratory_predictions.csv")
    c.add_argument("--current", required=True)
    c.add_argument("--annotations", default="data/annotations.csv")
    c.add_argument("--output", required=True)
    e=sub.add_parser("evaluate", help="Re-evaluate CSV saved from this project or historical prediction CSV")
    e.add_argument("--predictions", required=True)
    e.add_argument("--annotations", default="data/annotations.csv")
    e.add_argument("--output", required=True)
    b=sub.add_parser("ablate", help="Reaggregate saved 15-prompt logits without model inference")
    b.add_argument("--logits", required=True)
    b.add_argument("--predictions", required=True)
    b.add_argument("--output", required=True)
    x=sub.add_parser("analyze", help="Evidence-grounded per-case report from existing model outputs")
    x.add_argument("--predictions", required=True)
    x.add_argument("--logits", required=True)
    x.add_argument("--annotations", default="data/annotations.csv")
    x.add_argument("--interventions", default=None)
    x.add_argument("--hierarchical", default=None)
    x.add_argument("--output", required=True)
    r=sub.add_parser("report", help="Offline consolidated E0-E4 report, figures, provenance and per-case explanations")
    r.add_argument("--baseline", required=True, help="E0 saved predictions.csv (full+ROI, both prompt banks)")
    r.add_argument("--annotations", default="data/annotations.csv")
    r.add_argument("--e3", default=None, help="E3 hierarchical_predictions.csv, optional")
    r.add_argument("--e4-paired", default=None, help="E4 paired_e3_e4.csv, optional (requires --e3)")
    r.add_argument("--e4-logits", default=None, help="E4 operation_e4_logits.csv, optional")
    r.add_argument("--spatial-effects", default=None, help="E2 intervention_effects.csv, optional")
    r.add_argument("--plots", action="store_true", help="Save accuracy chart and ROI confusion matrices (matplotlib)")
    r.add_argument("--output", required=True, help="New, empty report directory")
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    try:
        if a.command == "report":
            from .study_report import build_study_report
            result = build_study_report(
                a.baseline, a.annotations, a.output,
                e3=a.e3, e4_paired=a.e4_paired, e4_logits=a.e4_logits,
                spatial_effects=a.spatial_effects, plots=a.plots,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if a.command == "ablate":
            from .aggregation import run_aggregation
            print(json.dumps(run_aggregation(Path(a.logits), Path(a.predictions), Path(a.output)), indent=2, ensure_ascii=False))
            return 0
        if a.command == "analyze":
            from .analysis import build_case_analysis
            print(json.dumps(build_case_analysis(a.predictions, a.logits, a.annotations,
                a.output, a.interventions, a.hierarchical), indent=2, ensure_ascii=False))
            return 0
        if a.command == "compare":
            from .comparison import compare_runs
            obj=compare_runs(a.reference,a.current,a.annotations,a.output)
            print(json.dumps(obj,indent=2,ensure_ascii=False))
            return 0
        if a.command == "evaluate":
            preds = load_predictions(a.predictions)
            ann = read_annotations(a.annotations)
            if any(p["file"] not in ann for p in preds):
                raise ValueError("Predictions refer to an unannotated image")
            # Use external annotation labels; never trust labels encoded in prediction files.
            for p in preds:
                p["label"] = ann[p["file"]]["label"]
            res=write_report(a.output, preds, ann, provenance="evaluation of provided predictions")
            print(json.dumps(res["summary"], indent=2, ensure_ascii=False))
            return 0
        cfg=load_config(a.config)
        if getattr(a, "rois_file", None):
            root=Path(cfg["project_root"])
            override=Path(a.rois_file)
            override=override if override.is_absolute() else root / override
            if not override.is_file():
                raise FileNotFoundError(f"Alternative ROI file not found: {override}")
            cfg["rois"]=str(override.resolve())
        if a.command == "auto-rois":
            from .auto_roi import AutoROIOptions, run_auto_roi
            opts=AutoROIOptions(method=a.method, center_prior=a.center_prior,
                                min_width_ratio=a.min_width_ratio,
                                owl_threshold=a.owl_threshold, owl_model=a.owl_model, owl_revision=a.owl_revision,
                                owl_device=a.owl_device)
            out=Path(a.output)
            if not out.is_absolute():
                out=Path(cfg["project_root"])/out
            manual=None if a.no_manual_comparison else cfg["rois"]
            result=run_auto_roi(cfg,out,opts,files=a.files,manual_rois=manual)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0
        if a.command == "operation-e4":
            from .operation_e4 import run_operation_e4, validate_e4_inputs
            def rooted(path):
                p=Path(path)
                return p if p.is_absolute() else Path(cfg["project_root"]) / p
            e3 = rooted(a.e3_dir)
            e3_prompts = rooted(a.e3_prompts)
            prompts = rooted(a.prompts)
            if a.dry_run:
                print(json.dumps(validate_e4_inputs(cfg,e3,e3_prompts,prompts,
                    region=a.region,limit=a.limit),indent=2,ensure_ascii=False))
                return 0
            date=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
            folder = rooted(a.output or f"runs/operation_e4_{date}")
            obj=run_operation_e4(cfg,e3,e3_prompts,prompts,folder,
                region=a.region,aggregation=a.aggregation,device=a.device,limit=a.limit)
            print(json.dumps(obj,indent=2,ensure_ascii=False))
            return 0
        if a.command == "validate":
            result=validate_inputs(cfg)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif a.command == "preview-rois":
            validate_inputs(cfg)
            p=create_roi_preview(cfg, a.output or "runs/roi_contact_sheet.png")
            print("ROI preview:",p)
        elif a.command in ("run", "diagnose", "hierarchical"):
            date=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
            folder = a.output or f"runs/{a.command}_{date}"
            if a.command == "run":
                res=run_experiment(cfg, folder, device=a.device, limit=a.limit)
            elif a.command == "diagnose":
                res=run_diagnostics(cfg, folder, device=a.device, files=a.files)
            else:
                from .hierarchical import run_hierarchical
                prompt_file=Path(a.prompts)
                if not prompt_file.is_absolute():
                    prompt_file=Path(cfg["project_root"]) / prompt_file
                res=run_hierarchical(cfg, prompt_file, folder, device=a.device,
                                     region=a.region, aggregation=a.aggregation, limit=a.limit)
            print(json.dumps(res,indent=2,ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
