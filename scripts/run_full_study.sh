#!/usr/bin/env bash
# Full E0–E4 study. No training. Never overwrite a previous experiment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR="${1:-runs/study_$(date -u +%Y%m%dT%H%M%SZ)}"
DEVICE="${DEVICE:-auto}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
PLOTS="${PLOTS:-0}"
if [[ -e "$RUN_DIR" ]]; then
  echo "ERROR: output exists, choose a new run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
{
  echo "[0/8] Validating 13 unmodified photographs and ROI annotations"
  python -m urban_vlm validate --config "$CONFIG"
  python -m urban_vlm preview-rois --config "$CONFIG" --output "$RUN_DIR/roi_review.png"
  echo "[1/8] E0 — Full image / ROI, generic / contextual prompts"
  python -m urban_vlm run --config "$CONFIG" --device "$DEVICE" --output "$RUN_DIR/E0_baseline"
  echo "[2/8] E1 — Aggregation ablations from E0 logits (no inference)"
  python -m urban_vlm ablate --logits "$RUN_DIR/E0_baseline/prompt_logits.csv" \
    --predictions "$RUN_DIR/E0_baseline/predictions.csv" --output "$RUN_DIR/E1_aggregation"
  echo "[3/8] E2 — Spatial interventions on full set"
  python -m urban_vlm diagnose --config "$CONFIG" --device "$DEVICE" --output "$RUN_DIR/E2_spatial"
  echo "[4/8] E3 — Two-stage presence/operation"
  python -m urban_vlm hierarchical --config "$CONFIG" \
    --prompts configs/prompts_hierarchical.json --region both \
    --aggregation mean --device "$DEVICE" --output "$RUN_DIR/E3_hierarchical"
  echo "[5/8] E4 — Refine only operation prompts, reuse E3 presence logits"
  python -m urban_vlm operation-e4 --config "$CONFIG" \
    --e3-dir "$RUN_DIR/E3_hierarchical" --region both --aggregation mean \
    --device "$DEVICE" --output "$RUN_DIR/E4_operation"
  echo "[6/8] Case analysis of baseline and E3"
  python -m urban_vlm analyze --predictions "$RUN_DIR/E0_baseline/predictions.csv" \
    --logits "$RUN_DIR/E0_baseline/prompt_logits.csv" \
    --interventions "$RUN_DIR/E2_spatial/interventions.csv" \
    --hierarchical "$RUN_DIR/E3_hierarchical/hierarchical_predictions.csv" \
    --output "$RUN_DIR/case_analysis"
  echo "[7/8] Scientific E0–E4 report with per-case evidence"
  OPTS=()
  if [[ "$PLOTS" == "1" ]]; then OPTS+=(--plots); fi
  python -m urban_vlm report --baseline "$RUN_DIR/E0_baseline/predictions.csv" \
    --e3 "$RUN_DIR/E3_hierarchical/hierarchical_predictions.csv" \
    --e4-paired "$RUN_DIR/E4_operation/paired_e3_e4.csv" \
    --e4-logits "$RUN_DIR/E4_operation/operation_e4_logits.csv" \
    --spatial-effects "$RUN_DIR/E2_spatial/intervention_effects.csv" \
    --annotations data/annotations.csv --output "$RUN_DIR/FINAL_REPORT" "${OPTS[@]}"
  echo "COMPLETED: $RUN_DIR"
  echo "Review: $RUN_DIR/FINAL_REPORT/study_report.md"
} 2>&1 | tee "$RUN_DIR/execution.log"
