#!/usr/bin/env bash
# Generate auditable scientific report from frozen 2026-11-21 development outputs, without GPU or photos.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${1:-runs/offline_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$OUT" ]] || { echo "ERROR: output exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
BASE="reference/historical_2026_11_21"
python -m urban_vlm ablate --logits "$BASE/baseline/prompt_logits.csv" \
  --predictions "$BASE/baseline/predictions.csv" --output "$OUT/E1_aggregation"
python -m urban_vlm analyze --predictions "$BASE/baseline/predictions.csv" \
  --logits "$BASE/baseline/prompt_logits.csv" --output "$OUT/CASE_ANALYSIS"
python -m urban_vlm report --baseline "$BASE/baseline/predictions.csv" \
  --e3 "$BASE/e3/hierarchical_predictions.csv" \
  --e4-paired "$BASE/e4/paired_e3_e4.csv" \
  --e4-logits "$BASE/e4/operation_e4_logits.csv" \
  --annotations data/annotations.csv --output "$OUT/FINAL_REPORT" "${@:2}" >/dev/null
printf 'Offline results: %s/FINAL_REPORT/study_report.md\n' "$OUT"
