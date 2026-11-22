#!/usr/bin/env bash
# E5: auto ROI construction + QA + separate experimental inference.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
METHOD="${METHOD:-opencv}"
DEVICE="${DEVICE:-auto}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
OUT="${1:-runs/E5_auto_rois}"
if [[ -e "$OUT" ]]; then
  echo "ERROR: $OUT already exists. Choose a new output path." >&2
  exit 2
fi
python -m urban_vlm auto-rois --config "$CONFIG" --method "$METHOD" \
    --output "$OUT/localization"
python -m urban_vlm validate --config "$CONFIG" \
    --rois-file "$OUT/localization/rois_auto.csv"
python -m urban_vlm preview-rois --config "$CONFIG" \
    --rois-file "$OUT/localization/rois_auto.csv" \
    --output "$OUT/roi_review_for_siglip.png"
printf '\nROI candidates generated. Inspect %s before inference.\n' "$OUT/roi_review_for_siglip.png"
if [[ "${RUN_INFERENCE:-0}" == "1" ]]; then
  python -m urban_vlm run --config "$CONFIG" \
      --rois-file "$OUT/localization/rois_auto.csv" \
      --device "$DEVICE" --output "$OUT/E5_inference"
fi
