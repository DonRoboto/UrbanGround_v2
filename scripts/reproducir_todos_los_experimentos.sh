#!/usr/bin/env bash
# Reproducción completa de la Parte B: E0–E5, sobre las 13 fotografías originales.
# No reentrena modelos y no altera data/rois.csv ni los datos de entrada.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${1:-runs/reproduccion_$(date -u +%Y%m%dT%H%M%SZ)}"
DEVICE="${DEVICE:-auto}"
OWL_DEVICE="${OWL_DEVICE:-$DEVICE}"
CONFIG="${CONFIG:-configs/experiment_pinned.yaml}"
PLOTS="${PLOTS:-1}"
ROI_SOURCE="${ROI_SOURCE:-regenerate}" # regenerate | historical
OWL_REVISION="${OWL_REVISION:-cfd3195ba4ea9592eec887ded089f4c08eff231d}"
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.01}"
if [[ "$ROI_SOURCE" != "regenerate" && "$ROI_SOURCE" != "historical" ]]; then
  echo "ROI_SOURCE debe ser regenerate o historical" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "ERROR: el directorio de salida ya existe: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
python -m pip freeze > "$OUT/environment.freeze.txt"
python --version > "$OUT/python_version.txt" 2>&1
{
  echo "UrbanGroundFloor | reproduccion integral E0-E5 | $(date -u +%FT%TZ)"
  echo "CONFIG=$CONFIG DEVICE=$DEVICE OWL_DEVICE=$OWL_DEVICE ROI_SOURCE=$ROI_SOURCE"
  echo "[1/7] Verificando archivos y hashes de las 13 fotografias originales"
  python scripts/check_entrega.py --require-images
  python -m urban_vlm validate --config "$CONFIG"
  echo "[2/7] E0-E4: formulaciones, agregacion, intervenciones y jerarquia"
  CONFIG="$CONFIG" DEVICE="$DEVICE" PLOTS="$PLOTS" \
    bash scripts/run_full_study.sh "$OUT/E0_E4"
  if [[ "$ROI_SOURCE" == "regenerate" ]]; then
    echo "[3/8] E5-A: regenerar ROI con OpenCV"
    python -m urban_vlm auto-rois --config "$CONFIG" --method opencv \
      --output "$OUT/E5/roi_opencv"
    echo "[4/8] E5-B: regenerar ROI con OWLv2 (commit congelado)"
    python -m urban_vlm auto-rois --config "$CONFIG" --method owlv2 \
      --owl-revision "$OWL_REVISION" --owl-device "$OWL_DEVICE" \
      --output "$OUT/E5/roi_owlv2"
  else
    echo "[3-4/8] E5: usar ROI historicas congeladas (reproduccion de inferencia; sin regenerar detectores)"
    mkdir -p "$OUT/E5/roi_opencv" "$OUT/E5/roi_owlv2"
    cp reference/historical_2026_11_21/e5/opencv/{rois_auto.csv,manifest.json,roi_diagnostics.csv,summary.json} "$OUT/E5/roi_opencv/"
    cp reference/historical_2026_11_21/e5/owlv2/{rois_auto.csv,manifest.json,roi_diagnostics.csv,summary.json} "$OUT/E5/roi_owlv2/"
  fi
  echo "[5/8] Validacion de ROI y procedencia antes de clasificar"
  python scripts/run_e5_siglip2.py --config "$CONFIG" \
    --opencv-rois "$OUT/E5/roi_opencv/rois_auto.csv" \
    --owlv2-rois "$OUT/E5/roi_owlv2/rois_auto.csv" --dry-run
  echo "[6/8] Clasificacion E5: imagen completa y tres ROI, dos bancos de prompts"
  python scripts/run_e5_siglip2.py --config "$CONFIG" \
    --opencv-rois "$OUT/E5/roi_opencv/rois_auto.csv" \
    --owlv2-rois "$OUT/E5/roi_owlv2/rois_auto.csv" \
    --device "$DEVICE" --output "$OUT/E5/clasificacion"
  echo "[7/8] Verificar E0--E5 contra los resultados historicos, por fotografia"
  python scripts/verificar_reproduccion.py --run "$OUT" --score-tolerance "$SCORE_TOLERANCE"
  echo "[8/8] Finalizado; informes principales:"
  echo " - $OUT/E0_E4/FINAL_REPORT/study_report.md"
  echo " - $OUT/E5/clasificacion/comparison/E5_comparison_report.md"
  echo " - $OUT/E5/clasificacion/comparison/metrics_summary.csv"
  echo " - $OUT/VERIFICATION/reproduction_report.md"
  echo "Nota: son resultados exploratorios de las 13 imagenes de desarrollo."
} 2>&1 | tee "$OUT/reproducir_todos.log"
