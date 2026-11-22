#!/usr/bin/env bash
# CPU-only: test code, archived file shape, and generation of a complete E0-E4 report.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
bash scripts/run_offline_audit.sh "runs/repro_check_$(date -u +%Y%m%dT%H%M%S)_$$"
if [[ -f data/images/fachada_01.jpg ]]; then
  python -m urban_vlm validate --config configs/experiment.yaml
else
  echo 'Photos absent: offline reproducibility verified; real-image inference NOT tested.'
fi
