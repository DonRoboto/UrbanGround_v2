#!/usr/bin/env bash
# Run the hierarchical classifier from this checkout, independent of an
# older globally installed `urban-vlm` command.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON:-python}"
exec "$PYTHON_BIN" -m urban_vlm hierarchical "$@"
