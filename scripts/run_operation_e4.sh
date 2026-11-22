#!/usr/bin/env bash
# Siempre importa el paquete local; evita el comando global urban-vlm v0.1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m urban_vlm operation-e4 "$@"
