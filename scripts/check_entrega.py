#!/usr/bin/env python3
"""Audita el repositorio Parte B sin GPU. --require-images exige los 13 originales exactos."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NAMES = [f"fachada_{i:02d}.jpg" for i in range(1, 14)]
M = ROOT / "reference/historical_2026_11_21/e5/owlv2/manifest.json"


def read_csv(p):
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def audit(require_images=False):
    failures = []
    def check(ok, message):
        (print("[OK]" if ok else "[ERROR]", message))
        if not ok: failures.append(message)

    for r in ("README.md", "requirements.txt", "pyproject.toml", "nota_tecnica.md",
              "nota_tecnica.pdf", "anotaciones.csv", "data/annotations.csv", "data/rois.csv",
              "src/urban_vlm/cli.py", "configs/prompts_generic.json", "configs/prompts_contextual.json",
              "reference/historical_2026_11_21/e5/metrics_summary.csv"):
        check((ROOT / r).is_file(), "Archivo presente: " + r)
    if failures:
        return False
    check(digest(ROOT / "data/annotations.csv") == digest(ROOT / "anotaciones.csv"),
          "anotaciones.csv es copia byte a byte de data/annotations.csv")
    anns = read_csv(ROOT / "data/annotations.csv")
    rois = read_csv(ROOT / "data/rois.csv")
    check(len(anns) == 13 and set(r["file"] for r in anns) == set(NAMES) and len(set(r["file"] for r in anns)) == 13,
          "13 etiquetas y nombres únicos")
    check(Counter(r["label"] for r in anns) == Counter({"A": 7, "I": 5, "N": 1}),
          "Distribución de etiquetas 7 A / 5 I / 1 N")
    check(len(rois) == 13 and {r["file"] for r in rois} == set(NAMES), "13 ROI manuales")
    for b in ("generic", "contextual"):
        d = json.loads((ROOT / f"configs/prompts_{b}.json").read_text())
        check(set(d) == {"A", "I", "N"} and all(len(d[c]) == 5 for c in d),
              f"Banco {b}: tres clases con 5 prompts")
    bdir = ROOT / "reference/historical_2026_11_21/baseline"
    baseline = read_csv(bdir / "predictions.csv")
    logits = read_csv(bdir / "prompt_logits.csv")
    check(len(baseline) == 52 and len(logits) == 780, "Históricos E0: 52 decisiones, 780 logits")
    check(len({(r["file"],r["region"],r["prompt_bank"]) for r in baseline}) == 52,
          "Sin decisiones E0 duplicadas")
    e5dir = ROOT / "reference/historical_2026_11_21/e5"
    e5 = read_csv(e5dir / "metrics_summary.csv")
    pp = read_csv(e5dir / "paired_predictions.csv")
    check(len(e5) == 6 and len(pp) == 26,
          "E5: 6 condiciones y 26 comparaciones pareadas")
    ck = json.loads((e5dir / "comparison_manifest_redacted.json").read_text())
    check(ck.get("partial_smoke_test") is False and ck.get("n_images") == 13,
          "E5 no es prueba parcial")
    check(ck.get("checkpoint_commit") == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
          "Commit SigLIP 2 guardado coincide con configuración de referencia")
    actual = {r["file"]:r for r in anns}
    check(all(r["label"] == actual[r["file"]]["label"] for r in baseline),
          "Etiquetas históricas E0 coinciden con anotaciones entregadas")
    expected=json.loads(M.read_text()).get("images_sha256", {})
    imdir=ROOT / "data/images"
    photos=[imdir / n for n in NAMES]
    n_present=sum(p.is_file() for p in photos)
    if n_present==13:
        try:
            from PIL import Image
            wrong=[]
            for p in photos:
                with Image.open(p) as im:
                    if im.size != (640,640): wrong.append(f"{p.name}: tamaño {im.size}")
                if expected.get(p.name) != digest(p): wrong.append(f"{p.name}: SHA256 distinto al archivo original utilizado en E5")
            check(not wrong, "13 imágenes auténticas 640x640 y SHA256 coincidente con E5")
            for e in wrong:print('  ',e)
        except ImportError:
            check(False,"Instala Pillow para verificar imágenes: pip install -r requirements.txt")
    else:
        missing=[p.name for p in photos if not p.exists()]
        if require_images:
            check(False,f"Faltan {len(missing)} imágenes originales: {', '.join(missing[:5])}")
        else:
            print(f"[PENDIENTE] Imágenes originales presentes: {n_present}/13. Para publicación con datos, necesitas autorización y --require-images.")
    print('\nRESULTADO: ', 'FALTAN REQUISITOS' if failures else ('ESTRUCTURA VERIFICADA (SIN IMÁGENES)' if n_present!=13 else 'DATOS Y ESTRUCTURA VERIFICADOS'))
    return not failures

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-images',action='store_true',help='falla si faltan imágenes o no coinciden con los SHA256 históricos')
    args=p.parse_args()
    sys.exit(0 if audit(require_images=args.require_images) else 1)
