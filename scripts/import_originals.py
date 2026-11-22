#!/usr/bin/env python3
"""Importa los 13 JPG originales sin transformar ni recomprimir; valida SHA256 histórico."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
NAMES=[f'fachada_{i:02d}.jpg' for i in range(1,14)]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path,help='Carpeta local con 13 JPG tal como se entregaron')
    args=ap.parse_args()
    expected=json.loads((ROOT/'reference/historical_2026_11_21/e5/owlv2/manifest.json').read_text())['images_sha256']
    source=args.source.expanduser().resolve()
    if not source.is_dir(): ap.error(f'Carpeta no encontrada: {source}')
    entries=[]
    for name in NAMES:
        p=source/name
        if not p.is_file():ap.error(f'Falta {p}')
        with Image.open(p) as im:
            if im.size!=(640,640):ap.error(f'{name}: dimensiones {im.size}, deben ser 640x640')
        if (h:=digest(p))!=expected.get(name):
            ap.error(f'{name}: SHA256 {h} difiere de los datos usados en los experimentos. No mezclar imágenes; buscar los originales exactos.')
        entries.append((name,p,h))
    d=ROOT/'data/images';d.mkdir(exist_ok=True)
    for name,p,h in entries:
        out=d/name
        if out.exists():
            if digest(out)!=h:ap.error(f'{out} ya existe con otros bytes; no se sobrescribe')
            print('Ya existe, idéntico:',name)
        else:
            shutil.copyfile(p,out)
            print('Copiado sin alterar:',name)
    print('Las 13 fotografías originales coinciden con las huellas E5. Revisa los permisos antes de distribuirlas públicamente.')

if __name__=='__main__':main()
