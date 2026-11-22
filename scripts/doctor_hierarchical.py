#!/usr/bin/env python3
"""Offline diagnostic: identify which UrbanGroundFloor CLI is being loaded.

Works without image files or downloading SigLIP. Execute from any directory:
    python scripts/doctor_hierarchical.py
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))


def main() -> int:
    print('Project:', ROOT)
    print('Python:', sys.executable)
    print('Global urban-vlm executable:', shutil.which('urban-vlm') or '(not found)')
    try:
        import urban_vlm
        from urban_vlm.cli import parser
        from urban_vlm.hierarchical import load_hierarchical_prompts
        from urban_vlm.data import load_config, validate_inputs
        print('urban_vlm package:', Path(urban_vlm.__file__).resolve())
        print('Package version:', urban_vlm.__version__)
        cli_help = parser().format_help()
        print('hierarchical registered:', 'hierarchical' in cli_help)
        if 'hierarchical' not in cli_help:
            print('ERROR: This project does not contain the v0.2 CLI. Check the folder.')
            return 2
        prompt_file = ROOT / 'configs' / 'prompts_hierarchical.json'
        prompts = load_hierarchical_prompts(prompt_file)
        print('Hierarchical prompts:', sum(len(x) for stage in prompts.values() for x in stage.values()))
        config_file = ROOT / 'configs' / 'experiment.yaml'
        cfg = load_config(config_file)
        print('Model checkpoint:', cfg['model_id'])
        validate_inputs(cfg, require_images=False)
        print('Labels, ROIs, and base prompt banks: OK')
        missing = [f'fachada_{i:02d}.jpg' for i in range(1, 14)
                   if not (Path(cfg['images_dir']) / f'fachada_{i:02d}.jpg').is_file()]
        print('Images: 13/13 found' if not missing else 'Images missing: ' + ', '.join(missing))
        for name in ('torch', 'transformers'):
            print(f'{name}:', 'available' if importlib.util.find_spec(name) is not None else 'NOT INSTALLED')
        print('The above checks do not execute the pretrained model.')
        return 0
    except Exception as exc:
        print(type(exc).__name__ + ':', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
