"""Ensure the reviewer audit passes on valid files and fails on tampered results."""
from __future__ import annotations
import csv
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('verification', PROJECT_ROOT/'scripts/verificar_reproduccion.py')
VERIFY=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def save(path, data):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=['file','predicted','score_A']); w.writeheader(); w.writerows(data)


def test_comparator_detects_prediction_mutation(tmp_path):
    ref=tmp_path/'ref.csv'; new=tmp_path/'new.csv'
    save(ref,[{'file':'facade.jpg','predicted':'A','score_A':'.4'}])
    save(new,[{'file':'facade.jpg','predicted':'N','score_A':'.4'}])
    audit=VERIFY.Audit(.01)
    audit.compare_table('test',new,ref,('file',),('predicted',),('score_A',))
    assert audit.summary().get('FAIL')==1
    assert audit.diff_rows[0]['field']=='predicted'


def test_comparator_allows_small_float_differences(tmp_path):
    ref=tmp_path/'ref.csv'; new=tmp_path/'new.csv'
    save(ref,[{'file':'facade.jpg','predicted':'A','score_A':'.4'}])
    save(new,[{'file':'facade.jpg','predicted':'A','score_A':'.402'}])
    audit=VERIFY.Audit(.01)
    audit.compare_table('test',new,ref,('file',),('predicted',),('score_A',))
    assert audit.summary().get('FAIL',0)==0


def test_reference_manifest_self_consistency():
    audit=VERIFY.Audit(.01)
    VERIFY.audit_reference_only(audit)
    assert audit.summary().get('FAIL',0)==0
    assert audit.summary()['PASS']>=8


def test_cli_exposes_owl_revision():
    from urban_vlm.cli import parser
    args=parser().parse_args(['auto-rois','--output','anywhere','--owl-revision','abc123'])
    assert args.owl_revision=='abc123'
