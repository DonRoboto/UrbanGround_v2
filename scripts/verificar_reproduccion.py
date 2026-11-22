#!/usr/bin/env python3
"""Audit E0--E5 against frozen historical references; no model inference.

A complete run is compared by (photo, region, bank) rather than just accuracy.
Binary/float differences are reported separately. E2 has no archived historical
logits, so only completeness and within-run E0 invariants can be checked.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'reference/historical_2026_11_21'
KEYS = ('file', 'region', 'prompt_bank')
NAMES = {f'fachada_{i:02d}.jpg' for i in range(1,14)}


def rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Falta archivo: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def index_table(data, key_columns, filename):
    result = {}
    for row in data:
        key = tuple(row[c] for c in key_columns)
        if key in result:
            raise ValueError(f'Duplicado en {filename}: {key}')
        result[key] = row
    return result


class Audit:
    def __init__(self, tol):
        self.events = []
        self.tol = tol
        self.diff_rows = []

    def check(self, stage, name, ok, detail=''):
        self.events.append({'stage':stage, 'check':name, 'status':'PASS' if ok else 'FAIL', 'detail':str(detail)})
        print(f"[{'OK' if ok else 'ERROR'}] {stage}: {name}" + (f' — {detail}' if detail else ''), flush=True)
        return ok

    def compare_table(self, stage, current, historical, keys, string_cols, num_cols=(), tol=None):
        ref = index_table(rows(historical), keys, str(historical))
        got = index_table(rows(current), keys, str(current))
        self.check(stage,'claves de filas idénticas',set(ref)==set(got),
                   f'ref={len(ref)}, actual={len(got)}, ausentes={len(ref.keys()-got.keys())}, extra={len(got.keys()-ref.keys())}')
        differences = []
        for key in sorted(set(ref)&set(got)):
            for col in string_cols:
                if str(ref[key].get(col)) != str(got[key].get(col)):
                    differences.append((key,col,ref[key].get(col),got[key].get(col)))
            for col in num_cols:
                a,b = float(ref[key][col]),float(got[key][col])
                if not (math.isfinite(a) and math.isfinite(b) and abs(a-b) <= (tol if tol is not None else self.tol)):
                    differences.append((key,col,a,b))
        for key,col,a,b in differences:
            self.diff_rows.append({'stage':stage,'key':' / '.join(key),'field':col,'reference':a,'actual':b})
        self.check(stage, 'predicciones/valores coinciden', not differences, f'diferencias={len(differences)}; ejemplos={differences[:3]}')

    def summary(self):
        return dict(Counter(item['status'] for item in self.events))


def metrics_by_bank(stage, path, ref, audit):
    audit.compare_table(stage, path, ref, ('region','prompt_bank'), ('correct','n'),
                        ('accuracy','macro_f1','recall_A','recall_I','recall_N'), tol=1e-8)


def audit_reference_only(audit):
    baseline=rows(REF/'baseline/predictions.csv')
    e3=rows(REF/'e3/hierarchical_predictions.csv')
    e4=rows(REF/'e4/paired_e3_e4.csv')
    e5=rows(REF/'e5/paired_predictions.csv')
    e5_stats=rows(REF/'e5/metrics_summary.csv')
    audit.check('reference','E0: 52 predicciones',len(baseline)==52)
    audit.check('reference','E3: 26 predicciones',len(e3)==26)
    audit.check('reference','E4: 26 pares',len(e4)==26)
    audit.check('reference','E5: 26 pares',len(e5)==26)
    audit.check('reference','E5: 6 filas históricas ROI',len(e5_stats)==6)
    expected= {(n,r,b) for n in NAMES for r in ('full','roi') for b in ('generic','contextual')}
    audit.check('reference','E0: 13×2×2 sin duplicados',set(index_table(baseline,KEYS,'E0'))==expected)
    audit.check('reference','E5: 13×2 sin duplicados',set(index_table(e5,('file','prompt_bank'),'E5'))=={(n,b) for n in NAMES for b in ('generic','contextual')})
    ref_anns={r['file']:r['label'] for r in rows(ROOT/'data/annotations.csv')}
    audit.check('reference','etiquetas verificadas',all(x['label']==ref_anns[x['file']] for x in baseline))
    from itertools import product
    e5idx=index_table(e5,('file','prompt_bank'),'E5')
    e0idx=index_table(baseline,KEYS,'E0')
    audit.check('reference','E5 manual = E0 manual', all(e5idx[n,b]['manual_prediction']==e0idx[n,'roi',b]['predicted'] for n,b in product(sorted(NAMES),('generic','contextual'))))


def validate_e2(audit, path, baseline_path):
    e2=rows(path)
    exp={(n,b,r) for n in NAMES for b in ('generic','contextual') for r in ('original','roi_crop','blur_upper','blur_inside_roi','blur_outside_roi')}
    got=index_table(e2,('file','prompt_bank','region'),str(path))
    audit.check('E2','13×2×5 intervenciones',set(got)==exp, f'filas={len(e2)}')
    base=index_table(rows(baseline_path),KEYS,str(baseline_path))
    mismatches=[]
    for n in sorted(NAMES):
        for b in ('generic','contextual'):
            for iv,region in (('original','full'),('roi_crop','roi')):
                if (n,b,iv) in got and (n,region,b) in base:
                    a=got[n,b,iv]; c=base[n,region,b]
                    if a['predicted']!=c['predicted']:
                        mismatches.append(f'{n}/{b}/{iv}')
    audit.check('E2','controles original/roi_crop concuerdan con E0',not mismatches, f'desacuerdos={mismatches[:6]}')
    audit.check('E2','no se atribuye igualdad con logits históricos inexistentes',True,
                'E2 verifica cobertura y controles, no coincidencia numérica histórica')


def verify_checkpoint(audit, fullrun):
    refhash=json.loads((REF/'e5/comparison_manifest_redacted.json').read_text())['checkpoint_commit']
    for stage,path in (
        ('E0',fullrun/'E0_E4/E0_baseline/manifest.json'),
        ('E3',fullrun/'E0_E4/E3_hierarchical/manifest.json'),
        ('E4',fullrun/'E0_E4/E4_operation/manifest.json'),
    ):
        if not path.is_file():
            audit.check(stage,'manifiesto',False,'no existe')
            continue
        m=json.loads(path.read_text())
        actual=m.get('model_commit_hash_detected')
        # E4 manifest might carry checkpoint nested; only verify if this field exists.
        if stage=='E4' and not actual:
            continue
        audit.check(stage,'checkpoint congelado',actual==refhash,f'{actual} vs {refhash}')


def verify_complete(audit, fullrun):
    base=fullrun/'E0_E4'
    e5=fullrun/'E5'
    audit.compare_table('E0',base/'E0_baseline/predictions.csv',REF/'baseline/predictions.csv',KEYS,
                        ('label','predicted','correct'),('score_A','score_I','score_N'))
    audit.compare_table('E1',base/'E1_aggregation/aggregation_summary.csv',ROOT/'reference/reviewer_e1/aggregation_summary.csv',
                        ('region','prompt_bank','aggregation'),('n','correct'),('accuracy','macro_f1','recall_A','recall_I','recall_N'),tol=1e-8)
    validate_e2(audit,base/'E2_spatial/interventions.csv',base/'E0_baseline/predictions.csv')
    audit.compare_table('E3',base/'E3_hierarchical/hierarchical_predictions.csv',REF/'e3/hierarchical_predictions.csv',
                        KEYS,('label','predicted','presence_predicted','operation_predicted','correct'),
                        ('score_presence_commercial','score_presence_noncommercial','score_operation_active','score_operation_unverifiable'))
    audit.compare_table('E4',base/'E4_operation/paired_e3_e4.csv',REF/'e4/paired_e3_e4.csv',
                        ('file','region'),('human','e3_predicted','e4_predicted','e3_correct','e4_correct','changed_operation'),
                        ('e3_margin_operation','e4_margin_operation'))
    e5root=e5/'clasificacion/comparison'
    audit.compare_table('E5',e5root/'paired_predictions.csv',REF/'e5/paired_predictions.csv',
                        ('file','prompt_bank'),('label','manual_prediction','opencv_prediction','owlv2_prediction'),
                        ('iou_opencv','iou_owlv2'),tol=0.00001)
    metrics=rows(e5root/'metrics_summary.csv')
    refmetrics=rows(REF/'e5/metrics_summary.csv')
    audit.check('E5','8 condiciones presentes',len(metrics)==8,f'{len(metrics)} condiciones')
    idx=index_table(metrics,('method','prompt_bank'),'E5 metrics')
    refindex=index_table(refmetrics,('method','prompt_bank'),'reference E5 metrics')
    e0ref=rows(REF/'baseline/predictions.csv')
    for b in ('generic','contextual'):
        matching=[r for r in e0ref if r['region']=='full' and r['prompt_bank']==b]
        assert len(matching)==13
        correct=sum(r['label']==r['predicted'] for r in matching)
        actual=idx.get(('full_image',b))
        audit.check('E5',f'full_image/{b} frente a E0 referencia',actual is not None and
                    int(actual['correct'])==correct and int(actual['n'])==13)
    for key,expected in refindex.items():
        actual=idx.get(key)
        ok=(actual is not None and
            all(abs(float(actual[v])-float(expected[v]))<1e-8 for v in
                ('accuracy','macro_f1','recall_A','recall_I','recall_N')) and
            actual['correct']==expected['correct'])
        audit.check('E5',f'métricas {key[0]}/{key[1]}',ok)
    for name in ('opencv','owlv2'):
        current=e5/f"roi_{'opencv' if name=='opencv' else 'owlv2'}/rois_auto.csv"
        ref=REF/f'e5/{name}/rois_auto.csv'
        audit.compare_table('E5-ROI',current,ref,('file',),('x1','y1','x2','y2'))
    verify_checkpoint(audit,fullrun)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',help='Directorio generado por reproducir_todos_los_experimentos.sh')
    p.add_argument('--reference-only',action='store_true',help='Validar archivos históricos sin GPU ni fotos')
    p.add_argument('--score-tolerance',type=float,default=0.01,help='Tolerancia absoluta para logits (default 0.01)')
    p.add_argument('--soft',action='store_true',help='Generar informe aunque haya discrepancias; no alterar status de fallos')
    args=p.parse_args(argv)
    if not args.reference_only and not args.run:
        p.error('Elige --run o --reference-only')
    audit=Audit(args.score_tolerance)
    dest=Path(args.run).resolve()/'VERIFICATION' if args.run else None
    try:
        audit_reference_only(audit)
        if args.run:
            verify_complete(audit,Path(args.run).resolve())
    except Exception as exc:
        audit.check('SISTEMA','Excepción en auditoría',False,f'{type(exc).__name__}: {exc}')
    summary=audit.summary()
    if dest:
        dest.mkdir(parents=True, exist_ok=True)
        with (dest/'checks.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['stage','check','status','detail']);w.writeheader();w.writerows(audit.events)
        with (dest/'differences.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['stage','key','field','reference','actual']);w.writeheader();w.writerows(audit.diff_rows)
        (dest/'verification.json').write_text(json.dumps({'checked_utc':datetime.now(timezone.utc).isoformat(),
            'score_tolerance':args.score_tolerance,'summary':summary,'checks':audit.events},ensure_ascii=False,indent=2)+'\n')
        lines=['# Informe de reproducibilidad E0–E5','',
               f"**Estado:** {'CONCORDANTE' if not summary.get('FAIL') else 'DIFERENCIAS DETECTADAS'}.",
               f"**Verificaciones:** {summary.get('PASS',0)} correctas, {summary.get('FAIL',0)} fallidas.",
               '', 'Se comparan decisiones por fotografía, métricas, coordenadas de ROI y checkpoint. Las puntuaciones logit',
               f'admiten una tolerancia absoluta de {args.score_tolerance:g}, porque CPU/GPU pueden diferir numéricamente.',
               'Los resultados E2 solo permiten verificar integridad y coherencia con E0: no están archivados todos sus logits históricos.',
               'La selección de las 13 fotografías para diseño de prompts y ROI fue exploratoria; estos resultados no son held-out.',
               '','| Etapa | Comprobación | Estado | Detalle |','|---|---|---|---|']
        for x in audit.events:
            lines.append('| ' + ' | '.join(str(x[k]).replace('|','/') for k in ('stage','check','status','detail'))+' |')
        (dest/'reproduction_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('\nINFORME:',dest/'reproduction_report.md')
    print('RESUMEN:',summary)
    return 0 if args.soft or not summary.get('FAIL') else 2

if __name__=='__main__':
    raise SystemExit(main())
