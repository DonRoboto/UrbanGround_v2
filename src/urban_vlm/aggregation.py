#!/usr/bin/env python3
"""Auditoría EXPLORATORIA de agregación textual con logits SigLIP 2 guardados.

No realiza inferencia ni ajusta pesos. Compara reglas fijas (mean, median,
max, top2, logmeanexp) sin seleccionar un ganador a partir de estas 13 fotos.
No utilice estos resultados como desempeño de generalización.

Ejemplo:
  python scripts/ablate_prompt_aggregation.py \\
    --logits runs/full_13/prompt_logits.csv \\
    --predictions runs/full_13/predictions.csv \\
    --out runs/ablation_aggregation
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

CLASSES = ('A', 'I', 'N')
METHODS = ('mean', 'median', 'top2', 'max', 'logmeanexp')
KEYS = ('file', 'region', 'prompt_bank')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: faltan columnas: {sorted(missing)}')
        return list(reader)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values: list[float], method: str) -> float:
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Logits inválidos o ausentes.')
    ordered = sorted(values)
    if method == 'mean':
        return float(mean(ordered))
    if method == 'median':
        return float(median(ordered))
    if method == 'top2':
        if len(values) < 2:
            raise ValueError('Top-2 requiere al menos dos prompts por clase.')
        return float(mean(ordered[-2:]))
    if method == 'max':
        return float(ordered[-1])
    if method == 'logmeanexp':
        high = ordered[-1]
        return float(high + math.log(sum(math.exp(v-high) for v in ordered)/len(ordered)))
    raise ValueError(f'Método desconocido: {method}')


def macro_f1(actual: list[str], predicted: list[str]) -> float:
    parts=[]
    for c in CLASSES:
        tp=sum(a==c and p==c for a,p in zip(actual,predicted))
        fp=sum(a!=c and p==c for a,p in zip(actual,predicted))
        fn=sum(a==c and p!=c for a,p in zip(actual,predicted))
        den=2*tp+fp+fn
        parts.append(2*tp/den if den else 0.)
    return sum(parts)/len(parts)


def run_aggregation(logits_path:Path,predictions_path:Path,out:Path) -> dict:
    logits_rows=read_csv(logits_path, set(KEYS)|{'class','prompt_index','prompt_text','logit'})
    pred_rows=read_csv(predictions_path, set(KEYS)|{'label','predicted','score_A','score_I','score_N'})
    labels={}; original_scores={}
    for row in pred_rows:
        key=tuple(row[k] for k in KEYS)
        if key in labels:
            raise ValueError(f'Predicción duplicada: {key}')
        if row['label'] not in CLASSES or row['predicted'] not in CLASSES:
            raise ValueError(f'Etiqueta desconocida: {row}')
        labels[key]=row['label']
        original_scores[key]={c:float(row['score_'+c]) for c in CLASSES}
    logits=defaultdict(list); per_class_index=defaultdict(set)
    for row in logits_rows:
        key=tuple(row[k] for k in KEYS)
        c=row['class']
        if key not in labels or c not in CLASSES:
            raise ValueError(f'Grupo sin etiqueta/clase válida: {key},{c}')
        idx=int(row['prompt_index'])
        if idx in per_class_index[key,c]:
            raise ValueError(f'Índice de prompt duplicado: {key},{c},{idx}')
        per_class_index[key,c].add(idx)
        logits[key,c].append(float(row['logit']))
    results=[]
    for key in sorted(labels):
        groups=[logits.get((key,c),[]) for c in CLASSES]
        counts=[len(z) for z in groups]
        if min(counts)<2 or len(set(counts))!=1:
            raise ValueError(f'Bancos por clase incompletos/desbalanceados: {key}: {counts}')
        for c,z in zip(CLASSES,groups):
            if abs(aggregate(z,'mean')-original_scores[key][c])>1e-4:
                raise ValueError(f'Inconsistencia entre logits y score de predicción: {key}, {c}')
        for method in METHODS:
            scores={c:aggregate(z,method) for c,z in zip(CLASSES,groups)}
            pred=max(CLASSES,key=lambda c:scores[c])
            ordered=sorted(scores.values(),reverse=True)
            result=dict(zip(KEYS,key))
            result.update(aggregation=method,label=labels[key],predicted=pred,
                          correct=int(pred==labels[key]),
                          score_A=scores['A'],score_I=scores['I'],score_N=scores['N'],
                          margin=ordered[0]-ordered[1])
            results.append(result)
    if len({(r['file'],r['region'],r['prompt_bank'],r['aggregation']) for r in results})!=len(results):
        raise ValueError('Grupos repetidos.')
    out.mkdir(parents=True,exist_ok=True)
    columns=list(KEYS)+['aggregation','label','predicted','correct','score_A','score_I','score_N','margin']
    write_csv(out/'aggregation_predictions.csv',results,columns)
    group_rows=[]
    by_group=defaultdict(list)
    for r in results:
        by_group[(r['region'],r['prompt_bank'],r['aggregation'])].append(r)
    for (region,bank,method),gr in sorted(by_group.items()):
        correct=sum(r['correct'] for r in gr)
        actual=[r['label'] for r in gr]; pred=[r['predicted'] for r in gr]
        detail={'region':region,'prompt_bank':bank,'aggregation':method,'n':len(gr),'correct':correct,
                'accuracy':correct/len(gr),'macro_f1':macro_f1(actual,pred)}
        for c in CLASSES:
            instances=[r for r in gr if r['label']==c]
            detail[f'recall_{c}']=(sum(r['predicted']==c for r in instances)/len(instances)) if instances else None
        group_rows.append(detail)
    summary_columns=['region','prompt_bank','aggregation','n','correct','accuracy','macro_f1','recall_A','recall_I','recall_N']
    write_csv(out/'aggregation_summary.csv',group_rows,summary_columns)
    mean_ref={(r['file'],r['region'],r['prompt_bank']):r for r in results if r['aggregation']=='mean'}
    changes=[]
    for r in results:
        if r['aggregation']=='mean':continue
        baseline=mean_ref[(r['file'],r['region'],r['prompt_bank'])]
        if r['predicted']!=baseline['predicted']:
            changes.append({**{k:r[k] for k in KEYS}, 'label':r['label'],
                            'aggregation':r['aggregation'],'mean_prediction':baseline['predicted'],
                            'new_prediction':r['predicted'],'mean_correct':baseline['correct'],'new_correct':r['correct']})
    change_columns=list(KEYS)+['label','aggregation','mean_prediction','new_prediction','mean_correct','new_correct']
    write_csv(out/'changes_vs_mean.csv',changes,change_columns)
    manifest={
        'analysis':'post-hoc exploratory aggregation ablation (NO independent validation)',
        'logits_sha256':sha256(logits_path),'predictions_sha256':sha256(predictions_path),
        'model_reinference':False,'weights_updated':False,'aggregation_methods':list(METHODS),
        'analysis_post_hoc':True,
        'warning':'Do not select the best aggregator on these 13 cases and report it as generalization.',
        'n_cases':len(labels),'n_logit_rows':len(logits_rows),
        'n_results':len(results)
    }
    (out/'aggregation_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logits',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    manifest=run_aggregation(args.logits,args.predictions,args.out)
    print(f"Evaluadas {manifest['n_cases']} combinaciones foto×región×banco; {manifest['n_logit_rows']} logits")
    print(f'Archivos guardados en {args.out}')
    print('IMPORTANTE: resultados exploratorios post hoc; sin validación independiente.')


if __name__=='__main__':
    main()
