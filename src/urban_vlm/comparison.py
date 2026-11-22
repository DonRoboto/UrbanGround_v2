"""Compare a new inference run with saved historical decisions, not score calibrations."""
from __future__ import annotations
from pathlib import Path
from .data import read_annotations, write_csv
from .evaluation import load_predictions, evaluate_predictions


def compare_runs(reference: str | Path, current: str | Path, annotations: str | Path, output: str | Path) -> dict:
    old, new = load_predictions(reference), load_predictions(current)
    ann = read_annotations(annotations)
    first = {(r['file'], r['region'], r['prompt_bank']): r for r in old}
    second = {(r['file'], r['region'], r['prompt_bank']): r for r in new}
    keys = sorted(set(first) & set(second))
    if not keys:
        raise ValueError('No shared image/region/prompt-bank rows to compare')
    rows = []
    for key in keys:
        file, region, bank = key
        label = ann[file]['label']
        old_pred, new_pred = first[key]['predicted'], second[key]['predicted']
        rows.append(dict(file=file, region=region, prompt_bank=bank,
                         label=label, historical=old_pred, current=new_pred,
                         changed=int(old_pred != new_pred),
                         historical_correct=int(label == old_pred), current_correct=int(label == new_pred)))
    dest = Path(output)
    write_csv(dest, rows, ['file','region','prompt_bank','label','historical','current','changed','historical_correct','current_correct'])
    return {'compared_rows':len(rows),'changed':sum(r['changed'] for r in rows),
            'historical_correct':sum(r['historical_correct'] for r in rows),
            'current_correct':sum(r['current_correct'] for r in rows),
            'unmatched_historical':len(set(first)-set(second)),
            'unmatched_current':len(set(second)-set(first)),
            'output':str(dest),
            'note':'This compares decisions only, not probabilities; partial overlaps must not be presented as n=13 evaluation.'}
