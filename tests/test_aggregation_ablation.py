from pathlib import Path
import csv
from urban_vlm.aggregation import run_aggregation, aggregate

ROOT=Path(__file__).resolve().parents[1]


def test_aggregation_rules():
    vals=[-7,-4,-1,0,5]
    assert aggregate(vals,'mean')==sum(vals)/5
    assert aggregate(vals,'top2')==2.5
    assert aggregate(vals,'max')==5


def test_saved_real_logit_aggregation(tmp_path):
    folder=ROOT/'reference/historical_2026_11_21/baseline'
    summary=run_aggregation(folder/'prompt_logits.csv',folder/'predictions.csv',tmp_path/'ablation')
    assert summary['n_cases']==52
    assert summary['n_logit_rows']==780
    assert summary['n_results']==260
    with (tmp_path/'ablation/aggregation_summary.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    mapping={(r['region'],r['prompt_bank'],r['aggregation']):r for r in rows}
    assert int(mapping['roi','generic','mean']['correct'])==10
    assert int(mapping['full','generic','mean']['correct'])==2
    assert int(mapping['roi','contextual','top2']['correct'])==10
