from pathlib import Path
from urban_vlm.comparison import compare_runs
ROOT=Path(__file__).resolve().parents[1]

def test_identical_reference(tmp_path):
    ref=ROOT/'reference/prior_exploratory_predictions.csv'
    x=compare_runs(ref,ref,ROOT/'data/annotations.csv',tmp_path/'compared.csv')
    assert x['compared_rows']==52 and x['changed']==0 and x['historical_correct']==21
    assert (tmp_path/'compared.csv').exists()
