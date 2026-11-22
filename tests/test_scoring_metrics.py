from pathlib import Path
from urban_vlm.data import read_prompt_bank,read_annotations
from urban_vlm.scoring import calculate_scores
from urban_vlm.evaluation import evaluate_predictions,diagnose_disagreements,load_predictions

ROOT=Path(__file__).resolve().parents[1]

def test_15_logits_aggregate():
    p=read_prompt_bank(ROOT/'configs/prompts_generic.json')
    scores,pred,margin,details=calculate_scores([3]*5+[2]*5+[1]*5,p)
    assert pred=='A' and scores['A']==3 and margin==1 and len(details)==15

def test_historical_metrics():
    rows=load_predictions(ROOT/'reference/prior_exploratory_predictions.csv')
    result=evaluate_predictions(rows)
    lookup={(x['region'],x['prompt_bank']):x for x in result['summary']}
    assert (lookup['full','generic']['correct'],lookup['roi','generic']['correct'])==(2,10)
    assert lookup['roi','contextual']['correct']==8
    assert abs(lookup['roi','generic']['macro_f1']-0.7043650793650794)<1e-8

def test_error_hypothesis_label_I():
    anns=read_annotations(ROOT/'data/annotations.csv')
    rows=load_predictions(ROOT/'reference/prior_exploratory_predictions.csv')
    d=diagnose_disagreements(rows,anns)
    residual={x['file']:x for x in d if x['region']=='roi' and x['prompt_bank']=='generic'}
    assert set(residual)=={'fachada_05.jpg','fachada_07.jpg','fachada_09.jpg'}
    assert 'Possible' in residual['fachada_05.jpg']['hypothesis_not_proof']
