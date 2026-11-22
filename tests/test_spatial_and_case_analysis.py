from pathlib import Path
import csv
from PIL import Image
from urban_vlm.data import load_config
from urban_vlm.pipeline import run_diagnostics
from urban_vlm.spatial import compute_spatial_effects
from urban_vlm.analysis import build_case_analysis

ROOT=Path(__file__).resolve().parents[1]

class FakeBackend:
    device='fake-cpu';commit=None
    def predict_logits(self,image,texts):
        assert len(texts)==15
        return [2.0]*5+[1.0]*5+[0.0]*5


def test_spatial_effects_account_for_view_changes():
    items=[
       {'file':'a.jpg','prompt_bank':'generic','region':'original','label':'A','predicted':'N','score_A':0,'score_I':1,'score_N':2},
       {'file':'a.jpg','prompt_bank':'generic','region':'roi_crop','label':'A','predicted':'A','score_A':4,'score_I':1,'score_N':2},
    ]
    out=compute_spatial_effects(items)
    assert len(out)==2
    assert out[1]['label_changed']==1
    assert out[1]['scale_or_canvas_changed']==1
    assert out[1]['delta_A_minus_N']==4


def test_full_case_report_from_saved_user_outputs(tmp_path):
    saved=ROOT/'reference/historical_2026_11_21/baseline'
    r=build_case_analysis(saved/'predictions.csv',saved/'prompt_logits.csv',
                          ROOT/'data/annotations.csv',tmp_path/'case')
    assert r['cases']==13 and r['rows']==26
    with (tmp_path/'case/per_image_analysis.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    mismatch={x['file'] for x in rows if x['prompt_bank']=='generic' and x['roi_correct']=='0'}
    assert mismatch=={'fachada_05.jpg','fachada_07.jpg','fachada_09.jpg'}
    assert (tmp_path/'case/case_report.md').is_file()


def test_diagnose_produces_effect_file(tmp_path):
    cfg=load_config(ROOT/'configs/experiment.yaml')
    imgdir=tmp_path/'images';imgdir.mkdir()
    for i in range(1,14):
        Image.new('RGB',(640,640),'#fedcab').save(imgdir/f'fachada_{i:02d}.jpg')
    cfg['images_dir']=str(imgdir)
    run=run_diagnostics(cfg,tmp_path/'diag',files=['fachada_05.jpg'],backend=FakeBackend())
    assert run['effect_rows']==10
    assert (tmp_path/'diag/intervention_effects.csv').is_file()
