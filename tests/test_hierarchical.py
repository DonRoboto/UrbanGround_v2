from pathlib import Path
import csv
import pytest
from PIL import Image
from urban_vlm.data import load_config
from urban_vlm.hierarchical import (
    load_hierarchical_prompts, flatten_hierarchical, hierarchical_decision,
    run_hierarchical,
)

ROOT=Path(__file__).resolve().parents[1]
PROMPTS=ROOT/'configs/prompts_hierarchical.json'

class HierarchicalFakeBackend:
    device='fake-cpu'
    commit=None
    def predict_logits(self, image, texts):
        assert len(texts)==20
        return [3.0]*5+[0.0]*5+[4.0]*5+[1.0]*5


def fixture_cfg(tmp_path):
    cfg=load_config(ROOT/'configs/experiment.yaml')
    imgdir=tmp_path/'images'
    imgdir.mkdir()
    for i in range(1,14):
        Image.new('RGB',(640,640),'#b0c4de').save(imgdir/f'fachada_{i:02d}.jpg')
    cfg['images_dir']=str(imgdir)
    return cfg


def test_prompt_structure_and_stage_outcomes():
    prompts=load_hierarchical_prompts(PROMPTS)
    texts,descriptors=flatten_hierarchical(prompts)
    assert len(texts)==20 and len(descriptors)==20
    yes,rows=hierarchical_decision([3]*5+[0]*5+[4]*5+[1]*5,prompts)
    assert yes['predicted']=='A' and len(rows)==20
    no_commerce,_=hierarchical_decision([-1]*5+[2]*5+[4]*5+[1]*5,prompts)
    assert no_commerce['predicted']=='N'
    uncertain,_=hierarchical_decision([3]*5+[0]*5+[1]*5+[4]*5,prompts)
    assert uncertain['predicted']=='I'
    ties,_=hierarchical_decision([0]*20,prompts)
    assert ties['predicted']=='N'
    with pytest.raises(ValueError):hierarchical_decision([1]*19,prompts)


def test_hierarchical_fake_inference_end_to_end(tmp_path):
    cfg=fixture_cfg(tmp_path)
    out=tmp_path/'hierarchical'
    result=run_hierarchical(cfg,PROMPTS,out,region='both',backend=HierarchicalFakeBackend())
    assert (result['images'],result['predictions'],result['prompt_logits'])==(13,26,520)
    with (out/'hierarchical_predictions.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert {x['predicted'] for x in rows}=={'A'}
    assert all(x['runner_up_margin']=='' for x in rows)
    assert (out/'manifest.json').exists()
    assert (out/'metrics.json').exists()
    with pytest.raises(FileExistsError):
        run_hierarchical(cfg,PROMPTS,out,region='roi',backend=HierarchicalFakeBackend())
