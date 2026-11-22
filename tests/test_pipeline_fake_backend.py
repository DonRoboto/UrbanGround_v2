from pathlib import Path
from PIL import Image
from urban_vlm.data import load_config,validate_inputs
from urban_vlm.pipeline import run_experiment,run_diagnostics

ROOT=Path(__file__).resolve().parents[1]

class FakeBackend:
    device='fake-cpu'
    commit=None
    def predict_logits(self,image,texts):
        assert len(texts)==15
        return [2.0]*5+[1.0]*5+[0.0]*5

def cfg_fixture(tmp_path):
    cfg=load_config(ROOT/'configs/experiment.yaml')
    imgdir=tmp_path/'images'
    imgdir.mkdir()
    for i in range(1,14):
        Image.new('RGB',(640,640),(200,200,200)).save(imgdir/f'fachada_{i:02d}.jpg')
    cfg['images_dir']=str(imgdir)
    return cfg

def test_end_to_end_fake_inference(tmp_path):
    cfg=cfg_fixture(tmp_path)
    output=tmp_path/'run'
    report=run_experiment(cfg,output,limit=2,backend=FakeBackend())
    assert (report['images'],report['predictions'],report['prompt_logits'])==(2,8,120)
    assert (output/'manifest.json').exists()
    assert (output/'disagreements.csv').exists()
    assert (output/'report.md').exists()

def test_full_run_shape_with_fake_backend(tmp_path):
    cfg=cfg_fixture(tmp_path)
    output=tmp_path/'full'
    report=run_experiment(cfg,output,backend=FakeBackend())
    assert (report['images'],report['predictions'],report['prompt_logits'])==(13,52,780)

def test_diagnostic_fake_backend(tmp_path):
    cfg=cfg_fixture(tmp_path)
    output=tmp_path/'diagnostics'
    report=run_diagnostics(cfg,output,files=['fachada_05.jpg'],backend=FakeBackend())
    assert (report['images'],report['predictions'],report['prompt_logits'])==(1,10,150)
