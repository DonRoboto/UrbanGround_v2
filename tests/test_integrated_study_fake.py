"""Entire workflow with synthetic images and stub scores; no model downloads."""
from pathlib import Path
from PIL import Image
from urban_vlm.data import load_config
from urban_vlm.pipeline import run_experiment, run_diagnostics
from urban_vlm.hierarchical import run_hierarchical
from urban_vlm.aggregation import run_aggregation
from urban_vlm.analysis import build_case_analysis
from urban_vlm.operation_e4 import run_operation_e4
from urban_vlm.study_report import build_study_report

ROOT=Path(__file__).resolve().parents[1]

class FakeSiglip:
    device='cpu-stub';commit='synthetic-no-weights'
    def predict_logits(self,img,texts):
        if len(texts)==15:
            return [3.0]*5+[2.0]*5+[1.0]*5
        if len(texts)==20:
            return [3.0]*5+[1.0]*5+[4.0]*5+[2.0]*5
        assert len(texts)==10
        return [2.0]*5+[3.0]*5


def test_integrated_all_study_components(tmp_path):
    cfg=load_config(ROOT/'configs/experiment.yaml')
    images=tmp_path/'images';images.mkdir()
    for i in range(1,14):
        Image.new('RGB',(640,640),(120+i,130,140)).save(images/f'fachada_{i:02}.jpg')
    cfg['images_dir']=str(images)
    fake=FakeSiglip()
    run_experiment(cfg,tmp_path/'base',backend=fake)
    run_diagnostics(cfg,tmp_path/'spatial',backend=fake)
    run_hierarchical(cfg,ROOT/'configs/prompts_hierarchical.json',tmp_path/'hier',region='both',backend=fake)
    run_operation_e4(cfg, tmp_path/'hier', ROOT/'configs/prompts_hierarchical.json',
                     ROOT/'configs/prompts_operation_e4.json', tmp_path/'e4',
                     region='both', backend=fake)
    agg=run_aggregation(tmp_path/'base/prompt_logits.csv',tmp_path/'base/predictions.csv',tmp_path/'agg')
    assert agg['n_results']==260
    report=build_case_analysis(
        tmp_path/'base/predictions.csv',tmp_path/'base/prompt_logits.csv',
        cfg['annotations'],tmp_path/'cases',
        tmp_path/'spatial/interventions.csv',
        tmp_path/'hier/hierarchical_predictions.csv',
    )
    assert report['cases']==13 and report['rows']==26
    assert (tmp_path/'cases/case_report.md').exists()

    study=build_study_report(tmp_path/'base/predictions.csv', cfg['annotations'], tmp_path/'full_report',
                             e3=tmp_path/'hier/hierarchical_predictions.csv',
                             e4_paired=tmp_path/'e4/paired_e3_e4.csv',
                             e4_logits=tmp_path/'e4/operation_e4_logits.csv',
                             spatial_effects=tmp_path/'spatial/intervention_effects.csv')
    assert study['comparisons']==8
    assert (tmp_path/'full_report/spatial_evidence.csv').exists()
    assert (tmp_path/'full_report/stage1_presence_metrics.csv').exists()
