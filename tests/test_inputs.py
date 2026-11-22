import json
from pathlib import Path
import pytest
from PIL import Image
from urban_vlm.data import read_annotations,read_rois,read_prompt_bank,crop_roi,load_config,validate_inputs,EXPECTED_NAMES
from urban_vlm.perturbations import make_variants

ROOT=Path(__file__).resolve().parents[1]

def test_historical_roi_and_annotations():
    anns=read_annotations(ROOT/'data/annotations.csv')
    rois=read_rois(ROOT/'data/rois.csv')
    assert set(anns)==set(EXPECTED_NAMES)==set(rois)
    assert [sum(a['label']==c for a in anns.values()) for c in 'AIN']==[7,5,1]
    assert rois['fachada_01.jpg']==(391,301,639,561)

def test_prompt_banks():
    for bank in ('generic','contextual'):
        prompts=read_prompt_bank(ROOT/f'configs/prompts_{bank}.json')
        assert sum(len(x) for x in prompts.values())==15

def test_config_no_images_option():
    cfg=load_config(ROOT/'configs/experiment.yaml')
    assert validate_inputs(cfg,require_images=False)['image_count']==13

def test_roi_hard_failure():
    img=Image.new('RGB',(640,640))
    with pytest.raises(ValueError): crop_roi(img,(0,0,641,200))

def test_variants_and_spatial_behavior():
    img=Image.new('RGB',(640,640),(255,0,0))
    from PIL import ImageDraw
    ImageDraw.Draw(img).rectangle((200,200,300,300),fill='blue')
    variants=make_variants(img,(200,200,300,300))
    assert len(variants)==5
    assert variants['roi_crop'].size==(100,100)
    assert variants['blur_outside_roi'].getpixel((250,250))==img.getpixel((250,250))
