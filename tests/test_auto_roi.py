"""E5 automatic ROI: no real photos or model downloads are required."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from urban_vlm.auto_roi import (
    AutoROIOptions, heuristic_roi, candidate_from_detections,
    select_auto_roi, iou, run_auto_roi,
)
from urban_vlm.data import read_rois, validate_inputs
from urban_vlm.cli import parser


def synthetic_photo(*, target='right') -> Image.Image:
    img=Image.new('RGB',(640,640),(195,201,208))
    d=ImageDraw.Draw(img)
    d.rectangle((0,0,639,255),fill=(158,169,173))
    for x in range(30,620,135):
        d.rectangle((x,80,x+63,160), fill=(37,67,77))
    d.rectangle((0,530,639,639),fill=(89,95,99))
    storefront=(340,275,600,524) if target=='right' else (40,275,300,524)
    d.rectangle(storefront,fill=(217,203,182),outline=(22,28,30),width=7)
    x1,y1,x2,y2=storefront
    d.rectangle((x1+30,y1+45,x2-30,y2-20),fill=(38,60,57))
    for x in range(x1+35,x2-35,18):
        d.line((x,y1+48,x,y2-21),fill=(187,194,198),width=2)
    d.rectangle((x1+25,y1+10,x2-25,y1+42),fill=(190,58,31))
    return img


@pytest.fixture
def small_project(tmp_path: Path):
    img_dir=tmp_path/'images'; img_dir.mkdir()
    names=['fachada_01.jpg','fachada_02.jpg']
    for name,kind in zip(names,['right','left']):
        synthetic_photo(target=kind).save(img_dir/name,quality=95)
    ann=tmp_path/'annotations.csv'
    with ann.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['file','label','evidence'])
        writer.writeheader(); writer.writerows([{'file':n,'label':'I','evidence':'synthetic example'} for n in names])
    reference=tmp_path/'rois_manual.csv'
    with reference.open('w',newline='',encoding='utf8') as f:
        wr=csv.writer(f);wr.writerow(['file','x1','y1','x2','y2'])
        wr.writerow([names[0],340,275,600,524]);wr.writerow([names[1],40,275,300,524])
    return dict(images_dir=str(img_dir),annotations=str(ann),rois=str(reference),
                expected_image_size=[640,640],strict_expected_images=False),reference


def test_opencv_candidate_valid_blank():
    opts=AutoROIOptions()
    cand=heuristic_roi(Image.new('RGB',(640,640),(130,130,130)),opts)
    x1,y1,x2,y2=cand.box
    assert 0<=x1<x2<=640 and 0<=y1<y2<=640
    assert cand.review=='manual_review'


def test_opencv_valid_structured():
    out=heuristic_roi(synthetic_photo(),AutoROIOptions())
    assert 0 <= out.box[0] < out.box[2] <= 640
    assert out.box[2]-out.box[0] >= 150
    assert out.method=='opencv'


def test_iou_exclusive():
    assert iou((0,0,100,100),(0,0,100,100)) == 1.0
    assert iou((0,0,10,10),(10,10,20,20)) == 0.0
    assert iou((0,0,100,100),(50,0,150,100)) == pytest.approx(1/3)


def test_detection_rejects_upper_windows():
    img=synthetic_photo()
    opts=AutoROIOptions(method='owlv2')
    detections=[{'box':[100,20,180,120],'score':.95,'query':'upper floor'},
                {'box':[310,310,530,480],'score':.4,'query':'ground floor'}]
    cand=candidate_from_detections(img,opts,detections)
    assert cand and cand.detector_score==.4 and cand.method=='owlv2'


def test_owl_fallback():
    class NoDetections:
        def detect(self,img): return []
    cand=select_auto_roi(synthetic_photo(),AutoROIOptions(method='owlv2'),NoDetections())
    assert cand.method=='opencv_fallback' and cand.review=='manual_review'


def test_invalid_options():
    with pytest.raises(ValueError): AutoROIOptions(center_prior=2.0)
    with pytest.raises(ValueError): AutoROIOptions(method='random')


def test_creates_diagnostics_and_never_changes_manual(small_project,tmp_path):
    cfg,ref=small_project
    original=ref.read_bytes()
    output=tmp_path/'out'
    result=run_auto_roi(cfg,output,AutoROIOptions(),manual_rois=ref)
    assert result['images']==2
    rois=read_rois(output/'rois_auto.csv')
    assert set(rois)=={'fachada_01.jpg','fachada_02.jpg'}
    assert (output/'roi_review.png').is_file()
    assert (output/'roi_diagnostics.csv').is_file()
    assert (output/'manifest.json').is_file()
    assert ref.read_bytes()==original
    assert json.loads((output/'manifest.json').read_text())['images_sha256']


def test_manual_ROI_does_not_influence_generation(small_project,tmp_path):
    cfg,reference=small_project
    first=tmp_path/'run_1';second=tmp_path/'run_2'
    run_auto_roi(cfg,first,AutoROIOptions(),manual_rois=reference)
    reference.write_text('file,x1,y1,x2,y2\nfachada_01.jpg,1,1,100,100\nfachada_02.jpg,1,1,100,100\n')
    run_auto_roi(cfg,second,AutoROIOptions(),manual_rois=reference)
    assert (first/'rois_auto.csv').read_bytes()==(second/'rois_auto.csv').read_bytes()
    assert (first/'roi_diagnostics.csv').read_bytes()!=(second/'roi_diagnostics.csv').read_bytes()


def test_auto_rois_valid_for_pipeline(small_project,tmp_path):
    cfg,_=small_project
    output=tmp_path/'auto'
    run_auto_roi(cfg,output,AutoROIOptions())
    cfg['rois']=str(output/'rois_auto.csv')
    # Mock prompt banks with five required entries per class.
    for name in ['generic','contextual']:
        (tmp_path/f'{name}.json').write_text(json.dumps({c:[f'{name} {c} {i}' for i in range(5)] for c in 'AIN'}))
    cfg['prompts']={name:str(tmp_path/f'{name}.json') for name in ['generic','contextual']}
    res=validate_inputs(cfg)
    assert res['image_count']==2


def test_no_overwrite(small_project,tmp_path):
    cfg,_=small_project
    output=tmp_path/'out'
    run_auto_roi(cfg,output,AutoROIOptions())
    with pytest.raises(FileExistsError):
        run_auto_roi(cfg,output,AutoROIOptions())


def test_cli_supports_roi_override_and_auto_rois():
    args=parser().parse_args(['run','--rois-file','runs/auto/rois_auto.csv'])
    assert args.rois_file=='runs/auto/rois_auto.csv'
    args=parser().parse_args(['auto-rois','--output','runs/new','--method','opencv'])
    assert args.method=='opencv'
