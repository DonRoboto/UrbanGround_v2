"""E5 comparison tests. Can run on CPU without torch or transformers."""
import importlib.util
import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from urban_vlm.data import file_sha256, load_config, read_rois

SOURCE = Path(__file__).resolve().parents[1] / 'scripts' / 'run_e5_siglip2.py'
spec = importlib.util.spec_from_file_location('e5_siglip2_script', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_iou():
    assert mod.roi_iou((0,0,10,10), (0,0,10,10)) == 1.0
    assert mod.roi_iou((0,0,10,10), (10,10,20,20)) == 0.0
    assert mod.roi_iou((0,0,10,10), (5,0,15,10)) == pytest.approx(1/3)


def test_refuses_modified_image_hash(tmp_path):
    box = (10,10,50,50)
    a = {'images_sha256': {'fachada_01.jpg': 'a'}}
    assert a['images_sha256'] != {'fachada_01.jpg': 'b'}


def test_read_run_rejects_duplicates(tmp_path):
    d = tmp_path / 'r'; d.mkdir()
    (d / 'manifest.json').write_text(json.dumps({'a': 1}),encoding='utf-8')
    (d / 'predictions.csv').write_text('file,region,prompt_bank,label,predicted\n' +
                                       'fachada_01.jpg,roi,generic,A,A\n' * 2, encoding='utf-8')
    with pytest.raises(ValueError):
        mod.read_run(d, ['fachada_01.jpg'])


def test_report_fake_runs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs' / 'experiment.yaml')
    # Synthetic photos; do not touch the real study dataset.
    imgs = tmp_path / 'imgs'; imgs.mkdir()
    anns = Path(cfg['annotations'])
    for i in range(1, 14):
        Image.new('RGB',(640,640),color=(i*11%255,120,100)).save(imgs / f'fachada_{i:02d}.jpg')
    cfg['images_dir'] = str(imgs)
    manual = Path(cfg['rois'])
    auto = {}
    for method in ('opencv','owlv2'):
        rd = tmp_path / method; rd.mkdir()
        p = rd / 'rois_auto.csv'
        p.write_bytes(manual.read_bytes())
        image_hash = {n: file_sha256(imgs / n) for n in sorted(x.name for x in imgs.glob('*.jpg'))}
        (rd / 'manifest.json').write_text(json.dumps({
            'method':method,'images_sha256':image_hash,
            'annotation_file_sha256':file_sha256(anns),
            'reference_rois_sha256':file_sha256(manual)
        }),encoding='utf-8')
        auto[method] = p
    preflight = mod.roi_preflight(cfg, {'manual':manual,**auto})
    runs = {}
    with (root / 'reference' / 'prior_exploratory_predictions.csv').open(newline='',encoding='utf-8') as f:
        orig = list(csv.DictReader(f))
    # Historical reference is only used to validate report logic here, not as E5 inference evidence.
    for method, roipath in {'manual':manual,**auto}.items():
        rp = tmp_path / 'runs' / method; rp.mkdir(parents=True)
        ch = {Path(cfg['annotations']).name: file_sha256(anns),
              Path(roipath).name:file_sha256(roipath),
              **{Path(v).name:file_sha256(v) for v in cfg['prompts'].values()}}
        man={'model_id':cfg['model_id'],'model_commit_hash_detected':'synthetic-test-commit',
             'image_hashes':preflight['image_hashes'],
             'image_count':13,'prediction_count':52,'config_and_label_hashes':ch}
        (rp/'manifest.json').write_text(json.dumps(man),encoding='utf-8')
        with (rp/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['file','region','prompt_bank','label','predicted'], extrasaction='ignore')
            w.writeheader(); w.writerows(orig)
        runs[method]=rp
    report = tmp_path / 'final_report'
    mod.compare_runs(cfg, runs, preflight, report)
    assert (report/'E5_comparison_report.md').exists()
    with (report/'metrics_summary.csv').open(newline='',encoding='utf-8') as f:
        stats = list(csv.DictReader(f))
    assert len(stats) == 8
    assert {r["method"] for r in stats} == {"full_image", "manual", "opencv", "owlv2"}
    assert all(float(r['accuracy']) <= 1 for r in stats)
    with (report/'paired_predictions.csv').open(newline='',encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 26
        assert all(r["full_prediction"] in {"A", "I", "N"} for r in rows)
    with (report/'changes_vs_full.csv').open(newline='',encoding='utf-8') as f:
        assert len(list(csv.DictReader(f))) == 13*2*3
