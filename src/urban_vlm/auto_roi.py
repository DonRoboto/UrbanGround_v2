"""Automatic *candidate* ground-floor ROIs (E5), not commerce classification.

Two methods are provided:
- opencv: label-free, CPU-only candidate from local facade structure;
- owlv2: optional zero-shot storefront/entrance cue detector, with CV fallback.

The output is deliberately independent of the human reference ROI coordinates.
Human ROIs are used AFTER selection for diagnostics only. A candidate is not
validated simply because it has a high score, nor is an OWLv2 score a
probability that a shop is operating.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
import csv
import json
import math
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .data import (
    read_image, read_annotations, read_rois, file_sha256, crop_roi, write_csv,
)

Box = tuple[int, int, int, int]

# Not designed to recognize an 'open business': includes closed shutters and
# ground-floor entrances, so the proposal does not depend on the target label A.
DEFAULT_OWL_QUERIES = (
    "a shop entrance on the ground floor of a building",
    "a small storefront in a city street",
    "a restaurant entrance at street level",
    "a market stall on the sidewalk",
    "a closed rolling metal shutter at a storefront",
    "a service workshop entrance",
    "a small street-side counter selling products",
    "an entrance to a commercial ground floor",
)


@dataclass(frozen=True)
class AutoROIOptions:
    method: str = "opencv"  # opencv or owlv2
    center_prior: float = 0.50  # location of front of interest; not a label
    min_width_ratio: float = 0.28
    min_height_ratio: float = 0.25
    owl_threshold: float = 0.12
    owl_model: str = "google/owlv2-base-patch16-ensemble"
    owl_revision: str | None = None  # freeze the historical model weights
    owl_device: str = "auto"

    def __post_init__(self):
        if self.method not in {"opencv", "owlv2"}:
            raise ValueError("method must be 'opencv' or 'owlv2'")
        if not 0.0 <= self.center_prior <= 1.0:
            raise ValueError("center_prior must be between 0 and 1")
        if not 0.15 <= self.min_width_ratio <= 0.95:
            raise ValueError("min_width_ratio must be between 0.15 and 0.95")
        if not 0.15 <= self.min_height_ratio <= 0.80:
            raise ValueError("min_height_ratio must be between 0.15 and 0.80")
        if not 0.0 < self.owl_threshold < 1.0:
            raise ValueError("owl_threshold must be in (0, 1)")


@dataclass(frozen=True)
class Candidate:
    box: Box
    method: str
    score: float
    review: str
    detail: str
    detector_query: str = ""
    detector_score: float | None = None


def clamp_box(box: tuple[float, float, float, float], size: tuple[int, int]) -> Box:
    w, h = size
    a, b, c, d = box
    x1 = max(0, min(w - 1, int(round(a))))
    y1 = max(0, min(h - 1, int(round(b))))
    x2 = max(x1 + 1, min(w, int(round(c))))
    y2 = max(y1 + 1, min(h, int(round(d))))
    return (x1, y1, x2, y2)


def iou(a: Box, b: Box) -> float:
    left = max(a[0], b[0]); top = max(a[1], b[1])
    right = min(a[2], b[2]); bottom = min(a[3], b[3])
    inter = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom else 0.0


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('OpenCV is required. Install: pip install -e ".[auto-roi]"') from exc
    return cv2


def _smooth(x: np.ndarray, n: int = 9) -> np.ndarray:
    if len(x) < n:
        return x
    return np.convolve(x, np.ones(n, dtype=np.float64) / n, mode="same")


def _local_peak(x: np.ndarray, lo: int, hi: int, fallback: int) -> int:
    lo = max(0, min(len(x), lo)); hi = max(lo, min(len(x), hi))
    if hi <= lo:
        return int(fallback)
    scores = _smooth(x.astype(np.float64))
    # Weak peaks should not shift the default floor boundary arbitrarily.
    segment = scores[lo:hi]
    if float(segment.max()) <= 0.0:
        return int(fallback)
    return int(lo + np.argmax(segment))


def _resize_cv(img: Image.Image, maxside: int = 840) -> np.ndarray:
    cv2 = _cv2()
    w, h = img.size
    scale = min(1.0, maxside / max(w, h))
    array = np.asarray(img.convert("RGB"))
    if scale < 1.0:
        array = cv2.resize(array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return array


def heuristic_roi(image: Image.Image, opts: AutoROIOptions) -> Candidate:
    """Image-only structural candidate; cannot identify an exact parcel.

    Use Canny/vertical edges, near-ground-floor y prior, horizontal scanning,
    and a spatial prior for the target frontage. Labels/prompts/manual ROIs are
    never accessed. Non-commercial facades will still receive a candidate.
    """
    cv2 = _cv2()
    array = _resize_cv(image)
    h, w = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    # Filter texture noise, edge map, color saturation (signs/merchandise proxy).
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = (cv2.Canny(gray, 65, 150) > 0).astype(np.float32)
    dx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    dy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    dx /= max(1.0, float(np.percentile(dx, 98)))
    dy /= max(1.0, float(np.percentile(dy, 98)))
    dx = np.clip(dx, 0, 1)
    dy = np.clip(dy, 0, 1)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0

    # Restrict vertical search to plausible street-level facades. Two peak
    # searches suggest ground-floor ceiling and the pavement/facade boundary.
    horizontal = 0.55 * dy.mean(axis=1) + 0.45 * edges.mean(axis=1)
    ytop = _local_peak(horizontal, int(.34*h), int(.53*h), int(.42*h))
    ybottom = _local_peak(horizontal, int(.72*h), int(.93*h), int(.85*h))
    ytop = max(int(.31*h), min(int(.52*h), ytop))
    ybottom = max(int(.75*h), min(int(.95*h), ybottom))
    if ybottom - ytop < int(opts.min_height_ratio * h):
        ytop, ybottom = int(.41*h), int(.85*h)

    # Rows just below the facade are allowed in the crop, but the road below is
    # not supposed to dominate the search objective.
    subedges = edges[ytop:ybottom]
    subdx = dx[ytop:ybottom]
    subsat = saturation[ytop:ybottom]
    if subedges.size == 0:
        return Candidate((0, int(.40*image.height), image.width, int(.88*image.height)),
                         'opencv_fallback', 0.0, 'manual_review', 'Empty ground-floor band')
    # Column score: shopfronts/counters/signs tend to contain fine structure,
    # but this is not a semantic detector and cannot disambiguate vehicles.
    column = .60 * subedges.mean(axis=0) + .32 * subdx.mean(axis=0) + .08 * subsat.mean(axis=0)
    column = _smooth(column, max(5, int(w * .018) | 1))
    cumsum = np.pad(np.cumsum(column, dtype=np.float64), (1, 0), mode="constant")
    anchor = opts.center_prior * w
    candidates: list[tuple[float, tuple[int,int]]] = []
    widths = sorted(set(max(16, int(w * fraction)) for fraction in
                        (opts.min_width_ratio, .35, .45, .55, .68)))
    for win in widths:
        win = min(w, win)
        step = max(4, w // 32)
        positions = list(range(0, max(1,w - win + 1), step))
        if w - win not in positions:
            positions.append(w - win)
        for x in positions:
            # Score means are comparable across widths. Small windows are
            # penalized, as they often lock onto a car or street advertisement.
            structure = float((cumsum[x+win] - cumsum[x]) / win)
            shift = abs((x + win/2 - anchor) / max(1, w))
            # fixed geometric prior, no labels; 0.28~0.68 wide preferred.
            score = structure + .018 * (win/w) - .040 * shift
            candidates.append((score, (x, x+win)))
    candidates.sort(key=lambda z: z[0], reverse=True)
    chosen_score, (x1, x2) = candidates[0]
    # Expand for whole frontage rather than grabbing only the high-frequency
    # subregion; do not artificially force an open/closed commercial class.
    padx = max(6, int((x2-x1)*.07))
    x1 = max(0, x1-padx); x2 = min(w, x2+padx)
    ytop = max(0, ytop-int(.025*h))
    ybottom = min(h, ybottom+int(.025*h))
    ow, oh = image.size
    box = clamp_box((x1*ow/w, ytop*oh/h, x2*ow/w, ybottom*oh/h), image.size)

    # Ambiguity: the best candidate and strongest spatially distinct runner-up
    # are close in structural score. This is NOT a calibrated confidence.
    chosen_mid = (x1+x2)/2
    distinct = [s for s,(lo,hi) in candidates if abs((lo+hi)/2-chosen_mid)>.18*w]
    gap = float(chosen_score - max(distinct)) if distinct else 0.0
    if float(np.std(column)) < .008 or gap < .008:
        review = 'manual_review'
    else:
        review = 'review_recommended'  # all automatic ROIs merit review
    detail = f'CV edges+vertical structure; score_gap={gap:.4f}; target prior={opts.center_prior:.2f}'
    return Candidate(box, 'opencv', float(chosen_score), review, detail)


class Detector(Protocol):
    def detect(self, image: Image.Image) -> list[dict[str, Any]]: ...


class OWLv2Detector:
    """Optional zero-shot storefront cue detector. No labeled training here."""
    def __init__(self, opts: AutoROIOptions):
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except (ImportError, AttributeError) as exc:
            raise RuntimeError('OWLv2 requires torch and transformers: pip install -e ".[inference,auto-roi]"') from exc
        self.torch = torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if opts.owl_device == 'auto' else opts.owl_device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA selected but unavailable')
        self.threshold = opts.owl_threshold
        self.queries = list(DEFAULT_OWL_QUERIES)
        self.processor = Owlv2Processor.from_pretrained(opts.owl_model, **({"revision": opts.owl_revision} if opts.owl_revision else {}))
        self.model = Owlv2ForObjectDetection.from_pretrained(opts.owl_model, **({"revision": opts.owl_revision} if opts.owl_revision else {})).to(self.device).eval()
        self.commit = getattr(self.model.config, '_commit_hash', None)

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        torch = self.torch
        texts = [self.queries]
        inputs = self.processor(text=texts, images=image, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, target_sizes=[(image.height, image.width)],
            threshold=self.threshold, text_labels=texts
        )[0]
        return [{"box": [float(v) for v in b.tolist()], "score": float(s.item()), "query": str(t)}
                for b,s,t in zip(result['boxes'], result['scores'], result['text_labels'])]


def candidate_from_detections(image: Image.Image, opts: AutoROIOptions,
                              detections: list[dict[str, Any]]) -> Candidate | None:
    w,h = image.size
    viable=[]
    for det in detections:
        raw=det['box']
        if not all(math.isfinite(float(t)) for t in raw):
            continue
        box = clamp_box(tuple(raw), image.size)
        x1,y1,x2,y2=box
        area=((x2-x1)*(y2-y1))/(w*h)
        midx=(x1+x2)/(2*w); midy=(y1+y2)/(2*h)
        # Avoid windows on upper floors, road objects, huge all-image bboxes.
        if not (.31 <= midy <= .93) or not (.0025 <= area <= .72):
            continue
        confidence=float(det['score'])
        if confidence < opts.owl_threshold:
            continue
        # Highest detection confidence, tempered by target-front ambiguity.
        rank = confidence - .28 * abs(midx-opts.center_prior) - .10 * max(0.0, .43-midy)
        viable.append((rank, confidence, box, str(det.get('query',''))))
    if not viable:
        return None
    viable.sort(key=lambda v: v[0], reverse=True)
    _, confidence, (x1,y1,x2,y2), query = viable[0]
    bw,bh=x2-x1,y2-y1
    # Include the wider storefront and its floor boundary; not merely a sign.
    minw=max(bw*1.25,w*opts.min_width_ratio)
    minh=max(bh*1.7,h*opts.min_height_ratio)
    cx=(x1+x2)/2
    cy=(y1+y2)/2 + .04*h
    # Restrict to target ground-floor region and avoid large road extension.
    x1=max(0,cx-minw/2);x2=min(w,cx+minw/2)
    y1=max(.29*h,cy-minh/2);y2=min(.95*h,cy+minh/2)
    if y2-y1 < opts.min_height_ratio*h:
        y1=max(.27*h,y2-opts.min_height_ratio*h)
    box=clamp_box((x1,y1,x2,y2),image.size)
    review='review_recommended' if confidence>=.30 else 'manual_review'
    return Candidate(box,'owlv2',confidence,review,'Open-vocabulary cue detection; verify the target frontage',query,confidence)


def select_auto_roi(image: Image.Image, opts: AutoROIOptions,
                    detector: Detector | None = None) -> Candidate:
    if opts.method=='opencv':
        return heuristic_roi(image,opts)
    if detector is None:
        detector=OWLv2Detector(opts)
    suggestion=candidate_from_detections(image,opts,detector.detect(image))
    if suggestion is None:
        a=heuristic_roi(image,opts)
        return Candidate(a.box,'opencv_fallback',a.score,'manual_review',
                         'OWLv2 found no suitable ground-floor object; '+a.detail)
    return suggestion


def _montage(entries: list[tuple[str,Image.Image,Candidate,Box|None]], output: Path):
    # Illustrative review sheet: full image with auto rectangle (green),
    # manual rectangle (orange, if available) and cropped candidate.
    cw,ch=610,530
    canvas=Image.new('RGB',(cw*3,ch*math.ceil(len(entries)/3)), '#f3f4f6')
    draw=ImageDraw.Draw(canvas)
    for idx,(name,img,cand,reference) in enumerate(entries):
        col,row=idx%3,idx//3
        left,top=col*cw,row*ch
        target=img.copy()
        pen=ImageDraw.Draw(target)
        pen.rectangle(cand.box,outline='#0ca678',width=5)
        if reference:
            pen.rectangle(reference,outline='#f08c00',width=3)
        target.thumbnail((295,430))
        crop=crop_roi(img,cand.box)
        crop.thumbnail((295,430))
        canvas.paste(target,(left+8,top+49))
        canvas.paste(crop,(left+310,top+49))
        draw.text((left+8,top+8),name+' | '+cand.method+' | '+cand.review,fill='#111827')
        if reference:
            draw.text((left+8,top+27),f'IoU vs manual: {iou(cand.box,reference):.3f}   Green=Auto, Orange=Manual',fill='#374151')
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output)


def _file_sha(path: str | Path) -> str:
    return file_sha256(path)


def run_auto_roi(cfg: dict, out: str | Path, opts: AutoROIOptions, *,
                 files: list[str]|None=None, manual_rois: str|Path|None=None,
                 detector: Detector|None=None) -> dict[str,Any]:
    """Create a new auto ROI CSV, QA sheet, diagnostics and manifests.

    IMPORTANT: ``manual_rois`` is consulted only after ALL automatic boxes are
    produced. No manual ROI coordinates are ever used to rank proposals.
    """
    output=Path(out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite non-empty directory: {output}')
    annotations=read_annotations(cfg['annotations'])
    all_names=sorted(annotations)
    names=files if files is not None else all_names
    if not names or len(names)!=len(set(names)) or any(n not in annotations for n in names):
        raise ValueError('Select existing, unique image filenames from annotations.csv')
    prior_rois=read_rois(manual_rois) if manual_rois else {}
    # Create a single detector instance rather than downloading/loading it for every image.
    if opts.method=='owlv2' and detector is None:
        detector=OWLv2Detector(opts)
    roi_rows=[]; diagnostic_rows=[]; sheets=[]
    for index,name in enumerate(names,1):
        file=Path(cfg['images_dir'])/name
        if not file.is_file():
            raise FileNotFoundError(file)
        img=read_image(file)
        expected=cfg.get('expected_image_size')
        if expected and list(img.size)!=list(expected):
            raise ValueError(f'{name}: got {img.size} instead of {expected}; review coordinates and preprocessing')
        cand=select_auto_roi(img,opts,detector)
        box=cand.box
        ref=prior_rois.get(name)
        overlap=iou(box,ref) if ref else None
        # The reference is used for QA only, AFTER a box has been selected;
        # low agreement flags an automatic proposal for review.
        review = 'manual_review' if overlap is not None and overlap < .30 else cand.review
        roi_rows.append({'file':name,**dict(zip(('x1','y1','x2','y2'),box))})
        diagnostic_rows.append({'file':name,'method':cand.method,'review':review,
            'score':f'{cand.score:.6f}','detail':cand.detail,
            'detector_query':cand.detector_query,
            'detector_score':(f'{cand.detector_score:.6f}' if cand.detector_score is not None else ''),
            'width':img.width,'height':img.height,
            'area_ratio':f'{(box[2]-box[0])*(box[3]-box[1])/(img.width*img.height):.6f}',
            'iou_vs_manual':f'{overlap:.6f}' if overlap is not None else ''})
        sheets.append((name,img, Candidate(cand.box,cand.method,cand.score,review,cand.detail,cand.detector_query,cand.detector_score),ref))
        print(f'[{index}/{len(names)}] {name} {cand.method}: {box}, review={review}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'rois_auto.csv',roi_rows,['file','x1','y1','x2','y2'])
    diagnostic_fields=['file','method','review','score','detail','detector_query','detector_score',
                       'width','height','area_ratio','iou_vs_manual']
    write_csv(output/'roi_diagnostics.csv',diagnostic_rows,diagnostic_fields)
    _montage(sheets,output/'roi_review.png')
    overlaps=[float(r['iou_vs_manual']) for r in diagnostic_rows if r['iou_vs_manual']]
    summary={'images':len(names),'method_requested':opts.method,
             'method_counts':{key:sum(r['method']==key for r in diagnostic_rows)
                              for key in sorted({r['method'] for r in diagnostic_rows})},
             'manual_review_count':sum(r['review']=='manual_review' for r in diagnostic_rows),
             'mean_iou_vs_manual':statistics.mean(overlaps) if overlaps else None,
             'median_iou_vs_manual':statistics.median(overlaps) if overlaps else None,
             'note':'IoU compares ROIs only; it is not commercial activity detection accuracy.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
        'pipeline':'ground_floor_roi_candidate_v1.1',
        'method':opts.method,'params':dict(vars(opts)),
        'owl_model_commit_detected':getattr(detector,'commit',None) if opts.method=='owlv2' else None,
        'images_sha256':{name:_file_sha(Path(cfg['images_dir'])/name) for name in names},
        'annotation_file_sha256':_file_sha(cfg['annotations']),
        'reference_rois_sha256':_file_sha(manual_rois) if manual_rois else None,
        'warning':'Automatic ROIs are proposals. Target frontage may be ambiguous; review all 13 before inference.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    rows=['# E5 — Automatic ground-floor ROI candidates','',
          f'Algorithm: **{opts.method}**; images: **{len(names)}**.',
          f'Methods used: {summary["method_counts"]}.',
          f'Requiring manual review: **{summary["manual_review_count"]}**.',
          '']
    if overlaps:
        rows.append(f'Mean IoU with historical manual boxes: **{summary["mean_iou_vs_manual"]:.3f}** (descriptive only).')
    rows.extend(['','## Interpretation','',
                 '- Automatic ROI localization is NOT a prediction of A, I or N.',
                 '- A high IoU only measures alignment with human ROI boxes, not recognition of commerce.',
                 '- OpenCV is structural and can select cars or neighboring facades.',
                 '- OWLv2 is a zero-shot cue detector; missing/closed shops may be missed.',
                 '- Review the target frontage (especially image 11) before reporting any classification metrics.',
                 '- Preserve original manual E0–E4 experiments for comparisons.',
                 ''])
    (output/'README_RESULTS.md').write_text('\n'.join(rows),encoding='utf-8')
    return {'output':str(output),**summary}
