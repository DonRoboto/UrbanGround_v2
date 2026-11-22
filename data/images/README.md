# Original photographs (not included in the public repository)

Place `fachada_01.jpg` to `fachada_13.jpg` here, retaining their original **640×640** resolution. These files are deliberately excluded by `.gitignore` to avoid redistributing third-party Street View imagery without verifying license/permission.

Before each full run:

```bash
python -m urban_vlm validate --config configs/experiment.yaml
python -m urban_vlm preview-rois --config configs/experiment.yaml --output runs/roi_review.png
```

The ROI coordinates in `data/rois.csv` are in **original pixel coordinates**, not normalized to [0,1]. If photographs are cropped or resized, define **new** ROIs with new identifiers and do not compare them without documenting the change.
