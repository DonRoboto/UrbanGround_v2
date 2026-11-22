# E5: Automatic ground-floor ROI localization (experimental)

## Goal and scope

Create a bounding-box **candidate** for the ground floor of a street facade, without reading its A/I/N human label or using its historical ROI coordinates. These ROIs are **not a ground-truth segmentation of commerce**. A *noncommercial* facade still needs an ROI to classify its frontage.

We retain `data/rois.csv` unchanged for E0–E4. E5 is a separate experiment, with its own image hashes, CSV, preview and model runs.

## Installation

From the repository root, use Python 3.10+:

```bash
python -m pip install -e ".[auto-roi,test]"
# If using the zero-shot detector or SigLIP 2 inference:
python -m pip install -e ".[auto-roi,inference,test]"
```

For **OpenCV-only** localization, no VLM, GPU or external downloads are required.

## Workflow: automated proposal → quality assurance → classification

### 1. Validate the original photographs

```bash
python -m urban_vlm validate --config configs/experiment.yaml
```

Original 640×640 photographs must be in `data/images/`, with the canonical filenames.

### 2A. Fast OpenCV candidate (CPU)

```bash
python -m urban_vlm auto-rois \
    --config configs/experiment.yaml \
    --method opencv \
    --output runs/E5_opencv_01
```

**How it works:** OpenCV Canny edges and vertical/horizontal gradients identify an approximate ground-floor band. A deterministic sliding-window score searches for a structurally rich frontage, with a weak horizontal prior toward the target front (default normalized x=0.5). It does **not** detect commerce, persons, sales, or shop operating status. Cars and neighboring facades can dominate the structure score. A high structural score is not a calibrated probability.

If the *target building* is known to be far right or left, you may supply only its approximate **horizontal center**, without manually outlining the full ROI:

```bash
python -m urban_vlm auto-rois --method opencv \
    --config configs/experiment.yaml --center-prior 0.75 \
    --output runs/E5_opencv_right_01
```

Keep the same center-prior policy across the dataset for a controlled comparison; don't tune per image based on SigLIP errors.

### 2B. OWLv2 proposal (GPU preferred; optional)

```bash
python -m urban_vlm auto-rois \
    --config configs/experiment.yaml \
    --method owlv2 \
    --owl-threshold 0.12 \
    --output runs/E5_owlv2_01
```

Checkpoint: `google/owlv2-base-patch16-ensemble`. Its preset zero-shot text queries include open shops, **closed shutters**, market stalls and service/workshop entrances. The detector's objective is to locate a candidate frontage, not to determine if it is actively operating. When no plausible detection is found, the implementation **falls back to OpenCV** and marks the case for review. Queries, threshold, model and fallback are logged. The first run downloads the checkpoint; ensure you have `torch` and `transformers` installed.

Because OWLv2 itself is text-conditioned, evaluate its ROI localization as a *separate pipeline* from the label-free OpenCV method. Neither selection guarantees identifying the exact target frontage; **F11 requires particular care** because it has commerce on adjacent facades.

### 3. Inspect outputs

Each run creates:

```text
runs/E5_opencv_01/
  rois_auto.csv          # file,x1,y1,x2,y2 (Pillow half-open coordinates)
  roi_diagnostics.csv    # method, structural/detection score, review state, IoU (optional)
  roi_review.png         # green=automatic; orange=original manual (if supplied)
  manifest.json          # input hashes, algorithm parameters
  summary.json
  README_RESULTS.md
```

The comparison against `data/rois.csv` is calculated **after** the automatic box has been chosen and does not influence its coordinates. IoU measures overlap with a manually drawn rectangle, **not** classification accuracy or generalization. In an evaluation dataset, pre-register the localization rule and review its outputs before looking at SigLIP predictions.

### 4. Use the proposed ROIs in SigLIP 2 without replacing the original dataset

```bash
python -m urban_vlm validate --config configs/experiment.yaml \
   --rois-file runs/E5_opencv_01/rois_auto.csv

python -m urban_vlm preview-rois --config configs/experiment.yaml \
   --rois-file runs/E5_opencv_01/rois_auto.csv \
   --output runs/E5_opencv_01/preview_for_model.png

python -m urban_vlm run --config configs/experiment.yaml \
   --rois-file runs/E5_opencv_01/rois_auto.csv \
   --device auto --output runs/E5_opencv_01/E5_inference
```

This executes the original E0 conditions (full/roi × generic/contextual) and saves **new** 52 predictions and 780 prompt logits. Their ROI source is traced in the run manifest's hashes. Compare with the unchanged historical manual-ROI E0 baseline. The `full` rows should agree when model/inference and image bytes agree.

For hierarchical tests, rerun E3 using the **same** `--rois-file`; run E4 against that E3 run and pass the **same** alternative file, since E4 hashes must match. Do not mix manual-ROI E3 with automatic-ROI E4.

### 5. One-command script

```bash
bash scripts/run_auto_roi_experiment.sh runs/E5_opencv_quick
# After reviewing proposals:
RUN_INFERENCE=1 DEVICE=auto METHOD=opencv \
  bash scripts/run_auto_roi_experiment.sh runs/E5_opencv_full
```

The second command regenerates its own ROIs and performs inference. It does not overwrite prior results.

## Scientific limitations

- E5 ROI proposals are *uncalibrated candidates*, not a validated ground-floor or storefront detector.
- OpenCV sees edges, texture and gradient structure; it has no information about which business is active.
- OWLv2 is a pre-trained open-vocabulary detector and may miss subtle street-level activity, closed businesses or the correct target facade.
- The 13 images were already used for exploratory development; any new 13-image classification or IoU is a *development result*, not generalization evidence.
- An image-only proposal often cannot distinguish *which building* is the target when multiple facades are visible; a target-center hint or human review may remain necessary.
- Comparing manual and automatic crops also changes scale/amount of context. Quantify changes in ROI area and IoU, and consider evaluating fixed-scale blur controls.
- Preserve the original labels and manual ROI set; publish the algorithm, parameters and both sets of predictions, including failures.
