# E4 — Operation Evidence Refinement

## Research question

Does replacing only the **operation-evidence descriptions** in the second
stage of the two-stage SigLIP 2 classifier change the errors associated with
commercial premises whose current operation is not visually verifiable?

We do **not** train SigLIP 2. We do **not** tune thresholds on the labels.
This is a post-hoc, exploratory prompt formulation on 13 development images.
The categories remain:

- `A`: visible evidence of an establishment ready to offer goods/services.
  A client, sale, or transaction is **not** required.
- `I`: commercial premises are recognizable but operation cannot be determined
  from this image; **does not mean permanently closed**.
- `N`: no recognizable commercial premises in the target frontage.

## Controlled comparison: E3 → E4

E3 implemented a two-stage decision: presence of commerce (commercial /
noncommercial), followed by evidence of operation (active / unverifiable).
In E3+ROI the first stage correctly separated the 12 commercial frontages
from the sole residential frontage in this exploratory sample. The four
remaining errors came from the second stage.

**E4 freezes the first stage exactly:** it reads E3's 10 archived presence
logits (5 per hypothesis) for each image-region, verifies their consistency
against E3's saved predictions and checksum of the original text bank, and
reuses those scores verbatim. The model is called only to evaluate **10 new
operation prompts**, listed in `configs/prompts_operation_e4.json`.

E3 and E4 use the same images, ROI coordinates, human labels, SigLIP 2
checkpoint, and **mean** aggregation. The outputs are kept in different
folders. There is no manual selection of per-image prompts or after-the-fact
score thresholds.

What E4 changes: *which text hypotheses the second stage compares*. What it
does **not** change: imagery, localization, commercial-presence logits,
category definitions, aggregation, or model weights.

## Install / update

If using the **full v0.3 package**:

```bash
python -m pip install -e '.[inference,test]'
```

If using the **patch**, extract its contents at the root of your existing
`UrbanGroundFloor_SigLIP2_Full` repository. The provided launcher forces
Python to load the local `src/` package, avoiding a stale `urban-vlm` command
from v0.1. You do not need to rerun E3.

```bash
bash scripts/run_operation_e4.sh --help
```

## Execution

Set `--e3-dir` to the folder produced by your successful **E3 hierarchical**
run. It must contain `hierarchical_predictions.csv`,
`hierarchical_logits.csv`, and `manifest.json`. E3 needs to have been run with
`--aggregation mean` and **the original prompts**.

1. **Validate sources without using GPU**:

```bash
bash scripts/run_operation_e4.sh \
  --config configs/experiment.yaml \
  --e3-dir runs/hierarchical_13 \
  --prompts configs/prompts_operation_e4.json \
  --region both \
  --dry-run
```

2. **Two-image smoke test** (new output folder; does not overwrite E3):

```bash
bash scripts/run_operation_e4.sh \
  --config configs/experiment.yaml \
  --e3-dir runs/hierarchical_13 \
  --prompts configs/prompts_operation_e4.json \
  --region roi --limit 2 --device auto \
  --output runs/e4_smoke_2
```

3. **Complete E4, same 26 image-regions as E3**:

```bash
bash scripts/run_operation_e4.sh \
  --config configs/experiment.yaml \
  --e3-dir runs/hierarchical_13 \
  --prompts configs/prompts_operation_e4.json \
  --region both --aggregation mean --device auto \
  --output runs/e4_operation_13
```

Outputs in `runs/e4_operation_13/`:

| File | Meaning |
|---|---|
| `operation_e4_predictions.csv` | 26 E4 predictions, two internal decisions and E3 labels |
| `operation_e4_logits.csv` | 520 rows: 260 **unchanged E3 presence** + 260 **new E4 operation** |
| `paired_e3_e4.csv` | 26 paired comparisons, including `corrected` and `regressed` |
| `metrics.csv`, `metrics.json` | E4 metrics by region |
| `e3_reference_metrics.json` | E3 metrics on the **same selected image-regions** |
| `E4_report.md` | Per-image paired E3↔E4 report |
| `report.md`, `disagreements.csv` | Standard evaluation and disagreement summary |
| `manifest.json` | Model info, hashes of images, prompts, E3 inputs, software, GPU |

For `--region roi` alone, there are 13 predictions, 130 new logits and 130
cached logits. The stage-1 scores are not recomputed. All output folders must
be new. The detector refuses to silently reuse E3 results with changed
photographs, annotations, ROI or model revision.

## How to read the findings

Do not assert that E4 is "better" solely because of more correct predictions
on these same 13 images. For the scientific discussion, inspect:

- F03, F06, F12: visible operation misclassified as I by E3.
- F05, F09: E3 correctly chose I despite commercial signage.
- F07: presence recognized by E3, operation over-inferred as A instead of I.
- F08: an E3 positive example close to the second-stage decision boundary.
- F11: the only negative example (N); stage-1 decision stays exactly frozen.

Use the paired CSV to report **all** corrected and newly incorrect cases,
not only favorable examples. These are controlled software interventions on
text hypotheses, **not causal proof of internal visual attention**.

## Known validity limits

- The prompts were written *after* reviewing these same photographs. E4 is
  development-set analysis, not an independently validated intervention.
- F02/F13 likely depict the same commercial premises from different viewpoints.
- N has only one example; its recall is not a robust measure of generalization.
- Scores/margins are logits of image–text compatibility, **not probabilities**.
- A static Street View image cannot verify opening hours or current operation.
- Source images are not included in the public repository; verify reuse rights.

For a real external validation, collect more fronts and pre-register the E4
prompt bank before assessing independent images/establishments.
