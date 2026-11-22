# Parte B — Pre-registered *after exploration*: experimental protocol v0.2

## Research question

**Is there visible evidence of active commercial activity on the target ground-floor frontage?**
This is an observability task, not an inference about operating hours or registered land use.

## Human annotations

- **A**: evidence of a commercial operation visible (entrance open/accessible, goods
  displayed, stall ready, service station visibly prepared). A customer or sale is
  **not** required.
- **I**: commercial premises recognizable but current activity **not verifiable**.
  A locked shutter or obscured interior does not imply the shop is permanently closed.
- **N**: no commercial premises/activity visible on the **target frontage**.
  Nearby buildings are not the target; the ground-floor use determines the label
  even when upper stories are residential.

Review and, if needed, version the manually assigned `data/annotations.csv`
**before** inspecting new two-stage predictions. The current annotations were
confirmed iteratively on the same 13 photos; they are not blind labels.
F02 and F13 appear to be two viewpoints of one business, not independent shops.

## Experiments and what they can establish

**E0 (historical control):** all 13 images × full/ROI × generic/contextual
prompts, SigLIP 2 model weights frozen. Five prompt logits averaged per class.

**E1 (offline aggregation ablation):** reaggregate E0 prompt logits with
mean, median, top-2, maximum, and log-mean-exp, fixed over all photos. No model
re-inference. These alternatives were inspected after seeing the 13 examples:
**post-hoc diagnostic, not a valid estimate of improvement/generalization**.

**E2 (spatial perturbation):** for both prompt banks, compare `original`,
`roi_crop`, `blur_upper`, `blur_inside_roi`, and `blur_outside_roi`. The blur-outside
variant keeps the *same canvas and original ROI pixel size*, whereas cropping
changes both scene context and relative business scale. Blur introduces an
out-of-distribution artifact and the ROI boxes may include vehicles or walls;
none of these interventions causally identifies a visual feature used by SigLIP.
13 images × 2 prompt banks × 5 variants = 130 decisions, 1,950 prompt logits.

**E3 (two-stage evidence-aware proposal):** new bank, independent of E0 prompts.
Stage 1 compares commercial-premises evidence against residential/noncommercial
frontage. Stage 2 compares visible operating evidence against operation that
cannot be verified. If Stage 1 selects noncommercial → N; otherwise Stage 2
selects A or I. Each comparison averages five frozen prompts by default.
New prompts were developed after observing the same 13 cases, so E3 is
**exploratory**. Neither output logits nor the margins of different stages are
calibrated probabilities or directly comparable across stages.

## Evaluation

- 13 fixed images; compare accuracy, macro-F1, per-class precision/recall/F1
  and complete confusion matrices, including all zeros.
- Report counts (e.g., 10/13) alongside percentages. `N` has only one
  human-labeled example; an N recall of 1/1 is not robust.
- For E3 include both stage scores/margins (not an invented three-class score).
- Explain each mismatch from (a) visible human evidence, (b) full/ROI class,
  (c) top class-specific prompt texts/logits, (d) spatial perturbation behavior,
  (e) cautious alternative hypotheses and limits.
- Changes of raw compatibility scores are descriptive: SigLIP scores are not
  confidence in a business being open or closed.

## Validity

No held-out Latin American or foreign-city comparison, no pretraining data audit,
no blind/inter-annotator assessment, and possible dependence of F02/F13.
Do not claim "regional bias proven" or that E3 has independent validation.

To make a future **generalization** claim, freeze model, prompts, cropping
protocol and aggregation, then evaluate on *new locations*, ideally by property
with grouped sampling so nearby views of the same business cannot leak across
train/validation/test.
