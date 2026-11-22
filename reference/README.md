# Historical development-set outputs

- `historical_2026_11_21/baseline/`: E0 `predictions.csv`, `prompt_logits.csv`, `metrics.json` from prior 13-image run (52 decisions/780 logits).
- `historical_2026_11_21/e3/`: E3 hierarchical predictions, logits, metrics; manifest with personal absolute path redacted (26 decisions/520 logits).
- `historical_2026_11_21/e4/`: E4 paired E3/E4 decisions and logits. No independent E4 inference performed by this release builder.
- `prior_exploratory_predictions.csv`: older recorded decisions, not a new model inference.

These saved scores support an **offline audit** without photos or downloaded model weights. They were collected in iterative experiments on the same 13 development photographs and **must not be described as generalization benchmarks**. The 13 original Street View images are deliberately omitted. Do not compare raw scores across E3 stages as if they were calibrated probabilities.

- `historical_2026_11_21/e5/`: 26 decisiones pareadas manual/OpenCV/OWLv2, métricas, boxes y manifiestos de ROI con rutas privadas suprimidas.
