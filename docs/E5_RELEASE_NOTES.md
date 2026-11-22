# E5 release notes — UrbanGroundFloor v1.1

This release adds an *automatic ROI candidate* stage while preserving the original manual ROI data and historical E0–E4 results.

- `auto-rois`: OpenCV-only (CPU) or OWLv2 zero-shot, with fallback and QA flags.
- `--rois-file` available on validation, previews, E0, E2, E3, E4 for isolated comparisons.
- `rois_auto.csv` is independent of the manual `data/rois.csv`; no previous label or manual rectangle is used to choose an automatic box.
- Outputs: image hashes, ROI diagnostics, reviewer montage and optional IoU with the manually drawn ROI.
- 11 E5 unit tests added; 44 tests passed total in the build environment.
- The optional OWLv2 model requires weights downloaded separately and was not run against the 13 actual photos in the build environment.
- A fully automatic assignment to the *correct* building cannot be guaranteed when multiple frontages appear. A spatial hint or review remains necessary.
