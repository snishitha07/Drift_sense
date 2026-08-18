# Citations & Supporting References

This document lists the techniques, structural choices, and noise/augmentation
models used in DriftSense-X, with credible public references for each —
required per the hackathon's Citation Requirement. Keep these in sync with
the citations used in the PPT presentation.

## DRAM structure (word-lines, bit-lines, contact/via array)

- Itoh, K. (2001). *VLSI Memory Chip Design.* Springer. — Describes the
  periodic word-line/bit-line array and per-cell contact structure that
  `generate_data.py`'s `draw_dram_pattern()` is modeled on (horizontal
  word-lines, vertical bit-lines crossing at right angles, a contact/via at
  every intersection).
- Sze, S. M., & Ng, K. K. (2007). *Physics of Semiconductor Devices*, 3rd ed.
  Wiley. — General reference for DRAM cell array geometry and periodicity,
  used to justify the fine-pitch, high-contrast, right-angle grid structure
  and the physical plausibility of near-identical repeating cells.

## SEM imaging effects (edge-brightening, noise)

- Reimer, L. (1998). *Scanning Electron Microscopy: Physics of Image
  Formation and Microanalysis*, 2nd ed. Springer. — Basis for the
  edge-brightening step (`edge_brighten()`): secondary-electron yield rises
  at feature edges/sidewalls, producing brighter edge contrast than flat
  regions in real SEM images. Also covers the shot-noise sources motivating
  the independent Gaussian sensor-noise model applied separately to each
  image.
- Goldstein, J. I., et al. (2018). *Scanning Electron Microscopy and X-Ray
  Microanalysis*, 4th ed. Springer. — Supporting reference for SEM signal
  and noise characteristics used to size the reference-vs-search noise gap
  (search image noisier than reference, per the task's stated test
  conditions).

## Core matching technique

- Lewis, J. P. (1995). *Fast Normalized Cross-Correlation.* Vision
  Interface. — Basis for `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`, used
  as the primary matching/ambiguity score across the exact-size,
  multi-rotation candidate search.

## Sub-pixel refinement

- Kuglin, C. D., & Hines, D. C. (1975). *The Phase Correlation Image
  Alignment Method.* Proceedings of IEEE International Conference on
  Cybernetics and Society. — Basis for `cv2.phaseCorrelate`, used to refine
  the chosen candidate to sub-pixel precision after the matching-region /
  center tie-break decision.

## Structural fingerprint (gradient orientation histogram)

- Dalal, N., & Triggs, B. (2005). *Histograms of Oriented Gradients for
  Human Detection.* CVPR. — Conceptual basis for the lightweight
  orientation-histogram fingerprint computed per candidate. Note: on a
  highly periodic layout this signal is intentionally NOT used as the
  primary ambiguity discriminator (see README §5 — every repeat has a
  near-identical fingerprint by construction, so NCC is used for that
  decision instead).

## Noise / degradation models (WaferTwin synthetic generator)

- Gaussian noise and Gaussian blur models follow standard OpenCV/NumPy image
  degradation practice (`cv2.GaussianBlur`, `numpy.random.Generator.normal`),
  used to emulate common SEM/optical inspection imaging artifacts (sensor
  noise, defocus) for robustness testing, consistent with the noise
  characteristics discussed in Reimer (1998) and Goldstein et al. (2018)
  above.
- Rotation and scale-ratio jitter are modeled as small random perturbations
  (`cv2.getRotationMatrix2D`, `cv2.resize`) around the nominal values,
  representing mechanical stage misalignment and imprecise magnification
  calibration — consistent with the drift/mechanical-slack description in
  the task background.

## Note on synthetic data

WaferTwin (`generate_data.py`) is our own synthetic generator, not derived
from any external dataset or proprietary wafer imagery. It is explicitly
**not** claimed to be a physically accurate SEM simulator — see Limitations
in `README.md`.
