#!/usr/bin/env python3
"""
generate_data.py — WaferTwin synthetic benchmark generator (DriftSense-X)
DRAM-style architecture.

Generates (Reference Image, Search Image) pairs simulating Applied
Materials' Navigation-Error Recovery scenario:

  Reference Image : a high-magnification DRAM patch — periodic horizontal
                     word-lines and vertical bit-lines crossing at right
                     angles, with a small contact/via dot at every
                     intersection. Fine pitch, high contrast, very regular.

  Search Image    : a 1000x1000 px "low-magnification overview" — the SAME
                     tiled DRAM grid, covering a ~10x larger physical field
                     of view. The reference pattern appears shrunk ~10x
                     somewhere inside it, surrounded by many periodic
                     look-alike regions (genuine navigation-error ambiguity).

Mandatory dataset properties implemented here:
  - Independent sensor noise on reference and search image (never reused).
  - Edge-brightening to mimic SEM secondary-electron edge contrast.
  - Ground-truth center coordinate of the reference pattern recorded.
  - Realistic degradation: Gaussian blur, small rotation, and scale jitter
    around the nominal 10x shrink factor.
  - Search image noise/blur is set higher than reference (test-condition
    realism — the organizer's hidden test set will be noisier still).

WaferTwin is a synthetic self-evaluation benchmark. It is NOT a physically
accurate SEM simulator — see CITATIONS.md and README.md for scope/limits.

Usage:
    python generate_data.py --architecture DRAM --samples 30 --seed 42 --output-dir data
"""

import argparse
import json
import os
import numpy as np
import cv2

REF_SIZE = 300          # reference (high-mag) image side length, px
SEARCH_SIZE = 1000      # search (low-mag overview) image side length, px
BASE_PITCH = 30         # word-line/bit-line pitch at reference resolution, px
NOMINAL_SHRINK = 10.0    # nominal magnification ratio between the two images
DRIFT_STD_PX = 20       # std-dev of the tool's navigation-error drift from the
                        # search image's center, in search-image pixels. The
                        # task background describes real drift as landing the
                        # tool "several pixels away from the intended
                        # location" — a SMALL offset, not a location
                        # scattered anywhere in the 1000x1000 field. This is
                        # also why "closest to center" is a valid tie-break
                        # for periodic look-alikes (see README).

VALID_VARIATIONS = [
    "clean", "noise", "blur", "rotation", "scale",
    "noise_ambiguity", "blur_ambiguity", "combined",
]


# --------------------------------------------------------------------------
# DRAM structure: word-lines + bit-lines + contact/via dots
# --------------------------------------------------------------------------

def draw_dram_pattern(h, w, pitch, rng, jitter_frac=0.04):
    """
    Draws a periodic DRAM-style cell array:
      - horizontal word-lines
      - vertical bit-lines (crossing at right angles)
      - a small contact/via dot at every intersection
    Small per-line/per-dot jitter emulates realistic fab line-edge roughness
    while preserving overall periodicity (see CITATIONS.md for structural
    references).
    """
    img = np.full((h, w), 40, dtype=np.float32)   # dark field background
    line_w = max(1, pitch // 8)
    dot_r = max(1, pitch // 6)

    # word-lines (horizontal)
    y = pitch // 2
    while y < h:
        jy = int(rng.normal(0, pitch * jitter_frac))
        yy = int(np.clip(y + jy, 0, h - 1))
        cv2.line(img, (0, yy), (w, yy), 170, line_w, lineType=cv2.LINE_AA)
        y += pitch

    # bit-lines (vertical)
    x = pitch // 2
    while x < w:
        jx = int(rng.normal(0, pitch * jitter_frac))
        xx = int(np.clip(x + jx, 0, w - 1))
        cv2.line(img, (xx, 0), (xx, h), 170, line_w, lineType=cv2.LINE_AA)
        x += pitch

    # contact/via dot at every word-line x bit-line intersection
    yy0 = pitch // 2
    while yy0 < h:
        xx0 = pitch // 2
        while xx0 < w:
            jx = int(rng.normal(0, pitch * jitter_frac * 0.5))
            jy = int(rng.normal(0, pitch * jitter_frac * 0.5))
            cx = int(np.clip(xx0 + jx, 0, w - 1))
            cy = int(np.clip(yy0 + jy, 0, h - 1))
            cv2.circle(img, (cx, cy), dot_r, 235, -1, lineType=cv2.LINE_AA)
            xx0 += pitch
        yy0 += pitch

    return img


# --------------------------------------------------------------------------
# SEM-realistic post-processing
# --------------------------------------------------------------------------

def edge_brighten(img, gain=0.18):
    """
    Mimic real SEM edge-brightening: secondary-electron yield increases at
    feature edges/sidewalls, producing brighter edge contrast than flat
    regions (see CITATIONS.md — Reimer, 1998).
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = mag / (mag.max() + 1e-6)
    return np.clip(img + gain * 255.0 * mag, 0, 255)


def add_noise(img, sigma, rng):
    """Independent additive Gaussian sensor noise (drawn fresh each call)."""
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255)


def apply_blur(img, sigma):
    if sigma <= 0:
        return img
    k = max(3, int(2 * round(sigma * 2) + 1))
    return cv2.GaussianBlur(img, (k, k), sigma)


VARIATION_PARAMS = {
    # (search_noise_sigma, search_blur_sigma, rotation_deg_std, scale_jitter_frac)
    "clean":            (4.0, 0.3, 0.5, 0.03),
    "noise":            (14.0, 0.3, 0.5, 0.03),
    "blur":             (4.0, 0.5, 0.5, 0.03),
    "rotation":         (4.0, 0.3, 4.0, 0.03),
    "scale":            (4.0, 0.3, 0.5, 0.12),
    "noise_ambiguity":  (16.0, 0.4, 2.0, 0.08),
    "blur_ambiguity":   (6.0, 0.6, 2.0, 0.08),
    "combined":         (12.0, 0.4, 3.0, 0.10),
}
REF_NOISE_SIGMA = 3.0   # reference image always has LESS noise than search
REF_BLUR_SIGMA = 0.3


# --------------------------------------------------------------------------
# Sample generation
# --------------------------------------------------------------------------

def generate_sample(rng, variation="clean"):
    search_noise, search_blur, rot_std, scale_jit = VARIATION_PARAMS[variation]

    # 1) Clean high-mag reference structure (ground-truth structure, no noise yet)
    clean_ref = draw_dram_pattern(REF_SIZE, REF_SIZE, BASE_PITCH, rng)

    # 2) Reference Image = clean structure + edge-brightening + its OWN
    #    independent (lower) noise/blur — an independent physical capture.
    reference = edge_brighten(clean_ref)
    reference = apply_blur(reference, REF_BLUR_SIGMA)
    reference = add_noise(reference, REF_NOISE_SIGMA, rng)
    reference = reference.astype(np.uint8)

    # 3) Shrink factor: nominal 10x with scale-jitter (mag. ratio isn't exact)
    shrink = NOMINAL_SHRINK * float(rng.uniform(1 - scale_jit, 1 + scale_jit))
    small_size = max(8, int(round(REF_SIZE / shrink)))
    small_true_patch = cv2.resize(clean_ref, (small_size, small_size), interpolation=cv2.INTER_AREA)

    # 4) Small rotation of the shrunk patch (stage/rotation misalignment)
    rot_deg = float(rng.normal(0, rot_std))
    if abs(rot_deg) > 0.05:
        M = cv2.getRotationMatrix2D((small_size / 2, small_size / 2), rot_deg, 1.0)
        small_true_patch = cv2.warpAffine(
            small_true_patch, M, (small_size, small_size),
            borderMode=cv2.BORDER_CONSTANT, borderValue=40,
        )

    # 5) Low-mag tiled background covering the full search field of view.
    #    Independently generated (different jitter draws) -> visually similar
    #    periodic look-alikes, NOT pixel-identical to the true patch.
    coarse_pitch = max(2, int(round(BASE_PITCH / shrink)))
    background = draw_dram_pattern(SEARCH_SIZE, SEARCH_SIZE, coarse_pitch, rng)

    # 6) Place the true (shrunk) reference occurrence near the search image's
    #    CENTER with a small random drift offset (see DRIFT_STD_PX above) —
    #    this matches the physical navigation-error scenario, not a location
    #    scattered uniformly across the whole field.
    margin = small_size
    center = SEARCH_SIZE / 2.0
    tx = int(np.clip(center + rng.normal(0, DRIFT_STD_PX), margin, SEARCH_SIZE - margin))
    ty = int(np.clip(center + rng.normal(0, DRIFT_STD_PX), margin, SEARCH_SIZE - margin))
    y0, x0 = ty - small_size // 2, tx - small_size // 2
    background[y0:y0 + small_size, x0:x0 + small_size] = small_true_patch

    # 7) Search Image = edge-brightening + ITS OWN independent (higher) noise/blur
    search = edge_brighten(background)
    search = apply_blur(search, search_blur)
    search = add_noise(search, search_noise, rng)
    search = search.astype(np.uint8)

    ground_truth = {
        "architecture": "DRAM",
        "variation": variation,
        "true_center_x": tx,
        "true_center_y": ty,
        "shrink_factor": shrink,
        "rotation_deg": rot_deg,
        "reference_size": REF_SIZE,
        "search_size": SEARCH_SIZE,
    }
    return reference, search, ground_truth


def main():
    ap = argparse.ArgumentParser(description="WaferTwin synthetic data generator for DriftSense-X")
    ap.add_argument("--architecture", choices=["DRAM"], default="DRAM",
                     help="Only DRAM-style is implemented in this generator.")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="data")
    ap.add_argument("--variation", type=str, default="clean", choices=VALID_VARIATIONS,
                     help="Degradation condition (see VARIATION_PARAMS).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = []
    for i in range(args.samples):
        reference, search, gt = generate_sample(rng, args.variation)
        sample_dir = os.path.join(args.output_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sample_dir, "reference.png"), reference)
        cv2.imwrite(os.path.join(sample_dir, "search.png"), search)
        with open(os.path.join(sample_dir, "ground_truth.json"), "w") as f:
            json.dump(gt, f, indent=2)
        manifest.append({"sample": f"sample_{i:04d}", **gt})

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {args.samples} DRAM sample(s) "
          f"[variation={args.variation}, seed={args.seed}] into '{args.output_dir}/'")


if __name__ == "__main__":
    main()
