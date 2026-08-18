#!/usr/bin/env python3
"""
infer.py — DriftSense-X standalone localization inference (DRAM)

Given a reference image and a larger search image, predicts the center
coordinate (x, y) of the reference pattern inside the search image.

Official task rules implemented:
  - Accepts a reference image path and a search image path.
  - Outputs a single (x, y) center coordinate.
  - The reference pattern appears in the search image shrunk ~10x
    (10x lower magnification) relative to the reference.
  - If more than one matching region is found, returns the one CLOSEST TO
    THE CENTER of the search image (official tie-break rule — this is NOT
    a highest-confidence pick).

Approach: classical computer vision (NOT a neural network).
  1. Multi-scale, multi-rotation candidate generation (cv2.matchTemplate)
     searched around the expected ~1/10 shrink ratio.
  2. Structural fingerprint scoring (gradient-orientation histogram).
  3. Matching-region selection: any candidate within a small tolerance of
     the best score is treated as a genuine "matching region".
  4. Tie-break: among matching regions, pick the one closest to the
     search image's geometric center (per official rule).
  5. Sub-pixel refinement of the chosen candidate (phase correlation).

Usage:
    python infer.py --reference reference.png --search search.png
    python infer.py --reference reference.png --search search.png --output result.json

Only values that are actually computed are reported (no invented metrics).
"""

import argparse
import json
import time
import numpy as np
import cv2

# Search space: expected reference->search shrink is ~10x (i.e. template
# side length ~ reference_side/10), but the exact ratio is jittered by the
# generator/organizer (+-10-15%). NCC is highly sensitive to even 1-pixel
# template-size mismatches at this scale, so we search over exact INTEGER
# output sizes (not shrink ratios) to guarantee the true size is hit exactly
# rather than falling between two tested ratios.
TEMPLATE_SIZE_CANDIDATES = list(range(20, 42))         # px, covers ~7x-15x shrink for a 300px ref
ROTATION_CANDIDATES_DEG = [-9, -6, -3, 0, 3, 6, 9]     # small rotation search
TOP_K_PER_PASS = 4
NMS_RADIUS = 6

# Matching-region criteria. NCC (normalized cross-correlation) is used as the
# PRIMARY discriminator for ambiguity, not the fingerprint-blended score: on
# a periodic layout every repeat has a near-identical structural fingerprint
# by construction, so fingerprint similarity alone cannot separate true
# match from look-alikes and would create false ambiguity if used here. NCC
# reflects literal pixel-pattern agreement and is far more discriminative
# (see empirical score gaps documented in README.md).
MIN_MATCH_QUALITY = 0.55     # NCC below this is not considered a real match at all
NCC_TIE_TOLERANCE = 0.04     # candidates within this NCC of the best are a genuine tie
MIN_PATCH_STD = 8.0          # reject candidates whose search patch is near-flat: NCC
                              # (normalized cross-correlation) is numerically unstable
                              # in low-variance/low-contrast regions and can report a
                              # spuriously high score there despite no real structure.


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def non_max_suppress_peaks(score_map, top_k, radius):
    scores = score_map.copy()
    peaks = []
    for _ in range(top_k):
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        val = scores[idx]
        if not np.isfinite(val) or val <= -1e8:
            break
        peaks.append((idx[1], idx[0], float(val)))  # (x, y, score)
        y0, y1 = max(0, idx[0] - radius), min(scores.shape[0], idx[0] + radius + 1)
        x0, x1 = max(0, idx[1] - radius), min(scores.shape[1], idx[1] + radius + 1)
        scores[y0:y1, x0:x1] = -1e9
    return peaks


def rotate_image(img, angle_deg):
    if angle_deg == 0:
        return img
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def multiscale_multirotation_candidates(search, reference):
    """Step 1: sweep exact template sizes x rotations, template-match, NMS -> candidates."""
    h, w = reference.shape
    all_candidates = []
    for size in TEMPLATE_SIZE_CANDIDATES:
        rw, rh = size, size
        shrink = w / size
        if rh >= search.shape[0] or rw >= search.shape[1]:
            continue
        base_tmpl = cv2.resize(reference, (rw, rh), interpolation=cv2.INTER_AREA)
        for angle in ROTATION_CANDIDATES_DEG:
            tmpl = rotate_image(base_tmpl, angle)
            result = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
            peaks = non_max_suppress_peaks(result, TOP_K_PER_PASS, NMS_RADIUS)
            for (px, py, score) in peaks:
                cx, cy = px + rw // 2, py + rh // 2
                patch = search[py:py + rh, px:px + rw]
                if patch.size == 0 or patch.std() < MIN_PATCH_STD:
                    continue  # reject: NCC unreliable on a near-flat region
                all_candidates.append({
                    "x": cx, "y": cy, "w": rw, "h": rh,
                    "ncc": score, "shrink": float(shrink), "rotation": angle,
                })

    # Merge duplicate detections of the SAME physical location found across
    # different scale/rotation passes (a real site is often re-detected at
    # several nearby scales) — keep only the highest-scoring one per location
    # so genuinely distinct sites, not scale-duplicates, drive the ambiguity
    # count used by the official center tie-break.
    MERGE_RADIUS = 15
    all_candidates.sort(key=lambda c: -c["ncc"])
    merged = []
    for c in all_candidates:
        if all(abs(c["x"] - m["x"]) > MERGE_RADIUS or abs(c["y"] - m["y"]) > MERGE_RADIUS for m in merged):
            merged.append(c)
    return merged


def gradient_fingerprint(patch, bins=16):
    patch = patch.astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) + np.pi)
    hist, _ = np.histogram(ang, bins=bins, range=(0, 2 * np.pi), weights=mag)
    norm = np.linalg.norm(hist) + 1e-8
    return hist / norm


def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def score_with_fingerprint(search, reference, candidates):
    """Step 2: combine NCC with structural fingerprint similarity."""
    ref_fp = gradient_fingerprint(reference)
    for c in candidates:
        y0, y1 = c["y"] - c["h"] // 2, c["y"] + c["h"] // 2
        x0, x1 = c["x"] - c["w"] // 2, c["x"] + c["w"] // 2
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(search.shape[0], y1), min(search.shape[1], x1)
        patch = search[y0:y1, x0:x1]
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            c["fingerprint_sim"] = 0.0
        else:
            patch_r = cv2.resize(patch, (reference.shape[1], reference.shape[0]))
            c["fingerprint_sim"] = cosine_sim(ref_fp, gradient_fingerprint(patch_r))
        c["combined_score"] = 0.7 * c["ncc"] + 0.3 * c["fingerprint_sim"]
    candidates.sort(key=lambda c: -c["combined_score"])
    return candidates


def subpixel_refine(search, reference, candidate):
    y0, y1 = candidate["y"] - candidate["h"] // 2, candidate["y"] + candidate["h"] // 2
    x0, x1 = candidate["x"] - candidate["w"] // 2, candidate["x"] + candidate["w"] // 2
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(search.shape[0], y1), min(search.shape[1], x1)
    patch = search[y0:y1, x0:x1].astype(np.float32)
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return float(candidate["x"]), float(candidate["y"])
    ref_r = cv2.resize(reference, (patch.shape[1], patch.shape[0])).astype(np.float32)
    win = cv2.createHanningWindow((patch.shape[1], patch.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ref_r * win, patch * win)
    return candidate["x"] + dx, candidate["y"] + dy


def select_by_official_tiebreak(search, candidates):
    """
    Step 3+4 (OFFICIAL RULE): a "matching region" is a candidate whose NCC
    both clears an absolute quality floor (MIN_MATCH_QUALITY) and sits
    within NCC_TIE_TOLERANCE of the best NCC found. If more than one
    matching region is found, return the one CLOSEST TO THE CENTER of the
    search image — not simply the single highest-scoring one.

    NCC (not the fingerprint-blended score) is used for this decision — see
    module docstring / README for why fingerprint similarity alone is not a
    reliable ambiguity signal on periodic layouts.
    """
    ranked_by_ncc = sorted(candidates, key=lambda c: -c["ncc"])
    best_ncc = ranked_by_ncc[0]["ncc"]

    matching = [c for c in ranked_by_ncc
                if c["ncc"] >= MIN_MATCH_QUALITY and c["ncc"] >= best_ncc - NCC_TIE_TOLERANCE]
    if not matching:
        matching = [ranked_by_ncc[0]]

    cx_img, cy_img = search.shape[1] / 2.0, search.shape[0] / 2.0
    for c in matching:
        c["dist_to_center"] = ((c["x"] - cx_img) ** 2 + (c["y"] - cy_img) ** 2) ** 0.5

    matching.sort(key=lambda c: c["dist_to_center"])
    routing = "fast_path" if len(matching) == 1 else "deep_path_center_tiebreak"
    return matching[0], routing, len(matching)


def localize(search, reference):
    t0 = time.time()

    candidates = multiscale_multirotation_candidates(search, reference)
    if not candidates:
        raise RuntimeError("No candidates found — check image sizes/paths.")

    candidates = score_with_fingerprint(search, reference, candidates)
    chosen, routing, n_matches = select_by_official_tiebreak(search, candidates)

    rx, ry = subpixel_refine(search, reference, chosen)
    elapsed = time.time() - t0

    return {
        "predicted_x": float(rx),
        "predicted_y": float(ry),
        "routing": routing,
        "num_matching_regions": n_matches,
        "score": float(chosen["combined_score"]),
        "shrink_used": float(chosen["shrink"]),
        "rotation_used_deg": float(chosen["rotation"]),
        "inference_time_sec": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description="DriftSense-X standalone localization inference (DRAM)")
    ap.add_argument("--reference", required=True, help="Path to reference image")
    ap.add_argument("--search", required=True, help="Path to search image")
    ap.add_argument("--output", default=None, help="Optional path to write result JSON")
    args = ap.parse_args()

    search = load_gray(args.search)
    reference = load_gray(args.reference)

    result = localize(search, reference)

    print(f"Predicted center: ({result['predicted_x']:.2f}, {result['predicted_y']:.2f})")
    print(f"Routing: {result['routing']} (matching regions found: {result['num_matching_regions']})")
    print(f"Score: {result['score']:.4f}")
    print(f"Inference time: {result['inference_time_sec']*1000:.1f} ms")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
