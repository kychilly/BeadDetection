#!/usr/bin/env python3
"""
microbead_detector_gradient_ld.py

Detects microbeads in videos with an approximation of NI Vision's
"IMAQ Match Pattern 4" and writes copies of each video with a red circle
drawn on the detected beads.

This version runs ONLY:
    - GradientPyramid
    - LowDiscrepancy
and ONLY on video1, video2 and video3.

Expected layout (next to this script, or pass --base-dir):
    VideoFiles/       video1.mp4, video2.mp4, video3.mp4, ...
    TemplateImages/   template1.png, template2.png, template3.png

Output folders (created automatically if missing):
    ProcessedGrayscalePyramid/          GradientPyramid runs
    ProcessedLowDiscrepancySampling/    LowDiscrepancy runs

For every video, every algorithm and every min match score (500..900 step 25)
one output file is written, e.g.:
    ProcessedLowDiscrepancySampling/video1_MinScore500_LowDiscrepancy_LDContrastSubpixel.mp4

Outputs that already exist are skipped, so the program can be stopped and
restarted at any time.

Dependencies:  pip install opencv-python numpy
"""

import argparse
import gc
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
VIDEO_DIR_NAME = "../VideoFiles"
TEMPLATE_DIR_NAME = "../TemplateImages"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")
TEMPLATE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

VIDEO_NUMBERS = (1, 2, 3)  # only video1, video2, video3 are processed

# Output folder for each algorithm (created if missing)
OUTPUT_DIR_NAMES = {
    "GradientPyramid": "ProcessedGrayscalePyramid",
    "LowDiscrepancy": "ProcessedLowDiscrepancySampling",
}

NUM_MATCHES = 2  # beads requested; if more, the farthest apart are kept
MIN_SCORES = list(range(500, 901, 25))  # 500, 525, ... 900 (scale 1-1000)
ANGLE_RANGE = (0, 0)  # degrees (min, max); (0, 0) = no rotation search
ANGLE_STEP = 5

ALGORITHMS = ["GradientPyramid", "LowDiscrepancy"]
ADVANCED_OPTIONS = ["LDContrastSubpixel"]

MIN_SEPARATION_FRACTION = 0.5  # min distance between matches, as a fraction of the template size
MAX_CANDIDATES = 10  # candidates kept per frame before bead selection
PEAKS_PER_TEMPLATE = 8  # coarse peaks refined per template and frame
PYRAMID_COARSE_MARGIN = 0.15  # coarse-stage threshold = lowest min score - margin
LD_COARSE_MARGIN = 0.20
LD_MIN_CONTRAST = 3.0  # gray levels (std dev); used when the advanced option is on

CIRCLE_COLOR = (0, 0, 255)  # BGR red
CIRCLE_THICKNESS = 2
CIRCLE_PADDING = 3  # px added to the template radius
SHIFT = 4  # fractional bits used when drawing (subpixel circles)


# =============================================================================
# DATA
# =============================================================================
@dataclass
class Match:
    score: float  # 0-1000
    cx: float
    cy: float
    angle: float
    w: int  # size of the original (unrotated) template
    h: int


# =============================================================================
# IMAGE HELPERS
# =============================================================================
def gradient_mag(gray_u8):
    g = cv2.GaussianBlur(gray_u8, (3, 3), 0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def pyr_down_n(img, levels):
    for _ in range(levels):
        img = cv2.pyrDown(img)
    return img


class FramePyramid:
    """Lazily built image pyramid."""

    def __init__(self, base):
        self.imgs = [base]

    def get(self, level):
        while len(self.imgs) <= level:
            self.imgs.append(cv2.pyrDown(self.imgs[-1]))
        return self.imgs[level]


def run_match(image, tmpl, mask):
    if image.shape[0] < tmpl.shape[0] or image.shape[1] < tmpl.shape[1]:
        return None
    if mask is not None:
        res = cv2.matchTemplate(image, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
    else:
        res = cv2.matchTemplate(image, tmpl, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(res, 0.0, 1.0).astype(np.float32)


def top_peaks(res, thresh, k, cap=PEAKS_PER_TEMPLATE):
    k = max(3, int(k) | 1)
    dil = cv2.dilate(res, np.ones((k, k), np.uint8))
    ys, xs = np.where((res >= dil) & (res >= thresh))
    peaks = list(zip(xs.tolist(), ys.tolist()))
    peaks.sort(key=lambda p: -res[p[1], p[0]])
    return peaks[:cap]


def subpixel_offset(res, x, y):
    """Parabolic peak interpolation, clamped to +/-0.5 px."""
    dx = dy = 0.0
    if 0 < x < res.shape[1] - 1:
        l, c, r = float(res[y, x - 1]), float(res[y, x]), float(res[y, x + 1])
        d = l - 2 * c + r
        if abs(d) > 1e-9:
            dx = 0.5 * (l - r) / d
    if 0 < y < res.shape[0] - 1:
        u, c, b = float(res[y - 1, x]), float(res[y, x]), float(res[y + 1, x])
        d = u - 2 * c + b
        if abs(d) > 1e-9:
            dy = 0.5 * (u - b) / d
    return max(-0.5, min(0.5, dx)), max(-0.5, min(0.5, dy))


def halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


# =============================================================================
# TEMPLATE VARIANTS
# =============================================================================
class Rep:
    """A template representation (grayscale or gradient) plus its coarse copy."""

    def __init__(self, img, mask255, levels):
        self.full = img
        self.fmask = None if mask255 is None else (mask255 > 127).astype(img.dtype)
        self.coarse = pyr_down_n(img, levels)
        if mask255 is None:
            self.cmask = None
        else:
            cm = pyr_down_n(mask255, levels)
            self.cmask = (cm > 127).astype(img.dtype)


class Variant:
    """One (possibly rotated) version of a template with everything precomputed."""

    def __init__(self, tmpl, angle):
        h, w = tmpl.shape
        self.orig_size = (w, h)
        self.angle = angle

        if angle == 0:
            img, mask255 = tmpl, None
        else:
            diag = int(math.ceil(math.hypot(w, h)))
            canvas = np.zeros((diag, diag), np.uint8)
            ones = np.zeros((diag, diag), np.uint8)
            ox, oy = (diag - w) // 2, (diag - h) // 2
            canvas[oy:oy + h, ox:ox + w] = tmpl
            ones[oy:oy + h, ox:ox + w] = 255
            M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
            img = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR)
            mask255 = cv2.warpAffine(ones, M, (diag, diag), flags=cv2.INTER_NEAREST)

        self.img = img
        self.shape = img.shape
        self.levels = self._auto_levels(min(self.shape))
        self.gray = Rep(img, mask255, self.levels)
        self.grad = Rep(gradient_mag(img), mask255, self.levels)
        self._build_ld(mask255)

    @staticmethod
    def _auto_levels(side):
        levels = 0
        while side // 2 >= 12 and levels < 3:
            side //= 2
            levels += 1
        return levels

    def _build_ld(self, mask255):
        """Low-discrepancy (Halton) sample points inside the template."""
        th, tw = self.shape
        valid = th * tw if mask255 is None else int((mask255 > 127).sum())
        target = int(np.clip(valid // 3, 32, 400))
        pts, seen, i = [], set(), 1
        while len(pts) < target and i < target * 12:
            x = int(halton(i, 2) * tw)
            y = int(halton(i, 3) * th)
            i += 1
            if (x, y) in seen:
                continue
            if mask255 is not None and mask255[y, x] <= 127:
                continue
            seen.add((x, y))
            pts.append((x, y))

        self.ld_t = None
        self.ld_stride = int(np.clip(min(th, tw) // 10, 1, 4))
        if len(pts) < 8:
            return
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        t = self.img[ys, xs].astype(np.float32)
        t -= t.mean()
        sum_t2 = float((t * t).sum())
        if sum_t2 < 1e-3:  # flat template, cannot be correlated
            return
        self.ld_dx, self.ld_dy, self.ld_t, self.ld_sum_t2 = xs, ys, t, sum_t2


# =============================================================================
# SEARCH STRATEGIES
# =============================================================================
def refine_peaks(frame, var, tmpl, tmask, origins, pad, fine_thr, subpixel):
    """Full-resolution correlation in a small window around each coarse origin."""
    H, W = frame.shape[:2]
    th, tw = tmpl.shape[:2]
    out = []
    for ox, oy in origins:
        x0, y0 = max(0, ox - pad), max(0, oy - pad)
        x1, y1 = min(W, ox + tw + pad), min(H, oy + th + pad)
        res = run_match(frame[y0:y1, x0:x1], tmpl, tmask)
        if res is None:
            continue
        _, maxv, _, (bx, by) = cv2.minMaxLoc(res)
        if maxv < fine_thr:
            continue
        dx, dy = subpixel_offset(res, bx, by) if subpixel else (0.0, 0.0)
        out.append(Match(maxv * 1000.0,
                         x0 + bx + dx + (tw - 1) / 2.0,
                         y0 + by + dy + (th - 1) / 2.0,
                         var.angle, *var.orig_size))
    return out


def pyramid_search(fpyr, rep, var, fine_thr, coarse_thr, subpixel):
    levels = var.levels
    scale = 2 ** levels
    res_c = run_match(fpyr.get(levels), rep.coarse, rep.cmask)
    if res_c is None:
        return []
    ch, cw = rep.coarse.shape[:2]

    if levels == 0:
        th, tw = rep.full.shape[:2]
        out = []
        for px, py in top_peaks(res_c, fine_thr, min(cw, ch) // 2):
            dx, dy = subpixel_offset(res_c, px, py) if subpixel else (0.0, 0.0)
            out.append(Match(float(res_c[py, px]) * 1000.0,
                             px + dx + (tw - 1) / 2.0, py + dy + (th - 1) / 2.0,
                             var.angle, *var.orig_size))
        return out

    peaks = top_peaks(res_c, coarse_thr, min(cw, ch) // 2)
    origins = [(px * scale, py * scale) for px, py in peaks]
    return refine_peaks(fpyr.get(0), var, rep.full, rep.fmask, origins,
                        scale * 2 + 2, fine_thr, subpixel)


def ld_scores(frame_f, var, min_std):
    """Normalized correlation using only the low-discrepancy sample points,
    evaluated on a strided grid of template positions."""
    th, tw = var.shape
    H, W = frame_f.shape
    if H < th or W < tw:
        return None
    s = var.ld_stride
    gy, gx = (H - th) // s + 1, (W - tw) // s + 1
    s1 = np.zeros((gy, gx), np.float32)
    s2 = np.zeros((gy, gx), np.float32)
    s12 = np.zeros((gy, gx), np.float32)
    for dy, dx, t in zip(var.ld_dy, var.ld_dx, var.ld_t):
        sl = frame_f[dy:dy + gy * s:s, dx:dx + gx * s:s]
        s1 += sl
        s2 += sl * sl
        s12 += sl * t
    n = len(var.ld_t)
    var_i = s2 - s1 * s1 / n
    ncc = s12 / np.sqrt(np.maximum(var_i, 1e-6) * var.ld_sum_t2)
    ncc[var_i < n * min_std * min_std] = 0.0
    return np.clip(np.nan_to_num(ncc), 0.0, 1.0)


def ld_search(gray_u8, gray_f, var, fine_thr, coarse_thr, subpixel, min_std):
    if var.ld_t is None:
        return []
    ncc = ld_scores(gray_f, var, min_std)
    if ncc is None:
        return []
    s = var.ld_stride
    peaks = top_peaks(ncc, coarse_thr, max(3, min(var.shape) // (2 * s)))
    origins = [(px * s, py * s) for px, py in peaks]
    return refine_peaks(gray_u8, var, var.gray.full, var.gray.fmask, origins,
                        s * 2 + 2, fine_thr, subpixel)


# =============================================================================
# DETECTION AND BEAD SELECTION
# =============================================================================
def suppress_close(cands, min_sep):
    cands.sort(key=lambda m: m.score, reverse=True)
    kept = []
    for m in cands:
        if all(math.hypot(m.cx - k.cx, m.cy - k.cy) >= min_sep for k in kept):
            kept.append(m)
            if len(kept) >= MAX_CANDIDATES:
                break
    return kept


def detect(frame_bgr, algo, variants, floor, subpixel, min_sep):
    """All candidate matches at or above the lowest min score, best first."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    fine_thr = floor
    cands = []

    if algo == "GradientPyramid":
        fp = FramePyramid(gradient_mag(gray))
        thr = max(0.0, floor - PYRAMID_COARSE_MARGIN)
        for v in variants:
            cands += pyramid_search(fp, v.grad, v, fine_thr, thr, subpixel)
    elif algo == "LowDiscrepancy":
        gray_f = gray.astype(np.float32) - 128.0
        thr = max(0.0, floor - LD_COARSE_MARGIN)
        min_std = LD_MIN_CONTRAST if subpixel else 0.0
        for v in variants:
            cands += ld_search(gray, gray_f, v, fine_thr, thr, subpixel, min_std)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    return suppress_close(cands, min_sep)


def choose_beads(cands, min_score, k):
    """Matches at/above min_score. If more than k, keep the k farthest apart."""
    pool = [m for m in cands if m.score >= min_score]
    if len(pool) <= k:
        return pool
    if k <= 1:
        return pool[:1]

    best, bi, bj = -1.0, 0, 1
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            d = math.hypot(pool[i].cx - pool[j].cx, pool[i].cy - pool[j].cy)
            if d > best:
                best, bi, bj = d, i, j
    chosen = [pool[bi], pool[bj]]
    while len(chosen) < k:
        rest = [m for m in pool if m not in chosen]
        if not rest:
            break
        chosen.append(max(rest, key=lambda m: min(math.hypot(m.cx - c.cx, m.cy - c.cy)
                                                  for c in chosen)))
    return chosen


def draw_bead(img, m):
    r = max(m.w, m.h) / 2.0 + CIRCLE_PADDING
    f = 1 << SHIFT
    cv2.circle(img, (int(round(m.cx * f)), int(round(m.cy * f))), int(round(r * f)),
               CIRCLE_COLOR, CIRCLE_THICKNESS, cv2.LINE_AA, SHIFT)


# =============================================================================
# FILE HANDLING
# =============================================================================
def find_numbered_files(folder, prefix, extensions, only_numbers=None):
    """prefix1.ext, prefix2.ext, ... in numeric order. Gaps and extras are fine.
    If only_numbers is given, only those numbers are returned."""
    pat = re.compile(rf"^{prefix}(\d+)$", re.IGNORECASE)
    found = {}
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in extensions:
            continue
        m = pat.match(stem)
        if m:
            num = int(m.group(1))
            if only_numbers is not None and num not in only_numbers:
                continue
            found.setdefault(num, Path(folder) / name)
    return [found[k] for k in sorted(found)]


def output_name(stem, score, algo, adv):
    return f"{stem}_MinScore{score}_{algo}_{adv}.mp4"


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def process_video(video_path, algo, adv, variants, out_dir, min_sep):
    """One pass over the video produces the output for every missing min score."""
    stem = video_path.stem
    todo = {}
    for s in MIN_SCORES:
        final = out_dir / output_name(stem, s, algo, adv)
        if not final.exists():
            todo[s] = final
    if not todo:
        print(f"  [skip] {stem} / {algo} / {adv}: all outputs already exist")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] could not open {video_path.name}, skipping")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    floor = min(MIN_SCORES) / 1000.0
    subpixel = (adv == "LDContrastSubpixel")
    writers = {}  # min score -> (writer, partial path, final path)
    ok_run = False
    frames = 0
    t0 = time.time()

    print(f"  [run ] {stem} / {algo} / {adv}: {len(todo)} of {len(MIN_SCORES)} outputs to make"
          f"  -> {out_dir.name}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if not writers:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                for s, final in todo.items():
                    part = final.with_name(final.stem + ".partial.mp4")
                    wr = cv2.VideoWriter(str(part), fourcc, fps, (w, h))
                    if not wr.isOpened():
                        raise RuntimeError("could not open a video writer (mp4v codec)")
                    writers[s] = (wr, part, final)

            cands = detect(frame, algo, variants, floor, subpixel, min_sep)

            drawn = {}  # identical selections share one annotated frame
            for s, (wr, _, _) in writers.items():
                chosen = choose_beads(cands, s, NUM_MATCHES)
                key = tuple(sorted((round(m.cx, 1), round(m.cy, 1), m.w, m.h)
                                   for m in chosen))
                img = drawn.get(key)
                if img is None:
                    if chosen:
                        img = frame.copy()
                        for m in chosen:
                            draw_bead(img, m)
                    else:
                        img = frame
                    drawn[key] = img
                wr.write(img)

            frames += 1
            if frames % 25 == 0:
                tot = f"/{total}" if total else ""
                print(f"         frame {frames}{tot}  ({frames / (time.time() - t0):.1f} fps)",
                      end="\r", flush=True)

        ok_run = frames > 0
        if not ok_run:
            print("  [warn] no frames could be read, skipping")
    finally:
        cap.release()
        for wr, _, _ in writers.values():
            wr.release()
        if ok_run:
            for _, part, final in writers.values():
                os.replace(part, final)
        else:
            for _, part, _ in writers.values():
                remove_quietly(part)  # interrupted or failed: leave no partial file
        writers.clear()
        gc.collect()

    if ok_run:
        print(f"  [done] {stem} / {algo} / {adv}: {frames} frames in {time.time() - t0:.0f}s"
              + " " * 20)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Microbead detection (GradientPyramid + LowDiscrepancy) on videos 1-3")
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent),
                    help="Folder containing VideoFiles and TemplateImages "
                         "(default: this script's folder)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    video_dir = base / VIDEO_DIR_NAME
    template_dir = base / TEMPLATE_DIR_NAME

    for d, label in ((video_dir, VIDEO_DIR_NAME), (template_dir, TEMPLATE_DIR_NAME)):
        if not d.is_dir():
            sys.exit(f"Folder not found: {d}  ({label} must be in {base})")

    # ---- output folders (created if missing) ----
    out_dirs = {}
    for algo in ALGORITHMS:
        out_dirs[algo] = base / OUTPUT_DIR_NAMES[algo]
        out_dirs[algo].mkdir(parents=True, exist_ok=True)
        for stale in out_dirs[algo].glob("*.partial.mp4"):  # from an interrupted run
            remove_quietly(stale)

    # ---- templates ----
    if ANGLE_RANGE[0] == ANGLE_RANGE[1] or ANGLE_STEP <= 0:
        angles = [ANGLE_RANGE[0]]
    else:
        lo, hi = sorted(ANGLE_RANGE)
        angles = [float(a) for a in np.arange(lo, hi + 1e-9, ANGLE_STEP)]

    variants, sizes = [], []
    for tp in find_numbered_files(template_dir, "template", TEMPLATE_EXTENSIONS):
        img = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            print(f"[warn] could not read template {tp.name}, skipping")
            continue
        sizes.append(min(img.shape))
        variants += [Variant(img, a) for a in angles]
        print(f"Loaded {tp.name} ({img.shape[1]}x{img.shape[0]})")
    if not variants:
        sys.exit(f"No usable templates (template1, template2, ...) in {template_dir}")
    min_sep = max(2.0, MIN_SEPARATION_FRACTION * min(sizes))

    # ---- videos (only video1, video2, video3) ----
    videos = find_numbered_files(video_dir, "video", VIDEO_EXTENSIONS,
                                 only_numbers=VIDEO_NUMBERS)
    found_nums = {int(re.search(r"(\d+)$", v.stem).group(1)) for v in videos}
    for n in VIDEO_NUMBERS:
        if n not in found_nums:
            print(f"[warn] video{n} not found in {video_dir}")
    if not videos:
        sys.exit(f"None of video{', video'.join(map(str, VIDEO_NUMBERS))} found in {video_dir}")
    print(f"Found {len(videos)} video(s). Algorithms: {', '.join(ALGORITHMS)}")
    for algo in ALGORITHMS:
        print(f"  {algo} -> {out_dirs[algo]}")
    print()

    t_start = time.time()
    try:
        for vp in videos:
            print(f"{vp.name}")
            for algo in ALGORITHMS:
                for adv in ADVANCED_OPTIONS:
                    try:
                        process_video(vp, algo, adv, variants, out_dirs[algo], min_sep)
                    except Exception as exc:  # one bad video/pass must not stop the run
                        print(f"  [error] {vp.name} / {algo}: {exc}")
                    gc.collect()
            gc.collect()
    except KeyboardInterrupt:
        print("\nStopped by user. Finished outputs are kept; rerun to continue.")
        return
    print(f"\nAll done in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()