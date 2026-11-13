
# !/usr/bin/env python3
"""
microbead_detector_optimized.py

Detects microbeads in videos with an approximation of NI Vision's
"IMAQ Match Pattern 4" (grayscale pyramid) and writes copies of each video
with a red circle drawn on the detected beads.

Motion rule: a candidate is NOT called a bead when it first appears.  It is
tracked frame to frame and only becomes a bead after it has been seen for
CONFIRM_FRAMES frames AND moved steadily (several px per frame, in a roughly
straight, consistent direction).  Smudges, dust and match "jitter" around a
fixed spot never satisfy that and are never drawn.

Expected layout (next to this script, or pass --base-dir):
    ../VideoFiles3/       video1.wmv, video2.mp4, ...
    ../TemplateImages/    template1.png, template2.png, template3.png
    ../ProcessedVideos3/  created automatically

Pipeline per video (every step maps onto LabVIEW / NI Vision):
    PASS 1  Every frame: grayscale-pyramid match at the lowest min score (300),
            keep all candidates (x, y, score).
    PASS 2  For every min score 300..700: keep candidates >= that score, link
            them frame to frame into tracks, confirm tracks that move.
    PASS 3  Re-read the video and write one annotated copy per min score.

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
VIDEO_DIR_NAME = "../VideoFiles3"
TEMPLATE_DIR_NAME = "../TemplateImages"
OUTPUT_DIR_NAME = "../ProcessedVideos3"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")
TEMPLATE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

NUM_MATCHES = 2  # beads requested; if more, the farthest apart are kept
MIN_SCORES = list(range(300, 701, 25))  # 300, 325, ... 700 (scale 1-1000)
ANGLE_RANGE = (0, 0)  # degrees (min, max); (0, 0) = no rotation search
ANGLE_STEP = 5

ALGORITHMS = ["GrayscalePyramid"]
ADVANCED_OPTIONS = ["LDContrastSubpixel"]

MIN_SEPARATION_FRACTION = 0.5  # min distance between matches, as a fraction of the template size
MAX_CANDIDATES = 20  # candidates kept per frame before bead selection
PEAKS_PER_TEMPLATE = 8  # coarse peaks refined per template and frame
PYRAMID_COARSE_MARGIN = 0.15  # coarse-stage threshold = lowest min score - margin

# ---- motion rule ---------------------------------------------------------------
MOTION_FILTER = True  # False = every candidate >= min score is a bead
CONFIRM_FRAMES = 5  # a candidate must be seen this many frames before it can be a bead
MIN_MEAN_STEP_PX = 0.8  # average frame-to-frame movement inside the confirm window
MIN_STEP_PX = 0.5  # a step counts as "moving" if it is at least this many px
MIN_MOVING_STEPS = 0.75  # fraction of steps that must count as moving (a 1-2 frame blip fails)
MAX_LINK_DIST = 60.0  # px: max jump between frames to still be the same object
MAX_GAP = 1  # frames an object may vanish and still keep its track
BEAD_GRACE = 5  # frames a confirmed bead stays a bead after it stops passing the test

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


# =============================================================================
# TEMPLATE VARIANTS
# =============================================================================
class Rep:
    """A template representation (grayscale) plus its coarse copy."""

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

    @staticmethod
    def _auto_levels(side):
        levels = 0
        while side // 2 >= 12 and levels < 3:
            side //= 2
            levels += 1
        return levels


# =============================================================================
# GRAYSCALE PYRAMID SEARCH
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


# =============================================================================
# DETECTION (PASS 1)
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


def detect(frame_bgr, variants, floor, subpixel, min_sep):
    """All candidate matches at or above the lowest min score, best first."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    fp = FramePyramid(gray)
    thr = max(0.0, floor - PYRAMID_COARSE_MARGIN)
    cands = []
    for v in variants:
        cands += pyramid_search(fp, v.gray, v, floor, thr, subpixel)
    return suppress_close(cands, min_sep)


# =============================================================================
# MOTION FILTER (PASS 2)
# =============================================================================
def window_is_moving(win):
    """win: last CONFIRM_FRAMES observations [(frame, Match)] of one track.

    A bead keeps moving (jitters / drifts) every frame; a smudge or stuck
    particle holds its position to a fraction of a pixel.  So the window must
    show movement in almost every step, not just one or two.
    """
    if len(win) < CONFIRM_FRAMES:
        return False
    if win[-1][0] - win[0][0] > CONFIRM_FRAMES - 1 + MAX_GAP:  # too many missing frames
        return False
    steps = [math.hypot(b[1].cx - a[1].cx, b[1].cy - a[1].cy) / (b[0] - a[0])
             for a, b in zip(win, win[1:])]
    if sum(steps) / len(steps) < MIN_MEAN_STEP_PX:
        return False
    return sum(s >= MIN_STEP_PX for s in steps) >= MIN_MOVING_STEPS * len(steps)


def find_beads(per_frame, min_score):
    """Returns beads[f] = list of Match drawn as beads in frame f for this min score.

    Candidates below min_score are ignored.  Each remaining candidate is linked
    to the nearest track (last position, or the position predicted from its
    velocity).  A track starts confirming once it has
    CONFIRM_FRAMES observations; from then on its matches are beads while the
    last CONFIRM_FRAMES observations pass window_is_moving (plus BEAD_GRACE).
    """
    beads = [[] for _ in per_frame]
    tracks = []  # {"obs": [...], "bead_until": frame}
    for f, cands in enumerate(per_frame):
        tracks = [t for t in tracks if f - t["obs"][-1][0] <= MAX_GAP + 1]
        used = set()
        for m in cands:
            if m.score < min_score:
                continue
            best, best_d = None, MAX_LINK_DIST
            for ti, tr in enumerate(tracks):
                if ti in used:
                    continue
                lf, lm = tr["obs"][-1]
                d = math.hypot(m.cx - lm.cx, m.cy - lm.cy)
                if len(tr["obs"]) >= 2:  # velocity prediction
                    pf, pm = tr["obs"][-2]
                    dt = lf - pf
                    vx, vy = (lm.cx - pm.cx) / dt, (lm.cy - pm.cy) / dt
                    k = f - lf
                    d = min(d, math.hypot(m.cx - (lm.cx + vx * k), m.cy - (lm.cy + vy * k)))
                if d <= best_d:
                    best, best_d = ti, d
            if best is None:
                tracks.append({"obs": [(f, m)], "bead_until": -1})
                used.add(len(tracks) - 1)
                continue
            tr = tracks[best]
            used.add(best)
            tr["obs"].append((f, m))
            if len(tr["obs"]) > 4 * CONFIRM_FRAMES:
                del tr["obs"][:-CONFIRM_FRAMES]
            if not MOTION_FILTER or window_is_moving(tr["obs"][-CONFIRM_FRAMES:]):
                tr["bead_until"] = f + BEAD_GRACE
            if f <= tr["bead_until"]:
                beads[f].append(m)
        if not MOTION_FILTER:  # no tracking needed: every candidate is a bead
            beads[f] = [m for m in cands if m.score >= min_score]
    return beads


# =============================================================================
# BEAD SELECTION AND DRAWING (PASS 3)
# =============================================================================
def choose_beads(cands, min_score, k):
    """Confirmed beads at/above min_score. If more than k, keep the k farthest apart."""
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
def find_numbered_files(folder, prefix, extensions):
    """prefix1.ext, prefix2.ext, ... in numeric order. Gaps and extras are fine."""
    pat = re.compile(rf"^{prefix}(\d+)$", re.IGNORECASE)
    found = {}
    for name in os.listdir(folder):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in extensions:
            continue
        m = pat.match(stem)
        if m:
            found.setdefault(int(m.group(1)), Path(folder) / name)
    return [found[k] for k in sorted(found)]


def output_name(stem, score, algo, adv):
    return f"{stem}_MinScore{score}_{algo}_{adv}.mp4"


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def detect_all(video_path, variants, floor, subpixel, min_sep, max_frames):
    """PASS 1: candidate list for every frame. Returns (per_frame, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    per_frame = []
    t0 = time.time()
    try:
        while max_frames <= 0 or len(per_frame) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            per_frame.append(detect(frame, variants, floor, subpixel, min_sep))
            n = len(per_frame)
            if n % 25 == 0:
                tot = f"/{total}" if total else ""
                print(f"         detect frame {n}{tot}  ({n / (time.time() - t0):.1f} fps)",
                      end="\r", flush=True)
    finally:
        cap.release()
    return per_frame, fps


def process_video(video_path, algo, adv, variants, out_dir, min_sep, max_frames=0):
    """One detection pass + one motion pass; writes every missing min-score output."""
    stem = video_path.stem
    todo = {}
    for s in MIN_SCORES:
        final = out_dir / output_name(stem, s, algo, adv)
        if not final.exists():
            todo[s] = final
    if not todo:
        print(f"  [skip] {stem} / {algo} / {adv}: all outputs already exist")
        return

    floor = min(MIN_SCORES) / 1000.0
    subpixel = (adv == "LDContrastSubpixel")
    t0 = time.time()
    print(f"  [run ] {stem} / {algo} / {adv}: {len(todo)} of {len(MIN_SCORES)} outputs to make")

    per_frame, fps = detect_all(video_path, variants, floor, subpixel, min_sep, max_frames)
    if per_frame is None:
        print(f"  [warn] could not open {video_path.name}, skipping")
        return
    if not per_frame:
        print("  [warn] no frames could be read, skipping")
        return

    beads_by_score = {}
    for s in todo:
        beads_by_score[s] = find_beads(per_frame, s)
        n_all = sum(1 for c in per_frame for m in c if m.score >= s)
        n_bead = sum(len(b) for b in beads_by_score[s])
        print(f"         min score {s}: {n_bead} bead detections from {n_all} candidates")

    cap = cv2.VideoCapture(str(video_path))
    writers = {}
    ok_run = False
    frames = 0
    t_write = time.time()
    print(f"         writing {len(todo)} output videos...")
    try:
        while frames < len(per_frame):
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

            drawn = {}  # identical selections share one annotated frame
            for s, (wr, _, _) in writers.items():
                chosen = choose_beads(beads_by_score[s][frames], s, NUM_MATCHES)
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
                print(f"         writing frame {frames}/{len(per_frame)}  "
                      f"({frames / (time.time() - t_write):.1f} fps)",
                      end="\r", flush=True)
        ok_run = frames > 0
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
    ap = argparse.ArgumentParser(description="Microbead detection on video files")
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent),
                    help="Folder the ../VideoFiles3 etc. paths are relative to "
                         "(default: this script's folder)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Debug: only process the first N frames of each video")
    args = ap.parse_args()

    base = Path(args.base_dir)
    video_dir = (base / VIDEO_DIR_NAME).resolve()
    template_dir = (base / TEMPLATE_DIR_NAME).resolve()
    out_dir = (base / OUTPUT_DIR_NAME).resolve()

    for d, label in ((video_dir, VIDEO_DIR_NAME), (template_dir, TEMPLATE_DIR_NAME)):
        if not d.is_dir():
            sys.exit(f"Folder not found: {d}  ({label} relative to {base})")
    out_dir.mkdir(exist_ok=True)

    for stale in out_dir.glob("*.partial.mp4"):  # left over from an interrupted run
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

    # ---- videos ----
    videos = find_numbered_files(video_dir, "video", VIDEO_EXTENSIONS)
    if not videos:
        sys.exit(f"No videos (video1, video2, ...) found in {video_dir}")
    print(f"Found {len(videos)} video(s). Output folder: {out_dir}\n")

    t_start = time.time()
    try:
        for vp in videos:
            print(f"{vp.name}")
            for algo in ALGORITHMS:
                for adv in ADVANCED_OPTIONS:
                    try:
                        process_video(vp, algo, adv, variants, out_dir, min_sep,
                                      args.max_frames)
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