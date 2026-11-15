#!/usr/bin/env python3
"""
microbead_detector_optimized.py

Detects microbeads in videos and writes copies of each video with circles drawn
on them:

    ORANGE circle  =  POTENTIAL bead.  Appears on the very first frame in which
                      something moves (a moving bead is blurry and cannot be
                      recognised from one image, but it shows up instantly as a
                      moving blob when the frame is compared with the frames
                      just before it).
    RED circle     =  CONFIRMED bead.  A potential bead that has been followed
                      for CONFIRM_SECONDS (0.33 s) and whose 0.33 s of track
                      history shows real travel across the solution.  Once red,
                      it stays red while it is tracked (also after it gets
                      trapped and stops moving).

How a potential bead is found (motion detector, every frame):
    * each frame is high-pass filtered (removes the slow brightness gradient)
    * a "background" is built as the median of the previous ~8 frames
      (skipping the last 2) -> anything that is not in that background is new
    * pixels that differ by more than MOTION_THRESH are grouped into blobs; the
      blobs of the right size/shape are the potential beads
    * blobs sitting right next to a long-standing static object (e.g. the ring
      in the left of the image, whose texture flickers) are ignored

How a potential bead becomes a real bead (judged only from 0.33 s of frames):
    * it was seen in most of those frames
    * it travelled (extent across the solution >= MIN_EXTENT_PX)
    * it kept moving (most frame-to-frame steps >= MIN_STEP_PX)
    * it did not just go back and forth (net / path >= MIN_EFFICIENCY)
  Smudges, dust, flicker and match jitter stay near one spot and fail this.

The grayscale-pyramid template match (IMAQ Match Pattern 4 replica) is still
used: matches at or above the min score keep a confirmed bead tracked after it
stops moving (trapped bead), where there is no motion blob any more.

Expected layout (next to this script, or pass --base-dir):
    ../VideoFiles3/       video1.wmv, video2.mp4, ...
    ../TemplateImages/    template1.png, template2.png, template3.png
    ../ProcessedVideos3/  created automatically

Dependencies:  pip install opencv-python numpy
"""

import argparse
import gc
import math
import os
import re
import sys
import time
from collections import deque
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

NUM_MATCHES = 2  # red beads drawn per frame; if more, the farthest apart are kept
MIN_SCORES = list(range(300, 701, 25))  # 300, 325, ... 700 (scale 1-1000)
ANGLE_RANGE = (0, 0)  # degrees (min, max); (0, 0) = no rotation search
ANGLE_STEP = 5

ALGORITHMS = ["GrayscalePyramid"]
ADVANCED_OPTIONS = ["LDContrastSubpixel"]

MIN_SEPARATION_FRACTION = 0.5  # min distance between matches, as a fraction of the template size
MAX_CANDIDATES = 20  # template candidates kept per frame
PEAKS_PER_TEMPLATE = 8  # coarse peaks refined per template and frame
PYRAMID_COARSE_MARGIN = 0.15  # coarse-stage threshold = lowest min score - margin

# ---- motion detector (potential beads, found from the very first frame) ----
MOTION_SCALE = 0.5          # motion analysis runs on a half-size image (speed)
MOTION_HP_SIGMA_SMALL = 1.5  # high-pass = blur(small) - blur(large)
MOTION_HP_SIGMA_LARGE = 15.0
BG_FRAMES = 8               # background = median of this many previous frames ...
BG_GAP = 2                  # ... skipping the most recent BG_GAP frames
MIN_BG_FRAMES = 3           # motion detection starts when this many are available
MOTION_THRESH = 4.0         # gray-level difference that counts as "changed"
MOTION_MIN_PEAK = 6.0       # a blob must reach this difference somewhere
MOTION_MIN_AREA = 150       # blob area range in full-resolution pixels
MOTION_MAX_AREA = 9000
MOTION_MIN_FILL = 0.30      # blob area / bounding box area
MOTION_MAX_ASPECT = 3.5     # long side / short side of the bounding box
MOTION_MIN_SEP = 100.0      # px; of two blobs closer than this only the stronger is kept
MAX_MOTION_BLOBS = 6
MOTION_R_MIN, MOTION_R_MAX = 27.0, 50.0  # circle radius range for motion blobs

# ---- potential -> confirmed bead (times in seconds, converted with the fps) ----
MOTION_FILTER = True        # False = every template candidate >= min score is a bead
CONFIRM_SECONDS = 0.33      # a potential bead is judged after this long
MIN_SEEN_FRACTION = 0.7     # it must have been seen in this fraction of those frames
MIN_EXTENT_PX = 60.0        # largest distance between any two positions in that time
MIN_EFFICIENCY = 0.15       # net distance / path length (very low = just back and forth)
MIN_STEP_PX = 8.0           # px per frame that counts as "really moving"
MIN_STEP_FRACTION = 0.6     # this fraction of the frame-to-frame steps must be that big
LINK_BASE_GATE_PX = 15.0    # a slow / still track only links to blobs this close (per frame)
LINK_SPEED_FACTOR = 2.5     # ...plus this multiple of its last speed (px/frame)
MAX_LINK_DIST = 170.0       # px per frame: gate for brand-new tracks and upper limit
TEMPLATE_JOIN_PX = 45.0     # a template match may only join a track this close to it
MAX_GAP_SECONDS = 0.2       # a track may go unobserved this long before it is dropped
POTENTIAL_HOLD_FRAMES = 2   # orange circle is kept this many frames without a new sighting

# ---- static objects (ignored by the motion detector) ----
STATIC_RADIUS_PX = 6.0      # a template match this close to the last one is "the same, not moving"
STATIC_SECONDS = 0.5        # ...for this long makes it a static object
ANCHOR_RADIUS_PX = 60.0     # new motion blobs this close to a static object are ignored
ANCHOR_MIN_SCORE = 400      # static objects are looked for in matches >= max(min score, this)

ORANGE = (0, 165, 255)      # BGR: potential bead
RED = (0, 0, 255)           # BGR: confirmed bead
CIRCLE_THICKNESS = 2
CIRCLE_PADDING = 3  # px added to the template radius
SHIFT = 4  # fractional bits used when drawing (subpixel circles)


# =============================================================================
# DATA
# =============================================================================
@dataclass
class Match:
    score: float  # 0-1000 (0 for motion blobs)
    cx: float
    cy: float
    angle: float
    w: int  # size of the original (unrotated) template (motion blobs: circle diameter)
    h: int
    src: str = "tmpl"  # "tmpl" = template match, "motion" = moving blob
    peak: float = 0.0  # motion blobs: strongest difference to the background


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


def suppress_close(cands, min_sep):
    cands.sort(key=lambda m: m.score, reverse=True)
    kept = []
    for m in cands:
        if all(math.hypot(m.cx - k.cx, m.cy - k.cy) >= min_sep for k in kept):
            kept.append(m)
            if len(kept) >= MAX_CANDIDATES:
                break
    return kept


def detect_templates(gray, variants, floor, subpixel, min_sep):
    """All template candidates at or above the lowest min score, best first."""
    fp = FramePyramid(gray)
    thr = max(0.0, floor - PYRAMID_COARSE_MARGIN)
    cands = []
    for v in variants:
        cands += pyramid_search(fp, v.gray, v, floor, thr, subpixel)
    return suppress_close(cands, min_sep)


# =============================================================================
# MOTION DETECTOR (potential beads, available from the first moving frame)
# =============================================================================
class MotionDetector:
    """Finds blobs that are new compared with the last few frames."""

    def __init__(self):
        self.buf = deque(maxlen=BG_FRAMES + BG_GAP)
        self.k_open = np.ones((3, 3), np.uint8)
        self.k_close = np.ones((7, 7), np.uint8)

    def update(self, gray):
        s = cv2.resize(gray, None, fx=MOTION_SCALE, fy=MOTION_SCALE,
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        hp = (cv2.GaussianBlur(s, (0, 0), MOTION_HP_SIGMA_SMALL)
              - cv2.GaussianBlur(s, (0, 0), MOTION_HP_SIGMA_LARGE))
        blobs = []
        if len(self.buf) >= BG_GAP + MIN_BG_FRAMES:
            hist = list(self.buf)[:len(self.buf) - BG_GAP]
            bg = np.median(np.stack(hist), axis=0)
            d = np.abs(hp - bg)
            blobs = self._blobs(d)
        self.buf.append(hp)
        return blobs

    def _blobs(self, d):
        m = (d > MOTION_THRESH).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k_close)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        inv = 1.0 / (MOTION_SCALE * MOTION_SCALE)
        found = []
        for k in range(1, n):
            x, y, w, h, a = (int(v) for v in st[k])
            area = a * inv
            if not (MOTION_MIN_AREA <= area <= MOTION_MAX_AREA):
                continue
            if a / float(w * h) < MOTION_MIN_FILL or max(w, h) / float(min(w, h)) > MOTION_MAX_ASPECT:
                continue
            sub = d[y:y + h, x:x + w][lab[y:y + h, x:x + w] == k]
            peak = float(sub.max())
            if peak < MOTION_MIN_PEAK:
                continue
            r = min(MOTION_R_MAX, max(MOTION_R_MIN, math.sqrt(area / math.pi)))
            size = int(round(2 * r))
            found.append(Match(0.0, cen[k][0] / MOTION_SCALE, cen[k][1] / MOTION_SCALE,
                               0.0, size, size, "motion", peak))
        found.sort(key=lambda m: -m.peak)
        kept = []
        for m in found:  # of two close blobs (bead + its faint tail) keep the stronger
            if all(math.hypot(m.cx - k.cx, m.cy - k.cy) >= MOTION_MIN_SEP for k in kept):
                kept.append(m)
                if len(kept) >= MAX_MOTION_BLOBS:
                    break
        return kept


# =============================================================================
# TRACKING: POTENTIAL (orange) -> CONFIRMED (red)
# =============================================================================
def track_gate(tr, f):
    """How far (px) an observation may be from the track's last position to join it."""
    dt = f - tr["obs"][-1][0]
    if len(tr["obs"]) < 2:
        return MAX_LINK_DIST * dt
    (f0, m0), (f1, m1) = tr["obs"][-2], tr["obs"][-1]
    speed = math.hypot(m1.cx - m0.cx, m1.cy - m0.cy) / (f1 - f0)
    return min(MAX_LINK_DIST, LINK_BASE_GATE_PX + LINK_SPEED_FACTOR * speed) * dt


def window_is_bead(win, n_frames):
    """win: the track's observations [(frame, Match)] over the last n_frames frames,
    judged together as one piece of evidence."""
    if len(win) < MIN_SEEN_FRACTION * n_frames or len(win) < 3:
        return False
    pts = [(o[1].cx, o[1].cy) for o in win]
    path = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    net = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    extent = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in pts for b in pts)
    if extent < MIN_EXTENT_PX or path <= 0 or net / path < MIN_EFFICIENCY:
        return False
    steps = [math.hypot(b[1].cx - a[1].cx, b[1].cy - a[1].cy) / (b[0] - a[0])
             for a, b in zip(win, win[1:])]
    return sum(s >= MIN_STEP_PX for s in steps) >= MIN_STEP_FRACTION * len(steps)


def nearest_track(tracks, used, m, f, limit):
    best, best_d = None, None
    for ti, tr in enumerate(tracks):
        if ti in used:
            continue
        lm = tr["obs"][-1][1]
        d = math.hypot(m.cx - lm.cx, m.cy - lm.cy)
        if d <= min(track_gate(tr, f), limit) and (best_d is None or d < best_d):
            best, best_d = ti, d
    return best


def find_beads(tmpl_cands, motion_cands, min_score, fps):
    """Returns out[f] = list of (Match, confirmed) to draw in frame f for this min score.

    confirmed False -> orange potential bead, True -> red bead.
    """
    n_conf = max(2, int(round(CONFIRM_SECONDS * fps)))
    max_gap = max(1, int(round(MAX_GAP_SECONDS * fps)))
    anchor_frames = max(2, int(round(STATIC_SECONDS * fps)))
    anchor_score = max(min_score, ANCHOR_MIN_SCORE)
    out = [[] for _ in tmpl_cands]

    if not MOTION_FILTER:
        for f, cands in enumerate(tmpl_cands):
            out[f] = [(m, True) for m in cands if m.score >= min_score]
        return out

    tracks = []   # {"obs": [(frame, Match)], "start": f, "last": f, "bead": bool}
    statics = []  # {"x", "y", "n", "last"}: template matches that do not move
    for f in range(len(tmpl_cands)):
        tracks = [t for t in tracks if f - t["last"] <= max_gap]
        statics = [s for s in statics if f - s["last"] <= 2]
        used = set()

        def add(ti, m):
            tr = tracks[ti]
            tr["obs"].append((f, m))
            tr["last"] = f
            if len(tr["obs"]) > 4 * n_conf:
                del tr["obs"][:-2 * n_conf]
            used.add(ti)

        # 1) moving blobs continue existing tracks, the rest may start new ones
        leftover = []
        for m in motion_cands[f]:
            ti = nearest_track(tracks, used, m, f, MAX_LINK_DIST)
            if ti is None:
                leftover.append(m)
            else:
                add(ti, m)

        # 2) template matches keep a track alive when the bead has stopped moving
        free = []
        for m in tmpl_cands[f]:
            if m.score < min_score:
                continue
            ti = nearest_track(tracks, used, m, f, TEMPLATE_JOIN_PX)
            if ti is None:
                free.append(m)
            else:
                add(ti, m)

        # 3) template matches that stay put frame after frame = static objects
        touched = set()
        for m in free:
            if m.score < anchor_score:
                continue
            best, best_d = None, None
            for si, s in enumerate(statics):
                if si in touched:
                    continue
                d = math.hypot(m.cx - s["x"], m.cy - s["y"])
                if d <= STATIC_RADIUS_PX and (best_d is None or d < best_d):
                    best, best_d = si, d
            if best is None:
                statics.append({"x": m.cx, "y": m.cy, "n": 1, "last": f})
                touched.add(len(statics) - 1)
            else:
                s = statics[best]
                s["x"], s["y"], s["n"], s["last"] = m.cx, m.cy, s["n"] + 1, f
                touched.add(best)
        anchors = [s for s in statics if s["n"] >= anchor_frames]

        # 4) new potential beads (orange from this very frame)
        for m in leftover:
            if any(math.hypot(m.cx - a["x"], m.cy - a["y"]) < ANCHOR_RADIUS_PX for a in anchors):
                continue
            tracks.append({"obs": [(f, m)], "start": f, "last": f, "bead": False})
            used.add(len(tracks) - 1)

        # 5) judge tracks that are old enough; emit what is drawn in this frame
        for tr in tracks:
            if not tr["bead"] and tr["last"] == f and f - tr["start"] >= n_conf - 1:
                win = [o for o in tr["obs"] if o[0] > f - n_conf]
                if window_is_bead(win, n_conf):
                    tr["bead"] = True
            hold = f - tr["last"]
            if tr["bead"] or hold <= POTENTIAL_HOLD_FRAMES:
                out[f].append((tr["obs"][-1][1], tr["bead"]))
    return out


# =============================================================================
# BEAD SELECTION AND DRAWING
# =============================================================================
def choose_beads(pool, k):
    """Confirmed beads of a frame. If more than k, keep the k farthest apart."""
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


def draw_circle(img, m, color):
    r = max(m.w, m.h) / 2.0 + CIRCLE_PADDING
    f = 1 << SHIFT
    cv2.circle(img, (int(round(m.cx * f)), int(round(m.cy * f))), int(round(r * f)),
               color, CIRCLE_THICKNESS, cv2.LINE_AA, SHIFT)


def frame_overlay(entries):
    """(reds, oranges) to draw for one frame from find_beads' entries."""
    reds = choose_beads([m for m, ok in entries if ok], NUM_MATCHES)
    oranges = [m for m, ok in entries if not ok]
    return reds, oranges


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
    """PASS 1: template candidates and motion blobs for every frame.
    Returns (tmpl_per_frame, motion_per_frame, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    tmpl_pf, motion_pf = [], []
    motion = MotionDetector()
    t0 = time.time()
    try:
        while max_frames <= 0 or len(tmpl_pf) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_pf.append(motion.update(gray))
            tmpl_pf.append(detect_templates(gray, variants, floor, subpixel, min_sep))
            n = len(tmpl_pf)
            if n % 25 == 0:
                tot = f"/{total}" if total else ""
                print(f"         detect frame {n}{tot}  ({n / (time.time() - t0):.1f} fps)",
                      end="\r", flush=True)
    finally:
        cap.release()
    return tmpl_pf, motion_pf, fps


def process_video(video_path, algo, adv, variants, out_dir, min_sep, scores, max_frames=0):
    """One detection pass + one tracking pass; writes every missing min-score output."""
    stem = video_path.stem
    todo = {}
    for s in scores:
        final = out_dir / output_name(stem, s, algo, adv)
        if not final.exists():
            todo[s] = final
    if not todo:
        print(f"  [skip] {stem} / {algo} / {adv}: all outputs already exist")
        return

    floor = min(MIN_SCORES) / 1000.0
    subpixel = (adv == "LDContrastSubpixel")
    t0 = time.time()
    print(f"  [run ] {stem} / {algo} / {adv}: {len(todo)} of {len(scores)} outputs to make")

    tmpl_pf, motion_pf, fps = detect_all(video_path, variants, floor, subpixel, min_sep, max_frames)
    if tmpl_pf is None:
        print(f"  [warn] could not open {video_path.name}, skipping")
        return
    if not tmpl_pf:
        print("  [warn] no frames could be read, skipping")
        return

    entries_by_score = {}
    for s in todo:
        ent = find_beads(tmpl_pf, motion_pf, s, fps)
        entries_by_score[s] = ent
        first_o = next((f for f, e in enumerate(ent) if any(not ok for _, ok in e)), None)
        first_r = next((f for f, e in enumerate(ent) if any(ok for _, ok in e)), None)
        fmt = lambda f: "-" if f is None else f"frame {f} ({f / fps:.2f}s)"
        print(f"         min score {s}: first orange {fmt(first_o)}, first red {fmt(first_r)}")

    cap = cv2.VideoCapture(str(video_path))
    writers = {}
    ok_run = False
    frames = 0
    t_write = time.time()
    print(f"         writing {len(todo)} output videos...")
    try:
        while frames < len(tmpl_pf):
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
                reds, oranges = frame_overlay(entries_by_score[s][frames])
                key = (tuple(sorted((round(m.cx, 1), round(m.cy, 1), m.w, m.h) for m in reds)),
                       tuple(sorted((round(m.cx, 1), round(m.cy, 1), m.w, m.h) for m in oranges)))
                img = drawn.get(key)
                if img is None:
                    if reds or oranges:
                        img = frame.copy()
                        for m in oranges:
                            draw_circle(img, m, ORANGE)
                        for m in reds:
                            draw_circle(img, m, RED)
                    else:
                        img = frame
                    drawn[key] = img
                wr.write(img)
            frames += 1
            if frames % 25 == 0:
                print(f"         writing frame {frames}/{len(tmpl_pf)}  "
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
    ap.add_argument("--scores", type=int, nargs="+", default=None,
                    help="Debug: only these min scores instead of the full sweep")
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

    scores = args.scores or MIN_SCORES
    t_start = time.time()
    try:
        for vp in videos:
            print(f"{vp.name}")
            for algo in ALGORITHMS:
                for adv in ADVANCED_OPTIONS:
                    try:
                        process_video(vp, algo, adv, variants, out_dir, min_sep, scores,
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