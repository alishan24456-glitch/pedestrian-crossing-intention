#!/usr/bin/env python3
"""
Pedestrian Crossing-Intention Prediction (ADAS)
===============================================

From a dashcam / elevated street view, YOLOv8n person detection +
centroid tracking watches pedestrians near the kerb and the zebra
crossing. Each pedestrian's position, heading and speed toward the
crossing feed a CROSSING / WAITING / WALKING AWAY intent state, and
a pedestrian judged about to step onto the road raises a
CROSSING AHEAD warning with a timestamped event log.

* **Crossing zones** -- the zebra crossings are documented IMAGE
  REGIONS for this scene (fractions of the 1280x720 frame, measured
  on the demo clip), not a calibrated road model: a main diagonal
  zebra in the centre, a second zebra on the right pavement, and a
  third zebra on the far-left carriageway. The kerb approach zone
  is the ring within 70 px of any zebra edge. A foot point inside a
  zebra is ON CROSSING; inside the ring is AT KERB; anywhere else
  is AWAY.
* **Intent** -- per pedestrian, position, heading and speed come
  from a least-squares velocity fit over the last ~1.5 s of
  tracked foot points (the bottom centre of the person box), plus
  the distance from the foot point to the nearest zebra edge:
    - CROSSING: on the zebra, or at the kerb / within 150 px of
      the zebra heading toward the road faster than 15 px/s;
    - WAITING: at the kerb standing still, or moving without a
      clear toward/away heading (standing, or walking along the
      pavement);
    - WALKING AWAY: moving away from the crossing faster than
      15 px/s (heading dot product with the direction to the
      crossing below -0.4).
  A candidate state must hold for ~0.7 s before the confirmed
  state changes, so tracker jitter cannot flip the board, and a
  pedestrian is not judged at all until it has been tracked for
  1.0 s.
* **Warning** -- any pedestrian whose confirmed state is CROSSING
  raises the CROSSING AHEAD banner (amber/red) and a timestamped
  event log entry; the warning clears when no pedestrian is
  CROSSING any more. The per-pedestrian intent board shows track
  ID, state, speed (px/s) and distance to the crossing (px).
* **Detection** -- YOLOv8n via OpenCV DNN (NOT ultralytics at run
  time). The stock ONNX model takes a fixed 640x640 input, so on
  this clip -- where people are only ~80 px tall -- detection runs
  TILED: two overlapping 704 px tiles across the 1280-wide frame,
  merged with NMS. Whole-frame detection at 640 finds only 0-5
  people per frame on this clip; tiled finds 5-11. YOLOv8's ONNX
  output is [1, 84, N] = 4 box coords + 80 class scores, with NO
  objectness channel: confidence is the max class score and the
  class is its argmax (class 0 = person).

Weights: ``yolov8n.onnx`` next to this script (12.8 MB, ships in
this folder). If it is missing, the script tries to export it from
``yolov8n.pt`` with Ultralytics (downloading the .pt if needed) and
otherwise exits with instructions.

Simulation mode (``--simulate``) renders a street with the same
zebra geometry and four figures: one walks up to the kerb, waits,
then crosses; one crosses the right zebra directly; one walks
along the pavement away from the crossing; one walks in and waits
at the kerb. The simulation's person source finds the figures
with a colour mask + contours -- the zones, the velocity fit and
the intent logic are the exact same code as real mode, and the
detector never sees the simulation's ground truth. Simulation is
an internal logic check; the delivered demo is real footage
(see README).

Honest framing: intent is a heuristic from motion and position,
NOT a mind-read. A pedestrian standing at the kerb can still step
out without warning, and a pedestrian judged WALKING AWAY can
turn back. The warning is a driver aid, never a guarantee.

Examples
--------
    python pedestrian_crossing_intention.py                 (file picker)
    python pedestrian_crossing_intention.py --source pick
    python pedestrian_crossing_intention.py --source street.mp4
    python pedestrian_crossing_intention.py --source 0      (webcam)
    python pedestrian_crossing_intention.py --simulate --max-seconds 30 --no-display

Press Q to quit. The video is written to ``output.mp4``, the
timestamped event log to ``output_events.log`` and the final
counters to ``output_summary.txt`` next to it.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout (the output video is always 1280x720)
# ---------------------------------------------------------------------------
W, H = 1280, 720
TITLE_H = 46
PANEL_H = 150                     # stats panel at the bottom
VIDEO_W, VIDEO_H = W, H - TITLE_H - PANEL_H
BG = (24, 26, 32)
PANEL_BG = (32, 35, 46)
WHITE = (255, 255, 255)
GREY = (160, 166, 178)
GREEN = (90, 200, 90)
AMBER = (60, 170, 255)
RED = (60, 60, 235)
YELLOW = (0, 210, 235)

# ---------------------------------------------------------------------------
# Detection + intent tuning
# ---------------------------------------------------------------------------
DET_W, DET_H = 1280, 720          # detection runs on this frame size
DETECT_EVERY = 3                  # run YOLO every 3rd frame (10 fps)
CONF = 0.30
NMS_THRESH = 0.30
PERSON_CLASS = 0                  # COCO person
MATCH_DIST = 60                   # px, centroid match radius (floor;
                                  # the gate is scale-aware, see update())
MAX_MISSED = 30                   # detection rounds (3 s) before a track
                                  # drops -- brief detection gaps on
                                  # small/blurred people must not kill
                                  # a track (raised from 15 after the
                                  # first demo run: 248 tracks for ~10
                                  # real people)
TILE_W = 704                      # tiled detection: two overlapping
TILE_X0 = (0, DET_W - TILE_W)     # 704 px tiles across the frame
VEL_WINDOW_S = 1.5                # velocity fit window
MIN_AGE_S = 1.0                   # track this long before any verdict
CONFIRM_S = 0.7                   # a candidate state must hold this
                                  # long before the confirmed state
                                  # changes (kills tracker jitter)
SPEED_GATE = 15.0                 # px/s: below this a pedestrian is
                                  # standing still
DOT_TOWARD = 0.4                  # heading dot dir-to-crossing above
                                  # this = moving toward the road
DOT_AWAY = -0.4                   # below this = moving away
APPROACH_PX = 150.0               # within this far, heading toward the
                                  # crossing also counts as CROSSING
KERB_MARGIN_PX = 70.0             # the kerb ring around each zebra
PT_URL = ("https://github.com/ultralytics/assets/releases/"
          "download/v8.2.0/yolov8n.pt")

# Zebra crossings, as fractions of the 1280x720 frame, measured on
# the demo clip (Pexels 37252001, elevated view over a city
# intersection). These are IMAGE REGIONS for this scene, not a
# calibrated road model.
#   main: the big diagonal zebra in the centre of the frame
#   (stripes run from upper-left to lower-right).
#   right: the zebra on the right pavement, vertical stripes.
#   far-left: the zebra on the far carriageway, top-left corner.
CROSSING_POLYS = [
    [(0.40, 0.38), (0.72, 0.42), (0.64, 0.68), (0.33, 0.66)],
    [(0.80, 0.33), (0.95, 0.33), (0.95, 0.63), (0.80, 0.63)],
    [(0.00, 0.16), (0.28, 0.20), (0.20, 0.42), (0.00, 0.38)],
]

# Pixel-space polygons, built once at import from the fractions.
CROSSING_PX = [[(x * DET_W, y * DET_H) for x, y in poly]
               for poly in CROSSING_POLYS]

STATE_COLOURS = {"CROSSING": RED, "WAITING": AMBER,
                 "WALKING AWAY": GREEN, "MEASURING": GREY}


# ---------------------------------------------------------------------------
# Geometry helpers -- point in polygon, distance to a polygon
# ---------------------------------------------------------------------------
def _point_in_poly(px, py, poly):
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            xin = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < xin:
                inside = not inside
    return inside


def _point_seg_dist(px, py, ax, ay, bx, by):
    """Distance from a point to a line segment."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _nearest_on_poly(px, py, poly):
    """(distance, nearest point) from (px, py) to a polygon's edges."""
    best_d, best_pt = float("inf"), None
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        d = _point_seg_dist(px, py, ax, ay, bx, by)
        if d < best_d:
            dx, dy = bx - ax, by - ay
            seg = dx * dx + dy * dy
            t = 0.0 if seg == 0 else max(
                0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg))
            best_d, best_pt = d, (ax + t * dx, ay + t * dy)
    return best_d, best_pt


def crossing_info(fx, fy):
    """Where a foot point sits relative to the zebras.

    Returns (zone, distance_px, nearest point) where zone is
    'ON CROSSING' (inside a zebra), 'AT KERB' (within the kerb
    ring around a zebra) or 'AWAY', and distance_px is 0 on the
    zebra, else the distance to the nearest zebra edge.
    """
    best_d, best_pt = float("inf"), None
    for poly in CROSSING_PX:
        if _point_in_poly(fx, fy, poly):
            return "ON CROSSING", 0.0, (fx, fy)
        d, pt = _nearest_on_poly(fx, fy, poly)
        if d < best_d:
            best_d, best_pt = d, pt
    if best_d <= KERB_MARGIN_PX:
        return "AT KERB", best_d, best_pt
    return "AWAY", best_d, best_pt


# ---------------------------------------------------------------------------
# Person sources -- real mode (YOLOv8 via OpenCV DNN, tiled) and
# simulation (colour mask). Both return boxes in detection-frame
# coordinates.
# ---------------------------------------------------------------------------
def ensure_weights(path):
    """Return a usable yolov8n.onnx path, exporting it if needed."""
    if os.path.exists(path):
        return path
    print(f"Weights not found at {path}; trying Ultralytics export ...")
    try:
        from ultralytics import YOLO
        pt = os.path.join(os.path.dirname(path) or ".", "yolov8n.pt")
        if not os.path.exists(pt):
            print("Downloading yolov8n.pt ...")
            urllib.request.urlretrieve(PT_URL, pt)
        print("Exporting yolov8n.pt -> yolov8n.onnx (one-time) ...")
        YOLO(pt).export(format="onnx", imgsz=640)
        exported = os.path.join(os.path.dirname(pt) or ".", "yolov8n.onnx")
        if os.path.exists(exported) and exported != path:
            os.replace(exported, path)
        return path
    except Exception as exc:  # noqa: BLE001 - report and stop cleanly
        print(f"Could not obtain weights: {exc}")
        print("Download yolov8n.onnx and place it next to this script, or")
        print("install ultralytics (`pip install ultralytics`) and re-run.")
        sys.exit(1)


class YoloPersonDetector:
    """YOLOv8n person detection through OpenCV DNN, tiled 2x.

    The stock ONNX model takes a fixed 640x640 input, so a whole
    frame at 640 finds only 0-5 people on this clip (people are
    ~80 px tall). Two overlapping 704 px tiles find 5-11 per
    frame (measured 1 frame/s over the demo clip). Boxes from the
    two tiles are merged with NMS.
    """

    def __init__(self, weights_path):
        self.net = cv2.dnn.readNetFromONNX(ensure_weights(weights_path))

    def _detect_tile(self, tile):
        blob = cv2.dnn.blobFromImage(tile, 1 / 255.0, (640, 640),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        # YOLOv8 ONNX output is [1, 84, N]: 4 box coords + 80 class
        # scores, NO objectness channel -- confidence is the max
        # class score, class is its argmax. (A v7-style objectness
        # read makes the detector dead; do not repeat that bug.)
        out = self.net.forward()[0].T  # [8400, 84]
        scores = out[:, 4:].max(axis=1)
        classes = out[:, 4:].argmax(axis=1)
        keep = (scores > CONF) & (classes == PERSON_CLASS)
        th, tw = tile.shape[:2]
        boxes = []
        for (cx, cy, bw, bh), s in zip(out[keep, :4], scores[keep]):
            boxes.append([float((cx - bw / 2) * tw / 640),
                          float((cy - bh / 2) * th / 640),
                          float(bw * tw / 640), float(bh * th / 640),
                          float(s)])
        return boxes

    def detect(self, frame):
        boxes = []
        for x0 in TILE_X0:
            tile = frame[:, x0:x0 + TILE_W]
            for b in self._detect_tile(tile):
                boxes.append([b[0] + x0, b[1], b[2], b[3], b[4]])
        if not boxes:
            return []
        idx = cv2.dnn.NMSBoxes([b[:4] for b in boxes],
                               [b[4] for b in boxes], CONF, NMS_THRESH)
        return [boxes[int(i)] for i in idx.flatten()] if len(idx) else []


def sim_person_boxes(frame):
    """Colour-blob person source for the rendered street."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([95, 120, 80]),
                       np.array([115, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 400:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([float(x), float(y), float(w), float(h), 1.0])
    return boxes


# ---------------------------------------------------------------------------
# Intent tracker -- centroid tracking + per-pedestrian intent state
# ---------------------------------------------------------------------------
def _fit_velocity(history):
    """Least-squares (vx, vy) in px/s over [(t, x, y), ...]; None if thin."""
    if len(history) < 5:
        return None
    t0 = history[0][0]
    ts = [t - t0 for t, _, _ in history]
    if ts[-1] - ts[0] < 0.5:
        return None
    n = len(ts)
    mt = sum(ts) / n
    mx = sum(x for _, x, _ in history) / n
    my = sum(y for _, _, y in history) / n
    den = sum((tt - mt) ** 2 for tt in ts)
    if den <= 0:
        return None
    vx = sum((tt - mt) * (xx - mx)
             for tt, xx in zip(ts, [x for _, x, _ in history])) / den
    vy = sum((tt - mt) * (yy - my)
             for tt, yy in zip(ts, [y for _, _, y in history])) / den
    return (vx, vy)


class IntentTracker:
    """Tracks pedestrians and judges each one's crossing intent."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.frames = 0
        self.video_frames = 0
        self.events = []
        self.detections = 0
        self.pedestrians_seen = 0
        self._last_t = None
        # Per-state distinct track segments: a track counts in every
        # state it is ever confirmed in (a crosser who waited first
        # counts in both WAITING and CROSSING).
        self.state_segments = {"WAITING": set(), "CROSSING": set(),
                               "WALKING AWAY": set()}
        self.state_changes = 0
        self.warning_active = False
        self.warning_activations = 0
        self.warning_seconds = 0.0
        self.first_warning_t = None
        self._warning_since = None

    def _log(self, t, text):
        self.events.append((t, text))

    @staticmethod
    def _centre(box):
        x, y, w, h, _ = box
        return (x + w / 2, y + h / 2)

    @staticmethod
    def _foot(box):
        x, y, w, h, _ = box
        return (x + w / 2, y + h)

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah, _ = a
        bx, by, bw, bh, _ = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def reset(self, t):
        """Drop all tracks: the source looped, so the scene restarted."""
        n = len(self.tracks)
        if self.warning_active:
            self._log(t, "CROSSING AHEAD warning cleared "
                         "(source looped)")
            self.warning_active = False
            self._warning_since = None
        self.tracks.clear()
        self._log(t, f"source looped -- {n} in-flight tracks reset "
                     f"(intent states restart from zero)")

    def _candidate_state(self, tr, zone, dist, nearest, t):
        """Intent from position, heading and speed (unconfirmed)."""
        v = _fit_velocity(tr["history"])
        if v is None or t - tr["born_t"] < MIN_AGE_S:
            # Hold a confirmed state through a thin history: a
            # detection gap thins the 1.5 s velocity window, so the
            # fit goes None and the track would otherwise regress
            # to MEASURING -- on the second demo run a WAITING
            # pedestrian flipped back to MEASURING at 15.5 s and
            # 24.4 s for exactly this reason. A young track (never
            # confirmed anything) stays MEASURING, which is what
            # tr["state"] already is.
            return tr["state"], 0.0, dist
        vx, vy = v
        speed = math.hypot(vx, vy)
        tr["vx"], tr["vy"], tr["speed"] = vx, vy, speed
        if zone == "ON CROSSING":
            return "CROSSING", speed, dist
        if speed < SPEED_GATE:
            # Standing still: at the kerb that is WAITING; further
            # away it is also WAITING -- a still pedestrian is not
            # crossing.
            return "WAITING", speed, dist
        # Heading is judged PER ZEBRA, not against the nearest edge
        # point alone: a pedestrian can be nearest to one zebra
        # (the right one) while walking toward another (the main
        # one), and a nearest-edge reference then reads WALKING
        # AWAY for someone clearly approaching the main crossing
        # -- found on the first simulation run, where the figure
        # walking up to the main zebra flipped to WALKING AWAY at
        # 3.3 s because the right zebra was nearer. So: CROSSING
        # if the pedestrian is near ANY zebra and heading toward
        # THAT zebra; WALKING AWAY if it is heading away from the
        # NEAREST zebra.
        best_toward_dot = -2.0
        nearest_dot = 0.0
        nearest_dist = float("inf")
        fx, fy = tr["fx"], tr["fy"]
        for poly in CROSSING_PX:
            d_i, npt = _nearest_on_poly(fx, fy, poly)
            dx, dy = npt[0] - fx, npt[1] - fy
            dlen = math.hypot(dx, dy)
            if dlen < 1.0:
                dot_i = 0.0
            else:
                dot_i = (vx * dx + vy * dy) / (speed * dlen)
            if d_i < nearest_dist:
                nearest_dist, nearest_dot = d_i, dot_i
            if d_i <= APPROACH_PX and dot_i > best_toward_dot:
                best_toward_dot = dot_i
        tr["dot"] = nearest_dot
        if best_toward_dot >= DOT_TOWARD:
            return "CROSSING", speed, dist
        if nearest_dot <= DOT_AWAY:
            return "WALKING AWAY", speed, dist
        return "WAITING", speed, dist

    def update(self, boxes, t):
        """One detection round. Returns the live state dict."""
        self.frames += 1
        self.detections += len(boxes)
        used = set()
        for box in boxes:
            cx, cy = self._centre(box)
            fx, fy = self._foot(box)
            best, best_d = None, float("inf")
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - tr["cx"], cy - tr["cy"])
                # Scale-aware match gate: a person box is only
                # ~30x80 px, and YOLO's box jitters by ~25 px on
                # these small blurred figures -- a fixed gate plus
                # a strict IoU floor spawned a new track for the
                # same person every ~1.2 s on the first demo run
                # (248 tracks for ~10 real people). The gate is
                # half the bigger box's size, so a jittery small
                # box may jump further than a steady one, while
                # two people walking side by side (centroids
                # ~40+ px apart with overlapping boxes) still
                # cannot steal each other's track.
                gate = max(MATCH_DIST,
                           0.5 * max(box[2], box[3],
                                     tr["box"][2], tr["box"][3]))
                if d >= gate:
                    continue
                if d >= best_d:
                    continue
                # A detection must also overlap the track's last box.
                # The threshold is low (0.02) for the same reason:
                # a jittery small box overlaps its predecessor by
                # only ~0.02, while two different people's boxes
                # essentially never overlap at all.
                if self._iou(box, tr["box"]) <= 0.02:
                    continue
                best, best_d = tid, d
            if best is None:
                tid = self.next_id
                self.next_id += 1
                self.pedestrians_seen += 1
                self.tracks[tid] = {"id": tid, "cx": cx, "cy": cy,
                                    "fx": fx, "fy": fy, "box": box,
                                    "missed": 0, "born_t": t,
                                    "history": [], "vx": 0.0,
                                    "vy": 0.0, "speed": 0.0,
                                    "dot": 0.0, "zone": "AWAY",
                                    "dist": 0.0,
                                    "state": "MEASURING",
                                    "pending": None,
                                    "pending_since": None}
                best = tid
            used.add(best)
            tr = self.tracks[best]
            tr.update({"cx": cx, "cy": cy, "fx": fx, "fy": fy,
                       "box": box, "missed": 0})
            tr["history"].append((t, fx, fy))
            tr["history"] = [(tt, xx, yy) for tt, xx, yy in tr["history"]
                             if t - tt <= VEL_WINDOW_S]
            zone, dist, nearest = crossing_info(fx, fy)
            tr["zone"], tr["dist"] = zone, dist
            cand, speed, _ = self._candidate_state(tr, zone, dist,
                                                   nearest, t)
            # Confirmation: a candidate must hold ~0.7 s before the
            # confirmed state changes, so tracker jitter cannot
            # flip the board.
            if cand == tr["state"]:
                tr["pending"] = None
                tr["pending_since"] = None
            elif cand == tr["pending"]:
                if (tr["pending_since"] is not None
                        and t - tr["pending_since"] >= CONFIRM_S):
                    prev = tr["state"]
                    tr["state"] = cand
                    tr["pending"] = None
                    tr["pending_since"] = None
                    self.state_changes += 1
                    if cand in self.state_segments:
                        self.state_segments[cand].add(best)
                    self._log(t, f"pedestrian #{best} intent "
                                 f"{prev} -> {cand} "
                                 f"(speed {speed:.0f} px/s, "
                                 f"{dist:.0f} px from crossing, "
                                 f"{zone.lower()})")
            else:
                tr["pending"] = cand
                tr["pending_since"] = t
        for tid in list(self.tracks):
            if tid in used:
                continue
            tr = self.tracks[tid]
            tr["missed"] += 1
            if tr["missed"] > MAX_MISSED:
                del self.tracks[tid]
        # Warning: active while any pedestrian is confirmed CROSSING.
        any_crossing = any(tr["state"] == "CROSSING"
                           for tr in self.tracks.values())
        if any_crossing and not self.warning_active:
            ids = [tid for tid, tr in self.tracks.items()
                   if tr["state"] == "CROSSING"]
            self.warning_active = True
            self._warning_since = t
            self.warning_activations += 1
            if self.first_warning_t is None:
                self.first_warning_t = t
            self._log(t, f"CROSSING AHEAD warning -- pedestrian "
                         f"#{', #'.join(map(str, ids))} judged "
                         f"about to cross")
        elif not any_crossing and self.warning_active:
            self.warning_active = False
            self._warning_since = None
            self._log(t, "CROSSING AHEAD warning cleared -- no "
                         "pedestrian crossing")
        live = {"CROSSING": 0, "WAITING": 0, "WALKING AWAY": 0,
                "MEASURING": 0}
        for tr in self.tracks.values():
            if tr["missed"] <= 2:
                live[tr["state"]] += 1
        return {"warning": self.warning_active, "live": live}

    def summary(self):
        lines = [
            "PEDESTRIAN CROSSING-INTENTION PREDICTION -- RUN SUMMARY",
            f"frames processed: {self.video_frames}",
            f"tracker updates: {self.frames}",
            f"person detections (summed): {self.detections}",
            f"pedestrians tracked: {self.pedestrians_seen}",
            "pedestrian segments ever confirmed in each state "
            "(one pedestrian can count in several states):",
            f"  WAITING: {len(self.state_segments['WAITING'])}",
            f"  CROSSING: {len(self.state_segments['CROSSING'])}",
            f"  WALKING AWAY: {len(self.state_segments['WALKING AWAY'])}",
            f"intent state changes logged: {self.state_changes}",
            f"CROSSING AHEAD warnings raised: "
            f"{self.warning_activations}",
            ("first warning at: "
             + (f"{self.first_warning_t:.1f} s"
                if self.first_warning_t is not None else "never")),
            f"total time with a warning active: "
            f"{self.warning_seconds:.1f} s",
            f"events logged: {len(self.events)}",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Simulation renderer -- a street with the same zebra geometry; four
# figures approach, wait, cross and walk away. The tracker only ever
# sees these pixels.
# ---------------------------------------------------------------------------
class StreetSim:
    """Renders a street scene with pedestrians at the zebra.

    The zebra polygons are drawn at exactly the fractions the
    pipeline judges against, so the sim exercises the same zones
    as the real clip. Figures are drawn in the crowd-density blue
    (255,130,30) so the colour-mask person source finds them.
    """

    BLUE = (255, 130, 30)         # figure colour (inside the mask)
    DARK = (200, 100, 20)         # darker blue, still inside the mask
    SKIN = (170, 190, 220)        # head (light, outside the mask)

    def __init__(self, w, h, fps=30):
        self.w, self.h = w, h
        self.fps = fps
        self.frame_idx = 0
        # Waypoints: (t, x, y) in detection-frame pixels. A figure
        # walks between waypoints at constant speed and holds the
        # last one. Figures are staged far apart so their blue
        # blobs never merge into one contour.
        self.figures = [
            # A: walks up to the kerb, waits, crosses the main zebra,
            # then walks away to the left.
            {"path": [(0.0, 1150, 500), (5.0, 900, 420),
                      (10.0, 900, 420), (17.0, 500, 430),
                      (22.0, 300, 480), (30.0, 200, 480)]},
            # B: crosses the right zebra directly, then walks away.
            {"path": [(0.0, 1240, 350), (4.0, 1230, 350),
                      (10.0, 1000, 350), (14.0, 980, 550),
                      (30.0, 980, 620)]},
            # C: starts at the kerb below the main zebra and walks
            # down-left, away from the crossing, the whole time
            # (heading points away from the nearest zebra).
            {"path": [(0.0, 700, 520), (12.0, 500, 700),
                      (30.0, 500, 700)]},
            # D: walks in from the left and waits at the kerb.
            {"path": [(0.0, 60, 320), (14.0, 100, 320),
                      (16.0, 470, 320), (30.0, 470, 320)]},
        ]

    def _pos(self, fig, t):
        path = fig["path"]
        if t <= path[0][0]:
            return path[0][1], path[0][2]
        for i in range(len(path) - 1):
            t0, x0, y0 = path[i]
            t1, x1, y1 = path[i + 1]
            if t0 <= t <= t1:
                f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return x0 + f * (x1 - x0), y0 + f * (y1 - y0)
        return path[-1][1], path[-1][2]

    def _draw_street(self, img):
        # Sky / buildings (top), pavement, road (bottom two-thirds).
        cv2.rectangle(img, (0, 0), (self.w, int(self.h * 0.22)),
                      (120, 140, 165), -1)
        cv2.rectangle(img, (0, int(self.h * 0.22)),
                      (self.w, int(self.h * 0.38)),
                      (150, 150, 145), -1)
        cv2.rectangle(img, (0, int(self.h * 0.38)), (self.w, self.h),
                      (70, 72, 76), -1)
        # Kerb line.
        cv2.line(img, (0, int(self.h * 0.38)),
                 (self.w, int(self.h * 0.38)), (200, 200, 195), 4)
        # Zebra stripes: white bars interpolated across each
        # crossing polygon (same fractions as the pipeline zones).
        for poly in CROSSING_PX:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(img, [pts], (95, 95, 98))
            n_bars = 6
            for i in range(n_bars):
                f0 = (2 * i + 0.15) / (2 * n_bars)
                f1 = (2 * i + 0.85) / (2 * n_bars)
                top0 = (poly[0][0] + f0 * (poly[1][0] - poly[0][0]),
                        poly[0][1] + f0 * (poly[1][1] - poly[0][1]))
                top1 = (poly[0][0] + f1 * (poly[1][0] - poly[0][0]),
                        poly[0][1] + f1 * (poly[1][1] - poly[0][1]))
                bot0 = (poly[3][0] + f0 * (poly[2][0] - poly[3][0]),
                        poly[3][1] + f0 * (poly[2][1] - poly[3][1]))
                bot1 = (poly[3][0] + f1 * (poly[2][0] - poly[3][0]),
                        poly[3][1] + f1 * (poly[2][1] - poly[3][1]))
                bar = np.array([top0, top1, bot1, bot0], dtype=np.int32)
                cv2.fillPoly(img, [bar], (235, 235, 230))
        # A parked car, far from the crossing, for scale.
        cv2.rectangle(img, (80, 560), (280, 640), (40, 60, 120), -1)

    def _draw_figure(self, img, cx, fy):
        """A walking figure ~80 px tall, feet at (cx, fy)."""
        x = int(cx)
        y = int(fy)
        # Legs.
        cv2.rectangle(img, (x - 11, y - 28), (x - 2, y), self.DARK, -1)
        cv2.rectangle(img, (x + 2, y - 28), (x + 11, y), self.DARK, -1)
        # Torso.
        cv2.rectangle(img, (x - 13, y - 62), (x + 13, y - 26),
                      self.BLUE, -1)
        # Arms.
        cv2.rectangle(img, (x - 19, y - 58), (x - 13, y - 30),
                      self.DARK, -1)
        cv2.rectangle(img, (x + 13, y - 58), (x + 19, y - 30),
                      self.DARK, -1)
        # Head.
        cv2.circle(img, (x, y - 70), 9, self.SKIN, -1)

    def frame(self):
        idx = self.frame_idx
        self.frame_idx += 1
        t = idx / self.fps
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_street(img)
        for fig in self.figures:
            cx, fy = self._pos(fig, t)
            self._draw_figure(img, cx, fy)
        return img


# ---------------------------------------------------------------------------
# Rendering -- draws tracker output onto the 1280x720 canvas
# ---------------------------------------------------------------------------
def draw_canvas(det_frame, tracker, res, elapsed, mode_label):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = BG
    cv2.rectangle(canvas, (0, 0), (W, TITLE_H), PANEL_BG, -1)
    cv2.putText(canvas, "PEDESTRIAN CROSSING-INTENTION  --  ADAS",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    cv2.putText(canvas, mode_label, (W - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)

    vh, vw = det_frame.shape[:2]
    scale = min(VIDEO_W / vw, VIDEO_H / vh)
    nw, nh = int(vw * scale), int(vh * scale)
    resized = cv2.resize(det_frame, (nw, nh))
    ox = (VIDEO_W - nw) // 2
    oy = TITLE_H + (VIDEO_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    sx, sy = nw / vw, nh / vh

    def _mx(x):
        return ox + int(x * sx)

    def _my(y):
        return oy + int(y * sy)

    # Zebra polygons (green) + kerb ring (amber outline, expanded).
    for poly in CROSSING_PX:
        pts = np.array([(_mx(x), _my(y)) for x, y in poly],
                       dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], GREEN)
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
        cv2.polylines(canvas, [pts], True, GREEN, 2)
    # Kerb ring: a 70 px margin around the main zebra, drawn as an
    # expanded outline (approx: scale the polygon about its centre).
    for poly in CROSSING_PX:
        cx = sum(x for x, _ in poly) / len(poly)
        cy = sum(y for _, y in poly) / len(poly)
        exp = []
        for x, y in poly:
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy) or 1.0
            exp.append((_mx(x + dx / d * KERB_MARGIN_PX),
                        _my(y + dy / d * KERB_MARGIN_PX)))
        cv2.polylines(canvas, [np.array(exp, dtype=np.int32)], True,
                      AMBER, 1)
    cv2.putText(canvas, "ZEBRA", (_mx(CROSSING_PX[0][0][0]),
                                   _my(CROSSING_PX[0][0][1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

    # Pedestrian boxes + intent tags. Tags are de-collided: people
    # stand close together at the kerb, and one tag per box printed
    # on top of the next is unreadable (first demo run). Tags are
    # collected first, sorted by height, and any tag that would
    # overlap the one above it is pushed down below it.
    tag_items = []
    for tid, tr in tracker.tracks.items():
        if tr["missed"] > 2:
            continue
        x, y, w, h, _ = tr["box"]
        colour = STATE_COLOURS.get(tr["state"], GREY)
        thick = 3 if tr["state"] == "CROSSING" else 2
        cv2.rectangle(canvas, (_mx(x), _my(y)), (_mx(x + w), _my(y + h)),
                      colour, thick)
        # Foot point.
        fx, fy = tr["fx"], tr["fy"]
        cv2.circle(canvas, (_mx(fx), _my(fy)), 4, colour, -1)
        tag = (f"#{tid} {tr['state']}  {tr['speed']:.0f} px/s  "
               f"{tr['dist']:.0f} px")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.55, 2)
        lx = _mx(x)
        if lx + tw + 8 > W - 4:
            lx = max(4, W - 4 - tw - 8)
        ly = _my(y) - 8
        if ly - th < oy + 4:
            ly = _my(y + h) + th + 10
        tag_items.append((ly, lx, tw, th, tag, colour))
    tag_items.sort(key=lambda item: item[0])
    placed = []  # (top, bottom) of tags already drawn
    for ly, lx, tw, th, tag, colour in tag_items:
        top = ly - th - 4
        # Push down past any tag this one would overlap.
        for p_top, p_bot in placed:
            if top < p_bot + 2 and ly + 4 > p_top:
                shift = p_bot + 2 - top
                top += shift
                ly += shift
        if ly + 4 > H - 4:
            ly = H - 8
            top = ly - th - 4
        cv2.rectangle(canvas, (lx, top), (lx + tw + 8, ly + 4),
                      (24, 26, 32), -1)
        cv2.putText(canvas, tag, (lx + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        placed.append((top, ly + 4))

    # Bottom panel: board + warning banner.
    py = TITLE_H + VIDEO_H
    cv2.rectangle(canvas, (0, py), (W, H), PANEL_BG, -1)
    warning = res["warning"]
    acol = RED if warning else GREEN
    cv2.rectangle(canvas, (0, py), (W, py + 6), acol, -1)
    banner = ("CROSSING AHEAD  --  PEDESTRIAN ABOUT TO CROSS"
              if warning else "CLEAR  --  NO PEDESTRIAN CROSSING")
    cv2.putText(canvas, banner, (16, py + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, acol, 2)
    live = res["live"]
    cv2.putText(canvas, f"t = {elapsed:.1f} s", (16, py + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas,
                f"pedestrians: {tracker.pedestrians_seen}   "
                f"CROSSING: {live['CROSSING']}   "
                f"WAITING: {live['WAITING']}   "
                f"WALKING AWAY: {live['WALKING AWAY']}   "
                f"MEASURING: {live['MEASURING']}",
                (300, py + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas,
                f"warnings: {tracker.warning_activations}   "
                f"warning time: {tracker.warning_seconds:.1f} s   "
                f"state changes: {tracker.state_changes}   "
                f"detections: {tracker.detections}",
                (16, py + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)
    return canvas


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return ""


def open_source(source):
    if source == "pick":
        path = pick_file()
        if not path:
            print("No file chosen.")
            sys.exit(0)
        source = path
    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        sys.exit(1)
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", default="pick",
                    help="video file, webcam index, or 'pick'")
    ap.add_argument("--simulate", action="store_true",
                    help="rendered street view instead of a camera/video")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true",
                    help="loop a short source until --max-seconds is reached")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--weights", default=None,
                    help="path to yolov8n.onnx (default: next to the script)")
    args = ap.parse_args()

    fps = 30
    here = os.path.dirname(os.path.abspath(__file__))
    detector = None if args.simulate else YoloPersonDetector(
        args.weights or os.path.join(here, "yolov8n.onnx"))
    tracker = IntentTracker()
    sim = StreetSim(DET_W, DET_H, fps) if args.simulate else None
    cap = None if args.simulate else open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else fps
    if not src_fps or src_fps <= 0:
        src_fps = fps
    step = max(1, round(src_fps / fps))  # sample the source at ~30 fps

    out_path = os.path.abspath(args.out)
    base, _ = os.path.splitext(out_path)
    events_path = base + "_events.log"
    summary_path = base + "_summary.txt"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    mode_label = "SIMULATION (logic check)" if args.simulate else "REAL MODE"

    n_out = 0
    max_frames = int(args.max_seconds * fps)
    src_idx = 0
    boxes = []
    res = {"warning": False,
           "live": {"CROSSING": 0, "WAITING": 0, "WALKING AWAY": 0,
                    "MEASURING": 0}}
    while n_out < max_frames:
        t = n_out / fps
        if args.simulate:
            det_frame = sim.frame()
            boxes = sim_person_boxes(det_frame)
            res = tracker.update(boxes, t)
        else:
            ok, frame = cap.read()
            src_idx += 1
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    src_idx = 1
                    tracker.reset(t)
                if not ok:
                    break
            if (src_idx - 1) % step != 0:
                continue
            det_frame = cv2.resize(frame, (DET_W, DET_H))
            if n_out % DETECT_EVERY == 0:
                boxes = detector.detect(det_frame)
                res = tracker.update(boxes, t)
            else:
                # Recompute only the warning/live view from the last
                # tracker state (no new detection round this frame).
                any_crossing = any(
                    tr["state"] == "CROSSING"
                    for tr in tracker.tracks.values())
                if tracker.warning_active and not any_crossing:
                    tracker.warning_active = False
                    tracker._warning_since = None
                    tracker._log(t, "CROSSING AHEAD warning cleared -- "
                                    "no pedestrian crossing")
                live = {"CROSSING": 0, "WAITING": 0, "WALKING AWAY": 0,
                        "MEASURING": 0}
                for tr in tracker.tracks.values():
                    if tr["missed"] <= 2:
                        live[tr["state"]] += 1
                res = {"warning": tracker.warning_active, "live": live}
        canvas = draw_canvas(det_frame, tracker, res, t, mode_label)
        writer.write(canvas)
        # Warning time is accumulated once per output frame, here --
        # not inside update(), which only runs on detection rounds
        # (every 3rd frame) in real mode and would under-count.
        if tracker.warning_active:
            tracker.warning_seconds += 1.0 / fps
        n_out += 1
        if not args.no_display:
            try:
                cv2.imshow("Pedestrian Crossing-Intention", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            except cv2.error:
                pass

    writer.release()
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    tracker.video_frames = n_out
    with open(events_path, "w") as f:
        for t, text in tracker.events:
            f.write(f"{t:6.1f}s  {text}\n")
    with open(summary_path, "w") as f:
        f.write(tracker.summary())
    print(tracker.summary())
    print(f"video:   {out_path}")
    print(f"events:  {events_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
