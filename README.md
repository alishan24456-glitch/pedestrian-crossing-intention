# Pedestrian Crossing-Intention Prediction (ADAS)

From a dashcam / elevated street view, YOLOv8n person detection +
centroid tracking watches pedestrians near the kerb and the zebra
crossing. Each pedestrian's position, heading and speed toward the
crossing feed a **CROSSING / WAITING / WALKING AWAY** intent state,
and a pedestrian judged about to step onto the road raises a
**CROSSING AHEAD** warning with a timestamped event log.

Pedestrian protection is a core ADAS function in every 2026 market
report, and crossing-intention prediction is active 2026 research
(the JAAD / PIE benchmarks). This is the prediction layer on top of
his blind-spot-detection (vehicles beside the car) and
crowd-density-heatmap (counting a crowd) — this one predicts what
one pedestrian is about to do.

## How intent is judged

- **Crossing zones** — the zebra crossings are documented **image
  regions for this scene** (fractions of the 1280×720 frame,
  measured on the demo clip), not a calibrated road model: a main
  diagonal zebra in the centre, a second zebra on the right
  pavement, and a third zebra on the far-left carriageway. The
  kerb approach zone is the ring within 70 px of any zebra edge.
  A foot point (bottom centre of the person box) inside a zebra is
  **ON CROSSING**; inside the ring is **AT KERB**; anywhere else
  is **AWAY**.
- **Motion** — per pedestrian, heading and speed come from a
  least-squares velocity fit over the last ~1.5 s of tracked foot
  points, plus the distance from the foot point to the nearest
  zebra edge.
- **States**
  | State | Meaning |
  |---|---|
  | `CROSSING` | On the zebra, or near a zebra (within 150 px) and heading toward *that* zebra faster than 15 px/s |
  | `WAITING` | At the kerb standing still, or moving without a clear toward/away heading (walking along the pavement) |
  | `WALKING AWAY` | Moving away from the nearest zebra faster than 15 px/s (heading dot product below −0.4) |
  | `MEASURING` | Tracked for less than 1.0 s — no verdict yet |
- **Confirmation** — a candidate state must hold for ~0.7 s
  before the confirmed state changes, so tracker jitter cannot
  flip the board.
- **Warning** — any pedestrian whose confirmed state is CROSSING
  raises the **CROSSING AHEAD** banner and a timestamped event log
  entry; it clears when no pedestrian is CROSSING any more. The
  per-pedestrian intent board shows track ID, state, speed (px/s)
  and distance to the crossing (px).
- **Detection** — YOLOv8n via OpenCV DNN (no torch/ultralytics at
  run time). The stock ONNX model takes a fixed 640×640 input, so
  on this clip — where people are only ~80 px tall — detection
  runs **tiled**: two overlapping 704 px tiles across the 1280-wide
  frame, merged with NMS. Whole-frame detection at 640 finds only
  0–5 people per frame on this clip; tiled finds 5–11.
  YOLOv8's ONNX output is [1, 84, N] = 4 box coords + 80 class
  scores, with **no objectness channel**: confidence is the max
  class score, class is its argmax (class 0 = person).
- **Tracking** — centroid tracking with a scale-aware match gate
  (half the bigger box's size) and a 0.02 IoU floor, holding a
  track through up to 3 s of missed detections.

## Honest framing — read this before trusting the board

**Intent is a heuristic from motion and position, NOT a
mind-read.** A pedestrian standing at the kerb can still step out
without warning, and a pedestrian judged WALKING AWAY can turn
back. The warning is a driver aid, never a guarantee.

- The zebra polygons are image regions for this scene, not a
  calibrated road model; on a different camera view the zones
  must be re-measured.
- Speeds and distances are in **pixels**, not metres — the camera
  is uncalibrated.
- The tracker fragments: counts are track **segments**, not
  distinct people (see the demo numbers below). A person whose
  track breaks and restarts re-enters as MEASURING.
- A pedestrian striding briskly up to the kerb reads CROSSING
  until they stop and the board settles to WAITING — the
  heuristic cannot tell "walking to the kerb to wait" from
  "walking to the kerb to cross" until the pedestrian stops
  (verified in simulation: figure D reads CROSSING for ~2 s
  while walking up to the kerb).
- People in the tilt-shift-blurred top and bottom bands of the
  clip are detected unreliably; the board is steadiest in the
  sharp middle band.

## Demo (real footage)

Real street footage (Pexels video 37252001, free license — title
and creator not recorded, the Pexels video page is blocked by a
Cloudflare check; the search listing reads
"cars-pedestrian-shift-speed"): an elevated fixed-camera view over
a city intersection with a zebra crossing, pedestrians waiting at
the kerb, crossing the zebra and walking along the pavements.
17.0 s clip at 59.94 fps, resampled to a 1280×720 @ 30 fps working
copy so the timers run at true speed; the demo loops it to 30 s
(`--loop`, loop point 17.0 s, logged in the event log).
Downloaded fresh for this build under the 14 Nov fresh-video rule
— never used by any earlier project. Three other candidates were
downloaded and frame-inspected first and rejected: 40260364
(black-and-white, legs only, no full person), 38917563 (two people
on a distant crossing, too small/far), 27376433 (corrupt download
— decoding errors, only 14 frames extracted), and 36133637
(vertical).

**Demo verified (30 s, looped, 900 frames, 300 tracker updates):**
1,891 person detections (summed), **193 pedestrians tracked**,
segments ever confirmed **WAITING 17, CROSSING 9, WALKING AWAY 3**,
30 intent state changes, **2 CROSSING AHEAD warnings, first at
6.0 s, warning active 17.9 s**, 34 events logged. H.264 LinkedIn
copy exported and codec-verified h264; frames inspected at 6 s
(warning banner active, pedestrian #26 CROSSING on the zebra at
13 px/s, #27 WAITING at the kerb), 12 s, 23 s and 28 s.

- **What the real demo shows:** pedestrians standing ON the zebra
  read CROSSING (e.g. #26 at 6.0 s, #60 at 9.7 s, #148 at 23.1 s);
  people waiting at the kerb read WAITING (e.g. #27, 61 px from the
  crossing, 0–1 px/s); the warning fires at 6.0 s and stays active
  while pedestrians are on or approaching the zebra, clearing only
  at the 17.0 s loop point, then firing again at 23.1 s on the
  second pass and holding to the end. Pedestrians approaching the
  zebra from the pavement read CROSSING too (e.g. #66 at 12.2 s,
  62 px/s, 125 px away; #166 at 29.1 s, 77 px/s, 73 px away).
- **Honest counts:** 193 tracks for ~10 real people per pass —
  the tracker still fragments (see the limits above), so the
  segment counts are track segments, not distinct people. Two
  speeds in the log are inflated by track jumps between nearby
  pedestrians (#68 at 137 px/s, #142 at 79 px/s); the speeds are
  px/s estimates from jittery small boxes, not surveyed
  measurements.

## Simulation (logic check only — not the demo)

`--simulate` renders a street with the same zebra geometry and
four figures: A walks up to the kerb, waits, then crosses the main
zebra and walks away; B crosses the right zebra directly, then
walks away; C starts at the kerb below the main zebra and walks
away down-left; D walks in from the left and waits at the kerb.
The person source finds the figures with a colour mask + contours
— the zones, the velocity fit and the intent logic are the exact
same code as real mode, and the detector never sees the
simulation's ground truth.

Verified on the final build (30 s, 900 frames): **4 pedestrians
tracked, WAITING 4, CROSSING 3, WALKING AWAY 3, 16 intent state
changes, 1 CROSSING AHEAD warning, first at 3.6 s, warning active
15.8 s.** The log reads as scripted: A approaches (CROSSING at
3.6 s) → waits at the kerb (WAITING at 6.7 s) → crosses the zebra
(CROSSING at 11.5 s) → walks away (WALKING AWAY at 19.4 s);
B crosses the right zebra (CROSSING at 5.1 s) → walks away
(13.0 s); C walks away from the start (WALKING AWAY at 1.7 s);
D reads CROSSING briefly while striding up to the kerb (15.9 s)
then WAITING (18.0 s) — the documented "walking to the kerb"
limit above.

## Bugs found and fixed during verification

1. **Nearest-zebra heading reference (intent logic):** the first
   build judged heading against the nearest zebra edge only, so
   the figure walking up to the main zebra flipped to WALKING
   AWAY at 3.3 s — it was nearest to the right zebra while
   walking toward the main one. Heading is now judged per zebra:
   CROSSING if the pedestrian is near ANY zebra and heading
   toward THAT zebra; WALKING AWAY if heading away from the
   NEAREST zebra.
2. **Max-dot over all zebras (second attempt at bug 1):** taking
   the largest dot product over all zebras made the figure walking
   along the bottom pavement read as heading toward the far-left
   zebra (500 px away) and never reach WALKING AWAY. Replaced by
   the per-zebra rule in (1): the toward test carries its own
   distance gate per zebra.
3. **Tracker fragmentation (matcher):** the first demo run
   produced 248 tracks for ~10 real people — YOLO's box jitters
   by ~25 px on these small (~30×80 px) blurred figures, and the
   fixed 60 px gate with a 0.05 IoU floor spawned a new track for
   the same person every ~1.2 s. Fixed with a scale-aware gate
   (half the bigger box's size), a 0.02 IoU floor and a 3 s
   missed-detection hold. The final run still counts 193 tracks
   for ~10 real people per pass — better, but fragmentation
   remains the main honest limit (see above); the detection
   itself is stable (1,891 detections, identical across runs),
   so the churn is in association, not detection.
4. **Warning-time under-count:** warning seconds were accumulated
   inside `update()`, which only runs on detection rounds (every
   3rd frame) in real mode, so the total under-counted by up to
   two-thirds. Warning time is now accumulated once per output
   frame in the main loop.
5. **Overlapping intent tags:** at the kerb, three or four tags
   printed on top of each other into unreadable mush (the same
   failure the wrong-way-driver-alert build fixed by dropping
   small-box labels). Tags are now collected, sorted by height,
   and any tag that would overlap the one above it is pushed down
   below it.
6. **Simulation figure C too slow:** C's first path ran at
   15.0 px/s — exactly on the 15 px/s standing gate — so it read
   WAITING for the whole run instead of WALKING AWAY. Its path
   was shortened to 12 s (22 px/s) and angled away from the
   crossing so its heading dot product clears −0.4.
7. **Confirmed state regressing to MEASURING:** a detection gap
   thins the 1.5 s velocity window, the fit goes None, and the
   candidate became MEASURING — on the second demo run a WAITING
   pedestrian flipped back to MEASURING at 15.5 s and 24.4 s. A
   confirmed state is now held through a thin history; only a
   track that has never been confirmed reads MEASURING. The
   final run has no MEASURING regressions.

## Files

- `pedestrian_crossing_intention.py` — the detector
  (`--source pick | <file> | <webcam>`, `--simulate`,
  `--max-seconds`, `--loop`, `--no-display`, `--out`)
- `yolov8n.onnx` — YOLOv8n weights (12.8 MB, under GitHub's 25 MB
  web-upload cap). If missing, the script exports it once from
  `yolov8n.pt` via Ultralytics or prints the download link.
- `requirements.txt`, `.gitignore`
- Demo: `../../project-demos/pedestrian-crossing-intention-demo.mp4`
  (+ H.264 copy in `../../project-demos/linkedin-h264/`,
  codec-verified h264) with
  `pedestrian-crossing-intention-demo_events.log` and
  `pedestrian-crossing-intention-demo_summary.txt` alongside it.
- Raw footage: `../../project-demos/real-footage/raw/
  pedestrian-crossing-raw.mp4` (see `SOURCES.md`).

## Run it

```bash
pip install -r requirements.txt
python pedestrian_crossing_intention.py --source street.mp4
python pedestrian_crossing_intention.py --simulate --max-seconds 30 --no-display
```

Press Q to quit. The video is written to `output.mp4` (or `--out`),
the timestamped event log to `output_events.log` and the final
counters to `output_summary.txt` next to it.

Author: Abdul Hanan — Computer Vision & AI Developer
