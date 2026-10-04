# sys_padel-court-detector

Finds the court in a padel recording and returns a camera you can use: the lens
coefficients, the court-to-image homography, and the 13 court keypoints in the
pixels of the original frame.

It works from a **plate** - a still of the empty court, built by taking the
pixel-wise median of keyframes spread across the whole recording. Players and
the ball are somewhere different in every frame and lose the median; the court
is in all of them and survives. No player detection, no per-frame tracking, and
one calibration is valid for the whole recording as long as the camera does not
move.

```
python detect_court.py match.mkv
python detect_court.py court_plate.png --json court.json --pictures out/
python batch_courts.py /path/to/videos
python batch_courts.py /home/jonathan/Projects/streamyoursport/raw-videos/padel/plates/normal-view   --kind plate   --ext .png   --out /home/jonathan/Projects/streamyoursport/raw-videos/padel/plates/normal-view/courts
```

For how the algorithm works, and the road that led to it, see
[how-it-works.MD](how-it-works.MD).

## Install

```
source .venv/bin/activate
uv pip install -r requirements.txt        # numpy, opencv-python
```

Plate building shells out to **ffmpeg**. It is used with `-skip_frame nokey`,
which is what makes a 21-minute file readable in ~25 s: these recordings are raw
H.264 in MKV with no index and no usable timestamps, so seeking does not work and
keyframes are the only way in. If `ffmpeg` is not on `PATH`, add its directory to
`FFMPEG_DIRS` in [padelcourt/plate.py](padelcourt/plate.py) - it already knows
about `/opt/AgentDVR/ffmpeg7/bin`.

Calibrating one plate takes 10-20 s on a laptop core. Building a plate from a
video takes about as long again, and is cached.

## One video or one plate

```
$ python detect_court.py match.mkv
video match.mkv
      316 keyframes seen, 45 kept (stride 4)
      wrote match_plate.png   (ghost 0 px)

plate       /path/match_plate.png
verdict     ACCEPT
quality     line rms 1.10 px   paint coverage 100%   probe 0.64 px

lens        fisheye   k = 0.030066, -0.041974   f = 960.0   centre = (960.0, 540.0)
            theta   = atan(r_ideal)
            theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + ...)
            r_dist  = theta_d
            u = f * x_dist + cx,   v = f * y_dist + cy

homography  court metres -> pixels of the UNDISTORTED image
            [     53.2602     -46.3366     717.6564]
            [      1.6165       3.9767      81.8755]
            [     -0.0010      -0.0462       1.0000]

keypoints   (original image pixels)
            far_corner_l    world ( 0.00,   0.00) m   image (  736.71,   117.89)
            ...
            near_corner_r   world (10.00,  20.00) m   image ( 2137.44,  1136.85)   derived   OFF FRAME
```

Useful flags: `--json out.json` (the full result), `--pictures out/` (the three
pictures), `--undistort flat.png` (the plate with the lens removed),
`--quiet` (only the JSON on stdout, progress on stderr, so it pipes).

The exit code is **0** accepted, **1** calibrated but refused, **2** no
calibration at all.

## A folder

```
python batch_courts.py /path/to/videos            # or a folder of plates
python batch_courts.py /path/to/videos --out /elsewhere --jobs 8
```

Plates are written **beside their video** as `<name>_plate.png` and reused if
already there. Everything else goes into one folder, `courts/` under the input
by default:

```
courts/<plate>_court_debug.jpg      what the detector walked, always written
courts/<plate>_court_success.jpg    the fitted court drawn over the plate, when accepted
courts/<plate>_court_fail.jpg       the same, when refused - the picture that shows why
courts/<plate>.json                 lens, homography, keypoints, quality
courts/index.json, index.csv        one row per input
```

`--kind plate` takes every image in the folder rather than only `*_plate.png`;
`--kind video` ignores images. In the default `auto` mode a plate belonging to a
video in the same run is skipped, so nothing is calibrated twice.

## The four courtside readers, and how one is chosen

A sideline is found by four different means - `canny`, `track`, `scans`,
`chord` - and choosing between them happens in two steps, both argued at
`walk.SIDE_SELECT`:

1. **The corner test is a veto.** A reader's curve has to reach the point where
   the near service line's paint stopped, or it is thrown out. It is the only
   measurement here that tests whether the right FEATURE was followed: a reader
   locked onto the base of the fence returns a beautifully clean arc, so no
   measure of smoothness can see it.
2. **The survivors are ranked by a provisional fit.** A bare camera is solved
   from the other four lines plus each surviving reader's two sidelines, and the
   reader kept is the one whose own samples land closest to `X = 0` and `X = 10`
   *in metres* through that camera. Pixels do not work here - a pixel at the net
   is worth several times the court a pixel at the near baseline is.

Over 65 labelled plates that scores 60 accepted against 58 for the older
first-past-the-corner cascade, gaining two and losing none, and 60 is what an
oracle with the truth file open also scores.

```
PADEL_SIDE_SELECT=cascade python detect_court.py plate.png    # fit|cascade|graded
PADEL_SIDE_SOURCES=chord python detect_court.py plate.png     # canny|track|scans|chord
python detect_court.py plate.png --sides-json                 # what all four found
python compare_sides.py /folder/of/labelled/plates            # every rule, scored
```

`compare_sides.py` takes the plates in a folder that carry a `_court-truth.json`,
runs each technique over every one of them, and scores each run by how far its 13
keypoints land from the labelled ones. With `--write-detections` it also leaves a
`<plate>_courtside-detection.json` beside each plate - every reader's points and
its own fit - which the label tool draws over the picture.

## What you get back

```jsonc
{
  "plate": "/path/match_plate.png",
  "image_size": {"width": 1920, "height": 1080},
  "accepted": true,
  "reason": "",                       // why it was refused, when it was
  "quality": {
    "line_rms_px": 1.10,              // how well the model fits the paint it was fitted to
    "paint_coverage": 1.0,            // how much of the model's lines land on real paint
    "probe_rms_px": 0.64,             // offset of the model from that paint, independently measured
    "max_anchor_px": 5.30,            // worst distance from a detected junction to its modelled place
    "n_anchors": 3, "n_lines": 5, "n_samples": 805
  },
  "lens": {
    "model": "fisheye", "k": [0.030066, -0.041974],
    "f": 960.0, "cx": 960.0, "cy": 540.0,
    "equation": "theta = atan(r_ideal) ..."
  },
  "homography": {
    "matrix_undistorted_px": [[...], [...], [...]],   // court metres -> UNDISTORTED pixels
    "matrix_normalised":     [[...], [...], [...]],   // court metres -> ideal normalised
    "valid_on": "an UNDISTORTED image - ..."
  },
  "keypoints": [
    {"name": "near_svc_c", "world_m": [5.0, 16.95], "image_px": [963.24, 896.07],
     "derived": false, "in_frame": true}
  ],
  "camera_model": {"Hn": [[...]], "k": [...], "model": "fisheye", "f": 960, "cx": 960, "cy": 540}
}
```

### The one thing to know about the homography

The camera is **two-stage on purpose**:

```
court (X, Y) --Hn--> ideal normalised --lens--> pixels
```

A homography maps straight lines to straight lines, so no 3x3 matrix can
describe a wide-angle camera - on this footage a straight service line bows
65 px. So `matrix_undistorted_px` is valid **only on an undistorted image**.
Applying it to a raw frame puts the whole lens error back.

Two correct ways to use the result:

```python
from padelcourt import undistort_image
from padelcourt.court import Camera
import json, numpy as np, cv2

d   = json.load(open("courts/match_plate.json"))
cam = Camera.from_dict(d["camera_model"])

# 1. raw frame, full model - no undistortion anywhere
uv = cam.project([[5.0, 10.0]])          # net centre  -> pixels
XY = cam.backproject([[963.0, 896.0]])   # a pixel     -> court metres

# 2. plain homography, on a frame that has been flattened first
flat = undistort_image(cv2.imread(d["plate"]), cam)
H    = np.array(d["homography"]["matrix_undistorted_px"])
p    = H @ [5.0, 10.0, 1.0]
uv_flat = p[:2] / p[2]
```

`Camera.project` / `backproject` handle the distortion in both directions
(the inverse is a Newton iteration, since the forward model is not invertible in
closed form).

### The keypoints

Court coordinates are metres on the floor: **X** 0 at the left sideline to 10 at
the right, **Y** 0 at the far baseline to 20 at the near one (nearest the
camera). Net at Y=10, service lines at Y=3.05 and Y=16.95, centre line at X=5.
Those are FIP standard dimensions and every regulation court matches them, which
is what lets one fixed model fit every venue.

11 keypoints lie on painted intersections; the two near corners are `derived` -
on a typical mount they are outside the frame, and projecting them through a
solved homography is both easier and more accurate than trying to detect them.
`in_frame` says whether a point actually lands inside the image; points outside
it are still meaningful, a homography being happy to be evaluated anywhere.

All keypoint pixels are in the **original** image, distortion included, so they
can be drawn straight onto the frame as it comes out of the camera.

## The verdict

`accepted` is the pipeline's own opinion, and it is deliberately strict:

| gate                        | threshold                                             |
| --------------------------- | ----------------------------------------------------- |
| paint coverage              | > 85%                                                 |
| independent paint probe rms | < 2.0 px                                              |
| line fit rms                | < 2.0 px                                              |
| worst anchor error          | ≤ 25 px                                               |
| line redundancy check       | every held-out line must measure where the model says |

A refused result still carries a full camera, and is still written out with its
`_court_fail` picture. On a plate whose far half is faint the fit can be
geometrically right and still miss the coverage gate - look at the picture
before believing the flag.

## How it works, in one paragraph

(The long version, including everything that was tried and discarded, is in
[how-it-works.MD](how-it-works.MD).)

A box scan finds the near service T - the junction of the near service line and
the centre line - and two ridge walks start from it, one along each line,
re-centring on the ridge crest every 4 px. A crest is not the middle of a wide
stripe, so each walk is then slid onto the middle of its own paint: both Canny
edges are read perpendicular to the line at every sample, and the line moves to
their midpoint - which is the follower's error exactly, with a RANSAC quadratic
through their *sum*, the stripe's width, deciding which pairs to believe. The
junction is then re-measured by
intersecting the two walked lines. Column scans cut the frame into court/gap/court
to locate the net's floor line, the far service line and the far baseline by
topology rather than by brightness; row scans find the sidelines, which on a
padel court are not painted at all but are the boundary between the surface and
the glass. The far service line is the hard one: the centre line has to cross the
net to reach it, and the net has a white vertical strap standing directly over
the line. What the walk crosses is recorded by a Hough pass that stops for
nothing, and the far service line is chosen afterwards as the clean line with a
gap just above it - because the centre line ends where the far service line
crosses it. Everything is then solved together by Levenberg-Marquardt for the
homography and the radial coefficients at once, with two model-seeded retrace
rounds, and checked against paint the fit never saw.

The full story, including what was tried and did not work, is in the research
tree's `CourtWalk/README.md`.

## The second solver: fit the pose, read the courtside where it says

`fit_court.py` is a **different way of solving the same plate**, not a stage of
the pipeline above. It is scored against it on the same labelled plates and by
the same measure, and both are kept.

```
python fit_court.py plate.png --pictures out/
python fit_court.py /folder/of/plates --out /folder/pose      # every image in it
python fit_court.py --compare /folder/of/labelled/plates      # scored against solve.py
```

Given a **folder** it calibrates every image in it and writes, per plate,
`<plate>.json` (the full result - lens, homography, keypoints, camera model),
`<plate>_court_debug`, and `<plate>_court_success` or `_court_fail` depending on
its own verdict, plus one `index.csv` / `index.json` across the run. Pictures it
has written itself are skipped on a second pass, so the output folder can live
inside the input one.

It writes the same three pictures `detect_court.py` does, under the same names -
`<plate>_court_debug`, and `_court_success` or `_court_fail` depending on its own
verdict - into whatever folder you point `--pictures` at. Same names on purpose:
put them in a different folder from `courts/` and the two solvers' pictures for
one plate can be flicked between with only the court changing. `--ext .jpg`
makes them about ten times smaller.

The debug picture shows a different kind of thing from `_court_debug` on the
other side, because the solver works a different way. There is no walked path
here - there is a **prediction and a band around it** - so what it draws is the
window the reader was allowed (green), the samples it decided were the boundary
inside that window (yellow), the net's foot (blue), the paint it was handed
(magenta/white/orange) and where the fitted court finally put every line
(orange). A fit that has gone wrong shows it one of two unmistakable ways: the
band sits somewhere that is not the sideline, or the band is right and the
samples inside it scatter or cling to one wall of it.

The verdict is `calibrate`'s four gates at `calibrate`'s numbers - paint
coverage > 85%, independent paint probe < 2 px, line rms < 2 px, worst anchor
<= 25 px, plus the per-line redundancy check. Not re-tuned: a verdict invented
alongside the fit it judges tends to be one the fit passes, and `paint_probe` is
the one measurement in this repository that no fit has ever seen.

**The line-rms gate is measured on the near half only**, with the far segment
reported beside it. That split is the one thing changed after seeing the gate
disagree with the truth, and the reason is that the two numbers move opposite
ways: the far samples scatter more in pixels *and* pin the line's direction
better, being at the end of a long lever arm, so adding them took the court's
error down by a third and the line rms up from p90 1.72 px to 2.55 - straddling
a gate of 2.0. Twenty plates were refused and all twenty were right. A mean
residual cannot tell "noisier" from "further away", so the gate keeps the
footing it was calibrated on and `far_rms_px` carries the rest.

On the 80 labelled plates it accepts 67 and refuses 13, and **all 13 refusals
are false** - every one is inside tolerance. Ten fail on coverage with probe rms
of 0.55-1.01 px against a gate of 2.0, meaning the model is demonstrably on the
paint and there simply is not enough of it lit for the probe to find; eight of
those are refused by `detect_court.py` too, so they are a property of the plate.
The gates fail in the safe direction - nothing is accepted that should not be -
but the README's older advice holds here as well: **look at the picture before
believing the flag.**

What changes is the parameterisation and where the sideline comes from.
`solve.py` fits an 8-DOF homography and detects the sidelines **once, blind**,
from a straight chord drawn between the four service-line endpoints - a chord
whose position is a bracket rather than an answer, because the far service
line's paint overruns the sideline by a median 0.27-0.32 m. `fitcourt.py` fits a
**6-DOF metric pose** (three angles, three metres) plus the two radial
coefficients, and never detects a sideline blind: the pose **projects** X = 0
and X = 10, lens bow included, and the edge is read in a band around that
prediction, perpendicular to it, with the same directional-Sobel-and-run-test
reader `walk.chord_lines` uses. Then re-read and re-solved, in a band that
narrows each round, until it stops moving.

Over the 80 labelled plates, by `compare_sides.court_error_m`:

| solver     | accepted  | worst visible point p50 | p90     | worst   |
| ---------- | --------- | ----------------------- | ------- | ------- |
| pose fit   | **80/80** | **0.072 m**             | 0.120 m | 0.246 m |
| `solve.py` | 74/80     | 0.161 m                 | 0.284 m | 0.666 m |

Each sideline is read in **two segments** - the near half (Y 10.3-17.6) and the
stretch between the far service line and the net (Y 3.6-9.7), with a half-metre
gap at Y=10 where the net post sits on the line and there is nothing to read.
Fitting the far segment as well is worth a third off every percentile, and most
where it was worst: the far band's p90 goes 0.293 m to 0.174. That contradicts
`walk._chord_read`, which measures the far half 0.034 m outside the line and
loses plates by including it - and the reason it does not carry over is the
reader, not the court. That one seeds from a straight chord and searches ±140 px;
this one seeds from a projected world line in a band that has narrowed to ±10 px
by the last round, so the same stretch of court is being asked a much narrower
question. `--no-far-half` turns it off.

74 plates are accepted by both, 6 by the pose fit alone, none by `solve.py`
alone. Caveats worth keeping in view: the tolerance is 0.30 m and the worst
plate here is 0.290, so 80/80 has a centimetre of headroom; and these are the
same labelled cameras `solve.py` was developed against, which is fair but is not
a held-out test set.

**Two things it set out to do and did not.** The lens was meant to come from
`straighten/` and be held fixed, which would have made the courtside answer
independent of any lens fitted to the court's own lines. It does not survive the
corpus - blind focal lengths run from 522 to 1050 px on frames whose own
calibrations all sit at 960, a metric pose cannot absorb that, and held to it
the fit accepts 42 of 80. The lens is fitted here instead. And pinning f at w/2,
which is `court.py`'s convention, turns out to cost twelve plates: the errors
then cluster at 0.22-0.29 m on nearly every plate, which is a systematic scale
offset rather than scatter.

**The net's floor line** is read a third way - blur hard *along* where the pose
says the line runs, until the mesh fuses into one dark band, then find the
darkening across it. Measured through each plate's own truth camera, against the
column-topology reader in `walk.detect`:

| reader                     | plates read | p50    | mad   | full range      |
| -------------------------- | ----------- | ------ | ----- | --------------- |
| walk column topology        | 65 of 80    | 10.123 | 0.071 | 8.783 .. 13.249 |
| blur along + step across    | 80 of 80    | 10.184 | 0.051 | 9.797 .. 10.301 |

It reads *long* - 18 cm toward the camera - because a net's foot is cord, skirt
and the shadow in front of them. But 18 cm that repeats to a mad of 0.051 m is a
constant, not an error, the same way `walk.ANCHOR_OUT` and `FAR_T_SHORT` are, so
it can be anchored where it actually lands rather than at Y = 10. Fitted there,
swept over the corpus:

| `--net-weight` | accepted  | p50         | p90         | worst   |
| -------------- | --------- | ----------- | ----------- | ------- |
| 0 (default)    | **80/80** | 0.104 m     | 0.182 m     | 0.290 m |
| 0.35           | 79/80     | **0.086 m** | **0.143 m** | 0.369 m |
| 1.0            | 78/80     | 0.087 m     | 0.169 m     | 0.392 m |

Monotone in the weight: the net pulls the centre in and pushes the tail out. It
is **check-only by default** because the accepted count is the gate the rest of
this repository is judged by - the same conclusion `solve.CHECK_ONLY` reached
about the net from different evidence - but if typical accuracy matters more
than worst-case coverage, `--net-weight 0.35` is the better setting and the
table is here to say so. What the reader is unambiguously good for either way is
telling you whether the fitted Y = 10 lands where the net is, on every plate
rather than on 65 of them.

## Layout

```
detect_court.py          one video or one plate
fit_court.py             the second solver, alone or scored against the first
batch_courts.py          a folder of either
compare_sides.py         each courtside technique on its own, scored against the
                         labelled plates in a folder
padelcourt/
  plate.py               a still of the empty court: median of the video's keyframes
  image.py               what the detector reads off the pixels - court colour,
                         directional top-hats, ridge and edge peaks, the RANSAC
                         curve trim, centring a line on its paint, undistort maps
  court.py               court dimensions, the six lines, and the Camera
  walk.py                the detector: walk out of the near service T, scan for
                         the rest, and choose the far service line
  solve.py               the camera fit (homography + radial distortion, jointly)
                         and the model-seeded re-trace that refines it
  fitcourt.py            the SECOND solver: a metric pose fitted against courtside
                         edges read where the pose predicts them, and the net's
                         foot read by blurring along it. Independent of solve.py -
                         shares walk.py's paint lines and nothing else
  calibrate.py           detect -> solve -> re-trace -> judge, and the paint probe
                         that judges it
  draw.py                the overlay and the walk picture
  api.py                 the shape of the answer
straighten/              the lens ALONE, with no court model and no homography -
                         solved from every straight edge in the frame, the hall's
                         and the court's. Separate because a rectified frame is
                         useful before anything is calibrated; see its README
```

Each module is named for what it does in the pipeline above, and contains only
code that pipeline reaches. The algorithm went through two earlier front ends -
a click-seeded fit and a region/top-hat line detector - and their remains have
been removed rather than left in place: an unused `detect()` that another
detector once called is a trap for the next reader, not history. The history is
in the research tree's `CourtWalk/README.md`, which is where it belongs.

What that pruning removed, for the record: the alternative line detector and its
region-outline and ridge-curve seeding, the click-seeded `fit_court`/
`trace_court` path, the batch tool's camera-grouping and cross-recording
comparison, and every `main()` belonging to a research CLI. 6493 lines became
3738, and the four reference plates produce byte-identical JSON before and after.
