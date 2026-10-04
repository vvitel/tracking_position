# straighten - the fisheye lens, and nothing else

Solves the lens distortion of a single frame from the straight edges in it: every
one the hall happens to contain, plus the court's own lines, which are walked by
[padelcourt/walk.py](../padelcourt/walk.py) and fed to the same objective. No
homography and no court model come out of it - the answer is a radial mapping.

```python
from straighten import fisheye_coefficients, undistort_image, undistort_points

lens = fisheye_coefficients("plate.png")      # path or BGR array
lens.f, lens.k1, lens.k2                      # the coefficients
lens.court                                    # {line: (bow before, after)} in px
lens.as_dict()                                # JSON-safe, incl. the mapping
flat, zoom, canvas = undistort_image(cv2.imread("plate.png"), lens)
xy = undistort_points([[100, 100]], lens)     # raw pixels -> straightened

fisheye_coefficients(img, court=False)        # hall edges only, ~4x faster
```

```bash
# every image in a folder -> <folder>/straighten/, non-recursive, overwriting
python straighten/batch_straighten.py /path/to/plates
python -m straighten.batch_straighten /path/to/plates --csv
```

`--csv` also writes `lens.csv` with the coefficients, the radial mapping, and
what the court contributed. `--no-court` is the hall-edges-only solve, `--nk`
sets how many radial coefficients to fit (2; three is not better), `--thr` the
Canny threshold, `--no-corners` keeps the frame corners, `--pad` the output
canvas size.

Requires `numpy` and `opencv-python`, both already in `requirements.txt`. The
court half also needs `padelcourt`, which is the rest of this repository; a copy
of this package on its own still runs, and silently gives the blind solve.

## The model

```
theta  = r_ideal / f          via   r_ideal = f tan(theta)
r_dist = f * theta * (1 + k1 theta^2 + k2 theta^4)
```

`cx, cy` is fixed at the image centre rather than fitted - a real principal
point can be tens of pixels off, but freeing it per frame costs more robustness
than it buys. `Rref` is the gauge radius: undistorted coordinates are defined
only up to scale, so the mapping is normalised to hold the image-corner radius
fixed. Anything consuming `f, k` must use the same convention.

## How it works, and the one trap

A straight world line is straight in the image only when the lens model is
right, so cut the Canny contours into arcs and minimise

```
bend = (rms distance off the arc's own best-fit line) / (arc extent)
```

weighted by extent squared, truncated so genuinely-curved things contribute a
constant. Coarse grid on `(f, k1)`, then Nelder-Mead on all of it.

**The trap is the objective.** Maximising Hough concentration is the obvious
choice and it is degenerate: it rewards any mapping that squashes the interior
of the frame, which is not the same thing as straightening it. On the reference
plate it scored the wrong lens twice as high as the truth and landed *further*
away than doing nothing (375 px against 278). `bend` cannot be gamed that way,
because both of its terms scale together.

Three details that mattered more than expected:

- **Arc length is the signal.** Sag grows with the square of length, so long
  arcs are worth far more. Spans of 192/384/768 together beat any single length
  by 3.9 px against 21 px.
- **The pre-filter must be loose.** Arcs are screened for straightness in the
  *distorted* frame. Screening at 0.06 throws away exactly the long peripheral
  arcs that carry the most information, because they are genuinely bent there.
- **The corners are erased first** (400x200 top, 250x200 bottom). That is where
  a broadcast frame keeps its banner, scoreboard and watermark; those graphics
  are composited in image space, so they are straight in the distorted frame by
  construction, and a plumb-line fit that believes them is being told to leave
  the lens alone exactly where it does most. `--no-corners` disables it, which
  you want for frames that are not broadcast-framed.

## The court's own lines

The hall's edges are enough on most plates, but they are not evidence about the
court, and where the periphery is mostly burnt-in overlay the blind fit can
over-correct and bow the court lines the *other* way while every straightness
measure it can see still reads healthy. The court's lines are the only straight
things in frame that are certainly straight, certainly not composited, and
certainly what the calibration will be judged against. [court.py](court.py)
walks them - a walk needs no lens knowledge - and hands them to the same
objective at 15% of the total weight, since the paint owns the middle of the
frame and the building still owns the periphery.

Two things about them are not like the hall's arcs, and both had to be measured:

- **The point sets must be fitted before they are measured.** A walked line
  carries about half a pixel of per-sample scatter, which on a line of extent
  500 px is a *bend* of 1e-3 - the same order as the truncation, so a raw point
  set arrives near saturation and barely pulls: fed raw, the court is worth
  6.3 → 6.2 px and 12.3 → 10.8 against the two calibrated cameras, and fed
  fitted it is worth 6.3 → 2.4 and 12.3 → 8.5. Worse, the one way to lower a
  noise-dominated ratio is to grow its denominator, so what pull there is points
  at magnifying the middle of the frame rather than at straightening anything.
  Re-laying each line on a **degree-4** fit of itself, with two passes of outlier
  rejection, removes the scatter without touching the bow. Degree 2 is worse
  than not smoothing at all (8.6/12.2 px against 6.2/10.8): a fisheye's image of
  a straight line is not a parabola over 1800 px, and a quadratic forced through
  it misrepresents the shape by more than the noise it removes. The lens model
  is itself 4th order in theta, and 4 is where both metrics below stop
  improving together. The rejection passes are worth as much as the degree - a
  least-squares fit of any order follows a stray tail sample, and they take the
  worst same-camera disagreement from 63 px to 38.
- **Uncapped is wrong.** Undistorted with a *click-calibrated* lens, these lines
  still bow 0.5-3 px - at both reference venues, on every line. The paint is laid
  by hand and the detector has its own systematic error along a line, and
  neither is something a lens can fix. Told to make them straight anyway, the
  fit answers by bending the lens, and rotonde goes from 12.3 px out to 30.
  `COURT_CAP` is what says how straight a court line is expected to be; past it
  a line stops pulling.

The far baseline is never fed - the glass hides the strip of floor in front of
it and the posts scallop what is left, and it measures 20-30x the bow of any
painted line. Anything else that bows more than 3x the near service line's is
dropped the same way, by `court.CURV_RATIO`.

`lens.court` reports the **signed** bow of each line before and after, in source
pixels. Signed on purpose: the failure this evidence exists to prevent is a line
that comes back bowed the *other* way, and every unsigned measure calls that an
improvement.

## Never compare two solves by their coefficients

`(f, k1, k2)` is badly degenerate. Over 103 plates of one camera model, `f`
spanned 575-1626 and `k1` spanned -0.47 to +0.29 including sign changes, while
the radial mapping those triples produce agreed to about **3%**. Two plates of
the *same* camera came back as `f=1626, k1=-0.472` and `f=840, k1=+0.119` - and
their mappings differ by 13 px at r=400 out of ~190 px of distortion.

Use `Lens.displacement()`, which reports how far a feature at a given radius
moves when the frame is straightened. That is what the rest of a pipeline
consumes, and it is what is actually determined.

## What it is worth

Two measurements, because neither is enough on its own. The first is accuracy
against a click-calibrated camera and there are only two of those; the second is
whether one fixed camera is given the same lens twice, which needs no ground
truth and can be run over the whole corpus.

Max radial deviation over the frame from the click-calibrated mapping:

```
                 blind    + court
galais            6.32       2.35      (doing nothing: 278 px)
rotonde          12.35       8.53      (doing nothing: 307 px)
```

Disagreement between two plates of one fixed camera, over 92 plates of 39
cameras, 71 pairs - the same quantity, measured where nobody clicked:

```
                 blind    + court
median           10.44       5.60
p90              32.07      16.25
worst            56.05      38.14
```

The walk gave usable lines on all 92 plates, a median of 6 each. What is left is real
and is reported: after the solve the worst line on a plate still bows a median
3.1 px and a p90 8.4 px, which is the bow that was already there under a
calibrated lens and is not the lens's to remove.

Three things measured and rejected, so they are not retried by accident:

- **Sub-pixel edge refinement makes the objective better and the answer worse.**
  It halves the residual sag of the hall arcs and moves the recovered lens away
  from the calibrated one, on rotonde by 2x. There is no radial bias in the
  shift; noise simply was never the limit. Two independent subsets of one frame's
  arcs imply lenses that differ by 15-44 px, far more than the 6-12 px by which
  the blind solve misses the truth.
- **More radial coefficients do not help.** The model is already degenerate at
  two; a third flattens the valley rather than determining the mapping.
- **Trusting the court more does not help.** At 25% of the weight instead of
  15%, or with the cap at 0.004, rotonde goes from 8.5 px out to 32 - the court's
  own residual bow becomes a bigger vote than the whole building. Nor does
  dropping the weaker lines instead of capping them: measured at a cap of 0.001,
  holding out the net line takes rotonde from 7.5 px to 32, and cutting back to
  the two paint lines the walk centres costs galais 4.5 → 5.4.
```
