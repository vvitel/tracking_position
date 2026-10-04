# training-tools

A GUI for putting a ground truth court on a folder of plates.

```
python training-tools/label_court.py                  # pick the folder in the window
python training-tools/label_court.py /path/to/plates --jobs 6
```

Needs nothing the detector does not already need: numpy, OpenCV and the
`tkinter` that ships with Python. Pixels reach the canvas through Tk's own PPM
reader, so there is no Pillow.

## What it writes

Three files beside each plate, named after it:

| file | who wrote it | what is in it |
| --- | --- | --- |
| `<stem>_court-lines.json` | the detector | the walk's samples, centred on the painted lines |
| `<stem>_court-sides.json` | you | the left, right and far boundaries, as clicked |
| `<stem>_court-truth.json` | the fit | the court, in the schema `batch_courts.py` writes |

`_court-truth.json` is deliberately the same shape as the detector's own output -
lens (`fisheye`, `k`, `f`, `cx`, `cy`), homography, and the 13 keypoints in
original image pixels - plus a `labelled` block saying where it came from, so a
truth file is never mistaken for a detection.

## The loop

1. **Folder...** lists every plate, with a `walk` and a `sides` column. The
   **Filter** box narrows the list by filename, case-insensitively; every
   whitespace-separated term has to appear, in any order, so `5f-c6 04-01` finds
   one camera on one day. **Next**, **Prev** and **Save + next** all stay inside
   the filtered set, so it works as a way of labelling one venue at a time.
2. **Generate walks** runs `padelcourt.walk.detect` over every plate that has no
   `_court-lines.json` yet, on a pool of worker processes; the list updates as
   each one lands. Plates that already have the file are skipped, including
   plates whose walk *failed* - the failure is written into the file, so it is
   recorded once rather than retried on every launch.
3. Open a plate, press **1** for the sidelines or **2** for the far side, and
   click along it. The quadratic appears from the third mark; five are required
   per side, and one of them should be the far corner.
4. With five on every side the court is fitted and drawn over the plate.
5. **Save + next** (**S**) writes both files and opens the next plate that still
   needs marking.

There is no left/right choice to make: a click on a sideline is filed by which
side of the court's midline it landed on, and the midline is the `centre_line`
the walk found - world `X = 5` - rather than the middle of the frame, because
nothing guarantees the camera is centred on the court and a wide-angle lens bows
that line by tens of pixels down the picture. It is drawn as a dotted line while
marking, so the split is visible rather than implied, and held at its endpoints
past the ends of the paint rather than extrapolated. Where the walk failed
entirely it falls back to the middle of the frame.

| | |
| --- | --- |
| wheel | zoom under the cursor |
| middle / right drag | pan |
| **F** | fit the plate to the window |
| **1** / **2** | sidelines / far side |
| **E** | eraser - click a mark to delete it |
| **C** / **W** | show the fitted court / show what the walk found |
| **D** | step the courtside dropdown - off, all four, then one at a time |
| **S**, **N**, **P** | save and next, next, previous |

Shortcuts are ignored while the filter box has the keyboard, so typing `s` in it
searches rather than saves.

## Comparing the courtside detectors against your marks

The detector has four ways of finding a sideline and tries them in order, taking
the first whose curve reaches the corner where the near service line's paint
stopped. Where a plate has a `<stem>_courtside-detection.json` beside it the
**Courtsides** dropdown comes alive and draws what each of them found - its
points, and its own quadratic through them - over the picture you are marking:

```
python compare_sides.py /path/to/plates --write-detections   # a folder
python detect_court.py plate.png --sides-json                # one plate
```

The one the cascade actually chose is drawn thick, a refused one dashed, and the
white crosses are the corners every reader is judged against. `all (4)` adds a
legend and puts the four corner misses in the status line, which is the number
the choice is made on. A reader with clean points and a curve dragged off them
looks different from one that followed the wrong feature entirely, and both look
different from your marks - which is the whole reason to draw them together.

Nothing about the file is required: no file, no dropdown.

The status line under the picture carries the numbers worth watching: how far
each side's marks sit off their own curve, then the fit's line rms, the paint
coverage and the probe residual - the detector's own three gates, applied to the
court you just marked. They do not refuse anything, because your marks are the
ground truth by definition; a gate that goes red is a plate to go and look at.

## Why the walk is needed to fit the marks

The three boundaries you mark are two degrees of freedom short of a camera on
their own. They give six constraints (two per line correspondence) against eight
for the homography plus two for the lens, and what they leave loose is the Y
scale: nothing in the left sideline, the right sideline and the far baseline says
where the near baseline is. The two far corners do not help - they are the
intersections of lines already marked.

So the fit takes your three boundaries *and* the three painted lines the walk
found (`near_service`, `centre_line`, `far_service`), which is three lines in each
direction - the six-line design `solve.py` was built around, with the half the
detector measures worst replaced by a person. The sidelines the walk found are
discarded; yours are used instead.

A plate whose walk failed can still be marked, and its marks are saved. It just
has no court until the walk is fixed or replaced.

## How the marks are weighed

A clicked point on the left sideline knows one thing about the world - that it
lies on `X = 0` - so it enters as a line sample. Six of those against the near
service line's four hundred would lose every argument under a plain least
squares, so **every line carries the same total weight regardless of how many
samples it has**: each one is a single correspondence and is treated as one.

That weighting cannot coexist with a Huber threshold, so the fit runs in two
passes - one robust and unweighted, purely to catch a walked line that is not the
line it claims to be, then one weighted and plain over what survived. Your marks
are never screened out.

The two far corners are the exception that *does* carry a full world position, so
they go in as real point anchors at `(0, 0)` and `(10, 0)`, alongside the walk's
own three. They are taken as the intersection of the two fitted curves rather
than as whichever click was outermost.

Measured against cameras it never saw - four plates calibrated the ordinary way,
their boundaries reprojected as six marks a side with 1.5 px of click jitter -
the labelled fit reproduces the reference keypoints to a median of 2.1 px and a
worst of 3.1 px, at 100% paint coverage on all four.

## Files

```
label_court.py        entry point
labeltool/store.py    the three sidecar files, and which plates need which
labeltool/walks.py    run the walk, write and read `_court-lines.json`
labeltool/sides.py    read `_courtside-detection.json` for the overlay
labeltool/fit.py      marks + paint -> Camera -> the two documents
labeltool/canvas.py   the zooming, panning image widget
labeltool/app.py      the window
```
