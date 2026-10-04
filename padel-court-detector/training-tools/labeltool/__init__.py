"""Ground-truth labelling for padel court plates.

The detector in `padelcourt` reads a plate and returns a court. This package is
the other direction: a human reads the plate and states where the court is, so
there is something to measure the detector against.

    store   the three sidecar files, and which plates still need which
    walks   run the detector's walk over a plate, write `_court-lines.json`
    fit     marked sides + walked paint lines -> a Camera -> `_court-truth.json`
    canvas  a zooming, panning image widget
    app     the GUI that ties them together

WHAT THE HUMAN LABELS, AND WHY IT IS ONLY HALF THE COURT. The three boundaries -
left, right and far - are exactly the parts the detector finds hardest, because
none of them is paint: they are the base of the glass, a step in colour rather
than a bright ridge, and `court.OCCLUDED_BY_DEFAULT` exists because the far one
lies outright when it is walked. They are also, on their own, two degrees of
freedom short of a camera. Three lines give six constraints against eight for a
homography plus two for the lens, and the pair they leave loose is the Y scale:
nothing in the left sideline, the right sideline and the far baseline says where
the near baseline is. The two far corners do not help - they are the
intersections of lines already marked, so they carry no new information about
the plane.

So the labelled fit takes the human's three boundaries AND the three painted
lines the walk found (`near_service`, `centre_line`, `far_service`), which is
three lines in each direction: the six-line design `solve` was built for, with
the half it measures worst replaced by a person. See `fit` for how the two are
weighed against each other.
"""
import os
import sys

#: The repo root, so `import padelcourt` works however this is launched - and,
#: because it runs at import time, in a spawned worker process too.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
