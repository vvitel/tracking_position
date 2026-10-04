#!/usr/bin/env python3
"""Mark the ground truth court on a folder of plates.

    python training-tools/label_court.py
    python training-tools/label_court.py /path/to/plates --jobs 6

Writes three files beside each plate, all named after it:

    <stem>_court-lines.json    the detector's walk, generated in the background
    <stem>_court-sides.json    the left, right and far boundaries, marked by you
    <stem>_court-truth.json    the court fitted from the two, in the same schema
                               `batch_courts.py` writes

Marking: press 1 for the sidelines or 2 for the far side and click along it - at
least five points a side, including the far corner - and the quadratic appears
from the third. Which SIDELINE a click belongs to is read off which side of the
court's midline it landed on, so there is no left/right to choose. E for the
eraser, then click a point to remove it. The court is drawn as soon as all three
sides have five. S saves and opens the next plate that still needs one.

The Filter box narrows the list by filename, case-insensitively; every
whitespace-separated term has to appear, so `5f-c6 04-01` finds one camera on one
day. Next, Prev and Save all stay inside the filtered set.

Wheel zooms under the cursor, middle or right drag pans, F fits the plate to the
window, W shows what the walk found, C toggles the fitted court.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=None,
                    help="folder of plates to open (otherwise pick one in the window)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes for the walks (default: cores - 2)")
    a = ap.parse_args()

    from labeltool.app import run
    run(folder=a.folder, jobs=a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
