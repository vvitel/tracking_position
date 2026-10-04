#!/usr/bin/env python3
"""Straighten every image in a folder.

    python straighten/batch_straighten.py /path/to/plates
    python -m straighten.batch_straighten /path/to/plates --csv

Reads the folder NON-recursively, solves the fisheye distortion of each image
independently, and writes the straightened result to `<folder>/straighten/`,
creating that folder if needed and overwriting whatever is already there. Output
files keep the input's name and extension.

Each image gets its own solve, from the hall's straight edges AND the court's
own lines, which are walked per image. If the folder holds several frames from
one fixed camera they should agree closely - measured over 39 such cameras here,
the median pair agrees to 4 px of radial mapping against the 12 px they managed
before the court was fed in. Disagreeing by much more is a sign that one of them
is too plain, or that its court walk failed. `--csv` writes the coefficients, the
court lines used and the bow left on them, so that can be checked.

`--no-court` is the hall-edges-only solve: about four times faster, and about
twice as far from the two cameras there is a click-calibrated answer for.
"""

import argparse
import csv
import os
import sys
import time

# runnable both as `python straighten/batch_straighten.py` and `python -m`
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from straighten import (IMAGE_EXT, DEFAULT_RADII, fisheye_coefficients,
                            undistort_image)
    from straighten.lens import CORNERS
else:
    from . import IMAGE_EXT, DEFAULT_RADII, fisheye_coefficients, undistort_image
    from .lens import CORNERS

import cv2                                                        # noqa: E402

OUT_DIR = "straighten"


def images_in(folder):
    """Every image directly in `folder`, minus this tool's own output folder."""
    if not os.path.isdir(folder):
        raise SystemExit(f"not a folder: {folder}")
    out = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMAGE_EXT:
            out.append(p)
    return out


def run(folder, out_dir=None, nk=2, thr=255, no_corners=False, csv_path=None,
        pad=1.35, court=True, quiet=False):
    """Straighten every image in `folder`. -> list of (path, Lens or None)"""
    folder = os.path.abspath(folder)
    paths = images_in(folder)          # validate the input before creating output
    out_dir = out_dir or os.path.join(folder, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    if not quiet:
        print(f"{len(paths)} images in {folder} -> {out_dir}", flush=True)

    results, t0 = [], time.time()
    for i, p in enumerate(paths, 1):
        name = os.path.basename(p)
        t1 = time.time()
        img = cv2.imread(p)
        if img is None:
            print(f"[{i:3d}/{len(paths)}] SKIP  unreadable        {name}", flush=True)
            results.append((p, None))
            continue
        try:
            lens = fisheye_coefficients(img, nk=nk, thr=thr, court=court,
                                        corners=None if no_corners else CORNERS)
        except (Exception, SystemExit) as ex:
            # SystemExit is a BaseException, so `except Exception` sails past it
            # and takes the whole batch down on one bad frame
            print(f"[{i:3d}/{len(paths)}] FAIL  {type(ex).__name__}: {str(ex)[:60]}"
                  f"   {name}", flush=True)
            results.append((p, None))
            continue
        lens.source = p
        flat, _, _ = undistort_image(img, lens, pad=pad)
        dst = os.path.join(out_dir, name)
        if not cv2.imwrite(dst, flat):
            print(f"[{i:3d}/{len(paths)}] FAIL  cannot write {dst}", flush=True)
            results.append((p, None))
            continue
        results.append((p, lens))
        if not quiet:
            print(f"[{i:3d}/{len(paths)}] ok  {time.time()-t1:5.1f}s  "
                  f"f={lens.f:7.1f} k1={lens.k1:+.4f} k2={lens.k2:+.4f}  "
                  f"sag {lens.sag_before:4.2f}->{lens.sag_after:4.2f} px  "
                  + (f"court {len(lens.court)} lines, worst bow "
                     f"{lens.court_sag:4.2f} px  " if lens.court
                     else "no court lines  ")
                  + name, flush=True)

    ok = [l for _, l in results if l is not None]
    if csv_path:
        write_csv(csv_path if isinstance(csv_path, str)
                  else os.path.join(out_dir, "lens.csv"), results)
    if not quiet:
        print(f"\n{len(ok)}/{len(paths)} straightened in "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    return results


def write_csv(path, results):
    """Coefficients and the radial displacement they imply, one row per image."""
    cols = (["image", "width", "height", "f"]
            + [f"k{i+1}" for i in range(max((len(l.k) for _, l in results if l), default=2))]
            + [f"d{int(r)}" for r in DEFAULT_RADII]
            + ["Rref", "arcs", "straight", "sag_before", "sag_after",
               "court_lines", "court_sag"])
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for p, lens in results:
            if lens is None:
                continue
            row = {"image": os.path.basename(p), "width": lens.width,
                   "height": lens.height, "f": round(lens.f, 1),
                   "Rref": round(lens.Rref, 1), "arcs": lens.arcs,
                   "straight": lens.straight,
                   "sag_before": round(lens.sag_before, 3),
                   "sag_after": round(lens.sag_after, 3),
                   "court_lines": len(lens.court),
                   "court_sag": round(lens.court_sag, 3)}
            for i, kv in enumerate(lens.k):
                row[f"k{i+1}"] = round(float(kv), 6)
            for r, d in zip(DEFAULT_RADII, lens.displacement()):
                row[f"d{int(r)}"] = round(float(d), 2)
            wr.writerow(row)
    print("wrote", path, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="folder of images, read non-recursively")
    ap.add_argument("--outdir", default=None,
                    help=f"where the results go (default: <folder>/{OUT_DIR})")
    ap.add_argument("--nk", type=int, default=2,
                    help="radial coefficients to fit (default 2; more is not better)")
    ap.add_argument("--thr", type=int, default=255, help="Canny threshold, both ends")
    ap.add_argument("--pad", type=float, default=1.35,
                    help="output canvas as a multiple of the input size")
    ap.add_argument("--no-corners", action="store_true",
                    help="keep the frame corners, instead of erasing the blocks "
                         "where broadcast graphics live - see blank_corners()")
    ap.add_argument("--no-court", action="store_true",
                    help="hall edges only: do not walk the court's own lines. "
                         "Faster and less accurate - see the README")
    ap.add_argument("--csv", nargs="?", const=True, default=None,
                    help="also write the coefficients (default <outdir>/lens.csv)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.folder, a.outdir, nk=a.nk, thr=a.thr, no_corners=a.no_corners,
              csv_path=a.csv, pad=a.pad, court=not a.no_court, quiet=a.quiet)
    return 0 if any(l for _, l in res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
