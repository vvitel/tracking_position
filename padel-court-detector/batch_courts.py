#!/usr/bin/env python3
"""Detect the court in every video (or plate) under a folder.

    python batch_courts.py /path/to/videos
    python batch_courts.py /path/to/plates --out /somewhere/else --jobs 8

Plates are built beside their video as `<name>_plate.png` and reused if they are
already there. Everything else goes into ONE output folder - `courts/` under the
input by default - three pictures and one JSON per plate, plus `index.json` and
`index.csv` for the whole run:

    courts/<plate>_court_debug.jpg     what the detector walked
    courts/<plate>_court_success.jpg   the fitted court, when accepted
    courts/<plate>_court_fail.jpg      the fitted court, when refused
    courts/<plate>.json                lens, homography, keypoints, quality

A refused plate still gets its JSON and its `_court_fail` picture: the fit can be
geometrically right and still miss a gate, and the picture is how you tell.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
#: Pictures this tool writes, so a re-run does not try to calibrate its own output.
MADE = ("_court_debug", "_court_success", "_court_fail")

_CFG = {}


def find_inputs(root, kind, recursive=False):
    """Videos and/or plate images in `root`, in a stable order.

    Subfolders are only walked when `recursive` is set; otherwise just the files
    sitting directly in `root` are taken.

    A plate that belongs to a video in the same run is dropped: the video is
    processed through its own plate, so keeping both would calibrate the same
    image twice and race to write the same pictures.
    """
    from padelcourt import VIDEO_EXT
    from padelcourt.plate import plate_path
    if os.path.isfile(root):
        return [root]
    if recursive:
        walk = os.walk(root)
    else:
        names = sorted(os.listdir(root))
        walk = [(root, [], [f for f in names
                            if os.path.isfile(os.path.join(root, f))])]
    videos, plates = [], []
    for d, dirs, files in walk:
        dirs[:] = [x for x in sorted(dirs) if x != _CFG.get("out_name", "courts")]
        for f in sorted(files):
            p = os.path.join(d, f)
            low = f.lower()
            if kind in ("auto", "video") and low.endswith(VIDEO_EXT):
                videos.append(p)
            elif kind in ("auto", "plate") and low.endswith(IMAGE_EXT):
                if any(m in f for m in MADE):
                    continue                     # our own pictures
                if kind == "auto" and not f.endswith("_plate.png"):
                    # In `auto` mode only plates we recognise are taken, so that
                    # pointing at a video folder full of thumbnails does the
                    # obvious thing. `--kind plate` takes every image.
                    continue
                plates.append(p)
    theirs = {plate_path(v) for v in videos}
    return sorted(videos) + sorted(p for p in plates if p not in theirs)


def one(path):
    """Calibrate one input and write its pictures. Never raises."""
    from padelcourt import (VIDEO_EXT, calibrate_image_file, calibrate_video,
                            write_pictures)
    row = {"input": path, "name": os.path.basename(path)}
    try:
        if path.lower().endswith(VIDEO_EXT):
            res = calibrate_video(path, frames=_CFG["frames"], force=_CFG["force"],
                                  verbose=False, rounds=_CFG["rounds"],
                                  model=_CFG["model"], nk=_CFG["nk"])
            row["ghost_px"] = getattr(res, "ghost", None)
        else:
            res = calibrate_image_file(path, rounds=_CFG["rounds"],
                                       model=_CFG["model"], nk=_CFG["nk"])
    except BaseException as e:
        row.update(ok=False, error=str(e)[:200])
        if _CFG.get("traceback"):
            row["traceback"] = traceback.format_exc()[-1500:]
        return row

    stem = os.path.splitext(os.path.basename(res.plate_path))[0]
    d = res.to_dict()
    row.update(ok=res.ok, reason=res.reason, plate=res.plate_path, stem=stem,
               line_rms_px=d["quality"]["line_rms_px"],
               coverage=d["quality"]["paint_coverage"],
               probe_rms_px=d["quality"]["probe_rms_px"],
               max_anchor_px=d["quality"]["max_anchor_px"])
    out = _CFG["out"]
    os.makedirs(out, exist_ok=True)
    row["pictures"] = write_pictures(res, out, stem=stem, ext=_CFG["ext"])
    row["json"] = res.save_json(os.path.join(out, stem + ".json"))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder of videos and/or plates, or a single file")
    ap.add_argument("--out", default="", help="output folder (default: <input>/courts)")
    ap.add_argument("--kind", default="auto", choices=["auto", "video", "plate"],
                    help="what to look for; 'auto' takes videos and *_plate.png")
    ap.add_argument("--recursive", action="store_true",
                    help="also look in subfolders (default: only the input folder)")
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--frames", type=int, default=45, help="keyframes to median per plate")
    ap.add_argument("--force-plate", action="store_true", help="rebuild cached plates")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--model", default="fisheye", choices=["fisheye", "poly"])
    ap.add_argument("--nk", type=int, default=2)
    ap.add_argument("--ext", default=".jpg", choices=[".jpg", ".png"])
    ap.add_argument("--traceback", action="store_true", help="keep tracebacks in index.json")
    a = ap.parse_args()

    root = os.path.abspath(a.input)
    out = os.path.abspath(a.out) if a.out else os.path.join(
        root if os.path.isdir(root) else os.path.dirname(root), "courts")
    _CFG.update(out=out, out_name=os.path.basename(out), frames=a.frames,
                force=a.force_plate, rounds=a.rounds, model=a.model, nk=a.nk,
                ext=a.ext, traceback=a.traceback)

    inputs = find_inputs(root, a.kind, a.recursive)
    if not inputs:
        raise SystemExit("nothing to do: no %s found under %s" % (a.kind, root))
    os.makedirs(out, exist_ok=True)
    print("%d input(s) -> %s\n" % (len(inputs), out))

    rows = []
    if a.jobs > 1 and len(inputs) > 1:
        with mp.Pool(a.jobs, initializer=_init, initargs=(_CFG,)) as pool:
            it = pool.imap(one, inputs)
            rows = _drain(it, len(inputs))
    else:
        rows = _drain((one(p) for p in inputs), len(inputs))

    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    cols = ["name", "ok", "reason", "line_rms_px", "coverage", "probe_rms_px",
            "max_anchor_px", "ghost_px", "error", "plate"]
    with open(os.path.join(out, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok = [r for r in rows if r.get("ok")]
    print("\n%d accepted, %d refused, %d could not be calibrated at all"
          % (len(ok), len(rows) - len(ok) - sum(1 for r in rows if r.get("error")),
             sum(1 for r in rows if r.get("error"))))
    for r in rows:
        if not r.get("ok"):
            print("  %-58s %s" % (r["name"][-58:], r.get("error") or r.get("reason")))
    print("\nwrote %s and %s" % (os.path.join(out, "index.json"),
                                 os.path.join(out, "index.csv")))
    return 0 if len(ok) == len(rows) else 1


def _init(cfg):
    _CFG.update(cfg)


def _drain(it, n):
    rows = []
    for i, r in enumerate(it):
        rows.append(r)
        print("[%3d/%d] %-7s %-58s %s"
              % (i + 1, n, "ACCEPT" if r.get("ok") else "REJECT", r["name"][-58:],
                 r.get("error") or r.get("reason", "")))
        sys.stdout.flush()
    return rows


if __name__ == "__main__":
    sys.exit(main())
