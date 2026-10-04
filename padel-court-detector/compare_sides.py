#!/usr/bin/env python3
"""Run the detector once per courtside technique and score each against truth.

    python compare_sides.py /path/to/plates
    python compare_sides.py /path/to/plates --variants auto,canny,chord --jobs 8

Only plates that have a `<stem>_court-truth.json` beside them are taken, because
the whole point is the comparison.

A FIT IS ACCEPTED WHEN NO VISIBLE PART OF THE COURT IT DRAWS IS MORE THAN 0.3 m
FROM THE REAL ONE - 0.6 m beyond the far service line, where the detector
measures nothing and the fit extrapolates; see `COURT_FAR_TOL_M`. The court is
sampled on a grid, each sample is projected through the fit and read back onto
the ground through the truth, and the worst of those distances in each band is
the plate's score - see `court_error_m`. Visible means the TRUTH puts it in the
frame, so the near corners, which sit outside the picture on a typical mount,
are not held against anything. The keypoint error in pixels is still measured
and reported, but it decides nothing.

Four variants are ways of CHOOSING a sideline reader and five hold the detector
to one reader, which is how the choosing was measured:

    auto        the shipped rule, whichever it currently is
    cascade     first reader past the corner test, each side alone
    graded      corner test as a veto, then rank on `provisional_world`
    fit         corner test as a veto, then a provisional camera per surviving
                reader, ranked on what each makes of its own samples
    canny       first strong edge out from centre
    track       boundary track from the corner
    scans       row scans
    chord_lines the first edge out that belongs to a line along the chord
    chord       the same chord read as a step in a blurred cross-profile

A variant that is refused on a side is not a failure of the harness - it is the
answer to "what is this technique alone worth", and it is counted as such.

Writes `sides-compare/` under the input: one JSON per variant with every plate's
row, `compare.csv` with all of them, and a summary on stdout.

ALSO WRITES `<plate>_courtside-detection.json` BESIDE EACH PLATE - every reader's
points, whether it qualified, and the fit through them, which is what the label
tool overlays. One file per plate, so it comes from one variant's run: the first
CHOOSER in `--variants`, because a run held to a single reader has nothing to say
about the other four. `--no-write-detections` turns it off.

It is nearly free under the shipped selector. Recording every reader is what
`walk.SIDE_RECORD_ALL` costs a second a plate for, but `SIDE_SELECT` is `fit`,
which has to read them all to rank them - so on a default run they are read
either way and only the writing is added.
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TRUTH = "_court-truth.json"
#: `auto` is the shipped selection rule, whatever it currently is; `cascade`,
#: `graded` and `fit` pin the three that exist, so a change to the chooser can be
#: measured against what it replaced rather than against a memory of it. `auto`
#: duplicates whichever of them `walk.SIDE_SELECT` names - `fit`, currently - and
#: that duplication is the point: it is what catches the default drifting away
#: from the rule someone last measured.
VARIANTS = ("auto", "cascade", "graded", "fit",
            "canny", "track", "scans", "chord_lines", "chord")

#: The variants that name a single reader rather than a way of choosing between
#: them. Kept in `walk.SIDE_SOURCE_KEYS`' order, and validated against it there:
#: a key here that walk does not know raises rather than silently running the
#: unrestricted cascade under a reader's name.
READERS = ("canny", "track", "scans", "chord_lines", "chord")

_CFG = {}


def plates_with_truth(root):
    """`(plate.png, truth.json)` for every labelled plate in `root`, sorted."""
    out = []
    for t in sorted(glob.glob(os.path.join(root, "*" + TRUTH))):
        stem = t[:-len(TRUTH)]
        for ext in (".png", ".jpg", ".jpeg"):
            if os.path.exists(stem + ext):
                out.append((stem + ext, t))
                break
    return out


#: A fit is accepted when no VISIBLE part of the court it draws lands further
#: than this from the real one, in metres on the ground. Metres rather than
#: pixels because a pixel is worth several times as much court at the net as it
#: is at the near baseline - the p90 pixel scatter of the chord's far half beats
#: its near half and is twice as bad in metres, which is the whole argument in
#: `walk._chord_read`. The court is what has to be right, so the court is the
#: thing measured.
COURT_TOL_M = 0.3

#: ...except beyond the far service line, where it is this instead. Not a
#: softening: it is where the evidence runs out and everyone knows it. Seen from
#: behind the near baseline the far strip is almost edge-on, the glass frame
#: hides the surface in front of it, and `CourtModel.OCCLUDED_BY_DEFAULT` already
#: holds out exactly this segment - Y in [0, 3.05] - for the far baseline and
#: both far sidelines, so the detector is not measuring there and the fit is
#: extrapolating.
#:
#: The corpus says the same thing without being asked: on 57 of 65 labelled
#: plates the WORST visible point of the fit is at Y = 0.0, on one of the two far
#: corners, and the error falls away monotonically towards the camera - a mean
#: of 0.45 m in this band against 0.15 m over the near baseline.
#:
#: 0.60 m rather than 0.50 is headroom, not a fitted number: sweeping the far
#: tolerance the accepted count goes 47 (at 0.30), 53 (0.40), 58 (0.50) and then
#: STOPS - 58 at 0.60, 0.70, 0.80 and 1.00 m alike, and equal to judging the far
#: band not at all. Past 0.5 m nothing in this corpus is decided here any more,
#: so 0.6 sits in flat ground rather than on a slope.
COURT_FAR_TOL_M = 0.6

#: Where the near tolerance stops and the far one starts, in metres from the far
#: baseline. The far service line, because that is the boundary the court model
#: already draws for the same reason.
COURT_FAR_Y_M = 3.05

#: Where the court is sampled, in metres. 0.5 m over 10x20 m is 861 points, and
#: the error surface is a smooth projective one - the maximum does not hide
#: between samples.
COURT_STEP_M = 0.5


def court_grid(step=COURT_STEP_M):
    """Every sample point on the court, in metres."""
    import padelcourt.court as CM
    xs = np.arange(0.0, CM.WIDTH + 1e-9, step)
    ys = np.arange(0.0, CM.LENGTH + 1e-9, step)
    return np.array([[x, y] for y in ys for x in xs], float)


def court_error_m(cam, truth_cam, size, step=COURT_STEP_M):
    """How far the fitted court lands from the real one, in metres.

    For each sample the fit projects to a pixel and that pixel is read back
    through the TRUTH camera onto the court plane: the answer is where the fit
    says a point is, measured on the real court, which is the error a downstream
    consumer of the homography actually suffers.

    ONLY THE VISIBLE PART COUNTS, and visible means the TRUTH puts it in the
    frame - the real court is what is or is not in shot. Visibility judged by
    the fit instead would let a fit that pushed the near baseline off the bottom
    of the picture excuse itself from the rows it got most wrong. Points the fit
    sends off frame are kept and scored, because that is an error, not an
    absence of evidence.

    THE COURT IS JUDGED IN TWO BANDS - see `COURT_FAR_TOL_M` - so the worst
    point is reported for each, and a plate passes only if both are inside their
    own tolerance.

    Returns a dict, all-None when nothing is visible.
    """
    W = court_grid(step)
    pt = truth_cam.project(W)
    vis = ((pt[:, 0] >= 0) & (pt[:, 0] < size[0])
           & (pt[:, 1] >= 0) & (pt[:, 1] < size[1]) & np.isfinite(pt).all(1))
    if not vis.any():
        return {"max": None, "median": None, "p95": None, "n_visible": 0,
                "near_max": None, "far_max": None}
    Wv = W[vis]
    back = truth_cam.backproject(cam.project(Wv))
    e = np.linalg.norm(back - Wv, axis=1)
    e = np.where(np.isfinite(e), e, 1e6)
    far = Wv[:, 1] < COURT_FAR_Y_M
    return {"max": float(e.max()), "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)), "n_visible": int(vis.sum()),
            "near_max": (float(e[~far].max()) if (~far).any() else None),
            "far_max": (float(e[far].max()) if far.any() else None)}


def keypoint_errors(got, want):
    """Per-name pixel distance between two keypoint lists, as `{name: px}`."""
    a = {k["name"]: k["image_px"] for k in got}
    b = {k["name"]: k["image_px"] for k in want}
    return {n: float(((a[n][0] - b[n][0]) ** 2 + (a[n][1] - b[n][1]) ** 2) ** 0.5)
            for n in a if n in b}


def _stats(vals):
    v = sorted(vals)
    if not v:
        return {}

    def pct(p):
        return v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))]
    return {"n": len(v), "median": pct(50), "p95": pct(95), "max": v[-1],
            "mean": sum(v) / len(v)}


def one(job):
    """One plate under one variant. Never raises."""
    plate, truth_path, variant = job
    # Read at CALL time by `walk.side_sources_wanted`, so setting it here - in
    # the worker, before the import that uses it - is enough, and survives the
    # fork that a pool worker is.
    os.environ["PADEL_SIDE_SOURCES"] = variant if variant in READERS else ""
    os.environ["PADEL_SIDE_SELECT"] = \
        variant if variant in ("cascade", "graded", "fit") else ""
    # Only the `auto` run reads every technique and writes the file: it is the
    # one whose cascade is unrestricted, so its four answers are the four this
    # plate actually has. A forced run would write the same points minus the
    # readers it was not allowed to use.
    write_sides = _CFG.get("write_sides") and variant == _CFG.get("sides_from")
    os.environ["PADEL_SIDE_RECORD_ALL"] = "1" if write_sides else ""
    from padelcourt import calibrate_image_file, courtside_detection_path
    from padelcourt import save_courtside_detection
    row = {"name": os.path.basename(plate), "variant": variant}
    try:
        res = calibrate_image_file(plate, rounds=_CFG["rounds"],
                                   model=_CFG["model"], nk=_CFG["nk"])
    except BaseException as e:
        row.update(ok=False, failed=True, error=str(e)[:200])
        if _CFG.get("traceback"):
            row["traceback"] = traceback.format_exc()[-1500:]
        return row

    if write_sides:
        row["sides_json"] = save_courtside_detection(
            res, courtside_detection_path(plate))
    d = res.to_dict()
    with open(truth_path) as fh:
        truth = json.load(fh)
    err = keypoint_errors(d["keypoints"], truth["keypoints"])
    how = res.trace.get("side_how", {})
    from padelcourt.court import Camera
    tcam = Camera.from_dict(truth["camera_model"])
    size = (truth["image_size"]["width"], truth["image_size"]["height"])
    q = court_error_m(res.camera, tcam, size, _CFG["step"])
    near, far = q["near_max"], q["far_max"]
    # THE VERDICT IS THE TOLERANCE RULE, not the pipeline's own gates. The gates
    # are what the detector can check without a truth file - coverage, rms,
    # probe - and they are kept beside it as `pipeline_ok` because the two
    # disagreeing on a plate is worth seeing: a gate that refuses a court which
    # is in fact within tolerance is a gate costing good fits.
    row.update(court_max_m=q["max"], court_median_m=q["median"],
               court_p95_m=q["p95"], n_visible=q["n_visible"],
               near_max_m=near, far_max_m=far,
               ok=((near is None or near <= _CFG["tol"])
                   and (far is None or far <= _CFG["far_tol"])
                   and q["max"] is not None),
               # Kept so the corpus can be re-scored against a different rule
               # without running the detector over it again.
               camera_model=res.camera.to_dict())
    row.update(pipeline_ok=bool(res.ok), failed=False, reason=res.reason,
               line_rms_px=d["quality"]["line_rms_px"],
               coverage=d["quality"]["paint_coverage"],
               probe_rms_px=d["quality"]["probe_rms_px"],
               max_anchor_px=d["quality"]["max_anchor_px"],
               kp=err, kp_median=_stats(err.values()).get("median"),
               kp_max=_stats(err.values()).get("max"),
               left_how=how.get("left_sideline", ""),
               right_how=how.get("right_sideline", ""),
               left_refused=how.get("left_sideline", "").startswith("REFUSED"),
               right_refused=how.get("right_sideline", "").startswith("REFUSED"))
    return row


def _init(cfg):
    _CFG.update(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder of plates, each with a _court-truth.json")
    ap.add_argument("--out", default="", help="output folder (default: <input>/sides-compare)")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="which techniques to run, comma separated (%s)" % ", ".join(VARIANTS))
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=0, help="only the first N plates")
    ap.add_argument("--no-write-detections", dest="write_detections",
                    action="store_false",
                    help="do NOT write <plate>_courtside-detection.json beside each "
                         "plate; by default one is written, holding every "
                         "technique's points and the fit through them for the label "
                         "tool to overlay")
    ap.add_argument("--tol", type=float, default=COURT_TOL_M,
                    help="a fit is accepted when no visible part of the court it "
                         "draws is further than this from the real one, in metres "
                         "(default %(default)s)")
    ap.add_argument("--far-tol", type=float, default=COURT_FAR_TOL_M,
                    help="the tolerance beyond the far service line (Y < %.2f m), "
                         "where the detector has no evidence and the fit "
                         "extrapolates (default %%(default)s)" % COURT_FAR_Y_M)
    ap.add_argument("--step", type=float, default=COURT_STEP_M,
                    help="court sampling pitch in metres (default %(default)s)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--model", default="fisheye", choices=["fisheye", "poly"])
    ap.add_argument("--nk", type=int, default=2)
    ap.add_argument("--traceback", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(a.input)
    out = os.path.abspath(a.out) if a.out else os.path.join(root, "sides-compare")
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit("unknown variant(s): %s" % ", ".join(bad))
    # ONE FILE PER PLATE, so exactly one variant may write it, and it should be
    # a variant that was allowed to read everything. A run held to a single
    # reader records that reader and calls the other four "not offered", which
    # is a true statement about that run and a misleading file to leave beside a
    # plate. So the first CHOOSER asked for wins, in the order they were asked
    # for, and only a run with no chooser at all falls back to a reader.
    sides_from = next((v for v in variants if v not in READERS), variants[0])
    _CFG.update(rounds=a.rounds, model=a.model, nk=a.nk, traceback=a.traceback,
                tol=a.tol, far_tol=a.far_tol, step=a.step,
                write_sides=a.write_detections, sides_from=sides_from)

    pairs = plates_with_truth(root)
    if a.limit:
        pairs = pairs[:a.limit]
    if not pairs:
        raise SystemExit("no plate with a %s under %s" % (TRUTH, root))
    os.makedirs(out, exist_ok=True)
    jobs = [(p, t, v) for v in variants for p, t in pairs]
    print("%d plate(s) x %d variant(s) = %d run(s) -> %s"
          % (len(pairs), len(variants), len(jobs), out))
    if a.write_detections:
        # Spelled out rather than imported from `padelcourt.api`: the workers
        # set their environment BEFORE importing padelcourt, and importing it
        # here would have them inherit it already loaded from the fork.
        print("courtside points from the `%s` run -> <plate>%s beside each plate"
              % (sides_from, "_courtside-detection.json"))
        if sides_from in READERS:
            print("  NOTE: no chooser variant was asked for, so that run is held to"
                  " one reader\n        and the file will call the others \"not"
                  " offered\". Add `auto` to get all of them.")
    print()

    rows = []
    if a.jobs > 1 and len(jobs) > 1:
        with mp.Pool(a.jobs, initializer=_init, initargs=(_CFG,)) as pool:
            it = pool.imap_unordered(one, jobs)
            rows = _drain(it, len(jobs))
    else:
        _init(_CFG)
        rows = _drain((one(j) for j in jobs), len(jobs))

    by_variant = {v: [r for r in rows if r["variant"] == v] for v in variants}
    for v, rs in by_variant.items():
        with open(os.path.join(out, "%s.json" % v), "w") as fh:
            json.dump(sorted(rs, key=lambda r: r["name"]), fh, indent=1, default=str)
    _write_csv(os.path.join(out, "compare.csv"), rows, variants, pairs)
    summary = _summarise(by_variant, variants, len(pairs), a.tol, a.far_tol)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    _write_refused_csv(os.path.join(out, "refused.csv"), summary, variants)
    _print_summary(summary, by_variant, variants, pairs, a.tol, a.far_tol)
    print("\nwrote %s" % out)
    return 0


def _drain(it, n):
    rows = []
    for i, r in enumerate(it):
        rows.append(r)
        m = r.get("court_max_m")
        print("[%3d/%d] %-11s %-7s %-50s %s"
              % (i + 1, n, r["variant"], "ACCEPT" if r.get("ok") else "REJECT",
                 r["name"][-50:],
                 ("near %5.2f m  far %5.2f m  median %5.2f m"
                  % (r["near_max_m"] or 0, r["far_max_m"] or 0, r["court_median_m"]))
                 if m is not None else (r.get("error") or "no fit")))
        sys.stdout.flush()
    return rows


def _write_csv(path, rows, variants, pairs):
    cols = ["name", "variant", "ok", "near_max_m", "far_max_m", "court_max_m",
            "court_median_m", "court_p95_m",
            "n_visible", "pipeline_ok", "reason", "kp_median", "kp_max",
            "line_rms_px", "coverage", "probe_rms_px", "max_anchor_px",
            "left_how", "right_how", "error"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        order = {v: i for i, v in enumerate(variants)}
        for r in sorted(rows, key=lambda r: (r["name"], order[r["variant"]])):
            w.writerow(r)


def _summarise(by_variant, variants, n_plates, tol=COURT_TOL_M,
               far_tol=COURT_FAR_TOL_M):
    out = {}
    for v in variants:
        rs = by_variant[v]
        fit = [r for r in rs if not r.get("failed")]
        med = [r["kp_median"] for r in fit if r.get("kp_median") is not None]
        cmax = [r["court_max_m"] for r in fit if r.get("court_max_m") is not None]
        cmed = [r["court_median_m"] for r in fit if r.get("court_median_m") is not None]
        near = [r["near_max_m"] for r in fit if r.get("near_max_m") is not None]
        far = [r["far_max_m"] for r in fit if r.get("far_max_m") is not None]
        all_kp = [e for r in fit for e in (r.get("kp") or {}).values()]
        out[v] = {
            "plates": len(rs),
            "no_fit": sum(1 for r in rs if r.get("failed")),
            "accepted": sum(1 for r in rs if r.get("ok")),
            "refused": _refused(rs, tol, far_tol),
            "pipeline_accepted": sum(1 for r in rs if r.get("pipeline_ok")),
            # Which band each variant is actually losing plates on.
            "near_fails": sum(1 for r in fit if (r.get("near_max_m") or 0) > tol),
            "far_fails": sum(1 for r in fit if (r.get("far_max_m") or 0) > far_tol),
            "sides_refused": sum(bool(r.get("left_refused")) + bool(r.get("right_refused"))
                                 for r in fit),
            "sides_total": 2 * len(fit),
            "court_max_m": _stats(cmax),
            "court_median_m": _stats(cmed),
            "near_max_m": _stats(near),
            "far_max_m": _stats(far),
            "within_010m": sum(1 for m in cmax if m <= 0.10),
            "within_020m": sum(1 for m in cmax if m <= 0.20),
            "worse_than_1m": sum(1 for m in cmax if m > 1.0),
            "plate_median_px": _stats(med),
            "keypoint_px": _stats(all_kp),
        }
    return out


def _refused(rows, tol, far_tol):
    """Every plate this variant did not get within `tol`, worst first.

    A plate that could not be fitted at all is refused too, and says so: "no
    fit" and "fitted, but 2 m out" are both failures to produce a court, and a
    list that only had the second would flatter the technique that dies early.
    """
    out = []
    for r in rows:
        if r.get("ok"):
            continue
        if r.get("failed"):
            why = "no fit: " + (r.get("error") or "?")
        else:
            bad = []
            if (r.get("near_max_m") or 0) > tol:
                bad.append("near %.2f m (>%.2f)" % (r["near_max_m"], tol))
            if (r.get("far_max_m") or 0) > far_tol:
                bad.append("far %.2f m (>%.2f)" % (r["far_max_m"], far_tol))
            why = ", ".join(bad) or "?"
        out.append({"name": r["name"],
                    "court_max_m": r.get("court_max_m"),
                    "near_max_m": r.get("near_max_m"),
                    "far_max_m": r.get("far_max_m"),
                    "court_median_m": r.get("court_median_m"),
                    "why": why,
                    "sides": "%s | %s" % (r.get("left_how", ""), r.get("right_how", "")),
                    "pipeline": "accepted" if r.get("pipeline_ok")
                                else (r.get("reason") or "no fit")})
    return sorted(out, key=lambda r: (r["court_max_m"] is not None,
                                      -(r["court_max_m"] or 0)))


def _print_summary(summary, by_variant, variants, pairs, tol, far_tol):
    n = len(pairs)
    print("\n%s\nCOURTSIDE TECHNIQUES, %d labelled plates" % ("=" * 104, n))
    print("ACCEPTED = the fitted court is within %.2f m of the real one over every "
          "visible point," % tol)
    print("           and within %.2f m beyond the far service line (Y < %.2f m), "
          "where nothing is measured\n%s" % (far_tol, COURT_FAR_Y_M, "=" * 104))
    head = ("%-11s %9s %7s %9s %9s %9s %7s %7s %8s"
            % ("variant", "accept", "no fit", "near m", "far m", "med m",
               "near-x", "far-x", "sides-x"))
    print(head)
    print("-" * len(head))
    for v in variants:
        s = summary[v]
        print("%-11s %6d/%-3d %7d %9s %9s %9s %7d %7d %4d/%-4d"
              % (v, s["accepted"], s["plates"], s["no_fit"],
                 _fmt(s["near_max_m"].get("median"), 3),
                 _fmt(s["far_max_m"].get("median"), 3),
                 _fmt(s["court_median_m"].get("median"), 3),
                 s["near_fails"], s["far_fails"],
                 s["sides_refused"], s["sides_total"]))
    print("\nnear m / far m are the per-plate WORST visible point in each band, and what")
    print("is printed is the MEDIAN of those over the plates that fitted. near-x and")
    print("far-x count the plates each band refused; a plate can fail both.")
    print("sides-x counts the sidelines the technique refused, out of the ones asked for.")
    print("\nThe pipeline's own gates, for comparison - coverage, rms and probe, which are")
    print("what it can check WITHOUT a truth file:")
    for v in variants:
        s = summary[v]
        print("  %-11s gates accept %2d/%-3d   %.2f m rule accepts %2d/%-3d"
              % (v, s["pipeline_accepted"], s["plates"], tol,
                 s["accepted"], s["plates"]))

    for v in variants:
        ref = summary[v]["refused"]
        print("\n%s\n%s - %d plate(s) refused\n%s" % ("-" * 104, v, len(ref), "-" * 104))
        if not ref:
            print("  none")
            continue
        for r in ref:
            print("  %-62s %-34s %s"
                  % (r["name"][-62:], r["why"][:34],
                     "gates: " + r["pipeline"][:36]))


def _write_refused_csv(path, summary, variants):
    """Every refusal, one row per (variant, plate) - the list, as a file."""
    cols = ["variant", "name", "near_max_m", "far_max_m", "court_max_m", "why", "sides",
            "pipeline"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for v in variants:
            for r in summary[v]["refused"]:
                w.writerow(dict(r, variant=v))


def _fmt(x, nd=2):
    return "-" if x is None else "%.*f" % (nd, x)


if __name__ == "__main__":
    sys.exit(main())
