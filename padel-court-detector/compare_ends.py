#!/usr/bin/env python3
"""Score where each walked line STOPS against truth, in metres on the court.

    python compare_ends.py /path/to/plates
    python compare_ends.py /path/to/plates --ends near_l,near_r --worst 20

Only plates that have a `<stem>_court-truth.json` beside them are taken.

WHAT THIS MEASURES AND WHY IT IS NOT `compare_sides.py`. That harness scores the
FIT - the court a plate ends up drawing - and a fit is a compromise between six
lines and three anchors, so a corner that is 0.2 m out reaches the verdict
diluted by everything else that was right. The endpoints are the one part of
this detector a truth camera can score DIRECTLY: backproject the detected pixel
through the plate's own truth model and read the world coordinate off it, which
is exactly how `ANCHOR_OUT`, `FAR_T_SHORT` and `NEAR_SVC_DROP` were derived.
This is that measurement, standing rather than ad hoc.

THE DETECTOR ONLY IS RUN - `walk.detect`, not `calibrate`. Three reasons, in
order of how much they matter:

    - An endpoint measured through the fit is confounded with the solver. What
      is wanted here is what the WALK produced, before anything reconciled it
      with five other lines.
    - `trace` is filled as the walk proceeds, so a plate whose detector dies
      still reports every endpoint it reached first. Those plates are not a
      rounding error in this population - they are the interesting half of it.
    - 1.3 s a plate against a full calibration's several, so the whole corpus is
      a 20-second loop rather than a coffee break.

SIGN CONVENTION, one for all five endpoints: `err_m` is positive when the walk
went TOO FAR along its own direction of travel and negative when it stopped
SHORT. So the two arms of a line are directly comparable and "overshoot" and
"short" mean the same thing at both ends of the court. `off_m` is the other
axis - across the line rather than along it - and is what `peel_ends` exists to
keep small.

WHAT THE NOMINALS ARE. Every endpoint here is a place PAINT STOPS, not a place
the court model has a corner, and the two differ by a measured constant that the
detector already carries:

    near_l, near_r   the near service line's paint, which runs a little PAST the
                     sideline onto the glass base - `walk.ANCHOR_OUT`
    far_t            the centre line's far end, which stops short of the far
                     service T by the width of the walk's look-ahead -
                     `walk.FAR_T_SHORT`
    far_l, far_r     the far service line's own two ends, from `chord_ends`

The far service ends are scored against the same `ANCHOR_OUT` as the near ones
and are the one place a large POSITIVE `err_m` is not necessarily a fault: that
walk is documented to run past the sideline where the glass base carries on at
the same image height, and is left to, because the overrun backprojects
collinear and a collinear point cannot move a line correspondence. It can move a
CHORD, though - `chord_ends` is what both chord readers draw their strip along -
so the tail is worth watching even where it is not worth fixing.

THE TRUTH MODEL IS NOT PERFECT AND THIS DOES NOT PRETEND IT IS. A click-seeded
camera carries its own error into every number here, so the ABSOLUTE median of a
population is a claim about the detector and the labelling together. Two things
are unaffected and both are printed: the SPREAD within an endpoint, which is
what a better estimator has to shrink, and the RANKING between plates, which is
what the worst-first list is for. The `off_m` column doubles as the floor -
across-line error is ~0 by construction once `peel_ends` has run, so whatever it
reads is roughly what the truth model and the backprojection contribute.

Writes `ends-compare/` under the input: `ends.csv` with every endpoint of every
plate, `ends.json` with the same plus the walk's own stop reason, and a summary
on stdout, worst plate first.
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TRUTH = "_court-truth.json"

#: Where each endpoint lives in the trace, what the world says it should be, and
#: which way "too far" points.
#:
#: `axis` is the coordinate the endpoint MEASURES - X for the three that stop
#: against a sideline, Y for the far T, which stops against the far service
#: line - and the other one is scored as `off_m`. That split is the whole
#: analysis: along-line and across-line error have different causes, different
#: sizes and different fixes, and a single distance in pixels hides both.
#:
#: `outward` is +1 where the walk travels along increasing `axis` and -1 where it
#: travels against it, so `err_m` is "the walk went too far" at all five.
#: Filled in from `walk`'s own constants rather than restated, so a change to
#: `ANCHOR_OUT` or `FAR_T_SHORT` moves the target here too - the alternative is a
#: harness that keeps scoring against a number the detector no longer aims at.
def _ends_table():
    import padelcourt.court as CM
    import padelcourt.walk as W
    return (
        {"key": "near_l", "name": "near_service_end_l", "line": "near service",
         "axis": 0, "outward": -1.0, "arm": "left",
         "nominal": (-W.ANCHOR_OUT, CM.NEAR_SERVICE_Y)},
        {"key": "near_r", "name": "near_service_end_r", "line": "near service",
         "axis": 0, "outward": +1.0, "arm": "right",
         "nominal": (CM.WIDTH + W.ANCHOR_OUT, CM.NEAR_SERVICE_Y)},
        {"key": "far_t", "name": "far_service_t", "line": "centre",
         "axis": 1, "outward": -1.0, "arm": "far",
         "nominal": (CM.CENTRE_X, CM.FAR_SERVICE_Y + W.FAR_T_SHORT)},
        {"key": "far_l", "name": "far_service_end_l", "line": "far service",
         "axis": 0, "outward": -1.0, "arm": "left",
         "nominal": (-W.ANCHOR_OUT, CM.FAR_SERVICE_Y)},
        {"key": "far_r", "name": "far_service_end_r", "line": "far service",
         "axis": 0, "outward": +1.0, "arm": "right",
         "nominal": (CM.WIDTH + W.ANCHOR_OUT, CM.FAR_SERVICE_Y)},
    )


#: An endpoint further than this from where the world says it is counts as OFF,
#: in metres along its own line. Not a tolerance the detector is held to - it is
#: a way of splitting one population into two so the list at the bottom is worth
#: reading. 0.10 m is a little over three times the median error of the near
#: corners and about where their legitimate population stops: the reference
#: models put the paint ends 0.03-0.08 m outside the sideline at the two venues
#: measured, so anything past 0.10 is outside the venue-to-venue spread of the
#: thing being aimed at.
OFF_M = 0.10

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


def short_name(path):
    """`venue/hh-mm-ss` - what the source comments call these plates.

    The filenames carry a venue, a MAC address and an ISO timestamp and are 90
    characters long, which is unreadable in a table and worse in a commit
    message. Every measurement note in `walk.py` names plates by the time alone
    (`13-09-39`, `22-17-42`), so that is what is printed, with the venue in front
    because two venues can record the same second.
    """
    base = os.path.basename(os.path.splitext(path)[0])
    venue = base.split("__")[0][:18]
    times = re.findall(r"(\d{2}-\d{2}-\d{2})", base)
    return "%s/%s" % (venue, times[-1] if times else base[-9:])


def endpoints_of(tr):
    """`{key: (px, note)}` for every endpoint the trace reached.

    `corners` is written near the END of `detect`, after the sidelines, so a
    plate that dies in between would report no near corners at all even though
    it walked them first. It is recomputed from the line itself where that
    happened - the same expression `detect` uses - because those plates are
    exactly the ones this harness exists to look at.
    """
    out = {}
    corners = tr.get("corners")
    svc = np.asarray(tr.get("near_service", np.zeros((0, 2))), float).reshape(-1, 2)
    if corners is None and len(svc):
        corners = (svc[np.argmin(svc[:, 0])], svc[np.argmax(svc[:, 0])])
    why = tr.get("near_why") or {}
    peel = tr.get("peeled") or (None, None)
    if corners is not None:
        out["near_l"] = (np.asarray(corners[0], float),
                         {"why": why.get("left", ""), "peeled": peel[0]})
        out["near_r"] = (np.asarray(corners[1], float),
                         {"why": why.get("right", ""), "peeled": peel[1]})
    if tr.get("far_t") is not None:
        out["far_t"] = (np.asarray(tr["far_t"], float),
                        {"why": tr.get("centre_why", ""),
                         # The far T is the one endpoint the pipeline already
                         # refuses to anchor on its own say-so, so record what it
                         # decided: `anchors` drops it unless the walk reached the
                         # end of the paint, and a chosen line can supply it
                         # instead. An error here means something different
                         # depending on which of the three happened.
                         "anchored": any(a["name"] == "far_service_t"
                                         for a in tr.get("anchors", [])),
                         "from_chooser": bool(tr.get("far_t_from_chooser"))})
    ce = tr.get("chord_ends")
    if ce is not None:
        for key, p in (("far_l", ce[0]), ("far_r", ce[1])):
            out[key] = (np.asarray(p, float), {})
    return out


def score(px, spec, tcam):
    """One endpoint, in metres on the court, split along and across its line."""
    W = tcam.backproject(np.asarray(px, float).reshape(1, 2))[0]
    if not np.isfinite(W).all():
        return None
    ax, off_ax = spec["axis"], 1 - spec["axis"]
    return {"world_x": float(W[0]), "world_y": float(W[1]),
            "err_m": float(spec["outward"] * (W[ax] - spec["nominal"][ax])),
            "off_m": float(W[off_ax] - spec["nominal"][off_ax])}


#: Ways of placing the near service line's corners, as `--variants`.
#:
#:     walk       the walk's own endpoint everywhere - what shipped before the
#:                refiner existed
#:     curve_max  `walk.corner_on_curve` on the near corners, taking the
#:                STRONGEST crossing edge in the window
#:     curve      the same, taking the FIRST qualifying one on the way out -
#:                what ships now
#:     curve_far  `curve`, and the far service line's own two ends read the same
#:                way off the other end of the same sideline
#:
#: The losing rule is kept rather than deleted because it is what makes the
#: winning rule's case, and because a rule that is only described is a rule
#: nobody can re-measure when the corpus changes.
#:
#: Only the near corners differ between them. The other three endpoints are
#: reported under all three and should come back identical, which is a check on
#: the harness as much as on the refiner: a change in `far_t` between variants
#: would mean the refiner had reached something it has no business touching.
VARIANTS = ("walk", "curve_max", "curve", "curve_far")


def one(job):
    """Every endpoint of one plate under one variant. Never raises."""
    plate, truth_path, variant = job
    # Read at CALL time by `walk.corner_refine_wanted`, so setting it here - in
    # the worker, before the import that uses it - is enough, and survives the
    # fork that a pool worker is.
    os.environ["PADEL_CORNER_REFINE"] = {"walk": "none", "curve_far": "near,far"}\
        .get(variant, "near")
    os.environ["PADEL_CORNER_RULE"] = "strongest" if variant == "curve_max" else "first"
    import cv2
    from padelcourt import walk as W
    from padelcourt.court import Camera
    row = {"name": short_name(plate), "plate": os.path.basename(plate),
           "variant": variant}
    img = cv2.imread(plate)
    if img is None:
        return dict(row, failed=True, error="cannot read plate", ends={})
    tr = {}
    try:
        W.detect(img, verbose=False, trace=tr)
        row["detector"] = "ok"
    except BaseException as e:                   # the detector raises SystemExit
        # NOT a lost plate: `trace` is filled as the walk proceeds, so whatever
        # was reached before the failure is still measurable and is still what
        # the detector would have handed on.
        row["detector"] = "died: " + str(e)[:120]
        if _CFG.get("traceback"):
            row["traceback"] = traceback.format_exc()[-1200:]
    with open(truth_path) as fh:
        truth = json.load(fh)
    tcam = Camera.from_dict(truth["camera_model"])
    size = (truth["image_size"]["width"], truth["image_size"]["height"])
    got = endpoints_of(tr)
    ends = {}
    for spec in _ends_table():
        k = spec["key"]
        if k not in got:
            ends[k] = {"present": False}
            continue
        px, note = got[k]
        s = score(px, spec, tcam)
        if s is None:
            ends[k] = {"present": False, "note": "backprojects to infinity"}
            continue
        slid = (tr.get({"near": "corner_refined",
                        "far": "far_ends_refined"}.get(k.split("_")[0], ""))
                or {}).get(spec.get("arm"), {}) if k != "far_t" else {}
        ends[k] = dict(s, present=True, px=[float(px[0]), float(px[1])],
                       slid_px=slid.get("along"), slid_why=slid.get("why"),
                       slid_ratio=slid.get("ratio"),
                       # Whether the walk stopped where the IMAGE ran out rather
                       # than where the paint did - `walk.ANCHOR_MARGIN` is the
                       # same test, and it is the one stop reason that makes an
                       # endpoint mean nothing at all.
                       at_frame=bool(px[0] < 14 or px[1] < 14
                                     or px[0] > size[0] - 14 or px[1] > size[1] - 14),
                       **note)
    row["ends"] = ends
    row["failed"] = False
    return row


def _init(cfg):
    _CFG.update(cfg)


def _stats(vals):
    v = sorted(vals)
    if not v:
        return {}

    def pct(p):
        return v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))]
    return {"n": len(v), "median": pct(50), "p90": pct(90), "max": v[-1]}


def _summarise(rows, off_m):
    """Per endpoint: where it lands, how far it scatters, and which way it fails."""
    out = {}
    for spec in _ends_table():
        k = spec["key"]
        got = [r["ends"][k] for r in rows if r["ends"].get(k, {}).get("present")]
        err = [e["err_m"] for e in got]
        out[k] = {
            "name": spec["name"], "line": spec["line"],
            "nominal": list(spec["nominal"]),
            "n": len(got), "missing": len(rows) - len(got),
            # Signed, because the two failure modes have different causes: a walk
            # that overshoots is following something that is not the line, a walk
            # that stops short lost the line it was following.
            "median_signed": float(np.median(err)) if err else None,
            "mad": float(np.median(np.abs(np.array(err) - np.median(err)))) if err else None,
            "abs": _stats([abs(v) for v in err]),
            "off": _stats([abs(e["off_m"]) for e in got]),
            "long": sum(1 for v in err if v > off_m),
            "short": sum(1 for v in err if v < -off_m),
            "at_frame": sum(1 for e in got if e.get("at_frame")),
        }
    return out


def _off_list(rows, keys, off_m):
    """Every (plate, endpoint) further out than `off_m`, worst first."""
    out = []
    for r in rows:
        for k in keys:
            e = r["ends"].get(k, {})
            if not e.get("present"):
                out.append({"name": r["name"], "end": k, "err_m": None,
                            "off_m": None, "why": "not reached: " + r.get("detector", "")})
                continue
            if abs(e["err_m"]) <= off_m:
                continue
            out.append({"name": r["name"], "end": k, "err_m": e["err_m"],
                        "off_m": e["off_m"], "at_frame": e.get("at_frame"),
                        "peeled": e.get("peeled"),
                        "why": e.get("why", "") or ""})
    return sorted(out, key=lambda r: -(abs(r["err_m"]) if r["err_m"] is not None else 1e9))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="folder of plates, each with a _court-truth.json")
    ap.add_argument("--out", default="", help="output folder (default: <input>/ends-compare)")
    ap.add_argument("--ends", default="", help="only these endpoints, comma separated "
                                               "(near_l, near_r, far_t, far_l, far_r)")
    ap.add_argument("--variants", default="walk",
                    help="how the near corners are placed, comma separated (%s); "
                         "default %%(default)s" % ", ".join(VARIANTS))
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=0, help="only the first N plates")
    ap.add_argument("--off", type=float, default=OFF_M,
                    help="an endpoint further than this along its own line counts "
                         "as off, in metres (default %(default)s)")
    ap.add_argument("--worst", type=int, default=0,
                    help="print only the N worst in each list (default: all)")
    ap.add_argument("--traceback", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(a.input)
    out = os.path.abspath(a.out) if a.out else os.path.join(root, "ends-compare")
    keys = [k.strip() for k in a.ends.split(",") if k.strip()] or \
        [s["key"] for s in _ends_table()]
    known = {s["key"] for s in _ends_table()}
    bad = [k for k in keys if k not in known]
    if bad:
        raise SystemExit("unknown endpoint(s): %s" % ", ".join(bad))

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit("unknown variant(s): %s" % ", ".join(bad))

    pairs = plates_with_truth(root)
    if a.limit:
        pairs = pairs[:a.limit]
    if not pairs:
        raise SystemExit("no plate with a %s under %s" % (TRUTH, root))
    os.makedirs(out, exist_ok=True)
    _CFG.update(traceback=a.traceback)
    jobs = [(p, t, v) for v in variants for p, t in pairs]
    print("%d labelled plate(s) x %d variant(s) -> %s\n"
          % (len(pairs), len(variants), out))

    if a.jobs > 1 and len(jobs) > 1:
        with mp.Pool(a.jobs, initializer=_init, initargs=(_CFG,)) as pool:
            rows = _drain(pool.imap_unordered(one, jobs), len(jobs), keys)
    else:
        _init(_CFG)
        rows = _drain((one(j) for j in jobs), len(jobs), keys)
    rows.sort(key=lambda r: (r["variant"], r["name"]))

    with open(os.path.join(out, "ends.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    _write_csv(os.path.join(out, "ends.csv"), rows)
    summary = {}
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        summary[v] = _summarise(rs, a.off)
        _print_summary(summary[v], rs, keys, a.off, a.worst, v, len(variants))
    if len(variants) > 1:
        _print_delta(summary, variants, keys, rows, a.off)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("\nwrote %s" % out)
    return 0


def _print_delta(summary, variants, keys, rows, off_m):
    """Variant against variant, per endpoint, ON THE TAIL.

    THE MEDIAN DECIDES NOTHING HERE and is printed last on purpose. The world
    position each endpoint is scored against is a measured constant, so a
    variant that shifts the whole population has moved the constant's target
    rather than found the paint - and the constant can simply be re-derived
    afterwards from the variant's own median. Nothing can be re-derived from a
    tail. So the columns that decide are the counts past `off_m` and 1.5x it,
    and the single worst endpoint in the corpus, which is the number a downstream
    consumer of one plate actually meets.

    `fell back` is a robustness column too, and reads the opposite way to the
    others: a refiner that declines and hands back the walk's own endpoint has
    not failed, it has refused to guess. What would be a failure is declining
    NEVER - a guard that never fires is not a guard.
    """
    base = variants[0]
    print("\n%s\nVARIANT AGAINST VARIANT, WORST CASE FIRST - %s is the baseline\n%s"
          % ("=" * 104, base, "=" * 104))
    head = ("%-18s %-10s %8s %8s %9s %9s %9s %10s"
            % ("endpoint", "variant", "off>%.2f" % off_m, "off>%.2f" % (1.5 * off_m),
               "worst", "|p90|", "mad", "fell back"))
    print(head)
    print("-" * len(head))
    for k in keys:
        for v in variants:
            s = summary[v][k]
            got = [r["ends"][k] for r in rows
                   if r["variant"] == v and r["ends"].get(k, {}).get("present")]
            big = sum(1 for e in got if abs(e["err_m"]) > 1.5 * off_m)
            # "never offered" and "offered and declined" are different answers,
            # and the far T is always the first - it is not a candidate for this
            # treatment at all, so a count there would read as 65 refusals.
            offered = [e for e in got if e.get("slid_px") or e.get("slid_why")]
            back = "%d/%d" % (sum(1 for e in offered if not e.get("slid_px")),
                              len(offered)) if offered else "n/a"
            print("%-18s %-10s %8s %8s %9s %9s %9s %10s"
                  % (s["name"] if v == base else "", v,
                     "%d/%d" % (s["long"] + s["short"], s["n"]),
                     "%d/%d" % (big, s["n"]),
                     _fmt(s["abs"].get("max")), _fmt(s["abs"].get("p90")),
                     _fmt(s["mad"]), "-" if v == base else back))
    print("\nfell back = the corner the refiner declined to move, so the walk's own "
          "endpoint\nstands. The median each variant would re-derive its world "
          "constant from:")
    for k in keys:
        print("  %-18s %s" % (summary[base][k]["name"],
                              "   ".join("%s %s" % (v, _fmt(summary[v][k]["median_signed"]))
                                         for v in variants)))


def _drain(it, n, keys):
    rows = []
    for i, r in enumerate(it):
        rows.append(r)
        cells = []
        for k in keys:
            e = r["ends"].get(k, {})
            cells.append("%s %s" % (k, "%+.2f" % e["err_m"] if e.get("present") else "  -  "))
        print("[%3d/%d] %-6s %-28s %s%s"
              % (i + 1, n, r.get("variant", ""), r["name"][-28:], "  ".join(cells),
                 "" if r.get("detector") == "ok" else "   <- " + r.get("detector", "")[:60]))
        sys.stdout.flush()
    return rows


def _write_csv(path, rows):
    cols = ["name", "end", "err_m", "off_m", "world_x", "world_y", "px_x", "px_y",
            "at_frame", "peeled", "why", "detector"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            for spec in _ends_table():
                e = r["ends"].get(spec["key"], {})
                if not e.get("present"):
                    w.writerow({"name": r["name"], "end": spec["key"],
                                "detector": r.get("detector", ""), "why": "not reached"})
                    continue
                w.writerow({"name": r["name"], "end": spec["key"],
                            "err_m": "%.4f" % e["err_m"], "off_m": "%.4f" % e["off_m"],
                            "world_x": "%.4f" % e["world_x"],
                            "world_y": "%.4f" % e["world_y"],
                            "px_x": "%.1f" % e["px"][0], "px_y": "%.1f" % e["px"][1],
                            "at_frame": e.get("at_frame"), "peeled": e.get("peeled"),
                            "why": e.get("why", ""), "detector": r.get("detector", "")})


def _print_summary(summary, rows, keys, off_m, worst, variant="", n_var=1):
    n = len(rows)
    print("\n%s\nWHERE THE WALKS STOP, %d labelled plates%s"
          % ("=" * 100, n, ("   [corners: %s]" % variant) if n_var > 1 else ""))
    print("err_m is signed ALONG the line: + the walk went too far, - it stopped short.")
    print("off_m is ACROSS the line, which is ~0 by construction and is therefore\n"
          "roughly what the truth model and the backprojection contribute.\n%s" % ("=" * 100))
    head = ("%-18s %6s %8s %8s %7s %7s %7s %7s %6s %6s"
            % ("endpoint", "n", "median", "mad", "|p50|", "|p90|", "|max|",
               "off p90", "long", "short"))
    print(head)
    print("-" * len(head))
    for k in keys:
        s = summary[k]
        print("%-18s %6s %8s %8s %7s %7s %7s %7s %6d %6d"
              % (s["name"], "%d/%d" % (s["n"], n),
                 _fmt(s["median_signed"]), _fmt(s["mad"]),
                 _fmt(s["abs"].get("median")), _fmt(s["abs"].get("p90")),
                 _fmt(s["abs"].get("max")), _fmt(s["off"].get("p90")),
                 s["long"], s["short"]))
    print("\nlong / short count the endpoints more than %.2f m past / before where the\n"
          "world says the paint stops. `median` is the systematic part - a constant\n"
          "there is a constant to re-derive, not an error - and `mad` is the scatter,\n"
          "which is what a better estimator has to shrink." % off_m)

    for k in keys:
        bad = _off_list(rows, [k], off_m)
        shown = bad if not worst else bad[:worst]
        print("\n%s\n%s - %d of %d plates off by more than %.2f m\n%s"
              % ("-" * 100, summary[k]["name"], len(bad), n, off_m, "-" * 100))
        if not bad:
            print("  none")
            continue
        for r in shown:
            print("  %-26s %8s  across %7s  %s%s"
                  % (r["name"][-26:],
                     "-" if r["err_m"] is None else "%+.3f m" % r["err_m"],
                     "-" if r["off_m"] is None else "%+.3f" % r["off_m"],
                     "AT FRAME EDGE  " if r.get("at_frame") else "",
                     (r["why"] or "")[:44]))
        if worst and len(bad) > worst:
            print("  ... %d more" % (len(bad) - worst))


def _fmt(x, nd=3):
    return "-" if x is None else "%+.*f" % (nd, x)


if __name__ == "__main__":
    sys.exit(main())
