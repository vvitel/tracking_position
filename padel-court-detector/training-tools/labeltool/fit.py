"""Marked sides + walked paint lines -> a Camera, and the truth document.

The stages are `solve`'s, and for its reasons: the lens is fitted first by
straightness alone, then the homography by DLT on line correspondences, then the
two are refined together. What is different is only who is believed.

WHAT THE MARKS ARE WORTH. A clicked point on the left sideline knows one thing
about the world - that it lies on X = 0 - so it enters as a line sample, which is
the only correspondence a point on a line can carry. Six of them against the near
service line's four hundred is where a plain least squares goes wrong: the loss
would happily trade seventy pixels of labelled boundary for one pixel of paint,
and the labelled boundary is the entire reason this tool exists. So every line
carries the same TOTAL weight regardless of how many samples it has - each is one
correspondence and is treated as one - which is what `_line_weight` does.

That weighting cannot coexist with a Huber threshold, because Huber measures the
residual it is handed and a six-point line scaled up by 5.8 would have its knee
at half a pixel. The fit is therefore in two passes: one robust and unweighted,
purely to catch a walked line that is not the line it claims to be, and one
weighted and plain over whatever survived. Human lines are never screened out -
they are the ground truth, and a mark the labeller can see on screen is not an
outlier to be voted down.

THE TWO FAR CORNERS ARE REAL ANCHORS. Unlike every other mark they have both
world coordinates - (0, 0) and (10, 0) - so they go in as point correspondences,
alongside the walk's own three. They are taken as the intersection of the fitted
side curves rather than as whichever click was outermost, because that is the
corner the labeller was drawing towards and it uses all the marks near it.
"""
import os

import numpy as np

from padelcourt import api
from padelcourt import court as CM
from padelcourt.calibrate import (ANCHOR_GATE, COVERAGE_MIN, PROBE_MAX, RMS_MAX,
                                  paint_probe)
from padelcourt.court import Camera, perp, unit
from padelcourt.solve import (_in_front, _line_residuals, _orientation_ok,
                              _point_resid, fit_distortion,
                              homography_from_lines, redundancy_check)

#: The three boundaries a person marks, and the world line each one is.
#: `swap` is the fit orientation, as everywhere else in this codebase: the
#: sidelines are x(y), the far baseline is y(x).
SIDES = {
    "left":  {"line": "left_sideline",  "world": ((0.0, 0.0), (0.0, CM.LENGTH)),
              "swap": True,  "label": "Left side (X = 0)"},
    "right": {"line": "right_sideline", "world": ((CM.WIDTH, 0.0), (CM.WIDTH, CM.LENGTH)),
              "swap": True,  "label": "Right side (X = 10)"},
    "far":   {"line": "far_baseline",   "world": ((0.0, 0.0), (CM.WIDTH, 0.0)),
              "swap": False, "label": "Far side (Y = 0)"},
}
SIDE_ORDER = ("left", "right", "far")

#: Marks per side before a court can be computed. Three fix a quadratic; the
#: extra two are what tell you the quadratic is the right one.
MIN_MARKS = 5

#: Marks per side before the curve is drawn at all - a quadratic through three
#: points passes through all three and says nothing.
MIN_CURVE = 3

#: Which of the walk's lines the labelled fit takes. The two sidelines are not
#: among them: the human has just marked those, and they are the ones the walk
#: measures worst. `net_line` is excluded for the reason `solve.CHECK_ONLY`
#: gives - it is a shadow-and-mesh boundary rather than paint, uncertain at the
#: few-centimetre level, and fitted at full weight it drags the solution ~5x.
FROM_WALK = ("near_service", "centre_line", "far_service")

#: The count every line is normalised to, so that "each line is one
#: correspondence" has a scale to be expressed in. Any value gives the same
#: relative weighting; this one is roughly a walked line's length, so the
#: reported residuals stay in the same range as the detector's.
N_REF = 200.0

#: How well a corner the labeller drew is believed, in pixels. Larger than the
#: ~1 px they can place it to at zoom, because the corner reported here is the
#: intersection of two fitted quadratics and inherits both curves' error; and
#: smaller than the walk's own 3 px near-service anchors (`solve.ANCHOR_SIGMA`),
#: which are the end of a painted stripe that overshoots the sideline by a
#: centimetre or two and has to be corrected for. A person clicking a corner is
#: clicking the corner.
CORNER_SIGMA = 2.0

#: How far past the marked span the corner intersection may be found. A corner
#: is meant to BE one of the marks, so an intersection well outside the marked
#: range means the two curves are not meeting where the labeller thought.
CORNER_REACH = 40.0


class FitError(Exception):
    """The marks and the walk together do not make a court, and why."""


# --------------------------------------------------------------- the sides --
def fit_curve(pts, swap):
    """Quadratic through the marks: x(y) for a sideline, y(x) for the far one.

    Returns (coeffs, u_lo, u_hi) in the parameter the fit is along, or None when
    there are too few marks or they are stacked on one value of it.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < MIN_CURVE:
        return None
    u = p[:, 1] if swap else p[:, 0]
    v = p[:, 0] if swap else p[:, 1]
    if len(np.unique(np.round(u, 3))) < MIN_CURVE:
        return None
    try:
        c = np.polyfit(u, v, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.isfinite(c).all():
        return None
    return c, float(u.min()), float(u.max())


def curve_points(curve, swap, n=120):
    """The fitted curve as an image polyline, for drawing."""
    c, lo, hi = curve
    u = np.linspace(lo, hi, n)
    v = np.polyval(c, u)
    return np.column_stack([v, u] if swap else [u, v])


def curve_rms(pts, curve, swap):
    """How far the marks sit off their own curve, in pixels.

    Shown to the labeller. Three marks give zero by construction; from the fifth
    on it is the number that says whether the side really is a smooth curve or
    whether one click landed somewhere else.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    c = curve[0]
    u = p[:, 1] if swap else p[:, 0]
    v = p[:, 0] if swap else p[:, 1]
    return float(np.sqrt(np.mean((v - np.polyval(c, u)) ** 2)))


def far_corner(side_curve, far_curve, y_start):
    """Where a sideline's curve meets the far baseline's, in pixels.

    x = S(y) and y = F(x) compose to y = F(S(y)), and near a corner the sideline
    is steep while the far baseline is shallow, so the composition's derivative
    is the product of two small numbers and the plain iteration converges in a
    handful of steps. Returns None if it wanders outside the marked spans, which
    is the case where the two curves do not actually meet where they were drawn.
    """
    cs, y_lo, y_hi = side_curve
    cf, x_lo, x_hi = far_curve
    y = float(y_start)
    for _ in range(60):
        x = float(np.polyval(cs, y))
        if not (x_lo - CORNER_REACH <= x <= x_hi + CORNER_REACH):
            return None
        y_new = float(np.polyval(cf, x))
        if not (y_lo - CORNER_REACH <= y_new <= y_hi + CORNER_REACH):
            return None
        if abs(y_new - y) < 1e-4:
            return np.array([float(np.polyval(cs, y_new)), y_new])
        y = y_new
    return None


def corners(curves):
    """The two far corners, as {name: (u, v)}, from whichever curves exist.

    The far corner of a sideline is its end nearest the far baseline, which on
    every plate this tool will ever see is the smaller image y - the camera is
    behind the near baseline looking up the court.
    """
    out = {}
    far = curves.get("far")
    if far is None:
        return out
    for side, name in (("left", "far_corner_l"), ("right", "far_corner_r")):
        c = curves.get(side)
        if c is None:
            continue
        p = far_corner(c, far, c[1])            # c[1] is the side's smallest y
        if p is not None:
            out[name] = p
    return out


# ------------------------------------------------------------ the residual --
def _line_weight(n):
    """Scale a line's residuals so every line carries the same total weight.

    Sum of squares over n samples scaled by sqrt(N_REF / n) is N_REF times the
    MEAN square, whatever n is - so a six-mark boundary and a four-hundred-sample
    painted line arrive at the loss as one correspondence each, which is what
    they are.
    """
    return float(np.sqrt(N_REF / max(int(n), 1)))


def _residual(lines, points, template, weighted):
    """The function `CM.lm` minimises: perpendicular line error, then anchors."""
    w = ([_line_weight(len(L["pts"])) for L in lines] if weighted
         else [1.0] * len(lines))

    def resid(p):
        c = CM.unpack(p, template, False)
        out = []
        for L, wi in zip(lines, w):
            xy = c.undistort(c.to_norm(L["pts"]))
            a, b = c.ideal(np.array([L["world_a"], L["world_b"]]))
            n = perp(unit(b - a))
            out.append(((xy - a) @ n) * c.f * wi)
        if points:
            e = _point_resid(c, points)
            s = np.array([P.get("sigma") or CORNER_SIGMA for P in points], float)
            out.append((e / s[:, None]).ravel())
        return np.concatenate(out)
    return resid


def _plausible(cam, shape):
    """`solve`'s two sanity tests: not mirrored, and the court all in front."""
    ok, why = _orientation_ok(cam, shape)
    if not ok:
        return False, why
    return _in_front(cam)


def _check(cam, shape, stage):
    ok, why = _plausible(cam, shape)
    if not ok:
        raise FitError("%s gave an implausible camera: %s" % (stage, why))


# ------------------------------------------------------------------ the fit --
def build_lines(sides, walk_lines):
    """The line correspondences, human ones first, each tagged with its source."""
    out = []
    for key in SIDE_ORDER:
        spec = SIDES[key]
        pts = np.asarray(sides.get(key, []), float).reshape(-1, 2)
        if len(pts) < MIN_MARKS:
            raise FitError("%s has %d of %d marks" % (spec["label"], len(pts), MIN_MARKS))
        a, b = spec["world"]
        out.append({"name": spec["line"], "pts": pts, "source": "marked",
                    "world_a": np.array(a, float), "world_b": np.array(b, float)})
    for L in walk_lines:
        if L["name"] in FROM_WALK and len(L["pts"]) >= 8:
            out.append(dict(L, source="walked"))
    return out


def fit_court(shape, sides, walk_lines, walk_anchors=(), model="fisheye", nk=2):
    """Return (Camera, report). Raises FitError with a sentence for the labeller."""
    h, w = shape[:2]
    lines = build_lines(sides, walk_lines)
    xs = [L for L in lines if L["world_a"][0] == L["world_b"][0]]
    ys = [L for L in lines if L["world_a"][1] == L["world_b"][1]]
    if len(xs) < 2 or len(ys) < 2:
        raise FitError("need 2 lines in each direction, have %d along X and %d "
                       "along Y - the walk did not give enough paint to fit the "
                       "marks against" % (len(xs), len(ys)))

    curves = {k: fit_curve(sides.get(k, []), SIDES[k]["swap"]) for k in SIDE_ORDER}
    if any(c is None for c in curves.values()):
        raise FitError("a side's marks do not make a curve - are several stacked "
                       "on one row or column?")
    corner_px = corners(curves)
    points = [{"name": n, "world": ((0.0, 0.0) if n.endswith("_l") else (CM.WIDTH, 0.0)),
               "px": p, "sigma": CORNER_SIGMA} for n, p in sorted(corner_px.items())]
    points += [dict(A) for A in walk_anchors]

    f = w * 0.5 if model == "fisheye" else float(w)
    cam = Camera(np.eye(3), np.zeros(nk), model, f, w / 2.0, h / 2.0)
    # Stage A, the lens, from straightness alone. Every line is handed over
    # unweighted here: this stage counts curvature evidence, and a six-mark
    # boundary genuinely carries less of it than a four-hundred-sample walk.
    k, (cx, cy), rms_straight = fit_distortion(lines, cam)
    cam = Camera(np.eye(3), k, model, f, cx, cy)
    try:
        Hn = homography_from_lines(lines, cam)
    except BaseException as e:
        raise FitError("the line-DLT failed: %s" % str(e)[:120])
    cam = Camera(Hn, k, model, f, cx, cy)
    _check(cam, shape, "the line-DLT")

    # Pass 1: robust, unweighted, and only to find out whether a walked line is
    # the line it claims to be. Its camera is thrown away.
    x, _ = CM.lm(_residual(lines, points, cam, False), CM.pack(cam), huber=3.0)
    screen = CM.unpack(x, cam, False)
    dropped = []
    if _plausible(screen, shape)[0]:
        for L in lines:
            if L["source"] == "marked":
                continue
            XY = screen.backproject(L["pts"])
            ax = 1 if L["world_a"][1] == L["world_b"][1] else 0
            off = abs(float(np.median(XY[:, ax])) - float(L["world_a"][ax]))
            if off > 0.25:                       # `solve.redundancy_check`'s tolerance
                dropped.append((L["name"], off))
    out = {n for n, _ in dropped}
    keep = [L for L in lines if L["name"] not in out]
    if (len([L for L in keep if L["world_a"][0] == L["world_b"][0]]) < 2
            or len([L for L in keep if L["world_a"][1] == L["world_b"][1]]) < 2):
        keep, dropped = lines, []                # rather no screening than no fit

    # Pass 2: every line worth one correspondence, no Huber - see the module
    # docstring for why those two go together.
    x, r = CM.lm(_residual(keep, points, cam, True), CM.pack(cam), huber=None)
    cam = CM.unpack(x, cam, False)
    _check(cam, shape, "the refined fit")

    d = _line_residuals(cam, keep)
    per_line = {}
    for L, di in zip(keep, d):
        XY = cam.backproject(L["pts"])
        ax = 1 if L["world_a"][1] == L["world_b"][1] else 0
        per_line[L["name"]] = {
            "n": int(len(di)), "source": L["source"],
            "rms": float(np.sqrt(np.mean(di ** 2))), "max": float(np.abs(di).max()),
            "axis": "Y" if ax else "X", "model": float(L["world_a"][ax]),
            "measured": float(np.median(XY[:, ax])), "check_only": False,
        }
    per_point = {}
    if points:
        for P, ei in zip(points, _point_resid(cam, points)):
            per_point[P["name"]] = {
                "px": float(np.linalg.norm(ei)),
                "sigma": float(P.get("sigma") or CORNER_SIGMA),
                "world": [float(v) for v in P["world"]],
                "at": [float(v) for v in P["px"]]}
    allr = np.concatenate(d)
    report = {
        "n_lines": len(keep), "n_checks": 0,
        "n_samples": int(sum(len(L["pts"]) for L in keep)),
        "n_rejected": 0,
        "rms_line_px": float(np.sqrt(np.mean(allr ** 2))),
        "rms_straightness_px": rms_straight,
        "per_line": per_line,
        "n_anchors": len(points),
        "max_anchor_px": max((v["px"] for v in per_point.values()), default=None),
        "per_point": per_point,
        "dropped_walk_lines": [{"name": n, "off_m": round(o, 3)} for n, o in dropped],
        "corners_px": {n: [round(float(v), 2) for v in p] for n, p in corner_px.items()},
        "k": [float(v) for v in cam.k], "f": cam.f, "cx": cam.cx, "cy": cam.cy,
    }
    return cam, report


# ---------------------------------------------------------- the documents --
def verdict(img, cam, report):
    """The detector's own four gates, applied to the labelled camera.

    Not to accept or refuse anything - the human's marks are the truth here by
    definition - but because a labelled court that fails a gate is a court with
    something wrong with it, and the labeller is the one person in a position to
    look at the picture and see what.
    """
    cov, prms = paint_probe(img, cam)
    cerr = report.get("max_anchor_px")
    bad = [b for b in redundancy_check(report)[0]]
    ok = ((not bad) and cov > COVERAGE_MIN and prms < PROBE_MAX
          and report["rms_line_px"] < RMS_MAX
          and not (cerr is not None and cerr > ANCHOR_GATE))
    why = []
    if cerr is not None and cerr > ANCHOR_GATE:
        why.append("anchors %.0f px (gate %.0f)" % (cerr, ANCHOR_GATE))
    if bad:
        why.append("lines " + ",".join(b[0] for b in bad))
    if cov <= COVERAGE_MIN:
        why.append("coverage %.0f%%" % (100 * cov))
    if prms >= PROBE_MAX:
        why.append("probe %.2f px" % prms)
    if report["rms_line_px"] >= RMS_MAX:
        why.append("rms %.2f px" % report["rms_line_px"])
    return cov, prms, ok, "; ".join(why)


def sides_document(plate_name, shape, sides, curves=None, complete=False):
    """`_court-sides.json` - the marks themselves, and nothing derived from them.

    The marks are what a person did and the only thing here that cannot be
    recomputed, so they are stored raw, in click order, at full precision. The
    quadratics are written alongside for a reader who wants the curve without
    re-deriving it, and are never read back: reopening a plate re-fits them from
    the marks, so there is one definition of the curve and it is the code's.
    """
    h, w = shape[:2]
    doc = {
        "plate": os.path.basename(plate_name),
        "image_size": {"width": int(w), "height": int(h)},
        "marked_by": "training-tools/label_court.py",
        "complete": bool(complete),
        "min_marks_per_side": MIN_MARKS,
        "note": "image pixels, in the order they were clicked",
        "sides": {},
    }
    for key in SIDE_ORDER:
        pts = np.asarray(sides.get(key, []), float).reshape(-1, 2)
        spec = SIDES[key]
        doc["sides"][key] = {
            "court_line": spec["line"],
            "world_a": [float(v) for v in spec["world"][0]],
            "world_b": [float(v) for v in spec["world"][1]],
            "n": int(len(pts)),
            "points": [[float(x), float(y)] for x, y in pts],
        }
        c = (curves or {}).get(key)
        if c is not None:
            doc["sides"][key]["quadratic"] = {
                "along": "y" if spec["swap"] else "x",
                "of": "x" if spec["swap"] else "y",
                "coeffs": [float(v) for v in c[0]],
                "span": [float(c[1]), float(c[2])],
            }
    return doc


def truth_document(img, plate_path, cam, report, sides, source):
    """`_court-truth.json` - the same schema `padelcourt` writes, plus provenance.

    Deliberately identical in shape to the detector's own output, so anything
    that reads a detected court reads a labelled one with no special case; the
    `labelled` block is the only addition, and it says where this court came
    from so a truth file is never mistaken for a detection.
    """
    cov, prms, ok, why = verdict(img, cam, report)
    res = api.CourtResult(img, plate_path, cam, report, cov, prms, ok, why,
                          trace=None, source=source)
    doc = res.to_dict()
    doc["labelled"] = {
        "by": "training-tools/label_court.py",
        "marks": {k: int(len(np.asarray(sides.get(k, []), float).reshape(-1, 2)))
                  for k in SIDE_ORDER},
        "corners_px": report.get("corners_px", {}),
        "dropped_walk_lines": report.get("dropped_walk_lines", []),
        "per_line": report.get("per_line", {}),
        "per_point": report.get("per_point", {}),
        "note": ("`accepted` is the detector's own verdict applied to a court a "
                 "person marked. It is a warning, not a filter: the marks are "
                 "the ground truth by definition, and a gate it misses is "
                 "something to go and look at."),
    }
    return doc
