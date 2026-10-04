"""Detect, solve, re-trace, and judge - one plate in, one camera out.

The verdict is deliberately strict and deliberately not the only output: a plate
whose far half is faint can be fitted correctly and still miss the coverage
gate, so `calibrate` returns the camera either way and says what failed.
"""
import cv2
import numpy as np

from . import court as CM
from .solve import CHECK_ONLY, redundancy_check, retrace, solve
from .walk import WORLD_LINES, detect


def paint_probe(img, cam):
    """Independent acceptance check: is there paint where the model says?"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def centre_of_band(u, v, vertical):
        s = np.arange(-20, 20.01, 0.5)
        p = (np.column_stack([u + s, np.full_like(s, v)]) if vertical
             else np.column_stack([np.full_like(s, u), v + s]))
        x0 = np.clip(p[:, 0].astype(int), 0, gray.shape[1] - 1)
        y0 = np.clip(p[:, 1].astype(int), 0, gray.shape[0] - 1)
        prof = gray[y0, x0]
        wgt = np.clip(prof - np.percentile(prof, 20), 0, None)
        if wgt.max() < 25:
            return None
        wgt = np.where(wgt < 0.45 * wgt.max(), 0, wgt)
        if (np.diff(np.concatenate([[0], (wgt > 0).astype(int), [0]])) == 1).sum() != 1:
            return None
        return float((s * wgt).sum() / wgt.sum())

    probes = ([(x, CM.NEAR_SERVICE_Y, False) for x in (1, 2, 3, 4, 6, 7, 8, 9)]
              + [(x, CM.FAR_SERVICE_Y, False) for x in (1.5, 3, 4, 6, 7, 8.5)]
              + [(CM.CENTRE_X, y, True) for y in (11, 12, 13, 14, 15)])
    found, errs, n = 0, [], 0
    for X, Y, vert in probes:
        uv = cam.project([[X, Y]])[0]
        if not (0 <= uv[0] < img.shape[1] and 0 <= uv[1] < img.shape[0]):
            continue
        n += 1
        v = centre_of_band(uv[0], uv[1], vert)
        if v is not None:
            found += 1
            errs.append(abs(v))
    if not n:
        return 0.0, 99.9
    return found / n, (float(np.sqrt(np.mean(np.square(errs)))) if errs else 99.9)


#: Measured and printed, never part of the verdict. The net's floor line reads
#: anywhere from 9.06 m to 10.26 m on fits independently confirmed correct to
#: 0.3-2.6 px - long where the net's bottom rests on the floor and its shadow
#: hides the strip in front of it, short where the lower mesh is translucent and
#: brightly lit so the court run overshoots into it. A 1.2 m spread on good fits
#: is not a 0.25 m gate. It stays visible because a net position that shifts
#: between recordings of one fixed camera means somebody moved the net, which is
#: worth knowing; it just must not decide anything.
ADVISORY = ("net_line",)


#: The anchors are now CONSTRAINTS, so this is the residual of a constraint that
#: was applied rather than a free-standing opinion about the answer - which is a
#: stronger test, not a weaker one. A fit that still misses an anchor it was
#: pulled toward is one where the line evidence actively contradicts it, and the
#: only ways that happens are a wrong anchor or a wrong line.
#:
#: This used to be measured against the NOMINAL corners at X=0 and X=10, which
#: carried `WalkDetect.ANCHOR_OUT` as a permanent ~5 px offence and needed a
#: 44 px gate to tolerate it. Measured against where the paint actually ends,
#: the same 41 accepted plates sit at median 3.0 px and max 22.8 BEFORE the
#: constraint existed to pull them in.
ANCHOR_GATE = 25.0

#: The other three gates the verdict is made of. Named because `_shortfall`
#: scores the refinement rounds against the same four numbers the verdict uses,
#: and two copies of a gate drift apart.
COVERAGE_MIN, PROBE_MAX, RMS_MAX = 0.85, 2.0, 2.0


def _shortfall(cov, prms, rms, cerr):
    """How close this camera is to being refused: 1.0 is exactly on a gate.

    Every gate divided by itself, so the four are comparable and the worst one
    is the score. No new constant is introduced and none is tuned - this is the
    verdict, read as a continuous quantity instead of a boolean.
    """
    return max((cerr or 0.0) / ANCHOR_GATE, prms / PROBE_MAX, rms / RMS_MAX,
               COVERAGE_MIN / max(cov, 1e-3))


def _pin_violation(L, pin):
    """Median offset of a re-traced line from a pinned one, over their overlap.

    Measured perpendicular in the only sense that matters here - these are
    near-horizontal lines fitted as y(x) - and only where the pin actually
    measured something, so a re-trace that reaches further is judged on the part
    the walk can speak for rather than on its extrapolation.
    """
    p = np.asarray(L["pts"], float).reshape(-1, 2)
    m = (p[:, 0] >= pin["x_lo"]) & (p[:, 0] <= pin["x_hi"])
    if m.sum() < 8:
        return float("inf")                      # no overlap: not the same line
    return float(np.median(np.abs(p[m, 1] - np.polyval(pin["coeffs"], p[m, 0]))))


def _finish(img, lines, ctx, want, rounds, model, nk, points=()):
    cam, rep = solve(lines, img.shape, model=model, nk=nk, points=points,
                     check_only=CHECK_ONLY)
    # A REFINEMENT ROUND IS A CANDIDATE, NOT A REPLACEMENT. Taking the last
    # round unconditionally assumes the re-trace can only add precision, and
    # `retrace` has a second failure mode besides finding the wrong line: it can
    # find NOTHING. It re-measures the sidelines over the near half only, where
    # the boundary is a colour step in the vignetted corner, and on 28% of the
    # sidelines in this corpus it comes back with too few samples to keep. When
    # it loses both, the round has fewer than four lines and is skipped below -
    # but when it loses ONE, the fit is handed two lines along X where the walk
    # gave three, and two parallel lines in a direction have no cross-ratio to
    # check each other with. That is the degeneracy the six-line design exists
    # to avoid, arriving after the design has done its job. On `17-02-10` the
    # walk's 52-point left sideline is dropped that way and the court closes 10
    # and 22 px inside its own detected corners, on both sides at once.
    #
    # So each round is scored and the best one is kept. The score is the verdict
    # itself (`_shortfall`), which is the only thing here entitled to say one
    # camera is better than another, and it introduces no constant of its own.
    # Ties go to the later round, so the refinement still wins wherever it is
    # doing what it is for.
    #
    # Honest about what this costs: the accepted camera is now the best of three
    # by the same gates that then judge it, so those numbers are very slightly
    # optimistic. Measured over the corpus, the price is small and the gain is
    # not - probe p50 1.30 -> 1.24, p90 1.91 -> 1.78, worst anchor 24.5 -> 22.7,
    # and two plates that were refused on a probe of 2.01 and 2.14 turn out to
    # have had a 1.53 and a 1.75 in hand all along.
    best = (cam, rep, _shortfall(*paint_probe(img, cam),
                                 rep["rms_line_px"], rep.get("max_anchor_px")), 0)
    pins = {k: v for k, v in (ctx.get("pinned") or {}).items() if v}
    by_name = {L["name"]: L for L in lines}
    for _round in range(rounds):
        rt = retrace(img, cam, want, ctx=ctx)
        # A LINE THAT WAS MEASURED WELL IS NOT UP FOR RE-MEASUREMENT. `retrace`
        # re-seeds every line from the current model, which is what makes the
        # far service line safe when it was detected blind - and is exactly what
        # sinks it when it was not. The far half is a thin sliver with the net's
        # top tape 25-30 px away and brighter, so a model whose far end is a
        # little wrong seeds the search there, the tape is found, the tape is
        # fitted, and the next round is seeded from a model that agrees. The
        # walked line and its anchor lose that argument: measured on two plates
        # here, the far service T ends up 29-30 px from where the walk put it,
        # against a sigma of 2 px.
        #
        # So the re-trace of a line has to agree with how that line was actually
        # measured, to within a tolerance that measurement earned
        # (`walk.pin_far_service`), or it is discarded and the measurement
        # stands. The tolerance is not a tolerance on the ANSWER - it is one on
        # the disagreement between two measurements of one line.
        for name, pin in pins.items():
            for i, L in enumerate(rt):
                if L["name"] != name:
                    continue
                if _pin_violation(L, pin) > pin["tol"] and name in by_name:
                    rt[i] = by_name[name]
                break
        rt += [L for L in lines if L["name"] == "net_line"]
        # Count what `solve` will FIT, not what it is handed. It removes its
        # check-only lines - the net - before applying this same test, so
        # counting the net here meant a 3-line retrace passed the guard as 4 and
        # then raised inside `solve`.
        if len([L for L in rt if L["name"] not in CHECK_ONLY]) < 4:
            break
        try:
            # The anchors go into every round, not just the first. `retrace`
            # re-measures the lines from the current model, so a round that
            # started from a drifted X scale would otherwise re-measure the
            # sidelines where that model put them and confirm its own drift.
            cam, rep = solve(rt, img.shape, model=model, nk=nk, points=points)
        except SystemExit:
            # A refinement round that cannot run is a round not taken, not a
            # reason to discard the answer we already have. Letting this escape
            # threw away pass 1 entirely and cost five videos their calibration.
            break
        s = _shortfall(*paint_probe(img, cam), rep["rms_line_px"],
                       rep.get("max_anchor_px"))
        if s <= best[2]:
            best = (cam, rep, s, _round + 1)
    cam, rep, _, kept = best
    rep["refine_round"] = kept
    return cam, rep


def anchor_error(rep):
    """Largest distance from the fitted court to an anchor it was given."""
    return rep.get("max_anchor_px")


def calibrate(img, rounds=2, model="fisheye", nk=2, trace=None, anchor=True):
    """Return (camera, report, coverage, probe rms, ok, note).

    The net's floor line is detected and reported but held OUT of the fit, and
    out of the verdict - see `ADVISORY`. Everything else the walk found is
    fitted, re-traced from the model `rounds` times, and then judged against
    paint the fit never saw.
    """
    lines, ctx = detect(img, verbose=False, with_ctx=True, trace=trace)
    want = {k: v for k, v in WORLD_LINES.items() if k != "net_line"}
    cam, rep = _finish(img, lines, ctx, want, rounds, model, nk,
                       points=ctx["anchors"] if anchor else ())
    cov, prms = paint_probe(img, cam)
    cerr = anchor_error(rep)
    bad = [b for b in redundancy_check(rep)[0] if b[0] not in ADVISORY]
    ok = ((not bad) and cov > COVERAGE_MIN and prms < PROBE_MAX
          and rep["rms_line_px"] < RMS_MAX)
    if cerr is not None and cerr > ANCHOR_GATE:
        ok = False
    why = []
    if cerr is not None and cerr > ANCHOR_GATE:
        why.append("anchors %.0f px (gate %.0f)" % (cerr, ANCHOR_GATE))
    if bad:
        why.append("lines " + ",".join(b[0] for b in bad))
    if cov <= COVERAGE_MIN:
        why.append("coverage %.0f%%" % (100 * cov))
    if prms >= PROBE_MAX:
        why.append("probe %.2f px" % prms)
    if rep["rms_line_px"] >= RMS_MAX:
        why.append("rms %.2f px" % rep["rms_line_px"])
    return cam, rep, cov, prms, ok, "; ".join(why)
