"""Court paint as plumb-line evidence, from this project's own walk detector.

The hall's own straight edges solve the lens on most plates, but they are not
evidence about the COURT in particular, and on a plate whose periphery is mostly
burnt-in overlay - straight in the distorted frame by construction - the blind
fit can over-correct and bow the court lines the *other* way. Every straightness
measure still reads healthy while it does, because the thing that was bowed is
not in the set being measured.

The court's own lines cannot lie about this. They are straight on the ground,
they are what any calibration downstream will be measured against, and
`padelcourt.walk` finds them on the raw frame with no lens knowledge at all -
`detect` is a walk out of one junction, so nothing it does assumes straightness
in the image. Feeding them into the same bend objective removes the failure at
its source instead of detecting it afterwards.

They do not replace the hall arcs, and must not: the paint sits in the middle of
the frame, where the lens does least. The periphery, where it does most, is
still only constrained by the building. How the two share the objective is
`lens.solve`'s decision - see `court_frac`.

WHAT IS FED, AND WHAT IS NOT. Everything `walk.detect` returns, minus the far
baseline. Measured against the two click-calibrated cameras - undistort a line
that is straight in the world with the TRUE lens, and the bow that is left is
that evidence's own error - the six detected lines come back 0.2-1.7 px rms over
150-520 px, while the far baseline comes back at 22 and 32. It is not a noisy
measurement of a straight thing; the glass hides the strip of floor in front of
it and the posts scallop what is left, so what is being measured is not straight.
`CURV` below is the gate that catches the same problem in any other line.
"""

import numpy as np

from .lens import _shape, _sagitta

#: Never fed, whatever it measures. See the module docstring.
NOT_STRAIGHT = ("far_baseline",)

#: A walked line is dropped if it bows this many times more, per unit length
#: squared, than the same plate's near service line. That line is the one the
#: detector gets right every time - it is the anchor's own line, walked to ~440
#: samples and then slid onto the middle of its own paint - so it measures what
#: THIS plate's lens actually does to a straight thing, which is the only
#: reference available before the lens is known. Anything bowing several times
#: harder is not a straight thing seen through that lens.
#:
#: Over this corpus the near service line's normalised curvature runs
#: 6.2-6.7e-4/px and the far baseline, the one line known to be crooked, sits at
#: 26-30e-4. The gate has two decades of room and is not a tuned threshold.
CURV_RATIO = 3.0
CURV_FLOOR = 20e-4

#: Fewer samples than this and a line says nothing about a bow: what is fitted
#: is a curve through whatever `walk.trim_to_curve` has left.
MIN_PTS = 12

#: Degree of the curve each walked line is re-laid on, and the two passes of
#: outlier rejection that fit it.
#:
#: THE SMOOTHING IS NOT COSMETIC - IT IS WHAT MAKES THE COURT USABLE AT ALL. On
#: the two click-calibrated cameras, feeding the raw point sets is worth
#: 6.3 -> 6.2 px and 12.3 -> 10.8; feeding them fitted is worth 6.3 -> 2.4 and
#: 12.3 -> 8.5.
#:
#: The objective is bend = rms-off-the-best-fit-line / extent, and a walked line
#: carries about half a pixel of per-sample scatter that no lens can remove. On
#: a line of extent ~500 px that is a floor of 1e-3 in bend, the same order as
#: `COURT_CAP`, so a raw point set arrives near saturation and barely pulls.
#: Worse, the one move that does lower a noise-dominated ratio is growing its
#: denominator, so what pull there is points at magnifying the middle of the
#: frame rather than at straightening anything.
#:
#: Fitting the line first removes the scatter - over 150-440 samples, by an
#: order of magnitude - without touching the bow, which is the only thing being
#: measured. The degree is the whole question. Two is WORSE than not smoothing
#: at all, 8.6/12.2 px, because a fisheye's image of a straight line is not a
#: parabola over 1800 px and a quadratic forced through it misrepresents the
#: shape by more than the noise it removes. The lens model is itself 4th order
#: in theta, and 4 is where the reference plates and the corpus stop improving
#: together - 5 and 6 each win one of them and lose the other.
#:
#: The rejection passes matter as much as the degree. A least-squares fit of any
#: degree follows a stray tail sample, and the tails of these walks are where
#: the strays are: at degree 4 they take the worst disagreement between two
#: plates of one fixed camera from 62.9 px to 38.1.
POLY_DEG, POLY_PASSES, POLY_KILL = 4, 2, 3.0


def _resample(pts, m, deg=POLY_DEG):
    """Order a line's samples along its own axis and re-lay them on a fitted curve.

    The walks return anything from 18 to 440 points at wildly different
    spacings; the bend objective wants one fixed-size tensor. Working in the
    line's own dominant coordinate keeps the shape, and evening out the stations
    also stops a line being judged mostly by whichever end happened to be
    sampled densest - the near service line carries a third of its samples in
    the 80 px around the anchor it was walked out of.

    The degree is dropped on a short line rather than being refused: a walk that
    came back with 18 samples still knows where it bowed, it just cannot afford
    five coefficients to say so.
    """
    d = pts - pts.mean(0)
    C = d.T @ d
    ax = 0 if C[0, 0] >= C[1, 1] else 1
    u, v = pts[:, ax], pts[:, 1 - ax]
    o = np.argsort(u)
    u, v = u[o], v[o]
    keep = np.concatenate([[True], np.diff(u) > 1e-9])
    u, v = u[keep], v[keep]
    deg = min(deg, max(2, len(u) // 4))
    if len(u) < deg + 3:
        return None
    good = np.ones(len(u), bool)
    for _ in range(POLY_PASSES):
        r = v - np.polyval(np.polyfit(u[good], v[good], deg), u)
        s = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        if s < 1e-9:
            break
        g = np.abs(r) <= POLY_KILL * s
        if g.sum() < deg + 3 or (g == good).all():
            break
        good = g
    t = np.linspace(u[0], u[-1], m)
    w = np.polyval(np.polyfit(u[good], v[good], deg), t)
    return np.column_stack([t, w] if ax == 0 else [w, t])


def walked_lines(bgr, verbose=False):
    """Raw frame -> {name: (N, 2) points}, or {} if the walk cannot start.

    Kept separate from `court_arcs` so a caller that has already walked - the
    court detector walks every plate it calibrates - can hand the result over
    instead of paying for a second walk, which is the expensive half of this.
    """
    try:
        from padelcourt.walk import detect
    except ImportError as e:                 # straighten copied out on its own
        if verbose:
            print("  no court detector available:", e)
        return {}
    try:
        out = detect(bgr, verbose=False)
    except (SystemExit, Exception) as e:     # the walk aborts on a bad anchor
        if verbose:
            print("  walk failed:", e)
        return {}
    return {L["name"]: np.asarray(L["pts"], float) for L in out}


def court_arcs(bgr, m=48, min_pts=MIN_PTS, lines=None, verbose=False):
    """Raw frame -> (arcs, weights, names), in the same form as `lens.arc_set`.

    Weights are the same extent^2 the hall arcs use, unscaled: how much
    authority the court gets against the building is `lens.solve`'s decision,
    made as a share of the total weight rather than as a multiplier, because the
    number of hall arcs varies by a factor of three across the corpus and a
    multiplier would mean something different on every plate.

    Returns empty arrays whenever the walk cannot start or every line is
    refused; the lens solve then falls back to the hall arcs alone, which is
    exactly what it did before this existed.

    The point sets are taken as the detector produced them, and are NOT
    re-centred on their paint here. Two of them - the near service line and the
    centre line - have already been slid onto the middle of their own stripe by
    `walk.detect`, which is what makes them the straightest evidence in the
    frame. The other four are deliberately not: the far service line's paint is
    1-2 px wide and has no middle to find, and the sidelines and the net line
    are not paint at all - a colour step at the foot of the glass and the edge of
    the net's shadow. Running a stripe-centring rule over those measures the
    width of something that is not a stripe.
    """
    if lines is None:
        lines = walked_lines(bgr, verbose=verbose)

    A, names = [], []
    for name, pts in lines.items():
        pts = np.asarray(pts, float).reshape(-1, 2)
        if name in NOT_STRAIGHT or len(pts) < min_pts:
            continue
        r = _resample(pts, m)
        if r is None:
            continue
        A.append(r)
        names.append(name)
    if not A:
        return np.zeros((0, m, 2)), np.zeros(0), []
    A = np.stack(A)

    ext, _ = _shape(A)
    w = ext ** 2
    curv = 2 * np.abs(_sagitta(A)) / np.maximum(ext, 1e-9) ** 2
    ref = curv[names.index("near_service")] if "near_service" in names else 0.0
    lim = max(CURV_RATIO * ref, CURV_FLOOR)
    keep = curv <= lim
    if verbose:
        for n, c, k in zip(names, curv, keep):
            if not k:
                print(f"  dropped {n}: bow {c*1e4:.1f}e-4/px against a limit "
                      f"of {lim*1e4:.1f}e-4")
    if not keep.any():
        return np.zeros((0, m, 2)), np.zeros(0), []
    return A[keep], w[keep], [n for n, k in zip(names, keep) if k]
