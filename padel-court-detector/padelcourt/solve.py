"""Solve the camera from LINE correspondences only - no points anywhere.

Why lines rather than points. Every earlier version funnelled the geometry
through seed keypoints, and every one of them failed the same way: a seed a few
tens of pixels out sent a trace onto the wrong feature, and because the two
service lines are parallel the fit absorbed the error by rescaling Y and still
reported a clean residual. A line correspondence needs no seed to exist, so the
failure has nowhere to enter.

Stages
    A  distortion alone, by making each point set straight   (plumb-line)
    B  undistort; every set is now genuinely straight, so fit a line to each
    C  homography from the six line correspondences, by DLT
    D  joint refine of (k, Hn) against every point, robustly
    E  reject and refit

Stage A works with no homography because "this is straight in the world"
constrains the lens alone. Stage C then needs no seed either: for a homography
H, a world line l appears as l' with H^T l' ~ l, which is two linear
constraints per line. Six lines give twelve constraints on eight degrees of
freedom, three lines in each direction, so each direction carries a redundant
constraint and a misidentified line RAISES the residual instead of hiding in it.

ANCHORS. Lines alone leave one thing loose. A line correspondence measures
perpendicular distance only, so it says nothing about where along itself a point
sits: the near service line pins its own direction and offset and leaves the X
scale entirely to the two sidelines. Where those are short, noisy or have to be
extrapolated to reach the near end of the court, X drifts, and the fit reports a
clean residual while doing it - the same failure the whole line-based design
exists to prevent, arriving through the one door it left open.

So `points` may carry a few honest point correspondences. They are NOT seeds:
nothing is traced from them and stage C never sees them, so a bad one cannot
send a detector onto the wrong feature. They enter only stage D, at their own
measured sigma, where they are 4 residuals against ~440 and therefore change
nothing in any direction the lines already determine - and everything in the one
they do not.
"""
import numpy as np

from . import court as CM
from . import image as IM
from .court import Camera, unit, perp


def _fit_line(xy):
    """Total-least-squares line through points, as homogeneous (a, b, c)."""
    c = xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy - c, full_matrices=False)
    n = Vt[1]                                   # unit normal
    return np.array([n[0], n[1], -float(n @ c)])


def homography_from_lines(lines, cam, verbose=False):
    """DLT on line correspondences. Returns Hn: court metres -> ideal normalised.

    For x' = H x, a world line l maps to l' with H^T l' ~ l. Writing A = H^T,
    the cross product l x (A l') = 0 gives two independent linear equations in
    the nine entries of A. Stack them, take the null vector, transpose back.

    Both sides are normalised first (world into [-1,1], image lines to unit
    normal); without it the 20-metre world scale against a ~1-unit image scale
    conditions the SVD badly enough to matter.
    """
    Tw = np.array([[2.0 / CM.WIDTH, 0, -1.0],
                   [0, 2.0 / CM.LENGTH, -1.0],
                   [0, 0, 1.0]])
    TwInvT = np.linalg.inv(Tw).T
    rows = []
    for L in lines:
        xy = cam.undistort(cam.to_norm(L["pts"]))
        lp = _fit_line(xy)
        lp = lp / np.linalg.norm(lp[:2])
        lw = np.cross(np.append(L["world_a"], 1.0), np.append(L["world_b"], 1.0))
        lw = TwInvT @ lw
        lw = lw / np.linalg.norm(lw[:2]) if np.linalg.norm(lw[:2]) > 1e-12 else lw
        p, q, r = lw
        z = np.zeros(3)
        # All THREE cross-product equations, not a chosen two. l x (A l') = 0
        # has rank 2, so two rows suffice in general - but not always: the world
        # line X=5 normalises to l=(1,0,0), and with q=r=0 the row pair I first
        # picked collapsed to a single equation. The system then came out rank 7
        # with a two-dimensional null space, and the DLT returned a mirrored
        # court. Handing the SVD the redundant third row costs nothing.
        rows.append(np.concatenate([z, -r * lp, q * lp]))
        rows.append(np.concatenate([r * lp, z, -p * lp]))
        rows.append(np.concatenate([-q * lp, p * lp, z]))
    Amat = np.array(rows, float)
    _, s, Vt = np.linalg.svd(Amat)
    A = Vt[-1].reshape(3, 3)
    Hn_n = A.T
    Hn = Hn_n @ Tw
    if abs(Hn[2, 2]) < 1e-12:
        raise SystemExit("degenerate line-DLT solution")
    Hn = Hn / Hn[2, 2]
    if verbose:
        print("  line-DLT: %d lines, %d rows, singular values %.3e / %.3e (ratio %.1f)"
              % (len(lines), len(rows), s[-2], s[-1],
                 s[-2] / max(s[-1], 1e-12)))
    return Hn


def _in_front(cam):
    """Is the WHOLE court in front of the camera?

    A homography does not know that it must be: it will happily place the
    plane's horizon across the middle of the court, and world points beyond it
    project through infinity and come back MIRRORED. One plate did exactly that
    - horizon at Y = 19.06 m on a court 20 m long, so the near baseline came out
    at depth -0.05 and was drawn off the top of the frame with left and right
    swapped, a fan of lines radiating from a vanishing point inside the playing
    surface.

    Nothing else caught it, and nothing else could: the fold is beyond the near
    SERVICE line at Y=16.95, and every other check - the paint probe, the metre
    grid, the per-line residuals, the anchors - lives between the two service
    lines. That fit reported line_rms 0.79 px and 89% paint coverage.

    Depth is affine in (X, Y), so the four corners bound the whole rectangle.
    Measured over 49 cameras the worst legitimate margin is +0.028 against this
    fold's -0.050, with nothing in between, so the test is the physical one
    rather than a fitted threshold.

    Applied ONLY to a refined camera, never to the line-DLT that seeds it. The
    DLT is an initialisation and is allowed to be silly: on one plate a
    refinement round starts from a DLT that is folded through 50% of the court
    and LM pulls it back to a fit 10 px better than the round before. Testing
    the seed threw that round away and cost the plate its calibration.
    """
    d = np.array([[0.0, 0.0], [CM.WIDTH, 0.0],
                  [CM.WIDTH, CM.LENGTH], [0.0, CM.LENGTH]]) @ cam.Hn[2, :2] + cam.Hn[2, 2]
    if cam.Hn[2, 2] < 0:
        d = -d
    if not np.isfinite(d).all() or d.min() <= 0:
        return False, ("the horizon crosses the court - %.0f%% of its length is "
                       "behind the camera" % (100 * (d <= 0).mean()))
    return True, ""


def _orientation_ok(cam, shape):
    """Reject a mirrored or collapsed solution before wasting a refinement."""
    h, w = shape[:2]
    p = cam.project([[CM.CENTRE_X, CM.NET_Y], [CM.CENTRE_X, CM.NEAR_SERVICE_Y],
                     [0.0, CM.NET_Y], [CM.WIDTH, CM.NET_Y]])
    if not np.isfinite(p).all():
        return False, "non-finite projection"
    if p[1][1] <= p[0][1]:
        return False, "near service line is not below the net in the image"
    if p[3][0] <= p[2][0]:
        return False, "court is mirrored left-to-right"
    if not (0 < p[0][0] < w and 0 < p[0][1] < h):
        return False, "court centre projects outside the frame"
    return True, ""


def _visible_span(cam, wa, wb, shape, margin=6, n=200):
    """Longest run of the world line whose projection stays inside the frame."""
    h, w = shape[:2]
    t = np.linspace(0.0, 1.0, n)[:, None]
    W = np.asarray(wa, float) + t * (np.asarray(wb, float) - np.asarray(wa, float))
    P = cam.project(W)
    ok = ((P[:, 0] > margin) & (P[:, 0] < w - margin) &
          (P[:, 1] > margin) & (P[:, 1] < h - margin) & np.isfinite(P).all(axis=1))
    if ok.sum() < 10:
        return None
    idx = np.where(ok)[0]
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    run = max(splits, key=len)
    return W[run[0]], W[run[-1]]


def retrace(bgr, cam, want, ctx=None, verbose=False):
    """Re-measure every line, seeded from the CURRENT model rather than a guess.

    This is the step that makes the far service line safe. Detected blind it
    came out as the far baseline (measured Y = -0.09m instead of 3.05m), because
    the far half segments as a thin sliver and any ridge inside it looks
    plausible. Projected from a model that is already good to a few pixels, the
    search window can be tight enough that no other feature is reachable.

    Uses CourtFit's tracer, so the samples are the same sub-pixel quality as the
    click-seeded path - the only difference is where the seed came from.

    AND THEN CENTRED ON THE PAINT, exactly as the walks are. `trace_segment`
    re-centres with `image._peak_ridge`, so it inherits that rule's habit of
    latching onto one side of a wide stripe and staying there - see
    `image.centre_on_paint`. Leaving it out here undoes the correction on most of
    the corpus rather than on none of it: a refinement round is kept whenever it
    scores better overall, 72 of 116 plates keep one, and an uncorrected round
    hands the fit back the 2 px the detection had just removed. Measured, the
    difference is the whole of the gain at the tail - with the rounds corrected
    too the probe's p90 is 0.97 px, with only the detection corrected it is 1.61.
    """
    ridge = IM.line_response(bgr)
    # The detector already built this and it costs a 51px median blur; taken
    # from `ctx` where there is one, so two refinement rounds do not rebuild it.
    edges = (ctx or {}).get("edges")
    if edges is None:
        edges = IM.white_edges(bgr)
    seeds = cam.project(CM.KP_WORLD)
    surf = IM.surface_score(bgr, seeds)
    out = []
    for name, (wa, wb) in want.items():
        wa, wb = CM.RETRACE_SPAN.get(name, (wa, wb))
        span = _visible_span(cam, wa, wb, bgr.shape)
        if span is None:
            continue
        A, B = cam.project([span[0]])[0], cam.project([span[1]])[0]
        kind = CM.LINE_KIND[name]
        field = surf if kind == CM.BOUNDARY else ridge
        if name == "net_line" and ctx is not None and ctx.get("net_field") is not None:
            field = ctx["net_field"]
        inward = 1.0
        if kind == CM.BOUNDARY:
            d = unit(B - A)
            nrm = perp(d)
            midw = 0.5 * (np.asarray(span[0]) + np.asarray(span[1]))
            insw = midw + np.array(CM.INWARD[name]) * 0.6
            midi, insi = cam.project([midw])[0], cam.project([insw])[0]
            inward = -np.sign(float((insi - midi) @ nrm)) or 1.0
        pts = IM.trace_segment(field, A, B, kind=kind, inward=inward,
                               step=4.0, half=8.0, acquire=12.0,
                               min_contrast=10.0, shape=bgr.shape)
        if len(pts) < 12:
            if verbose:
                print("    retrace %-15s only %d samples" % (name, len(pts)))
            continue
        if kind == CM.PAINT:
            pts = IM.centre_on_paint(edges, pts,
                                     abs(B[0] - A[0]) < abs(B[1] - A[1]))
        wa2, wb2 = CM.WORLD_LINES[name]
        out.append({"name": name, "pts": pts,
                    "world_a": np.array(wa2, float), "world_b": np.array(wb2, float)})
    return out


#: Measured and reported, but kept OUT of the residual. The net's floor line is
#: a segmentation edge rather than paint (rms ~2.6px against ~1px for the paint
#: lines), and where it sits relative to Y=10 is uncertain at the few-centimetre
#: level - the net has thickness and its base carries a shadow. Fitted at full
#: weight it drags the solution ~5x; held out, it still answers the question the
#: whole line-based design exists to answer, namely whether a Y line is the line
#: it claims to be. Same conclusion the far baseline forced in Court/.
CHECK_ONLY = ("net_line",)

#: How well a point correspondence is believed, in pixels, when it does not say.
#:
#: Measured on the five recordings of the one installation with a click-seeded
#: ground truth, where the fit is known right to 0.4 px and the anchor residual
#: is therefore the ANCHOR's error and nothing else: 1.8, 2.0, 2.2, 2.8, 3.1 px.
#: The error budget agrees - detection repeats to ~1 px within a pose, the paint
#: overshoot varies ~2 cm (1.9 px) between venues, and the left/right asymmetry
#: deliberately left unmodelled is another 1.9, for 2.9 px combined.
#:
#: Reading it off the whole corpus instead gives 4.4, and that number is wrong:
#: it is measured against unconstrained fits, so it counts the very drift this
#: anchor exists to remove as if it were noise in the anchor. Believing it costs
#: real plates. `CM.lm`'s Huber knee sits at 3 sigma, and in the linear regime
#: past it the pull is proportional to 1/sigma, so a sigma inflated by fit drift
#: pulls proportionally less exactly where the drift is worst - at sigma 5 one
#: of the three plates this feature was built for lands at 48 px, at sigma 3 it
#: lands at 2.8. Below the knee it makes no difference at all: on the ground
#: truth group every sigma from 2 to 5 reproduces the same camera to 0.07 px.
ANCHOR_SIGMA = 3.0


def _point_resid(c, pset):
    """(n, 2) error of each correspondence, in REAL image pixels.

    Forwards through the distortion rather than undistorting the observation,
    which is what the line residuals do. The two are not interchangeable here:
    these anchors sit at the far left and right of the frame, where a fisheye's
    undistortion magnifies by ~3.5x, so measuring them the line way would apply
    a sigma of 5 as 1.4 at the very place the anchor is least certain - and by a
    factor that changes with the lens and with where in the frame the corner
    happens to fall. A line sample can be measured either way because it is a
    distance to a line that is straight in undistorted space; a point has a
    position in the image, and that is what this reports.
    """
    px = np.array([P["px"] for P in pset], float)
    return c.project(np.array([P["world"] for P in pset], float)) - px


def solve(lines, shape, model="fisheye", nk=2, f=None, free_centre=False,
          reject_px=3.0, check_only=CHECK_ONLY, points=(), verbose=False):
    """Point sets (+ optional anchors) -> (Camera, report).

    `points` is [{name, world, px, sigma}] - see the module docstring. Still no
    keypoints in the sense that mattered: none of this seeds a trace.
    """
    h, w = shape[:2]
    points = [dict(P) for P in points]
    checks = [L for L in lines if L["name"] in check_only]
    lines = [L for L in lines if L["name"] not in check_only]
    if len(lines) < 4:
        raise SystemExit("only %d lines detected - need at least 4" % len(lines))
    xs = [L for L in lines if L["world_a"][0] == L["world_b"][0]]
    ys = [L for L in lines if L["world_a"][1] == L["world_b"][1]]
    if len(xs) < 2 or len(ys) < 2:
        raise SystemExit("need at least 2 lines in each direction, have %d/%d"
                         % (len(xs), len(ys)))
    f = float(f) if f else (w * 0.5 if model == "fisheye" else float(w))
    cam = Camera(np.eye(3), np.zeros(nk), model, f, w / 2.0, h / 2.0)

    lines = [dict(L) for L in lines]                 # own the point arrays
    k, (cx, cy), rms_a = fit_distortion(lines, cam, free_centre, verbose)
    cam = Camera(np.eye(3), k, model, f, cx, cy)

    Hn = homography_from_lines(lines, cam, verbose)
    cam = Camera(Hn, k, model, f, cx, cy)
    ok, why = _orientation_ok(cam, shape)
    if not ok:
        raise SystemExit("line-DLT gave an implausible camera: %s" % why)

    if points:
        pt_world = np.array([P["world"] for P in points], float)
        pt_px = np.array([P["px"] for P in points], float)
        pt_sigma = np.array([P.get("sigma") or ANCHOR_SIGMA for P in points], float)

    def resid_for(lset):
        """The residual `lm` differentiates, with everything constant hoisted out.

        `lm` builds its Jacobian numerically, so this is called eleven-odd times
        per iteration on point sets that never change. Two things follow. The
        samples are normalised and their radii taken ONCE, here, rather than per
        call; and the undistortion - the whole cost - is memoised on `k`, which
        is what it depends on, so the eight columns that perturb the homography
        alone reuse the one the base point already computed. Same arithmetic,
        about a fifth of it.
        """
        counts = [len(L["pts"]) for L in lset]
        allpts = (np.concatenate([L["pts"] for L in lset]) if lset
                  else np.zeros((0, 2)))
        # World endpoints, homogeneous and stacked, so mapping them through the
        # candidate homography is one matmul rather than one per line.
        ends = np.array([[list(L["world_a"]) + [1.0], list(L["world_b"]) + [1.0]]
                         for L in lset], float).reshape(-1, 3) if lset \
            else np.zeros((0, 3))
        n_end = len(ends)
        # The anchors ride along in the same matmul - they are world points too.
        world_h = (np.vstack([ends, np.column_stack([pt_world, np.ones(len(pt_world))])])
                   if points else ends)
        fixed = None if free_centre else cam.to_norm(allpts)
        fixed_rd = None if fixed is None else np.linalg.norm(fixed, axis=1)
        memo = {}

        def resid(p):
            c = CM.unpack(p, cam, free_centre)
            xn = fixed if fixed is not None else c.to_norm(allpts)
            rd = fixed_rd if fixed_rd is not None else np.linalg.norm(xn, axis=1)
            key = c.k.tobytes() if fixed is not None else (c.k.tobytes(), c.cx, c.cy)
            xy = memo.get(key)
            if xy is None:
                if len(memo) > 32:               # one LM iteration's worth
                    memo.clear()
                xy = memo[key] = xn * c.undistort_scale(rd)[:, None]
            q = world_h @ c.Hn.T
            xyi = q[:, :2] / np.where(np.abs(q[:, 2]) < 1e-12, 1e-12, q[:, 2])[:, None]
            ab = xyi[:n_end]
            a, b = ab[0::2], ab[1::2]
            d = b - a
            u = d / np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)[:, None]
            nrm = np.column_stack([-u[:, 1], u[:, 0]])
            # Every line's samples measured in one pass: the per-line loop this
            # replaces was six slices and a dot product each, on point sets of a
            # few hundred, eleven times per LM iteration.
            ap = np.repeat(a, counts, axis=0)
            npt = np.repeat(nrm, counts, axis=0)
            out = [((xy - ap) * npt).sum(axis=1) * c.f]
            if points:
                # Divided by sigma so the anchors and the line samples arrive at
                # the loss in the same currency, and so `CM.lm`'s single Huber
                # threshold means "3 sigma" for both. Both components are kept
                # even though only the one along the line carries information -
                # the across-the-line component is what the line samples already
                # measure, so it costs nothing and it is a free consistency
                # check that the anchor belongs to the line it claims to end.
                # `Camera.project`'s three stages, written out on the ideal
                # coordinates the matmul above already produced: two points do
                # not justify twenty numpy calls, and this runs on every column
                # of every Jacobian.
                xi = xyi[n_end:]
                rr = np.linalg.norm(xi, axis=1)
                th = np.arctan(rr)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sc = th * c._radial(th) / rr
                xd = xi * np.where(rr < 1e-12, 1.0, sc)[:, None]
                e = xd * c.f + np.array([c.cx, c.cy]) - pt_px
                out.append((e / pt_sigma[:, None]).ravel())
            return np.concatenate(out)
        return resid

    x, r = CM.lm(resid_for(lines), CM.pack(cam, free_centre), huber=3.0, verbose=verbose)
    cam = CM.unpack(x, cam, free_centre)

    n_drop = 0
    for _ in range(2):
        d = _line_residuals(cam, lines)
        scale = max(1.0, 1.4826 * float(np.median(np.abs(np.concatenate(d)))))
        cut = max(reject_px, 4.0 * scale)
        for L, di in zip(lines, d):
            L["pts"] = L["pts"][np.abs(di) <= cut]
            n_drop += int((np.abs(di) > cut).sum())
        lines = [L for L in lines if len(L["pts"]) >= 8]
        # Anchors are never rejected here. There are two of them and they are
        # the only evidence about the thing they constrain, so an outlier test
        # run against a fit that has already drifted would drop exactly the
        # anchor that was about to correct it. A wrong one is handled instead by
        # the Huber knee above, and reported below so it can be seen.
        x, r = CM.lm(resid_for(lines), CM.pack(cam, free_centre), huber=3.0, verbose=verbose)
        cam = CM.unpack(x, cam, free_centre)

    ok, why = _orientation_ok(cam, shape)
    if not ok:
        raise SystemExit("refined camera is implausible: %s" % why)
    ok, why = _in_front(cam)
    if not ok:
        raise SystemExit("refined camera is implausible: %s" % why)

    per_line, d = {}, _line_residuals(cam, lines + checks)
    for L, di in zip(lines + checks, d):
        XY = cam.backproject(L["pts"])
        ax = 1 if L["world_a"][1] == L["world_b"][1] else 0
        per_line[L["name"]] = {
            "n": int(len(di)), "rms": float(np.sqrt(np.mean(di ** 2))),
            "max": float(np.abs(di).max()),
            "axis": "Y" if ax else "X", "model": float(L["world_a"][ax]),
            "measured": float(np.median(XY[:, ax])),
            "check_only": L["name"] in check_only,
        }
    per_point = {}
    if points:
        e = _point_resid(cam, points)
        for P, ei in zip(points, e):
            per_point[P["name"]] = {
                "px": float(np.linalg.norm(ei)), "sigma": float(P.get("sigma") or ANCHOR_SIGMA),
                "world": [float(v) for v in P["world"]],
                "at": [float(v) for v in P["px"]]}
    allr = np.concatenate(d[:len(lines)]) if lines else np.array([0.0])
    return cam, {
        "n_lines": len(lines), "n_checks": len(checks), "n_samples": int(sum(len(L["pts"]) for L in lines)),
        "n_rejected": n_drop,
        "rms_line_px": float(np.sqrt(np.mean(allr ** 2))),
        "rms_straightness_px": rms_a,
        "per_line": per_line,
        "n_anchors": len(points),
        "max_anchor_px": max((v["px"] for v in per_point.values()), default=None),
        "per_point": per_point,
        "k": [float(v) for v in cam.k], "f": cam.f, "cx": cam.cx, "cy": cam.cy,
    }


def redundancy_check(report, tol_m=0.25):
    """With 3 parallel lines per direction, a wrong one cannot hide.

    Each direction has one redundant constraint, so a line that is not the line
    it claims to be forces a compromise the residual has to pay for. This turns
    that into a verdict without needing anything outside the fit.
    """
    bad = []
    for name, s in report["per_line"].items():
        if abs(s["measured"] - s["model"]) > tol_m:
            bad.append((name, s["axis"], s["model"], s["measured"]))
    per_axis = {}
    for s in report["per_line"].values():
        per_axis[s["axis"]] = per_axis.get(s["axis"], 0) + 1
    return bad, per_axis


# ------------------------------------------ fitting the lens, and trimming --
def fit_distortion(traced, cam, free_centre=False, verbose=False):
    """Stage B - plumb line: choose k so that every traced line is straight.

    No homography is involved. "Straight in the world" constrains the lens
    alone, and ~800 samples pin 2-3 parameters extremely well.
    """
    # Same hoist as `resid_for`: the samples never move, so normalise them and
    # take their radii once instead of once per numerical-Jacobian column.
    cut = np.cumsum([0] + [len(t["pts"]) for t in traced])
    allpts = np.concatenate([t["pts"] for t in traced])
    fixed = None if free_centre else cam.to_norm(allpts)
    fixed_rd = None if fixed is None else np.linalg.norm(fixed, axis=1)

    def resid(p):
        c = Camera(np.eye(3), p[:len(cam.k)], cam.model, cam.f,
                   p[-2] if free_centre else cam.cx, p[-1] if free_centre else cam.cy)
        xn = fixed if fixed is not None else c.to_norm(allpts)
        rd = fixed_rd if fixed_rd is not None else np.linalg.norm(xn, axis=1)
        xya = xn * c.undistort_scale(rd)[:, None]
        out = []
        for i in range(len(traced)):
            xy = xya[cut[i]:cut[i + 1]]
            m = xy - xy.mean(0)
            _, _, Vt = np.linalg.svd(m, full_matrices=False)
            out.append((m @ Vt[1]) * cam.f)          # perpendicular error, px
        return np.concatenate(out)

    x0 = list(cam.k) + ([cam.cx, cam.cy] if free_centre else [])
    x, r = lm_wrap(resid, np.array(x0, float), huber=3.0, verbose=verbose)
    return (x[:len(cam.k)],
            (x[-2], x[-1]) if free_centre else (cam.cx, cam.cy),
            float(np.sqrt(np.mean(r ** 2))))


def lm_wrap(fun, x0, **kw):
    return CM.lm(fun, x0, **kw)


def _line_residuals(cam, lines):
    """Signed perpendicular error, in pixels, of every traced sample."""
    out = []
    for t in lines:
        xy = cam.undistort(cam.to_norm(t["pts"]))
        a, b = cam.ideal(np.array([t["world_a"], t["world_b"]]))
        n = perp(unit(b - a))
        out.append(((xy - a) @ n) * cam.f)
    return out
