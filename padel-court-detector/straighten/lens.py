"""Blind fisheye undistortion from the straight edges the hall is made of.

No court model, no seeds, no colour, no line detection. The only assumption is
that the scene contains straight lines - which a padel hall is full of: glass
frames, posts, roof beams, court paint. A straight world line is straight in the
image only when the lens model is right, so sweep the lens parameters and keep
the ones that make the edges straightest.

Straightest, NOT most Hough-concentrated. Maximising Hough concentration is the
obvious objective and it is degenerate: it rewards any mapping that squashes the
interior of the frame, which is not the same thing as straightening it, and on
the reference plate it scored the wrong lens twice as high as the truth and
landed further from it than doing nothing at all. The plumb-line residual below
is the objective that works.

The lens model is

    theta   = r_ideal / f  ... via  r_ideal = f tan(theta)
    r_dist  = f * theta * (1 + k1 theta^2 + k2 theta^4)

Undistortion inverts it: pixel radius -> theta (Newton) -> f tan(theta).

Scale gauge. Undistorted coordinates are only defined up to a global scale (any
scale is absorbed by whatever homography follows), and an unconstrained scale
would let the optimiser cheat by shrinking every edge. Every candidate mapping
is therefore renormalised to leave the image-corner radius where it was, which
pins the gauge exactly.

Do not compare two cameras by their coefficients. (f, k1, k2) is badly
degenerate: over 103 plates of one camera model, f spans 575-1626 and k1 spans
-0.47 to +0.29 with sign changes, while the radial mapping those triples produce
agrees to about 3%. Two plates of one camera came back as f=1626, k1=-0.47 and
f=840, k1=+0.12 with mappings 13 px apart at r=400. Compare `displacement()`.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------- lens model

def radial(th, k):
    """1 + k1 t^2 + k2 t^4 + ... for as many coefficients as were given."""
    s = np.ones_like(th)
    p = np.ones_like(th)
    for ki in k:
        p = p * th ** 2
        s = s + ki * p
    return s


def dradial(th, k):
    """d/dtheta of theta * radial(theta)."""
    d = np.ones_like(th)
    for i, ki in enumerate(k):
        d = d + (2 * i + 3) * ki * th ** (2 * i + 2)
    return d


def theta_of(rd, f, k, iters=12):
    """distorted pixel radius -> incidence angle theta (Newton).

    Stops when the update is below an ulp of theta, which on this model is the
    third or fourth pass - the remaining eight moved nothing and this is the
    inner loop of every objective in the package.
    """
    t = np.asarray(rd, float) / f
    th = t.copy()
    for i in range(iters):
        # `radial` and `dradial` written out together: they walk the same powers
        # of theta, and this loop is the inner loop of every objective here.
        t2 = th ** 2
        p, s, d = t2, 1.0, 1.0
        for j, kj in enumerate(k):
            if j:
                p = p * t2
            s = s + kj * p
            d = d + (2 * j + 3) * kj * p
        step = (th * s - t) / np.where(np.abs(d) < 1e-9, 1e-9, d)
        th -= step
        # Never converged before the third pass, so the test is not worth its
        # own three passes over the array until then.
        if i >= 2 and not np.any(np.abs(step) > 1e-13):
            break
    return th


def r_ideal(rd, f, k, th_max=1.45):
    """distorted pixel radius -> ideal (pinhole) pixel radius."""
    return f * np.tan(np.clip(theta_of(rd, f, k), -th_max, th_max))


def gauge(f, k, Rref):
    """Scale that leaves radius Rref fixed under the undistortion."""
    return Rref / float(r_ideal(np.array([Rref], float), f, k)[0])


def undistort_pts(pts, c, f, k, Rref):
    """Pixels -> centred, gauge-fixed undistorted pixels."""
    d = np.atleast_2d(np.asarray(pts, float)) - c
    rd = np.hypot(d[:, 0], d[:, 1])
    s = gauge(f, k, Rref) * r_ideal(rd, f, k) / np.maximum(rd, 1e-9)
    return d * np.where(rd < 1e-9, 1.0, s)[:, None]


def undistort_maps(shape, c, f, k, Rref, zoom=1.0, canvas=None):
    """cv2.remap maps that straighten a whole frame.

    The gauge that pins Rref is the right one for fitting but a poor one for
    looking at: it holds the corners still, so everything inside is squeezed.
    `zoom` scales the output about the centre and `canvas` gives it room to
    spread into; zoom = 1/gauge with a bigger canvas keeps the middle of the
    frame at its original resolution.
    """
    H, W = shape[:2]
    OH, OW = canvas if canvas else (H, W)
    co = np.array([OW / 2.0, OH / 2.0])
    gx, gy = np.meshgrid(np.arange(OW, dtype=np.float64), np.arange(OH, dtype=np.float64))
    dx, dy = gx - co[0], gy - co[1]
    rout = np.hypot(dx, dy) / zoom                     # radius in gauge units
    ri = rout / gauge(f, k, Rref)                      # -> f tan(theta)
    th = np.arctan(ri / f)
    rd = f * th * radial(th, k)                        # forward model -> source radius
    s = np.where(rout < 1e-9, 1.0 / zoom, rd / np.maximum(rout * zoom, 1e-9))
    return (dx * s + c[0]).astype(np.float32), (dy * s + c[1]).astype(np.float32)


def fwd_map(pts, c, f, k, Rref, zoom=1.0, canvas=None, shape=None):
    """Source pixels -> rectified-image pixels, matching undistort_maps()."""
    H, W = (canvas if canvas else shape[:2])
    co = np.array([W / 2.0, H / 2.0])
    return undistort_pts(pts, c, f, k, Rref) * zoom + co


def inv_map(pts, c, f, k, Rref, zoom=1.0, canvas=None, shape=None):
    """Rectified-image pixels -> source pixels. Inverse of fwd_map.

    Needed to hand something found in the rectified frame back to a solver that
    works on the raw one: the rectification is a viewing aid, the calibration
    still has to be expressed against the pixels the camera actually produced.
    """
    H, W = (canvas if canvas else shape[:2])
    co = np.array([W / 2.0, H / 2.0])
    d = np.atleast_2d(np.asarray(pts, float)) - co
    rout = np.hypot(d[:, 0], d[:, 1])
    ri = rout / (zoom * gauge(f, k, Rref))
    th = np.arctan(ri / f)
    rd = f * th * radial(th, k)
    s = np.where(rout < 1e-9, 1.0, rd / np.maximum(rout, 1e-9))
    return d * s[:, None] + c


# ---------------------------------------------------------------- edge input

def canny(img, thr=255, aperture=3):
    """Strict Canny. Both thresholds at the same value = no hysteresis, only
    edges that clear the bar on their own. A 3x3 L2 Sobel magnitude tops out
    near 1442, so 255 is a high bar and deliberately so: the fit wants a few
    hundred unimpeachable arcs, not every edge in the frame."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(g, thr, thr, apertureSize=aperture, L2gradient=True)


#: (width, height) of the erased block at the two top corners and the two
#: bottom corners of the frame.
CORNERS = ((400, 200), (250, 200))


def blank_corners(edge, corners=CORNERS):
    """Zero the edge map in the four corners.

    The corners are where a broadcast frame keeps its furniture: channel
    banner, scoreboard, watermark. Those graphics are composited in image space,
    so they are straight in the DISTORTED frame by construction, and a plumb-
    line fit that believes them is being asked to leave the lens uncorrected
    exactly where the lens does most. They also sit at the largest radius, which
    is where each arc counts for most.
    """
    e = edge.copy()
    H, W = e.shape[:2]
    (tw, th), (bw, bh) = corners
    e[:th, :tw] = 0
    e[:th, W - tw:] = 0
    e[H - bh:, :bw] = 0
    e[H - bh:, W - bw:] = 0
    return e


def arcs(edge, span=384, m=48, min_extent=25.0, max_bend=0.15):
    """Edge map -> (K, m, 2) stack of edge arcs that could be bits of a line.

    Contours of a thin edge map trace each edge out and back, so every arc gets
    sampled roughly twice; that costs nothing but a factor in the weights. Arcs
    are cut to a fixed point count so the whole set evaluates as one tensor.

    `max_bend` pre-filters in the *distorted* image, so it must stay loose: a
    long arc across the corner of a fisheye frame is genuinely bent, and those
    are exactly the arcs that carry the signal. Screening at 0.06 throws away
    the long peripheral arcs that matter most. It is here to drop blobs, text
    and player outlines - things no lens model could have straightened.
    """
    cs, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    step, stride = max(1, span // m), max(1, span // 2)
    out = []
    for c in cs:
        p = c[:, 0, :].astype(np.float64)
        for s in range(0, len(p) - span + 1, stride):
            out.append(p[s:s + span:step][:m])
    if not out:
        return np.zeros((0, m, 2)), np.zeros(0)
    A = np.stack(out)
    ext, bend = _shape(A)
    keep = (ext >= min_extent) & (bend <= max_bend)
    return A[keep], ext[keep]


def arc_set(edge, spans=(192, 384, 768), **kw):
    """Arcs at several lengths at once.

    Short arcs are numerous and spread over the whole frame; long ones are few
    but each says far more about the lens, since the sag of a straight line
    grows with the SQUARE of its length. Taking both, weighted by extent^2, beat
    every single-length set that was tried - 3.9 px against 21 px for the first.
    """
    As, ws = [], []
    for s in spans:
        A, w = arcs(edge, span=s, **kw)
        if len(A):
            As.append(A)
            ws.append(w)
    if not As:
        return np.zeros((0, 48, 2)), np.zeros(0)
    return np.concatenate(As), np.concatenate(ws) ** 2


def _shape(A):
    """Per-arc (extent, bend) = (sqrt of major eigenvalue, minor/major ratio)."""
    D = A - A.mean(1, keepdims=True)
    x, y = D[:, :, 0], D[:, :, 1]
    m = A.shape[1]
    # The 2x2 covariance written out. `einsum` on this shape falls back to its
    # own C loop, and this is evaluated a few hundred times per solve.
    cxx = (x * x).sum(1) / m
    cyy = (y * y).sum(1) / m
    cxy = (x * y).sum(1) / m
    tr = cxx + cyy
    det = cxx * cyy - cxy ** 2
    dsc = np.sqrt(np.maximum(tr ** 2 - 4 * det, 0.0))
    lo, hi = (tr - dsc) / 2, (tr + dsc) / 2
    hi = np.maximum(hi, 1e-12)
    return np.sqrt(hi), np.sqrt(np.maximum(lo, 0.0) / hi)


def _sagitta(A):
    """Signed bow of each arc: how far its middle sits off its own chord.

    From a least-squares quadratic in the arc's own frame, not from the three
    points at the ends and the middle. Those three are the most fragile choice
    available - one stray endpoint in a scattered walk (the far service line
    routinely has a few) and the reading is nonsense, with a plausible magnitude
    and an arbitrary sign.

    Signed on purpose. The failure the court evidence exists to prevent is not a
    large residual but a residual of the WRONG SIGN - a line that was bowed one
    way and comes out bowed the other, which every unsigned measure calls an
    improvement.
    """
    D = A - A.mean(1, keepdims=True)
    C = np.einsum("kmi,kmj->kij", D, D)
    out = np.zeros(len(A))
    for i in range(len(A)):
        _, V = np.linalg.eigh(C[i])
        u = D[i] @ V[:, 1]                 # along the arc
        v = D[i] @ V[:, 0]                 # across it
        out[i] = -np.polyfit(u, v, 2)[0] * ((u.max() - u.min()) / 2) ** 2
    return out


# ---------------------------------------------------------------- the metric

def bend_cost(A, w, c, Rref, cap=0.02):
    """Plumb-line objective: weighted mean arc bend, truncated.

    bend = (rms distance off the arc's own best-fit line) / (arc extent), so it
    is invariant to the scale of the undistorted frame - the shrink-everything
    cheat that breaks a Hough score has no purchase here.

    Two details that mattered more than expected. The weights come from the
    *distorted* image and never move, so the optimiser cannot down-weight an arc
    it fails to straighten by shrinking it. And truncating at `cap` makes the
    genuinely-curved arcs - heads, balls, cables, painted logos - contribute a
    constant instead of steering the fit; they are not outliers to down-weight,
    they are curves, and no lens model will ever straighten them.

    `cap` may be per-arc, and has to be: court paint needs a much looser one
    than a hall arc. A 1500 px court line bowed by 30 px already reads bend
    0.023, past the default truncation, so at one shared cap the very evidence
    that matters most arrives saturated and never pulls at all.
    """
    K, m, _ = A.shape
    wn = w / w.sum()
    cap = np.broadcast_to(np.asarray(cap, float), (K,))
    # `undistort_pts`, with the half of it that does not depend on the lens
    # lifted out: the arcs do not move, so their offsets from the centre and
    # their radii are the same on all several hundred evaluations of this.
    d = A.reshape(-1, 2) - c
    rd = np.hypot(d[:, 0], d[:, 1])
    rsafe = np.maximum(rd, 1e-9)
    at_centre = rd < 1e-9

    def cost(p):
        f, k = float(np.exp(p[0])), np.asarray(p[1:], float)
        if not np.isfinite(f) or f < 50 or f > 1e6:
            return 1e9
        s = gauge(f, k, Rref) * r_ideal(rd, f, k) / rsafe
        xy = d * np.where(at_centre, 1.0, s)[:, None]
        if not np.isfinite(xy).all():
            return 1e9
        _, bend = _shape(xy.reshape(K, m, 2))
        return float((wn * np.minimum(bend, cap)).sum())

    return cost


# ---------------------------------------------------------------- optimiser

def nelder_mead(cost, x0, step, iters=200, tol=1e-7):
    """Downhill simplex. Three parameters do not need scipy."""
    x0 = np.asarray(x0, float)
    n = len(x0)
    sim = np.vstack([x0] + [x0 + np.eye(n)[i] * step[i] for i in range(n)])
    val = np.array([cost(s) for s in sim])
    for _ in range(iters):
        o = np.argsort(val)
        sim, val = sim[o], val[o]
        if abs(val[-1] - val[0]) < tol * (abs(val[0]) + tol):
            break
        cen = sim[:-1].mean(0)
        xr = cen + (cen - sim[-1])
        vr = cost(xr)
        if vr < val[0]:
            xe = cen + 2.0 * (cen - sim[-1])
            ve = cost(xe)
            sim[-1], val[-1] = (xe, ve) if ve < vr else (xr, vr)
        elif vr < val[-2]:
            sim[-1], val[-1] = xr, vr
        else:
            xc = cen + 0.5 * (sim[-1] - cen)
            vc = cost(xc)
            if vc < val[-1]:
                sim[-1], val[-1] = xc, vc
            else:
                sim[1:] = sim[0] + 0.5 * (sim[1:] - sim[0])
                val[1:] = [cost(s) for s in sim[1:]]
    o = np.argsort(val)
    return sim[o][0], val[o][0]


#: Truncation for the court's own arcs, in the same sag/extent units as `cap`.
#:
#: Uncapped is wrong, and it is the first thing to try. The court's lines are
#: not straight to begin with: undistorted with a CLICK-CALIBRATED lens they
#: still bow 0.5-3 px, at both reference venues, on every line - the detector has
#: a systematic error along a line that no lens explains, and the paint itself is
#: laid by hand. Told to make them straight, the fit answers by bending the lens,
#: and rotonde goes from 12.3 px out to 30.
#:
#: So the cap is what says how straight a court line is EXPECTED to be, and
#: everything past it stops pulling. 0.002 is "about 3 px of sag on a 1500 px
#: line", which is where the measured bow of this detector's lines sits. It was
#: chosen on the one number that separates it from 0.001: the worst
#: disagreement between two plates of one fixed camera, 56 px at 0.001 against
#: 38 at 0.002, because at the tighter cap there are plates whose court is
#: truncated away entirely and which fall back to the blind answer. On
#: everything else the two are within a pixel of each other.
COURT_CAP = 0.002

#: The share of the total arc weight the court gets, whatever the arc counts
#: happen to be. A share rather than a multiplier because the number of hall
#: arcs varies by a factor of three across the corpus, so a multiplier means
#: something different on every plate. The court owns the middle of the frame
#: and the building owns the periphery, and neither can be allowed to silence
#: the other: all-court leaves the corners unconstrained, all-building is the
#: failure the court evidence is here to fix.
#:
#: Six lines against three hundred arcs, so this is a large multiplier and it
#: has to be. Raising it further does not keep helping: at 0.25 rotonde goes
#: from 8.5 px out to 32, because the court's own residual bow (see `COURT_CAP`)
#: is then a bigger vote than the whole building.
COURT_FRAC = 0.15


def solve(img, nk=2, thr=255, cap=0.02, f_grid=(500, 2500, 15),
          k_grid=(-0.03, 0.03, 7), corners=CORNERS, log=None, court=True,
          court_frac=COURT_FRAC, court_cap=COURT_CAP, court_lines=None,
          **arc_kw):
    """Image -> (f, k, report). Coarse grid on (f, k1), then simplex on all.

    A padel hall is full of straight lines that are not court paint - glass
    frames, posts, roof trusses. They all constrain the lens just as well, and
    the arc stage neither knows nor cares which is which.

    `court=True` additionally walks the court's own lines (`straighten.court`,
    which calls `padelcourt.walk.detect`) and adds them to the same objective at
    a looser cap and a fixed share of the weight. They are the only straight
    things in frame that are certainly straight and certainly not composited,
    and they are the lines the answer will eventually be judged against. It
    costs the walk - seconds, not milliseconds - and falls back silently to the
    hall arcs alone when the walk cannot start. `court_lines` accepts a walk
    already done, as {name: points}, so a caller that walks anyway pays once.
    """
    H, W = img.shape[:2]
    c = np.array([W / 2.0, H / 2.0])
    Rref = float(np.hypot(W, H) / 2.0)
    e = canny(img, thr)
    if corners:
        e = blank_corners(e, corners)
    A, w = arc_set(e, **arc_kw)
    if len(A) < 20:
        raise RuntimeError(f"only {len(A)} usable arcs - image too plain or thr too high")
    n_hall = len(A)
    caps = np.full(n_hall, float(cap))
    court_names = []
    if court or court_lines is not None:
        from .court import court_arcs
        CA, cw, court_names = court_arcs(img, m=A.shape[1], lines=court_lines,
                                         verbose=log is not None)
        if len(CA):
            cw = cw / cw.sum() * (court_frac / (1.0 - court_frac)) * w.sum()
            A = np.concatenate([A, CA])
            w = np.concatenate([w, cw])
            caps = np.concatenate([caps, np.full(len(CA), float(court_cap))])
            if log:
                log(f"  court lines: {' '.join(court_names)}"
                    f"  ({100 * cw.sum() / w.sum():.0f}% of the weight)")
        elif log:
            log("  court lines: none - the hall's edges alone")
    cost = bend_cost(A, w, c, Rref, caps)

    best, bv = None, np.inf
    for f in np.exp(np.linspace(np.log(f_grid[0]), np.log(f_grid[1]), f_grid[2])):
        for k1 in np.linspace(k_grid[0], k_grid[1], k_grid[2]):
            p = np.array([np.log(f), k1] + [0.0] * (nk - 1))
            v = cost(p)
            if v < bv:
                best, bv = p, v
    if log:
        log(f"  arcs {n_hall}+{len(court_names)}  grid best "
            f"f={np.exp(best[0]):7.1f} k1={best[1]:+.4f} bend={bv*1e3:.3f}e-3")

    step = np.array([0.10, 0.006] + [0.002] * (nk - 1))
    p, v = nelder_mead(cost, best, step, iters=500)
    f, k = float(np.exp(p[0])), np.asarray(p[1:], float)

    # readable version of the same thing: rms sag, in pixels, over the arcs the
    # solution actually straightened. Extent is the one measured in the source
    # frame, so before and after are quoted in the same units. Long arcs only -
    # a 60-px arc is straight under any lens, and including those buries the
    # signal under a pile of numbers that never had any.
    W1 = undistort_pts(A.reshape(-1, 2), c, f, k, Rref).reshape(A.shape)
    ext0, bend0 = _shape(A)
    ext1, bend1 = _shape(W1)
    ok = np.zeros(len(A), bool)              # hall arcs only, and they come first
    ok[:n_hall] = ((bend1[:n_hall] < cap)
                   & (ext0[:n_hall] >= np.percentile(ext0[:n_hall], 75)))
    rep = {"edges": int((e > 0).sum()), "arcs": n_hall, "straight": int(ok.sum()),
           "bend": float(v),
           "sag_before": float(np.median((bend0 * ext0)[ok])) if ok.any() else float("nan"),
           "sag_after": float(np.median((bend1 * ext0)[ok])) if ok.any() else float("nan"),
           "c": c, "Rref": Rref, "edge_map": e, "court": {}}

    # What the court paint did, quoted in source pixels: signed sagitta before
    # and after. This is the one number that says whether the solve helped the
    # thing the calibration will be measured against, and it is reported whether
    # or not the court steered the fit - a line that comes back bowed the other
    # way is the failure mode, and only the sign shows it.
    if court_names:
        s0 = _sagitta(A[n_hall:])
        s1 = _sagitta(W1[n_hall:]) * ext0[n_hall:] / np.maximum(ext1[n_hall:], 1e-9)
        rep["court"] = {n: (float(a), float(b)) for n, a, b in zip(court_names, s0, s1)}
    return f, k, rep
