"""What the detector reads off the pixels.

The court's own colour, the directional top-hats that separate paint from
surface, the ridge and edge peak finders the walks re-centre on, the RANSAC trim
that says which samples lie on one curve, and the maps that take the lens out of
a frame. Everything here is a measurement on an image; nothing here knows what a
court is beyond `court.py`'s dimensions.
"""
import math

import cv2
import numpy as np

from . import court as CM
from .court import perp


# ------------------------------------------------------------- 1. colour ----
def court_colour(bgr, w_frac=0.30, h_frac=0.30, cy_frac=0.58, n=1000, seed=0):
    """Robust (location, inverse-covariance) of the surface colour in Lab chroma.

    The sample box is mostly court but never purely court - the centre line
    crosses it, and a shadow or a ball may too - so a plain mean is pulled off
    the true colour. This is an iteratively trimmed estimator: start from the
    coordinatewise median, keep the closest 70% by Mahalanobis distance,
    re-estimate, repeat. Cheap stand-in for MCD, and with ~5% outliers it lands
    in the same place.

    Chroma only (a, b): these cameras vignette hard and a floodlit court runs
    from near-white to near-black in L, but the hue barely moves.
    """
    h, w = bgr.shape[:2]
    x0, x1 = int(w * (0.5 - w_frac / 2)), int(w * (0.5 + w_frac / 2))
    y0, y1 = int(h * (cy_frac - h_frac / 2)), int(h * (cy_frac + h_frac / 2))
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 1.0), cv2.COLOR_BGR2LAB)
    patch = lab[y0:y1, x0:x1, 1:].reshape(-1, 2).astype(np.float64)
    rng = np.random.RandomState(seed)
    sel = patch[rng.choice(len(patch), min(n, len(patch)), replace=False)]

    mu = np.median(sel, axis=0)
    S = np.cov((sel - mu).T) + np.eye(2) * 1e-3
    for _ in range(12):
        d = np.einsum("ij,jk,ik->i", sel - mu, np.linalg.inv(S), sel - mu)
        keep = sel[d <= np.quantile(d, 0.70)]
        if len(keep) < 20:
            break
        mu_new = keep.mean(axis=0)
        S_new = np.cov((keep - mu_new).T) + np.eye(2) * 1e-3
        if np.allclose(mu_new, mu, atol=1e-4):
            mu, S = mu_new, S_new
            break
        mu, S = mu_new, S_new
    # trimming shrinks the scatter; undo it so the threshold means what it says
    S = S / 0.55
    return mu, np.linalg.inv(S), (x0, y0, x1, y1)


# -------------------------------------------------------------- 3. paint ----
def directional_tophat(bgr, horizontal=True, length=27):
    """Isolate horizontal OR vertical paint.

    Opening with a tall thin kernel erases horizontal lines and leaves vertical
    ones, so subtracting it isolates the horizontal paint. Needed because an
    isotropic top-hat lights up both, and a column crossing the centre line then
    responds along its entire length.
    """
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                  (1, length) if horizontal else (length, 1))
    return cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k).astype(np.float32)


def bilinear(img, pts):
    """Sample an image at float coordinates; NaN outside. -> (N,) or (N, C)

    Multi-channel images are sampled in one pass rather than a plane at a time.
    The four corner lookups and the weights are the same work for one channel or
    three, and this runs some ten thousand times per frame on a handful of
    points each, where the per-call overhead is the whole cost.
    """
    x, y = pts[:, 0], pts[:, 1]
    h, w = img.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    # In bounds is exactly "the clip did nothing", which is two comparisons
    # instead of four and matters at thirty numpy calls a sample.
    ok = (x0 == x0c) & (y0 == y0c)
    fx, fy = x - x0c, y - y0c
    gx, gy = 1 - fx, 1 - fy
    x1c, y1c = x0c + 1, y0c + 1
    a, b = img[y0c, x0c], img[y0c, x1c]
    c, d = img[y1c, x0c], img[y1c, x1c]
    if img.ndim == 3:
        fx, gx, fy, gy = fx[:, None], gx[:, None], fy[:, None], gy[:, None]
        ok = ok[:, None]
    v = a * gx * gy + b * fx * gy + c * gx * fy + d * fx * fy
    return v if ok.all() else np.where(ok, v, np.nan)


def bilinear_at(img, x, y):
    """`bilinear` for a single point of a single-channel image. -> float or nan

    Same four neighbours and the same weights in the same order, in plain
    Python. The walkers read one pixel per step and a numpy call on one point is
    thirty times the arithmetic it performs.
    """
    h, w = img.shape[:2]
    x0, y0 = math.floor(x), math.floor(y)
    x0c = 0 if x0 < 0 else (w - 2 if x0 > w - 2 else x0)
    y0c = 0 if y0 < 0 else (h - 2 if y0 > h - 2 else y0)
    if x0 != x0c or y0 != y0c:
        return math.nan
    fx, fy = x - x0c, y - y0c
    gx, gy = 1 - fx, 1 - fy
    return (float(img[y0c, x0c]) * gx * gy + float(img[y0c, x0c + 1]) * fx * gy +
            float(img[y0c + 1, x0c]) * gx * fy + float(img[y0c + 1, x0c + 1]) * fx * fy)


def line_response(bgr):
    """Bright-thin-structure response: white paint stands out, court does not.

    A white top-hat with a kernel wider than the paint keeps the lines and
    removes the court surface, floodlights and shadow gradients, so the same
    contrast threshold works on a sunlit blue court and a floodlit one.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    return cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k).astype(np.float32)


def surface_score(bgr, seed_pts, softness=0.6):
    """Per-pixel "is this the playing surface" score in [0, 1].

    The colour model is seeded from inside the clicked keypoint hull rather than
    hardcoded, so a blue, green or terracotta court all work with no tuning.
    Only the Lab chroma channels are used: these cameras vignette hard and a
    floodlit court runs from near-white under the lamps to near-black at the
    near baseline, but the hue barely moves. The score is deliberately soft, so
    the 0.5 crossing at the wall base interpolates to sub-pixel.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 1.0), cv2.COLOR_BGR2LAB)
    ab = lab[:, :, 1:].astype(np.float32)
    hull = cv2.convexHull(np.asarray(seed_pts, np.float32).reshape(-1, 1, 2))
    inside = np.zeros(bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, hull.astype(np.int32), 255)
    inside = cv2.erode(inside, np.ones((25, 25), np.uint8))
    sel = ab[inside > 0]
    if len(sel) < 500:
        raise SystemExit("clicked points enclose too little area to model the surface")
    med = np.median(sel, axis=0)
    mad = np.median(np.abs(sel - med), axis=0) + 1.0     # paint is a minority: median survives it
    dist = np.abs(ab - med).max(axis=2) / (3.0 * mad.max())
    score = 1.0 / (1.0 + np.exp((dist - 1.0) / softness))
    # The paint is achromatic, so it scores as non-surface and carves the court
    # into pieces. Close it back up, with a kernel wider than the paint but
    # narrow enough that the corners only round by a few pixels.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(score.astype(np.float32), cv2.MORPH_CLOSE, k)


#: How much lighter than ITS OWN SURROUNDINGS a pixel has to be to count as
#: paint for the edge map, in L units, and how wide a neighbourhood that is
#: measured over.
#:
#: A single threshold taken from the court's median L cannot work, and the
#: reason is not venue-to-venue variation - it is variation across one frame.
#: On `08-06-59` the near half's paint reads L=237 while the far half's reads
#: 130 against a floor of 110: a gate set to see the near line is 30 units above
#: the far line, and a gate set to see the far line floods the near half. That
#: one gate is why the far service line was invisible to the Hough on the plates
#: where it matters most - the far half is exactly where the light runs out.
#:
#: Against its own neighbourhood the same line stands +20, and the near line
#: stands +100; one number covers both. The neighbourhood is a median wide
#: enough that the widest painted line is a minority of it - the near service
#: line runs ~18px here, so a 51px window is two thirds floor even straddling
#: it - and narrow enough to follow the lighting across the frame.
#:
#: Swept over the 90 plates whose far service line was walked well enough to be
#: ground truth, counting the plates where the line is among the candidates an
#: unstoppable walk collects: the old global gate finds it on 85, this finds it
#: on 89. See HZ_VOTES for the other half of that measurement.
WHITE_OVER, WHITE_K = 12.0, 51


def white_edges(bgr, over=WHITE_OVER, k=WHITE_K):
    """Strict edges of whatever is lighter than the floor around it.

    Everything not standing clear of its own neighbourhood is zeroed before
    Canny sees it, so the only edges left are the borders of paint, net tape and
    glare - and then the thresholds are set as high as Canny allows, so that
    what survives is a step, not a gradient.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 1.0), cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    floor = cv2.medianBlur(L, k)
    white = np.where(L.astype(np.int16) - floor.astype(np.int16) >= over,
                     L, 0).astype(np.uint8)
    return cv2.Canny(white, 255, 255, apertureSize=3, L2gradient=True)


#: Profile stations by half-width. The perpendicular probes below run tens of
#: thousands of times per frame at a handful of distinct widths, and rebuilding
#: the same `arange` each time was a measurable share of the walk.
_STATIONS = {}


def stations(half):
    s = _STATIONS.get(half)
    if s is None:
        if len(_STATIONS) > 512:
            _STATIONS.clear()
        s = _STATIONS[half] = np.arange(-half, half + 0.001, 0.5)
        s.flags.writeable = False
    return s


def polyfit_fast(x, y, deg):
    """`np.polyfit(x, y, deg)`, arithmetic for arithmetic, minus the wrapper.

    Same column scaling and the same `lstsq` with the same `rcond`, so it
    returns the identical coefficients; what it skips is polyfit's own argument
    handling, which is about half of its cost and is paid a thousand times per
    frame by the walkers' shape test.
    """
    lhs = np.vander(x, deg + 1)
    scale = np.sqrt((lhs * lhs).sum(axis=0))
    lhs /= scale
    c = np.linalg.lstsq(lhs, y, rcond=len(x) * np.finfo(np.float64).eps)[0]
    return c / scale


def distinct_at_least(v, n):
    """Does `v` hold at least `n` distinct values? `len(np.unique(v)) >= n`."""
    if len(v) < n:
        return False
    if len(v) <= 8:                             # a RANSAC's minimal sample
        return len(set(v.tolist())) >= n
    return int((np.diff(np.sort(v)) != 0).sum()) >= n - 1


def _mid(v):
    """`np.median(v)` for a 1-D array, without the dispatch around it."""
    n = len(v)
    p = np.partition(v, [(n - 1) // 2, n // 2])
    return (p[(n - 1) // 2] + p[n // 2]) / 2.0


def _peak_ridge(field, c, n, half, min_contrast):
    """Sub-pixel crest of the NEAREST qualifying bright line along c + s*n.

    Nearest, not brightest. Over a wide acquisition window the brightest ridge
    is often the wrong one - the net's tape band outshines the far service line
    sitting 25px away - and a global argmax silently walks the whole trace onto
    it. Taking the closest local maximum above the contrast floor mirrors the
    nearest-crossing rule used for boundaries and removes the brightness/
    distance tradeoff entirely.
    """
    s = stations(half)
    prof = bilinear(field, c + s[:, None] * n)
    if np.isnan(prof).any():
        return None
    prof = prof - _mid(prof)
    inner = prof[1:-1]
    loc = np.flatnonzero((inner >= min_contrast) & (inner >= prof[:-2])
                         & (inner > prof[2:])) + 1
    if not len(loc):
        return None
    # One candidate is the usual case, and picking the nearest of one costs
    # four numpy calls that these walks make tens of thousands of times.
    i = int(loc[0]) if len(loc) == 1 else int(loc[np.argmin(np.abs(s[loc]))])
    y0, y1, y2 = float(prof[i - 1]), float(prof[i]), float(prof[i + 1])
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
    return c + (s[i] + min(1.0, max(-1.0, d)) * 0.5) * n


def _peak_edge(field, c, n, half, inward):
    """Sub-pixel 0.5 crossing of the surface score, i.e. the wall base.

    `n` is oriented so that -n points into the court. Several crossings can fall
    in the window (a fence post, a kerb); the one nearest the predicted position
    is taken, which is what keeps the trace on the true boundary.
    """
    s = stations(half)
    prof = bilinear(field, c + s[:, None] * n)
    if np.isnan(prof).any():
        return None
    if inward < 0:
        s, prof = s[::-1], prof[::-1]        # index 0 always inside the court
    hi = prof >= 0.5
    cross = np.where(hi[:-1] & ~hi[1:])[0]   # inside -> outside transitions
    if not len(cross):
        return None
    i = cross[np.argmin(np.abs(s[cross]))]
    # Demand a decisive local swing. In the vignetted corners the chroma washes
    # out and the score drifts through 0.5 without a real edge there, which
    # produced 20px outliers; a marginal crossing is worse than no sample.
    lo, hi_ = max(0, i - 8), min(len(prof), i + 9)
    win = prof[lo:hi_]
    if win.max() < 0.7 or win.min() > 0.3:
        return None
    a, b = prof[i], prof[i + 1]
    frac = (a - 0.5) / (a - b) if abs(a - b) > 1e-9 else 0.0
    return c + (s[i] + frac * (s[i + 1] - s[i])) * n


def trace_segment(field, A, B, kind=CM.PAINT, inward=1.0, step=5.0, half=14.0,
                  acquire=50.0, min_contrast=12.0, extend=0.0, shape=None):
    """Walk A->B (plus `extend` px beyond B) recording the line position.

    Search is perpendicular to the straight A->B direction, but centred on a
    running perpendicular offset, so a line that bows 65px across the frame is
    still tracked with a +-14px window: the offset absorbs the curvature and
    only the residual has to be found. Error *along* the line is irrelevant,
    which is why sloppy 10px clicks still yield sub-pixel samples.

    The window opens up to `acquire` while cold or after a run of misses. A
    click that is 50px off its line - which is roughly what hand-clicking a
    fisheye frame gives you - otherwise never gets a first hit, and with no
    first hit the running offset never starts and the whole line is lost.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    L = float(np.linalg.norm(B - A))
    if L < 5:
        return np.zeros((0, 2))
    d0 = (B - A) / L
    n0 = perp(d0)
    h, w = (shape[:2] if shape is not None else field.shape[:2])
    out, off, misses = [], 0.0, 0
    t = step
    while t <= L + extend:
        c = A + d0 * t + n0 * off
        if not (0 <= c[0] < w and 0 <= c[1] < h):
            break
        win = acquire if not out else min(acquire, half + 10.0 * misses)
        if kind == CM.BOUNDARY:
            p = _peak_edge(field, c, n0, win, inward)
        else:
            p = _peak_ridge(field, c, n0, win, min_contrast)
        if p is None:
            misses += 1
            if misses > 8 and t > L:      # past the clicked span and lost: stop
                break
        else:
            off = 0.5 * off + 0.5 * float((p - (A + d0 * t)) @ n0)
            out.append(p)
            misses = 0
        t += step
    return np.array(out) if out else np.zeros((0, 2))


def interpolating_fits(nodes, values):
    """One exact polynomial per row of (nodes, values). -> (coeffs, usable)

    `nodes` is (K, deg+1), so each fit is interpolation rather than least
    squares and the whole batch is a single stacked solve. That is the point:
    the RANSACs here draw hundreds of minimal samples apiece and used to spend
    all of their time inside `polyfit`, which is a least-squares problem set up
    and solved from scratch every time.

    Rows with a repeated node have no unique fit; they come back zero and
    `usable` False, which is the hypothesis those loops used to skip.
    """
    K, m = nodes.shape
    s = np.sort(nodes, axis=1)                  # distinct nodes, row by row
    usable = (np.diff(s, axis=1) != 0).all(axis=1)
    C = np.zeros((K, m))
    if usable.any():
        V = nodes[usable][:, :, None] ** np.arange(m - 1, -1, -1)
        try:
            C[usable] = np.linalg.solve(V, values[usable][:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:           # distinct nodes, so not expected
            usable[:] = False
    return C, usable & np.isfinite(C).all(axis=1)


def _best_by_inliers(C, usable, a, b, tol):
    """The hypothesis of `C` with the most inliers. -> (count, mask or None)

    `argmax` keeps the first of any equal counts, which is the hypothesis a
    strictly-greater running comparison over the same order would have kept.
    """
    Va = a[:, None] ** np.arange(C.shape[1] - 1, -1, -1)
    counts = np.where(usable, (np.abs(b - C @ Va.T) <= tol).sum(1), -1)
    n = int(counts.max()) if len(counts) else 0
    if n <= 0:
        return n, None
    return n, np.abs(b - np.polyval(C[int(np.argmax(counts))], a)) <= tol


def trim_to_curve(pts, swap, tol=3.0, deg=2, iters=400, seed=0):
    """Keep only the points that lie on one smooth curve, by RANSAC.

    A quadratic is enough: the lens bows a straight line by up to 65px, but it
    bows it smoothly, so anything off that curve is a different feature.

    RANSAC rather than iterative trimming, because the contamination is not a
    sprinkle of outliers. A region boundary leaks into the dark vignetted frame
    corner, where "leftmost court pixel" has nothing to do with the sideline,
    and the left sideline came out as a 50/50 mixture of two features - a
    quadratic rms of 55px. Trimming starts from a fit to that mixture and never
    escapes it; RANSAC only ever fits 3 points at a time, so the true curve wins
    on inlier count.
    """
    if len(pts) < 12:
        return pts
    a = pts[:, 1] if swap else pts[:, 0]
    b = pts[:, 0] if swap else pts[:, 1]
    rng = np.random.RandomState(seed)
    n = len(pts)
    # All `iters` hypotheses at once. Drawing the samples still costs what it
    # always did - the draw order is the algorithm and must not change - but
    # fitting deg+1 points to a degree-deg curve is exact interpolation, so the
    # whole batch is one `solve` on stacked Vandermondes instead of `iters`
    # trips through `polyfit`, which was 400 least-squares problems per line.
    idx = np.array([rng.permutation(n)[:deg + 1] for _ in range(iters)])
    C, good = interpolating_fits(a[idx], b[idx])
    best_n, best = _best_by_inliers(C, good, a, b, tol)
    if best is None or best_n < 12:
        return pts
    keep = best
    for _ in range(3):                      # polish on the inlier set
        c = np.polyfit(a[keep], b[keep], deg)
        keep = np.abs(b - np.polyval(c, a)) <= tol
        if keep.sum() < deg + 2:
            break
    return pts[keep]


# ------------------------------------------- 4. the middle of the paint ----
#: How far either side of a walked sample its own line's edges are looked for,
#: in pixels. Has to clear the widest half-stripe in the frame - the near
#: service line runs 8-9 px across most of its length and up to 18 px where it
#: meets the anchor, the closest point on it to the camera - plus the couple of
#: pixels the walk itself may be off centre. It costs nothing to be generous:
#: the nearest edge is taken, so a window that reaches past the paint finds the
#: paint first anyway.
EDGE_HALF = 14.0

#: The narrowest stripe there is a middle to find in, in pixels, measured as the
#: fitted distance between its two edges. Below this the two edges are one
#: structure and the pair says nothing: the far service line measures 1.2 px
#: wide on all 116 plates, and its samples sit a median 0.12 px off the midpoint
#: of the two Canny pixels either side of them - there is no bias there to
#: remove, because a 1-2 px ridge has only one crest to find.
EDGE_MIN_WIDTH = 3.5

#: How many usable edge pairs a line needs, and how far a pair's WIDTH may sit
#: off the width curve before the pair is discarded.
#:
#: The tolerance is on the width and on nothing else, and that is the whole
#: reason it can be this tight. Writing a sample as the paint's centre plus the
#: follower's error, `p = c + e`, the two distances it measures are `a = w - e`
#: and `b = w + e`, so their SUM is `2w` with the error algebraically absent.
#: A width is therefore a property of the court - a painted stripe a few pixels
#: across, changing smoothly with distance - and 1.5 px off its own curve means
#: the pair is not measuring one stripe. A tolerance on either distance alone,
#: or on the midpoint, would instead be a statement about how noisy the walk is,
#: and would throw out exactly the samples worth correcting.
EDGE_MIN_N, EDGE_TOL = 24, 1.5

#: What the width curve has to come back as before it is believed, in pixels of
#: fit residual. Deliberately equal to `EDGE_TOL`, because this is not a second
#: opinion on the fit - the inliers cannot be further off than the tolerance
#: that selected them. It is here for the ONE path where they can: `trim_to_curve`
#: hands back its input unchanged when RANSAC cannot reach twelve inliers, so
#: "these are all on one curve" and "I could not find a curve in this" come back
#: the same length and only the residual tells them apart.
EDGE_MAX_RMS = 1.5

#: HOW FAR THIS PASS MAY MOVE A LINE is not a constant, and setting one was a
#: mistake this has already made once. 6 px looked generous against a median
#: move of 1.6 px and turned out to sit 0.2 px above the corpus maximum: the
#: centre line is ~10 px wide where it meets the near T and its walk latches
#: 4-5 px off centre there, so one slightly wider court would have crossed the
#: threshold and had its whole correction discarded - a cliff, not a guard.
#:
#: The geometry already bounds it. Both distances are measured OUTWARD from the
#: sample, so both are positive and the sample lies between the two edges by
#: construction; their midpoint is inside the paint, and the move cannot exceed
#: the stripe's own half-width however wide the stripe is. What is worth
#: refusing is a pair that does not measure the stripe, and `EDGE_TOL` refuses
#: it on the width - a statement about the paint rather than a number of pixels.


def _normals(pts, swap, deg=2):
    """Unit normals to the smooth curve through an ordered set of samples.

    From the fitted curve rather than from neighbouring samples: the walk
    re-centres up to 2 px sideways at every step, which over a 4 px step is tens
    of degrees, and a normal that noisy would scatter the edge distances by more
    than the bias they are being measured to remove.
    """
    u = pts[:, 1] if swap else pts[:, 0]
    v = pts[:, 0] if swap else pts[:, 1]
    try:
        dv = np.polyval(np.polyder(np.polyfit(u, v, deg)), u)
    except (np.linalg.LinAlgError, ValueError):
        dv = np.zeros(len(u))
    t = np.column_stack([np.ones_like(dv), dv])
    if swap:
        t = t[:, ::-1]
    t = t / np.linalg.norm(t, axis=1)[:, None]
    return np.column_stack([-t[:, 1], t[:, 0]])


def _edge_out(E, pts, n, half=EDGE_HALF, sub=0.25):
    """Distance from each sample to the nearest edge pixel along +n. NaN if none."""
    s = np.arange(sub, half + 1e-9, sub)
    P = pts[:, None, :] + s[None, :, None] * n[:, None, :]
    h, w = E.shape[:2]
    xi = np.clip(np.rint(P[:, :, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(P[:, :, 1]).astype(int), 0, h - 1)
    hit = (E[yi, xi] > 0) & (P[:, :, 0] >= 0) & (P[:, :, 0] < w) \
        & (P[:, :, 1] >= 0) & (P[:, :, 1] < h)
    out = np.full(len(pts), np.nan)
    got = hit.any(axis=1)
    out[got] = s[hit.argmax(axis=1)][got]
    return out


def centre_on_paint(E, pts, swap, half=EDGE_HALF, tol=EDGE_TOL, min_n=EDGE_MIN_N,
                    min_width=EDGE_MIN_WIDTH, max_rms=EDGE_MAX_RMS, trace=None):
    """Slide a walked line sideways onto the middle of its own paint.

    THE WALK DOES NOT FOLLOW THE MIDDLE OF A WIDE LINE, and on the two lines
    that are wide here it misses it by about 2 px for their entire length.
    `image._peak_ridge` takes the NEAREST local maximum of the top-hat, which is
    the right rule for telling one line from its neighbour and the wrong one for
    placing a sample inside a single stripe: an 8 px stripe's top-hat is a
    plateau with a few units of ripple on it, several local maxima wide, and the
    nearest one to the previous sample is whichever the walk was already
    standing over. So the error does not average out - it is latched at the
    anchor and carried the whole way. Measured over 24 plates, the near service
    line's samples sit 1.0-2.4 px off the middle of their own paint, at a
    consistent offset within each plate and with a sign that flips between two
    recordings of the SAME fixed camera. That signature is the giveaway: a real
    feature does not change sides when nothing moved, a latched starting
    position does.

    AND IT IS NOT ALWAYS A LATCH. On a lower-contrast court the same plateau
    makes the follower change its mind at every step instead of sticking: on
    `13-34-53` the near service line's samples are a median 0.00 px off centre
    and scatter 0.90 px about it, the centre line's 1.49 px with a worst sample
    at 6.6, and the points visibly cross from one edge of the stripe to the
    other and back. Both faces of that are the same defect and both are worth
    removing, but only one of them is a curve.

    A stripe's two edges are what say where its middle is, and they are the one
    thing about it that is not a plateau - `white_edges` puts them within a pixel
    on every sample. Write a sample as the paint's centre plus the follower's
    error, `p = c + e`; the two distances it measures outward are then

        a = w - e        b = w + e        (w = the stripe's half-width)

    and the two combinations of them say completely different things:

        (a - b) / 2 = -e         the follower's error, EXACTLY, per sample
         a + b      = 2w         the stripe's width, with `e` algebraically gone

    So the midpoint is not something to fit. It is already the answer at every
    sample where both edges were found, with the walk's own error - latched,
    scattered, or both - cancelling on its own. Fitting it, which is what this
    did at first, low-passes the correction and removes only the part of the
    error that is smooth: on `13-34-53` that is a 0.4 px move against a 0.9 px
    scatter it leaves entirely alone.

    WHAT IS FITTED IS THE WIDTH, and it is fitted to decide which pairs to
    believe rather than to produce the answer. A pair that has picked up a
    crossing line - the centre line meets the near service line at the anchor
    and the net lower down - measures a width that is nothing like the stripe's,
    while a pair straddling the true stripe measures the right width no matter
    how far off centre the sample it was measured from is. That is what makes a
    1.5 px tolerance meaningful here and meaningless on the midpoint itself; see
    `EDGE_TOL`. The width is also what the narrowness test needs, since a 1-2 px
    line has no middle to look for.

    Where there is no credible pair - one edge missing, or a width that does not
    belong to this stripe - the correction the REST of the line implies is used
    instead, from a quadratic through the midpoints that were credible. That is
    1-6% of a walk, it is the old behaviour, and beyond the outermost credible
    sample it is held rather than extrapolated: the ends of the near service
    line are the two corners the sidelines are anchored at, and a quadratic run
    past its own evidence is not the thing to move them with.

    Returns the shifted points, or the input unchanged when the paint is too
    narrow to have a middle, or too few credible pairs were found.
    """
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) < min_n:
        return pts
    n = _normals(pts, swap)
    u = pts[:, 1] if swap else pts[:, 0]
    a = _edge_out(E, pts, n, half)
    b = _edge_out(E, pts, -n, half)
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.sum() < min_n:
        return _no_shift(trace, "both edges found on only %d samples" % int(both.sum()), pts)
    a, b = np.nan_to_num(a), np.nan_to_num(b)

    W = trim_to_curve(np.column_stack([u[both], (a + b)[both]]), False, tol=tol)
    if len(W) < min_n:
        return _no_shift(trace, "only %d widths hold a curve" % len(W), pts)
    cw = np.polyfit(W[:, 0], W[:, 1], 2)
    rms = float(np.sqrt(np.mean((np.polyval(cw, W[:, 0]) - W[:, 1]) ** 2)))
    if rms > max_rms:                            # see EDGE_MAX_RMS
        return _no_shift(trace, "width curve rms %.1f px" % rms, pts)
    wide = np.polyval(cw, np.clip(u, W[:, 0].min(), W[:, 0].max()))
    width = float(np.median(wide))
    if width < min_width:
        return _no_shift(trace, "paint only %.1f px wide" % width, pts)

    # A pair belonging to one stripe measures that stripe's width, wherever
    # inside it the sample happens to sit. A pair that has found a crossing line
    # does not, and that is the only thing being tested for here.
    mid = 0.5 * (a - b)
    good = both & (np.abs(a + b - wide) <= tol)
    if good.sum() < min_n:
        return _no_shift(trace, "only %d credible edge pairs" % int(good.sum()), pts)
    cm = np.polyfit(u[good], mid[good], 2)
    fill = np.polyval(cm, np.clip(u, u[good].min(), u[good].max()))
    # Both distances are measured outward, so a credible pair brackets its own
    # sample and `mid` is inside the paint by construction. The fallback is a fit
    # and carries no such guarantee, so it is held to the same bound explicitly.
    shift = np.where(good, mid, np.clip(fill, -0.5 * wide, 0.5 * wide))
    if trace is not None:
        trace.update(moved=True, width=width, shift=float(np.median(shift)),
                     shift_max=float(np.abs(shift).max()),
                     scatter=float(np.std(shift)), fell_back=int((~good).sum()),
                     why="")
    return pts + shift[:, None] * n


def _no_shift(trace, why, pts):
    if trace is not None:
        trace.update(moved=False, why=why)
    return pts


# ------------------------------------------------------------- 5. lens ----
def undistort_maps(cam, shape, balance=1.0):
    """remap() tables that straighten a frame, so a plain homography applies.

    `balance` scales the output focal length: 1.0 keeps the centre scale and
    pushes the corners outside the frame, lower values pull more of the barrel
    back into view.
    """
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xy = np.column_stack([(xs.ravel() - w / 2) / (cam.f * balance),
                          (ys.ravel() - h / 2) / (cam.f * balance)])
    d = cam.distort(xy)
    mx = (d[:, 0] * cam.f + cam.cx).reshape(h, w).astype(np.float32)
    my = (d[:, 1] * cam.f + cam.cy).reshape(h, w).astype(np.float32)
    return mx, my


def peak_ridge_lab(lab, c, n, half, min_contrast):
    """`_peak_ridge`, but the profile is a COLOUR distance rather than a top-hat.

    The top-hat is a brightness detector, and on a wet court so is the glare: at
    `10-05-24` the mottling beside the far service line scores 19-21 against the
    paint's 21, so the paint stops being a crest and `_peak_ridge` returns None
    for seven steps with its prediction 0.2 px from the line.

    Colour separates them where brightness cannot, because paint differs from
    the glare in TWO ways at once - lighter AND less chromatic - while the glare
    differs from the court in only one. Measured across the dead band, the paint
    dips 6-8 chroma units below its surroundings while the glare dips none.

    Signed, and that is not a detail: the unsigned norm peaks 2.5 px off the
    line on 4 of 5 dead-band columns, because the paint's own dark shoulder is
    16 L units from the background where the paint is 9, and a norm cannot tell
    darker from lighter. Gated on L, it peaks where the top-hat does on every
    column measured.

    The property this has and the top-hat has not is that its contrast does not
    decay as the window opens - 17/13/13/13 across half 4-9 px against the
    top-hat's 14/10/8/7 - which is what makes coasting over a gap work at all:
    every miss widens the window, and with a top-hat that lowers the contrast
    and guarantees the next miss.
    """
    s = stations(half)
    pts = c + s[:, None] * n
    LAB = bilinear(lab, pts)
    if np.isnan(LAB).any():
        return None
    L, A, B = LAB[:, 0], LAB[:, 1], LAB[:, 2]
    mL, mA, mB = _mid(L), _mid(A), _mid(B)
    prof = np.where(L > mL, np.sqrt((L - mL) ** 2 + (A - mA) ** 2 + (B - mB) ** 2), 0.0)
    prof = prof - _mid(prof)
    inner = prof[1:-1]
    loc = np.flatnonzero((inner >= min_contrast) & (inner >= prof[:-2])
                         & (inner > prof[2:])) + 1
    if not len(loc):
        return None
    # One candidate is the usual case, and picking the nearest of one costs
    # four numpy calls that these walks make tens of thousands of times.
    i = int(loc[0]) if len(loc) == 1 else int(loc[np.argmin(np.abs(s[loc]))])
    y0, y1, y2 = float(prof[i - 1]), float(prof[i]), float(prof[i + 1])
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
    return c + (s[i] + min(1.0, max(-1.0, d)) * 0.5) * n
