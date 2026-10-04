"""Padel court geometry, the six lines, and the camera that maps them.

Court coordinates are metres on the floor plane, seen from the camera:

    X: 0 at the left sideline, 10 at the right sideline
    Y: 0 at the far baseline, 20 at the near baseline (closest to the camera)

Net at Y=10, service lines 6.95m either side of it (Y=3.05 and Y=16.95), and a
centre service line at X=5 running between each service line and the net. Those
are the FIP standard dimensions and every regulation court matches them, which
is what makes a single fixed model fit every venue.

The camera model is deliberately two-stage:

    court (X, Y) --Hn--> ideal normalised --distort--> pixels

Hn is a homography onto *ideal* (undistorted) normalised coordinates, so the
lens distortion is applied after it. A homography maps straight lines to
straight lines, so without the distortion stage no Hn can describe a wide-angle
camera - on the sample footage a straight service line bows 65px.

f/cx/cy are held fixed by default. f is not identifiable from line straightness
alone (scaling f is absorbed exactly by rescaling k), so pinning it costs
nothing for ground-plane work. It would matter for 3D solvePnP.
"""
import math

import numpy as np

LENGTH = 20.0
WIDTH = 10.0
NET_Y = LENGTH / 2
SERVICE_OFFSET = 6.95
FAR_SERVICE_Y = NET_Y - SERVICE_OFFSET
NEAR_SERVICE_Y = NET_Y + SERVICE_OFFSET
CENTRE_X = WIDTH / 2

# The 11 keypoints a detector should predict: every one lies on the floor plane
# and on a painted intersection. The near corners are deliberately absent - they
# are off-frame on a typical mount, and projecting them through Hn once it is
# solved is both easier and more accurate than regressing them.
KEYPOINTS = [
    ("far_corner_l",   0.0,     0.0),
    ("far_corner_r",   WIDTH,   0.0),
    ("far_svc_l",      0.0,     FAR_SERVICE_Y),
    ("far_svc_c",      CENTRE_X, FAR_SERVICE_Y),
    ("far_svc_r",      WIDTH,   FAR_SERVICE_Y),
    ("net_l",          0.0,     NET_Y),
    ("net_c",          CENTRE_X, NET_Y),
    ("net_r",          WIDTH,   NET_Y),
    ("near_svc_l",     0.0,     NEAR_SERVICE_Y),
    ("near_svc_c",     CENTRE_X, NEAR_SERVICE_Y),
    ("near_svc_r",     WIDTH,   NEAR_SERVICE_Y),
]
KP_NAMES = [n for n, _, _ in KEYPOINTS]
KP_WORLD = np.array([[x, y] for _, x, y in KEYPOINTS], float)
KP_INDEX = {n: i for i, n in enumerate(KP_NAMES)}

DERIVED = [("near_corner_l", 0.0, LENGTH), ("near_corner_r", WIDTH, LENGTH)]

# Court lines, each given by the keypoints lying on it in order. Tracing walks
# between consecutive keypoints, so a line only needs two to be usable; the
# extras on the sidelines just keep the trace anchored over a long span.
# `extend` is how far past the last keypoint the line continues, in metres.
#
# `kind` matters because real padel courts paint only the service lines and the
# centre lines. The perimeter is the wall itself: the playing surface runs up to
# the glass and the "sideline" is the surface/wall boundary, a step edge in
# colour rather than a bright ridge. Both are traced, by different detectors.
#
# The five core lines are all spanned by the six service-line keypoints alone:
# those six contain three collinear PAIRS (far/near_svc_l on X=0, _c on X=5,
# _r on X=10) as well as the two service lines themselves. So a detector that
# only reports the six easy, always-visible points seeds every line, in both
# orientations, with no point it had to guess.
#
# Each of those spans crosses the net, which hides the far half's floor. That is
# fine: the trace has locked onto the line long before it reaches the net and
# simply coasts over the gap. What is NOT fine is starting a trace at net_l or
# net_r - those sit on the sideline AND on the net's own floor line, so
# acquisition picks the wrong one immediately and the fit is dragged >100px.
# Seed traces in a clean stretch of line, never at a junction of two lines.
PAINT, BOUNDARY = "paint", "boundary"
LINES = [
    ("left_sideline",       ["far_svc_l", "near_svc_l"], LENGTH - NEAR_SERVICE_Y, BOUNDARY),
    ("right_sideline",      ["far_svc_r", "near_svc_r"], LENGTH - NEAR_SERVICE_Y, BOUNDARY),
    ("centre_line",         ["far_svc_c", "near_svc_c"], 0.0, PAINT),
    ("far_service",         ["far_svc_l", "far_svc_c", "far_svc_r"], 0.0, PAINT),
    ("near_service",        ["near_svc_l", "near_svc_c", "near_svc_r"], 0.0, PAINT),
    # far-end extras, off by default - see OCCLUDED_BY_DEFAULT
    ("far_baseline",        ["far_corner_l", "far_corner_r"], 0.0, BOUNDARY),
    ("left_sideline_far",   ["far_corner_l", "far_svc_l"], 0.0, BOUNDARY),
    ("right_sideline_far",  ["far_corner_r", "far_svc_r"], 0.0, BOUNDARY),
]

# Court outline for masking, in order.
OUTLINE = np.array([[0, 0], [WIDTH, 0], [WIDTH, LENGTH], [0, LENGTH]], float)

# Boundary lines at the far end are traceable but LIE, so they are off by
# default. From a camera mounted high behind the near baseline the floor is seen
# almost edge-on at the far end, and the few centimetres of frame at the bottom
# of the glass hide the strip of surface directly in front of it. Held out of
# the fit, the far baseline measures Y = +0.75m with a spread of only 0.14m: a
# straight, consistent, confidently wrong feature - exactly the kind that a
# robust loss cannot save you from, because it is not an outlier.
# The near sidelines are viewed steeply and measure 0.000m and 10.000m exactly.
OCCLUDED_BY_DEFAULT = {"far_baseline", "left_sideline_far", "right_sideline_far"}


def unit(v):
    if getattr(v, "shape", None) == (2,):
        # `np.linalg.norm`'s own arithmetic without its dispatch. Nearly every
        # call here is a two-vector and there are tens of thousands per frame.
        return v / math.sqrt(v[0] * v[0] + v[1] * v[1])
    return v / np.linalg.norm(v)


def perp(v):
    return np.array([-v[1], v[0]])


class Camera:
    """court metres <-> image pixels, via a homography plus radial distortion.

    model="poly"    Brown radial: r_d = r * (1 + k1 r^2 + k2 r^4 + k3 r^6)
    model="fisheye" equidistant:  theta_d = theta (1 + k1 t^2 + k2 t^4), r_d = theta_d
    """

    def __init__(self, Hn, k, model="poly", f=1920.0, cx=960.0, cy=540.0):
        self.Hn = np.asarray(Hn, float).reshape(3, 3)
        self.k = np.asarray(k, float)
        self.model = model
        self.f, self.cx, self.cy = float(f), float(cx), float(cy)

    # -- distortion -------------------------------------------------------
    def _radial(self, t):
        """1 + k1 t^2 + k2 t^4 + ... for as many coefficients as were given."""
        if not len(self.k):
            return np.ones_like(t)
        t2 = t ** 2
        p = t2
        s = 1.0 + self.k[0] * p
        for ki in self.k[1:]:
            p = p * t2
            s = s + ki * p
        return s

    def _dradial(self, t):
        """d/dt of t * _radial(t)."""
        if not len(self.k):
            return np.ones_like(t)
        t2 = t ** 2
        p = t2
        d = 1.0 + 3.0 * self.k[0] * p
        for i, ki in enumerate(self.k[1:], 1):
            p = p * t2
            d = d + (2 * i + 3) * ki * p
        return d

    def distort(self, xy):
        """ideal normalised -> distorted normalised."""
        xy = np.atleast_2d(np.asarray(xy, float))
        r = np.linalg.norm(xy, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.model == "fisheye":
                th = np.arctan(r)
                s = th * self._radial(th) / r
            else:
                s = self._radial(r)
        s = np.where(r < 1e-12, 1.0, s)
        return xy * s[:, None]

    def undistort_scale(self, rd):
        """distorted radius -> ideal/distorted ratio. The radial core of `undistort`.

        Split out because it is the whole cost of a bundle iteration and it
        depends on `k` alone: a caller refining a homography against fixed
        samples can hoist the radii out of its loop and reuse this per `k`.
        """
        k = self.k
        if self.model == "fisheye":
            th = rd.copy()
            for _ in range(12):          # Newton on theta_d(theta) - rd = 0
                # `_radial` and `_dradial` written out together: they walk the
                # same powers of theta and this is the innermost loop of every
                # bundle iteration, so sharing them halves the passes over it.
                t2 = th ** 2
                p, s, dth = t2, 1.0, 1.0
                for i, ki in enumerate(k):
                    if i:
                        p = p * t2
                    s = s + ki * p
                    dth = dth + (2 * i + 3) * ki * p
                step = (th * s - rd) / np.where(np.abs(dth) < 1e-9, 1e-9, dth)
                th -= step
                # Quadratic convergence gets here in three or four passes; the
                # rest were moving theta by less than an ulp of tan(theta).
                if not np.any(np.abs(step) > 1e-13):
                    break
            r = np.tan(np.clip(th, -1.5, 1.5))
        else:
            r = rd.copy()
            for _ in range(12):          # r = rd / (1 + k1 r^2 + ...)
                s = self._radial(r)
                rn = rd / np.where(np.abs(s) < 1e-9, 1e-9, s)
                done = not np.any(np.abs(rn - r) > 1e-13)
                r = rn
                if done:
                    break
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(rd < 1e-12, 1.0, r / rd)

    def undistort(self, xy):
        """distorted normalised -> ideal normalised (fixed-point / Newton inverse)."""
        xy = np.atleast_2d(np.asarray(xy, float))
        rd = np.linalg.norm(xy, axis=1)
        return xy * self.undistort_scale(rd)[:, None]

    # -- pixel <-> normalised --------------------------------------------
    def to_px(self, xyd):
        xyd = np.atleast_2d(np.asarray(xyd, float))
        return np.column_stack([xyd[:, 0] * self.f + self.cx, xyd[:, 1] * self.f + self.cy])

    def to_norm(self, uv):
        uv = np.atleast_2d(np.asarray(uv, float))
        return np.column_stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f])

    # -- court <-> image ---------------------------------------------------
    def ideal(self, XY):
        """court metres -> ideal normalised (no distortion)."""
        XY = np.atleast_2d(np.asarray(XY, float))
        p = np.column_stack([XY, np.ones(len(XY))]) @ self.Hn.T
        w = np.where(np.abs(p[:, 2]) < 1e-12, 1e-12, p[:, 2])
        return p[:, :2] / w[:, None]

    def project(self, XY):
        """court metres -> pixels."""
        return self.to_px(self.distort(self.ideal(XY)))

    def backproject(self, uv):
        """pixels -> court metres."""
        xy = self.undistort(self.to_norm(uv))
        Hi = np.linalg.inv(self.Hn)
        p = np.column_stack([xy, np.ones(len(xy))]) @ Hi.T
        w = np.where(np.abs(p[:, 2]) < 1e-12, 1e-12, p[:, 2])
        return p[:, :2] / w[:, None]

    def polyline(self, A, B, n=64):
        """Pixel polyline for the world segment A->B, curvature included."""
        t = np.linspace(0, 1, n)[:, None]
        return self.project(np.asarray(A, float) + t * (np.asarray(B, float) - np.asarray(A, float)))

    # -- serialisation -----------------------------------------------------
    def to_dict(self):
        return {"Hn": self.Hn.tolist(), "k": self.k.tolist(), "model": self.model,
                "f": self.f, "cx": self.cx, "cy": self.cy}

    @staticmethod
    def from_dict(d):
        return Camera(d["Hn"], d["k"], d.get("model", "poly"), d["f"], d["cx"], d["cy"])

    def homography_px(self):
        """3x3 court-metres -> pixels, valid ONLY on an undistorted image.

        Provided because downstream code usually wants a plain H. Anything using
        it must first remap the frame with undistort_maps(); applying it to a
        raw frame reintroduces the full lens error.
        """
        K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]], float)
        return K @ self.Hn


def pack(cam, free_centre=False):
    p = list(cam.k) + list((cam.Hn / cam.Hn[2, 2]).ravel()[:8])
    if free_centre:
        p += [cam.cx, cam.cy]
    return np.array(p, float)


def unpack(p, template, free_centre=False):
    nk = len(template.k)
    k = p[:nk]
    Hn = np.append(p[nk:nk + 8], 1.0).reshape(3, 3)
    cx, cy = (p[nk + 8], p[nk + 9]) if free_centre else (template.cx, template.cy)
    return Camera(Hn, k, template.model, template.f, cx, cy)


def lm(fun, x0, iters=80, huber=None, verbose=False):
    """Levenberg-Marquardt with a numerical Jacobian and IRLS Huber weighting.

    Self-contained on purpose: the problem is ~13 parameters against ~1000
    residuals, which does not justify pulling scipy into the runtime deps.
    `huber` is the residual magnitude (in pixels) beyond which a sample is
    downweighted - it is what stops a player standing on a line from tilting
    the whole calibration.
    """
    def weights(r):
        if huber is None:
            return np.ones_like(r)
        a = np.abs(r)
        return np.where(a <= huber, 1.0, np.sqrt(huber / np.maximum(a, 1e-12)))

    def cost(r):
        w = weights(r)
        return float(np.sum((w * r) ** 2))

    x = np.asarray(x0, float).copy()
    r = fun(x)
    c = cost(r)
    lam = 1e-3
    for it in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            h = max(1e-7, abs(x[j]) * 1e-6)
            xj = x.copy()
            xj[j] += h
            J[:, j] = (fun(xj) - r) / h
        w = weights(r)
        Jw, rw = J * w[:, None], r * w
        A, g = Jw.T @ Jw, Jw.T @ rw
        improved = False
        for _ in range(40):
            try:
                dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x + dx
            rn = fun(xn)
            cn = cost(rn)
            if np.isfinite(cn) and cn < c:
                x, r, c = xn, rn, cn
                lam = max(lam * 0.3, 1e-10)
                improved = True
                break
            lam *= 10
            if lam > 1e12:
                break
        if verbose:
            print("  lm it=%2d rms=%.3f lam=%.1e" % (it, np.sqrt(np.mean(r ** 2)), lam))
        if not improved:
            break
    return x, r


# ------------------------------------------------- the six lines, in detail --
# The same six lines as LINES above, in the form the detector and the solver
# want them: how each one appears in the image, which way the court lies from
# it, and how far along it a re-trace should reach.
# how each line appears, and which way the court lies from it. PAINT lines are
# bright ridges; BOUNDARY lines are where the playing surface stops.
LINE_KIND = {
    "left_sideline": BOUNDARY, "right_sideline": BOUNDARY,
    "net_line": BOUNDARY, "centre_line": PAINT,
    "near_service": PAINT, "far_service": PAINT,
}


INWARD = {                       # unit direction in court metres, into the court
    "left_sideline": (1.0, 0.0), "right_sideline": (-1.0, 0.0),
    "net_line": (0.0, 1.0),      # the near half is the side that survives
}


# The stretch of each line worth re-measuring once a model exists. Not the full
# world line: a sideline projected over its whole 20m stays inside the frame
# well past Y=0, out where the court has ended, and a tight tracer started there
# never finds anything to lock onto. The sidelines are cut to the near half
# because the net hides the far half's floor and the far end is inset behind the
# glass frame - the same reason Court/CourtModel.OCCLUDED_BY_DEFAULT exists.
#
# A third reason, measured rather than reasoned, and the one that actually
# decides it: `IM.trace_segment` starts COLD, and until it gets a first hit its
# running perpendicular offset never starts, so where a span begins matters far
# more than how long it is. Extending the sidelines back to Y=3.5 - roughly what
# the walk covers - takes `17-02-10`'s right sideline from 12 samples to ZERO,
# because the extra 55 steps at the far end all miss and the tracer reaches the
# part it could have followed with nothing to follow it from. Longer is not
# more; it is a different, worse, starting point.
RETRACE_SPAN = {
    "left_sideline":  ((0.0, 10.3), (0.0, 17.6)),
    "right_sideline": ((WIDTH, 10.3), (WIDTH, 17.6)),
    "net_line":       ((0.6, NET_Y), (WIDTH - 0.6, NET_Y)),
    "centre_line":    ((CENTRE_X, 3.6), (CENTRE_X, 16.6)),
    "near_service":   ((0.3, NEAR_SERVICE_Y), (WIDTH - 0.3, NEAR_SERVICE_Y)),
    "far_service":    ((0.6, FAR_SERVICE_Y), (WIDTH - 0.6, FAR_SERVICE_Y)),
}


# world line as (a, b): any two distinct points on it, in court metres
WORLD_LINES = {
    "left_sideline":  ((0.0, 0.0), (0.0, LENGTH)),
    "centre_line":    ((CENTRE_X, 0.0), (CENTRE_X, LENGTH)),
    "right_sideline": ((WIDTH, 0.0), (WIDTH, LENGTH)),
    "far_service":    ((0.0, FAR_SERVICE_Y), (WIDTH, FAR_SERVICE_Y)),
    "net_line":       ((0.0, NET_Y), (WIDTH, NET_Y)),
    "near_service":   ((0.0, NEAR_SERVICE_Y), (WIDTH, NEAR_SERVICE_Y)),
}
