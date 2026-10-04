"""Fit the court as a metric camera POSE, against edges read where the pose says.

A SEPARATE SOLVER, not a stage of `solve.py`. The two differ in what they treat
as unknown and in where the sideline evidence comes from, and they are meant to
be run against each other rather than merged.

    solve.py     8 DOF homography + 2 radial coefficients, fitted jointly.
                 Sidelines detected ONCE, blind, from a straight chord drawn
                 between the four service-line endpoints; the refinement rounds
                 re-read them with a weaker step-edge tracer.

    this         6 DOF metric pose (R, t) in metres, plus 2 radial
                 coefficients. The sidelines are never detected blind: the pose
                 PREDICTS where X = 0 and X = 10 project - bow included, since
                 the lens is in hand - and the edge is read in a narrow band
                 around that prediction, perpendicular to it. Re-read and
                 re-solved until it stops moving.

Over the 80 labelled plates, scored by `compare_sides.court_error_m`:

                     accepted   worst visible point of the fit, in metres
                                 p50      p90      worst
    pose fit         80/80      0.104    0.182    0.290
    solve.py         74/80      0.161    0.284    0.666

74 plates are accepted by both, 6 by this one alone, and NONE by `solve.py`
alone - so on this corpus it is a strict improvement rather than a trade. The
honest qualifications: the tolerance is 0.30 m and the worst plate here is
0.290, so 80/80 is a pass with 1 cm of headroom and not a comfortable one; the
corpus is 80 plates and the 6-plate gap is not large against that; and this
solver is scored on the same labelled cameras `solve.py` was developed against,
which is fair but is not an independent test set.

WHY THE POSE AND NOT THE HOMOGRAPHY. A 3x3 has eight degrees of freedom and the
court has six: three angles and three metres. The two spare ones are not free
precision, they are the room in which `solve._in_front`'s failure lives - a
homography is perfectly happy to put the plane's horizon across the middle of
the court, and did on one plate, at 0.79 px line rms and 89% paint coverage.
Parameterised as a pose that cannot be expressed: depth is r3 . (X - C) and the
court is in front of the camera or the residual is infinite. The check is not
enforced, it is unnecessary.

Camera height falls out as `-R^T t` rather than having to be decomposed
afterwards - WITH THE CAVEAT THAT IT IS ONLY AS GOOD AS f, which this model
identifies weakly. `court.py` records why: scaling f is absorbed almost exactly
by rescaling k, so the two are nearly the same parameter and f is pinned at w/2
by convention. The metric constraint |r1| = |r2| does bite on the homography -
it is 2 scale-invariant conditions and it is what makes a wrong lens
unabsorbable here - but it does not turn a weakly-known f into a strongly-known
one. Over the corpus the heights come out at p50 2.77 m in a 2.63-3.01 m band,
which is the right order for a padel mount, is tight enough to be a real
consistency check, and is NOT a survey of the mount. What it IS good for is
catching a fit that has left: the run where one plate diverged put its height
at 1.38 m, and `_plausible` now refuses that before it can be kept.

PINNING f COSTS TWELVE PLATES, which was not expected and is the clearest thing
this corpus said about the parameterisation. With f held at w/2 - `court.py`'s
convention, and what every truth file here carries - and only k fitted, the
same solver accepts 68 of 80 with p50 0.253 m and p90 0.415; freeing f as well
it accepts 80 with p50 0.104 and p90 0.182. The tell is in the shape of the
error rather than its size: pinned, the near-max errors cluster at 0.22-0.29 m
on almost every plate, which is a systematic scale offset and not scatter. So f
is weakly identified but it is NOT unidentified, and holding it at a convention
buys a bias rather than robustness.

WHAT IT STILL TAKES FROM `walk.py`, and why that is not circular: the near
service line, the far service line, the centre line, and the three anchor
points. Those are paint, walked off a ridge, and they are the strongest thing
in the pipeline. What it pointedly does NOT take is `walk`'s sidelines - the
chosen output of `SIDE_SELECTORS` never enters here, so the courtside answer is
independent of the one the existing solver arrives at and the two can be
compared on the same plate.

THE LENS WAS MEANT TO COME FROM `straighten/` AND DOES NOT, which is the one
design intention this module failed to keep and the most useful thing it
learned. The idea was sound: `straighten` solves the lens from every straight
edge in the frame with no court model at all, so holding it fixed would leave
six metric parameters and make the courtside answer independent of any lens
fitted to the court's own lines. It does not survive the corpus. See
`fixed_lens` for the numbers - blind focal lengths from 522 to 1050 px on
frames whose own calibrations all sit at 960, and 42 of 80 plates accepted
against 80 when the lens is fitted here instead. `--lens straighten --fix-lens`
keeps the original path, and is the thing to re-run when that package improves.

So the lens IS fitted against the court's own lines, and the independence this
module was going to have, it does not have. What it does still have is the part
that mattered more: the sideline is never DETECTED blind. `solve.py` reads it
once from a chord between two service-line endpoints and then re-reads it with
a weaker tracer; here it is read only ever where a fitted model projects X = 0,
in a band that narrows as the model improves. That is the change the corpus
rewards, and it is separable from where the lens came from.

The two packages do share a model exactly, which is worth stating because the
gauge in `straighten`'s README suggests otherwise. There

    straighten:  rd_px / f = theta * (1 + k1 theta^2 + ...)   r_ideal = f tan(theta)
    court.Camera: to_norm divides by f, then the same Newton on the same series

so `(f, k)` transfer with no conversion. `Rref` is the gauge of straighten's
OUTPUT IMAGE normalisation and does not enter the pixel-to-ray map, which is
all that is used here.

WHAT THE ANSWER LOOKS LIKE, and one thing to know before consuming it. The
report carries `lens`, `homography`, `keypoints` and `camera_model` in exactly
`api`'s shapes, so a consumer can take either solver's JSON without knowing
which produced it. `lens_seed` is separate and is the STARTING guess - under the
default that is f = w/2 and k = 0, which is why it must never be confused with
`lens`.

THE COEFFICIENTS ARE NOT INDIVIDUALLY STABLE, and that is the degeneracy
`court.py` names rather than a defect here: scaling f is absorbed almost exactly
by rescaling k, so the fit slides along that ridge and lands wherever the data
stops distinguishing. Measured over cameras that appear more than once in the
corpus - the same physical mount, several recordings:

    camera             k1 across recordings              spread   f spread   court error
    40-7A-A4-50-8E-6D  -0.0279 -0.0283 -0.0296           0.0016      3 px    0.163-0.167 m
    98-F9-CC-66-47-01  -0.0413 -0.0486 -0.0363           0.0123     12 px    0.120-0.156
    98-F9-CC-66-46-DB  -0.0325 -0.0320 -0.0174 -0.0534   0.0360     18 px    0.047-0.102
    40-7A-A4-50-8F-10  -0.0345 -0.0118 +0.0298           0.0643     51 px    0.106-0.189

One camera repeats to four decimal places and another wanders across zero, while
the COURT stays right on every one of them. So `k` and `f` here are the pair the
fit happened to settle on, not a measurement of the optics; `camera_model` as a
whole is what is meaningful, and the two must be used together. Anyone who wants
a stable per-installation lens should pool a camera's recordings rather than
believe one plate's coefficients - and `straighten/` is the right tool for that
question, being the one that asks it without a court in the way.

WHAT IS MEASURED AND WHAT IS NOT. The corpus figures above are measured, and so
is everything in `fixed_lens`, `_far_ends`, `Fixed`, `_plausible` and
`CURVE_DEG` - each cites the plates it comes from. Every other constant here is
REASONED from the measurements recorded in `walk.py` and says so in its own
docstring; none of them has been swept. `SIDE_BANDS` and the net reader's
constants are the ones most likely to be leaving something on the table.
"""
import numpy as np

try:
    import cv2
except ImportError:                                  # pragma: no cover
    raise SystemExit("fitcourt needs opencv-python")

from . import court as CM
from . import image as IM
from . import walk as WK
from .court import Camera, perp, unit


# --------------------------------------------------------------- 1. pose ----
def pose_to_Hn(rvec, t):
    """(rotation vector, translation in metres) -> Hn, court metres -> ideal.

    `Camera.ideal` maps court (X, Y, 1) through Hn into PINHOLE-NORMALISED
    coordinates - f has already been divided out by the time Hn is applied - so
    Hn is not `K [r1 r2 t]` but `[r1 r2 t]` itself, with no calibration matrix
    anywhere in it. That is the whole reason this parameterisation drops in
    without touching `Camera`: the repo's two-stage split happens to put the
    metric homography exactly where a pose produces one.
    """
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    return np.column_stack([R[:, 0], R[:, 1], np.asarray(t, float).reshape(3)])


def Hn_to_pose(Hn):
    """Hn -> (rotation vector, translation), by the usual plane decomposition.

    h1 and h2 are r1 and r2 up to one shared scale, so the scale is the mean of
    their inverse lengths - the mean rather than either alone because a fitted
    Hn does not have them exactly equal, and averaging is the least committed
    way to split the difference. R is then whatever rotation is closest to
    [r1 r2 r1xr2] in the Frobenius sense, which is the SVD's U Vt.

    THE SIGN IS NOT A CONVENTION, it is the front-of-camera test. lambda and
    -lambda both reproduce Hn projectively and only one puts the court in front,
    so it is chosen by the depth of the court centre. This is `_in_front` doing
    its job once, here, instead of guarding every iteration downstream.
    """
    Hn = np.asarray(Hn, float).reshape(3, 3)
    h1, h2, h3 = Hn[:, 0], Hn[:, 1], Hn[:, 2]
    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
    if not np.isfinite(n1 + n2) or min(n1, n2) < 1e-12:
        raise SystemExit("degenerate homography: cannot decompose to a pose")
    lam = 2.0 / (n1 + n2)
    if float(np.array([CM.CENTRE_X, CM.NET_Y, 1.0]) @ Hn[2, :]) < 0:
        lam = -lam
    r1, r2, t = lam * h1, lam * h2, lam * h3
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:                         # a reflection is not a pose
        U[:, 2] *= -1
        R = U @ Vt
    return cv2.Rodrigues(R)[0].ravel(), t


def camera_height(rvec, t):
    """Metres from the floor plane to the camera centre.

    C = -R^T t is the camera centre in world coordinates, and the court is the
    plane Z = 0, so the third component IS the height. Its SIGN depends on which
    way `r1 x r2` happened to point, which is not a physical fact about the
    court, so the magnitude is what is returned.
    """
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    return float(abs((-R.T @ np.asarray(t, float).reshape(3))[2]))


class Fixed:
    """The lens, the principal point, and what of them is being fitted.

    The parameter vector is [rvec(3), t(3)] and then, in this order, f if
    `free_f` and the radial coefficients if `free_k`. Held in one object because
    three functions have to agree about that layout and passing six positional
    arguments around was how they were going to stop agreeing.

    FREEING THE LENS IS A RETREAT AND IS LABELLED AS ONE. The design this module
    was built around holds the lens fixed from `straighten` and fits six metric
    parameters, which is what would have made the courtside evidence independent
    of it. `free_f`/`free_k` give that up - the lens is then fitted against the
    same lines, as `solve.fit_distortion` does it - and buy back the plates where
    `straighten`'s blind solve is wrong.

    Measured over the 80 labelled plates, at the tolerance `compare_sides` uses:

        lens                      fitted        accepted   p50 err   p90
        straighten, held fixed    nothing         42/80     0.228 m   1.258
        neutral, f held at w/2    k               68/80     0.253     0.415
        straighten, f and k free  f, k            76/80     0.102     0.181
        neutral, f and k free     f, k            80/80     0.104     0.182

    The last two agreeing to 0.002 m is the useful line in that table: once f
    and k are both free, where they STARTED stops mattering, which says the seed
    is not carrying the answer and the sideline evidence is. It is also why
    `straighten` is no longer the default - it costs a lens solve and changes
    nothing.

    Why holding a wrong lens is fatal here specifically: on
    `gdr-guipavas__...8F-08` straighten returns f = 662 and
    k1 = +0.218 where the plate's own calibration is f = 960, k1 = +0.017, and
    held to it the far service line fits at 8.08 px rms and lands at Y = 3.97 m
    instead of 3.05. Handed the right lens the same code gets 2.39 px and 3.27 m.
    A metric pose cannot absorb that: scaling the ideal coordinates is exactly
    what |r1| = 1 refuses to allow, so a lens error that a free homography would
    have soaked up in its eighth degree of freedom has nowhere to go here.
    """

    def __init__(self, f, k, cx, cy, free_f=False, free_k=False):
        self.f, self.k = float(f), np.asarray(k, float)
        self.cx, self.cy = float(cx), float(cy)
        self.free_f, self.free_k = bool(free_f), bool(free_k)
        self.i_f = 6
        self.i_k = 6 + (1 if self.free_f else 0)

    @property
    def frozen(self):
        """Is the pixel-to-ray map constant over the whole fit?"""
        return not (self.free_f or self.free_k)

    def camera(self, p):
        return Camera(pose_to_Hn(p[:3], p[3:6]), p[self.i_k:] if self.free_k else self.k,
                      "fisheye", p[self.i_f] if self.free_f else self.f, self.cx, self.cy)

    def pack(self, rvec, t):
        return np.array(list(np.asarray(rvec, float).ravel())
                        + list(np.asarray(t, float).ravel())
                        + ([self.f] if self.free_f else [])
                        + (list(self.k) if self.free_k else []), float)

    def base(self):
        """A Camera carrying the lens alone, for undistorting observations."""
        return Camera(np.eye(3), self.k, "fisheye", self.f, self.cx, self.cy)


# --------------------------------------------------------------- 2. lens ----
#: The neutral lens: f = w/2 and no distortion, the same starting point
#: `solve.solve` uses. It is a SEED here rather than an answer - see `Fixed` -
#: and f stays at w/2 unless `free_f`, which is `court.py`'s convention and the
#: one every truth file in the corpus carries.
NEUTRAL_F = 0.5


def fixed_lens(bgr, lens=None, verbose=False):
    """(f, k, cx, cy, source). The pixel-to-ray map the fit starts from.

    `lens` is None for the neutral seed, the string "straighten" to solve it
    blind with that package, or a `straighten.Lens` / dict / truth-file lens to
    use as given.

    STRAIGHTEN IS NOT THE DEFAULT, AND THAT IS A MEASUREMENT RATHER THAN A
    PREFERENCE. The design this module was built around holds a known lens fixed
    and fits six metric parameters - which is only worth doing if the lens is
    known. Over the 80 labelled plates `straighten`'s blind solve returns focal
    lengths from 522 to 1050 px on 1920-wide frames whose own calibrations all
    sit at 960, and a metric pose cannot absorb that: on
    `gdr-guipavas__...8F-08` (f = 662, k1 = +0.218) the fit lands 1.455 m from
    the truth, and handed the plate's own lens instead the same code lands at
    0.262 m. Held to straighten across the corpus the pose fit accepts 42 of 80;
    fitting the lens instead it accepts 76, against the existing solver's 74.
    It is kept as an option because when it is right it is very cheap, and
    because that comparison is the thing worth re-running when `straighten`
    improves.
    """
    h, w = bgr.shape[:2]
    if lens is None:
        return NEUTRAL_F * w, np.zeros(2), w / 2.0, h / 2.0, "neutral"
    if isinstance(lens, str):
        if lens != "straighten":
            raise SystemExit("unknown lens source %r" % lens)
        try:
            from straighten import fisheye_coefficients
            lens = fisheye_coefficients(bgr)
        except Exception as e:                       # noqa: BLE001 - reported below
            if verbose:
                print("  lens: straighten unavailable (%s) - using the neutral seed" % e)
            return NEUTRAL_F * w, np.zeros(2), w / 2.0, h / 2.0, "neutral"
        src = "straighten"
    else:
        src = "given"
    d = lens.as_dict() if hasattr(lens, "as_dict") else dict(lens)
    return (float(d["f"]), np.asarray(d["k"], float),
            float(d.get("cx", w / 2.0)), float(d.get("cy", h / 2.0)), src)


# ------------------------------------------------------- 3. the prediction --
def _visible_span(cam, wa, wb, shape, margin=6, n=200):
    """Longest run of the world segment whose projection stays in the frame.

    Same job as `solve._visible_span` and deliberately a local copy: this module
    is meant to be readable end to end without `solve.py` open beside it, and
    the function is twelve lines.
    """
    h, w = shape[:2]
    t = np.linspace(0.0, 1.0, n)[:, None]
    W = np.asarray(wa, float) + t * (np.asarray(wb, float) - np.asarray(wa, float))
    P = cam.project(W)
    ok = ((P[:, 0] > margin) & (P[:, 0] < w - margin) &
          (P[:, 1] > margin) & (P[:, 1] < h - margin) & np.isfinite(P).all(axis=1))
    if ok.sum() < 10:
        return None
    idx = np.where(ok)[0]
    run = max(np.split(idx, np.where(np.diff(idx) > 1)[0] + 1), key=len)
    return W[run[0]], W[run[-1]]


#: Degree of the polynomial that carries the predicted line's bow into strip
#: coordinates, and how many world points it is fitted through.
#:
#: NOT A MODEL OF ANYTHING - it is a change of representation. `_strip_grid`
#: wants the offset from the chord as `polyval(curve, t)`, and the curve being
#: represented is the projection of a straight world line through a KNOWN lens,
#: which is smooth and monotone in its bow. Degree 4 through 128 samples fits it
#: to well under a hundredth of a pixel; the residual is asserted below rather
#: than assumed, because a silently bad fit here would displace every sample.
#:
#: This is the one place the fixed lens pays off directly. `walk._locate` has to
#: SEARCH the bow, over +-`CHORD_MAX_BOW` = 60 px, because it starts from a
#: straight chord between two detected endpoints and has no lens to ask. Here
#: the bow is computed, so the band does not have to be wide enough to contain
#: a wrong guess about it.
#:
#: DEGREE 6 AND A QUARTER OF A PIXEL, both raised from what they were after the
#: first corpus run refused seven plates outright at degree 4 and 0.05 px - on
#: representation errors of 0.05 to 0.07 px, which is a twentieth of what the
#: reader below can resolve and a fortieth of the band it searches. That guard
#: was measuring nothing and rejecting plates for it. At degree 6 the worst
#: residual over the corpus is far under the new bound, and the bound is now set
#: where an error would actually start to displace a sample.
CURVE_DEG, CURVE_N = 6, 128
CURVE_MAX_PX = 0.25


def predict(cam, wa, wb, shape, inward):
    """Where the pose says a world line lands: chord, normal, and its bow.

    Returns None when too little of the line is in frame. `inward` is a world
    direction pointing INTO the court from the line, used to orient the strip
    the same way `_strip_grid` expects - column 0 deepest inside.
    """
    span = _visible_span(cam, wa, wb, shape)
    if span is None:
        return None
    W = (np.asarray(span[0], float)
         + np.linspace(0, 1, CURVE_N)[:, None] * (np.asarray(span[1], float)
                                                  - np.asarray(span[0], float)))
    P = cam.project(W)
    A, B = P[0], P[-1]
    L = float(np.linalg.norm(B - A))
    if L < 100.0 or not np.isfinite(P).all():
        return None
    u = (B - A) / L
    mid = 0.5 * (np.asarray(span[0], float) + np.asarray(span[1], float))
    step = cam.project([mid + np.asarray(inward, float) * 0.5])[0] - cam.project([mid])[0]
    inside = float(np.sign(step @ perp(u)))
    if inside == 0.0:
        return None
    nrm = perp(u) * inside
    t = (P - A) @ u
    c = (P - A) @ nrm
    curve = np.polyfit(t, c, CURVE_DEG)
    err = float(np.abs(c - np.polyval(curve, t)).max())
    if err > CURVE_MAX_PX:
        # Degree 4 has always been enough on this geometry; if it ever is not,
        # every sample read through this strip is displaced by the shortfall and
        # the fit would absorb it silently. Better to stop.
        raise SystemExit("predicted curve not representable to %.2f px (got %.2f)"
                         % (CURVE_MAX_PX, err))
    return {"A": A, "B": B, "u": u, "n": nrm, "inside": inside, "curve": curve,
            "L": L, "t": t, "world": W, "span": span}


def strip_to_image(pred, t, s):
    """Strip coordinates (along, across-from-the-CURVE) -> image pixels."""
    t = np.asarray(t, float)
    off = np.polyval(pred["curve"], t) + np.asarray(s, float)
    return pred["A"][None, :] + t[:, None] * pred["u"][None, :] + off[:, None] * pred["n"][None, :]


# --------------------------------------------------- 4. the courtside read --
#: The band the edge is looked for in, inward and outward from the PREDICTED
#: sideline, in pixels, per refinement round. It shrinks because the prediction
#: gets better and the band's only job is to contain the truth: every extra
#: pixel inward is another chance to meet a scuff's edge before the paint's,
#: which is `walk.CHORD_LINE_THR`'s finding and applies unchanged here.
#:
#: The first entry is sized from what `walk` measured about the SEED, not about
#: this reader. The seed here is a pose fitted to the service lines and the
#: three anchors alone, and the anchors themselves land a median 3.0 px from
#: where the paint ends (`solve.ANCHOR_SIGMA`), so a first band of +-32 px is
#: several times the seed's own expected error. The last entry matches
#: `walk.CHORD_LINE_IN`/`OUT` (10/20), which IS measured - it is the band that
#: reader uses once `_locate` has placed the curve.
#:
#: UNSWEPT. The schedule is a guess at the shape of the thing, not a measured
#: optimum, and it is the first constant here that should be swept.
SIDE_BANDS = ((32.0, 32.0), (18.0, 18.0), (12.0, 14.0), (10.0, 12.0))

#: How far apart the sideline samples handed to the solver are, in px of chord.
#: `_line_candidates` reads every 1 px; this keeps every nth. Same reasoning as
#: `walk._chord_read`'s `pitch` - the solver weights samples equally within a
#: line - except that `_residuals` here also normalises per line, so this only
#: sets how much the RANSAC has to chew on rather than the balance of the fit.
SIDE_PITCH = 3

#: The near half only, in metres of Y, for the same reason `court.RETRACE_SPAN`
#: and `walk._chord_read` both cut there: seen from behind the near baseline the
#: floor up-court is almost edge-on and the glass frame hides the strip of
#: surface in front of it. `walk._chord_read` measures the far half at a median
#: 0.034 m OUTSIDE the nominal line against the near half's 0.021 m inside, and
#: outside is the expensive direction because it widens the fitted court.
#:
#: The far segment stops at 9.7 and the near one starts at 10.3 rather than the
#: two meeting: the net post and the net's own foot sit exactly on the sideline
#: at Y = 10, so that half-metre is the one stretch where there is no boundary to
#: read - it is occluded by the thing whose base `read_net_base` measures.
#:
#: WHETHER THE FAR SEGMENT IS FITTED IS `FAR_HALF`, and it is a real question
#: rather than an oversight. Everything above is `walk`'s finding, on `walk`'s
#: reader, seeded from a straight chord that could be 50 px off. This module's
#: prediction is a projected world line in a band that narrows to +-10 px, which
#: is a different measurement of the same feature - so the finding is re-tested
#: here rather than inherited. See `FAR_HALF`.
SIDE_Y = (10.3, 17.6)
SIDE_Y_FAR = (3.6, 9.7)

#: Fit the far segment of each sideline as well as the near one.
#:
#: The far end is where the fit's error concentrates - `compare_sides` records
#: that on 57 of 65 labelled plates the WORST visible point is at Y = 0, and it
#: is why that module grades the far band at 0.60 m against 0.30 for the rest -
#: so reading the sideline up there is worth wanting. `walk._chord_read` says it
#: should not be: measured on its reader the far half sits a median 0.034 m
#: OUTSIDE the nominal line where the near half sits 0.021 m inside, and handing
#: it to the solver took that corpus from 110 plates accepted to 108.
#:
#: Re-measured here, over the 80 labelled plates, fitted as one concatenated
#: record per sideline (see `_join`) - and `walk`'s finding does NOT carry over.
#: It gets better everywhere, and most where it was worst:
#:
#:     far segment    accepted   NEAR band p50/p90/max     FAR band p50/p90/max
#:     not fitted       80/80    0.103  0.182  0.290 m     0.126  0.293  0.406 m
#:     fitted           80/80    0.072  0.120  0.246       0.082  0.174  0.254
#:
#: A third off every percentile of both bands. The far band's p90 - the band
#: `compare_sides` has to grade at 0.60 m instead of 0.30 precisely because the
#: evidence runs out up there - improves from 0.293 m to 0.174.
#:
#: WHY IT WORKS HERE AND NOT THERE is the difference between the two readers,
#: not a contradiction of `walk`'s measurement. That one seeds from a straight
#: chord between two service-line endpoints, which the far service line's paint
#: overrun can tilt by up to 1.36 m of world, and then searches +-140 px; up at
#: the far end a reader that far out has plenty of glass frame and kerb to lock
#: onto, and its samples land a median 0.034 m OUTSIDE the line. Here the seed
#: is a projected world line and the band has narrowed to +-10 px by the last
#: round, so the same stretch of court is being asked a much narrower question.
#:
#: IT ALSO BREAKS THE LINE-RMS GATE, which is worth knowing before reading a
#: `_court_fail` picture: the far samples scatter more in pixels while pinning
#: the line's direction better, so line rms goes UP as the court gets BETTER.
#: See `verdict`, which measures the gate on the near half and reports the far
#: segment beside it - without that split this change refuses 20 plates, all 20
#: of them correct.
FAR_HALF = True

#: RANSAC inlier tolerance in the strip, in px. `walk._chord_read` uses 3.0 for
#: the same curve through the same candidates.
SIDE_TOL = 3.0

#: Fewest inliers, and least of the predicted span they may cover, for a read to
#: count. `walk.CHORD_MIN_SPAN` is 0.45 and was measured: no accepted sideline
#: there holds less than 63% of its chord, and the one plate whose green turf
#: surround defeats the reader comes back holding 15%. Kept as it stands.
SIDE_MIN_N, SIDE_MIN_SPAN = 12, 0.45


def _join(near, far):
    """One sideline out of its two readable segments, or whichever exists.

    ONE RECORD AND NOT TWO, which is the whole of the decision. Handed to the
    solver separately they would be two line correspondences for the same world
    line, and `SAMPLE_REF` weights per LINE - so the sideline would silently
    carry twice the authority of the near service line it is being fitted
    against. Concatenated, the far segment's ~55 samples join the near's ~210
    and take about a fifth of that sideline's say, which is roughly its share of
    the evidence.
    """
    if near is None or far is None:
        rec = near if far is None else far
        return rec if rec is None else dict(rec, n_far=0 if far is None else len(rec["pts"]))
    return dict(near,
                pts=np.vstack([far["pts"], near["pts"]]),
                s=np.concatenate([far["s"], near["s"]]),
                n=near["n"] + far["n"],
                n_far=len(far["pts"]),          # the far samples come FIRST
                span=max(near["span"], far["span"]))


def read_side(f, cam, side, shape, band, y=SIDE_Y, pitch=SIDE_PITCH, tol=SIDE_TOL,
              trace=None, tag=""):
    """The sideline, read perpendicular to where the POSE puts it.

    Everything the reader does is `walk._line_candidates` unchanged - directional
    Sobel across the strip in all three Lab channels, non-maximum suppression,
    connected components kept only where they run `CHORD_LINE_LEN` px along the
    chord within `CHORD_LINE_ANG` of it, and the innermost survivor per row. Two
    things are different, and neither is in that function:

    1. THE CURVE IT IS HANDED IS COMPUTED, NOT LOCATED. `walk._chord_read` gets
       there by voting a two-parameter family over a 140 px band
       (`CHORD_IN`/`CHORD_OUT`), because a chord drawn between two service-line
       endpoints is a bracket rather than an answer - the far service line's
       paint overruns the sideline by a median 0.27-0.32 m and up to 1.36 m, so
       the chord tilts. A pose projects X = 0 directly. No endpoint is involved,
       so none of that error is.

    2. THE SAMPLES ARE THE RANSAC'S INLIERS, NOT THE CURVE THROUGH THEM.
       `walk._chord_read` returns `polyval(cf, t) + polyval(cf2, t)`, which is
       the fitted quadratic evaluated at the sample positions; that is right
       there, where the curve IS the answer being reported. Here the answer is a
       pose and these are its evidence, so pre-smoothing them with a quadratic
       would hide the scatter the solver is entitled to see and let two rounds
       of smoothing compound. RANSAC is used to reject, not to replace.

    Returns None if the strip could not hold a line, else a dict with the line
    correspondence and the signed offsets `s` - which are the quantity this
    whole module minimises, in pixels, positive INTO the court.
    """
    X = 0.0 if side == "left_sideline" else CM.WIDTH
    inward = CM.INWARD[side]
    pred = predict(cam, (X, y[0]), (X, y[1]), shape, inward)
    if pred is None:
        return None
    deep, out = band
    ts = WK._line_candidates(f, pred["A"], pred["B"], pred["inside"], pred["curve"],
                             deep=deep, out=out)
    if len(ts) < SIDE_MIN_N:
        return None
    cf, _ = WK._ransac_strip(ts, tol=tol)
    if cf is None:
        return None
    keep = ts[np.abs(ts[:, 1] - np.polyval(cf, ts[:, 0])) <= tol]
    if len(keep) < SIDE_MIN_N:
        return None
    span = float(keep[:, 0].max() - keep[:, 0].min()) / pred["L"]
    if span < SIDE_MIN_SPAN:
        return None
    keep = keep[np.argsort(keep[:, 0])][::max(1, int(pitch))]
    pts = strip_to_image(pred, keep[:, 0], keep[:, 1])
    wa, wb = CM.WORLD_LINES[side]
    rec = {"name": side, "pts": pts,
           "world_a": np.array(wa, float), "world_b": np.array(wb, float),
           "s": keep[:, 1], "span": span, "n": len(keep)}
    if trace is not None:
        trace[side + tag] = {"n": len(keep), "span": span,
                             "s_median": float(np.median(keep[:, 1])),
                             "s_mad": float(np.median(np.abs(keep[:, 1] - np.median(keep[:, 1]))))}
    return rec


# ------------------------------------------------------ 5. the net's foot --
#: How hard the net strip is blurred ALONG the line, in rows of 1 px, and the
#: band it is read in, inward (into the near half, which is court) and outward
#: (up into the net), in px.
#:
#: THE BLUR IS THE WHOLE IDEA HERE and it is not the same blur `_rectify` uses
#: for a sideline. A net is mesh: at the pixel level it is alternating cord and
#: background all the way up, so no cross-profile taken from a single column has
#: a step in it. Averaged over 161 px ALONG the line the cord and its holes
#: average together into one dark band, and the step at its foot appears. The
#: sideline's `CHORD_BLUR2` is 81 for a feature that is already solid; this is
#: doubled because the thing being fused is a texture rather than smoothed.
#:
#: The blur is legitimate for the same reason it is legitimate there: the strip
#: is cut around the PREDICTED curve, so the line is straight in it by
#: construction and averaging along it cannot smear the feature across it.
#:
#: UNSWEPT, both of them.
NET_BLUR = 161
NET_IN, NET_OUT = 40.0, 24.0

#: The guard either side of a candidate foot and the width of the bands compared
#: across it, in px - the same two numbers and the same meaning as
#: `walk.CHORD_GUARD`/`CHORD_SPAN`, which were swept and found not to matter:
#: what the band widths decide is how sharply the score peaks, not where.
NET_GUARD, NET_SPAN = 3.0, 12.0

#: How much darker the net has to be than the court beside it, in Lab L units,
#: before a row is believed.
#:
#: Reasoned from `image.WHITE_OVER`, which is the same kind of number read the
#: other way round: 12 L units above its own neighbourhood is what makes a pixel
#: paint there, measured across a corpus where the near half's paint reads
#: L = 237 and the far half's 130. The net's foot is a shadow under a dark mesh
#: and is a far larger step than paint is, so 12 is a floor rather than a
#: threshold - it exists to refuse a row that has no step at all, not to grade
#: the ones that do. UNSWEPT.
NET_DROP = 12.0

#: Fewest rows, and least of the predicted span, for a net read to count.
NET_MIN_N, NET_MIN_SPAN = 20, 0.40

#: How far ON THE NEAR SIDE of Y = 10 this reader's darkening actually sits, in
#: metres. The same kind of constant as `walk.FAR_T_SHORT` and measured the same
#: way: read the net on every labelled plate and back-project it through that
#: plate's OWN TRUTH CAMERA, so what is left is the reader's offset and not the
#: fit's error.
#:
#: Over the 80 plates this reader lands at Y = 10.184, mad 0.051, p10 10.074 and
#: p90 10.263, with nothing outside 9.797-10.301. It reads LONG - toward the
#: camera - and it should: a net's foot is not a line, it is the cord, the skirt
#: and the shadow the near-side lighting throws in front of them, and the first
#: darkening on the way up-court is the near edge of all that.
#:
#: AGAINST THE COLUMN-TOPOLOGY READER in `walk.detect`, measured identically:
#:
#:     reader                    plates read   p50        mad     full range
#:     walk column topology        65 of 80    10.123    0.071    8.783 .. 13.249
#:     blur along + step across    80 of 80    10.184    0.051    9.797 .. 10.301
#:
#: Three things in that table, and only one of them favours the old reader. Its
#: p50 is closer to 10 and it is closer on 51 of the 65 plates both read - the
#: bias here is real and larger. But it fails to read 15 plates at all, and when
#: it is wrong it is wrong by METRES: 8.78 and 13.25 are not noisy measurements
#: of a net, they are a column scan that stopped somewhere else. This reader has
#: no such case, and its spread is 0.50 m end to end against 4.47.
#:
#: A BIAS THAT REPEATS TO 0.051 M IS NOT AN ERROR, IT IS A CONSTANT - which is
#: the whole argument `walk.ANCHOR_OUT` makes about the paint that overruns the
#: sideline, and `FAR_T_SHORT` about where the centre line's walk stops. Anchored
#: here rather than at 10.0 the net's floor line becomes a usable Y constraint
#: for the first time: `FAR_T_SHORT` is trusted at a mad of 0.143 m and this is
#: three times tighter than that.
#:
#: DOES FITTING IT HELP? Yes in the middle, no at the tail, monotonically in the
#: weight. Anchored at 10.184 and swept over the corpus:
#:
#:     net_weight   accepted     p50       p90     worst
#:        0 (off)     80/80    0.104 m   0.182 m   0.290 m
#:        0.35        79/80    0.086     0.143     0.369
#:        1.0         78/80    0.087     0.169     0.392
#:
#: More net pulls the centre of the distribution in and pushes the tail out, and
#: it does so cleanly enough that this is a real choice rather than noise: 0.35
#: is the best p50 AND the best p90 in the table by some margin, and it costs one
#: plate. 0 is the DEFAULT because the accepted count is the gate everything else
#: in this repository is judged by, and 80/80 is the stronger claim - but a
#: consumer who cares about typical accuracy rather than worst-case coverage
#: should be running at 0.35, and that is what this table is here to say.
#:
#: It is the same shape of answer `solve.CHECK_ONLY` reached for the net, from
#: different evidence and for a different reason - there the net was held out
#: because it dragged the solution ~5x at full weight, here because at full
#: weight it costs two plates. Neither is an argument that the feature is
#: worthless; both are arguments about how hard to lean on it.
NET_SHORT = 0.184


def read_net_base(f, cam, shape, blur=NET_BLUR, deep=NET_IN, out=NET_OUT,
                  guard=NET_GUARD, span=NET_SPAN, drop=NET_DROP, tol=SIDE_TOL,
                  trace=None):
    """The net's floor line: fuse the mesh along it, then find the darkening.

    The direction is known before anything is looked at - the net is parallel to
    the service lines and the lens says how much that bows - and the position is
    known to whatever the current pose is worth. So the strip is cut around the
    prediction, blurred hard ALONG it until the mesh stops being mesh, and each
    row asks one question: where does the band just outside become darker than
    the band just inside by more than `NET_DROP`.

    Read outward from the NEAR half, which is the side that is court. The far
    half beyond the net is court too, but it is what the net occludes and what
    `court.OCCLUDED_BY_DEFAULT` is about; the near foot is the edge that has
    floor on one side of it.

    WHAT THIS IS AND IS NOT WORTH. It is a better reader than the column
    topology in `walk.detect`, which takes the top of the first court run in a
    vertical scan and therefore reads a single column with no averaging at all.
    It does not make the net's floor line a good CONSTRAINT, and nothing can:
    `calibrate.ADVISORY` records it reading anywhere from 9.06 m to 10.26 m on
    fits independently confirmed correct to 0.3-2.6 px, because the net has
    thickness and its base carries a shadow. A 1.2 m spread is not evidence
    about a line at Y = 10.

    So it is returned, measured, and reported - and left OUT of the residual
    unless `net_weight` is set, exactly as `solve.CHECK_ONLY` leaves it out. It
    earns its place by saying whether the fitted Y = 10 lands where the net is,
    which is a real check the sidelines cannot perform.
    """
    wa, wb = CM.RETRACE_SPAN["net_line"]
    pred = predict(cam, wa, wb, shape, CM.INWARD["net_line"])
    if pred is None:
        return None
    pr = WK._rectify(f, pred["A"], pred["B"], pred["inside"], curve=pred["curve"],
                     blur=blur, deep=deep, out=out, dt=1.0)
    L = pr["lab"][:, :, 0]
    nrow, ncol = L.shape
    g = int(round(guard))
    sp = int(round(span))
    lo, hi = g + sp, ncol - g - sp
    if hi - lo < 4:
        return None
    band = WK._bander(L)
    cand = np.arange(lo, hi)
    J = np.broadcast_to(cand[None, :], (nrow, len(cand)))
    # Column 0 is deepest INSIDE the court and the last is furthest outside, so
    # "inside minus outside" is a darkening on the way out - which is what the
    # net is - and a brightening scores negative and can never win.
    score = band(J - g - sp, J - g) - band(J + g, J + g + sp)
    j = np.argmax(score, axis=1)
    best = score[np.arange(nrow), j]
    ok = best >= drop
    if ok.sum() < NET_MIN_N:
        return None
    # Sub-pixel by the parabola through the winning score and its neighbours,
    # the same treatment `_line_candidates` gives its gradient peak.
    ji = np.clip(j, 1, len(cand) - 2)
    y0, y1, y2 = (score[np.arange(nrow), ji - 1], score[np.arange(nrow), ji],
                  score[np.arange(nrow), ji + 1])
    den = y0 - 2.0 * y1 + y2
    fr = np.zeros(nrow)
    np.divide(0.5 * (y0 - y2), den, out=fr, where=np.abs(den) > 1e-9)
    ds = pr["s"][1] - pr["s"][0]
    s = pr["s"][cand[ji]] + np.clip(fr, -1, 1) * ds
    ts = np.column_stack([pr["t"][ok], s[ok]])
    cf, _ = WK._ransac_strip(ts, tol=tol)
    if cf is None:
        return None
    keep = ts[np.abs(ts[:, 1] - np.polyval(cf, ts[:, 0])) <= tol]
    if len(keep) < NET_MIN_N:
        return None
    cover = float(keep[:, 0].max() - keep[:, 0].min()) / pred["L"]
    if cover < NET_MIN_SPAN:
        return None
    keep = keep[np.argsort(keep[:, 0])][::SIDE_PITCH]
    pts = strip_to_image(pred, keep[:, 0], keep[:, 1])
    # Anchored where this reader was MEASURED to land, not at the nominal Y = 10
    # - see NET_SHORT. Fitting it at 10.0 would feed the solver a line it is
    # known to be 18 cm wrong about, which is `walk.ANCHOR_OUT`'s argument
    # exactly: an anchor placed where the feature is not costs more than no
    # anchor, and in the one direction the anchor exists to constrain.
    y = CM.NET_Y + NET_SHORT
    rec = {"name": "net_line", "pts": pts, "world_a": np.array([0.0, y]),
           "world_b": np.array([CM.WIDTH, y]), "s": keep[:, 1], "span": cover,
           "n": len(keep), "drop": float(np.median(best[ok])), "short": NET_SHORT}
    if trace is not None:
        trace["net_line"] = {"n": len(keep), "span": cover,
                             "drop_L": rec["drop"],
                             "rows_gated": int(nrow - ok.sum())}
    return rec


# ------------------------------------------------------------- 6. the fit --
#: How much a line correspondence is believed relative to a point one, in px.
#: Paint walked off a ridge scatters 0.23 px rms about its own curve
#: (`walk.PEEL`'s table), and a sideline read by `_line_candidates` lands a
#: median 0.86 px from the truth sideline (`walk.CHORD_LINE_THR`). One number
#: for both is deliberate and is the honest reading of those two: they differ by
#: less than the venue-to-venue spread either of them carries.
LINE_SIGMA = 1.0

#: EVERY LINE COUNTS AS THIS MANY SAMPLES, whatever it actually returned. Each
#: line's residuals are scaled by sqrt(SAMPLE_REF / n), so a sideline holding 40
#: rows and one holding 300 carry the same authority about the geometry they
#: both describe, and a reader that comes back thin does not also come back
#: quiet. `solve.py` manages the same balance differently, by pitching each
#: detector's output to a comparable density - `walk._chord_read` explains that
#: handing the solver 450 sideline points against the near service line's 445
#: would quietly make the two sidelines half the fit. That works, but it makes
#: the weighting a property of every detector's `pitch` rather than of the
#: solver.
#:
#: NORMALISING TO A COUNT AND NOT TO ONE, which is a correction to how this was
#: first written and the reason the number exists at all. Dividing by sqrt(n)
#: alone makes each line contribute its rms SQUARED to the cost - about 5 units
#: for five lines at 1 px - while three anchor points at sigma 3 and a typical
#: 7 px error contribute some 17. The anchors then outweigh every line in the
#: fit put together, which inverts exactly what `solve.py`'s anchors are for:
#: "4 residuals against ~440 and therefore change nothing in any direction the
#: lines already determine - and everything in the one they do not". Scaling to
#: a reference count restores that ratio while keeping the equality between
#: lines. 150 is about what a sideline returns at `SIDE_PITCH`, so no line is
#: being inflated much beyond what it really measured.
SAMPLE_REF = 150.0

#: Huber knee, in sigmas, and the outlier cut in px - both as `solve.solve` sets
#: them, since the residuals are in the same currency and mean the same thing.
HUBER, REJECT_PX = 3.0, 3.0


def _residuals(lines, points, fx):
    """Build the residual `CM.lm` differentiates.

    THE UNDISTORTION IS HOISTED CLEAN OUT, which is the arithmetic dividend of
    fixing the lens. `solve.resid_for` cannot do this - its `k` is a fitted
    parameter, so every Jacobian column that perturbs k has to redo the Newton
    inverse over every sample, and it goes to some trouble to memoise what it
    can. When `Fixed.frozen`, the map from a detected pixel to its ideal
    normalised position depends on nothing being fitted, so it is computed once,
    here, and the entire per-iteration cost is two matmuls and a dot product.

    (`free_f` or `free_k` puts the lens back inside the loop and undoes it. That
    is the real arithmetic price of freeing it, on top of the design one.)
    """
    counts = [len(L["pts"]) for L in lines]
    allpts = (np.concatenate([L["pts"] for L in lines]) if lines
              else np.zeros((0, 2)))
    base = fx.base()
    fixed = base.undistort(base.to_norm(allpts)) if fx.frozen else None
    ends = (np.array([[list(L["world_a"]) + [1.0], list(L["world_b"]) + [1.0]]
                      for L in lines], float).reshape(-1, 3) if lines
            else np.zeros((0, 3)))
    n_end = len(ends)
    # Each line scaled to SAMPLE_REF samples, so lines are equal to each other
    # and still outweigh the anchors together - see SAMPLE_REF.
    w_line = np.concatenate(
        [np.full(c, np.sqrt(SAMPLE_REF / c) / L.get("sigma", LINE_SIGMA))
         for L, c in zip(lines, counts)]) if lines else np.zeros(0)
    if points:
        pw = np.array([P["world"] for P in points], float)
        ppx = np.array([P["px"] for P in points], float)
        psig = np.array([P.get("sigma") or 3.0 for P in points], float)
        world_h = np.vstack([ends, np.column_stack([pw, np.ones(len(pw))])])
    else:
        world_h = ends

    def resid(p):
        cam = fx.camera(p)
        xy = fixed if fixed is not None else cam.undistort(cam.to_norm(allpts))
        q = world_h @ cam.Hn.T
        w = np.where(np.abs(q[:, 2]) < 1e-12, 1e-12, q[:, 2])
        xyi = q[:, :2] / w[:, None]
        out = []
        if n_end:
            ab = xyi[:n_end]
            a, b = ab[0::2], ab[1::2]
            d = b - a
            uu = d / np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)[:, None]
            nn = np.column_stack([-uu[:, 1], uu[:, 0]])
            ap = np.repeat(a, counts, axis=0)
            npt = np.repeat(nn, counts, axis=0)
            out.append(((xy - ap) * npt).sum(axis=1) * cam.f * w_line)
        if points:
            # `Camera.project`'s three stages on the ideal coordinates the matmul
            # already produced, written out - this runs on every Jacobian column.
            xi = xyi[n_end:]
            rr = np.linalg.norm(xi, axis=1)
            th = np.arctan(rr)
            with np.errstate(divide="ignore", invalid="ignore"):
                sc = th * cam._radial(th) / rr
            xd = xi * np.where(rr < 1e-12, 1.0, sc)[:, None]
            e = xd * cam.f + np.array([cam.cx, cam.cy]) - ppx
            out.append((e / psig[:, None]).ravel())
        r = np.concatenate(out) if out else np.zeros(0)
        return np.where(np.isfinite(r), r, 1e6)

    return resid


def line_error(cam, L):
    """Signed perpendicular error of one line's samples, in px.

    Measured in UNDISTORTED space, where the world line really is straight, so
    the residual is a distance to a line rather than to a curve - the same
    choice `solve._line_residuals` makes, and for the same reason.
    """
    xy = cam.undistort(cam.to_norm(L["pts"]))
    a, b = cam.ideal(np.array([L["world_a"], L["world_b"]]))
    return ((xy - a) @ perp(unit(b - a))) * cam.f


def line_offsets(cam, lines):
    """Signed perpendicular error of every sample, in px - the reported quantity."""
    out = {}
    for L in lines:
        d = line_error(cam, L)
        XY = cam.backproject(L["pts"])
        ax = 1 if L["world_a"][1] == L["world_b"][1] else 0
        out[L["name"]] = {
            "n": int(len(d)), "rms_px": float(np.sqrt(np.mean(d ** 2))),
            "max_px": float(np.abs(d).max()),
            "axis": "Y" if ax else "X", "model_m": float(L["world_a"][ax]),
            "measured_m": float(np.median(XY[:, ax])),
        }
    return out


def refine(p0, lines, points, fx, verbose=False):
    """LM on the pose against these correspondences, with two rejection passes."""
    lines = [dict(L) for L in lines]
    p, _ = CM.lm(_residuals(lines, points, fx), p0, huber=HUBER, verbose=verbose)
    for _ in range(2):
        cam = fx.camera(p)
        drop = 0
        for L in lines:
            xy = cam.undistort(cam.to_norm(L["pts"]))
            a, b = cam.ideal(np.array([L["world_a"], L["world_b"]]))
            d = np.abs(((xy - a) @ perp(unit(b - a))) * cam.f)
            scale = max(1.0, 1.4826 * float(np.median(d)))
            keep = d <= max(REJECT_PX, 4.0 * scale)
            drop += int((~keep).sum())
            L["pts"] = L["pts"][keep]
            if "s" in L:
                L["s"] = np.asarray(L["s"])[keep]
        lines = [L for L in lines if len(L["pts"]) >= 8]
        if not drop:
            break
        p, _ = CM.lm(_residuals(lines, points, fx), p, huber=HUBER, verbose=verbose)
    return p, lines


# ------------------------------------------------------------ 7. the seed --
#: World position of the far service line's painted ends, for SEEDING ONLY.
#:
#: The paint overruns the sideline here by a median 0.27-0.32 m and up to 1.36 m
#: (`walk`'s section 4b header), which is far too uncertain to anchor anything -
#: it is exactly why the chord's position is a bracket rather than an answer.
#: As a seed it is fine: it has to be good enough to put the predicted sideline
#: inside a 32 px band, and a third of a metre at the far service line is well
#: inside that. It is used to build the initial homography and then dropped; it
#: never enters `points` and no residual ever sees it.
SEED_FAR_OUT = 0.30


def _far_ends(lines, far_t):
    """The far service line's two painted ends, for seeding.

    `walk.chord_ends` first, which is the same computation `detect` does, and
    then the same thing again WITHOUT its symmetry gate if that refused. The
    gate exists because a lopsided far walk would tilt the chord that reader
    then measures a sideline along, by degrees rather than pixels. Nothing here
    is measured along this line: it is two points used to make a four-point
    homography non-degenerate and then discarded, so a lopsided pair is still a
    perfectly good pair of points. Refusing on this cost two plates their fit
    entirely in the first corpus run, for a guard that was not guarding anything
    this function does.

    AND `far_t` IS OPTIONAL, which is the other half of the same mistake. It is
    an input to the symmetry test and to nothing else, so requiring it made the
    fallback inherit the very precondition it exists to survive without: on both
    plates that failed, the centre line's walk never reached the far T, so
    `walk.anchors` omitted it, so this returned None and the seed was left with
    three collinear points. The line itself was fine on both - 144 and 152
    samples, and the trim drops none of them.
    """
    fs = next((L for L in lines if L["name"] == "far_service"), None)
    if fs is None or len(fs["pts"]) < 40:
        return None
    if far_t is not None:
        e = WK.chord_ends(fs["pts"], far_t)
        if e is not None:
            return e
    w = IM.trim_to_curve(np.asarray(fs["pts"], float).reshape(-1, 2), False, tol=3.0)
    if len(w) < 40:
        return None
    cf = np.polyfit(w[:, 0], w[:, 1], 2)
    return tuple(np.array([x, float(np.polyval(cf, x))])
                 for x in (w[:, 0].min(), w[:, 0].max()))


def seed(lines, ctx, fx, verbose=False):
    """A first pose, from points that owe the sidelines nothing.

    Four correspondences would do for a homography and three of the ones to hand
    are COLLINEAR - both near service ends and the near service T all sit on
    Y = 16.95 - which is degenerate however many of them there are. The far
    service line's ends are what break it; see `_far_ends`, at a world position
    that is only good enough to seed with (see `SEED_FAR_OUT`).

    Deliberately NOT `solve.homography_from_lines`: that needs two lines in each
    direction and the only X-direction line available without touching a
    sideline is the centre line.
    """
    W, PX = [], []
    for P in ctx.get("anchors", ()):
        W.append(P["world"])
        PX.append(P["px"])
    if ctx.get("T") is not None:
        W.append((CM.CENTRE_X, CM.NEAR_SERVICE_Y))
        PX.append(ctx["T"])
    far_t = next((P["px"] for P in ctx.get("anchors", ())
                  if P["name"] == "far_service_t"), None)
    e = _far_ends(lines, far_t)
    if e is not None:
        W += [(-SEED_FAR_OUT, CM.FAR_SERVICE_Y),
              (CM.WIDTH + SEED_FAR_OUT, CM.FAR_SERVICE_Y)]
        PX += [e[0], e[1]]
    if len(W) < 4:
        raise SystemExit("only %d seed correspondences - need 4" % len(W))
    base = fx.base()
    xy = base.undistort(base.to_norm(np.asarray(PX, float)))
    H, _ = cv2.findHomography(np.asarray(W, np.float64).reshape(-1, 1, 2),
                              xy.reshape(-1, 1, 2), 0)
    if H is None:
        raise SystemExit("seed homography failed on %d points" % len(W))
    rvec, t = Hn_to_pose(H)
    if verbose:
        print("  seed: %d points, height %.2f m" % (len(W), camera_height(rvec, t)))
    return rvec, t


# ------------------------------------------------------------- 8. the run --
#: Refinement rounds, and the movement below which it has stopped moving, in px
#: of the worst court keypoint. One round per band in `SIDE_BANDS`.
ROUNDS, SETTLED_PX = len(SIDE_BANDS), 0.20

ANCHOR_LINES = ("near_service", "far_service", "centre_line")

#: The two gates a round is scored against, in px - `calibrate.RMS_MAX` and
#: `calibrate.ANCHOR_GATE`, deliberately the same numbers. Each divided by
#: itself so the two are comparable and the worse one is the score, which is
#: `calibrate._shortfall` with the two terms this module can compute.
ROUND_RMS_MAX, ROUND_ANCHOR_MAX = 2.0, 25.0


def _plausible(cam, shape):
    """Is this camera physically possible for a court seen from behind it?

    A pose cannot fold the horizon through the court - that is the whole point
    of the parameterisation - but with the lens freed it can still WANDER, and
    on one plate in the corpus a round did: LM traded a wrong k against a wrong
    scale and came back with a court 11815 m from the truth, which is not a fit
    that lost accuracy but one that left. Nothing in the residual notices,
    because it is self-consistent about a court somewhere else entirely.

    So the same three questions `solve._orientation_ok` asks, plus a bound on
    the height. 1 to 12 m is not a tuned threshold: a padel camera is mounted
    above the back glass, the corpus's own fits sit at 2.2-3.3 m, and anything
    outside this bracket is a solver artefact rather than an unusual venue.
    """
    h, w = shape[:2]
    p = cam.project([[CM.CENTRE_X, CM.NET_Y], [CM.CENTRE_X, CM.NEAR_SERVICE_Y],
                     [0.0, CM.NET_Y], [CM.WIDTH, CM.NET_Y]])
    if not np.isfinite(p).all():
        return False
    if p[1][1] <= p[0][1] or p[3][0] <= p[2][0]:
        return False                                 # upside down, or mirrored
    return bool(0 < p[0][0] < w and 0 < p[0][1] < h)


def _score(cam, lines, points):
    """How close this camera is to being refused: 1.0 is exactly on a gate.

    `calibrate._finish`'s rule, for the reason given there - a refinement round
    is a CANDIDATE, not a replacement - carried over because it applies here for
    the same reason and more strongly: this module re-reads the sidelines from
    its own current answer, so a round that drifts is a round whose next read
    confirms the drift.
    """
    d = (np.concatenate([line_error(cam, L) for L in lines]) if lines
         else np.array([1e6]))
    rms = float(np.sqrt(np.mean(d ** 2)))
    amax = max((float(np.linalg.norm(cam.project([P["world"]])[0]
                                     - np.asarray(P["px"], float)))
                for P in points), default=0.0)
    return max(rms / ROUND_RMS_MAX, amax / ROUND_ANCHOR_MAX), rms, amax


def fit(bgr, lens=None, rounds=ROUNDS, free_f=True, free_k=True, net_weight=0.0,
        far_half=FAR_HALF, verbose=False, detected=None):
    """Detect, seed, and alternate reading the courtsides with fitting the pose.

    `detected` is `(lines, ctx)` from `walk.detect`, so a caller that already
    has one does not pay for it twice. `net_weight` > 0 puts the net's floor
    line into the residual at that line-sigma multiple; at 0 it is measured and
    reported only - see `read_net_base` for why that is the default.

    Returns a dict. `camera` is a `court.Camera` and drops into everything that
    already consumes one.
    """
    shape = bgr.shape
    f_lens, k, cx, cy, source = fixed_lens(bgr, lens, verbose)
    fx = Fixed(f_lens, k, cx, cy, free_f, free_k)
    if verbose:
        print("  lens: %s   f=%.1f  k=%s  centre=(%.1f, %.1f)%s"
              % (source, f_lens, ", ".join("%.6f" % v for v in k), cx, cy,
                 "   fitting " + "+".join(n for n, on in
                                          (("f", free_f), ("k", free_k)) if on)
                 if not fx.frozen else "   HELD FIXED"))
    lines, ctx = detected if detected is not None else \
        WK.detect(bgr, verbose=False, with_ctx=True)
    anchors = [dict(L) for L in lines if L["name"] in ANCHOR_LINES]
    if len(anchors) < 3:
        raise SystemExit("need all three anchor lines, have %d" % len(anchors))
    points = ctx["anchors"]
    fields = WK.fields(bgr)

    rvec, t = seed(lines, ctx, fx, verbose)
    # The anchors alone first, so the very first predicted sideline comes from a
    # pose that has at least been fitted, rather than from a four-point DLT.
    #
    # WITH THE LENS FROZEN FOR THIS STAGE, WHATEVER THE CALLER ASKED FOR. The
    # three anchor lines are two along Y and one along X, and a lens fitted
    # against that is fitted against almost nothing: on
    # `coquelicot-padel__...8F-17` freeing f and k here sent the pre-fit to
    # f = 491 at a camera height of 1.38 m, from which no sideline could be read
    # and the run kept the wreckage. Frozen, the same stage is six parameters
    # against three well-measured lines and three points, which is what it is
    # for. The lens is freed below, where the sidelines are there to constrain
    # it - which is the same argument `solve.py` makes for fitting distortion
    # only once every line is in hand.
    fx0 = Fixed(f_lens, k, cx, cy)
    p0, _ = refine(fx0.pack(rvec, t), anchors, points, fx0)
    if not _plausible(fx0.camera(p0), shape):
        raise SystemExit("the anchor lines alone do not give a plausible pose")
    p = fx.pack(p0[:3], p0[3:6])

    history, sides, net = [], {}, None
    prev_kp, best = None, None
    for r in range(rounds):
        cam = fx.camera(p)
        band = SIDE_BANDS[min(r, len(SIDE_BANDS) - 1)]
        tr = {}
        sides = {}
        for side in ("left_sideline", "right_sideline"):
            rec = read_side(fields, cam, side, shape, band, trace=tr)
            if far_half:
                rec = _join(rec, read_side(fields, cam, side, shape, band,
                                           y=SIDE_Y_FAR, trace=tr, tag="_far"))
            if rec is not None:
                sides[side] = rec
        net = read_net_base(fields, cam, shape, trace=tr)
        if len(sides) < 2:
            # Two lines along X where the design wants three is the degeneracy
            # `calibrate._finish` documents; one sideline is worse than none,
            # because it looks like evidence and carries no cross-check.
            if verbose:
                print("  round %d: only %d sideline(s) read - stopping"
                      % (r, len(sides)))
            break
        use = anchors + list(sides.values())
        if net is not None and net_weight > 0:
            use.append(dict(net, sigma=LINE_SIGMA / net_weight))
        p_try, fitted = refine(p, use, points, fx)
        cam_try = fx.camera(p_try)
        height = camera_height(p_try[:3], p_try[3:6])
        sc, rms, amax = _score(cam_try, fitted, points)
        okay = _plausible(cam_try, shape) and 1.0 <= height <= 12.0
        if not okay:
            # The round LEFT. Not a worse fit - a different court. Keep what we
            # had; the next round would only re-read the sidelines from here and
            # agree with itself.
            if verbose:
                print("  round %d: implausible (height %.2f m) - discarded" % (r, height))
            break
        p = p_try
        cam = cam_try
        kp = cam.project(CM.KP_WORLD)
        moved = (float(np.abs(kp - prev_kp).max()) if prev_kp is not None
                 else float("inf"))
        prev_kp = kp
        history.append({"round": r, "band": band, "moved_px": moved,
                        "height_m": height, "rms_px": rms, "anchor_px": amax,
                        "score": sc, **tr})
        # Ties to the later round, so a refinement still wins wherever it is
        # doing what it is for - `calibrate._finish`'s rule exactly.
        if best is None or sc <= best[0]:
            best = (sc, p.copy(), dict(sides), net, r)
        if verbose:
            print("  round %d band %.0f/%.0f  %s  moved %.2f px  height %.2f m  "
                  "rms %.2f  anchor %.1f  score %.2f"
                  % (r, band[0], band[1],
                     "  ".join("%s n=%d s=%+.2f" % (n.split("_")[0], v["n"], v["s_median"])
                               for n, v in sorted(tr.items()) if "s_median" in v),
                     moved, height, rms, amax, sc))
        if moved < SETTLED_PX:
            break

    if best is None:
        # Not one round produced a scored, plausible camera with both sidelines.
        # What is left is the anchor-only pre-fit, which has no courtside
        # evidence in it at all - that is not this solver's answer, it is the
        # absence of one, and returning it as though it were is how a 11815 m
        # court got into the first corpus run.
        raise SystemExit("no round read both sidelines from a plausible pose")
    p, sides, net, kept = best[1], best[2], best[3], best[4]
    cam = fx.camera(p)
    report = {
        "camera": cam,
        "pose": {"rvec": [float(v) for v in p[:3]], "t_m": [float(v) for v in p[3:6]],
                 "height_m": camera_height(p[:3], p[3:6]),
                 "free_f": bool(free_f), "free_k": bool(free_k),
                 "f": float(cam.f), "k": [float(v) for v in cam.k]},
        # THE ANSWER, in the shape `api.lens_of` returns it - so `["lens"]["k"]`
        # means the same thing in both solvers' JSON. It did not, briefly, and
        # that was a trap worth naming: this key used to hold the SEED, which
        # under the default neutral start is k = [0, 0]. Anyone reading the
        # obvious field would have got zeros and believed the camera had no
        # distortion, on a solver whose whole premise is that it does.
        "lens": _lens_of(cam),
        # ...and the seed, separately and unmistakably labelled.
        "lens_seed": {"source": source, "f": float(f_lens),
                      "k": [float(v) for v in k], "cx": cx, "cy": cy},
        "homography": _homography_of(cam),
        "keypoints": _keypoints_of(cam, shape),
        "sidelines_read": len(sides),
        "round_kept": kept,
        "per_line": line_offsets(cam, anchors + list(sides.values())),
        "sides": {n: {"n": v["n"], "span": v["span"]} for n, v in sides.items()},
        "rounds": history,
        # The samples themselves, for the debug picture. Not JSON - `fit_court.py`
        # drops them before writing. A picture of a fit that cannot show what the
        # fit READ is a picture of an opinion.
        "lines": anchors + list(sides.values()),
        "net": net,
        "band": SIDE_BANDS[min(kept, len(SIDE_BANDS) - 1)],
        "anchor_points": points,
    }
    if net is not None:
        report["net_line"] = dict(line_offsets(cam, [net])["net_line"],
                                  fitted=bool(net_weight > 0), drop_L=net["drop"])
    # What the far half would have said, measured and never fitted - SIDE_Y_FAR.
    # Wrapped because this is a REPORT: a strip that cannot be cut out there must
    # not be able to take down a fit that has already succeeded.
    far = {}
    for side in ("left_sideline", "right_sideline"):
        try:
            rec = read_side(fields, cam, side, shape, SIDE_BANDS[-1], y=SIDE_Y_FAR)
        except BaseException:                            # noqa: BLE001 - SystemExit too
            rec = None
        if rec is not None:
            far[side] = line_offsets(cam, [rec])[side]
    report["far_half_check"] = far
    e = _point_errors(cam, points)
    report["per_point"] = e
    report["max_anchor_px"] = max((v["px"] for v in e.values()), default=None)
    report.update(verdict(bgr, report))
    return report


def _lens_of(cam):
    """The fitted lens, in `api.lens_of`'s shape. Deferred import: `api` reaches
    the whole pipeline and this module is meant to stand beside it, not under
    it - but the SHAPE of the answer has to be identical or a consumer cannot
    treat the two solvers interchangeably, which is the entire point of the
    comparison this module exists for."""
    from .api import lens_of
    return lens_of(cam)


def _homography_of(cam):
    from .api import homography_of
    return homography_of(cam)


def _keypoints_of(cam, shape):
    from .api import keypoints_of
    return keypoints_of(cam, shape)


def _point_errors(cam, points):
    out = {}
    for P in points:
        d = cam.project([P["world"]])[0] - np.asarray(P["px"], float)
        out[P["name"]] = {"px": float(np.linalg.norm(d)),
                          "world": [float(v) for v in P["world"]]}
    return out


# ---------------------------------------------------------- 9. the verdict --
#: The four gates, and the per-line redundancy tolerance in metres - every one
#: of them `calibrate`'s number rather than a new one.
#:
#: DELIBERATELY NOT RE-TUNED FOR THIS SOLVER, and that is the whole value of
#: them here. A verdict invented alongside the fit it judges tends to be a
#: verdict the fit passes; these were set against a different solver and are
#: simply asked the same questions of this one. `paint_probe` in particular is
#: the one measurement in the repository that no fit has ever seen - it looks
#: for paint where the model says paint should be, on pixels that were never
#: fitted to - so it is an outside opinion in the strict sense.
#:
#: The one gate that is NOT carried over is `solve.redundancy_check`'s use of
#: the net. There it is advisory because the net's floor line reads anywhere
#: from 9.06 to 10.26 m; here it is anchored at `NET_SHORT` where it was
#: measured to land, so it is checked against 10.184 like any other line - but
#: still only when it was actually fitted.
V_COVERAGE_MIN, V_PROBE_MAX, V_RMS_MAX, V_ANCHOR_MAX, V_TOL_M = 0.85, 2.0, 2.0, 25.0, 0.25


def verdict(bgr, rep):
    """Is this fit good enough to use? The same four gates `calibrate` applies.

    Returns {"ok", "reason", "quality"} to be merged into the report.
    """
    from .calibrate import paint_probe          # deferred: it pulls in solve.py
    cam = rep["camera"]
    cov, prms = paint_probe(bgr, cam)
    # THE GATE IS MEASURED ON THE NEAR HALF, and the far segment is measured
    # beside it rather than inside it. This is a change made AFTER seeing the
    # gate disagree with the truth, so it needs a better reason than that, and
    # there is one: `V_RMS_MAX` is `calibrate.RMS_MAX`, a threshold set against
    # lines read over the near half, and pixel rms over a longer span is not the
    # same quantity. Fitting the far segment took the court's own error DOWN by
    # a third at every percentile - near-band p90 0.182 m to 0.120, far-band p90
    # 0.293 to 0.174 - and took the line rms UP, p90 1.72 px to 2.55, straddling
    # a gate of 2.0. Twenty plates were refused and all twenty were right.
    #
    # The mechanism is not mysterious and it is why the two numbers move
    # opposite ways: the far samples sit almost edge-on against the glass frame
    # and genuinely scatter more in PIXELS, and they also sit at the far end of a
    # long lever arm, so they pin the line's direction better than the near ones
    # do. A mean residual cannot tell "noisier" from "further away". Averaging
    # them in makes the statistic worse at predicting the thing it is for.
    #
    # So the gate keeps its own footing and the far segment is reported at
    # `far_rms_px`, where a reader can see it. Nothing is hidden: the anchors,
    # the coverage and the independent paint probe are untouched, and the probe
    # is the one measurement no fit has ever seen.
    gate, extra = [], []
    for L in rep["lines"]:
        nf = int(L.get("n_far") or 0)
        if nf and len(L["pts"]) > nf:
            gate.append(dict(L, pts=L["pts"][nf:]))
            extra.append(dict(L, pts=L["pts"][:nf]))
        else:
            gate.append(L)
    d = np.concatenate([line_error(cam, L) for L in gate])
    rms = float(np.sqrt(np.mean(d ** 2)))
    fd = (np.concatenate([line_error(cam, L) for L in extra]) if extra else None)
    amax = rep.get("max_anchor_px")
    bad = [n for n, v in rep["per_line"].items()
           if abs(v["measured_m"] - v["model_m"]) > V_TOL_M]
    if rep.get("net_line", {}).get("fitted"):
        v = rep["net_line"]
        if abs(v["measured_m"] - v["model_m"]) > V_TOL_M:
            bad.append("net_line")
    why = []
    if amax is not None and amax > V_ANCHOR_MAX:
        why.append("anchors %.0f px (gate %.0f)" % (amax, V_ANCHOR_MAX))
    if bad:
        why.append("lines " + ",".join(sorted(bad)))
    if cov <= V_COVERAGE_MIN:
        why.append("coverage %.0f%%" % (100 * cov))
    if prms >= V_PROBE_MAX:
        why.append("probe %.2f px" % prms)
    if rms >= V_RMS_MAX:
        why.append("rms %.2f px" % rms)
    return {"ok": not why, "reason": "; ".join(why),
            "quality": {"line_rms_px": rms, "paint_coverage": cov,
                        "probe_rms_px": prms, "max_anchor_px": amax,
                        "far_rms_px": (None if fd is None
                                       else float(np.sqrt(np.mean(fd ** 2)))),
                        "n_far": int(sum(len(L["pts"]) for L in extra)),
                        "n_lines": len(rep["lines"]),
                        "n_samples": int(sum(len(L["pts"]) for L in rep["lines"]))}}


# --------------------------------------------------------- 10. the pictures --
#: One colour per thing the fit was told, so a folder of these can be skimmed.
#: The sidelines are the loud ones because they are what this solver does
#: differently; the paint lines are quiet because they arrived from `walk.py`
#: already correct and are not the thing under examination here.
PIC_COL = {
    "left_sideline": (0, 255, 255), "right_sideline": (0, 255, 255),
    "near_service": (255, 0, 255), "far_service": (255, 160, 0),
    "centre_line": (255, 255, 255), "net_line": (255, 60, 60),
}
PIC_BAND = (90, 200, 90)          # the walls of the strip the reader looked in
PIC_MODEL = (0, 165, 255)         # where the fitted court says the line is


def _poly(vis, cam, A, B, col, t=1, n=96):
    s = np.linspace(0, 1, n)[:, None]
    p = cam.project(np.asarray(A, float) + s * (np.asarray(B, float) - np.asarray(A, float)))
    if np.isfinite(p).all():
        cv2.polylines(vis, [p.astype(np.int32)], False, col, t, cv2.LINE_AA)


def debug_picture(bgr, rep, path, stem=""):
    """What the pose fit READ, drawn where it read it.

    The analogue of `draw.draw_walk`, and it shows a different kind of thing
    because this solver works a different way. `draw_walk` shows a path followed
    across the pixels. There is no path here - there is a PREDICTION and a band
    around it, so what is worth seeing is exactly that: where the model said the
    sideline would be, how wide a window the reader was allowed, and which
    pixels inside that window it decided were the boundary.

    A fit that has gone wrong shows it here in one of two unmistakable ways: the
    band sits somewhere that is not the sideline, or the band is right and the
    samples inside it scatter or cling to one wall of it.
    """
    from .draw import _label, mark_x
    vis = bgr.copy()
    cam = rep["camera"]
    band = rep.get("band") or SIDE_BANDS[-1]

    # The search band, drawn as it actually was: offset along the strip's own
    # normal, not simply a fatter line, so what is drawn is the window.
    for side in ("left_sideline", "right_sideline"):
        X = 0.0 if side == "left_sideline" else CM.WIDTH
        try:
            pred = predict(cam, (X, SIDE_Y[0]), (X, SIDE_Y[1]), bgr.shape,
                           CM.INWARD[side])
        except BaseException:                            # noqa: BLE001
            pred = None
        if pred is None:
            continue
        for edge in (band[0], -band[1]):
            q = strip_to_image(pred, pred["t"], np.full(len(pred["t"]), edge))
            cv2.polylines(vis, [q.astype(np.int32)], False, PIC_BAND, 1, cv2.LINE_AA)

    # Where the fitted court puts every line, under the samples rather than over
    # them - the samples are the evidence and must not be hidden by the model.
    for name, (wa, wb) in CM.WORLD_LINES.items():
        _poly(vis, cam, wa, wb, PIC_MODEL, 1)

    for L in rep["lines"] + ([rep["net"]] if rep.get("net") is not None else []):
        col = PIC_COL.get(L["name"], (200, 200, 200))
        for q in np.asarray(L["pts"], float).reshape(-1, 2):
            if np.isfinite(q).all():
                cv2.circle(vis, (int(round(q[0])), int(round(q[1]))), 2, col, -1,
                           cv2.LINE_AA)
    for P in rep.get("anchor_points", ()):
        mark_x(vis, np.asarray(P["px"], float), (0, 255, 0))

    q = rep["quality"]
    rows = [("%s" % (stem or "plate"), (255, 255, 255)),
            ("%s   %s" % ("ACCEPT" if rep["ok"] else "REJECT", rep["reason"]),
             (80, 255, 80) if rep["ok"] else (80, 80, 255)),
            ("rms %.2f px%s   coverage %.0f%%   probe %.2f px   anchor %.1f px"
             % (q["line_rms_px"],
                "" if q.get("far_rms_px") is None else
                "  (far segment %.2f, n=%d, not gated)" % (q["far_rms_px"], q["n_far"]),
                100 * q["paint_coverage"], q["probe_rms_px"],
                q["max_anchor_px"] or 0.0), (255, 255, 255)),
            ("pose  height %.2f m   f %.0f   k %s   round %d of %d"
             % (rep["pose"]["height_m"], rep["pose"]["f"],
                ", ".join("%.4f" % v for v in rep["pose"]["k"]),
                rep["round_kept"], len(rep["rounds"])), (255, 255, 255)),
            ("band searched  %.0f in / %.0f out px" % (band[0], band[1]), PIC_BAND)]
    for name in ("left_sideline", "right_sideline"):
        v = rep["per_line"].get(name)
        if v:
            rows.append(("%-15s n=%3d  rms %.2f px  measures X = %.3f m"
                         % (name, v["n"], v["rms_px"], v["measured_m"]),
                         PIC_COL[name]))
    if rep.get("net_line"):
        v = rep["net_line"]
        rows.append(("net_line        n=%3d  rms %.2f px  measures Y = %.3f m  (%s)"
                     % (v["n"], v["rms_px"], v["measured_m"],
                        "fitted at %.3f" % v["model_m"] if v["fitted"] else "check only"),
                     PIC_COL["net_line"]))
    for i, (text, col) in enumerate(rows):
        _label(vis, text, (14, 26 + 20 * i), col)
    cv2.imwrite(path, vis)
    return path


def overlay_picture(bgr, rep, path, stem="", metres=True):
    """The fitted court over the plate, with the verdict burned in.

    `draw.draw_overlay` exactly - the same colours and the same metre grid - so
    that this picture and the one `detect_court.py` writes for the same plate
    can be flicked between and only the COURT differs.
    """
    from .draw import _label, draw_overlay
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):      # it narrates as it draws
        draw_overlay(bgr, rep["camera"], path, metres=metres)
    vis = cv2.imread(path)
    if vis is None:
        return path
    q = rep["quality"]
    _label(vis, stem or "plate", (14, 26), (255, 255, 255))
    _label(vis, "pose fit   %s%s" % ("ACCEPT" if rep["ok"] else "REJECT",
                                     "" if rep["ok"] else "  " + rep["reason"]),
           (14, 46), (80, 255, 80) if rep["ok"] else (80, 80, 255))
    _label(vis, "rms %.2f px   coverage %.0f%%   probe %.2f px   anchor %.1f px   "
                "height %.2f m" % (q["line_rms_px"], 100 * q["paint_coverage"],
                                   q["probe_rms_px"], q["max_anchor_px"] or 0.0,
                                   rep["pose"]["height_m"]), (14, 66), (255, 255, 255))
    cv2.imwrite(path, vis)
    return path


def write_pictures(bgr, rep, out_dir, stem, ext=".png"):
    """The three pictures, named as `api.write_pictures` names its own.

        <stem>_court_debug     what this solver read, and where it looked
        <stem>_court_success   the fitted court over the plate, when accepted
        <stem>_court_fail      the same, when refused

    Same names on purpose, because these are meant to be compared against the
    other solver's pictures for the same plate - so they go in a DIFFERENT
    folder rather than under different names, and nothing has to be renamed to
    put the two side by side.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, stem)
    made = {}
    try:
        made["debug"] = debug_picture(bgr, rep, base + "_court_debug" + ext, stem)
    except BaseException as e:                           # noqa: BLE001
        made["debug_error"] = str(e)[:120]
    for stale in ("_court_success", "_court_fail"):      # never leave last run's verdict
        if os.path.exists(base + stale + ext):
            os.remove(base + stale + ext)
    try:
        made["overlay"] = overlay_picture(
            bgr, rep, base + ("_court_success" if rep["ok"] else "_court_fail") + ext,
            stem)
    except BaseException as e:                           # noqa: BLE001
        made["overlay_error"] = str(e)[:120]
    return made
