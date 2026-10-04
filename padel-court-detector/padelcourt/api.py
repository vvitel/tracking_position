"""What a caller wants back: a plate, a lens, a homography and 13 keypoints.

The detector lives in `walk`, the fit in `solve`, the verdict in `calibrate`;
this module is only the shape of the answer. It exists so that no caller has to
know that the camera is a homography onto IDEAL normalised coordinates followed
by a radial distortion - which is the one thing about this model that surprises
people, because it means the plain 3x3 homography is valid on an undistorted
image and nowhere else.

The keypoints, by contrast, are given in ORIGINAL image pixels: they are
projected through the full model, distortion included, so they can be drawn on
the frame as it comes out of the camera.
"""
import json
import os

import cv2
import numpy as np

from . import court as CM
from . import plate as plate_mod
from . import walk as walk_mod
from .calibrate import calibrate
from .draw import draw_overlay, draw_walk
from .image import undistort_maps

#: The 11 painted keypoints, plus the two near corners. The near corners are
#: DERIVED rather than detected - on a typical mount they are outside the frame -
#: so they are reported with `in_frame: false` and are still useful, because a
#: homography is happy to be evaluated outside the image.
ALL_KEYPOINTS = [(n, x, y) for n, x, y in CM.KEYPOINTS] + list(CM.DERIVED)

#: Written into the result so that whoever reads the JSON can undo the lens
#: without reading this code. `r` is the radius in normalised coordinates,
#: i.e. ((u - cx) / f, (v - cy) / f).
LENS_EQUATIONS = {
    "fisheye": ("theta   = atan(r_ideal)\n"
                "theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + ...)\n"
                "r_dist  = theta_d\n"
                "u = f * x_dist + cx,   v = f * y_dist + cy"),
    "poly": ("r_dist = r_ideal * (1 + k1*r_ideal^2 + k2*r_ideal^4 + ...)\n"
             "u = f * x_dist + cx,   v = f * y_dist + cy"),
}


class CourtResult(object):
    """One calibrated plate.

    `ok` is the pipeline's own verdict and `reason` says what failed. A result
    that is not `ok` still carries a full camera: on a plate whose far half is
    faint the fit can be geometrically right and still miss a coverage gate, so
    the caller is given the numbers rather than a silent None.
    """

    def __init__(self, plate, plate_path, camera, report, coverage, probe_rms,
                 ok, reason, trace=None, source=None):
        self.plate = plate
        self.plate_path = plate_path
        self.camera = camera
        self.report = report
        self.coverage = float(coverage)
        self.probe_rms = float(probe_rms)
        self.ok = bool(ok)
        self.reason = reason or ""
        self.trace = trace or {}
        self.source = source

    # -- the three things a caller asked for -------------------------------
    def keypoints(self):
        return keypoints_of(self.camera, self.plate.shape)

    def homography(self):
        return homography_of(self.camera)

    def lens(self):
        return lens_of(self.camera)

    def to_dict(self):
        rep = self.report or {}
        return {
            "source": self.source,
            "plate": self.plate_path,
            "image_size": {"width": int(self.plate.shape[1]),
                           "height": int(self.plate.shape[0])},
            "accepted": self.ok,
            "reason": self.reason,
            "quality": {
                "line_rms_px": float(rep.get("rms_line_px", float("nan"))),
                "paint_coverage": self.coverage,
                "probe_rms_px": self.probe_rms,
                "max_anchor_px": (None if rep.get("max_anchor_px") is None
                                  else float(rep["max_anchor_px"])),
                "n_anchors": int(rep.get("n_anchors", 0) or 0),
                "n_lines": int(rep.get("n_lines", 0) or 0),
                "n_samples": int(rep.get("n_samples", 0) or 0),
            },
            "lens": self.lens(),
            "homography": self.homography(),
            "keypoints": self.keypoints(),
            "camera_model": self.camera.to_dict(),
            "court_dimensions_m": {"width": CM.WIDTH, "length": CM.LENGTH,
                                   "net_y": CM.NET_Y,
                                   "service_line_y": [CM.FAR_SERVICE_Y, CM.NEAR_SERVICE_Y],
                                   "centre_line_x": CM.CENTRE_X},
        }

    def save_json(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1, sort_keys=False)
        return path


# ---------------------------------------------------------------- pieces ----
def keypoints_of(cam, shape=None):
    """The court's keypoints in ORIGINAL image pixels, distortion included."""
    W = np.array([[x, y] for _, x, y in ALL_KEYPOINTS], float)
    P = cam.project(W)
    out = []
    for (name, X, Y), (u, v) in zip(ALL_KEYPOINTS, P):
        row = {"name": name, "world_m": [float(X), float(Y)],
               "image_px": [float(u), float(v)],
               "derived": name not in CM.KP_INDEX}
        if shape is not None:
            row["in_frame"] = bool(0 <= u < shape[1] and 0 <= v < shape[0])
        out.append(row)
    return out


def lens_of(cam):
    """The distortion coefficients, and the equation they belong to."""
    return {"model": cam.model,
            "k": [float(v) for v in cam.k],
            "f": float(cam.f), "cx": float(cam.cx), "cy": float(cam.cy),
            "equation": LENS_EQUATIONS.get(cam.model, ""),
            "note": ("r_ideal is the radius in normalised coordinates, "
                     "x_ideal = (u - cx) / f. Solve the equation for theta (or "
                     "r_ideal) to undistort; Camera.undistort() does it by "
                     "Newton iteration.")}


def homography_of(cam):
    """Court metres -> pixels, and where each of the two forms is valid."""
    return {
        "matrix_undistorted_px": [[float(v) for v in row]
                                  for row in cam.homography_px()],
        "matrix_normalised": [[float(v) for v in row] for row in cam.Hn],
        "maps": "court (X, Y, 1) in metres -> homogeneous pixels",
        "valid_on": ("an UNDISTORTED image - remap the frame with "
                     "undistort_image() first. On a raw frame use the full "
                     "model (Camera.project), which applies the lens after "
                     "this homography."),
    }


def undistort_image(img, cam, balance=1.0):
    """The frame with the lens taken out, so the plain homography applies."""
    mx, my = undistort_maps(cam, img.shape, balance)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR)


# ------------------------------------------------------------- entry points --
def calibrate_plate(plate, rounds=2, model="fisheye", nk=2, plate_path=None, source=None):
    """Calibrate one plate (a BGR image). Raises RuntimeError if the walk dies."""
    trace = {}
    try:
        cam, rep, cov, prms, ok, why = calibrate(plate, rounds=rounds, model=model,
                                                 nk=nk, trace=trace)
    except BaseException as e:                   # the detector raises SystemExit
        raise RuntimeError("court detection failed: %s" % e)
    return CourtResult(plate, plate_path, cam, rep, cov, prms, ok, why, trace, source)


def calibrate_image_file(path, **kw):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("cannot read %s" % path)
    return calibrate_plate(img, plate_path=os.path.abspath(path),
                           source=os.path.abspath(path), **kw)


def build_plate(video, frames=45, force=False, verbose=True):
    """Median plate for a video, written beside it as `<name>_plate.png`."""
    plate, path, ghost = plate_mod.build(video, target=frames, force=force, verbose=verbose)
    return plate, path, ghost


def calibrate_video(video, frames=45, force=False, verbose=True, **kw):
    """Build the plate beside the video, then calibrate it."""
    plate, path, ghost = build_plate(video, frames=frames, force=force, verbose=verbose)
    res = calibrate_plate(plate, plate_path=os.path.abspath(path),
                          source=os.path.abspath(video), **kw)
    res.ghost = ghost
    return res


# -------------------------------------------------- the courtside readers ----
#: Where the sideline JSON's fits are sampled, in image rows, so a reader can
#: draw the curve without re-implementing `np.polyval`. Only ever drawn over the
#: span the points themselves cover - a quadratic extrapolated past its own
#: evidence is the failure `SIDE_CORNER_MAX` exists to catch, and a picture that
#: hides it would defeat the point of looking.
SIDE_FIT_SAMPLES = 40


def _side_fit(pts):
    """The quadratic `add()` would fit to these sideline points, and its rms.

    Sidelines are fitted with `swap=True` - x as a function of y - because they
    run down the picture, so a fit in the other direction is vertical where it
    matters most. Same trim and same degree as `walk.detect`'s own, so what is
    reported here is the fit that would be used, not a lookalike.
    """
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) < 10:
        return None
    kept = walk_mod.trim_to_curve(pts, True)
    if len(kept) < 10:
        return None
    y, x = kept[:, 1], kept[:, 0]
    try:
        c = np.polyfit(y, x, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    rms = float(np.sqrt(((x - np.polyval(c, y)) ** 2).mean()))
    ys = np.linspace(float(y.min()), float(y.max()), SIDE_FIT_SAMPLES)
    return {"form": "x = c[0]*y^2 + c[1]*y + c[2]",
            "coeffs": [float(v) for v in c],
            "y_range": [float(y.min()), float(y.max())],
            "rms_px": rms,
            "n_kept": int(len(kept)),
            "kept": [[float(a), float(b)] for a, b in kept],
            "polyline": [[float(np.polyval(c, t)), float(t)] for t in ys]}


def courtside_detection(res):
    """Every courtside reader's points and fit, for one plate.

    Needs a run made with `walk.SIDE_RECORD_ALL` (or `PADEL_SIDE_RECORD_ALL`),
    which is what puts all four answers in the trace instead of just the winner.
    Returns None when the trace does not carry them.

    The point of the file is comparison, so nothing here is filtered: a reader
    that was refused keeps its points and says why it was refused, because the
    picture of a detector on the wrong feature is the thing worth seeing.
    """
    cand = res.trace.get("side_candidates")
    if not cand:
        return None
    corners = res.trace.get("side_corners", {})
    how = res.trace.get("side_how", {})
    chosen = {}
    for side, label in how.items():
        chosen[side] = next((k for k, v in cand.get(side, {}).items()
                             if v["label"] == label), None)
    techniques = {}
    for key in walk_mod.SIDE_SOURCE_KEYS:
        sides = {}
        for side in ("left_sideline", "right_sideline"):
            c = cand.get(side, {}).get(key)
            if c is None:
                # The chord is the only reader that can be absent rather than
                # refused: it needs both service walks to have stopped
                # symmetrically before there is a chord to read along at all.
                sides[side] = {"available": False,
                               "why": "not offered on this plate", "points": []}
                continue
            pts = np.asarray(c["points"], float).reshape(-1, 2)
            sides[side] = {
                "available": True,
                "n_points": int(len(pts)),
                "points": [[float(a), float(b)] for a, b in pts],
                "corner_miss_px": (None if c["corner_miss"] is None
                                   else float(c["corner_miss"])),
                "qualifies": bool(c["qualifies"]),
                "why": c["why"],
                "chosen": chosen.get(side) == key,
                "fit": _side_fit(pts),
            }
        techniques[key] = {
            "label": next((c["label"] for side in sides
                           for c in [cand.get(side, {}).get(key)] if c), key),
            "sides": sides,
        }
    return {
        "plate": res.plate_path,
        "source": res.source,
        "image_size": {"width": int(res.plate.shape[1]),
                       "height": int(res.plate.shape[0])},
        "accepted": res.ok,
        "reason": res.reason,
        # The one point every reader is judged against: where the near service
        # line's paint actually stopped. `walk.SIDE_CORNER_MAX` is the gate.
        "corners": {k: [float(v[0]), float(v[1])] for k, v in corners.items()},
        "corner_gate_px": walk_mod.SIDE_CORNER_MAX,
        "cascade": list(walk_mod.SIDE_SOURCE_KEYS),
        "chosen": chosen,
        "side_how": how,
        "techniques": techniques,
    }


def save_courtside_detection(res, path):
    """Write `courtside_detection` to `path`. Returns the path, or None."""
    d = courtside_detection(res)
    if d is None:
        return None
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=False)
    return path


#: `<plate>.png` -> `<plate>_courtside-detection.json`, beside the plate.
SIDES_SUFFIX = "_courtside-detection.json"


def courtside_detection_path(plate_path):
    return os.path.splitext(plate_path)[0] + SIDES_SUFFIX


# ----------------------------------------------------------------- pictures --
def write_pictures(res, out_dir, stem=None, ext=".jpg", debug=True, overlay=True,
                   metres=True):
    """The three pictures, named after the plate.

    `<stem>_court_debug`   what the detector walked, written from the trace, so
                           it exists even when the fit was refused
    `<stem>_court_success` the fitted court drawn over the plate, when accepted
    `<stem>_court_fail`    the same, when refused - the picture that shows WHY
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = stem or os.path.splitext(os.path.basename(res.plate_path or "plate"))[0]
    base = os.path.join(out_dir, stem)
    made = {}
    note = ("accepted" if res.ok else "rejected: " + res.reason)
    if debug:
        try:
            draw_walk(res.plate, res.trace, base + "_court_debug" + ext,
                         title=stem, note=note)
            made["debug"] = base + "_court_debug" + ext
        except BaseException as e:
            made["debug_error"] = str(e)[:120]
    for stale in ("_court_success", "_court_fail"):
        p = base + stale + ext
        if os.path.exists(p):
            os.remove(p)                         # never leave last run's verdict behind
    if overlay:
        path = base + ("_court_success" if res.ok else "_court_fail") + ext
        try:
            draw_overlay(res.plate, res.camera, path, metres=metres)
            _caption(path, stem, res)
            made["overlay"] = path
        except BaseException as e:
            made["overlay_error"] = str(e)[:120]
    return made


def _caption(path, stem, res):
    """Burn the verdict into the picture - these get skimmed a hundred at a time."""
    im = cv2.imread(path)
    if im is None:
        return
    rep = res.report or {}
    tag = "%s   %s" % (stem, "ACCEPT" if res.ok else "REJECT")
    sub = "line_rms %.2f px   coverage %.0f%%   probe %.2f px   anchors %s   %s" % (
        rep.get("rms_line_px", float("nan")), 100 * res.coverage, res.probe_rms,
        ("%.1f px (%d)" % (rep["max_anchor_px"], rep.get("n_anchors", 0))
         if rep.get("max_anchor_px") is not None else "none"),
        res.reason)
    cv2.rectangle(im, (0, 0), (im.shape[1], 62), (0, 0, 0), -1)
    col = (0, 255, 0) if res.ok else (0, 0, 255)
    cv2.putText(im, tag, (12, 26), 0, 0.72, col, 2, cv2.LINE_AA)
    cv2.putText(im, sub, (12, 51), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, im)
