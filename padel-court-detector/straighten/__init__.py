"""Fisheye undistortion - the lens, and nothing else.

    from straighten import fisheye_coefficients, undistort_image

    lens = fisheye_coefficients("plate.png")
    print(lens.f, lens.k1, lens.k2)
    flat = undistort_image(cv2.imread("plate.png"), lens)

The lens is solved from straight edges: every one the scene happens to contain -
glass frames, posts, roof beams - plus the court's own lines, walked by
`padelcourt.walk` and fed to the same objective. See `lens.py` for the
objective, `court.py` for what the court contributes and why it needs handling
the building's edges do not, and README.md for what it is worth.

No homography and no court MODEL: what comes out is a radial mapping, not a
calibration. `fisheye_coefficients(img, court=False)` is the solve with the
court left out, which needs nothing outside this package.
"""

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import lens as _lens
from .lens import (CORNERS, blank_corners, canny, fwd_map, gauge, inv_map,  # noqa: F401
                   r_ideal, radial, theta_of, undistort_maps, undistort_pts)

__all__ = ["Lens", "fisheye_coefficients", "solve_lens", "undistort_image",
           "undistort_points", "straighten_file", "IMAGE_EXT"]

#: Extensions `batch_straighten` will pick up.
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class Lens:
    """The solved fisheye distortion.

        r_dist = f * theta * (1 + k1 theta^2 + k2 theta^4)

    `f` is in pixels and `k` is however many coefficients were asked for (two by
    default). `cx, cy` is the distortion centre, which this solver fixes at the
    image centre rather than fitting - a real principal point can be tens of
    pixels off, but freeing it per image costs more robustness than it buys.

    `Rref` is the gauge radius: undistorted coordinates are defined only up to
    scale, and the mapping is normalised to hold that radius (the image corner)
    fixed. Anything that consumes `f, k` must use the same convention.

    Do not compare two of these by their coefficients - see `displacement`.
    """

    f: float
    k: tuple
    cx: float
    cy: float
    Rref: float
    width: int = 0
    height: int = 0
    sag_before: float = float("nan")
    sag_after: float = float("nan")
    arcs: int = 0
    straight: int = 0
    source: str = ""
    #: {line name: (signed sagitta before, after)} in source pixels, for the
    #: walked court lines that were fed to the fit. Empty when the solve was
    #: blind. See `court_sag` for the one-number version.
    court: dict = field(default_factory=dict, repr=False)
    report: dict = field(default_factory=dict, repr=False)

    @property
    def k1(self):
        return self.k[0] if len(self.k) > 0 else 0.0

    @property
    def k2(self):
        return self.k[1] if len(self.k) > 1 else 0.0

    @property
    def centre(self):
        return np.array([self.cx, self.cy], float)

    @property
    def court_sag(self):
        """Worst |sagitta| left on a court line, in source pixels. nan if blind.

        The honest headline for a court-informed solve: how far from straight
        the lines that will be calibrated against still are. Taken as the worst
        rather than the median because one line bowed the wrong way is the
        failure this evidence exists to prevent, and a median hides it.
        """
        if not self.court:
            return float("nan")
        return float(max(abs(b) for _, b in self.court.values()))

    def as_dict(self):
        """Plain JSON-safe dict, including the radial mapping it implies."""
        d = {"model": "fisheye", "f": float(self.f),
             "k": [float(x) for x in self.k],
             "cx": float(self.cx), "cy": float(self.cy), "Rref": float(self.Rref),
             "width": int(self.width), "height": int(self.height),
             "sag_before": float(self.sag_before), "sag_after": float(self.sag_after),
             "arcs": int(self.arcs), "straight": int(self.straight),
             "court": {n: [float(a), float(b)] for n, (a, b) in self.court.items()},
             "court_sag": self.court_sag}
        if self.source:
            d["source"] = self.source
        for r, v in zip(DEFAULT_RADII, self.displacement()):
            d[f"d{int(r)}"] = round(float(v), 3)
        return d

    def displacement(self, radii=None):
        """How far a feature at radius r moves when the frame is straightened.

        THIS is the thing to compare between two solves, not (f, k1, k2). Those
        three are badly degenerate: across 103 plates of one camera model f
        spanned 575-1626 and k1 spanned -0.47 to +0.29 with sign changes, while
        these displacements agreed to about 3%. Two plates of one camera came
        back as f=1626, k1=-0.47 and f=840, k1=+0.12 with mappings 13 px apart
        at r=400. The parameters are a parameterisation; the mapping is the
        measurement.

        Zero at `Rref` by construction - that is the gauge.
        """
        r = np.asarray(DEFAULT_RADII if radii is None else radii, float)
        return gauge(self.f, self.k, self.Rref) * r_ideal(r, self.f, self.k) - r


#: Radii, in pixels, that `as_dict` and `displacement` report at by default.
DEFAULT_RADII = (200, 400, 600, 800, 1000)


def _read(image):
    if isinstance(image, np.ndarray):
        return image, ""
    img = cv2.imread(str(image))
    if img is None:
        raise SystemExit(f"cannot read {image}")
    return img, str(image)


def fisheye_coefficients(image, nk=2, thr=255, corners=CORNERS, court=True,
                         log=None, **kw):
    """Solve the fisheye distortion of one image. -> Lens

    `image` is a path or a BGR array. `nk` is how many radial coefficients to
    fit; two is the default and three is not better - the model is already
    degenerate at two, and more parameters flatten the valley rather than
    determine the mapping.

    `court=True` walks the court's own lines and feeds them to the same
    objective alongside the hall's edges, which is worth a factor of 1.5-4 on
    the two cameras there is a click-calibrated answer for. It costs the walk,
    and falls back to the blind solve wherever the walk cannot start.
    `court=False` is the hall-edges-only solve, which needs nothing outside this
    package.

    Raises RuntimeError if the frame has too few usable straight edges.
    """
    img, src = _read(image)
    f, k, rep = _lens.solve(img, nk=nk, thr=thr, corners=corners, court=court,
                            log=log, **kw)
    H, W = img.shape[:2]
    return Lens(f=float(f), k=tuple(float(x) for x in k),
                cx=float(rep["c"][0]), cy=float(rep["c"][1]), Rref=float(rep["Rref"]),
                width=W, height=H, sag_before=rep["sag_before"],
                sag_after=rep["sag_after"], arcs=rep["arcs"],
                straight=rep["straight"], source=src, court=rep.get("court") or {},
                report=rep)


#: `fisheye_coefficients` under the name the rest of this project uses.
solve_lens = fisheye_coefficients


def undistort_image(img, lens, zoom=None, pad=1.35, interpolation=cv2.INTER_LINEAR):
    """Straighten a whole frame. -> (image, zoom, canvas)

    The fitting gauge holds the image corners still, which squeezes everything
    inside; for looking at, `zoom = 1/gauge` on a larger canvas keeps the middle
    of the frame at its original resolution instead. `pad` sets how much larger.
    Pass `zoom=1.0, pad=1.0` to get the gauge-exact rectification at the input
    size, which is what `fwd_map`/`inv_map` assume by default.
    """
    if zoom is None:
        zoom = 1.0 / gauge(lens.f, lens.k, lens.Rref)
    H, W = img.shape[:2]
    canvas = (int(round(H * pad)), int(round(W * pad)))
    mx, my = undistort_maps(img.shape, lens.centre, lens.f, lens.k, lens.Rref,
                            zoom, canvas)
    return cv2.remap(img, mx, my, interpolation), zoom, canvas


def undistort_points(pts, lens):
    """Pixels -> centred, gauge-fixed undistorted pixels. -> (N, 2)"""
    return undistort_pts(pts, lens.centre, lens.f, lens.k, lens.Rref)


def straighten_file(path, out, **kw):
    """Solve one image file and write the straightened version. -> Lens"""
    img, _ = _read(path)
    lens = fisheye_coefficients(img, **kw)
    lens.source = str(path)
    flat, _, _ = undistort_image(img, lens)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if not cv2.imwrite(str(out), flat):
        raise SystemExit(f"cannot write {out}")
    return lens
