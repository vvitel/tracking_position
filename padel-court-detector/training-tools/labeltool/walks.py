"""Run the detector's walk over a plate and write `_court-lines.json`.

The walk is the expensive half of `padelcourt` - roughly seven seconds a plate -
and it is entirely deterministic, so it is done once per plate and cached in a
file beside it. What is saved is what `walk.detect` returns: for each line it
found, the pixel coordinates of the samples it centred on the paint, together
with the world line they belong to. That is exactly what `solve` consumes, so
the labelled fit later re-reads this file and needs nothing else from the
detector.

The walk's own point anchors are saved too. They are the near service line's two
ends and the far service T - the only things in an otherwise line-based pipeline
that say where ALONG a line something sits (see `walk.anchors`), and the
labelled fit wants them for the same reason the detector does.

So are the two service lines' four ends, which is a longer list than the anchors
for the reason `_ends` gives: they are drawn on the picture as X's, and what is
worth looking at there is every end the walk placed, including the ones the fit
was not allowed to use.

`generate` never raises. It runs in a worker process, and a plate the walk dies
on is a plate to mark and move past, not a reason to take the pool down with it.
"""
import os
import traceback

import cv2
import numpy as np

from . import store


def _round(a, nd=2):
    """Points as plain lists at sub-pixel precision.

    Two decimals is a hundredth of a pixel, which is far below anything the walk
    claims, and it takes a 450-point line from ~11 kB of float repr to ~4.
    """
    return [[round(float(x), nd), round(float(y), nd)] for x, y in np.asarray(a, float).reshape(-1, 2)]


#: The two service lines' four ends, keyed as `compare_ends.py` keys them.
#:
#: SAVED SEPARATELY FROM THE ANCHORS even though two of the four are also
#: anchors, because the two sets answer different questions. `anchors` is what
#: the FIT is allowed to be pulled by, so it is filtered - a corner too near the
#: frame edge is dropped, and the far service ends were never in it at all,
#: since what they hold is the chord readers' strip and not the solve. This is
#: every end the walk placed, whatever became of it, which is what somebody
#: checking the endpoints wants to look at.
def _ends(trace):
    out = {}
    for key, val in (("near", trace.get("corners")),
                     ("far", trace.get("chord_ends"))):
        if val is None:
            continue
        p = np.asarray(val, float).reshape(-1, 2)
        # The far pair is None on a plate whose far service walk stopped
        # asymmetrically - there is no chord there - so a plate legitimately has
        # two ends or four.
        if len(p) == 2:
            out[key + "_l"], out[key + "_r"] = _round(p)
    return out


def _document(plate, shape, lines, anchors, ends=None, error=None):
    h, w = shape[:2]
    return {
        "plate": os.path.basename(plate),
        "image_size": {"width": int(w), "height": int(h)},
        "generated_by": "padelcourt.walk.detect",
        "ok": error is None and len(lines) > 0,
        "error": error,
        "note": ("`points` are image pixels, centred on the line the walk was "
                 "following. `world_a`/`world_b` are two points on the court "
                 "line they belong to, in metres. `service_ends` are where each "
                 "service line's paint stopped, in image pixels."),
        "lines": lines,
        "anchors": anchors,
        "service_ends": ends or {},
    }


def generate(plate):
    """Walk `plate`, write its `_court-lines.json`, return a small summary.

    Never raises: the return value carries the failure instead.
    """
    out = store.sidecar(plate, store.LINES)
    try:
        from padelcourt.walk import detect

        bgr = cv2.imread(plate)
        if bgr is None:
            raise RuntimeError("cannot read %s" % os.path.basename(plate))
        trace = {}
        found, ctx = detect(bgr, verbose=False, with_ctx=True, trace=trace)
        lines = [{"name": L["name"],
                  "world_a": [float(v) for v in L["world_a"]],
                  "world_b": [float(v) for v in L["world_b"]],
                  "n": int(len(L["pts"])),
                  "points": _round(L["pts"])}
                 for L in found]
        anchors = [{"name": A["name"],
                    "world": [float(v) for v in A["world"]],
                    "image_px": [round(float(v), 2) for v in A["px"]],
                    "sigma": (None if A.get("sigma") is None else float(A["sigma"]))}
                   for A in ctx.get("anchors", [])]
        doc = _document(plate, bgr.shape, lines, anchors, _ends(trace))
    except BaseException as e:                    # the detector raises SystemExit
        shape = (0, 0)
        try:
            im = cv2.imread(plate)
            if im is not None:
                shape = im.shape
        except BaseException:
            pass
        doc = _document(plate, shape, [], [], error=str(e)[:300] or e.__class__.__name__)
        doc["traceback"] = traceback.format_exc()[-1200:]
    store.write_json(out, doc)
    return {"plate": plate, "ok": doc["ok"], "error": doc["error"],
            "n_lines": len(doc["lines"]),
            "kept": {L["name"]: L["n"] for L in doc["lines"]}}


def load(plate):
    """The saved walk as `solve` wants it: [{name, pts, world_a, world_b}].

    Returns (lines, anchors, ends, error). `error` is a sentence to show the
    labeller, not an exception - a plate whose walk failed can still be marked,
    it just cannot be fitted.
    """
    doc = store.read_json(store.sidecar(plate, store.LINES))
    if doc is None:
        return [], [], {}, "no %s - generate the walks first" % store.LINES
    if not doc.get("ok"):
        return [], [], {}, "the walk failed on this plate: %s" % (doc.get("error") or "?")
    lines = [{"name": L["name"], "pts": np.asarray(L["points"], float).reshape(-1, 2),
              "world_a": np.asarray(L["world_a"], float),
              "world_b": np.asarray(L["world_b"], float)}
             for L in doc.get("lines", [])]
    anchors = [{"name": A["name"], "world": tuple(A["world"]),
                "px": np.asarray(A["image_px"], float), "sigma": A.get("sigma")}
               for A in doc.get("anchors", [])]
    ends = {k: np.asarray(v, float) for k, v in (doc.get("service_ends") or {}).items()}
    # A FILE WRITTEN BEFORE THE ENDS WERE SAVED still has two of them, under
    # their anchor names, and showing those beats showing nothing: a folder
    # walked last week would otherwise mark no endpoints at all until every
    # plate in it had been walked again. The far pair is only in the newer files,
    # so a stale plate marks two X's and a fresh one four.
    for A in anchors:
        for name, key in (("near_service_end_l", "near_l"),
                          ("near_service_end_r", "near_r")):
            if A["name"] == name and key not in ends:
                ends[key] = A["px"]
    return lines, anchors, ends, None
