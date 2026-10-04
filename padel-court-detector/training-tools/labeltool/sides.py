"""Read `<stem>_courtside-detection.json` so the marker can look at it.

The file is written by `compare_sides.py` (unless `--no-write-detections`) or by
`detect_court.py --sides-json`, and holds, for one plate, what EVERY courtside
reader found and the quadratic through it. This module only reshapes it for the
canvas; it never fits anything, because a picture that re-derives the fit is a
picture of this module rather than of the detector.

Nothing here is required. A folder with no detection files simply has the
dropdown disabled, which is the state every folder starts in.
"""
import os

import numpy as np

from padelcourt.walk import SIDE_SOURCE_KEYS

from . import store

DETECTION = "_courtside-detection.json"

#: The order the dropdown offers them in - the cascade's own order, so the list
#: reads as "what would be tried first" top to bottom.
#:
#: TAKEN FROM THE DETECTOR RATHER THAN REPEATED HERE. A second copy of this
#: tuple is exactly how `chord_lines` came to be a registered reader, written
#: into every detection file, and still not offered in this dropdown: nothing
#: failed, the list simply did not know about it. Anything keyed by reader below
#: is derived from this one, so a sixth reader needs a colour and nothing else.
ORDER = tuple(SIDE_SOURCE_KEYS)

#: One colour per reader, held apart from the marking colours (red/green/blue)
#: on purpose: what is being compared here is the detectors against each other,
#: and against the marks, so neither set may be mistaken for the other. A reader
#: with no colour here still draws, in white - visible and obviously unstyled,
#: which is the right way for this to fail.
COLOUR = {"canny": "#f59e0b", "track": "#a78bfa", "scans": "#f472b6",
          "chord_lines": "#fb923c", "chord": "#22d3ee"}

#: Where the near service line's paint stopped - the one point every reader is
#: judged against. Drawn as an X converging on it so a reader that missed it is
#: obvious, and so it looks the same as the ends the walk overlay marks: it is
#: the same point, arriving from a different file.
CORNER_COLOUR = "#ffffff"

#: Radius of a detected sample on screen, in pixels, with one reader shown.
#: Drawn with a dark outline rather than bare: a bare fill at this size
#: disappears into the court and into its own reader's curve, and the outline is
#: what separates a SAMPLE from the line fitted through it, which is the
#: comparison the overlay exists for.
DOT_R = 4
DOT_OUTLINE = "#111111"


def dot_radius(key, choice):
    """Screen radius for one reader's samples under this dropdown choice.

    With every reader shown, one radius each in `ORDER`, largest first.

    NOT COSMETIC. In the ordinary case the readers agree to within a couple of
    pixels, so drawn at one size they land on top of each other and all you see
    is whichever was drawn last - the picture says "the chord found this", when
    what it should say is "they all agree". Stepping the radius down in draw
    order nests them: agreement reads as a target of concentric rings, and a
    reader that wandered off leaves its own ring behind on its own. Which is the
    question this overlay is opened to answer.

    Computed from `ORDER` rather than tabulated, so the rings stay nested however
    many readers there are - a fixed table indexed by position is a crash waiting
    for the next one.
    """
    if choice != ALL or key not in ORDER:
        return DOT_R
    return DOT_R + (len(ORDER) - 1 - ORDER.index(key))


OFF, ALL = "off", "all (%d)" % len(ORDER)

#: What the dropdown offers. `off` first, so the default costs nothing to read.
CHOICES = (OFF, ALL) + ORDER


def path_for(plate):
    return store.sidecar(plate, DETECTION)


def load(plate):
    """`(doc, note)` for one plate, or `(None, note)` when there is no file.

    `note` is a sentence for the status line, never an exception: a plate with
    no detection file is the ordinary case, not an error.
    """
    p = path_for(plate)
    if not os.path.exists(p):
        return None, None
    doc = store.read_json(p)
    if doc is None:
        return None, "%s is unreadable" % os.path.basename(p)
    return doc, None


def layers(doc, choice):
    """What to draw for this dropdown choice.

    Returns a list of `{key, colour, side, points, polyline, chosen, miss,
    rms, qualifies, why}`, one per (reader, sideline) that has anything to show.
    A reader that was REFUSED is still returned - seeing where a rejected
    detector went is most of the value of looking at all.
    """
    if not doc or choice == OFF:
        return []
    keys = ORDER if choice == ALL else (choice,)
    out = []
    for key in keys:
        t = (doc.get("techniques") or {}).get(key)
        if not t:
            continue
        for side, s in sorted((t.get("sides") or {}).items()):
            if not s.get("available") or not s.get("points"):
                continue
            fit = s.get("fit") or {}
            out.append({
                "key": key, "side": side, "colour": COLOUR.get(key, "#ffffff"),
                "label": t.get("label", key),
                "points": np.asarray(s["points"], float).reshape(-1, 2),
                "polyline": np.asarray(fit.get("polyline") or [], float).reshape(-1, 2),
                "chosen": bool(s.get("chosen")),
                "miss": s.get("corner_miss_px"),
                "rms": fit.get("rms_px"),
                "qualifies": bool(s.get("qualifies")),
                "why": s.get("why") or "",
            })
    return out


def corners(doc):
    """The two points the readers are judged against, as `{side: (x, y)}`."""
    return {k: np.asarray(v, float) for k, v in (doc.get("corners") or {}).items()}


def summary(doc, choice):
    """One line for the status bar: how each shown reader did, per side.

    Every reader at once will not fit as sentences, so `all` is compressed to
    the one number that decides anything - how far the reader's own curve lands
    from the corner it had to reach - with the refused ones marked. One reader
    on its own gets the room to say the rest.
    """
    if not doc or choice == OFF:
        return ""
    L = layers(doc, choice)
    if not L:
        return "nothing recorded for this reader"
    if choice == ALL:
        per = {}
        for x in L:
            per.setdefault(x["key"], {})["L" if x["side"].startswith("left")
                                         else "R"] = x
        bits = []
        for key in ORDER:
            if key not in per:
                continue
            bits.append("%s %s" % (key, "/".join(
                ("-" if per[key][s]["miss"] is None else "%.0f" % per[key][s]["miss"])
                + ("" if per[key][s]["qualifies"] else "x")
                for s in ("L", "R") if s in per[key])))
        return "corner miss L/R px:   " + "   ".join(bits)
    return "   ".join(
        "%s%s %d pts, corner %s px%s%s"
        % (x["key"], "L" if x["side"].startswith("left") else "R",
           len(x["points"]),
           "-" if x["miss"] is None else "%.0f" % x["miss"],
           "" if x["rms"] is None else ", fit %.2f px" % x["rms"],
           "" if x["qualifies"] else " REFUSED")
        for x in L)
