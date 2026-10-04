"""The two pictures, and the measurement that decides which one is written.

`draw_overlay` puts the fitted court back on the plate; `draw_walk` shows what
the detector actually walked, which is the picture to look at when a fit is
refused - it is drawn from the trace, so it exists even when the walk threw.
"""
import cv2
import numpy as np

from . import court as CM
from .walk import T_BOX, WALK_LINES


def draw_overlay(img, cam, path, thickness=1, metres=False):
    out = img.copy()

    def poly(A, B, colour, t=thickness):
        s = np.linspace(0, 1, 96)[:, None]
        p = cam.project(np.asarray(A, float) + s * (np.asarray(B, float) - np.asarray(A, float)))
        cv2.polylines(out, [p.astype(np.int32)], False, colour, t, cv2.LINE_AA)

    poly([0, 0], [CM.WIDTH, 0], (0, 255, 0)); poly([0, CM.LENGTH], [CM.WIDTH, CM.LENGTH], (0, 255, 0))
    poly([0, 0], [0, CM.LENGTH], (0, 255, 0)); poly([CM.WIDTH, 0], [CM.WIDTH, CM.LENGTH], (0, 255, 0))
    poly([0, CM.FAR_SERVICE_Y], [CM.WIDTH, CM.FAR_SERVICE_Y], (0, 255, 255))
    poly([0, CM.NEAR_SERVICE_Y], [CM.WIDTH, CM.NEAR_SERVICE_Y], (0, 255, 255))
    poly([CM.CENTRE_X, CM.FAR_SERVICE_Y], [CM.CENTRE_X, CM.NEAR_SERVICE_Y], (0, 255, 255))
    poly([0, CM.NET_Y], [CM.WIDTH, CM.NET_Y], (255, 128, 0))
    if metres:
        for X in np.arange(1, CM.WIDTH):
            poly([X, 0], [X, CM.LENGTH], (90, 90, 90), 1)
        for Y in np.arange(1, CM.LENGTH):
            poly([0, Y], [CM.WIDTH, Y], (90, 90, 90), 1)
    cv2.imwrite(path, out)
    print("  wrote %s" % path)


#: Where the halo is stroked from, relative to the glyph: the eight neighbours
#: at one pixel plus the four axes at two, which closes into a ~2 px outline.
_HALO = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1),
         (-2, 0), (2, 0), (0, -2), (0, 2))


def _label(vis, text, org, col, scale=0.52):
    """Text straight onto the image, haloed so it stays readable anywhere.

    A filled caption bar is simpler but covers the top of the frame, which on
    this footage is exactly where the far half of the court is - the part a
    failed far-service detection has to be read against. Outlining each glyph
    obscures a few percent of the pixels instead of all of them. The halo
    inverts for dark text, so the black centre-line entry is legible too.

    THE HALO IS STROKED, NOT DRAWN THICK, and that is not a stylistic choice.
    `cv2.putText` widens the ADVANCE between characters along with the stroke, so
    the same string at thickness 4 is wider than at thickness 1 - 331 px against
    312 for a typical legend entry here - and the difference accumulates a
    fraction of a pixel per character rather than arriving as one offset that
    could be subtracted. Drawing a thick halo under a thin fill therefore lets
    the two de-register as they go, and by the end of a long line the halo is a
    whole word ahead: every legend entry ended in a ghost of its own last word,
    which read as "curve rms 0.2 px px". Stroking the halo at the fill's own
    thickness keeps both on identical metrics, so they can only ever coincide.
    """
    lum = 0.114 * col[0] + 0.587 * col[1] + 0.299 * col[2]
    halo = (255, 255, 255) if lum < 90 else (0, 0, 0)
    for dx, dy in _HALO:
        cv2.putText(vis, text, (org[0] + dx, org[1] + dy), 0, scale, halo, 1,
                    cv2.LINE_AA)
    cv2.putText(vis, text, org, 0, scale, col, 1, cv2.LINE_AA)


#: An X on a detected line END: how far its arms reach from the point, how far
#: they stop SHORT of it, and how thick they are, in pixels along each diagonal.
END_R, END_GAP, END_W = 12, 4, 2


def mark_x(vis, p, col, r=END_R, gap=END_GAP, width=END_W, halo=(0, 0, 0)):
    """Four arms pointing at exactly `p`, at sub-pixel precision.

    The endpoints are the one thing in this picture placed to a fraction of a
    pixel - `peel_ends` fits a local line to put them there, `corner_on_curve`
    then slides them along it, and `compare_ends.py` scores the result in
    centimetres - so a marker rounded to the nearest whole pixel throws away
    part of what there is to look at. `cv2.line` reads fixed-point coordinates
    when handed a `shift`, so each arm is aimed to within a sixteenth of a pixel
    of where the walk left the endpoint.

    THE CENTRE IS LEFT OPEN, which is the whole reason this is not a `drawMarker`
    call. An X drawn through its own crossing covers four or five pixels of the
    thing being marked - and on this picture the question at an endpoint is
    always "where does the paint stop", so the marker was hiding its own answer.
    Arms that stop `gap` short of the point converge on it without touching it:
    the pixel under the mark stays evidence.

    AN X AND NOT A PLUS, which is what the near corners used to get. A plus's
    arms lie along the two features that MAKE an endpoint - the service line runs
    horizontally through it, the sideline vertically - so even stopped short they
    run down the paint and read as part of it. Turned 45 degrees they cross open
    floor at every corner of the court.

    The halo is what keeps it readable on both the white paint it usually sits on
    and the dark floor just outside it; pass `halo=None` for bare arms.
    """
    S, k = 4, 16.0                               # sixteenths of a pixel
    x, y = float(p[0]), float(p[1])

    def q(dx, dy):
        return int(round((x + dx) * k)), int(round((y + dy) * k))

    for c, w in ((halo, width + 2), (col, width)) if halo else ((col, width),):
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.line(vis, q(sx * gap, sy * gap), q(sx * r, sy * r), c, w,
                     cv2.LINE_AA, S)


#: The chord and everything read along it. A colour of its own rather than the
#: sideline's, because it is a second opinion and not a second helping: the
#: question this part of the picture answers is whether two methods that share no
#: pixel-level machinery arrived at the same line, and that is unreadable if they
#: are drawn as though they were one set.
CHORD_COL = (0, 190, 255)

#: `chord_lines` gets a colour of its own for the same reason again: it and
#: `chord_side` read one strip two ways, and on almost every plate they land on
#: top of each other, so the only thing worth seeing is the plate where they do
#: not. Drawn as crosses against the other's diamonds, so a disagreement is
#: legible even where the two colours are hard to tell apart.
CHORD_LINE_COL = (255, 150, 0)

#: How far the chord's answer may sit from the sideline actually used, in px,
#: before the legend calls it out. NOT A GATE - nothing is refused on it, no fit
#: changes because of it - only the point at which two independent answers differ
#: enough that somebody should look at the picture.
#:
#: The two populations are almost completely separated, which is what makes the
#: number worth printing at all. Over the 223 sidelines where both exist they
#: agree to a median 2.3 px and a p90 of 5.0; the disagreements are 20, 30, 52,
#: 55, 80, 153 and 175 px. Nothing in the corpus lands between 15 and 20, so this
#: sits in open space rather than being fitted to either side of it.
CHORD_DISAGREE = 15.0


def _draw_chord(vis, tr, name):
    """The chord to the far service line's end, and what was read along it.

    Drawn on every plate, including the great majority where some other source
    won - see `walk.detect`. Three things, and they fail in three different ways:

    the CHORD itself, corner to far service end, as a thin straight line. It is
    the input, not the output: if it is not lying roughly along the court's edge
    then the far service walk overran, and nothing read along it can be right.

    the POINTS, as diamonds - heavy where they were offered to the fit, which is
    the near half, and light above the net, where the floor is seen almost
    edge-on and this module refuses to believe any sideline detector, this one
    included. Where the light ones carry on straight past the heavy ones the cut
    is costing nothing; where they veer off, it is the reason the fit is intact.

    their DISAGREEMENT with whatever was actually used, in pixels, returned for
    the sideline's own legend entry. That number is the reason to draw any of
    this: a few pixels means two methods sharing no pixel-level machinery found
    the same edge, and tens of pixels means one of them is on the wrong feature
    and the diamonds will show which.

    Returns (text, worth_a_look) for the caller's legend.
    """
    ends = tr.get("chord_ends")
    corners = np.asarray(tr.get("corners", np.zeros((0, 2))), float).reshape(-1, 2)
    i = 0 if name == "left_sideline" else 1
    if ends is not None and len(corners) == 2:
        cv2.line(vis, tuple(np.round(corners[i]).astype(int)),
                 tuple(np.round(np.asarray(ends[i], float)).astype(int)),
                 CHORD_COL, 1, cv2.LINE_AA)
    y_min = tr.get("chord_y_min")
    used = (tr.get("side_how") or {}).get(name, "")
    bits, look = [], False
    for key, store, col, mark in (("chord", "chord_sides", CHORD_COL,
                                   cv2.MARKER_DIAMOND),
                                  ("chord_lines", "chord_line_sides",
                                   CHORD_LINE_COL, cv2.MARKER_TILTED_CROSS)):
        pts = np.asarray((tr.get(store) or {}).get(name, np.zeros((0, 2))),
                         float).reshape(-1, 2)
        if not len(pts):
            bits.append("%s read nothing" % key)
            continue
        near = pts[:, 1] >= y_min if y_min is not None else np.ones(len(pts), bool)
        for p, is_near in zip(pts, near):
            cv2.drawMarker(vis, tuple(np.round(p).astype(int)), col, mark, 5,
                           2 if is_near else 1, cv2.LINE_AA)
        n = int(near.sum())
        gap = _chord_gap(tr, name, store)
        if key in used or (key == "chord" and "the chord" in used):
            bits.append("%s %d pts, IS this line" % (key, n))
        elif gap is None:
            bits.append("%s %d pts, not comparable" % (key, n))
        else:
            bits.append("%s %d pts, %.1f px away" % (key, n, gap))
            look = look or gap > CHORD_DISAGREE
    if ends is None:
        return "no chord", False
    return "; ".join(bits), look


def _chord_gap(tr, name, store="chord_sides"):
    """Median px between one chord reader's answer and the sideline used."""
    a = np.asarray((tr.get(store) or {}).get(name, np.zeros((0, 2))),
                   float).reshape(-1, 2)
    b = np.asarray((tr.get("tracked") or {}).get(name, np.zeros((0, 2))),
                   float).reshape(-1, 2)
    y_min = tr.get("chord_y_min")
    if y_min is not None:
        a = a[a[:, 1] >= y_min]
    if len(a) < 4 or len(b) < 4:
        return None
    lo, hi = max(a[:, 1].min(), b[:, 1].min()), min(a[:, 1].max(), b[:, 1].max())
    a = a[(a[:, 1] >= lo) & (a[:, 1] <= hi)]
    if len(a) < 4 or len(np.unique(b[:, 1])) < 3:
        return None
    try:
        cf = np.polyfit(b[:, 1], b[:, 0], 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return float(np.median(np.abs(a[:, 0] - np.polyval(cf, a[:, 1]))))


def draw_walk(bgr, tr, path, title="", note=""):
    """Draw ONLY what walking produced, on the plate it was walked on.

    Nothing fitted, nothing solved, nothing projected back from a camera model:
    a point is here because a ridge tracker or a column scan put it here. When a
    calibration comes out wrong the question is always which of the two happened
    - the walk found the wrong thing, or the walk was right and the solve went
    astray - and no overlay of the fitted court can answer it, because the fitted
    court is what is in doubt.

    Points RANSAC would later discard are drawn as hollow rings. A line whose
    raw points are one clean arc and whose rings are a scatter was walked
    correctly; a line that is half rings was not, whatever its residual says.
    """
    vis = bgr.copy()
    h, w = vis.shape[:2]
    kept = tr.get("kept", {})

    def dim(c, k=0.45):
        return tuple(int(v * k) for v in c)

    legend = []
    for name, col, swap in WALK_LINES:
        pts = np.asarray(tr.get(name, np.zeros((0, 2))), float).reshape(-1, 2)
        if not len(pts):
            legend.append(("%s: nothing walked" % name.replace("_", " "), dim(col, 0.8)))
            continue
        # the path in order, so a walk that jumped shows as a spike
        cv2.polylines(vis, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
                      False, dim(col), 1, cv2.LINE_AA)
        k = kept.get(name)
        keep = set() if k is None else {(round(float(p[0]), 2), round(float(p[1]), 2))
                                        for p in k}
        n_in = 0
        for p in pts:
            c = tuple(np.round(p).astype(int))
            if (round(float(p[0]), 2), round(float(p[1]), 2)) in keep:
                cv2.circle(vis, c, 2, col, -1, cv2.LINE_AA)
                n_in += 1
            else:                                # dropped, or never offered at all
                cv2.circle(vis, c, 3, col, 1, cv2.LINE_AA)
        # How straight the surviving set actually is. Counting inliers is not
        # enough: `_trim_to_curve` returns its INPUT UNCHANGED when RANSAC cannot
        # reach 12 inliers, so a set it could make no sense of comes back looking
        # exactly like a set it fully endorsed. Measured here, a far service line
        # scattered over the net tape reads "19 walked, 19 kept" and 51 px rms.
        note_rms, bad = "", False
        if k is not None and len(k) >= 4:
            a = k[:, 1] if swap else k[:, 0]
            b = k[:, 0] if swap else k[:, 1]
            try:
                res = b - np.polyval(np.polyfit(a, b, 2), a)
                rms = float(np.sqrt((res ** 2).mean()))
                bad = rms > 3.0
                note_rms = ", curve rms %.1f px%s" % (rms, "  <-- NOT A LINE" if bad else "")
            except (np.linalg.LinAlgError, ValueError):
                pass
        # How far this line was slid onto the middle of its own paint, and how
        # wide that paint measured. Worth a line of its own because it is a
        # correction applied to something already drawn as an answer: without it
        # the picture shows samples 2 px off the stripe they are marking and
        # gives no hint that anything moved them there.
        #
        # The MEDIAN move and its SCATTER, because the follower goes wrong in two
        # ways and the median alone hides one of them: on a low-contrast court
        # the walk has no bias at all and wanders a pixel either side instead, so
        # a correction reading "moved 0.1 px" can be the one doing the most work.
        # `fell_back` is how many samples had no credible pair of edges and took
        # the rest of the line's fitted correction instead - a large number there
        # means the paint is not being read, whatever the move says.
        mid = tr.get("centred", {}).get(name)
        note_mid = ""
        if mid and mid.get("moved"):
            note_mid = (", centred on %.1f px paint (moved %+.1f +-%.1f px, up to "
                        "%.1f; %d fell back)"
                        % (mid["width"], mid["shift"], mid.get("scatter", 0.0),
                           mid["shift_max"], mid.get("fell_back", 0)))
        elif mid:
            note_mid = ", NOT centred: " + mid.get("why", "")
        legend.append(("%s: %d walked, %d kept%s%s"
                       % (name.replace("_", " "), len(pts), n_in, note_rms, note_mid),
                       (60, 160, 255) if bad else col))

    # --- the sidelines: where they were looked for, and what was found -------
    # Two independent candidate sets, drawn differently on purpose. A failure is
    # almost always one of: the corner is wrong, the track died in the first few
    # points, or the row scans found a feature well inside the real boundary -
    # and those three look nothing alike here.
    scan = tr.get("scanned_sides", {})
    trk = tr.get("tracked", {})
    cann = tr.get("canny_cands", {})
    how = tr.get("side_how", {})
    for name, col in (("left_sideline", (80, 80, 255)), ("right_sideline", (80, 255, 80))):
        s = np.asarray(scan.get(name, np.zeros((0, 2))), float).reshape(-1, 2)
        for p in s:                              # old row scans: hollow squares
            c = np.round(p).astype(int)
            cv2.rectangle(vis, (c[0] - 4, c[1] - 4), (c[0] + 4, c[1] + 4), col, 1, cv2.LINE_AA)
        e = np.asarray(cann.get(name, np.zeros((0, 2))), float).reshape(-1, 2)
        for p in e:                              # every candidate Canny picked
            cv2.circle(vis, tuple(np.round(p).astype(int)), 1, dim(col, 0.75), -1, cv2.LINE_AA)
        t = np.asarray(trk.get(name, np.zeros((0, 2))), float).reshape(-1, 2)
        if len(t) > 1:
            cv2.polylines(vis, [np.round(t).astype(np.int32).reshape(-1, 1, 2)],
                          False, dim(col, 0.6), 1, cv2.LINE_AA)
        for p in t:                              # the ones kept, re-measured
            cv2.circle(vis, tuple(np.round(p).astype(int)), 3, col, -1, cv2.LINE_AA)
        # Which SOURCE won, not just how many points it produced: the sources
        # are tried in order and each has to prove it reached the corner, so
        # "63 points, boundary track from the corner" and a refusal reason are
        # the two things worth reading here. Counting the Canny candidates as
        # though they were the points is only true of the first source.
        # ...and, on the same line, the chord's independent answer for this same
        # sideline and how far it sits from the one that won - see `_draw_chord`.
        # On its own line it pushed the legend to thirteen entries; the number
        # only means anything next to the source it disagrees with anyway.
        chord, odd = _draw_chord(vis, tr, name)
        ok = len(t) >= 12
        legend.append(("%s: %d points, %s  (%d edge candidates; %s)"
                       % (name.replace("_", " "), len(t),
                          how.get(name, "row scans"), len(e), chord),
                       col if ok and not odd else (60, 160, 255)))

    # --- the far edge of the centre line: the far service line's depth -------
    far_t = tr.get("far_t")
    rej = tr.get("far_t_rejected")
    if far_t is not None:
        c = tuple(np.round(np.asarray(far_t, float)).astype(int))
        cv2.drawMarker(vis, c, (0, 0, 0), cv2.MARKER_DIAMOND, 30, 5, cv2.LINE_AA)
        cv2.drawMarker(vis, c, (255, 160, 0), cv2.MARKER_DIAMOND, 26, 2, cv2.LINE_AA)
        cv2.line(vis, (0, c[1]), (vis.shape[1], c[1]), (255, 160, 0), 1, cv2.LINE_AA)
        anch = any(a["name"] == "far_service_t" for a in tr.get("anchors", []))
        cv2.putText(vis, "far T" + ("  Y=3.40 ANCHOR" if anch else "  (not used)"),
                    (c[0] + 18, c[1] - 10), 0, 0.5, (255, 160, 0), 1, cv2.LINE_AA)
        if tr.get("centre_why"):
            legend.append(("centre line stopped: " + tr["centre_why"], (255, 160, 0)))
        legend.append(("centre line ends at (%.0f, %.0f) -> far service %s"
                       % (far_t[0], far_t[1],
                          tr.get("far_anchored")
                          or "NOT anchored (too few columns agreed)"),
                       (255, 160, 0) if tr.get("far_anchored") else (60, 160, 255)))
    elif rej:
        cv2.line(vis, (0, int(rej[1])), (vis.shape[1], int(rej[1])), (60, 160, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (0, int(rej[2])), (vis.shape[1], int(rej[2])), (255, 160, 0), 1, cv2.LINE_AA)
        cv2.line(vis, (0, int(rej[3])), (vis.shape[1], int(rej[3])), (255, 160, 0), 1, cv2.LINE_AA)
        legend.append(("centre line ends at y=%.0f but the far half is y=%.0f..%.0f"
                       " - anchor REFUSED, fell back to brightest ridge"
                       % (rej[1], rej[2], rej[3]), (60, 160, 255)))

    T = tr.get("T")
    if T is not None:
        cv2.drawMarker(vis, tuple(np.round(T).astype(int)), (0, 165, 255),
                       cv2.MARKER_TILTED_CROSS, 44, 2, cv2.LINE_AA)
        legend.append(("anchor T at (%.0f, %.0f)" % (T[0], T[1]), (0, 165, 255)))
    else:
        legend.append(("anchor T: NOT FOUND - the walk never started", (0, 165, 255)))

    # The two corners the sidelines are tracked FROM, and that the fit is now
    # CONSTRAINED by. If these are wrong nothing downstream can be right - and
    # since the solver is pulled toward them, a wrong one is no longer merely
    # unhelpful. Drawn last and largest for that reason.
    #
    # The ring is a FINDING AID and the X is the measurement: at plate scale two
    # bare X's in a 4k frame take a while to spot, and the whole point of the
    # picture is being able to go straight to them. Radius chosen to clear the
    # arms so it never touches the crossing.
    corners = np.asarray(tr.get("corners", np.zeros((0, 2))), float).reshape(-1, 2)
    used = {tuple(np.round(np.asarray(P["px"], float), 3)) for P in tr.get("anchors", [])}
    for i, p in enumerate(corners):
        c = tuple(np.round(p).astype(int))
        on = tuple(np.round(p, 3)) in used
        mark_x(vis, p, (0, 255, 255))
        cv2.circle(vis, c, 20, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "X=%s%s" % ("-0.05" if i == 0 else "10.05", "" if on else " (not used)"),
                    (c[0] - 34, c[1] + 36), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if len(corners) == 2:
        legend.append(("near service ends X at (%.1f, %.1f) and (%.1f, %.1f) - %d used to "
                       "ANCHOR the fit"
                       % (corners[0][0], corners[0][1], corners[1][0], corners[1][1],
                          len(tr.get("anchors", []))), (0, 255, 255)))

    # THE FAR SERVICE LINE'S OWN TWO ENDS, marked the same way because they are
    # the same kind of thing - the place that line's paint stops - and scored the
    # same way by `compare_ends.py`. In the chord's colour and not the corners',
    # because they are not solver anchors: what they do is fix the far end of the
    # strip both chord readers draw along, so an end in the wrong place tilts a
    # reader rather than pulling the fit. No ring: the chord line already drawn
    # from each near corner leads the eye to them.
    far_ends = tr.get("chord_ends")
    if far_ends is not None:
        f = np.asarray(far_ends, float).reshape(-1, 2)
        for p in f:
            mark_x(vis, p, CHORD_COL)
        if len(f) == 2:
            legend.append(("far service ends X at (%.1f, %.1f) and (%.1f, %.1f) - where "
                           "both chord readers draw from"
                           % (f[0][0], f[0][1], f[1][0], f[1][1]), CHORD_COL))
    # the box find_T searched: an anchor on the wrong junction is only readable
    # against the region it was allowed to look in.
    cv2.rectangle(vis, (int(T_BOX[0] * w), int(T_BOX[1] * h)),
                  (int(T_BOX[2] * w), int(T_BOX[3] * h)), (0, 165, 255), 1)

    _label(vis, "%s   walked lines only (no fit)" % title, (14, 26),
           (245, 245, 245), 0.62)
    y = 50
    if note:
        _label(vis, note, (14, y), (60, 160, 255), 0.55)
        y += 23
    for text, col in legend:
        _label(vis, text, (14, y), col)
        y += 23
    cv2.imwrite(path, vis)
    return path
