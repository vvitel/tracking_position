"""Find the court by walking it, starting from one anchor: the near service T.

Nothing here segments the court as a whole. Everything is anchored at a single
junction - where the near service line meets the centre line - and reached from
it by following paint or by 1-D scans outward. Lines are then identified by
TOPOLOGY (what you meet, in what order, walking away from the anchor) rather
than by photometry (what looks like a line).

That distinction is the point. `CourtLines/` identifies the far service line by
segmenting the far half and ridge-searching inside it, and the far half is a
40-60px sliver at both venues measured, so the blind pass lands on the far
baseline every time. Walking up a column instead gives:

    court ON  ... near half ...          top of this run IS the net's floor line
    court OFF ... the net ...
    court ON  ... far half ...           top of this run IS the far baseline
    court OFF ... beyond the court ...

and the far service line is the one bright horizontal ridge inside the second
ON run. A 47px window with exactly one candidate in it cannot pick the wrong
line, however thin the sliver is.

Outputs the same six line point sets `CourtLines/LineDetect.detect` does, so
`CourtLines/LineSolve` consumes them unchanged - the solver is held fixed on
purpose, so any difference in the result comes from the front end alone.
"""
import math
import os
import warnings

import cv2
import numpy as np

from . import court as CM
from . import image as IM
from .image import centre_on_paint, trim_to_curve, white_edges

WORLD_LINES = CM.WORLD_LINES     # same court, same six lines

#: How far OUTSIDE the nominal sideline the near service line's paint stops, in
#: metres. It does stop outside, and this is not a detector bias: backprojected
#: through the two CLICK-SEEDED reference models, which never saw a walked
#: sideline, the painted ends land at X = -0.070 / +10.082 at GALAIS and -0.038 /
#: +10.073 at ROTONDE, and the 41 accepted batch fits agree at -0.032 / +10.065.
#: Three independent sources, two venues, same sign at both ends. A padel
#: sideline is the surface/glass boundary rather than paint, and the paint is
#: laid up to the glass, so a few centimetres of overshoot is what the court is
#: actually like.
#:
#: It matters because anchoring the paint end at the nominal sideline would feed
#: the solver a point it knows to be 5 cm wrong, in the one direction the anchor
#: exists to constrain, and every court would come out ~10 cm too wide. Measured
#: over the 41: anchoring at the nominal position leaves a median 8.5 px error,
#: anchoring here leaves 3.0 px. The left/right asymmetry in those numbers is
#: real but smaller than the venue-to-venue spread (2 cm against a 2.3-4.5 cm
#: mad), so it is deliberately NOT modelled - one symmetric constant measures as
#: well as two and claims less.
ANCHOR_OUT = 0.05

#: The same distance for a corner `corner_on_curve` has placed, which is a
#: DIFFERENT POINT and therefore a different constant: the walk's endpoint is
#: where the paint stopped being bright enough to follow, and the refined one is
#: the crossing edge that ends it - the glass base, which IS the sideline.
#:
#: Measured the same way as `ANCHOR_OUT`, over the 65 labelled plates, as the
#: world X the corner backprojects to through each plate's own truth camera:
#: -0.009 (mad 0.009) at the left end and +10.004 (mad 0.012) at the right. Both
#: within a centimetre of the nominal sideline, from opposite ends of the court
#: and seventeen venues, so this is 0 rather than a small fitted number - and one
#: symmetric constant, for `ANCHOR_OUT`'s reason: the left/right difference is
#: 1.3 cm against a 0.9-1.2 cm mad and modelling it would claim more than the
#: measurement supports.
#:
#: That the two constants differ by 5 cm is not a discrepancy between them; it is
#: the 5 cm of paint laid past the sideline that `ANCHOR_OUT` is named for, now
#: measured from the other side.
CORNER_ANCHOR_OUT = 0.0

#: An endpoint this close to the frame edge is where the IMAGE stopped, not
#: where the paint did, and it is not a corner. Never triggers on this corpus
#: (the nearest is 67 px) but a wider mount would put it in play, and an anchor
#: that silently means something else is the failure this whole module is
#: organised to avoid.
ANCHOR_MARGIN = 14.0

#: How far SHORT of the far service T the centre line's walk stops, in metres.
#: The end-of-line rule fires when there is no paint 8px ahead, so it stops
#: about that far back, and at the far end of the court 8px is a large distance
#: on the ground - which is exactly why this end is worth anchoring and also why
#: the offset cannot be ignored.
#:
#: Measured three ways, as with `ANCHOR_OUT`: the 47 accepted plates put it at
#: world Y 3.380 (mad 0.143), and the two CLICK-SEEDED models, which never saw a
#: walked centre line, put it at 3.573 (galais) and 3.361 (rotonde). Anchored at
#: 3.40 the residual over the corpus is median 1.40 px, p90 2.20, max 2.86 -
#: eight times tighter than the near-service corners, because this walk stops
#: against a junction rather than against the open end of a painted stripe.
#:
#: Correcting it in the IMAGE instead - pushing the endpoint 4px further along
#: the walk and anchoring at the nominal (5.00, 3.05) - was measured and is
#: exactly as good (mad 0.137 against 0.143), so it buys nothing; the detected
#: point is left where it was actually detected.
FAR_T_SHORT = 0.35

#: The far T is believed harder than the near corners (3.0 px) because it is
#: measured better: p68 1.64 px over the corpus against their 4.4, on a feature
#: whose world X repeats to a 0.015 m mad. It needs the authority - at this
#: distance the whole far half of the court hangs off it.
FAR_T_SIGMA = 2.0


def anchors(ends, far_t, shape, outs=None):
    """The walk's own endpoints as world point correspondences.

    Returned for `LineSolve.solve(points=...)`. These are the ONLY points in a
    pipeline that is otherwise entirely line-based, and they earn it by being
    the one thing lines cannot express: a line correspondence measures
    perpendicular distance, so nothing in the fit says where ALONG a line
    anything sits. For the near service line that is the court's X scale, left
    entirely to the two sidelines; for the far service line it is the depth of
    the whole far half, which the refinement rounds are otherwise free to slide
    because a re-traced line confirms wherever the model already put it.

    `far_t` is the centre line's far end, or None when that walk stopped for a
    reason other than reaching the end of the paint - in which case its last
    point is wherever it happened to die, and means nothing.

    `outs` is how far outside the sideline each near corner sits, left and
    right, and it is a parameter because the answer depends on HOW THE CORNER
    WAS FOUND rather than on the court: a walked endpoint is the end of the
    paint and a refined one is the sideline the paint runs past. See
    `ANCHOR_OUT` and `CORNER_ANCHOR_OUT`. Defaults to the walked pair.
    """
    h, w = shape[:2]
    lo, hi = outs if outs is not None else (ANCHOR_OUT, ANCHOR_OUT)
    out = []
    for p, world, name in (
            (ends[0], (-lo, CM.NEAR_SERVICE_Y), "near_service_end_l"),
            (ends[1], (CM.WIDTH + hi, CM.NEAR_SERVICE_Y), "near_service_end_r"),
            (far_t, (CM.CENTRE_X, CM.FAR_SERVICE_Y + FAR_T_SHORT), "far_service_t")):
        if p is None:
            continue
        p = np.asarray(p, float)
        if not (ANCHOR_MARGIN < p[0] < w - ANCHOR_MARGIN
                and ANCHOR_MARGIN < p[1] < h - ANCHOR_MARGIN):
            continue
        out.append({"name": name, "world": world, "px": p,
                    "sigma": FAR_T_SIGMA if name == "far_service_t" else None})
    return out


# ------------------------------------------------------------ 1. fields ----
#: The window a horizontal line is looked for in, and how many votes it takes.
#:
#: 35 votes out of a 50px window asks for a line that crosses the window nearly
#: unbroken, and the far service line does not: measured at the far T of
#: `08-06-59`, its two edges contribute 32 and 30 collinear pixels - the centre
#: line crossing it and one small break cost it the rest - so a 35-vote gate
#: misses it by two pixels' worth of votes.
#:
#: Swept with `image.WHITE_OVER` over the 90 plates with a walk-pinned far
#: service line, as "is the true line among the candidates an unstoppable walk
#: collects":
#:
#:     gate/votes   global/35   6/30   8/25   12/22   16/25   22/35
#:     finds it       85/90    89/90  88/90  89/90   82/90   50/90
#:     candidates         5        9     11      10       8       4
#:
#: 12/22 is taken over 6/30 because asking for more contrast and fewer votes is
#: the more robust half of the trade: a faint line still votes, while noise has
#: to clear 12 L units before it can vote at all.
#:
#: TWO VOTE COUNTS, because the same primitive is read for two different
#: purposes and they want opposite errors. FINDING a line the chooser will judge
#: later wants recall - a spurious candidate is discarded downstream. LATCHING
#: the walk's leash on a crossing wants precision - a spurious crossing sets
#: `crossed` in the near half, where the flanks are already floor, so the leash
#: tightens with the whole court still ahead and the walk ends at the next
#: two-step gap. Measured: running the latch at 22 votes moved 13 far Ts down by
#: 32 to 124 px and cost 11 accepted plates (92 -> 81). At 35 the latch sees the
#: net's tape, which is the widest, brightest horizontal line in the frame, and
#: little else.
HZ_W, HZ_H, HZ_VOTES, HZ_DEG = 50, 20, 35, 12.0
HZ_VOTES_FIND = 22

#: Where a flank has to sit, relative to the floor beside the anchor, for the
#: walk to call itself back on the court. Crossing a horizontal line is itself a
#: gap - the line has two edges, a shadow under it and, at the top of the net, a
#: tape several pixels thick - so tightening the leash the instant one is
#: crossed ends the walk ON the crossing: measured that way the far T lands
#: 32-44 px short on 22 plates, every one stopping "at a 2-step gap past the
#: last horizontal line" while still standing on it.
#:
#: A fixed number of clean samples was tried instead and does not separate: 8
#: fixes 19 of those 22 and 24 fixes all of them, but 24 is longer than the
#: whole far half on the plate this exists to fix, so its far service line falls
#: inside the grace and is never reached. What actually distinguishes "off the
#: tape" from "still on the net" is not distance, it is whether there is floor
#: beside the line again - the same measurement the column scans use to tell the
#: net's mesh from the court.
HZ_SMOOTH = 1.0


def horizontals(edges, cx, cy, w=HZ_W, h=HZ_H, votes=HZ_VOTES, max_deg=HZ_DEG):
    """y of every near-horizontal line through a window on the edge map, upper first."""
    x0, y0 = int(cx - w // 2), int(cy - h // 2)
    if x0 < 0 or y0 < 0 or x0 + w > edges.shape[1] or y0 + h > edges.shape[0]:
        return []
    sub = edges[y0:y0 + h, x0:x0 + w]
    if int(sub.sum()) // 255 < votes:            # cannot reach the vote count
        return []
    lines = cv2.HoughLines(sub, 1, np.pi / 180, votes)
    if lines is None:
        return []
    out = []
    for rho, th in lines[:, 0, :]:
        if abs(np.degrees(th) - 90.0) > max_deg:
            continue
        a, b = np.cos(th), np.sin(th)
        if abs(b) > 1e-6:
            out.append(float(y0 + (rho - a * (w / 2.0)) / b))
    out.sort()
    merged = []
    for y in out:                                # a painted stripe has two edges
        if not merged or y - merged[-1] > 4.0:
            merged.append(y)
    return merged


#: How the far service line is told apart from everything else the walk crosses.
#:
#: GAP ABOVE IT, in px. The centre line ENDS at the far service T - there is no
#: paint beyond it - so the true line is the one with nothing above it. Measured
#: over the 90 plates with a walk-pinned far service line, the gap opens a median
#: 4.8 px above the true line and 35.7 px above the decoys.
#:
#: RMS AND REACH. Cleanliness alone picks the NET'S TOP TAPE every time - it is
#: the longest, straightest, brightest horizontal line in the frame (rms 0.7-1.2
#: against the true line's 0.4 median but 5-9 on a distant far half). Ordering
#: alone (the first line crossed with a gap above it) picks a decoy below the
#: true line on 5 of 90. The three together:
#:
#:     rule                                         right  wrong  abstain
#:     cleanest fit alone                              63     27        0
#:     upper-most alone                                17     68        0
#:     first crossed with a gap <= 12px                81      5        4
#:     gap <= 12px, rms <= 1.5, reach >= 100, cleanest 78      0       12
#:
#: An abstention costs nothing - it leaves the walk's own answer in place - so
#: the rule to want is the one that is never wrong, and its worst error on the
#: 78 it does answer is 2 px.
FIND_GAP, FIND_RMS, FIND_REACH, FIND_MIN = 12.0, 1.5, 100.0, 12
#: DID THE WALK STAY ON THE LINE IT STARTED FROM? A quadratic through the
#: walked far service line has to pass within this many pixels of the far T it
#: was seeded at. At the far T the service line is the faintest thing in the
#: frame and the court's far boundary runs parallel a few pixels beyond it, so
#: the follower can change lines without ever losing a ridge - `10-05-24` walks
#: off onto the court's far edge and reports it as the service line.
#:
#: This one test, and deliberately not the two obvious companions:
#:
#:     plate                  rms  seed off  chooser off   walk is
#:     10-05-24             11.49      19.4            -   ON THE WRONG LINE
#:     13-30-43             15.62      31.4            -   ON THE WRONG LINE
#:     8F-0F 19-04-58        0.54       2.8         60.6   fine
#:     e135a85cef74          6.10       1.3        222.8   fine
#:     7cfd634d113f          7.22       0.5          1.0   fine
#:
#: The walk's own rms does not separate - 7.22 on a good plate against 11.49 on a
#: broken one - because a walk that overruns the sidelines onto the glass base
#: is still on its own line. And the CHOSEN line cannot referee it: it is wrong
#: by 60 and 222 px on two plates where the walk is right, and trusting it there
#: cost 3 accepted plates. The seed is the only party to this that is known good.
FIND_SEED = 8.0

#: How far, in degrees, the far service walk may leave the direction it set out
#: in before it is stopped (`walk`'s `max_tilt`). Only this walk gets it, and
#: only because its geometry is known in advance: it starts at the far service T
#: with the whole line within a few degrees of horizontal in the image, and the
#: features it can be lost to - the court's far edge, the glass base outside the
#: far corners - all leave that direction steeply. The near service line is
#: given the same treatment nowhere near as safely (it bows ~100 px end to end
#: and its walk is already stopped by paint), and the centre line runs the other
#: way through the net, where a steep local chord is normal.
#:
#: THE THRESHOLD IS NOT WHAT SEPARATES, and measuring is the only way to find
#: that out. A healthy far service walk runs 0.3-4.3 degrees off its seed on
#: average, but the per-sample chord is noisy for the reason in `walk`'s
#: docstring - `_peak_ridge` re-centres up to 2 px sideways, which is several
#: degrees over a 32 px chord - and the WORST single sample of a healthy walk
#: is a median of 16 degrees and reaches 43. A bare 20 degree gate would fire
#: on most of the corpus.
#:
#: `confirm` is what does the work. Over 232 far service walks: a walk that is
#: not turning never puts more than TWO consecutive samples over the gate (178
#: walks at zero, 28 at one, 4 at two), and every walk that reaches three is
#: turning - the escapes hold 22-59 degrees over 20+ consecutive samples. The
#: rule stops 22 walks and the run-length gap between the two populations is
#: exactly one sample wide, which is narrow, so it is the CONSECUTIVE count
#: that should be raised if this ever misfires, not the angle.
FAR_TILT = 20.0

#: How many consecutive missed steps the far service walk may coast over,
#: against `walk`'s default of 6. The default is what a walk needs when it may
#: have lost the line; this one no longer may - it is bounded by `FAR_TILT`,
#: searched over a window derived from that bound, and read on a colour field
#: whose contrast does not decay as that window opens.
#:
#: Measured rather than assumed to be safe: across 462 far service walks the
#: gaps actually coasted run 285 of one step, 107 of two, 96 of three, 42 of
#: four, 51 of five and 25 of six, so a leash of 3 does cut 103 of those walks
#: short. It costs nothing, because this line is over-determined long before it
#: runs out - 110 of 116 accepted at 2, 3, 4 and 6 alike, probe and rms and
#: anchor identical to three decimals, and the shortest far service line in the
#: corpus is 144 samples either way. What it does buy is a smaller worst case:
#: max probe 2.069 -> 2.043, and the search window can now open to 9.8 px
#: instead of 14.2 before the walk gives up.
#:
#: 3 rather than 2 because 2 is the one setting that is measurably worse (probe
#: p90 1.772 against 1.758) and cuts twice as many walks for it.
FAR_MAX_MISS = 3

#: The contrast a colour ridge has to stand above its own profile, for the
#: far service walk (`walk`'s `colour_ridge`, `image.peak_ridge_lab`). A
#: SEPARATE constant from the top-hat's 14 because the two fields are on
#: different scales and a shared number transfers by luck rather than by
#: meaning: at 14 the colour ridge loses `10-05-24`'s rightward walk entirely
#: (66 samples to 1), at 16 it loses both. Swept, 8/10/12 all walk every one of
#: the four hard plates end to end; 10 is the middle of that plateau.
FAR_COLOUR_CONTRAST = 10.0

#: How far the ridge may fall below its own running median before the NEAR
#: SERVICE walk calls the paint finished, against `walk`'s default of 0.45.
#:
#: That default is what a walk needs where a gap in the middle of a line is
#: normal - the centre line passes under the net's cord and its shadow - and
#: this line has no such feature on it: it is plain paint on plain floor from
#: the T to the glass. What it does have at both ends is the thing 0.45 cannot
#: refuse. The paint stops at the glass base, and beyond it is the sponsor
#: banner and the frame's shadow, which the ridge follower reads as a dimmer
#: line leaving at a slightly different angle. Measured along the left end of
#: `22-17-42`, where the paint stops at x=104: the ridge holds 152-160 over the
#: last 20 px of paint and then reads 94, 100, 86, 44, 39, 88, 52, 71, 65 - a
#: 0.45 gate needs three CONSECUTIVE samples under 70 and the banner never
#: supplies them, so the walk runs 31 px past the corner and the fit is handed
#: an anchor 0.35 m outside the court.
#:
#: THE GATE IS ALREADY RELATIVE and stays that way - it is a fraction of the
#: median of the walk's own last 15 accepted samples, so vignetting, venue
#: colour and exposure never enter, and a shadow that dims the paint dims the
#: median behind it too. Only the fraction changes here.
#:
#: Normalising it further, by the LIGHTNESS OF THE FLOOR BESIDE THE LINE, was
#: tried and is worse for a reason worth recording: the flanks are only the
#: court while the walk is on the court. Past the left end of `22-17-42` they
#: are the banner (L 144 against the floor's 100) and the ratio falls twice as
#: fast, but past the right end they are the frame's shadow (L 62), and
#: dividing by that lifts the two samples beyond the paint from 0.50 and 0.60
#: back to 1.21 and 1.41 - the rule stops firing exactly where it is needed. A
#: lagged flank median fixes that and then measures identically to this, since
#: it tracks illumination no faster than the ridge median it would replace.
#:
#: Swept over the 116 plates, as the world X the two endpoints backproject to
#: through each plate's own camera - the same measurement `ANCHOR_OUT` is
#: derived from, where the truth is 0.05 m OUTSIDE the sideline:
#:
#:     drop        accepted   past the sideline (m)      short of it (m)
#:                            p50    p90    max          p90    max
#:     0.45 (was)      108   0.044  0.166  0.349        0.008  0.435
#:     0.55            108   0.037  0.142  0.331        0.020  0.435
#:     0.65            108   0.035  0.133  0.269        0.020  0.435
#:     0.70            108   0.033  0.130  0.269        0.022  0.428
#:     0.75            108   0.029  0.106  0.269        0.027  0.428
#:     0.80            107   0.026  0.099  0.614        0.035  1.891
#:
#: A plateau from 0.55 to 0.75 that costs nothing and a cliff at 0.80, where one
#: plate's walk is cut 1.9 m short and its anchor error goes to 228 px. 0.70 is
#: the middle of the plateau, which is where a threshold belongs when the
#: populations either side of it are this far apart. Probe and line rms do not
#: move at any setting (p50 0.626/0.591 against 0.625/0.592) - this changes
#: where the line STOPS, not where it is.
#:
#: What it does NOT fix, and no photometric gate can: the walks that end at
#: "ridge lost" or at the frame edge with the paint still fading. Eight of the
#: ten worst endpoints in the corpus stop that way, identically at 0.45 and
#: 0.70, out where the fisheye turns 5 px into 0.2 m.
NEAR_SVC_DROP = 0.70

#: How far off its own line the LAST sample of the near service walk may sit
#: before it is thrown away, in pixels, and how many samples the line is judged
#: from. `walk` finds the line; this decides whether its final sample is a point
#: ON it - which is a different question, and only the corners have to answer it.
#:
#: THE LAST SAMPLE IS SPECIAL AND THE MEASUREMENT SAYS SO. Fitting a quadratic
#: to the 30 walked samples 4..34 in from each end and asking how far the ones
#: outside that window sit off it, over the 232 arms in the corpus:
#:
#:     sample, counting in from the end     p50    p90    p99     max   n>3px
#:     the endpoint itself                 0.75   3.63  11.58   14.62      33
#:     one in                              0.34   1.72   4.76    8.12      10
#:     two in                              0.29   1.11   3.77   10.94       7
#:     three in                            0.22   0.61   2.33    2.96       0
#:
#: against a line whose own scatter is 0.23 px rms (p90 0.38). Three samples in,
#: the walk is on the paint to a fifth of a pixel and NOTHING in the corpus is
#: more than 2.96 px off; the endpoint is 14.6 px off on the worst plate, which
#: is sixty sigma. So 3.0 is not a tuned threshold, it is the top of the
#: legitimate population - and the `6 * rms` term in `peel_ends` takes over
#: wherever a plate's own walk is noisier than this corpus's.
#:
#: WHY IT IS THIS ONE SAMPLE. The walk's curve rule needs `confirm` consecutive
#: offences and the walk ends before it can collect them, so the sample that
#: stopped the walk is always kept. On `07-37-14` the right arm's last sample
#: drops 8 px in one 5 px step onto the glass base; RANSAC duly rejects it, and
#: the corner is read before RANSAC ever runs, so the reject is what anchors the
#: sidelines.
#:
#: TAKING THE OUTERMOST RANSAC INLIER INSTEAD - the obvious fix, and the one
#: `ends` already argues against - does not work, for a reason that is about the
#: fit rather than about the endpoint: `trim_to_curve` fits ONE quadratic across
#: the whole 1780 px span, a fisheye bows that span by more than a quadratic can
#: follow, and what it clips is therefore the ends of a curve that is
#: legitimately not quadratic. Measured there: corpus 110 accepted -> 68. This
#: test is local instead - `PEEL_FIT` samples immediately inside the point being
#: judged, interpolating rather than extrapolating - so it can be strict about
#: the endpoint without being wrong about the bow.
#:
#: Swept over the 116 plates, with the corner's own offence measured as above:
#:
#:     gate     accepted   corner off its line (px)        past the sideline (m)
#:                         p90    p99    max    n>3px      p90      worst anchor
#:     off          108   3.63  11.58  14.62       33     0.130         12.69 px
#:     2.0          108   1.89   3.21   3.53        4     0.096         11.26
#:     2.5          108   2.14   3.24   3.53        5     0.096         11.03
#:     3.0          108   2.19   3.26   4.20        6     0.093         11.03
#:     4.0          108   2.36   3.99   4.20       14     0.106         11.90
#:     6.0          108   2.81   4.65   4.84       21     0.109         11.90
#:
#: Nothing is lost at any setting and the whole range 2.0-3.0 is a plateau. It
#: touches 31 of the 232 arms - 24 give up one sample and 7 give up two - so the
#: line it is trimming is 0.1% shorter and its corner is right.
PEEL, PEEL_FIT = 3.0, 120


def peel_ends(pts, swap, gate=None, fit=PEEL_FIT):
    """Drop samples off each end of a walked line that are not ON it.

    The walk's own curve rule cannot reach them, and the reason is structural
    rather than a matter of tuning: that rule needs `confirm` consecutive
    offences before it will act, and at the end of a line there is nothing left
    to confirm with. The walk stops, and the sample that stopped it is kept.
    Harmless for the LINE - one sample in four hundred, and `trim_to_curve`
    discards it before anything is fitted - and not harmless at all for the
    CORNER, which is that sample and nothing else.

    Run backwards, once the line is finished, the test costs nothing: fit the
    line WITHOUT the end under examination and ask whether that end belongs to
    it. Nothing is extrapolated - the fit is interpolated over `fit` samples
    immediately inside the point it judges - and the gate is the fit's own
    residual wherever that exceeds `PEEL`, so a plate whose walk is noisier is
    judged by its own scatter rather than by this corpus's.

    AFTER `centre_on_paint`, NOT BEFORE, and that is the whole reason this is
    here and not in `walk`. Where the paint ends, one of the two Canny edges the
    centring reads is the END of the stripe rather than its side, so the
    midpoint of the pair is not the middle of the line: measured on `18-09-21`,
    the last sample is 5.4 px off the raw walk's own curve and 10.4 px off after
    centring. A peel inside `walk` would see the 5.4 and hand on the 10.4.
    """
    pts = np.asarray(pts, float)
    gate = PEEL if gate is None else gate
    lo, hi = 0, len(pts)
    for front in (False, True):
        while hi - lo > max(12, fit // 4):
            w = pts[lo + 1:lo + 1 + fit] if front else pts[max(lo, hi - 1 - fit):hi - 1]
            a = w[:, 1] if swap else w[:, 0]
            b = w[:, 0] if swap else w[:, 1]
            if len(np.unique(a)) < 3:
                break
            cf = np.polyfit(a, b, 2)
            g = max(gate, 6.0 * float(np.sqrt(((b - np.polyval(cf, a)) ** 2).mean())))
            q = pts[lo] if front else pts[hi - 1]
            qa, qb = (q[1], q[0]) if swap else (q[0], q[1])
            if abs(qb - np.polyval(cf, qa)) <= g:
                break
            if front:
                lo += 1
            else:
                hi -= 1
    return pts[lo:hi], lo, len(pts) - hi


# ------------------------------------ 3b. the corner, slid along its line ----
# `peel_ends` above answers "is this sample ON the line" and `centre_on_paint`
# answers "where across the paint is it". Neither answers the question the
# corner actually poses, which is WHERE ALONG THE LINE THE PAINT STOPS - and
# that is the one the walk is worst at, because the walk has to decide it going
# forwards, one 4 px step at a time, from the line's brightness alone.
#
# WHAT ENDS THE PAINT IS A LINE, NOT A DIMMING. The service line stops at the
# glass, and the glass base is a hard step running across it - the same feature
# the sideline readers are looking for, met end-on. So the corner is an
# INTERSECTION, and intersections are worth far more than stopping points: they
# are found by fitting two things that each have hundreds of pixels of evidence,
# rather than by noticing that one of them has gone quiet.
#
# CONSTRAINED TO THE LINE'S OWN CURVE, which is what makes it safe. Left free in
# two dimensions this is a small Hough transform at the corner, and a Hough in a
# 100 px box has no idea which of the parallel edges out there - the glass base,
# its own reflection, the frame's shadow, the fence base a metre further out - it
# has landed on. `13-09-39` is the recorded case: the first strong edge out from
# centre is the fence base and it returns a clean 16-point arc a metre outside
# the court. Held to the near service line's own quadratic, which is a 0.23 px
# rms fit over 400 samples, the search decides ONE scalar - how far along - and
# the across-line accuracy `peel_ends` bought cannot be given back.
#
# The reach is what keeps it off the wrong parallel edge: 30 px along the line at
# the near corner is under 0.2 m of court, and everything the box could confuse
# the glass base with is further out than that.

#: How far along its own line the corner may move, in pixels, and the pitch the
#: search runs at. The reach is the same 30 px as the candidate window the walk
#: would need to collect its own alternatives, and it is a claim about how wrong
#: the walk can be rather than a tuning knob: measured over the corpus the walk's
#: endpoint is a median 3 px and p90 12 px from where this lands, so 30 px is the
#: far tail of the population it has to be able to correct, and no more.
CORNER_REACH, CORNER_STEP = 30.0, 0.5

#: How much of the crossing line is integrated, in pixels each way from the
#: service line, and how far it may lean from the chord, in degrees.
#:
#: The lean is needed because the chord is a CHORD - it runs from this corner to
#: the far service line's own end, and the lens bows the sideline between them -
#: so the sideline's direction AT the corner is not the chord's direction. 20
#: degrees covers the bow with room to spare and is still nowhere near the
#: service line itself, which crosses at 60-90 degrees; a window that reached
#: that far could score the stripe's own long edges instead of what ends them.
CORNER_SPAN, CORNER_ANG, CORNER_ANG_STEP = 40.0, 20.0, 2.0

#: How many samples of its own line the local curve is fitted from - `PEEL_FIT`,
#: for `PEEL_FIT`'s reason: near enough the end to be a local fit, far enough in
#: not to be dragged by the sample under examination.
CORNER_FIT = PEEL_FIT

#: How much a position must beat the search window's own median score before it
#: counts as an edge at all. A ratio rather than a level, because the score is a
#: gradient and a gradient's units are the venue's contrast: what is being asked
#: is not "is there an edge here" but "is there an edge HERE and not everywhere
#: along this line", which is the only form of the question that means anything
#: on a plate where the whole corner sits in glare.
#:
#: THE FIRST QUALIFYING EDGE ON THE WAY OUT WINS, NOT THE STRONGEST, and that is
#: the whole difference between this working and not. Taking the global maximum
#: over the window was tried first and improves 96 of the 130 corners while
#: wrecking 26: the wrecked ones are unmistakable in the trace because they all
#: slide 21-30 px, out to the end of the reach, where the glass frame's outer
#: edge and the fence base sit - features that are STRONGER than the glass base
#: because they are further from the paint's own glare, and that a scoring rule
#: has no way to refuse. Outwardness is the thing that identifies the right edge,
#: exactly as it is for `chord_lines`, whose rule this is: the sideline is the
#: first qualifying edge out, and everything past it belongs to something else.
#:
#: Swept over the 65 labelled plates with `compare_ends.py`, as the world X each
#: corner backprojects to through that plate's own TRUTH camera - a model this
#: detector never saw - with the scatter measured about the population's own
#: median, because the median is a constant to re-derive and not an error:
#:
#:     rule                      left arm                  right arm
#:                           mad    p90|e|  off>0.1    mad    p90|e|  off>0.1
#:     the walk's endpoint  0.012   0.091      4      0.021   0.131     17
#:     strongest in window  0.019   0.170      8      0.015   0.084      4
#:     first out, gate 1.8  0.010   0.076      2      0.011   0.075      1
#:     first out, gate 2.0  0.010   0.069      1      0.011   0.070      1
#:     first out, gate 2.2  0.009   0.064      1      0.012   0.076      0
#:     first out, gate 2.4  0.009   0.069      1      0.012   0.076      0
#:     first out, gate 2.6  0.014   0.077      1      0.013   0.080      1
#:     first out, gate 2.8  0.014   0.077      1      0.014   0.088      5
#:     first out, gate 4.0  0.022   0.088      5      0.023   0.094      6
#:
#: 2.0 to 2.4 is a plateau and 2.2 is the middle of it, which is where a
#: threshold belongs when the populations either side are this far apart. Below
#: it the rule stops at scuffs INSIDE the paint - the failures at 1.6 all slide
#: backwards by 2-17 px at a ratio of 1.6-2.1, barely over their own gate - and
#: above it the real edge stops qualifying on the dimmer plates and the corner
#: falls back to the walk, taking the walk's tail with it.
#:
#: Both arms end up alike, and that is the result worth noting: the walk's two
#: arms differ by a factor of two in scatter and by 4 against 17 in the tail,
#: because one of them faces the sponsor banner and the other the frame's
#: shadow. An edge is an edge either way.
CORNER_PEAK = 2.2

#: Which line's ends are refined this way. Read at call time so a worker process
#: can be told without an import; `PADEL_CORNER_REFINE` overrides it with a comma
#: list, or `none` for the walk's own endpoints throughout.
#:
#: BOTH SERVICE LINES, because both end against the same two sidelines and this
#: reads them off the same feature from opposite ends. The far T is deliberately
#: NOT in this list and is not a candidate for it: it does not end against a
#: crossing edge at all, it ends against the far service line, which the detector
#: finds independently and by other means.
#:
#: Measured with `compare_ends.py` over the 65 labelled plates, as the world X
#: each endpoint backprojects to through that plate's own truth camera. The
#: columns that decide are the TAIL ones - a median is a constant to re-derive,
#: and nothing can be re-derived from a tail:
#:
#:     endpoint            rule        off>0.10  off>0.15   worst    p90    mad
#:     near_service_end_l  the walk        4/65      3/65   0.202  0.091  0.012
#:                         strongest       7/65      6/65   0.252  0.138  0.018
#:                         first out       1/65      0/65   0.138  0.066  0.009
#:     near_service_end_r  the walk       17/65      5/65   0.193  0.131  0.021
#:                         strongest       4/65      4/65   0.258  0.084  0.017
#:                         first out       0/65      0/65   0.096  0.078  0.012
#:     far_service_end_l   the walk       27/65     23/65   0.471  0.297  0.072
#:                         first out      13/65     11/65   0.471  0.221  0.030
#:     far_service_end_r   the walk       33/65     28/65   0.679  0.346  0.130
#:                         first out      17/65     15/65   0.679  0.267  0.040
#:
#: Per corner on the near line, 104 of 130 improve, 6 worsen and none of the six
#: by more than 0.041 m or across the 0.10 m line. The far ends improve less
#: completely and for a reason the trace states out loud: that line is 1-2 px of
#: paint at that distance and the crossing edge often does not stand out, so the
#: refiner DECLINES on 23-24 of 65 and hands back the walk's endpoint. Every
#: residual failure past 0.4 m out there is one of those declines - the far
#: walk's own overrun, untouched - which is the guard working, not failing.
#:
#: END TO END, against the walk's own endpoints, over the same 65 plates
#: (`compare_sides.py --variants auto`):
#:
#:     corners        accepted   court error: median    p90     max   flips
#:     the walk         58/65                  0.227  0.470   0.769       -
#:     near             59/65                  0.225  0.469   0.781   3 for, 2 against
#:     near, far        59/65                  0.219  0.454   0.972   2 for, 1 against
#:
#: The gain at the verdict is one plate, and that is the honest size of it: the
#: anchors are three residuals against some 440 line samples, so the fit dilutes
#: them however well they are placed. What the far ends buy is not in that column
#: at all - they are where both chord readers draw their strip from, and the
#: median and p90 above are where a straighter strip shows up.
#:
#: THE ONE BAD PLATE IS WORTH NAMING because it is not this change's fault and
#: the number in the `max` column belongs to it: `17-45-42` goes from 0.745 m
#: (refused) at the baseline to 0.115 m with the near corners alone and to 0.972 m
#: with the far ends as well - and the far end that did it MOVED HALF A PIXEL,
#: from -0.030 to -0.039 m. A plate whose court swings 0.86 m on a half-pixel
#: chord anchor is fragile in a way no endpoint rule can fix, and finding that is
#: worth more than the plate is.
CORNER_REFINE = ("near", "far")

#: Which qualifying position wins - `first` on the way out, or the `strongest`
#: in the window. Kept as a switch rather than deleted because the losing rule
#: is the one that makes the winning rule's argument: see `CORNER_PEAK`, and
#: `compare_ends.py --variants walk,curve_max,curve`, which measures all three.
CORNER_RULE = "first"


CORNER_ENDS = ("near", "far")


def corner_refine_wanted():
    """Which line's ends `corner_on_curve` places, as a frozenset."""
    env = os.environ.get("PADEL_CORNER_REFINE")
    want = CORNER_REFINE if env is None else env
    if not isinstance(want, str):
        want = ",".join(want)
    keys = tuple(k.strip().lower() for k in want.replace(" ", ",").split(",") if k.strip())
    if keys in ((), ("none",), ("off",), ("0",)):
        return frozenset()
    bad = [k for k in keys if k not in CORNER_ENDS]
    if bad:
        raise ValueError("unknown corner end %s; know %s"
                         % (", ".join(bad), ", ".join(CORNER_ENDS)))
    return frozenset(keys)


def corner_rule():
    r = (os.environ.get("PADEL_CORNER_RULE") or CORNER_RULE).strip().lower()
    if r not in ("first", "strongest"):
        raise ValueError("unknown corner rule %r; know first, strongest" % r)
    return r


def corner_on_curve(gray, pts, corner, chord, outward, reach=CORNER_REACH,
                    span=CORNER_SPAN, ang=CORNER_ANG, ang_step=CORNER_ANG_STEP,
                    step=CORNER_STEP, fit=CORNER_FIT, peak=CORNER_PEAK,
                    rule="first", trace=None):
    """Slide a corner ALONG its own line onto the edge that ends it.

    `chord` is a point further along the line the corner sits ON - the far
    service line's end, on the same side - so that the crossing line's direction
    is known before it is looked for. `outward` is +1 where the walk left along
    increasing x and -1 where it left along decreasing x.

    SCORED ON THE DIRECTIONAL DERIVATIVE ACROSS THE CROSSING LINE, not on an
    edge map. Two reasons, and the second is the one that matters:

    - A binary Canny map is 1 px wide, so a line integral over it is a spike
      with no shoulders and nothing to interpolate between; the derivative is
      smooth and has a sub-pixel maximum, which is the whole point of doing this
      instead of taking a walk sample.
    - It is SELECTIVE for free. An edge running along the crossing line has its
      gradient perpendicular to it, and the service line's own two long edges -
      which the integral passes over at every candidate position - have theirs
      at right angles to that, so they contribute nothing. No masking, no
      excluded band: the projection does it.

    UNSIGNED, and that is measured rather than assumed. What lies beyond the
    paint is not consistently darker: `NEAR_SVC_DROP` records a sponsor banner
    at L 144 past one end of `22-17-42` and the frame's shadow at L 62 past the
    other, against a floor at 100. A signed test tuned on either end is wrong at
    the other, and wrong at whichever venue paints its base the other colour.

    Returns (corner, moved_px) - the input corner unchanged when there is no
    chord, too little line, or nothing at the peak worth moving to.
    """
    pts = np.asarray(pts, float).reshape(-1, 2)
    corner = np.asarray(corner, float)
    if chord is None or len(pts) < 12:
        return corner, 0.0
    w = pts[np.argsort(np.abs(pts[:, 0] - corner[0]))[:fit]]
    if not IM.distinct_at_least(w[:, 0], 3):
        return corner, 0.0
    cf = IM.polyfit_fast(w[:, 0], w[:, 1], 2)

    # The box the search can reach, plus a margin for the gradient's own kernel.
    # Cropped rather than run on the frame: this is a 100 px question asked twice
    # a plate, and a full-frame Sobel to answer it would cost more than the
    # search does.
    pad = reach + span + 6.0
    h, wd = gray.shape[:2]
    x0, y0 = int(max(0, corner[0] - pad)), int(max(0, corner[1] - pad))
    x1, y1 = int(min(wd, corner[0] + pad + 1)), int(min(h, corner[1] + pad + 1))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return corner, 0.0
    box = gray[y0:y1, x0:x1]
    gx = cv2.Sobel(box, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(box, cv2.CV_32F, 0, 1, ksize=3)

    ss = np.arange(-reach, reach + 1e-9, step)
    xs = corner[0] + outward * ss
    P = np.column_stack([xs, np.polyval(cf, xs)]) - np.array([x0, y0], float)
    ts = np.arange(-span, span + 1e-9, 1.0)
    u0 = CM.unit(np.asarray(chord, float) - corner)
    a0 = math.atan2(u0[1], u0[0])
    M = []
    for a in np.radians(np.arange(-ang, ang + 1e-9, ang_step)) + a0:
        u = np.array([math.cos(a), math.sin(a)])
        n = CM.perp(u)
        Q = (P[:, None, :] + ts[None, :, None] * u).reshape(-1, 2)
        g = (IM.bilinear(gx, Q) * n[0]
             + IM.bilinear(gy, Q) * n[1]).reshape(len(ss), len(ts))
        # `nanmean` because the box is clipped at the frame edge and some of the
        # crossing line can fall outside it - the alternative is refusing every
        # corner near the edge, which is where the corners are.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            M.append(np.nanmean(np.abs(g), axis=1))
    # The best lean AT EACH POSITION rather than one lean for the window: the
    # crossing line's direction is a property of where it is, and a single angle
    # chosen from the strongest position would then be asked to explain an edge
    # 30 px away that the lens has turned by a degree or two.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = np.nanmax(np.array(M), axis=0)
    if np.isnan(sc).all():
        return corner, 0.0
    med = float(np.nanmedian(sc))
    v = np.nan_to_num(sc, nan=-np.inf)
    # THE FIRST QUALIFYING PEAK ON THE WAY OUT - see `CORNER_PEAK`. The search
    # starts `reach` INSIDE the walk's endpoint, not at it, because the walk can
    # be wrong in both directions and a rule that only looked outward could never
    # pull back a corner that had coasted onto the reflection.
    i = None
    if rule == "strongest":
        j = int(np.argmax(v))
        i = j if (med > 0 and v[j] >= peak * med) else None
    else:
        for j in range(1, len(v) - 1):
            if v[j] >= peak * med and v[j] >= v[j - 1] and v[j] > v[j + 1]:
                i = j
                break
    if i is None or med <= 0:
        if trace is not None:
            trace.update(moved=0.0, ratio=(float(np.nanmax(sc)) / med) if med > 0 else None,
                         why="no edge stands out at %.1fx the window's median" % peak)
        return corner, 0.0
    top = float(sc[i])
    # Sub-pixel, which is most of what this buys over choosing a walk sample:
    # the walk quantises the corner at its 4 px step and out here 4 px is 2 cm of
    # court at the near sideline and rather more at the far one.
    d = 0.0
    if 0 < i < len(sc) - 1 and np.isfinite(sc[i - 1:i + 2]).all():
        den = sc[i - 1] - 2 * sc[i] + sc[i + 1]
        if abs(den) > 1e-9:
            d = float(np.clip(0.5 * (sc[i - 1] - sc[i + 1]) / den, -1.0, 1.0))
    x = corner[0] + outward * (ss[i] + d * step)
    q = np.array([x, float(np.polyval(cf, x))])
    if trace is not None:
        # Signed OUTWARD along the line, not along x: "+2.5" means the walk
        # stopped 2.5 px short of the edge whichever end of the court it is.
        trace.update(moved=float(np.linalg.norm(q - corner)),
                     along=float(ss[i] + d * step), ratio=top / med, why="")
    return q, float(np.linalg.norm(q - corner))


#: How much of the perpendicular search window is there to cover the DETECTOR
#: rather than the line, in pixels - `_peak_ridge`'s own ~2 px of re-centring,
#: on the previous sample and on this one. The line's share is computed from
#: `max_tilt` instead of assumed, so the whole window is 4.0 + 1.5 = 5.5 px at
#: 4 px steps and 20 degrees, against the 9 px - growing to 21 on misses - that
#: a walk with no tilt limit has to use.
#:
#: WHAT THIS BUYS AND WHAT IT DOES NOT, because the two are different. It fixes
#: the failure it was aimed at: on `10-05-24` the far service line is a 31-40
#: ridge with the court's far edge at 66 and an unbroken glare ramp joining
#: them, so a 9 px window has an uphill path out of the line and a 5.5 px one
#: does not - the walk goes from 15 samples to 75 at 0.18 px curve rms, and
#: `13-30-43` from 69 to 78.
#:
#: It buys NOTHING at the level of the verdict: 109 of 116 accepted with it and
#: 109 without, and swept over 2.5/4.0/5.5/7.0 it is 109 at every setting. The
#: aggregate residuals are a wash - probe p50 1.241 -> 1.245, p90 1.776 ->
#: 1.790, max 2.051 -> 1.991; line rms p50 1.003 -> 0.990, p90 1.284 -> 1.325.
#: A tighter window costs misses on a noisy ridge, and misses are what the stop
#: rules key on, so 74 plates end their walk a little earlier.
#:
#: Kept because the mechanism is right rather than because the corpus asked for
#: it: a search radius is a claim about how far the line could have moved, and
#: where that is known it should be computed. 4.0 over 2.5 for the max-probe and
#: max-rms tail; nothing else separates the settings.
TILT_NOISE = 4.0


def scan_horizontals(f, p0, d0, edges, shape, step=4.0, half=9.0, min_contrast=14.0,
                     max_len=4000, coast_half=24.0, votes=HZ_VOTES_FIND, need=3):
    """Walk up the ridge stopping for NOTHING, and write down what it crosses.

    This is the same follower as `walk`, with every stop rule removed and every
    gap coasted over: it is not trying to find an endpoint, it is collecting
    evidence for a choice made afterwards. That inversion is what lets the Hough
    run at `HZ_VOTES_FIND` - a spurious candidate here is discarded by
    `choose_far_service`, while a spurious crossing inside `walk` would end the
    walk in the wrong place.

    Returns (candidates, gap) where each candidate is {y, x, seen, gap} - `seen`
    counting the steps that saw it and `gap` the distance up to the first run of
    `need` missing steps above it, or None if the ridge never goes missing there.
    """
    field = f["vert"]
    p, d = np.asarray(p0, float).copy(), CM.unit(np.asarray(d0, float))
    seq, hits, recent, miss, n = [], [], [], 0, 0
    while n * step < max_len:
        n += 1
        c = p + step * d
        if not (2 < c[0] < shape[1] - 3 and 2 < c[1] < shape[0] - 3):
            break
        # The search widens while coasting, as in `walk`, but is capped: coasting
        # for a long way must not start dragging in whatever else is nearby.
        q = IM._peak_ridge(field, c, CM.perp(d), min(coast_half, half + 2.0 * miss),
                           min_contrast)
        if q is None:
            miss += 1
            seq.append((float(c[1]), True))
            p = c
            continue
        miss = 0
        seq.append((float(q[1]), False))
        for y in horizontals(edges, q[0], q[1], votes=votes):
            if abs(y - q[1]) <= 12.0:
                hits.append((float(y), float(q[0])))
        recent.append(q)
        if len(recent) > 8:
            recent.pop(0)
        if len(recent) >= 4:
            v = recent[-1] - recent[0]
            d = CM.unit(0.4 * d + 0.6 * CM.unit(v)) if np.linalg.norm(v) > 1e-6 else d
        p = q

    def gap_above(y):
        run, start = 0, None
        for yy, gap in seq:
            if yy > y - 2:                       # not above this candidate yet
                continue
            if gap:
                start = yy if run == 0 else start
                run += 1
                if run >= need:
                    return float(y - start)
            else:
                run, start = 0, None
        return None

    cands, group = [], []
    for y, x in sorted(hits):                    # one line is seen over several steps
        if group and y - group[-1][0] <= 6.0:
            group.append((y, x))
            continue
        if group:
            cands.append(group)
        group = [(y, x)]
    if group:
        cands.append(group)
    out = []
    for g in cands:
        y = float(np.median([v[0] for v in g]))
        out.append({"y": y, "x": float(np.median([v[1] for v in g])),
                    "seen": len(g), "gap": gap_above(y)})
    return out, seq


def choose_far_service(f, cands, shape):
    """Which of the crossed lines is the far service line, if any.

    Each survivor of the gap test is WALKED, left and right, and judged on what
    it turns out to be: a line that reaches `FIND_REACH` either side and holds a
    quadratic to `FIND_RMS` is a court line, and the cleanest such line is the
    answer. Returns the candidate with its walked points, or None to abstain.
    """
    best = None
    for c in cands:
        if c["gap"] is None or c["gap"] > FIND_GAP:
            continue
        p0 = np.array([c["x"], c["y"]], float)
        lf = walk(f["horiz"], p0, (-1, 0), shape=shape, lab=f["lab"])
        rt = walk(f["horiz"], p0, (1, 0), shape=shape, lab=f["lab"])
        if not len(lf) and not len(rt):
            continue
        P = np.vstack([lf[::-1], [p0], rt])
        if len(P) < FIND_MIN:
            continue
        if (p0[0] - P[:, 0].min() < FIND_REACH) or (P[:, 0].max() - p0[0] < FIND_REACH):
            continue
        co = np.polyfit(P[:, 0], P[:, 1], 2)
        rms = float(np.sqrt(np.mean((np.polyval(co, P[:, 0]) - P[:, 1]) ** 2)))
        if rms > FIND_RMS:
            continue
        if best is None or rms < best["rms"]:
            best = {"y": c["y"], "x": c["x"], "gap": c["gap"], "seen": c["seen"],
                    "rms": rms, "n": len(P), "coeffs": co, "pts": P}
    return best


def fields(bgr):
    """Everything the walks read, computed once."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gb = cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32)
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    # local mean absolute deviation: the floor is smooth, the net's mesh is not.
    # This is what keeps a column scan from re-acquiring "court" inside the net,
    # whose navy mesh has almost the same chroma as a blue court.
    rough = cv2.blur(np.abs(gb - cv2.blur(gb, (5, 5))), (9, 9))
    iso = IM.line_response(bgr)
    return {
        "gray": gb, "lab": lab, "rough": rough, "iso": iso,
        # A scan crossing a painted line must step over its SHOULDERS too. The
        # far service line carries a dark shadow along its upper edge that is
        # neither paint nor court, and undilated it split the far half into two
        # runs of 15px each - the far half stopped existing as a single thing.
        "bridge": cv2.dilate(iso, np.ones((7, 7), np.uint8)),
        "horiz": IM.directional_tophat(bgr, horizontal=True),
        "vert": IM.directional_tophat(bgr, horizontal=False),
    }


def _sample(field, pts):
    return IM.bilinear(field, np.asarray(pts, float).reshape(-1, 2))


def _sample_lab(lab, pts):
    return IM.bilinear(lab, np.asarray(pts, float).reshape(-1, 2))


# ------------------------------------------------------- 2. the anchor T ----
#: The near service T sits at (0.511w, 0.778h) at one venue and (0.485w, 0.780h)
#: at the other - the geometry that puts a camera behind the near baseline puts
#: this junction in the same place every time. The box is wide enough to absorb
#: a lot of that, and narrow enough to exclude the net, which is the only other
#: junction of white lines in the frame.
T_BOX = (0.30, 0.55, 0.70, 0.96)


def find_T(f, shape, box=T_BOX, arm=(14, 60), stub=(55, 95), spread=5,
           step=3, verbose=False):
    """Locate the near service line x centre line junction. No model, no seed.

    Scored by ARMS, not by white density. A "grow a square and take the highest
    white fraction" score peaks in the middle of the largest white area, which
    on this footage is the net's top tape - a junction is not where there is the
    most white, it is where white leaves in a particular set of directions.

    The near service T has three arms: left and right along the service line,
    and one pointing away from the camera along the centre line. The fourth
    direction - towards the camera - is bare court past a short stub, and that
    asymmetry is what tells this junction apart from the net's centre strap
    crossing its tape, which has four.

    Each arm reads the directional top-hat that isolates ITS orientation, and
    takes the max over a few pixels across the arm so that a service line bowed
    by the lens (~65px end to end here) still registers at 60px out.
    """
    h, w = shape[:2]
    x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    y1 = min(y1, h - int(stub[1]) - 2)           # keep the stub arm inside the frame
    r = np.arange(arm[0], arm[1] + 1, 4.0)
    rs = np.arange(stub[0], stub[1] + 1, 4.0)
    off = np.arange(-spread, spread + 1, 1.0)

    # The candidates that sit on paint, in the order the scan used to visit them
    # - rows outward from the top of the box, columns left to right - because
    # the winner is the FIRST of any equal scores and that order is the
    # tie-break. All four arms of all of them are then read in four calls
    # instead of four per candidate; at a few hundred candidates of 130-odd
    # samples each, the per-call overhead was the whole cost.
    ys = np.arange(y0, y1, step)
    xs = np.arange(x0, x1, step)
    on = f["iso"][np.ix_(ys, xs)] >= 20
    iy, ix = np.nonzero(on)
    if not len(iy):
        raise SystemExit("no white junction inside the search box")
    cxs, cys = xs[ix].astype(float), ys[iy].astype(float)

    def arms_of(field, ax, sgn, rad):
        """ax=0 reads along x with the spread across y; ax=1 the other way."""
        along = np.repeat(sgn * rad, len(off))            # len(rad)*len(off)
        across = np.tile(off, len(rad))
        if ax == 0:
            P = np.stack([cxs[:, None] + along, cys[:, None] + across], axis=2)
        else:
            P = np.stack([cxs[:, None] + across, cys[:, None] + along], axis=2)
        v = _sample(field, P.reshape(-1, 2)).reshape(len(cxs), len(rad), len(off))
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.nanmax(v, axis=2), axis=1)

    with warnings.catch_warnings():              # all-NaN arms are legitimate
        warnings.simplefilter("ignore", RuntimeWarning)
        L = arms_of(f["horiz"], 0, -1, r)
        R = arms_of(f["horiz"], 0, +1, r)
        U = arms_of(f["vert"], 1, -1, r)
        D = arms_of(f["vert"], 1, +1, rs)
    # `min(L, R, U)` as Python evaluates it, NaNs and all: a NaN in R or U loses
    # the comparison and leaves the running minimum alone, and only a NaN in L
    # carries through. It matters because an arm that falls off the frame is NaN
    # and the loop this replaces then skipped the candidate outright.
    m = np.where(R < L, R, L)
    m = np.where(U < m, U, m)
    s = m - 0.5 * D
    s = np.where(np.isnan(s), -np.inf, s)        # `s > best` was never true of NaN
    i = int(np.argmax(s))
    best = float(s[i])
    if not best > -1e18:
        raise SystemExit("no white junction inside the search box")
    bestp = (cxs[i], cys[i], float(L[i]), float(R[i]), float(U[i]), float(D[i]))
    p = np.array(bestp[:2])
    if verbose:
        print("  service T seed at (%.1f, %.1f)  score %.1f   arms L=%.0f R=%.0f U=%.0f D=%.0f"
              % (p[0], p[1], best, bestp[2], bestp[3], bestp[4], bestp[5]))
    return p


#: How far from the junction the walked samples used to locate it are taken.
#: `IN` skips the blob where the two ridges have merged into one. `OUT` bounds
#: how much of each line the quadratic has to model; measured over the corpus it
#: matters and 400 is a minimum rather than a compromise - the junction's y
#: error against the fitted cameras runs 0.80 / 0.72 / 0.57 / 0.60 / 0.92 px at
#: 200 / 300 / 400 / 600 / unbounded. Too little and the fit is short and noisy,
#: too much and a quadratic stops describing a fisheye's bow.
JOIN_IN, JOIN_OUT = 15.0, 400.0
JOIN_GUARD = 30.0


def _junction_from_walks(near_svc, centre, p, deg=2):
    """The junction, from the two walks that pass through it.

    A local refinement has to invent its own samples, and at this one place in
    the image they are the hardest samples in the frame to get. The obvious one
    - fit each ridge from 12 to 55px out and intersect - was measured over 104
    plates and does not work: `_peak_ridge` subtracts the median of its window,
    so a 9px half-window that fits INSIDE the centre line's paint (~18px wide
    where it meets the junction, this being the closest point on it to the
    camera) has no contrast left and returns nothing. Median zero samples out of
    43. It silently did nothing on 95 plates of 104, and on the 9 where it did
    fire it fitted a quadratic to a 15px span, extrapolated it 40px, and moved
    the junction 17px sideways on two of them. Widening the window to 12px finds
    43/43 samples and is no better overall (junction repeatability 5.10px
    against 4.24) - the estimator is weak because its samples all lie on ONE
    side of the answer, not because it cannot see them.

    The walks have no such problem. They follow each line for hundreds of
    pixels, re-centring at every step and widening their own window whenever a
    step misses, and the near service line runs THROUGH the junction and out the
    other side, so its fit interpolates. Fitting each walked line over a window
    either side of the seed and intersecting gives ~130 sub-pixel samples per
    line instead of 43 and 0, and measures better on both scales available:
    junction repeatability across same-pose recordings 2.49px against 4.24, and
    y error against the fitted cameras 0.57px against 3.16.

    (x error is 2.4-4.3px for every estimator including the raw box scan, so
    that column says nothing about any of them - it is the fitted camera's own
    uncertainty in x, which is where this calibration is weakest.)

    Returns `p` unchanged if either line is too thin here or the answer moves
    further than a box-scan cell plus a lens bow could explain - a junction that
    disagrees with the seed by 30px means one of the walks left its line, and
    the seed is then the only thing still known to sit on paint.
    """
    a, b = np.asarray(near_svc, float), np.asarray(centre, float)
    if len(a) < 40 or len(b) < 20:
        return p
    du, dv = np.abs(a[:, 0] - p[0]), p[1] - b[:, 1]
    a = a[(du > JOIN_IN) & (du < JOIN_OUT)]
    b = b[(dv > JOIN_IN) & (dv < JOIN_OUT)]
    if len(a) < 30 or len(b) < 15 or np.ptp(a[:, 0]) < 60 or np.ptp(b[:, 1]) < 40:
        return p
    a1 = np.polyfit(a[:, 0], a[:, 1], deg)       # y = a(x), the service line
    a2 = np.polyfit(b[:, 1], b[:, 0], deg)       # x = b(y), the centre line
    q = np.asarray(p, float).copy()
    for _ in range(20):                          # two curves, alternate substitution
        q = np.array([np.polyval(a2, q[1]), np.polyval(a1, np.polyval(a2, q[1]))])
    if not np.isfinite(q).all() or np.linalg.norm(q - p) >= JOIN_GUARD:
        return p
    return q


# --------------------------------------------------------- 3. the walks ----
def walk(field, p0, d0, step=4.0, half=9.0, min_contrast=14.0, max_miss=6,
         max_len=4000, shape=None, stop=None, drop=0.45, tol=4.0, warm=12,
         fit=120, confirm=3, win=15, gap=3, resume=0.6, trace=None,
         lab=None, end_drop=0.75, ahead=(8.0, 16.0, 24.0), gap_curve=False,
         edges=None, tight_miss=1, rough=None, rough_max=None, max_tilt=None, colour_ridge=False):
    """Follow a ridge from a point in a direction, and STOP where the line ends.

    The direction is re-estimated from the last few accepted samples rather than
    from the previous step alone: a single noisy re-centring then bends the walk
    by a few degrees instead of derailing it, which matters over the 900px the
    near service line runs here.

    Knowing where to stop is what makes the endpoints usable. A padel court is
    walled in glass, and the glass REFLECTS the service line: run past the court
    boundary and there is still a ridge to follow, dimmer and leaving at a
    different angle. Unstopped, this walk ran 60-96px past the real corner at
    both reference venues and put the court's edge at X=-1.26m and X=11.11m
    instead of 0 and 10. Two cues stop it, both measured against the walk's own
    recent history so that vignetting and venue colour never enter:

    BRIGHTNESS - the ridge response collapses where paint stops. Measured across
    the galais left corner: ridge 140 -> 45, grey 225 -> 120, inside 8px. A
    reflection is about a third as bright as the paint making it.

    SHAPE - a straight world line seen through this lens is a quadratic to well
    under a pixel, so anything that leaves that curve is a different feature.
    Testing one 4px step's DIRECTION does not work, though it is the obvious
    thing to try: `_peak_ridge` re-centres up to 2px sideways, which is 27
    degrees of noise per step, and such a test trips beside the anchor where the
    service and centre ridges are still merged. Extrapolating a quadratic fitted
    to the last ~120 accepted points averages that away and still catches a
    43-degree kink within a few pixels. Its gate is the fitted curve's OWN
    residual, because the two venues differ 2x in scatter on the same line and a
    fixed 3px trips mid-line on the noisier one.

    Measured against click-seeded ground truth, the four near-service endpoints
    land at X = -0.07, 10.08, -0.04, 10.07 m against a true 0 and 10.

    WHAT IS ON THE OTHER SIDE (`lab`, optional) is a third cue, and for the
    centre line it is the decisive one, because that line ends against the far
    service line and its endpoint is the anchor the whole far half hangs off.
    The question there is not "is this sample still good" - it is, right up to
    the junction - but "is there any line left in front of it". Measured as L a
    little way ahead over the walk's OWN median L, which is what makes it
    survive four venues at four exposures.

    Three things had to be right, and each was measured:

    LIGHTNESS, not chroma. Chroma looks spectacular on the first venue tried -
    61-67 units from the court colour along the line, 1.4 at the T - and then
    fails completely at an unsaturated one, where court and paint have the SAME
    chroma, every sample reads 0.1-1.3 from the court colour, paint included,
    and the test fires on all 120 steps. Lightness is the one channel where a
    court cannot be mistaken for the line painted on it: whatever colour it is,
    it is darker than its own markings.

    THE BRIGHTEST OF SEVERAL DISTANCES, not one. The net's cord crosses this
    line and is as dark as bare court, but there is paint again 16px further on.
    One distance leaves the worst legitimate sample at 0.33 against a median
    0.59 at the real end, which overlaps; the brightest of 8/16/24 separates
    0.51 from 0.64 and lifts the 5th percentile of legitimate samples to 0.83.

    AFTERWARDS, not at the first reading. Even so a wide occlusion can read as
    an end, and no test applied forward in time can tell the two apart - the
    difference is whether paint ever comes back, which is not knowable yet. So
    the walk runs on and is then cut back to the first sample of the TRAILING
    run with nothing ahead of it. This costs nothing: the walk was going to take
    those steps anyway, and letting it take them is what makes the answer safe.
    """
    p, d = np.asarray(p0, float).copy(), CM.unit(np.asarray(d0, float))
    d_seed = d.copy()                            # the direction the line set out in
    swap = abs(d[0]) < abs(d[1])                 # x = f(y) for a vertical walk
    out, recent, rhist, miss, weak, bent = [], [], [], 0, 0, 0
    lhist, abad, tilted = [], [], 0
    crossed, lines, at_gap, clear = False, [], False, False  # lines walked over
    why, at_end = "ran to the frame edge", False

    def curve_off(q):
        """How far `q` sits off the quadratic through the recent walk, and the
        gate for that, which is the curve's own residual rather than a constant."""
        w = np.array(out[-fit:])
        a = w[:, 1] if swap else w[:, 0]
        b = w[:, 0] if swap else w[:, 1]
        if not IM.distinct_at_least(a, 3):
            return None, None
        cf = IM.polyfit_fast(a, b, 2)
        gate = max(tol, 6.0 * float(np.sqrt(((b - np.polyval(cf, a)) ** 2).mean())))
        qa, qb = (q[1], q[0]) if swap else (q[0], q[1])
        return abs(qb - np.polyval(cf, qa)), gate

    tilt_reach = None if max_tilt is None else float(np.tan(np.radians(max_tilt)))
    while len(out) < max_len:
        c = p + step * d
        if shape is not None and not (2 < c[0] < shape[1] - 3 and 2 < c[1] < shape[0] - 3):
            break
        if stop is not None and stop(c):
            why = "caller said stop"
            break
        # HOW FAR SIDEWAYS MAY THE NEXT SAMPLE BE? `half` is a search radius, and
        # a search radius is a statement about how far the line could have moved.
        # Where `max_tilt` says the line's direction is known, that statement can
        # be computed instead of assumed: over one `step` a line held to
        # `max_tilt` moves at most `step * tan(max_tilt)` perpendicular - 1.5 px
        # at 4 px steps and 20 degrees - and the rest of the window is only there
        # to cover `_peak_ridge`'s own ~2 px of re-centring noise. Coasting over
        # `miss` gaps, the budget grows by one step's worth per step SKIPPED,
        # which is the same geometry, rather than by a flat 2 px that has nothing
        # to do with the line.
        reach = (half + 2.0 * miss if max_tilt is None else
                 TILT_NOISE + (miss + 1) * step * tilt_reach)
        q = (IM.peak_ridge_lab(lab, c, CM.perp(d), reach, min_contrast)
             if colour_ridge and lab is not None else
             IM._peak_ridge(field, c, CM.perp(d), reach, min_contrast))
        if q is None:
            miss += 1
            # A tight leash, where a caller asks for one, is right in the BODY
            # of the line and wrong at its start: the first steps away from the
            # anchor run through the junction itself, where the two ridges are
            # still one blob and the perpendicular peak search finds nothing.
            # Measured, the centre line misses its first two steps on most
            # plates and three on some. Until the walk is warm it is acquiring.
            # A GAP MEANS TWO DIFFERENT THINGS EITHER SIDE OF THE NET, so the
            # leash changes length when the walk crosses it. Before: the net's
            # strap and tape interrupt the line and the walk has to be allowed
            # through them, which is also what lets it coast off the end. After:
            # the line ahead is plain paint on plain floor for its last few
            # centimetres and there is nothing left to coast over, so the first
            # real gap IS the end. `crossed` is latched by walking over a
            # horizontal white line, which on this course happens exactly once
            # before the far half - at the top of the net.
            leash = tight_miss if (crossed and clear) else max_miss
            if miss > (leash if len(out) >= warm else max(leash, 6)):
                if crossed and clear:
                    # Past the last horizontal line there is nothing left to
                    # coast over, so this gap is not a gap - it is the end, and
                    # the last sample is the far service T.
                    why = ("the line stopped at a %d-step gap past the last "
                           "horizontal line it crossed" % miss)
                    at_gap = True
                else:
                    why = "ridge lost"
                break
            p = c                                # coast straight through the gap
            continue
        gapped, miss = miss, 0
        r = IM.bilinear_at(field, q[0], q[1])

        # A RUN OF MISSES is the strongest evidence the line has ended, and
        # coasting through it is the one thing that hid the end completely: the
        # brightness rule needs three consecutive ACCEPTED samples below its
        # threshold, and the samples that would have supplied them are exactly
        # the ones the walk skipped. Measured over the corpus, a real scuff in
        # the paint resumes at 0.93-1.13 of the brightness before it, while the
        # walk that ran 130px past the far service T into the background resumed
        # at 0.13, then 0.43, then 0.61. Real paint continues; a different
        # feature does not have to.
        if gapped >= gap and len(out) >= warm:
            med = _med(rhist[-win:])
            if med > 0 and r < resume * med:
                why = ("line did not resume after a %d-step gap (%.0f%% of its "
                       "brightness)" % (gapped, 100 * r / med))
                break

        # Requiring the re-acquired point to sit on the line's own curve was
        # tried here and does NOT separate: measured over the corpus a genuine
        # resume after a scuff lands up to 4.9 px off the extrapolation, and the
        # escape this was written to catch is 6.1 px off. A gate between them is
        # a coin toss, and the version that caught the escape also truncated a
        # good plate half way up the far half. The colour cue is what actually
        # ends this walk, and it ends it two steps BEFORE the gap even opens.
        if gap_curve and gapped and len(out) >= warm:
            off, gate = curve_off(q)
            if off is not None and off > gate:
                why = ("did not resume where it left off: %.1f px off its own "
                       "curve after a %d-step gap (gate %.1f)" % (off, gapped, gate))
                break

        # NO LINE LEFT AHEAD - recorded per sample, acted on after the walk.
        # `ahead` is a LIST of distances and the test takes the brightest of
        # them, so a dark thing the line passes UNDER does not read as the line
        # ending: the net's cord crosses the centre line and is as dark as bare
        # court, but there is paint again 16px further on. Measured over the
        # corpus, one distance alone leaves the worst legitimate sample at 0.33
        # against a median 0.59 at the real end - overlapping - while the
        # brightest of three separates 0.51 from 0.64 and lifts the 5th
        # percentile of legitimate samples from 0.49 to 0.83.
        nothing_ahead = False
        # This sample's own L and the three ahead of it, read in ONE pass. They
        # are four separate points of one image and the sampler is elementwise,
        # so the arithmetic is unchanged; what goes is three quarters of the
        # per-call overhead, which at four reads per step of every walk was the
        # largest single share of the frame.
        Lq = None
        if lab is not None:
            P = np.empty((len(ahead) + 1, 2))
            P[0] = q
            for j, kk in enumerate(ahead):
                P[j + 1] = q + kk * d
            Lv = _sample_lab(lab, P)[:, 0]
            Lq = float(Lv[0])
            if len(out) >= warm:
                med = _med(lhist[-win:])
                La = max(float(v) for v in Lv[1:])
                nothing_ahead = med > 0 and La < end_drop * med

        if len(out) >= warm:
            med = _med(rhist[-win:])
            if r < drop * med:
                weak += 1
                if weak >= confirm:
                    del out[-(confirm - 1):], abad[-(confirm - 1):]
                    why = "brightness fell to %.0f%% of its running median" % (100 * r / med)
                    break
            else:
                weak = 0
            off, gate = curve_off(q)
            if off is not None:
                if off > gate:
                    bent += 1
                    if bent >= confirm:
                        del out[-(confirm - 1):], abad[-(confirm - 1):]
                        why = "left its own curve by %.1f px (gate %.1f)" % (off, gate)
                        break
                else:
                    bent = 0

        # HOW STEEPLY IS THE WALK LEAVING THE DIRECTION IT SET OUT IN? The two
        # rules above are both RELATIVE to the walk's own recent history, and at
        # the far service T that is exactly what fails. Measured per sample on
        # the four plates where this walk is lost:
        #
        #   - `10-05-24` escapes onto the court's far edge inside EIGHT samples,
        #     before `warm` arms the curve rule at all: `off` and its gate are
        #     still None while the walk climbs 29 px.
        #   - `17-28-34` and `13-30-43` escape later, and the curve rule watches
        #     it happen and disarms itself. Its gate is `6 * rms` over the last
        #     `fit` samples, and those samples now include the escape, so the
        #     gate inflates 4.0 -> 30.4 px in twelve steps while the offence
        #     holds at 13-15 px. `bent` never reaches `confirm`; tightening the
        #     multiplier does not help, because the reference is contaminated.
        #
        # A court line's tilt in the IMAGE is not a fact about this walk, so no
        # amount of drift can move it - which is the whole point. See `FAR_TILT`
        # for what the gate can and cannot separate on its own.
        #
        # Over a chord rather than a step, and with `confirm` behind it, for the
        # reason in this function's docstring: `_peak_ridge` re-centres up to
        # 2 px sideways, which is 27 degrees of noise on one 4 px step. Eight
        # steps average that away, and a single-sample spike - `13-30-43` throws
        # one of 16.5 degrees at i=55 and is back to 3.3 at i=56 - must not end
        # a walk with 120 good samples still ahead of it.
        if max_tilt is not None and len(out) >= 3:
            v = q - out[-8 if len(out) >= 8 else 0]
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                cos = abs(float(np.dot(v / n, d_seed)))
                if np.degrees(np.arccos(min(1.0, cos))) > max_tilt:
                    tilted += 1
                    if tilted >= confirm:
                        del out[-(confirm - 1):], abad[-(confirm - 1):]
                        why = ("left the direction it set out in by more than "
                               "%.0f degrees" % max_tilt)
                        break
                else:
                    tilted = 0

        # WHAT HAVE I JUST WALKED OVER? A horizontal white line through the
        # sample's own window, found on the strict edge map. Recorded in order,
        # so the LAST one before the walk ends is the far service line - the
        # line the centre line ends against.
        if edges is not None and len(out) >= warm:
            for y in horizontals(edges, q[0], q[1]):
                if abs(y - q[1]) <= 4.0 and (not lines or abs(y - lines[-1][1]) > 8.0):
                    lines.append((float(q[0]), float(y)))
                    if not crossed:
                        crossed, clear = True, False
        # ...and am I off it yet? Only once there is court either side of the
        # line again is the walk past the thing it crossed, and only then does a
        # gap ahead mean the end rather than the crossing.
        if crossed and not clear and rough is not None and rough_max is not None:
            F = np.vstack([q + t * k * CM.perp(d) for k in (10.0, 16.0, 24.0) for t in (-1, 1)])
            clear = float(np.median(_sample(rough, F))) <= HZ_SMOOTH * rough_max

        out.append(q)
        abad.append(nothing_ahead)               # index-aligned with `out`
        recent.append(q)
        rhist.append(r)
        if lab is not None:
            lhist.append(Lq)
        if len(recent) > 8:
            recent.pop(0)
        if len(recent) >= 4:
            rr = np.array(recent)
            v = rr[-1] - rr[0]
            d = CM.unit(0.4 * d + 0.6 * CM.unit(v)) if np.linalg.norm(v) > 1e-6 else d
        p = q
    # WHERE THE LINE STOPPED AND DID NOT COME BACK. The walk is allowed to run
    # on past the end - it will, the ridge follower always finds something - and
    # is then cut back to the first sample of the trailing run with no paint in
    # front of it. Breaking out at the first such sample instead cannot work:
    # the same reading is produced by the net crossing the line, and telling the
    # two apart needs to know whether paint ever returned, which is only knowable
    # afterwards. This is what "the white stops" means when the walk cannot see
    # into the future - and it costs nothing, because the walk was going to run
    # those extra steps anyway.
    i = len(out)
    while i > 0 and abad[i - 1]:
        i -= 1
    if 0 < i < len(out):
        dropped = len(out) - i - 1
        out = out[:i + 1]                        # keep the endpoint itself
        at_end = True
        if dropped:
            why = "the line stopped %d step(s) before the walk did" % dropped
        else:
            why = "the line stopped here - no paint ahead of it"
    elif abad and abad[-1]:
        at_end = True                            # ended exactly where it stopped
    if at_gap:
        at_end = True

    if trace is not None:
        # `at_end` is the machine-readable half of `why`: it says the walk
        # stopped because the LINE stopped, not because the ridge got lost or
        # the frame ran out. Only then does its last point mean anything as a
        # landmark - the other reasons leave it wherever the walk happened to
        # die, and those two populations sit 0.35 m apart in world Y.
        trace["why"], trace["at_end"] = why, at_end
        trace["lines"] = lines
    return np.array(out) if out else np.zeros((0, 2))


# ------------------------------------------------- 4. runs along a scan ----
def _med(v):
    """Median of a short list of floats - `np.median(v)`, without numpy.

    It is here because at fifteen numbers a numpy call is entirely overhead, and
    this one runs three times per pixel of every column scan - a hundred columns
    of several hundred pixels per frame - which made it the single largest cost
    in the whole detector.
    """
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n & 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def scan_runs(f, mu, gate_ab, rough_max, p0, d, n, tol_ab=9.0, tol_l=30.0,
              confirm=4, reacquire=5, bridge=22, max_runs=2, min_run=(150, 18)):
    """Walk one pixel at a time and cut the walk into court-ON runs.

    Three tests, and which one applies depends on what is being asked.

    STAYING in a run is NEIGHBOUR-RELATIVE plus ROUGHNESS - deliberately not the
    absolute colour test. These cameras vignette hard enough that the court's
    chroma has drifted 9 units by the frame corners, so an absolute gate tight
    enough to stop at the wall also stops in the middle of the court: measured,
    a column at x=300 ended 220px early. Against a running estimate of the
    colour just behind it, a gradient never accumulates. Same reason
    `CourtAuto.court_region` flood-fills instead of thresholding.

    ROUGHNESS is the local mean absolute deviation of the grey image, and it is
    here because colour alone cannot see the net. At one venue the navy mesh is
    14 chroma units from the blue court, close enough that a neighbour-relative
    scan ran straight up the column, through the net and out the other side
    without noticing. The floor is smooth (~1) and a mesh is not (~6-9),
    whatever colour either happens to be.

    RE-ACQUIRING the next run after a gap is ABSOLUTE - raw Lab chroma distance
    from the surface colour - because across the net there is no neighbour left
    to compare to. Raw, not Mahalanobis: the trimmed estimator's scatter is
    ~0.8 units wide, so a chi-squared gate loose enough for the far end of the
    court is 60 sigma and the number stops meaning anything. Court measures
    under 4 at both venues and mesh 14 and 26, so a threshold near 10 is not a
    tuned constant, it is the gap.

    Paint is bridged rather than counted as leaving the court, but only for
    `bridge` pixels: the service lines are ~10px across and must not end a run,
    while the net's top tape is ~25px and must.

    `min_run` is per slot - the near half is most of the frame, the far half is
    a 36-47px sliver. Runs too short for their slot are discarded WITHOUT
    consuming it, so "run 1 is the near half, run 2 is the far half" survives
    the fragmentation that the dark frame corners produce.
    """
    P = np.asarray(p0, float) + np.arange(n)[:, None] * CM.unit(np.asarray(d, float))
    lab = _sample_lab(f["lab"], P)
    paint = _sample(f["bridge"], P) > 18.0
    rough = _sample(f["rough"], P)
    ok = np.isfinite(lab).all(axis=1)
    dist = np.linalg.norm(lab[:, 1:] - mu, axis=1)
    smooth = ok & (rough < rough_max)
    fresh = smooth & (dist < gate_ab)
    if np.isscalar(min_run):
        min_run = (min_run,) * max_runs

    def need(k):
        return min_run[min(k, len(min_run) - 1)]

    # Read out into plain Python before the loop: every test below is on one
    # sample of three numbers, where a numpy call costs more than the whole
    # iteration around it. See `_med`.
    labL, okL, paintL = lab.tolist(), ok.tolist(), paint.tolist()
    smoothL = smooth.tolist()
    freshL = fresh.tolist()

    # The window is held as three bounded columns rather than a growing list of
    # samples, so the running median is three sorts of fifteen floats and no
    # allocation at all - it is taken once per pixel and there are tens of
    # thousands of pixels per frame.
    h0, h1, h2 = [], [], []
    runs, ref, state = [], None, "out"
    start, offc, onc, bridged, ran_off, i = 0, 0, 0, 0, False, 0
    for i in range(n):
        if not okL[i]:
            break
        li = labL[i]
        if state == "in":
            if paintL[i]:
                bridged += 1
                if bridged > bridge:              # too wide to be a painted line
                    _close(runs, start, i - bridged, need(len(runs)))
                    state, offc, onc, bridged = "out", 0, 0, 0
                continue
            bridged = 0
            da, db = li[1] - ref[1], li[2] - ref[2]
            jump = (math.sqrt(da * da + db * db) > tol_ab
                    or abs(li[0] - ref[0]) > tol_l)
            if not smoothL[i] or jump:
                offc += 1
                if offc >= confirm:
                    _close(runs, start, i - confirm + 1, need(len(runs)))
                    state, offc, onc, bridged = "out", 0, 0, 0
                    if len(runs) >= max_runs:
                        break
            else:
                offc = 0
                if len(h0) == 15:
                    del h0[0], h1[0], h2[0]
                h0.append(li[0])
                h1.append(li[1])
                h2.append(li[2])
                # Median of a window, not an exponential average. An EMA tracks
                # what it is supposed to be detecting: the court-to-wall ramp
                # here spans ~8px, and an EMA lagging 2-3 samples behind a
                # 3-units-per-pixel ramp never sees more than 8 units of
                # difference, so every sideline scan walked straight through the
                # wall and off the frame. A 15-sample median lags the ramp by
                # half its window and trips immediately.
                ref = (_med(h0), _med(h1), _med(h2))
            continue
        # state == "out": looking for the next run to open
        if freshL[i] and not paintL[i]:
            onc += 1
            if onc >= reacquire:
                state, start = "in", i - reacquire + 1
                win = labL[start:i + 1]
                h0 = [r[0] for r in win]
                h1 = [r[1] for r in win]
                h2 = [r[2] for r in win]
                ref = (_med(h0), _med(h1), _med(h2))
                offc, bridged = 0, 0
        else:
            onc = 0
    if state == "in":
        # Ran out of samples still on the court: this scan never found an edge,
        # so whatever it returns is the frame border rather than a court line.
        ran_off = True
        if len(runs) < max_runs:
            _close(runs, start, min(i, n - 1), need(len(runs)))
    return runs, P, lab, dist, paint, ran_off


def edge_at(f, c, n, rough_max, inner=34.0, outer=26.0, tol=9.0, confirm=3):
    """Sub-pixel court -> outside crossing nearest `c`, searching along +n.

    Both gates come from THIS profile's own inner samples. A roughness gate
    measured once beside the anchor reads 4.1 at one venue while the bare court
    out by the sideline runs 2.6-4.5, so a single global gate trips on court and
    kills the track after 14 points. Same failure mode as an absolute colour
    gate under vignetting, and the same fix.
    """
    n = CM.unit(np.asarray(n, float))
    s = np.arange(-inner, outer + 0.01, 1.0)
    P = np.asarray(c, float) + s[:, None] * n
    lab = _sample_lab(f["lab"], P)
    rough = _sample(f["rough"], P)
    if not np.isfinite(lab).all():
        return None
    k = int(inner * 0.4)
    ref = np.median(lab[:k, 1:], axis=0)
    base = float(np.median(rough[:k]))
    if base > 4.0 * rough_max:                    # not starting on court at all
        return None
    rgate = max(rough_max, 3.0 * base + 1.5)
    dist = np.linalg.norm(lab[:, 1:] - ref, axis=1)
    bad = (dist > tol) | (rough > rgate)
    run = 0
    for i in range(k, len(s)):
        run = run + 1 if bad[i] else 0
        if run >= confirm:
            return P[0] + _edge_subpixel(lab, i - confirm + 1) * n
    return None


def canny_edges(bgr, lo=255, hi=255):
    """Only the strongest gradients in the frame.

    Both thresholds at 255 with an L2 gradient is a very high bar - a 3x3 Sobel
    magnitude tops out near 1442 - and that is the point. The court's boundary
    is a hard step from playing surface to wall base; the wear marks, shading
    and colour drift that the row scans kept stopping at are gradual and simply
    do not fire. It costs nothing to be this strict because the boundary is not
    a subtle feature, it is the only hard vertical edge out there.
    """
    return cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), lo, hi,
                     apertureSize=3, L2gradient=True)


def _line_kernel(angle_deg, length):
    """A 1-px anti-aliased line through the centre of an NxN kernel."""
    n = int(length) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    a = np.radians(angle_deg)
    d = np.array([np.cos(a), -np.sin(a)])
    for t in np.linspace(-length / 2.0, length / 2.0, int(length) * 4):
        p = np.array([c, c]) + t * d
        x0, y0 = int(np.floor(p[0])), int(np.floor(p[1]))
        fx, fy = p[0] - x0, p[1] - y0
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= y0 + dy < n and 0 <= x0 + dx < n:
                k[y0 + dy, x0 + dx] += wt
    s = k.sum()
    return k / s if s > 0 else k


def canny_along(gray, angle_deg, box, shape, length=15, lo=255, hi=255):
    """Smooth ALONG the edge we are hunting, then Canny.

    Averaging parallel to an edge cancels the speckle and short wear marks that
    litter a worn court, and does NOT soften the step across the edge, because
    no averaging happens in that direction. Measured on the two worn plates in
    this corpus, candidate purity goes from 12-34% to 92-96% - and the measured
    X does not move (galais -0.113 -> -0.116 m), so it sharpens the SELECTION
    without biasing the position.

    Direction matters entirely: smoothing across the same edge, or isotropically,
    destroys it - 0-7 usable candidates against 83-107. Only the crop that will
    actually be searched is filtered, which keeps the cost near one full-frame
    pass per plate.
    """
    y0, y1, x0, x1 = box
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return np.zeros(shape[:2], np.uint8)
    b = cv2.filter2D(sub.astype(np.float32), -1, _line_kernel(angle_deg, length))
    e = cv2.Canny(np.clip(b, 0, 255).astype(np.uint8), lo, hi,
                  apertureSize=3, L2gradient=True)
    out = np.zeros(shape[:2], np.uint8)
    out[y0:y1, x0:x1] = e
    return out


def _edge_candidates(E, centre, lo, hi, side, w, gap=10, ylo=-1e9, yhi=1e9):
    """First lit pixel walking outward from the centre line, per row."""
    out = []
    for q in centre:
        y = int(round(q[1]))
        if lo < y < hi or y < 40 or y >= E.shape[0] - 2 or not (ylo <= y <= yhi):
            continue
        x = int(round(q[0])) + side * gap
        if not (2 < x < w - 3):
            continue
        # The whole outward run in one pass over the row, rather than a Python
        # step per pixel: the scan can cross most of the frame and there is one
        # per row of the centre line.
        seg = E[y, x:w - 3] if side > 0 else E[y, 3:x + 1][::-1]
        j = int(np.argmax(seg != 0))
        if seg[j]:
            out.append((float(x + side * j), float(y)))
    return np.array(out, float).reshape(-1, 2)


def _ransac_through(a, anchor, tol=3.0, iters=600, need=12, seed=0):
    """Best quadratic FORCED through `anchor`; returns the inlier mask."""
    if len(a) < need:
        return None, 0
    rng = np.random.RandomState(seed)
    q = np.asarray(anchor, float).reshape(2)
    # `choice(n, k, replace=False)` IS `permutation(n)[:k]` - same draws, same
    # stream, half the call overhead, and there are hundreds of them per line.
    idx = np.array([rng.permutation(len(a))[:2] for _ in range(iters)])
    P = a[idx]                                   # (iters, 2, 2), then the anchor
    ny = np.concatenate([P[:, :, 1], np.full((iters, 1), q[1])], axis=1)
    nx = np.concatenate([P[:, :, 0], np.full((iters, 1), q[0])], axis=1)
    C, ok = IM.interpolating_fits(ny, nx)
    bn, best = IM._best_by_inliers(C, ok, a[:, 1], a[:, 0], tol)
    return (best, bn) if bn >= need else (None, bn)


def side_edge(f, E, centre, near_svc, anchor, far_t, side, rough_max,
              tol=3.0, iters=600, need=12, seed=0, gap=10):
    """The court's side boundary, as the FIRST strong edge outward from centre.

    Four facts pin it down, none of which needs a model:

    1. The near service line ENDS on it, so `anchor` is a point known to be on
       the line - forced into every RANSAC hypothesis rather than merely fitted.
    2. Walking outward from the centre line, the court boundary is the FIRST
       strong edge met. Everything beyond it - fence posts, the far court seen
       through the glass, the surround - is outside and irrelevant. This is what
       the old row scans got wrong in the other direction: they stopped at the
       first COLOUR change, which on a worn court is a wear mark well inside.
    3. It runs more vertically than the chord from `anchor` to the far T. That
       is a projective certainty, not a heuristic: both lines start at the same
       point and end at the same depth, one at X=0 and one at X=5, so the
       sideline's far end is always outboard of the far T's.
    4. Its position is a colour step, not a gradient peak. Canny SELECTS, and
       `edge_at` then measures - taking Canny's own pixel biases the court 8-10
       cm wide at both reference venues, and re-measuring removes it.

    `centre` supplies the ROWS to search, one candidate per row, and every one of
    them lies ABOVE the near service line - so no sideline is ever measured near
    the corner it is anchored at. Two things put that region out of reach and
    neither is a bug here:

    The band `lo..hi` around the near service line is skipped, because a row
    crossing that bowed line meets its paint at a grazing angle and stops on it
    instead of on the boundary. The band is ~146 px tall, and BOTH corners fall
    inside it - at one venue the left corner sits at y=761 in a band spanning
    742..888. Below the band there is nothing to find either: the court widens
    towards the camera and the sidelines have already left the frame, so
    extending the rows down there yields exactly zero extra candidates.

    So every sideline is extrapolated from the bottom of its measured range down
    to its anchor - a median of 46 px and up to 176. The anchor pins the
    endpoint; the curve between is inferred.
    """
    if not len(centre) or anchor is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    h, w = E.shape[:2]
    lo, hi = near_svc[:, 1].min() - 15, near_svc[:, 1].max() + 15
    a = _edge_candidates(E, centre, lo, hi, side, w, gap)
    best, bn = _ransac_through(a, anchor, tol, iters, need, seed)
    if best is None:
        return np.zeros((0, 2)), a

    # --- second pass: smooth ALONG the line before looking again -----------
    # The first pass is only needed for its DIRECTION. Once a curve exists,
    # each third of it can be re-searched on an image averaged parallel to that
    # curve's local tangent, which cancels the wear marks the first pass was
    # tripping over. The tangent turns ~12 degrees end to end, enough that three
    # bands beat one angle on six sides of eight.
    cf = np.polyfit(a[best][:, 1], a[best][:, 0], 2)
    gray = f["gray"]
    # Bands span every row that CAN be searched, not the rows pass A managed to
    # find something on. Pass A is the weak pass - it is the one looking at an
    # unblurred worn court - so inheriting its extent lets it cap the strong
    # pass. Taken from its inliers the sidelines came back covering y=396..701
    # with their corner at y=761, extrapolated 60 px to reach their own anchor,
    # and the X scale drifted until the near corners projected off the top of
    # the frame with left and right swapped. Taken from its candidates, one
    # plate's right sideline still spanned only y=386..510 because that is all
    # pass A could see there at all.
    #
    # Widening further, to every row that CAN be searched, was tried and is
    # worse: each band then covers more rows, the tangent varies more within it,
    # the blur is misaligned over part of the band, and the far rows contribute
    # candidates from behind the net. Two plates lost coverage (68% -> 53%) for
    # one that gained. Pass A's candidate extent is the useful compromise.
    y0, y1 = float(a[:, 1].min()), float(a[:, 1].max())
    xin = int(np.clip(np.median(centre[:, 0]) - side * 20, 0, w))
    xout = 0 if side < 0 else w
    bx0, bx1 = (min(xin, xout), max(xin, xout))
    def tangent(y):
        return float(np.degrees(np.arctan2(1.0, -(2 * cf[0] * y + cf[1]))))

    # The angle comes from pass A's own curve, NOT from sweeping for whichever
    # angle yields the most candidates. Sweeping was tried: it rescued one plate
    # whose tangent was 14 degrees out, and on another it locked the blur onto a
    # parallel distractor - a fence line - and produced a court whose near
    # service ends missed their own detected corners by 273 px, against the 64
    # it had fixed. Maximising inlier count over an angle is only safe when
    # nothing else in the frame is straight and parallel, and here things are.
    grown = []
    for j in range(3):
        lo_b = y0 + (y1 - y0) * j / 3.0
        hi_b = y0 + (y1 - y0) * (j + 1) / 3.0
        ang = tangent(0.5 * (lo_b + hi_b))
        box = (max(0, int(lo_b) - 16), min(h, int(hi_b) + 16), bx0, bx1)
        Eb = canny_along(gray, ang, box, (h, w))
        grown.append(_edge_candidates(Eb, centre, lo, hi, side, w, gap,
                                      lo_b - 2, hi_b + 2))
    grown = [g for g in grown if len(g)]
    if grown:
        a2 = np.vstack(grown)
        b2, n2 = _ransac_through(a2, anchor, tol, iters, need, seed)
        if b2 is not None and n2 > bn:
            a, best, bn = a2, b2, n2
    k = a[best]
    cf = np.polyfit(k[:, 1], k[:, 0], 2)
    # fact 3: refuse a curve that leans out further than the chord to the far T
    if far_t is not None and abs(far_t[1] - anchor[1]) > 50:
        chord = abs((far_t[0] - anchor[0]) / (far_t[1] - anchor[1]))
        slope = abs(2 * cf[0] * anchor[1] + cf[1])
        if slope > chord:
            return np.zeros((0, 2)), a
    out = []
    for y in k[:, 1]:
        c = np.array([float(np.polyval(cf, y)), float(y)])
        q = edge_at(f, c, np.array([float(side), 0.0]), rough_max,
                    inner=30.0, outer=14.0)
        out.append(q if q is not None else c)
    return np.array(out), a


#: How far a sideline's own curve may land from the corner where the near
#: service line ends, in pixels, before the sideline is refused.
#:
#: The corner is a point KNOWN to be on that sideline, so this is the same test
#: the centre line's walk already gets: a curve fitted to what a detector
#: returned has to pass through the point it was anchored at, or the detector
#: followed something else. Every sideline detector here is *seeded* with the
#: corner - `side_edge` forces it into every RANSAC hypothesis, `track_side`
#: starts on it - and none of them is *checked* against it, which is not the
#: same thing: a quadratic through the corner has two free parameters left, and
#: two are enough to reach a parallel feature a metre outside the court.
#:
#: Measured on the TRIMMED set, because trimming is what exposes the failure.
#: On `13-09-39` the right sideline arrives as 23 points spanning 245 rows whose
#: own quadratic reaches its corner to 15 px; RANSAC then keeps the 16 that lie
#: on ONE curve, and what is left is a 92-row stub asked to speak for the 240
#: rows between it and the corner. Tested before the trim it looks fine; tested
#: after, it misses by 167 px.
#:
#: A line and a parabola both get to make the case and the smaller miss counts.
#: The parabola is the shape the lens makes, but over a short span it
#: extrapolates wildly for reasons that have nothing to do with which feature
#: was followed - one accepted plate's right sideline stops 10 rows short of its
#: corner and its parabola still misses by 43 px where a straight line misses by
#: 2.5. Refusing a measurement should take both readings agreeing.
#:
#: Over the 116-plate corpus the worst sideline on an ACCEPTED plate misses by
#: 33 px and the 98th percentile is 31, against 167 px for the one that is on
#: the wrong feature. There is nothing between 54 and 167, so the gate sits in
#: open space at both ends rather than being fitted to either.
SIDE_CORNER_MAX = 60.0

#: The courtside readers, by the key an experiment names them with. The cascade
#: below is tried in THIS order and the first one that reaches its own corner
#: wins - the order is measured, not reasoned, and the argument for it is at the
#: call site.
#:
#: `chord_lines` sits immediately before `chord` because the two read the same
#: strip and it is the more accurate of the pair - swapping which of them fills
#: that slot takes 106 accepted plates to 107 - but only just, and they fail in
#: different places, so both are kept and `SIDE_SELECTORS` decides.
SIDE_SOURCE_KEYS = ("canny", "track", "scans", "chord_lines", "chord")

#: Which readers a run may use, in the order it tries them; `None` is the
#: measured cascade above. Set it to one key to hold the sideline to a single
#: technique and see what that technique alone is worth - `PADEL_SIDE_SOURCES`
#: in the environment does the same, and is what survives a `multiprocessing`
#: worker. Nothing in the pipeline sets either, so a normal run is unaffected.
SIDE_SOURCES = None


def side_sources_wanted():
    """The reader keys this run may use, in order. Empty tuple = all of them."""
    want = SIDE_SOURCES or os.environ.get("PADEL_SIDE_SOURCES") or ""
    if not isinstance(want, str):
        want = ",".join(want)
    keys = tuple(k.strip() for k in want.replace(" ", ",").split(",") if k.strip())
    bad = [k for k in keys if k not in SIDE_SOURCE_KEYS]
    if bad:
        raise ValueError("unknown courtside source %s; know %s"
                         % (", ".join(bad), ", ".join(SIDE_SOURCE_KEYS)))
    return keys


#: Read EVERY courtside reader and keep all of their answers in the trace, under
#: `side_candidates`, instead of stopping at the first one that qualifies. Costs
#: the `track_side` call the cascade normally skips - about a second a plate -
#: and changes nothing about which answer is used. `PADEL_SIDE_RECORD_ALL` in
#: the environment does the same, and is what survives a worker process.
SIDE_RECORD_ALL = False


def side_record_all():
    """Whether to read every courtside reader rather than stop at the winner."""
    if SIDE_RECORD_ALL:
        return True
    return os.environ.get("PADEL_SIDE_RECORD_ALL", "").strip().lower() \
        in ("1", "true", "yes", "on")


#: How a courtside reader is chosen, of the three in `SIDE_SELECTORS`.
#: `PADEL_SIDE_SELECT` overrides it.
#:
#:     cascade  the old rule: first reader past the corner test wins, and each
#:              sideline is decided on its own
#:     graded   corner test as a veto, then rank on `provisional_world`
#:     fit      corner test as a veto, then solve a provisional camera per
#:              surviving reader and rank on what it makes of its own samples
#:
#: Over the 65 labelled plates, accepted within 0.30 m (0.60 m past the far
#: service line): cascade 58, graded 54, fit 60. `fit` gains two and loses none,
#: and 60 is the CEILING - best-of-four chosen by an oracle with the truth file
#: open scores 60 as well, so on this corpus it never picks a worse reader than
#: the best one available. It costs the four readers being read rather than
#: three, and one bare solve per surviving pair: 6.1 s a plate against 5.6.
#:
#: THE CORNER TEST SURVIVED THE CHANGE AS A VETO, and that is the point of the
#: arrangement. It is the only one of these measurements that tests whether the
#: right FEATURE was followed: a reader locked onto the base of the fence
#: returns a beautifully clean arc, so no measure of smoothness can see it. What
#: the corner test cannot do is rank the survivors, because among readers that
#: all reached their corner it carries almost nothing - over 191 (plate, reader)
#: pairs on the labelled corpus its rank correlation with the resulting court
#: error is +0.09, and the curve fit's own rms is +0.02. Ranking on either
#: selects WORSE than the plain cascade (41 and 42 plates against 47).
#:
#: What does rank, measured OFFLINE, is the same residual in METRES on the court
#: - +0.27, picking the better reader on 10 of the 18 plates where the readers
#: disagree by more than 0.10 m against 4 for the rms.
#:
#: IT DOES NOT SURVIVE BEING MOVED INSIDE THE DETECTOR, and the default is
#: `cascade` because of it. The offline number was computed through each
#: variant's OWN FITTED CAMERA, which does not exist yet when the reader has to
#: be chosen. The only metre map available here is `provisional_world`, built
#: from the four ends of the two service lines - and `chord_side` draws its
#: strip between exactly those same two points, so the chord's samples lie near
#: the line joining two points the map sends to X=0 by construction. Its grade
#: is small whether or not it found the boundary. The rule duly stops choosing
#: on evidence and starts choosing the chord: 82 of 130 sidelines against the
#: cascade's 1, and on the corpus it takes 58 accepted plates to 54, losing four
#: and gaining none.
#:
#: THE FAULT WAS IN THE MAP AND NOT IN THE IDEA, which `fit` then showed by
#: making the same measurement through a camera solved from the other four lines
#: and the anchors - a map that owes the chord nothing - and gaining two plates
#: where this one lost four. `graded` is kept because the pair of them is the
#: evidence for that sentence, and a knob that only ever wins teaches nobody
#: which half of an idea was the good one.
SIDE_SELECT = "fit"


#: The three ways of choosing, by the name an experiment gives them.
SIDE_SELECTORS = ("cascade", "graded", "fit")


def side_select():
    v = (os.environ.get("PADEL_SIDE_SELECT") or SIDE_SELECT or "cascade").strip()
    if v not in SIDE_SELECTORS:
        raise ValueError("PADEL_SIDE_SELECT must be one of %s, not %r"
                         % (", ".join(SIDE_SELECTORS), v))
    return v


def provisional_world(near_ends, far_ends):
    """Image -> court metres, from the four ends of the two service lines.

    A plain 4-point homography with no lens in it, so it is WRONG - by tens of
    centimetres at the far corners, which is where a fisheye bows most. It is
    not used to measure the court. It is used to convert a residual in pixels
    into a residual in metres, and for that only the SCALE GRADIENT has to be
    roughly right: a pixel at the net is worth several times the court a pixel
    at the near baseline is, and that is the whole of what a pixel residual gets
    wrong when readers are compared across the depth of the frame.

    The error it does carry is very nearly common to every reader on one plate -
    they are all looking at the same court through the same lens - and the score
    is only ever used to rank readers WITHIN one plate, so it cancels where it
    matters. Returns None when the far service line's ends are not known.
    """
    if far_ends is None or near_ends is None:
        return None
    src = np.array([near_ends[0], near_ends[1], far_ends[0], far_ends[1]], np.float32)
    dst = np.array([[0.0, CM.NEAR_SERVICE_Y], [CM.WIDTH, CM.NEAR_SERVICE_Y],
                    [0.0, CM.FAR_SERVICE_Y], [CM.WIDTH, CM.FAR_SERVICE_Y]], np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    return H if np.isfinite(H).all() else None


def side_grade(pts, H, want):
    """How far this sideline sits from the line it claims to be, in metres.

    The rms of the samples' court X about the nominal `want` (0 or 10), so it
    counts a reader that is parallel but displaced - the glass base instead of
    the paint - as well as one that is scattered. Returns None when there is no
    map to measure through, which is what puts the plate back on the cascade.
    """
    if H is None or len(pts) < 3:
        return None
    p = cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2),
                                 H).reshape(-1, 2)
    if not np.isfinite(p).all():
        return None
    return float(np.sqrt(((p[:, 0] - want) ** 2).mean()))


#: Which world X each sideline is supposed to sit on.
SIDE_WANT = {"left_sideline": 0.0, "right_sideline": CM.WIDTH}


def fit_grade(base, pair, points, shape):
    """Solve a provisional camera from these sidelines and grade what it makes.

    `base` is the four lines the sideline choice does not touch; `pair` is one
    reader's two sidelines. The camera is a bare `solve` - no re-trace, no
    rounds - and the grade is the rms of the sidelines' own samples about the
    line they claim to be, IN METRES on the court, worst side counting.

    THIS IS THE SAME MEASUREMENT `side_grade` MAKES AND A DIFFERENT MAP, and the
    map is the whole point. `provisional_world` is built from the four service
    line ends, which are also the two points `chord_side` draws its strip
    between - so the chord scores well there whether or not it found the
    boundary, and grading on it hands the chord 82 of 130 sidelines and loses
    four plates. A camera solved from six lines and the anchors owes the chord
    nothing.

    The fit is still pulled ONTO the samples being graded, which is why this
    cannot detect a sideline that is wrong in the way the fit can absorb - both
    of them displaced outward, say, which just widens the court. What it does
    see is a sideline the other four lines and the anchors CONTRADICT. Measured
    offline against each plate's own final camera the same quantity ranks the
    readers at rho +0.27 and picks the better one on 10 of the 18 plates where
    they disagree by more than 0.10 m.

    Returns None when the solve fails, which is itself information: a candidate
    pair that will not solve is not a candidate.
    """
    from .solve import solve                    # circular at module scope
    lines = list(base) + [L for _, L in sorted(pair.items())]
    try:
        cam, _rep = solve(lines, shape, points=points)
    except BaseException:                       # solve raises SystemExit
        return None
    worst = 0.0
    for name, L in pair.items():
        W = cam.backproject(L["pts"])
        if not np.isfinite(W).all():
            return None
        worst = max(worst, float(np.sqrt(((W[:, 0] - SIDE_WANT[name]) ** 2).mean())))
    return worst


def choose_sides(cand, ends, far_ends, select="graded", order=(),
                 base=(), points=(), shape=None):
    """Pick a reader for each sideline. Returns (points, how, grades).

    Under `graded`: every reader that reached its corner is graded in metres on
    both sidelines, and the reader taken is the one with the best WORST side -
    the worst side, because a court is only as good as the side that is furthest
    out, and a mean would let a perfect left excuse a right on the fence.

    A READER THAT HAS BOTH SIDELINES IS PREFERRED TO ONE THAT HAS EITHER,
    outright, before any grade is compared. Two readers on one court is two
    features being called one line, and the fit has no way to know: it is handed
    six lines and told they are the court. In practice this settles almost
    nothing - the cascade mixed readers on 6 of 65 labelled plates and the mixed
    answer tied or beat every single-reader run on 5 of them - so it is a
    tie-break with a reason, not a rule with a body count.

    Falls back to the cascade, per side, when nothing has both sides or there is
    no map to grade through.
    """
    names = ("left_sideline", "right_sideline")
    empty = np.zeros((0, 2))
    ok = {n: {k: c for k, c in cand.get(n, {}).items() if c["qualifies"]}
          for n in names}
    if select == "fit":
        pairs, grades = {}, {}
        for k in SIDE_SOURCE_KEYS:
            if not all(k in ok[n] for n in names):
                continue
            pair = {}
            for n in names:
                pts = trim_to_curve(
                    np.asarray(ok[n][k]["points"], float).reshape(-1, 2), True)
                if len(pts) < 10:
                    pair = None
                    break
                a, b = WORLD_LINES[n]
                pair[n] = {"name": n, "pts": pts, "world_a": np.array(a, float),
                           "world_b": np.array(b, float)}
            if pair is None:
                continue
            g = fit_grade(base, pair, points, shape)
            if g is not None:
                pairs[k], grades[k] = pair, {"worst_m": g}
        if pairs:
            best = min(pairs, key=lambda k: grades[k]["worst_m"])
            return ({n: np.asarray(ok[n][best]["points"], float).reshape(-1, 2)
                     for n in names},
                    {n: ok[n][best]["label"] for n in names},
                    grades)
        # Nothing solved, or no reader has both sides: the cascade still has an
        # answer and a plate with a sideline beats a plate without one.
        select = "cascade"
    H = provisional_world(ends, far_ends) if select == "graded" else None
    grades = {}
    if H is not None:
        for n in names:
            for k, c in ok[n].items():
                grades.setdefault(k, {})[n] = side_grade(
                    trim_to_curve(np.asarray(c["points"], float).reshape(-1, 2), True),
                    H, SIDE_WANT[n])
    both = [k for k in grades
            if all(grades[k].get(n) is not None for n in names)]
    if both:
        best = min(both, key=lambda k: max(grades[k][n] for n in names))
        return ({n: np.asarray(ok[n][best]["points"], float).reshape(-1, 2)
                 for n in names},
                {n: ok[n][best]["label"] for n in names},
                grades)
    # Nothing has both sides: take each side on its own, by grade where there is
    # one and by cascade order where there is not.
    pts, how = {}, {}
    for n in names:
        graded = {k: grades[k][n] for k in grades if grades[k].get(n) is not None}
        if graded:
            k = min(graded, key=graded.get)
        else:
            k = next((k for k in (order or SIDE_SOURCE_KEYS) if k in ok[n]), None)
        if k is None:
            pts[n], how[n] = empty, "REFUSED - " + "; ".join(
                c["why"] for c in cand.get(n, {}).values() if c["why"])
        else:
            pts[n] = np.asarray(ok[n][k]["points"], float).reshape(-1, 2)
            how[n] = ok[n][k]["label"]
    return pts, how, grades


def _corner_miss(pts, corner):
    """How far this set's own curve lands from a point it must pass through."""
    p = np.asarray(pts, float).reshape(-1, 2)
    out = []
    for deg in (1, 2):
        if len(np.unique(p[:, 1])) <= deg:
            continue
        try:
            cf = np.polyfit(p[:, 1], p[:, 0], deg)
        except (np.linalg.LinAlgError, ValueError):
            continue
        out.append(abs(float(np.polyval(cf, corner[1])) - float(corner[0])))
    return min(out) if out else float("inf")


def track_side(f, p0, rough_max, side=-1.0, step=6.0, first=40.0, win=22.0,
               max_miss=4, max_len=400, shape=None, tol=9.0, trace=None):
    """Follow a sideline UP-COURT from the corner where the service line ends.

    The row scans ask "walking out from the middle of the court, where does the
    court stop?", and on eight plates in this corpus the answer is a feature
    33-141px inside the real boundary, every time toward the inside, with no
    second court run beyond it to recover from. Anchoring at the endpoint asks
    a much easier question: the corner IS a point on the sideline, known to
    ~8cm, so the sideline never has to be found, only followed - and a tracker
    that only ever looks +-22px around its own prediction never sees whatever
    the row scans were stopping at.

    `side` is -1 for the left sideline and +1 for the right. The first step
    searches wide because there is no direction yet; after two points there is
    one, and after six a quadratic, which is the shape the lens makes.
    """
    p = np.asarray(p0, float).copy()
    n = np.array([side, 0.0])
    pts, miss = [p.copy()], 0
    why = "ran to the frame edge"
    while len(pts) < max_len:
        if len(pts) == 1:
            pred, half = p + np.array([0.0, -step]), first
        else:
            a = np.array(pts)
            if len(a) >= 6:
                cf = np.polyfit(a[-80:, 1], a[-80:, 0], 2)
                y = a[-1, 1] - step
                pred = np.array([np.polyval(cf, y), y])
            else:
                pred = a[-1] + step * CM.unit(a[-1] - a[0])
            half = win
        if shape is not None and not (3 < pred[0] < shape[1] - 4 and 3 < pred[1] < shape[0] - 4):
            break
        q = edge_at(f, pred, n, rough_max, inner=34.0, outer=half, tol=tol)
        if q is None:
            q = edge_at(f, pred, n, rough_max, inner=34.0 + half, outer=half, tol=tol)
        if q is None or np.linalg.norm(q - pred) > half + 6.0:
            miss += 1
            if miss > max_miss:
                why = "boundary lost"
                break
            p = pred
            continue
        miss = 0
        pts.append(q)
        p = q
    if trace is not None:
        trace["why"] = why
    return np.array(pts)


# ------------------------------------- 4b. the sideline, along its chord ----
# Both service lines are now walked to their own ends, and those ends are points
# on the SIDELINES: the near service line stops at (0, 16.95) and (10, 16.95),
# the far one at (0, 3.05) and (10, 3.05). So each sideline arrives with a chord
# already drawn along 13.9 m of it, before anything looks at a pixel out there.
#
# What that chord is good for, measured over the corpus against the fitted
# cameras, is its DIRECTION: it lies within 0.51-0.57 degrees of the reference
# sideline's own chord at the median, 1.29 at p90 and 5.0 at worst. Its POSITION
# is a bracket rather than an answer - the true sideline sits anywhere from 50 px
# outside it to 34 px inside - because the far service line's paint runs past the
# sideline before it stops, by a median 0.27-0.32 m and up to 1.36 m.
#
# A direction that good is worth more than it looks, because it licenses a blur.
# Averaging ALONG a line cancels everything that is not that line - speckle, wear
# marks, the mottling on a wet court - and does not soften the step across it,
# since no averaging happens in that direction. `canny_along` already does this
# for the sideline, but it has to bootstrap its angle from a RANSAC pass run on
# the unblurred, worn court, which is the weakest thing in the chain. Here the
# angle is geometry, known before the first look.
#
# So the band around the chord is RECTIFIED into a strip - one axis along the
# chord, one across it - and everything happens there: the blur is one separable
# pass along the strip, the boundary is a step in each row's cross-profile, and
# the curve is a low-order polynomial in strip coordinates. Measured over the
# corpus the blur is what does the work, not the rectification: turned off, the
# same detector puts 41 of 232 sidelines more than 10 px from their reference
# against 29 with it on.
#
# WHAT IT READS IN THE FAR HALF IS NOT OFFERED TO THE SOLVER, and that has now
# been tried three ways rather than assumed - see `chord_side` for why the far
# half measures worse than it looks. Adding those samples to whichever sideline
# was chosen takes the corpus from 110 plates accepted to:
#
#     109   only where the chosen sideline has no sample past the net at all
#     108   only where it has fewer than six
#     108   always
#
# Monotone in how much of it is added, every flip on paint coverage, and no
# threshold buys anything back - at 109 nothing is gained at all, one plate is
# simply lost. It is worth knowing that the far half IS thinly covered otherwise,
# which is the reason to keep wanting this: over the accepted corpus only 8% of
# the samples the solver gets for a sideline lie past the net, and 28 of 220
# sidelines have none there whatsoever. Thin and unbiased still beats thick and
# 3.4 cm outboard.
CHORD_IN, CHORD_OUT = 75.0, 65.0
#: The band, in pixels, inward and outward from the chord. Sized from the bracket
#: above (+34 inside, -50 outside) with margin, and ASYMMETRIC because the two
#: directions are not the same question. Every reader below measures "unlike the
#: court" against a stretch of profile that has to BE court, and that reference
#: is taken from the innermost `CHORD_REF` of the band - which only works if the
#: band reaches past the boundary's worst inward excursion. Reaching further
#: inward than this starts to cost: at the net the inward direction runs along
#: the net's own dark foot, and at 100 px that contaminated reference put 13 of
#: 232 sidelines past 25 px against 8 at 75.
CHORD_REF = 0.25

#: How far along the chord one strip row is, and how hard the strip is blurred
#: along itself, in rows - so the two passes average over 21*2 = 42 px and
#: 81*2 = 162 px of line.
#:
#: TWO PASSES because a hard blur along a STRAIGHT chord smears a line that is
#: bowed. The lens bows this one by 20-30 px end to end, which is a slope of
#: ~0.13 near the ends of the strip and 8 px of smear over a 61-row kernel. Pass
#: one finds the curve with a blur short enough not to care; pass two re-cuts the
#: strip AROUND that curve, so the line is straight in it by construction and the
#: blur runs exactly along the line however much the lens bent it.
CHORD_DT, CHORD_BLUR, CHORD_BLUR2 = 2.0, 21, 81

#: How far the located curve may start from the corner it is anchored at, and how
#: far it may bow away from the chord, in pixels.
#:
#: The corner is a point KNOWN to be on this sideline, so the curve's value at
#: the near end is not a free parameter - it is a measurement whose error is
#: known. Against the fitted cameras it lands within 3.2-4.4 px at the median,
#: 12.2-14.0 at p90 and 48.1 once, so 25 px admits every corner in the corpus
#: while refusing a fifth of the band.
#:
#: The bow is the lens and is not free either: the reference sideline never
#: leaves the chord by more than 50 px, so 60 covers it and still refuses the
#: curve that would have to leave the court to reach a fence. Both are search
#: BOUNDS, not tolerances - nothing is penalised for sitting near them.
CHORD_AT_CORNER, CHORD_MAX_BOW = 25.0, 60.0

#: How wide a window the per-row trace searches around the located curve, in px,
#: and how many consecutive samples have to be outside before `_find_gate` calls
#: a boundary.
#:
#: THIS WIDENED WHEN `_find_step` ARRIVED, and the two facts are the same fact.
#: Against the old absolute gate the window had to be narrow, because widening it
#: only offered the gate more chances to trip early on something the located
#: curve had already rejected - at 20 px it let 41 of 232 sidelines past 10 px
#: from their reference instead of 29. A contrast peak cannot be bought that way:
#: it is not asking whether a sample is un-court-like, it is asking where the
#: largest step in the window is, and a bigger window mostly means the real step
#: is now inside it. Swept with the find as it stands, 20 px measures best on
#: every statistic - the sideline's scatter about its own median goes 1.25 px to
#: 1.04 against a window of 14, its p90 4.4 to 4.1, its p99 16.0 to 13.4 - and it
#: is where the one side in the corpus that came back empty stops doing so.
#: Wider still buys nothing and starts to cost: at 28 px the p99 improves by
#: 0.8 px while the worst side loses 4.4.
CHORD_HALF, CHORD_NEED = 20.0, 3

#: The guard either side of a candidate boundary, and the width of the bands
#: compared across it, in pixels. Used by `_locate` to score a whole curve and by
#: `_find_step` to score one row, and deliberately the SAME two numbers in both -
#: the per-row search exists to refine the answer the whole strip voted for, and
#: a search that scored positions differently from the vote could only wander off
#: it. Nothing was tuned here: swept over guards of 1-5 px and spans of 10-22 the
#: sideline's scatter about its own median does not move (mad 1.13-1.15 px, p90
#: 4.3-4.6), because what the band widths decide is how sharply the score peaks
#: and not where.
CHORD_GUARD, CHORD_SPAN = 3.0, 15.0

# ---- `chord_lines`: edges that form a line parallel to the chord ------------
# Everything above reads a BLURRED PROFILE - it asks each row where the biggest
# step across the boundary is, having first averaged 162 px of line to make that
# step legible. `chord_lines` asks a different question of the same strip: where
# are the hard edges, which of them join up into something long and parallel to
# the chord, and which of those comes first on the way out. It never averages
# across rows at all; what stands in for that is the requirement that an edge
# belong to a run of edges - a topological substitute for a photometric one.
#
# The two are worth having together because they fail differently, and the
# numbers for that are in `chord_lines`. The constants below are its own.
CHORD_LINE_IN, CHORD_LINE_OUT = 10.0, 20.0
#: The band `chord_lines` looks in, inward and outward from the located curve,
#: in pixels. Far narrower than `CHORD_IN`/`CHORD_OUT` because this reader
#: runs SECOND, on a curve `_locate` has already placed - and it has to be, since
#: its rule is "the first qualifying edge on the way out" and every extra pixel
#: inward is another chance to meet a scuff's edge before the paint's.
#:
#: Measured against the raw chord instead of the located curve, a band this
#: narrow holds a median 93% of the true sideline but only 55% at p10, and less
#: than half of it on 15 of 232 sides - the chord tilts off the sideline because
#: the far service line's paint overruns it. On the located curve the same band
#: holds a median 96% and 100% over the near half.
CHORD_LINE_THR = 40.0
#: How steep a gradient across the chord counts as an edge, as the length of the
#: (dL, da, db) vector on `fields`' Lab.
#:
#: COLOUR, BUT NOT FOR THE REASON IT LOOKS LIKE. Read in gray this reader gets a
#: median 0.98 px from the truth sideline holding 0.90 of its rows; read in all
#: three Lab channels, 0.86 px holding 0.94, and 0.0155 m against 0.0169. But
#: most of that is L against gray and not the chroma at all - L alone already
#: gets 0.93 px at 0.92 - because Lab's L is perceptually uniform where gray is
#: a weighted sum of the sensor's channels, and a blue court is exactly where
#: those two part company. Chroma adds the rest, and cannot do the job by
#: itself: the chroma gradient alone gets 18-23 of the 130 sidelines wrong at
#: every threshold, since a worn patch is as much of a chroma step as paint is
#: and the boundary OUTSIDE the court - glass foot, kerb - has almost no chroma
#: step at all.
#:
#: THE COURT DOES NOT NOTICE, which is the honest half of this. The corpus sits
#: at the same 58 plates for `chord_lines` alone and 59 under the chooser either
#: way, with no plate flipping in either direction; the near band's median
#: improves 0.171 m to 0.166 and its p90 is a shade worse. What colour buys is
#: the reader, not the fit: better coverage, a lower median, and a WIDER
#: PLATEAU - in Lab the p90 error stays under 2 px from threshold 30 to 55,
#: where in gray it only does so from 30 to 45 - so this number is less critical
#: than it was. That, at the cost of two more Sobels on a 30 px strip, is why it
#: is kept.
#:
#: STRICTER MEASURES WORSE, ALL THE WAY DOWN, and that is the useful thing this
#: number knows. Swept against the TRUTH cameras - 130 labelled sidelines, each
#: scored by how far the fitted curve lands from where the truth camera puts the
#: paint - the sides that end up more than 3 px out go 3 at threshold 40, 7 at
#: 90, 13 at 120 and 19 at 160, and the worst side goes 11.4 px, 11.8, 441, 467.
#: Coverage falls with it, a median 0.93 of rows holding a candidate at 40
#: against 0.70 at 160. (Swept in gray, before the change above; the Lab sweep
#: has the same shape and the same optimum, which is why the number did not have
#: to move with the channels.)
#:
#: THE RUN TEST IS WHAT SEPARATES PAINT FROM SCUFFING, not the edge strength, and
#: everything above follows from that. A scuff does not fail here for being a
#: weak edge; it fails for not being 30 px of edge lying along the chord. What a
#: high threshold removes is worn PAINT - which is a weak edge too - and removing
#: it breaks the paint's own run, so the component test drops the line the
#: reader was looking for and the RANSAC below fits whatever else survived. That
#: is where the 441 px comes from, and it is why the honest direction here was
#: down and not up.
#:
#: 90 was chosen on the wrong metric. The earlier sweep found 50-90 a flat
#: optimum because it measured the sideline's scatter about its OWN median,
#: which cannot see a curve that is displaced as a whole - exactly the error a
#: threshold high enough to lock onto the wrong feature produces. Against truth
#: the optimum is 40-50 and 90 is off the end of it.
#:
#: NEITHER A WIDER KERNEL NOR A SHARPER ONE. Normalised so the threshold means
#: the same slope in gray levels per pixel for each - the raw response to a unit
#: ramp is 8 at ksize 3, 128 at 5, 2048 at 7 and 32 for Scharr, so comparing
#: kernels at one raw number compares them at wildly different strictness -
#: ksize 5 and 7 are equal or worse at every threshold and lose coverage faster
#: (0.84 and 0.78 at the shipped strictness, against 0.87), and Scharr is
#: indistinguishable from ksize 3 (p90 1.96 px against 2.00). There is nothing
#: for a bigger derivative to find: the feature is 5 cm of paint seen at a few
#: pixels across, and 3x3 already resolves it.
#:
#: AND NOT A LADDER EITHER. Grading each edge by the strictest threshold it
#: still forms a line at, and letting the highest grade beat the innermost,
#: measures worse - per-side median 1.02 px -> 1.44, p90 2.39 -> 3.41 - because
#: the hardest edge in a 30 px band is regularly the kerb or the glass foot
#: outside the court rather than the paint, and grading promotes it over the
#: paint the innermost-first rule would have taken. Dropping a level only in
#: rows where the strict map is empty keeps the ordering but still measures
#: worse (median 1.13-1.27), for the same reason as the plain sweep: the rows a
#: strict map empties are the worn ones, and they are where the answer is.
#:
#: NOT Canny either, which was the first thing tried: both thresholds at 255 is
#: too strict inside a 30 px band, and on the sides where it finds nothing to
#: align, its worst sideline is 493 px out.
CHORD_LINE_LEN, CHORD_LINE_ANG = 30.0, 15.0
#: How long a run of edges has to be to count as a line, in pixels, and how far
#: its bounding box may lean from the chord, in degrees.
#:
#: THIS IS THE STRICTNESS THAT WORKS, which is the other half of what
#: `CHORD_LINE_THR` says. Length and threshold were swept together, because a
#: lower threshold admits more junk and a longer run is what would have to
#: absorb it, and that is what happens: at threshold 40 the tail improves from
#: 20 px of run to 30 - p99 3.43 px from the truth sideline to 3.01, worst 11.4
#: to 3.3, sides more than 3 px out 3 to 2 - and then gives it back at 40 (4
#: sides) and 60 (9), where the requirement starts costing whole sidelines
#: rather than scuffs, coverage falling 0.90 to 0.84.
#:
#: End to end the two are worth about the same: on the 65 labelled plates
#: `chord_lines` alone accepts 56 at the old 90/20, 57 at 40/20, and 58 at
#: 40/30, with no plate lost at either step. Under the shipped chooser the
#: corpus goes 58 to 59.
#:
#: GAPS ARE NOT BRIDGED, and that was the surprise here. Closing runs across a
#: 4 px gap first - the obvious kindness to worn paint - measures worse on every
#: tail statistic at every threshold (p99 9.4 px against 14.2 at threshold 120),
#: because a closing that reaches over a gap in the paint also reaches from a
#: scuff to the paint and hands the alignment test one merged component that
#: passes.
CHORD_LINE_GATE = 10.0
#: How far the two readers may disagree, in pixels, before the edges are refused
#: and the blurred step is kept: the median gap between the two curves over the
#: strip.
#:
#: A guard on divergence, not a tolerance - it is set from what ordinary
#: agreement looks like rather than from what fixes anything. The two curves sit
#: a median 0.14 px apart, p90 1.29 and p99 6.15, so 10 px is past the far end of
#: normal and fires on 2 of 232 sides. What it buys is the one case where the
#: edge reader locks onto the wrong feature with 58% coverage and lands 44.7 px
#: out, where the blurred reader - which votes with the whole strip and cannot be
#: captured by one long mark - is merely 29.4 px out. With the guard the mix is
#: equal or better than the blurred reader alone on every statistic.

#: How much of the chord the trace has to survive on, as a fraction. A detector
#: reading a strip cannot walk off the end of the line - the strip stops where
#: the chord does - so the only way it says "I could not see this" is by holding
#: few rows. Measured, the one plate in the corpus where a green artificial-turf
#: surround leaves nothing for the roughness cue to catch comes back holding 15%
#: of its chord and 370 px from the truth, and no accepted sideline holds less
#: than 63%.
CHORD_MIN_SPAN = 0.45

#: How lopsided the far service walk may be about its own T before its ends are
#: refused as chord anchors: |left - right| / max, in px along the line.
#:
#: This is a guard, and it is honest about being one. The far service walk stops
#: symmetrically on every plate in this corpus - p50 0.013, p90 0.053, worst
#: 0.132 - so a gate anywhere above 0.15 admits all 116 and separates nothing.
#: It is emphatically NOT what makes the chord work: bucketed by symmetry the
#: sideline error does not move (p90 13.3 px at sym < 0.02 against 12.0 at sym >
#: 0.08) and the four worst sidelines come from walks symmetric to 0.000-0.025.
#: What it does catch is the failure it is named for - one side of the walk
#: running away down the glass base while the other stops at the paint - which
#: would tilt the chord by degrees rather than displace it by pixels, and which
#: no other test here would notice.
CHORD_SYM = 0.25


def _bump(t):
    """0 at both ends, 1 in the middle - the shape a lens bows a straight line."""
    x = (t - t[0]) / max(float(t[-1] - t[0]), 1e-9)
    return 4.0 * x * (1.0 - x)


def _outward(lim, step):
    """0, +step, -step, +2*step, ... out to +-lim: search order, not a range."""
    out = [0.0]
    v = step
    while v <= lim + 1e-9:
        out += [v, -v]
        v += step
    return out


def _strip_grid(A, B, inside, curve=None, deep=CHORD_IN, out=CHORD_OUT,
                dt=CHORD_DT, ds=1.0):
    """Sampling grid for the band along A->B, or along a curve through it.

    ORIENTED, not signed: column 0 is always the deepest court sample and the
    last column the furthest outside, whichever way round the chord runs, so
    nothing downstream has to know which sideline it is looking at.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    L = float(np.linalg.norm(B - A))
    u = (B - A) / L
    n = np.array([-u[1], u[0]]) * inside         # n always points INTO the court
    t = np.arange(0.0, L + 1e-6, dt)
    s = np.arange(deep, -out - 1e-6, -ds)        # descending: inside -> outside
    c = np.zeros_like(t) if curve is None else np.polyval(curve, t)
    P = (A[None, None, :] + t[:, None, None] * u[None, None, :]
         + (c[:, None] + s[None, :])[:, :, None] * n[None, None, :])
    return P, t, s, u, n


def _rectify(f, A, B, inside, curve=None, blur=CHORD_BLUR, deep=CHORD_IN,
             out=CHORD_OUT, dt=CHORD_DT, ref=CHORD_REF):
    """The band around the chord, resampled into a strip and blurred along it."""
    P, t, s, u, n = _strip_grid(A, B, inside, curve, deep, out, dt)
    mx = np.ascontiguousarray(P[:, :, 0].astype(np.float32))
    my = np.ascontiguousarray(P[:, :, 1].astype(np.float32))

    def cut(img):
        v = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return (v if blur < 3 else
                cv2.GaussianBlur(v, (1, int(blur) | 1), 0, borderType=cv2.BORDER_REPLICATE))

    return {"t": t, "s": s, "u": u, "n": n, "A": np.asarray(A, float),
            "k": max(6, int(ref * len(s))),
            "lab": cut(f["lab"]), "rough": cut(f["rough"])}


def _outsideness(pr, tol=9.0, mult=3.0):
    """How un-court-like each strip sample is: 1.0 is exactly on the gate.

    The same two cues `edge_at` uses, each divided by its own gate so they arrive
    on one scale, and then MAXED rather than added - because they answer the same
    question at different venues and each is blind at some of them. Outside the
    glass is a steel fence at one court and a strip of green artificial turf at
    another: the turf is as smooth as the court and the fence is nearly its
    colour, so either cue alone loses a venue and the larger of the two loses
    none. Both gates are measured against this row's own inside samples, never
    against a global constant, for the reason in `edge_at`'s docstring.
    """
    ref = np.median(pr["lab"][:, :pr["k"], 1:], axis=1)
    d = np.linalg.norm(pr["lab"][:, :, 1:] - ref[:, None, :], axis=2) / tol
    gate = mult * np.median(pr["rough"][:, :pr["k"]], axis=1) + 1.0
    return np.maximum(d, pr["rough"] / gate[:, None])


def _bander(O):
    """`band(j0, j1)` -> the mean of each row of `O` over its own [j0, j1).

    A cumulative sum, so a whole search over candidate positions costs one pass
    over the strip rather than one per candidate, and out-of-strip indices are
    clipped rather than refused - a row whose band runs off the end is a row near
    the corner, and it still has most of a band to average.
    """
    n = O.shape[1]
    C = np.concatenate([np.zeros((len(O), 1)), np.cumsum(O, axis=1)], axis=1)

    def band(j0, j1):
        return (np.take_along_axis(C, np.clip(j1, 0, n), 1)
                - np.take_along_axis(C, np.clip(j0, 0, n), 1)) / np.maximum(j1 - j0, 1)

    return band


def _locate(pr, span=CHORD_SPAN, guard=CHORD_GUARD, at_corner=CHORD_AT_CORNER,
            max_bow=CHORD_MAX_BOW, step=2.0):
    """Where the boundary runs, decided by the WHOLE strip at once.

    A two-parameter family - offset at the corner, and bow - is scored on how
    much more un-court-like the band just outside a candidate curve is than the
    band just inside it, and the score is the MEDIAN of that over every row. That
    is the point of doing it here rather than row by row: a fence, a bench or a
    doorway that fires on a fifth of the rows cannot outvote the boundary, and
    the per-row trace that follows never has to consider it, because it only ever
    looks within `CHORD_HALF` of the answer this returns.

    THE CUE IS AVERAGED AS IT COMES, not clipped or thresholded first, and that
    was measured rather than assumed. Several venues here lay a strip of green
    artificial turf between the playing surface and the fence, so a row crossing
    outward meets TWO boundaries - court to turf, then turf to fence - and a
    contrast score has an obvious incentive to prefer the second, which is much
    the bigger step. Capping the cue at twice its own gate removes that incentive
    and makes things WORSE, not better - 102 plates accepted against 107, worst
    anchor residual 52 px against 40, measured with this detector tried first,
    which is the arrangement those differences show up in. Thresholding it
    outright is worse again (p90 11.6 px -> 18.1) because a binary score is flat
    wherever the inside is clean and the outside is dirty, so the answer inside
    that plateau is whichever offset the loop happened to reach first.

    What keeps this off the fence is not the shape of the score, it is the band:
    `CHORD_AT_CORNER` and `CHORD_MAX_BOW` only ever offer it curves that a point
    known to be on the sideline could plausibly belong to.

    Returns (score, offset, bow).
    """
    O = _outsideness(pr)
    ds = abs(pr["s"][1] - pr["s"][0])
    ph = _bump(pr["t"])
    g, w = int(round(guard / ds)), int(round(span / ds))

    # One cumulative sum over the strip, then every candidate is two gathers.
    # `_bander`'s job, done for a whole row of offsets at a time: the search is a
    # couple of thousand candidates and at one median each it was the most
    # expensive thing in the sideline reader.
    ncol = O.shape[1]
    C = np.concatenate([np.zeros((len(O), 1)), np.cumsum(O, axis=1)], axis=1)
    rows = np.arange(len(O))

    def band(j0, j1):
        return (C[rows, np.clip(j1, 0, ncol)] - C[rows, np.clip(j0, 0, ncol)]) \
            / np.maximum(j1 - j0, 1)

    # Searched outward from the undisplaced, unbowed chord in both parameters, so
    # that a tie - and clipping makes ties likely - is broken toward the curve
    # that assumes least: the one running through the corner along the chord.
    # `argmax` takes the first of equal scores, which is that same order.
    offs = np.array(_outward(at_corner, step))
    best = None
    for b in _outward(max_bow, step):
        c = b * ph
        # s descends from deep inside, so the column index grows outward
        jc = np.round((pr["s"][0] - (offs[:, None] + c[None, :])) / ds).astype(int)
        sc = np.median(band(jc + g, jc + g + w) - band(jc - g - w, jc - g), axis=1)
        j = int(np.argmax(sc))
        if best is None or sc[j] > best[0]:
            best = (float(sc[j]), float(offs[j]), float(b))
    return best


def _find_gate(O, idx, ds, need=CHORD_NEED):
    """The first run of `need` consecutive samples past the row's own gate.

    An ABSOLUTE question - is this sample un-court-like yet - and the cheap right
    answer wherever the court stops being court abruptly, which is most of the
    time and is nearly all of the far half. `need` in a row, so a single noisy
    sample cannot call a boundary. -1 for a row that never trips.
    """
    hit = np.take_along_axis(O, idx, 1) > 1.0
    if hit.shape[1] < need:
        return np.full(len(O), -1)
    run = hit[:, :hit.shape[1] - need + 1].copy()
    for j in range(1, need):
        run &= hit[:, j:hit.shape[1] - need + 1 + j]
    return np.where(run.any(axis=1), np.argmax(run, axis=1), -1)


def _find_step(O, idx, ds, guard=CHORD_GUARD, span=CHORD_SPAN):
    """Where the step is largest: `_locate`'s own score, asked one row at a time.

    A RELATIVE question, and the one to ask on a worn court. Near the camera the
    surface by the sideline is scuffed, and measured over this corpus what that
    does to a row's cross-profile is not to add a mark the reader could reject -
    it is to replace the step with a RAMP. Profiles aligned on the true line and
    medianed: where the answer comes out right the outsideness sits flat at 0.42
    from 30 px inside the line to 6 px inside it and then steps to 4.3; where it
    comes out more than 15 px inside, the same profile is already at 1.34 thirty
    pixels in and climbs monotonically the whole way. There is no mark there to
    find. An absolute gate on a ramp fires wherever the ramp happens to cross it,
    which is at the foot of it, and that is the whole of the near-half error:
    `_locate` puts the curve at a median 1.7 px OUTSIDE the reference, and the
    gate trace run from it lands 1.7 px inside, turning 9 sidelines pulled inward
    by more than 5 px into 35.

    A contrast score cannot be fooled that way, because a ramp answers it with
    the same number everywhere along itself and only a step answers it with a
    peak. Swapping the find in pass one, at the search window the gate needed and
    with nothing else touched, takes the sideline's scatter about its own median
    from mad 1.27 px to 1.13, its p90 from 5.0 to 4.7, its p99 from 28.8 to 16.4
    and its worst from 92.0 px to 30.6, and leaves the far half's core where it
    was (mad 1.03 to 1.02, p90 3.4 either way). Most of the rest of the gain is
    in `CHORD_HALF`, which this is what allowed to widen.

    It is NOT what the dispersion of the strip along the line says, which was the
    first thing tried: the standard deviation of the unblurred outsideness over
    82 px of line is 0.55 at the positions this gets wrong against 0.71 at the
    ones it gets right, so a scuff is if anything the STEADIER of the two and no
    threshold on it separates them. Nor is the blur to blame for that: it is one
    separable pass along the strip's first axis, so it cannot move anything
    across the line, and a ramp 30 px wide across the line is in the photograph.

    -1 for a row whose best position is not a step at all.
    """
    band = _bander(O)
    g, w = int(round(guard / ds)), int(round(span / ds))
    sc = band(idx + g, idx + g + w) - band(idx - g - w, idx - g)
    k = np.argmax(sc, axis=1)
    return np.where(sc[np.arange(len(O)), k] > 0.0, k, -1)


def _trace_strip(pr, curve, find=_find_gate, half=CHORD_HALF, back=6, fwd=6):
    """Per row, the sub-pixel boundary, searched only near `curve`.

    Two stages, exactly as `edge_at` and `_edge_subpixel` split the job, and only
    the first of them is `find`'s business. The second re-reads the position as
    the halfway point between the profile's own level inside the step and its own
    level outside, because a gate crossing is not where the step is - it is where
    the step happened to reach a particular height, which lands early on a strong
    step and late on a weak one.

    THAT HALFWAY POINT IS ONLY EVER LOOKED FOR BEHIND THE FOUND COLUMN, and on
    most rows it is not there. `_find_gate` returns the first column past 1.0,
    and 1.0 is BELOW the halfway level between a court at 0.4 and an outside at
    4.5, so the crossing has not happened yet and the sample stays where the find
    put it. That reads like a defect and measures like a feature: letting the
    search look forward as well moves 94% of the second pass's rows, by a median
    1.1 px, and takes the sideline's scatter about its own median from mad 1.06
    px to 1.37. By the second pass the strip has been re-cut around a curve
    already on the line and blurred over 162 px, so the step height barely varies
    from row to row and the find's bias is very nearly a constant - while the
    halfway level, which depends on each row's own outside median, is not, and
    inherits the difference between a fence, a kerb and a strip of turf. In the
    first pass, where `_find_step` returns the middle of a step rather than its
    foot and the crossing could legitimately lie either side, searching forward
    as well is a wash and was dropped rather than kept as a knob: mad 1.06 px
    against 1.07, p90 4.3 against 4.5, worst 30.2 against 29.4.

    All of it on a profile averaged over 162 px of line rather than measured at
    one point on it - which is the only reason the medians either side of the
    step are stable enough to define a halfway point at all.
    """
    O = _outsideness(pr)
    ds = abs(pr["s"][1] - pr["s"][0])
    w, n = int(round(half / ds)), O.shape[1]
    lo = np.clip(np.round((pr["s"][0] - curve) / ds).astype(int) - w, 0, n - 1)
    idx = np.clip(lo[:, None] + np.arange(2 * w + 1)[None, :], 0, n - 1)
    j = find(O, idx, ds)
    ok = (j >= back) & (j <= idx.shape[1] - fwd - 1)
    if not ok.any():
        return np.zeros((0, 2))
    i = np.arange(len(O))[ok]
    jj, r = j[ok], np.arange(int(ok.sum()))
    prof, cols = np.take_along_axis(O, idx, 1)[i], idx[i]
    win = np.take_along_axis(prof, (jj[:, None] + np.arange(-back, fwd + 1)[None, :]), 1)
    inn = np.median(win[:, :back - 1], axis=1)     # its level inside the step
    out = np.median(win[:, back + 2:], axis=1)     # ...and outside it
    mid = 0.5 * (inn + out)
    # scanning outward over the window behind the found column, the first sample
    # that had already reached the halfway level - so the crossing is between it
    # and the one before. A row where none had is left at the found column.
    kk = np.arange(-back, 1)
    seg = np.take_along_axis(prof, (jj[:, None] + kk[None, :]), 1)
    above = seg >= mid[:, None]
    first = np.where(above.any(axis=1), np.argmax(above, axis=1), back)
    q = np.clip(jj + kk[first], 1, prof.shape[1] - 1)
    a, b = prof[r, q - 1], prof[r, q]
    fr = np.where(np.abs(b - a) > 1e-9, (mid - a) / (b - a), 0.0)
    s0, s1 = pr["s"][cols[r, q - 1]], pr["s"][cols[r, q]]
    # a step too shallow to have a halfway point is left where `find` put it
    weak = (out - inn) < 0.5
    s_out = np.where(weak, pr["s"][cols[r, jj]],
                     s0 + np.clip(fr, 0, 1) * (s1 - s0))
    return np.column_stack([pr["t"][ok], s_out])


def _line_candidates(f, A, B, inside, curve, thr=CHORD_LINE_THR,
                     min_len=CHORD_LINE_LEN, ang=CHORD_LINE_ANG,
                     deep=CHORD_LINE_IN, out=CHORD_LINE_OUT):
    """The first hard edge out that belongs to a line running along the chord.

    Four steps, and the rectification is what makes three of them trivial.

    1. EDGES ACROSS THE CHORD ONLY. The strip's second axis IS the perpendicular
       to the chord, so one Sobel along it is a directional edge detector aimed
       at the boundary and blind to everything running across it - a tramline of
       scuffing left by a player's run-up scores nothing here unless it happens
       to lie along the sideline. Non-maximum suppression across the same axis
       leaves one column per step rather than a fat ridge.

       IN ALL THREE LAB CHANNELS, not in gray, and the difference is not the
       chroma - see `CHORD_LINE_THR`. Taken as the length of the vector
       (dL, da, db) this is the ordinary multi-channel directional derivative,
       and it is on the same scale as the gray one it replaced, so `thr` did not
       have to move.
    2. EDGES THAT FORM A LINE. Connected components of that map; a component is
       a line if it spans at least `min_len` rows - rows are 1 px of chord here,
       not the 2 px the rest of this section reads at - and if its bounding box
       leans less than `ang` from the chord, which in the strip means a box much
       taller than it is wide. This is the only thing standing in for averaging
       along the line, and it is a topological substitute rather than a
       photometric one: an edge earns its place by having neighbours, not by
       being strong.
    3. THE FIRST ONE ON THE WAY OUT, per row. `_strip_grid` orders the columns
       from deep inside to far outside, so that is the first surviving column.
    4. is `chord_lines`, which puts the quadratic through what is left.

    The position is read to sub-pixel off the gradient itself, by the parabola
    through the suppressed peak and its two neighbours - the same treatment
    `IM._peak_ridge` gives a paint ridge. It is worth little (mad 1.37 px against
    1.38 without it), which says the scatter left here is not quantisation but
    rows locking onto genuinely different features - the paint's outer edge on
    some, the kerb beyond it on others.
    """
    P, t, s, u, n = _strip_grid(A, B, inside, curve=curve, deep=deep, out=out,
                                dt=1.0, ds=1.0)
    c = cv2.remap(f["lab"],
                  np.ascontiguousarray(P[:, :, 0].astype(np.float32)),
                  np.ascontiguousarray(P[:, :, 1].astype(np.float32)),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # `Sobel` runs per channel, so this is one derivative across the chord in
    # each of L, a and b; the norm makes them one edge map again.
    a = np.linalg.norm(cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3), axis=2)
    pad = np.pad(a, ((0, 0), (1, 1)), mode="edge")
    M = ((a >= thr) & (a >= pad[:, :-2]) & (a >= pad[:, 2:])).astype(np.uint8)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(M, 8)
    h = stats[:, cv2.CC_STAT_HEIGHT].astype(float)   # its extent ALONG the chord
    w = stats[:, cv2.CC_STAT_WIDTH].astype(float)    # ...and across it
    keep = (h >= min_len) & ((w - 1) <= np.tan(np.radians(ang)) * (h - 1))
    keep[0] = False                                  # the background component
    S = keep[lab]
    has = S.any(axis=1)
    if has.sum() < 12:
        return np.zeros((0, 2))
    r = np.arange(len(t))
    j = np.clip(np.argmax(S, axis=1), 1, a.shape[1] - 2)
    y0, y1, y2 = a[r, j - 1], a[r, j], a[r, j + 1]
    den = y0 - 2.0 * y1 + y2
    fr = np.zeros(len(r), np.float32)
    np.divide(0.5 * (y0 - y2), den, out=fr, where=np.abs(den) > 1e-9)
    ds = s[1] - s[0]
    return np.column_stack([t[has], (s[j] + np.clip(fr, -1, 1) * ds)[has]])


def _ransac_strip(ts, deg=2, tol=3.0, iters=300, need=12, seed=0):
    """Best low-order s(t) through the strip candidates; RANSAC, no anchor.

    No anchor is forced here, unlike `_ransac_through`, because the anchor is
    already spent: `_locate` searched only curves that start within
    `CHORD_AT_CORNER` of the corner, so every candidate this sees was found
    inside a window the corner defined.
    """
    if len(ts) < need:
        return None, 0
    rng = np.random.RandomState(seed)
    t0, t1 = ts[:, 0], ts[:, 1]
    best, bn = None, 0
    for _ in range(iters):
        i = rng.permutation(len(ts))[:deg + 1]   # `choice(..., replace=False)`
        if not IM.distinct_at_least(ts[i, 0], deg + 1):
            continue
        try:
            cf = IM.polyfit_fast(ts[i, 0], ts[i, 1], deg)
        except (np.linalg.LinAlgError, ValueError):
            continue
        # Horner, in the order `polyval` evaluates it, without its wrapper
        inl = np.abs(t1 - ((cf[0] * t0 + cf[1]) * t0 + cf[2])) <= tol
        if inl.sum() > bn:
            best, bn = inl, int(inl.sum())
    if best is None or bn < need:
        return None, 0
    return IM.polyfit_fast(ts[best, 0], ts[best, 1], deg), bn


def _read_step(f, A, B, inside, pr, loc, tol):
    """Pass one for `chord_side`: the biggest step in each blurred cross-profile.

    `_locate` has already chosen the curve; this refines it row by row against a
    profile averaged over 42 px of line, taking the position where the band just
    outside is most unlike the band just inside. See `_find_step`.
    """
    cf, _ = _ransac_strip(_trace_strip(pr, np.polyval(loc, pr["t"]),
                                       find=_find_step), tol=tol)
    return cf


def _read_lines(f, A, B, inside, pr, loc, tol):
    """Pass one for `chord_lines`: the first aligned edge out of each row.

    Never averages across rows at all - what stands in for that is `min_len`
    edges in a row along the chord. See `_line_candidates`, which does the
    looking; this only puts the quadratic through it and returns the answer in
    the same strip coordinates `_read_step` uses, so the two are comparable.
    """
    cf, _ = _ransac_strip(_line_candidates(f, A, B, inside, loc), tol=tol)
    return None if cf is None else np.polyadd(loc, cf)


def _chord_read(f, T, corner, far_end, first, y_min=None, pitch=3, tol=3.0,
                trace=None):
    """The sideline, read out of a strip rectified along its own chord.

    Everything the two chord readers share: cutting the strip, `_locate`'s vote
    on where the boundary runs, the second pass that re-cuts the strip around
    whatever pass one found, and the rules about what is worth reporting. `first`
    is the only difference between them - `_read_step` or `_read_lines` - and it
    is handed the rectified strip and the located curve and returns a curve in
    strip coordinates, or None.

    `corner` is where the near service line's paint stopped and `far_end` where
    the far service line's did, on the same side - see this section's header for
    what that chord is and is not good for. Returns the boundary samples in image
    pixels, or an empty array when the strip could not hold a curve.

    THE WHOLE CHORD IS READ AND ONLY THE NEAR HALF IS REPORTED (`y_min`, the net's
    floor line). Those are two different uses of it and only one of them survives
    contact with a padel court. Direction and position - the blur's angle, and
    the band `_locate` votes inside - want every pixel of line they can get, and
    the far half is half the line. As MEASUREMENTS the far half's samples are the
    same feature `RETRACE_SPAN` cuts out of every sideline re-trace, for the
    reason given there and in `CourtModel.OCCLUDED_BY_DEFAULT`: seen from behind
    the near baseline the floor up-court is almost edge-on and the glass frame
    hides the strip of surface in front of it. (Not the same SEGMENT, though -
    `OCCLUDED_BY_DEFAULT` holds out Y in [0, 3.05], beyond where this chord even
    reaches. The chord's far half is part of `left_sideline`, which is fitted.)

    IN PIXELS THE FAR HALF LOOKS LIKE THE BETTER HALF, and that is an illusion of
    scale worth spelling out, because it is the reason this cut looks wrong every
    time someone checks it on a debug picture. Against the fitted cameras the far
    half scatters less in pixels - p90 3.1 px against the near half's 4.9. Back-
    projected to METRES, where the fit actually lives, it is twice the worse half
    on both counts: its median offset from the nominal line runs to 0.147 m at
    p90 against 0.061, and its spread WITHIN one sideline to 0.052 m against
    0.023. A pixel at the net is simply worth several times the court a pixel at
    the near baseline is. And the offset is not noise but a bias with a sign: the
    near half sits a median 0.021 m inside the nominal line and the far half
    0.034 m OUTSIDE it, the two disagreeing by a median 0.064 m.

    Outside is the expensive direction, because it widens the fitted court, which
    is what paint coverage measures. Handed the far half as well, on every
    sideline, the corpus goes from 110 plates accepted to 108 - three lost and one
    gained, and all four flips are coverage, at 79-84% against the gate's 85%.
    Handing it over only where the chosen sideline has nothing up there at all is
    no better; see the sweep in this section's header.

    `pitch` keeps every nth traced row. The strip is read at 2 px along the line
    because `_locate` and the RANSAC want the statistics, and the SOLVER does not:
    it weights every sample equally, so handing it 450 sideline points against
    the near service line's 445 would quietly make the two sidelines half the
    fit. Every third sample is a 6 px pitch, which is what `track_side` returns.
    """
    A, B = np.asarray(corner, float), np.asarray(far_end, float)
    L = float(np.linalg.norm(B - A))
    if L < 100:
        return np.zeros((0, 2))
    u = (B - A) / L
    inside = float(np.sign(np.dot(np.asarray(T, float) - A, np.array([-u[1], u[0]]))))
    if inside == 0:
        return np.zeros((0, 2))

    pr = _rectify(f, A, B, inside)
    score, s0, bow = _locate(pr)
    # `_bump` is exactly quadratic, so refitting the located curve as a
    # polynomial in t is a change of form and not of value. Both readers want it
    # in that form: one to sample it per row, the other to cut a strip round it.
    loc = np.polyfit(pr["t"], s0 + bow * _bump(pr["t"]), 2)
    cf = first(f, A, B, inside, pr, loc, tol)
    if trace is not None:
        trace.update(score=score, s0=s0, bow=bow, length=L)
    if cf is None:
        return np.zeros((0, 2))
    # Pass two: the strip re-cut around the curve pass one found, so the line is
    # straight in it and the hard blur runs exactly along the line.
    #
    # AND BACK TO THE GATE HERE, which is not an oversight. `_find_step` earns
    # its keep in pass one, where the curve it starts from can be 14 px out and
    # the ramp is between the two; by pass two the strip is cut around a curve
    # already on the line and is only 26 px deep either way, so the bands the
    # step score needs no longer fit inside it and the ramp is mostly outside it.
    # Measured over the corpus, using the step score in pass two as well takes
    # the sideline's scatter about its own median from mad 1.07 px to 1.54 and
    # its p90 from 4.5 to 5.8, and costs the far half - where the gate is simply
    # right, the surface there being seen edge-on against a glass frame rather
    # than worn - mad 0.97 px to 1.39, p90 3.1 to 3.9 and worst 15.6 to 23.0.
    pr2 = _rectify(f, A, B, inside, curve=cf, blur=CHORD_BLUR2, deep=26.0, out=26.0)
    ts = _trace_strip(pr2, np.zeros_like(pr2["t"]))
    cf2, n2 = _ransac_strip(ts, tol=tol)
    if cf2 is None:
        return np.zeros((0, 2))
    keep = ts[np.abs(ts[:, 1] - np.polyval(cf2, ts[:, 0])) <= tol]
    if len(keep) < 12:
        return np.zeros((0, 2))
    span = float(keep[:, 0].max() - keep[:, 0].min()) / L
    if trace is not None:
        trace.update(n2=n2, span=span, n_kept=len(keep))
    if span < CHORD_MIN_SPAN:
        return np.zeros((0, 2))
    t = np.sort(keep[:, 0])[::max(1, int(pitch))]
    s = np.polyval(cf, t) + np.polyval(cf2, t)
    out = pr2["A"] + t[:, None] * pr2["u"] + s[:, None] * pr2["n"]
    if y_min is not None:
        out = out[out[:, 1] >= y_min]
        if trace is not None:
            trace["n_near_half"] = len(out)
    return out


def chord_side(f, T, corner, far_end, y_min=None, pitch=3, tol=3.0, trace=None):
    """The `chord` reader: the sideline as a step in a blurred cross-profile.

    Averages 162 px of line into each profile before asking where the boundary
    is, which is what makes a worn court legible at all - and is also its
    weakness, since a step is all it can see and a worn near court replaces the
    step with a 30 px ramp. See `_read_step` and `_find_step`.
    """
    return _chord_read(f, T, corner, far_end, _read_step, y_min, pitch, tol, trace)


def chord_lines(f, T, corner, far_end, y_min=None, pitch=3, tol=3.0, trace=None):
    """The `chord_lines` reader: the sideline as the first aligned edge out.

    The same strip and the same located curve as `chord_side`, read the opposite
    way round: hard edges taken across the chord, kept only where they belong to
    a run at least `CHORD_LINE_LEN` long lying within `CHORD_LINE_ANG` of the
    chord, and then the innermost survivor in each row. Where `chord_side`
    averages along the line and then looks for a step, this looks for edges and
    then asks which of them have neighbours - a topological substitute for the
    averaging rather than a photometric one.

    THE TWO FAIL DIFFERENTLY, which is the whole reason to keep both. Where they
    part it is usually this one that is right: held to one reader each over the
    65 labelled plates, `chord` accepts 55 courts inside 0.30 m and this accepts
    58, and its near band is the tighter of the two (worst visible point a median
    0.171 m against 0.176, p90 0.293 against 0.325).

    (The paragraph above is measured at the constants this file now carries. The
    scatter figures that used to stand here - mad 1.25 px against 1.00, and a
    106-to-107 corpus swap - were taken at `CHORD_LINE_THR` 90 and
    `CHORD_LINE_LEN` 20, and are not restated because the sweep that replaced
    those two showed the metric behind them, scatter about the reader's own
    median, cannot see the error that matters. See `CHORD_LINE_THR`.)

    It is NOT strictly better, which is why both are readers rather than one
    being a fallback inside the other: the sidelines it gets wrong it gets wrong
    on its own, by finding a long mark that is not the paint, where `chord_side`
    votes with the whole strip and cannot be captured that way. Both are read on
    every plate and `SIDE_SELECTORS` decides.
    """
    return _chord_read(f, T, corner, far_end, _read_lines, y_min, pitch, tol, trace)


def chord_ends(walked, far_t, sym=CHORD_SYM):
    """Where the far service line's paint stopped, left and right, or None.

    None unless the walk got there symmetrically - see `CHORD_SYM` - and unless
    there is enough of it to say where its own ends are: 40 samples is the same
    bar `pin_far_service` sets before it will hold the refinement to this walk.

    Trimmed to its own curve first: an endpoint read straight off a walk is
    whatever its outermost sample happened to be, and that can be a sample the
    line's own quadratic rejects.

    The near service corners are pointedly NOT treated this way - see `detect`,
    where the same change measures out backwards - and the difference between
    the two cases is how much a quadratic has to explain. That one spans both
    arms of a 900 px line and a fisheye bows it past what a quadratic follows to
    3 px, so RANSAC clips its ends and scatters the corner by 0.8 m. This one is
    held near-horizontal by `FAR_TILT`, is a third as long, and its trim drops a
    median of ONE sample against the near line's sixteen. So here the outlier
    that gets dropped really is an outlier: the outermost sample is one on 34 of
    116 left ends and 19 of 116 right ones, and dropping it moves the anchor by a
    p90 of 12.5 px and 5.3 px.

    Neutral on the corpus - 110 accepted either way, every quality figure equal
    to three decimals - because the chord is the last source tried and wins on
    two sidelines. It is kept for what it does to those two, and to the anchor
    the chord hands the strip on every other plate, rather than for a number in
    the index.
    """
    w = np.asarray(walked, float).reshape(-1, 2)
    if far_t is None or len(w) < 40:
        return None
    w = trim_to_curve(w, False, tol=3.0)
    if len(w) < 40:
        return None
    left, right = float(far_t[0] - w[:, 0].min()), float(w[:, 0].max() - far_t[0])
    if min(left, right) <= 0 or abs(left - right) / max(left, right) > sym:
        return None
    cf = np.polyfit(w[:, 0], w[:, 1], 2)
    return tuple(np.array([x, float(np.polyval(cf, x))])
                 for x in (w[:, 0].min(), w[:, 0].max()))


def _ridge_candidates(f, ys, resp, frac=0.4, floor=10.0):
    """Every local peak of the paint response inside one column's far half."""
    out = []
    for i in range(1, len(resp) - 1):
        if resp[i] >= resp[i - 1] and resp[i] > resp[i + 1] and resp[i] >= floor:
            out.append((float(ys[i]), float(resp[i])))
    if not out:
        return out
    top = max(r for _, r in out)
    return [(y, r) for y, r in out if r >= frac * top]


def pick_far_service(f, cands, far_t, over=(-10.0, 45.0), win0=22.0, win=11.0, need=10):
    """Choose the far service line by DEPTH, anchored on the centre line's end.

    The centre line runs from the near service T to the far service T, so where
    it stops IS the far service line's depth - measured against click-seeded
    models its far end back-projects to (5.00, 2.75) and (5.02, 2.95) against a
    true (5.00, 3.05), overshooting up-court by 0.1-0.3 m because the tracker
    coasts a few pixels past the junction.

    Taking the BRIGHTEST ridge in the far half instead is what put this line on
    the net's top tape and its sponsor logos: at two plates here the brightest
    candidate sat 90 px and 230 px below the centre line's end, and the fitted
    "line" scattered over 40-76 px. Brightness is the wrong question when a
    brighter parallel feature is 20 px away. Depth is not.

    Columns are visited outward from the anchor, so the prediction is always an
    extrapolation of a curve already fitted to nearer columns - the same growth
    the sideline tracker uses.
    """
    if far_t is None or not cands:
        return None
    order = sorted(range(len(cands)), key=lambda i: abs(cands[i][0] - far_t[0]))
    pts = []
    for k in order:
        x, ys, resp, span = cands[k]
        if len(pts) >= 6:
            a = np.array(pts)
            cf = np.polyfit(a[:, 0], a[:, 1], min(2, len(a) - 1))
            pred, lo, hi = float(np.polyval(cf, x)), -win, win
        elif len(pts) >= 3:
            a = np.array(pts)
            cf = np.polyfit(a[:, 0], a[:, 1], 1)
            pred, lo, hi = float(np.polyval(cf, x)), -win0, win0
        else:
            # ASYMMETRIC while anchoring. The centre line is followed to where
            # its paint stops, and it always coasts a little PAST the junction,
            # never short of it - 5 px at one venue, 24-36 px at another. So the
            # far service line is always a bit BELOW the far edge in image
            # terms, and a window centred on that edge misses it outright.
            pred, lo, hi = float(far_t[1]), over[0], over[1]
        best = None
        for y, r in _ridge_candidates(f, ys, resp):
            d = y - pred
            if lo <= d <= hi and (best is None or abs(d) < best[0]):
                best = (abs(d), y)
        if best is None:
            continue
        q = IM._peak_ridge(f["horiz"], np.array([x, best[1]]), np.array([0.0, 1.0]), 3.0, 8.0)
        pts.append((x, float(q[1]) if q is not None else best[1]))
    if len(pts) < need:
        return None
    return sorted(pts)


def probe_far_service(f, far_t, x_lo, x_hi, step=20, below=8.0, skip=40.0,
                      win0=26.0, win=12.0, need=20, max_rms=4.0):
    """Find the far service line by probing straight below the centre line's end.

    The bounded search in `pick_far_service` is stronger where it applies,
    because the court's own far half brackets it. But it can only see what the
    column scans bracketed, and on two plates here those scans put the "far
    half" inside the net's mesh - so the real far service line, 70 px higher,
    was invisible to them however good the anchor was.

    Probing does not need the runs at all: it asks the far edge of the centre
    line where the line is and looks there. Its own evidence is the answer -
    55 and 56 of 60 columns landing on one curve on exactly those two plates.
    Accepted only on that evidence, never as a guess.
    """
    if far_t is None:
        return None
    xs = [x for x in range(int(x_lo), int(x_hi) + 1, step) if abs(x - far_t[0]) >= skip]
    pts = []
    for x in sorted(xs, key=lambda v: abs(v - far_t[0])):
        if len(pts) >= 6:
            a = np.array(pts)
            pred = float(np.polyval(np.polyfit(a[:, 0], a[:, 1], 2), x))
            half = win
        elif len(pts) >= 3:
            a = np.array(pts)
            pred = float(np.polyval(np.polyfit(a[:, 0], a[:, 1], 1), x))
            half = win0
        else:
            pred, half = float(far_t[1]) + below, win0
        q = IM._peak_ridge(f["horiz"], np.array([float(x), pred]), np.array([0.0, 1.0]),
                           half, 8.0)
        if q is None or abs(q[1] - pred) > half:
            continue
        pts.append((float(x), float(q[1])))
    if len(pts) < need:
        return None
    a = np.array(sorted(pts))
    keep = trim_to_curve(a, False, tol=3.0)
    if len(keep) < need:
        return None
    rms = float(np.sqrt(((keep[:, 1] - np.polyval(
        np.polyfit(keep[:, 0], keep[:, 1], 2), keep[:, 0])) ** 2).mean()))
    if rms > max_rms:
        return None
    return [tuple(q) for q in keep]


#: What makes a WALKED far service line good enough to hold the refinement to.
#: `REACH` is how far the walk must have got on BOTH sides of the far T, in
#: pixels; `RMS` is how well those samples must lie on one quadratic; `KEEP` is
#: how much of the walk has to survive the trim onto that quadratic.
#:
#: Reach either side is the load-bearing one, and it is a statement about
#: leverage rather than about quantity. A line correspondence constrains the
#: model perpendicular to itself, so what a short stub cannot fix is its own
#: ANGLE - and the refinement's failure mode here is exactly a rotation, the far
#: service line pivoting away onto the net's top tape. 100px either side of a
#: point that is itself known to 2.9 px worst case pins that angle to under two
#: degrees; a stub on one side of the T pins nothing.
PIN_REACH, PIN_RMS, PIN_KEEP = 100.0, 1.5, 0.8

#: How far the refinement may move the line away from each kind of measurement,
#: in pixels, and the fewest detected points worth leashing at all.
#:
#: Two tolerances because there are two qualities of evidence, and measured over
#: 104 plates they are wildly different things. Where the walk earns a pin (81
#: plates, median 150 samples) the re-trace lands 0.11 px away, worst 0.22 - the
#: strong gate is holding a door nothing pushes on. Where it does not (24
#: plates, median 24 samples, all of them a thin far half seen from a low
#: camera) the re-trace slides a median 0.14 px, p90 3.6, and on one plate
#: 31.7 px - onto the net's top tape, which is what the loose gate exists to
#: stop. The corpus has a clean gap there: next worst is 5.0 px, then 4.8.
PIN_TOL_WALK, PIN_TOL_SCAN, PIN_MIN_PTS = 5.0, 8.0, 12


def pin_far_service(walked, far_t, detected):
    """How far the refinement may move the far service line, and from what.

    Returns the quadratic coefficients and x range of the line the refinement is
    held to (see `CourtBatch._finish`), with the tolerance that measurement has
    earned, or None if there is not enough of a line to hold it to at all.

    The test is deliberately about the MEASUREMENT and not about the fit it will
    go into: it asks how well this line was seen, which is knowable here, rather
    than whether the answer looks right, which is not.

    The weak case matters more than the strong one. `retrace` re-seeds from the
    current model with a 12 px acquisition window, which is a good deal when the
    detection was blind and a bad one when it was topological: the scans bracket
    this line between the net and the far baseline by cutting a column into
    court / gap / court, and a window seeded from a model that is a few pixels
    out has no such protection. Where the far half is thin, the tape wins, the
    next round is seeded from a model that agrees with the tape, and the line
    ends up on the net. Holding a 14-point bracketed detection to 8 px is
    trusting topology over photometry, which is the whole basis of this module.
    """
    a = None
    if far_t is not None and len(walked) >= 40:
        w = np.asarray(walked, float)
        keep = trim_to_curve(w, False, tol=3.0)
        if len(keep) >= max(40, PIN_KEEP * len(w)):
            left = float(far_t[0] - keep[:, 0].min())
            right = float(keep[:, 0].max() - far_t[0])
            cf = np.polyfit(keep[:, 0], keep[:, 1], 2)
            rms = float(np.sqrt(((keep[:, 1] - np.polyval(cf, keep[:, 0])) ** 2).mean()))
            if min(left, right) >= PIN_REACH and rms <= PIN_RMS:
                a = (keep, cf, rms, PIN_TOL_WALK, [left, right])
    if a is None:
        d = np.asarray(detected, float).reshape(-1, 2)
        if len(d) < PIN_MIN_PTS:
            return None
        cf = np.polyfit(d[:, 0], d[:, 1], 2 if len(d) >= 8 else 1)
        rms = float(np.sqrt(((d[:, 1] - np.polyval(cf, d[:, 0])) ** 2).mean()))
        a = (d, cf, rms, PIN_TOL_SCAN, None)
    keep, cf, rms, tol, reach = a
    return {"coeffs": [float(v) for v in cf], "rms": rms, "tol": tol,
            "x_lo": float(keep[:, 0].min()), "x_hi": float(keep[:, 0].max()),
            "reach": reach, "n": int(len(keep)), "walked": reach is not None}


def _close(runs, a, b, min_run):
    """Accept a finished run only if it is long enough for the slot it lands in.

    Discarding a short run rather than recording it is what keeps the ordering
    meaningful: a 10px flicker in a dark frame corner must not become "run 1"
    and push the near half into the far half's slot.
    """
    if b - a >= min_run:
        runs.append((a, b))


def _edge_subpixel(lab, i, back=6, fwd=6):
    """Where the colour actually changed, between sample i-1 and i.

    Interpolated on the local chroma step - the midpoint between the median
    inside and the median outside - so it is a real boundary position rather
    than the first integer sample that tripped a threshold.
    """
    a, b = max(0, i - back), min(len(lab), i + fwd)
    if b - a < 4 or i - a < 2 or b - i < 2:
        return float(i)
    inside = np.median(lab[a:i, 1:], axis=0)
    prof = np.linalg.norm(lab[a:b, 1:] - inside, axis=1)
    prof = np.convolve(prof, np.ones(3) / 3.0, mode="same")
    lo = float(np.median(prof[:max(2, i - a - 1)]))
    hi = float(np.median(prof[i - a + 1:]))
    if hi - lo < 3.0:
        return float(i)
    mid = 0.5 * (lo + hi)
    for j in range(max(1, i - a - 3), len(prof)):
        if prof[j] >= mid > prof[j - 1]:
            t = (mid - prof[j - 1]) / (prof[j] - prof[j - 1])
            return a + j - 1 + t
    return float(i)


# ----------------------------------------------------------- 5. detector ----
def detect(bgr, verbose=True, debug=None, with_ctx=False, col_step=22,
           row_step=6, trace=None):
    """Return [{name, pts, world_a, world_b}] - the same six lines, walked.

    `trace` is an optional dict filled in AS THE WALK PROCEEDS rather than
    returned at the end, so a caller still holds everything found up to the
    point of failure when this raises. That is the only way to see a failure:
    the interesting cases are exactly the ones that never reach a return.
    """
    h, w = bgr.shape[:2]
    tr = trace if trace is not None else {}
    tr["shape"] = bgr.shape
    f = fields(bgr)
    mu, Sinv, box = IM.court_colour(bgr)
    # How tightly this image's court colour clusters. Needed before the walks
    # now, not just by the run scans below, because the centre line's walk uses
    # it to recognise bare court beyond the end of the paint.
    gate_ab = max(9.0, 5.0 * float(np.sqrt(np.trace(np.linalg.inv(Sinv)) / 2)))
    T = find_T(f, bgr.shape, verbose=verbose)
    tr["T"] = T

    # How rough a patch of THIS court's bare floor is, measured beside the
    # anchor. Used by the run scans below, and by the centre line's walk, which
    # needs it to know when it is off the net - so it is measured here, before
    # any walk, rather than after them.
    band = f["rough"][int(T[1]) - 90:int(T[1]) - 20, int(T[0]) - 220:int(T[0]) - 40]
    rough_max = float(np.median(band)) * 3.0 + 1.5

    # --- the two lines that meet at the anchor ------------------------------
    # Both ends of this line stop against the glass, with a reflection and a
    # sponsor banner beyond them for the follower to run on to, and nothing on
    # the line itself to coast over - so it is walked on a tighter brightness
    # gate than any other. See `NEAR_SVC_DROP`.
    # WHY EACH ARM STOPPED IS RECORDED, and it is not decoration: these two
    # endpoints are the court's X scale and they are anchored unconditionally,
    # where the far T is dropped unless its walk says it reached the end of the
    # paint. `NEAR_SVC_DROP` measured the population this cannot fix - the walks
    # that end at "ridge lost" or against the frame edge with the paint still
    # fading, which are eight of the ten worst endpoints in the corpus - and
    # nothing downstream can currently tell them from a walk that stopped where
    # the paint did. Kept in the trace so that it can.
    l_tr, r_tr = {}, {}
    right = walk(f["horiz"], T, (1, 0), shape=bgr.shape, drop=NEAR_SVC_DROP,
                 trace=r_tr)
    left = walk(f["horiz"], T, (-1, 0), shape=bgr.shape, drop=NEAR_SVC_DROP,
                trace=l_tr)
    tr["near_why"] = {"left": l_tr.get("why", ""), "right": r_tr.get("why", "")}
    near_svc = np.vstack([left[::-1], [T], right]) if len(left) or len(right) else np.zeros((0, 2))
    # The centre line gets the end-of-line cue, because its far end is not just an
    # endpoint - it IS the far service T, the anchor everything at the far end
    # of the court is measured from, and it is now a constraint on the fit.
    #
    # The obvious companion change, forbidding missed steps outright on the
    # grounds that this line is continuous, was measured and is wrong. To the
    # ridge detector it is not continuous: over 49 plates the genuine line needs
    # 661 gaps of one step and 82 of two, every plate needs at least one, and
    # the first two steps out of the anchor miss on most of them because the two
    # ridges are still one blob there. `max_miss=2` truncated a dozen plates
    # half way up the far half - world Y 6.7 against a T at 3.05 - and returned
    # an empty walk on two. What actually lets the walk escape is not the gap,
    # it is being allowed to resume ANYWHERE after one; that is fixed in `walk`
    # by the cue above, which fires two steps BEFORE the gap even opens.
    # `edges` is what tells this walk it has crossed the net - see `walk`. Only
    # this walk needs it: the two service lines are horizontal themselves and
    # cross nothing on their way.
    ctr_tr = {}
    edges = white_edges(bgr)
    up = walk(f["vert"], T, (0, -1), shape=bgr.shape, lab=f["lab"],
              edges=edges, rough=f["rough"],
              rough_max=rough_max, trace=ctr_tr)
    centre = np.vstack([[T], up]) if len(up) else np.zeros((0, 2))
    tr["centre_why"] = ctr_tr.get("why", "")
    tr["centre_lines"] = ctr_tr.get("lines", [])
    # ...AND THEN SLID ONTO THE MIDDLE OF THEIR OWN PAINT. Both of these lines
    # are wide enough here for the ridge follower to have latched onto one side
    # of them and stayed there - see `centre_on_paint`. Done before the junction
    # is measured, so that `_junction_from_walks` intersects the two lines'
    # middles rather than two arbitrary chords of them, and before the endpoints
    # are read off, so that the corners the sidelines hang from are on the paint's
    # centre line too.
    #
    # Only these two. The far service line's paint is 1-2 px wide at that
    # distance and has no middle to find; the sidelines and the net line are not
    # paint at all - they are a colour step at the foot of the glass and the
    # boundary of the net's shadow - so a white stripe's two edges is not what
    # either of them is made of.
    i_T = len(left)                              # where T sits in `near_svc`
    ctr_mid, svc_mid = {}, {}
    near_svc = centre_on_paint(edges, near_svc, False, trace=svc_mid)
    centre = centre_on_paint(edges, centre, True, trace=ctr_mid)
    tr["centred"] = {"near_service": svc_mid, "centre_line": ctr_mid}
    # BOTH ENDS OF THIS LINE ARE CORNERS, so the sample each one ends on has to
    # be a sample that is actually on the line - see `peel_ends`. Only this
    # line: the centre line's far end is cut back by the walk's own end-of-line
    # cue and needs no help (its outermost sample moves 0.0 px on all 116
    # plates), and `FAR_T_SHORT` is calibrated to where that walk stops.
    if len(near_svc):
        near_svc, n_lo, n_hi = peel_ends(near_svc, False)
        i_T -= n_lo
        tr["peeled"] = (n_lo, n_hi)
    # A SECOND PASS up the same column that stops for nothing, reading the same
    # edge map at a lower vote count. The walk above has to decide where to stop
    # while it is walking; this one decides nothing and just records, so it can
    # afford to see faint lines. What it collects is judged by
    # `choose_far_service` once the whole column has been seen - which is the
    # only vantage point from which "the line with nothing above it" is a
    # question that can be asked at all.
    hz_cands, _ = scan_horizontals(f, T, (0, -1), edges, bgr.shape)
    chosen = choose_far_service(f, hz_cands, bgr.shape)
    tr["hz_cands"], tr["far_chosen"] = hz_cands, chosen
    # With both lines in hand the junction can be measured properly rather than
    # guessed at from a 3px scan grid: see `_junction_from_walks`. The seed has
    # done its job - it got the walks onto the paint - and the point put back
    # into the two line sets should be the one the walks agree on.
    Tj = _junction_from_walks(near_svc, centre, T)
    if not np.array_equal(Tj, T):
        T = Tj
        # Written back in place rather than rebuilt from the raw walks, which is
        # what this used to do: those walks are no longer what either line set
        # holds - `centre_on_paint` has moved every sample of both onto the
        # middle of its paint - and rebuilding would silently discard that.
        if len(near_svc):
            near_svc[i_T] = T
        if len(centre):
            centre[0] = T
        tr["T"] = T
    tr["near_service"], tr["centre_line"] = near_svc, centre
    # Everything downstream is measured relative to these two. Their acquisition
    # radius is ~20px (a 9px window widening by 2px per missed step, six times),
    # so an anchor further off the paint than that starts a walk that never locks
    # on - and it must say so rather than hand an empty array to the scans.
    if len(near_svc) < 60 or len(centre) < 30:
        raise SystemExit("anchor at (%.0f, %.0f) did not lead anywhere: "
                         "service line %d samples, centre line %d"
                         % (T[0], T[1], len(near_svc), len(centre)))

    def runs_at(p0, d, n, **kw):
        return scan_runs(f, mu, gate_ab, rough_max, p0, d, n, **kw)

    # --- column scans: net floor line, far baseline, far service line -------
    # Started just above the near service line - the one row known to be court -
    # and run upward until the court has ended twice. Columns near the centre
    # line are skipped: a vertical scan up the centre line is paint for its
    # whole length, so it never opens a run at all.
    cx_lo, cx_hi = (centre[:, 0].min() - 22, centre[:, 0].max() + 22) if len(centre) else (0, 0)
    svc_y = {}
    for p in near_svc:
        svc_y.setdefault(int(round(p[0])), []).append(p[1])
    net_pts, far_svc_pts, far_base_pts, far_cands = [], [], [], []
    xs, last = [], -1e9
    for x in sorted(svc_y):                       # a fixed pitch in pixels
        if not (cx_lo < x < cx_hi) and x - last >= col_step:
            xs.append(x)
            last = x
    for x in xs:
        y_start = float(np.mean(svc_y[x])) - 25.0
        if y_start < 30:
            continue
        runs, P, lab, dist, paint, _ = runs_at((x, y_start), (0, -1), int(y_start) - 2)
        if len(runs) < 2:
            # Run 1 ended, but with nothing above it: this column left the court
            # through a SIDELINE, not at the net. The court narrows towards the
            # far end, so an outer column exits sideways hundreds of pixels
            # before it would have reached the net - measured, at x=300 the run
            # ends at y=554 against y=350 in the middle. Recording those tops as
            # net points put 38 sideline samples into a 60-point set, and they
            # form a perfectly smooth curve of their own; RANSAC then had two
            # competing curves to choose between and sometimes chose the wing,
            # reporting the net at 13.2m and rejecting four calibrations that
            # were within 0.4-2.6px of a click-seeded model. A second run above
            # is the topological proof that the gap was the net.
            continue
        net_pts.append((float(x), y_start - _edge_subpixel(lab, runs[0][1])))
        a, b = runs[1]
        far_base_pts.append((float(x), y_start - _edge_subpixel(lab, b)))
        # Exactly one bright horizontal ridge lives inside the far half. This is
        # the whole reason for walking: the window is bounded by the court
        # itself, so however thin the far half is (47px here, 36px at the other
        # venue), there is nothing else inside it to lock onto.
        seg = _sample(f["horiz"], P[a + 3:b - 2])
        if len(seg) < 6 or not np.isfinite(seg).any() or np.nanmax(seg) < 10.0:
            continue
        far_cands.append((float(x), P[a + 3:b - 2][:, 1], np.nan_to_num(seg),
                          (float(P[b][1]), float(P[a][1]))))
        yy = P[a + 3 + int(np.nanargmax(seg))][1]
        q = IM._peak_ridge(f["horiz"], np.array([float(x), yy]), np.array([0.0, 1.0]),
                           3.0, 8.0)
        far_svc_pts.append((float(x), q[1] if q is not None else yy))

    # --- the far service line, chosen by depth rather than by brightness -----
    # The anchor is only trusted if it sits inside the far half the column scans
    # measured - court, gap, court - because a centre line that overshot into
    # the background is worse than no anchor at all. Measured, one plate here
    # ends its centre line at y=90 where the far half spans 200-260.
    far_t = centre[np.argmin(centre[:, 1])] if len(centre) else None
    # WHERE THE WALK NEVER REACHED AN END, the chooser supplies one. `at_end` is
    # the walk's own statement that it stopped because the LINE stopped; without
    # it the last sample is wherever the ridge died, which on `08-06-59` is the
    # net's near foot, 120px short of the far service T, and the far service is
    # then probed BELOW that and lands on the net. A chosen line is a line that
    # was walked 100px either side at under 1.5px rms with nothing above it,
    # which is a far better endpoint than a walk that admits it lost the ridge.
    if chosen is not None and not ctr_tr.get("at_end") and len(centre):
        far_t = np.array([chosen["x"], chosen["y"]], float)
        tr["far_t_from_chooser"] = True
    bracketed = far_t
    if far_t is not None and far_cands:
        near = sorted(far_cands, key=lambda c: abs(c[0] - far_t[0]))[:5]
        lo = float(np.median([c[3][0] for c in near])) - 12.0
        hi = float(np.median([c[3][1] for c in near])) + 12.0
        if not (lo <= far_t[1] <= hi):
            tr["far_t_rejected"] = (float(far_t[0]), float(far_t[1]), lo, hi)
            bracketed = None
    anchored = pick_far_service(f, far_cands, bracketed)
    how, walked = "bounded by the far half", np.zeros((0, 2))
    if anchored is None:
        # The runs disagreed with the anchor, or bracketed too few columns. Ask
        # the image directly instead, and believe it only on its own evidence.
        anchored = probe_far_service(f, far_t, near_svc[:, 0].min(), near_svc[:, 0].max())
        how = "probed below the centre line's end"
    tr["far_service_bright"] = np.asarray(far_svc_pts, float).reshape(-1, 2)
    if anchored:
        far_svc_pts = anchored

    # --- ...and then WALKED, now that its T is reliable ----------------------
    # The column scans give a median of 18 points over 530 px, on every plate in
    # the corpus - they sample one column every 22 px and only where a column
    # cuts cleanly through the far half, so this line has always been the
    # thinnest thing in the fit. The far service T is a point ON it, known to
    # 2.86 px worst case, and the near service line is already walked out of its
    # own T to 440 points. So walk this one the same way: 150 points over 600 px,
    # median 105% of the line's true projected length against the scans' 93%.
    #
    # It overruns the sidelines on some plates, out to world X +16 where the
    # glass base carries on at the same image height - the far-end version of
    # the reflection that used to extend the near service line. Left alone,
    # deliberately: measured against cameras that never saw these points, the
    # overrun backprojects to Y = 3.05 +- 0.11 m at 0.2-0.3 px of curve rms, so
    # it is COLLINEAR, and collinear points cannot move a line correspondence.
    # Shortening the walk's shape memory to catch it does the opposite of what
    # it looks like it should: `fit` 120 -> 40 takes the overrun from 21 plates
    # to 47 and from world X +16 to +71, because a short window tracks the bend
    # instead of resisting it. The long memory IS the stiffness.
    if far_t is not None and (ctr_tr.get("at_end") or tr.get("far_t_from_chooser")):
        rt = walk(f["horiz"], far_t, (1, 0), shape=bgr.shape, lab=f["lab"],
                  max_tilt=FAR_TILT, colour_ridge=True,
                  min_contrast=FAR_COLOUR_CONTRAST,
                  max_miss=FAR_MAX_MISS)
        lf = walk(f["horiz"], far_t, (-1, 0), shape=bgr.shape, lab=f["lab"],
                  max_tilt=FAR_TILT, colour_ridge=True,
                  min_contrast=FAR_COLOUR_CONTRAST,
                  max_miss=FAR_MAX_MISS)
        walked = np.vstack([lf[::-1], [far_t], rt]) if len(lf) or len(rt) else np.zeros((0, 2))
        tr["far_service_walked"] = walked
        # DID THE WALK STAY ON THE LINE IT STARTED ON? At the far T the far
        # service line is the faintest thing in the frame and the far court's
        # boundary is a few pixels beyond it, bright and parallel: on `10-05-24`
        # the walk leaves the service line within 20px and reports the court's
        # far edge instead. The chosen line is an independent measurement of
        # where the line runs, so the walk can be checked against it.
        # It is kept as a BACKSTOP rather than as the working guard. With
        # `FAR_TILT` stopping the walk where it turns, this fires on no plate in
        # the 116-plate corpus - it used to fire on four, and on two of those the
        # walk it threw away was correct for 300 px either side of the T and only
        # wrong in its tail. A test that judges a whole walk by one quadratic
        # through all of it cannot tell those two failures apart; stopping the
        # walk at the turn removes the need to.
        if len(walked) >= 12:
            co = np.polyfit(walked[:, 0], walked[:, 1], 2)
            if abs(float(np.polyval(co, far_t[0])) - far_t[1]) > FIND_SEED:
                # Say nothing rather than something wrong: the column scans
                # cannot drift onto a neighbouring line, because they bracket the
                # far half by topology - court, gap, court - so they are what
                # this falls back to.
                walked = np.zeros((0, 2))
                tr["far_walk_vetoed"] = True
        if len(walked) >= 40:
            # REPLACES the scans, rather than joining them. Taking the union
            # looks free - both sets are filtered by the same `_trim_to_curve`,
            # and the scans reach further on 11 plates - and it is not: measured
            # over the corpus it puts a camera back into disagreement with its
            # own repeat recording, 1.40 px to 4.55. The scan points are sparse
            # enough that a few bad ones carry real weight in the fit, and the
            # walk does not need them.
            far_svc_pts = [tuple(q) for q in walked]
            how = "walked out of the far T"
    # Held to whatever was actually measured, walked or scanned - see
    # `pin_far_service`. This runs after both, so the weak case is covered too,
    # and the weak case is the one that fails.
    pinned = pin_far_service(walked, far_t, far_svc_pts)
    tr["far_pinned"] = pinned
    tr["far_t"] = far_t
    tr["far_anchored"] = how if anchored is not None else None
    tr["far_service"] = np.asarray(far_svc_pts, float).reshape(-1, 2)
    tr["net_line"] = np.asarray(net_pts, float).reshape(-1, 2)
    tr["far_baseline"] = np.asarray(far_base_pts, float).reshape(-1, 2)
    tr["columns"] = len(xs)

    # --- row scans: the sidelines ------------------------------------------
    # Outward from the middle of the court, so the scan always starts on court
    # and the answer depends only on where it stops. Rows below the anchor are
    # included even though there is no centre line to start them from - the
    # court is widest there, which is where a sideline is worth the most.
    net_y = float(np.median([p[1] for p in net_pts])) if net_pts else T[1] - 100
    mid = {int(round(p[1])): p[0] for p in centre}
    for y in range(int(near_svc[:, 1].max()) + 10, int(h * 0.98), 2):
        mid[y] = T[0]
    # The near service line is bowed by ~100px end to end, so a horizontal scan
    # anywhere inside that band crosses it at a grazing angle - hundreds of
    # pixels of paint, far past what `bridge` will step over, and the scan ends
    # on the service line instead of the sideline. Measured, that put the left
    # edge at x=373 where the truth is x=156. The whole band is skipped.
    svc_lo, svc_hi = near_svc[:, 1].min() - 15, near_svc[:, 1].max() + 15
    side_l, side_r = [], []
    for y in sorted(mid)[::row_step]:
        if (net_y + 20 < y < svc_lo) or y > svc_hi:
            for d, acc in ((-1, side_l), (1, side_r)):
                x0 = mid[y] + 8 * d
                n = int(min(w * 0.6, (w - 4 - x0) if d > 0 else (x0 - 4)))
                if n < 60:
                    continue
                runs, P, lab, dist, paint, ran_off = runs_at((x0, y), (d, 0), n,
                                                             max_runs=1, min_run=40)
                if not runs or ran_off:           # walked out of frame, not off court
                    continue
                e = _edge_subpixel(lab, runs[0][1])
                if e < 40 or e > n - 4:
                    continue
                acc.append((x0 + d * e, float(y)))

    # --- sidelines again, this time anchored at the service line's corners ---
    # The near service line ends ON the sidelines, at world (0, 16.95) and
    # (10, 16.95), so each corner is a point known to be on the line we are
    # looking for. Measured against click-seeded models the corners land within
    # 0.08 m, and tracking from them gives ~80 points per sideline against the
    # row scans' 13, at X = -0.03 and 10.03 m. Preferred whenever the track runs
    # far enough to be worth more than the scans; the scans stay as the fallback
    # because the track needs a boundary that exists continuously, and at some
    # venues it does not.
    tr["scanned_sides"] = {"left_sideline": np.array(side_l, float).reshape(-1, 2),
                           "right_sideline": np.array(side_r, float).reshape(-1, 2)}
    # THE RAW OUTERMOST SAMPLE, AND DELIBERATELY NOT THE LAST RANSAC INLIER.
    # These two are the corners - they anchor the fit, they start both sideline
    # chords, and every sideline detector is forced through them - and taking
    # them from the trimmed line is the obvious improvement that measures out
    # backwards. `add` does trim this line before fitting it, so the corner IS a
    # point the next stage may discard; the mistake is inferring from that that
    # the trim knows better where the paint stops.
    #
    # It does not, and cannot: `trim_to_curve` fits ONE quadratic to ~440 samples
    # spanning both arms, a fisheye bows that span by more than a quadratic can
    # follow to 3 px, and what RANSAC clips is therefore the ENDS of a curve that
    # is legitimately not quadratic out there. A curve trim constrains where a
    # line IS; it says nothing about where it STOPS.
    #
    # Measured, backprojected through 110 accepted cameras that never saw either
    # variant - the left end's world X, whose spread is what `ANCHOR_OUT` has to
    # stand for:
    #
    #     definition                       median     mad   p5..p95        worst
    #     outermost raw sample             -0.035   0.030   -0.21..+0.03   +0.52
    #     outermost RANSAC inlier          -0.001   0.076   -0.11..+0.80   +1.53
    #     raw x, y read off the curve      -0.021   0.037   -0.19..+0.05   +0.53
    #
    # The trim does not bias the corner, it scatters it - 2.5x the mad, with a
    # p95 tail 0.8 m inside the court - so there is no constant to re-derive,
    # and the corpus goes from 110 accepted to 68, 45 of them refused on the
    # anchor gate. Snapping the raw endpoint onto the trimmed curve keeps the
    # along-line position and fixes nothing either: the quadratic is extrapolated
    # past its own inliers to get there and overshoots, adding a 2 cm bias
    # across the line for no reduction in scatter.
    #
    # The far service T is not treated this way either, and there the reason is
    # that it needs no treatment: its outermost sample moves 0.0 px on all 116
    # plates, because `walk` already cuts that walk back to the last sample with
    # paint ahead of it. Where an endpoint rule belongs is in the walk's own stop
    # conditions, which know what the line is doing; not in a fit afterwards,
    # which only knows what shape it has.
    #
    # WHICH IS AN ARGUMENT ABOUT WHERE THE LINE STOPS, AND NOT ABOUT WHETHER THE
    # SAMPLE IT STOPS ON IS ON THE LINE - two questions this comment used to run
    # together. The trim cannot answer the second one either, for the reason
    # above, but a LOCAL fit can and `peel_ends` has already done it by the time
    # this runs: the endpoint below is the outermost sample that its own line,
    # fitted over the 120 samples just inside it, agrees is a point on it. That
    # leaves the along-line position measured exactly as the table above says it
    # should be, and takes the across-line blunder out of it - worst corner 14.6
    # px off its own line, against 0.23 px of scatter, down to 4.2.
    ends = (near_svc[np.argmin(near_svc[:, 0])], near_svc[np.argmax(near_svc[:, 0])])
    E = canny_edges(bgr)
    scans = tr["scanned_sides"]
    # BOTH SERVICE LINES HAVE NOW BEEN WALKED TO THEIR OWN ENDS, and those four
    # ends are points on the two sidelines - so each sideline can be read along a
    # chord drawn between its own two, on a strip blurred hard along that chord.
    # See section 4b. None unless the far service walk stopped symmetrically.
    far_ends = chord_ends(walked, far_t)
    # ...AND EACH NEAR CORNER IS THEN SLID ALONG ITS OWN LINE onto the edge that
    # ends it, using the chord to the far service line's end on the same side as
    # the direction that edge must run in. Before the sideline readers, because
    # the corner is what they are all seeded from and vetoed by - and after
    # `chord_ends`, because that is where the direction comes from. See
    # `corner_on_curve`; a plate whose far service walk gave no chord keeps the
    # walk's own endpoint, as does one where nothing at the corner stands out.
    anchor_out = [ANCHOR_OUT, ANCHOR_OUT]
    refine = corner_refine_wanted()
    if far_ends is not None and refine:
        moved, cor_tr = [], {}
        for i, sgn in ((0, -1.0), (1, 1.0)):
            t = {}
            if "near" in refine:
                # `peak` passed rather than defaulted so that the module constant
                # is read at call time - which is what lets it be swept from
                # outside without touching this file.
                q, d = corner_on_curve(f["gray"], near_svc, ends[i], far_ends[i], sgn,
                                       peak=CORNER_PEAK, rule=corner_rule(), trace=t)
                ends = (q, ends[1]) if i == 0 else (ends[0], q)
                # A corner the refiner declined is still the walk's endpoint and
                # is still the end of the PAINT, so it keeps the walked constant.
                # Only the ones actually placed on the crossing edge move to the
                # sideline.
                if t.get("moved"):
                    anchor_out[i] = CORNER_ANCHOR_OUT
            moved.append(t.get("along", 0.0))
            cor_tr["left" if i == 0 else "right"] = t
        tr["corner_refined"] = cor_tr
        # THE FAR SERVICE LINE'S OWN ENDS, by the same rule and off the same
        # sideline - met at the other end of it. The chord is the near corner
        # this time, and it is taken AFTER that corner has been placed, so the
        # better-known of the two anchors the search for the worse-known.
        #
        # These are not solver anchors: they are where both chord readers draw
        # their strip from, so what an error here costs is a tilted strip rather
        # than a pulled fit - which is why the gain shows up in the court error's
        # median and p90 rather than in the accepted count. See `CORNER_REFINE`.
        if "far" in refine:
            far_tr, far_moved = {}, []
            new = list(far_ends)
            # The same samples `chord_ends` read the endpoints off, rather than
            # the raw walk: the local curve this search is held to should be the
            # line's, and out here the walk's tail can leave it.
            fw = trim_to_curve(walked, False, tol=3.0)
            for i, sgn in ((0, -1.0), (1, 1.0)):
                t = {}
                new[i], _ = corner_on_curve(f["gray"], fw, far_ends[i], ends[i],
                                            sgn, peak=CORNER_PEAK, rule=corner_rule(),
                                            trace=t)
                far_tr["left" if i == 0 else "right"] = t
                far_moved.append(t.get("along", 0.0))
            far_ends = tuple(new)
            tr["far_ends_refined"] = far_tr
        if verbose:
            print("  corners  slid %+.1f / %+.1f px along their own line "
                  "(+ = further out)" % (moved[0], moved[1]))
            if "far" in refine:
                print("  far ends slid %+.1f / %+.1f px" % (far_moved[0], far_moved[1]))
    wanted = side_sources_wanted()
    record_all = side_record_all()
    select = side_select()
    cand = {}
    tr["chord_ends"] = far_ends
    tr["chord_y_min"] = net_y + 20
    tracked, canny, how = {}, {}, {}
    chord_tr, chord_pts, line_pts = {}, {}, {}
    for corner, sgn, name, far_end in ((ends[0], -1, "left_sideline",
                                        far_ends[0] if far_ends else None),
                                       (ends[1], 1, "right_sideline",
                                        far_ends[1] if far_ends else None)):
        cand[name] = {}
        # Read WHETHER OR NOT it is going to be used. It is the last source
        # tried, so on most plates its answer is never asked for - and its answer
        # is the one worth having beside the others in the debug picture, because
        # it is the only one arrived at a different way. An independent second
        # opinion that agrees is how you know the first one is right, and it
        # costs 0.56 s against the walk's 7.2.
        chord_tr[name] = {"chord": {}, "chord_lines": {}}
        chord_pts[name], line_pts[name] = (
            (chord_side(f, T, corner, far_end, trace=chord_tr[name]["chord"]),
             chord_lines(f, T, corner, far_end, trace=chord_tr[name]["chord_lines"]))
            if far_end is not None else (np.zeros((0, 2)), np.zeros((0, 2))))
        pts, cands = side_edge(f, E, centre, near_svc, corner, far_t, sgn, rough_max)
        canny[name] = cands
        # THE CORNER IS EVIDENCE, NOT A HINT. Each detector below is asked for a
        # sideline and then made to prove it reached the one point on that line
        # that is already known - see `_corner_miss`. Having enough points is not
        # the same as having the right ones, and it is the only test the wrong
        # ones fail: on `13-09-39` the first strong edge out from centre is the
        # base of the fence rather than the base of the glass, and it returns a
        # perfectly clean 16-point arc a metre outside the court. The first
        # source that agrees with its corner wins; the boundary track needs a
        # continuous colour step, which some courts do not have, and the row
        # scans stop at the first colour change, which on a worn court is inside.
        #
        # THE CHORD GOES LAST, AND THAT ORDER WAS MEASURED RATHER THAN REASONED.
        # It is the only one of the four told where to look before it looks - the
        # other three all search outward from inside the court and stop at the
        # first thing that qualifies, so they can be lost to whatever is nearer
        # than the boundary or, where the boundary is faint, to whatever lies
        # beyond it. Judged as a detector it is duly the more robust: over the 232
        # sidelines here, against each plate's own fitted camera, its worst is
        # 63 px and it never comes back empty, where the first-strong-edge pass
        # reaches 308 px, puts its p99 at 102, and comes back empty twice.
        #
        # It is still the WORSE first choice, and the reason is that robustness
        # is not what is scarce here. In the ordinary case the two agree to a few
        # pixels and the Canny pass is the more accurate of the two - it selects
        # on a strict edge map and then re-measures with `edge_at`, against this
        # one's step read off a blurred profile - and a few pixels of sideline is
        # a few centimetres of court width, which is exactly what `paint_probe`
        # is sensitive to. Put first, in five variants tried, it accepts 102-108
        # plates of 116 against 110; the losses are almost all paint coverage, and
        # on the plate that is easiest to read it sits a median 4.9 px outboard of
        # the Canny answer on one side while agreeing to 0.06 px on the other.
        # Constraining the Canny pass to a 30-50 px band around it instead of
        # replacing it costs two plates as well.
        #
        # So it is here to answer the question none of the other three can, which
        # is what to do when all of them are refused: on this corpus that is two
        # left sidelines, both of which now have one where they had none.
        sources = [("canny", "first strong edge out from centre", lambda: pts),
                   ("track", "boundary track from the corner",
                    lambda: track_side(f, corner, rough_max, side=float(sgn),
                                       shape=bgr.shape)),
                   ("scans", "row scans", lambda: scans[name])]
        # Both chord readers, already read above; only the near half of either is
        # offered - see `_chord_read` for why the far half is not a measurement.
        for key, label, p in (
                ("chord_lines", "aligned edges along the chord", line_pts[name]),
                ("chord", "the chord to the far service line's end",
                 chord_pts[name])):
            if len(p):
                sources.append((key, label,
                                lambda p=p: p[p[:, 1] >= net_y + 20]))
        if wanted:
            # An experiment asked for a subset, in its own order. A key with no
            # source behind it - `chord`, when the far service walk stopped
            # asymmetrically - drops out here and the sideline is refused, which
            # is the honest answer for "what is this technique alone worth".
            by_key = {k: s for k, s in ((s[0], s) for s in sources)}
            sources = [by_key[k] for k in wanted if k in by_key]
        # EVERY SOURCE IS READ unless the cascade is running and nobody wants
        # the rest: the graded rule cannot rank what it has not seen, and the
        # recording exists to be looked at. Under `cascade` the old laziness
        # stands, which is the ~1 s a plate the `track_side` call costs.
        eager = record_all or select in ("graded", "fit")
        got = np.zeros((0, 2))
        for key, label, source in sources:
            if len(got) and not eager:
                break
            p = np.asarray(source(), float).reshape(-1, 2)
            why, miss = "", None
            if len(p) < 12:
                why = "%s: %d points" % (label, len(p))
            else:
                miss = _corner_miss(trim_to_curve(p, True), corner)
                if miss > SIDE_CORNER_MAX:
                    why = "%s: misses its corner by %.0f px" % (label, miss)
            cand[name][key] = {"label": label, "points": p, "corner_miss": miss,
                               "qualifies": not why, "why": why}
            if not why and not len(got):
                got = p
    tr["corners"] = np.array(ends)
    tr["anchors"] = anchors(ends, far_t if ctr_tr.get("at_end") else None, bgr.shape,
                            outs=tuple(anchor_out))

    out = []
    tr["kept"] = {}

    def add(name, pts, swap, tol=3.0, max_rms=8.0):
        pts = np.asarray(pts, float)
        if len(pts) < 10:
            return 0
        pts = trim_to_curve(pts, swap, tol=tol)
        tr["kept"][name] = pts
        if len(pts) < 10:
            return 0
        # `_trim_to_curve` hands back its INPUT UNCHANGED when RANSAC cannot
        # reach twelve inliers, so "I could not find a curve in this" is
        # indistinguishable from "these are all on one" - and counting points,
        # as the test above does, cannot tell them apart. Measure what came
        # back instead. Over 40 plates every line type peaks at 3.9 px while the
        # sets the trimmer gave up on run 23-72 px, so 8 px separates them with
        # room to spare in both directions.
        u = pts[:, 1] if swap else pts[:, 0]
        v = pts[:, 0] if swap else pts[:, 1]
        try:
            rms = float(np.sqrt(((v - np.polyval(np.polyfit(u, v, 2), u)) ** 2).mean()))
        except (np.linalg.LinAlgError, ValueError):
            rms = 0.0
        if rms > max_rms:
            tr.setdefault("refused", {})[name] = rms
            return 0
        a, b = WORLD_LINES[name]
        out.append({"name": name, "pts": pts,
                    "world_a": np.array(a, float), "world_b": np.array(b, float)})
        return len(pts)

    # THE FOUR LINES THE SIDELINE CHOICE DOES NOT DEPEND ON GO IN FIRST, because
    # under `fit` they are what the candidate sidelines are tried against: a
    # provisional camera is solved from these plus each candidate pair, and the
    # pair judged by what its own camera makes of it. See `choose_sides`.
    n = {
        "near_service": add("near_service", near_svc, False),
        "centre_line": add("centre_line", centre, True),
        # The net line is a shadow-and-mesh boundary, not paint: a 3px
        # inlier band throws away two thirds of it for no reason.
        "net_line": add("net_line", net_pts, False, tol=5.0),
        "far_service": add("far_service", far_svc_pts, False, tol=2.0),
    }
    tracked, how, grades = choose_sides(cand, ends, far_ends, select, wanted,
                                        base=out, points=tr["anchors"],
                                        shape=bgr.shape)
    side_l = [tuple(q) for q in tracked["left_sideline"]]
    side_r = [tuple(q) for q in tracked["right_sideline"]]
    n["left_sideline"] = add("left_sideline", side_l, True)
    n["right_sideline"] = add("right_sideline", side_r, True)
    tr["side_grades"] = grades
    tr["side_select"] = select
    tr["tracked"] = tracked
    tr["canny_cands"] = canny
    tr["side_how"] = how
    tr["chord"] = chord_tr
    tr["chord_sides"] = chord_pts
    tr["chord_line_sides"] = line_pts
    if record_all:
        tr["side_candidates"] = cand
        tr["side_corners"] = {"left_sideline": ends[0], "right_sideline": ends[1]}

    if verbose:
        print("  colour mu=(%.1f, %.1f)   roughness gate %.1f   net floor y=%.0f"
              % (mu[0], mu[1], rough_max, net_y))
        print("  walked   near_service=%d  centre_line=%d   (%d columns scanned)"
              % (len(near_svc), len(centre), len(xs)))
        for nm, c in (("near_service", svc_mid), ("centre_line", ctr_mid)):
            print("  centred  %-13s %s"
                  % (nm, ("moved %+.2f +-%.2f px onto %.1f px of paint "
                          "(up to %.2f, %d fell back)"
                          % (c["shift"], c["scatter"], c["width"],
                             c["shift_max"], c["fell_back"]))
                     if c.get("moved") else "left alone - " + c.get("why", "")))
        print("  scanned  net=%d  far_baseline=%d  far_service=%d  left=%d  right=%d"
              % (len(net_pts), len(far_base_pts), len(far_svc_pts), len(side_l), len(side_r)))
        print("  kept     " + "  ".join("%s=%d" % (k, v) for k, v in n.items()))
        xn = sum(1 for L in out if L["world_a"][0] == L["world_b"][0])
        print("  %d lines along X, %d along Y%s"
              % (xn, len(out) - xn, "" if xn >= 3 and len(out) - xn >= 3
                 else "   <-- fewer than 3 in a direction: no redundancy there"))

    if debug is not None:
        vis = bgr.copy()
        cols = {"left_sideline": (0, 0, 255), "right_sideline": (0, 255, 0),
                "net_line": (255, 0, 0), "near_service": (255, 0, 255),
                "centre_line": (255, 255, 255), "far_service": (0, 255, 255)}
        for L in out:
            for p in L["pts"]:
                cv2.circle(vis, tuple(np.round(p).astype(int)), 2, cols[L["name"]], -1)
        for p in far_base_pts:
            cv2.circle(vis, tuple(np.round(p).astype(int)), 1, (128, 128, 255), -1)
        # THE FOUR SERVICE LINE ENDS, on the crossing of an X and to a sixteenth
        # of a pixel - see `draw.mark_x`. `draw` imports this module, so the
        # import is deferred to here rather than taken at the top; by the time a
        # debug picture is being written this module is long since loaded, so
        # there is no cycle to walk into.
        from .draw import CHORD_COL, mark_x
        for p in np.asarray(ends, float).reshape(-1, 2):
            mark_x(vis, p, (0, 255, 255))
        for p in np.asarray(far_ends if far_ends is not None else
                            np.zeros((0, 2)), float).reshape(-1, 2):
            mark_x(vis, p, CHORD_COL)
        cv2.drawMarker(vis, tuple(np.round(T).astype(int)), (0, 165, 255),
                       cv2.MARKER_TILTED_CROSS, 40, 2)
        cv2.rectangle(vis, (int(T_BOX[0] * w), int(T_BOX[1] * h)),
                      (int(T_BOX[2] * w), int(T_BOX[3] * h)), (0, 165, 255), 1)
        cv2.imwrite(debug, vis)
        print("  wrote %s" % debug)

    if with_ctx:
        return out, {"T": T, "net_y": net_y, "mu": mu, "Sinv": Sinv, "gate_ab": gate_ab,
                     "rough_max": rough_max, "far_baseline": np.array(far_base_pts),
                     "anchors": tr["anchors"], "pinned": {"far_service": pinned},
                     "edges": edges}
    return out


# --------------------------------------------------- 6. the walk, drawn ----
#: Only the three lines that come out of walking alone. The sidelines are not
#: here on purpose: they are 1-D colour scans that stop at an edge, not paint
#: followed by a ridge tracker, and mixing them in would make a picture of the
#: detector rather than of the walk. The flag is the fit orientation: the two
#: service lines are y(x), the centre line is x(y).
#: Black for the centre line: it is drawn ON white paint, so any light colour
#: disappears into what it is marking.
WALK_LINES = (("near_service", (255, 0, 255), False),
              ("centre_line", (0, 0, 0), True),
              ("far_service", (0, 255, 255), False))


