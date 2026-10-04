# How it works

Two parts. The first is the algorithm as it stands: what each stage does and why
it exists at all. The second is how it got that way - the attempts, the
measurements that killed them, and the ideas that survived. Both stay at the
level of ideas; the details, the constants and the numbers behind every choice
are in the comments of the module named at the end of each section.

---

# Part 1 - the algorithm

## The problem, and why the obvious approach fails

One padel recording, one fixed camera, and the question is where the court is.
Four things make it harder than it sounds.

**The camera is a fisheye.** A straight service line bows up to 65 px across the
frame. A homography maps straight lines to straight lines, so no 3x3 matrix can
describe this view. The lens has to be part of the model.

**A padel court paints almost nothing.** Only the two service lines and the two
centre lines are painted. The perimeter is the wall: the playing surface simply
runs up to the glass, so a "sideline" is a step in colour, not a bright ridge.
Two different kinds of feature, one fit.

**The far half is tiny and half-hidden.** From a mount behind the near baseline
the far half occupies 30-50 px of a 1080-line frame - under 1% - and the net
stands in front of it. The far service line is the single most valuable
measurement in the frame, because it fixes the far end of the court, and it is
also the faintest thing in it.

**Nothing about the lighting is uniform.** Court L varies by 64 units across
venues and by 100 units across a single frame. Any fixed threshold works at one
end of the court and fails at the other.

The court geometry itself is the one thing that is certain: 20 x 10 m, net at
10 m, service lines 6.95 m either side of it, centre lines at X=5. Every
regulation court matches, so one fixed model fits every venue.

## Stage 0 - the plate

A per-pixel median of ~45 keyframes spread across the whole recording. Anything
that moves is somewhere different in every frame and loses the median; the court
is identical in all of them and survives. Players and the ball disappear.

Two details matter: the frames are spread over the *whole* recording rather than
a window (a player who stands still for 15 s survives a 15-frame window), and
they are keyframes decoded with `-skip_frame nokey` (these recordings have no
index and no usable timestamps, so seeking does not work at all).

→ `plate.py`

## Stage 1 - the anchor

One point starts everything: the **near service T**, where the near service line
meets the centre line. It is the best-conditioned feature in the frame - closest
to the camera, largest, and a junction of two lines that are both painted.

It is found by *arm directions*, not by brightness. A box scan asks, at each
candidate, in which directions white leaves it: the near service T has three
arms and, towards the camera, bare court past a short stub. "Where is the most
white" would answer the net's top tape.

The seed only has to be good enough to start the walks - about 20 px. The
junction itself is measured properly later, in stage 3.

→ `walk.py: find_T`

## Stage 2 - two walks

From the anchor, a ridge follower walks left and right along the near service
line, and away along the centre line. Every 4 px it re-centres on the ridge crest
perpendicular to its own direction, and re-estimates its direction from the last
few accepted samples, so it tracks the lens bow instead of fighting it. The near
service line yields ~440 sub-pixel points.

Knowing where to **stop** is what makes the endpoints usable, and it is harder
than following: the court is walled in glass, and the glass reflects the service
line. Three cues stop the walk, all measured against the walk's own recent
history so that vignetting and venue colour never enter: the ridge response
collapsing, the walk leaving the quadratic fitted to its own recent samples, and
- the decisive one - there being no more paint ahead. The last is applied
*afterwards*: the walk runs on, and is then cut back to the first sample of the
trailing run with nothing in front of it, because "the line ended" and "something
crossed the line" look identical until you know whether paint came back.

A ridge follower does not follow the *middle* of a wide line, and both of these
lines are wide. The crest it re-centres on is the nearest local maximum, which
inside an 8 px stripe means whichever ripple of the plateau the walk was already
standing over: on a high-contrast court that latches at the anchor and is carried
the whole way, about 2 px, with a sign that flips between two recordings of the
same fixed camera; on a low-contrast one it changes its mind every step and
scatters by ~1 px, the samples visibly crossing from one edge of the stripe to
the other.

So each walk is then **slid sideways onto the middle of its own paint**. The two
strict Canny edges of the stripe are read perpendicular to the line at every
sample, and the two combinations of those distances say different things: their
*difference* is the follower's error exactly, per sample, and their *sum* is the
stripe's width with that error algebraically absent. So the width is what gets
the RANSAC quadratic - not to produce the answer but to decide which pairs are
measuring one stripe rather than a crossing line - and the midpoint is taken per
sample, where the walk's error cancels on its own. Only where there is no
credible pair does a fitted curve supply the correction.

With both walks in hand, the junction is re-measured by intersecting them, which
is a far better estimate than any local scan of the blob where two lines merge -
and it now intersects two lines' middles rather than two arbitrary chords of them.

→ `walk.py: walk, _junction_from_walks`, `image.py: centre_on_paint`

## Stage 3 - the far end of the centre line

The centre line's far end **is** the far service T, the anchor the entire far
half hangs from. Reaching it means crossing the net - and a padel net has a
white vertical strap standing directly over the centre line, brighter than the
paint. A ridge follower climbs it without hesitating.

Three mechanisms, each doing one job:

**A gap means two different things either side of the net.** Before the net, gaps
have to be tolerated - the strap, the tape and their shadows interrupt the line.
After it, the line ahead is plain paint on plain floor and the first real gap
*is* the end. The walk latches when it crosses a horizontal white line, and
tightens its leash once there is smooth floor beside it again. Not after a fixed
number of steps: one venue needs a longer wait than another venue's entire far
half.

**A second pass that stops for nothing.** The same ridge follower runs the column
again with every stop rule removed, recording each horizontal line it crosses at
a lower Hough vote count. It decides nothing, which is exactly why it can afford
to be sensitive.

**A choice made afterwards.** The far service line is the one with a *gap just
above it* - the centre line ends where they cross, so nothing is above the true
line - that also walks 100 px either side at under 1.5 px rms. Cleanliness alone
picks the net's top tape every time; ordering alone picks a decoy below the true
line. The two cues together are what separate them.

→ `walk.py: scan_horizontals, choose_far_service`

## Stage 4 - column scans: topology, not photometry

The net's floor line, the far baseline and the bracket for the far service line
come from vertical scans, one every 22 px, each starting on a row known to be
court and walking up. Each column is cut into court-ON runs: on, off (the net),
on again (the far half), off for good.

The scans never ask "does this look like a line". They ask what is met, and in
what order. A 47 px window containing exactly one candidate cannot pick the wrong
line however thin the sliver is, and no threshold in that sentence has to be
right. It is the strongest idea in the pipeline.

Staying inside a run is judged against the *neighbouring* pixels, not an absolute
colour: these cameras vignette hard enough that an absolute gate tight enough to
stop at the wall stops in the middle of the court. Re-acquiring after the net has
to be absolute, because across the net there is no neighbour left. And colour
cannot see the net at all at some venues - navy mesh against blue court - so the
net is recognised by *roughness*, the local mean absolute deviation of the grey
image, which is ~1 on a floor and 6-9 on a mesh whatever colour either is.

→ `walk.py: scan_runs, pick_far_service`

## Stage 5 - the sidelines

The near service line ends **on** the sidelines, at world (0, 16.95) and
(10, 16.95). Once the walk stops at the real corner rather than in the glass
reflection, each corner is a point known to be on the line being looked for - so
the sideline never has to be found, only followed. A tracker follows it up-court
from there, looking only +-22 px around its own prediction, which is why whatever
the older row scans used to stop at is simply never in view: 13-14 points per
sideline became ~80.

Both of its references are local. An absolute colour model tops out below the
threshold in the vignetted corner where the track has to start, and a global
roughness gate measured beside the anchor trips on bare court out by the wall.

Three detectors are tried in order - the first strong edge outward from the
centre line, the boundary track from the corner, the old row scans - and each
has to **prove it reached the corner it was seeded from**. Being seeded with a
point and being checked against it are not the same thing, and only the second
catches a sideline that is on the wrong feature: a quadratic forced through the
corner still has two free parameters, and two are enough to reach a fence base
a metre outside the court. The first source that agrees with its corner wins,
and a sideline that no source can find is left out of the fit rather than
guessed at.

→ `walk.py: track_side, side_edge, SIDE_CORNER_MAX`

## Stage 6 - the fit

Six lines go to the solver: three along X (two sidelines, centre line) and three
along Y (both service lines, the net's floor line). Three parallel lines in a
direction have an invariant cross-ratio and two do not - that is the whole reason
for six rather than four.

The fit is staged so that nothing needs a good first guess:

1. **Distortion alone**, by making each point set straight. No homography is
   needed to know that a court line is straight in the world.
2. **The homography**, by DLT on line correspondences, on the now-straight sets.
3. **Joint refine** of the radial coefficients and the homography together
   against every sample at once, robustly, with reject-and-refit.

The two detected corners of the near service line enter as **point constraints**.
That looks like it should be swamped by 440 line points and is not: a line
correspondence measures perpendicular distance only, so it is structurally blind
to where along itself the court sits. The X scale is left entirely to the
sidelines, and where those are short or extrapolated, the corners are the only
observation in that contest.

→ `solve.py`

## Stage 7 - re-trace

Twice, the five paint and boundary lines are re-measured from the model just
solved, and the fit is redone. This is what turns a good fit into a precise one.

Each round is a **candidate, not a replacement**: all of them are scored by the
stage 8 verdict, read as a continuous quantity rather than a boolean, and the
best one is kept. A re-trace can fail by finding the wrong line, and it can also
fail by finding nothing at all - and a round that loses one sideline hands the
solver two lines along X where the walk gave three, which is exactly the
degeneracy the six-line design exists to avoid.

It has one failure mode, and one guard. Re-seeding a line from the model is
exactly what rescues a line that was detected blind, and exactly what sinks one
that was measured well: in a thin far half the net's tape is inside the search
window and brighter, so the re-trace finds the tape, fits the tape, and the next
round is seeded from a model that agrees with it. So a line the walk measured
*well* is pinned: the re-trace must agree with it to a tolerance that measurement
earned, or it is discarded and the measurement stands.

The re-traced paint lines are centred on their paint exactly as the walks are,
and for the same reason: the tracer re-centres with the same nearest-crest rule,
so it inherits the same off-centre latch. Leaving it out undoes the correction on
the 72 plates of 116 that keep a refinement round.

The net's floor line is not re-traced at all - it is carried through from the
column scans, which located it topologically.

→ `calibrate.py: _finish`, `solve.py: retrace`

## Stage 8 - the verdict

Four gates: paint coverage above 85%, an independent paint probe under 2 px, line
fit rms under 2 px, and no anchor further than 25 px from where the fit put it.
Plus a redundancy check - every line held out of the fit must measure where the
model says it should.

The point of holding a line out is that no measurement taken *on the paint the
fit used* can see a wrong far-field geometry. Measured on a deliberately blind
pass: the paint probe scored it **better** than the good fit (0.57 px vs 0.73),
because the blind fit really does put the painted lines in the right places. Only
the held-out net line saw the error.

A refused result still carries a full camera. A plate whose far half is faint can
be fitted correctly and still miss the coverage gate, so the numbers and the
`_court_fail` picture are the output, not a silent `None`.

→ `calibrate.py: calibrate, paint_probe`

## Four principles, if you only remember four

1. **Topology beats photometry.** What is met, and in what order, walking from
   somewhere known - not what looks brightest.
2. **Everything relative, nothing absolute.** Every threshold that matters is
   measured against the walk's own recent history, the neighbouring pixels, or a
   patch of this court's own floor.
3. **Decide afterwards.** "The line ended" and "something crossed it" are the
   same reading until you know whether paint came back. Record, then choose.
4. **Hold something out.** A fit judged only on what it was fitted to will pass
   while being wrong.

---

# Part 2 - how we got here

Each step here was a genuine attempt, and most of what follows is the record of
what measurement said about it. The order is chronological.

## 1. Clicked keypoints and traced lines

The first working version asked a human for 6-11 keypoints and traced the lines
outward from them. Two findings from it survived everything since.

**The clicks are only a seed.** Accuracy comes from the tracing: ~1500 sub-pixel
samples constrain the same parameters that 11 keypoints would, and noise averages
down with sqrt(N). Corners are the worst features on a court - two 5 cm bands
crossing at a grazing angle, frequently stood on - while the middle of a long
line is clean. Clicks entered the fit weakly enough that a 15 px click error did
not matter.

**The far end lies.** Held out of the fit, the far baseline measures Y = +0.75 m
with a spread of only 0.14 m: straight, consistent and confidently wrong, because
the bottom few centimetres of the glass hide the strip of surface in front of it.
A robust loss cannot save you from that - it is not an outlier. It is excluded by
default to this day.

## 2. Trying to remove the human

Two attempts. A blind global search over camera poses did not work - the space is
too large and the acceptance function too flat. Region-growing from the court's
own colour worked better: sample a central rectangle, estimate the surface colour
robustly in Lab chroma, flood-fill it with Canny edges as barriers, and read the
region's outline as three lines.

That produced a real detector, and it produced the failure that redirected the
whole project: **a seed a few tens of pixels out sends a trace onto the wrong
feature, and because the two service lines are parallel, the fit absorbs the
error by rescaling Y and still reports a clean residual.** A wrong answer with a
good residual is the worst possible outcome.

## 3. Points out, lines in

If seeds are the problem, remove the concept. A line correspondence needs no seed
to exist. Six lines, three per direction, because three parallel lines have an
invariant cross-ratio and two do not - a misidentified line now *raises* the
residual instead of being absorbed.

The staged fit dates from here: distortion first by plumb-line (no homography
needed), then a DLT on lines, then joint refinement. RANSAC on quadratics rather
than a Hough transform, because these lines bow 65 px and a Hough accumulator
rewards short straight artefacts over long bowed real ones.

This is the solver still in use, unchanged.

## 4. The far service line refuses to be found blind

The line pipeline finds the far service line by segmenting the far half and
searching inside it. The far half is a 47 px sliver at one venue and 36 px at
another. Blind, that search is wrong every time; it was only ever recovered
because the model-seeded re-trace went back for it afterwards.

Turn the re-trace off and ask each pipeline where the far service line is:

| | galais | rotonde |
|---|---|---|
| walking | 3.06 m | 3.05 m |
| region + search | not found at all | 2.83 m (22 cm out) |

That is what motivated the walk: stop depending on a good enough first guess.

## 5. Walking, and the scans

The walk pipeline replaced the front end and kept the solver. Its ideas -
anchor by arm directions, free-running ridge walks, column scans that cut the
court by topology - are described in Part 1. What was *not* obvious, and cost
real time:

- **Two colour tests for two questions.** Neighbour-relative to stay in a run
  (an absolute gate stopped a column 220 px early, mid-court); absolute to
  re-acquire after the net, where there is no neighbour left.
- **The neighbour reference must be a median, not an exponential average.** An
  EMA tracks the very thing it is meant to detect: lagging two samples behind a
  3-units-per-pixel ramp, it never saw more than 8 units of change, so every
  sideline scan walked through the wall and off the frame.
- **Roughness sees the net when colour cannot.** Navy mesh sits 14 chroma units
  from a blue court; the local mean absolute deviation separates them cleanly at
  every venue.
- **A scan crossing a line must step over its shoulders.** The far service line
  carries a dark shadow along its upper edge; bridging only the pixels the
  top-hat called paint split the far half into two 15 px runs, and the far half
  stopped existing as a single thing.

## 6. The glass reflects the court, and the walk believed it

Unstopped, the walks ran 60-96 px past the real corner into the reflection of the
service line in the glass and put the court's edge at X = -1.26 and 11.11 m. The
three stop cues came from this, and so did the ordering lesson: the end-of-line
test cannot be applied at the first bad reading, because a wide occlusion reads
identically. Run on, then cut back.

## 7. The corners: from check to constraint

Once the walk stopped at the real corner, the two ends of the near service line
were points known to lie on the sidelines. First they were used as a *check*, on
the reasoning that 440 line points would swamp two corner points in the fit.

**That reasoning was wrong, and it is the most useful mistake in the project.**
Least squares does not apportion influence by counting residuals; it does so by
the derivative of the residual with respect to the parameter. A line
correspondence measures perpendicular distance and nothing else, so it is
structurally blind to where the court sits *along* that line. The X scale rests
entirely on the sidelines, and the 440 points contribute nothing to it. A corner
does not compete with them - it competes with nothing.

Promoted to constraints, the anchor residual on the three plates this was built
for fell from 82, 75 and 60 px to 8.8, 2.8 and 3.0 - and changed the answer by at
most 0.07 px where the fit was already pinned. Inert where it should be, decisive
where it should be.

The same corners then anchored the sidelines themselves (Part 1, stage 5).

## 8. A refinement that had never run

The junction of the two walked lines had a dedicated refinement step. Measured
over the corpus, it found a median of **zero** samples: its search window was
narrower than the paint at the junction, so the median-subtraction inside it had
no contrast left to work with. It was a no-op on 95 plates of 104 and harmful on
the 9 where it did fire.

Replaced by intersecting the two walked lines, the y error at the junction fell
from 3.16 px to **0.53**, and the whole downstream court moved by a median of
0.05 px - which is the real lesson: a step that appears to be doing nothing
should be checked, not trusted.

## 9. A measure that had to be retired

Comparing two recordings of the same fixed camera was the strongest check
available - a fixed camera must produce the same answer twice. It was retired
deliberately, for two reasons that are properties of the world rather than of the
code: **the camera may have been re-aimed between recordings, and the court's
wear marks change.** Two recordings disagreeing no longer means the algorithm was
wrong. The phase-correlation test that was supposed to detect a moved camera also
turned out to read only the peak's *location* and discard its response, so a
re-aimed camera passed it at (+0.01, -0.03) px while the response fell from 1.000
to 0.186.

## 10. The centre line versus the net

The remaining failures all had the same shape: the centre walk climbing the net's
white strap instead of stopping at the far service T. **Five fixes were tried and
all five were net-negative.** They are worth keeping because their reasons are
properties of the problem:

- Rough flanks are the *normal* reading at the far service T - the line ends
  against another line with the court boundary a few pixels beyond - so any rule
  keyed on end-roughness fires on the plates that were already right.
- Suspending the stop rules over the net also suspends the one that finds the
  end, and the walk forgets it ever reached one.
- A suspension is a licence to escape: with the rules off, one plate ran 129 px
  past the end of the court and folded its camera.

What worked was the crossing latch, and specifically **not** the version of it
that ends the walk on the crossing - crossing a line is itself a gap, and that
version cost 18 accepted plates. Nor a fixed number of clean samples after the
crossing: 8 repairs 19 of 22 regressions and 24 repairs all of them, but 24 steps
is longer than one venue's entire far half. Only "is there floor beside the line
again" is venue-independent.

## 11. One threshold cannot see a whole frame

The far service line was declared "not in the image" on the three plates where it
mattered most. It was in the image. The claim came from counting pixels lighter
than the court's median L plus a margin - the same gate the edge map used - and
that gate is set from the whole frame while the light is not evenly spread over
it:

| | near half | far half |
|---|---|---|
| paint L | 237 | 130 |
| floor L | 135 | 110 |
| the court's median L, one number for both | 134 | 134 |

Against its own neighbourhood the far line stands +20 and the near line +100, and
one number covers both. There was a second gate behind the first: with the local
gate the line's edges *do* appear, and the Hough window still returned nothing,
because its rows carried 32 and 30 collinear pixels against a threshold of 35.
Two pixels' worth of votes.

Lowering that threshold everywhere cost 11 accepted plates - a spurious crossing
in the near half tightens the leash with the whole court still ahead. The
resolution was to read the same primitive at two vote counts, because latching a
policy wants precision and collecting candidates for later judgement wants
recall. That is what made the record-and-choose design of stage 3 possible.

## 12. The walk that changes lines without noticing

The last failure: at the far T the service line is the faintest thing in the
frame and the court's far boundary runs parallel a few pixels beyond it, so the
follower slides from one to the other without ever losing a ridge. It reports the
court's far edge as the service line.

Neither the walk's own rms (7.2 on a good plate, 11.5 on a broken one) nor the
independently chosen line (wrong by 60 and 222 px on two plates where the walk
was right) could referee it. What could was the walk's own seed: a quadratic
through the walked line must pass through the point it started from. 0.5-2.8 px
when the walk stays put, 19-31 px when it does not.

## 13. A sideline that outvoted the anchor it was anchored at

`13-09-39` fitted a court half a metre too wide on the right: the near service
line's painted end sat at x=1867 and the model put its own X=10.05 at x=1960, a
99 px anchor residual against a 25 px gate. The anchor was right, the fit had
seen it, and the fit had overruled it.

Not a weighting problem, though it looks like one. Forcing the anchor - sigma 3
down to 0.3 - does produce the correct court, and the line residual barely
moves, which says the two are not really in tension. The 16-point right sideline
was on the base of the **fence** rather than the base of the glass, a clean arc
a metre outside the court, and 16 samples of one wrong line outvote two right
points because least squares counts residuals. Weakening the sideline's voice
everywhere to survive this would be paying for one plate with the corpus.

What the wrong sideline could not do is reach its own corner. Every sideline
detector here is *seeded* with the corner - `side_edge` forces it into every
RANSAC hypothesis - and none of them was *checked* against it. The check has to
come after RANSAC trims the set, and that is the whole subtlety: untrimmed, the
23 points this returned reach their corner to 15 px; trimmed to the 16 that
actually lie on one curve, what is left is a 92-row stub asked to speak for the
240 rows between it and the corner, and it misses by 167 px. Over the corpus the
worst sideline on an accepted plate misses by 33, and nothing at all lands
between 54 and 167.

Refused, the plate falls through to the boundary track from the corner, which
here runs the full 346 rows down to the corner and misses it by 0.7 px. 116
plates: 104 accepted to 105, every other plate's numbers bit-identical.

## 14. The re-trace that quietly removed the redundancy

`17-02-10` closed its court 10 px and 22 px inside its own detected corners, on
both sides at once - the sidelines visibly inside the glass base, the fitted
service line cutting across its own paint rather than ending on it.

The obvious reading is that the corner anchors are outvoted and want more
weight. Measured, they do not: on the *final* line set, dropping sigma from 3.0
to 1.0 - three times the pull - moves the right anchor from 22.4 px to 21.5.
Only sigma 0.5, which claims the corner is known to half a pixel against a
measured 1.8-3.1, forces it, and then only by making the solver reject the
sideline as an outlier. Weight was never the lever.

It was not the anchors' fight to lose, because they had already won it. The fit
*before* the re-trace sits 4.2 px and 9.2 px from its corners. The two
refinement rounds are what walk the court inward, because `retrace` re-measures
the sidelines over the near half only - a colour step in the vignetted corner -
and on 28% of the sidelines in this corpus it comes back with too few samples to
keep. Losing both is survivable: the round then has fewer than four lines and is
skipped. Losing exactly ONE leaves two lines along X, no cross-ratio, and an X
scale free to drift.

The general fix is to stop treating a refinement round as automatically an
improvement. Every round is scored by the four gates the verdict already uses,
normalised by themselves so the worst one is the score, and the best round is
kept - ties to the later round, so the re-trace still wins wherever it is doing
its job. No new constant, and nothing tuned: it is the verdict deciding which of
its own candidates to submit.

The one honest cost is that the accepted camera is now the best of three by the
same gates that judge it, so those numbers are slightly optimistic. Over 116
plates the trade is probe p50 1.30 -> 1.24 and p90 1.91 -> 1.78, worst anchor
24.5 -> 22.7, and 105 accepted -> 107, the two gained having been refused on
probe residuals of 2.01 and 2.14 while holding a 1.53 and a 1.75.

## 15. The one absolute threshold left, and why fixing it did not pay

§14 leaves an obvious question: why does the re-trace lose the sidelines at all?
It has a specific answer, and the answer is a genuine defect.

`image._peak_edge` looks for the **0.5 crossing** of the surface score. That
score is a sigmoid of Lab chroma distance from the court's own median, so it is
near 0 outside the court and *ought* to be near 1 on it. Measured along
`17-02-10`'s sidelines it is 0.60-0.75 near the net, 0.40-0.57 by the near
service line, and 0.27-0.47 on the right. Outside it is 0.02-0.05, so the step
is there and it is huge - but the level being hunted for is one the court itself
never reaches. The left sideline's trace makes 166 attempts and gets its first
hit on the 139th; the right gets 12 hits out of 152.

Which is precisely the failure `track_side` was written to avoid - "an absolute
colour model tops out below the threshold in the vignetted corner where the
track has to start" - and principle 2 of this document. The walk's `edge_at`
takes its reference from the profile's own inner samples; the re-trace's edge
finder was the one place the lesson had not been applied.

Making it relative works, as a measurement. Referenced to the profile's own
court level and keeping every other rule unchanged, `17-02-10`'s sidelines go
from 0 and 12 samples to 159 and 120, and across the corpus the number of plates
fitted with the full three lines along X rises from 95 to 112 of 116.

**And the answer gets no better.** 107 accepted becomes 105; probe p50 1.24
becomes 1.29; the anchors improve only slightly, p90 13.5 to 12.7. The reason is
structural rather than a matter of tuning: a re-trace seeded from the current
model is self-confirming, so 150 samples entrench that model far more firmly
than 12 did. Two controls confirm it. Making the finder relative and then taking
the last round unconditionally - the pre-§14 behaviour - scores **100**, the
worst of anything measured. Dropping the sideline re-trace entirely and always
keeping the walk scores 104, with one plate at a 167 px anchor residual that the
re-trace had been rescuing.

So neither source is reliably the better one, and that is exactly why the guard
that works is the one at the level of the *fit* rather than of the line. The
absolute threshold is still there, still wrong in principle, and now measured:
it is worth fixing the day something else depends on those samples, and it is
not worth two plates today.

## 16. The walk was never on the middle of the line

Every line in this document is followed by re-centring on a ridge crest, and the
question of *which* crest was settled long ago: `image._peak_ridge` takes the
**nearest** local maximum above the contrast floor, not the brightest, because
over a wide acquisition window the brightest one is the net's tape rather than
the far service line 25 px away.

That rule is right for telling one line from its neighbour and wrong for placing
a sample inside a single stripe. The near service line is 8-9 px of paint across
most of its length; its top-hat there is not a peak, it is a plateau with a few
units of ripple on it and several local maxima in a row. The nearest one to the
previous sample is whichever the walk was already standing over, so the choice is
made once, at the anchor, and then carried for 900 px. Measured against the two
Canny edges of the paint itself, the walked samples sit 1.0-2.4 px off the middle
of their own stripe, at a near-constant offset within a plate.

The signature is what makes this a defect rather than a property of the paint:
**the sign flips between two recordings of the same fixed camera**. Nothing
moved, the light changed slightly, the walk latched onto the other side of the
plateau, and the line moved 4 px. A real feature does not do that.

And on a lower-contrast court the same plateau produces the opposite symptom.
`13-34-53` is a grey surface where the follower changes its mind at every step
instead of sticking: its near service line is a median **0.00** px off centre
and scatters 0.90 px about it, its centre line 1.49 px with a worst sample at
6.6, and the drawn points cross from one edge of the stripe to the other and
back. Same defect, no bias at all - which matters, because the first version of
this fix could only see the bias.

The fix does not touch the follower. The stripe's two edges are the one thing
about it that is not a plateau, and `white_edges` - already built, for a
different purpose - puts them within a pixel on every sample. Write a sample as
the paint's centre plus the follower's error, `p = c + e`, and the two distances
it measures outward are `a = w - e` and `b = w + e`. The two combinations of
those are not equally useful:

```
(a - b) / 2  =  -e      the follower's error, exactly, at this sample
 a + b       =  2w      the stripe's width, with e algebraically absent
```

So **the midpoint is not something to fit** - it is already the answer wherever
both edges were found, and the walk's error cancels on its own whether it is
latched or scattered. What gets the RANSAC quadratic is the **width**, and it is
fitted in order to decide which pairs to believe rather than to produce the
answer: a pair that has picked up a crossing line - the centre line meets the
near service line at the anchor and the net lower down - measures a width that is
nothing like the stripe's, while a pair straddling the true stripe measures the
right width no matter how far off centre the sample it came from is. That is what
makes a 1.5 px tolerance meaningful on the width and meaningless on the midpoint.

Getting that the wrong way round is the mistake this went through. Fitting each
*side* and moving the line to the midpoint of the two curves is the obvious
reading of "RANSAC fit a curve to both sets of points", it is what was built
first, and it is a low-pass filter on the correction: it removes the smooth part
of the error and leaves the rest. On `13-34-53` it reported moving the line
0.4 px and left a 0.90 px scatter completely untouched, which is exactly what the
picture showed. Per-sample midpoints take the scatter out too - 0.90 to **0.18**,
and the centre line's 1.49 to **0.12**.

Two details still earned their place. Where there is no credible pair - one edge
missing, or a width that does not belong to this stripe, 1-6% of a walk - the
correction the rest of the line implies is used, from a quadratic through the
midpoints that were credible. And past the outermost credible sample that curve
is **held rather than extrapolated**: the ends of the near service line are the
corners the sidelines hang from.

It applies to the two wide lines and declines everywhere else on its own
evidence: the far service line measures 1.2 px wide on all 116 plates and its
samples are already a median 0.12 px from the midpoint of their own edges, so the
width test turns it down. The sidelines and the net line are not paint.

After it, the residual off-centre is a median 0.00-0.12 px - the measurement
floor - on every plate in the corpus, and against the independent paint probe:

| | before | fitted sides | per-sample midpoint |
|---|---|---|---|
| probe rms p50 | 1.258 | 0.702 | **0.626** |
| probe rms p90 | 1.758 | 0.878 | **0.751** |
| probe rms max | 2.043 | 1.451 | **1.168** |
| line rms p50 | 1.001 | 0.961 | **0.591** |
| line rms p90 | 1.331 | 1.272 | **0.798** |
| repeatability, stable pairs | 1.82 | 1.80 | **1.64** |

The line residual is the one that moves most, and it should be: it is the only
one of these that measures per-sample scatter rather than where the line sits.

Two things this did *not* do, both worth saying. It barely moved the anchor
residuals - p50 5.47 to 5.59, worst 22.8 to 22.6 - because a perpendicular
correction does not change where along a line the paint stops, which is what
`ANCHOR_OUT` and `FAR_T_SHORT` measure. And accepted went **110 to 108**: three
plates crossed the 85% coverage gate the wrong way while their probe residuals
improved, and one crossed it the right way. Every one of those losses is
`paint_probe` abstaining because a second bright object entered its ±20 px
window, not because the paint is missing or the model is off it - corpus-wide,
coverage is flat at 0.979 -> 0.976, nine plates losing a probe and four gaining
one. The gate is doing what §15 already noted: measuring how much of the court
could be *seen*, on a rule that is knife-edged when something else is nearby.
Widening it to recover the three would be tuning the judge to the answer, so it
stays.

## Where it stands

On a 116-plate corpus spanning many venues: **108 accepted**, at a probe residual
of 0.63 px median and 1.17 px worst, and a line residual of 0.59 px median
against a 2.00 px gate. Every one of the eight refusals is on coverage and
nothing else - five at 84% against an 85% gate, three at two venues whose far
half is faint enough to stay in the 53-68% range even though the fitted court is
geometrically correct. All eight probe under 0.95 px and fit under 0.86 px rms.
Not one is refused on its anchors, its line residual or its probe: every refusal
is a gate on how much of the court could be *seen*, not on where the fit put it.

Known limits, stated rather than hidden: the net-position check has a 1.2 m
spread on fits independently confirmed correct, so it catches gross failures and
nothing finer; the sidelines are extrapolated to their near corners, which is the
weakest part of the geometry; and coverage is not weighted by how much of each
line is actually visible, which is what holds those last three plates below the
gate.
