"""Preference Bayesian optimization - the search engine behind CNPro A/B.

WHAT PROBLEM THIS IS
--------------------
A CNPro configuration has no loss function. "Better" is whatever the person
looking at the image says it is, they can only say it about images they have
actually seen, and each opinion costs two full generations plus their
attention. Meanwhile the space is combinatorial: four models x six profile
offsets x five prompts x two LoRA weights on a 0.05 grid is already ~5 million
points. Sweeping it is not slow, it is impossible - which is exactly why this
is not an X/Y grid with more axes.

So the user's taste is treated as a latent UTILITY f over the configuration
space, given a Gaussian process prior; every graded comparison is evidence
about f; and the next pair to ask about is chosen so that the answer is worth
the most. Three consequences shape everything below:

* **The observation is a comparison, not a value.** People are reliable about
  "this one, not that one" and unreliable about "7 out of 10", so the model
  never assumes a grade means the same thing twice - only that a higher grade
  means the second image was preferred more strongly.
* **f is never observed.** Only its ordering is, so the posterior is not
  Gaussian and is approximated (Laplace) rather than written down.
* **Queries are the scarce resource.** Which pair to ask about is as much of
  the algorithm as the model is - see `next_duel`.

THE MODEL
---------
    f ~ GP(0, k)                        latent utility over configurations
    z = (f(B) - f(A)) / (sqrt(2)*s)     a duel's signed utility gap
    P(B preferred) = Phi(z)             Thurstone / probit comparison model

with `s` the scale of the noise in a single human judgement. A grade of 0..10
enters as a SOFT LABEL p = grade/10, i.e. the log-likelihood of one duel is

    p*log Phi(z) + (1 - p)*log Phi(-z)

which is the cross-entropy of a fractional preference. This is what makes 5 =
"cannot decide" informative rather than discarded: it is the statement
f(A) = f(B), and it pulls the two configurations' utilities TOGETHER - "both
were great" and "both were awful" say the same thing about the difference,
which is the only thing a comparison can ever say. 0 and 10 are clamped to
p = 0.02 / 0.98: a human who is certain is still occasionally wrong, and an
un-clamped 0 asks the model for an infinite utility gap on one sample.

Both terms are concave in z (log Phi is), so the negative log posterior is
convex and the Laplace mode is unique - Newton's method below converges to the
one answer rather than to whichever local optimum it started nearest.

THE KERNEL, AND WHY MIXED SPACES ARE THE WHOLE POINT
----------------------------------------------------
Dimensions are of two kinds and they are not interchangeable:

* a **choice** dimension (models, prompts, profile lines) has no metric - two
  models are simply different - so its kernel factor is an OVERLAP kernel:
  1 when equal, exp(-theta) when not.
* a **range** dimension (a LoRA weight) is a number, and 0.55 genuinely does
  tell you something about 0.60 - so its factor is the usual squared
  exponential.

...with two refinements that matter in practice. A choice dimension whose
labels are all NUMBERS (an offset list `-0.2, -0.1, 0, 0.1`) is encoded as a
range even though the user only gets to pick from the list. The values are
ordered and the ordering is real; treating them as unrelated categories throws
away the single most useful piece of structure the user handed us. And a
weight dimension can be CONDITIONAL on a pick dimension (`Dimension.parent`):
a LoRA slot's weight means nothing across two different LoRAs - the good
weight for one is not evidence about the other - so between two points whose
picks differ the weight contributes nothing to the kernel, and each pick
learns its own weight response from its own observations alone.

The similarity rows calibrate this kernel as well as the diversity filters.
Their noisy-OR rates act as perceptual ARD: evidence transfers farther across
a dimension the user calls visually inert and less across one they call vivid.
A conditional weight has one visual rate per parent pick, matching its utility
semantics. With no similarity labels every rate is exactly 1, the prior kernel.

CHOOSING THE NEXT DUEL
----------------------
Two independent samples are drawn from the posterior over a candidate pool and
each one's argmax becomes one side of the duel (dueling Thompson sampling, aka
self-sparring). This is worth more than it looks:

* it needs no hand-tuned exploration constant - the exploration comes from the
  posterior's own uncertainty, which shrinks exactly where questions have been
  answered;
* it naturally produces CHAMPION vs CHALLENGER duels once the posterior is
  confident, because the incumbent is the argmax of most samples - so the user
  keeps seeing the current best rather than two configurations they have
  already rejected; and
* it degrades gracefully: with no data the samples are prior draws and the
  duels are random, which is exactly the right behaviour at the start.

Thompson sampling alone is a SINGLE-ANSWER machine, and taste is not a
single answer - a person who likes two unrelated looks is the normal case,
not a degenerate one. Three mechanisms widen it without giving up
convergence, on a fixed cycle (see EXPLORE_CYCLE):

* **The frontier.** `frontier()` is the diverse top of the posterior: its
  public result is capped at FRONTIER_SIZE readable recipes, but acquisition
  retains up to ATTRACTOR_ARCHIVE_SIZE polished basins internally, each at
  least FRONTIER_SEPARATION from the others. The candidate pool is built
  around the full archive rather than one incumbent or the four-entry report,
  so a fifth strong basin does not starve because of a presentation limit.
* **Cross-basin duels.** Every cycle one duel is champion-of-one-basin vs
  champion-of-another. Comparisons are the only thing that anchors two
  regions' utilities to the same scale; without these, two basins that were
  each polished against their own neighbours drift apart and the ranking
  between them is fiction.
* **Uncertainty duels.** Every cycle one duel is the champion vs the pool
  point the posterior knows least about. Thompson exploration shrinks as
  the posterior sharpens - which is right near the top and wrong far from
  it, where "never asked" and "confidently bad" would otherwise become
  indistinguishable.

Three further tactics are STOCHASTIC - each fires at a fixed MEAN rate
(see BLUFF_MEAN / VOID_MEAN / RIVAL_MEAN) rather than on the cycle, so
their arrivals cannot be anticipated, cannot phase-lock with the
schedule, and cannot rut into a beat the rest of the search learns to
compensate for. They exist because every mechanism above is a function
of the history, and a search fed only by its own history can lock onto
local similarities it has no way to see past:

* **The bluff** (1 duel in ~7): one pair drawn uniformly at random - no
  posterior, no locality, no history. Deliberately "suboptimal", like a
  poker bluff: every other query's answer partly echoes what the model
  already believes, and the random probe is the one question whose
  answer cannot - novel information that no history-driven acquisition
  would ever have asked for.
* **Void probes** (1 in ~12): the space is scanned for regions nothing
  ever shown comes near, and the duel is placed inside them - the two
  sides preferably in two DIFFERENT voids. The uncertainty duel asks
  about the least-known POOL point, and the pool is itself shaped by
  the history; this asks about the parts of the space the whole session
  has simply never visited.
* **Rival samples** (1 in ~14): two attractors are drawn and each side
  is a fresh GOOD sample from one of them - the attractors compared
  between themselves through their populations, not only through their
  champions (which is the cross-basin duel's job). Two basins whose
  champions are settled can still hide that one is broad and reliable
  while the other is a single lucky point.

TWO DUELS ARE NEVER RE-ASKED PAST A CAP. Asked pairs are remembered and a
pair that has been graded MAX_PAIR_REPEATS times is skipped when the next
challenger is chosen: a third asking of the same question measures the
user's consistency, not their taste, and it is exactly the "keeps showing
me the same two images" failure a session-long search cannot afford.

THE PRICE OF A QUESTION
-----------------------
Every duel shows two images, but not every image costs a generation: the
host caches renders by recipe, so a configuration already rendered this
session is roughly TWENTY TIMES cheaper to show again than to generate
fresh (~30s a generation against a lookup). The engine learns what is
currently reusable through `reuse_probe`, a host-injected callable
(point -> bool); with no probe it is cost-blind, which is exactly the
behaviour everything else in this docstring describes.

What the probe buys is a swap rule in the Thompson slot, and only there:
either side may be replaced by a reusable candidate whose SAMPLED utility
lags the ideal pick by less than REUSE_MARGIN units of the learned
judgement noise. That margin is the whole argument: a deficit smaller
than the noise of a single human judgement is a difference the grader
could not resolve anyway, so the cheaper question is - to the person
answering it - the same question, at a twentieth of the wait. Information
per second, not information per duel, is what a 30-second generation
makes the right objective.

What the rule refuses to become is an all-to-all tournament of the cache,
because 20 reused comparisons cost the time of one new pair - and all of
the user's attention with it. Three guards: the margin itself (relative
to the noise, never an absolute "close enough"), the repeat cap (a cached
PAIR is still never re-asked past it), and REUSE_STREAK_MAX - after that
many consecutive duels in which both sides came from the cache, the
preference switches off until a duel brings a fresh image, because only
generation produces genuinely new evidence about new configurations. The
exploration slots (seeds, bluff, void, rival, cross-basin, develop,
uncertainty) are never biased: their value is where they land, not what
they cost.

THE TREND MODEL - A SECOND OPINION THAT ONLY PROPOSES
-----------------------------------------------------
A posterior evaluation costs microseconds; a generation costs ~30s. That
asymmetry means practically unlimited model evaluations are worth one
avoided generation, so a SECOND model - cheap, global, deliberately
different in its bias - screens the space at a scale the duel loop never
could: a ridge-regularized additive model (one-hot per opaque category,
value and value-squared per numeric dimension, and per-parent-pick weight
curves for conditional weights) fitted to the GP's own latent utilities
at the observed points.

Different bias on purpose. The GP's product kernel is local: about a
combination of levels that were each seen only in OTHER company it can
say little, because the overlap factors discount across every differing
category at once. The additive model extrapolates main effects to
never-visited combinations - "this model kept winning, that factor kept
winning, so try them together" - which is exactly the guess a human would
pencil in and the product kernel cannot.

Symbiosis, not authority: each duel the trend model scores
SURROGATE_BATCH uniform draws in one matrix multiply and nominates its
top slice into the candidate pool (SURROGATE_SHARE), where the GP
posterior and Thompson sampling still decide everything - nominees win
duels or vanish, exactly like the interesting-mark hybrids. A wrong trend
therefore costs pool slots and nothing else; a right one jumps the search
to a promising unvisited combination without paying a generation per
intermediate step. It is gated behind SURROGATE_MIN_DUELS because a trend
fitted to three comparisons is a guess wearing a lab coat.

THE THREE ROWS, AND ABSOLUTE QUALITY
------------------------------------
A single comparison cannot say "both of these are bad" - "both great" and
"both awful" are the same statement about the difference. The ROW supplies
the missing absolute statement. Top and middle both say "on track" and add a
soft above-par probit observation for EACH side; bottom says both are unusable
and adds a below-par one for each. The 0..10 grade still orders the pair inside
either region. Nothing extra is asked: the click already lands on exactly one
of those rows. This anchors both ends of utility, lets bad subspaces be avoided
without ranking their interiors, and prevents N-GOOD from calling the least
awful member of an all-bad session "good" merely because every set has a
relative winner.

SIMILARITY: WHAT A COMPARISON CANNOT SAY
-----------------------------------------
A graded duel encodes a DIFFERENCE in utility. It cannot encode a DISTANCE -
"these two are the same image" and "these two are equally good" are not the
same statement, and no amount of grading produces the first. So the panel's
three grade rows partition the answer differently: the top row is "these are
rather distinct, and we are on track", the middle is "rather SIMILAR, and we
are on track", the bottom is "I will rate them but both are bad". One click
still answers, whichever row it lands on.

The middle row is the new channel, and it is the cheapest click in the panel:
"do these look alike" is a first-glance impression that costs no deliberation
at all, and because the top two rows PARTITION the on-track case, every
graded duel yields a similarity label rather than only the ones somebody
bothered to mark. That density is what makes the label worth fitting on.

WHAT IT BUYS. Separation - "how far apart are two configurations" - was a
hand-written prior: a differing category counts 1, a range dimension counts
the fraction of its span. It is what decides whether a duel is worth asking
(MIN_DUEL_SEPARATION), how different the frontier's keepers have to be
(FRONTIER_SEPARATION), which configurations a collage may hold together
(`population`) and how many distinct good answers the space has (`capacity`).
All eighteen of those tests call one function, so making that function
LEARNED upgrades every one of them at once. Numeric choice lists use their
actual normalized spacing; conditional weights contribute only under a shared
parent pick and learn one rate per pick. `_fit_metric` shrinks every rate
toward the old constant so a session with no labels keeps the prior.

The model is noisy-OR: each dimension independently has some rate of
producing a visible change, distance is the sum of those rates, and
P(distinct) = 1 - exp(-distance). Concave in the distance, linear in the
rates, roughly one parameter per row (one per parent level for a conditional
weight) - so it fits from a handful of labels and cannot overfit into anything
exotic.

TWO HONEST LIMITS. Outside declared pick/weight relationships the metric is
LINEAR IN THE DIMENSIONS, so it cannot learn arbitrary visual interactions;
that is a real loss and interaction terms are the escape hatch if it ever
bites. And the labels are CENSORED: every duel path filters on separation, so a dimension
the metric has shrunk stops being asked about and can never be un-shrunk by
ordinary play - the estimate would be self-confirming. `_probe_duel` is the
answer to that, and the reason it exists: at a fixed mean rate it asks
exactly the duel the learned metric now refuses and the prior would have
allowed, so a wrongly-shrunk dimension gets the one question that can
restore it. The weights are also floored (METRIC_MIN) so nothing can vanish
outright.

A similarity label pays two dividends on the UTILITY side. Two configurations
that look the same must be worth the same, so the middle row records an extra
tie; and the fitted visual rates enter the GP kernel itself, deciding how far
the grade transfers through each parameter rather than only filtering future
pairs.

"INTERESTING" IS NOT A GRADE
----------------------------
The panel's per-image toggle marks a sample whose OVERALL quality is
whatever the grades said - usually bad - but which touches a characteristic
the user wants to see in good samples. The mark therefore never enters the
likelihood: no observation, no utility, the sample stays exactly as bad as
it was graded. It enters the ACQUISITION instead, as a donor: a share of
every candidate pool is hybrids that transplant a few of the donor's semantic
genes into an attractor - the tempting characteristic, tried inside a good
configuration - and the hybrids then earn their place through ordinary duels
or vanish. A pick and its conditional weight transplant together. See
mark_interesting / _hybrid / INTERESTING_SHARE.

THE SEARCH IS ALSO A GENERATOR
------------------------------
A converged solver is not one recipe, it is a REGION - and a region has a
shape as well as a size. Three questions are asked of it, and they are one
piece of machinery:

    islands()      how many SEPARATE good regions are there? Components of
                   the good set, joined by hops the fitted similarity model
                   cannot tell apart, so "isolated" means isolated to the
                   eye rather than distant in parameters.
    capacity()     how many mutually DISTINCT samples can be had? The count
                   the collage button wears - "3 GOOD" - obtained by running
                   that button's own selection and taking its length, not by
                   estimating a related quantity.
    population(n)  give me n of them. Island by island, round robin, every
                   pair at least COLLAGE_CONFIDENCE likely to LOOK
                   different, ranked by a fresh posterior DRAW so that
                   asking twice gives two samples of one taste.

All three share `_good_pool` and one distance, and the distance is a
probability the user's own similar/distinct clicks calibrated. That last
point is the whole reason the numbers mean anything: a bar written as a
distance can be - and was - set to a value at which the model itself
expected 82% of the pairs to look identical.

The good pool is absolute as well as relative: a candidate must be supported
above par and within one judgement-noise margin of the champion. The latter is
computed from the posterior variance of their DIFFERENCE, including their
covariance; marginal variances cannot describe that event.

Nothing in this module imports gradio, the host, or the rest of CNPro. It is
numpy and stdlib, so `tests/test_ab_search.py` runs it directly.
"""

import math

import numpy as np

try:
    # scipy rides along with the host and is ~40x faster on the tail values
    # this uses. Nothing here NEEDS it - the fallback is exact, just slower -
    # so its absence must not take the search down with it.
    from scipy.special import erfc as _erfc
except Exception:  # pragma: no cover - exercised only without scipy
    _erfc = np.vectorize(math.erfc, otypes=[float])

#: Utility gaps are clamped to this many "judgement noise" units before the
#: likelihood sees them. Phi(-8) is 6e-16, which is representable and whose
#: log is -35 - so the clamp costs nothing in the range that matters and buys
#: two things: the tail arithmetic can never underflow to a zero denominator,
#: and one confidently-wrong grade cannot drag a whole region of the space
#: with it (the gradient saturates instead of growing without bound).
Z_CLIP = 8.0

#: How far a grade of exactly 0 or 10 is pulled off certainty. See the module
#: docstring: a human who is sure is still sometimes wrong, and p in {0, 1} is
#: a likelihood with no finite maximum.
P_CLAMP = 0.02

#: Newton iterations for the Laplace mode, and the convergence threshold on
#: max|df|. The objective is convex and n is small, so this converges in a
#: handful of steps; the cap is a guard, not a schedule.
NEWTON_STEPS = 40
NEWTON_TOL = 1e-7

#: Hyperparameters are re-selected by marginal likelihood every this many
#: observations. Every observation would be affordable too (n is tiny), but
#: the selected values swing early on and a search whose kernel changes under
#: it every single duel produces duels that look erratic to the user.
HYPER_EVERY = 3

#: The grids marginal likelihood picks from. Deliberately coarse: with a dozen
#: comparisons there is not enough evidence to justify a fine grid, and a
#: hyperparameter fitted to noise is worse than a reasonable constant.
#: - LENGTHSCALES are in units of the normalized [0, 1] range dimensions
#: - THETAS: exp(-theta) is how much utility two different categories share,
#:   so 0.5 -> 0.61 ("models behave somewhat alike"), 4.0 -> 0.02 ("nothing
#:   transfers")
#: - NOISES is the scale of a single human judgement, in utility units
#:
#: THE GRID IS NARROW BECAUSE THE SELECTION IS BIASED, and that was measured
#: rather than assumed. Marginal likelihood (ML-II) picked the SMOOTHEST, most
#: forgiving corner of every grid it was offered - on 8 runs out of 8, at
#: three different grids. That is the small-sample Occam effect: with a dozen
#: comparisons, "nothing much matters and the human is noisy" explains the
#: answers while spending the least prior mass, so it wins - and it is exactly
#: the model that cannot tell two models apart, whose posterior is nearly flat,
#: and a flat posterior recommends almost anything.
#:
#: A log-normal prior over the hyperparameters (MAP-II instead of ML-II) was
#: written and measured against this: it did move the selection off the corner,
#: and it did NOT improve the configuration found (better at 16 duels, worse at
#: 32, across 8 seeds). So it is not here. What is here instead is a grid whose
#: smooth end is still a usable model: at lengthscale 0.5 a knob's neighbours
#: are related without being interchangeable, and at theta 0.5 two categories
#: share 61% rather than 74%. Widening either end again re-opens the same hole.
LENGTHSCALES = (0.15, 0.3, 0.5)
THETAS = (0.5, 1.5, 4.0)
NOISES = (0.4, 0.9)

#: Candidate pool size per duel. The pool is what Thompson sampling maximizes
#: over, so it is the resolution of the search: too small and the argmax is
#: noise, too large and the m x m posterior covariance stops being free.
POOL_SIZE = 320

#: How much of the pool is drawn NEAR the incumbent rather than uniformly, as
#: a function of how many comparisons are in. This is annealed rather than
#: constant because the two ends of the search want opposite things and both
#: were measured to matter:
#:
#:   a pool that is mostly local from the start   finds a worse answer early
#:                                                (it polishes the first thing
#:                                                it likes)
#:   a pool that stays uniform to the end         never polishes at all - in a
#:                                                million-point space a uniform
#:                                                draw agrees with the incumbent
#:                                                on nothing
#:
#: So it starts near-uniform and tightens as evidence accumulates, which is
#: also what the person grading sees: unrelated configurations at first,
#: variations on a theme once the theme is established.
POOL_LOCAL_BASE = 0.05
POOL_LOCAL_GROWTH = 0.015
POOL_LOCAL_MAX = 0.45

#: Duels before the model is trusted with the choice. Under three comparisons
#: the posterior is essentially the prior and Thompson sampling would be an
#: expensive way to pick at random - so it picks at random directly, which at
#: least lets the pairs be chosen for SPREAD.
SEED_DUELS = 3

#: How different the two sides of a duel have to be before it is worth asking.
#: One differing category counts 1; a range dimension counts the fraction of
#: its span that separates them.
#:
#: WITHOUT THIS, A QUARTER OF THE DUELS ARE UNANSWERABLE, and that is measured:
#: in a space of one LoRA row (which LoRA, and its weight) 19% of duels came
#: back differing by a SINGLE 0.05 weight step, and in a space of one weight
#: alone, 25% did. Two images from weights 0.60 and 0.65 are the same image.
#: The user is then asked to grade a difference they cannot see, and whatever
#: they answer is noise the model has to spend later duels unlearning.
#:
#: 0.2 is four steps of a 0.05 knob - the point where a LoRA weight starts to
#: show. It costs the search the ability to ask about finer differences, which
#: is not a loss: a person cannot answer those, so the comparison was never
#: worth a generation. Any categorical difference clears the bar on its own.
MIN_DUEL_SEPARATION = 0.2

#: How often a duel that has already been graded may be asked again. Twice is
#: replication - a noisy judgement is worth confirming once - and a third time
#: is the search measuring the user's consistency instead of their taste. The
#: cap is per PAIR, so the champion still re-fights (against new challengers)
#: as often as the posterior wants it to.
MAX_PAIR_REPEATS = 2

#: How far a REUSABLE side may lag the Thompson pick it replaces, in units
#: of the learned judgement noise s. The trade is only ever made inside
#: this margin, and the unit is the argument for it: a sampled-utility
#: deficit under one judgement's noise is a difference the grader cannot
#: resolve, so the cached question is - to the person answering - the same
#: question at ~1/20th the wait (see THE PRICE OF A QUESTION). Written in
#: noise units rather than raw utility because the posterior's scale
#: breathes with the anchors: an absolute margin would silently mean
#: "anything" early and "nothing" late.
REUSE_MARGIN = 0.75

#: Consecutive duels in which BOTH sides came from the cache before the
#: reuse preference switches itself off. Reused duels are 20x cheaper in
#: generation time but full price in the user's attention, and only a
#: generation produces new evidence about a new configuration - so after
#: this many all-cached duels in a row the preference stands down and the
#: Thompson choice runs cost-blind until a duel brings a fresh image. The
#: cap removes the PREFERENCE, it does not force freshness: Thompson
#: remains free to ask an all-cached question when that is genuinely the
#: best question, which is what keeps the guard from costing convergence.
REUSE_STREAK_MAX = 3

#: The trend model's knobs - see THE TREND MODEL in the module docstring.
#: MIN_DUELS gates it: under a handful of comparisons the "trend" is
#: noise, and nominating from noise spends pool slots that the uniform
#: fill uses better. BATCH is how many uniform draws it screens per duel -
#: large on purpose, because a screen is one matrix multiply and the whole
#: reason the model exists is that its evaluations are ~free next to a
#: 30-second generation. SHARE is the slice of the pool its nominees get:
#: modest, like the interesting-hybrids', because nominees are proposals
#: to be tested, not answers. RIDGE is the regularization - the features
#: are all 0..1, so one constant serves every space.
SURROGATE_MIN_DUELS = 5
SURROGATE_BATCH = 4096
SURROGATE_SHARE = 0.15
SURROGATE_RIDGE = 1e-2

#: The duel schedule, as a cycle over graded duels. Within each cycle of this
#: many duels, one is a CROSS-BASIN duel (two attractors' champions - the only
#: thing that keeps separate basins on one utility scale) and one is an
#: UNCERTAINTY duel (champion vs the point the posterior knows least about -
#: the guard against "never asked" quietly becoming "confidently bad").
#: The rest are ordinary Thompson duels, which is what converges.
EXPLORE_CYCLE = 5

#: The stochastic tactics, as MEAN periods. One roll of a single coin per
#: duel is partitioned into three bands of width 1/period, so each tactic
#: fires with exactly its own probability, at most one fires per duel, and
#: the arrivals are geometric - "on average every Nth duel", never exactly
#: every Nth. On average rather than on schedule on purpose: a fixed beat
#: could phase-lock with EXPLORE_CYCLE (a tactic forever landing on a slot
#: another mechanism already owns), and a predictable probe is one the
#: rest of the schedule quietly organizes itself around. A tactic whose
#: coin fires but which has nothing valid to ask falls through to Thompson
#: like every other slot, so none of them can stall the search.
#:
#: BLUFF - a fully random pair, the poker bluff. Every other query is a
#: function of the history and its answer partly echoes what the model
#: already believes; the uniform pair is the one question that cannot.
#: VOID - a pair from the largest region(s) of the space nothing shown
#: comes near, preferably one side in each of two different voids.
#: RIVAL - fresh good samples from two different attractors, so basins
#: are compared through their populations, not only their champions.
BLUFF_MEAN = 7
VOID_MEAN = 12
RIVAL_MEAN = 14

#: What counts as a void: a candidate at least this far (in separation
#: units - 1.0 is one whole categorical change, or a range dimension's
#: full span) from EVERYTHING ever shown. Once the session has covered a
#: space densely enough that nothing clears the bar, the tactic retires
#: itself - correct, because a space with no void left has nothing for
#: this tactic to find, and its coin's duels go back to Thompson.
VOID_SEPARATION = 1.0

#: Uniform draws scored per void probe. More than SEED_BATCH because the
#: void is found by the EMPTIEST of the batch, and in a large space a thin
#: batch's emptiest candidate is usually just a mediocre gap.
VOID_BATCH = 128

#: Neighbourhood samples drawn per attractor when a rival duel picks each
#: basin's best current member.
RIVAL_SAMPLES = 16

#: How many diverse near-optima the frontier keeps polished, and how far
#: apart two of them have to be to count as different answers rather than the
#: same answer twice. 0.5 is half a range dimension's span; any categorical
#: difference clears it on its own.
FRONTIER_SIZE = 4
FRONTIER_SEPARATION = 0.5

#: Basins retained internally for acquisition and N-GOOD. FRONTIER_SIZE is a
#: presentation limit - four recipes are enough for a STOP report a human can
#: read - and must not be the model's memory limit. In a fifteen-degree space
#: it is entirely plausible to have more than four supported modes; dropping
#: the fifth from every local pool makes it decay from under-sampling rather
#: than evidence. The archive is refreshed once per graded duel and cached, so
#: widening it does not multiply status()/capacity() work.
ATTRACTOR_ARCHIVE_SIZE = 12

#: THE COLLAGE'S PROMISE, WRITTEN AS A PROBABILITY. Every pair of entries
#: N-GOOD returns is at least this likely to LOOK different - not "this far
#: apart in some units", which is what every version of this before it said
#: and why every version of it was wrong.
#:
#: The similarity row makes this sayable. `_fit_metric` fits exactly one
#: model, and the model is a probability:
#:
#:     P(the pair looks distinct) = 1 - exp(-separation)
#:
#: so a distance IS a probability, and the two can be converted. Doing that
#: to the constants this used to be set to is the whole diagnosis of the
#: original complaint - N-GOOD returning N of the same image:
#:
#:     MIN_DUEL_SEPARATION 0.2 -> 18% likely to look different
#:     FRONTIER_SEPARATION 0.5 -> 39%
#:                         1.0 -> 63%
#:                         2.3 -> 90%
#:
#: A collage built at 0.2 was a set of pairs each MORE LIKELY THAN NOT to
#: look identical, by the model's own fitted rates, and the sheet came back
#: exactly as the model predicted. No amount of better selection fixes a bar
#: that says the wrong thing; the bar had to be moved into the currency of
#: the promise.
#:
#: 0.9 rather than 0.99: the last stretch is expensive in a way the user
#: pays for. P is per PAIR, so a sheet of n entries holds n(n-1)/2 of them
#: and demanding near-certainty on every one shrinks a collage of twelve to
#: a collage of three in most spaces. At 0.9 an occasional near-pair on a
#: big sheet is the price of the sheet being big, and the note says what was
#: achieved rather than leaving it to be discovered.
COLLAGE_CONFIDENCE = 0.9

#: When two GOOD configurations are the same PLACE rather than two places -
#: the hop that joins them into one island (see `islands`). Derived from the
#: same model and deliberately at the coin-flip: if the fit cannot say which
#: way a pair would be judged, they are not two answers. A hop of its own
#: would be a third constant nobody could calibrate.
ISLAND_CONFIDENCE = 0.5


def similarity_distance(confidence):
    """The learned-metric distance at which the fitted similarity model puts
    `confidence` probability on the pair LOOKING different.

    The inverse of `distinct_probability`, and the reason both exist: every
    distance threshold in this engine is really a statement about what the
    user will see, and stating it in probability is the only form in which
    it can be checked against the data that was collected.
    """
    confidence = min(max(float(confidence), 0.0), 0.999)
    return -math.log(1.0 - confidence)


def distinct_probability(distance):
    """The fitted model's chance that a pair `distance` apart looks
    different - `1 - exp(-d)`, the noisy-OR of `_fit_metric`."""
    return 1.0 - math.exp(-max(float(distance), 0.0))


#: The two thresholds above, in the units every distance test uses.
POPULATION_SEPARATION = similarity_distance(COLLAGE_CONFIDENCE)
ISLAND_HOP = similarity_distance(ISLAND_CONFIDENCE)

#: A keeper has to be WORTH KEEPING, and the test is probabilistic because
#: the utility scale is not fixed - with few absolute anchors the whole
#: posterior compresses toward 0, and any threshold written in raw utility
#: units silently changes meaning with it. So a keeper needs at least
#: KEEP_VS_CHAMPION probability of matching the champion in one judgement
#: (under the learned judgement noise), and at least KEEP_VS_PAR probability
#: of beating par. Without the floor the frontier's tail fills with the best
#: of mediocre regions - "diversity" that is really filler, measured at the
#: 44th-67th percentile of a 62k-point bench space. Quality prevails;
#: diversity is diversity OF THE GOOD.
KEEP_VS_CHAMPION = 0.25
KEEP_VS_PAR = 0.35

#: HOW MANY GOOD ANSWERS ARE THERE? - the capacity estimate, and the one
#: number the frontier deliberately cannot give. The frontier answers "which
#: few configurations should I keep", capped at FRONTIER_SIZE because a STOP
#: report a human reads has to be short. `capacity` answers the other
#: question: of the WHOLE space, how many mutually distinguishable
#: configurations does the model currently believe are good? That is not four
#: - a taste that likes one model at any prompt has as many good answers as
#: there are prompts - and it is the number the N-GOOD button spends
#: generations on, so it is estimated rather than guessed.
#:
#: Counted rather than estimated: `capacity` runs the collage's own
#: selection over the collage's own pool and reports how many entries it
#: could place. See `capacity` - a Monte Carlo estimate of the good region's
#: packing number lived here for a while, and it answered a different
#: question than the one the box is asked.
#:
#: How many Thompson orderings the selection tries, and the asymmetry
#: between the two callers is the whole point.
#:
#: A greedy maximal set is not unique: how many entries fit depends on which
#: one is taken first, and every press starts from a fresh draw. Measured on
#: one seed, presses of 2, 2, 2, 1, 1 against a box reading 2. So the two
#: callers lean opposite ways - `capacity` reports the SMALLEST count any
#: ordering yields, `population` returns the LARGEST set any ordering finds -
#: and the box under-promises against the press by construction rather than
#: by luck.
#:
#: The press leaning that way is right on its own terms too: it is about to
#: spend a generation per entry, so a few milliseconds of retrying to place
#: one more distinct sample is the cheapest work in the whole button. The
#: draws share one Cholesky, so the extra orderings are nearly free.
SELECTION_DRAWS = 3

#: What counts as good, and how sure the model has to be of it: a
#: configuration is good when its POSTERIOR puts it within QUALITY_MARGIN
#: judgement noises of the champion with probability at least
#: CAPACITY_CONFIDENCE.
#:
#: MEASURED AGAINST THE CHAMPION, IN NOISE UNITS, and both halves of that are
#: load-bearing. An absolute threshold ("utility above 0.5") means nothing
#: here: with few dislike anchors the whole posterior compresses toward 0 -
#: measured at a champion of 0.065 after fifty duels of a taste the model had
#: correctly learned - so any fixed number silently becomes "everything" or
#: "nothing" depending on how the session was graded. The margin is the gap
#: KEEP_VS_CHAMPION already implies (Phi^-1(0.25)*sqrt(2) = 0.95 noises), so
#: "good" means the same thing here as it does on the frontier.
#:
#: The probability is over the POSTERIOR ALONE, not the judgement noise, and
#: that is what makes this a confidence test rather than a preference one: a
#: configuration nobody has evidence about sits at the prior, its probability
#: is near a half however tempting its mean, and it counts for nothing until
#: the search has actually learned something about its neighbourhood.
#:
#: THE THRESHOLD IS DERIVED, NOT TUNED. k(x, x) = 1, so an unexplored point
#: sits at mean 0 with variance 1 - and at the very start the champion is 0
#: too, which makes its test value exactly Phi(QUALITY_MARGIN * s). With the
#: largest s the hyperparameter grid offers (NOISES[-1] = 0.9) that is 0.816,
#: so any threshold at or below it lets the WHOLE SPACE pass at duel one:
#: measured, and it reported 56 to 155 good configurations before a single
#: comparison had been graded. 0.85 clears it for every noise on the grid,
#: which is the property to preserve if either constant is ever moved.
QUALITY_MARGIN = 1.0
CAPACITY_CONFIDENCE = 0.85

#: A collage entry must also have posterior support for being ABOVE PAR.
#: Closeness to the champion alone is a relative statement: after a run of
#: bottom-row verdicts the least awful point is still the champion, and the old
#: keeper override admitted it to N-GOOD unconditionally. The top/middle rows
#: now provide the positive anchors this test consumes. 0.60 is deliberately
#: evidence-seeking rather than severe: "on track" is a viability verdict, and
#: a candidate already has to clear the stricter near-champion test above.
GOOD_VS_PAR_CONFIDENCE = 0.60

#: The largest capacity ever reported. A space can hold more good
#: configurations than anybody will ever render, and past a few thousand the
#: exact number stops being information - it is "more than you will use".
#: The cap keeps the panel's box narrow and the estimate honest about its own
#: resolution.
CAPACITY_MAX = 9999

#: `population` oversamples: candidates drawn per configuration asked for,
#: and the pool it is allowed to grow to. The pool is what the diverse pick
#: chooses from, so it has to hold several times the wanted count for the
#: separation filter to have anything to reject - and it is capped because
#: the pick prices a pool x pool posterior draw.
POPULATION_OVERSAMPLE = 8
POPULATION_POOL_MAX = 1024

#: THE LEARNED SEPARATION METRIC - see SIMILARITY in the module docstring.
#: One rate per ordinary dimension and one per parent level for a conditional
#: weight, scaling its contribution to `Space.separation`.
#:
#: SHRUNK TOWARD 1, which is the old hand-written metric exactly: with no
#: similarity labels every rate sits at the prior and nothing in the engine
#: behaves differently than it did before the row existed. METRIC_SHRINK is
#: the strength of that pull in the fit; the data outgrows it naturally, since
#: the likelihood grows with the label count and the penalty does not.
#:
#: FLOORED AND CAPPED, and the floor is the load-bearing half. A weight at 0
#: is a dimension the search has gone blind to - it would stop asking about
#: it, therefore stop being told about it, and could never recover (see
#: _probe_duel for the other half of that defence). 0.05 still means "four
#: differing categories here are worth one elsewhere", which is as inert as
#: anything needs to be. The cap is only there to stop one emphatic label
#: from making a single row dominate the whole metric.
METRIC_SHRINK = 2.0
METRIC_MIN = 0.05
METRIC_MAX = 3.0

#: Gradient steps and step size for that fit. The objective is smooth, tiny
#: (roughly one parameter per row), warm-started from the previous fit and
#: re-run only when a label arrives, so this is a rounding error next to one
#: generation.
METRIC_STEPS = 60
METRIC_RATE = 0.15

#: How much a "distinct" label is worth against a "similar" one. The top row
#: is the DEFAULT answer - the one an idle hand parks on - while the middle
#: row is the exceptional claim somebody had to mean. Weighting them equally
#: would let a habit of never leaving row one read as evidence that every
#: dimension is vividly visible. At 0.35 a distinct label still counts, but
#: three of them are needed to answer one deliberate "these are the same".
#:
#: THE DISCOUNT DOES NOT APPLY TO A PROBE'S ANSWER. A probe duel (see
#: _probe_duel) is shown precisely BECAUSE the metric expects it to look
#: identical, so "distinct" there is not an idle hand parking on the default
#: row - it is the model being contradicted on a question it asked. Measured:
#: discounted, ten such answers moved a wrongly-shrunk weight from 0.134 to
#: 0.170, and since probes arrive once in PROBE_MEAN duels that is a recovery
#: path of several hundred duels, i.e. none at all. At full weight the same
#: ten answers clear the gate.
DISTINCT_WEIGHT = 0.35

#: Mean period of the censoring probe - the duel the LEARNED metric refuses
#: and the PRIOR metric would have allowed. Rarer than the other tactics
#: because it deliberately asks a question that may be unanswerable ("these
#: two look identical"), which is a real cost to the user; often enough that
#: a dimension shrunk by a few early labels gets re-examined within a session
#: rather than being written off for good. Inert until the metric has
#: actually moved: with every weight at its prior there is no pair that one
#: gate refuses and the other allows, so the tactic finds nothing and falls
#: through to Thompson like any other coin.
PROBE_MEAN = 16

#: Candidates drawn per space-filling pick during the seed duels.
SEED_BATCH = 64

#: "Interesting" marks kept, and the share of the candidate pool spent on
#: their hybrids. An interesting sample is NOT a good sample - the user's
#: definition is "overall bad, we can't use it, but this characteristic is
#: tempting" - so the mark never touches the likelihood: the sample stays
#: exactly as bad as it was graded. What it does is donate: hybrid
#: candidates transplant a few of the marked configuration's semantic genes
#: into a good base (an attractor), which is literally "this characteristic,
#: in a good sample", and the hybrids then live or die by ordinary duels.
#: The share is deliberately modest and the transplant deliberately small
#: (1-3 genes): a characteristic is usually carried by few genes,
#: and flooding the pool with a bad sample's genome would be valuing the
#: mark as if it meant "good".
INTERESTING_MAX = 12
INTERESTING_SHARE = 0.15

#: A duel is a SURPRISE when a side the model gave less than this probability
#: wins with a decisive grade (<=2 or >=8). A surprise arms one immediate
#: follow-up duel for the winner - against the champion when they can be
#: told apart, else inside its own neighbourhood. This is what "a 10 is not
#: 'here we are'" means operationally, in BOTH directions: the grade neither
#: settles the search (it is one noisy comparison) nor evaporates into the
#: smoothing prior (a lone spike in hostile territory is otherwise averaged
#: away before anything acts on it - measured on the hidden-gem bench, where
#: gems were found and then LOST for exactly that reason).
SURPRISE_P = 0.35

#: What either ON-TRACK row says about EACH side against par: the probability
#: that the configuration is usable/better than an unjudged one. The row
#: wording is explicit - distinct *and on track*, or similar *and on track* -
#: and throwing that half away left the model with negative anchors only. It
#: could identify "less bad" while having no evidence that anything was good,
#: then N-GOOD force-admitted that relative winner. Kept softer than a dislike
#: (0.70 versus 0.15) because "on track" means viable, not exceptional; the
#: 0..10 comparison still decides how the two are ordered inside that region.
ON_TRACK_P = 0.70

#: What a dislike-both click says about EACH side against par: the
#: probability that the configuration beats a par one. 0.15 rather than the
#: P_CLAMP floor because a gut "I dislike both" is a real signal with real
#: noise in it - strong enough that a region marked this way twice sinks
#: well below anything merely ungraded, soft enough that one exasperated
#: click cannot bury a subspace beyond recovery.
#:
#: THE MAGNITUDE SELF-CALIBRATES. The probability is fixed, but the utility
#: gap it implies is Phi^-1(0.15) * sqrt(2) * s - in units of the judgement
#: noise s, which marginal likelihood re-selects as the session goes. A
#: decisive grader's dislikes therefore cut deep and an erratic grader's
#: are automatically softened, with no absolute severity ever hand-tuned.
DISLIKED_P = 0.15

#: Observations kept. Beyond this the oldest are dropped: the fit is O(n^3)
#: and, more to the point, taste drifts over a long session - the last 250
#: comparisons describe what the user wants now.
MAX_OBSERVATIONS = 250

#: Range dimensions are enumerated on their own step grid when the incumbent
#: is polished coordinate by coordinate; this caps that enumeration.
MAX_GRID = 41


# ---------------------------------------------------------------------------
# The normal distribution, in the tail
# ---------------------------------------------------------------------------

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _pdf(z):
    return np.exp(-0.5 * z * z - _LOG_SQRT_2PI)


def _cdf(z):
    """Phi(z), through erfc rather than erf.

    `0.5 * (1 + erf(z))` loses every significant digit for z below about -6:
    the two terms cancel. erfc is computed directly in that tail and stays
    accurate to full precision, which is what makes Z_CLIP = 8 safe.
    """
    return 0.5 * _erfc(-z / _SQRT_2)


def _mills(z):
    """phi(z)/Phi(z) - the inverse Mills ratio, i.e. d/dz log Phi(z)."""
    return _pdf(z) / np.maximum(_cdf(z), 1e-300)


# ---------------------------------------------------------------------------
# The space
# ---------------------------------------------------------------------------

class Dimension:
    """One degree of freedom.

    A point in the space is a tuple with one entry per dimension: the INDEX of
    the chosen label for a choice dimension, the value itself for a range one.
    Indices rather than labels because the labels are the caller's objects
    (prompt strings, model names) and this module never needs to look at them.
    """

    CHOICE = "choice"
    RANGE = "range"

    def __init__(self, name, kind, labels=None, ordinals=None,
                 low=0.0, high=1.0, step=0.05):
        self.name = name
        self.kind = kind
        self.labels = list(labels or [])
        # Present only when every label parsed as a number - see `numeric`.
        self.ordinals = None if ordinals is None else [float(v) for v in ordinals]
        self.low = float(low)
        self.high = float(high)
        self.step = float(step) if step else 0.0
        # Another Dimension INSTANCE in the same space, or None. Set on a
        # weight dimension whose meaning depends on a pick dimension (which
        # LoRA, which prompt): a weight is not transferable between two
        # different LoRAs, and the kernel honours that by ignoring this
        # dimension entirely between two points whose parent picks differ.
        self.parent = None

    @classmethod
    def choice(cls, name, labels):
        """A pick from a list. Numeric labels are recognised, not required.

        A list of offsets and a list of model names are the same KIND of
        control for the user - "one of these" - and it would be an odd UI that
        made them declare which. So the detection is here, once: if every
        label reads as a number the dimension is ordered and the kernel is
        told so, otherwise the labels are opaque and it is not.
        """
        ordinals = []
        for label in labels:
            try:
                ordinals.append(float(str(label).strip()))
            except (TypeError, ValueError):
                ordinals = None
                break
        # An ordering of one distinct value is not an ordering; it would also
        # make the normalization below divide by zero.
        if ordinals is not None and len(set(ordinals)) < 2:
            ordinals = None
        return cls(name, cls.CHOICE, labels=labels, ordinals=ordinals)

    @classmethod
    def range(cls, name, low, high, step=0.05):
        low, high = (float(low), float(high))
        if high < low:
            low, high = high, low
        return cls(name, cls.RANGE, low=low, high=high, step=step)

    # -- membership ------------------------------------------------------

    @property
    def numeric(self):
        """Does this dimension carry a metric the kernel can use?"""
        return self.kind == Dimension.RANGE or self.ordinals is not None

    @property
    def trivial(self):
        """A dimension with nothing to choose. It is kept (so the caller's row
        numbering survives) but contributes no variation."""
        if self.kind == Dimension.CHOICE:
            return len(self.labels) < 2
        return self.high - self.low < 1e-12

    def snap(self, value):
        """A range value on its own step grid, clamped to the bounds.

        Snapping is not cosmetic. It bounds the number of distinct points the
        search can ever visit, which is what lets repeated visits accumulate
        evidence instead of scattering it over 0.6237 and 0.6241 - and it also
        stops the user being shown two "different" LoRA weights that produce
        the same image.
        """
        value = min(max(float(value), self.low), self.high)
        if self.step > 0:
            value = self.low + round((value - self.low) / self.step) * self.step
            value = min(max(value, self.low), self.high)
        return round(value, 6)

    def sample(self, rng):
        if self.kind == Dimension.CHOICE:
            return int(rng.integers(max(len(self.labels), 1)))
        return self.snap(rng.uniform(self.low, self.high))

    def alternatives(self, current):
        """Every value this dimension could take instead of `current`."""
        if self.kind == Dimension.CHOICE:
            return [i for i in range(len(self.labels)) if i != current]
        if self.step > 0:
            count = int(round((self.high - self.low) / self.step)) + 1
        else:
            count = MAX_GRID
        count = max(2, min(count, MAX_GRID))
        grid = [self.snap(self.low + (self.high - self.low) * i / (count - 1))
                for i in range(count)]
        return [v for v in dict.fromkeys(grid) if abs(v - current) > 1e-9]

    def baseline(self):
        """The value this dimension starts at: the first listed choice, or the
        bottom of a range - which for a LoRA weight is the LoRA switched off.
        Both read as "what the user would have had without this row"."""
        return 0 if self.kind == Dimension.CHOICE else self.snap(self.low)

    def encode(self, value):
        """The kernel's coordinate for a value: normalized to [0, 1] when the
        dimension has a metric, the bare index when it does not (only equality
        is ever asked of it)."""
        if self.kind == Dimension.RANGE:
            span = self.high - self.low
            return 0.0 if span <= 0 else (float(value) - self.low) / span
        if self.ordinals is not None:
            lo, hi = min(self.ordinals), max(self.ordinals)
            index = int(value) if 0 <= int(value) < len(self.ordinals) else 0
            return (self.ordinals[index] - lo) / (hi - lo)
        return float(value)

    def describe(self, value):
        if self.kind == Dimension.CHOICE:
            index = int(value)
            if 0 <= index < len(self.labels):
                return str(self.labels[index])
            return "?"
        return f"{float(value):g}"


class Space:
    """The declared degrees of freedom, and the points they span."""

    def __init__(self, dimensions):
        self.dimensions = list(dimensions)
        self._numeric = np.array([d.numeric for d in self.dimensions], dtype=bool)
        # (weight dim index, pick dim index) for every conditional weight -
        # resolved by IDENTITY, because the parent is the instance the caller
        # paired it with, not any dimension that happens to look alike.
        self.conditional = []
        for index, dimension in enumerate(self.dimensions):
            if dimension.parent is None:
                continue
            for parent_index, candidate in enumerate(self.dimensions):
                if candidate is dimension.parent:
                    self.conditional.append((index, parent_index))
                    break
        self._conditional_parent = dict(self.conditional)

        # THE PERCEPTUAL METRIC'S PARAMETERS. Ordinary dimensions get one
        # rate. A conditional weight gets one rate PER PARENT PICK: 0.8 of
        # LoRA A can be visually inert while the same movement on LoRA B is
        # decisive, and averaging those into one coefficient is exactly the
        # kind of false distance that lets N-GOOD buy "diversity" without a
        # visible change. The public `weights` list remains one value per UI
        # dimension for status/reporting; a conditional entry is the mean of
        # its contextual rates. `_metric_rates` is what the model uses.
        self._metric_specs = []
        for index, dimension in enumerate(self.dimensions):
            parent = self._conditional_parent.get(index)
            if parent is not None:
                parent_dim = self.dimensions[parent]
                if parent_dim.kind == Dimension.CHOICE and parent_dim.labels:
                    self._metric_specs.extend(
                        (index, level) for level in range(len(parent_dim.labels)))
                    continue
            self._metric_specs.append((index, None))
        self._metric_rates = np.ones(len(self._metric_specs), dtype=float)
        self.weights = [1.0] * len(self.dimensions)
        self.conditional_weights = {}
        self.set_metric_rates(self._metric_rates)

    def __len__(self):
        return len(self.dimensions)

    @property
    def live(self):
        """Indices of the dimensions that actually vary."""
        return [i for i, d in enumerate(self.dimensions) if not d.trivial]

    @property
    def atomic_groups(self):
        """Live coordinates that form one semantic gene for transplantation.

        A LoRA/prompt pick and its conditional weight are two kernel
        coordinates but one characteristic. An Interesting donor must never
        contribute one without the other: "this LoRA was interesting" paired
        with some unrelated weight is not the feature the user marked.
        """
        live = set(self.live)
        children = {}
        for child, parent in self.conditional:
            if child in live:
                children.setdefault(parent, []).append(child)
        groups, used = [], set()
        for index in self.live:
            if index in used or index in self._conditional_parent:
                continue
            group = [index]
            group.extend(child for child in children.get(index, [])
                         if child not in used)
            used.update(group)
            groups.append(tuple(group))
        # A live child of a trivial/non-live parent still has a meaningful
        # within-parent weight and therefore remains its own gene.
        for index in self.live:
            if index not in used:
                groups.append((index,))
        return groups

    @property
    def size(self):
        """How many distinct points the space holds, or None if that overflows
        being a useful number. Reported to the user, never used in the maths."""
        total = 1
        for d in self.dimensions:
            if d.kind == Dimension.CHOICE:
                total *= max(len(d.labels), 1)
            elif d.step > 0:
                total *= max(int(round((d.high - d.low) / d.step)) + 1, 1)
            else:
                return None
            if total > 10 ** 12:
                return None
        return total

    def baseline(self):
        return tuple(d.baseline() for d in self.dimensions)

    def sample(self, rng):
        return tuple(d.sample(rng) for d in self.dimensions)

    def perturb(self, point, rng, count=1):
        """`point` with `count` of its live dimensions re-drawn.

        The local half of the candidate pool. A pool of purely uniform draws
        is fine for finding the right region and hopeless for polishing inside
        it: with eight dimensions, a uniform draw agrees with the incumbent on
        nothing, so every candidate near the incumbent has to be built on
        purpose.
        """
        live = self.live
        if not live:
            return tuple(point)
        values = list(point)
        for index in rng.choice(live, size=min(count, len(live)), replace=False):
            values[index] = self.dimensions[index].sample(rng)
        return tuple(values)

    def key(self, point):
        """Identity of a point. Range values are already snapped to their grid,
        so rounding here only guards against float noise from arithmetic."""
        return tuple(
            int(v) if d.kind == Dimension.CHOICE else round(float(v), 6)
            for d, v in zip(self.dimensions, point))

    @staticmethod
    def _delta(dimension, x, y):
        """One dimension's normalized structural distance.

        Numeric CHOICES are numeric here as well as in the utility kernel.
        Treating the list ``0, .25, .5, .75, 1`` as five unrelated labels
        made one adjacent notch buy the same collage separation as the whole
        span, despite the class docstring promising the opposite.
        """
        if dimension.numeric:
            return abs(dimension.encode(x) - dimension.encode(y))
        return 0.0 if int(x) == int(y) else 1.0

    def deltas(self, a, b):
        """The UNWEIGHTED per-dimension distances between two points.

        A differing OPAQUE category counts 1 - two models are either the same
        model or a different one, there is no half. A range or numeric-choice
        dimension counts the fraction of its own span that separates the two
        values, so the number means the same thing whether a typed grid runs
        0..1 or 0.5..1.5. A conditional child contributes zero across two
        parent picks because its values are not comparable there.

        These are also the FEATURES the similarity fit runs on (see
        _fit_metric): "which rows differ between the two images the user just
        called the same, and by how much" is exactly this vector.
        """
        values = []
        for index, (dimension, x, y) in enumerate(
                zip(self.dimensions, a, b)):
            parent = self._conditional_parent.get(index)
            # A child's values are not comparable across parent picks. This
            # is already the utility kernel's rule; failing to mirror it here
            # let every LoRA row contribute TWO apparent differences when
            # only the pick had a defined meaning.
            if parent is not None and int(a[parent]) != int(b[parent]):
                values.append(0.0)
            else:
                values.append(self._delta(dimension, x, y))
        return values

    @property
    def metric_rates(self):
        return self._metric_rates.copy()

    def set_metric_rates(self, rates):
        """Install fitted noisy-OR rates and publish their compact reading."""
        values = np.asarray(rates, dtype=float)
        if values.shape != (len(self._metric_specs),):
            raise ValueError("metric rate count does not match the space")
        self._metric_rates = values.copy()
        compact = [1.0] * len(self.dimensions)
        contextual = {}
        for index, (dimension_index, level) in enumerate(self._metric_specs):
            value = float(values[index])
            if level is None:
                compact[dimension_index] = value
            else:
                contextual.setdefault(dimension_index, []).append(value)
        for dimension_index, per_level in contextual.items():
            contextual[dimension_index] = list(per_level)
            compact[dimension_index] = float(np.mean(per_level))
        self.weights = compact
        self.conditional_weights = contextual

    def metric_features(self, a, b):
        """Expanded unweighted features used by the perceptual fit.

        Conditional children occupy one column per parent level. Only the
        column belonging to a shared parent pick can be non-zero; across two
        picks every child column is zero because the values have no common
        meaning there.
        """
        base = self.deltas(a, b)
        features = []
        for dimension_index, level in self._metric_specs:
            value = base[dimension_index]
            if level is not None:
                parent = self._conditional_parent[dimension_index]
                value = (value if int(a[parent]) == level
                         and int(b[parent]) == level else 0.0)
            features.append(value)
        return features

    def separation(self, a, b):
        """How far apart two configurations are, in "visible change" units -
        the structural distances above, each scaled by what the similarity
        rows taught about that dimension and, for a conditional weight, its
        shared parent pick.

        THE ONE FUNCTION EVERY DISTANCE TEST CALLS. Whether a duel is worth
        asking, how different a keeper has to be, what a collage may hold
        together and how many distinct good answers the space has are all
        this number against a threshold - so `weights` moving is the whole of
        how a learned metric reaches them. See MIN_DUEL_SEPARATION.
        """
        return float(np.dot(self._metric_rates, self.metric_features(a, b)))

    def raw_separation(self, a, b):
        """`separation` under the PRIOR weights - the metric as it was before
        any similarity label. Only _probe_duel wants this: the pair worth
        probing is the one these two disagree about."""
        return sum(self.deltas(a, b))

    def separations(self, points):
        """`separation` for a whole batch at once: the pairwise matrix.

        Same arithmetic as the scalar version, in numpy - the capacity
        estimate asks about every pair of a few hundred draws, which is tens
        of thousands of calls per graded duel, and that is a tenth of a
        second of Python for a number nobody would wait for.
        """
        if not points:
            return np.zeros((0, 0))
        values = np.asarray(points, dtype=float)
        encoded = self.encode(points)
        total = np.zeros((len(points), len(points)))
        for rate, (index, level) in zip(self._metric_rates,
                                        self._metric_specs):
            dimension = self.dimensions[index]
            column = encoded[:, index] if dimension.numeric else values[:, index]
            delta = np.abs(column[:, None] - column[None, :])
            if not dimension.numeric:
                delta = (delta > 1e-9).astype(float)
            if level is not None:
                parent = self._conditional_parent[index]
                at_level = np.abs(values[:, parent] - level) < 0.5
                delta = delta * (at_level[:, None] & at_level[None, :])
            total += float(rate) * delta
        return total

    def encode(self, points):
        if not points:
            return np.zeros((0, len(self.dimensions)))
        return np.array([[d.encode(v) for d, v in zip(self.dimensions, p)]
                         for p in points], dtype=float)

    @property
    def numeric_mask(self):
        return self._numeric

    def describe(self, point):
        """A one-line reading of a point, live dimensions only."""
        parts = []
        for index in self.live:
            d = self.dimensions[index]
            parts.append(f"{d.name} = {d.describe(point[index])}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

class PreferenceSearch:
    """Graded duels in, a configuration out.

    Usage is a loop: `next_duel()` -> show both -> `observe(a, b, grade)`, and
    `best()` at any point (including mid-loop - it is what the STOP button
    prints, so it must be meaningful after every single answer).
    """

    # Export/replay code receives an engine instance rather than this module;
    # exposing the schema's anchor probability here lets it upgrade old states
    # without copying a model constant into the UI layer.
    ON_TRACK_P = ON_TRACK_P

    def __init__(self, space, seed=0, seed_duels=SEED_DUELS,
                 pool_size=POOL_SIZE):
        self.space = space
        self.rng = np.random.default_rng(seed)
        self.seed_duels = seed_duels
        self.pool_size = pool_size

        self.points = []            # unique configurations, in first-seen order
        self._index = {}            # space key -> row in self.points
        # (index_a, index_b, p); index_a is None for an ABSOLUTE observation,
        # a verdict's statement about index_b against the prior mean.
        self.observations = []
        self.duels = 0              # graded duels + verdicts, NOT len(observations)
        # (point_a, point_b, distinct, mass) from the panel's top two rows -
        # see SIMILARITY in the module docstring. BY VALUE, not by index into
        # self.points: these
        # outlive the observation cap (_forget_oldest rebuilds the point
        # list, which would silently reindex them), and what they are about
        # is a pair of CONFIGURATIONS, not a pair of graded duels. `mass` is
        # how much the fit believes each one - see DISTINCT_WEIGHT.
        self.similarities = []
        # Pairs the censoring probe asked about, by unordered key. An answer
        # to one of these is worth full weight however it lands: the model
        # asked, so it does not get to discount the reply.
        self._probed = set()

        self._asked = {}            # unordered pair key -> times asked
        self._shown = []            # every point ever put in front of the user
        self._shown_keys = set()
        self._levels_shown = set()  # (dimension index, choice index) seen so far
        self._attractors = []       # every distinct basin, proven or not
        self._keepers = []          # the floored, public frontier
        self._frontier_public = []  # floored archive, before display slicing
        self._frontier_stamp = -1   # duel at which that archive was computed
        self._frontier_limit = 0
        self._redirect = False      # last duel was disliked - change the subject
        self._dislike_streak = 0    # consecutive disliked duels
        self._surprise = None       # unexpected winner awaiting its follow-up
        self._interesting = []      # donor configurations, oldest first
        #: What the last `population` call actually achieved - how many
        #: islands its entries came from, how many the good region holds,
        #: and the lowest pairwise probability of looking different on the
        #: sheet. Read by the panel and printed with the collage: the
        #: promise the button makes is only worth making if the press can
        #: say whether it kept it.
        self.population_report = {"islands": 0, "total_islands": 0,
                                  "confidence": 1.0}

        # Host-injected cost oracle: point -> "an image of this configuration
        # is already made and reusable". None (the default, and what every
        # test without a host runs under) means every side costs a
        # generation - the cost-blind behaviour. See THE PRICE OF A QUESTION.
        # Which tactic produced the duel on screen, when that is something
        # the user needs told. Only the censoring probe sets it: that duel
        # is SUPPOSED to look like the same image twice, which without a
        # word of explanation reads as the search having broken.
        self.last_tactic = None

        self.reuse_probe = None
        self._reuse_streak = 0      # consecutive duels with both sides cached
        self._probe_memo = {}       # this duel's probe answers, by point key

        self.f = np.zeros(0)        # Laplace mode of the latent utility
        self._alpha = np.zeros(0)   # K^-1 f, the predictive weights
        self._W = np.zeros((0, 0))  # likelihood curvature at the mode
        self._K = np.zeros((0, 0))
        self._hyper = (LENGTHSCALES[1], THETAS[1], NOISES[0])
        self._logz = float("-inf")
        self._dirty = True

    # -- observations ----------------------------------------------------

    def _row(self, point):
        key = self.space.key(point)
        index = self._index.get(key)
        if index is None:
            index = len(self.points)
            self._index[key] = index
            self.points.append(tuple(point))
        return index

    def mark_interesting(self, point):
        """Register a configuration as a DONOR - see INTERESTING_MAX.

        Not an observation: the mark carries no opinion about how good the
        configuration is (the grade already said that, and the user's own
        definition of the button is "overall bad"). It only makes the
        configuration's coordinates available for hybridization in the
        candidate pool, so its tempting characteristic gets tried inside
        good samples. Marking the same configuration again is a no-op.
        """
        key = self.space.key(point)
        if any(self.space.key(existing) == key
               for existing in self._interesting):
            return
        self._interesting.append(tuple(point))
        if len(self._interesting) > INTERESTING_MAX:
            del self._interesting[0]

    def observe(self, point_a, point_b, grade, disliked=False,
                interesting_a=False, interesting_b=False, similar=None):
        """Record one graded duel. `grade` is 0..10, 0 = A far better.

        `similar` is which of the panel's top two rows the click landed on -
        True for "rather SIMILAR samples, and we are on track", False for
        "rather distinct, and we are on track", None when the answer carries
        no such verdict (the dislike row, or an old caller that does not ask
        the question). It trains the SEPARATION METRIC - see SIMILARITY in
        the module docstring and _fit_metric - which is what decides whether a
        duel is worth asking at all, how different a keeper has to be, and
        what a collage may hold together. The shared "on track" half also
        anchors BOTH samples above par; without it the model knows only
        relative winners and cannot distinguish good from merely less bad.

        `similar=True` also records an extra TIE observation on the pair, and
        that is not double-counting: two configurations that look the same
        must be worth the same, which is a stronger and more reliable claim
        than the grade beside it (a soft label a person assigns loosely). If
        the grade on that row says one side is slightly ahead, both
        statements are on the record and the posterior settles between them,
        which is the honest outcome.

        `disliked` is the panel's bottom row: the SAME 0..10 comparison,
        clicked there instead of the normal row when the user dislikes BOTH
        sides. That is a feeling a person has the moment a pair appears and
        can report at no extra cost - the click that grades the duel simply
        lands one row lower - and it carries exactly what a comparison
        cannot: "both are awful" and "both are great" are the same statement
        about the difference. The grade itself still matters (which side is
        LESS bad steers the model inside the region like any other duel);
        what `disliked` adds is one absolute observation per side against
        the prior mean 0 - par, what the GP believes about a configuration
        nobody has asked about. That is what turns a bad subspace into
        something the search can mark once and then AVOID, rather than rank
        carefully from the inside at two generations a question - and it is
        what gives `portfolio(good=False)` an address for "bad".

        `interesting_a` / `interesting_b` are the per-image toggles: "this
        one has a characteristic worth carrying into good samples" - fully
        compatible with the same side being graded down and disliked, which
        is the expected combination. They route to `mark_interesting` and
        touch nothing else; see it for why a mark is not evidence.
        """
        p = min(max(float(grade) / 10.0, P_CLAMP), 1.0 - P_CLAMP)

        # Was this outcome a SURPRISE? Checked against the model as it stood
        # BEFORE this answer, and only when that model is current (`_dirty`
        # is the tell) - a stale posterior's expectations are not
        # expectations. A disliked duel cannot surprise upward: its winner
        # is merely less bad. See SURPRISE_P for what arming this does.
        self._surprise = None
        if self.observations and not self._dirty and not disliked:
            means, _ = self._posterior([tuple(point_a), tuple(point_b)],
                                       with_covariance=False)
            scale = _SQRT_2 * max(self._hyper[2], 1e-3)
            expected_b = float(_cdf((means[1] - means[0]) / scale))
            if grade >= 8 and expected_b < SURPRISE_P:
                self._surprise = tuple(point_b)
            elif grade <= 2 and expected_b > 1.0 - SURPRISE_P:
                self._surprise = tuple(point_a)

        index_a, index_b = self._row(point_a), self._row(point_b)
        self.observations.append((index_a, index_b, p))
        if disliked:
            for index in (index_a, index_b):
                self.observations.append((None, index, DISLIKED_P))
        elif similar is not None:
            # The top two rows partition the ON-TRACK case. This is absolute
            # information just as the bottom row's "both bad" is: the grade
            # orders the pair, while these two anchors say both belong on the
            # usable side of par. Old callers that pass similar=None retain
            # comparison-only behavior because they never asked this question.
            for index in (index_a, index_b):
                self.observations.append((None, index, ON_TRACK_P))
        if similar is not None:
            mass = 1.0
            if not similar and self._pair_key(point_a, point_b) not in self._probed:
                mass = DISTINCT_WEIGHT
            self.similarities.append(
                (tuple(point_a), tuple(point_b), not bool(similar), mass))
            if similar:
                # "The same image" implies "worth the same" - see the
                # docstring. Recorded as a second tie on the pair, which is
                # the vocabulary the likelihood already speaks.
                self.observations.append((index_a, index_b, 0.5))
            self._fit_metric()
        if interesting_a:
            self.mark_interesting(point_a)
        if interesting_b:
            self.mark_interesting(point_b)
        # A dislike is also read at the PROCESS level - "this is not going
        # well" - so the next duel changes the subject (see next_duel), and a
        # STREAK of them un-anneals the pool (see _pool) rather than letting
        # the search keep polishing a direction being declared a dead end.
        self._redirect = bool(disliked)
        self._dislike_streak = self._dislike_streak + 1 if disliked else 0
        self.duels += 1
        if len(self.observations) > MAX_OBSERVATIONS:
            self._forget_oldest()
        self._dirty = True

    def _forget_oldest(self):
        """Drop observations past the cap and rebuild the point set.

        Points are rebuilt rather than kept: a configuration nothing refers to
        any more still costs a row and a column in every matrix below, and the
        whole reason for the cap is that those are cubic.
        """
        kept = self.observations[-MAX_OBSERVATIONS:]
        old_points = self.points
        self.points, self._index = [], {}
        rebuilt = []
        for a, b, p in kept:
            rebuilt.append((None if a is None else self._row(old_points[a]),
                            self._row(old_points[b]), p))
        self.observations = rebuilt

    # -- the separation metric -------------------------------------------

    def _fit_metric(self):
        """Refit `space.weights` from the similarity labels.

        THE MODEL IS NOISY-OR. Each dimension independently has some rate of
        producing a visible change; the distance between two configurations
        is the sum of the rates of the dimensions that differ (scaled, for a
        range dimension, by how far); and

            P(the pair looks distinct) = 1 - exp(-distance)

        which is just "distinct unless every differing dimension happened to
        be invisible". It is the right shape for the question - a difference
        somewhere is enough, and differences accumulate rather than average -
        and it has roughly ONE PARAMETER PER ROW (one per parent level for a
        conditional weight), so a dozen labels are already informative and
        there is nothing exotic for it to overfit into.

        The log-likelihood is concave in the distance and the distance is
        linear in the weights, so the objective is smooth and well behaved;
        weights are optimized in the log so they stay positive, warm-started
        from the previous fit, and pulled toward 1 - the hand-written metric
        - by METRIC_SHRINK. With no labels every weight IS 1 and every
        distance test in the engine behaves exactly as it did before this
        existed, which is the property that makes the feature safe to ship.

        The clamps at the end are not cosmetic: see METRIC_MIN for why a
        dimension must never be allowed to reach zero.
        """
        if not self.similarities:
            self.space.set_metric_rates(
                np.ones(len(self.space._metric_specs), dtype=float))
            return
        features = np.array(
            [self.space.metric_features(a, b)
             for a, b, _d, _m in self.similarities],
            dtype=float)
        distinct = np.array([1.0 if d else 0.0
                             for _a, _b, d, _m in self.similarities])
        # Assigned when the label was recorded - see DISTINCT_WEIGHT: the top
        # row is where an idle hand parks, so an undirected "distinct" label
        # is worth less than a deliberate "similar" one, while an answer to a
        # duel the probe ASKED counts in full.
        mass = np.array([m for _a, _b, _d, m in self.similarities])

        v = np.log(np.clip(self.space.metric_rates,
                           METRIC_MIN, METRIC_MAX))
        for _ in range(METRIC_STEPS):
            weights = np.exp(v)
            distance = np.maximum(features @ weights, 1e-6)
            # d/d(distance) of the log-likelihood: the distinct term wants
            # distance LARGER (its derivative is the inverse odds of having
            # been visible), the similar term wants it smaller, flatly.
            survival = np.exp(-distance)
            slope = mass * (distinct * survival / np.maximum(1.0 - survival, 1e-12)
                            - (1.0 - distinct))
            # Chain through distance = features @ exp(v), plus the shrinkage
            # pull toward v = 0 (weight 1).
            gradient = (slope @ features) * weights - 2.0 * METRIC_SHRINK * v
            step = METRIC_RATE * gradient / max(len(self.similarities), 1)
            # A hard cap per step rather than a line search: the objective is
            # tiny and re-fitted every label, so a slightly slow climb costs
            # nothing and an overshoot on one emphatic label costs a metric.
            v = v + np.clip(step, -0.25, 0.25)
            v = np.clip(v, math.log(METRIC_MIN), math.log(METRIC_MAX))
        self.space.set_metric_rates(np.exp(v))

    # -- the model -------------------------------------------------------

    def _kernel(self, u, v, lengthscale, theta):
        """k(u, v) as a product over perceptually weighted dimensions.

        The similarity rows do not merely decide which pairs may appear on a
        collage. They say where utility should transfer: two configurations
        the user repeatedly calls visually identical should share evidence,
        while a vivid change should decorrelate quickly. With every metric
        rate at its prior value 1 this is the original mixed kernel exactly.
        """
        if u.size == 0 or v.size == 0:
            return np.zeros((u.shape[0], v.shape[0]))
        delta = u[:, None, :] - v[None, :, :]
        log_k = np.zeros(delta.shape[:2])
        for index, dimension in enumerate(self.space.dimensions):
            difference = delta[:, :, index]
            rate = self.space.weights[index]
            parent = self.space._conditional_parent.get(index)
            if parent is not None:
                # A conditional weight is not transferable across its parent
                # pick, and its learned visibility is contextual as well.
                parent_difference = np.abs(delta[:, :, parent]) > 1e-9
                difference = np.where(parent_difference, 0.0, difference)
                contextual = self.space.conditional_weights.get(index)
                if contextual:
                    rate_matrix = np.zeros_like(difference)
                    parent_dimension = self.space.dimensions[parent]
                    for level, level_rate in enumerate(contextual):
                        encoded_level = parent_dimension.encode(level)
                        same_level = ((np.abs(u[:, None, parent] - encoded_level)
                                       < 1e-9)
                                      & (np.abs(v[None, :, parent] - encoded_level)
                                         < 1e-9))
                        rate_matrix = np.where(same_level, level_rate,
                                               rate_matrix)
                    rate = rate_matrix
            if dimension.numeric:
                scaled = difference / max(lengthscale, 1e-6)
                log_k -= 0.5 * rate * scaled * scaled
            else:
                log_k -= theta * rate * (np.abs(difference) > 1e-9)
        return np.exp(log_k)

    def _likelihood(self, f, noise):
        """log p(D|f), its gradient, and the curvature W = -d2/df2.

        W is not diagonal: every duel couples two points, contributing a 2x2
        block. It is positive semi-definite because each duel's log-likelihood
        is concave in the gap - which is what makes the Newton iteration below
        a descent on a convex objective rather than a hopeful fixed point.
        """
        n = len(self.points)
        scale = _SQRT_2 * max(noise, 1e-3)
        g = np.zeros(n)
        W = np.zeros((n, n))
        total = 0.0
        for a, b, p in self.observations:
            # An absolute observation (a is None) is the same probit statement
            # with the anchor - utility 0, the prior mean - on side A. Its
            # curvature touches only the diagonal: nothing couples to a point
            # whose utility is not a variable.
            gap = f[b] if a is None else f[b] - f[a]
            z = np.clip(gap / scale, -Z_CLIP, Z_CLIP)
            cdf_pos, cdf_neg = _cdf(z), _cdf(-z)
            total += p * math.log(max(cdf_pos, 1e-300)) \
                + (1.0 - p) * math.log(max(cdf_neg, 1e-300))

            mills_pos, mills_neg = _mills(z), _mills(-z)
            # d/dz of the duel's log-likelihood, and its second derivative
            d1 = p * mills_pos - (1.0 - p) * mills_neg
            d2 = (-p * mills_pos * (z + mills_pos)
                  + (1.0 - p) * mills_neg * (z - mills_neg))

            block = -d2 / (scale * scale)     # >= 0; the duel's contribution
            g[b] += d1 / scale
            W[b, b] += block
            if a is None:
                continue
            g[a] -= d1 / scale
            W[a, a] += block
            W[a, b] -= block
            W[b, a] -= block
        return total, g, W

    def _laplace(self, K, noise, warm=None):
        """The Laplace approximation for one set of hyperparameters.

        Returns (f_hat, W, log_marginal). The Newton step is written in the
        `f <- K (WK + I)^-1 (Wf + g)` form so that only K itself is ever
        inverted, and it is damped by a backtracking line search: an early
        step with almost no data can otherwise overshoot into the saturated
        tail of Phi, where the gradient is ~0 and the iteration stalls at a
        point that is not the mode.
        """
        n = K.shape[0]
        eye = np.eye(n)
        jitter = 1e-6 * (np.trace(K) / max(n, 1) + 1.0)
        K = K + jitter * eye

        f = np.zeros(n) if warm is None or warm.shape[0] != n else warm.copy()

        def objective(vec):
            total, _, _ = self._likelihood(vec, noise)
            return total - 0.5 * float(vec @ np.linalg.solve(K, vec))

        current = objective(f)
        total, g, W = self._likelihood(f, noise)
        for _ in range(NEWTON_STEPS):
            try:
                target = np.linalg.solve(W @ K + eye, W @ f + g)
            except np.linalg.LinAlgError:
                break
            step = K @ target - f
            for shrink in (1.0, 0.5, 0.25, 0.1, 0.02):
                candidate = f + shrink * step
                value = objective(candidate)
                if value >= current - 1e-12:
                    break
            else:
                break
            moved = float(np.max(np.abs(candidate - f)))
            f, current = candidate, value
            total, g, W = self._likelihood(f, noise)
            if moved < NEWTON_TOL:
                break

        sign, logdet = np.linalg.slogdet(eye + W @ K)
        if sign <= 0:
            logdet = 0.0
        log_marginal = total - 0.5 * float(f @ np.linalg.solve(K, f)) - 0.5 * logdet
        return f, W, K, log_marginal

    def _fit(self):
        if not self._dirty or not self.observations:
            return
        u = self.space.encode(self.points)

        search_hypers = (len(self.observations) % HYPER_EVERY == 1
                         or self._logz == float("-inf"))
        candidates = ([(l, t, s) for l in LENGTHSCALES for t in THETAS for s in NOISES]
                      if search_hypers else [self._hyper])

        best = None
        for lengthscale, theta, noise in candidates:
            K = self._kernel(u, u, lengthscale, theta)
            f, W, K, logz = self._laplace(K, noise, warm=self.f)
            if best is None or logz > best[0]:
                best = (logz, f, W, K, (lengthscale, theta, noise))

        self._logz, self.f, self._W, self._K, self._hyper = best
        self._alpha = np.linalg.solve(self._K, self.f)
        self._dirty = False

    # -- prediction ------------------------------------------------------

    def _posterior(self, points, with_covariance=True):
        """Posterior mean (and covariance) of f at `points`.

        At the Laplace mode the predictive mean is the ordinary GP one with
        alpha = K^-1 f_hat, and the covariance loses `Ks' (WK + I)^-1 W Ks` -
        the amount the comparisons pinned down. With no observations at all
        this is the prior, which is the correct answer and not a special case
        worth branching on: mean 0 everywhere, so `next_duel` draws two
        unrelated points and the loop starts.
        """
        u = self.space.encode(points)
        lengthscale, theta, _ = self._hyper
        if not self.observations:
            mean = np.zeros(len(points))
            if not with_covariance:
                return mean, None
            return mean, self._kernel(u, u, lengthscale, theta)

        self._fit()
        observed = self.space.encode(self.points)
        cross = self._kernel(u, observed, lengthscale, theta)
        mean = cross @ self._alpha
        if not with_covariance:
            return mean, None

        prior = self._kernel(u, u, lengthscale, theta)
        eye = np.eye(self._K.shape[0])
        try:
            reduction = np.linalg.solve(self._W @ self._K + eye, self._W)
        except np.linalg.LinAlgError:
            reduction = np.zeros_like(self._W)
        cov = prior - cross @ reduction @ cross.T
        cov = 0.5 * (cov + cov.T)
        np.fill_diagonal(cov, np.maximum(np.diag(cov), 1e-12))
        return mean, cov

    def _sample(self, mean, cov, count):
        """`count` independent draws from N(mean, cov), Cholesky where it can.

        The covariance is a difference of two kernel matrices and is only PSD
        in exact arithmetic, so the factorization is retried with growing
        jitter and finally falls back to sampling the diagonal alone. That
        fallback is not a silent degradation: dropping the correlations makes
        the draws NOISIER, so the duel it produces is more exploratory than
        intended, never falsely confident.
        """
        size = mean.shape[0]
        scale = float(np.mean(np.diag(cov))) + 1e-12
        for exponent in range(-10, -3):
            try:
                factor = np.linalg.cholesky(cov + (10.0 ** exponent) * scale * np.eye(size))
            except np.linalg.LinAlgError:
                continue
            return [mean + factor @ self.rng.standard_normal(size)
                    for _ in range(count)]
        deviation = np.sqrt(np.maximum(np.diag(cov), 0.0))
        return [mean + deviation * self.rng.standard_normal(size)
                for _ in range(count)]

    # -- queries ---------------------------------------------------------

    def _pair_key(self, a, b):
        """The identity of a duel, unordered: A vs B is the same question as
        B vs A, and the repeat cap must see it as one."""
        key_a, key_b = self.space.key(a), self.space.key(b)
        return (key_a, key_b) if key_a <= key_b else (key_b, key_a)

    def _times_asked(self, a, b):
        return self._asked.get(self._pair_key(a, b), 0)

    def _rendered(self, point):
        """Is this configuration's image already made? Asked through the
        host-injected `reuse_probe`, memoized per duel - the host answers by
        building the point's whole recipe string, and one duel's selection
        may ask about the same pool point several times. A probe that is
        absent, or that BREAKS, reads as "nothing is reusable": the reuse
        preference is an economy, and an economy that can take the search
        down with it costs more than it saves.
        """
        if self.reuse_probe is None:
            return False
        key = self.space.key(point)
        cached = self._probe_memo.get(key)
        if cached is None:
            try:
                cached = bool(self.reuse_probe(tuple(point)))
            except Exception:
                cached = False
            self._probe_memo[key] = cached
        return cached

    def _remember(self, a, b):
        """Book-keep a duel about to be shown; returns it for the caller.

        Every path out of next_duel comes through here, which is what makes
        the repeat cap and the coverage bonus honest: they are about what the
        user has SEEN, and a duel the user then skips was still seen.
        """
        key = self._pair_key(a, b)
        self._asked[key] = self._asked.get(key, 0) + 1
        # The reuse streak counts what the user is about to SEE: a duel made
        # entirely of images already rendered, however it came about - by
        # the reuse preference or by Thompson honestly wanting two cached
        # points. Both spend the user's attention without generating new
        # evidence, and both is what REUSE_STREAK_MAX bounds the run of.
        if self.reuse_probe is not None:
            if self._rendered(a) and self._rendered(b):
                self._reuse_streak += 1
            else:
                self._reuse_streak = 0
        for point in (a, b):
            point_key = self.space.key(point)
            if point_key not in self._shown_keys:
                self._shown_keys.add(point_key)
                self._shown.append(tuple(point))
            for index, dimension in enumerate(self.space.dimensions):
                if dimension.kind == Dimension.CHOICE and not dimension.trivial:
                    self._levels_shown.add((index, int(point[index])))
        return a, b

    def _coverage_bonus(self, point):
        """How many categorical levels this point would show for the first
        time, in separation units (a categorical difference counts 1, so half
        of one per unseen level keeps the bonus subordinate to spread)."""
        bonus = 0.0
        for index, dimension in enumerate(self.space.dimensions):
            if dimension.kind == Dimension.CHOICE and not dimension.trivial \
                    and (index, int(point[index])) not in self._levels_shown:
                bonus += 0.5
        return bonus

    def _spread_pick(self, anchor=None):
        """A space-filling draw: the candidate farthest from everything shown.

        This is the design of experiments the seed duels run on. A uniform
        draw is unbiased and also careless - three random duels can put six
        points in one corner and never show half the models - so each seed
        point is the best of SEED_BATCH candidates, scored by its distance to
        every point already shown plus a bonus for categorical levels not yet
        seen. With `anchor`, the pick must also be a visible distance from it
        (it is the other side of the same duel).
        """
        best, widest = None, None
        for _ in range(SEED_BATCH):
            candidate = self.space.sample(self.rng)
            if anchor is not None:
                separation = self.space.separation(anchor, candidate)
                if separation < MIN_DUEL_SEPARATION:
                    if widest is None or separation > widest[0]:
                        widest = (separation, candidate)
                    continue
            spread = min((self.space.separation(shown, candidate)
                          for shown in self._shown), default=3.0)
            # Capped so that in a huge space the coverage bonus still moves
            # the pick, rather than drowning under raw distance.
            score = min(spread, 3.0) + self._coverage_bonus(candidate)
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None:
            return best[1]
        # Nothing cleared the anchor bar - the space is too small to hold a
        # visible difference - so the widest miss is the best question left.
        if widest is not None:
            return widest[1]
        return self.space.perturb(anchor, self.rng, count=len(self.space))

    def _hybrid(self, donor, base):
        """`base` with 1-3 semantic genes transplanted from `donor`.

        The donor is an "interesting" configuration - overall bad, one
        characteristic worth keeping - and the base is a good one. A
        characteristic is usually carried by FEW genes, so the transplant is
        small: a hybrid that took half the donor's genome would mostly inherit
        what made the donor bad. A conditional pick+weight pair is one gene,
        never two independently sampled coordinates; splitting it can create
        a LoRA/weight combination the user did not mark at all.
        """
        groups = self.space.atomic_groups
        if not groups:
            return tuple(base)
        child = list(base)
        count = min(int(self.rng.integers(1, 4)), len(groups))
        for group_index in self.rng.choice(
                len(groups), size=count, replace=False):
            for index in groups[int(group_index)]:
                child[index] = donor[index]
        return tuple(child)

    def _surrogate_features(self, points):
        """The trend model's design matrix - see THE TREND MODEL.

        Additive by construction, which is the model's whole reason for
        existing: an intercept, then per live dimension either a one-hot
        block (opaque category) or the encoded value and its square
        (numeric - the square is what lets a knob's trend PEAK inside its
        range instead of only at an end). A conditional weight mirrors the
        kernel's no-transfer rule feature-wise: its value and square appear
        once PER PARENT LEVEL, masked to the rows that picked that level,
        so each pick's weight curve is its own pair of coefficients and
        duels graded on one LoRA's weight never shape the trend of another.
        """
        if not points:
            return np.zeros((0, 1))
        u = self.space.encode(points)
        raw = np.array([[float(v) for v in p] for p in points], dtype=float)
        live = set(self.space.live)
        children = {child for child, _parent in self.space.conditional}
        columns = [np.ones(len(points))]
        for index in sorted(live):
            dimension = self.space.dimensions[index]
            if index in children:
                continue
            if dimension.numeric:
                value = u[:, index]
                columns.append(value)
                columns.append(value * value)
            else:
                for level in range(len(dimension.labels)):
                    columns.append(
                        (np.abs(raw[:, index] - level) < 0.5).astype(float))
        for child, parent in self.space.conditional:
            if child not in live:
                continue
            value = u[:, child]
            parent_dim = self.space.dimensions[parent]
            if parent_dim.kind != Dimension.CHOICE \
                    or len(parent_dim.labels) < 2:
                # A trivial (or non-choice) parent has one effective level,
                # so the child is an ordinary numeric dimension.
                columns.append(value)
                columns.append(value * value)
                continue
            for level in range(len(parent_dim.labels)):
                mask = (np.abs(raw[:, parent] - level) < 0.5).astype(float)
                columns.append(mask * value)
                columns.append(mask * value * value)
        return np.stack(columns, axis=1)

    def _surrogate_fit(self):
        """Ridge weights fitted to the Laplace mode at the observed points,
        or None while there is too little to fit.

        The target is `self.f` - the GP's own belief - rather than raw
        grades, and that is the symbiosis: the main solver has already done
        the work of turning noisy fractional comparisons and dislike
        anchors into per-point utilities, so the trend model inherits all
        of it for the cost of a least-squares solve, and the two models can
        never disagree about what the DATA said, only about what it implies
        elsewhere.
        """
        if len(self.points) < 4 or self.f.shape[0] != len(self.points):
            return None
        X = self._surrogate_features(self.points)
        try:
            weights = np.linalg.solve(
                X.T @ X + SURROGATE_RIDGE * np.eye(X.shape[1]),
                X.T @ self.f)
        except np.linalg.LinAlgError:
            return None
        return weights

    def _surrogate_nominees(self, count):
        """The trend model's shortlist: the top `count` of SURROGATE_BATCH
        uniform draws, scored in one matrix multiply.

        Nominees enter the candidate pool and nothing else - the GP
        posterior and Thompson sampling still decide what is asked, so a
        wrong trend costs pool slots and a right one puts the cross-
        combination guess in front of the real judge without a generation
        spent discovering each combination separately.
        """
        if count <= 0:
            return []
        self._fit()
        weights = self._surrogate_fit()
        if weights is None:
            return []
        batch = [self.space.sample(self.rng) for _ in range(SURROGATE_BATCH)]
        scores = self._surrogate_features(batch) @ weights
        picked, keys = [], set()
        for index in np.argsort(-scores):
            point = batch[int(index)]
            key = self.space.key(point)
            if key in keys:
                continue
            keys.add(key)
            picked.append(point)
            if len(picked) >= count:
                break
        return picked

    def _pool(self):
        """Candidates for this duel: every attractor's neighbourhood, fresh
        uniform draws, everything already seen - and, when the user has
        marked configurations as interesting, HYBRIDS of those donors with
        the attractors: the marked characteristic, tried inside good
        samples. The hybrids get a modest fixed share (INTERESTING_SHARE)
        and no other privilege - they win duels or they disappear. Once
        the trend model has enough duels behind it, its NOMINEES join on
        the same terms (SURROGATE_SHARE): the additive extrapolation's
        best cross-combination guesses, offered to the posterior, never
        imposed on it.

        The observed points are in there so that the current champion can be
        re-selected. A duel between two never-seen configurations is a
        measurement with no common reference, and a chain of them lets the
        utility scale drift; re-fighting the incumbent is what keeps the
        comparisons anchored - and it is also what the user expects to see.

        The local half is split over the whole FRONTIER, not spent on one
        incumbent: the champion keeps half (it is the answer being polished),
        and the other attractors share the rest, so a close second basin goes
        on being refined instead of starving the moment it falls behind.
        """
        pool, keys = [], set()

        def add(point):
            key = self.space.key(point)
            if key not in keys:
                keys.add(key)
                pool.append(tuple(point))

        for point in self.points:
            add(point)
        if self.observations:
            centers = self._attractors or [self.best()]
            for center in centers:
                add(center)
            share = min(POOL_LOCAL_MAX,
                        POOL_LOCAL_BASE + POOL_LOCAL_GROWTH * self.duels)
            # "This is not going well", sustained, un-anneals the search:
            # every consecutive disliked duel halves the polishing share, so
            # the pool falls back toward broad exploration until something
            # lands outside the territory being disliked. Recovery is
            # automatic - the streak resets on the first normal grade.
            share /= 2 ** min(self._dislike_streak, 3)
            for _ in range(int(self.pool_size * share)):
                if len(centers) == 1 or self.rng.random() < 0.5:
                    center = centers[0]
                else:
                    center = centers[int(self.rng.integers(1, len(centers)))]
                add(self.space.perturb(center, self.rng,
                                       count=int(self.rng.integers(1, 3))))
            if self._interesting:
                for _ in range(int(self.pool_size * INTERESTING_SHARE)):
                    donor = self._interesting[
                        int(self.rng.integers(len(self._interesting)))]
                    base = centers[int(self.rng.integers(len(centers)))]
                    add(self._hybrid(donor, base))
            # The trend model's nominees - the cross-combination guesses no
            # neighbourhood perturbation and no uniform draw would put in
            # front of the posterior at any useful rate. A fixed modest
            # share, like the hybrids', and the same deal: they win duels
            # or they vanish. See THE TREND MODEL.
            if self.duels >= SURROGATE_MIN_DUELS:
                for point in self._surrogate_nominees(
                        int(self.pool_size * SURROGATE_SHARE)):
                    add(point)
        while len(pool) < self.pool_size:
            before = len(pool)
            add(self.space.sample(self.rng))
            if len(pool) == before and len(pool) >= max(2, self.space.size or 2):
                break     # the space itself is smaller than the pool
        return pool

    def _seed_duel(self):
        """The opening duels, before there is a posterior worth sampling.

        The first one is anchored: the BASELINE (first choice of every list,
        bottom of every range - i.e. the configuration the user would have had
        without any of this) against the most spread-out draw available. That
        makes the first question a meaningful one - "is any of this an
        improvement?" - and it puts a reference point into the data that later
        duels can be compared against. Every seed point after that is a
        space-filling pick (see _spread_pick), so the opening duels are a
        design over the space rather than a handful of coin flips.
        """
        if not self._shown and not self.observations:
            first = self.space.baseline()
        else:
            first = self._spread_pick()
        return first, self._spread_pick(anchor=first)

    def _cross_duel(self, attractors):
        """Two basins' champions, or None when there is no pair worth asking.

        The pair asked least often wins the slot; a pair already at the
        repeat cap, or too alike to grade, is not a question.
        """
        best = None
        for i in range(len(attractors)):
            for j in range(i + 1, len(attractors)):
                a, b = attractors[i], attractors[j]
                if self.space.separation(a, b) < MIN_DUEL_SEPARATION:
                    continue
                asked = self._times_asked(a, b)
                if asked >= MAX_PAIR_REPEATS:
                    continue
                rank = (asked, i + j)
                if best is None or rank < best[0]:
                    best = (rank, (a, b))
        return best[1] if best else None

    def _develop_duel(self, unproven):
        """A duel INSIDE the least-observed unproven basin: its
        representative against its own neighbourhood, or None.

        An unproven attractor cannot prove itself through cross-basin duels
        alone - those rank its current representative, and its current
        representative is wherever the thin evidence happens to sit, usually
        nowhere near the basin's actual peak. Finding the peak takes duels
        between points of the SAME basin, which Thompson sampling almost
        never asks for (its argmax lives in the champion's basin). Fired
        only while an unproven basin exists, so a settled search never pays
        for it.
        """
        if not unproven:
            return None
        counts = [sum(1 for point in self.points
                      if self.space.separation(target, point)
                      < FRONTIER_SEPARATION)
                  for target in unproven]
        target = unproven[int(np.argmin(counts))]
        for _ in range(64):
            candidate = self.space.perturb(target, self.rng,
                                           count=int(self.rng.integers(1, 3)))
            if self.space.separation(target, candidate) \
                    >= MIN_DUEL_SEPARATION \
                    and self._times_asked(target, candidate) \
                    < MAX_PAIR_REPEATS:
                return target, candidate
        return None

    def _explore_duel(self, pool, mean, cov):
        """The champion against an OPTIMISTIC unknown, or None when
        everything visible from here has been asked.

        Ranked by mean + sd, not by variance alone: pure max-variance points
        to wherever is unexplored, which in a large space spreads the budget
        thin; optimism points to promising-and-uncertain, which is what
        develops a second basin into something the frontier can hold - and a
        region sunk by dislikes has a low mean, so it is passed over without
        any special case."""
        champion = (self._attractors[0] if self._attractors
                    else pool[int(np.argmax(mean))])
        optimism = mean + np.sqrt(np.maximum(np.diag(cov), 0.0))
        for index in np.argsort(-optimism):
            candidate = pool[int(index)]
            if self.space.separation(champion, candidate) < MIN_DUEL_SEPARATION:
                continue
            if self._times_asked(champion, candidate) >= MAX_PAIR_REPEATS:
                continue
            return champion, candidate
        return None

    def _bluff_duel(self):
        """A pair drawn uniformly at random - the bluff - or None.

        No posterior, no attractors, no history: both sides are pure
        uniform draws, subject only to being tellable-apart and not asked
        past the cap. See BLUFF_MEAN for why being locally suboptimal on
        purpose is the point.
        """
        for _ in range(32):
            a = self.space.sample(self.rng)
            b = self.space.sample(self.rng)
            if self.space.separation(a, b) < MIN_DUEL_SEPARATION:
                continue
            if self._times_asked(a, b) >= MAX_PAIR_REPEATS:
                continue
            return a, b
        return None

    def _probe_duel(self):
        """A pair the LEARNED metric refuses and the PRIOR metric would have
        allowed - or None when the two agree, which is most of the time.

        THE ANSWER TO CENSORED FEEDBACK, and the reason the learned metric is
        safe to act on. Every duel path in this class filters on separation,
        the bluff included, so the moment a dimension's weight drops below
        the gate the search stops asking about it - and therefore stops being
        TOLD about it. The estimate would confirm itself forever, on however
        few labels shrank it, and a dimension wrongly written off in the
        first ten duels would stay written off for the session.

        So this asks exactly that question and nothing else: a pair differing
        in the shrunk directions, one the user would have been shown before
        the metric moved. A "distinct" answer pulls the weight back up; a
        "similar" answer confirms it and costs the one duel. Inert by
        construction while the weights are at their prior - the two gates
        then agree everywhere and there is no such pair to find.
        """
        for _ in range(64):
            a = self.space.sample(self.rng)
            b = self.space.perturb(a, self.rng, count=int(self.rng.integers(1, 3)))
            if self.space.key(a) == self.space.key(b):
                continue
            if self.space.separation(a, b) >= MIN_DUEL_SEPARATION:
                continue          # the learned metric would ask this anyway
            if self.space.raw_separation(a, b) < MIN_DUEL_SEPARATION:
                continue          # both metrics refuse it - genuinely too alike
            if self._times_asked(a, b) >= MAX_PAIR_REPEATS:
                continue
            # Remembered so that the answer counts in full - see
            # DISTINCT_WEIGHT. Without this the probe asks a question whose
            # reply it then discounts, and the recovery path it exists to
            # provide is hundreds of duels long.
            self._probed.add(self._pair_key(a, b))
            return a, b
        return None

    def _void_duel(self):
        """A pair from the unvisited reaches of the space, or None.

        Side A is the emptiest of VOID_BATCH uniform draws - the candidate
        farthest from EVERYTHING ever shown - and only counts when that
        distance clears VOID_SEPARATION: a "void" is a region the session
        has genuinely never come near, not merely the widest gap in a
        well-covered space. Side B maximizes the smaller of (its own
        emptiness, its distance to A), which lands it in a DIFFERENT void
        whenever one exists and on the far side of the same void when only
        one does - either way the duel brings back two observations from
        territory no history-driven mechanism was ever going to visit.
        """
        if not self._shown:
            return None
        candidates = [self.space.sample(self.rng) for _ in range(VOID_BATCH)]
        emptiness = [min(self.space.separation(shown, candidate)
                         for shown in self._shown)
                     for candidate in candidates]
        first = int(np.argmax(emptiness))
        if emptiness[first] < VOID_SEPARATION:
            return None
        a = candidates[first]
        best = None
        for candidate, empty in zip(candidates, emptiness):
            distance = self.space.separation(a, candidate)
            if distance < MIN_DUEL_SEPARATION:
                continue
            if self._times_asked(a, candidate) >= MAX_PAIR_REPEATS:
                continue
            score = min(empty, distance)
            if best is None or score > best[0]:
                best = (score, candidate)
        return (a, best[1]) if best else None

    def _rival_duel(self, attractors):
        """Fresh good samples from two DIFFERENT attractors, or None.

        The cross-basin duel ranks two basins' champions; this ranks their
        POPULATIONS: each side is the best of RIVAL_SAMPLES perturbations
        of its own attractor, kept only if it is strictly closer to its own
        center than to the rival's (a perturbation that flipped the basin-
        defining coordinate belongs to the other side's argument, not this
        side's). The attractor pair is drawn at random, so over a session
        every pair of basins gets compared, not only the two the fixed
        schedule keeps picking.
        """
        if len(attractors) < 2:
            return None
        picked = self.rng.choice(len(attractors), size=2, replace=False)
        home, away = attractors[int(picked[0])], attractors[int(picked[1])]
        sides = []
        for center, rival in ((home, away), (away, home)):
            members = [center]
            for _ in range(RIVAL_SAMPLES):
                candidate = self.space.perturb(center, self.rng,
                                               count=int(self.rng.integers(1, 3)))
                if self.space.separation(candidate, center) \
                        < self.space.separation(candidate, rival):
                    members.append(candidate)
            means, _ = self._posterior(members, with_covariance=False)
            sides.append([members[int(i)] for i in np.argsort(-means)])
        for a in sides[0][:4]:
            for b in sides[1][:4]:
                if self.space.separation(a, b) >= MIN_DUEL_SEPARATION \
                        and self._times_asked(a, b) < MAX_PAIR_REPEATS:
                    return a, b
        return None

    def next_duel(self):
        """The pair to show next: (A, B)."""
        # One duel, one set of probe answers: the render cache this reflects
        # changes between duels (this duel's own renders enter it), never
        # during one selection.
        self._probe_memo = {}
        self.last_tactic = None
        if self.duels < self.seed_duels:
            return self._remember(*self._seed_duel())

        # For the side effect as much as the answer: refreshes the INTERNAL
        # attractor list the pool and the duels below draw on - which
        # includes promising-but-unproven basins the public frontier floors
        # out, because a basin proves itself through exactly these duels.
        self.frontier()
        attractors = self._attractors
        pool = self._pool()
        if len(pool) < 2:
            return self._remember(*self._seed_duel())

        # The schedule: within every EXPLORE_CYCLE graded duels, one duel
        # ranks two basins against each other, one develops an unproven
        # basin from the inside (only while one exists), and one probes the
        # optimistic unknown; the rest are Thompson duels, which is what
        # converges. Every special slot falls through to Thompson when it
        # has nothing to ask, so none of them can stall the search.
        phase = self.duels % EXPLORE_CYCLE
        if phase == EXPLORE_CYCLE - 3 and len(attractors) >= 2:
            duel = self._cross_duel(attractors)
            if duel is not None:
                return self._remember(*duel)
        if phase == EXPLORE_CYCLE - 2:
            keeper_keys = {self.space.key(k) for k in self._keepers}
            unproven = [attractor for attractor in attractors
                        if self.space.key(attractor) not in keeper_keys]
            duel = self._develop_duel(unproven)
            if duel is not None:
                return self._remember(*duel)

        # A surprising winner gets its follow-up FIRST - before the schedule,
        # before exploration - while the trail is hot: against the champion
        # when the two can be told apart (which both verifies the upset and
        # ranks it), else against its own neighbourhood (which finds what
        # else is in there).
        if self._surprise is not None:
            target, self._surprise = self._surprise, None
            champion = attractors[0] if attractors else None
            if champion is not None \
                    and self.space.separation(target, champion) \
                    >= MIN_DUEL_SEPARATION \
                    and self._times_asked(target, champion) < MAX_PAIR_REPEATS:
                return self._remember(target, champion)
            duel = self._develop_duel([target])
            if duel is not None:
                return self._remember(*duel)

        mean, cov = self._posterior(pool)
        # A disliked duel redirects the very next one to exploration: "this
        # is not going well" answered by more of the same would be the search
        # not listening. One duel, not a mode - the posterior has already
        # absorbed the dislike, and Thompson resumes from wherever it leads.
        if self._redirect:
            self._redirect = False
            duel = self._explore_duel(pool, mean, cov)
            if duel is not None:
                return self._remember(*duel)

        # The stochastic tactics - see BLUFF_MEAN / VOID_MEAN / RIVAL_MEAN /
        # PROBE_MEAN. One roll, four bands, so each fires with exactly its
        # own rate and at most one per duel. Below the surprise follow-up and
        # the dislike redirect (a hot trail and "this is not going well" both
        # outrank a coin), above the Thompson draw - whose share the coins
        # dilute, which is the point: Thompson is the mechanism that locks in.
        roll = self.rng.random()
        edge = 1.0 / BLUFF_MEAN
        if roll < edge:
            duel = self._bluff_duel()
        elif roll < (edge := edge + 1.0 / VOID_MEAN):
            duel = self._void_duel()
        elif roll < (edge := edge + 1.0 / RIVAL_MEAN):
            duel = self._rival_duel(attractors)
        elif roll < edge + 1.0 / PROBE_MEAN:
            duel = self._probe_duel()
            # The one tactic the panel has to announce - see last_tactic.
            self.last_tactic = "probe" if duel is not None else None
        else:
            duel = None
        if duel is not None:
            return self._remember(*duel)

        if phase == EXPLORE_CYCLE - 1:
            duel = self._explore_duel(pool, mean, cov)
            if duel is not None:
                return self._remember(*duel)

        first, second = self._sample(mean, cov, 2)

        # The reuse preference - see REUSE_MARGIN and THE PRICE OF A
        # QUESTION. Active only when the host wired a probe and the streak
        # cap has not tripped, and only HERE, in the Thompson slot: the
        # exploration slots above are about where they land, not what they
        # cost. The margin is REUSE_MARGIN judgement-noise units under the
        # LEARNED noise, so what is ever traded away is a sampled-utility
        # difference the grader could not have resolved.
        margin = None
        if self.reuse_probe is not None \
                and self._reuse_streak < REUSE_STREAK_MAX:
            margin = REUSE_MARGIN * max(self._hyper[2], 1e-3)

        index_a = int(np.argmax(first))
        if margin is not None:
            # Walking down the first sample's own ranking keeps this a
            # Thompson choice: the first CACHED configuration inside the
            # margin asks (to the grader) the argmax's question at a
            # twentieth of the cost, and if none is, the argmax stands.
            for candidate in np.argsort(-first):
                candidate = int(candidate)
                if float(first[candidate]) < float(first[index_a]) - margin:
                    break
                if self._rendered(pool[candidate]):
                    index_a = candidate
                    break

        # B is the highest-sampled configuration that is VISIBLY different
        # from A, not simply the highest-sampled one. Walking down the second
        # sample's own ranking keeps this a Thompson choice - the posterior
        # still decides which challenger is worth asking about - while
        # guaranteeing the question can be answered at all. Both samples
        # peaking on the same point is the same case and needs no branch of
        # its own: a point's separation from itself is 0. A pair already
        # asked MAX_PAIR_REPEATS times is walked past the same way: the
        # posterior's next-favourite challenger is a new question, the
        # favourite re-asked a third time is not. With the reuse preference
        # active, the walk continues past the first valid challenger - but
        # never below the margin - looking for a valid CACHED one; a fresh
        # side costs a whole generation, so the same
        # question-the-grader-cannot-tell-apart rule applies to B as to A.
        index_b, fallback = None, None
        for candidate in np.argsort(-second):
            candidate = int(candidate)
            if index_b is not None and (
                    margin is None
                    or float(second[candidate])
                    < float(second[index_b]) - margin):
                break
            separation = self.space.separation(pool[index_a], pool[candidate])
            if separation >= MIN_DUEL_SEPARATION:
                if self._times_asked(pool[index_a],
                                     pool[candidate]) >= MAX_PAIR_REPEATS:
                    continue
                if index_b is None:
                    index_b = candidate
                    if margin is None or self._rendered(pool[candidate]):
                        break
                elif self._rendered(pool[candidate]):
                    index_b = candidate
                    break
                continue
            if index_b is None and separation > 0 \
                    and (fallback is None or separation > fallback[0]):
                fallback = (separation, candidate)
        if index_b is None:
            # The whole pool is within a hair of A. That means the space
            # itself is too small to hold a visible difference, so the widest
            # pair available is the best question there is.
            index_b = fallback[1] if fallback else (index_a + 1) % len(pool)
        return self._remember(pool[index_a], pool[index_b])

    # -- the answer ------------------------------------------------------

    def _polish(self, start, frozen=()):
        """Coordinate ascent on the posterior mean from `start`: for each
        dimension in turn, try every alternative and keep the best. A greedy
        search over the FULL combinatorial space (not over the pool), which is
        what turns a dozen comparisons into an answer from a space of
        millions. Returns (point, posterior mean).

        Dimensions in `frozen` are not moved. That is how the frontier keeps
        a second basin a second basin: from an imperfect start, the single
        move "switch to the champion's model" often looks better than any
        within-basin move - the champion's neighbourhood is simply better
        mapped - and unrestricted ascent walks the runner-up straight into
        the champion. Freezing the dims that make it a DIFFERENT answer
        leaves the ascent free to do what it is for: tuning the detail.
        """
        current = list(start)
        score = float(self._posterior([tuple(current)], with_covariance=False)[0][0])
        for _ in range(len(self.space) + 2):
            improved = False
            for index in self.space.live:
                if index in frozen:
                    continue
                dimension = self.space.dimensions[index]
                alternatives = dimension.alternatives(current[index])
                if not alternatives:
                    continue
                candidates = []
                for value in alternatives:
                    trial = list(current)
                    trial[index] = value
                    candidates.append(tuple(trial))
                means, _ = self._posterior(candidates, with_covariance=False)
                pick = int(np.argmax(means))
                if float(means[pick]) > score + 1e-9:
                    current = list(candidates[pick])
                    score = float(means[pick])
                    improved = True
            if not improved:
                break
        return tuple(current), score

    def best(self):
        """The configuration with the highest posterior utility.

        Not simply the best-graded point seen: the posterior generalizes -
        it can prefer a configuration nobody has looked at, because the model
        has learned which model, which offset and which LoRA weight each
        carried their own weight - and coordinate ascent is what walks there.

        Defined as the frontier's champion, not a separate computation: the
        frontier polishes from DIVERSE starts and can summit a hill the
        single strongest observed point's ascent never reaches - measured on
        the hidden-gem bench, where the two disagreed and best() was the one
        that was wrong. One answer, one code path.
        """
        if not self.observations:
            return self.space.baseline()
        keepers = self.frontier()
        return keepers[0] if keepers else self.space.baseline()

    def _candidates(self, centers, per_center=24, pool=POOL_SIZE,
                    uniform=None):
        """A candidate set for frontier/portfolio: everything observed, the
        neighbourhoods of `centers`, and uniform draws up to `pool`.

        `pool` and `per_center` are arguments rather than constants because
        `population` asks for as many diverse answers as the panel's count
        offers: a set sized for choosing ONE duel cannot hold sixty
        configurations that are all different from each other.

        `uniform` MAKES THE UNIFORM DRAWS A QUOTA instead of whatever is left
        over under `pool`, and a caller that wants to see the whole space has
        to ask for it. The default is the leftover behaviour, which is right
        for a duel - the pool is meant to tighten around the attractors as
        evidence accumulates (see POOL_LOCAL_MAX) - and quietly wrong for a
        collage: `population` scales `per_center` with the count asked for,
        so at N=20 the neighbourhoods alone already exceeded `pool` and the
        uniform loop never ran once. The far half of the space was not
        rejected for being bad, it was never a candidate, and no amount of
        cleverness downstream can spread a selection over points that are
        not in the set.
        """
        candidates, keys = [], set()

        def add(point):
            key = self.space.key(point)
            if key not in keys:
                keys.add(key)
                candidates.append(tuple(point))

        for point in self.points:
            add(point)
        for center in centers:
            for _ in range(per_center):
                add(self.space.perturb(center, self.rng,
                                       count=int(self.rng.integers(1, 3))))
        target = pool if uniform is None else len(candidates) + int(uniform)
        # x4 because `add` dedupes: in a small space the draws collide, and
        # the loop has to be allowed to give up rather than spin.
        for _ in range(max(pool, int(uniform or 0)) * 4):
            if len(candidates) >= target:
                break
            add(self.space.sample(self.rng))
        return candidates

    def _diverse_top(self, candidates, means, count, separation, best_first=True):
        """Greedy selection down a ranking, keeping only candidates at least
        `separation` from everything already kept.

        RANKING FIRST, distance only as a filter - which is what a duel
        wants (the next pair has to be worth asking about, and the best
        available answer is the interesting one) and NOT what a collage
        wants: near the top of a posterior every candidate is a
        perturbation of the champion, so this fills up inside one basin and
        every entry clears the filter against its neighbour while the sheet
        is N views of one answer. `population` selects by island instead.
        """
        picked = []
        order = np.argsort(-means) if best_first else np.argsort(means)
        for index in order:
            point = candidates[int(index)]
            if all(self.space.separation(point, kept) >= separation
                   for kept in picked):
                picked.append(point)
            if len(picked) >= count:
                break
        return picked

    @staticmethod
    def _components(pairwise, hop):
        """Index groups of `pairwise`, joined transitively at `hop`.

        The ISLANDS: two good configurations belong to the same one when a
        CHAIN of small steps connects them - which is the definition that
        matches what the word means. Two points a mile apart at opposite
        ends of one long plateau are one island (you can walk between them
        without leaving the good region); two points the same distance apart
        with badness between them are two.

        Union-find over the matrix rather than clustering: there is no k to
        choose, no centroid to place, nothing that can split a plateau down
        the middle because a parameter said there should be three of
        something. The one input is the hop, and the hop is a probability -
        see ISLAND_CONFIDENCE.
        """
        size = len(pairwise)
        parent = list(range(size))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        close = np.asarray(pairwise) < hop
        for row in range(size):
            for column in np.flatnonzero(close[row][row + 1:]) + row + 1:
                a, b = find(row), find(int(column))
                if a != b:
                    parent[a] = b
        groups = {}
        for index in range(size):
            groups.setdefault(find(index), []).append(index)
        return list(groups.values())

    def islands(self):
        """The ISOLATED HIGH-QUALITY DOMAINS: one entry per separate good
        region the model believes in, best first.

        Returns `[(peak, size)]` - the best configuration in each island
        under the posterior mean, and how many of the pooled good points
        landed in it (a rough read of how much room that island has).

        WHAT THE NUMBER MEANS, and what it does not. This counts SEPARATE
        answers, not distinguishable images: a taste where only the model
        matters has one island the size of a subspace, and every point in it
        is reachable from every other without passing through anything bad.
        That is why the button's count is not this number - see `capacity`,
        which counts how many mutually-distinct samples the region can hold -
        but the two are complementary and the panel says both: three islands
        holding forty distinct samples between them is a different thing to
        know than either half alone.

        OVER `_good_pool`, LIKE EVERYTHING ELSE HERE. Frontier members seed
        that pool and the full internal basin archive supplies their
        neighbourhoods, but none bypasses the absolute quality test: the
        champion of a session whose every verdict was "bad" is still bad.
        The difference-variance calculation in `_quality` preserves a
        supported runner-up without granting a presentation keeper immunity
        from the evidence.

        IT NEEDS THE SIMILARITY ROW. Under the prior metric every dimension
        counts 1.0, which is over ISLAND_HOP, so every categorical
        difference disconnects and "island" degenerates into "cell of the
        categorical grid". The labels are what shrink the rows that do not
        change the picture (measured: an inert row falls from 1.0 to ~0.4
        over ninety graded duels) until only the ones that do can separate
        two islands.
        """
        if not self.observations:
            return []
        chosen, scores, _thompson = self._good_pool()
        if not chosen:
            return []
        groups = self._components(self.space.separations(chosen), ISLAND_HOP)
        peaks = []
        for group in groups:
            best = int(max(group, key=lambda index: scores[index]))
            peaks.append((chosen[best], len(group), float(scores[best])))
        peaks.sort(key=lambda triple: -triple[2])
        return [(point, size) for point, size, _score in peaks]

    def _diverse_centers(self, utilities, count):
        """Maximin pick among the DECENT observed points: the strongest
        first, then always the decent point FARTHEST from everything chosen.

        Diversity-first, not utility-first, and that is the load-bearing
        choice: ranked by utility, the runners-up are always the champion's
        own basin wearing a different prompt - one categorical difference
        clears any separation gate - and a frontier grown from them collapses
        to one basin the moment it is polished. Farthest-point selection
        reaches the OTHER basin whenever it holds any decent point at all.

        "Decent" is the top HALF of what was observed - and deliberately NOT
        "above par": an underexplored basin idles just below zero until the
        cross-basin duels prove it, and those duels only happen to basins
        this selection admits. Excluding it here starves it in exactly the
        state it needs feeding - measured on the two-peak test taste, where
        the second peak's whole population sat at -0.05. "Above par" is the
        PUBLIC keeper floor's test (see frontier), where it belongs.
        Permissive is cheap: a mediocre center costs only a polish - it
        climbs into a real basin and the dedupe collapses it.
        """
        if not len(utilities):
            return []
        threshold = float(np.quantile(utilities, 0.5))
        decent = [index for index in range(len(self.points))
                  if utilities[index] >= threshold]
        centers = [int(np.argmax(utilities))]
        while len(centers) < count and decent:
            picked, widest = None, 0.0
            for index in decent:
                separation = min(
                    self.space.separation(self.points[index],
                                          self.points[center])
                    for center in centers)
                if separation > widest:
                    widest, picked = separation, index
            if picked is None or widest < FRONTIER_SEPARATION:
                break
            centers.append(picked)
        return [self.points[index] for index in centers]

    def frontier(self, count=FRONTIER_SIZE):
        """The diverse top of the posterior: up to `count` configurations,
        each polished by coordinate ascent, each at least FRONTIER_SEPARATION
        from the others, best first.

        This is the multi-answer form of `best()`, and it exists because
        taste is not unimodal: a user who likes two unrelated looks is the
        normal case. The frontier is recomputed every duel and is what the
        candidate pool polishes around - so a strong second basin keeps
        being refined - and what the cross-basin duels rank against each
        other.
        """
        if not self.observations:
            return [self.space.baseline()]
        count = max(int(count), 1)
        archive_limit = max(count, ATTRACTOR_ARCHIVE_SIZE)
        if self._frontier_stamp == self.duels \
                and self._frontier_limit >= archive_limit:
            return list(self._frontier_public[:count])
        self._fit()

        # One polish per maximin center (see _diverse_centers), with twice as
        # many centers as ARCHIVE slots: two centers can still share a basin
        # (their distinguishing dims may be numeric, which the freeze below
        # leaves loose), and without the surplus every such collapse would
        # cost a basin. FRONTIER_SIZE is only how many recipes STOP prints;
        # acquisition and N-GOOD retain the wider archive. Non-champion
        # centers are polished with the OPAQUE categorical dims that
        # distinguish them from every retained member frozen - see _polish
        # for why letting those move hands the runner-up basin to the champion.
        result = []
        for center in self._diverse_centers(self.f, archive_limit * 2):
            if len(result) >= archive_limit:
                break
            frozen = {index for index, dimension
                      in enumerate(self.space.dimensions)
                      if dimension.kind == Dimension.CHOICE
                      and dimension.ordinals is None
                      and any(int(center[index]) != int(kept[index])
                              for kept, _score in result)}
            point, score = self._polish(center, frozen=frozen)
            if all(self.space.separation(point, kept) >= FRONTIER_SEPARATION
                   for kept, _score in result):
                result.append((point, score))
        result.sort(key=lambda pair: -pair[1])
        # TWO lists come out of this, on purpose. The INTERNAL one
        # (_attractors) keeps every distinct basin including the promising-
        # but-unproven - it feeds the pool split and the cross-basin duels,
        # which are exactly how an unproven basin gets the evidence to prove
        # itself, and flooring it would starve the runner-up the moment it
        # fell behind. The PUBLIC one - the return value, the keepers
        # rendered on stop - applies the keeper floor: the champion at
        # whatever it is worth, everything behind it at FRONTIER_KEEP of the
        # champion and never below par. Diversity of the GOOD, not one
        # answer plus filler.
        self._attractors = [point for point, _score in result]
        if result:
            # Scale-free floor - see KEEP_VS_CHAMPION / KEEP_VS_PAR. The
            # probabilities are computed under the learned judgement noise,
            # so the test means the same thing whether the posterior is
            # stretched by strong anchors or compressed toward 0 without
            # them.
            scale = _SQRT_2 * max(self._hyper[2], 1e-3)
            champion_score = result[0][1]
            result = result[:1] + [
                (point, score) for point, score in result[1:]
                if float(_cdf((score - champion_score) / scale))
                >= KEEP_VS_CHAMPION
                and float(_cdf(score / scale)) >= KEEP_VS_PAR]
        self._frontier_public = [point for point, _score in result]
        self._keepers = list(self._frontier_public[:FRONTIER_SIZE])
        self._frontier_stamp = self.duels
        self._frontier_limit = archive_limit
        return list(self._frontier_public[:count])

    def suggest(self, good=True):
        """ONE configuration from the good (or bad) end, varied per call.

        The GOOD/BAD buttons' engine: a draw from the portfolio rather than
        its top entry, because the whole point of asking twice is getting
        two different answers - the generator is a distribution over the
        wanted (or unwanted) region, not a lookup of its argmax.
        """
        picks = self.portfolio(6, good=good)
        if not picks:
            return self.space.sample(self.rng)
        return picks[int(self.rng.integers(len(picks)))]

    def portfolio(self, count=8, good=True,
                  min_separation=MIN_DUEL_SEPARATION):
        """`count` diverse configurations from the good (or bad) end of the
        posterior - the "generator" the search converges towards.

        `good=True` samples the high end: diverse, acceptable-or-better
        configurations near the attractors, for generating variety rather
        than one recipe. `good=False` samples the low end - useful as a
        check ("is THIS what it thinks I hate?") and as a probe of whether
        the bad regions were actually mapped. The low end only means
        anything once dislike clicks have anchored it: with comparisons
        alone the posterior is translation-invariant and "bad" has no
        address.
        """
        if not self.observations:
            return []
        self._fit()
        centers = self._diverse_centers(self.f if good else -self.f,
                                        FRONTIER_SIZE)
        candidates = self._candidates(centers)
        means, _ = self._posterior(candidates, with_covariance=False)
        return self._diverse_top(candidates, means, count, min_separation,
                                 best_first=good)

    # -- how many good answers are there, and what are they ----------------
    #
    # The frontier answers "which few should I keep". These two answer the
    # other half of the same question - how many good answers this taste has
    # in it at all, and give me that many - which is what turns a search into
    # a generator: a converged solver is not one recipe, it is a REGION, and
    # the size of that region is a thing the model can be asked about rather
    # than a number the user has to guess.

    def _quality(self, means, covariance, champion=None):
        """Candidates confidently near the champion AND above par.

        The first event is ``f(x) >= f(champion) - margin``. Its uncertainty
        is therefore the variance of a DIFFERENCE,

            var(x) + var(champion) - 2 cov(x, champion),

        not x's marginal variance. Near points are strongly correlated with
        the champion; discarding that covariance made the model reject the
        well-supported interior of a good basin while retaining whatever
        isolated points happened to clear the scalar approximation.

        The second event is ``f(x) > 0``. It consumes the absolute signal in
        the three rows: top/middle anchor both samples on the usable side of
        par, bottom anchors both below it. A relative winner from an entirely
        bad session can no longer enter N-GOOD merely because every search has
        to have a champion.

        A one-dimensional variance array is accepted for the historical test
        harness and treated conservatively (no covariance available). Live
        selection always passes the full posterior covariance.
        """
        means = np.asarray(means, dtype=float)
        uncertainty = np.asarray(covariance, dtype=float)
        if means.size == 0:
            return np.zeros(0, dtype=bool)
        if uncertainty.ndim == 2:
            variances = np.maximum(np.diag(uncertainty), 1e-12)
            champion_index = int(np.argmax(means))
            champion_score = float(means[champion_index])
            difference_variance = np.maximum(
                variances + variances[champion_index]
                - 2.0 * uncertainty[:, champion_index], 1e-12)
        else:
            variances = np.maximum(uncertainty, 1e-12)
            champion_score = (float(champion) if champion is not None
                              else float(means.max()))
            difference_variance = variances
        margin = QUALITY_MARGIN * max(self._hyper[2], 1e-3)
        near = _cdf((means - champion_score + margin)
                    / np.sqrt(difference_variance))
        above_par = _cdf(means / np.sqrt(variances))
        return ((near >= CAPACITY_CONFIDENCE)
                & (above_par >= GOOD_VS_PAR_CONFIDENCE))

    def _champion_score(self, means):
        """The utility the quality floor is measured against.

        The best posterior mean anywhere it has been evaluated - the observed
        points included, because a batch of uniform draws in a large space
        will usually miss the basin the search spent its duels proving, and a
        floor set from the draws alone would drift down with them.
        """
        observed = float(self.f.max()) if len(self.f) else 0.0
        return max(observed, float(means.max()) if len(means) else 0.0)

    def capacity(self, min_separation=POPULATION_SEPARATION):
        """How many DISTINCTLY DIFFERENT good configurations the model
        believes the space holds, as one integer.

        IT IS WHAT THE BUTTON CAN DELIVER, counted by running the button's
        own selection and taking the length. One algorithm, two callers:
        `population(n)` places entries until it runs out or reaches n, and
        this places them until it runs out. The number in the box is then
        the answer to the only question the box is asked - "how many
        distinct good samples can I have" - rather than a separate estimate
        of a related quantity that the button then fails to match.

        THAT IT USED TO BE A SEPARATE ESTIMATE is the reason this is spelled
        out. It was a volume-and-packing calculation: what share of uniform
        draws clear the quality floor, times the space's size, divided by
        how many good points sit within one separation ball of each other.
        A fine estimator of the good region's packing number, and not the
        same quantity as "configurations this solver can hand you that are
        all mutually distinct" - which is what N-GOOD spends generations on.
        It reported 40 where the button could produce 6, at which point the
        box is not information, it is a target the button misses.

        SMALL NUMBERS ARE THE HONEST ONES here, and they will be smaller
        than the old estimate by a lot. At COLLAGE_CONFIDENCE the entries
        have to be different ANSWERS, and most spaces hold a handful of
        those and thousands of variations on them. The variations are still
        reachable - ask for more than the box says and `population` gives
        what it can, reporting the confidence it actually achieved.

        NO KEEPER FLOOR any more, and that is deliberate rather than an
        omission: keepers only have to clear FRONTIER_SEPARATION, which is
        a 39% chance of looking different. Flooring this at their count
        would put a number in the box that the collage cannot honour, which
        is the exact failure being removed.

        IT MUST NOT OVER-PROMISE, which is a weaker claim than "it equals
        what the button returns" and the right one. Both run the same
        selection over the same KIND of pool, but the button's pool is
        sized by what was asked for and is therefore larger, so a press for
        more than the box says can legitimately find more. Under-promising
        costs nothing; over-promising is a target missed on every press.

        THE WORST OF SEVERAL DRAWS, because a greedy maximal set is not a
        unique quantity: how many entries fit depends on which one is taken
        first, and the button takes its first from a Thompson draw that
        differs every press. Measured on one seed, a box reading 2 was met
        by presses returning 2, 2, 2, 1, 1 - so the box was wrong two times
        in five while being a perfectly good estimate of the typical case.
        SELECTION_DRAWS orderings from the ONE covariance factorisation, and
        the smallest set any of them yields is what the box says. The press
        takes the LARGEST of the same number of tries, so the inequality
        between them holds by construction and not by luck.
        """
        if not self.observations:
            return 0
        eligible, means, thompson = self._good_pool()
        if not eligible:
            return 0
        pairwise = self.space.separations(eligible)
        placed = len(self._distinct_indices(pairwise, means, CAPACITY_MAX,
                                            min_separation))
        for scores in thompson(SELECTION_DRAWS):
            placed = min(placed, len(self._distinct_indices(
                pairwise, scores, CAPACITY_MAX, min_separation)))
        return int(min(max(placed, 1), CAPACITY_MAX))

    def _good_pool(self, count=0):
        """The candidate set the count and the collage are both computed
        over: `(eligible, means, thompson)`, everything already filtered to
        what clears the quality floor.

        `means` is the posterior mean of each eligible candidate, which is
        what `capacity` counts by. `thompson(n)` draws n posterior SAMPLES
        over the same candidates - what `population` ranks by, so that two
        presses of one solver state differ, and what `capacity` uses to see
        how much the count moves with the ordering. A callable rather than
        an array because the draw costs a Cholesky, and n draws share it.

        ONE CONSTRUCTION, because the box and the button disagreeing was
        traced to two: `capacity` used to sample the space UNIFORMLY while
        `population` sampled around the attractors. Uniform draws in a large
        space land almost entirely in unexplored territory, where the
        posterior is uncertain and the quality floor rejects on principle -
        so the box counted over a handful of points while the button chose
        over hundreds, and the two numbers had no reason to resemble each
        other. Measured at box 1 / button 17 on one seed.

        Half the pool is neighbourhoods of the attractors and half is
        uniform - see `_candidates` - so both halves of the question ("what
        do we already believe in" and "is there anything good out there we
        have not looked at") are represented at every size.
        """
        self._fit()
        if not self._attractors or self._frontier_stamp != self.duels:
            self.frontier()
        keepers = list(self._keepers)
        # The internal archive, not the four-entry STOP presentation. A fifth
        # supported basin must keep receiving local candidates or it is erased
        # by the acquisition policy rather than rejected by evidence.
        centers = list(self._attractors) or self._diverse_centers(
            self.f, ATTRACTOR_ARCHIVE_SIZE)
        seen = {self.space.key(point) for point in keepers}
        pool = int(min(max(POOL_SIZE, int(count) * POPULATION_OVERSAMPLE),
                       POPULATION_POOL_MAX))
        candidates = keepers + [
            point for point in self._candidates(
                centers,
                per_center=max(8, pool // (2 * max(1, len(centers)))),
                pool=pool,
                uniform=pool // 2)
            if self.space.key(point) not in seen]
        means, cov = self._posterior(candidates)
        good = self._quality(means, cov)
        keep = np.flatnonzero(good)
        if not len(keep):
            return [], np.zeros(0), (lambda draws=1: np.zeros((draws, 0)))
        return ([candidates[int(index)] for index in keep], means[keep],
                lambda draws=1: np.asarray(
                    self._sample(means, cov, draws))[:, keep])

    def population(self, count, min_separation=POPULATION_SEPARATION):
        """Up to `count` high-quality configurations, SPREAD ACROSS the good
        region rather than crowded at the top of it - the extraction
        interface behind N-GOOD.

        THE PROMISE IS A PROBABILITY, and it is checked against the model
        that was fitted from the user's own similar/distinct clicks: every
        PAIR of entries is at least COLLAGE_CONFIDENCE likely to look
        different. Not "0.2 apart", not "0.5 apart" - both of those were the
        bar at one time, and in the fitted model they mean an 82% and a 61%
        chance of the pair looking IDENTICAL. The sheet came back exactly as
        those numbers predicted, which is what "N-GOOD returns N of the same
        image" was.

        ISLANDS FIRST, then within them. `islands` finds the separate good
        regions - components of the good set, joined by hops the model
        cannot tell apart - and this walks them ROUND ROBIN: the best
        unpicked entry of island 1, then of island 2, and so on, lap after
        lap. So a collage of three over three islands is one sample of each
        separate answer, and a collage of twelve is four looks at each. The
        structure of the good region is what orders the sheet, rather than a
        ranking that never leaves the champion's basin (the previous
        failure) or a farthest-point walk that prefers the region's outer
        margins (the one before that).

        THE PAIRWISE BAR STILL HOLDS ACROSS ISLANDS, not only within them.
        Two islands are separate places, which does not by itself make their
        peaks look different - a chain of bad configurations can lie between
        two good ones that resemble each other - so every candidate is
        checked against EVERYTHING already picked, wherever it came from.

        FEWER THAN ASKED IS AN ANSWER. Nothing here relaxes the quality floor
        or the bar to fill a quota: a collage padded with mediocre or
        near-duplicate entries would misreport the model's own belief. If the
        model cannot yet support any good entry, the result is empty and the
        panel says so.

        WHAT WAS ACHIEVED IS REPORTED, in `population_report`: how many
        islands the entries came from, the lowest pairwise probability on
        the sheet, and whether the bar had to be given up on. The panel
        prints it. A promise nobody can check is not a promise.

        AS MANY AS IT CAN PLACE. Which entries fit depends on which one is
        taken first, so several orderings are tried and the fullest sheet
        wins - see SELECTION_DRAWS. Without that the press could come back
        under the number the box had just offered, not because the region
        had shrunk but because one draw happened to open with a point that
        crowded out the rest.

        VARIED PER CALL: the island order and the pick inside each island
        follow a DRAW from the posterior rather than its mean (the same
        Thompson logic the duels are chosen by), so two presses of one
        solver state are two different samples of one taste.

        KEEPERS ARE CANDIDATES, NOT EXEMPTIONS. Every entry clears the same
        near-champion and above-par posterior tests. This deliberately allows
        an empty result early, or after an all-bad session: rendering the
        relative winner under a GOOD label would be a stronger false claim
        than saying the model has no supported good population yet.
        """
        count = max(int(count), 1)
        if not self.observations:
            return []
        # The same pool the count is taken over, sized by what was asked
        # for - see `_good_pool`.
        eligible, _means, thompson = self._good_pool(count)
        if not eligible:
            self.population_report = {"islands": 0, "total_islands": 0,
                                      "confidence": 0.0}
            return []
        # ONE pairwise matrix, shared by the island decomposition and every
        # bar test below - the same numbers, so the sheet cannot be selected
        # under one notion of distance and reported under another.
        pairwise = self.space.separations(eligible)
        # THE BEST OF SEVERAL ORDERINGS - see SELECTION_DRAWS. Greedy sets
        # differ in size by which entry is taken first, and the press is
        # about to spend a generation per entry, so it is worth a few
        # milliseconds to find the ordering that places the most. Ties keep
        # the FIRST draw, which is the Thompson one, so two presses of one
        # solver state still differ.
        scores, picked = None, []
        for candidate_scores in thompson(SELECTION_DRAWS):
            attempt = self._distinct_indices(pairwise, candidate_scores,
                                             count, min_separation)
            if len(attempt) > len(picked):
                scores, picked = candidate_scores, attempt
            if len(picked) >= count:
                break
        groups = self._components(pairwise, ISLAND_HOP)
        island_of = {index: number for number, group in enumerate(
            sorted(groups, key=lambda g: -max(scores[i] for i in g)))
            for index in group}

        # NO FALLBACK, and the absence is the feature. A "the bar could not
        # be met, here are the most different ones available" path was
        # written here and deleted: it fired exactly when the good region
        # holds one or two answers and the user asked for twelve, which is
        # the case N-GOOD was reported broken for, and it filled the sheet
        # with look-alikes under a disclaimer. A short collage is the
        # honest answer to "give me twelve distinct samples" when there are
        # two - and it is the only answer that cannot waste an hour.
        gaps = [pairwise[a][b] for position, a in enumerate(picked)
                for b in picked[position + 1:]]
        self.population_report = {
            "islands": len({island_of[index] for index in picked}),
            "total_islands": len(groups),
            "confidence": (distinct_probability(min(gaps)) if gaps else 1.0),
        }
        return [eligible[index] for index in picked]

    def _distinct_indices(self, pairwise, scores, count, bar):
        """Up to `count` indices, every pair at least `bar` apart, taken
        ISLAND BY ISLAND - the selection behind both N-GOOD and the count
        on its label.

        The islands come first because they are the structure of the answer:
        one lap of the round robin puts one sample of every separate good
        region on the sheet, and the laps after it deepen each island in the
        same order, so a collage of three over three islands is three
        answers and a collage of twelve is four looks at each. Inside an
        island the best candidate under `scores` wins - so the peaks lead
        and the margins fill in behind them, which is the opposite of what a
        farthest-point walk does.

        The bar is checked against EVERY index already taken, not just the
        ones from the same island: two separate places can still look alike,
        and the promise is about pairs on the sheet, not about topology.

        A lap that places nothing ends it. That is what "fewer than asked is
        an answer" is, mechanically - there is no relaxation step, and the
        caller decides what to say about a short result.
        """
        groups = self._components(pairwise, ISLAND_HOP)
        groups.sort(key=lambda group: -max(scores[index] for index in group))
        order = [sorted(group, key=lambda index: -scores[index])
                 for group in groups]
        taken, picked = set(), []
        while len(picked) < count:
            placed = False
            for group in order:
                if len(picked) >= count:
                    break
                for index in group:
                    if index in taken:
                        continue
                    if all(pairwise[index][other] >= bar for other in picked):
                        taken.add(index)
                        picked.append(index)
                        placed = True
                        break
            if not placed:
                break
        return picked

    def confidence(self):
        """How sure the model is that `best()` beats a typical rival, 0..1.

        Read as a probability: it is P(best preferred) against the median of
        the current candidate pool, under the posterior. Shown to the user
        because "12 duels in" says nothing on its own - eight duels over two
        dimensions is a finished search and eight over eight is barely
        started.
        """
        if not self.observations:
            return 0.0
        pool = self._pool()
        mean, cov = self._posterior(pool)
        best_index = int(np.argmax(mean))
        median = float(np.median(mean))
        spread = math.sqrt(max(float(cov[best_index, best_index]), 0.0)
                           + self._hyper[2] ** 2)
        if spread <= 0:
            return 1.0
        return float(_cdf((float(mean[best_index]) - median) / (_SQRT_2 * spread)))

    def status(self):
        return {
            # Graded duels, not observations: a quality grade adds extra
            # observation rows, and the user counts questions answered.
            "duels": self.duels,
            "points": len(self.points),
            # The PUBLIC count - basins that cleared the keeper floor. The
            # internal list also holds unproven ones, which would make "N
            # distinct looks" a promise the keepers then fail to keep.
            "attractors": len(self._keepers),
            # How many good answers the space holds, not how many the
            # frontier keeps - see `capacity`. This is the count the panel
            # puts on the collage button and what a press spends.
            "capacity": self.capacity(),
            # ISOLATED good regions - separate answers rather than
            # distinguishable samples. A plateau taste is one island holding
            # dozens of distinct samples; two unrelated looks that both work
            # are two islands. Knowing both numbers is the difference
            # between "vary this" and "choose between these".
            "islands": len(self.islands()),
            "confidence": self.confidence(),
            # What the similarity row has taught about each row's visibility
            # - all ones until it has been used. Conditional rows expose the
            # compact mean here and their per-pick rates separately below.
            "metric": list(self.space.weights),
            "metric_contexts": {
                self.space.dimensions[index].name: list(rates)
                for index, rates in self.space.conditional_weights.items()
            },
            "similarities": len(self.similarities),
            "lengthscale": self._hyper[0],
            "theta": self._hyper[1],
            "noise": self._hyper[2],
        }
