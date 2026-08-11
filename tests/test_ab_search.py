"""CNPro A/B: the search converges, and the recipe it prints is the truth.

WHAT IS PINNED HERE
-------------------
1. **The comparison model is oriented correctly.** 0 means A, 10 means B, and a
   sign error there is a search that converges on the configuration the user
   liked least - while looking exactly like a working search from the outside.
   So the convergence check is run TWICE, the second time with the grades
   deliberately reversed, and the reversed run has to converge on the WORST
   configuration. A test that only checked "it finds something good" would pass
   on a build with the sign flipped and no data.
2. **5 is an answer, not a refusal.** A tie has to pull two configurations'
   utilities together; if it were being discarded, the posterior after a tie
   would be indistinguishable from the posterior with no observation at all.
3. **It beats not having a model.** The recommended configuration has to be
   substantially better than an average draw from the same space - otherwise
   the whole apparatus is an expensive random number generator.
4. **The recipe reconstructs the configuration.** The string the STOP button
   prints is the only durable output of a search, and Set parses it back. So
   the round trip is checked on values chosen to break it: a prompt containing
   the separator, a percent sign, several rows touching the prompt at once.
5. **The field table names fields that exist.** the script's list of searchable unit
   settings is written out by hand; a field renamed in the dataclass would
   leave a row that offers a setting nothing reads, silently.
6. **The three rows mean what they say.** A bottom-row grade pushes both sides
   below par while the grade still orders them; either on-track row pushes
   both above par and also supplies its similar/distinct label. A legacy
   comparison with no row verdict remains comparison-only. A dislike arms the
   redirect: the next duel explores, and the streak clears on the first
   on-track grade.
7. **Taste is allowed to be bimodal.** Against a two-peak synthetic taste the
   frontier has to hold BOTH peaks, visibly separated, and portfolio() has to
   draw good samples from the good end and bad samples from the bad end.
8. **No duel is re-asked past the repeat cap, and the seed duels are a
   design.** The opening duels have to show most of a categorical dimension's
   levels rather than land six points wherever uniform chance puts them.
9. **An upset is followed up, not filed away.** A decisive win by a
   configuration the model wrote off arms one immediate verification duel -
   a 10 is neither "here we are" nor one comparison to be averaged into the
   smoothing prior; it is the instruction to look closer, now.
10. **"Interesting" is not evidence.** The per-image mark ("overall bad, but
    this characteristic is tempting") must leave the posterior bit-identical
    - and must surface in the candidate pool as hybrids that carry the
    donor's coordinates into a good base, or the button is a placebo.
11. **The session file is a resume, not a lookalike.** The exported HTML
    round-trips its JSON island verbatim (prompts containing "</script>"
    included), keeps a human-readable half, and a replayed solver holds the
    SAME posterior as the search it came from - hyperparameters included -
    while RESUME refuses state from different rows. GOOD/N-GOOD deliberately
    differ: edited rows remain live generator controls, so saved coordinates
    are projected onto their current values.
12. **Reuse is an economy, not a rut.** With a host probe wired, already-
    rendered sides are shown measurably more often (they are ~20x cheaper
    than generating) while the SAME convergence bar still holds - the swap
    only ever trades away sampled-utility differences under one
    judgement's noise. At the reuse streak cap the duel is bit-identical
    to the cost-blind engine's (the preference switches off rather than
    becoming an all-to-all tour of the cache), and a probe that throws
    reads as "nothing cached", never as a dead search.
13. **The trend model proposes; the posterior disposes.** It nominates
    nothing before SURROGATE_MIN_DUELS. Fed lessons where the best model
    and the best factor were each proven only in OTHER company, its
    nominees are enriched for both winners AND contain the never-shown
    winning combination - the cross-combination guess the GP's product
    kernel cannot make. A conditional weight's trend is learned per pick:
    one LoRA preferring low weights must not drag another's trend down.
14. **Set GOOD and Reset have an origin.** One sampled good point is applied
    on top of the exact settings captured at search launch; untouched fields
    survive, Reset's snapshot is copied rather than referenced, and that
    snapshot rides in the retained solver payload through JSON.
15. **RESUME means these rows, and N-GOOD means one row.** A retained solver
    enables RESUME only when its dimension signature matches the visible
    search rows. N-GOOD composes every returned sample into one horizontal
    strip, never a near-square grid.

Needs numpy (the search engine). Everything else is stubbed, because the parts
under test are arithmetic and string building - the host is not involved in
either. Exit code 0 = pass.
"""

import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)

FAILURES = []
SKIPS = []


def fail(message):
    FAILURES.append(message)


def check(condition, message):
    if not condition:
        fail(message)
    return condition


# ---------------------------------------------------------------------------
# Enough of the host to import the script
#
# The script is a gradio panel wrapped around pure functions, and it is the
# pure functions that are worth testing. Stubbing is legitimate HERE and would
# not be for the UI: nothing below calls a stub, they exist only so that the
# module-level imports and the two callback registrations succeed.
# ---------------------------------------------------------------------------

def install_host_stubs():
    def module(name, **attributes):
        mod = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    class _Component:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

        def __call__(self, *args, **kwargs):
            return self

        def change(self, *args, **kwargs):
            return None

        click = change

    gradio = module("gradio")
    for name in ("Dropdown", "Textbox", "Number", "Checkbox", "Radio", "Row",
                 "Group", "Image", "Button", "Markdown", "HTML", "State",
                 "Timer", "Slider", "update"):
        setattr(gradio, name, _Component)

    class _Script:
        is_img2img = False
        args_from = None
        args_to = None

    module("modules")
    module("modules.scripts", Script=_Script, AlwaysVisible=object())
    module("modules.errors", display=lambda *a, **k: None)
    module("modules.processing", Processed=object, process_images=None,
           fix_seed=lambda p: None)
    state = types.SimpleNamespace(interrupted=False, stopping_generation=False,
                                  job="", job_count=0)
    opts = types.SimpleNamespace(data={}, add_option=lambda *a, **k: None)
    module("modules.shared", state=state, opts=opts,
           OptionInfo=lambda *a, **k: None)
    module("modules.ui_components", ToolButton=_Component)
    module("modules.script_callbacks",
           on_ui_settings=lambda *a, **k: None,
           on_after_component=lambda *a, **k: None)

    import modules
    modules.scripts = sys.modules["modules.scripts"]
    modules.errors = sys.modules["modules.errors"]
    modules.processing = sys.modules["modules.processing"]
    modules.shared = sys.modules["modules.shared"]
    modules.script_callbacks = sys.modules["modules.script_callbacks"]
    modules.ui_components = sys.modules["modules.ui_components"]


def load_script():
    import importlib.util
    path = os.path.join(EXTENSION, "scripts", "CNPro_AB.py")
    spec = importlib.util.spec_from_file_location("cnpro_ab_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1-3. The search
# ---------------------------------------------------------------------------

#: A synthetic taste over a space with all three kinds of dimension: opaque
#: categories (a model), ordered categories (profile factors), and a continuous
#: LoRA weight. Written as a smooth function of each so that a working search
#: can generalize - which is the property being tested. The optimum is
#: model 1, factor 1.0, prompt 1, weight 0.65.
MODELS = ["canny", "depth", "openpose", "tile"]
FACTORS = ["0.5", "0.75", "1", "1.25", "1.5"]
PROMPTS = ["a house", "a house in fog", "a house at night"]


def taste(point):
    model, factor, prompt, weight = point
    return ({0: 0.2, 1: 1.0, 2: -0.3, 3: 0.0}[int(model)]
            - 2.0 * (float(FACTORS[int(factor)]) - 1.0) ** 2
            + {0: 0.0, 1: 0.5, 2: -0.2}[int(prompt)]
            - 3.0 * (float(weight) - 0.65) ** 2)


def build_space(search):
    return search.Space([
        search.Dimension.choice("model", MODELS),
        search.Dimension.choice("factor", FACTORS),
        search.Dimension.choice("prompt", PROMPTS),
        search.Dimension.range("lora", 0.0, 1.0, 0.05),
    ])


def run_search(search, numpy, seed, duels, reverse=False, noise=0.3):
    """A whole session against the synthetic taste.

    `reverse` grades the same duels with the scale inverted, which is what a
    sign error in the model would look like from the inside.
    """
    space = build_space(search)
    rng = numpy.random.default_rng(1000 + seed)
    engine = search.PreferenceSearch(space, seed=seed)
    for _ in range(duels):
        a, b = engine.next_duel()
        if space.key(a) == space.key(b):
            fail("next_duel returned the same configuration twice - a duel "
                 "between a point and itself measures nothing, and the grade "
                 "the user gives it is noise the model then has to unlearn")
        if space.separation(a, b) < search.MIN_DUEL_SEPARATION - 1e-9:
            fail("next_duel asked about two configurations %.3f apart, under "
                 "the %.2f it takes to see a difference. The user is then "
                 "grading two images that look the same, and whatever they "
                 "answer is noise"
                 % (space.separation(a, b), search.MIN_DUEL_SEPARATION))
        gap = taste(b) - taste(a) + rng.normal(0, noise)
        if reverse:
            gap = -gap
        engine.observe(a, b, round(min(max(5 + 5 * gap / 1.2, 0), 10)))
    return engine


def test_search(search, numpy):
    space = build_space(search)

    # -- the encoding ------------------------------------------------------
    numeric = list(space.numeric_mask)
    check(numeric == [False, True, False, True],
          "the kernel is treating the wrong dimensions as ordered: %r. A list "
          "of numbers ('0.5, 0.75, 1') carries an ORDER and a list of model "
          "names does not; getting this backwards either invents a metric "
          "between two unrelated models or throws away the one real piece of "
          "structure the user handed over" % (numeric,))

    baseline = space.baseline()
    check(baseline == (0, 0, 0, 0.0),
          "the baseline is %r, not the first choice of every list and the "
          "bottom of every range. It is the opening duel's anchor - 'is any of "
          "this better than what I already had' - and an arbitrary point does "
          "not ask that question" % (baseline,))

    weight = space.dimensions[3]
    check(abs(weight.snap(0.6237) - 0.60) < 1e-9,
          "a range value is not being snapped to its step grid (0.6237 -> %r). "
          "Without snapping the same configuration is never visited twice, so "
          "evidence scatters over values that produce identical images"
          % (weight.snap(0.6237),))

    # -- a tie is an observation ------------------------------------------
    tie = search.PreferenceSearch(build_space(search), seed=0)
    left, right = (0, 0, 0, 0.0), (1, 4, 2, 1.0)
    tie.observe(left, right, 5)
    mean_tie, _ = tie._posterior([left, right], with_covariance=False)
    decided = search.PreferenceSearch(build_space(search), seed=0)
    decided.observe(left, right, 0)
    mean_decided, _ = decided._posterior([left, right], with_covariance=False)

    check(abs(float(mean_tie[0] - mean_tie[1])) < 1e-6,
          "a grade of 5 left the two configurations %g apart. 5 is the "
          "statement that they are worth the same - 'both great' and 'both "
          "awful' say the same thing about the DIFFERENCE - so it has to pull "
          "their utilities together, not be discarded"
          % abs(float(mean_tie[0] - mean_tie[1])))
    check(float(mean_decided[0]) > float(mean_decided[1]),
          "after a grade of 0 (A much better) the model ranks B at least as "
          "high as A. The scale is reversed somewhere between the button and "
          "the likelihood")

    # -- convergence -------------------------------------------------------
    best_utilities, worst_utilities = [], []
    for seed in range(4):
        best_utilities.append(taste(run_search(search, numpy, seed, 30).best()))
        worst_utilities.append(
            taste(run_search(search, numpy, seed, 30, reverse=True).best()))

    optimum = max(taste((m, f, p, round(0.05 * w, 3)))
                  for m in range(len(MODELS)) for f in range(len(FACTORS))
                  for p in range(len(PROMPTS)) for w in range(21))
    worst = min(taste((m, f, p, round(0.05 * w, 3)))
                for m in range(len(MODELS)) for f in range(len(FACTORS))
                for p in range(len(PROMPTS)) for w in range(21))
    rng = numpy.random.default_rng(7)
    typical = float(numpy.mean([taste(space.sample(rng)) for _ in range(4000)]))

    found = float(numpy.mean(best_utilities))
    check(found >= typical + 0.6 * (optimum - typical),
          "30 graded duels found configurations worth %.2f on average. An "
          "average configuration in this space is worth %.2f and the best is "
          "%.2f, so the search is not buying much over picking at random - "
          "which is the only thing that justifies its existence"
          % (found, typical, optimum))

    reversed_found = float(numpy.mean(worst_utilities))
    check(reversed_found <= typical - 0.4 * (typical - worst),
          "grading every duel BACKWARDS still produced configurations worth "
          "%.2f, against %.2f for an average one. The search should then "
          "converge on the WORST corner of the space (%.2f) - that it does "
          "not means the grades are not actually steering it, and the "
          "convergence check above would pass on a build that ignores them"
          % (reversed_found, typical, worst))

    # -- what the panel reports --------------------------------------------
    engine = run_search(search, numpy, 0, 30)
    status = engine.status()
    check(0.0 <= status["confidence"] <= 1.0,
          "confidence is %r, which is reported to the user as a percentage"
          % (status["confidence"],))
    check(status["duels"] == 30,
          "the engine counted %r duels after 30 observations" % (status["duels"],))

    # -- the small spaces where near-duplicate duels actually happen -------
    #
    # A five-dimensional space almost never produces two configurations that
    # differ by one step, so the guard above would be untested by the runs
    # here. The spaces where it matters are the small ones - "one LoRA row" is
    # a whole search - and those are exactly what a first-time user builds.
    for name, dimensions in (
            ("one LoRA row",
             [search.Dimension.choice("lora", ["a", "b"]),
              search.Dimension.range("weight", 0.0, 1.0, 0.05)]),
            ("one weight", [search.Dimension.range("weight", 0.0, 1.0, 0.05)]),
    ):
        small = search.Space(dimensions)
        engine = search.PreferenceSearch(small, seed=3)
        rng = numpy.random.default_rng(3)
        closest = 99.0
        for _ in range(20):
            a, b = engine.next_duel()
            closest = min(closest, small.separation(a, b))
            engine.observe(a, b, 10 if float(b[-1]) > float(a[-1]) else 0)
        check(closest >= search.MIN_DUEL_SEPARATION - 1e-9,
              "in a space of %s, a duel came back %.3f apart - two images from "
              "weights 0.60 and 0.65 are the same image, and asking about them "
              "spends two generations and a judgement on nothing"
              % (name, closest))

    # -- an explicit row is anchored; a row-less legacy call is not ----------
    #
    # "I dislike both" has to have an ADDRESS: both configurations below the
    # prior mean. Without the anchor the posterior is translation-invariant
    # and a pair graded on the dislike row would look identical to a pair
    # the user loved - the difference is all a comparison can see.
    left, right = (0, 0, 0, 0.0), (0, 4, 2, 1.0)
    bad_pair = search.PreferenceSearch(build_space(search), seed=0)
    bad_pair.observe(left, right, 5, disliked=True)
    mean_bad, _ = bad_pair._posterior([left, right], with_covariance=False)
    check(float(mean_bad[0]) < -0.05 and float(mean_bad[1]) < -0.05,
          "a disliked tie left the pair at (%.3f, %.3f). 'I dislike both' "
          "has to push BOTH sides below par, or a bad subspace can only "
          "ever be ranked from the inside, never avoided"
          % (float(mean_bad[0]), float(mean_bad[1])))

    plain = search.PreferenceSearch(build_space(search), seed=0)
    plain.observe(left, right, 5)
    mean_plain, _ = plain._posterior([left, right], with_covariance=False)
    check(abs(float(mean_plain[0])) < 0.05 and abs(float(mean_plain[1])) < 0.05,
          "a row-less legacy tie moved the pair off par to (%.3f, %.3f) - "
          "similar=None means the caller supplied no absolute row verdict, "
          "so only the comparison may be recorded"
          % (float(mean_plain[0]), float(mean_plain[1])))

    lessbad = search.PreferenceSearch(build_space(search), seed=0)
    lessbad.observe(left, right, 0, disliked=True)   # A less bad - and both bad
    mean_lb, _ = lessbad._posterior([left, right], with_covariance=False)
    check(float(mean_lb[0]) > float(mean_lb[1]),
          "on a disliked duel graded 0, A is not ranked above B (%.3f vs "
          "%.3f). WHICH side is less bad still steers the model inside the "
          "region - the dislike must not erase the comparison"
          % (float(mean_lb[0]), float(mean_lb[1])))
    check(float(mean_lb[1]) < -0.05,
          "on a disliked duel the loser sits at %.3f, not below par - the "
          "dislike's anchor is not landing" % float(mean_lb[1]))

    # -- a shocking upset gets its follow-up -------------------------------
    #
    # A 10 for a configuration the model wrote off must arm one immediate
    # verification duel for the winner. Without it, a lone spike in hostile
    # territory is averaged away by the smoothing prior before anything acts
    # on it - the hidden gem is found and then LOST, which is worse than
    # never finding it, because two generations were spent looking at it.
    shock = search.PreferenceSearch(build_space(search), seed=0)
    strong = (1, 2, 1, 0.6)
    for loser in ((0, 0, 0, 0.0), (2, 4, 2, 1.0), (3, 0, 2, 0.3),
                  (0, 4, 0, 0.9)):
        shock.observe(strong, loser, 0)      # the champion earns its rank
    shock.best()                             # model current, expectations live
    dark = (3, 4, 2, 1.0)
    shock.observe(strong, dark, 10)          # ...and is crushed by a nobody
    check(shock._surprise is not None
          and shock.space.key(shock._surprise) == shock.space.key(dark),
          "a decisive win by a configuration the model wrote off did not arm "
          "the follow-up (surprise=%r). The upset is then one comparison "
          "against a mountain of smoothing, and the gem it points at is "
          "averaged away" % (shock._surprise,))
    follow_a, follow_b = shock.next_duel()
    followed = {shock.space.key(follow_a), shock.space.key(follow_b)}
    check(shock.space.key(dark) in followed,
          "the duel after a shocking upset does not involve the upset winner "
          "- the follow-up went to %r instead, so 'look closer, now' is not "
          "actually happening" % (followed,))

    # -- "interesting" is not evidence -------------------------------------
    #
    # The mark means "overall bad, but this characteristic is tempting", so
    # it must not move the posterior by a hair: an engine fed the same duels
    # WITH marks has to believe exactly what one fed them without marks
    # believes. Valuing the mark as utility would smuggle "actually it is
    # a bit good" into a statement the user explicitly did not make.
    left, right = (0, 0, 0, 0.0), (1, 4, 2, 1.0)
    plain_eng = search.PreferenceSearch(build_space(search), seed=0)
    marked_eng = search.PreferenceSearch(build_space(search), seed=0)
    plain_eng.observe(left, right, 2, disliked=True)
    marked_eng.observe(left, right, 2, disliked=True,
                       interesting_b=True)
    mean_plain2, _ = plain_eng._posterior([left, right], with_covariance=False)
    mean_marked, _ = marked_eng._posterior([left, right], with_covariance=False)
    check(abs(float(mean_plain2[0]) - float(mean_marked[0])) < 1e-12
          and abs(float(mean_plain2[1]) - float(mean_marked[1])) < 1e-12,
          "an interesting mark moved the posterior (%.4f/%.4f vs %.4f/%.4f). "
          "The mark is an acquisition hint, not a grade - the sample is "
          "still exactly as bad as the user said it was"
          % (float(mean_plain2[0]), float(mean_plain2[1]),
             float(mean_marked[0]), float(mean_marked[1])))
    check(len(marked_eng._interesting) == 1 and not plain_eng._interesting,
          "the interesting mark was not recorded as a donor (%d donors)"
          % len(marked_eng._interesting))

    # -- ...but its characteristic reaches the candidate pool --------------
    #
    # The donor's coordinates have to appear in HYBRIDS: pool candidates
    # that carry a piece of the marked configuration inside a good base.
    # Otherwise the button is a placebo.
    hybrid_eng = search.PreferenceSearch(build_space(search), seed=1)
    good_pt, other = (0, 2, 0, 0.5), (0, 1, 0, 0.3)
    donor = (2, 4, 2, 1.0)          # model 2: nothing else in the data has it
    hybrid_eng.observe(good_pt, other, 3)
    hybrid_eng.observe(good_pt, donor, 1, disliked=True, interesting_b=True)
    pool = hybrid_eng._pool()
    donor_key = hybrid_eng.space.key(donor)
    hybrids = [pt for pt in pool
               if int(pt[0]) == 2 and hybrid_eng.space.key(pt) != donor_key]
    check(len(hybrids) > 0,
          "no pool candidate carries the marked donor's model into another "
          "configuration - the interesting button changes nothing the "
          "search will ever look at")

    # -- a dislike changes the subject -------------------------------------
    #
    # "This is not going well" answered with more polishing of the same
    # neighbourhood is the search not listening. After a disliked duel the
    # local (polishing) share of the pool collapses; it recovers on the
    # first normal grade.
    redirected = search.PreferenceSearch(build_space(search), seed=7)
    for _ in range(redirected.seed_duels + 3):
        a, b = redirected.next_duel()
        redirected.observe(a, b, 5)
    check(redirected._dislike_streak == 0 and not redirected._redirect,
          "normal grades left a dislike streak of %d and redirect=%r - the "
          "un-annealing would then be permanent instead of a response"
          % (redirected._dislike_streak, redirected._redirect))
    a, b = redirected.next_duel()
    redirected.observe(a, b, 5, disliked=True)
    check(redirected._dislike_streak == 1 and redirected._redirect,
          "a disliked duel did not arm the redirect (streak %d, redirect "
          "%r) - the next duel would keep polishing the direction the user "
          "just declared a dead end"
          % (redirected._dislike_streak, redirected._redirect))
    a, b = redirected.next_duel()
    redirected.observe(a, b, 5)
    check(redirected._dislike_streak == 0,
          "a normal grade did not clear the dislike streak (%d) - recovery "
          "from 'this is not going well' has to be automatic"
          % redirected._dislike_streak)

    # -- taste is allowed to be bimodal ------------------------------------
    #
    # Two unrelated looks, nearly equally good, on OPAQUE categories (the
    # model), so no smoothness can carry evidence between them. The search
    # has to end holding both - a single-answer search that silently drops
    # the second-best basin is the failure the frontier exists to prevent.
    def two_peak_taste(point):
        model, factor, prompt, weight = point
        f = float(FACTORS[int(factor)])
        w = float(weight)
        peaks = {1: 1.0 - 2.0 * (f - 1.0) ** 2 - 3.0 * (w - 0.65) ** 2,
                 3: 0.9 - 2.0 * (f - 0.75) ** 2 - 3.0 * (w - 0.25) ** 2}
        return peaks.get(int(model), -0.8) \
            + {0: 0.0, 1: 0.2, 2: 0.0}[int(prompt)]

    # WHAT THE PAIR LOOKS LIKE, which is not what it is WORTH. The panel's
    # three grade rows partition the on-track case, so a real session hands
    # back a similarity verdict with every single grade - and any test that
    # grades without one is running the engine in a state the UI cannot
    # produce. Here the look is carried by the model and the weight; factor
    # and prompt change the score without changing the picture, which is the
    # case the learned metric exists to discover.
    def looks_alike(a, b):
        return int(a[0]) == int(b[0]) and abs(float(a[3]) - float(b[3])) < 0.25

    both_found = 0
    engines = []
    for seed in range(3):
        space2 = build_space(search)
        engine = search.PreferenceSearch(space2, seed=seed)
        rng = numpy.random.default_rng(2000 + seed)
        for _ in range(40):
            a, b = engine.next_duel()
            gap = two_peak_taste(b) - two_peak_taste(a) + rng.normal(0, 0.3)
            grade = round(min(max(5 + 5 * gap / 1.2, 0), 10))
            # The dislike row, the way a person actually uses it: clicked
            # when BOTH sides clearly miss what they want.
            disliked = max(two_peak_taste(a), two_peak_taste(b)) < -0.2
            engine.observe(a, b, grade, disliked=disliked,
                           similar=looks_alike(a, b))
        engines.append(engine)
        keepers = engine.frontier()
        for i, one in enumerate(keepers):
            for other in keepers[i + 1:]:
                if space2.separation(one, other) < search.FRONTIER_SEPARATION:
                    fail("two frontier members are %.2f apart, under the %.2f "
                         "that makes them different answers - the frontier is "
                         "then the same answer twice, dressed as diversity"
                         % (space2.separation(one, other),
                            search.FRONTIER_SEPARATION))
        models = {int(keeper[0]) for keeper in keepers}
        strong = [keeper for keeper in keepers
                  if two_peak_taste(keeper) > 0.3]
        if {1, 3} <= models and len(strong) >= 2:
            both_found += 1
    check(both_found >= 2,
          "against a two-peak taste the frontier held both peaks on only "
          "%d of 3 seeds. Thompson sampling alone polishes whichever basin "
          "is ahead and starves the other; the frontier, the attractor-split "
          "pool and the cross-basin duels exist precisely so that a user who "
          "likes two unrelated looks ends the search holding both"
          % both_found)

    # -- the two peaks are two ISLANDS -------------------------------------
    #
    # The frontier check above says the search HOLDS both answers. This says
    # it can NAME them as separate places: `islands` is the machinery N-GOOD
    # samples from, and a taste built out of two disjoint peaks with a wall
    # of -0.8 between them is the case it has to get right. Two of three
    # seeds, like the frontier claim beside it - one search is one sample of
    # a stochastic process.
    named = 0
    for engine in engines:
        found = engine.islands()
        models = {int(peak[0]) for peak, _size in found}
        if len(found) >= 2 and {1, 3} <= models:
            named += 1
    check(named >= 2,
          "islands() separated the two peaks on only %d of 3 seeds. The good "
          "set here is two disjoint blobs around models 1 and 3 with "
          "everything between them at -0.8, so a component count that cannot "
          "see two is not measuring isolation - and N-GOOD, which samples one "
          "entry per island before it takes a second from any, would offer "
          "one answer where the user has two.\n"
          "      THE SIMILARITY ROW IS WHAT MAKES THIS WORK, so suspect the "
          "metric first: under the PRIOR weights every row counts 1.0 and "
          "the island hop is %.2f, so a change of factor or prompt splits a "
          "blob as surely as a change of model does and 'isolated' means "
          "nothing. It is the labels that shrink the inert rows until only "
          "the ones that change the picture can separate two islands"
          % (named, search.ISLAND_HOP))

    # -- portfolio: good samples on demand, bad samples on demand ----------
    engine = engines[0]
    for wanted in (True, False):
        pick = engine.suggest(good=wanted)
        check(len(pick) == len(build_space(search).dimensions),
              "suggest(good=%r) returned a point of the wrong arity: %r"
              % (wanted, pick))
    good_draw = engine.portfolio(6, good=True)
    bad_draw = engine.portfolio(6, good=False)
    check(len(good_draw) >= 3 and len(bad_draw) >= 3,
          "portfolio() returned %d good / %d bad configurations from a "
          "40-duel session - the generator half of the search has nothing "
          "to offer" % (len(good_draw), len(bad_draw)))
    if good_draw and bad_draw:
        good_worth = float(numpy.mean([two_peak_taste(g) for g in good_draw]))
        bad_worth = float(numpy.mean([two_peak_taste(g) for g in bad_draw]))
        check(good_worth > bad_worth + 0.8,
              "portfolio(good=True) draws are worth %.2f against %.2f for "
              "portfolio(good=False) - the two ends of the posterior are not "
              "actually mapped to the two ends of the taste, so 'generate "
              "good/bad samples on demand' would generate noise"
              % (good_worth, bad_worth))

    # -- no duel is re-asked past the cap ----------------------------------
    hammered = run_search(search, numpy, 1, 40)
    over = max(hammered._asked.values())
    check(over <= search.MAX_PAIR_REPEATS,
          "one pair was asked %d times (cap %d). A third asking of the same "
          "question measures the user's consistency, not their taste - and "
          "it is exactly the 'it keeps showing me the same two images' "
          "failure a session-long search cannot afford"
          % (over, search.MAX_PAIR_REPEATS))

    # -- the seed duels are a design, not coin flips -----------------------
    covered = search.PreferenceSearch(build_space(search), seed=5)
    seen_models = set()
    for _ in range(covered.seed_duels):
        a, b = covered.next_duel()
        seen_models.update((int(a[0]), int(b[0])))
        covered.observe(a, b, 5)
    check(len(seen_models) >= 3,
          "the %d seed duels showed only models %r of 4. The opening duels "
          "are the design of experiments - six points that never leave one "
          "corner teach the model nothing about most of the space, and the "
          "user then watches it rediscover the rest one duel at a time"
          % (covered.seed_duels, sorted(seen_models)))

    # -- a conditional weight does not transfer across its pick ------------
    #
    # A LoRA slot's weight means nothing across two different LoRAs. The
    # kernel has to treat (A, 0.2) vs (B, 0.2) and (A, 0.2) vs (B, 0.9) as
    # EQUALLY similar - the pick difference is the whole difference - or
    # duels graded on lora B's weight quietly shape what the model believes
    # about lora A's.
    pick_dim = search.Dimension.choice("pick", ["a", "b"])
    weight_dim = search.Dimension.range("weight", 0.0, 1.0, 0.05)
    weight_dim.parent = pick_dim
    cond = search.PreferenceSearch(search.Space([pick_dim, weight_dim]), seed=0)
    u = cond.space.encode([(0, 0.2)])
    v = cond.space.encode([(1, 0.2), (1, 0.9), (0, 0.9)])
    kernel = cond._kernel(u, v, lengthscale=0.3, theta=1.5)
    check(abs(float(kernel[0, 0]) - float(kernel[0, 1])) < 1e-12,
          "across two picks, matching weights (%.4f) and clashing weights "
          "(%.4f) give different kernel values - weight evidence is leaking "
          "between LoRAs whose weights are not transferable"
          % (float(kernel[0, 0]), float(kernel[0, 1])))
    check(float(kernel[0, 2]) < 1.0 - 1e-6,
          "within ONE pick, two different weights score %.4f - the weight "
          "kernel stopped working where it is supposed to work"
          % float(kernel[0, 2]))

    # -- the observation cap still leaves a working model ------------------
    capped = search.PreferenceSearch(build_space(search), seed=0)
    for index in range(search.MAX_OBSERVATIONS + 20):
        a, b = space.sample(capped.rng), space.sample(capped.rng)
        if space.key(a) == space.key(b):
            continue
        capped.observe(a, b, 10 if taste(b) > taste(a) else 0)
    check(len(capped.observations) <= search.MAX_OBSERVATIONS,
          "the observation cap is not holding: %d kept, cap %d. The fit is "
          "cubic in the number of distinct configurations, so an unbounded "
          "session eventually stops between duels rather than generating"
          % (len(capped.observations), search.MAX_OBSERVATIONS))
    check(all(index < len(capped.points)
              for a, b, _p in capped.observations for index in (a, b)),
          "dropping old observations left an observation pointing at a "
          "configuration that was pruned away - the next fit will index out "
          "of range")


def test_similarity_metric(search, numpy):
    """The third row: what a comparison cannot say.

    Separation decides whether a duel is worth asking, how different a
    keeper has to be, and what a collage may hold together - and it used to
    be a hand-written constant. The similarity row makes it LEARNED, so what
    is tested is the whole loop, in the order it can break:

    * NOTHING CHANGES WITHOUT LABELS. Every weight sits at the prior, so a
      session that never uses the row is bit-identical to one from before
      the row existed. This is the property that makes the feature safe, and
      it is the first thing a bad fit would break.
    * The fit SEPARATES an inert dimension from a vivid one, and leaves
      dimensions nobody labelled alone.
    * A "similar" mark also ties the pair on the UTILITY side - two images
      that look the same are worth the same, which is a stronger claim than
      the soft grade beside it.
    * THE CENSORING DEFENCE HOLDS. Once a weight drops under the gate the
      search stops asking about that dimension and would never hear about it
      again - the estimate would confirm itself forever. The probe asks
      exactly the duel the learned metric now refuses, its answers count in
      full, and it goes quiet again once the weight recovers.
    """
    space = build_space(search)          # model, factor, prompt, lora
    fresh = search.PreferenceSearch(space, seed=1)
    check(list(fresh.space.weights) == [1.0] * len(space.dimensions),
          "a fresh space did not start at the hand-written metric (%r) - "
          "every distance threshold in the engine is calibrated against "
          "'one differing category counts 1', so anything else silently "
          "recalibrates all of them" % (list(fresh.space.weights),))

    rng = numpy.random.default_rng(0)
    engine = search.PreferenceSearch(build_space(search), seed=3)
    for _ in range(6):
        a = engine.space.sample(rng)
        engine.observe(a, engine.space.perturb(a, rng), 6)
    check(list(engine.space.weights) == [1.0] * len(space.dimensions),
          "six graded duels with no similarity verdict moved the separation "
          "metric to %r. A session that never touches the new row has to "
          "behave exactly as it did before the row existed"
          % (list(engine.space.weights),))

    # Teach it that the prompt row is visually inert and the model row is not.
    def label(engine, index, similar, grade=5, count=20):
        for _ in range(count):
            a = engine.space.sample(rng)
            b = list(a)
            b[index] = (int(a[index]) + 1) % len(space.dimensions[index].labels)
            engine.observe(a, tuple(b), grade, similar=similar)

    label(engine, 2, True)               # prompt: same image every time
    label(engine, 0, False)              # model: always tells them apart
    weights = engine.space.weights
    check(weights[2] < 0.5 * weights[0],
          "after twenty 'these look the same' labels on the prompt row and "
          "twenty 'distinct' on the model row, the metric weighs them %.3f "
          "and %.3f. The whole point of the row is that the search stops "
          "spending two generations a question on a knob the user has said "
          "does nothing" % (weights[2], weights[0]))
    check(abs(weights[1] - 1.0) < 1e-6 and abs(weights[3] - 1.0) < 1e-6,
          "dimensions nobody labelled drifted to %r - the fit is shrunk to "
          "the prior precisely so that evidence about one row cannot "
          "silently restate the others" % ([weights[1], weights[3]],))

    # The utility dividend: a "similar" mark records a tie of its own.
    tied = search.PreferenceSearch(build_space(search), seed=7)
    a = tied.space.baseline()
    b = tied.space.perturb(a, numpy.random.default_rng(2))
    before = len(tied.observations)
    tied.observe(a, b, 7, similar=True)
    check(len(tied.observations) == before + 4,
          "a 'similar and on-track' verdict added %d observation(s), not "
          "four: the comparison, its visual tie, and two above-par anchors. "
          "Dropping the tie wastes the more reliable similarity statement; "
          "dropping the anchors leaves N-GOOD unable to distinguish good "
          "from merely less bad"
          % (len(tied.observations) - before))

    # The censoring defence, end to end.
    blind = search.PreferenceSearch(build_space(search), seed=5)
    check(blind._probe_duel() is None,
          "the censoring probe found a pair to ask about while the metric "
          "was still at its prior. It exists to ask what the LEARNED metric "
          "refuses and the prior allows - with the two identical there is no "
          "such pair, and firing anyway spends duels on nothing")
    label(blind, 2, True, count=40)
    gate = search.MIN_DUEL_SEPARATION
    shrunk = blind.space.separation((0, 0, 0, 0.5), (0, 0, 1, 0.5))
    check(shrunk < gate,
          "forty 'these look the same' labels left the prompt row at %.3f, "
          "still above the %.2f gate - the metric is not actually reaching "
          "the decision it is supposed to inform" % (shrunk, gate))
    probed = blind._probe_duel()
    check(probed is not None
          and blind.space.separation(*probed) < gate <= blind.space.raw_separation(*probed),
          "with the prompt row shrunk under the gate the probe did not "
          "produce the one duel that can un-shrink it (%r). Every duel path "
          "filters on separation, so without this the search never asks "
          "again, never hears otherwise, and the estimate confirms itself "
          "for the rest of the session" % (probed,))
    for _ in range(6):
        pair = blind._probe_duel()
        if pair is None:
            break
        blind.observe(pair[0], pair[1], 8, similar=False)
    check(blind.space.separation((0, 0, 0, 0.5), (0, 0, 1, 0.5)) >= gate,
          "six probe answers of 'actually these are distinct' did not lift "
          "the prompt row back over the gate (%.3f). Probes arrive once in "
          "PROBE_MEAN duels, so a recovery needing more than a handful of "
          "them is a recovery path hundreds of duels long - which is none"
          % blind.space.separation((0, 0, 0, 0.5), (0, 0, 1, 0.5)))
    check(blind._probe_duel() is None,
          "the probe kept firing after the weight it was defending had "
          "recovered - it asks deliberately hard-to-answer duels, so it has "
          "to go quiet the moment the two metrics agree again")


def test_capacity(search, numpy):
    """HOW MANY good answers are there, and can they be extracted?

    `capacity` is the number the panel's N box shows and the number N-GOOD
    spends generations on, so what is tested is that it MEANS something:

    * it never force-admits a keeper. A frontier always has a relative
      champion, including after an all-bad session; N-GOOD requires separate
      posterior support that an entry is above par;
    * it TRACKS THE TASTE. A taste with one sharp optimum has a handful of
      good answers and a taste where only the model matters has a whole
      subspace of them. This is the test with teeth: a "capacity" that
      reported the same number for both would be reporting the search's
      length, or the space's size, or a constant - all of which are easier
      to compute and none of which is the question;
    * it does not run ahead of the evidence. Before the model can tell a
      champion from a typical configuration there is no meaningful count,
      and reporting the space's size at duel one would send the user off to
      render a thousand images of nothing.

    And `population(n)` is the extractor that has to agree with it: every
    entry a different ANSWER, not merely a visible notch away (the
    POPULATION_SEPARATION bar, which is the keeper's - a collage of
    near-duplicates is the one failure that wastes a GPU-hour and looks
    like success), never more than asked, and DIFFERENT between presses,
    which is the whole reason the button can be pressed twice.

    THE BAR ALONE DOES NOT CATCH THE BUG THIS TEST NOW CARRIES, which is
    why the pre-fix selection is run alongside and has to lose. N-GOOD
    returned N slight variations of one image while satisfying a pairwise
    floor perfectly: walking the posterior ranking downward and keeping
    whatever clears the floor fills up inside the champion's basin, where
    every neighbour is one notch from the last. Pairwise-floor tests pass on
    that. What separates the two selections is COVERAGE - how far the set
    reaches across the good region - so that is what is measured.
    """
    space = build_space(search)

    #: Only the model matters - so every combination of factor, prompt and
    #: weight on the right model is equally good, and there are hundreds.
    def broad(point):
        return {0: -0.6, 1: 1.0, 2: -0.6, 3: -0.6}[int(point[0])]

    def session(taste, duels, seed=5):
        rng = numpy.random.default_rng(1000 + seed)
        engine = search.PreferenceSearch(build_space(search), seed=seed)
        for _ in range(duels):
            a, b = engine.next_duel()
            gap = taste(b) - taste(a) + rng.normal(0, 0.3)
            disliked = taste(a) < 0 and taste(b) < 0
            # The live panel always answers exactly one of its three rows.
            # Top/middle both mean ON TRACK and differ only in whether the
            # render looks alike; bottom means both bad and intentionally has
            # no similarity label. A capacity bench that passes no row for a
            # usable pair withholds the absolute-quality signal N-GOOD is
            # specifically supposed to consume.
            similar = (None if disliked else
                       (int(a[0]) == int(b[0])
                        and int(a[1]) == int(b[1])
                        and int(a[2]) == int(b[2])
                        and abs(float(a[3]) - float(b[3])) < 0.25))
            engine.observe(a, b, round(min(max(5 + 5 * gap / 1.2, 0), 10)),
                           disliked=disliked, similar=similar)
        engine.frontier()
        return engine

    fresh = search.PreferenceSearch(space, seed=1)
    check(fresh.capacity() == 0,
          "a solver with no observations claimed %d good configurations - it "
          "has not been told anything, so the honest answer is none and the "
          "N box must not offer to spend generations on it"
          % fresh.capacity())

    # Over several seeds, because one search is one sample of a stochastic
    # process and the claim is about the ESTIMATOR, not about a run.
    narrows, broads = [], []
    for seed in (5, 6, 7):
        one, other = session(taste, 50, seed), session(broad, 50, seed)
        for engine, counts, name in ((one, narrows, "one sharp optimum"),
                                     (other, broads, "a good subspace")):
            count = engine.capacity()
            counts.append(count)
            # THE BOX IS WHAT THE BUTTON DELIVERS. Ask for more than the
            # box says and the collage should come back at the box's number,
            # because the two are the same selection run to different
            # limits. Slack of one for the sampling: the two calls draw
            # their own candidate pools.
            #
            # NOT floored at the keeper count, which is what this used to
            # assert. Keepers only clear FRONTIER_SEPARATION - a 39% chance
            # of looking different - so a capacity floored at their number
            # is the box promising samples the collage cannot produce, the
            # exact mismatch that made "N-GOOD returns N of the same image"
            # look like a selection bug rather than a units bug.
            delivered = len(engine.population(count + 5))
            check(delivered >= count,
                  "the N box says %d under a taste with %s and N-GOOD asked "
                  "for %d delivered only %d. The box may UNDER-promise - the "
                  "button sizes its pool by what was asked for and can find "
                  "more - but a box the button cannot reach is a target "
                  "missed on every press" % (count, name, count + 5, delivered))
    narrows.sort()
    broads.sort()
    check(broads[1] > narrows[1],
          "capacity's median over three seeds was %r for a taste with ONE "
          "optimum and %r for a taste where three whole dimensions are free "
          "- the second space genuinely holds hundreds of good answers and "
          "the first holds a handful, so a number that does not separate "
          "them is not measuring the good region at all" % (narrows, broads))

    early = session(broad, 1)
    space_size = build_space(search).size
    check(early.capacity() * 20 <= space_size,
          "after ONE graded duel capacity claimed %d of the space's %d "
          "configurations were good. Nothing is known yet - every unexplored "
          "point sits at the prior - so a number like that is the flat "
          "posterior being counted rather than a taste, and it invites the "
          "user to spend a GPU-hour on it (measured at 56-155 before the "
          "quality floor was made to clear the prior; see "
          "CAPACITY_CONFIDENCE)" % (early.capacity(), space_size))

    broadly = session(broad, 50)
    wanted = min(broadly.capacity(), 24)
    picks = broadly.population(wanted)
    check(0 < len(picks) <= wanted,
          "population(%d) returned %d configurations - it must never return "
          "more than asked (the count is a GPU budget) and never nothing "
          "when the capacity it agrees with says there are answers"
          % (wanted, len(picks)))
    def gaps(points):
        return [broadly.space.separation(one, other)
                for index, one in enumerate(points)
                for other in points[index + 1:]]

    report = broadly.population_report
    for gap in gaps(picks):
        check(gap >= search.POPULATION_SEPARATION - 1e-9,
              "two entries of a collage are %.3f apart - the fitted "
              "similarity model puts them at %.0f%% likely to look "
              "different, under the %.0f%% the button promises. A grid of "
              "the same image rendered twenty times costs a GPU-hour to "
              "discover" % (gap, 100 * search.distinct_probability(gap),
                            100 * search.COLLAGE_CONFIDENCE))
    check(report["confidence"] >= search.COLLAGE_CONFIDENCE - 1e-9,
          "population_report claims %.0f%%, under the %.0f%% the button "
          "promises. The report is what the panel prints beside the collage, "
          "so it has to be the measured truth about the sheet - and since "
          "there is no relaxation path, a sheet that cannot make the bar is "
          "short rather than disclaimed"
          % (100 * report["confidence"], 100 * search.COLLAGE_CONFIDENCE))

    # THE PRE-FIX SELECTION, run on the same solver, and it has to FAIL the
    # assertion above - otherwise the assertion proves nothing about this
    # engine and would pass just as happily on the code that produced the
    # bug. This is `population` as it was: rank by the posterior draw, keep
    # whatever clears MIN_DUEL_SEPARATION.
    broadly._fit()
    keepers = list(broadly._keepers or broadly.frontier())
    seen = {broadly.space.key(point) for point in keepers}
    pool = keepers + [point for point in broadly._candidates(
        broadly._diverse_centers(broadly.f, search.FRONTIER_SIZE),
        per_center=96, pool=512)
        if broadly.space.key(point) not in seen]
    means, cov = broadly._posterior(pool)
    good = broadly._quality(means, numpy.diag(cov),
                            broadly._champion_score(means))
    good[:len(keepers)] = True
    keep = numpy.flatnonzero(good)
    eligible = [pool[int(i)] for i in keep]
    draw = broadly._sample(means, cov, 1)[0][keep]
    ranked = broadly._diverse_top(eligible, draw, max(wanted, 8),
                                  search.MIN_DUEL_SEPARATION)
    if len(ranked) > 1:
        worst = min(gaps(ranked))
        check(worst < search.POPULATION_SEPARATION,
              "the pre-fix selection (ranking-first at MIN_DUEL_SEPARATION) "
              "produced %d entries whose worst pair is %.3f - already over "
              "the collage bar. Then the assertion above cannot distinguish "
              "the fix from the bug on this solver, and it is measuring "
              "nothing" % (len(ranked), worst))

    # DIFFERENT BETWEEN PRESSES - ASKED WITH SLACK, which is the only form
    # of the claim that means anything. Ask for everything the region holds
    # and there is nothing to choose: the selection finds the same maximal
    # set every time, and returning it twice is CORRECT - those are the
    # answers, and a press that shuffled in a near-duplicate to look busy
    # would be the original bug wearing a disguise. The claim is about
    # WHICH of several possible answers a press shows, so it is asked one
    # short of the maximum, where the choice exists.
    if wanted >= 2:
        fewer = wanted - 1
        # Three presses, at least two outcomes: the selection is stochastic
        # (a fresh pool and a fresh posterior draw per press), so demanding
        # that two CONSECUTIVE presses differ would be a coin flip dressed
        # as an assertion. What is being pinned is that the button is not
        # DETERMINISTIC, and three tries settle that without flakiness.
        outcomes = {frozenset(broadly.space.key(point)
                              for point in broadly.population(fewer))
                    for _ in range(3)}
        check(len(outcomes) >= 2,
              "three presses of N-GOOD for %d of the %d distinct answers the "
              "region holds produced one single outcome - with a choice "
              "available the button has to make different ones, or pressing "
              "again is a GPU-hour spent re-rendering the first collage"
              % (fewer, wanted))

    status = broadly.status()
    check(status.get("capacity") == broadly.capacity() or
          abs(status["capacity"] - broadly.capacity()) <= 2,
          "status() reported a capacity of %r against capacity()'s %r - the "
          "panel reads the first and the button spends the second"
          % (status.get("capacity"), broadly.capacity()))


def test_stochastic_tactics(search, numpy):
    """The three coin-fired tactics: they fire at their mean rates, and each
    one asks the question it claims to ask.

    * The BLUFF is a uniform pair - its validity (separation, repeat cap) is
      already asserted for every duel by the session loops above; what is
      pinned here is that the coins actually land at roughly 1/BLUFF_MEAN,
      1/VOID_MEAN and 1/RIVAL_MEAN per duel, because a rate silently wired
      to 0 is three dead tactics that no other test would ever notice.
    * A VOID probe has to land in genuinely unvisited territory - both sides
      at least VOID_SEPARATION from everything ever shown when the space has
      the room - and has to RETIRE (return None) in a space with no void
      left, rather than dressing the widest gap of a covered space up as
      discovery.
    * A RIVAL duel has to draw its two sides from two DIFFERENT attractors'
      basins - one side each - or it is not comparing attractors at all.
    """
    space = build_space(search)

    # -- the coins land at their advertised rates --------------------------
    engine = search.PreferenceSearch(build_space(search), seed=11)
    rng = numpy.random.default_rng(3011)
    fired = {"bluff": 0, "void": 0, "rival": 0}
    for name in list(fired):
        original = getattr(engine, "_%s_duel" % name)

        def wrapped(*args, _original=original, _name=name):
            fired[_name] += 1
            return _original(*args)

        setattr(engine, "_%s_duel" % name, wrapped)
    duels = 140
    for _ in range(duels):
        a, b = engine.next_duel()
        gap = taste(b) - taste(a) + rng.normal(0, 0.3)
        engine.observe(a, b, round(min(max(5 + 5 * gap / 1.2, 0), 10)))
    expected = {"bluff": duels / search.BLUFF_MEAN,
                "void": duels / search.VOID_MEAN,
                "rival": duels / search.RIVAL_MEAN}
    for name, count in fired.items():
        check(count >= max(2, expected[name] // 3),
              "the %s tactic fired %d times in %d duels (expected around "
              "%.0f). A tactic that never fires is indistinguishable from a "
              "working one in every other test - the search just quietly "
              "loses its guard against locking into local similarities"
              % (name, count, duels, expected[name]))
        check(count <= 3 * expected[name],
              "the %s tactic fired %d times in %d duels (expected around "
              "%.0f) - at that rate it is not seasoning the schedule, it IS "
              "the schedule, and convergence is what pays for it"
              % (name, count, duels, expected[name]))

    # -- a void probe lands in the void ------------------------------------
    voider = search.PreferenceSearch(build_space(search), seed=2)
    cluster = [(0, f, 0, w) for f in (0, 1) for w in (0.0, 0.1, 0.2)]
    for point in cluster:
        voider._shown.append(point)
        voider._shown_keys.add(voider.space.key(point))
    duel = voider._void_duel()
    if check(duel is not None,
             "with the whole session clustered in one corner (model 0, "
             "prompt 0, low weights), _void_duel found no void - most of "
             "the space is one, so the scan is not looking"):
        a, b = duel
        for side, point in (("A", a), ("B", b)):
            emptiness = min(space.separation(shown, point)
                            for shown in cluster)
            check(emptiness >= search.VOID_SEPARATION - 1e-9,
                  "void side %s sits %.2f from the shown cluster, under the "
                  "%.2f that makes it a void - the probe is sampling the "
                  "explored region and calling it discovery"
                  % (side, emptiness, search.VOID_SEPARATION))
        check(space.separation(a, b) >= search.MIN_DUEL_SEPARATION - 1e-9,
              "a void duel came back %.2f apart - under the separation "
              "gate, so the user is asked to grade an invisible difference"
              % space.separation(a, b))

    # -- ...and retires when there is none ---------------------------------
    tiny = search.Space([
        search.Dimension.choice("x", ["a", "b"]),
        search.Dimension.range("w", 0.0, 1.0, 0.5),
    ])
    covered = search.PreferenceSearch(tiny, seed=0)
    for x in (0, 1):
        for w in (0.0, 0.5, 1.0):
            covered._shown.append((x, w))
            covered._shown_keys.add(tiny.key((x, w)))
    check(covered._void_duel() is None,
          "in a fully-shown space _void_duel still claims to have found a "
          "void - whatever it returns is a re-visit dressed as exploration, "
          "and the duel it spends on it was Thompson's")

    # -- a rival duel draws one side from each basin -----------------------
    rival = search.PreferenceSearch(build_space(search), seed=4)
    home, away = (1, 2, 1, 0.6), (3, 1, 0, 0.3)
    rival.observe(home, (0, 0, 0, 0.0), 0)
    rival.observe(away, (2, 4, 2, 1.0), 0)
    check(rival._rival_duel([home]) is None,
          "_rival_duel produced a duel from a single attractor - there is "
          "no rivalry to measure, so the slot must fall through")
    duel = rival._rival_duel([home, away])
    if check(duel is not None,
             "two well-separated attractors produced no rival duel - the "
             "basins are then never compared through their populations"):
        a, b = duel

        def basin(point):
            return (0 if space.separation(point, home)
                    <= space.separation(point, away) else 1)

        check(basin(a) != basin(b),
              "both sides of a rival duel came from the same basin "
              "(A=%r, B=%r) - that is a develop duel wearing a rival "
              "duel's name, and the attractors stay uncompared" % (a, b))
        check(space.separation(a, b) >= search.MIN_DUEL_SEPARATION - 1e-9,
              "a rival duel came back %.2f apart - under the separation "
              "gate" % space.separation(a, b))


def test_reuse_and_trend(search, numpy):
    """The two economies - see items 12 and 13 in the module docstring.

    Reuse: showing an already-rendered side again is ~20x cheaper than
    generating one, so a search told (via `reuse_probe`) what is cached
    has to cash that in - visibly more reused sides - without paying for
    it in the answer, and without degenerating into a tour of the cache.
    The trend model: solver arithmetic is ~free next to a 30s generation,
    so a second cheap model screens thousands of configurations per duel
    and nominates into the pool - and its additive bias has to deliver
    the one thing the GP's product kernel cannot: the never-shown
    combination of separately-proven winners.
    """
    space = build_space(search)

    # -- the reuse preference raises the reuse rate, not the regret --------
    def run_priced(seed, duels, probe_on):
        space_ = build_space(search)
        rng = numpy.random.default_rng(1000 + seed)
        engine = search.PreferenceSearch(space_, seed=seed)
        rendered = set()
        reused = total = 0
        if probe_on:
            # The host's probe, minus the host: a point is reusable once
            # shown (vary-seed off, no cache eviction).
            engine.reuse_probe = lambda pt: space_.key(pt) in rendered
        for _ in range(duels):
            a, b = engine.next_duel()
            for point in (a, b):
                total += 1
                if space_.key(point) in rendered:
                    reused += 1
                rendered.add(space_.key(point))
            gap = taste(b) - taste(a) + rng.normal(0, 0.3)
            engine.observe(a, b, round(min(max(5 + 5 * gap / 1.2, 0), 10)))
        return engine, reused / max(total, 1)

    blind_rates, priced_rates, priced_found = [], [], []
    for seed in range(4):
        _, blind_rate = run_priced(seed, 30, probe_on=False)
        engine, priced_rate = run_priced(seed, 30, probe_on=True)
        blind_rates.append(blind_rate)
        priced_rates.append(priced_rate)
        priced_found.append(taste(engine.best()))
    blind = float(numpy.mean(blind_rates))
    priced = float(numpy.mean(priced_rates))
    check(priced >= blind + 0.08,
          "with the reuse probe wired, %.0f%% of shown sides were reused "
          "against %.0f%% without it. Reusing is ~20x cheaper than "
          "generating; a preference that does not move the rate is not an "
          "economy, it is a no-op wearing one's name"
          % (100 * priced, 100 * blind))

    optimum = max(taste((m, f, p, round(0.05 * w, 3)))
                  for m in range(len(MODELS)) for f in range(len(FACTORS))
                  for p in range(len(PROMPTS)) for w in range(21))
    rng = numpy.random.default_rng(7)
    typical = float(numpy.mean([taste(space.sample(rng)) for _ in range(4000)]))
    found = float(numpy.mean(priced_found))
    check(found >= typical + 0.6 * (optimum - typical),
          "with the reuse preference on, 30 duels found configurations "
          "worth %.2f (typical %.2f, optimum %.2f) - the same bar the "
          "cost-blind search clears. The swap rule may only trade away "
          "differences under one judgement's noise; losing the answer to "
          "save render time is the one trade that never pays"
          % (found, typical, optimum))

    # -- at the streak cap the duel IS the cost-blind duel -----------------
    #
    # Two engines on the same seed, fed identically, so their rng streams
    # are in lockstep. From duel 6 on, one gets the REALISTIC probe (shown
    # points are cached - the situation where the preference genuinely
    # moves choices, measured above) with its streak pinned at the cap
    # before every selection. If the cap properly switches the preference
    # off, every duel stays bit-identical to the twin's; any divergence
    # means the cap is not actually standing between the search and an
    # all-to-all tour of the cache. (A probe claiming EVERYTHING is cached
    # would prove nothing here: the unbiased pick is then itself "cached"
    # and the swap never moves - this test caught its own first draft.)
    capped = search.PreferenceSearch(build_space(search), seed=13)
    twin = search.PreferenceSearch(build_space(search), seed=13)
    rng = numpy.random.default_rng(77)
    shown_keys = set()
    diverged_at = None
    for index in range(18):
        if index >= 6:
            capped.reuse_probe = \
                lambda pt: capped.space.key(pt) in shown_keys
            capped._reuse_streak = search.REUSE_STREAK_MAX
        a, b = capped.next_duel()
        twin_a, twin_b = twin.next_duel()
        if (capped.space.key(a), capped.space.key(b)) \
                != (twin.space.key(twin_a), twin.space.key(twin_b)):
            diverged_at = index
            break
        for point in (a, b):
            shown_keys.add(capped.space.key(point))
        gap = taste(b) - taste(a) + rng.normal(0, 0.3)
        grade = round(min(max(5 + 5 * gap / 1.2, 0), 10))
        capped.observe(a, b, grade)
        twin.observe(twin_a, twin_b, grade)
    check(diverged_at is None,
          "with the reuse streak pinned at the cap, duel %s differed from "
          "the cost-blind twin's - the cap must switch the preference OFF, "
          "or %d cheap duels in a row simply buy a %dth instead of a fresh "
          "image" % (diverged_at, search.REUSE_STREAK_MAX,
                     search.REUSE_STREAK_MAX + 1))

    # -- the streak counts, and clears, on what is actually shown ----------
    streaky = search.PreferenceSearch(build_space(search), seed=6)
    streaky.reuse_probe = lambda pt: True
    a, b = streaky.next_duel()
    check(streaky._reuse_streak == 1,
          "a duel with both sides cached did not count toward the streak "
          "(%d) - the cap then never trips and the guard is decorative"
          % streaky._reuse_streak)
    streaky.observe(a, b, 5)
    streaky.reuse_probe = lambda pt: False
    a, b = streaky.next_duel()
    check(streaky._reuse_streak == 0,
          "a duel that brought a fresh image did not clear the streak (%d) "
          "- recovery has to be automatic, exactly like the dislike streak"
          % streaky._reuse_streak)

    # -- a broken probe is a missing probe, not a dead search --------------
    hostile = search.PreferenceSearch(build_space(search), seed=5)
    hostile.reuse_probe = lambda pt: 1 // 0
    rng = numpy.random.default_rng(505)
    for _ in range(6):
        a, b = hostile.next_duel()
        gap = taste(b) - taste(a) + rng.normal(0, 0.3)
        hostile.observe(a, b, round(min(max(5 + 5 * gap / 1.2, 0), 10)))
    check(hostile._reuse_streak == 0,
          "a probe that throws was counted as a cache hit (streak %d) - a "
          "broken economy must read as 'nothing is reusable', never as "
          "'everything is'" % hostile._reuse_streak)

    # -- the trend model waits for evidence --------------------------------
    gated = search.PreferenceSearch(build_space(search), seed=8)
    nominated_at = []
    original = gated._surrogate_nominees

    def noting(count):
        nominated_at.append(gated.duels)
        return original(count)

    gated._surrogate_nominees = noting
    rng = numpy.random.default_rng(4008)
    for _ in range(search.SURROGATE_MIN_DUELS + 6):
        a, b = gated.next_duel()
        gap = taste(b) - taste(a) + rng.normal(0, 0.3)
        gated.observe(a, b, round(min(max(5 + 5 * gap / 1.2, 0), 10)))
    if check(bool(nominated_at),
             "the trend model never nominated in %d duels - the pool then "
             "carries no cross-combination guesses at all, and the second "
             "model exists only in this file's imagination"
             % (search.SURROGATE_MIN_DUELS + 6)):
        check(min(nominated_at) >= search.SURROGATE_MIN_DUELS,
              "the trend model nominated at duel %d, before the %d-duel "
              "gate - a trend fitted to three comparisons is a guess "
              "wearing a lab coat, and its nominees crowd out the uniform "
              "fill that actually explores"
              % (min(nominated_at), search.SURROGATE_MIN_DUELS))

    # -- ...and then makes the guess the kernel cannot ---------------------
    #
    # Model 1 proves itself only at factor "0.5"; factor "1" proves itself
    # only on model 0; (model 1, factor "1") is never shown. The additive
    # trend has to put the two proven winners TOGETHER in its nominees -
    # and the proven-loser factors at the ends have to fade.
    trend = search.PreferenceSearch(build_space(search), seed=9)
    lessons = [
        ((1, 0, 0, 0.5), (0, 0, 0, 0.5), 0),   # model 1 beats model 0
        ((1, 0, 1, 0.5), (2, 0, 1, 0.5), 0),   # ...and model 2
        ((1, 0, 2, 0.5), (3, 0, 2, 0.5), 0),   # ...and model 3
        ((0, 2, 0, 0.5), (0, 0, 0, 0.5), 0),   # factor "1" beats "0.5"
        ((0, 2, 1, 0.5), (0, 4, 1, 0.5), 0),   # ...and "1.5"
        ((0, 2, 2, 0.5), (0, 1, 2, 0.5), 0),   # ...and "0.75"
    ]
    for winner, loser, grade in lessons:
        trend.observe(winner, loser, grade)
    nominees = trend._surrogate_nominees(40)
    if check(bool(nominees),
             "six perfectly consistent lessons produced no nominees"):
        model_share = float(numpy.mean(
            [1.0 if int(point[0]) == 1 else 0.0 for point in nominees]))
        edge_share = float(numpy.mean(
            [1.0 if int(point[1]) in (0, 4) else 0.0 for point in nominees]))
        check(model_share >= 0.5,
              "only %.0f%% of nominees carry the proven-best model "
              "(uniform would give 25%%) - the trend model is not reading "
              "the one main effect the lessons spelled out"
              % (100 * model_share))
        check(edge_share <= 0.2,
              "%.0f%% of nominees still carry a proven-loser factor "
              "(uniform would give 40%%) - the quadratic factor trend is "
              "not steering away from the ends the lessons rejected"
              % (100 * edge_share))
        check(any(int(point[0]) == 1 and int(point[1]) == 2
                  for point in nominees),
              "no nominee tries the proven-best model WITH the proven-best "
              "factor - the never-shown winning combination is precisely "
              "the guess the trend model exists to make and the product "
              "kernel cannot")

    # -- a conditional weight's trend is learned per pick ------------------
    #
    # Pick a prefers LOW weights, pick b prefers HIGH - mirroring the
    # kernel's no-transfer rule. A single global weight trend (no
    # interaction features) would force both picks into one direction and
    # fail here.
    pick_dim = search.Dimension.choice("pick", ["a", "b"])
    weight_dim = search.Dimension.range("weight", 0.0, 1.0, 0.05)
    weight_dim.parent = pick_dim
    cond = search.PreferenceSearch(search.Space([pick_dim, weight_dim]),
                                   seed=10)
    cond.observe((0, 0.2), (0, 0.9), 0)     # a: low beats high
    cond.observe((0, 0.1), (0, 0.8), 0)
    cond.observe((1, 0.2), (1, 0.9), 10)    # b: high beats low
    cond.observe((1, 0.1), (1, 0.8), 10)
    cond.observe((0, 0.2), (1, 0.9), 5)     # one scale across the picks
    cond._fit()
    weights = cond._surrogate_fit()
    if check(weights is not None,
             "the trend model failed to fit a two-dimensional conditional "
             "space with eight observed points"):
        probes = [(0, 0.1), (0, 0.9), (1, 0.1), (1, 0.9)]
        scores = cond._surrogate_features(probes) @ weights
        check(float(scores[0]) > float(scores[1]),
              "pick a was taught to prefer low weights and the trend "
              "scores its high end above its low end (%.3f vs %.3f)"
              % (float(scores[1]), float(scores[0])))
        check(float(scores[3]) > float(scores[2]),
              "pick b was taught to prefer high weights and the trend "
              "scores its low end above its high end (%.3f vs %.3f) - "
              "one pick's weight lessons are leaking into the other's "
              "trend, the exact false transfer the per-pick features "
              "exist to prevent"
              % (float(scores[2]), float(scores[3])))


# ---------------------------------------------------------------------------
# 4-5. The recipe
# ---------------------------------------------------------------------------

def unit(**overrides):
    values = dict(enabled=True, model="canny_xl [0123abcd]", module="canny",
                  weight_profile="0@1;0.5@0.5;1@0|0~1", weight=1.0,
                  guidance_start=0.0, guidance_end=1.0, processor_res=1024)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_recipe(dna, search):
    Gene = dna.Gene

    # Two prompt rows and two LoRA rows - the composition the user asked for,
    # and the one place several rows write the same field.
    genes = dna._build_genes([
        Gene(Gene.KIND_SETTING, 0, unit_index=0, field="model",
             values=["canny_xl [0123abcd]", "depth_xl [4567ef01]"]),
        Gene(Gene.KIND_PROFILE, 1, unit_index=0, line="Main",
             values=[0.5, 1.0, 1.5]),
        # Both characters the encoding has to survive, in one prompt: `[a|b]`
        # is prompt-editing syntax and `100%` is English, so neither is exotic.
        Gene(Gene.KIND_PROMPT, 2, values=["a house | a 100% shed", "a barn"],
             prompt_mode=dna.PROMPT_REPLACE, weights=[0.5, 1.0, 1.5]),
        Gene(Gene.KIND_PROMPT, 3, values=["golden hour", "moonlight"],
             prompt_mode=dna.PROMPT_APPEND, weights=[0.8, 1.2]),
        Gene(Gene.KIND_LORA, 4, loras=["add_detail", "film_grain"],
             weights=[0.0, 0.6, 1.0]),
        Gene(Gene.KIND_LORA, 5, loras=["sharpen"],
             weights=[0.0, 0.6, 1.0]),
    ], search)

    space = dna._space(genes, search)
    check(len(space) == 10,
          "six rows declared %d dimensions, not 10. A LoRA row and a Prompt "
          "row are TWO parameters each - which one, and how strongly - and "
          "the Model and Profile rows are one each" % len(space))

    # model=depth, factor x1.5, prompt 0 at weight 1 (the one with the
    # separator in it), 'moonlight' at 1.2, film_grain at 0.6, sharpen off.
    # The weight entries are INDICES into each row's own weight list - the
    # weight dimension is an ordered choice now, not a continuous range.
    point = (1, 2, 0, 1, 1, 1, 1, 1, 0, 0)

    prompt = dna._compose_prompt(genes, point, "BASE")
    check(prompt.startswith("a house | a 100% shed, (moonlight:1.2)"),
          "the composed prompt is %r. Replace has to land first and the "
          "appends after it, or a second prompt row silently overwrites the "
          "first - and the weight the search chose has to reach the prompt in "
          "A1111's own `(text:w)` form" % prompt)
    check("(a house" not in prompt,
          "a prompt at weight exactly 1 was still wrapped in `(text:1)`: %r. "
          "That is the same prompt spelled differently, and it turns up in "
          "every recipe and every infotext as noise" % prompt)
    check("<lora:film_grain:0.6>" in prompt,
          "the LoRA row's pick is not in the prompt: %r" % prompt)
    check("sharpen" not in prompt,
          "a LoRA at weight 0 was written into the prompt as a tag: %r. "
          "`<lora:x:0>` still loads the network, so 0 has to mean the tag is "
          "not emitted at all - otherwise the bottom of every weight range is "
          "a value that costs time and can still change the image" % prompt)

    base_units = {0: unit()}
    recipe = dna._config_string(genes, point, base_units, "BASE", seed=12345)
    values = dna._parse_config_string(recipe)

    check(values.get("u0.model") == "depth_xl [4567ef01]",
          "the recipe names model %r" % values.get("u0.model"))
    check(values.get("u0.enabled") == "True",
          "the recipe does not say the unit has to be enabled (%r). A recipe "
          "that configures a disabled unit configures nothing, and the person "
          "pasting it cannot see that from the images"
          % values.get("u0.enabled"))
    check(values.get("prompt") == prompt,
          "the prompt did not survive the round trip:\n  wrote %r\n  read  %r"
          % (prompt, values.get("prompt")))
    check(values.get("seed") == "12345",
          "the recipe lost the seed (%r) - two of the same configuration on "
          "different seeds are different images, so a recipe without it does "
          "not reproduce what was graded" % values.get("seed"))

    profile = values.get("u0.weight_profile", "")
    check(profile.endswith("|0~1.5"),
          "scaling the main profile by 1.5 produced %r. The factor multiplies "
          "the scale range - anything else means the recipe does not carry "
          "the profile that was actually generated with" % profile)
    check(profile.startswith("0@1;0.5@0.5;1@0"),
          "scaling the profile moved the drawn points (%r). The whole reason "
          "the factor lives in the range is that the points, mids, waves and "
          "the response exponent have to survive it untouched" % profile)

    # The separator and the percent sign are the two characters the encoding
    # has to survive, and prompts contain both in the wild - `[a|b]` is prompt
    # editing and `100%` is English.
    check("%7C" in recipe and "%25" in recipe,
          "the recipe did not escape the separator or the percent sign:\n  %r"
          % recipe)
    check(recipe.count("|") == recipe.count(" | "),
          "an unescaped separator is inside a value, so the recipe splits into "
          "the wrong number of tokens when it is read back:\n  %r" % recipe)

    # -- the recipe names the WHOLE generation, negative prompt included ---
    #
    # The negative prompt steers every image exactly as the positive one
    # does, and its omission from the export was reported the same way the
    # prompt's own omission once was; the sampler settings are what a pasted
    # recipe regenerates WITH. All of it has to survive the round trip,
    # escapes included, and an absent knob has to stay absent - not be
    # guessed into a value the recipe then reproduces as fact.
    extras = dna._generation_extras(types.SimpleNamespace(
        negative_prompt="blurry | 100% jpeg", steps=30,
        sampler_name="DPM++ 2M", scheduler="Karras", cfg_scale=5.5,
        width=1024, height=768, denoising_strength=None))
    full = dna._config_string(genes, point, base_units, "BASE", seed=12345,
                              extras=extras)
    read = dna._parse_config_string(full)
    check(read.get("neg_prompt") == "blurry | 100% jpeg",
          "the negative prompt did not survive the recipe round trip (%r) - "
          "a recipe without it regenerates against a DIFFERENT negative, "
          "which looks exactly like a recommendation that does not work"
          % read.get("neg_prompt"))
    for token, expected in (("steps", "30"), ("sampling", "DPM++ 2M"),
                            ("scheduler", "Karras"), ("cfg_scale", "5.5"),
                            ("width", "1024"), ("height", "768")):
        check(read.get(token) == expected,
              "the recipe lost %s (%r, wanted %r) - the settings the duels "
              "actually generated with are part of what the recipe names"
              % (token, read.get(token), expected))
    check("denoising_strength" not in read,
          "a txt2img run with no denoising strength still wrote one (%r) - "
          "an absent knob must stay absent, not be guessed"
          % read.get("denoising_strength"))
    check(dna._coerce("steps", "30") == 30
          and dna._coerce("cfg_scale", "5.5") == 5.5
          and dna._coerce("width", "1024") == 1024
          and dna._coerce("neg_prompt", "x") == "x",
          "Set/Import hand a numeric setting back to its component as text "
          "- a Number component handed a string breaks or quietly reverts")
    for name in ("neg_prompt", "steps", "sampling", "scheduler", "cfg_scale",
                 "width", "height", "denoising_strength"):
        check(name in dna._HOST_FIELD_NAMES,
              "%s is not on the host capture list - the recipe would name a "
              "value that Export cannot read and Set cannot apply" % name)

    # -- reading the panel back -------------------------------------------
    # Row slots: target, mode, choices, line, values, prompt_mode, loras,
    # weights, points - see ROW_ARGS.
    args = [1] + [None] * (dna.MAX_ROWS * dna.ROW_ARGS) + [False, True]
    args[1:1 + dna.ROW_ARGS] = ["Unit 0", dna.MODE_PROFILE, [],
                                "Main", "0.5:1.5:3", dna.PROMPT_REPLACE, "",
                                "", "0"]
    rows = dna._read_rows(args, 1)
    check(len(rows) == 1 and rows[0].kind == Gene.KIND_PROFILE,
          "a Profile row did not read back as one: %r" % (rows,))
    check([round(v, 4) for v in rows[0].values] == [0.5, 1.0, 1.5],
          "'0.5:1.5:3' expanded to %r" % (rows[0].values,))

    args[1:1 + dna.ROW_ARGS] = ["Unit 0", dna.MODE_MODEL,
                                ["canny_xl", "depth_xl"], "Main", "",
                                dna.PROMPT_REPLACE, "", "", "0"]
    rows = dna._read_rows(args, 1)
    check(len(rows) == 1 and rows[0].kind == Gene.KIND_SETTING
          and rows[0].field == "model"
          and rows[0].values == ["canny_xl", "depth_xl"],
          "a Model row did not read back as a model gene: %r" % (rows,))

    args[1:1 + dna.ROW_ARGS] = ["LoRA", dna.MODE_MODEL, [], "Main",
                                "", dna.PROMPT_REPLACE,
                                "<lora:add_detail:0.8>\nfilm_grain",
                                "0.2, 0.9", "0"]
    rows = dna._read_rows(args, 1)
    check(len(rows) == 1 and rows[0].loras == ["add_detail", "film_grain"],
          "a pasted LoRA tag did not reduce to its name: %r"
          % (rows[0].loras if rows else None,))
    check(rows and rows[0].weights == [0.2, 0.9],
          "the row's weight list was not read: %r"
          % (rows[0].weights if rows else None,))

    # The weights box takes the same interval notation as every other
    # numeric list, and an EMPTY box means the kind's default grid.
    args[8] = "0.2:1:5"
    rows = dna._read_rows(args, 1)
    check(rows and [round(w, 4) for w in rows[0].weights]
          == [0.2, 0.4, 0.6, 0.8, 1.0],
          "'0.2:1:5' expanded to %r" % (rows[0].weights if rows else None,))
    args[8] = ""
    rows = dna._read_rows(args, 1)
    check(rows and rows[0].weights == list(dna.LORA_WEIGHTS),
          "an empty weights box did not fall back to the default grid: %r"
          % (rows[0].weights if rows else None,))

    # -- two Replace rows are refused -------------------------------------
    try:
        dna._build_genes([
            Gene(Gene.KIND_PROMPT, 0, values=["x"], prompt_mode=dna.PROMPT_REPLACE),
            Gene(Gene.KIND_PROMPT, 1, values=["y"], prompt_mode=dna.PROMPT_REPLACE),
        ], search)
        fail("two Prompt rows both set to Replace were accepted. Only one of "
             "them can ever reach the model, so the other is a dimension the "
             "search spends comparisons on for nothing - and the recommendation "
             "names a prompt that never ran")
    except RuntimeError:
        pass

    # -- a negative factor is refused, not silently reinterpreted ---------
    from lib_cnpro import profile_scale
    try:
        profile_scale.parse_factors("-1")
        fail("a negative profile factor was accepted. The scale range cannot "
             "express an inverted curve - `|0~-1` is read back as `|-1~0`, "
             "which is the curve slid DOWN - so it has to be refused rather "
             "than quietly become something else")
    except ValueError:
        pass

    # -- the recipe is self-contained -------------------------------------
    #
    # Both of these were reported as bugs, and they had one cause: a recipe
    # that names only what the search VARIED. Set then leaves the prompt box
    # and the seed box alone, the user generates with whatever was in them,
    # and gets images that have nothing to do with what they graded - which
    # reads as the search having recommended something that does not work.
    plain = dna._build_genes([
        Gene(Gene.KIND_SETTING, 0, unit_index=0, field="model",
             values=["canny_xl [0123abcd]", "depth_xl [4567ef01]"]),
    ], search)
    bare = dna._parse_config_string(
        dna._config_string(plain, (1,), {0: unit()}, "a photo", seed=99))
    check(bare.get("prompt") == "a photo",
          "a recipe from a search that varied no prompt did not carry the "
          "prompt (%r). What it names has to be the WHOLE configuration, or "
          "Set reproduces something else" % bare.get("prompt"))
    check(bare.get("seed") == "99",
          "a recipe without the seed (%r) generates on a fresh noise field "
          "every time, which looks exactly like a recommendation that does "
          "not work" % bare.get("seed"))

    # -- prompt weights are written in the host's own syntax ---------------
    weighted = dna._build_genes([
        Gene(Gene.KIND_PROMPT, 0, values=["a (round) house"],
             prompt_mode=dna.PROMPT_APPEND, weights=[0.5, 1.3, 1.5]),
    ], search)
    text = dna._compose_prompt(weighted, (0, 1), "BASE")
    check(text == "BASE, (a \\(round\\) house:1.3)",
          "a weighted prompt came out as %r. Parentheses inside the text have "
          "to be escaped or they close the weight early and turn the rest of "
          "the prompt into an expression" % text)

    # -- a recipe saved before the header went away still pastes -----------
    header_form = "DNA1 | u0.model=depth_xl [4567ef01] | seed=7"
    old = dna._parse_config_string(header_form)
    check(old.get("u0.model") == "depth_xl [4567ef01]" and old.get("seed") == "7",
          "a configuration string carrying the old DNA1 header no longer "
          "parses (%r). It is not written any more, but strings saved with it "
          "are in people's notes" % (old,))
    check(not dna._config_string(plain, (1,), {0: unit()}, "a photo",
                                 seed=1).startswith("DNA"),
          "the header is being written again - it was on the front of every "
          "recipe and every trace line and nothing ever read it")

    # -- the trace says what was CHOSEN, not the whole configuration -------
    line = dna._choices_line(genes, point)
    check("weight_profile" not in line and "0@1;" not in line
          and "seed=" not in line,
          "a trace line is carrying the resolved configuration (%r). At a "
          "profile string and a composed prompt per row that is kilobytes a "
          "line, which is neither readable nor bounded" % line)
    check("x1.5" in line and "film_grain @ 0.6" in line,
          "a trace line does not name the choices it was supposed to (%r) - "
          "which value each row landed on is the whole point of it" % line)
    long_prompt = dna._build_genes([
        Gene(Gene.KIND_PROMPT, 0, values=["z" * 400], prompt_mode=dna.PROMPT_APPEND,
             weights=[0.5, 1.0, 1.5]),
    ], search)
    trace_line = dna._choices_line(long_prompt, (0, 1))
    check(len(trace_line) < 120,
          "a 400-character prompt produced a %d-character trace line - prompts "
          "have to be cut to %d characters there, or one row's choice fills "
          "the box" % (len(trace_line), dna.CHOICE_PROMPT_CHARS))

    # -- the session file round-trips --------------------------------------
    #
    # The export is one HTML file that is simultaneously a JSON island this
    # script reads back VERBATIM and a page a browser renders. The payload
    # here carries the two things most likely to break it: "</script>"
    # inside a prompt (would end the island early un-escaped) and the
    # recipe escapes' own characters.
    payload = {
        "app": dna.WHO, "version": dna.EXPORT_VERSION, "tab": "txt2img",
        "row_count": 2,
        "rows": [{"target": "LoRA", "mode": "Model",
                  "choices": [], "line": "Main",
                  "values": "a house | 100% </script> shed",
                  "prompt_mode": "Replace", "loras": "add_detail",
                  "weights": "0.2, 0.6, 1", "points": "0"}],
        "tail": {"vary_seed": False, "resume": True},
        "settings": {"prompt": "</script><script>alert(1)</script>",
                     "u0.enabled": True, "u0.model": "depth_xl [4567ef01]"},
        "canvases": {},
        "solver": None,
        "result": "u0.enabled=True | prompt=x %7C y | seed=7",
    }
    round_tripped = dna._parse_export(dna._export_html(payload))
    check(round_tripped == payload,
          "the session file did not round-trip:\n  wrote %r\n  read  %r"
          % (payload, round_tripped))
    page = dna._export_html(payload)
    check("<h2>Settings</h2>" in page and "<h2>Search rows</h2>" in page,
          "the session file has no human-readable half - a version this "
          "build cannot import is then a dead file instead of a page the "
          "user can reproduce by hand")

    # -- the solver state replays into an identical model ------------------
    import numpy
    replay_space = build_space(search)
    original = search.PreferenceSearch(replay_space, seed=9)
    rng = numpy.random.default_rng(99)
    for index in range(8):
        a, b = original.next_duel()
        original.observe(a, b, int(rng.integers(0, 11)),
                         disliked=(index % 3 == 0),
                         interesting_b=(index == 2))
    state = json.loads(json.dumps(
        dna._solver_payload(original, 9, replay_space)))
    resumed_space = build_space(search)
    resumed = search.PreferenceSearch(resumed_space, seed=9)
    count = dna._restore_solver(resumed, state, resumed_space)
    check(count == len(original.observations),
          "the replay restored %r of %d observations" %
          (count, len(original.observations)))
    check(resumed.duels == original.duels
          and len(resumed._interesting) == len(original._interesting),
          "the replay lost the duel count (%r vs %r) or the interesting "
          "marks (%r vs %r)" % (resumed.duels, original.duels,
                                len(resumed._interesting),
                                len(original._interesting)))
    probes = original.points[:4]
    means_original = original._posterior(probes, with_covariance=False)[0]
    means_resumed = resumed._posterior(probes, with_covariance=False)[0]
    check(all(abs(float(x) - float(y)) < 1e-9
              for x, y in zip(means_original, means_resumed)),
          "a replayed search believes something different from the search "
          "it was exported from - the import is then a lookalike, not a "
          "resume")

    other_space = search.Space([
        search.Dimension.choice("model", ["different", "labels"]),
        search.Dimension.range("w", 0.0, 1.0, 0.05)])
    check(dna._restore_solver(search.PreferenceSearch(other_space, seed=0),
                              state, other_space) is None,
          "RESUME replayed solver state onto a DIFFERENT space - old "
          "comparisons must retain their meaning there, even though GOOD "
          "is allowed to project them onto edited rows")

    projected = search.PreferenceSearch(other_space, seed=0)
    projected_count = dna._restore_solver(
        projected, state, other_space, allow_row_changes=True)
    check(projected_count == len(original.observations),
          "GOOD-style replay refused a different set of rows instead of "
          "projecting the learned coordinates onto their current values")
    check(all(len(point) == len(other_space.dimensions)
              and 0 <= int(point[0]) < len(other_space.dimensions[0].labels)
              and other_space.dimensions[1].low <= float(point[1])
                  <= other_space.dimensions[1].high
              for point in projected.points),
          "projecting retained knowledge produced a point outside the "
          "currently edited rows: %r" % (projected.points,))

    # -- the default weighted-channel grids -------------------------------
    check(dna.LORA_WEIGHTS == [0.2, 0.6, 1.0],
          "the default LoRA weight grid is %r, not the three states "
          "0.2 / 0.6 / 1.0. Coarse is the contract: three visible steps per "
          "LoRA keeps a five-LoRA search at 3^5 weight combinations instead "
          "of 21^5, and the 0.2 floor avoids the weight-0 degeneracy where "
          "every LoRA in a slot is the same image" % (dna.LORA_WEIGHTS,))
    check(dna.PROMPT_WEIGHTS == [0.4, 0.8, 1.2],
          "the default prompt weight grid is %r, not 0.4 / 0.8 / 1.2 - "
          "straddling 1 because de-emphasis and emphasis both live near it"
          % (dna.PROMPT_WEIGHTS,))
    # An ordered choice, not a bag of categories: the weight labels are
    # numbers, so evidence still transfers between neighbouring weights.
    lora_dims = Gene(Gene.KIND_LORA, 0, loras=["a", "b"]).dimensions(search)
    check(len(lora_dims) == 2 and lora_dims[1].numeric
          and lora_dims[1].parent is lora_dims[0],
          "a LoRA row's weight list is not an ordered, pick-conditional "
          "dimension (numeric=%r, parent ok=%r) - either the metric was "
          "lost (no transfer between 0.5 and 0.6) or the weight is shared "
          "across picks (the false-transfer the parent link prevents)"
          % (lora_dims[1].numeric if len(lora_dims) == 2 else None,
             len(lora_dims) == 2 and lora_dims[1].parent is lora_dims[0]))

    # -- the checkbox coercion --------------------------------------------
    check(dna._coerce("u0.enabled", "False") is False,
          "'False' was not coerced to a boolean. A Checkbox handed the STRING "
          "'False' renders as checked, because a non-empty string is truthy - "
          "so Set would enable a unit the recipe says to disable")
    check(dna._coerce("u0.processor_res", "1024") == 1024,
          "a numeric field did not come back as a number")


def test_demo_session(dna):
    """GOOD/N-GOOD outlive the search - the session and staging mechanics.

    The buttons are the USE of a trained solver, and three behaviours make
    that true rather than aspirational:

    * with no loop alive, request_demo REFUSES - the refusal is the panel's
      cue to take the idle path (stage the request, click Generate itself),
      so a session that accepted instead would strand the press silently;
    * with a loop alive, requests QUEUE in arrival order - a press during a
      render is a request, not a mistimed click - and never overtake a
      grade that landed in the same instant: the grade answers the question
      on screen, the demo is a side request;
    * a staged idle request is served one per run and a stale one is
      dropped, so a press that never became a run cannot turn a later,
      unrelated Generate into a sample render nobody asked for.

    A request carries WHAT WAS ASKED FOR, not merely that something was: the
    two buttons differ only in the count, and a queue that forgot it would
    serve a 12-sample collage as one sample.
    """
    import time as _time

    session = dna._Session()
    check(not session.request_demo(),
          "a session with no loop alive accepted a GOOD request - the panel "
          "then never takes the idle path, and pressing the button while "
          "idle does nothing at all")

    session.start()
    check(session.request_demo() and session.request_demo(12),
          "a running session refused a GOOD/N-GOOD press that did not land "
          "exactly while a duel was awaiting a grade - a press during a "
          "render is a request, not a mistimed click")
    first, second = session.await_grade(), session.await_grade()
    check(isinstance(first, dna._Demo) and first.count is None,
          "the first queued press did not come back as the single GOOD "
          "sample it was (%r)" % (first,))
    check(isinstance(second, dna._Demo) and second.count == 12,
          "an N-GOOD press for 12 samples came back as %r - the count is "
          "what the whole button is, and a queue that drops it renders one "
          "image where a collage was asked for" % (second,))

    session.publish(1, None, None, "waiting")
    session.request_demo()
    session.grade(7, similar=True)
    first = session.await_grade()
    check(isinstance(first, tuple) and first[0] == 7.0,
          "a grade and a queued demo raced and the demo won (%r) - the "
          "grade answers the question on screen and must be served first"
          % (first,))
    check(first[1] is False and first[2] is True,
          "the grade came back as %r - the ROW is the verdict, so which one "
          "the click landed on has to survive the trip to run(): a similar "
          "click arriving as a plain one silently drops the only signal a "
          "comparison cannot carry" % (first,))
    check(isinstance(session.await_grade(), dna._Demo),
          "the queued demo was lost once a grade overtook it")

    session.request_demo()
    session.start()
    check(not session.pending_demos,
          "start() carried a stale demo request into the new search - the "
          "first duel would be interrupted by a sample nobody asked this "
          "session for")
    session.finish("done")
    check(not session.request_demo(),
          "a finished session still accepted a demo request onto its dead "
          "queue - nothing will ever serve it, and the idle path (which "
          "would) is never taken")

    script = dna.Script()
    staged = dna._PENDING_DEMO["txt2img"]
    fresh = dna._Demo(4)
    staged[:] = [(dna._Demo(), _time.time() - dna.DEMO_STALE_SECONDS - 1),
                 (fresh, _time.time())]
    check(script._pop_demo_request() is fresh,
          "the staged queue served a stale entry (or lost the fresh one "
          "behind it) - a press that never became a run must not hijack a "
          "later Generate")
    check(script._pop_demo_request() is None and not staged,
          "the staged queue was not drained one-per-run")


def test_demo_accepts_edited_rows(dna, search):
    """A finished search remains a generator after its rows are edited."""
    old_space = search.Space([
        search.Dimension.choice("old choice", ["a", "b", "c"]),
        search.Dimension.range("old range", 0.0, 1.0, 0.1),
    ])
    payload = {
        "signature": dna._space_signature(old_space),
        "observations": [[[2, 1.0], [0, 0.0], 0.8],
                         [None, [2, 1.0], search.ON_TRACK_P]],
        "similarities": [],
        "interesting": [[2, 1.0]],
        "on_track_anchors": True,
        "duels": 1,
    }
    edited_space = search.Space([
        search.Dimension.choice("edited choice", ["new"]),
        search.Dimension.range("edited range", 0.2, 0.4, 0.1),
        search.Dimension.choice("added row", ["first", "second"]),
    ])

    script = dna.Script()
    session = dna._Session()
    rendered = types.SimpleNamespace(images=["rendered"], infotexts=["info"])
    seen = []

    def samples(*args, **_kwargs):
        seen.append(args[1])
        return [rendered], "GOOD sample rendered"

    script._samples = samples
    p = types.SimpleNamespace(extra_generation_params={})
    tab = "txt2img"
    marker = object()
    previous = dna._PENDING_SOLVER.get(tab, marker)
    dna._PENDING_SOLVER[tab] = payload
    try:
        result = script._demo_run(
            dna._Demo(), p, search, edited_space, [], [],
            types.SimpleNamespace(args_from=0), {}, "prompt", 7, session)
    finally:
        if previous is marker:
            dna._PENDING_SOLVER.pop(tab, None)
        else:
            dna._PENDING_SOLVER[tab] = previous

    check(result is rendered and len(seen) == 1,
          "GOOD refused the retained solver after the rows were edited "
          "instead of rendering from it")
    projected = set(seen[0].points) if seen else set()
    check(projected == {(0, 0.2, 0), (0, 0.4, 0)},
          "GOOD did not project old coordinates onto the edited rows: %r"
          % (projected,))
    check("refused" not in session.status.lower(),
          "GOOD still reports a row-mismatch refusal: %r" % session.status)


def test_collage_cache_policy(dna):
    """N-GOOD reads paid-for duel renders without evicting them on misses."""
    script = dna.Script()
    p = types.SimpleNamespace(styles=[], override_settings={},
                              extra_generation_params={})
    recipe = dna._config_string([], (), {}, "base prompt", 11)
    hit = types.SimpleNamespace(images=["already rendered"])
    cache = {recipe: hit}
    calls = []
    original = dna.process_images

    def render(processing):
        calls.append(processing)
        return types.SimpleNamespace(images=["new render"], infotexts=[""])

    dna.process_images = render
    try:
        got = script._render(
            p, [], types.SimpleNamespace(args_from=0), [], (), {},
            "base prompt", 11, cache=cache, cache_store=False)
        check(got is hit and not calls,
              "a read-only N-GOOD cache lookup regenerated an image already "
              "paid for during a duel")

        empty = {}
        script._render(
            p, [], types.SimpleNamespace(args_from=0), [], (), {},
            "base prompt", 11, cache=empty, cache_store=False)
        check(len(calls) == 1 and not empty,
              "a fresh collage render entered the duel cache - a large "
              "collage would evict the very images reuse is meant to save")
    finally:
        dna.process_images = original

    # The N box is free text beside a button, and the press has already
    # happened by the time it is read.
    for text, wanted in (("12", 12), ("  7 ", 7), ("", dna.CAPACITY_DEFAULT),
                         ("twelve", dna.CAPACITY_DEFAULT), ("-3", 1),
                         ("0", 1), ("100000", dna.COLLAGE_MAX)):
        got = dna._collage_count(text)
        check(got == wanted,
              "the N box read %r as %r rather than %r - it is free text next "
              "to a button that spends GPU minutes, so every reading of it "
              "has to be a number in range" % (text, got, wanted))


def test_collage_layout(dna):
    """N-GOOD is one horizontal comparison strip, whatever N is."""
    try:
        from PIL import Image
    except Exception as exc:
        SKIPS.append("Pillow is absent, so collage geometry was not checked "
                     "(%s)" % exc)
        return

    samples = [Image.new("RGB", (20 + index, 12 + 2 * index),
                         (40 * index, 0, 0))
               for index in range(1, 6)]
    sheet = dna._collage(samples)
    cell_w = max(image.width for image in samples)
    cell_h = max(image.height for image in samples)
    wanted = (len(samples) * cell_w
              + (len(samples) + 1) * dna.COLLAGE_GAP,
              cell_h + 2 * dna.COLLAGE_GAP)
    check(sheet is not None and sheet.size == wanted,
          "N-GOOD composed five samples as %r instead of the one-row strip "
          "%r - the collage must be N columns by one row, not a square-ish "
          "grid" % (getattr(sheet, "size", None), wanted))


def test_resume_matches_visible_rows(dna, search):
    """RESUME is available only for the exact space shown in the panel."""
    matcher = getattr(dna, "_solver_matches_rows", None)
    if not check(callable(matcher),
                 "RESUME has no shared row-compatibility predicate - its "
                 "paint and click paths can then disagree about whether the "
                 "visible search can continue"):
        return
    values = [1] + [None] * (dna.MAX_ROWS * dna.ROW_ARGS)
    values[1:1 + dna.ROW_ARGS] = [
        dna.TARGET_PROMPT, dna.MODE_MODEL, [], "Main", "first\nsecond",
        dna.PROMPT_REPLACE, "", "", "0"]
    genes = dna._build_genes(dna._read_rows(values, 1), search)
    state = {
        "signature": dna._space_signature(dna._space(genes, search)),
        "observations": [[None, [0, 0.4], search.ON_TRACK_P]],
    }
    check(matcher(state, values, search),
          "RESUME rejected solver state learned over the visible rows")

    edited = list(values)
    edited[5] = "first\nthird"
    check(not matcher(state, edited, search),
          "RESUME accepted solver state from different visible row values - "
          "pressing it would advertise a continuation and start fresh")
    check(not matcher({**state, "observations": []}, values, search),
          "RESUME accepted a matching shell with no observations to resume")


def test_loop_controls(dna):
    """START/STOP and RESUME, and the promise that Generate is not one of them.

    THE PROPERTY WORTH PINNING IS THE NEGATIVE ONE. Selecting this script
    used to mean the next Generate spent itself on a search; it does not any
    more, and "a plain Generate is handed back to the host" is invisible from
    the outside until the day it regresses - at which point pressing Generate
    to check a prompt opens a duel loop instead. `run()` returning None is
    the whole of that contract, so it is checked first and directly.

    The rest is the state machine the two buttons share:

    * a press STAGES its intent and clicks Generate, because a gradio handler
      cannot invoke the host's pipeline - so what the press meant has to
      survive the trip, and START and RESUME must not arrive as each other;
    * the session reads BUSY from the press rather than from the loop, or the
      window between the click and the host picking the job up is a window in
      which the panel offers START again and a second search gets staged;
    * a burst of presses is still one search - they all asked for the same
      single thing, and a queue of them would start another one every time
      the host drained a job;
    * a stale press is dropped, exactly as a stale GOOD press is, for a
      strictly worse failure: a hijacked Generate that starts asking for
      grades rather than one that renders an unasked-for sample.
    """
    import time as _time

    tab = "txt2img"
    session = dna._SESSIONS[tab]
    staged = dna._PENDING_START[tab]
    script = dna.Script()

    def reset():
        staged[:] = []
        dna._PENDING_DEMO[tab][:] = []
        dna._PENDING_SOLVER.pop(tab, None)
        session.running = session.launching = session.stopping = False
        session.solver_state = None

    reset()
    check(script.run(None) is None,
          "a Generate with nothing staged was not handed back to the host - "
          "selecting this script would then turn the host's own button into "
          "a search, which is exactly what it must not do")

    for resume in (False, True):
        reset()
        staged.append((resume, _time.time()))
        got = script._pop_start_request()
        check(got is resume,
              "a staged %s press came back as %r - which button was pressed "
              "is decided at the press and travels with the entry, and the "
              "two mean opposite things about the retained state"
              % ("RESUME" if resume else "START", got))
        check(not staged, "the staged press was not consumed by the run it "
                          "started")

    reset()
    staged[:] = [(False, _time.time()), (False, _time.time()),
                 (True, _time.time())]
    check(script._pop_start_request() is False and not staged,
          "a burst of presses left searches queued behind the first - each "
          "would start another one as the host drained the queue")

    reset()
    staged[:] = [(False, _time.time() - dna.DEMO_STALE_SECONDS - 1)]
    check(script._pop_start_request() is None,
          "a stale START press survived to hijack a later, unrelated "
          "Generate into a search")

    reset()
    dna._PENDING_DEMO[tab].append((dna._Demo(), _time.time()))
    staged.append((False, _time.time()))
    try:
        script.run(None)
    except Exception:
        # No rows against the stubs, so this run dies just after the two
        # pops - which is all this case is about.
        pass
    check(len(staged) == 1,
          "a GOOD generation swallowed the START staged behind it - that "
          "press queued a generation of its own and never gets served")

    # The panel reads busy from the PRESS, not from the loop.
    reset()
    session.arm("starting")
    check(session.snapshot()["launching"] and not session.snapshot()["running"],
          "arm() did not put the session in the staged-but-not-started state "
          "the button and the poll both read")
    session.request_stop()
    check(not session.launching,
          "STOP over a press that never became a run left the panel armed - "
          "the button reads STOP for as long as it is, so it has to be able "
          "to take that back")
    session.arm("starting")
    script.run(None)
    check(not session.launching,
          "reaching run() did not disarm the session - a press whose click "
          "never became a run would leave the panel reading 'starting' "
          "forever")
    reset()


def test_set_good_and_reset_state(dna, search):
    """Set GOOD samples from the learned taste; Reset has an exact origin."""
    initial = {
        "u0.enabled": True,
        "u0.model": "canny",
        "u0.weight_profile": "0@1;1@1",
        "u0.module": "none",
        "prompt": "the initial prompt",
        "neg_prompt": "blur",
        "steps": 28,
        "seed": 123,
    }

    # Applying a point starts from the snapshot, not from whatever currently
    # happens to be in the UI. Both a unit row and a prompt row are included
    # so this catches an implementation that only rewrites the easy scalar.
    model = dna.Gene(dna.Gene.KIND_SETTING, 0, unit_index=0,
                     field="model", values=["canny", "depth"])
    prompt = dna.Gene(dna.Gene.KIND_PROMPT, 1,
                      values=["golden hour", "rain"],
                      prompt_mode=dna.PROMPT_APPEND, weights=[0.8, 1.2])
    model.offset = 0
    prompt.offset = 1
    configured = dna._configuration_for_point(
        [model, prompt], (1, 0, 1), initial)
    check(configured.get("u0.model") == "depth",
          "Set GOOD did not apply the sampled unit value")
    check(configured.get("prompt") ==
          "the initial prompt, (golden hour:1.2)",
          "Set GOOD composed its prompt against something other than the "
          "initial search prompt: %r" % configured.get("prompt"))
    check(configured.get("neg_prompt") == "blur"
          and configured.get("steps") == 28
          and configured.get("u0.module") == "none",
          "Set GOOD lost untouched initial settings: %r" % configured)
    check(initial["u0.model"] == "canny"
          and initial["prompt"] == "the initial prompt",
          "sampling mutated the memorized Reset snapshot")

    # The complete snapshot uses Set's existing grammar, so Reset and the
    # visible Configuration box cannot disagree about escaping or types.
    snapshot_text = dna._settings_string(initial, list(initial))
    snapshot_read = dna._parse_config_string(snapshot_text)
    check(snapshot_read.get("prompt") == "the initial prompt"
          and snapshot_read.get("u0.model") == "canny"
          and snapshot_read.get("steps") == "28",
          "the initial snapshot does not round-trip through Set's grammar: %r"
          % snapshot_read)

    # One positively anchored model and one negatively anchored model make
    # the sampled GOOD unambiguous. This exercises retained-state replay, not
    # merely the deterministic point-to-settings helper above.
    one_gene = dna.Gene(dna.Gene.KIND_SETTING, 0, unit_index=0,
                        field="model", values=["canny", "depth"])
    one_gene.offset = 0
    space = dna._space([one_gene], search)
    learned = search.PreferenceSearch(space, seed=123)
    bad, good = (0,), (1,)
    learned.observations.extend([
        (None, learned._row(bad), 0.02),
        (None, learned._row(good), 0.98),
    ])
    learned._dirty = True
    payload = dna._solver_payload(learned, 123, space, initial)
    payload = json.loads(json.dumps(payload))
    check(payload.get("initial_settings") == initial,
          "the retained solver payload lost the settings Reset needs after "
          "a reload/restart: %r" % payload.get("initial_settings"))
    sampled, error = dna._sample_good_configuration(
        payload, [one_gene], initial, serial=1)
    check(error is None and sampled is not None,
          "Set GOOD could not sample the retained solver: %r" % error)
    if sampled is not None:
        check(sampled.get("u0.model") == "depth",
              "Set GOOD sampled %r instead of the only positively anchored "
              "configuration" % sampled.get("u0.model"))
        check(sampled.get("prompt") == initial["prompt"]
              and sampled.get("steps") == initial["steps"],
              "Set GOOD did not rebuild from the memorized initial settings")

    session = dna._Session()
    mutable = {"prompt": "before", "input_order": [1, 2]}
    session.remember_initial(mutable)
    mutable["prompt"] = "after"
    mutable["input_order"].append(3)
    check(session.initial_settings ==
          {"prompt": "before", "input_order": [1, 2]},
          "Reset remembered a live reference rather than the initial values")


def test_point_gene(dna, search):
    """The Profile-points row, at the gene level.

    Three things make it a search dimension rather than a gimmick: the
    offsets are an ORDERED choice (evidence transfers between neighbouring
    values), the recipe names the REWRITTEN weight_profile (a recipe that
    omits it reproduces the un-offset image), and offset 0 reproduces the
    drawn profile verbatim - the baseline duel really is the baseline.
    """
    unit = types.SimpleNamespace(weight_profile="0@0;0.5@0.5;1@1|0~2",
                                 enabled=True, weight=1.0,
                                 guidance_start=0.0, guidance_end=1.0)
    gene = dna.Gene(dna.Gene.KIND_POINT, 0, unit_index=0, line="Main",
                    values=[-0.5, 0.0, 0.5], point_indices=[1])
    dims = gene.dimensions(search)
    check(len(dims) == 1 and dims[0].numeric,
          "a point row's offsets are not an ordered dimension - the search "
          "then cannot transfer evidence between -0.1 and -0.2, which is "
          "the single most useful structure the numbers carry")

    moved = dna._config_string([gene], (2,), {0: unit}, "prompt", seed=1,
                               extras={})
    check("u0.weight_profile=" in moved and "0.5@0.75" in moved,
          "the recipe does not name the offset profile - pasting it back "
          "reproduces the UN-offset image: %r" % (moved,))
    baseline = dna._config_string([gene], (1,), {0: unit}, "prompt", seed=1,
                                  extras={})
    check("0@0;0.5@0.5;1@1%7C0~2" in baseline,
          "offset 0 did not reproduce the drawn profile verbatim in the "
          "recipe: %r" % (baseline,))
    # The neutral offset prints as a bare "0", not as words: the trace is a
    # column of these, one per row per image, and "0" among signed offsets
    # already reads as "the profile untouched" for a fraction of the width.
    # What has to hold is that the line says WHICH offset the duel chose.
    check(gene.describe((1,)).endswith(" 0")
          and "+0.5" in gene.describe((2,)),
          "the trace line does not say which offset a duel chose: %r / %r"
          % (gene.describe((1,)), gene.describe((2,))))

    # A LIST of indices moves as a GROUP: one offset, every listed knot -
    # that is what edits a profile interval instead of a single point.
    group = dna.Gene(dna.Gene.KIND_POINT, 0, unit_index=0, line="Main",
                     values=[-0.5, 0.0, 0.5], point_indices=[0, 1])
    together = dna._config_string([group], (2,), {0: unit}, "prompt", seed=1,
                                  extras={})
    check("0@0.25" in together and "0.5@0.75" in together,
          "offsetting points [0, 1] by +0.5 did not move BOTH knots "
          "(%r) - the group is the feature: the listed points shift "
          "together, or the row is just the old single-point edit with a "
          "longer label" % (together,))

    # The index-list parser: the box's own grammar.
    check(dna._point_indices("0, 2, -1", 0) == [0, 2, -1]
          and dna._point_indices("", 0) == [0]
          and dna._point_indices("  1  3 ", 0) == [1, 3],
          "the points box does not parse comma/whitespace index lists "
          "(with empty meaning [0], the box's initial value)")
    for bad in ("1.5", "one", "0, 0"):
        try:
            dna._point_indices(bad, 0)
            fail("%r was accepted as a point-index list - garbage or a "
                 "repeated index has to be refused before it silently "
                 "becomes a double-moved knot or a dropped row" % (bad,))
        except ValueError:
            pass


def test_persistence(dna, search, numpy):
    """Nothing ends a search irrecoverably - the fix for "A/B and GOOD/N-GOOD
    stop working" after a STOP or an Interrupt.

    Four properties carry it:

    * start() KEEPS the retained solver state - clearing it there meant that
      starting a search and interrupting it before the first grade destroyed
      the PREVIOUS search's learned taste, and GOOD/N-GOOD then reported "no
      solver state" about work that had cost dozens of duels;
    * a RESUMED start keeps the RECORD too - the Tried lines, the
      recommendation and the summary describe the very duels the resumed
      observations came from, and wiping them left a record that began
      mid-session while the first line of it claimed to be resuming 51
      observations;
    * a FRESH start still clears all of it, or a search on new rows would
      open showing another search's history;
    * the state is mirrored to disk and read back, and the retained-state
      lookup prefers the explicit sources over the mirror: staged import
      first, then the session's own state, then the disk.

    There is no timeout to test: a duel waits for its grade indefinitely -
    see _Session.await_grade.
    """
    session = dna._Session()
    session.solver_state = {"observations": [[None, [0], 0.5]], "duels": 1}
    session.start()
    check(session.solver_state is not None,
          "start() cleared the retained solver state - interrupting a fresh "
          "search before its first grade then destroys the previous search's "
          "learned taste, and GOOD/N-GOOD go dead with it")

    session.record("A", "u0.model=x")
    session.record("B", "u0.model=y")
    session.set_result("u0.model=x", "2 graded, confidence 40%")
    session.start(resumed=True)
    check(len(session.trace) == 2,
          "a RESUMED start wiped the Tried record (%d lines left). The "
          "observations it is resuming from are the very duels those lines "
          "describe, so the panel comes back claiming to resume dozens of "
          "duels above an empty history - which is how a stop-and-continue "
          "silently costs the session's whole record"
          % len(session.trace))
    check(session.result and session.summary,
          "a RESUMED start cleared the recommendation it is about to "
          "recompute from the very same state - the panel blinks back to "
          "'nothing graded yet' over a search that knows dozens of duels")

    session.start()
    check(not session.trace and not session.result and not session.summary,
          "a FRESH start kept the previous search's record - these rows have "
          "never been tried, so a Tried box full of another space's recipes "
           "is worse than an empty one")

    # PRE-SCHEMA REPLAY: the old file retained the similarity row but not the
    # shared "on track" half of its meaning.  That is enough information to
    # recover the two above-par anchors exactly once.  Current payloads carry
    # the marker and must not receive the anchors twice on every resume.
    space = build_space(search)
    a, b = space.baseline(), space.perturb(
        space.baseline(), numpy.random.default_rng(17))
    old_state = {
        "signature": dna._space_signature(space),
        "observations": [[list(a), list(b), 0.7],
                         [list(a), list(b), 0.5]],
        "similarities": [[list(a), list(b), False, 1.0]],
        "interesting": [], "duels": 1,
    }
    restored = search.PreferenceSearch(space, seed=1)
    dna._restore_solver(restored, old_state, space)
    means, _ = restored._posterior([a, b], with_covariance=False)
    check(len(restored.observations) == 4 and float(min(means)) > 0.0,
          "a pre-anchor saved search did not recover both samples' old "
          "on-track verdict during replay")
    current_state = dict(old_state)
    current_state["observations"] = [
        *old_state["observations"], [None, list(a), search.ON_TRACK_P],
        [None, list(b), search.ON_TRACK_P]]
    current_state["on_track_anchors"] = True
    restored_current = search.PreferenceSearch(space, seed=1)
    dna._restore_solver(restored_current, current_state, space)
    check(len(restored_current.observations) == 4,
          "a current saved search duplicated its on-track anchors on replay")

    tab = "persistence-test"
    payload = {"observations": [[None, [1], 0.9]], "duels": 3,
               "signature": [], "seed": 1, "interesting": [],
               "hyper": [0.3, 1.5, 0.4]}
    try:
        dna._store_solver(tab, payload)
        check(dna._stored_solver(tab) == payload,
              "the disk mirror did not round-trip - a restart then loses the "
              "learned state the mirror exists to keep")
        check(dna._stored_solver("no-such-tab-ever") is None,
              "a missing mirror did not read as None")

        probe = dna._Session()
        dna._SESSIONS[tab] = probe
        check(dna._retained_solver(tab) == payload,
              "the retained-state lookup did not fall back to the disk "
              "mirror - GOOD/BAD and Resume then die with a restart")
        probe.solver_state = {"observations": [["s"]], "duels": 9}
        check(dna._retained_solver(tab) == probe.solver_state,
              "the session's own state did not win over the disk mirror - "
              "the mirror is the fallback, not the authority")
        dna._PENDING_SOLVER[tab] = {"observations": [["i"]], "duels": 2}
        check(dna._retained_solver(tab) == dna._PENDING_SOLVER[tab],
              "a staged import did not win the retained-state lookup - it is "
              "the one source the user pointed at explicitly")
    finally:
        dna._SESSIONS.pop(tab, None)
        dna._PENDING_SOLVER.pop(tab, None)
        try:
            os.remove(dna._solver_file(tab))
        except OSError:
            pass


def test_phase_css(dna):
    """The tick's phase CSS: per-side inference progress on the interesting
    toggles, and the grade-here nudge exactly while a grade is awaited.

    The toggles STAY toggles - the fill is a pseudo-element behind the
    label - and the nudge is opacity-only, so neither can move the rows a
    fast hand is aiming at.
    """
    def snap(**kw):
        base = {"running": False, "awaiting": False, "generating": None}
        base.update(kw)
        return base

    check(dna._phase_css("cnpro_ab_txt2img", snap(), 0.5) == "",
          "an idle panel got phase CSS - the tick then re-renders a style "
          "element every poll for nothing")
    a = dna._phase_css("cnpro_ab_txt2img",
                       snap(running=True, generating="A"), 0.43)
    check("#cnpro_ab_txt2img_interesting_a{--cnpro-ab-fill:43%}" in a
          and "interesting_b" not in a,
          "generating A did not fill A (and only A): %r" % (a,))
    b = dna._phase_css("cnpro_ab_txt2img",
                       snap(running=True, generating="B"), 0.25)
    check("#cnpro_ab_txt2img_interesting_a{--cnpro-ab-fill:100%}" in b
          and "#cnpro_ab_txt2img_interesting_b{--cnpro-ab-fill:25%}" in b,
          "generating B must show A complete and B in progress - the pair "
          "is what reads as 'A done, B under way': %r" % (b,))
    check("grade-hint" not in a and "grade-hint" not in b,
          "the grade-here nudge showed while an image was still rendering - "
          "it must mark the user's turn, not the machine's")
    waiting = dna._phase_css("cnpro_ab_txt2img", snap(awaiting=True), 0.0)
    check(".cnpro-ab-grade-hint{opacity:1}" in waiting
          and "#cnpro_ab_txt2img_interesting_a{--cnpro-ab-fill:100%}" in waiting
          and "#cnpro_ab_txt2img_interesting_b{--cnpro-ab-fill:100%}" in waiting,
          "a duel awaiting its grade must fade the nudge in and keep both "
          "completed fills at 100%%: %r" % (waiting,))
    clamped = dna._phase_css("x", snap(running=True, generating="A"), 2.0)
    check("100%" in clamped and "200%" not in clamped,
          "progress above 1 was not clamped: %r" % (clamped,))

    # THE COLLAGE BUTTON'S WIDTH rides the same channel, as a DIGIT COUNT.
    # style.css turns it into a width in `ch` - the width of a digit in the
    # font the button is drawn in - and GOOD above it reads the same
    # variable, which is what keeps the two one column at any count. A pixel
    # width computed in python would be the panel guessing at a font it
    # never sees, and it is the one thing this payload must never contain.
    for count, digits in ((3, 1), (12, 2), (9999, 4)):
        sized = dna._phase_css("x", snap(), 0.0, count)
        check(f"--cnpro-ab-digits:{digits}" in sized,
              "a count of %d did not publish %d digit(s) for the parked "
              "column to size itself by: %r" % (count, digits, sized))
        check("px" not in sized,
              "the phase CSS carried a PIXEL width for the collage button "
              "(%r). The panel cannot know the theme's font metrics; the "
              "digit count is the whole contract" % (sized,))
    check(dna._phase_css("x", snap(), 0.0) == "",
          "an idle panel with no count yet still got phase CSS")

    # The session-side lifecycle the CSS keys off.
    session = dna._Session()
    session.start()
    session.say("duel 1: generating A", generating="A")
    check(session.snapshot()["generating"] == "A",
          "say(generating='A') did not mark the side being rendered")
    # EXACT REPORTED FLICKER: terminal progress, the host's finalization
    # reset, then terminal again. Once this render has started the middle 0
    # must be unable to move the latch backwards. A stale terminal value at
    # the START of the next side is the opposite case and must be ignored.
    check(session.track_progress(1.0) == 0.0,
          "a new A render inherited the previous render's stale 100%")
    check(abs(session.track_progress(0.42) - 0.42) < 1e-9
          and session.track_progress(1.0) == 1.0
          and session.track_progress(0.0) == 1.0
          and session.track_progress(1.0) == 1.0,
          "render progress did not stay monotone across the reported "
          "42% -> 100% -> 0% -> 100% sequence")
    session.say("duel 1: generating B", generating="B")
    check(session.snapshot()["progress"] == 0.0
          and session.track_progress(1.0) == 0.0
          and abs(session.track_progress(0.2) - 0.2) < 1e-9,
          "starting B did not reset A's completion latch, or accepted A's "
          "stale terminal counter as B's progress")
    session.render_complete("B")
    check(session.snapshot()["progress"] == 1.0,
          "a successfully returned render was not forced to terminal progress")
    session.publish(1, None, None, "waiting")
    published = session.snapshot()
    check(published["generating"] is None and published["awaiting"],
          "publish() must clear the active rendering side and set awaiting; "
          "the completed fill is now derived from the published images")
    completed = dna._phase_css("cnpro_ab_txt2img", published,
                               published["progress"])
    check("interesting_a{--cnpro-ab-fill:100%}" in completed
          and "interesting_b{--cnpro-ab-fill:100%}" in completed,
          "publishing the completed duel cleared its terminal fills: %r"
          % completed)
    session.say("rendering keeper 1")
    check(session.snapshot()["generating"] is None,
          "an unrelated say() kept a stale rendering side")
    session.start_demo("go")
    check(session.snapshot()["generating"] == "A",
          "a demo run renders into the A slot, and A's chrome must carry "
          "its progress")
    session.track_progress(0.6)
    session.render_complete("A")
    session.publish_demo("sample", "done")
    demo_done = session.snapshot()
    check(demo_done["generating"] is None
          and demo_done["progress"] == 1.0
          and "interesting_a{--cnpro-ab-fill:100%}" in dna._phase_css(
              "cnpro_ab_txt2img", demo_done, demo_done["progress"]),
          "a completed GOOD sample did not keep its A indicator at 100%")
    session.finish("done")
    check(session.snapshot()["generating"] is None,
          "finish() kept a stale rendering side")


def test_field_table(dna):
    """Every searchable field has to be a field the unit actually carries.

    Read out of the source rather than off the imported dataclass: importing
    external_code drags in the host, and the failure this guards against is a
    RENAME, which is visible in the text.
    """
    path = os.path.join(EXTENSION, "lib_cnpro", "external_code.py")
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except OSError:
        SKIPS.append("lib_cnpro/external_code.py was not readable, so the "
                     "field table could not be checked against the dataclass")
        return
    body = source.partition("class ControlNetUnit:")[2]
    declared = set(re.findall(r"^    ([a-z_0-9]+)\s*:", body, re.M))
    if not declared:
        SKIPS.append("no dataclass fields were found in external_code.py - the "
                     "field table check needs the ControlNetUnit declaration")
        return
    unknown = sorted(name for name in dna.UNIT_FIELDS if name not in declared)
    check(not unknown,
          "the searchable-field table names %d field(s) the ControlNetUnit "
          "dataclass does not have: %r. A row offering one of those looks "
          "exactly like a row that works - it just sets an attribute nothing "
          "reads, and produces a search over identical images"
          % (len(unknown), unknown))

    # The three deliberate exclusions, asserted so that adding them back is a
    # decision rather than an accident. See the table's own comment.
    for name, why in (
            ("weight", "the weight PROFILE overrides it, so a row on it is inert"),
            ("guidance_start", "the weight profile overrides it"),
            ("control_mode", "the UI no longer exposes it - the balance profile "
                             "replaced it"),
            ("image", "it is an image, not a value")):
        check(name not in dna.UNIT_FIELDS,
              "%r is searchable again, and it should not be: %s" % (name, why))


# ---------------------------------------------------------------------------

def main():
    sys.path.insert(0, EXTENSION)
    try:
        import numpy
    except Exception as exc:
        print("SKIPPED - the DNA search needs numpy (%s)" % exc)
        return 0

    install_host_stubs()
    from lib_cnpro import ab_search

    test_search(ab_search, numpy)
    test_similarity_metric(ab_search, numpy)
    test_capacity(ab_search, numpy)
    test_stochastic_tactics(ab_search, numpy)
    test_reuse_and_trend(ab_search, numpy)
    try:
        dna = load_script()
    except Exception as exc:
        fail("scripts/CNPro_AB.py did not import against the host stubs (%r). "
             "Everything below this point went unchecked." % (exc,))
        return report()
    test_recipe(dna, ab_search)
    test_point_gene(dna, ab_search)
    test_demo_session(dna)
    test_demo_accepts_edited_rows(dna, ab_search)
    test_collage_cache_policy(dna)
    test_collage_layout(dna)
    test_resume_matches_visible_rows(dna, ab_search)
    test_loop_controls(dna)
    test_set_good_and_reset_state(dna, ab_search)
    test_persistence(dna, ab_search, numpy)
    test_phase_css(dna)
    test_field_table(dna)
    return report()


def report():
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for failure in FAILURES:
            print("  -", failure)
        return 1
    for skip in SKIPS:
        print("PARTIAL SKIP -", skip)
    print("ok - the search converges (and converges the other way on reversed "
          "grades), a tie is an observation, and the printed recipe "
          "reconstructs the configuration it names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
