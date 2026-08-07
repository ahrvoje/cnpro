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
6. **The dislike row means what it says.** A grade on the second row ("and I
   dislike both") pushes both sides below the prior mean while the grade
   still orders them - with comparisons alone the posterior is translation-
   invariant and a bad region could not be marked, only ranked. A normal
   grade carries NO verdict about the pair, and a dislike arms the redirect:
   the next duel explores, and the streak clears on the first normal grade.
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
    while a state from different rows is refused rather than silently
    reindexed.
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

    # -- a dislike is an anchored statement, not a comparison --------------
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
          "a NORMAL tie moved the pair off par to (%.3f, %.3f) - the top "
          "row must carry no verdict about the pair, only the comparison, "
          "or every ordinary grade quietly marks a region"
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
            engine.observe(a, b, grade, disliked=disliked)
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
             prompt_mode=dna.PROMPT_REPLACE, weight_low=0.5, weight_high=1.5,
             weight_step=0.05),
        Gene(Gene.KIND_PROMPT, 3, values=["golden hour", "moonlight"],
             prompt_mode=dna.PROMPT_APPEND, weight_low=0.5, weight_high=1.5,
             weight_step=0.05),
        Gene(Gene.KIND_LORA, 4, loras=["add_detail", "film_grain"],
             weight_low=0.0, weight_high=1.0, weight_step=0.05),
        Gene(Gene.KIND_LORA, 5, loras=["sharpen"],
             weight_low=0.0, weight_high=1.0, weight_step=0.05),
    ], search)

    space = dna._space(genes, search)
    check(len(space) == 10,
          "six rows declared %d dimensions, not 10. A LoRA row and a Prompt "
          "row are TWO parameters each - which one, and how strongly - and "
          "the Setting and Profile rows are one each" % len(space))

    # model=depth, factor x1.5, prompt 0 at weight 1 (the one with the
    # separator in it), 'moonlight' at 1.2, film_grain at 0.6, sharpen off.
    point = (1, 2, 0, 1.0, 1, 1.2, 1, 0.6, 0, 0.0)

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
    args = [1] + [None] * (dna.MAX_ROWS * dna.ROW_ARGS) + [0, False, True, 0.05]
    args[1:1 + dna.ROW_ARGS] = ["Unit 0", dna.MODE_PROFILE, "Model", [],
                                "Main", "0.5:1.5:3", dna.PROMPT_REPLACE, "",
                                0.0, 1.0]
    rows = dna._read_rows(args, 1, 0.05)
    check(len(rows) == 1 and rows[0].kind == Gene.KIND_PROFILE,
          "a Profile row did not read back as one: %r" % (rows,))
    check([round(v, 4) for v in rows[0].values] == [0.5, 1.0, 1.5],
          "'0.5:1.5:3' expanded to %r" % (rows[0].values,))

    args[1:1 + dna.ROW_ARGS] = ["LoRA", dna.MODE_SETTING, "Model", [], "Main",
                                "", dna.PROMPT_REPLACE,
                                "<lora:add_detail:0.8>\nfilm_grain", 0.2, 0.9]
    rows = dna._read_rows(args, 1, 0.05)
    check(len(rows) == 1 and rows[0].loras == ["add_detail", "film_grain"],
          "a pasted LoRA tag did not reduce to its name: %r"
          % (rows[0].loras if rows else None,))
    check(rows and (rows[0].weight_low, rows[0].weight_high) == (0.2, 0.9),
          "the row's weight range was not read: %r"
          % ((rows[0].weight_low, rows[0].weight_high) if rows else None,))

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
             prompt_mode=dna.PROMPT_APPEND, weight_low=0.5, weight_high=1.5,
             weight_step=0.05),
    ], search)
    text = dna._compose_prompt(weighted, (0, 1.3), "BASE")
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
             weight_low=0.5, weight_high=1.5, weight_step=0.05),
    ], search)
    trace_line = dna._choices_line(long_prompt, (0, 1.0))
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
        "rows": [{"target": "LoRA", "mode": "Setting", "field": "Model",
                  "choices": [], "line": "Main",
                  "values": "a house | 100% </script> shed",
                  "prompt_mode": "Replace", "loras": "add_detail",
                  "weight_low": 0.2, "weight_high": 1.0}],
        "tail": {"vary_seed": False, "render_winner": True,
                 "weight_step": 0.4},
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
          "solver state replayed onto a DIFFERENT space - a changed label "
          "list silently reindexes every choice, which is worse than "
          "starting fresh")

    # -- the default weighted-channel grids -------------------------------
    check((dna.LORA_WEIGHT_MIN, dna.LORA_WEIGHT_MAX,
           dna.LORA_WEIGHT_STEP) == (0.2, 1.0, 0.4),
          "the default LoRA weight grid is %r, not the three states "
          "0.2 / 0.6 / 1.0. Coarse is the contract: three visible steps per "
          "LoRA keeps a five-LoRA search at 3^5 weight combinations instead "
          "of 21^5, and the 0.2 floor avoids the weight-0 degeneracy where "
          "every LoRA in a slot is the same image"
          % ((dna.LORA_WEIGHT_MIN, dna.LORA_WEIGHT_MAX,
              dna.LORA_WEIGHT_STEP),))
    check((dna.PROMPT_WEIGHT_MIN, dna.PROMPT_WEIGHT_MAX) == (0.4, 1.2),
          "the default prompt weight range is %r, not 0.4..1.2 - with the "
          "0.4 step that is the three states 0.4 / 0.8 / 1.2, straddling 1 "
          "because de-emphasis and emphasis both live near it"
          % ((dna.PROMPT_WEIGHT_MIN, dna.PROMPT_WEIGHT_MAX),))

    # -- the checkbox coercion --------------------------------------------
    check(dna._coerce("u0.enabled", "False") is False,
          "'False' was not coerced to a boolean. A Checkbox handed the STRING "
          "'False' renders as checked, because a non-empty string is truthy - "
          "so Set would enable a unit the recipe says to disable")
    check(dna._coerce("u0.processor_res", "1024") == 1024,
          "a numeric field did not come back as a number")


def test_demo_session(dna):
    """GOOD/BAD outlive the search - the session and staging mechanics.

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
    """
    import time as _time

    session = dna._Session()
    check(not session.request_demo(True),
          "a session with no loop alive accepted a GOOD request - the panel "
          "then never takes the idle path, and pressing the button while "
          "idle does nothing at all")

    session.start()
    check(session.request_demo(True) and session.request_demo(False),
          "a running session refused a GOOD/BAD press that did not land "
          "exactly while a duel was awaiting a grade - a press during a "
          "render is a request, not a mistimed click")
    check(session.await_grade(1) == "demo_good"
          and session.await_grade(1) == "demo_bad",
          "queued GOOD/BAD requests did not come back in arrival order")

    session.publish(1, None, None, "waiting")
    session.request_demo(True)
    session.grade(7)
    first = session.await_grade(1)
    check(isinstance(first, tuple) and first[0] == 7.0,
          "a grade and a queued demo raced and the demo won (%r) - the "
          "grade answers the question on screen and must be served first"
          % (first,))
    check(session.await_grade(1) == "demo_good",
          "the queued demo was lost once a grade overtook it")

    session.request_demo(False)
    session.start()
    check(not session.pending_demos,
          "start() carried a stale demo request into the new search - the "
          "first duel would be interrupted by a sample nobody asked this "
          "session for")
    session.finish("done")
    check(not session.request_demo(True),
          "a finished session still accepted a demo request onto its dead "
          "queue - nothing will ever serve it, and the idle path (which "
          "would) is never taken")

    script = dna.Script()
    staged = dna._PENDING_DEMO["txt2img"]
    staged[:] = [(True, _time.time() - dna.DEMO_STALE_SECONDS - 1),
                 (False, _time.time())]
    check(script._pop_demo_request() is False,
          "the staged queue served a stale entry (or lost the fresh one "
          "behind it) - a press that never became a run must not hijack a "
          "later Generate")
    check(script._pop_demo_request() is None and not staged,
          "the staged queue was not drained one-per-run")


def test_point_gene(dna, search):
    """The Profile-point row, at the gene level.

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
                    values=[-0.5, 0.0, 0.5], point_index=1)
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
    check("as drawn" in gene.describe((1,))
          and "+0.5" in gene.describe((2,)),
          "the trace line does not say which offset a duel chose: %r / %r"
          % (gene.describe((1,)), gene.describe((2,))))


def test_persistence(dna):
    """Nothing ends a search irrecoverably - the fix for "A/B and GOOD/BAD
    stop working" after an idle timeout, a STOP or an Interrupt.

    Three properties carry it:

    * start() KEEPS the retained solver state - clearing it there meant that
      starting a search and interrupting it before the first grade destroyed
      the PREVIOUS search's learned taste, and GOOD/BAD then reported "no
      solver state" about work that had cost dozens of duels;
    * the idle timeout is a PAUSE, not an end: it marks the session
      timed_out, which is what lets run() say "Generate resumes" instead of
      "stopped";
    * the state is mirrored to disk and read back, and the retained-state
      lookup prefers the explicit sources over the mirror: staged import
      first, then the session's own state, then the disk.
    """
    session = dna._Session()
    session.solver_state = {"observations": [[None, [0], 0.5]], "duels": 1}
    session.start()
    check(session.solver_state is not None,
          "start() cleared the retained solver state - interrupting a fresh "
          "search before its first grade then destroys the previous search's "
          "learned taste, and GOOD/BAD go dead with it")

    check(not session.timed_out,
          "a fresh session already reads as timed out")
    session.publish(1, None, None, "waiting")
    check(session.await_grade(0.001) is None and session.timed_out,
          "the idle timeout did not mark the session as PAUSED - run() then "
          "reports a stop, and nothing tells the user that Generate resumes")
    session.start()
    check(not session.timed_out,
          "start() carried a stale timed_out mark into the new search")

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
          and "--cnpro-ab-fill" not in waiting,
          "a duel awaiting its grade must fade the nudge in and carry no "
          "fill: %r" % (waiting,))
    clamped = dna._phase_css("x", snap(running=True, generating="A"), 2.0)
    check("100%" in clamped and "200%" not in clamped,
          "progress above 1 was not clamped: %r" % (clamped,))

    # The session-side lifecycle the CSS keys off.
    session = dna._Session()
    session.start()
    session.say("duel 1: generating A", generating="A")
    check(session.snapshot()["generating"] == "A",
          "say(generating='A') did not mark the side being rendered")
    session.publish(1, None, None, "waiting")
    published = session.snapshot()
    check(published["generating"] is None and published["awaiting"],
          "publish() must clear the rendering side and set awaiting - the "
          "fill would otherwise persist over the finished duel")
    session.say("rendering keeper 1")
    check(session.snapshot()["generating"] is None,
          "an unrelated say() kept a stale rendering side")
    session.start_demo("go")
    check(session.snapshot()["generating"] == "A",
          "a demo run renders into the A slot, and A's chrome must carry "
          "its progress")
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
    test_persistence(dna)
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
