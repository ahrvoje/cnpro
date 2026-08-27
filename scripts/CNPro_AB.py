"""CNPro A/B - a human-in-the-loop search for the configuration you want.

Drop this file into `<forge-neo>/scripts/` (or into `<this extension>/scripts/`)
and it appears in the Script dropdown as "CNPro A/B".

WHAT IT IS, AGAINST WHAT X/Y IS
-------------------------------
CNPro X/Y takes two axes and renders every cell. That is the right tool when
the answer fits on a contact sheet - twelve cells, one glance. It stops being a
tool at all around four axes: five models x six profile factors x four prompts x
a LoRA weight on a 0.05 grid is 10,080 images, which is not a grid, it is a
week.

DNA takes the SAME kinds of degree of freedom - as many rows as you care to add
- and never renders the space. It renders two configurations, asks which one is
closer to what you want, and uses the answer to decide what to ask next. The
model of your taste it builds along the way is what lets it recommend a
configuration NOBODY LOOKED AT: after a dozen answers it has usually worked out
which model, which factor and which LoRA weight each carried their own weight,
and the recommendation is the best combination of those, not the best image you
were shown. `lib_cnpro/ab_search.py` is that model and explains itself.

THE ROWS ARE THE DEGREES OF FREEDOM
-----------------------------------
One row is one thing that is allowed to vary. Add and remove them with the
buttons at the top - the row count is the shape of the search space, so it is
the first thing the panel asks. Four kinds:

* **Model** - the unit's model, and a LIST of candidates the search may pick
  from. Categorical, exactly as listed - DNA picks one of them, never
  something between. (Recipes and Set can still name every unit field -
  see UNIT_FIELDS - but a row only ever varies the model: the other fields
  proved to be noise dimensions in practice, and a chooser full of them
  buried the one that matters.)
* **Profile** - one of the unit's six profile lines, scaled by a factor from
  the row's list. See `lib_cnpro/profile_scale.py`: the factor multiplies the
  curve, so 1 is untouched, 0 switches the line off, 2 is twice the control.
* **Profile points** - a GROUP of drawn points of one of those lines, moved
  vertically TOGETHER by an offset from the row's list: 0 is the profile
  exactly as drawn, +0.1 is every listed point 0.1 higher in the plot's own
  axis units, negative moves them down - the same edit dragging each point
  in the editor would make. Point indices count drawn points left to right
  from 0, negative from the end (-1 = rightmost); listing several indices is
  what turns the offset into an edit of a whole profile INTERVAL rather than
  of one knot. Offsets are numbers, so the search knows they are ordered and
  evidence transfers between neighbouring values.
* **Prompt** - one prompt per line, one of which is picked, either REPLACING
  the prompt or being prepended/appended to it. Several Prompt rows compose
  (subject in one, lighting in another), which is why at most one of them may
  Replace.
* **LoRA** - one LoRA per line, of which the search picks ONE, and its weight
  is picked from the row's own weight list (interval notation welcome:
  `0.2:1:5`). The weights are numbers, so the search knows they are ordered
  and evidence transfers between neighbours - and pinning a LoRA to one
  guessed weight is how a good LoRA gets rejected, which is why the list is
  a row control and not a constant. Two LoRAs at once means two rows - a row
  is one LoRA slot, and its list is the candidates for that slot.
* **Canvas layer** - ONE layer of ONE canvas, at an opacity from the row's
  list (percent, as the layer row's own field shows it: 100 is opaque, 0 is
  the layer gone). The canvas is any of this tab's canvases that holds an
  image - a unit's Input tabs, and on img2img the host's own init-image
  canvas - and the layer is one of that canvas' layers. Nothing else about
  the canvas is varied; to vary two layers, add two rows. The stack of a
  canvas exists only in the browser (gradio receives the flattened
  composite), so the row's choices come from the page and every duel side
  asks the page for its composite - see THE CANVAS LAYER ROUND TRIP.

WHAT AN ANSWER IS
-----------------
Each round shows image A and image B and the SAME 0..10 scale THREE times.
One click answers, whichever row it lands on. The scale is the comparison: 0
means A is much better, 10 means B is much better, 5 means you cannot pick
between them - and 5 is a real answer, not a refusal: it says the two
configurations are worth the same, which is as informative as any other
grade.

The ROW is the verdict about the pair itself - the things a person feels
instantly when a pair appears and a comparison cannot carry, each asking for
no memory of any reference and no absolute number:

* **Distinct samples** - the normal answer. These two look rather different,
  and we are on track. "On track" is an absolute signal as well as prose:
  both samples are softly anchored above par, while the 0..10 grade orders
  them inside that usable region.
* **Similar samples** - ...rather ALIKE. This is the cheapest click in the
  panel and among the most informative: "do these look the same" is a
  first-glance impression that costs no deliberation, and because the top
  two rows PARTITION the on-track case, every graded duel yields a
  similarity verdict rather than only the ones somebody bothered to mark.
  That is what trains the SEPARATION METRIC (`lib_cnpro/ab_search.py`,
  `_fit_metric`) - the thing that decides whether a duel is worth asking at
  all, how different the STOP keepers have to be, and what an N-GOOD collage
  may hold together. It was a hand-written constant until this row existed;
  it now also calibrates the utility kernel, so grades transfer farther over
  visually inert rows and less over vivid ones. Like the first row, this one
  anchors both samples above par.
  It also pins the two configurations together in the utility model: two
  images that look the same are worth the same, which is a stronger claim
  than the grade beside it.
* **Bad samples** - I will rate them, but both are bad. The grade still
  steers (which side is LESS bad is information); what the row adds is that
  the whole region sinks below par and the search changes the subject - the
  next duel explores, and a streak of dislikes rolls the candidate pool back
  toward broad exploration. It carries no similarity verdict: both at once
  would need a fourth row, and a bad region is the cheapest place to lack
  that data.

Once in a while the search will show a pair that looks like the same image
twice. That is deliberate - it is re-checking a difference you called
invisible, because every duel is filtered by the learned metric and a
dimension wrongly written off would otherwise never get asked about again.
Grade the row you actually see.

GOOD and N-GOOD, parked beside START/STOP and RESUME, are not answers: each
renders fresh samples drawn from the good end of the solver's current
belief - varied per press, pure inference, no observation recorded - prints
their full recipes into the trace, and, during a search, returns to the same
duel. They are the "show me what you think I want" probes, and the images
join the gallery at stop.

    GOOD    one sample, from the good end - quality first.
    N-GOOD  a COLLAGE of N samples that are all good AND SPREAD ACROSS the
            good region, laid out as one horizontal strip. Spread, not merely
            "each different from its neighbour": the entries are chosen to
            cover the region rather than to rank inside it, and they clear
            the keeper's distance from each other rather than the duel's -
            see ab_search.population, which returned N views of the
            champion until both of those were true.

N IS ON THE BUTTON - "3 GOOD" - and it is the solver's own count of how many
mutually distinguishable good configurations it can actually hand over
(`ab_search.capacity`). Recomputed after every graded duel and written into
the LABEL, so the button always says what pressing it will do: it starts at
one before a solver has spoken, becomes "0 GOOD" (disabled) when the evidence
supports no above-par population, and grows as the good region is mapped.
Capped at COLLAGE_MAX. Press again for another collage: each press enters the good
region at a fresh draw from the posterior and covers it from there, so two
presses are two samples of one taste rather than the same list twice - until
the region is exhausted, at which point the same sheet IS the answer.

IT USED TO BE AN EDITABLE BOX beside the button, and the box is gone because
the number stopped being an estimate. It was one: a Monte Carlo reading of
the good region's size that could report forty while the button delivered
six, and choosing what to spend on that region was the user's call. The box
and the button now run the SAME selection to different limits (see
ab_search.capacity), so the box could only ever say what the button was
about to do - asking for more returns the same sheet, and the only edit it
still supported was asking for fewer of the few answers there are. A button
that states its own count is the whole control, and the width the box was
using went back to the scale.

During a live search N-GOOD reads the duel render cache: an island peak already
shown costs a lookup rather than another generation. Fresh collage members do
NOT enter that 32-image cache, or one large collage would evict every duel
image and destroy the reuse economy after the press. Reuse without pollution
is deliberate; the generated configurations remain subject to the same
quality and pairwise-distinctness tests either way.

THE BUTTONS DO NOT DIE WITH THE SEARCH, and that is the point of them: a
finished search is a reached state, not a spent one. With no search running
- after STOP, or right after an Import staged a solver - a GOOD/N-GOOD press
stages its request and clicks Generate client-side (the same route START
and RESUME take - see HOW THE SEARCH IS STARTED AND STOPPED), and run()
serves it by rendering those samples from the
retained solver state instead of opening duels: nothing is asked, nothing
is observed, the state is left exactly as it was. The rows are read fresh too:
editing them after the search reinterprets the learned coordinates on their
current values, with added coordinates starting at their first value and
out-of-range choices clamped. Prompt, canvases, resolution and seed are likewise
read fresh from the UI by that generation, so the learned taste can be applied
under freely retuned settings for as long as it is useful. The search is how
the generator is TRAINED; the buttons are how it is USED.

The panel's Session file accordion holds Export and Import: the whole session
- rows, unit settings, canvases, prompt AND negative prompt, the sampler
settings (sampler, scheduler, steps, CFG, size, denoising), seed, and the
solver's learned state - as ONE deterministic HTML file. The file renders in a browser as readable
tables and images, so a version mismatch still leaves a page a human can
reproduce by hand; imported, it reconfigures everything and stages the solver
state, which RESUME then continues onto the same rows.

Under each image sits an "✦ interesting" TOGGLE, and it is deliberately not a
grade: it says "overall this sample is whatever I graded it - usually bad -
but it touches a characteristic I want to see in good samples". The mark adds
no utility whatsoever; it donates the configuration's coordinates to hybrid
candidates (the characteristic, transplanted into the current attractors),
which then earn their way through ordinary duels or vanish. A LoRA/prompt pick
and its conditional weight transplant as one semantic gene, never as two
unrelated coordinates. Toggled marks are
sent with the grade, so the natural gesture is: mark ✦ on the tempting side,
then grade the pair on the dislike row.

HOW THE SEARCH IS STARTED AND STOPPED
-------------------------------------
TWO BUTTONS, and the host's Generate is not one of them.

    START/STOP  one button wearing the state it is in. Idle it reads START
                and begins a FRESH search - a new record, a new model, the
                retained state left untouched. Alive it reads STOP and ends
                the loop.
    RESUME      under it, and always on screen. It continues from the
                retained solver state - the last search on these rows, the
                disk mirror that survived a restart, or a state an Import
                staged - and it is ENABLED only while nothing is running and
                that state exactly matches the visible rows. Starting a search
                greys it out again.

GENERATE STAYS THE HOST'S OWN BUTTON. Selecting this script does not turn
Generate into "start a search" any more: a plain press renders the current
setup exactly as it would with no script selected - `run()` returns None and
the host generates normally. That is what makes the panel usable AS a panel,
because a search is a long-lived thing to be inside of and the prompt, the
canvases and the sampler settings still want trying out while you are in one.

The three buttons that DO start work still go THROUGH Generate, because only
the Generate event reads every component the generation pipeline needs: a
press stages its intent (see _PENDING_START and _PENDING_DEMO) and clicks the
host's button client-side, and `run()` serves the staged intent instead of a
plain render. There is no second path into the pipeline, and a gradio handler
cannot invoke one.

ONE SEED FOR THE WHOLE SEARCH. Both sides of every duel render on the seed in
the host's own box, which is what makes a duel a comparison of two
CONFIGURATIONS rather than of two noise fields - and it is also what makes
the render cache worth having, since a side already shown can be shown again
for a lookup instead of a generation. There used to be a "new seed each duel"
switch and it is gone: it traded a noisier comparison for nothing the panel
could show, and it disabled the cache outright (a fresh seed is a fresh
recipe, so no side could ever repeat).

THE SEARCH HAS NO LENGTH. It asks until STOP, because how many comparisons an
answer needs is not knowable in advance - it depends on how many degrees of
freedom were declared and on how consistently they are graded - and a number
guessed up front would either cut the search off mid-way or ask for answers
after it had stopped learning anything. The recommendation is recomputed after
EVERY answer and printed into the box below the moment it changes, so stopping
is never premature: whatever has been learned is already there.

STOP prints that recommendation into the box at the bottom - and prints the
whole FRONTIER into the trace: up to four keeper configurations, each polished,
each visibly different from the others, best first (★1, ★2, ...), each a full
recipe that can be copied out and pasted back. Taste is not unimodal, and the
search deliberately holds every strong basin it finds rather than the single
one that was ahead at STOP. Nothing is rendered at stop: the keepers are
recipes, and rendering them is one paste-and-Generate away when it is wanted -
a stop used to trigger up to four unasked generations, which made STOP the
one button that cost more GPU time than it saved. Set applies a recipe
string to the whole
Forge Neo configuration - unit fields, prompt and negative prompt, LoRAs,
sampler settings, seed. It is plain text
and can be saved, pasted back into that box, or handed to somebody else, which
is the point of it being a string rather than a button that only works in this
session.

SET GOOD samples one configuration from the learned good posterior and applies
it without generating an image. RESET restores every writable setting to the
exact values captured when START was pressed. That snapshot travels with the
retained solver state, so stop/reload/restart and Export/Import do not change
what Reset means. Both actions write through the same visible-component map as
Set; neither touches a unit State wholesale.

HOW IT RUNS, AND WHY THAT IS UNUSUAL
------------------------------------
A script's `run()` is one blocking call, so an interactive loop inside it has
no way to reach the page. This one publishes each duel to a module-level
session and BLOCKS on a condition variable; a gr.Timer in the panel polls that
session a few times a second and paints the images, the status and the result
box; the grade buttons write the score back and wake the loop. Nothing else in
the host is touched - the queue runs those handlers concurrently with the
generation exactly as it runs the Interrupt button.

The panel therefore stays live while the job is "running", which is what makes
the loop possible at all. Two consequences worth knowing:

* Interrupt (the host's own button) ends the search wherever it is, and the
  recommendation from the answers so far is still printed.
* NOTHING ELSE ENDS IT. A duel waits for its grade for as long as it takes -
  there is no idle timeout, and there is deliberately no setting for one. A
  pair on screen is a question that has already cost two generations, and a
  clock that expires it only ever means walking away from the desk costs the
  answer. STOP and Interrupt are the two ways out, and both are things the
  user meant to do.

NOTHING ENDS A SEARCH IRRECOVERABLY. Every graded duel refreshes the retained
solver state - on the session, and mirrored to a small JSON file that
survives a UI restart - and RESUME picks it up on the same rows. So a STOP,
an Interrupt, a page reload or a restarted server cost nothing but the
interruption itself: press RESUME and the search continues where it left
off, or press GOOD/N-GOOD and sample from whatever it had learned. State
learned on DIFFERENT rows never resumes - the signature check refuses it -
so changing the rows disables RESUME; START is the explicit fresh-search
action. The reloaded page finds its way back too: the poll timer runs once at
page load and repaints whatever the server-side session is doing, then
switches itself off if that is nothing.

A RESUMED SEARCH KEEPS ITS RECORD. Tried, the recommendation and the summary
survive a stop-and-continue exactly as the learned state does - they describe
the very duels the resumed observations came from, so clearing them would
leave a record that starts mid-session. START opens a new record; RESUME
never does.

THE CANVAS LAYER ROUND TRIP
---------------------------
A Canvas layer row is the one kind whose value python cannot apply by
itself: a canvas' layers, their blend modes, the levels, the edges tool and
the crop all live in javascript/canvas_extra.js, and gradio only ever holds
the flattened result. Re-compositing here would be a second copy of that
pipeline, which drifts - so the page does it, through three hidden
components of the panel (javascript/cnpro_ab.js is the other half):

    inventory   the page lists this tab's canvases that hold an image and
                each one's layers with their current opacity, once a second
                and only when it changes; the row's canvas and layer
                dropdowns are built from it, and Reset remembers the
                opacity a search started from out of it.
    request     before a side renders, run() writes the composites it needs
    reply       - canvas, layer, opacity, for every Canvas layer row - into
                the tick's request channel and BLOCKS; the page renders each
                composite OFF-SCREEN (the live canvas does not change) and
                writes the PNGs back; run() puts them where the generation
                reads its image from - the unit's Input slot, or
                `p.init_images` for the img2img canvas - and generates.
    set         Set, Set GOOD and Reset hand a layer opacity to the page,
                which writes it into the live canvas' layer, the one place
                that holds it.

The composite for a given opacity is part of the recipe (the
`<canvas>.layer<n>.opacity=` token), so the render cache still keys on the
whole generation, and a request is only ever made for a side that is about
to be generated. The wait has a deadline, unlike a grade's: the page
answers a request by itself, so silence means the page is gone or the
script did not load, and the search stops saying so.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
Nothing is written back to the UI while searching. Every duel gets a shallow
COPY of the unit dataclass in a copy of `p.script_args`; the gr.State the unit
panel is bound to is never touched, so an abandoned search leaves the panel
exactly as it was. Set, Set GOOD and Reset are the ONLY paths that write to the
UI, each runs only when pressed, and each writes to the visible components
rather than to the unit State - the same route an infotext paste takes.
"""

import base64
import copy
import html
import io
import json
import logging
import os
import random
import re
import sys
import tempfile
import threading
import time
import types

import gradio as gr

import modules.scripts as scripts
from modules import errors, processing, script_callbacks
from modules.processing import Processed, process_images
from modules.shared import state
from modules.ui_components import ToolButton

logger = logging.getLogger("CNPro_AB")

WHO = "CNPro A/B"

REFRESH_SYMBOL = "\U0001f504"
DICE_SYMBOL = "\U0001f3b2"
ADD_SYMBOL = "➕"
REMOVE_SYMBOL = "➖"

#: Rows built up front. Gradio 4.40 cannot create components after the page is
#: built, so "dynamic row count" is a fixed stack of rows whose VISIBILITY the
#: +/- buttons move.
#:
#: Fifteen, sized off the search somebody actually wants to run rather than off
#: a round number: three profile points on each of five ControlNet units is a
#: reasonable thing to try, and at ten rows it could not be expressed at all.
#: Every extra degree of freedom still costs comparisons, so a fifteen-row
#: search is a long one - but that is the user's trade to make, and the ceiling
#: is here to stop the component stack from growing without bound, not to
#: ration ambition.
MAX_ROWS = 15

#: Grades on offer. 0..10 with 5 meaning "cannot pick" - see the docstring.
GRADES = 11

TARGET_PROMPT = "Prompt"
TARGET_LORA = "LoRA"
TARGET_CANVAS = "Canvas layer"
GLOBAL_TARGETS = [TARGET_PROMPT, TARGET_LORA, TARGET_CANVAS]

#: A canvas of this tab, named by WHERE IT LIVES - `u<unit>.in<slot>` for a
#: unit's Input tab (slot 1-based, as the tabs are labelled), `img2img` for
#: the host's own init-image canvas. Minted by javascript/cnpro_ab.js
#: (keyOf) from the canvas' block id; read here for the row's label, the
#: recipe token and the place the composite goes. A uuid would not do: it is
#: new on every page load, and the row's choice has to outlive one.
RE_CANVAS_KEY = re.compile(r"^(?:u(\d+)\.in(\d+)|img2img)$")

#: The recipe token of one layer's opacity: `<canvas>.layer<n>.opacity`,
#: n 1-based as the layer list labels it (L1, L2, ...). One expression,
#: read by Set and written by the recipe, so the two cannot disagree.
RE_LAYER_TOKEN = re.compile(r"^(u\d+\.in\d+|img2img)\.layer(\d+)\.opacity$")

#: How long a duel side waits for the page to hand back a composite. A
#: grade waits forever because a person answers it; this is answered by the
#: page itself within a second, so silence means the page is closed or the
#: script is not loaded, and a search that waits on it forever looks like
#: a search that hung.
CANVAS_WAIT_SECONDS = 60

MODE_MODEL = "Model"
MODE_PROFILE = "Profile"
MODE_POINTS = "Profile points"
MODES = [MODE_MODEL, MODE_PROFILE, MODE_POINTS]

PROMPT_REPLACE = "Replace"
PROMPT_PREPEND = "Prepend"
PROMPT_APPEND = "Append"
PROMPT_MODES = [PROMPT_REPLACE, PROMPT_PREPEND, PROMPT_APPEND]

#: Default LoRA weight list: three states, 0.2 / 0.6 / 1.0. Used when a
#: weighted row's weights box is left empty.
#:
#: COARSE ON PURPOSE. Three visible steps per LoRA keeps a five-LoRA search
#: at 3^5 weight combinations instead of 21^5 - matched to a human's duel
#: budget, and each step is large enough to SEE, which is what makes a duel
#: answerable at all. The floor is 0.2 rather than 0 because 0 is a
#: degenerate identity: every LoRA in a slot at weight 0 is the same image,
#: while the model would hold them as different points and scatter evidence
#: across them. "This LoRA is not helping" is still discoverable - it is the
#: search preferring the bottom of the list - and a recommendation at 0.2
#: is cheap to try by hand at 0. The row's weights box takes any list, with
#: interval notation (`0.2:1:5`) for the finer grids; these are the
#: DEFAULTS, not the limits.
LORA_WEIGHTS = [0.2, 0.6, 1.0]

#: Default weight list for a PROMPT row, written with A1111's own `(text:w)`
#: syntax. Straddling 1 rather than starting at 0, because the two knobs
#: mean different things: a LoRA at 0 is simply absent, while a prompt term
#: at 0.6 is still in the prompt and still steering - de-emphasis and
#: emphasis are both useful and both live near 1. Three states, mirroring
#: the LoRA grid: three visible steps per row is what keeps a many-row
#: search inside a human's duel budget.
PROMPT_WEIGHTS = [0.4, 0.8, 1.2]

#: Characters that would end the `(text:w)` wrapper early, and so have to be
#: escaped inside it. A1111 reads a backslash before either as a literal.
RE_PROMPT_ESCAPE = re.compile(r"([()])")

#: How often the panel polls the session, in seconds. Fast enough that grading
#: does not feel laggy, slow enough to be invisible next to a generation.
POLL_SECONDS = 0.6

#: Lines kept in the trace box. Ten are shown; the rest scroll. A long search
#: is a few hundred lines of text, which is nothing - the cap is there so that
#: "run it overnight" cannot grow without bound.
TRACE_LIMIT = 2000

#: Rendered results kept for reuse, per search. The recipe string is the
#: cache key - it names the WHOLE generation, prompt and seed included, so a
#: hit is bit-identical by construction. Repeats are real: the champion
#: re-fights new challengers by design (that is what anchors the utility
#: scale), and the keepers rendered on stop are usually images a duel
#: already produced. The cap bounds memory, not correctness - images are a
#: few MB each and a miss only costs the generation it always cost.
RENDER_CACHE = 32

#: Recognised at the START of a pasted configuration and then ignored. NOT
#: written any more: it was on every line of the trace and at the front of
#: every recipe, and nothing ever read it - the tokens are self-describing and
#: the parser can tell a configuration from a sentence by whether it has any.
#: Still accepted so that a string saved before it went away still pastes.
CONFIG_HEADER = "DNA1"

#: Version of the exported session file. Bumped when the payload SHAPE
#: changes; import still applies whatever fields it recognises from other
#: versions and names what it could not, because the file's whole reason for
#: being HTML is that a human can finish the job by hand.
#: v2: rows carry "weights" (a list as text) and "points" (index list as
#: text) instead of v1's weight_low/weight_high/point and the panel-global
#: weight step; the "field" key and the render_winner/weight_step tail
#: entries are gone.
#: v3: rows carry "canvas" and "layer" (the Canvas layer row's choices).
EXPORT_VERSION = 3

RE_MODEL_HASH = re.compile(r"\s*\[[0-9a-fA-F]{6,}\]\s*$")
RE_UNIT_LABEL = re.compile(r"^Unit\s+(\d+)")
RE_LORA_TAG = re.compile(r"^\s*<\s*lora\s*:\s*([^:>]+?)\s*(?::[^>]*)?>\s*$", re.I)


# ---------------------------------------------------------------------------
# Reaching CNPro
#
# This file can live in the webui's own scripts/ directory, where the
# extension's basedir is NOT on sys.path. Every CNPro import is therefore
# deferred to call time, by which point cnpro.py has been imported and its
# modules are in sys.modules - which is what `import` consults first. Absence
# is handled rather than assumed away: with the extension disabled this script
# has to say so, not raise ImportError while the page is being built.
# ---------------------------------------------------------------------------

def _global_state():
    """CNPro's model and preprocessor registry, or None."""
    try:
        from lib_cnpro import global_state
    except Exception:
        return None
    return global_state


def _external_code():
    """CNPro's unit dataclass and profile grammar, or None."""
    try:
        from lib_cnpro import external_code
    except Exception:
        return None
    return external_code


def _profile_scale():
    """The profile-scaling arithmetic shared with CNPro X/Y, or None."""
    try:
        from lib_cnpro import profile_scale
    except Exception:
        return None
    return profile_scale


def _search():
    """The preference-search engine, or None.

    Separate from the CNPro imports above because it fails for a DIFFERENT
    reason - it needs numpy, not the extension - and the message the user
    needs to read is different in each case.
    """
    try:
        from lib_cnpro import ab_search
    except Exception as exc:
        logger.error(f"{WHO}: the search engine did not import ({exc}).")
        return None
    return ab_search


def _ui_groups(is_img2img):
    """The unit UI groups of THIS tab, in unit order (== script_args order)."""
    try:
        from lib_cnpro.controlnet_ui.controlnet_ui_group import ControlNetUiGroup
    except Exception:
        return []
    return [g for g in ControlNetUiGroup.all_ui_groups
            if bool(g.is_img2img) == bool(is_img2img)]


def _cnpro_script(p):
    """The running CNPro script instance, for its args_from/args_to window."""
    for script in getattr(getattr(p, "scripts", None), "alwayson_scripts", None) or []:
        try:
            if (script.title() or "").strip().lower() == "cnpro":
                return script
        except Exception:
            continue
    return None


def _networks():
    """The host's LoRA registry, or None.

    Found in sys.modules rather than imported: it lives in
    `extensions-builtin/sd_forge_lora/networks.py`, whose directory is only on
    sys.path while that extension is loading, and whose module name
    ("networks") is generic enough that importing it blind could pick up
    something else. The two attributes are the identity test.
    """
    module = sys.modules.get("networks")
    if module is not None and hasattr(module, "available_networks"):
        return module
    for module in list(sys.modules.values()):
        if hasattr(module, "available_networks") \
                and hasattr(module, "list_available_networks"):
            return module
    return None


def _lora_names():
    networks = _networks()
    if networks is None:
        return []
    return sorted(networks.available_networks, key=str.lower)


# ---------------------------------------------------------------------------
# The unit fields a recipe can name
#
# The vocabulary of `u<N>.<field>=` tokens: _set_targets builds Set, Export
# and Import from this table, and _coerce types values on the way back in.
# The ROWS no longer offer these - a row varies the unit's MODEL and nothing
# else (the "Model" mode); the chooser that offered every field here buried
# the one that matters under two dozen that were noise dimensions in every
# real search. The table stays curated all the same, because a recipe that
# names a field this build cannot place has to be reported, not invented:
#
# * IMAGES AND MASKS - they are not values, they are megabytes, and a search
#   over them is a search over things the user would have to paint first.
# * weight / guidance_start / guidance_end - the weight PROFILE overrides all
#   three, and every unit built by the editor has one.
# * control_mode - the UI no longer exposes it (the balance profile did), so
#   setting it would move a field nothing reads.
# ---------------------------------------------------------------------------

#: The shape of a field's value - descriptive now that rows no longer offer
#: these fields: CHOICE comes from a known list, the rest are free text or a
#: number. Kept on the table because a human reading a recipe token needs to
#: know what kind of value it takes.
KIND_CHOICE = "choice"
KIND_TEXT = "text"
KIND_NUMBER = "number"

BOOL_CHOICES = ["True", "False"]


class _Field:
    def __init__(self, label, kind, choices=None, cast=None):
        self.label = label
        self.kind = kind
        self._choices = choices
        self.cast = cast

    def choices(self, control_type):
        if self._choices is None:
            return []
        if callable(self._choices):
            return self._choices(control_type)
        return list(self._choices)


def _models_for_type(control_type):
    """The model list a unit's own dropdown is showing under `control_type`.

    Mirrors global_state.select_control_type's model half rather than reading
    the dropdown component: the choices the user sees were pushed there by a
    gr.update and never landed back on the Python object. The Control Type
    value IS live (it arrives as a gradio input), so this is exact.
    """
    global_state = _global_state()
    if global_state is None:
        return ["None"]
    pattern = str(control_type or "All").lower()
    if pattern == "all":
        return list(global_state.controlnet_names)
    filtered = global_state.get_filtered_controlnet_names(control_type)
    if pattern == "none" and "None" not in filtered:
        filtered = filtered + ["None"]
    return filtered or ["None"]


def _modules_for_type(control_type):
    global_state = _global_state()
    if global_state is None:
        return ["None"]
    if str(control_type or "All").lower() == "all":
        return global_state.get_all_preprocessor_names()
    return global_state.get_filtered_preprocessor_names(control_type) or ["None"]


def _enum_values(name, fallback):
    """The `.value`s of one of CNPro's enums, or a fallback with CNPro absent.

    The fallback is what keeps the field dropdown populated on a page built
    before (or without) the extension, so the row still says what it would
    offer instead of showing an empty list that reads as "nothing to vary".
    """
    external_code = _external_code()
    enum = getattr(external_code, name, None) if external_code else None
    if enum is None:
        return list(fallback)
    try:
        return [e.value for e in enum]
    except Exception:
        return list(fallback)


UNIT_FIELDS = {
    "model": _Field("Model", KIND_CHOICE, _models_for_type),
    "module": _Field("Preprocessor", KIND_CHOICE, _modules_for_type),
    "weight_profile": _Field("Weight profile", KIND_TEXT),
    "balance_profile": _Field("Balance profile", KIND_TEXT),
    "enabled": _Field("Enabled", KIND_CHOICE, BOOL_CHOICES,
                      cast=lambda v: str(v) == "True"),
    "pixel_perfect": _Field("Pixel perfect", KIND_CHOICE, BOOL_CHOICES,
                            cast=lambda v: str(v) == "True"),
    "resize_mode": _Field("Resize mode", KIND_CHOICE,
                          lambda _t: _enum_values("ResizeMode", [
                              "Just Resize", "Crop and Resize", "Resize and Fill"])),
    "hr_option": _Field("Hires-Fix option", KIND_CHOICE,
                        lambda _t: _enum_values("HiResFixOption", [
                            "Both", "Low res only", "High res only"])),
    "processor_res": _Field("Preprocessor resolution", KIND_NUMBER, cast=int),
    "threshold_a": _Field("Threshold A", KIND_NUMBER, cast=float),
    "threshold_b": _Field("Threshold B", KIND_NUMBER, cast=float),
    "unit_prompt": _Field("Unit prompt", KIND_TEXT),
    "unit_negative_prompt": _Field("Unit negative prompt", KIND_TEXT),
    "unit_prompt_emb_strength": _Field("Unit prompt strength", KIND_NUMBER,
                                       cast=float),
    "unit_prompt_delta_scale": _Field("Unit prompt delta", KIND_NUMBER,
                                      cast=float),
    "unit_negative_prompt_emb_strength": _Field(
        "Unit negative strength", KIND_NUMBER, cast=float),
    "unit_negative_prompt_delta_scale": _Field(
        "Unit negative delta", KIND_NUMBER, cast=float),
    "unit_prompt_retention": _Field("Prompt retention", KIND_NUMBER, cast=float),
    "input_order": _Field("Input order", KIND_TEXT),
    "image_enabled": _Field("Use input 1", KIND_CHOICE, BOOL_CHOICES,
                            cast=lambda v: str(v) == "True"),
    "image_2_enabled": _Field("Use input 2", KIND_CHOICE, BOOL_CHOICES,
                              cast=lambda v: str(v) == "True"),
    "image_3_enabled": _Field("Use input 3", KIND_CHOICE, BOOL_CHOICES,
                              cast=lambda v: str(v) == "True"),
    "image_4_enabled": _Field("Use input 4", KIND_CHOICE, BOOL_CHOICES,
                              cast=lambda v: str(v) == "True"),
    "image_5_enabled": _Field("Use input 5", KIND_CHOICE, BOOL_CHOICES,
                              cast=lambda v: str(v) == "True"),
}

RE_INPUT_ENABLED = re.compile(r"^image_(\d+)_enabled$")


def _field_component(group, field_name):
    """The gradio component a unit field is edited through, or None.

    The per-input mute checkboxes are the one field family that is not an
    attribute of the group - they are a list, one per input slot - so they are
    resolved here rather than everywhere that writes a field.
    """
    if field_name == "image_enabled":
        checks = getattr(group, "input_enabled_checks", None) or []
        return checks[0] if checks else None
    match = RE_INPUT_ENABLED.match(field_name)
    if match:
        checks = getattr(group, "input_enabled_checks", None) or []
        slot = int(match.group(1)) - 1
        return checks[slot] if 0 <= slot < len(checks) else None
    return getattr(group, field_name, None)


# ---------------------------------------------------------------------------
# Rows -> genes
# ---------------------------------------------------------------------------

def _unit_label(index):
    return f"Unit {index}"


def _unit_index(label):
    match = RE_UNIT_LABEL.match(str(label or ""))
    return int(match.group(1)) if match else None


def _target_choices(flags):
    """What a row can vary: the activated units, then Prompt and LoRA.

    The unit half comes from the enable checkboxes' own values, NOT from the
    unit gr.States. `enabled` reaches a State through CNPro's own field_updater,
    which is a SEPARATE handler on the very same checkbox event - so a handler
    that reads the State sees the value from before the click and the unit list
    runs permanently one step behind. The checkbox value arrives with the event
    that fired it and cannot be stale.
    """
    return [_unit_label(i) for i, on in enumerate(flags) if on] + GLOBAL_TARGETS


def _canvas_unit_slot(key):
    """(unit index, input slot 1-based) of a unit canvas key, or None for
    the host's img2img canvas."""
    match = RE_CANVAS_KEY.match(str(key or ""))
    if match is None:
        raise ValueError(f"{WHO}: {key!r} is not a canvas key.")
    if match.group(1) is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _canvas_label(key):
    """What the row's dropdown shows for a canvas key."""
    where = _canvas_unit_slot(key)
    if where is None:
        return "img2img image"
    return f"Unit {where[0]} · Input {where[1]}"


def _layer_index(text):
    """The 0-based layer index behind a layer dropdown value ("L2" -> 1)."""
    match = re.match(r"^L(\d+)", str(text or "").strip())
    return int(match.group(1)) - 1 if match else None


def _parse_inventory(text):
    """The page's canvas inventory: [{key, layers: [{size, opacity}]}].

    Only well-formed entries survive: the text crossed a browser boundary
    and is read by dropdown builders and the run's validation, neither of
    which should raise over a malformed line.
    """
    try:
        entries = json.loads(text) if text else []
    except (TypeError, ValueError):
        return []
    inventory = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) \
                or not RE_CANVAS_KEY.match(str(entry.get("key", ""))):
            continue
        layers = []
        for layer in entry.get("layers") or []:
            if not isinstance(layer, dict):
                continue
            try:
                opacity = int(layer.get("opacity", 100))
            except (TypeError, ValueError):
                opacity = 100
            layers.append({"size": str(layer.get("size", "")),
                           "opacity": min(max(opacity, 0), 100)})
        if layers:
            inventory.append({"key": entry["key"], "layers": layers})
    return inventory


#: The page's last inventory per tab, as _parse_inventory returns it. Read
#: by the row's dropdown builders, by _setup to validate a Canvas layer row
#: before a duel is spent, and by run() for the opacity Reset restores.
_CANVAS_INVENTORY = {"txt2img": [], "img2img": []}


def _canvas_choices(inventory):
    return [(_canvas_label(entry["key"]), entry["key"]) for entry in inventory]


def _layer_choices(inventory, key):
    for entry in inventory:
        if entry["key"] == key:
            return [f"L{index + 1} · {layer['size']}"
                    for index, layer in enumerate(entry["layers"])]
    return []


def _channel_html(message):
    """A python -> page message as the body of a hidden gr.HTML.

    Escaped into a <pre>, which the page reads back as textContent: the
    JSON never becomes markup, and an HTML update is a DOM mutation the
    page's observer sees at once - a textbox write would be neither (see
    javascript/cnpro_ab.js).
    """
    if not message:
        return ""
    return ("<pre class='cnpro-ab-channel-body'>"
            + html.escape(json.dumps(message, sort_keys=True)) + "</pre>")


def _layer_message(seq, values):
    """{key: {layer index: alpha}} out of `token: percent` pairs, wrapped
    for the page. The page reads alpha 0..1 (the layer's own unit); the
    recipe writes percent (the layer row's)."""
    layers = {}
    for token, value in values.items():
        match = RE_LAYER_TOKEN.match(str(token))
        if match is None:
            continue
        try:
            alpha = min(max(float(value), 0.0), 100.0) / 100.0
        except (TypeError, ValueError):
            continue
        layers.setdefault(match.group(1), {})[
            str(int(match.group(2)) - 1)] = alpha
    return {"seq": seq, "layers": layers}


def _short_model(name):
    """Model name without its ` [hash]` suffix - trace summaries are narrow."""
    return RE_MODEL_HASH.sub("", str(name or "")).strip() or "None"


def _lines(text):
    """One value per non-blank line, in the order they were typed."""
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


#: Longest prompt shown in a trace line or a status summary. A prompt can be
#: kilobytes; the trace is a record of WHICH choice was made, and 64
#: characters identify one at a glance. The whole of it is in the recipe.
CHOICE_PROMPT_CHARS = 64


def _elide(text, width=40):
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[:width - 1] + "…"


def _choices_line(genes, point):
    """What this configuration CHOSE, one short phrase per row.

    Not the resolved configuration: that is the recipe, it carries whole
    weight-profile strings and the fully composed prompt, and at a few
    kilobytes per line a trace of it is unreadable and unbounded. What is
    useful while a search runs is which value each row landed on.
    """
    return " · ".join(gene.describe(point) for gene in genes) or "(no choices)"


def _join_prompt(head, tail):
    """Two prompt fragments, joined the way a prompt is actually written.

    With ', ', not with a space: a prompt is a comma-separated list of terms,
    and gluing 'golden hour' onto 'a house on a hill' with a space makes ONE
    five-word term rather than two. The seam is normalized rather than trusted
    - both boxes are typed by hand and 'trailing comma' and 'no trailing comma'
    are equally natural spellings, so a comma already there is absorbed instead
    of doubled. An empty side yields the other side untouched.
    """
    head = str(head or "").rstrip().rstrip(",").rstrip()
    tail = str(tail or "").lstrip().lstrip(",").lstrip()
    if not head:
        return tail
    if not tail:
        return head
    return f"{head}, {tail}"


class Gene:
    """One row, resolved: the dimensions it declares and what a value means.

    A gene owns a SLICE of the search point - one entry for most kinds, two for
    LoRA (which LoRA, and at what weight) - so that adding a kind with more
    parameters later does not touch the loop. `take` is the only place that
    knows how wide its own slice is.
    """

    KIND_SETTING = "setting"
    KIND_PROFILE = "profile"
    KIND_POINT = "point"
    KIND_PROMPT = "prompt"
    KIND_LORA = "lora"
    KIND_LAYER = "layer"

    def __init__(self, kind, row, unit_index=None, field=None, values=None,
                 line=None, point_indices=None, prompt_mode=PROMPT_REPLACE,
                 loras=None, weights=None, canvas=None, layer=None):
        self.kind = kind
        self.row = row
        self.unit_index = unit_index
        self.field = field
        self.values = list(values or [])
        self.line = line
        # A Canvas layer row: which canvas (a key, see RE_CANVAS_KEY) and
        # which of its layers (0-based, bottom first, as the page holds
        # them); `values` is then the opacity list, in percent.
        self.canvas = canvas
        self.layer = layer
        # The GROUP of drawn points a Profile-points row moves together -
        # one offset, several knots, which is what edits an interval of the
        # curve instead of one point of it.
        self.point_indices = [int(i) for i in (point_indices or [0])]
        self.prompt_mode = prompt_mode
        self.loras = list(loras or [])
        self.weights = [float(w) for w in (weights or LORA_WEIGHTS)]
        self.offset = 0        # where this gene's slice starts, set by _space

    @property
    def on_unit(self):
        return self.kind in (Gene.KIND_SETTING, Gene.KIND_PROFILE,
                             Gene.KIND_POINT)

    @property
    def width(self):
        # Two for the kinds that pick a thing AND how strongly to apply it.
        return 2 if self.kind in (Gene.KIND_LORA, Gene.KIND_PROMPT) else 1

    @property
    def name(self):
        if self.kind == Gene.KIND_SETTING:
            return f"U{self.unit_index} {UNIT_FIELDS[self.field].label}"
        if self.kind == Gene.KIND_PROFILE:
            return f"U{self.unit_index} {self.line}"
        if self.kind == Gene.KIND_POINT:
            indices = ",".join(str(i) for i in self.point_indices)
            return f"U{self.unit_index} {self.line}[{indices}]"
        if self.kind == Gene.KIND_PROMPT:
            return f"Prompt ({self.prompt_mode.lower()})"
        if self.kind == Gene.KIND_LAYER:
            return f"{_canvas_label(self.canvas)} L{self.layer + 1}"
        return "LoRA"

    @property
    def layer_token(self):
        """The recipe token a Canvas layer row writes - see RE_LAYER_TOKEN."""
        return f"{self.canvas}.layer{self.layer + 1}.opacity"

    def override(self, point):
        """(canvas key, layer index, alpha 0..1) - what the page composites."""
        return self.canvas, self.layer, float(self.chosen(point)) / 100.0

    def dimensions(self, ab_search):
        """The search dimensions this row declares.

        A LoRA row and a Prompt row each declare TWO: which one of the listed
        candidates, and how strongly to apply it. The strength is the reason
        both are here - pinning a LoRA to a weight you guessed is how a good
        LoRA gets rejected, and the same is true of a prompt term nobody
        thought to emphasise.
        """
        if self.kind in (Gene.KIND_LORA, Gene.KIND_PROMPT):
            picks = self.loras if self.kind == Gene.KIND_LORA else self.values
            pick = ab_search.Dimension.choice(f"{self.name} pick", picks)
            # The row's own weight LIST, as an ordered choice: the labels are
            # numbers, so Dimension.choice keeps their metric and evidence
            # still transfers between neighbouring weights - the search just
            # never lands between the values the user listed, which is what a
            # typed list means.
            weight = ab_search.Dimension.choice(
                f"{self.name} weight", [f"{w:g}" for w in self.weights])
            # The weight belongs to the pick: 0.6 of one LoRA says nothing
            # about 0.6 of another, and the kernel is told so - each
            # candidate learns its own weight curve from its own duels.
            weight.parent = pick
            return [pick, weight]
        # Numeric labels for the numeric kinds: Dimension.choice keeps their
        # metric, so evidence transfers between neighbouring factors,
        # offsets and opacities.
        labels = ([f"{v:g}" for v in self.values]
                  if self.kind in (Gene.KIND_PROFILE, Gene.KIND_POINT,
                                   Gene.KIND_LAYER)
                  else self.values)
        return [ab_search.Dimension.choice(self.name, labels)]

    def take(self, point):
        """This gene's own values out of a whole search point."""
        return point[self.offset:self.offset + self.width]

    # -- what a value means ---------------------------------------------

    def chosen(self, point):
        """The gene's value in human terms, for labels and the config string.

        The two-dimensional kinds return a (what, how strongly) pair; the rest
        return the value itself.
        """
        slice_ = self.take(point)
        index = int(slice_[0])
        if self.kind in (Gene.KIND_LORA, Gene.KIND_PROMPT):
            # The weight entry is an INDEX into the row's weight list, not
            # the weight itself - the dimension is a choice now.
            windex = int(slice_[1])
            weight = (self.weights[windex]
                      if 0 <= windex < len(self.weights) else self.weights[0])
            if self.kind == Gene.KIND_LORA:
                name = self.loras[index] if 0 <= index < len(self.loras) else ""
                return name, weight
            text = self.values[index] if 0 <= index < len(self.values) else ""
            return text, weight
        if self.kind == Gene.KIND_PROFILE:
            return self.values[index] if 0 <= index < len(self.values) else 1.0
        if self.kind == Gene.KIND_POINT:
            # The neutral offset: the profile exactly as drawn.
            return self.values[index] if 0 <= index < len(self.values) else 0.0
        if self.kind == Gene.KIND_LAYER:
            # Opaque: the layer as it composes with no row at all.
            return self.values[index] if 0 <= index < len(self.values) else 100.0
        return self.values[index] if 0 <= index < len(self.values) else ""

    def describe(self, point):
        """This row's choice, short enough to sit in a line with the others."""
        if self.kind in (Gene.KIND_LORA, Gene.KIND_PROMPT):
            what, weight = self.chosen(point)
            what = (what if self.kind == Gene.KIND_LORA
                    else _elide(what, CHOICE_PROMPT_CHARS))
            return f"{self.name}: {what} @ {weight:g}"
        value = self.chosen(point)
        if self.kind == Gene.KIND_PROFILE:
            return f"{self.name} x{value:g}"
        if self.kind == Gene.KIND_POINT:
            # "+0.1", "-0.2" ... and a bare "0" for the neutral offset rather
            # than "as drawn". The trace is a column of these, one per row per
            # image, and the words cost more width than they carry: 0 in a
            # column of signed offsets already reads as "the profile
            # untouched", and the box's own header says what 0 means.
            return f"{self.name} {value:+g}" if value else f"{self.name} 0"
        if self.kind == Gene.KIND_LAYER:
            return f"{self.name} {value:g}%"
        if self.kind == Gene.KIND_SETTING and self.field == "model":
            return f"{self.name} = {_short_model(value)}"
        return f"{self.name} = {_elide(value)}"

    def apply_unit(self, unit, point):
        if self.kind == Gene.KIND_SETTING:
            cast = UNIT_FIELDS[self.field].cast
            value = self.chosen(point)
            setattr(unit, self.field, cast(value) if cast else value)
        elif self.kind == Gene.KIND_PROFILE:
            module = _profile_scale()
            if module is not None:
                unit.weight_profile = module.scale_profile(
                    unit, self.unit_index, self.line, self.chosen(point), who=WHO)
        elif self.kind == Gene.KIND_POINT:
            module = _profile_scale()
            if module is not None:
                # The listed points move as a GROUP: one offset, applied to
                # each in turn. Sequential rewrites are safe because moving a
                # point vertically never changes the point count, so the
                # remaining indices keep naming what they named.
                for index in self.point_indices:
                    unit.weight_profile = module.offset_profile_point(
                        unit, self.unit_index, self.line, index,
                        self.chosen(point), who=WHO)

    def prompt_fragment(self, point):
        """What this gene contributes to the prompt, or None."""
        if self.kind == Gene.KIND_PROMPT:
            text, weight = self.chosen(point)
            if not text:
                return None
            if abs(weight - 1.0) < 1e-6:
                # Exactly 1 is no emphasis at all, and `(text:1)` is a
                # different STRING from `text` even though it is the same
                # prompt - it would show up in every recipe and every
                # infotext as noise, and it changes how the text tokenizes
                # for anyone who reads it back.
                return text
            # Parentheses inside the text would close this wrapper early and
            # turn the rest of the prompt into a weight expression, so they
            # are escaped the way A1111 itself escapes them. Weights the user
            # already wrote INSIDE the text keep working: nesting multiplies,
            # which is the syntax's own rule and the useful behaviour here.
            escaped = RE_PROMPT_ESCAPE.sub(r"\\\1", text)
            return f"({escaped}:{weight:g})"
        if self.kind == Gene.KIND_LORA:
            name, weight = self.chosen(point)
            # A weight that rounds to zero is the LoRA not being there, and
            # `<lora:x:0>` is not the same thing: the host still loads the
            # network, which costs time and can still shift the result through
            # its text-encoder side. Emitting nothing is the honest encoding of
            # "this slot is off", and it is what makes 0 a usable end of the
            # range rather than a decorative one.
            if not name or abs(weight) < 1e-6:
                return None
            return f"<lora:{name}:{weight:g}>"
        return None


def _build_genes(rows, ab_search):
    """The rows that declare something, as genes, with their slices assigned.

    A row that names NOTHING is dropped. A row that names exactly one value is
    kept, and the difference matters: one value is not a search, it is a
    constant, and applying it is useful (pin the model, then search the rest).
    The search knows it cannot vary - `Space.live` skips it - so it costs no
    comparisons, and the count of degrees of freedom logged at the start is
    what tells the user which of their rows are actually being searched.
    """
    genes, offset = [], 0
    replace_rows = []
    for gene in rows:
        if gene is None:
            continue
        if gene.kind == Gene.KIND_LORA:
            if not gene.loras:
                continue
        elif len(gene.values) < 1:
            continue
        if gene.kind == Gene.KIND_PROMPT and gene.prompt_mode == PROMPT_REPLACE:
            replace_rows.append(gene.row + 1)
        gene.offset = offset
        offset += gene.width
        genes.append(gene)

    if len(replace_rows) > 1:
        # Two Replace rows means one of them never reaches the model, and the
        # search would spend comparisons on a dimension that does nothing -
        # while the recommendation names a prompt that never ran. Refused up
        # front, because it cannot be seen in the images.
        raise RuntimeError(
            f"{WHO}: rows {', '.join(str(r) for r in replace_rows)} are all "
            f"Prompt rows set to {PROMPT_REPLACE}, so only one of them could "
            f"ever reach the model and the others would be searched over for "
            f"nothing. Set all but one to {PROMPT_PREPEND} or {PROMPT_APPEND}.")
    return genes


def _space(genes, ab_search):
    dimensions = []
    for gene in genes:
        dimensions.extend(gene.dimensions(ab_search))
    return ab_search.Space(dimensions)


# ---------------------------------------------------------------------------
# The configuration string
#
# `u0.model=... | u0.weight_profile=... | prompt=... | neg_prompt=... |
#  steps=... | sampling=... | scheduler=... | cfg_scale=... | width=... |
#  height=... | seed=...`
#
# Readable, one line, and complete: what it names is what Set applies, so a
# string that came out of somebody else's session sets the same thing here. It
# lists RESOLVED values rather than "row 3 chose option 2" for exactly that
# reason - the rows are not part of it and do not have to be recreated.
#
# The prompt is emitted already composed (prompt genes applied, LoRA tags
# appended), which is also what makes Set idempotent: pressing it twice cannot
# append the same LoRA twice, because there is no append, only a value.
#
# Only `|`, `%` and newlines are escaped. Commas are left alone even though the
# string goes into the infotext, because the host's own `quote()` wraps any
# value containing one in JSON - so escaping them here would make every prompt
# unreadable to buy nothing.
# ---------------------------------------------------------------------------

def _escape(value):
    return (str(value).replace("%", "%25")
            .replace("|", "%7C")
            .replace("\r", "")
            .replace("\n", "%0A"))


def _unescape(value):
    return (str(value).replace("%0A", "\n")
            .replace("%7C", "|")
            .replace("%25", "%"))


def _config_string(genes, point, base_units, base_prompt, seed=None,
                   extras=None):
    """The one-line recipe for `point`.

    `extras` is the generation-settings half of the recipe's identity - the
    negative prompt, sampler, steps, CFG, size - built once per run by
    _generation_extras and emitted between the prompt and the seed.
    """
    units = {}
    for gene in genes:
        if not gene.on_unit:
            continue
        index = gene.unit_index
        if index not in units:
            unit = base_units.get(index)
            units[index] = copy.copy(unit) if unit is not None else None
        if units[index] is not None:
            gene.apply_unit(units[index], point)

    tokens = []
    for index in sorted(units):
        unit = units[index]
        if unit is None:
            continue
        # Enabled is emitted for every unit the search touched, even when no
        # row varies it: a recipe that sets a disabled unit's model and leaves
        # it disabled is a recipe that does nothing, and the person pasting it
        # cannot see which units it expects to be on.
        tokens.append(f"u{index}.enabled={_escape(bool(getattr(unit, 'enabled', True)))}")
        fields = sorted({gene.field for gene in genes
                         if gene.on_unit and gene.unit_index == index
                         and gene.kind == Gene.KIND_SETTING})
        if any(gene.kind in (Gene.KIND_PROFILE, Gene.KIND_POINT)
               and gene.unit_index == index for gene in genes):
            fields.append("weight_profile")
        for field in dict.fromkeys(f for f in fields if f != "enabled"):
            tokens.append(f"u{index}.{field}={_escape(getattr(unit, field, ''))}")

    # A layer's opacity is what the page composited the canvas at for this
    # image: without it the recipe reproduces the canvas as it is on screen,
    # which is exactly the configuration the row exists to search away from.
    for gene in genes:
        if gene.kind == Gene.KIND_LAYER:
            tokens.append(f"{gene.layer_token}={gene.chosen(point):g}")

    # ALWAYS, even when the search never touched them. A recipe that omits the
    # prompt because no row varied it is a recipe that reproduces nothing: the
    # person pasting it has some other prompt in the box, and one that omits
    # the seed generates on a fresh noise field - which looks exactly like the
    # search having recommended something that does not work. Both were
    # reported - and so, later, was the NEGATIVE prompt, which steers every
    # image exactly as the positive one does and rode in `extras` from then
    # on together with the sampler settings. The rule is now simple enough
    # to state: what the recipe names is the whole of what was generated.
    tokens.append(f"prompt={_escape(_compose_prompt(genes, point, base_prompt))}")
    for token, value in (extras or {}).items():
        tokens.append(f"{token}={_escape(value)}")
    if seed is not None:
        tokens.append(f"seed={int(seed)}")
    return " | ".join(tokens)


def _generation_extras(p):
    """The host generation settings a recipe has to name - see the rule in
    _config_string: what the recipe names is the whole of what was
    generated.

    The negative prompt is the reason this exists (its omission was reported
    exactly the way the prompt's own omission once was); the sampler,
    scheduler, steps, CFG, size and - when the pipeline carries one - the
    denoising strength decide the rest of the image that the rows do not.
    Read from `p`, what this generation ACTUALLY runs with (styles already
    applied), rather than from the UI components. Token names match the
    host's elem_id suffixes so the very same names serve Set, Export and
    Import through _set_targets - one vocabulary, four verbs.

    An attribute this host build does not have is OMITTED rather than
    guessed: an absent token reads as "no such knob here", which the
    parsers already handle, and a guessed value would be a lie the recipe
    then reproduces.
    """
    extras = {"neg_prompt": str(getattr(p, "negative_prompt", "") or "")}
    for token, attribute, cast in (
            ("steps", "steps", lambda v: int(float(v))),
            ("sampling", "sampler_name", str),
            ("scheduler", "scheduler", str),
            ("cfg_scale", "cfg_scale", lambda v: f"{float(v):g}"),
            ("width", "width", lambda v: int(float(v))),
            ("height", "height", lambda v: int(float(v))),
            ("denoising_strength", "denoising_strength",
             lambda v: f"{float(v):g}")):
        value = getattr(p, attribute, None)
        if value is None:
            continue
        try:
            extras[token] = cast(value)
        except (TypeError, ValueError):
            continue
    return extras


def _parse_config_string(text):
    """`{token: value}` from a configuration string, or {} if it is not one.

    Unknown tokens are KEPT rather than dropped, so that a string written by a
    later version does not silently lose the half this version does understand;
    the caller decides what it can apply and says what it could not.
    """
    text = str(text or "").strip()
    if not text:
        return {}
    parts = [part.strip() for part in text.split("|")]
    if parts and parts[0].upper().startswith("DNA"):
        parts = parts[1:]
    values = {}
    for part in parts:
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            continue
        values[key.strip()] = _unescape(value.strip())
    return values


def _settings_string(settings, order=None):
    """A complete settings snapshot in the same grammar Set already reads.

    Search recommendations intentionally name only the fields required to
    reproduce one sampled point. The initial snapshot has a different job:
    Reset must put *every* writable setting back, including fields no search
    row varied. It therefore serializes the whole `_set_targets` map while
    keeping that map's UI order when one is supplied.
    """
    settings = settings if isinstance(settings, dict) else {}
    keys = []
    for token in order or settings:
        if token in settings and token not in keys:
            keys.append(token)
    keys.extend(token for token in settings if token not in keys)
    return " | ".join(
        f"{token}={_escape(_jsonable(settings[token]))}" for token in keys)


def _configuration_for_point(genes, point, initial_settings):
    """The exact initial settings with one search `point` applied.

    Set GOOD is not a generation, so it has no `p` object or live unit
    dataclass to copy. The initial snapshot already contains every component
    Set can write. Rebuild only the small unit-shaped surface Gene.apply_unit
    needs, apply the sampled coordinates, then fold those changed fields back
    into a copy of the snapshot. Untouched settings consequently return to
    what the search actually started from instead of inheriting whatever was
    changed after it.
    """
    initial_settings = (initial_settings
                        if isinstance(initial_settings, dict) else {})
    settings = copy.deepcopy(initial_settings)
    units = {}
    changed = {}
    for gene in genes:
        if not gene.on_unit:
            continue
        index = gene.unit_index
        if index not in units:
            prefix = f"u{index}."
            fields = {
                token[len(prefix):]: _coerce(token, value)
                for token, value in settings.items()
                if token.startswith(prefix)
            }
            # Main-profile arithmetic consults the legacy scalars only when
            # the profile string is empty. Current CNPro units always carry a
            # profile, but explicit fallbacks keep old saved sessions
            # readable instead of turning Set GOOD into an AttributeError.
            fields.setdefault("weight_profile", "")
            fields.setdefault("weight", 1.0)
            fields.setdefault("guidance_start", 0.0)
            fields.setdefault("guidance_end", 1.0)
            fields.setdefault("enabled", True)
            units[index] = types.SimpleNamespace(**fields)
            changed[index] = set()
        gene.apply_unit(units[index], point)
        changed[index].add(
            "weight_profile"
            if gene.kind in (Gene.KIND_PROFILE, Gene.KIND_POINT)
            else gene.field)

    for index, fields in changed.items():
        unit = units[index]
        for field in fields:
            settings[f"u{index}.{field}"] = _jsonable(getattr(unit, field))

    for gene in genes:
        if gene.kind == Gene.KIND_LAYER:
            settings[gene.layer_token] = float(gene.chosen(point))

    if any(gene.kind in (Gene.KIND_PROMPT, Gene.KIND_LORA) for gene in genes):
        settings["prompt"] = _compose_prompt(
            genes, point, str(initial_settings.get("prompt", "") or ""))
    return settings


def _sample_good_configuration(payload, genes, initial_settings, serial=0):
    """One posterior sample as a complete settings mapping, or an error.

    The retained payload is replayed exactly like idle GOOD, including its
    deliberate projection onto edited rows. No render is submitted and no
    observation is added; only `suggest(good=True)` is consumed.
    """
    if not isinstance(payload, dict) or not payload.get("observations"):
        return None, "no solver state to sample from"
    ab_search = _search()
    if ab_search is None:
        return None, "the search engine is unavailable"
    try:
        space = _space(genes, ab_search)
        seed = int(payload.get("seed", 0)) + 104729 * max(int(serial), 0)
        search = ab_search.PreferenceSearch(space, seed=seed)
        restored = _restore_solver(
            search, payload, space, allow_row_changes=True)
        if restored is None or restored < 1:
            return None, "the retained solver has no observations to sample"
        point = search.suggest(good=True)
        return _configuration_for_point(
            genes, point, initial_settings), None
    except Exception as exc:
        return None, str(exc)


def _compose_prompt(genes, point, base_prompt):
    """The prompt a configuration actually generates with.

    Order is fixed and is the whole reason several rows can touch the prompt at
    once: REPLACE first (there is at most one - see _build_genes), then the
    prepends and appends in row order, then the LoRA tags. So "subject" in one
    row and "lighting" in another compose instead of racing, and a LoRA row
    lands on whatever prompt the other rows produced.
    """
    prompt = base_prompt
    for gene in genes:
        if gene.kind == Gene.KIND_PROMPT and gene.prompt_mode == PROMPT_REPLACE:
            prompt = gene.prompt_fragment(point) or ""
    for gene in genes:
        if gene.kind != Gene.KIND_PROMPT or gene.prompt_mode == PROMPT_REPLACE:
            continue
        fragment = gene.prompt_fragment(point)
        if not fragment:
            continue
        prompt = (_join_prompt(fragment, prompt)
                  if gene.prompt_mode == PROMPT_PREPEND
                  else _join_prompt(prompt, fragment))
    for gene in genes:
        if gene.kind != Gene.KIND_LORA:
            continue
        fragment = gene.prompt_fragment(point)
        if fragment:
            # APPENDED, never joined with a comma: a LoRA tag is not a prompt
            # term, and `, <lora:x:1>` puts an empty term in the prompt.
            prompt = f"{prompt.rstrip()} {fragment}".strip()
    return prompt


# ---------------------------------------------------------------------------
# Export / import - the whole session as ONE deterministic HTML file
#
# The file is three things at once: a JSON island this script reads back
# verbatim, a page a browser renders into readable settings tables and canvas
# images (so a failed import of a newer/older file still leaves the user able
# to reproduce everything by hand), and a record whose content depends only
# on the state it names - same state, byte-identical file.
# ---------------------------------------------------------------------------

RE_EXPORT_ISLAND = re.compile(
    r'<script type="application/json" id="cnpro-ab-state">(.*?)</script>',
    re.S)

#: Solver state staged by Import, consumed by the next run() on that tab.
#: Module-level because Import runs in a gradio handler and run() in the
#: generation thread - the session object is the only other thing they
#: share, and this must survive a session that has not started yet.
_PENDING_SOLVER = {}

#: GOOD/N-GOOD presses made while NO search is running, per tab: a list of
#: (_Demo, staged_at) pairs. The press stages its request here and
#: then clicks Generate itself (see _wire_duel); the run() that click
#: starts consumes ONE entry and renders that sample instead of starting
#: duels. One per run because one press queues one generation - a second
#: press while the first still waits in the host's queue stages its own
#: entry and its own click, and the pairing stays exact.
_PENDING_DEMO = {"txt2img": [], "img2img": []}

#: START and RESUME presses, per tab: a list of (resume_flag, staged_at).
#: The same stage-then-click-Generate route the demo presses take, and for
#: the same reason - a gradio handler cannot invoke the host's generation
#: pipeline, and only the Generate event reads every component that pipeline
#: needs. THIS LIST IS ALSO WHAT MAKES A PLAIN GENERATE PLAIN: run() starts
#: a search only when it finds an entry here, and returns None otherwise, so
#: the host renders the current setup exactly as it would with no script
#: selected.
_PENDING_START = {"txt2img": [], "img2img": []}

#: A staged press older than this is dropped with a log line rather than
#: served. The click that staged it fires Generate immediately, so a
#: surviving entry means that click never turned into a run (page reloaded,
#: script switched away) - and a stale entry would otherwise turn some
#: LATER, unrelated Generate press into a sample render, or into a search,
#: that the user did not ask for.
DEMO_STALE_SECONDS = 30 * 60

#: Idle presses counted per tab, forever. Two jobs: each press writes a
#: DISTINCT token into the hidden trigger box (a repeated value fires no
#: change event, so the second press would never click Generate), and each
#: demo run seeds its solver replay with it - the replayed engine is
#: deterministic, so without a varying seed every press of GOOD would
#: render the SAME "varied" sample.
_DEMO_SERIAL = {"txt2img": 0, "img2img": 0}

#: START/RESUME presses counted per tab, forever, for the first of those two
#: jobs only: the trigger box fires no change event on a repeated value, so a
#: second press has to write something the first did not. Its own counter
#: rather than the demo one because the demo counter is ALSO a seed - see
#: _demo_run - and a START must not move the sample a later GOOD renders.
_START_SERIAL = {"txt2img": 0, "img2img": 0}

#: Set-GOOD presses counted per tab. Replaying a retained solver is
#: deterministic, so every press needs a different seed or the button would
#: keep setting the same supposedly sampled configuration.
_SET_GOOD_SERIAL = {"txt2img": 0, "img2img": 0}

#: PRESSES one demo run will serve at most, however many landed while it was
#: working. Presses beyond the cap are dropped: a runaway "leaned on the
#: button" burst should cost a bounded number of generations, and pressing
#: again is one click. What each press costs is its own business: one sample
#: for GOOD, up to COLLAGE_MAX for N-GOOD.
DEMO_BATCH_MAX = 8

#: Samples ONE N-GOOD press will render, whatever count the button carries.
#: The solver's count is honest about how many distinct answers it holds and
#: a mapped region can offer a great many - which is a number worth KNOWING
#: and not a number worth rendering unasked. Sixty-four is a very wide
#: collage and roughly half an hour of GPU time; a request beyond it is
#: clamped, so pressing again is how more is asked for rather than the first
#: press quietly becoming an overnight job.
COLLAGE_MAX = 64

#: What the button offers before any solver has spoken. One, because a panel
#: that opens offering to spend eight generations on a taste it has not
#: learned yet is offering the wrong thing.
CAPACITY_DEFAULT = 1

#: Padding between collage cells, in pixels, and the colour behind them. The
#: gap is what makes the strip read as N separate samples rather than as one
#: wide image - the whole point of a collage is comparing its cells.
COLLAGE_GAP = 8
COLLAGE_BACKGROUND = (24, 24, 27)


class _Demo:
    """A GOOD press: an answer that answers nothing.

    Queued on the session (or staged for an idle Generate) and served by the
    loop, which renders it and returns to the very same duel. `count` is
    None for the single-sample GOOD button and an integer for N-GOOD, whose
    samples are composed into one collage image.

    An OBJECT rather than a flag, because the two requests differ in what
    they cost and in what they produce, and because `await_grade` returns it
    down the same channel a grade comes back on - where a tuple is a grade,
    so this has to be neither a tuple nor None.
    """

    def __init__(self, count=None):
        self.count = None if count is None else max(int(count), 1)

    @property
    def label(self):
        return "GOOD" if self.count is None else f"{self.count}-GOOD"


def _collage_count(text):
    """The count a press carries, as a number of samples to render.

    Clamped rather than validated, and that is the right severity here: the
    press has already happened by the time this runs, and the alternatives
    are a traceback in the generation thread or a press that silently does
    nothing. The value has crossed a session boundary - an imported file, a
    disk mirror written by an older build - so it is not trusted even though
    the panel is the only thing that writes it. An unreadable one falls back
    to the default, which costs one generation and is obviously wrong on
    screen.
    """
    try:
        count = int(float(str(text).strip()))
    except (TypeError, ValueError):
        return CAPACITY_DEFAULT
    return max(min(count, COLLAGE_MAX), 1)


def _collage_promise(report):
    """The claim the collage makes, as one clause, from
    `ab_search.population_report`.

    A PROBABILITY THE SHEET CAN BE CHECKED AGAINST. The entries came from a
    selection that enforces it, so this is not a boast - it is the number
    the next glance at the images either confirms or refutes, which is the
    only form in which "distinctly different samples" is worth saying at
    all. It also says how many separate good regions the entries came from,
    because that is what decides whether a short sheet is a disappointment
    or the answer: three entries from three islands is the good region
    having three answers in it.
    """
    if not report:
        return ""
    islands = int(report.get("islands") or 0)
    total = int(report.get("total_islands") or 0)
    confidence = float(report.get("confidence") or 0.0)
    if not islands:
        return ""
    where = f" from {islands} isolated good region{'' if islands == 1 else 's'}"
    if total > islands:
        where += f" of {total}"
    return f"{where}, every pair {confidence:.0%}+ likely to look different"


def _collage(images):
    """`images` as ONE horizontal strip, or None if there are none.

    A collage rather than N gallery entries because the question it answers
    is comparative - "what does the whole good region look like" - and that
    is not a question anybody answers by clicking through a gallery. The
    Every image occupies one cell in the same row, and every cell is the size
    of the largest image. Mixed resolutions therefore stay aligned without
    changing the strip into a second row.

    The sheet is what the gallery gets, alone: each sample was already
    written to the output folder by the generation that made it, and
    sixty-four gallery entries is a gallery nobody can use.
    """
    images = [image for image in images if image is not None]
    if not images:
        return None
    try:
        from PIL import Image
    except Exception as exc:
        logger.warning(f"{WHO}: could not compose the collage ({exc}) - the "
                       f"samples are in the gallery individually.")
        return None
    columns = len(images)
    rows = 1
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    sheet = Image.new(
        "RGB",
        (columns * cell_w + (columns + 1) * COLLAGE_GAP,
         rows * cell_h + (rows + 1) * COLLAGE_GAP),
        COLLAGE_BACKGROUND)
    for index, image in enumerate(images):
        column, row = index % columns, index // columns
        # Centred in its cell: an image smaller than the largest would
        # otherwise hang off one corner and break the strip the eye is
        # reading the comparison across.
        x = COLLAGE_GAP + column * (cell_w + COLLAGE_GAP) \
            + (cell_w - image.width) // 2
        y = COLLAGE_GAP + row * (cell_h + COLLAGE_GAP) \
            + (cell_h - image.height) // 2
        sheet.paste(image.convert("RGB"), (x, y))
    return sheet


def _save_collage(p, image, seed, info):
    """Write the collage next to the host's own grids.

    Best-effort: the individual samples were already saved by the
    generations that made them, so a failure here costs the sheet and
    nothing else - and it is worth attempting, because a collage that
    exists only in the gallery is gone the moment the next job runs.
    """
    if image is None:
        return
    try:
        from modules import images as host_images
        host_images.save_image(image, p.outpath_grids, "cnpro_ab_collage",
                               seed=seed, prompt=p.prompt, info=info, p=p,
                               grid=True)
    except Exception as exc:
        logger.warning(f"{WHO}: the collage could not be saved to disk "
                       f"({exc}) - it is still in the gallery.")


def _solver_file(tab):
    """Where this tab's solver state is mirrored on disk.

    The mirror exists because everything else holding the learned state is a
    Python object: a UI restart (or a crash) used to be the one ending no
    button could recover from, while a session file the user happened to
    Export could. The mirror is that export's solver half, written without
    being asked - same directory the Export button already writes to.
    """
    return os.path.join(tempfile.gettempdir(), f"cnpro_ab_{tab}_solver.json")


def _store_solver(tab, payload):
    """Mirror a solver payload to disk. Failure costs the mirror, never the
    search - a full disk must not turn a grade into an error."""
    path = _solver_file(tab)
    try:
        with open(path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(path + ".tmp", path)
    except OSError as exc:
        logger.warning(f"{WHO}: could not mirror the solver state to "
                       f"{path} ({exc}) - a UI restart will lose it.")


def _stored_solver(tab):
    """The disk mirror's payload, or None when there is none worth having."""
    try:
        with open(_solver_file(tab), encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if isinstance(payload, dict) and payload.get("observations"):
        return payload
    return None


def _retained_solver(tab):
    """The freshest solver state this tab can sample from or resume.

    Staged import first - it is the one source the user pointed at
    explicitly - then the last search's own state, then the disk mirror,
    which is what makes GOOD/N-GOOD and Resume outlive a restart.
    """
    session = _SESSIONS.get(tab)
    return (_PENDING_SOLVER.get(tab)
            or (session.solver_state if session is not None else None)
            or _stored_solver(tab))


def _solver_matches_rows(state, row_values, ab_search=None):
    """Whether `state` can resume the exact search rows now on screen.

    A retained solver may still be useful to GOOD/N-GOOD after the rows are
    edited: that path deliberately projects its coordinates. RESUME cannot do
    that. It promises to continue the same search, so both the button state and
    its click handler use this one predicate before making that promise.
    """
    if not isinstance(state, dict) or not state.get("observations"):
        return False
    ab_search = ab_search or _search()
    if ab_search is None:
        return False
    try:
        values = list(row_values)
        count = int(values[0] or 1)
        genes = _build_genes(_read_rows(values, count), ab_search)
        if not genes:
            return False
        signature = _space_signature(_space(genes, ab_search))
    except Exception:
        # Row edits pass through short-lived incomplete states. Eligibility is
        # a diagnostic and must not break the panel while one is on screen.
        return False
    return state.get("signature") == signature


def _image_to_data_uri(value):
    """A canvas/mask component's value as a PNG data URI, or None.

    Canvas channels arrive as PIL images, numpy arrays, editor dicts or
    nothing, depending on which widget and which gradio path produced them -
    so this accepts all of those rather than assuming one. None means "no
    image here", which the export simply omits: an omitted canvas imports
    as a CLEARED canvas, which is what reproduction means.
    """
    try:
        if value is None:
            return None
        if isinstance(value, str):
            return value if value.startswith("data:image") else None
        if isinstance(value, dict):
            value = value.get("background") or value.get("composite") \
                or value.get("image")
            if value is None:
                return None
        if not hasattr(value, "save"):
            import numpy as np
            if isinstance(value, np.ndarray):
                from PIL import Image
                value = Image.fromarray(value)
            else:
                return None
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return ("data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii"))
    except Exception as exc:
        logger.warning(f"{WHO}: could not serialize a canvas image ({exc}).")
        return None


def _data_uri_to_image(text):
    """The PIL image behind a data URI, or None."""
    try:
        _header, _sep, body = str(text or "").partition("base64,")
        if not body:
            return None
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(body)))
    except Exception as exc:
        logger.warning(f"{WHO}: could not read a canvas image ({exc}).")
        return None


def _input_field(slot):
    """The unit field holding Input `slot` (1-based, as the canvas key
    names it): `image` for the first tab, `image_<slot>` after - the
    dataclass' own naming (external_code.ControlNetUnit.input_images)."""
    return "image" if slot == 1 else f"image_{slot}"


def _layer_initial_settings(genes, tab):
    """`token: percent` of every Canvas layer row's layer, as the page
    last listed it - the opacity a search starts from, for Reset."""
    inventory = {entry["key"]: entry for entry in _CANVAS_INVENTORY[tab]}
    settings = {}
    for gene in genes:
        if gene.kind != Gene.KIND_LAYER:
            continue
        entry = inventory.get(gene.canvas)
        if entry is not None and 0 <= gene.layer < len(entry["layers"]):
            settings[gene.layer_token] = float(
                entry["layers"][gene.layer]["opacity"])
    return settings


def _apply_composites(pc, args, base_args, cnpro, composites):
    """The page's composites into the generation, by canvas key.

    Decoded the way the host decodes a canvas channel
    (modules_forge/forge_canvas/canvas.py, base64_to_image): PNG to RGBA,
    as a numpy array for a unit's Input slot - the unit canvases are
    numpy=True - and as a PIL image for `p.init_images`, which the img2img
    canvas hands the host as PIL. Same bytes the live composite would have
    produced, same decoding, so the generation cannot tell the two apart.
    """
    from PIL import Image
    import numpy as np
    for key, uri in composites.items():
        image = Image.open(io.BytesIO(
            base64.b64decode(uri.partition("base64,")[2]))).convert("RGBA")
        where = _canvas_unit_slot(key)
        if where is None:
            pc.init_images = [image]
            continue
        unit_index, slot = where
        index = cnpro.args_from + unit_index
        if args[index] is base_args[index]:
            args[index] = copy.copy(args[index])
        setattr(args[index], _input_field(slot), np.array(image))


def _space_signature(space):
    """What makes two searches comparable: the dimensions, exactly.

    Solver state only means anything over the space it was learned on, so
    import refuses to replay observations onto a different one - a changed
    label list silently reindexes every choice, which would be worse than
    starting fresh.
    """
    signature = []
    for dimension in space.dimensions:
        signature.append([
            dimension.name, dimension.kind,
            [str(label) for label in dimension.labels],
            round(float(dimension.low), 6), round(float(dimension.high), 6),
            round(float(dimension.step), 6)])
    return signature


def _coerce_point(space, values):
    """A JSON point (lists, floats) back into a space point (ints, floats)."""
    return tuple(
        int(v) if d.kind == "choice" else float(v)
        for d, v in zip(space.dimensions, values))


def _project_point(space, values):
    """A saved point reinterpreted on rows the user edited after searching.

    GOOD/N-GOOD are generators, not a continuation of the search. Their rows
    are therefore live controls: an existing coordinate keeps its position,
    choices and ranges clamp to what the edited row can express, removed
    coordinates disappear, and newly added ones take their baseline. This is
    deliberately different from RESUME, where reinterpreting observations
    would change what the old comparisons meant.
    """
    saved = list(values or [])
    projected = []
    for index, dimension in enumerate(space.dimensions):
        if index >= len(saved):
            projected.append(dimension.baseline())
            continue
        value = saved[index]
        try:
            if dimension.kind == "choice":
                last = max(len(dimension.labels) - 1, 0)
                projected.append(min(max(int(float(value)), 0), last))
            else:
                projected.append(dimension.snap(float(value)))
        except (TypeError, ValueError, OverflowError):
            projected.append(dimension.baseline())
    return tuple(projected)


def _solver_payload(search, seed, space, initial_settings=None):
    """Everything needed to rebuild the search: the observations by VALUE.

    Points are stored as tuples, not indices, so the replay does not depend
    on the original insertion order surviving - and `interesting` marks ride
    along because they steer the pool. The rng stream position is not saved:
    re-import reproduces the learned state exactly, and everything after it
    is deterministic in that state plus the seed.
    """
    return {
        "seed": int(seed),
        "duels": int(search.duels),
        # A REPORT, not state: it is recomputed from the observations below
        # and nothing replays it. It rides along so that the collage
        # button's count means something on a page that loaded after the
        # search that learned it - a reload, or a restart reading the disk
        # mirror.
        "capacity": int(search.capacity()),
        # Reset and Set GOOD are UI actions, but the settings they act on are
        # search state: exactly what every writable component held when the
        # search was launched. Keep it beside the observations so a stop,
        # reload, server restart or exported/imported session does not turn
        # Reset into a best-effort guess.
        "initial_settings": {
            str(token): _jsonable(value)
            for token, value in (initial_settings or {}).items()
        },
        "signature": _space_signature(space),
        "observations": [
            [None if a is None else list(search.points[a]),
             list(search.points[b]), float(p)]
            for a, b, p in search.observations],
        "interesting": [list(point) for point in search._interesting],
        # The similarity row's labels, by value like the observations. These
        # are what the separation metric is fitted from, and the metric
        # decides which duels are worth asking at all - so a resumed search
        # that dropped them would go straight back to asking the questions
        # this session already answered.
        "similarities": [[list(a), list(b), bool(d), float(m)]
                         for a, b, d, m in search.similarities],
        # Schema marker for the absolute half of the top/middle rows. Builds
        # before this marker stored their similarity labels but discarded the
        # shared "on track" verdict; restore can upgrade those states once,
        # while a current state must not receive duplicate anchors.
        "on_track_anchors": True,
        # The hyperparameters as SELECTED, not just the data they were
        # selected from: re-selection runs every few observations, so the
        # exporting search and a replay are usually between selections -
        # and a replay that re-selects immediately believes measurably
        # different things than the search it claims to resume.
        "hyper": [float(v) for v in search._hyper],
    }


def _restore_solver(search, state, space, allow_row_changes=False):
    """Replay solver state; optionally project it onto currently edited rows."""
    changed = state.get("signature") != _space_signature(space)
    if changed and not allow_row_changes:
        return None
    restore_point = _project_point if changed else _coerce_point
    count = 0
    for point_a, point_b, p in state.get("observations", []):
        index_b = search._row(restore_point(space, point_b))
        index_a = (None if point_a is None
                   else search._row(restore_point(space, point_a)))
        search.observations.append((index_a, index_b, float(p)))
        count += 1
    search.duels = int(state.get("duels", count))
    for point in state.get("interesting", []):
        search.mark_interesting(restore_point(space, point))
    # Replayed then fitted ONCE, rather than refitting per label the way
    # observe() does: the fit is over the whole set either way, and a replay
    # of two hundred labels would otherwise run it two hundred times.
    has_on_track_anchors = bool(state.get("on_track_anchors", False))
    for record in state.get("similarities", []):
        point_a, point_b, distinct = record[0], record[1], record[2]
        mass = float(record[3]) if len(record) > 3 else 1.0
        restored_a = restore_point(space, point_a)
        restored_b = restore_point(space, point_b)
        search.similarities.append((restored_a, restored_b,
                                    bool(distinct), mass))
        if not has_on_track_anchors:
            # Both similarity-labelled rows mean "on track". Old payloads
            # retained enough information to recover that signal exactly;
            # add it once instead of resuming a positive-only session with a
            # model that still knows only relative winners.
            for point in (restored_a, restored_b):
                search.observations.append(
                    (None, search._row(point), search.ON_TRACK_P))
    search._fit_metric()
    hyper = state.get("hyper")
    if hyper and len(hyper) == 3:
        search._hyper = tuple(float(v) for v in hyper)
        # A finite marker instead of the fresh engine's -inf, which is the
        # "never fitted, re-select immediately" trigger: the restored
        # hyperparameters hold until the ordinary cadence re-selects, the
        # same way they would have held in the exporting session.
        search._logz = 0.0
    search._dirty = True
    return count


def _export_html(payload):
    """The session file: readable page around a verbatim JSON island."""
    esc = html.escape

    def table(rows):
        body = "".join(f"<tr><th>{esc(str(k))}</th><td>{esc(str(v))}</td></tr>"
                       for k, v in rows)
        return f"<table>{body}</table>"

    sections = []
    row_count = int(payload.get("row_count", 1))
    row_lines = []
    for index, row in enumerate(payload.get("rows", [])[:row_count]):
        parts = [f"{key} = {row.get(key)!r}" for key in ROW_FIELDS]
        row_lines.append((f"row {index + 1}", " · ".join(parts)))
    sections.append("<h2>Search rows</h2>" + table(row_lines))
    sections.append("<h2>Settings</h2>" + table(
        sorted(payload.get("settings", {}).items())))
    # A "Loop" section used to sit here, holding the two switches the panel
    # no longer has. Files that still carry them keep them in the island (a
    # payload is written back verbatim) - they are simply nothing this build
    # has a control for, so there is nothing to lay out for a human either.

    canvases = payload.get("canvases", {})
    if canvases:
        images = "".join(
            f"<figure><figcaption>{esc(token)}</figcaption>"
            f"<img src='{esc(uri)}' alt='{esc(token)}'></figure>"
            for token, uri in sorted(canvases.items()))
        sections.append(f"<h2>Canvases</h2><div class='imgs'>{images}</div>")

    solver = payload.get("solver")
    if solver:
        sections.append("<h2>Solver</h2>" + table([
            ("graded duels", solver.get("duels", 0)),
            ("observations", len(solver.get("observations", []))),
            ("interesting marks", len(solver.get("interesting", []))),
            ("similarity verdicts", len(solver.get("similarities", []))),
            ("distinctly different good samples",
             solver.get("capacity", "not recorded")),
            ("seed", solver.get("seed")),
        ]))
    if payload.get("result"):
        sections.append("<h2>Recommendation</h2><pre>"
                        + esc(payload["result"]) + "</pre>")

    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # "</script>" inside a prompt would end the island early; JSON reads the
    # escaped form back as the same string.
    blob = blob.replace("</", "<\\/")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>CNPro A/B session (v{payload.get('version', EXPORT_VERSION)})"
        "</title><style>"
        "body{font-family:sans-serif;margin:2em;max-width:70em}"
        "table{border-collapse:collapse;margin:0.5em 0}"
        "th,td{border:1px solid #ccc;padding:2px 8px;text-align:left;"
        "vertical-align:top}th{white-space:nowrap}"
        ".imgs{display:flex;flex-wrap:wrap;gap:8px}"
        ".imgs img{max-width:220px;max-height:220px;display:block}"
        "figure{margin:0}figcaption{font-size:0.8em;opacity:0.7}"
        "pre{white-space:pre-wrap;border:1px solid #ccc;padding:8px}"
        "</style></head><body>"
        f"<h1>CNPro A/B session <small>v{payload.get('version', EXPORT_VERSION)}"
        f" · {esc(str(payload.get('tab', '')))}</small></h1>"
        "<p>Import this file back through the CNPro A/B panel's Import "
        "button. Everything below is the same data, laid out for a human: "
        "if the import ever refuses this file, these tables and images are "
        "enough to reproduce the session by hand.</p>"
        + "".join(sections)
        + '<script type="application/json" id="cnpro-ab-state">'
        + blob + "</script></body></html>")


def _parse_export(text):
    """The payload out of a session file, or None if it is not one."""
    match = RE_EXPORT_ISLAND.search(str(text or ""))
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except ValueError:
        return None


def _jsonable(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _set_targets(groups, is_img2img):
    """(token, component) for every scalar the panel can read or write -
    the unit fields plus the host's own boxes: prompt, negative prompt,
    seed, and the sampler settings (_HOST_FIELD_NAMES). The SAME list
    serves Set (writes), Export (reads) and Import (writes): one mapping,
    three verbs, so the three can never drift apart."""
    targets = []
    for index, group in enumerate(groups):
        for field_name in UNIT_FIELDS:
            component = _field_component(group, field_name)
            if component is not None:
                targets.append((f"u{index}.{field_name}", component))
    unreachable = []
    for name in _HOST_FIELD_NAMES:
        component = _host_component(name, is_img2img)
        if component is None:
            unreachable.append(name)
        else:
            targets.append((name, component))
    return targets, unreachable


def _canvas_targets(groups):
    """(token, component) for every image channel a unit carries: the input
    canvases (background and drawn foreground), the mask canvas, the output
    masks and the per-input weight masks. Guarded lookups throughout - a
    build of CNPro without one of these simply exports without it."""
    targets = []
    for index, group in enumerate(groups):
        for slot, canvas in enumerate(getattr(group, "image_canvases", []) or []):
            targets.append((f"u{index}.canvas{slot}.bg", canvas.background))
            targets.append((f"u{index}.canvas{slot}.fg", canvas.foreground))
        mask = getattr(group, "mask_image", None)
        if mask is not None:
            targets.append((f"u{index}.mask.bg", mask.background))
            targets.append((f"u{index}.mask.fg", mask.foreground))
        for band, component in (getattr(group, "output_masks", {}) or {}).items():
            targets.append((f"u{index}.outmask.{band}", component))
        for slot, channels in enumerate(getattr(group, "weight_masks", []) or []):
            component = (channels.get("global")
                         if isinstance(channels, dict) else None)
            if component is not None:
                targets.append((f"u{index}.wmask{slot}", component))
    return targets


# ---------------------------------------------------------------------------
# The session
#
# One per tab. This is the whole channel between the blocked run() and the live
# panel: run() publishes duels and waits, the timer reads, the buttons write.
# Everything crossing that line goes through the condition variable's lock -
# not for the images (a reference swap is atomic enough) but because the STOP
# flag, the pending score and the "what are we waiting for" state are read
# together and must agree with each other.
# ---------------------------------------------------------------------------

class _Session:

    def __init__(self):
        self.condition = threading.Condition()
        # Held while a poll is being answered - see `tick`. Not the condition
        # above: that one is the run/panel handshake and is held while
        # WAITING, which is exactly when a poll must still be served.
        self.tick_lock = threading.Lock()
        self.running = False
        # STAGED, BUT NOT YET PICKED UP. A START/RESUME press cannot start
        # the loop itself - it stages its intent and clicks Generate, and the
        # host may take a while to get to the job (it may be finishing
        # another one). Between the press and run() calling `start` there is
        # therefore a window in which nothing is running and yet the search
        # has been asked for, and without this flag the panel spends that
        # window offering START again - which stages a SECOND search.
        self.launching = False
        self.stopping = False
        self.pending_score = None
        self.pending_demos = []    # queued _Demo requests, served in order
        self.interesting = [False, False]     # A, B - toggled, sent with the grade
        self.awaiting = False
        self.generating = None     # "A"/"B" while that side's image renders
        # Monotone within ONE explicitly started render. The host's sampler
        # counters briefly fall back to zero during finalization; reading them
        # raw made a bar that had reached 100% flash empty before the image was
        # published. `render_started` also rejects a stale 100% left by the
        # previous side when the next side has only just been announced.
        self.render_progress = 0.0
        self.render_started = False
        self.generation = 0        # bumped whenever the images change
        self.image_a = None
        self.image_b = None
        self.duel = 0
        self.graded = 0
        self.status = ""
        self.result = ""
        self.summary = ""
        self.trace = []
        self.solver_state = None   # refreshed after every grade, for Export
        # Exact values of every component Set can write at the START/RESUME
        # press. Reset replays this mapping; Set GOOD uses it as the base on
        # which a sampled point is applied. It also rides in solver_state so
        # both buttons survive a page/server restart and session Import.
        self.initial_settings = None
        # The solver's capacity estimate - how many distinctly different good
        # configurations it believes it can produce. None until a search has
        # said; the panel then falls back to the retained state's own copy,
        # which is what keeps the button's count meaningful on a freshly
        # loaded page.
        self.capacity = None
        # The canvas round trip - see THE CANVAS LAYER ROUND TRIP. A request
        # is `{seq, overrides}` while run() waits for the page's composites
        # and None otherwise; the tick paints it into the request channel,
        # the page answers into `canvas_reply` with the same seq.
        self.canvas_seq = 0
        self.canvas_request = None
        self.canvas_reply = None

    # -- written by run() ------------------------------------------------

    def composites(self, overrides):
        """The page's composites for `overrides` ({key: {index: alpha}}),
        as {key: data URL}. Blocks until they arrive.

        A deadline, unlike await_grade: nobody has to act for this answer,
        so silence past CANVAS_WAIT_SECONDS means the page is closed or
        javascript/cnpro_ab.js did not load, and the search is told so
        rather than left waiting on a composite nothing will send. The
        wait is a 0.25s poll for the same reason await_grade's is: the
        host's Interrupt writes `shared.state`, not this condition.
        """
        with self.condition:
            self.canvas_seq += 1
            seq = self.canvas_seq
            self.canvas_request = {"seq": seq, "overrides": overrides}
            self.canvas_reply = None
            deadline = time.monotonic() + CANVAS_WAIT_SECONDS
            try:
                while True:
                    reply = self.canvas_reply
                    if isinstance(reply, dict) and reply.get("seq") == seq:
                        if reply.get("error"):
                            raise RuntimeError(
                                f"{WHO}: the page could not composite a "
                                f"canvas - {reply['error']}.")
                        images = reply.get("images")
                        missing = [key for key in overrides
                                   if not isinstance(images, dict)
                                   or not images.get(key)]
                        if missing:
                            raise RuntimeError(
                                f"{WHO}: the page returned no composite for "
                                f"{', '.join(missing)}.")
                        return images
                    if self.stopping or state.interrupted \
                            or getattr(state, "stopping_generation", False):
                        return None
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            f"{WHO}: the page did not answer the canvas "
                            f"request within {CANVAS_WAIT_SECONDS}s. A "
                            f"Canvas layer row needs the page open with "
                            f"javascript/cnpro_ab.js loaded (hard-refresh "
                            f"the browser).")
                    self.condition.wait(0.25)
            finally:
                self.canvas_request = None

    def extend_initial(self, settings):
        """More initial settings, learned after the press - the layer
        opacities the page reported, which no component holds."""
        with self.condition:
            if not isinstance(self.initial_settings, dict):
                self.initial_settings = {}
            for token, value in (settings or {}).items():
                self.initial_settings.setdefault(str(token), _jsonable(value))

    def start(self, resumed=False):
        # solver_state is deliberately NOT cleared: it is the tab's learned
        # taste, not this run's scratch. Clearing it here meant that starting
        # a search and interrupting it before the first grade destroyed the
        # PREVIOUS search's state - GOOD/N-GOOD then reported "no solver state"
        # about a taste that had cost dozens of duels to learn. A resumed or
        # graded run overwrites it; a run that learns nothing leaves it be.
        #
        # AND NEITHER IS THE RECORD, when this run RESUMES the last one. The
        # trace is the answer to "what has this search actually tried", and a
        # search that continues where it left off has tried all of it - the
        # observations it is resuming from are the very duels those lines
        # describe. Wiping it here meant that every STOP-and-continue (and
        # every Interrupt, and every pause) silently threw away the record of
        # the session it was continuing, leaving a Tried box that began at
        # duel 42 with nothing before it. Only a START press starts a new
        # record. A mismatched RESUME is refused before start() rather than
        # silently being changed into START.
        with self.condition:
            self.running = True
            self.launching = False
            self.stopping = False
            self.pending_score = None
            self.pending_demos = []
            self.awaiting = False
            self.generating = None
            self.render_progress = 0.0
            self.render_started = False
            self.generation += 1
            self.image_a = self.image_b = None
            self.duel = self.graded = 0
            self.canvas_request = self.canvas_reply = None
            self.status = "starting"
            if not resumed:
                self.result = ""
                self.summary = ""
                self.trace = []

    def record(self, side, recipe):
        """One generated image, in the order it was made.

        The trace is what answers "what did it actually try", which neither
        the images nor the recommendation can: two duels can look alike and
        differ in a field the eye does not read, and the winner says nothing
        about what it beat. Trimmed from the FRONT - a long session should
        cost a bounded amount of memory, and the recent rounds are the ones
        anybody scrolls back to.
        """
        with self.condition:
            self.trace.append(f"{side}: {recipe}")
            if len(self.trace) > TRACE_LIMIT:
                del self.trace[:len(self.trace) - TRACE_LIMIT]

    def finish(self, status):
        with self.condition:
            self.running = False
            self.launching = False
            self.awaiting = False
            self.generating = None
            self.canvas_request = None
            self.status = status
            self.generation += 1

    def say(self, status, generating=None):
        """Record the phase and, while rendering, which side owns progress.

        Callers that say anything else clear the side with the same call:
        whatever was rendering is not any more.
        """
        with self.condition:
            self.status = status
            self.generating = generating
            if generating is not None:
                # `say(..., generating=side)` is the render boundary. Reset
                # even when the side is A twice (successive GOOD samples).
                self.render_progress = 0.0
                self.render_started = False

    def track_progress(self, progress):
        """Latch host progress monotonically for the active render.

        A fresh render may initially see the previous render's terminal 1.0;
        ignore that until a sub-terminal value proves the new sampler has
        started. Once started, host finalization may report 0 again, but a
        completed fraction never moves backwards. `render_complete` is the
        authoritative terminal edge for cached and one-step renders.
        """
        with self.condition:
            if self.generating not in ("A", "B"):
                return self.render_progress
            value = min(max(float(progress or 0.0), 0.0), 1.0)
            if not self.render_started:
                if value >= 1.0:
                    return self.render_progress
                self.render_started = True
            self.render_progress = max(self.render_progress, value)
            return self.render_progress

    def render_complete(self, side):
        """The render returned successfully: its only truthful state is 1."""
        with self.condition:
            if self.generating == side:
                self.render_started = True
                self.render_progress = 1.0

    def publish(self, duel, image_a, image_b, status):
        with self.condition:
            self.duel = duel
            self.image_a = image_a
            self.image_b = image_b
            self.awaiting = True
            self.generating = None
            self.render_started = True
            self.render_progress = 1.0
            # Marks belong to the images they were toggled on.
            self.interesting = [False, False]
            self.status = status
            self.generation += 1

    def start_demo(self, status):
        """A sample run: alive for the poller, without wiping the session.

        Unlike a FRESH `start`, everything the finished search left behind -
        the recommendation, the trace, the solver state, the last images, the
        duel counts - is KEPT: a demo run adds to that record rather than
        opening a new one, and only a search on new rows resets it.
        """
        with self.condition:
            self.running = True
            self.launching = False
            self.stopping = False
            self.pending_score = None
            self.pending_demos = []
            self.awaiting = False
            # Demo samples render into the A slot, so A's chrome carries
            # their progress for the whole batch.
            self.generating = "A"
            self.render_progress = 0.0
            self.render_started = False
            self.status = status
            self.generation += 1

    def publish_demo(self, image, status):
        """A freshly rendered sample, into the A slot.

        The slot doubles as the panel's anchor: `image_a` is what keeps the
        duel group visible once nothing is running, so after a demo run the
        GOOD/N-GOOD buttons stay on screen for the next press instead of
        folding the panel away under the user's pointer.
        """
        with self.condition:
            self.image_a = image
            self.image_b = None
            self.awaiting = False
            self.generating = None
            self.render_started = True
            self.render_progress = 1.0
            self.status = status
            self.generation += 1

    def set_result(self, result, summary):
        with self.condition:
            self.result = result
            self.summary = summary

    def set_capacity(self, count):
        """How many good answers the solver can currently hand over.

        Written after every graded duel and painted into the collage
        button's own label, so the button always advertises what pressing it
        will do. Nothing else writes it: the editable box this used to feed
        is gone, because the count and the collage are now the same
        selection and the box could only ever repeat the button.
        """
        with self.condition:
            self.capacity = int(count)

    def remember_initial(self, settings):
        """Memorize the settings this search began from, by value."""
        if not isinstance(settings, dict):
            return False
        with self.condition:
            self.initial_settings = {
                str(token): _jsonable(value)
                for token, value in settings.items()
            }
        return True

    def await_grade(self):
        """Block until the user grades, stops or interrupts. FOREVER, if that
        is how long it takes.

        THERE IS NO DEADLINE, and there used to be one: the loop paused after
        an idle timeout (a setting, 30 minutes by default) on the theory that
        a search nobody answers should not hold the tab. It should. A duel on
        screen is a question that has been asked and paid for - two
        generations - and the only thing a clock adds is that walking away
        from it costs the answer. STOP and the host's own Interrupt already
        end a search, both are one click, and both are things the user MEANT.
        Coming back to a duel hours later and grading it is not an error
        state.

        Returns the grade, or None for "the search is over". The wait is a
        POLL of 0.25s rather than a plain wait, because the host's Interrupt
        button does not go through this session at all - it writes
        `shared.state`, which a condition variable cannot be notified by.
        """
        with self.condition:
            while True:
                if self.stopping:
                    return None
                if self.pending_score is not None:
                    score = self.pending_score
                    self.pending_score = None
                    self.awaiting = False
                    # Only a real answer counts as graded - not a GOOD
                    # sample request, which leaves the duel on screen still
                    # waiting.
                    if isinstance(score, tuple):
                        self.graded += 1
                    return score
                # After the grade, never instead of it: a queued GOOD press
                # is a side request, and a grade the user managed to land in
                # the same instant is the answer to the question on screen.
                if self.pending_demos:
                    self.awaiting = False
                    return self.pending_demos.pop(0)
                self.condition.wait(0.25)
                if state.interrupted or getattr(state, "stopping_generation", False):
                    return None

    # -- written by the panel --------------------------------------------

    def deliver_composites(self, text):
        """The page's answer to the request channel - `{seq, images}` or
        `{seq, error}` as JSON. A stale seq is dropped: the request it
        answers is over, and its images belong to a side already rendered
        or abandoned."""
        try:
            reply = json.loads(text) if text else None
        except (TypeError, ValueError):
            return False
        with self.condition:
            request = self.canvas_request
            if not isinstance(reply, dict) or request is None \
                    or reply.get("seq") != request.get("seq"):
                return False
            self.canvas_reply = reply
            self.condition.notify_all()
            return True

    def grade(self, score, disliked=False, similar=None):
        """An A-vs-B grade, 0..10. ONE click, whichever of the three rows it
        lands on.

        The row is the verdict the comparison itself cannot carry, and there
        are three of them because there are two such verdicts:

            similar=False   distinct samples, and we are on track
            similar=True    SIMILAR samples, and we are on track
            disliked=True   both are bad (and the row says nothing about
                            whether they looked alike - see below)

        The bottom row leaves `similar` as None rather than guessing it.
        Four rows would be needed to carry both verdicts at once, which is
        forty-four buttons, and a bad region is the cheapest place to lack
        similarity data - the search is about to stop asking about it anyway.

        Any interesting toggles set since the duel was published ride along.
        """
        with self.condition:
            if not self.awaiting:
                return False
            self.pending_score = (float(score), bool(disliked), similar,
                                  tuple(self.interesting))
            self.interesting = [False, False]
            self.status = "recorded - working out what to ask next"
            self.condition.notify_all()
            return True

    def request_demo(self, count=None):
        """Ask the loop for on-demand sample(s) from the good end - one for
        GOOD, a collage of `count` for N-GOOD.

        Mechanically a grade - it wakes the blocked loop - but it answers
        nothing: the loop renders the sample(s), prints their recipes to the
        trace, and returns to the SAME duel, still waiting.

        QUEUED, not gated on a duel being on screen: a press that lands
        while a duel is still rendering is served the moment the loop next
        listens, and several presses are served in order - the user asking
        for three GOOD samples is asking for three, not for three chances
        to hit the right instant. Returns False only when no loop is alive
        at all, which is the panel's cue to take the idle path instead.
        """
        with self.condition:
            if not self.running or self.stopping:
                return False
            request = _Demo(count)
            self.pending_demos.append(request)
            what = ("a fresh guess from the good end" if count is None
                    else f"a collage of {request.count} good samples")
            self.status = (f"rendering {what}" if self.awaiting else
                           f"queued {what} - after this render")
            self.condition.notify_all()
            return True

    def toggle_interesting(self, side):
        """Flip the "interesting" mark on side 0 (A) or 1 (B).

        A toggle, not a submission: it waits for the grade, like the dislike
        row it usually accompanies - the expected click is "this one is bad,
        AND it has something".
        """
        with self.condition:
            if not self.awaiting:
                return False
            self.interesting[side] = not self.interesting[side]
            marked = [name for name, on in zip("AB", self.interesting) if on]
            self.status = (f"✦ interesting: {' and '.join(marked)} - "
                           f"sent with your grade" if marked
                           else "interesting marks cleared")
            return True

    def arm(self, status):
        """A START/RESUME press that has staged itself and clicked Generate.

        Not `running` - nothing is - but the panel must read as busy from the
        press rather than from the host getting round to the job, or the
        button spends the wait offering to start a second search.
        """
        with self.condition:
            self.launching = True
            self.status = status

    def disarm(self):
        """The host reached run(), so the wait above is over either way.

        Called for EVERY generation this script sees, staged or plain: a
        press whose click never became a run would otherwise leave the panel
        reading "starting" forever, and the next Generate is exactly the
        evidence that it did not.
        """
        with self.condition:
            self.launching = False

    def request_stop(self):
        with self.condition:
            # `launching` too, and it is the way out of a press that never
            # became a run: STOP is on screen for as long as the panel thinks
            # something is coming, so it has to be able to take that back.
            self.launching = False
            self.stopping = True
            self.status = "stopping"
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {
                "running": self.running,
                "launching": self.launching,
                "awaiting": self.awaiting,
                "generating": self.generating,
                "progress": self.render_progress,
                "generation": self.generation,
                "image_a": self.image_a,
                "image_b": self.image_b,
                "duel": self.duel,
                "graded": self.graded,
                "capacity": self.capacity,
                "status": self.status,
                "result": self.result,
                "summary": self.summary,
                "trace": "\n".join(self.trace),
                "canvas_request": self.canvas_request,
            }


#: One session per tab. Keyed rather than a single global because both tabs
#: build the panel and their States are separate; only one can be generating,
#: but the OTHER tab's panel must not paint this one's duel.
_SESSIONS = {"txt2img": _Session(), "img2img": _Session()}


def _session(is_img2img):
    return _SESSIONS["img2img" if is_img2img else "txt2img"]


# ---------------------------------------------------------------------------
# Reaching the host's own prompt and seed boxes
#
# For the Set button only. Captured through the after_component callback rather
# than looked up later: the components exist by the time any script's ui() runs
# (the toprow is built above the script accordions), and a gradio event has to
# name its outputs at build time, so "find it when Set is pressed" is not a
# thing that can work.
# ---------------------------------------------------------------------------

_HOST_COMPONENTS = {}

#: The host boxes the panel reads and writes, by elem_id suffix - the same
#: names are the recipe/export tokens (see _generation_extras). Prompt and
#: seed were here first; the negative prompt and the sampler settings were
#: added after an exported session came back without its negative prompt -
#: everything on this list steers the image as surely as the prompt does,
#: and a session file that omits any of it "reproduces" something else.
_HOST_FIELD_NAMES = (
    "prompt", "neg_prompt", "seed", "steps", "sampling", "scheduler",
    "cfg_scale", "width", "height", "denoising_strength",
)

_WANTED_ELEM_IDS = tuple(
    f"{tab}_{name}"
    for tab in ("txt2img", "img2img")
    for name in _HOST_FIELD_NAMES)


def _on_after_component(component, **_kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id in _WANTED_ELEM_IDS:
        _HOST_COMPONENTS[elem_id] = component


script_callbacks.on_after_component(_on_after_component)


def _component_by_elem_id(elem_id):
    """The live Blocks' component with this elem_id, or None.

    THE FALLBACK, and it exists because the callback above is registered when
    this module is imported - so it only sees components created after that.
    That is the normal order (scripts load before the page is built) and it is
    not the only order: a script dropped in beside a running instance, a
    Reload UI that re-imports in a different sequence, or a host that builds
    its toprow earlier all end with an empty dict and a Set button that
    silently declines to touch the prompt. Asking the Blocks being built right
    now cannot be early or late.
    """
    try:
        from gradio.context import get_blocks_context
    except Exception:
        return None
    root = get_blocks_context()
    for block in getattr(root, "blocks", {}).values():
        if getattr(block, "elem_id", None) == elem_id:
            return block
    return None


def _host_component(name, is_img2img):
    """The host's own prompt / seed box for this tab, or None.

    Called at ui() build time: a gradio event has to name its outputs when it
    is wired, so "find it when Set is pressed" is not a thing that can work.
    """
    tab = "img2img" if is_img2img else "txt2img"
    elem_id = f"{tab}_{name}"
    component = _HOST_COMPONENTS.get(elem_id) or _component_by_elem_id(elem_id)
    if component is None:
        # LOUD, at build time. Without this the only symptom is a Set button
        # that appears to work and leaves the prompt alone - which is worse
        # than a missing button, because the user then generates with the
        # prompt and seed they happened to have and gets images that have
        # nothing to do with what they graded.
        logger.error(
            f"{WHO}: the host's #{elem_id} box was not found, so Set will not "
            f"be able to write the {name} it recommends. Everything else it "
            f"sets still applies; the {name} has to be copied by hand.")
    return component


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

#: Components per row that run() reads. The LoRA picker and its two tool
#: buttons are UI-only (they fill the textbox next to them and are never
#: read), so they are not among them.
#: ONE ROW'S VALUES, IN ORDER - the order gradio hands them to run(), the
#: order Export writes them and Import reads them, and the names the
#: session file uses for them. Every enumeration of a row's controls goes
#: through `_Row.args()`, which follows this tuple, so adding a control is
#: one entry here plus one attribute on _Row: nothing else can be left at
#: the old width.
ROW_FIELDS = ("target", "mode", "choices", "line", "values", "prompt_mode",
              "loras", "weights", "points", "canvas", "layer")
ROW_ARGS = len(ROW_FIELDS)

#: THE ROWS ARE THE WHOLE ARGUMENT LIST. There used to be two trailing
#: switches - "New seed each duel" and "Resume search" - and both are gone:
#: the seed is fixed by construction (see the module docstring), and
#: resuming is what the RESUME button says at the moment it is pressed, which
#: a checkbox read at Generate time could only ever say earlier and worse.
ARG_COUNT = 1 + MAX_ROWS * ROW_ARGS


class Script(scripts.Script):

    def title(self):
        return "CNPro A/B"

    # -- the panel -------------------------------------------------------

    def ui(self, is_img2img):
        groups = _ui_groups(is_img2img)
        if not groups:
            gr.Markdown("**ControlNet Pro is not loaded**, so there are no "
                        "units to search over. Enable the extension and "
                        "restart.")
            return []

        tab = "img2img" if is_img2img else "txt2img"
        eid = f"cnpro_ab_{tab}"
        enables = [g.enabled for g in groups]
        type_filters = [g.type_filter for g in groups]
        count = len(groups)

        # --- header: how many degrees of freedom -------------------------
        #
        # The two switches that used to share this line are gone - the seed
        # is fixed by construction, and resuming is the RESUME button - so
        # the row is back to the count and the three tools that change it.
        with gr.Row(variant="compact",
                    elem_classes=["cnpro-ab-headline-row"]):
            add = ToolButton(value=ADD_SYMBOL, elem_id=f"{eid}_add",
                             tooltip="One more degree of freedom")
            remove = ToolButton(value=REMOVE_SYMBOL, elem_id=f"{eid}_remove",
                                tooltip="One fewer degree of freedom")
            rescan = ToolButton(value=REFRESH_SYMBOL, elem_id=f"{eid}_rescan",
                                tooltip="Rescan the ControlNet and LoRA "
                                        "directories")
            headline = gr.Markdown(_headline(1), elem_id=f"{eid}_headline",
                                   elem_classes=["cnpro-ab-headline"])

        row_count = gr.State(1)
        rows = [self._row(i, eid, visible=(i == 0)) for i in range(MAX_ROWS)]

        # --- the duel ----------------------------------------------------
        #
        # ALWAYS ON SCREEN, and it did not use to be: the group was revealed
        # by the first duel, because a grading station with nothing to grade
        # is a screenful of empty boxes in a UI that is short of screenfuls.
        # START lives in this group, so hiding it would hide the one control
        # that could ever make a duel appear.
        #
        # What folds away instead is the pair of IMAGE boxes, which is where
        # essentially all of that screenful was - two gr.Images with no
        # picture in them. The rest is one row of buttons per line and reads
        # as the search's console, which is what it is.
        with gr.Group(elem_id=f"{eid}_duel",
                      elem_classes=["cnpro-ab-duel"]):
            with gr.Row(visible=False,
                        elem_classes=["cnpro-ab-images"]) as images_row:
                # NO fixed height. The panel is as wide as the user's window
                # lets it be, and a duel is decided on detail - so the images
                # take the width they are given and stop at their own
                # resolution, rather than being pinned to a thumbnail on a
                # screen with room for the real thing. The "never upscale"
                # half of that is CSS (`width: auto; max-width: 100%`), which
                # is why the size is not set here - see style.css.
                image_a = gr.Image(label="A", type="pil", interactive=False,
                                   show_download_button=False,
                                   elem_id=f"{eid}_image_a",
                                   elem_classes=["cnpro-ab-image"])
                image_b = gr.Image(label="B", type="pil", interactive=False,
                                   show_download_button=False,
                                   elem_id=f"{eid}_image_b",
                                   elem_classes=["cnpro-ab-image"])
            # One toggle under each image: "overall bad, we can't use it,
            # but this characteristic is tempting - try it in good samples".
            # NOT a grade and never valued as one: the mark adds no utility
            # (the grade already said how good the image is); it donates the
            # configuration's coordinates to hybrid candidates - see
            # ab_search.mark_interesting. A toggle so a stray click can be
            # unclicked; sent with the grade, like the dislike row it
            # usually accompanies.
            with gr.Row(elem_classes=["cnpro-ab-interesting"]):
                interesting_a = gr.Button(value="✦ A interesting",
                                          elem_id=f"{eid}_interesting_a",
                                          elem_classes=["cnpro-ab-mark"])
                interesting_b = gr.Button(value="✦ B interesting",
                                          elem_id=f"{eid}_interesting_b",
                                          elem_classes=["cnpro-ab-mark"])
            # THE SAME 0..10 SCALE, THREE TIMES, and the ROW is the verdict
            # the comparison itself cannot carry. One click answers, whichever
            # row it lands on - no second gesture, no memory of a reference,
            # no absolute number:
            #
            #   Distinct samples   these two look rather different, and we
            #                      are on track
            #   Similar samples    ...rather ALIKE. The cheapest click in the
            #                      panel and the most informative one: it is
            #                      a first-glance impression that costs no
            #                      deliberation, and because the top two rows
            #                      PARTITION the on-track case, every graded
            #                      duel yields a similarity label rather than
            #                      only the ones somebody bothered to mark.
            #                      That is what trains the separation metric -
            #                      see ab_search._fit_metric - which decides
            #                      whether a duel is worth asking, how
            #                      different a keeper has to be, and what a
            #                      collage may hold together.
            #   Bad samples        I will rate them, but both are bad.
            #
            # Ordered so the ANCHORS do not move: twenty duels of muscle
            # memory say "top = normal, bottom = both bad", and the new state
            # goes in the middle where it costs neither.
            #
            # The numbered buttons are the whole scale. There is deliberately
            # no live explanatory block underneath: replacing that output on
            # every phase change flickered and moved everything below it.
            with gr.Row(elem_classes=["cnpro-ab-grades", "cnpro-ab-distinct"]):
                # "Your turn", hugging the scale's left edge. The block is a
                # ZERO-WIDTH flex item sitting right before the first grade
                # button, and the text hangs off it leftward (style.css) -
                # anchored to the strip by construction, so it stays glued
                # to button 0 at any panel width, while the centred strip
                # itself does not move a pixel. Faded in by the tick's phase
                # CSS exactly while a grade is awaited: the one moment the
                # panel goes quiet and the next move is the user's, which
                # nothing used to say.
                gr.HTML("<span class='cnpro-ab-grade-hint'>Your turn "
                        "- grade "
                        "<span class='cnpro-ab-grade-hint-arrow'>➜</span>"
                        "</span>",
                        elem_classes=["cnpro-ab-grade-hint-wrap"])
                grade_buttons = [
                    gr.Button(value=str(score), min_width=34,
                              elem_id=f"{eid}_grade_{score}",
                              elem_classes=["cnpro-ab-grade"]
                                           + (["cnpro-ab-grade-tie"]
                                              if score == GRADES // 2 else []))
                    for score in range(GRADES)]
                gr.HTML("<span class='cnpro-ab-row-label'>Distinct samples"
                        "</span>",
                        elem_classes=["cnpro-ab-row-label-wrap"])
                # GOOD asks the solver for one fresh sample from what it
                # currently believes you want - pure inference, varied per
                # press, no effect on the duel or the model. The recipe
                # lands in Tried; the image lands in the output folder and
                # the final gallery.
                #
                # NO min_width, HERE OR ON THE OTHER THREE. The parked block
                # is one 2x2 grid whose cell size lives in exactly one place
                # - `--cnpro-ab-parked-w` in style.css, which every one of
                # the four reads for its width and both columns read for
                # their offsets. A number repeated here would be a second
                # opinion about the same box: it loses (the CSS carries
                # !important, which is what beats the inline style gradio
                # writes for min_width), so it could only ever be wrong
                # silently.
                good = gr.Button(value="GOOD",
                                 elem_id=f"{eid}_good",
                                 elem_classes=["cnpro-ab-demo"])
                # ONE BUTTON FOR BOTH ENDS OF THE LOOP, wearing whichever
                # state it is in - START while nothing runs, STOP while
                # something does. Two buttons would mean one of them is
                # always the wrong one to press, and a disabled START beside
                # a live STOP says nothing a single label does not.
                #
                # It starts as START: the panel opens with nothing running,
                # and the tick repaints it from the session on every poll.
                run_toggle = gr.Button(value="START", variant="primary",
                                       elem_id=f"{eid}_run",
                                       elem_classes=["cnpro-ab-run"])
            # "Rather similar, and we are on track". Same scale, same click,
            # one row lower - and the label it adds is the one thing no
            # amount of grading can produce: a graded duel encodes a
            # DIFFERENCE in utility, never a DISTANCE.
            with gr.Row(elem_classes=["cnpro-ab-grades", "cnpro-ab-similar"]):
                # A zero-width SPACER matching the hint's wrap on the row
                # above. It draws nothing; it exists so all three rows have
                # the same number of flex items before button 0. Without it
                # the hint's own flex gap shifts row one 2px right of the
                # other two - measured - and three scales that disagree by
                # 2px are three instruments rather than one ruler.
                gr.HTML("", elem_classes=["cnpro-ab-grade-hint-wrap"])
                similar_buttons = [
                    gr.Button(value=str(score), min_width=34,
                              elem_id=f"{eid}_similar_{score}",
                              elem_classes=["cnpro-ab-grade"]
                                           + (["cnpro-ab-grade-tie"]
                                              if score == GRADES // 2 else []))
                    for score in range(GRADES)]
                gr.HTML("<span class='cnpro-ab-row-label'>Similar samples"
                        "</span>",
                        elem_classes=["cnpro-ab-row-label-wrap"])
                # N-GOOD: a COLLAGE of N good samples that are all visibly
                # different from each other - the whole learned region at
                # once rather than one draw from it. Press again for another
                # collage; the entries are ranked by a fresh posterior draw
                # each time. See ab_search.population.
                #
                # THE COUNT IS THE LABEL - "3 GOOD" - and it used to be an
                # editable box beside it. The box was worth its width while
                # `capacity` was an estimate of the good region's size: it
                # could report forty, and the number worth spending was the
                # user's call. It is not that any more. The box and the
                # button now run the same selection (see ab_search.capacity),
                # so the box could only ever say what the button was about
                # to do - asking for more returns the same sheet, and the
                # only edit it still supported was asking for fewer of the
                # few answers there are. A button that says what it will do
                # is the whole control.
                #
                # The number is written by the tick, into the LABEL - and it
                # is the ONE thing that resizes the parked block. style.css
                # turns the digit count into the grid's cell width, which
                # all four buttons and both column offsets read, so the 2x2
                # stays a 2x2 at 3 or at 12 rather than the collage button
                # growing out of its column.
                n_good = gr.Button(value=f"{CAPACITY_DEFAULT} GOOD",
                                   elem_id=f"{eid}_n_good",
                                   elem_classes=["cnpro-ab-demo"])
                # What the press will actually ask for. A State rather than
                # the label it mirrors: the click reads a number, and
                # parsing it back out of "12 GOOD" would make the button's
                # wording load-bearing.
                capacity_state = gr.State(CAPACITY_DEFAULT)
                # RESUME, on THIS row rather than the one below it, and the
                # row it is declared in is the row it is painted on: the
                # parked controls are absolutely positioned inside their own
                # grade row (style.css), so being declared last would put it
                # alone at the bottom right with an empty slot above it - the
                # block would read as three buttons and a straggler. Here the
                # four make one 2x2 block, GOOD over N GOOD and START/STOP
                # over RESUME, and the third row's parked band is simply
                # empty.
                #
                # ALWAYS SHOWN, never hidden - only enabled or not. A control
                # that appears and disappears with the state is a control the
                # user has to hunt for; one that is greyed out says "this is
                # the thing you would press, and it is not available yet",
                # which is the more useful of the two statements. Disabled at
                # build time because a freshly opened panel has nothing
                # running and nothing to continue; the tick turns it on the
                # moment a retained state exists (see paint).
                resume = gr.Button(value="RESUME", interactive=False,
                                   elem_id=f"{eid}_resume",
                                   elem_classes=["cnpro-ab-resume"])
            # "I will rate them, but both are bad." The grade still matters
            # (which side is LESS bad steers the search inside the region);
            # the row is what lets it mark the whole region as avoid-this and
            # change the subject - see ab_search.observe. It carries no
            # similarity verdict: four rows would be needed for both, and a
            # bad region is the cheapest place to lack that data.
            with gr.Row(elem_classes=["cnpro-ab-grades", "cnpro-ab-dislike"]):
                gr.HTML("", elem_classes=["cnpro-ab-grade-hint-wrap"])
                dislike_buttons = [
                    gr.Button(value=str(score), min_width=34,
                              elem_id=f"{eid}_dislike_{score}",
                              elem_classes=["cnpro-ab-grade"]
                                           + (["cnpro-ab-grade-tie"]
                                              if score == GRADES // 2 else []))
                    for score in range(GRADES)]
                gr.HTML("<span class='cnpro-ab-row-label'>Bad samples</span>",
                        elem_classes=["cnpro-ab-row-label-wrap"])
        # --- the answer --------------------------------------------------
        with gr.Row(variant="compact", elem_classes=["cnpro-ab-answer"]):
            config = gr.Textbox(
                label="Configuration", value="", lines=1, max_lines=3, scale=9,
                placeholder="the search prints its recommendation here - or "
                            "paste one back in",
                elem_id=f"{eid}_config",
                elem_classes=["cnpro-ab-configuration"])
            # One compact 2x2 action block. Set GOOD is a posterior sample
            # applied without rendering; RESET spans the row below because it
            # is the inverse of either top action, restoring the exact values
            # memorized at search launch.
            with gr.Column(scale=2, min_width=180,
                           elem_classes=["cnpro-ab-set-actions"]):
                with gr.Row(variant="compact"):
                    apply_button = gr.Button(
                        value="Set", variant="primary", scale=1,
                        min_width=80, elem_id=f"{eid}_set")
                    set_good_button = gr.Button(
                        value="Set GOOD", scale=1, min_width=80,
                        elem_id=f"{eid}_set_good")
                reset_button = gr.Button(
                    value="RESET", scale=1, min_width=160,
                    elem_id=f"{eid}_reset")
        # Set's own report, OUTSIDE the duel group. It used to share the
        # status line in there, which is hidden until a search runs and hidden
        # again on a fresh page - so "Set applied nothing, and here is what it
        # could not place" was written where nobody could read it, and a Set
        # that did nothing looked exactly like a Set that worked. Empty, this
        # takes no height.
        set_note = gr.Markdown("", elem_id=f"{eid}_set_note",
                               elem_classes=["cnpro-ab-note"])

        # --- what was tried ----------------------------------------------
        #
        # Collapsed, because it is a record rather than a control: it matters
        # after the search, or when a duel looked wrong and the question is
        # what it actually generated. One line per image, in the order they
        # were made.
        with gr.Accordion("Tried", open=False, elem_id=f"{eid}_trace_group"):
            trace = gr.Textbox(
                label="", value="", lines=10, max_lines=10, interactive=False,
                show_label=False, elem_id=f"{eid}_trace",
                elem_classes=["cnpro-ab-trace"])

        # --- the whole session as one file --------------------------------
        #
        # Export writes rows, unit settings, canvases, prompt and negative
        # prompt, sampler settings, seed and the
        # solver's learned state into a single HTML file - readable in a
        # browser AND machine-importable, so a version this build cannot
        # read is still a page a human can reproduce by hand. Import applies
        # everything it recognises and stages the solver state for the next
        # Generate on this tab.
        with gr.Accordion("Session file", open=False,
                          elem_id=f"{eid}_io_group"):
            with gr.Row(variant="compact"):
                session_file = gr.File(
                    label="session (.html)", file_types=[".html"], scale=8,
                    elem_id=f"{eid}_session_file")
                export_button = gr.Button(value="Export", scale=1,
                                          min_width=70,
                                          elem_id=f"{eid}_export")
                import_button = gr.Button(value="Import", scale=1,
                                          min_width=70,
                                          elem_id=f"{eid}_import")
            io_note = gr.Markdown("", elem_id=f"{eid}_io_note",
                                  elem_classes=["cnpro-ab-note"])

        # A gr.Timer, not a JS interval: it is the host's own gradio, it stops
        # when told to, and it needs no element to hang off. It starts ACTIVE
        # so that a freshly loaded page repaints whatever the server-side
        # session is doing - a reload is how the user recovers from a
        # connection gone stale, and a panel that comes back blank over a
        # search still waiting for a grade is the search "stopping working".
        # The first tick switches it off again when nothing is running, so
        # an idle panel still costs one request per page load, not one per
        # second forever. What switches it back on is any press that means
        # work - START, RESUME, GOOD/N-GOOD - and the Generate click besides.
        poller = gr.Timer(POLL_SECONDS, active=True)
        # What THIS browser was last sent, so the poll can send only what
        # changed - see `tick`. A dict rather than a counter because the parts
        # move independently: the images change once per duel, the recipe once
        # per answer, and the controls change with the search phase.
        seen = gr.State({})

        # The Canvas layer row's channels to and from the page - see THE
        # CANVAS LAYER ROUND TRIP and javascript/cnpro_ab.js. The two the
        # page WRITES are textboxes (updateInput reaches gradio); the two
        # it READS are HTML, hidden by class rather than by visible=False
        # so their content still updates in the DOM for the observer.
        canvases = gr.Textbox(value="", visible=False,
                              elem_id=f"{eid}_canvases")
        canvas_reply = gr.Textbox(value="", visible=False,
                                  elem_id=f"{eid}_canvas_reply")
        canvas_request = gr.HTML(value="", elem_id=f"{eid}_canvas_request",
                                 elem_classes=["cnpro-ab-channel"])
        canvas_set = gr.HTML(value="", elem_id=f"{eid}_canvas_set",
                             elem_classes=["cnpro-ab-channel"])

        targets, unreachable = _set_targets(groups, is_img2img)
        self._wire_rows(rows, row_count, headline, add, remove, rescan,
                        enables, type_filters, count)
        self._wire_canvases(canvases, rows, tab)
        self._wire_duel(poller, seen, images_row, image_a, image_b,
                        config, trace,
                        (grade_buttons, similar_buttons, dislike_buttons),
                        (interesting_a, interesting_b), (good, n_good),
                        capacity_state, run_toggle, resume, rows, row_count,
                        is_img2img, targets, canvas_request, canvas_reply)
        self._wire_set(apply_button, set_good_button, reset_button,
                       config, set_note, targets, unreachable,
                       rows, row_count, is_img2img, canvas_set)
        self._wire_io(export_button, import_button, session_file, io_note,
                      rows, row_count, headline,
                      targets, _canvas_targets(groups), len(groups),
                      is_img2img, resume)

        # Returned in the order run() unpacks them - see _read_rows.
        controls = [row_count]
        for row in rows:
            controls.extend(row.args())
        return controls

    # -- one row ---------------------------------------------------------

    def _row(self, index, eid, visible):
        rid = f"{eid}_r{index}"
        with gr.Row(variant="compact", visible=visible,
                    elem_classes=["cnpro-ab-row"]) as container:
            target = gr.Dropdown(
                label=f"{index + 1}. varies", choices=list(GLOBAL_TARGETS),
                value=None, scale=2, elem_id=f"{rid}_target")
            # EVERY dependent control starts hidden, this one included. With a
            # blank target the row used to open showing the unit controls,
            # which reads as "this needs a ControlNet unit" when it does not -
            # Prompt and LoRA need none.
            # A Dropdown, not a Radio: three chips needed 310px to stay on one
            # line, a dropdown says the same thing in a label's width - and
            # the panel is short of width, not of clicks.
            mode = gr.Dropdown(
                label="how", choices=MODES, value=MODE_MODEL, visible=False,
                scale=2, elem_id=f"{rid}_mode")
            # A Canvas layer row's two picks. Their CHOICES come from the
            # page (see _wire_canvases): which canvases hold an image and
            # how many layers each has is browser state, and the panel is
            # built before any of it exists. The canvas value is its key
            # (RE_CANVAS_KEY), shown under its label; the layer value is
            # "L<n> · <size>", read back by _layer_index.
            canvas = gr.Dropdown(
                label="canvas", choices=[], value=None, visible=False,
                scale=3, elem_id=f"{rid}_canvas")
            layer = gr.Dropdown(
                label="layer", choices=[], value=None, visible=False,
                scale=2, elem_id=f"{rid}_layer")
            choices = gr.Dropdown(
                label="models", choices=[], value=[], multiselect=True,
                visible=False, scale=6, elem_id=f"{rid}_choices")
            line = gr.Dropdown(
                label="profile", choices=list(_profile_lines()), value="Main",
                visible=False, scale=2, elem_id=f"{rid}_line")
            # Which drawn points a "Profile points" row moves - a LIST, and
            # the listed points move together by the row's offset: 0-based,
            # left to right, negative from the end (-1 = rightmost).
            point = gr.Textbox(
                label="points", value="0", visible=False,
                scale=1, min_width=70, placeholder="0, 2, -1",
                elem_id=f"{rid}_point")
            # DECLARED BEFORE the values box, which is what puts it to the LEFT
            # of it - a hidden component takes no space, so declaration order
            # is the only thing that decides where a row's visible controls
            # land. That mirrors the LoRA row, where the picker sits left of
            # its textbox, and it reads correctly besides: the mode says how
            # the box below is going to be USED, so it comes first.
            prompt_mode = gr.Dropdown(
                label="prompt mode", choices=list(PROMPT_MODES),
                value=PROMPT_REPLACE, visible=False, scale=2,
                elem_id=f"{rid}_prompt_mode")
            values = gr.Textbox(
                label="values (one per line)", value="", visible=False,
                scale=6, lines=2, max_lines=8,
                elem_id=f"{rid}_values")
            lora_pick = gr.Dropdown(
                label="add LoRA", choices=_lora_names(), value=None,
                visible=False, scale=3, elem_id=f"{rid}_lora_pick")
            # Beside the picker, because that is where the hands already are:
            # the refresh spares a trip to the header's rescan when a LoRA
            # was just downloaded mid-setup, and the dice adds a candidate
            # the user did NOT think of - the row's list is the search's
            # whole horizon, and a cheap way to widen it is worth one button.
            lora_refresh = ToolButton(
                value=REFRESH_SYMBOL, visible=False,
                elem_id=f"{rid}_lora_refresh",
                tooltip="Rescan the LoRA directory")
            lora_dice = ToolButton(
                value=DICE_SYMBOL, visible=False,
                elem_id=f"{rid}_lora_dice",
                tooltip="Add a random LoRA not already listed")
            loras = gr.Textbox(
                label="LoRAs (one per line)", value="", visible=False, scale=5,
                lines=2, max_lines=8, placeholder="add_detail\nfilm_grain",
                elem_id=f"{rid}_loras")
            # ONE box for the whole weight list, interval notation included -
            # it replaced a min/max pair plus a panel-global step, which could
            # only ever say "an evenly spaced grid" and said it in three
            # places. Empty means the kind's default grid; the placeholder
            # names it (see _row_visibility).
            weights = gr.Textbox(
                label="weights", value="", visible=False, scale=2,
                min_width=110, elem_id=f"{rid}_weights")

        row = _Row(container, target, mode, choices, line, values,
                   prompt_mode, lora_pick, loras, weights,
                   lora_refresh, lora_dice, point, canvas, layer)

        def visibility(target_value, mode_value):
            return [gr.update(**props) for props in _row_visibility(
                target_value, mode_value)]

        switches = [mode, choices, line, values, prompt_mode,
                    lora_pick, loras, weights,
                    lora_refresh, lora_dice, point, canvas, layer]
        for component in (target, mode):
            component.change(fn=visibility,
                             inputs=[target, mode],
                             outputs=switches, show_progress=False)

        # THE FALLBACK PATH for the LoRA picker, not the normal one:
        # javascript/cnpro_ab.js appends the line client-side so that the
        # option list does not close and the filter typed into it survives.
        # This is what runs when that file has not loaded - a browser cache
        # away - and the two cannot double-insert, because the JS only appends
        # when it has swallowed the event that would have triggered this.
        #
        # Clearing the picker is what lets the same LoRA be picked twice: a
        # dropdown set to the value it already holds fires no change event.
        def add_lora(name, current):
            if not name:
                return gr.update(), gr.update()
            body = (current or "").rstrip()
            return (f"{body}\n{name}" if body else name), gr.update(value=None)

        lora_pick.change(fn=add_lora, inputs=[lora_pick, loras],
                         outputs=[loras, lora_pick], show_progress=False)

        # The same walk the header's rescan does for LoRAs, without also
        # re-hashing every ControlNet file - refreshing a picker mid-setup
        # should cost what it names.
        def refresh_loras():
            networks = _networks()
            if networks is not None:
                try:
                    networks.list_available_networks()
                except Exception as exc:
                    logger.warning(f"{WHO}: could not rescan LoRAs ({exc}).")
            return gr.update(choices=_lora_names())

        lora_refresh.click(fn=refresh_loras, inputs=[],
                           outputs=[lora_pick], show_progress=False)

        # Drawn from the LoRAs NOT already listed, so leaning on the button
        # walks through the library instead of stuttering on repeats; with
        # nothing left to add it does nothing rather than duplicating a line
        # the search would then hold as two identical candidates.
        def dice_lora(current):
            listed = {_lora_name(line) for line in _lines(current)}
            fresh = [name for name in _lora_names() if name not in listed]
            if not fresh:
                return gr.update()
            body = str(current or "").rstrip()
            pick = random.choice(fresh)
            return f"{body}\n{pick}" if body else pick

        lora_dice.click(fn=dice_lora, inputs=[loras],
                        outputs=[loras], show_progress=False)
        return row

    # -- wiring ----------------------------------------------------------

    def _wire_rows(self, rows, row_count, headline, add, remove, rescan,
                   enables, type_filters, count):
        """The +/- buttons, and the choice lists that follow the units."""

        def resize(delta):
            def handler(current):
                wanted = min(max(int(current or 1) + delta, 1), MAX_ROWS)
                return ([wanted, _headline(wanted)]
                        + [gr.update(visible=i < wanted) for i in range(MAX_ROWS)])
            return handler

        outputs = [row_count, headline] + [row.container for row in rows]
        add.click(fn=resize(+1), inputs=[row_count], outputs=outputs,
                  show_progress=False)
        remove.click(fn=resize(-1), inputs=[row_count], outputs=outputs,
                     show_progress=False)

        def choices_update(target_value, selected, filters):
            """One row's model list, narrowed by its unit's Control Type."""
            index = _unit_index(target_value)
            control_type = filters[index] if index is not None and index < len(filters) else "All"
            available = _models_for_type(control_type)
            return gr.update(choices=available,
                             value=[v for v in (selected or []) if v in available])

        def rebuild(*args):
            """Every row's target list and value list, after a unit changed.

            Which units are activated and which models each one offers are both
            LIVE UI state, so they are tracked rather than snapshotted - and
            both are read as gradio INPUTS, never off the component objects:
            the model dropdown's visible choices were pushed by a gr.update and
            never landed back on the Python object.
            """
            targets = list(args[0:MAX_ROWS])
            selected = list(args[MAX_ROWS:2 * MAX_ROWS])
            rest = args[2 * MAX_ROWS:]
            flags, filters = list(rest[:count]), list(rest[count:])

            available = _target_choices(flags)
            updates = []
            for target_value in targets:
                # A unit that has just been switched off cannot stay selected,
                # and falling back to nothing would make the row silently
                # inert - so it falls back to the first target that IS
                # available. Prompt and LoRA are always in the list.
                value = target_value if target_value in available else available[0]
                updates.append(gr.update(choices=available, value=value))
            for target_value, current in zip(targets, selected):
                updates.append(choices_update(target_value, current, filters))
            loras = _lora_names()
            updates.extend(gr.update(choices=loras) for _ in range(MAX_ROWS))
            return updates

        def rescan_all(*args):
            # Only from the button: get_all_models hashes every control file
            # and list_available_networks walks every LoRA, so neither can ride
            # on a Control Type change.
            global_state = _global_state()
            if global_state is not None:
                global_state.update_controlnet_filenames()
            networks = _networks()
            if networks is not None:
                try:
                    networks.list_available_networks()
                except Exception as exc:
                    logger.warning(f"{WHO}: could not rescan LoRAs ({exc}).")
            return rebuild(*args)

        live_inputs = ([row.target for row in rows]
                       + [row.choices for row in rows]
                       + enables + type_filters)
        live_outputs = ([row.target for row in rows]
                        + [row.choices for row in rows]
                        + [row.lora_pick for row in rows])

        for subscribe, handler in ([(rescan.click, rescan_all)]
                                   + [(c.change, rebuild)
                                      for c in enables + type_filters]):
            subscribe(fn=handler, inputs=live_inputs, outputs=live_outputs,
                      show_progress=False)

        # Per row: picking a unit re-reads only THAT row's lists. The LoRA
        # names are re-read here rather than only at build time - the panel
        # is built once at startup and the host may not have scanned its
        # LoRAs yet, which left every picker permanently empty on a page
        # where nothing else happened to rebuild it.
        for row in rows:
            def on_row(target_value, selected, *values):
                filters = list(values[count:])
                return (choices_update(target_value, selected, filters),
                        gr.update(choices=_lora_names())
                        if target_value == TARGET_LORA else gr.update())

            row.target.change(
                fn=on_row,
                inputs=[row.target, row.choices] + enables + type_filters,
                outputs=[row.choices, row.lora_pick], show_progress=False)

    def _wire_canvases(self, canvases, rows, tab):
        """The page's canvas inventory into every row's canvas and layer
        dropdowns.

        Written by javascript/cnpro_ab.js whenever what it lists changes;
        each write rebuilds the choices of all MAX_ROWS rows and keeps a
        row's current pick when it is still on offer. A pick that is gone
        (the canvas emptied, the layer deleted) is cleared rather than kept
        as a value the dropdown can no longer show - the run would refuse it
        anyway (_setup), and a visibly blank pick says why sooner.
        """

        def rebuild(text, *picks):
            inventory = _parse_inventory(text)
            _CANVAS_INVENTORY[tab] = inventory
            keys = [key for _label, key in _canvas_choices(inventory)]
            selected = list(picks[:MAX_ROWS])
            layers = list(picks[MAX_ROWS:])
            updates = []
            for key in selected:
                updates.append(gr.update(
                    choices=_canvas_choices(inventory),
                    value=key if key in keys else None))
            for key, current in zip(selected, layers):
                offered = _layer_choices(inventory, key)
                updates.append(gr.update(
                    choices=offered,
                    value=current if current in offered else None))
            return updates

        inputs = ([canvases] + [row.canvas for row in rows]
                  + [row.layer for row in rows])
        outputs = [row.canvas for row in rows] + [row.layer for row in rows]
        canvases.change(fn=rebuild, inputs=inputs, outputs=outputs,
                        show_progress=False, queue=False)

        # Picking a canvas narrows THAT row's layer list, from the
        # inventory last written - no round trip through the page.
        for row in rows:
            def on_canvas(key, current):
                offered = _layer_choices(_CANVAS_INVENTORY[tab], key)
                return gr.update(choices=offered,
                                 value=current if current in offered else None)

            row.canvas.change(fn=on_canvas, inputs=[row.canvas, row.layer],
                              outputs=[row.layer], show_progress=False,
                              queue=False)

    def _wire_duel(self, poller, seen, images_row, image_a, image_b,
                   config, trace, grade_rows,
                   interesting_buttons, demo_buttons, capacity_state,
                   run_toggle, resume, rows, row_count, is_img2img,
                   setting_targets, canvas_request, canvas_reply):
        """The poll, the three grade rows, the interesting toggles,
        GOOD/N-GOOD and the count in its label, START/STOP and RESUME -
        and the canvas round trip's request and reply."""
        session = _session(is_img2img)
        tab = "img2img" if is_img2img else "txt2img"
        eid = f"cnpro_ab_{tab}"
        row_inputs = [row_count]
        for row in rows:
            row_inputs.extend(row.args())

        # The page's composites, back into the session - see
        # _Session.deliver_composites. queue=False like the grade buttons:
        # the loop is blocked waiting for exactly this.
        canvas_reply.change(fn=session.deliver_composites,
                            inputs=[canvas_reply], outputs=None,
                            show_progress=False, queue=False)
        # The tick's style channel: a <style> element whose rules paint the
        # PHASE onto chrome a gradio update cannot reach - the inference
        # fill on the interesting toggles, the grade-here nudge's opacity.
        # An HTML component because a style element applies from anywhere
        # in the document, its hidden wrapper included, and the alternative
        # is client JS polling a second channel for the same numbers the
        # tick already carries. See _phase_css for what it says.
        phase_css = gr.HTML(value="", elem_id=f"{eid}_phase_css",
                            elem_classes=["cnpro-ab-phase-css"])

        def tick(last, *row_values):
            """Paint whatever the loop is doing - and ONLY what changed.

            NOTHING here is sent unconditionally, and that is the whole point.
            Re-sending a value gradio already holds still re-renders the block:
            a textbox steals a selection the user was making inside it, and two
            images re-encoded every tick cost more bandwidth than the search
            costs GPU.

            `last` is a per-browser gr.State holding what this client was last
            sent, so the comparison is against what IT has - not against a
            server-side "current", which would be wrong the moment a second
            tab joined.
            """
            # ONE POLL AT A TIME, and this is the whole of the flicker fix.
            #
            # The timer fires on a fixed interval whether or not the last one
            # has been answered, and `queue=False` puts every poll on its own
            # thread - so while the GPU is busy and responses take longer than
            # the interval, several polls are in flight at once. They all read
            # the SAME `last` (it only updates when a response lands), so they
            # all conclude that everything has changed and all of them send a
            # full payload. Gradio replaces the DOM of every output for each
            # one: measured at 126 replacements in 20 seconds across five
            # components while generating, against zero while idle. That is
            # the flicker, and it appeared only during inference for exactly
            # this reason.
            #
            # A poll that arrives while one is being served has nothing to add
            # - the one in flight is already carrying the current state - so
            # it answers with no-ops. `gr.update()` on a gr.State leaves the
            # State alone, so `last` stays truthful.
            if not session.tick_lock.acquire(blocking=False):
                return [gr.update()] * len(tick_outputs)
            try:
                return paint(last, row_values)
            finally:
                session.tick_lock.release()

        def paint(last, row_values):
            # The host briefly resets its sampler counters during render
            # finalization. Feed them through the session's per-render
            # monotone latch before taking the coherent phase snapshot.
            progress = session.track_progress(_inference_progress())
            snap = session.snapshot()
            last = last if isinstance(last, dict) else {}
            # BUSY means "the loop is alive or about to be": a START press
            # that has staged itself and clicked Generate is not `running`
            # until the host picks the job up, and every control below has
            # to read it as running from the press. See _Session.arm.
            busy = bool(snap["running"] or snap["launching"])
            # Retained solver state is what RESUME continues from and what
            # GOOD/N-GOOD sample from with nothing running - a reload, or a
            # restarted server whose disk mirror still holds a taste worth
            # having. Checked only while idle, so a live search never pays
            # the disk read.
            retained = None if busy else _retained_solver(tab)
            resumable = (retained is not None
                         and _solver_matches_rows(retained, row_values))
            # The IMAGES fold away when there is no duel to look at; the
            # rest of the group - the scale, GOOD/N-GOOD, START/STOP,
            # RESUME - is the search's console and is always on screen.
            visible = busy or bool(snap["image_a"])
            # The retained state's own count is what keeps the button's
            # number meaningful on a page that loaded AFTER the search that
            # learned it - a reload, or a restarted server reading its disk
            # mirror.
            capacity = snap["capacity"]
            if capacity is None and isinstance(retained, dict):
                capacity = retained.get("capacity")
            # Zero is meaningful now: every observed configuration can be
            # below par. Do not turn "no supported good population" into a
            # button promising one sample; that was the last UI route by
            # which the least-bad point could be rendered under a GOOD label.
            count = None if capacity is None else max(int(capacity), 0)
            now = {
                "generation": snap["generation"],
                "visible": visible,
                "result": snap["result"],
                "trace": snap["trace"],
                "capacity": count,
                # ONE FLAG FOR THE POLL AND FOR THE BUTTON, because they say
                # the same thing: the timer runs exactly while there is
                # something to paint, and the button reads STOP exactly then
                # too. Note it covers the staged-but-not-started window at
                # both ends - a poll that only woke for `running` would leave
                # the panel showing START until the first duel landed.
                "running": busy,
                # RESUME is offered only when both halves are true - nothing
                # is running, and there is something to continue.
                "can_resume": (not busy) and resumable,
                "css": _phase_css(eid, snap, progress, count),
                # The canvas request, painted while run() waits on it and
                # cleared after - one message per side that needs one.
                "canvas_request": _channel_html(snap["canvas_request"]),
            }

            def when(key, value):
                return gr.update(value=value) if now[key] != last.get(key) else gr.update()

            fresh = now["generation"] != last.get("generation")
            return [
                now,
                (gr.update(visible=now["visible"])
                 if now["visible"] != last.get("visible") else gr.update()),
                gr.update(value=snap["image_a"]) if fresh else gr.update(),
                gr.update(value=snap["image_b"]) if fresh else gr.update(),
                # An empty recipe is only painted while a search is RUNNING:
                # there it means "this session has not recommended anything
                # yet", and leaving the previous session's recipe on screen
                # instead reads as the search not knowing the rows changed -
                # reported with a freshly added LoRA row, whose recommendation
                # kept naming the old two-LoRA setup. Idle, an empty result
                # must not be sent: the box may hold a configuration the user
                # pasted in for Set.
                (when("result", now["result"])
                 if now["result"] or now["running"] else gr.update()),
                when("trace", now["trace"]),
                # The count, into the BUTTON'S LABEL and into the state the
                # press reads - and only when it has actually changed. Both
                # carry the same number for the same reason they always
                # did: the label is what the user aims at, the state is
                # what the click spends, and a press must never render a
                # different count from the one it advertised.
                (gr.update(value=f"{now['capacity']} GOOD",
                           interactive=now["capacity"] > 0)
                 if now["capacity"] is not None
                 and now["capacity"] != last.get("capacity") else gr.update()),
                (now["capacity"] if now["capacity"] is not None
                 else gr.update()),
                # The timer switches ITSELF off. Nothing else knows when the
                # loop ended: run() returns on its own thread, and a handler
                # bound to the Generate click would have to guess whether this
                # script was the one that ran.
                (gr.update(active=now["running"])
                 if now["running"] != last.get("running") else gr.update()),
                when("css", now["css"]),
                # ONE BUTTON, TWO STATES, painted from the session rather
                # than from what the last click meant to do: the loop can
                # also end without a click (Interrupt, a failed generation,
                # the search stopping itself), and a label that tracked the
                # clicks would then read STOP over a search that is gone.
                (gr.update(value="STOP" if now["running"] else "START",
                           variant="stop" if now["running"] else "primary")
                 if now["running"] != last.get("running") else gr.update()),
                (gr.update(interactive=now["can_resume"])
                 if now["can_resume"] != last.get("can_resume")
                 else gr.update()),
                when("canvas_request", now["canvas_request"]),
            ]

        good_button, n_good_button = demo_buttons
        tick_outputs = [seen, images_row, image_a, image_b, config, trace,
                        n_good_button, capacity_state, poller, phase_css,
                        run_toggle, resume, canvas_request]
        poller.tick(fn=tick, inputs=[seen] + row_inputs, outputs=tick_outputs,
                    show_progress=False, queue=False)

        # A disk mirror can belong to rows other than the ones a new page
        # shows. Re-evaluate RESUME whenever any part of the visible space
        # changes; one shared handler reads the whole row list, so no field can
        # be added to a row and silently escape the eligibility decision.
        def refresh_resume(*row_values):
            snap = session.snapshot()
            busy = bool(snap["running"] or snap["launching"])
            retained = None if busy else _retained_solver(tab)
            return gr.update(
                interactive=(not busy)
                and _solver_matches_rows(retained, row_values))

        gr.on(triggers=[component.change for component in row_inputs],
              fn=refresh_resume, inputs=row_inputs, outputs=[resume],
              show_progress=False, queue=False)

        # The Generate button starts the poll. It fires for EVERY generation,
        # including ones this script is not part of - which is why `tick`
        # deactivates itself as soon as it sees no session running, rather
        # than trying to work out here whether this script was selected.
        submit = _submit_button(is_img2img)
        if submit is not None:
            submit.click(fn=lambda: gr.update(active=True), inputs=[],
                         outputs=[poller], show_progress=False, queue=False)
        else:
            logger.warning(
                f"{WHO}: the Generate button was not found, so the panel "
                f"cannot start polling by itself. The search will still run "
                f"and its images will still be saved, but the duels will not "
                f"appear in the panel.")

        def grade(score, disliked=False, similar=None):
            def handler():
                session.grade(score, disliked, similar)
            return handler

        # The three rows carry the same eleven grades and differ only in the
        # verdict the row itself is - see _Session.grade. Wired from one
        # table so the three can never drift apart in what a click means.
        distinct_buttons, similar_buttons, dislike_buttons = grade_rows
        for buttons, verdict in ((distinct_buttons, {"similar": False}),
                                 (similar_buttons, {"similar": True}),
                                 (dislike_buttons, {"disliked": True})):
            for score, button in enumerate(buttons):
                button.click(fn=grade(score, **verdict), inputs=[],
                             outputs=None, show_progress=False,
                             queue=False)

        def toggle(side):
            def handler():
                session.toggle_interesting(side)
            return handler

        for side, button in enumerate(interesting_buttons):
            button.click(fn=toggle(side), inputs=[], outputs=None,
                         show_progress=False, queue=False)

        # GOOD/N-GOOD press two very different machines depending on whether
        # a loop is alive, and the user should never have to know which:
        #
        # * a RUNNING search takes the request directly - queued, served at
        #   the loop's next listen, back to the duel afterwards;
        # * IDLE (search finished, or a session just Imported), the press
        #   ITSELF becomes the generation: it stages the request and clicks
        #   Generate through the hidden trigger box below, and run() sees
        #   the staged entry and renders those samples from the retained
        #   solver state instead of starting duels. That is what makes the
        #   solver a GENERATOR once a good state is reached: the search
        #   ending does not end sampling from it, and everything the rows do
        #   not own - prompt, canvases, resolution, seed - is read fresh
        #   from the UI at that Generate, tweakable at will between presses.
        #
        # The box holds a distinct token per press (a repeated value fires
        # no change event), and the click happens client-side because a
        # gradio handler cannot invoke the host's generation pipeline - the
        # Generate button's own event is the one path that reads every
        # component the pipeline needs.
        #
        # SHARED WITH START AND RESUME, which reach the pipeline the same way
        # and for the same reason. One box rather than three: what the press
        # MEANT is already staged server-side, so the token only has to be
        # new, and three boxes would be three copies of the same three lines
        # of JS.
        demo_go = gr.Textbox(value="", visible=False,
                             elem_id=f"cnpro_ab_{tab}_demo_go")
        demo_go.change(
            fn=None, inputs=[demo_go], outputs=None,
            js="(token) => { if (token) { const b = document.getElementById("
               f"'{tab}_generate'); if (b) b.click(); }} }}")

        def press(request):
            """One GOOD or N-GOOD press, down whichever of the two paths is
            live. Both paths end with the panel saying what it did, because
            the two look identical from the button and only one of them
            produces an image within the second."""
            if session.request_demo(request.count):
                return gr.update()
            payload = _retained_solver(tab)
            if not payload or not payload.get("observations"):
                with session.condition:
                    session.status = (
                        "no solver state to sample from - run a search "
                        "first, or Import a session file that carries one")
                return gr.update()
            _DEMO_SERIAL[tab] += 1
            _PENDING_DEMO[tab].append((request, time.time()))
            with session.condition:
                session.status = (f"generating a {request.label} sample from "
                                  f"the learned taste" if request.count is None
                                  else f"generating a collage of "
                                       f"{request.count} good samples from "
                                       f"the learned taste")
            return gr.update(value=f"go-{_DEMO_SERIAL[tab]}")

        good_button.click(fn=lambda: press(_Demo()), inputs=[],
                          outputs=[demo_go],
                          show_progress=False, queue=False)
        # THE COUNT THE BUTTON WAS ADVERTISING is what the press spends: the
        # state is an INPUT to this click, and the tick writes it in the same
        # breath as the label, so the two cannot disagree about what a press
        # is going to do. Still clamped rather than trusted - the value has
        # crossed a session boundary (an import, a disk mirror written by an
        # older build), and a bad one must be clamped rather than raise in the
        # generation thread.
        n_good_button.click(fn=lambda count: press(_Demo(_collage_count(count))),
                            inputs=[capacity_state],
                            outputs=[demo_go],
                            show_progress=False, queue=False)

        def painted(go=None):
            """The trigger box and poll updates shared by the loop controls.

            The POLL is on this list because a press is the one moment the
            panel knows something is about to happen and the timer may be
            asleep - it switches itself off whenever nothing is running, and
            the tick that would switch it back on is the tick that is not
            firing. Waking it here rather than relying on the Generate click
            also covers the press that never reaches Generate: the panel then
            repaints, sees nothing running, and folds back to START.
            """
            return (gr.update() if go is None else gr.update(value=go),
                    gr.update(active=True))

        setting_tokens = [token for token, _component in setting_targets]
        setting_inputs = [component for _token, component in setting_targets]

        def launch(resume, setting_values, row_values=()):
            """START or RESUME: stage the intent, then click Generate.

            `resume` is decided HERE, at the press, and travels with the
            staged entry - it is not a control run() reads later. That is the
            whole reason the "Resume search" checkbox could go: a checkbox is
            a standing setting that has to be read at Generate time and can
            therefore disagree with the button that was actually pressed,
            while two buttons cannot.
            """
            snap = session.snapshot()
            if snap["running"] or snap["launching"]:
                # The button reads STOP in this state, so getting here means
                # a stale click - a second one in the same instant, or a page
                # whose poll had not repainted yet. Starting a second search
                # is the one thing it must not do.
                return painted()
            retained = _retained_solver(tab) if resume else None
            if resume and not _solver_matches_rows(retained, row_values):
                with session.condition:
                    session.status = (
                        "nothing to resume for these rows - START begins a "
                        "fresh search, or restore the rows that produced the "
                        "retained solver state")
                return painted()
            # Capture at the PRESS, not on the first grade and not from code
            # defaults. These are the exact visible settings Generate is
            # about to read. A successful RESUME replaces this fallback with
            # the original search's saved snapshot inside run().
            session.remember_initial(dict(zip(setting_tokens, setting_values)))
            _START_SERIAL[tab] += 1
            _PENDING_START[tab].append((bool(resume), time.time()))
            session.arm("resuming the search - waiting for the host"
                        if resume else
                        "starting a fresh search - waiting for the host")
            return painted(f"run-{_START_SERIAL[tab]}")

        def on_run_toggle(*setting_values):
            """One button, so one handler, and it reads the session rather
            than a remembered mode: whichever label the browser happens to be
            showing, what the press MEANS is decided by what is actually
            running at the moment it lands."""
            snap = session.snapshot()
            if snap["running"] or snap["launching"]:
                session.request_stop()
                return painted()
            return launch(False, setting_values)

        run_toggle.click(fn=on_run_toggle, inputs=setting_inputs,
                         outputs=[demo_go, poller],
                         show_progress=False, queue=False)
        setting_count = len(setting_inputs)

        def on_resume(*values):
            return launch(True, values[:setting_count],
                          values[setting_count:])

        resume.click(fn=on_resume,
                     inputs=setting_inputs + row_inputs,
                     outputs=[demo_go, poller],
                     show_progress=False, queue=False)

    def _wire_set(self, apply_button, set_good_button, reset_button,
                  config, note, targets, unreachable,
                  rows, row_count, is_img2img, canvas_set):
        """Set, Set GOOD and Reset onto the actual Forge Neo controls.

        Writes to the VISIBLE components, never to the unit gr.State. That is
        the same route an infotext paste takes, and it is the safe one: each
        component's own change handler patches its one field into the State
        (MAINTENANCE invariant - a handler that rebuilds the unit from a
        snapshot reverts whatever landed while it was in flight), and the
        Generate click re-reads every light field from its component anyway,
        which is what makes the sliders among them arrive.

        EVERY OUTCOME IS REPORTED, into a line that is always on screen. This
        button silently did nothing once - its report went to the status line
        inside the duel panel, which is hidden until a search runs - and the
        cost of that was not one button press: the recipe was then copied into
        the PROMPT by hand, and the generations afterwards had nothing to do
        with anything that had been graded. A control that writes to five
        different places has to say which of them it reached.
        """

        tab = "img2img" if is_img2img else "txt2img"
        session = _session(is_img2img)
        target_tokens = [token for token, _component in targets]
        target_outputs = [component for _token, component in targets]
        # Distinct per press: a repeated HTML value is no DOM change, and
        # the page's observer would not see a second Set of the same
        # opacity after the user moved the layer's spinner in between.
        set_serial = [0]

        def no_updates(message):
            return [gr.update(value=message)] \
                + [gr.update() for _ in targets] + [gr.update()]

        def apply_values(values, action):
            updates, applied = [], []
            for token, _component in targets:
                if token not in values:
                    updates.append(gr.update())
                    continue
                updates.append(gr.update(value=_coerce(token, values[token])))
                applied.append(token)

            # Layer opacities go to the PAGE - the layer is the only thing
            # that holds one - through the set channel, and count as
            # applied: the page's console names any it could not reach.
            handed = sorted(token for token in values
                            if RE_LAYER_TOKEN.match(str(token)))
            if handed:
                set_serial[0] += 1
                updates.append(gr.update(value=_channel_html(_layer_message(
                    set_serial[0], {token: values[token] for token in handed}))))
                applied.extend(handed)
            else:
                updates.append(gr.update())

            missing = sorted(set(values) - set(applied))
            report = (f"**{action}** {', '.join(applied)}."
                      if applied else f"**{action} set nothing.**")
            if action == "Reset" and applied:
                report += " Restored the settings captured when this search started."
            if missing:
                # Named, not swallowed: a token nothing here can place is
                # either a newer version's, or a unit this tab does not have,
                # or - the case that actually happened - the host's own prompt
                # box not having been found at build time. All three are
                # things the user has to know before they generate.
                report += (" **Not applied** (nothing here holds them): "
                           + ", ".join(missing) + ".")
                if any(token in unreachable for token in missing):
                    report += (" The console says which box was not found; "
                               f"copy that value by hand into the box above the "
                               f"Generate button.")
            logger.info(f"{WHO}: {action} applied {applied or 'nothing'}"
                        + (f"; could not place {missing}" if missing else ""))
            return [gr.update(value=report)] + updates

        def apply_config(text):
            values = _parse_config_string(text)
            if not values:
                return no_updates("**Nothing to set** - the box above holds "
                                  "no configuration string.")
            return apply_values(values, "Set")

        def remembered_initial():
            with session.condition:
                current = copy.deepcopy(session.initial_settings)
            if current:
                return current
            retained = _retained_solver(tab)
            saved = (retained.get("initial_settings")
                     if isinstance(retained, dict) else None)
            return copy.deepcopy(saved) if isinstance(saved, dict) else None

        row_inputs = [row_count]
        for row in rows:
            row_inputs.extend(row.args())

        def set_good(*row_values):
            payload = _retained_solver(tab)
            if not payload or not payload.get("observations"):
                return [gr.update()] + no_updates(
                    "**Nothing to set GOOD** - run and grade a search first, "
                    "or Import a session carrying solver state.")
            initial = remembered_initial()
            if not initial:
                return [gr.update()] + no_updates(
                    "**Nothing to set GOOD** - this retained search predates "
                    "the initial-settings snapshot. Start a fresh search once "
                    "so its settings can be memorized.")
            try:
                count = int(row_values[0] or 1)
                genes = _build_genes(
                    _read_rows(list(row_values), count), _search())
                if not genes:
                    raise ValueError("the visible search rows describe no values")
            except Exception as exc:
                return [gr.update()] + no_updates(
                    f"**Set GOOD refused** - {exc}.")

            _SET_GOOD_SERIAL[tab] += 1
            sampled, error = _sample_good_configuration(
                payload, genes, initial, _SET_GOOD_SERIAL[tab])
            if sampled is None:
                return [gr.update()] + no_updates(
                    f"**Set GOOD refused** - {error}.")
            recipe = _settings_string(sampled, target_tokens)
            applied = apply_values(sampled, "Set GOOD")
            return [gr.update(value=recipe)] + applied

        def reset_initial():
            initial = remembered_initial()
            if not initial:
                return no_updates(
                    "**Nothing to reset** - start or resume a search first "
                    "so its initial settings can be memorized.")
            return apply_values(initial, "Reset")

        apply_button.click(
            fn=apply_config, inputs=[config],
            outputs=[note] + target_outputs + [canvas_set],
            show_progress=False)
        set_good_button.click(
            fn=set_good, inputs=row_inputs,
            outputs=[config, note] + target_outputs + [canvas_set],
            show_progress=False)
        reset_button.click(
            fn=reset_initial, inputs=[],
            outputs=[note] + target_outputs + [canvas_set],
            show_progress=False)

    def _wire_io(self, export_button, import_button, session_file, note,
                 rows, row_count, headline, targets, canvas_targets,
                 group_count, is_img2img, resume_button):
        """Export and Import: the whole session as one HTML file.

        Export READS the same components Set writes (plus the canvases and
        the search rows) and needs no running search - though when one has
        graded anything, its learned state rides along. Import applies every
        recognised field, repaints the rows' dependent controls (a
        programmatic update fires no change events, so the visibility logic
        is re-run here through _row_visibility), and stages the solver state
        for the next Generate on this tab.
        """
        tab = "img2img" if is_img2img else "txt2img"
        session = _session(is_img2img)

        export_inputs = [row_count]
        for row in rows:
            export_inputs.extend(row.args())
        export_inputs.extend(component for _token, component in targets)
        export_inputs.extend(component for _token, component in canvas_targets)

        def do_export(*values):
            it = iter(values)
            count = int(next(it) or 1)
            row_values = [{key: _jsonable(next(it)) for key in ROW_FIELDS}
                          for _ in range(MAX_ROWS)]
            settings = {token: _jsonable(next(it)) for token, _c in targets}
            canvases = {}
            for token, _component in canvas_targets:
                uri = _image_to_data_uri(next(it))
                if uri:
                    canvases[token] = uri
            with session.condition:
                solver = session.solver_state
                result = session.result
            # NO "tail" KEY ANY MORE - the two switches it carried are gone.
            # A file written by an older build still has one and is still
            # read: nothing looks for it, so it rides along in the island as
            # a field this build does not use. See do_import.
            payload = {
                "app": WHO, "version": EXPORT_VERSION, "tab": tab,
                "row_count": count, "rows": row_values,
                "settings": settings,
                "canvases": canvases, "solver": solver, "result": result,
            }
            path = os.path.join(tempfile.gettempdir(),
                                f"cnpro_ab_{tab}.html")
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(_export_html(payload))
            except OSError as exc:
                return (gr.update(),
                        gr.update(value=f"**Export failed**: {exc}"))
            solved = (f", solver state at {solver['duels']} duel(s)"
                      if solver else "")
            logger.info(f"{WHO}: exported the session to {path}")
            return (gr.update(value=path),
                    gr.update(value=f"**Exported** {count} row(s), "
                                    f"{len(settings)} setting(s), "
                                    f"{len(canvases)} canvas image(s)"
                                    f"{solved}. Download above."))

        export_button.click(fn=do_export, inputs=export_inputs,
                            outputs=[session_file, note],
                            show_progress=False)

        import_outputs = [note, row_count, headline]
        import_outputs.extend(row.container for row in rows)
        for row in rows:
            import_outputs.extend(row.import_components())
        import_outputs.extend(component for _token, component in targets)
        import_outputs.extend(component for _token, component in canvas_targets)
        # LAST: RESUME. An import that carries solver state makes it pressable,
        # and it has to be done HERE rather than left to the tick because the
        # poll switches itself off whenever nothing is running: on an idle
        # panel the next tick is the one that never comes.
        import_outputs.append(resume_button)

        def no_ops(message):
            return [gr.update(value=message)] \
                + [gr.update() for _ in import_outputs[1:]]

        def do_import(file_value):
            if isinstance(file_value, (list, tuple)):
                file_value = file_value[0] if file_value else None
            path = (file_value.get("path")
                    if isinstance(file_value, dict) else file_value)
            path = getattr(path, "name", path)
            if not path:
                return no_ops("**Nothing to import** - pick a session file "
                              "above first.")
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = _parse_export(handle.read())
            except OSError as exc:
                return no_ops(f"**Import failed**: {exc}")
            if payload is None:
                return no_ops("**Not a CNPro A/B session file** - no state "
                              "island found in it.")

            count = min(max(int(payload.get("row_count", 1)), 1), MAX_ROWS)
            updates = [gr.update(), count, gr.update(value=_headline(count))]
            resume_rows = [count]

            for index in range(MAX_ROWS):
                updates.append(gr.update(visible=index < count))

            # Which units the imported settings enable decides what the
            # target dropdowns may offer - plus the imported target itself,
            # so a row aimed at a unit this build cannot see still shows
            # what it meant instead of blanking.
            settings = payload.get("settings", {})
            flags = [str(settings.get(f"u{i}.enabled", False)) == "True"
                     for i in range(group_count)]
            available = _target_choices(flags)
            imported_rows = payload.get("rows", [])
            for index in range(MAX_ROWS):
                data = (imported_rows[index]
                        if index < len(imported_rows) else None)
                if not isinstance(data, dict):
                    updates.extend(gr.update()
                                   for _ in rows[index].import_components())
                    resume_rows.extend([None] * ROW_ARGS)
                    continue
                # Files from before the mode chooser slimmed down: "Setting"
                # rows only ever varied the model in practice and "Profile
                # point" is the singular spelling of the same mode - both
                # map forward rather than blanking the dropdown.
                mode_value = {"Setting": MODE_MODEL,
                              "Profile point": MODE_POINTS}.get(
                    data.get("mode"), data.get("mode"))
                props = _row_visibility(data.get("target"), mode_value)
                target_value = data.get("target")
                choices = (available + [target_value]
                           if target_value and target_value not in available
                           else available)
                selected = [v for v in (data.get("choices") or [])]
                updates.append(gr.update(choices=choices, value=target_value))
                updates.append(gr.update(**{**props[0],
                                            "value": mode_value}))
                # The value list's OPTIONS live in this build's model
                # registry; the imported selection is offered as-is so it
                # renders even where the registry differs - rescan restores
                # the full list.
                updates.append(gr.update(**{**props[1], "choices": selected,
                                            "value": selected}))
                updates.append(gr.update(**{**props[2],
                                            "value": data.get("line")}))
                updates.append(gr.update(**{**props[3],
                                            "value": data.get("values")}))
                updates.append(gr.update(**{**props[4],
                                            "value": data.get("prompt_mode")}))
                updates.append(gr.update(**props[5]))          # lora_pick
                updates.append(gr.update(**{**props[6],
                                            "value": data.get("loras")}))
                # A v1 file has no "weights" (it had a min/max pair and a
                # panel-global step); the box is left as it is - empty means
                # the default grid, which is the closest honest reading.
                weights_value = ({"value": data["weights"]}
                                 if isinstance(data.get("weights"), str)
                                 else {})
                updates.append(gr.update(**{**props[7], **weights_value}))
                updates.append(gr.update(**props[8]))          # lora refresh
                updates.append(gr.update(**props[9]))          # lora dice
                # "points" (v2, a list as text) or the old single "point";
                # files with neither predate the mode and 0 is its neutral.
                points_value = data.get("points",
                                        str(data.get("point", 0)))
                updates.append(gr.update(**{**props[10],
                                            "value": str(points_value)}))
                # The canvas and layer picks (v3), offered AS IMPORTED so
                # they render before the page has listed its canvases; the
                # inventory's next write re-offers whatever is really there
                # (_wire_canvases) and blanks a pick that is not. A key an
                # older file never wrote, or one this build does not parse,
                # is left blank.
                canvas_value = data.get("canvas")
                if not RE_CANVAS_KEY.match(str(canvas_value or "")):
                    canvas_value = None
                layer_value = data.get("layer")
                if _layer_index(layer_value) is None:
                    layer_value = None
                updates.append(gr.update(**{
                    **props[11],
                    "choices": ([(_canvas_label(canvas_value), canvas_value)]
                                if canvas_value else []),
                    "value": canvas_value}))
                updates.append(gr.update(**{
                    **props[12],
                    "choices": [layer_value] if layer_value else [],
                    "value": layer_value}))
                resume_rows.extend([
                    target_value, mode_value, selected, data.get("line"),
                    data.get("values"), data.get("prompt_mode"),
                    data.get("loras"), data.get("weights"),
                    str(points_value), canvas_value, layer_value])

            # A v2 file's "tail" - the two switches this panel no longer has
            # - is deliberately not read: there is nothing to put it in.

            applied = []
            for token, _component in targets:
                if token in settings:
                    updates.append(gr.update(value=_coerce(token,
                                                           settings[token])))
                    applied.append(token)
                else:
                    updates.append(gr.update())
            missing = sorted(set(settings) - set(applied))

            canvases = payload.get("canvases", {})
            painted = 0
            for token, _component in canvas_targets:
                uri = canvases.get(token)
                if uri:
                    image = _data_uri_to_image(uri)
                    painted += 1 if image is not None else 0
                    updates.append(gr.update(value=image))
                else:
                    # Absent means EMPTY at export time, and reproduction
                    # includes the emptiness - a leftover canvas would
                    # silently change what every duel generates.
                    updates.append(gr.update(value=None))

            solver = payload.get("solver")
            if solver:
                _PENDING_SOLVER[tab] = solver
                if isinstance(solver.get("initial_settings"), dict):
                    session.remember_initial(solver["initial_settings"])
            retained = _retained_solver(tab)
            can_resume = _solver_matches_rows(retained, resume_rows)
            note_parts = [f"**Imported** {count} row(s), {len(applied)} "
                          f"setting(s), {painted} canvas image(s)."]
            if solver:
                if can_resume:
                    note_parts.append(
                        f"Solver state ({solver.get('duels', 0)} duel(s)) is "
                        f"staged - press RESUME in the panel above to "
                        f"continue that search, or GOOD/N-GOOD to render "
                        f"samples from it without searching.")
                else:
                    note_parts.append(
                        f"Solver state ({solver.get('duels', 0)} duel(s)) is "
                        f"staged for GOOD/N-GOOD, but its search rows do not "
                        f"match this file, so RESUME is unavailable.")
            if missing:
                note_parts.append("**Not applied** (nothing here holds "
                                  "them): " + ", ".join(missing) + ".")
            if int(payload.get("version", 0)) > EXPORT_VERSION:
                note_parts.append(
                    "The file is from a NEWER version - anything it could "
                    "not hand over is in the file itself: open it in a "
                    "browser and copy the rest by hand.")
            logger.info(f"{WHO}: imported a session file"
                        + (f"; could not place {missing}" if missing else ""))
            updates[0] = gr.update(value=" ".join(note_parts))
            # Imported solver state is always something GOOD/N-GOOD can
            # project onto edited rows. RESUME is stricter: it goes live only
            # when that state describes the exact rows imported beside it.
            if solver:
                with session.condition:
                    session.status = (
                        "solver state imported - RESUME continues that "
                        "search, GOOD/N-GOOD render samples from it"
                        if can_resume else
                        "solver state imported for GOOD/N-GOOD, but its "
                        "search rows do not match this file, so RESUME is "
                        "unavailable")
            updates.append(gr.update(interactive=can_resume))
            return updates

        import_button.click(fn=do_import, inputs=[session_file],
                            outputs=import_outputs, show_progress=False)

    # -- the search ------------------------------------------------------

    def _setup(self, p, args, ab_search):
        """Rows, space and base units out of the run's arguments.

        Everything in here can raise about a misconfiguration, which is why
        run() calls it BEFORE deciding whether this Generate is a search or
        a GOOD/N-GOOD sample render: the two want different error handling - a
        search wants the stack trace, a sample press wants a handled refusal,
        and both want the staged request consumed either way.
        """
        row_count = int(args[0] or 1)
        rows = _read_rows(args, row_count)
        genes = _build_genes(rows, ab_search)
        if not genes:
            raise RuntimeError(
                f"{WHO}: no row declares anything to vary, so there is nothing "
                f"to search. Give at least one row a list of values.")

        space = _space(genes, ab_search)
        if len(space.live) < 1:
            raise RuntimeError(
                f"{WHO}: every row offers exactly one value, so the search "
                f"space holds a single configuration and no comparison is "
                f"possible. Give at least one row a second value.")

        cnpro = _cnpro_script(p)
        needs_units = any(gene.on_unit for gene in genes)
        if needs_units and (cnpro is None or cnpro.args_from is None):
            raise RuntimeError(
                f"{WHO}: a row varies a ControlNet unit, but the ControlNet "
                f"Pro script is not running. Enable the extension, or put "
                f"every row on Prompt or LoRA.")

        base_args = list(p.script_args)
        base_units = {}
        for gene in genes:
            if not gene.on_unit or gene.unit_index in base_units:
                continue
            slot = cnpro.args_from + gene.unit_index
            if slot >= cnpro.args_to:
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} targets unit "
                    f"{gene.unit_index}, which does not exist.")
            unit = base_args[slot]
            if not getattr(unit, "enabled", False):
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} targets unit "
                    f"{gene.unit_index}, which is not enabled. A disabled "
                    f"unit is dropped before the search is applied, so every "
                    f"duel would be identical. Enable it first.")
            base_units[gene.unit_index] = unit

        # A Canvas layer row, against what the page last listed and what
        # the generation holds - before a duel is spent on a canvas that
        # emptied, a layer that was deleted, or a unit that is off. The page
        # checks again when asked to composite (the inventory is a second
        # old at most), but that answer arrives inside the loop, where it
        # can only stop the search rather than refuse the start.
        inventory = {entry["key"]: entry
                     for entry in _CANVAS_INVENTORY[
                         "img2img" if self.is_img2img else "txt2img"]}
        for gene in genes:
            if gene.kind != Gene.KIND_LAYER:
                continue
            entry = inventory.get(gene.canvas)
            if entry is None:
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} varies a layer of "
                    f"{_canvas_label(gene.canvas)}, which holds no image "
                    f"(or the page has not listed it yet). Put an image "
                    f"there, or pick another canvas.")
            if not 0 <= gene.layer < len(entry["layers"]):
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} varies layer "
                    f"{gene.layer + 1} of {_canvas_label(gene.canvas)}, "
                    f"which has {len(entry['layers'])} layer(s).")
            where = _canvas_unit_slot(gene.canvas)
            if where is None:
                if not self.is_img2img:
                    raise RuntimeError(
                        f"{WHO}: row {gene.row + 1} varies the img2img "
                        f"canvas, which the txt2img tab does not generate "
                        f"from.")
                continue
            unit_index, slot = where
            if cnpro is None or cnpro.args_from is None:
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} varies a ControlNet unit's "
                    f"canvas, but the ControlNet Pro script is not running.")
            if cnpro.args_from + unit_index >= cnpro.args_to:
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} targets unit {unit_index}, "
                    f"which does not exist.")
            unit = base_args[cnpro.args_from + unit_index]
            if not getattr(unit, "enabled", False):
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} varies a canvas of unit "
                    f"{unit_index}, which is not enabled. Enable it first.")
            if getattr(unit, _input_field(slot), None) is None:
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} varies unit {unit_index}'s "
                    f"Input {slot}, which reached the generation empty.")
        module = _profile_scale()
        for gene in genes:
            if gene.kind in (Gene.KIND_PROFILE, Gene.KIND_POINT) \
                    and module is not None:
                module.warn_if_inert(base_units[gene.unit_index],
                                     gene.unit_index, gene.line, who=WHO)
            if gene.kind == Gene.KIND_POINT and module is not None:
                # Before any duel is spent: a bad index would leave every
                # offset falling on no point - a whole dimension of
                # identical images, warned about once per render where
                # nobody connects it back to the row. The group moves
                # together, so every listed index has to name an existing
                # point, and no two may name the SAME one (0 and -N on an
                # N-point line are the same knot spelled twice - it would
                # move by double the offset the row promises).
                count = module.profile_point_count(
                    base_units[gene.unit_index], gene.line)
                resolved = set()
                for point_index in gene.point_indices:
                    if not (-count <= point_index < count):
                        raise RuntimeError(
                            f"{WHO}: row {gene.row + 1} offsets point "
                            f"{point_index} of unit {gene.unit_index}'s "
                            f"{gene.line} profile, which has {count} "
                            f"point(s) - indices 0..{count - 1}, or negative "
                            f"from the end. Pick existing points.")
                    knot = point_index if point_index >= 0 \
                        else point_index + count
                    if knot in resolved:
                        raise RuntimeError(
                            f"{WHO}: row {gene.row + 1} names the same drawn "
                            f"point twice ({point_index} aliases an index "
                            f"already listed on this {count}-point line) - "
                            f"it would move by double the offset.")
                    resolved.add(knot)
        return genes, space, cnpro, base_args, base_units

    def run(self, p, *args):
        session = _session(self.is_img2img)
        tab = "img2img" if self.is_img2img else "txt2img"
        # WHAT THE PANEL ASKED FOR, CONSUMED FIRST - before the rows are even
        # looked at, and before anything can raise. Two reasons, one for each
        # list:
        #
        # * an entry left behind by a run that died in validation used to
        #   hijack a later, unrelated Generate into a one-sample render -
        #   "the search stopped working", long after the press that caused
        #   it. Popping here means a failed run consumes its own request.
        # * a press stages its intent and clicks Generate, so reaching run()
        #   at all is the evidence that the wait is over - launched or not.
        demo = self._pop_demo_request()
        # ONE GENERATION SERVES ONE PRESS, so a Generate already claimed by a
        # GOOD/N-GOOD press does not also swallow a START staged a moment
        # later: that press queued a generation of its own, and this is not
        # it. Left staged, the start is served by the next one.
        start = None if demo is not None else self._pop_start_request()
        session.disarm()

        # NOBODY PRESSED ANYTHING IN THIS PANEL, so this is the host's own
        # Generate and it stays the host's own Generate: returning None hands
        # the job back and Forge renders the current setup exactly as it
        # would with no script selected. Selecting this script is how the
        # panel is opened, not a decision to spend the next press on a
        # search - see HOW THE SEARCH IS STARTED AND STOPPED.
        #
        # BEFORE every check below, including the ones that raise: a plain
        # Generate must not be refused for a search's reasons - missing
        # numpy, absent CNPro, or rows that do not describe a search yet.
        if demo is None and start is None:
            return None

        if len(args) < ARG_COUNT:
            # ui() returned nothing, which it only does when CNPro is absent.
            raise RuntimeError(
                f"{WHO}: ControlNet Pro is not loaded, so there is nothing to "
                f"search over. Enable the extension and restart.")
        ab_search = _search()
        if ab_search is None:
            raise RuntimeError(
                f"{WHO}: the search engine could not be imported - it needs "
                f"numpy, which the host normally provides. See the console.")

        try:
            genes, space, cnpro, base_args, base_units = self._setup(
                p, args, ab_search)
        except Exception as exc:
            if demo is None:
                # Clear START's armed state before the traceback goes up, or
                # the button would keep claiming a search exists when setup
                # never began.
                session.finish(f"refused - {exc}")
                raise
            # The press asked for a sample, not for a stack trace. Rows that
            # do not validate - or no rows at all, on a freshly started UI -
            # mean the solver's space cannot be rebuilt here.
            session.start_demo("")
            session.finish(f"sample refused - {exc} GOOD/N-GOOD need the rows "
                           f"the solver was trained on: recreate them, or "
                           f"Import the session file that carries them.")
            logger.warning(f"{WHO}: a GOOD/N-GOOD sample was requested but the "
                           f"rows did not validate ({exc})")
            return Processed(p, [])

        # ONE SEED FOR THE WHOLE SEARCH. A comparison on two different seeds
        # compares the seeds, and a search on a different seed per duel
        # compares each duel against a different noise field - so the seed in
        # the host's box is resolved once here and every render of this run
        # uses it. It is also the render cache's key material: a fixed seed is
        # what lets a side already shown be shown again for a lookup instead
        # of a generation (see RENDER_CACHE and search.reuse_probe).
        processing.fix_seed(p)
        base_seed = int(p.seed)
        base_prompt = p.prompt
        base_extras = _generation_extras(p)
        # What Reset restores for a Canvas layer row: the opacity the page
        # listed for that layer, which the press could not capture because
        # no component holds it. `extend_initial` never overwrites - a
        # RESUME's own saved snapshot stays the one that counts.
        session.extend_initial(_layer_initial_settings(genes, tab))

        # A GOOD/N-GOOD press with no search running: render those samples
        # from the retained solver state. The search does not restart,
        # nothing is asked, and the state is left exactly as it was - for the
        # next press, or for the next real search.
        if demo is not None:
            return self._demo_run(demo, p, ab_search, space, genes, base_args,
                                  cnpro, base_units, base_prompt, base_seed,
                                  session)

        search = ab_search.PreferenceSearch(space, seed=base_seed)
        # WHICH BUTTON WAS PRESSED IS THE WHOLE OF THIS DECISION. RESUME
        # continues, in order of intent: a solver state staged by Import is
        # the one source the user pointed at explicitly, then the tab's own
        # retained state - the last search here, or the disk mirror that
        # survived a restart. START continues nothing, and leaves staged
        # state STAGED (for a later RESUME) rather than consuming it.
        #
        # Either way the state only ever resumes onto the SAME search space:
        # observations replayed onto different rows would silently reindex
        # every choice, which is worse than starting over, so a mismatch is
        # refused.
        staged, source = None, "imported"
        if start:
            # Peek first. A failed compatibility check must not consume an
            # imported state: after the user restores its rows, that same
            # explicit import is still the state RESUME should use.
            staged = _PENDING_SOLVER.get(tab)
            if staged is None:
                staged = session.solver_state or _stored_solver(tab)
                source = "retained"
        resumed = 0
        if staged is not None:
            restored = _restore_solver(search, staged, space)
            if restored is None:
                message = (
                    f"resume refused - the {source} solver state belongs to "
                    f"different search rows; restore those rows or press "
                    f"START for a fresh search")
                session.finish(message)
                logger.warning(f"{WHO}: {message}.")
                return Processed(p, [])
            else:
                if source == "imported" \
                        and _PENDING_SOLVER.get(tab) is staged:
                    _PENDING_SOLVER.pop(tab, None)
                resumed = restored
                saved_initial = staged.get("initial_settings")
                if isinstance(saved_initial, dict) and saved_initial:
                    # A real RESUME belongs to the original search, so Reset
                    # must keep pointing at that search's beginning. The
                    # current values captured at the press are only the
                    # fallback for legacy payloads.
                    session.remember_initial(saved_initial)
                logger.info(f"{WHO}: resumed the {source} search - "
                            f"{restored} observation(s), "
                            f"{search.duels} graded duel(s).")

        p.extra_generation_params["Script"] = self.title()
        # THE SEARCH HAS NO LENGTH, so this number is a placeholder and the
        # progress bar reads as one. "Until you say stop" has no total, and
        # job_count is what the host divides by - a small number would show a
        # bar that fills and then keeps going, which is worse than one that
        # crawls. The duel number in the panel is the honest progress report.
        state.job_count = 2 * 100
        size = space.size
        logger.info(
            f"{WHO}: {len(space.live)} degree(s) of freedom over "
            f"{size if size is not None else 'continuously many'} "
            f"configurations, asking until stopped.")

        best = space.baseline()
        result = ""
        last = None
        stopped = "stopped"
        duel = 0
        renders = {}     # recipe -> Processed; see RENDER_CACHE
        demos = []       # GOOD samples rendered on demand, last 8 kept
        # LAST, so that nothing between here and the loop can raise with the
        # session marked running: the panel polls while it is, and a session
        # that is never finished is a panel that polls until the page is
        # reloaded. Every exit below goes through `finish`.
        #
        # `resumed` is also what decides whether the panel's RECORD survives:
        # a search continuing from retained observations keeps the Tried list
        # that describes them - see _Session.start.
        session.start(resumed=bool(resumed))
        if resumed:
            # A resumed search already knows things: say so before the first
            # duel, or the panel opens looking like a fresh start.
            best = search.best()
            result = _config_string(genes, best, base_units, base_prompt,
                                    base_seed, extras=base_extras)
            session.set_result(result, _summary(genes, best, search))
            session.record("↻", f"resumed the {source} search: "
                                f"{resumed} observation(s), "
                                f"{search.duels} graded duel(s)")
            payload = _solver_payload(
                search, base_seed, space, session.initial_settings)
            with session.condition:
                session.solver_state = payload
            session.set_capacity(payload["capacity"])
            _store_solver(tab, payload)

        # What is REUSABLE for the duel about to be chosen: a side whose
        # recipe is in the render cache costs a lookup instead of a ~30s
        # generation, and the engine's reuse preference (see
        # ab_search.REUSE_MARGIN) trades on that. Wired ONCE rather than per
        # duel: the recipe is the cache key, the seed is part of the recipe
        # and it does not move, so the only thing that changes under this
        # closure is `renders` - which it reads live.
        search.reuse_probe = lambda pt: (
            _config_string(genes, pt, base_units, base_prompt,
                           base_seed, extras=base_extras) in renders)
        try:
            while True:
                if session.stopping or state.interrupted \
                        or getattr(state, "stopping_generation", False):
                    break

                point_a, point_b = search.next_duel()
                duel += 1
                session.say(f"duel {duel}: generating A", generating="A")
                state.job = f"DNA duel {duel} - A"
                processed_a = self._render(p, base_args, cnpro, genes, point_a,
                                           base_units, base_prompt, base_seed,
                                           side="A", session=session,
                                           cache=renders, extras=base_extras)
                if session.stopping or state.interrupted:
                    break
                session.say(f"duel {duel}: generating B", generating="B")
                state.job = f"DNA duel {duel} - B"
                processed_b = self._render(p, base_args, cnpro, genes, point_b,
                                           base_units, base_prompt, base_seed,
                                           side="B", session=session,
                                           cache=renders, extras=base_extras)
                if session.stopping or state.interrupted:
                    break
                if not processed_a.images or not processed_b.images:
                    logger.error(f"{WHO}: a duel produced no image; stopping.")
                    stopped = "stopped - a generation failed"
                    break

                last = (processed_a, processed_b)
                # The censoring probe SHOWS a pair it expects to look
                # identical - that is the whole question it is asking (see
                # ab_search._probe_duel). Unannounced it reads as the search
                # having rendered the same image twice, which is a bug
                # report, so the one duel that needs a word gets one.
                session.publish(
                    duel, processed_a.images[0], processed_b.images[0],
                    "waiting for your grade"
                    if search.last_tactic != "probe" else
                    "waiting for your grade - these two are MEANT to look "
                    "alike: it is re-checking a difference you called "
                    "invisible, so grade the row you actually see")
                answer = session.await_grade()
                # GOOD/N-GOOD are not answers: render fresh guesses from the
                # good end of the solver's belief - pure inference, no
                # observation, no effect on the duel - then return to the
                # SAME duel, still waiting. The full recipes go to the
                # trace; the images go to the output folder now and the
                # gallery at stop.
                #
                # THE DUEL STAYS ON SCREEN, which is why nothing is
                # published here: the question is still unanswered, and
                # painting a collage over it would cost the user the pair
                # they were in the middle of judging. A press mid-search is
                # a look at the model, not a change of subject.
                while isinstance(answer, _Demo):
                    made, note = self._samples(
                        answer, search, p, base_args, cnpro, genes,
                        base_units, base_prompt, base_seed, session,
                        base_extras, cache=renders)
                    demos.extend(made)
                    del demos[:-8]
                    if session.stopping or state.interrupted:
                        break
                    session.publish(duel, processed_a.images[0],
                                    processed_b.images[0],
                                    f"{note} (recipes in Tried, images in the "
                                    f"output folder) - back to the duel, "
                                    f"waiting for your grade")
                    answer = session.await_grade()
                if answer is None:
                    break

                score, disliked, similar, (mark_a, mark_b) = answer
                search.observe(point_a, point_b, score, disliked=disliked,
                               interesting_a=mark_a, interesting_b=mark_b,
                               similar=similar)
                best = search.best()
                result = _config_string(genes, best, base_units, base_prompt,
                                        base_seed, extras=base_extras)
                session.set_result(result, _summary(genes, best, search))
                # Every grade lands in the retained state AND its disk
                # mirror, so nothing that can end this loop - STOP, an
                # Interrupt, a crash, even a restart - costs more than the
                # interruption itself.
                payload = _solver_payload(
                    search, base_seed, space, session.initial_settings)
                with session.condition:
                    session.solver_state = payload
                # AFTER every answer, which is what makes the button's count
                # a live reading of the search rather than a figure fixed at
                # the start: the good region is what each grade redraws.
                session.set_capacity(payload["capacity"])
                _store_solver(tab, payload)
        except Exception as exc:
            errors.display(exc, "running the CNPro A/B search")
            stopped = f"stopped - {exc}"

        interrupted = state.interrupted or getattr(state, "stopping_generation", False)
        if interrupted:
            stopped = "interrupted - RESUME continues the search"
        graded = len(search.observations)
        ending = f"{stopped}, {graded} graded duel{'' if graded == 1 else 's'}"

        if not graded:
            # Nothing was answered, so there is no recommendation - only the
            # baseline the search would have started from, which is not an
            # answer and must not be printed as one.
            session.finish(f"{ending} - nothing to recommend")
            logger.warning(f"{WHO}: no duel was graded, so there is nothing to "
                           f"recommend.")
            return Processed(p, [])

        logger.info(f"{WHO}: {result}")
        p.extra_generation_params["CNPro DNA"] = result

        # The search's real output is not one point, it is the FRONTIER: the
        # diverse top of the posterior, up to a handful of configurations
        # that are each polished and each visibly different from the others.
        # Taste is not unimodal, and "the answer" being singular would throw
        # away every basin except the one that happened to be ahead at STOP.
        # Each keeper goes into the trace as a full recipe - the box the
        # frontier members' recipes can be COPIED out of, since all but the
        # champion appear nowhere else. RECIPES, not renders: a stop renders
        # nothing (it used to trigger up to four unasked generations), and
        # any keeper is one Set-and-Generate away when it is wanted.
        keepers = search.frontier()
        for rank, keeper in enumerate(keepers, 1):
            recipe = _config_string(genes, keeper, base_units, base_prompt,
                                    base_seed, extras=base_extras)
            session.record(f"★{rank}", recipe)
            logger.info(f"{WHO}: keeper {rank}: {recipe}")

        session.finish(ending)

        # Nothing new was rendered, so the last duel is what the gallery gets -
        # a search that ends with an empty gallery looks like a search that
        # failed, and the recommendation in the box would be the only evidence
        # otherwise.
        if last is None and not demos:
            return Processed(p, [])
        sources = ([last[0], last[1]] if last is not None else []) + demos
        merged = copy.copy(sources[0])
        merged.images = [s.images[0] for s in sources]
        merged.infotexts = [s.infotexts[0] for s in sources]
        merged.all_prompts = [s.all_prompts[0] for s in sources]
        merged.all_seeds = [s.all_seeds[0] for s in sources]
        return merged

    def _pop_demo_request(self):
        """The oldest fresh staged GOOD/N-GOOD request for this tab, or None.

        Stale entries are dropped with a log line - see DEMO_STALE_SECONDS:
        a press stages its entry and clicks Generate in the same instant, so
        an old survivor means that click never became a run, and honouring
        it would turn a later, unrelated Generate into a sample render the
        user did not ask for.
        """
        staged = _PENDING_DEMO["img2img" if self.is_img2img else "txt2img"]
        while staged:
            request, when = staged.pop(0)
            if time.time() - when <= DEMO_STALE_SECONDS:
                return request
            logger.warning(
                f"{WHO}: dropped a stale GOOD request - the press that "
                f"staged it never became a run, so this Generate is treated "
                f"as the plain generation it looks like.")
        return None

    def _pop_start_request(self):
        """Whether this generation is a search, and whether it RESUMES.

        None when the panel staged nothing, which is the common case and the
        one that keeps Generate the host's own button - see run(). Otherwise
        True for RESUME and False for START, decided at the press rather than
        read off a control here.

        Stale entries are dropped exactly as the demo ones are, and the cost
        of not doing so is higher: a survivor would turn some later,
        unrelated Generate into a search that starts asking for grades.
        """
        staged = _PENDING_START["img2img" if self.is_img2img else "txt2img"]
        while staged:
            resume, when = staged.pop(0)
            if time.time() - when <= DEMO_STALE_SECONDS:
                # THE REST GO WITH IT. Several entries mean several presses
                # landed before any of them became a run, and they all asked
                # for the same single thing - one search. Left in the list
                # they would each start another one as the queue drained.
                del staged[:]
                return resume
            logger.warning(
                f"{WHO}: dropped a stale START/RESUME press - the click that "
                f"staged it never became a run, so this Generate is treated "
                f"as the plain generation it looks like.")
        return None

    def _samples(self, request, search, p, base_args, cnpro, genes,
                 base_units, base_prompt, base_seed, session, extras,
                 cache=None, publish=None):
        """The images ONE press of GOOD or N-GOOD asks for.

        Returns `(made, note)`: the Processed objects the caller should carry
        into its gallery, and one line saying what came back. Shared by the
        two paths a press can take - queued into a running search's loop, or
        an idle Generate that the press started itself - because the only
        thing that differs between them is where the images are painted,
        which is what `publish` is for.

        FEWER SAMPLES THAN ASKED FOR IS A RESULT, not an error. `population`
        will not pad a collage with near-duplicates or with configurations
        it does not believe in (see ab_search.population), so a request for
        forty against a solver that knows six good answers comes back with
        six - and the note says so, because the difference between "six is
        all there is" and "something went wrong" is the whole value of the
        number in the box.

        THE NOTE CARRIES THE PROMISE THE COLLAGE MAKES: how many separate
        good regions the entries came from, and the model's own lowest
        probability that any two of them look different. Both come from
        `population_report`, and printing them is not decoration - a sheet
        that claims distinct samples and delivers near-duplicates is the one
        failure of this button that costs an hour and looks like success, so
        the claim is stated in a form the next glance at the sheet can
        falsify.
        """
        report = {}
        wanted = None if request.count is None else min(request.count,
                                                        COLLAGE_MAX)
        if wanted is None:
            points = [search.suggest(good=True)]
            cache_store = True
        else:
            # Clamped HERE, where the generations are actually spent, and not
            # only in the box's parser: this is the one place that knows a
            # number is about to become GPU minutes.
            points = search.population(wanted)
            report = dict(getattr(search, "population_report", None) or {})
            # READ THE DUEL CACHE, DO NOT FILL IT. A population is mutually
            # distinct, not necessarily unseen: its island peaks and observed
            # high-quality members are often images the user already paid for,
            # and reusing those is ~20x cheaper than generating them again.
            # But inserting up to 64 fresh collage entries into a 32-entry LRU
            # would evict every duel image and destroy the search's reuse
            # economy. A read-only cache is both halves of the requirement.
            cache_store = False
        made = []
        for index, point in enumerate(points, 1):
            if session.stopping or state.interrupted \
                    or getattr(state, "stopping_generation", False):
                break
            recipe = _config_string(genes, point, base_units, base_prompt,
                                    base_seed, extras=extras)
            session.record(f"?{request.label}", recipe)
            state.job = (f"DNA {request.label} sample" if wanted is None
                         else f"DNA collage {index}/{len(points)}")
            session.say(
                (f"rendering {request.label} sample"
                 if wanted is None else
                 f"rendering {request.label} sample {index} of {len(points)}"),
                generating="A")
            shown = self._render(p, base_args, cnpro, genes, point,
                                 base_units, base_prompt, base_seed,
                                 session=session, cache=cache,
                                 cache_store=cache_store, extras=extras)
            if not shown.images:
                continue
            session.render_complete("A")
            made.append(shown)
            if publish is not None:
                publish(index, len(points), shown.images[0])

        if not made:
            return [], (f"nothing rendered - the solver has no good "
                        f"configuration to offer yet")
        if wanted is None:
            return made, "GOOD sample rendered"

        # The collage replaces its components in the gallery - see _collage.
        sheet = _collage([shown.images[0] for shown in made])
        if sheet is None:
            return made, f"{len(made)} good sample(s) rendered"
        _save_collage(p, sheet, base_seed, made[0].infotexts[0])
        collage = copy.copy(made[0])
        collage.images = [sheet]
        collage.infotexts = [made[0].infotexts[0]]
        # A short collage has two very different causes and they must not
        # read alike: the solver honestly having fewer distinct good answers
        # than were asked for, and the run having been cut off part way.
        if len(made) >= wanted:
            short = ""
        elif len(points) < wanted:
            short = (f" - {wanted} were asked for, and that is all the "
                     f"distinctly different good configurations the solver "
                     f"has")
        else:
            short = f" - stopped after {len(made)} of {wanted}"
        return [collage], (f"collage of {len(made)} good samples"
                           f"{_collage_promise(report)}{short}")

    def _demo_run(self, request, p, ab_search, space, genes, base_args, cnpro,
                  base_units, base_prompt, base_seed, session):
        """GOOD/N-GOOD with no search running: render sample(s) from the
        retained solver state, and touch nothing else.

        The state comes from an Imported session, the last search on this
        tab, or the disk mirror a restart left behind - see _retained_solver
        - and is only ever READ: no observation
        is added, the staged import stays staged for a later real search, and
        a plain Generate afterwards still starts the search it always
        started. The rows themselves, plus prompt, canvases, resolution and
        seed, are read fresh from the UI by THIS generation. Saved coordinates
        are projected onto edited rows rather than refused, which is what makes
        the buttons a generator the rest of the UI can be tweaked under at
        will: reach a good state once, then sample from it for as long as it
        is useful.

        The replayed engine is deterministic, so the solver is rebuilt on a
        seed that changes per press (_DEMO_SERIAL) - without that, "varied
        per press" would render the same sample every time.
        """
        tab = "img2img" if self.is_img2img else "txt2img"
        payload = _retained_solver(tab)
        label = request.label
        if not payload or not payload.get("observations"):
            session.start_demo("")
            session.finish("no solver state to sample from - run a search "
                           "first, or Import a session file that carries one")
            logger.warning(f"{WHO}: a {label} sample was requested with no "
                           f"solver state on this tab.")
            return Processed(p, [])

        search = ab_search.PreferenceSearch(
            space, seed=base_seed + 7919 * _DEMO_SERIAL[tab])
        _restore_solver(search, payload, space, allow_row_changes=True)

        session.start_demo(f"rendering {label} from the learned taste")
        p.extra_generation_params["Script"] = self.title()
        extras = _generation_extras(p)
        # The A slot is this run's viewport: each sample lands there as it
        # finishes, and the collage replaces them all at the end. A collage
        # of sixty-four is twenty minutes of GPU, and a panel that shows
        # nothing until it is done is a panel nobody trusts halfway through.
        def publish(index, total, image):
            session.publish_demo(
                image, (f"{label}: sample {index} of {total} - the collage "
                        f"is composed when they are all in"
                        if total > 1 else
                        f"GOOD sample rendered - press again for another, "
                        f"varied per press; the solver state is kept"))

        # Presses that landed while this one was rendering queued themselves
        # on the session (request_demo sees a live run) - served here rather
        # than costing one host job each.
        made, notes = [], []
        queue = [request]
        spent = 0
        try:
            while queue and len(notes) < DEMO_BATCH_MAX:
                if session.stopping or state.interrupted \
                        or getattr(state, "stopping_generation", False):
                    break
                served = queue.pop(0)
                # The host's progress bar divides by this, so it is the
                # number of GENERATIONS this run still owes - which for a
                # collage is its whole count, not one job per press.
                state.job_count = spent + (served.count or 1) + sum(
                    item.count or 1 for item in queue)
                spent += served.count or 1
                batch, note = self._samples(
                    served, search, p, base_args, cnpro, genes,
                    base_units, base_prompt, base_seed, session, extras,
                    publish=publish)
                made.extend(batch)
                notes.append(note)
                with session.condition:
                    queue.extend(session.pending_demos)
                    session.pending_demos = []
        except Exception as exc:
            errors.display(exc, "rendering a CNPro A/B sample")
            session.finish(f"stopped - {exc}")
            return Processed(p, [])

        dropped = (f" ({len(queue)} press(es) beyond the cap dropped - "
                   f"press again)" if queue else "")
        if made:
            # The collage is the artifact - it goes on screen last, over the
            # individual samples the publish callback painted on the way.
            session.publish_demo(made[-1].images[0], notes[-1])
        session.finish(f"{'; '.join(notes)}{dropped} - the solver state is "
                       f"kept: GOOD/N-GOOD keep working, and the rest of the "
                       f"UI can be changed freely between presses")
        if not made:
            return Processed(p, [])
        final = made[0]
        for extra in made[1:]:
            final.images.extend(extra.images)
            final.infotexts.extend(extra.infotexts)
        return final

    def _render(self, p, base_args, cnpro, genes, point, base_units,
                base_prompt, seed, side=None, session=None, cache=None,
                cache_store=True, extras=None):
        """One image for one configuration - or the one already made.

        The recipe is computed FIRST because it is the identity of the
        generation: same recipe, same image, so a cache hit skips the whole
        generation. The trace still gets a line (marked "reused") - it is
        the record of what each duel showed, and a duel that showed a cached
        image still showed it.
        """
        recipe = _config_string(genes, point, base_units, base_prompt, seed,
                                extras=extras)
        if cache is not None:
            hit = cache.get(recipe)
            if hit is not None and hit.images:
                if session is not None and side:
                    session.record(side,
                                   _choices_line(genes, point) + " (reused)")
                    session.render_complete(side)
                return hit
        pc = copy.copy(p)
        pc.styles = pc.styles[:]
        pc.override_settings = copy.copy(p.override_settings)
        # Per image, not shared. CNPro writes the unit's description into
        # extra_generation_params, so a shared dict carries one duel's note
        # into the infotext of every image after it.
        pc.extra_generation_params = copy.copy(p.extra_generation_params)
        # `p` was set up when the host assigned its script_args; saying so
        # keeps the property's setter from running setup_scripts again.
        pc.scripts_setup_complete = True
        # A duel is one image against one image. Anything else would be asking
        # the user to grade a batch against a batch.
        pc.n_iter = 1
        pc.batch_size = 1
        pc.seed = seed

        args = list(base_args)
        for gene in genes:
            if not gene.on_unit:
                continue
            slot = cnpro.args_from + gene.unit_index
            # The gr.State unit objects are the LIVE ones the panel writes to,
            # so they are copied before anything is set on them - once per
            # unit, not once per gene, since several rows may target the same
            # unit and a second copy would drop the first one's edits.
            if args[slot] is base_args[slot]:
                args[slot] = copy.copy(args[slot])
            gene.apply_unit(args[slot], point)
        # The Canvas layer rows: ask the page for each canvas' composite at
        # this point's opacities, and put it where the generation reads
        # the canvas from. One request for all of them - several rows on
        # one canvas are one composite - and only for a side that is about
        # to be generated (the cache hit above never asks).
        overrides = {}
        for gene in genes:
            if gene.kind == Gene.KIND_LAYER:
                key, index, alpha = gene.override(point)
                overrides.setdefault(key, {})[str(index)] = alpha
        if overrides:
            if session is None:
                raise RuntimeError(
                    f"{WHO}: a Canvas layer row needs the panel session to "
                    f"reach the page.")
            composites = session.composites(overrides)
            if composites is None:
                return Processed(p, [])   # stopped or interrupted mid-wait
            _apply_composites(pc, args, base_args, cnpro, composites)
        pc.script_args = args
        pc.prompt = _compose_prompt(genes, point, base_prompt)
        # Every image carries its own recipe, so a picture pulled out of the
        # output folder a week later still says what made it - and the same
        # string is what the trace box shows, so the two cannot disagree.
        pc.extra_generation_params["CNPro DNA"] = recipe
        if session is not None and side:
            # The IMAGE carries the whole recipe; the trace carries the
            # choices. Same generation, two audiences: one is read a week
            # later by whoever found the file, the other is read now by
            # somebody wondering what the last four duels varied.
            session.record(side, _choices_line(genes, point))
        try:
            processed = process_images(pc)
        except Exception as exc:
            errors.display(exc, "generating an image for CNPro A/B")
            return Processed(p, [])
        if session is not None and side:
            session.render_complete(side)
        if cache is not None and cache_store and processed.images:
            cache[recipe] = processed
            while len(cache) > RENDER_CACHE:
                cache.pop(next(iter(cache)))
        return processed


class _Row:
    """The components of one row, by name rather than by position."""

    def __init__(self, container, target, mode, choices, line, values,
                 prompt_mode, lora_pick, loras, weights,
                 lora_refresh, lora_dice, point, canvas, layer):
        self.container = container
        self.target = target
        self.mode = mode
        self.choices = choices
        self.line = line
        self.values = values
        self.prompt_mode = prompt_mode
        self.lora_pick = lora_pick
        self.loras = loras
        self.weights = weights
        self.lora_refresh = lora_refresh
        self.lora_dice = lora_dice
        self.point = point
        self.canvas = canvas
        self.layer = layer

    #: ROW_FIELDS name -> the attribute holding that field's component.
    #: "points" is the one name that differs from its attribute: the session
    #: file has called it that since v2.
    ATTRIBUTES = {"points": "point"}

    def args(self):
        """The row's value-carrying components, in ROW_FIELDS order."""
        return [getattr(self, self.ATTRIBUTES.get(name, name))
                for name in ROW_FIELDS]

    def import_components(self):
        """Every component Import repaints, in one fixed order: the value
        components plus the visibility-only ones (the LoRA picker and its
        two buttons). `_row_visibility` is the other half of this list."""
        return [self.target, self.mode, self.choices, self.line,
                self.values, self.prompt_mode, self.lora_pick, self.loras,
                self.weights, self.lora_refresh, self.lora_dice, self.point,
                self.canvas, self.layer]


def _profile_lines():
    return getattr(_profile_scale(), "PROFILE_LINES", {"Main": None})


def _read_rows(args, row_count):
    """The visible rows, as genes.

    Rows past the count are skipped rather than read: they are still in the
    args (gradio sends every component's value, visible or not) and they still
    hold whatever was typed into them before they were hidden. Reading them
    would make "-" a button that hides a row and keeps searching over it.
    """
    genes = []
    for index in range(min(row_count, MAX_ROWS)):
        base = 1 + index * ROW_ARGS
        (target, mode, choices, line, values, prompt_mode, loras,
         weights_text, points_text, canvas, layer_text) = \
            args[base:base + ROW_ARGS]

        def weight_list(default):
            # The row's own weight list, interval notation included; an
            # empty box means the kind's default grid - which is what the
            # placeholder shows (WEIGHT_PLACEHOLDERS).
            listed = [float(v) for v in _numbers(weights_text)]
            return listed or list(default)

        if target == TARGET_PROMPT:
            prompts = _lines(values)
            if prompts:
                genes.append(Gene(Gene.KIND_PROMPT, index, values=prompts,
                                  prompt_mode=prompt_mode or PROMPT_REPLACE,
                                  weights=weight_list(PROMPT_WEIGHTS)))
            continue

        if target == TARGET_LORA:
            names = [_lora_name(item) for item in _lines(loras)]
            names = [name for name in names if name]
            if names:
                genes.append(Gene(Gene.KIND_LORA, index, loras=names,
                                  weights=weight_list(LORA_WEIGHTS)))
            continue

        if target == TARGET_CANVAS:
            # Canvas, layer AND at least one opacity, or the row declares
            # nothing - like a Profile row with an empty factors box.
            layer_index = _layer_index(layer_text)
            if not RE_CANVAS_KEY.match(str(canvas or "")) \
                    or layer_index is None:
                continue
            opacities = [float(v) for v in _numbers(values)]
            bad = [v for v in opacities if not 0 <= v <= 100]
            if bad:
                raise ValueError(
                    f"{WHO}: row {index + 1}: {bad[0]:g} is not an opacity "
                    f"- the list is in percent, 0 to 100.")
            if opacities:
                genes.append(Gene(Gene.KIND_LAYER, index, canvas=canvas,
                                  layer=layer_index, values=opacities))
            continue

        unit_index = _unit_index(target)
        if unit_index is None:
            continue

        if mode in (MODE_PROFILE, MODE_POINTS):
            module = _profile_scale()
            if module is None:
                raise RuntimeError(
                    f"{WHO}: row {index + 1} varies a profile, but ControlNet "
                    f"Pro is not loaded.")
            if line not in module.PROFILE_LINES:
                raise ValueError(
                    f"{WHO}: row {index + 1}: unknown profile line {line!r}.")
            if mode == MODE_POINTS:
                offsets = module.parse_offsets(values)
                if offsets:
                    genes.append(Gene(Gene.KIND_POINT, index,
                                      unit_index=unit_index, line=line,
                                      values=offsets,
                                      point_indices=_point_indices(
                                          points_text, index)))
                continue
            factors = module.parse_factors(values)
            if factors:
                genes.append(Gene(Gene.KIND_PROFILE, index,
                                  unit_index=unit_index, line=line,
                                  values=factors))
            continue

        # Model mode: the multiselect IS the value list.
        picked = [v for v in (choices or []) if v]
        if picked:
            genes.append(Gene(Gene.KIND_SETTING, index, unit_index=unit_index,
                              field="model", values=picked))
    return genes


def _point_indices(text, row):
    """The point-index list of one Profile-points row.

    Integers, comma or whitespace separated, 0-based left to right, negative
    from the end - and the LIST is the feature: every listed point moves by
    the row's offset TOGETHER, which is what edits a profile interval
    instead of one knot. Empty means [0], matching the box's initial value.
    A repeated index is refused here in its typed form; collisions through
    negative aliases (0 and -N on an N-point line) are caught in _setup,
    where the point count is known.
    """
    tokens = [t for t in re.split(r"[\s,;]+", str(text or "").strip()) if t]
    if not tokens:
        return [0]
    indices = []
    for token in tokens:
        try:
            indices.append(int(token))
        except ValueError:
            raise ValueError(
                f"{WHO}: row {row + 1}: {token!r} is not a point index - use "
                f"integers, 0-based left to right, negative from the end, "
                f"e.g. '0, 2, -1'.")
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"{WHO}: row {row + 1} lists a point index twice - each listed "
            f"point moves once per offset, so a repeat would move it double.")
    return indices


def _lora_name(text):
    """A LoRA name, whether it was typed bare or pasted as a tag.

    `<lora:add_detail:0.8>` is what a user has in their clipboard nine times
    out of ten, and rejecting it would be pedantry - but the weight in it is
    NOT honoured, because the weight is what this row exists to search for.
    """
    match = RE_LORA_TAG.match(text)
    return match.group(1).strip() if match else text.strip()


def _numbers(text):
    """Numbers from a values box: one per line, commas, or an `a:b:n` run."""
    values = []
    for line in str(text or "").replace(",", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            parts = [t.strip() for t in line.split(":")]
            if len(parts) != 3:
                raise ValueError(
                    f"{WHO}: '{line}' is not a value: use a number, or "
                    f"'start:end:count' for an evenly spaced run.")
            start, end, count = float(parts[0]), float(parts[1]), int(float(parts[2]))
            if count < 1:
                raise ValueError(f"{WHO}: '{line}': count must be at least 1.")
            if count == 1:
                values.append(start)
            else:
                step = (end - start) / (count - 1)
                values.extend(start + step * i for i in range(count))
        else:
            values.append(float(line))
    return [f"{round(v, 6):g}" for v in values]


def _coerce(token, value):
    """A configuration string's text back into the type its component wants.

    Only the fields whose components refuse a string need this, and a Checkbox
    is the reason it exists: handed "False" it renders as CHECKED (a non-empty
    string is truthy) and then reports that string onward, so the unit ends up
    holding "False" and behaving as True. Everything else is left as text,
    which is what dropdowns and textboxes hold anyway.
    """
    if token == "seed":
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return -1
    # The host's Number/Slider components refuse text the same way the
    # Checkbox does - handed "1024" they either break or quietly revert.
    if token in ("steps", "width", "height"):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if token in ("cfg_scale", "denoising_strength") \
            or RE_LAYER_TOKEN.match(token):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    field = UNIT_FIELDS.get(token.partition(".")[2])
    if field is None or field.cast is None:
        return value
    try:
        return field.cast(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------------
# What the panel says
# ---------------------------------------------------------------------------

def _headline(count):
    # SHORT, because the line it sits on is shared with the loop's two
    # switches now. "Add rows for more" was the +/- buttons' own tooltips
    # said twice; the buttons are an inch to the left of this text.
    return f"**{count}** degree{'s' if count != 1 else ''} of freedom, one per row"


#: What the weights box means when it is left EMPTY, by row kind - shown as
#: the placeholder, applied by _read_rows. A placeholder rather than a value,
#: so switching a row between LoRA and Prompt never has to decide whether the
#: text in the box was the user's or the previous kind's default.
WEIGHT_PLACEHOLDERS = {
    TARGET_LORA: ", ".join(f"{w:g}" for w in LORA_WEIGHTS) + "   or   0.2:1:5",
    TARGET_PROMPT: ", ".join(f"{w:g}" for w in PROMPT_WEIGHTS) + "   or   0.4:1.2:5",
}


def _row_visibility(target_value, mode_value):
    """Props for a row's dependent controls, in `switches` order.

    A module function returning plain prop dicts rather than gr.updates,
    because it has two callers with different needs: the row's own change
    wiring (wraps each dict in gr.update) and Import (merges each dict with
    the imported VALUE into one update - naming a component twice in one
    output list is a gradio internal not to be relied on).

    Tested POSITIVELY, against the "Unit N" form. Testing "not one of the
    global targets" made a blank target - the state every page opens in -
    look like a unit and show the unit controls.
    """
    is_unit = _unit_index(target_value) is not None
    model = is_unit and mode_value == MODE_MODEL
    profile = is_unit and mode_value == MODE_PROFILE
    points = is_unit and mode_value == MODE_POINTS
    is_lora = target_value == TARGET_LORA
    is_prompt = target_value == TARGET_PROMPT
    is_canvas = target_value == TARGET_CANVAS
    weighted = is_lora or is_prompt
    return [
        {"visible": is_unit},                                   # mode
        {"visible": model},                                     # choices
        {"visible": profile or points},                         # line
        # ONE update carries visibility, label and placeholder together: the
        # values box is shared by four kinds of row.
        {"visible": profile or points or is_prompt or is_canvas,
         "label": _values_label(target_value, mode_value),
         "placeholder": _values_placeholder(target_value,
                                            mode_value)},       # values
        {"visible": is_prompt},                                 # prompt mode
        {"visible": is_lora},                                   # picker
        {"visible": is_lora},                                   # loras
        # The weights box serves BOTH weighted kinds; only its placeholder
        # follows the kind (empty = that kind's default grid), so a list the
        # user typed is never overwritten by changing the row's target.
        {"visible": weighted,
         "placeholder": WEIGHT_PLACEHOLDERS.get(target_value,
                                                WEIGHT_PLACEHOLDERS[
                                                    TARGET_LORA])},  # weights
        {"visible": is_lora},                                   # refresh
        {"visible": is_lora},                                   # dice
        {"visible": points},                                    # point
        {"visible": is_canvas},                                 # canvas
        {"visible": is_canvas},                                 # layer
    ]


def _values_label(target, mode):
    if target == TARGET_PROMPT:
        return "prompts (one per line)"
    if target == TARGET_CANVAS:
        return "opacities (%)"
    if _unit_index(target) is not None and mode == MODE_PROFILE:
        return "factors"
    if _unit_index(target) is not None and mode == MODE_POINTS:
        return "offsets (0 = as drawn)"
    return "values (one per line)"


def _values_placeholder(target, mode):
    if target == TARGET_PROMPT:
        return "a house on a hill\na house in a storm\n…"
    if target == TARGET_CANVAS:
        return "0, 50, 100   or   0:100:5"
    if _unit_index(target) is not None and mode == MODE_PROFILE:
        return "0.5, 0.75, 1, 1.5   or   0.5:1.5:5"
    if _unit_index(target) is not None and mode == MODE_POINTS:
        return "-0.2, 0, 0.2   or   -0.2:0.2:5"
    return "one value per line"


def _summary(genes, point, search):
    status = search.status()
    # More than one attractor is worth a word: it is the difference between
    # "converging on an answer" and "holding several answers", and the STOP
    # button's keepers are where the others will appear.
    looks = (f", {status['attractors']} distinct looks"
             if status.get("attractors", 0) > 1 else "")
    return (f"{status['duels']} graded, confidence {status['confidence']:.0%}"
            f"{looks} - best so far: " + _choices_line(genes, point))


def _inference_progress():
    """Where the current render is, 0..1, straight off the host's counters.

    Read live at tick time; the session does not count sampling steps itself.
    It only enforces the lifecycle fact the host counters cannot express:
    progress within one render is monotone, and the render returning means
    1.0 even while the host briefly resets its counters during finalization.
    Which RENDER the raw number belongs to is the session's `generating`.
    """
    steps = int(getattr(state, "sampling_steps", 0) or 0)
    if steps <= 0:
        return 0.0
    step = float(getattr(state, "sampling_step", 0) or 0)
    return min(max(step / steps, 0.0), 1.0)


def _phase_css(eid, snap, progress, capacity=None):
    """The tick's <style> payload: what the current PHASE paints onto chrome
    gradio components cannot carry themselves.

    * While a side renders, that side's "interesting" toggle wears the
      inference progress as a background fill (`--cnpro-ab-fill`, style.css)
      - the button stays a button, the fill is a pseudo-element behind its
      label. While B renders, A shows a FULL bar: the pair then reads as
      "A finished, B under way" at a glance.
      Progress is monotone within the render (latched by `_Session`) and a
      completed displayed image stays FULL after the render ends. Removing
      the rule at completion made 100% look like 0%, and exposed the host's
      brief finalization reset as an end -> zero -> end flash.
    * While a duel awaits its grade, the grade-here nudge beside the scale
      is faded in - generation over, the next move is the user's. It lives
      in CSS opacity rather than component visibility so that its appearing
      cannot move a single pixel of the rows it points at.

    * The parked 2x2's CELL SIZE, as a digit count - "how many characters",
      never a pixel width. style.css turns it into one, in `ch`, which is
      the width of a digit in the button's own font, so the cell fits
      "3 GOOD" and "9999 GOOD" exactly in whatever theme and at whatever
      font size. A pixel width computed here would be this file guessing at
      a font it never sees.

      ONE NUMBER SIZES THE WHOLE BLOCK. All four buttons and both column
      offsets are written in terms of that cell, so this is the only thing
      that can resize the parked controls, and it cannot resize one of them
      without resizing the rest - which is what keeps them a grid rather
      than four buttons that happen to line up.

    Empty when the phase needs nothing painted and the count is unknown,
    which is the idle common case the tick then dedupes away entirely. Note
    that a KNOWN count makes the payload non-empty for the rest of the
    session - that is not churn, because the tick only re-sends this string
    when it CHANGES, and the digit count changes about as often as the
    number of good answers does.
    """
    rules = []
    if capacity is not None:
        rules.append(f"#{eid}_duel .cnpro-ab-grades"
                     f"{{--cnpro-ab-digits:{len(str(int(capacity)))}}}")
    if snap["generating"] in ("A", "B"):
        percent = int(round(min(max(progress, 0.0), 1.0) * 100))
        if snap["generating"] == "A":
            rules.append(f"#{eid}_interesting_a{{--cnpro-ab-fill:{percent}%}}")
        else:
            rules.append(f"#{eid}_interesting_a{{--cnpro-ab-fill:100%}}")
            rules.append(f"#{eid}_interesting_b{{--cnpro-ab-fill:{percent}%}}")
    elif snap.get("awaiting") or snap.get("image_a") is not None:
        # The fill is progress, not activity. A displayed completed render is
        # 100%, including after the loop finishes; clearing the CSS here reads
        # visually as progress falling back to zero.
        rules.append(f"#{eid}_interesting_a{{--cnpro-ab-fill:100%}}")
        if snap.get("awaiting") or snap.get("image_b") is not None:
            rules.append(f"#{eid}_interesting_b{{--cnpro-ab-fill:100%}}")
    if snap["awaiting"]:
        rules.append(f"#{eid}_duel .cnpro-ab-grade-hint{{opacity:1}}")
    return f"<style>{''.join(rules)}</style>" if rules else ""


def _submit_button(is_img2img):
    try:
        from lib_cnpro.controlnet_ui.controlnet_ui_group import ControlNetUiGroup
    except Exception:
        return None
    context = ControlNetUiGroup.a1111_context
    return (context.img2img_submit_button if is_img2img
            else context.txt2img_submit_button)


# This script registers NO settings. There was one - the A/B search's idle
# timeout - and it is gone rather than defaulted to "never": a saved value
# outlives the default that produced it, so a build that merely changed the
# default would keep pausing the searches of everyone whose config.json had
# already recorded 30. See _Session.await_grade for why there is no clock.
