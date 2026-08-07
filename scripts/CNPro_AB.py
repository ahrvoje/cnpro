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

* **Setting** - a CNPro unit field, and a LIST of values it may take. Models,
  preprocessors, whole weight-profile strings, resolutions, per-input mutes:
  anything the unit carries. Categorical throughout, exactly as typed - DNA
  picks one of the listed values, never something between them. A list of
  numbers is recognised as ordered, so `768, 1024, 1280` tells the search that
  1024 sits between the other two and evidence transfers accordingly.
* **Profile** - one of the unit's six profile lines, scaled by a factor from
  the row's list. See `lib_cnpro/profile_scale.py`: the factor multiplies the
  curve, so 1 is untouched, 0 switches the line off, 2 is twice the control.
* **Profile point** - ONE drawn point of one of those lines, moved vertically
  by an offset from the row's list: 0 is the profile exactly as drawn, +0.1
  is that point 0.1 higher in the plot's own axis units, negative moves it
  down - the same edit dragging the point in the editor would make. Point
  indices count drawn points left to right from 0, negative from the end
  (-1 = rightmost). Offsets are numbers, so the search knows they are
  ordered and evidence transfers between neighbouring values.
* **Prompt** - one prompt per line, one of which is picked, either REPLACING
  the prompt or being prepended/appended to it. Several Prompt rows compose
  (subject in one, lighting in another), which is why at most one of them may
  Replace.
* **LoRA** - one LoRA per line, of which the search picks ONE, and its weight
  is a CONTINUOUS parameter searched over the row's range. This is the one
  place DNA is not categorical, and deliberately: a LoRA weight is a number
  where 0.55 genuinely tells you something about 0.60, and pinning a LoRA to a
  weight you guessed is how a good LoRA gets rejected. Two LoRAs at once means
  two rows - a row is one LoRA slot, and its list is the candidates for that
  slot.

WHAT AN ANSWER IS
-----------------
Each round shows image A and image B and the SAME 0..10 scale twice. One
click answers, whichever row it lands on. The scale is the comparison: 0
means A is much better, 10 means B is much better, 5 means you cannot pick
between them - and 5 is a real answer, not a refusal: it says the two
configurations are worth the same, which is as informative as any other
grade. The ROW is the verdict about the pair itself: the top row is the
normal answer, the bottom row says "and I dislike BOTH of these". That is
the one thing a person feels instantly when a pair appears and a comparison
cannot carry ("both are great" and "both are awful" are the same statement
about the DIFFERENCE) - and it asks for no memory of any reference and no
absolute number, both of which people answer differently every time. A
bottom-row grade still steers (which side is LESS bad is information); what
the row adds is that the whole region sinks below par and the search changes
the subject - the next duel explores, and a streak of dislikes rolls the
candidate pool back toward broad exploration. skip, beside the lower row, is
the third kind of answer: this pair could not be judged at all - a broken
render, an unreadable duel - and records nothing, which is NOT what 5
records.

GOOD and BAD, parked beside STOP and skip, are not answers either: each
renders ONE fresh sample drawn from the corresponding end of the solver's
current belief - varied per press, pure inference, no observation recorded -
prints its full recipe into the trace, and, during a search, returns to the
same duel. They are the "show me what you think I want (or hate)" probes,
and the images join the gallery at stop.

THE BUTTONS DO NOT DIE WITH THE SEARCH, and that is the point of them: a
finished search is a reached state, not a spent one. With no search running
- after STOP, or right after an Import staged a solver - a GOOD/BAD press
starts a generation ITSELF (it stages the request and clicks Generate
client-side), and run() serves it by rendering that one sample from the
retained solver state instead of opening duels: nothing is asked, nothing
is observed, the state is left exactly as it was. Everything the rows do
not own - prompt, canvases, resolution, seed - is read fresh from the UI by
that generation, so the learned taste can be applied under freely retuned
settings for as long as it is useful. The search is how the generator is
TRAINED; the buttons are how it is USED.

The panel's Session file accordion holds Export and Import: the whole session
- rows, unit settings, canvases, prompt AND negative prompt, the sampler
settings (sampler, scheduler, steps, CFG, size, denoising), seed, and the
solver's learned state - as ONE deterministic HTML file. The file renders in a browser as readable
tables and images, so a version mismatch still leaves a page a human can
reproduce by hand; imported, it reconfigures everything and stages the solver
state, which the next Generate resumes onto the same rows.

Under each image sits an "✦ interesting" TOGGLE, and it is deliberately not a
grade: it says "overall this sample is whatever I graded it - usually bad -
but it touches a characteristic I want to see in good samples". The mark adds
no utility whatsoever; it donates the configuration's coordinates to hybrid
candidates (the characteristic, transplanted into the current attractors),
which then earn their way through ordinary duels or vanish. Toggled marks are
sent with the grade, so the natural gesture is: mark ✦ on the tempting side,
then grade the pair on the dislike row.

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
one that was ahead at STOP; with "Render the winner on stop" ticked, all the
keepers are rendered into the gallery, so a stop ends with the diverse answers
in view rather than one of them. Set applies a recipe string to the whole
Forge Neo configuration - unit fields, prompt and negative prompt, LoRAs,
sampler settings, seed. It is plain text
and can be saved, pasted back into that box, or handed to somebody else, which
is the point of it being a string rather than a button that only works in this
session.

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
* A search nobody answers does not wedge the tab forever - it PAUSES after
  the idle timeout in Settings > CNPro: the loop ends exactly as STOP would
  end it, and everything it learned stays behind.

NOTHING ENDS A SEARCH IRRECOVERABLY. Every graded duel refreshes the retained
solver state - on the session, and mirrored to a small JSON file that
survives a UI restart - and the next Generate on the same rows RESUMES from
it. "Resume search" (on by default, in the loop controls) is the way out:
untick it to start over. So an idle timeout, a STOP, an Interrupt, a page
reload or a restarted server cost nothing but the interruption itself: press
Generate and the search continues where it left off, or press GOOD/BAD and
sample from whatever it had learned. State learned on DIFFERENT rows never
resumes - the signature check refuses it - so changing the rows is also a
fresh start, with a log line saying so. The reloaded page finds its way back
too: the poll timer runs once at page load and repaints whatever the
server-side session is doing, then switches itself off if that is nothing.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
Nothing is written back to the UI while searching. Every duel gets a shallow
COPY of the unit dataclass in a copy of `p.script_args`; the gr.State the unit
panel is bound to is never touched, so an abandoned search leaves the panel
exactly as it was. The Set button is the ONE path that writes to the UI, it
runs only when pressed, and it writes to the visible components rather than to
the unit State - the same route an infotext paste takes.
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

import gradio as gr

import modules.scripts as scripts
from modules import errors, processing, script_callbacks, shared
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
#: +/- buttons move. Ten is well past useful: every extra degree of freedom
#: costs comparisons, and a human runs out of patience long before a search
#: with ten of them has enough evidence to speak.
MAX_ROWS = 10

#: Grades on offer. 0..10 with 5 meaning "cannot pick" - see the docstring.
GRADES = 11

TARGET_PROMPT = "Prompt"
TARGET_LORA = "LoRA"
GLOBAL_TARGETS = [TARGET_PROMPT, TARGET_LORA]

MODE_SETTING = "Setting"
MODE_PROFILE = "Profile"
MODE_POINT = "Profile point"
MODES = [MODE_SETTING, MODE_PROFILE, MODE_POINT]

PROMPT_REPLACE = "Replace"
PROMPT_PREPEND = "Prepend"
PROMPT_APPEND = "Append"
PROMPT_MODES = [PROMPT_REPLACE, PROMPT_PREPEND, PROMPT_APPEND]

#: Default LoRA weight range and grid: three states, 0.2 / 0.6 / 1.0.
#:
#: COARSE ON PURPOSE. Three visible steps per LoRA keeps a five-LoRA search
#: at 3^5 weight combinations instead of 21^5 - matched to a human's duel
#: budget, and each step is large enough to SEE, which is what makes a duel
#: answerable at all. The floor is 0.2 rather than 0 because 0 is a
#: degenerate identity: every LoRA in a slot at weight 0 is the same image,
#: while the model would hold them as different points and scatter evidence
#: across them. "This LoRA is not helping" is still discoverable - it is the
#: search preferring the bottom of the range - and a recommendation at 0.2
#: is cheap to try by hand at 0. Type over the boxes for a finer or wider
#: search; these are the DEFAULTS, not the limits.
LORA_WEIGHT_MIN = 0.2
LORA_WEIGHT_MAX = 1.0
LORA_WEIGHT_STEP = 0.4

#: Default range for a PROMPT row's weight, written with A1111's own
#: `(text:w)` syntax. Straddling 1 rather than starting at 0, because the two
#: knobs mean different things: a LoRA at 0 is simply absent, while a prompt
#: term at 0.6 is still in the prompt and still steering - de-emphasis and
#: emphasis are both useful and both live near 1. With the default 0.4 step
#: this is the three states 0.4 / 0.8 / 1.2, mirroring the LoRA grid: three
#: visible steps per row is what keeps a many-row search inside a human's
#: duel budget.
PROMPT_WEIGHT_MIN = 0.4
PROMPT_WEIGHT_MAX = 1.2

#: Characters that would end the `(text:w)` wrapper early, and so have to be
#: escaped inside it. A1111 reads a backslash before either as a literal.
RE_PROMPT_ESCAPE = re.compile(r"([()])")

#: How long the loop waits for a grade before pausing, in minutes. Overridden
#: from Settings > CNPro. A search left open should not hold the tab hostage;
#: the loop ends as STOP would end it - the answers so far still produce a
#: recommendation - and the retained state means the next Generate resumes
#: exactly where it left off, so the timeout costs nothing.
IDLE_TIMEOUT_DEFAULT = 30

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
EXPORT_VERSION = 1

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
# The unit fields a Setting row can vary
#
# A curated table rather than "every field on the dataclass". Three kinds are
# excluded on purpose and the exclusions are the interesting part:
#
# * IMAGES AND MASKS - they are not values, they are megabytes, and a search
#   over them is a search over things the user would have to paint first.
# * weight / guidance_start / guidance_end - the weight PROFILE overrides all
#   three, and every unit built by the editor has one. A row on them would be
#   inert, and inert rows are the failure this whole file is trying to avoid:
#   they produce identical images and blame nothing. Profile rows are what
#   replaces them.
# * control_mode - the UI no longer exposes it (the balance profile did), so
#   varying it would move a field nothing reads.
# ---------------------------------------------------------------------------

#: How a field's candidate values are entered. CHOICE fields have a known list
#: and get a multiselect; the rest are typed, one value per line, because their
#: interesting values are strings nobody can enumerate (profiles, prompts) or
#: numbers the user has in mind (resolutions, thresholds).
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

FIELD_LABELS = {field.label: name for name, field in UNIT_FIELDS.items()}
FIELD_LABEL_LIST = [field.label for field in UNIT_FIELDS.values()]

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


def _short_model(name):
    """Model name without its ` [hash]` suffix - status lines are narrow."""
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

    def __init__(self, kind, row, unit_index=None, field=None, values=None,
                 line=None, point_index=0, prompt_mode=PROMPT_REPLACE,
                 loras=None, weight_low=LORA_WEIGHT_MIN,
                 weight_high=LORA_WEIGHT_MAX, weight_step=LORA_WEIGHT_STEP):
        self.kind = kind
        self.row = row
        self.unit_index = unit_index
        self.field = field
        self.values = list(values or [])
        self.line = line
        self.point_index = int(point_index or 0)
        self.prompt_mode = prompt_mode
        self.loras = list(loras or [])
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.weight_step = weight_step
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
            return f"U{self.unit_index} {self.line}[{self.point_index}]"
        if self.kind == Gene.KIND_PROMPT:
            return f"Prompt ({self.prompt_mode.lower()})"
        return "LoRA"

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
            weight = ab_search.Dimension.range(f"{self.name} weight",
                                               self.weight_low,
                                               self.weight_high,
                                               self.weight_step)
            # The weight belongs to the pick: 0.6 of one LoRA says nothing
            # about 0.6 of another, and the kernel is told so - each
            # candidate learns its own weight curve from its own duels.
            weight.parent = pick
            return [pick, weight]
        labels = ([f"{v:g}" for v in self.values]
                  if self.kind in (Gene.KIND_PROFILE, Gene.KIND_POINT)
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
        if self.kind == Gene.KIND_LORA:
            name = self.loras[index] if 0 <= index < len(self.loras) else ""
            return name, float(slice_[1])
        if self.kind == Gene.KIND_PROMPT:
            text = self.values[index] if 0 <= index < len(self.values) else ""
            return text, float(slice_[1])
        if self.kind == Gene.KIND_PROFILE:
            return self.values[index] if 0 <= index < len(self.values) else 1.0
        if self.kind == Gene.KIND_POINT:
            # The neutral offset: the profile exactly as drawn.
            return self.values[index] if 0 <= index < len(self.values) else 0.0
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
            return (f"{self.name} {value:+g}" if value
                    else f"{self.name} as drawn")
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
                unit.weight_profile = module.offset_profile_point(
                    unit, self.unit_index, self.line, self.point_index,
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

#: GOOD/BAD presses made while NO search is running, per tab: a list of
#: (wanted_good, staged_at) pairs. The press stages its request here and
#: then clicks Generate itself (see _wire_duel); the run() that click
#: starts consumes ONE entry and renders that sample instead of starting
#: duels. One per run because one press queues one generation - a second
#: press while the first still waits in the host's queue stages its own
#: entry and its own click, and the pairing stays exact.
_PENDING_DEMO = {"txt2img": [], "img2img": []}

#: A staged idle request older than this is dropped with a log line rather
#: than served. The click that staged it fires Generate immediately, so a
#: surviving entry means that click never turned into a run (page reloaded,
#: script switched away) - and a stale entry would otherwise turn some
#: LATER, unrelated Generate press into a sample render the user did not
#: ask for.
DEMO_STALE_SECONDS = 30 * 60

#: Idle presses counted per tab, forever. Two jobs: each press writes a
#: DISTINCT token into the hidden trigger box (a repeated value fires no
#: change event, so the second press would never click Generate), and each
#: demo run seeds its solver replay with it - the replayed engine is
#: deterministic, so without a varying seed every press of GOOD would
#: render the SAME "varied" sample.
_DEMO_SERIAL = {"txt2img": 0, "img2img": 0}

#: Samples one demo run will render at most, however many presses landed
#: while it was working. Presses beyond the cap are dropped and the status
#: line says so - a runaway "leaned on the button" burst should cost a
#: bounded number of generations, and pressing again is one click.
DEMO_BATCH_MAX = 8


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
    which is what makes GOOD/BAD and Resume outlive a restart.
    """
    session = _SESSIONS.get(tab)
    return (_PENDING_SOLVER.get(tab)
            or (session.solver_state if session is not None else None)
            or _stored_solver(tab))


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


def _solver_payload(search, seed, space):
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
        "signature": _space_signature(space),
        "observations": [
            [None if a is None else list(search.points[a]),
             list(search.points[b]), float(p)]
            for a, b, p in search.observations],
        "interesting": [list(point) for point in search._interesting],
        # The hyperparameters as SELECTED, not just the data they were
        # selected from: re-selection runs every few observations, so the
        # exporting search and a replay are usually between selections -
        # and a replay that re-selects immediately believes measurably
        # different things than the search it claims to resume.
        "hyper": [float(v) for v in search._hyper],
    }


def _restore_solver(search, state, space):
    """Replay an exported solver state, or None on a space mismatch."""
    if state.get("signature") != _space_signature(space):
        return None
    count = 0
    for point_a, point_b, p in state.get("observations", []):
        index_b = search._row(_coerce_point(space, point_b))
        index_a = (None if point_a is None
                   else search._row(_coerce_point(space, point_a)))
        search.observations.append((index_a, index_b, float(p)))
        count += 1
    search.duels = int(state.get("duels", count))
    for point in state.get("interesting", []):
        search.mark_interesting(_coerce_point(space, point))
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
        parts = [f"{key} = {row.get(key)!r}" for key in (
            "target", "mode", "field", "choices", "line", "values",
            "prompt_mode", "loras", "weight_low", "weight_high", "point")]
        row_lines.append((f"row {index + 1}", " · ".join(parts)))
    sections.append("<h2>Search rows</h2>" + table(row_lines))
    sections.append("<h2>Settings</h2>" + table(
        sorted(payload.get("settings", {}).items())))
    sections.append("<h2>Loop</h2>" + table(
        sorted(payload.get("tail", {}).items())))

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
        self.stopping = False
        self.timed_out = False     # the last end was the idle timeout - a pause
        self.pending_score = None
        self.pending_demos = []    # queued GOOD/BAD requests, served in order
        self.interesting = [False, False]     # A, B - toggled, sent with the grade
        self.awaiting = False
        self.generating = None     # "A"/"B" while that side's image renders
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

    # -- written by run() ------------------------------------------------

    def start(self):
        # solver_state is deliberately NOT cleared: it is the tab's learned
        # taste, not this run's scratch. Clearing it here meant that starting
        # a search and interrupting it before the first grade destroyed the
        # PREVIOUS search's state - GOOD/BAD then reported "no solver state"
        # about a taste that had cost dozens of duels to learn. A resumed or
        # graded run overwrites it; a run that learns nothing leaves it be.
        with self.condition:
            self.running = True
            self.stopping = False
            self.timed_out = False
            self.pending_score = None
            self.pending_demos = []
            self.awaiting = False
            self.generating = None
            self.generation += 1
            self.image_a = self.image_b = None
            self.duel = self.graded = 0
            self.status = "starting"
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
            self.awaiting = False
            self.generating = None
            self.status = status
            self.generation += 1

    def say(self, status, generating=None):
        """A status line - and, when a side's image is being rendered, WHICH
        side, so the panel can wear the inference progress on that side's
        own chrome. Callers that say anything else clear the side with the
        same call: whatever was rendering is not any more."""
        with self.condition:
            self.status = status
            self.generating = generating

    def publish(self, duel, image_a, image_b, status):
        with self.condition:
            self.duel = duel
            self.image_a = image_a
            self.image_b = image_b
            self.awaiting = True
            self.generating = None
            # Marks belong to the images they were toggled on.
            self.interesting = [False, False]
            self.status = status
            self.generation += 1

    def start_demo(self, status):
        """A sample run: alive for the poller, without wiping the session.

        Unlike `start`, everything the finished search left behind - the
        recommendation, the trace, the solver state, the last images, the
        duel counts - is KEPT: a demo run adds to that record rather than
        opening a new one, and the next real search is what resets it.
        """
        with self.condition:
            self.running = True
            self.stopping = False
            self.timed_out = False
            self.pending_score = None
            self.pending_demos = []
            self.awaiting = False
            # Demo samples render into the A slot, so A's chrome carries
            # their progress for the whole batch.
            self.generating = "A"
            self.status = status
            self.generation += 1

    def publish_demo(self, image, status):
        """A freshly rendered sample, into the A slot.

        The slot doubles as the panel's anchor: `image_a` is what keeps the
        duel group visible once nothing is running, so after a demo run the
        GOOD/BAD buttons stay on screen for the next press instead of
        folding the panel away under the user's pointer.
        """
        with self.condition:
            self.image_a = image
            self.image_b = None
            self.awaiting = False
            self.status = status
            self.generation += 1

    def set_result(self, result, summary):
        with self.condition:
            self.result = result
            self.summary = summary

    def await_grade(self, timeout_minutes):
        """Block until the user grades, stops, interrupts, or gives up.

        Returns the grade, or None for "the search is over". The wait is a
        POLL of 0.25s rather than a plain wait, because two of the four ways
        out do not go through this session at all: the host's Interrupt button
        writes `shared.state`, and the idle timeout is the absence of any
        event whatsoever. A condition variable cannot be notified by either.
        """
        deadline = time.time() + max(float(timeout_minutes or 0), 0) * 60.0
        with self.condition:
            while True:
                if self.stopping:
                    return None
                if self.pending_score is not None:
                    score = self.pending_score
                    self.pending_score = None
                    self.awaiting = False
                    # Only a real answer counts as graded - not a skip, and
                    # not a GOOD/BAD sample request, which leaves the duel
                    # on screen still waiting.
                    if isinstance(score, tuple):
                        self.graded += 1
                    return score
                # After the grade, never instead of it: a queued GOOD/BAD is
                # a side request, and a grade the user managed to land in
                # the same instant is the answer to the question on screen.
                if self.pending_demos:
                    self.awaiting = False
                    return ("demo_good" if self.pending_demos.pop(0)
                            else "demo_bad")
                self.condition.wait(0.25)
                if state.interrupted or getattr(state, "stopping_generation", False):
                    return None
                if timeout_minutes and time.time() > deadline:
                    logger.warning(
                        f"{WHO}: no grade for {timeout_minutes:g} minutes - "
                        f"pausing the search. The learned state is kept and "
                        f"the next Generate resumes it; the timeout is in "
                        f"Settings > CNPro.")
                    self.stopping = True
                    self.timed_out = True
                    return None

    # -- written by the panel --------------------------------------------

    def grade(self, score, disliked=False):
        """An A-vs-B grade (0..10) or the "skip" token. ONE click, whichever
        row it lands on: `disliked` is simply which row - the lower one says
        "and I dislike both of these". Any interesting toggles set since the
        duel was published ride along with it."""
        with self.condition:
            if not self.awaiting:
                return False
            self.pending_score = (
                "skip" if score == "skip"
                else (float(score), bool(disliked), tuple(self.interesting)))
            self.interesting = [False, False]
            self.status = "recorded - working out what to ask next"
            self.condition.notify_all()
            return True

    def request_demo(self, good):
        """Ask the loop for one on-demand sample from the good or bad end.

        Mechanically a grade - it wakes the blocked loop - but it answers
        nothing: the loop renders the sample, prints its recipe to the
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
            self.pending_demos.append(bool(good))
            end = "good" if good else "bad"
            self.status = (f"rendering a fresh guess from the {end} end"
                           if self.awaiting else
                           f"queued a fresh guess from the {end} end - "
                           f"after this render")
            self.condition.notify_all()
            return True

    def toggle_interesting(self, side):
        """Flip the "interesting" mark on side 0 (A) or 1 (B).

        A toggle, not a submission: it waits for the grade, like the dislike
        row it usually accompanies - the expected click is "this one is bad,
        AND it has something". The status line echoes the current state, so
        on-and-off-again is visible and trustworthy.
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

    def request_stop(self):
        with self.condition:
            self.stopping = True
            self.status = "stopping"
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {
                "running": self.running,
                "awaiting": self.awaiting,
                "generating": self.generating,
                "generation": self.generation,
                "image_a": self.image_a,
                "image_b": self.image_b,
                "duel": self.duel,
                "graded": self.graded,
                "status": self.status,
                "result": self.result,
                "summary": self.summary,
                "trace": "\n".join(self.trace),
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
ROW_ARGS = 11

#: Trailing controls after the rows: seed policy, winner render, weight step,
#: and whether Generate resumes the retained solver state.
TAIL_ARGS = 4

ARG_COUNT = 1 + MAX_ROWS * ROW_ARGS + TAIL_ARGS


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
        with gr.Row(variant="compact"):
            add = ToolButton(value=ADD_SYMBOL, elem_id=f"{eid}_add",
                             tooltip="One more degree of freedom")
            remove = ToolButton(value=REMOVE_SYMBOL, elem_id=f"{eid}_remove",
                                tooltip="One fewer degree of freedom")
            rescan = ToolButton(value=REFRESH_SYMBOL, elem_id=f"{eid}_rescan",
                                tooltip="Rescan the ControlNet and LoRA "
                                        "directories")
            headline = gr.Markdown(_headline(1), elem_id=f"{eid}_headline")

        row_count = gr.State(1)
        rows = [self._row(i, eid, visible=(i == 0)) for i in range(MAX_ROWS)]

        # --- the loop's own controls -------------------------------------
        with gr.Row(variant="compact"):
            # ONE grid for both weighted row kinds - LoRA weights and prompt
            # weights - because they are the same kind of question asked
            # about different things. Two controls would be two ways to say
            # the same number. The default 0.4 gives each weighted row three
            # visible states (a LoRA's 0.2 / 0.6 / 1.0), which is what keeps
            # a many-row search inside a human's duel budget; set it finer
            # to let the recommendation land between the values that were
            # compared, which is what the model is for - a duel is only ever
            # asked at MIN_DUEL_SEPARATION or more apart regardless.
            weight_step = gr.Number(label="Weight step",
                                    value=LORA_WEIGHT_STEP, minimum=0.01,
                                    maximum=0.5, step=0.01, scale=2,
                                    elem_id=f"{eid}_step")
            vary_seed = gr.Checkbox(
                label="New seed each duel", value=False, scale=2,
                elem_id=f"{eid}_vary_seed")
            render_winner = gr.Checkbox(
                label="Render the winner on stop", value=True, scale=2,
                elem_id=f"{eid}_render_winner")
            # ON BY DEFAULT: a search that ended - STOP, Interrupt, the idle
            # timeout, even a restart (the state is mirrored to disk) - is a
            # reached state, and Generate continues from it as long as the
            # rows still match what it was learned on. Unticking is the one
            # way to say "same rows, start over".
            resume_search = gr.Checkbox(
                label="Resume search", value=True, scale=2,
                elem_id=f"{eid}_resume")

        # --- the duel ----------------------------------------------------
        #
        # Hidden until there is one. The panel is a configuration form until
        # the search starts and a grading station afterwards, and showing the
        # grading half while there is nothing to grade costs a screenful of
        # empty boxes in a UI that is short of screenfuls.
        with gr.Group(visible=False, elem_id=f"{eid}_duel",
                      elem_classes=["cnpro-ab-duel"]) as duel_group:
            with gr.Row(elem_classes=["cnpro-ab-images"]):
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
            # The scale's ends are NOT labelled here, they are labelled in the
            # status line directly underneath. A gr.HTML in this row carries
            # gradio's `.block { width: 100% }`, so each caption claims the
            # whole row as its flex base and pushes STOP onto a line of its
            # own - measured, not guessed. Two fewer components, and the
            # legend still sits against the buttons it explains.
            with gr.Row(elem_classes=["cnpro-ab-grades"]):
                # "Grade here", in the padding the row already reserves on
                # its left for seam symmetry (see style.css) - absolutely
                # positioned like STOP in the right one, so the nudge costs
                # no layout and its appearing moves nothing. Faded in by the
                # tick's phase CSS exactly while a grade is awaited: the one
                # moment the panel goes quiet and the next move is the
                # user's, which nothing used to say.
                gr.HTML("<span class='cnpro-ab-grade-hint'>Grade here "
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
                # GOOD asks the solver for one fresh sample from what it
                # currently believes you want, BAD from what it believes you
                # do not - pure inference, varied per press, no effect on
                # the duel or the model. The recipe lands in Tried; the
                # image lands in the output folder and the final gallery.
                good = gr.Button(value="GOOD", min_width=80,
                                 elem_id=f"{eid}_good",
                                 elem_classes=["cnpro-ab-demo"])
                stop = gr.Button(value="STOP", variant="stop", min_width=80,
                                 elem_id=f"{eid}_stop",
                                 elem_classes=["cnpro-ab-stop"])
            # The SAME scale twice, and the row is the verdict: grading on
            # this second row says "and I dislike BOTH of these". That is a
            # feeling the user has the moment a pair appears - no memory of
            # any reference, no absolute number, no second click: the one
            # click that grades the duel simply lands one row lower. The
            # grade still matters (which side is LESS bad steers the search
            # inside the region); the row is what lets it mark the whole
            # region as avoid-this and change the subject - see
            # ab_search.observe.
            with gr.Row(elem_classes=["cnpro-ab-grades", "cnpro-ab-dislike"]):
                dislike_buttons = [
                    gr.Button(value=str(score), min_width=34,
                              elem_id=f"{eid}_dislike_{score}",
                              elem_classes=["cnpro-ab-grade"])
                    for score in range(GRADES)]
                bad = gr.Button(value="BAD", min_width=80,
                                elem_id=f"{eid}_bad",
                                elem_classes=["cnpro-ab-demo"])
                skip = gr.Button(value="skip", min_width=80,
                                 elem_id=f"{eid}_skip",
                                 elem_classes=["cnpro-ab-skip"])
            status = gr.Markdown("", elem_id=f"{eid}_status",
                                 elem_classes=["cnpro-ab-status"])

        # --- the answer --------------------------------------------------
        with gr.Row(variant="compact"):
            config = gr.Textbox(
                label="Configuration", value="", lines=1, max_lines=3, scale=9,
                placeholder="the search prints its recommendation here - or "
                            "paste one back in",
                elem_id=f"{eid}_config")
            apply_button = gr.Button(value="Set", variant="primary", scale=1,
                                     min_width=70, elem_id=f"{eid}_set")
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
        # second forever; the Generate click switches it back on.
        poller = gr.Timer(POLL_SECONDS, active=True)
        # What THIS browser was last sent, so the poll can send only what
        # changed - see `tick`. A dict rather than a counter because the parts
        # move independently: the images change once per duel, the status
        # several times per duel, the recipe once per answer.
        seen = gr.State({})

        targets, unreachable = _set_targets(groups, is_img2img)
        self._wire_rows(rows, row_count, headline, add, remove, rescan,
                        enables, type_filters, count)
        self._wire_duel(poller, seen, duel_group, image_a, image_b,
                        status, config, trace, grade_buttons, dislike_buttons,
                        (interesting_a, interesting_b), (good, bad), skip,
                        stop, is_img2img)
        self._wire_set(apply_button, config, set_note, targets, unreachable)
        self._wire_io(export_button, import_button, session_file, io_note,
                      rows, row_count, headline,
                      (vary_seed, render_winner, weight_step, resume_search),
                      targets, _canvas_targets(groups), len(groups),
                      is_img2img, duel_group, status)

        # Returned in the order run() unpacks them - see _read_rows.
        controls = [row_count]
        for row in rows:
            controls.extend([row.target, row.mode, row.field, row.choices,
                             row.line, row.values, row.prompt_mode, row.loras,
                             row.weight_low, row.weight_high, row.point])
        controls.extend([vary_seed, render_winner, weight_step, resume_search])
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
            # min_width fits all THREE options on one line - at the old 200
            # the third chip wrapped and the row grew a line taller for
            # nothing, exactly what the width was there to prevent.
            mode = gr.Radio(
                label="how", choices=MODES, value=MODE_SETTING, visible=False,
                scale=2, min_width=310, elem_id=f"{rid}_mode")
            field = gr.Dropdown(
                label="setting", choices=list(FIELD_LABEL_LIST),
                value=UNIT_FIELDS["model"].label, visible=False, scale=3,
                elem_id=f"{rid}_field")
            choices = gr.Dropdown(
                label="values", choices=[], value=[], multiselect=True,
                visible=False, scale=6, elem_id=f"{rid}_choices")
            line = gr.Dropdown(
                label="profile", choices=list(_profile_lines()), value="Main",
                visible=False, scale=2, elem_id=f"{rid}_line")
            # Which drawn point a "Profile point" row moves: 0-based, left to
            # right, negative from the end (-1 = rightmost).
            point = gr.Number(
                label="point #", value=0, precision=0, visible=False,
                scale=1, min_width=70, elem_id=f"{rid}_point")
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
            weight_low = gr.Number(label=WEIGHT_LABELS[0], value=LORA_WEIGHT_MIN,
                                   minimum=-2, maximum=4, step=0.05,
                                   visible=False, scale=1, min_width=70,
                                   elem_id=f"{rid}_wmin")
            weight_high = gr.Number(label=WEIGHT_LABELS[1], value=LORA_WEIGHT_MAX,
                                    minimum=-2, maximum=4, step=0.05,
                                    visible=False, scale=1, min_width=70,
                                    elem_id=f"{rid}_wmax")

        row = _Row(container, target, mode, field, choices, line, values,
                   prompt_mode, lora_pick, loras, weight_low, weight_high,
                   lora_refresh, lora_dice, point)

        def visibility(target_value, mode_value, field_label, low, high):
            return [gr.update(**props) for props in _row_visibility(
                target_value, mode_value, field_label, low, high)]

        switches = [mode, field, choices, line, values, prompt_mode,
                    lora_pick, loras, weight_low, weight_high,
                    lora_refresh, lora_dice, point]
        for component in (target, mode, field):
            component.change(fn=visibility,
                             inputs=[target, mode, field, weight_low, weight_high],
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

        def choices_update(target_value, field_label, selected, filters):
            """The value list for one Setting row, narrowed by its unit."""
            field_name = FIELD_LABELS.get(field_label, "model")
            field = UNIT_FIELDS[field_name]
            if field.kind != KIND_CHOICE:
                return gr.update()
            index = _unit_index(target_value)
            control_type = filters[index] if index is not None and index < len(filters) else "All"
            available = field.choices(control_type)
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
            fields = list(args[MAX_ROWS:2 * MAX_ROWS])
            selected = list(args[2 * MAX_ROWS:3 * MAX_ROWS])
            rest = args[3 * MAX_ROWS:]
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
            for target_value, field_label, current in zip(targets, fields, selected):
                updates.append(choices_update(target_value, field_label,
                                              current, filters))
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
                       + [row.field for row in rows]
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

        # Per row: picking a unit or a field re-reads only THAT row's lists.
        # The LoRA names are re-read here rather than only at build time - the
        # panel is built once at startup and the host may not have scanned its
        # LoRAs yet, which left every picker permanently empty on a page where
        # nothing else happened to rebuild it.
        for row in rows:
            def on_row(target_value, field_label, selected, *values):
                filters = list(values[count:])
                return (choices_update(target_value, field_label, selected, filters),
                        gr.update(choices=_lora_names())
                        if target_value == TARGET_LORA else gr.update())

            for component in (row.target, row.field):
                component.change(
                    fn=on_row,
                    inputs=[row.target, row.field, row.choices] + enables + type_filters,
                    outputs=[row.choices, row.lora_pick], show_progress=False)

    def _wire_duel(self, poller, seen, duel_group, image_a, image_b,
                   status, config, trace, grade_buttons, dislike_buttons,
                   interesting_buttons, demo_buttons, skip, stop, is_img2img):
        """The poll, the two grade rows, the interesting toggles, GOOD/BAD,
        skip and STOP."""
        session = _session(is_img2img)
        tab = "img2img" if is_img2img else "txt2img"
        eid = f"cnpro_ab_{tab}"
        # The tick's style channel: a <style> element whose rules paint the
        # PHASE onto chrome a gradio update cannot reach - the inference
        # fill on the interesting toggles, the grade-here nudge's opacity.
        # An HTML component because a style element applies from anywhere
        # in the document, its hidden wrapper included, and the alternative
        # is client JS polling a second channel for the same numbers the
        # tick already carries. See _phase_css for what it says.
        phase_css = gr.HTML(value="", elem_id=f"{eid}_phase_css",
                            elem_classes=["cnpro-ab-phase-css"])

        def tick(last):
            """Paint whatever the loop is doing - and ONLY what changed.

            NOTHING here is sent unconditionally, and that is the whole point.
            Re-sending a value gradio already holds still re-renders the
            block, so a status line rewritten with the same text 1.6 times a
            second FLICKERS, a textbox does the same and steals a selection
            the user was making inside it, and two images re-encoded every
            tick cost more bandwidth than the search costs GPU.

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
                return paint(last)
            finally:
                session.tick_lock.release()

        def paint(last):
            snap = session.snapshot()
            last = last if isinstance(last, dict) else {}
            # Retained solver state keeps the group (and the GOOD/BAD
            # buttons inside it) reachable on a page that loaded AFTER the
            # search - a reload, or a restarted server whose disk mirror
            # still holds a taste worth sampling. Checked last, so a live
            # search never pays the disk read.
            visible = (snap["running"] or bool(snap["image_a"])
                       or _retained_solver(tab) is not None)
            if visible and not snap["running"] and not snap["status"]:
                # Only a fresh session is this blank; say why the group is
                # on screen at all.
                snap["status"] = ("solver state from an earlier session is "
                                  "available - GOOD/BAD render samples from "
                                  "it; Generate resumes the search")
            now = {
                "generation": snap["generation"],
                "visible": visible,
                "status": _status_markdown(snap),
                "result": snap["result"],
                "trace": snap["trace"],
                "active": bool(snap["running"]),
                "css": _phase_css(eid, snap, _inference_progress()),
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
                when("status", now["status"]),
                # An empty recipe is only painted while a search is RUNNING:
                # there it means "this session has not recommended anything
                # yet", and leaving the previous session's recipe on screen
                # instead reads as the search not knowing the rows changed -
                # reported with a freshly added LoRA row, whose recommendation
                # kept naming the old two-LoRA setup. Idle, an empty result
                # must not be sent: the box may hold a configuration the user
                # pasted in for Set.
                (when("result", now["result"])
                 if now["result"] or now["active"] else gr.update()),
                when("trace", now["trace"]),
                # The timer switches ITSELF off. Nothing else knows when the
                # loop ended: run() returns on its own thread, and a handler
                # bound to the Generate click would have to guess whether this
                # script was the one that ran.
                (gr.update(active=now["active"])
                 if now["active"] != last.get("active") else gr.update()),
                when("css", now["css"]),
            ]

        tick_outputs = [seen, duel_group, image_a, image_b, status, config,
                        trace, poller, phase_css]
        poller.tick(fn=tick, inputs=[seen], outputs=tick_outputs,
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

        def grade(score, disliked=False):
            def handler():
                if not session.grade(score, disliked):
                    return gr.update()
                return gr.update(value=_status_markdown(session.snapshot()))
            return handler

        for score, button in enumerate(grade_buttons):
            button.click(fn=grade(score), inputs=[], outputs=[status],
                         show_progress=False, queue=False)
        for score, button in enumerate(dislike_buttons):
            button.click(fn=grade(score, disliked=True), inputs=[],
                         outputs=[status], show_progress=False, queue=False)

        # skip is a grade in every mechanical sense - it wakes the loop, the
        # loop moves on - it just records nothing: the duel could not be
        # judged (a broken render, an unreadable pair), and "could not judge"
        # is NOT the same answer as 5, which says they are worth the same.
        skip.click(fn=grade("skip"), inputs=[], outputs=[status],
                   show_progress=False, queue=False)

        def toggle(side):
            def handler():
                if not session.toggle_interesting(side):
                    return gr.update()
                return gr.update(value=_status_markdown(session.snapshot()))
            return handler

        for side, button in enumerate(interesting_buttons):
            button.click(fn=toggle(side), inputs=[], outputs=[status],
                         show_progress=False, queue=False)

        # GOOD/BAD press two very different machines depending on whether a
        # loop is alive, and the user should never have to know which:
        #
        # * a RUNNING search takes the request directly - queued, served at
        #   the loop's next listen, back to the duel afterwards;
        # * IDLE (search finished, or a session just Imported), the press
        #   ITSELF becomes the generation: it stages the request and clicks
        #   Generate through the hidden trigger box below, and run() sees
        #   the staged entry and renders one sample from the retained solver
        #   state instead of starting duels. That is what makes the solver a
        #   GENERATOR once a good state is reached: the search ending does
        #   not end sampling from it, and everything the rows do not own -
        #   prompt, canvases, resolution, seed - is read fresh from the UI
        #   at that Generate, tweakable at will between presses.
        #
        # The box holds a distinct token per press (a repeated value fires
        # no change event), and the click happens client-side because a
        # gradio handler cannot invoke the host's generation pipeline - the
        # Generate button's own event is the one path that reads every
        # component the pipeline needs.
        demo_go = gr.Textbox(value="", visible=False,
                             elem_id=f"cnpro_ab_{tab}_demo_go")
        demo_go.change(
            fn=None, inputs=[demo_go], outputs=None,
            js="(token) => { if (token) { const b = document.getElementById("
               f"'{tab}_generate'); if (b) b.click(); }} }}")

        def demo(wanted):
            def handler():
                if session.request_demo(wanted):
                    return (gr.update(value=_status_markdown(session.snapshot())),
                            gr.update())
                payload = _retained_solver(tab)
                if not payload or not payload.get("observations"):
                    with session.condition:
                        session.status = (
                            "no solver state to sample from - run a search "
                            "first, or Import a session file that carries one")
                    return (gr.update(value=_status_markdown(session.snapshot())),
                            gr.update())
                _DEMO_SERIAL[tab] += 1
                _PENDING_DEMO[tab].append((bool(wanted), time.time()))
                with session.condition:
                    session.status = ("generating a "
                                      + ("GOOD" if wanted else "BAD")
                                      + " sample from the learned taste")
                return (gr.update(value=_status_markdown(session.snapshot())),
                        gr.update(value=f"go-{_DEMO_SERIAL[tab]}"))
            return handler

        good_button, bad_button = demo_buttons
        good_button.click(fn=demo(True), inputs=[], outputs=[status, demo_go],
                          show_progress=False, queue=False)
        bad_button.click(fn=demo(False), inputs=[], outputs=[status, demo_go],
                         show_progress=False, queue=False)

        def on_stop():
            session.request_stop()
            return gr.update(value=_status_markdown(session.snapshot()))

        stop.click(fn=on_stop, inputs=[], outputs=[status],
                   show_progress=False, queue=False)

    def _wire_set(self, apply_button, config, note, targets, unreachable):
        """Set: a configuration string onto the actual Forge Neo controls.

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

        def apply_config(text):
            values = _parse_config_string(text)
            if not values:
                return [gr.update(value="**Nothing to set** - the box above "
                                        "holds no configuration string.")] \
                    + [gr.update() for _ in targets]
            updates, applied = [], []
            for token, _component in targets:
                if token not in values:
                    updates.append(gr.update())
                    continue
                updates.append(gr.update(value=_coerce(token, values[token])))
                applied.append(token)

            missing = sorted(set(values) - set(applied))
            report = f"**Set** {', '.join(applied)}." if applied else "**Set nothing.**"
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
            logger.info(f"{WHO}: Set applied {applied or 'nothing'}"
                        + (f"; could not place {missing}" if missing else ""))
            return [gr.update(value=report)] + updates

        apply_button.click(
            fn=apply_config, inputs=[config],
            outputs=[note] + [component for _token, component in targets],
            show_progress=False)

    def _wire_io(self, export_button, import_button, session_file, note,
                 rows, row_count, headline, tail, targets, canvas_targets,
                 group_count, is_img2img, duel_group, duel_status):
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
        row_keys = ("target", "mode", "field", "choices", "line", "values",
                    "prompt_mode", "loras", "weight_low", "weight_high",
                    "point")

        export_inputs = [row_count]
        for row in rows:
            export_inputs.extend([row.target, row.mode, row.field,
                                  row.choices, row.line, row.values,
                                  row.prompt_mode, row.loras,
                                  row.weight_low, row.weight_high,
                                  row.point])
        export_inputs.extend(tail)
        export_inputs.extend(component for _token, component in targets)
        export_inputs.extend(component for _token, component in canvas_targets)

        def do_export(*values):
            it = iter(values)
            count = int(next(it) or 1)
            row_values = [{key: _jsonable(next(it)) for key in row_keys}
                          for _ in range(MAX_ROWS)]
            tail_values = {"vary_seed": _jsonable(next(it)),
                           "render_winner": _jsonable(next(it)),
                           "weight_step": _jsonable(next(it)),
                           "resume": _jsonable(next(it))}
            settings = {token: _jsonable(next(it)) for token, _c in targets}
            canvases = {}
            for token, _component in canvas_targets:
                uri = _image_to_data_uri(next(it))
                if uri:
                    canvases[token] = uri
            with session.condition:
                solver = session.solver_state
                result = session.result
            payload = {
                "app": WHO, "version": EXPORT_VERSION, "tab": tab,
                "row_count": count, "rows": row_values,
                "tail": tail_values, "settings": settings,
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
            import_outputs.extend([row.target, row.mode, row.field,
                                   row.choices, row.line, row.values,
                                   row.prompt_mode, row.lora_pick, row.loras,
                                   row.weight_low, row.weight_high,
                                   row.lora_refresh, row.lora_dice,
                                   row.point])
        import_outputs.extend(tail)
        import_outputs.extend(component for _token, component in targets)
        import_outputs.extend(component for _token, component in canvas_targets)
        # LAST: the duel group and its status line. An import that carries
        # solver state reveals them, because GOOD/BAD live inside the group
        # and "import a generator and sample from it" must not require a
        # search to run first just to make the buttons reachable.
        import_outputs.extend([duel_group, duel_status])

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
                    updates.extend(gr.update() for _ in range(14))
                    continue
                props = _row_visibility(data.get("target"), data.get("mode"),
                                        data.get("field"),
                                        data.get("weight_low"),
                                        data.get("weight_high"))
                target_value = data.get("target")
                choices = (available + [target_value]
                           if target_value and target_value not in available
                           else available)
                selected = [v for v in (data.get("choices") or [])]
                updates.append(gr.update(choices=choices, value=target_value))
                updates.append(gr.update(**{**props[0],
                                            "value": data.get("mode")}))
                updates.append(gr.update(**{**props[1],
                                            "value": data.get("field")}))
                # The value list's OPTIONS live in this build's model
                # registry; the imported selection is offered as-is so it
                # renders even where the registry differs - rescan restores
                # the full list.
                updates.append(gr.update(**{**props[2], "choices": selected,
                                            "value": selected}))
                updates.append(gr.update(**{**props[3],
                                            "value": data.get("line")}))
                updates.append(gr.update(**{**props[4],
                                            "value": data.get("values")}))
                updates.append(gr.update(**{**props[5],
                                            "value": data.get("prompt_mode")}))
                updates.append(gr.update(**props[6]))          # lora_pick
                updates.append(gr.update(**{**props[7],
                                            "value": data.get("loras")}))
                updates.append(gr.update(**{**props[8],
                                            "value": data.get("weight_low")}))
                updates.append(gr.update(**{**props[9],
                                            "value": data.get("weight_high")}))
                updates.append(gr.update(**props[10]))         # lora refresh
                updates.append(gr.update(**props[11]))         # lora dice
                # Files written before the point row existed have no "point";
                # 0 is its neutral and the mode that reads it is not theirs.
                updates.append(gr.update(**{**props[12],
                                            "value": data.get("point", 0)}))

            tail_values = payload.get("tail", {})
            updates.append(gr.update(value=bool(tail_values.get("vary_seed",
                                                                False))))
            updates.append(gr.update(value=bool(tail_values.get(
                "render_winner", True))))
            updates.append(gr.update(value=float(tail_values.get(
                "weight_step", LORA_WEIGHT_STEP) or LORA_WEIGHT_STEP)))
            updates.append(gr.update(value=bool(tail_values.get("resume",
                                                                True))))

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
            note_parts = [f"**Imported** {count} row(s), {len(applied)} "
                          f"setting(s), {painted} canvas image(s)."]
            if solver:
                note_parts.append(
                    f"Solver state ({solver.get('duels', 0)} duel(s)) is "
                    f"staged - press Generate with {WHO} to resume the "
                    f"search, or GOOD/BAD in the panel above to render "
                    f"samples from it without searching.")
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
            # Imported solver state makes the duel group useful with nothing
            # running - GOOD/BAD sample from it directly - so reveal it and
            # say so where those buttons are. Without state the group stays
            # as it was: revealing empty grading controls would promise a
            # duel that is not coming.
            if solver:
                with session.condition:
                    session.status = ("solver state imported - GOOD/BAD "
                                      "render samples from it; Generate "
                                      "resumes the search")
                updates.append(gr.update(visible=True))
                updates.append(gr.update(
                    value=_status_markdown(session.snapshot())))
            else:
                updates.extend([gr.update(), gr.update()])
            return updates

        import_button.click(fn=do_import, inputs=[session_file],
                            outputs=import_outputs, show_progress=False)

    # -- the search ------------------------------------------------------

    def _setup(self, p, args, ab_search):
        """Rows, space and base units out of the run's arguments.

        Everything in here can raise about a misconfiguration, which is why
        run() calls it BEFORE deciding whether this Generate is a search or
        a GOOD/BAD sample render: the two want different error handling - a
        search wants the stack trace, a sample press wants a status line -
        and both want the staged request consumed either way.
        """
        row_count = int(args[0] or 1)
        vary_seed, render_winner, weight_step, resume_search = args[
            1 + MAX_ROWS * ROW_ARGS:ARG_COUNT]
        rows = _read_rows(args, row_count,
                          float(weight_step or LORA_WEIGHT_STEP))
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
        # A unit that starts disabled is fine IF a row is what switches it on -
        # "is this control worth having at all" is a legitimate thing to search
        # over, and it is the one case where the check below would be wrong.
        switched = {gene.unit_index for gene in genes
                    if gene.kind == Gene.KIND_SETTING and gene.field == "enabled"}
        for gene in genes:
            if not gene.on_unit or gene.unit_index in base_units:
                continue
            slot = cnpro.args_from + gene.unit_index
            if slot >= cnpro.args_to:
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} targets unit "
                    f"{gene.unit_index}, which does not exist.")
            unit = base_args[slot]
            if not getattr(unit, "enabled", False) \
                    and gene.unit_index not in switched:
                raise RuntimeError(
                    f"{WHO}: row {gene.row + 1} targets unit "
                    f"{gene.unit_index}, which is not enabled. A disabled unit "
                    f"is dropped before the search is applied, so every duel "
                    f"would be identical. Enable it, or give a row its "
                    f"'{UNIT_FIELDS['enabled'].label}' setting so that the "
                    f"search is the thing that switches it on.")
            base_units[gene.unit_index] = unit
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
                # nobody connects it back to the row.
                count = module.profile_point_count(
                    base_units[gene.unit_index], gene.line)
                if not (-count <= gene.point_index < count):
                    raise RuntimeError(
                        f"{WHO}: row {gene.row + 1} offsets point "
                        f"{gene.point_index} of unit {gene.unit_index}'s "
                        f"{gene.line} profile, which has {count} point(s) - "
                        f"indices 0..{count - 1}, or negative from the end. "
                        f"Pick an existing point.")
        return (genes, space, cnpro, base_args, base_units,
                bool(vary_seed), bool(render_winner), bool(resume_search))

    def run(self, p, *args):
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

        session = _session(self.is_img2img)
        tab = "img2img" if self.is_img2img else "txt2img"
        # A GOOD/BAD press with no search running started THIS generation
        # itself (see _wire_duel). Its staged request is consumed FIRST,
        # before the rows are even looked at: an entry left behind by a run
        # that died in validation used to hijack a later, unrelated Generate
        # into a one-sample render - "the search stopped working", long
        # after the press that actually caused it.
        demo = self._pop_demo_request()
        try:
            (genes, space, cnpro, base_args, base_units,
             vary_seed, render_winner, resume_search) = \
                self._setup(p, args, ab_search)
        except Exception as exc:
            if demo is None:
                raise
            # The press asked for a sample, not for a stack trace. Rows that
            # do not validate - or no rows at all, on a freshly started UI -
            # mean the solver's space cannot be rebuilt here.
            session.start_demo("")
            session.finish(f"sample refused - {exc} GOOD/BAD need the rows "
                           f"the solver was trained on: recreate them, or "
                           f"Import the session file that carries them.")
            logger.warning(f"{WHO}: a GOOD/BAD sample was requested but the "
                           f"rows did not validate ({exc})")
            return Processed(p, [])

        # A comparison on two different seeds compares the seeds. Fixed unless
        # the user asks otherwise, and even then both sides of one duel share
        # the seed - what varies within a duel is only ever the configuration.
        processing.fix_seed(p)
        base_seed = int(p.seed)
        base_prompt = p.prompt
        base_extras = _generation_extras(p)

        # A GOOD/BAD press with no search running: render that sample from
        # the retained solver state. The search does not restart, nothing is
        # asked, and the state is left exactly as it was - for the next
        # press, or for the next real search.
        if demo is not None:
            return self._demo_run(demo, p, ab_search, space, genes, base_args,
                                  cnpro, base_units, base_prompt, base_seed,
                                  session)

        search = ab_search.PreferenceSearch(space, seed=base_seed)
        # What Generate continues from, in order of intent: a solver state
        # staged by Import is the one source the user pointed at explicitly;
        # otherwise, with "Resume search" ticked, the tab's own retained
        # state - the last search here, or the disk mirror that survived a
        # restart. Unticked, staged state STAYS staged (for a later Generate
        # with the box ticked) and this search starts fresh. Either way the
        # state only ever resumes onto the SAME search space: observations
        # replayed onto different rows would silently reindex every choice,
        # which is worse than starting over, so a mismatch is refused.
        staged, source = None, "imported"
        if resume_search:
            staged = _PENDING_SOLVER.pop(tab, None)
            if staged is None:
                staged = session.solver_state or _stored_solver(tab)
                source = "retained"
        resumed = 0
        if staged is not None:
            restored = _restore_solver(search, staged, space)
            if restored is None:
                if source == "imported":
                    logger.warning(
                        f"{WHO}: the imported solver state was learned on a "
                        f"DIFFERENT set of rows, so it cannot be replayed "
                        f"here - starting fresh. Recreate the rows it names "
                        f"(they are readable in the session file) to resume "
                        f"it.")
                else:
                    logger.info(
                        f"{WHO}: the retained solver state was learned on a "
                        f"different set of rows - starting fresh on these.")
            else:
                resumed = restored
                logger.info(f"{WHO}: resumed the {source} search - "
                            f"{restored} observation(s), "
                            f"{search.duels} graded duel(s).")
        idle = float(shared.opts.data.get("cnpro_ab_idle_timeout",
                                          IDLE_TIMEOUT_DEFAULT) or 0)

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
        demos = []       # GOOD/BAD samples rendered on demand, last 8 kept
        # LAST, so that nothing between here and the loop can raise with the
        # session marked running: the panel polls while it is, and a session
        # that is never finished is a panel that polls until the page is
        # reloaded. Every exit below goes through `finish`.
        session.start()
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
            payload = _solver_payload(search, base_seed, space)
            with session.condition:
                session.solver_state = payload
            _store_solver(tab, payload)
        try:
            while True:
                if session.stopping or state.interrupted \
                        or getattr(state, "stopping_generation", False):
                    break

                # What is REUSABLE for the duel about to be chosen: a side
                # whose recipe is in the render cache costs a lookup instead
                # of a ~30s generation, and the engine's reuse preference
                # (see ab_search.REUSE_MARGIN) trades on that - but only if
                # it is told the truth, which hangs on THIS duel's seed. The
                # recipe is the cache key, so with "vary seed" on every duel
                # renders on a fresh seed, no recipe can repeat, and the
                # probe correctly answers "nothing is reusable" - the
                # engine then runs exactly as it did before the probe
                # existed. Rebuilt per duel because both the seed and the
                # cache's contents move under it.
                next_seed = self._seed(base_seed, duel + 1, vary_seed)
                search.reuse_probe = lambda pt, _seed=next_seed: (
                    _config_string(genes, pt, base_units, base_prompt,
                                   _seed, extras=base_extras) in renders)

                point_a, point_b = search.next_duel()
                duel += 1
                session.say(f"duel {duel}: generating A", generating="A")
                state.job = f"DNA duel {duel} - A"
                processed_a = self._render(p, base_args, cnpro, genes, point_a,
                                           base_units, base_prompt,
                                           self._seed(base_seed, duel, vary_seed),
                                           side="A", session=session,
                                           cache=renders, extras=base_extras)
                if session.stopping or state.interrupted:
                    break
                session.say(f"duel {duel}: generating B", generating="B")
                state.job = f"DNA duel {duel} - B"
                processed_b = self._render(p, base_args, cnpro, genes, point_b,
                                           base_units, base_prompt,
                                           self._seed(base_seed, duel, vary_seed),
                                           side="B", session=session,
                                           cache=renders, extras=base_extras)
                if session.stopping or state.interrupted:
                    break
                if not processed_a.images or not processed_b.images:
                    logger.error(f"{WHO}: a duel produced no image; stopping.")
                    stopped = "stopped - a generation failed"
                    break

                last = (processed_a, processed_b)
                session.publish(duel, processed_a.images[0], processed_b.images[0],
                                "waiting for your grade")
                answer = session.await_grade(idle)
                # GOOD/BAD are not answers: render one fresh guess from the
                # asked-for end of the solver's belief - pure inference, no
                # observation, no effect on the duel - then return to the
                # SAME duel, still waiting. The full recipe goes to the
                # trace; the image goes to the output folder now and the
                # gallery at stop.
                while answer in ("demo_good", "demo_bad"):
                    wanted = answer == "demo_good"
                    guess = search.suggest(good=wanted)
                    tag = "GOOD" if wanted else "BAD"
                    recipe = _config_string(genes, guess, base_units,
                                            base_prompt, base_seed,
                                            extras=base_extras)
                    session.record(f"?{tag}", recipe)
                    state.job = f"DNA {tag} sample"
                    shown = self._render(p, base_args, cnpro, genes, guess,
                                         base_units, base_prompt, base_seed,
                                         session=session, cache=renders,
                                         extras=base_extras)
                    if shown.images:
                        demos.append(shown)
                        del demos[:-8]
                    if session.stopping or state.interrupted:
                        break
                    session.publish(duel, processed_a.images[0],
                                    processed_b.images[0],
                                    f"{tag} sample rendered (recipe in Tried, "
                                    f"image in the output folder) - back to "
                                    f"the duel, waiting for your grade")
                    answer = session.await_grade(idle)
                if answer is None:
                    break
                if answer == "skip":
                    # Nothing observed, nothing recomputed: a skipped duel is
                    # the statement that the PAIR could not be judged, which
                    # says nothing about either configuration.
                    continue

                score, disliked, (mark_a, mark_b) = answer
                search.observe(point_a, point_b, score, disliked=disliked,
                               interesting_a=mark_a, interesting_b=mark_b)
                best = search.best()
                # base_seed even when the seed varies per duel: the winner is
                # rendered on it, so it is the seed that reproduces the image
                # this recipe is about.
                result = _config_string(genes, best, base_units, base_prompt,
                                        base_seed, extras=base_extras)
                session.set_result(result, _summary(genes, best, search))
                # Every grade lands in the retained state AND its disk
                # mirror, so nothing that can end this loop - STOP, an
                # Interrupt, the idle timeout, a crash, even a restart -
                # costs more than the interruption itself.
                payload = _solver_payload(search, base_seed, space)
                with session.condition:
                    session.solver_state = payload
                _store_solver(tab, payload)
        except Exception as exc:
            errors.display(exc, "running the CNPro A/B search")
            stopped = f"stopped - {exc}"

        interrupted = state.interrupted or getattr(state, "stopping_generation", False)
        if interrupted:
            stopped = "interrupted - Generate resumes the search"
        if session.timed_out:
            # Not an ending, a pause: the state this search reached is
            # retained (and mirrored to disk), so coming back hours later
            # costs nothing but pressing Generate.
            stopped = (f"paused - no grade for {idle:g} min; "
                       f"Generate resumes where it left off")
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
        # champion appear nowhere else.
        keepers = search.frontier()
        for rank, keeper in enumerate(keepers, 1):
            recipe = _config_string(genes, keeper, base_units, base_prompt,
                                    base_seed, extras=base_extras)
            session.record(f"★{rank}", recipe)
            logger.info(f"{WHO}: keeper {rank}: {recipe}")

        # The keepers are rendered LAST and only on a deliberate stop: an
        # Interrupt is the user asking for no more generations, and honouring
        # it by starting more would be the opposite of what they pressed.
        # `finish` comes after, so the panel keeps polling and says what is
        # happening instead of going quiet for the whole render.
        if render_winner and not interrupted:
            rendered = []
            for rank, keeper in enumerate(keepers, 1):
                # STOP already happened - that is how this path is reached -
                # so the abort signal DURING the keeper renders is the host's
                # Interrupt, and it is honoured between generations.
                if state.interrupted or getattr(state, "stopping_generation",
                                                False):
                    break
                state.job = f"DNA keeper {rank}"
                session.say(f"rendering keeper {rank} of {len(keepers)}")
                rendered.append(self._render(
                    p, base_args, cnpro, genes, keeper, base_units,
                    base_prompt, base_seed, side=f"★{rank}",
                    session=session, cache=renders, extras=base_extras))
            rendered = [r for r in rendered if r.images]
            plural = "" if len(rendered) == 1 else "s"
            session.finish(f"{ending}; {len(rendered)} keeper{plural} in the "
                           f"gallery")
            rendered.extend(demos)
            if rendered:
                merged = copy.copy(rendered[0])
                merged.images = [r.images[0] for r in rendered]
                merged.infotexts = [r.infotexts[0] for r in rendered]
                merged.all_prompts = [r.all_prompts[0] for r in rendered]
                merged.all_seeds = [r.all_seeds[0] for r in rendered]
                return merged
        else:
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
        """The oldest fresh staged GOOD/BAD request for this tab, or None.

        Stale entries are dropped with a log line - see DEMO_STALE_SECONDS:
        a press stages its entry and clicks Generate in the same instant, so
        an old survivor means that click never became a run, and honouring
        it would turn a later, unrelated Generate into a sample render the
        user did not ask for.
        """
        staged = _PENDING_DEMO["img2img" if self.is_img2img else "txt2img"]
        while staged:
            wanted, when = staged.pop(0)
            if time.time() - when <= DEMO_STALE_SECONDS:
                return wanted
            logger.warning(
                f"{WHO}: dropped a stale GOOD/BAD request - the press that "
                f"staged it never became a run, so this Generate is treated "
                f"as the search it looks like.")
        return None

    def _demo_run(self, wanted, p, ab_search, space, genes, base_args, cnpro,
                  base_units, base_prompt, base_seed, session):
        """GOOD/BAD with no search running: render sample(s) from the
        retained solver state, and touch nothing else.

        The state comes from an Imported session, the last search on this
        tab, or the disk mirror a restart left behind - see _retained_solver
        - and is only ever READ: no observation
        is added, the staged import stays staged for a later real search, and
        a plain Generate afterwards still starts the search it always
        started. Everything the rows do not own - prompt, canvases,
        resolution, seed - was read fresh from the UI by THIS generation,
        which is what makes the buttons a generator the rest of the UI can
        be tweaked under at will: reach a good state once, then sample from
        it for as long as it is useful.

        The replayed engine is deterministic, so the solver is rebuilt on a
        seed that changes per press (_DEMO_SERIAL) - without that, "varied
        per press" would render the same sample every time.
        """
        tab = "img2img" if self.is_img2img else "txt2img"
        payload = _retained_solver(tab)
        label = "GOOD" if wanted else "BAD"
        if not payload or not payload.get("observations"):
            session.start_demo("")
            session.finish("no solver state to sample from - run a search "
                           "first, or Import a session file that carries one")
            logger.warning(f"{WHO}: a {label} sample was requested with no "
                           f"solver state on this tab.")
            return Processed(p, [])

        search = ab_search.PreferenceSearch(
            space, seed=base_seed + 7919 * _DEMO_SERIAL[tab])
        if _restore_solver(search, payload, space) is None:
            session.start_demo("")
            session.finish(
                f"{label} sample refused - the solver state was learned on a "
                f"DIFFERENT set of rows, so its knowledge does not fit them. "
                f"Recreate the rows it names (readable in the session file), "
                f"or run a fresh search on these.")
            logger.warning(f"{WHO}: the retained solver state does not fit "
                           f"the current rows, so it cannot generate samples "
                           f"for them.")
            return Processed(p, [])

        session.start_demo(f"rendering a {label} sample from the learned "
                           f"taste")
        p.extra_generation_params["Script"] = self.title()
        extras = _generation_extras(p)
        made = []
        queue = [wanted]
        try:
            while queue and len(made) < DEMO_BATCH_MAX:
                if session.stopping or state.interrupted \
                        or getattr(state, "stopping_generation", False):
                    break
                good = queue.pop(0)
                tag = "GOOD" if good else "BAD"
                guess = search.suggest(good=good)
                recipe = _config_string(genes, guess, base_units, base_prompt,
                                        base_seed, extras=extras)
                session.record(f"?{tag}", recipe)
                p.extra_generation_params["CNPro DNA"] = recipe
                state.job = f"DNA {tag} sample"
                state.job_count = len(made) + len(queue) + 1
                shown = self._render(p, base_args, cnpro, genes, guess,
                                     base_units, base_prompt, base_seed,
                                     session=session, extras=extras)
                if shown.images:
                    made.append(shown)
                    session.publish_demo(
                        shown.images[0],
                        f"{tag} sample rendered - GOOD/BAD render more, "
                        f"varied per press; the solver state is kept")
                # Presses that landed while rendering queued themselves on
                # the session (request_demo sees a live run) - serve them
                # here rather than costing one host job per press.
                with session.condition:
                    queue.extend(session.pending_demos)
                    session.pending_demos = []
        except Exception as exc:
            errors.display(exc, "rendering a CNPro A/B sample")
            session.finish(f"stopped - {exc}")
            return Processed(p, [])

        dropped = (f" ({len(queue)} press(es) beyond the cap dropped - "
                   f"press again)" if queue else "")
        session.finish(f"{len(made)} sample(s) rendered{dropped} - the "
                       f"solver state is kept: GOOD/BAD keep working, and "
                       f"the rest of the UI can be changed freely between "
                       f"presses")
        if not made:
            return Processed(p, [])
        final = made[0]
        for extra in made[1:]:
            final.images.extend(extra.images)
            final.infotexts.extend(extra.infotexts)
        return final

    @staticmethod
    def _seed(base_seed, duel, vary_seed):
        """The seed BOTH sides of duel `duel` run on.

        Varying it across duels trades a noisier comparison for a
        recommendation that is about the configuration rather than about one
        lucky noise field; varying it WITHIN a duel would compare the seeds,
        which is why that is not on offer.
        """
        return base_seed + duel if vary_seed else base_seed

    def _render(self, p, base_args, cnpro, genes, point, base_units,
                base_prompt, seed, side=None, session=None, cache=None,
                extras=None):
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
        if cache is not None and processed.images:
            cache[recipe] = processed
            while len(cache) > RENDER_CACHE:
                cache.pop(next(iter(cache)))
        return processed


class _Row:
    """The components of one row, by name rather than by position."""

    def __init__(self, container, target, mode, field, choices, line, values,
                 prompt_mode, lora_pick, loras, weight_low, weight_high,
                 lora_refresh, lora_dice, point):
        self.container = container
        self.target = target
        self.mode = mode
        self.field = field
        self.choices = choices
        self.line = line
        self.values = values
        self.prompt_mode = prompt_mode
        self.lora_pick = lora_pick
        self.loras = loras
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.lora_refresh = lora_refresh
        self.lora_dice = lora_dice
        self.point = point


def _profile_lines():
    return getattr(_profile_scale(), "PROFILE_LINES", {"Main": None})


def _read_rows(args, row_count, weight_step):
    """The visible rows, as genes.

    Rows past the count are skipped rather than read: they are still in the
    args (gradio sends every component's value, visible or not) and they still
    hold whatever was typed into them before they were hidden. Reading them
    would make "-" a button that hides a row and keeps searching over it.
    """
    genes = []
    for index in range(min(row_count, MAX_ROWS)):
        base = 1 + index * ROW_ARGS
        (target, mode, field_label, choices, line, values, prompt_mode, loras,
         weight_low, weight_high, point_index) = args[base:base + ROW_ARGS]

        def weights(fallback):
            low = weight_low if weight_low is not None else fallback[0]
            high = weight_high if weight_high is not None else fallback[1]
            return float(low), float(high)

        if target == TARGET_PROMPT:
            prompts = _lines(values)
            if prompts:
                low, high = weights((PROMPT_WEIGHT_MIN, PROMPT_WEIGHT_MAX))
                genes.append(Gene(Gene.KIND_PROMPT, index, values=prompts,
                                  prompt_mode=prompt_mode or PROMPT_REPLACE,
                                  weight_low=low, weight_high=high,
                                  weight_step=weight_step))
            continue

        if target == TARGET_LORA:
            names = [_lora_name(item) for item in _lines(loras)]
            names = [name for name in names if name]
            if names:
                low, high = weights((LORA_WEIGHT_MIN, LORA_WEIGHT_MAX))
                genes.append(Gene(Gene.KIND_LORA, index, loras=names,
                                  weight_low=low, weight_high=high,
                                  weight_step=weight_step))
            continue

        unit_index = _unit_index(target)
        if unit_index is None:
            continue

        if mode in (MODE_PROFILE, MODE_POINT):
            module = _profile_scale()
            if module is None:
                raise RuntimeError(
                    f"{WHO}: row {index + 1} varies a profile, but ControlNet "
                    f"Pro is not loaded.")
            if line not in module.PROFILE_LINES:
                raise ValueError(
                    f"{WHO}: row {index + 1}: unknown profile line {line!r}.")
            if mode == MODE_POINT:
                offsets = module.parse_offsets(values)
                if offsets:
                    genes.append(Gene(Gene.KIND_POINT, index,
                                      unit_index=unit_index, line=line,
                                      values=offsets,
                                      point_index=int(point_index or 0)))
                continue
            factors = module.parse_factors(values)
            if factors:
                genes.append(Gene(Gene.KIND_PROFILE, index,
                                  unit_index=unit_index, line=line,
                                  values=factors))
            continue

        field_name = FIELD_LABELS.get(field_label)
        if field_name is None:
            continue
        field = UNIT_FIELDS[field_name]
        if field.kind == KIND_CHOICE:
            picked = [v for v in (choices or []) if v]
        elif field.kind == KIND_NUMBER:
            picked = _numbers(values)
        else:
            picked = _lines(values)
        if picked:
            genes.append(Gene(Gene.KIND_SETTING, index, unit_index=unit_index,
                              field=field_name, values=picked))
    return genes


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
    if token in ("cfg_scale", "denoising_strength"):
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
    return (f"**{count}** degree{'s' if count != 1 else ''} of freedom "
            f"- one per row. Add rows for more.")


#: The two ends of each weighted row kind's default range, by (min, max).
WEIGHT_DEFAULTS = {
    TARGET_LORA: (LORA_WEIGHT_MIN, LORA_WEIGHT_MAX),
    TARGET_PROMPT: (PROMPT_WEIGHT_MIN, PROMPT_WEIGHT_MAX),
}


#: The two weight boxes' labels. Abbreviated, and both on ONE line each, which
#: is the whole requirement: a label of "prompt weight" wraps to two lines in a
#: 70px box, so the pair rendered as two identical two-line captions with
#: nothing saying which end was which.
WEIGHT_LABELS = ("min. w.", "max. w.")


def _weight_default(target, current, end):
    """`{"value": ...}` when this end should be reset, `{}` when it should not.

    A row that changes kind should offer the new kind's usual range - a LoRA's
    0..1 is a de-emphasis-only range for a prompt term, and a prompt's 0.5..1.5
    cannot switch a LoRA off. But a row whose range the USER typed must keep
    it, and the handler cannot see what the row used to be. So the test is on
    the value itself: reset it only while it still holds one of the defaults,
    which is exactly the case where nobody has expressed an opinion.
    """
    defaults = WEIGHT_DEFAULTS.get(target)
    if defaults is None:
        return {}
    untouched = {values[end] for values in WEIGHT_DEFAULTS.values()}
    try:
        current = float(current)
    except (TypeError, ValueError):
        return {"value": defaults[end]}
    if any(abs(current - value) < 1e-9 for value in untouched):
        return {"value": defaults[end]}
    return {}


def _row_visibility(target_value, mode_value, field_label, low, high):
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
    setting = is_unit and mode_value == MODE_SETTING
    profile = is_unit and mode_value == MODE_PROFILE
    point_mode = is_unit and mode_value == MODE_POINT
    is_lora = target_value == TARGET_LORA
    is_prompt = target_value == TARGET_PROMPT
    field_name = FIELD_LABELS.get(field_label, "model")
    kind = UNIT_FIELDS[field_name].kind
    weighted = is_lora or is_prompt
    return [
        {"visible": is_unit},                                   # mode
        {"visible": setting},                                   # field
        {"visible": setting and kind == KIND_CHOICE},           # choices
        {"visible": profile or point_mode},                     # line
        # ONE update carries visibility, label and placeholder together: the
        # values box is shared by four kinds of row.
        {"visible": (setting and kind != KIND_CHOICE) or profile
                    or point_mode or is_prompt,
         "label": _values_label(target_value, mode_value, field_label),
         "placeholder": _values_placeholder(target_value, mode_value,
                                            field_label)},      # values
        {"visible": is_prompt},                                 # prompt mode
        {"visible": is_lora},                                   # picker
        {"visible": is_lora},                                   # loras
        # The weight range serves BOTH weighted kinds, and its sensible
        # bounds differ between them; _weight_default resets an end only
        # while it still holds one of the defaults, so a range the user has
        # typed is never overwritten by changing the row's target.
        {"visible": weighted, **_weight_default(target_value, low, 0)},
        {"visible": weighted, **_weight_default(target_value, high, 1)},
        {"visible": is_lora},                                   # refresh
        {"visible": is_lora},                                   # dice
        {"visible": point_mode},                                # point
    ]


def _values_label(target, mode, field_label):
    if target == TARGET_PROMPT:
        return "prompts (one per line)"
    if _unit_index(target) is not None and mode == MODE_PROFILE:
        return "factors"
    if _unit_index(target) is not None and mode == MODE_POINT:
        return "offsets (0 = as drawn)"
    return "values (one per line)"


def _values_placeholder(target, mode, field_label):
    if target == TARGET_PROMPT:
        return "a house on a hill\na house in a storm\n…"
    if _unit_index(target) is not None and mode == MODE_PROFILE:
        return "0.5, 0.75, 1, 1.5   or   0.5:1.5:5"
    if _unit_index(target) is not None and mode == MODE_POINT:
        return "-0.2, 0, 0.2   or   -0.2:0.2:5"
    field_name = FIELD_LABELS.get(field_label, "model")
    if UNIT_FIELDS[field_name].kind == KIND_NUMBER:
        return "768, 1024, 1280   or   768:1280:5"
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


def _status_markdown(snap):
    """The three lines under the duel. Always three, and always one line each.

    THE HEIGHT OF THIS BLOCK IS PART OF ITS CONTRACT. It used to say more
    while waiting for a grade than while generating - the scale legend and the
    best-so-far appeared and disappeared with the phase - which changed its
    height by 50px twice per duel and shoved the Configuration box, the Set
    button and the trace down and back up again every few seconds, for as long
    as the search ran. Nothing was flickering: the panel was resizing.

    So the lines are fixed. The phase changes inside line one, the other two
    are always present, and line one and the summary are both clipped to a
    single line with the whole of it on hover (a long status used to WRAP
    inside its fixed-height line and paint over the legend under it) - the
    full recipe is in the box below, which is where anybody reading it in
    detail is going to look anyway.

    Emitted as HTML rather than markdown because those three properties belong
    to specific lines: markdown's `  \\n` gives `<br>`s inside one paragraph,
    which cannot be clipped individually. Every value that comes from the user
    - prompts, model names, LoRA names - is escaped on the way in.
    """
    head = f"<b>Duel {snap['duel']}</b> · {snap['graded']} graded"
    head_title = ""
    if snap["status"]:
        head_title = html.escape(str(snap["status"]))
        head += f" · {head_title}"
    legend = ("0 = A much better · 5 = even · 10 = B much better &nbsp; "
              "same grade on the lower row = and I dislike both")
    summary = html.escape(snap["summary"]) if snap["summary"] else \
        "nothing graded yet - the first answer starts the model off"
    return (f"<div class='cnpro-ab-line cnpro-ab-clip' "
            f"title='{head_title}'>{head}</div>"
            f"<div class='cnpro-ab-line cnpro-ab-dim'>{legend}</div>"
            f"<div class='cnpro-ab-line cnpro-ab-dim cnpro-ab-clip' "
            f"title='{summary}'>{summary}</div>")


def _inference_progress():
    """Where the current render is, 0..1, straight off the host's counters.

    Read live at tick time rather than book-kept on the session: the sampler
    already publishes its step for the host's own progress bar, and a second
    account of the same number could only ever disagree with it. Which RENDER
    the number belongs to is the session's `generating` - the tick never uses
    this without it.
    """
    steps = int(getattr(state, "sampling_steps", 0) or 0)
    if steps <= 0:
        return 0.0
    step = float(getattr(state, "sampling_step", 0) or 0)
    return min(max(step / steps, 0.0), 1.0)


def _phase_css(eid, snap, progress):
    """The tick's <style> payload: what the current PHASE paints onto chrome
    gradio components cannot carry themselves.

    * While a side renders, that side's "interesting" toggle wears the
      inference progress as a background fill (`--cnpro-ab-fill`, style.css)
      - the button stays a button, the fill is a pseudo-element behind its
      label. While B renders, A shows a FULL bar: the pair then reads as
      "A finished, B under way" at a glance.
    * While a duel awaits its grade, the grade-here nudge beside the scale
      is faded in - generation over, the next move is the user's. It lives
      in CSS opacity rather than component visibility so that its appearing
      cannot move a single pixel of the rows it points at.

    Empty when the phase needs nothing painted, which is the idle common
    case the tick then dedupes away entirely.
    """
    rules = []
    if snap["running"] and snap["generating"] in ("A", "B"):
        percent = int(round(min(max(progress, 0.0), 1.0) * 100))
        if snap["generating"] == "A":
            rules.append(f"#{eid}_interesting_a{{--cnpro-ab-fill:{percent}%}}")
        else:
            rules.append(f"#{eid}_interesting_a{{--cnpro-ab-fill:100%}}")
            rules.append(f"#{eid}_interesting_b{{--cnpro-ab-fill:{percent}%}}")
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


def _on_ui_settings():
    section = ("cnpro", "CNPro")
    shared.opts.add_option("cnpro_ab_idle_timeout", shared.OptionInfo(
        IDLE_TIMEOUT_DEFAULT,
        "A/B search: pause after this many minutes with no grade - the "
        "learned state is kept and the next Generate resumes it "
        "(0 = wait forever, which holds the tab until Interrupt)",
        gr.Slider, {"minimum": 0, "maximum": 240, "step": 5}, section=section))


script_callbacks.on_ui_settings(_on_ui_settings)
