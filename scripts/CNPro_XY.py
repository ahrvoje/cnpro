"""CNPro X/Y - a two-axis sweep over ControlNet Pro units.

Drop this file into `<forge-neo>/scripts/` (or into `<this extension>/scripts/`)
and it appears in the Script dropdown as "CNPro X/Y".

WHAT EACH AXIS CAN SWEEP
------------------------
Each axis picks ONE target. A target is either an enabled CNPro unit, refined
by one of two modes -

* **Model** - the models the chosen unit's own model dropdown is currently
  offering, i.e. the list its Control Type filter has narrowed to. "None" is
  offered deliberately: it is the control-off baseline row.
* **Profile** - one of the unit's SIX profile lines (Main, Coarse, Mid, Fine,
  Depth, Drift) scaled by each listed factor: 1 is the curve untouched, 0
  switches that line off, 2 is twice the control.
* **Profile point** - ONE drawn point of one of those lines, moved vertically
  by each listed offset: 0 is the profile exactly as drawn, +0.1 is that
  point 0.1 higher in the plot's own axis units, negative moves it down. The
  same edit dragging the point in the editor would make, swept - "this
  curve, but with the step-0.75 knee higher/lower" is a question about a
  point that no whole-line factor can ask. Point indices count drawn points
  left to right from 0; negative counts from the end (-1 = rightmost).

- or one of the two targets that belong to the generation rather than to any
control:

* **Prompt** - one prompt per line, which either REPLACES the prompt or is
  prepended/appended to it. Replace is the substitution grid; the other two make
  the axis a modifier, so "the same scene under four lightings" does not mean
  retyping the scene four times.
* **LoRA** - one cell per line, each APPENDED to whatever prompt the cell ended
  up with, so it composes with a Prompt axis on the other side.

The axes are independent: the same unit on both (models across, factors down),
two different units, or a unit against a prompt - in any combination.

HOW A PROFILE LINE IS SCALED
----------------------------
Not by moving the drawn points: the factor is applied to the profile's `|lo~hi`
scale range, which is an exact scaling of the effective weight that leaves
every drawn feature intact. `lib_cnpro/profile_scale.py` is where that lives
and why - including why the clamp to [-1, 2] makes an extreme factor squash the
curve rather than scale it (logged per cell), and why a negative factor is
refused instead of silently becoming something else.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
Nothing is written back to the UI. Each cell gets a shallow COPY of the unit
dataclass, placed in a copy of `p.script_args`; the gr.State the unit panel is
bound to is never touched. A sweep therefore leaves the panel exactly as the
user set it, and an interrupted sweep leaves nothing to undo.
"""

import copy
import logging
import os
import random
import re
import shutil
import subprocess
import sys

import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules import errors, images, processing, script_callbacks, shared
from modules.processing import Processed, process_images
from modules.shared import opts, state
from modules.ui_components import ToolButton

logger = logging.getLogger("CNPro_XY")

REFRESH_SYMBOL = "\U0001f504"  # the same glyph the unit's own model refresh uses
DICE_SYMBOL = "\U0001f3b2"

MODE_MODEL = "Model"
MODE_PROFILE = "Profile"
MODE_POINT = "Profile point"
MODES = [MODE_MODEL, MODE_PROFILE, MODE_POINT]

#: What a Prompt axis does with the prompt already in the box. Replace is the
#: original behaviour and stays the default; the other two make a prompt axis
#: a MODIFIER of the user's prompt rather than a substitute for it, which is
#: what "same prompt, four different lighting phrases" needs.
PROMPT_REPLACE = "Replace"
PROMPT_PREPEND = "Prepend"
PROMPT_APPEND = "Append"
PROMPT_MODES = [PROMPT_REPLACE, PROMPT_PREPEND, PROMPT_APPEND]

#: The first dropdown picks WHAT an axis varies, and that is not always a unit.
#: A unit is then refined by the Model/Profile mode; these two are whole targets
#: on their own because they belong to the generation rather than to any one
#: control - the main prompt, and the LoRAs riding in it.
TARGET_PROMPT = "Prompt"
TARGET_LORA = "LoRA"
GLOBAL_TARGETS = [TARGET_PROMPT, TARGET_LORA]

ARG_COUNT = 22  # controls returned by ui(); see the guard in run()

#: Weight a freshly picked LoRA is inserted at. The point of the textbox is that
#: this is then edited by hand, so the default only has to be neutral.
#:
#: MIRRORED in javascript/cnpro_xy.js, which is what actually inserts the line
#: when that file is loaded - see the note on `add_lora` below.
LORA_DEFAULT_WEIGHT = 1

#: What the dice draws a weight from. The low end is 0.1 rather than 0: a LoRA
#: at 0 is the LoRA not being there, which is a cell the grid already has for
#: free in every row that does not name it.
LORA_RANDOM_WEIGHTS = (0.1, 1.0)

RE_MODEL_HASH = re.compile(r"\s*\[[0-9a-fA-F]{6,}\]\s*$")
RE_UNIT_LABEL = re.compile(r"^Unit\s+(\d+)")

#: How a legend cell is broken into lines.
#:
#: THE HOST WRAPS ON SPACES AND NOTHING ELSE. `images.draw_grid_annotations`
#: splits with `text.split()`, so a token with no space in it - which is what
#: every model name, every profile string and every LoRA tag is - cannot wrap
#: at all. It then SHRINKS the font until that one line fits the cell, one
#: point at a time, with no floor worth the name. A long enough name is
#: therefore drawn at a size nobody can read, and the grid looks like it
#: truncated something.
#:
#: So the lines are decided here and handed over already broken (the API takes
#: a LIST of annotations per cell, one per line). The host still shrinks each
#: line to fit, which is the right behaviour once the lines are short enough
#: for that to be a nudge rather than a collapse.
#:
#: LEGEND_LINE_CHARS is a target, not a maximum: the line count is chosen from
#: it and the text is then divided EVENLY, so a 30-character name comes out as
#: two 15s rather than a 22 and an 8.
LEGEND_LINE_CHARS = 22
LEGEND_MAX_LINES = 4

#: Prompts are the one thing not shown in full, and they are excluded for a
#: reason no wrapping can fix: a prompt is routinely hundreds of characters,
#: and a legend that honours that is a grid with more caption than image. 64
#: is about what identifies a prompt at a glance; the rest is elided.
PROMPT_LEGEND_CHARS = 64

#: Where a legend line may break. Spaces, and also the punctuation that holds
#: machine-generated names together - `diffusers_xl_canny_full` has no space
#: in it and four perfectly good places to break.
RE_LEGEND_CHUNK = re.compile(r"[^\s_\-/:.,|]+[\s_\-/:.,|]*|[\s_\-/:.,|]+")


# ---------------------------------------------------------------------------
# Reaching CNPro
#
# This file can live in the webui's own scripts/ directory, where the
# extension's basedir is NOT on sys.path. Every CNPro import is therefore
# deferred to call time, by which point cnpro.py has been imported and its
# modules are in sys.modules - which is what `import` consults first. Absence is
# handled rather than assumed away: with the extension disabled this script has
# to say so, not raise ImportError while the page is being built.
# ---------------------------------------------------------------------------

def _global_state():
    """CNPro's model registry, or None if CNPro is not loaded.

    Separate from _profile_scale on purpose: these two are needed by different
    halves of this script (model choices vs profile arithmetic) and one failing
    to import must not take the other's feature down with it.
    """
    try:
        from lib_cnpro import global_state
    except Exception:
        return None
    return global_state


def _profile_scale():
    """The shared profile-scaling arithmetic, or None if CNPro is not loaded.

    Shared with CNPro A/B, which scales the same six lines while searching
    rather than while sweeping. One implementation because the clamping rule
    there is the kind of thing two copies drift on silently - both would still
    produce a curve, just not the same one.
    """
    try:
        from lib_cnpro import profile_scale
    except Exception:
        return None
    return profile_scale


def _profile_lines():
    """The six profile lines, in the editor's own order.

    A function rather than a module constant, for the same reason every other
    CNPro import here is deferred: this file is importable with the extension
    absent, and a constant would have to be resolved at import time. Empty is
    the CNPro-absent case, and every caller is already behind a check that the
    axis targets a unit - which needs CNPro anyway.
    """
    return getattr(_profile_scale(), "PROFILE_LINES", {})


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
    sys.path while that extension is loading, and whose module name ("networks")
    is generic enough that importing it blind could pick up something else. The
    two attributes are the identity test.
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
# Choices
# ---------------------------------------------------------------------------

def _unit_label(index):
    return f"Unit {index}"


def _unit_index(label):
    match = RE_UNIT_LABEL.match(str(label or ""))
    return int(match.group(1)) if match else None


def _target_choices(flags):
    """What an axis can vary: the activated units, then Prompt and LoRA.

    The unit half comes from the enable checkboxes' own values, NOT from the
    unit gr.States. `enabled` reaches a State through CNPro's own field_updater,
    which is a SEPARATE handler on the very same checkbox event - so a handler
    that reads the State sees the value from before the click and the unit list
    runs permanently one step behind. The checkbox value arrives with the event
    that fired it and cannot be stale.

    Prompt and LoRA are always offered: they need no unit, and an axis on either
    is useful even with no control enabled at all.
    """
    return [_unit_label(i) for i, on in enumerate(flags) if on] + GLOBAL_TARGETS


def _models_for_type(control_type):
    """The model list a unit's own dropdown is showing under `control_type`.

    Mirrors global_state.select_control_type's model half rather than reading
    the dropdown component: the choices the user sees were pushed there by a
    gr.update and never landed back on the Python object. The Control Type value
    IS live (it arrives as a gradio input), so this is exact.
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


def _short_model(name):
    """Model name without its ` [hash]` suffix - grid annotations are narrow."""
    return RE_MODEL_HASH.sub("", str(name or "")).strip() or "None"


# ---------------------------------------------------------------------------
# Profile factors
#
# The arithmetic lives in lib_cnpro/profile_scale.py, shared with CNPro A/B -
# see the note on _profile_scale above. These are the thin local names the axis
# code reads with, plus the CNPro-absent case: a Profile axis targets a unit,
# so a missing CNPro is already a hard error by the time any of them runs, and
# the guard is here so that the failure SAYS that rather than raising
# AttributeError on None three frames deeper.
# ---------------------------------------------------------------------------

WHO = "CNPro X/Y"


def _profile_scale_or_die():
    module = _profile_scale()
    if module is None:
        raise RuntimeError(
            "CNPro X/Y: an axis scales a profile, but ControlNet Pro is not "
            "loaded. Enable the extension and restart.")
    return module


def _parse_factors(text):
    return _profile_scale_or_die().parse_factors(text)


def _parse_offsets(text):
    return _profile_scale_or_die().parse_offsets(text)


def _scale_profile(unit, unit_index, line, factor):
    return _profile_scale_or_die().scale_profile(
        unit, unit_index, line, factor, who=WHO)


def _offset_profile_point(unit, unit_index, line, point_index, offset):
    return _profile_scale_or_die().offset_profile_point(
        unit, unit_index, line, point_index, offset, who=WHO)


def _warn_if_inert(unit, index, line):
    module = _profile_scale()
    if module is not None:
        module.warn_if_inert(unit, index, line, who=WHO)


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

class Cell:
    """One value on one axis: a legend label, and what it changes.

    Two kinds of thing get varied here and they are applied in different places.
    `model` and `profile` edit a COPY OF A UNIT in the CNPro args window;
    `prompt` and `lora` edit the StableDiffusionProcessing itself. Keeping both
    in one class keeps the grid loop from having to know which is which, but the
    split is real - see apply_unit / apply_processing.
    """

    KIND_NONE, KIND_MODEL, KIND_PROFILE = "none", "model", "profile"
    KIND_POINT = "point"
    KIND_PROMPT, KIND_LORA = "prompt", "lora"

    def __init__(self, kind=KIND_NONE, unit_index=None, model=None,
                 line=None, factor=None, point_index=None, offset=None,
                 text=None, prompt_mode=PROMPT_REPLACE):
        self.kind = kind
        self.unit_index = unit_index
        self.model = model
        self.line = line
        self.factor = factor
        self.point_index = point_index
        self.offset = offset
        self.text = text
        self.prompt_mode = prompt_mode

    @property
    def on_unit(self):
        return self.kind in (Cell.KIND_MODEL, Cell.KIND_PROFILE,
                             Cell.KIND_POINT)

    @property
    def label(self):
        """What this cell says in the legend, IN FULL.

        Nothing is shortened here except a prompt - see PROMPT_LEGEND_CHARS.
        The text is broken into lines later, by `_legend_lines`, because how
        many lines it should occupy depends on how long it is and that is a
        question about the whole label rather than about any one field.
        """
        if self.kind == Cell.KIND_MODEL:
            return f"U{self.unit_index} {_short_model(self.model)}"
        if self.kind == Cell.KIND_PROFILE:
            return f"U{self.unit_index} {self.line} x{self.factor:g}"
        if self.kind == Cell.KIND_POINT:
            # `+0` never occurs - a zero offset is a legitimate cell (the
            # baseline) and its label must say so rather than print "+0".
            offset = (f"{self.offset:+g}" if self.offset else "as drawn")
            return (f"U{self.unit_index} {self.line}"
                    f"[{self.point_index}] {offset}")
        if self.kind == Cell.KIND_PROMPT:
            return _elide(self.text, PROMPT_LEGEND_CHARS) or "(empty prompt)"
        if self.kind == Cell.KIND_LORA:
            return " ".join(str(self.text or "").split()) or "(no LoRA)"
        return ""

    @property
    def legend(self):
        """The cell's legend as the host wants it: one annotation per line."""
        return [images.GridAnnotation(line) for line in _legend_lines(self.label)]

    def apply_unit(self, unit):
        if self.kind == Cell.KIND_MODEL:
            unit.model = self.model
        elif self.kind == Cell.KIND_PROFILE:
            unit.weight_profile = _scale_profile(
                unit, self.unit_index, self.line, self.factor)
        elif self.kind == Cell.KIND_POINT:
            unit.weight_profile = _offset_profile_point(
                unit, self.unit_index, self.line, self.point_index,
                self.offset)

    def apply_processing(self, pc):
        if self.kind == Cell.KIND_PROMPT:
            if self.prompt_mode == PROMPT_PREPEND:
                pc.prompt = _join_prompt(self.text, pc.prompt)
            elif self.prompt_mode == PROMPT_APPEND:
                pc.prompt = _join_prompt(pc.prompt, self.text)
            else:
                pc.prompt = self.text
        elif self.kind == Cell.KIND_LORA and self.text:
            # APPENDED, never assigned: a LoRA axis adds to whatever prompt the
            # cell already has, so it composes with a Prompt axis on the other
            # side (which is why prompts are applied first) and with LoRAs the
            # user already typed into the prompt box.
            pc.prompt = f"{pc.prompt.rstrip()} {self.text}".strip()


def _elide(text, width=44):
    """One line, short enough for a grid annotation."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[:width - 1] + "…"


def _legend_lines(text, per_line=LEGEND_LINE_CHARS, max_lines=LEGEND_MAX_LINES):
    """`text` broken into legend lines of roughly equal length.

    The line COUNT is chosen first, from the length, and the text is then
    divided into that many lines - which is what keeps the block balanced. The
    obvious greedy fill (pack each line to the limit, start a new one when it
    overflows) leaves a full line above a two-word stub, and in a legend that
    reads as a mistake rather than as a wrap.

    Breaks are taken at the separators machine-generated names actually use,
    and a run with no separator at all is cut mid-word rather than allowed to
    set the width of the whole block: an unbreakable 40-character token would
    otherwise shrink the font for every line beside it. Nothing is dropped -
    the lines concatenate back to `text`.
    """
    text = " ".join(str(text or "").split())
    if not text:
        return [""]
    count = max(1, min(max_lines, (len(text) + per_line - 1) // per_line))
    budget = max(1, (len(text) + count - 1) // count)

    lines, current = [], ""
    for chunk in RE_LEGEND_CHUNK.findall(text):
        while len(chunk) > budget:
            room = budget - len(current)
            if room <= 0:
                lines.append(current)
                current, room = "", budget
            current, chunk = current + chunk[:room], chunk[room:]
            lines.append(current)
            current = ""
        if current and len(current) + len(chunk) > budget:
            lines.append(current)
            current = ""
        current += chunk
    if current:
        lines.append(current)

    lines = [line.strip() or line for line in lines if line.strip()]
    if len(lines) > max_lines:
        # The division can still overshoot when the separators fall badly.
        # Everything past the cap is folded into the last line, which the
        # host then shrinks - one small line beats a caption taller than the
        # picture it labels.
        lines = lines[:max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return lines or [text]


def _join_prompt(head, tail):
    """Two prompt fragments, joined the way a prompt is actually written.

    With ', ', not with a space: a prompt is a comma-separated list of terms,
    and gluing 'golden hour' onto 'a house on a hill' with a space makes ONE
    five-word term rather than two. The seam is normalized rather than trusted -
    both boxes are typed by hand and 'trailing comma' and 'no trailing comma'
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


def _check_axes_do_not_collide(x, y):
    """Reject the two axis configurations that would write over each other.

    Both axes are applied to the SAME unit copy, X first (that is what lets
    "models across, factors down" work on one unit). Two axes writing the same
    thing is the one case where that is wrong, and it is wrong SILENTLY - the
    grid comes out looking like a proper sweep:

    * same unit, both on Model - Y's model wins in every cell, so the columns
      are identical and the X labels name models that never ran;
    * same unit, both on the same profile line - the two factors COMPOSE, so
      the cell labelled 'x0.5 / x2' actually ran at x1.

    Two different lines on one unit (depth across, main down) compose on
    purpose and are left alone - that is the interesting sweep. So do two LoRA
    axes (each APPENDS, so the grid is every combination of the two sets) and
    LoRA against Prompt (prompts are applied first, LoRAs append to the result).

    Two PROMPT axes are judged on the Y axis alone, because Y is applied second:
    prompts x prompts is a real grid as long as the second one COMBINES
    (subject across, lighting down). Only a Y that replaces is rejected - that
    one throws X's value away and prints an X legend naming text that never
    reached the model. X replacing is fine: Y then modifies the result.
    """
    if x.kind == Cell.KIND_PROMPT and y.kind == Cell.KIND_PROMPT \
            and y.prompt_mode == PROMPT_REPLACE:
        raise RuntimeError(
            f"CNPro X/Y: both axes sweep the PROMPT and the Y axis is set to "
            f"{PROMPT_REPLACE}, so the Y prompt would win in every cell and the "
            f"X legend would name text that never ran. Set the Y axis to "
            f"{PROMPT_PREPEND} or {PROMPT_APPEND}, or put one axis on LoRA or "
            f"on a unit.")
    if not (x.on_unit and y.on_unit) or x.unit_index != y.unit_index:
        return
    if x.kind == Cell.KIND_MODEL and y.kind == Cell.KIND_MODEL:
        raise RuntimeError(
            f"CNPro X/Y: both axes sweep the MODEL of unit {x.unit_index}, so "
            f"the Y model would win in every cell and the X labels would name "
            f"models that never ran. Put one axis on a different unit, or on "
            f"Profile.")
    if x.kind == Cell.KIND_PROFILE and y.kind == Cell.KIND_PROFILE \
            and x.line == y.line:
        raise RuntimeError(
            f"CNPro X/Y: both axes scale unit {x.unit_index}'s {x.line} "
            f"profile, so the two factors would MULTIPLY - a cell labelled "
            f"'xa / xb' would really run at x(a*b). Pick a different profile "
            f"line for one axis, or a different unit.")
    # A point offset against a whole-line scale of the same line composes on
    # purpose (the scale rewrites the range, the offset moves a drawn point)
    # - that is a legitimate 2D sweep. The SAME point twice is not: the
    # second offset reads the first one's moved value.
    if x.kind == Cell.KIND_POINT and y.kind == Cell.KIND_POINT \
            and x.line == y.line and x.point_index == y.point_index:
        raise RuntimeError(
            f"CNPro X/Y: both axes offset point {x.point_index} of unit "
            f"{x.unit_index}'s {x.line} profile, so the two offsets would "
            f"ADD - a cell labelled '+a / +b' would really run at +(a+b). "
            f"Pick a different point for one axis, or a different line.")


def _lines(text):
    """One axis value per non-blank line, in the order they were typed."""
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _build_axis(name, target, mode, models, line, point, factors, prompts,
                prompt_mode, loras):
    """The cells of one axis. An axis with nothing selected is one no-op cell."""
    if target == TARGET_PROMPT:
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"{name} axis: unknown prompt mode "
                             f"{prompt_mode!r}.")
        return [Cell(Cell.KIND_PROMPT, text=t, prompt_mode=prompt_mode)
                for t in _lines(prompts)] or [Cell()]

    if target == TARGET_LORA:
        return [Cell(Cell.KIND_LORA, text=t) for t in _lines(loras)] or [Cell()]

    index = _unit_index(target)
    if index is None:
        return [Cell()]

    if mode == MODE_MODEL:
        chosen = [m for m in (models or []) if m]
        return [Cell(Cell.KIND_MODEL, unit_index=index, model=m)
                for m in chosen] or [Cell()]

    if line not in _profile_lines():
        raise ValueError(f"{name} axis: unknown profile line {line!r}.")

    if mode == MODE_POINT:
        values = _parse_offsets(factors)
        return [Cell(Cell.KIND_POINT, unit_index=index, line=line,
                     point_index=int(point or 0), offset=v)
                for v in values] or [Cell()]

    values = _parse_factors(factors)
    return [Cell(Cell.KIND_PROFILE, unit_index=index, line=line, factor=v)
            for v in values] or [Cell()]


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

class Script(scripts.Script):

    def title(self):
        return "CNPro X/Y"

    def ui(self, is_img2img):
        groups = _ui_groups(is_img2img)
        if not groups:
            gr.Markdown("**ControlNet Pro is not loaded**, so there are no "
                        "units to sweep. Enable the extension and restart.")
            return []

        type_filters = [g.type_filter for g in groups]
        enables = [g.enabled for g in groups]
        count = len(groups)
        # Both tabs build this panel, so the ids have to carry the tab or the
        # img2img copy silently duplicates every txt2img id in one document.
        tab = "img2img" if is_img2img else "txt2img"

        def axis_row(name):
            eid = f"cnpro_xy_{tab}_{name.lower()}"
            with gr.Row(variant="compact"):
                # Seeded with the global targets, not left empty: the unit half
                # of this list only arrives when an enable checkbox fires, so a
                # page opened with no unit enabled would otherwise show an empty
                # dropdown - and Prompt and LoRA need no unit at all.
                target = gr.Dropdown(
                    label=f"{name} target", choices=list(GLOBAL_TARGETS),
                    value=None, scale=2, elem_id=f"{eid}_target")
                # EVERY mode-dependent control starts hidden, this one included.
                # With a blank target the panel used to open showing the mode
                # radio and the model list - unit-only controls - which reads as
                # "this script needs a ControlNet unit" when it does not: Prompt
                # and LoRA need none. Blank target now shows only the target
                # dropdown, so the first question the panel asks is the first
                # question it actually has.
                #
                # min_width, not scale alone: too narrow and gradio wraps the
                # radio's chips onto a second line and the panel grows a row
                # taller for nothing - 310 is what fits all three options.
                mode = gr.Radio(
                    label=f"{name} mode", choices=MODES, value=MODE_MODEL,
                    visible=False, scale=2, min_width=310,
                    elem_id=f"{eid}_mode")
                models = gr.Dropdown(
                    label=f"{name} models", choices=[], value=[],
                    multiselect=True, visible=False, scale=5,
                    elem_id=f"{eid}_models")
                line = gr.Dropdown(
                    label=f"{name} profile", choices=list(_profile_lines()),
                    value="Main", visible=False, scale=2,
                    elem_id=f"{eid}_line")
                # Which drawn point a "Profile point" axis moves: 0-based,
                # left to right, negative from the end (-1 = rightmost).
                point = gr.Number(
                    label=f"{name} point #", value=0, precision=0,
                    visible=False, scale=1, min_width=70,
                    elem_id=f"{eid}_point")
                # ONE textbox serves both profile modes - a factor list and
                # an offset list are the same kind of input asked about
                # different things, and the label/placeholder swap in
                # `visibility` below is what says which question is being
                # answered.
                factors = gr.Textbox(
                    label=f"{name} factors", value="", visible=False, scale=3,
                    lines=1, max_lines=1,
                    placeholder="0.5, 0.75, 1, 1.5   or   0.5:1.5:5",
                    elem_id=f"{eid}_factors")
                # Left of the prompts box, because it says how that box is
                # READ: Replace is the box being the prompt, Prepend/Append make
                # it a modifier of the prompt the user already typed upstairs.
                prompt_mode = gr.Dropdown(
                    label=f"{name} prompt mode", choices=list(PROMPT_MODES),
                    value=PROMPT_REPLACE, visible=False, scale=2,
                    elem_id=f"{eid}_prompt_mode")
                prompts = gr.Textbox(
                    label=f"{name} prompts (one per line)", value="",
                    visible=False, scale=7, lines=3, max_lines=12,
                    placeholder="a house on a hill\na house in a storm\n…",
                    elem_id=f"{eid}_prompts")
                # Seeded here AND refreshed when the target becomes LoRA. The
                # seed can be empty (whether the host has scanned its LoRAs by
                # the time this panel is built is not ours to rely on), so the
                # list is re-read at the moment it is first needed - see
                # on_axis_target.
                lora_pick = gr.Dropdown(
                    label=f"{name} add LoRA", choices=_lora_names(), value=None,
                    visible=False, scale=3, elem_id=f"{eid}_lora_pick")
                # DECLARED HERE, after both list-bearing controls. A hidden
                # component takes no space, so the row's declaration order is
                # what decides where this lands: in Model mode the only visible
                # thing before it is the model dropdown, and in LoRA mode it is
                # the LoRA picker. Declared earlier, it floated between the
                # target and the picker whenever the models list was hidden.
                #
                # elem_classes, not elem_id, for the styling: the two buttons
                # share one rule and the ids already carry the tab and the axis.
                # style.css top-anchors them - the host's own `.tool` rule ends
                # `align-self: end`, which in LoRA mode is the bottom of a
                # three-line textbox rather than the picker they belong to.
                refresh = ToolButton(
                    value=REFRESH_SYMBOL, visible=False,
                    elem_id=f"{eid}_refresh", elem_classes=["cnpro-xy-tool"],
                    tooltip="Rescan the ControlNet and LoRA directories")
                dice = ToolButton(
                    value=DICE_SYMBOL, visible=False,
                    elem_id=f"{eid}_dice", elem_classes=["cnpro-xy-tool"],
                    tooltip=f"Add a random LoRA at a random weight "
                            f"({LORA_RANDOM_WEIGHTS[0]:g}-"
                            f"{LORA_RANDOM_WEIGHTS[1]:g})")
                loras = gr.Textbox(
                    label=f"{name} LoRAs (one cell per line)", value="",
                    visible=False, scale=6, lines=3, max_lines=12,
                    placeholder="<lora:detail:0.4>\n"
                                "<lora:detail:0.8>\n"
                                "<lora:detail:0.8> <lora:film:0.3>",
                    elem_id=f"{eid}_loras")

            # WHAT the axis varies is the first question, so the mode radio and
            # everything under it follows the target rather than sitting beside
            # it: Prompt and LoRA have no Model/Profile distinction to make.
            def visibility(target_value, mode_value):
                # Tested POSITIVELY, against the "Unit N" form. Testing "not one
                # of the global targets" made a blank target - the state every
                # page opens in - look like a unit and show the unit controls.
                is_unit = _unit_index(target_value) is not None
                model_mode = is_unit and mode_value == MODE_MODEL
                profile_mode = is_unit and mode_value == MODE_PROFILE
                point_mode = is_unit and mode_value == MODE_POINT
                is_lora = target_value == TARGET_LORA
                is_prompt = target_value == TARGET_PROMPT
                return [gr.update(visible=is_unit),          # mode
                        gr.update(visible=model_mode),       # models
                        gr.update(visible=model_mode or is_lora),   # refresh
                        gr.update(visible=is_lora),          # dice
                        gr.update(visible=profile_mode or point_mode),  # line
                        gr.update(visible=point_mode),       # point
                        # the shared values box, relabelled per question
                        gr.update(visible=profile_mode or point_mode,
                                  label=(f"{name} offsets (0 = as drawn)"
                                         if point_mode else f"{name} factors"),
                                  placeholder=("-0.2, 0, 0.2   or   -0.2:0.2:5"
                                               if point_mode else
                                               "0.5, 0.75, 1, 1.5   or   "
                                               "0.5:1.5:5")),      # factors
                        gr.update(visible=is_prompt),        # prompt_mode
                        gr.update(visible=is_prompt),        # prompts
                        gr.update(visible=is_lora),          # lora_pick
                        gr.update(visible=is_lora)]          # loras

            switches = [mode, models, refresh, dice, line, point, factors,
                        prompt_mode, prompts, lora_pick, loras]
            for component in (target, mode):
                component.change(fn=visibility, inputs=[target, mode],
                                 outputs=switches, show_progress=False)

            # THE FALLBACK PATH, not the normal one.
            #
            # javascript/cnpro_xy.js intercepts the click on an option and adds
            # the line client-side, precisely so this round trip does NOT
            # happen: the round trip is what closes the option list, drops the
            # filter the user typed and scrolls the list back to the selection -
            # i.e. costs three interactions per LoRA on an axis whose whole
            # point is picking several. With the JS loaded the picker's value
            # never changes and nothing below ever fires.
            #
            # It is kept because the browser caches JS and an extension can be
            # installed without a hard refresh: without it the picker would be a
            # dropdown that does nothing at all. The two cannot both run - one
            # inserts only when the other has suppressed the event that would
            # have inserted.
            #
            # Clearing the picker is what lets the same LoRA be picked twice
            # here: gradio fires no change event when a dropdown is set to the
            # value it already holds, and "the same LoRA at several weights" is
            # the main thing this axis is for.
            def add_lora(name_picked, current):
                if not name_picked:
                    return gr.update(), gr.update()
                entry = f"<lora:{name_picked}:{LORA_DEFAULT_WEIGHT}>"
                body = (current or "").rstrip()
                return (f"{body}\n{entry}" if body else entry), gr.update(value=None)

            lora_pick.change(fn=add_lora, inputs=[lora_pick, loras],
                             outputs=[loras, lora_pick], show_progress=False)

            # The dice is a whole cell in one click: a LoRA nobody chose, at a
            # weight nobody chose. Server-side rather than in the JS because the
            # LoRA NAMES live here - the browser only ever sees the list while
            # the dropdown is open, and the point of the button is not having to
            # open it.
            def add_random_lora(current):
                names = _lora_names()
                if not names:
                    # Distinct from "the button did nothing": the host has no
                    # LoRAs scanned, which the refresh button next door fixes.
                    logger.warning(
                        "CNPro X/Y: the dice has no LoRAs to pick from - the "
                        "host has none scanned. Press the refresh button beside "
                        "it, or check the LoRA directory.")
                    return gr.update()
                weight = round(random.uniform(*LORA_RANDOM_WEIGHTS), 2)
                entry = f"<lora:{random.choice(names)}:{weight:g}>"
                body = (current or "").rstrip()
                return f"{body}\n{entry}" if body else entry

            dice.click(fn=add_random_lora, inputs=[loras], outputs=[loras],
                       show_progress=False)

            return (target, mode, models, line, point, factors, prompts,
                    prompt_mode, lora_pick, loras, refresh)

        (x_target, x_mode, x_models, x_line, x_point, x_factors, x_prompts,
         x_prompt_mode, x_lora_pick, x_loras, x_refresh) = axis_row("X")
        (y_target, y_mode, y_models, y_line, y_point, y_factors, y_prompts,
         y_prompt_mode, y_lora_pick, y_loras, y_refresh) = axis_row("Y")

        with gr.Row(variant="compact"):
            draw_legend = gr.Checkbox(label="Draw legend", value=True)
            include_lone_images = gr.Checkbox(label="Keep individual images",
                                              value=False)
            no_fixed_seeds = gr.Checkbox(label="Keep -1 for seeds", value=False)
            margin_size = gr.Slider(label="Grid margin (px)", minimum=0,
                                    maximum=64, value=0, step=1)

        # --- live choices ------------------------------------------------
        #
        # Which units are activated, and which models each one offers, are both
        # live UI state, so they are tracked rather than snapshotted: the unit
        # list follows the enable checkboxes and the model lists follow the
        # Control Type dropdowns. Both are read as gradio INPUTS, never off the
        # component objects - the model dropdown's visible choices were pushed
        # by a gr.update and never landed back on the Python object, and the
        # enable flag on the unit State is written by a sibling handler of the
        # same event (see _enabled_unit_labels).

        def _models_update(unit_label, selected, filters):
            index = _unit_index(unit_label)
            if index is None or index >= len(filters):
                return gr.update(choices=[], value=[])
            choices = _models_for_type(filters[index])
            return gr.update(choices=choices,
                             value=[m for m in (selected or []) if m in choices])

        def rebuild(x_label, y_label, x_sel, y_sel, *values):
            flags, filters = list(values[:count]), list(values[count:])
            choices = _target_choices(flags)
            # A unit that has just been switched off cannot stay selected, and
            # falling back to "nothing" would make the axis silently inert - so
            # it falls back to the first target that IS available. Prompt and
            # LoRA are always in the list, so there is always one.
            x_label = x_label if x_label in choices else choices[0]
            y_label = y_label if y_label in choices else choices[0]
            loras = _lora_names()
            return (gr.update(choices=choices, value=x_label),
                    gr.update(choices=choices, value=y_label),
                    _models_update(x_label, x_sel, filters),
                    _models_update(y_label, y_sel, filters),
                    gr.update(choices=loras),
                    gr.update(choices=loras))

        def rescan(*args):
            # Only from the buttons: get_all_models hashes every control file
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
                    logger.warning(f"CNPro X/Y: could not rescan LoRAs ({exc}).")
            return rebuild(*args)

        def on_axis_target(target_value, selected, *values):
            # The LoRA list is re-read HERE rather than only at build time: the
            # panel is built once at startup and the host may not have scanned
            # its LoRAs yet, which left the picker permanently empty on any page
            # where no other event happened to rebuild it.
            return (_models_update(target_value, selected, list(values[count:])),
                    gr.update(choices=_lora_names())
                    if target_value == TARGET_LORA else gr.update())

        live_inputs = ([x_target, y_target, x_models, y_models]
                       + enables + type_filters)
        live_outputs = [x_target, y_target, x_models, y_models,
                        x_lora_pick, y_lora_pick]

        for subscribe, handler in ([(x_refresh.click, rescan),
                                    (y_refresh.click, rescan)]
                                   + [(c.change, rebuild)
                                      for c in enables + type_filters]):
            subscribe(fn=handler, inputs=live_inputs, outputs=live_outputs,
                      show_progress=False)

        for axis_target, axis_models, axis_pick in (
                (x_target, x_models, x_lora_pick),
                (y_target, y_models, y_lora_pick)):
            axis_target.change(fn=on_axis_target,
                               inputs=[axis_target, axis_models] + enables + type_filters,
                               outputs=[axis_models, axis_pick], show_progress=False)

        return [x_target, x_mode, x_models, x_line, x_point, x_factors,
                x_prompts, x_prompt_mode, x_loras,
                y_target, y_mode, y_models, y_line, y_point, y_factors,
                y_prompts, y_prompt_mode, y_loras,
                draw_legend, include_lone_images, no_fixed_seeds, margin_size]

    def run(self, p, *args):
        if len(args) < ARG_COUNT:
            # ui() returned nothing, which it only does when CNPro is absent.
            raise RuntimeError(
                "CNPro X/Y: ControlNet Pro is not loaded, so this script has "
                "no units to sweep. Enable the extension and restart.")
        (x_target, x_mode, x_models, x_line, x_point, x_factors, x_prompts,
         x_prompt_mode, x_loras,
         y_target, y_mode, y_models, y_line, y_point, y_factors, y_prompts,
         y_prompt_mode, y_loras,
         draw_legend, include_lone_images, no_fixed_seeds,
         margin_size) = args[:ARG_COUNT]

        base_args = list(p.script_args)
        xs = _build_axis("X", x_target, x_mode, x_models, x_line, x_point,
                         x_factors, x_prompts, x_prompt_mode, x_loras)
        ys = _build_axis("Y", y_target, y_mode, y_models, y_line, y_point,
                         y_factors, y_prompts, y_prompt_mode, y_loras)
        _check_axes_do_not_collide(xs[0], ys[0])

        # CNPro is only REQUIRED when an axis actually varies a unit. A Prompt
        # or LoRA grid touches no control at all, and refusing to run one
        # because no unit is configured would be a demand the work does not
        # make. Resolved unconditionally so `_cell_args` has it when it is
        # needed, but only insisted on below.
        cnpro = _cnpro_script(p)
        needs_units = xs[0].on_unit or ys[0].on_unit
        if needs_units and (cnpro is None or cnpro.args_from is None):
            raise RuntimeError(
                "CNPro X/Y: an axis varies a ControlNet unit, but the "
                "ControlNet Pro script is not running. Enable the extension, "
                "or put both axes on Prompt or LoRA.")

        # Both axes are checked before the first image, not on the cell that
        # trips over it: a sweep that dies on cell 7 of 12 has already spent
        # six generations. Every cell on an axis shares one unit, so one check
        # per axis is the whole check.
        for name, cells in (("X", xs), ("Y", ys)):
            cell = cells[0]
            if not cell.on_unit:
                continue
            slot = cnpro.args_from + cell.unit_index
            if slot >= cnpro.args_to:
                raise RuntimeError(
                    f"CNPro X/Y: the {name} axis targets unit "
                    f"{cell.unit_index}, which does not exist.")
            unit = base_args[slot]
            if not getattr(unit, "enabled", False):
                raise RuntimeError(
                    f"CNPro X/Y: the {name} axis targets unit "
                    f"{cell.unit_index}, which is not enabled. A disabled unit "
                    f"is dropped before the sweep is applied, so every cell "
                    f"would be identical.")
            if cell.kind in (Cell.KIND_PROFILE, Cell.KIND_POINT):
                _warn_if_inert(unit, cell.unit_index, cell.line)
            if cell.kind == Cell.KIND_POINT:
                # Before the first image, like everything else here: a bad
                # index would leave every offset falling on no point, i.e. a
                # whole axis of identical cells - and the offset arithmetic
                # only warns per cell, which nobody reads twelve times.
                count = _profile_scale_or_die().profile_point_count(
                    unit, cell.line)
                if not (-count <= cell.point_index < count):
                    raise RuntimeError(
                        f"CNPro X/Y: the {name} axis offsets point "
                        f"{cell.point_index} of unit {cell.unit_index}'s "
                        f"{cell.line} profile, which has {count} point(s) - "
                        f"indices 0..{count - 1}, or negative from the end. "
                        f"Pick an existing point.")

        if not no_fixed_seeds:
            # A comparison grid on a random seed compares nothing.
            processing.fix_seed(p)

        p.extra_generation_params["Script"] = self.title()
        _record(p, "X", x_target, x_models, x_line, xs)
        _record(p, "Y", y_target, y_models, y_line, ys)

        total = len(xs) * len(ys)
        state.job_count = total * p.n_iter
        logger.info(f"CNPro X/Y: {len(xs)} x {len(ys)} = {total} cells")

        result = None
        for iy, y_cell in enumerate(ys):
            for ix, x_cell in enumerate(xs):
                if state.interrupted or getattr(state, "stopping_generation", False):
                    break
                index = iy * len(xs) + ix
                state.job = f"{index + 1} out of {total}"

                pc = copy.copy(p)
                pc.styles = pc.styles[:]
                pc.override_settings = copy.copy(p.override_settings)
                # Per cell, not shared. CNPro writes this cell's unit
                # description into extra_generation_params - and appends
                # "CNPro skipped units" when a cell's model is None - so a
                # shared dict carries one cell's note into the infotext of
                # every cell after it. Each image has to describe the unit IT
                # was generated with, which is the whole point of sweeping.
                pc.extra_generation_params = copy.copy(p.extra_generation_params)
                # `p` was set up when the host assigned its script_args; saying
                # so keeps the property's setter from running setup_scripts
                # again for every cell.
                pc.scripts_setup_complete = True
                pc.script_args = self._cell_args(base_args, cnpro, x_cell, y_cell)
                # Prompts first, LoRAs second: a LoRA cell appends to whatever
                # prompt this cell ended up with, so Prompt-across / LoRA-down
                # gives every combination rather than one axis erasing the other.
                for cell in (x_cell, y_cell):
                    if cell.kind == Cell.KIND_PROMPT:
                        cell.apply_processing(pc)
                for cell in (x_cell, y_cell):
                    if cell.kind == Cell.KIND_LORA:
                        cell.apply_processing(pc)
                try:
                    processed = process_images(pc)
                except Exception as exc:
                    errors.display(exc, "generating image for CNPro X/Y")
                    processed = Processed(p, [], p.seed, "")

                if result is None:
                    result = copy.copy(processed)
                    result.images = [None] * total
                    result.all_prompts = [None] * total
                    result.all_seeds = [None] * total
                    result.infotexts = [None] * total
                    result.index_of_first_image = 1

                if processed.images:
                    result.images[index] = processed.images[0]
                    result.all_prompts[index] = processed.prompt
                    result.all_seeds[index] = processed.seed
                    result.infotexts[index] = processed.infotexts[0]
            else:
                continue
            break

        if result is None or not any(result.images):
            logger.error("CNPro X/Y: no cell produced an image.")
            return Processed(p, [])

        # A cell that failed, or that an interrupt never reached, still has to
        # occupy its place or every image after it shifts into the wrong row.
        template = next(img for img in result.images if img is not None)
        for i, img in enumerate(result.images):
            if img is None:
                result.images[i] = Image.new(template.mode, template.size)
                result.all_prompts[i] = result.all_prompts[0]
                result.all_seeds[i] = result.all_seeds[0]
                result.infotexts[i] = result.infotexts[0]

        grid = images.image_grid(result.images, rows=len(ys))
        if draw_legend:
            cell_w, cell_h = map(max, zip(*(img.size for img in result.images)))
            grid = images.draw_grid_annotations(
                grid, cell_w, cell_h,
                [c.legend for c in xs],
                [c.legend for c in ys],
                margin_size)

        result.images.insert(0, grid)
        result.all_prompts.insert(0, result.all_prompts[0])
        result.all_seeds.insert(0, result.all_seeds[0])
        result.infotexts.insert(0, result.infotexts[0])

        if opts.grid_save:
            _save_grid(result, p)

        if not include_lone_images:
            for attr in ("images", "infotexts", "all_prompts", "all_seeds"):
                setattr(result, attr, getattr(result, attr)[:1])

        return result

    @staticmethod
    def _cell_args(base_args, cnpro, *cells):
        """`p.script_args` for one cell, with both axes applied.

        The gr.State unit objects are the LIVE ones the panel writes to, so they
        are copied before anything is set on them. One copy per unit per cell,
        not per axis: X and Y are allowed to target the SAME unit (models across,
        profile factors down), and two copies would mean the second overwrote
        the first.
        """
        args = list(base_args)
        for cell in cells:
            if not cell.on_unit:
                continue
            slot = cnpro.args_from + cell.unit_index
            if args[slot] is base_args[slot]:
                args[slot] = copy.copy(args[slot])
            cell.apply_unit(args[slot])
        return args


# ---------------------------------------------------------------------------
# Saving the grid
# ---------------------------------------------------------------------------

#: Where cjxl lives when nothing says otherwise. Resolved through LOCALAPPDATA
#: rather than a literal user directory so the same file works on any machine
#: with the standard install.
CJXL_DEFAULT_DIR = ("Programs", "jxl", "bin")
CJXL_EXE = "cjxl.exe" if os.name == "nt" else "cjxl"


def _cjxl_executable():
    """Path to cjxl: the configured one, then PATH, then the standard install."""
    candidates = []
    configured = str(shared.opts.data.get("cnpro_xy_cjxl_path", "") or "").strip()
    if configured:
        # a directory is accepted too - that is how the tool is usually spoken
        # about ("it's in .../jxl/bin"), and rejecting it would be pedantry
        candidates.append(os.path.join(configured, CJXL_EXE)
                          if os.path.isdir(configured) else configured)
    found = shutil.which("cjxl")
    if found:
        candidates.append(found)
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    candidates.append(os.path.join(local, *CJXL_DEFAULT_DIR, CJXL_EXE))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _to_jxl(png_path):
    """Re-encode a saved grid PNG as JPEG XL and drop the PNG.

    Returns the .jxl path, or None when the PNG was kept - and the PNG is kept
    on EVERY failure. A grid that cost twelve generations is not worth losing to
    a missing encoder, and "the file is still there, in the format you already
    had" is the only safe way to fail here.

    NOTE, since it is a real trade and not an oversight: the grid's parameters
    do not survive this. cjxl carries no PNG tEXt chunk across (JXL has no
    equivalent), and the host's PIL cannot open a .jxl anyway, so PNG Info could
    not read them back even if they were embedded. Per-cell infotext is
    unaffected - every individual image still carries its own, this reaches only
    the contact sheet.
    """
    exe = _cjxl_executable()
    if exe is None:
        logger.warning(
            "CNPro X/Y: JPEG XL conversion is on but cjxl was not found - "
            "looked on PATH and in %LOCALAPPDATA%\\Programs\\jxl\\bin. The grid "
            "was kept as PNG; set the path in Settings > CNPro, or turn "
            "the conversion off there.")
        return None

    quality = int(shared.opts.data.get("cnpro_xy_grid_jxl_quality", 50))
    jxl_path = os.path.splitext(png_path)[0] + ".jxl"
    base = [exe, png_path, jxl_path, "-q", str(quality), "--quiet"]
    # Level 5 caps a codestream at 2^28 pixels, which a large enough sweep grid
    # genuinely exceeds - the retry is the whole reason the grid is a PNG in the
    # first place, so it is not left to chance.
    for args in (base, base + ["--codestream_level=10"]):
        try:
            done = subprocess.run(args, capture_output=True, timeout=1800)
        except Exception as exc:
            logger.warning(f"CNPro X/Y: could not run cjxl ({exc}). The grid "
                           f"was kept as PNG.")
            return None
        if done.returncode == 0 and os.path.isfile(jxl_path) \
                and os.path.getsize(jxl_path) > 0:
            break
    else:
        message = (done.stderr or done.stdout or b"").decode("utf-8", "replace").strip()
        logger.warning(f"CNPro X/Y: cjxl failed ({message or 'no output'}). "
                       f"The grid was kept as PNG.")
        return None

    try:
        os.remove(png_path)
    except OSError as exc:
        logger.warning(f"CNPro X/Y: wrote {jxl_path} but could not remove the "
                       f"PNG ({exc}); both files are on disk.")
    before = os.path.getsize(jxl_path)
    logger.info(f"CNPro X/Y: grid saved as {os.path.basename(jxl_path)} "
                f"(JPEG XL q{quality}, {before / 1024:.0f} KB)")
    return jxl_path


def _save_grid(result, p):
    """Write the grid to disk - PNG, then JPEG XL if that is configured.

    ALWAYS PNG, never `opts.grid_format`. A sweep grid is the one image in the
    host that has no size bound: it is the cell size times the axis lengths, and
    the default WebP cannot hold it - the format caps either side at 16383 px,
    which a 16-wide grid of 1024 px cells passes exactly, and it is saved LOSSY
    by default, which is the wrong trade for an image whose whole content is
    small differences between neighbouring cells plus thin legend text. PNG has
    neither limit. Shrinking it afterwards is what the JPEG XL pass is for, and
    that runs on a file that already exists.

    Two of the host's side files are suppressed for the duration, because both
    are noise for a grid: `export_for_4chan` writes a downscaled .jpg beside any
    image over the size threshold - which a grid always is, and a lossy
    thumbnail of a contact sheet is unreadable - and `save_txt` writes a .txt of
    the parameters. Suppressed by overriding the two options rather than by
    deleting the files afterwards, so they are never created in the first place.
    """
    guarded = ("export_for_4chan", "save_txt")
    saved = {key: shared.opts.data.get(key, None) for key in guarded}
    for key in guarded:
        shared.opts.data[key] = False
    try:
        fullfn, _ = images.save_image(
            result.images[0], p.outpath_grids, "cnpro_xy",
            info=result.infotexts[0], extension="png",
            prompt=result.all_prompts[0], seed=result.all_seeds[0],
            grid=True, p=result)
    finally:
        for key, value in saved.items():
            # `None` means the option was never in `data` and was reading its
            # declared default; putting False back would silently change the
            # user's setting from "unset" to "off".
            if value is None:
                shared.opts.data.pop(key, None)
            else:
                shared.opts.data[key] = value
    if not fullfn or not shared.opts.data.get("cnpro_xy_grid_jxl", True):
        return fullfn
    return _to_jxl(fullfn) or fullfn


def _record(p, name, target, models, line, cells):
    """What this axis swept, into the grid's infotext.

    Values are joined with ' | ' and never ',': the host separates infotext
    fields with ', ', so a comma inside one would split it into two nonsense
    fields - and prompts are full of commas.
    """
    kind = cells[0].kind
    if kind == Cell.KIND_NONE:
        return
    params = p.extra_generation_params
    params[f"CNPro {name} target"] = target
    if kind == Cell.KIND_MODEL:
        params[f"CNPro {name} models"] = " | ".join(_short_model(m) for m in models)
    elif kind == Cell.KIND_PROFILE:
        params[f"CNPro {name} profile"] = line
        params[f"CNPro {name} factors"] = " ".join(f"x{c.factor:g}" for c in cells)
    elif kind == Cell.KIND_POINT:
        params[f"CNPro {name} profile"] = line
        params[f"CNPro {name} point"] = cells[0].point_index
        params[f"CNPro {name} offsets"] = " ".join(
            f"{c.offset:+g}" for c in cells)
    else:
        if kind == Cell.KIND_PROMPT:
            # Recorded even when it is the default: the values below are the
            # same text either way, and whether they REPLACED the prompt or were
            # bolted onto it is the difference between two entirely different
            # grids. Reading that back off the image is guesswork.
            params[f"CNPro {name} prompt mode"] = cells[0].prompt_mode
        params[f"CNPro {name} values"] = " | ".join(
            (c.text or "").replace(",", " ") for c in cells)


def _on_ui_settings():
    # CNPro's own section, not ControlNet's. These three are settings of THIS
    # script, which sweeps CNPro units; putting them under ControlNet buried
    # them among the host-facing model/path options they have nothing to do
    # with. The option KEYS are unchanged, so a config.json written before the
    # move still applies.
    section = ("cnpro", "CNPro")
    shared.opts.add_option("cnpro_xy_grid_jxl", shared.OptionInfo(
        True,
        "X/Y grids: re-encode the saved PNG as JPEG XL and delete the PNG "
        "(the grid loses its parameters - a .jxl cannot carry them; the "
        "individual images keep theirs)",
        gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("cnpro_xy_grid_jxl_quality", shared.OptionInfo(
        50, "X/Y grids: JPEG XL quality (100 = lossless, 90 = visually lossless)",
        gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}, section=section))
    shared.opts.add_option("cnpro_xy_cjxl_path", shared.OptionInfo(
        "", "X/Y grids: path to cjxl, file or its directory "
            "(blank: PATH, then %LOCALAPPDATA%\\Programs\\jxl\\bin)",
        section=section))


script_callbacks.on_ui_settings(_on_ui_settings)
