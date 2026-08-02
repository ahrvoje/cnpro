"""The output weight coverage panel - LAYOUT AND CHROME ONLY.

One collapsed accordion directly under the CNPro title, holding a static canvas
that shows what every enabled unit together will do to the output, aggregated in
OUTPUT geometry: violet 0 ... red 1 (the same ramp the weight masks are painted
in), contours at 0 / 0.25 / 0.5 / 0.75, an orange contour at 1 and red ones
above it, where control starts to oversaturate.

IT IS BUILT LIKE A UNIT, because it is read like one: settings in a column on
the LEFT (the same `controlnet_main_options_column` a unit uses, so the two
line up and share every width rule), the picture on the RIGHT, and the action
buttons in a `controlnet_image_controls` row under the canvas - the same
`cnet-toolbutton` group, the same glyphs (⤵I / ⤵O) as the unit's own
insert-image buttons, with the status line taking the space they leave. The
controls are ordinary gradio components rather than hand-rolled HTML, so they
inherit the widget vocabulary instead of imitating it.

EVERYTHING HERE IS CHROME. The map is computed entirely in the browser by
javascript/coverage_map.js, out of the values already on the page - the unit
enables, the profile strings, the painted mask channels, the resize modes and
the width/height sliders. That is not an implementation detail but the design:
those values reach the server only when Generate is pressed, so a server-side
preview would show the state of the LAST generation, which is precisely the
question this panel is not being asked. Nothing here writes to a unit, and none
of the components below carries a server-side event handler.

The ids/classes are a contract with coverage_map.js:
  .cnet-coverage-panel        the root row; its elem_id is
                              `<tab prefix>_coverage`, which is where the script
                              takes the units (`<tab prefix>_accordions`) and the
                              tab name from - no data-* carrier element, because
                              an element that exists only to hold two strings is
                              exactly the dead chrome the audit hunts for
  .cnet-coverage-channel      cross-attention / residual radio
  .cnet-coverage-metric       mean / peak radio
  .cnet-coverage-alpha        how strongly the map paints over a backdrop
  .cnet-coverage-status       the numbers, under the controls
  .cnet-coverage-canvas       sized to the OUTPUT resolution at paint time
  .cnet-coverage-readout      the weight under the pointer, follows it
  .cnet-coverage-max/-min     maximize / minimize the stage (the host's glyphs)
  .cnet-coverage-insert-*     backdrop from the img2img input / current output
  .cnet-coverage-clear-bg     drop the backdrop
  .cnet-coverage-refresh      recompute now, dropping the decoded-mask cache
"""
import gradio as gr
from modules.ui_components import ToolButton


# The two insert glyphs are the unit's own (ControlNetUiGroup
# tossdown_input_symbol / tossdown_output_symbol). Same action, same picture -
# duplicated as literals only because importing the UI group from here would be
# a circular import for two strings.
TOSSDOWN_INPUT = "⤵I"
TOSSDOWN_OUTPUT = "⤵O"
CLEAR = "✕"
REFRESH = "↻"

# The maximize pair is the HOST's, character for character and title for title
# (`modules_forge/forge_canvas/canvas.html`, `.forge-toolbar-box-a`). This canvas
# is read like the unit canvases and now maximizes like them, so it says so with
# the same two glyphs in the same corner rather than inventing a third
# vocabulary for the same gesture.
MAXIMIZE = "⛶"
MINIMIZE = "➖"


def _hue_ramp_css() -> str:
    """The mask hue ramp as CSS stops: weight 0 (violet, hue 270) to 1 (red).

    Emitted as hsl() at every 30 degrees rather than as a colour list, because
    hsl() IS the formula weight_mask.js paints with (full saturation, half
    lightness, hue = (1 - weight) * 270). A hand-written list of hex stops here
    would be a second definition of the ramp, free to drift from the one that
    decides what the picture looks like.
    """
    stops = []
    for i in range(10):
        weight = i / 9.0
        hue = (1.0 - weight) * 270.0
        stops.append(f"hsl({hue:.0f} 100% 50%) {weight * 100:.0f}%")
    return "linear-gradient(to right, " + ", ".join(stops) + ")"


def _contour_key_html() -> str:
    """What the lines drawn ON the picture mean. Left end of the legend row.

    The CONTOURS come first because they are read first: they are what is drawn
    on top, and the ramp is the background the eye has already learnt from the
    masks it was painted in.
    """
    return (
        '<span class="cnet-coverage-key">'
        '<span class="cnet-coverage-swatch cnet-coverage-swatch-quarter"></span>0.25/0.5/0.75'
        '<span class="cnet-coverage-swatch cnet-coverage-swatch-one"></span>1'
        '<span class="cnet-coverage-swatch cnet-coverage-swatch-over"></span>&gt;1'
        '</span>'
    )


def _ramp_html() -> str:
    """The weight ramp, its own block so the legend row can CENTRE it.

    It belongs under the middle of the picture it explains, and the row it
    lives in also carries the contour key (left) and the action buttons
    (right) - three grid cells, `1fr auto 1fr`, so the middle one lands on the
    row's centre, which is the canvas's centre (the canvas is centred in the
    same column). Two side cells of equal width is the only thing that has to
    hold for that, and `fr` is what guarantees it.
    """
    return (
        '<span class="cnet-coverage-scale">'
        f'<span class="cnet-coverage-ramp" style="background: {_hue_ramp_css()};"></span>'
        '<span class="cnet-coverage-ticks"><span>0</span><span>0.5</span><span>1</span></span>'
        '</span>'
    )


def _stage_html() -> str:
    """The picture, plus the three things that sit ON it.

    ORDER MATTERS, and only between the readout and the hint: the readout hides
    the hint while it is up (`~` in style.css, and CSS sibling combinators only
    look FORWARD), because both are centred pills on the same picture and the
    hint would land under the number exactly when the number is being read.

    The maximize pair is a plain pair of `<button>`s here rather than two
    ToolButtons in the action row below, for a reason that is not style: a
    maximized stage is `position: fixed` over the whole viewport, so it covers
    that row. An exit control the overlay hides is not an exit control.
    """
    return (
        '<div class="cnet-coverage-stage"'
        ' title="Total control weight per output pixel, aggregated over every enabled'
        ' unit and every Input that will run, in output geometry.'
        ' Violet 0, red 1; contours at 0, 0.25, 0.5, 0.75, orange at 1 and red above it.'
        ' Hover for the value under the pointer.'
        ' Not included: the unit Use-Mask, the balance profile and the preprocessor -'
        ' this is about WEIGHT, not content.">'
        '<canvas class="cnet-coverage-canvas" width="1" height="1"></canvas>'
        '<div class="cnet-coverage-readout">'
        '<span class="cnet-coverage-readout-swatch"></span>'
        '<span class="cnet-coverage-readout-value"></span>'
        '</div>'
        '<div class="cnet-coverage-hint">drop an image here for context</div>'
        '<div class="cnet-coverage-tools">'
        f'<button type="button" class="cnet-coverage-max" title="Maximize">{MAXIMIZE}</button>'
        f'<button type="button" class="cnet-coverage-min" title="Minimize"'
        f' style="display: none;">{MINIMIZE}</button>'
        '</div>'
        '</div>'
    )


def render_coverage_panel(elem_id_tabname: str, gen_type: str) -> None:
    """Build one tab's coverage panel, unit-shaped.

    `elem_id_tabname` is the prefix the units row carries
    (`<prefix>_accordions`); `gen_type` is txt2img/img2img, which decides whose
    width/height sliders and whose gallery the script reads - both are recovered
    from the root's elem_id in the browser.

    Labels are deliberately distinctive ("Coverage metric", not "Metric"):
    ui-config.json keys component defaults by LABEL, per tab, so a generic one
    here would collide with any other extension's (AGENTS.md 1b).
    """
    is_img2img = gen_type == "img2img"
    with gr.Row(elem_id=f"{elem_id_tabname}_coverage",
                elem_classes=["controlnet_image_and_options", "cnet-coverage-panel"]):
        with gr.Column(scale=0, min_width=300,
                       elem_classes=["controlnet_main_options_column",
                                     "cnet-coverage-options"]):
            # The two metrics answer different questions and neither derives
            # from the other (max of a sum is not the sum of maxima), which is
            # why this is a choice and not a fixed reading. No `info` paragraph:
            # gradio renders it above the control, where it cost more height
            # than the control itself - the explanation lives in the panel's
            # tooltip and in the status line, which names the metric in force.
            # WHICH INJECTION MECHANISM. Adding a ControlNet's residual weight
            # to an IP-Adapter's attention weight is a category error - they
            # land in different places in the UNet and never sum with each
            # other - so the channels are read one at a time and never
            # combined. Cross-attention first and by default: a session
            # typically holds several IP-Adapters and one canny or depth, and
            # the IP-Adapter side is where the arithmetic (per-input shares,
            # bands, several units on the same attention sites) is impossible
            # to do in your head.
            #
            # WHICH MODELS ARE ON WHICH CHANNEL is stated in the tooltip, and
            # that tooltip is GENERATED in coverage_map.js from the same
            # kind -> mechanism table the routing uses (MECHANISM_BY_KIND /
            # CHANNEL_DOC), then attached here from JS. It is not written out in
            # this file: a second copy of the list is a second thing to keep in
            # step, and gr.Radio has no tooltip argument to hold it anyway - its
            # `info` paragraph renders ABOVE the control, where it costs more
            # height than the control itself (the reason there is no `info` on
            # any control in this column).
            gr.Radio(
                choices=["cross-attention", "residual"],
                value="cross-attention",
                label="Coverage channel",
                elem_id=f"{elem_id_tabname}_coverage_channel",
                elem_classes=["cnet-coverage-channel"],
            )
            gr.Radio(
                choices=["mean", "peak"],
                value="mean",
                label="Coverage metric",
                elem_id=f"{elem_id_tabname}_coverage_metric",
                elem_classes=["cnet-coverage-metric"],
            )
            gr.Slider(
                minimum=10, maximum=100, step=5, value=50,
                label="Map opacity over backdrop",
                elem_id=f"{elem_id_tabname}_coverage_alpha",
                elem_classes=["cnet-coverage-alpha"],
            )
            # The numbers sit directly under the controls, at the bottom of the
            # settings column: they are what the panel SAYS, and the column is
            # where the eye already is. Under the canvas they competed with the
            # buttons for one line and pushed the picture up.
            gr.HTML(
                value='<p class="cnet-coverage-status">not computed yet</p>',
                elem_classes=["cnet-coverage-status-host"],
            )
        with gr.Column(scale=1, elem_classes=["cnet-coverage-canvas-column"]):
            gr.HTML(value=_stage_html())
            # ONE line under the canvas carries the whole legend AND the
            # buttons: contour key left, ramp centred, actions right. They were
            # two rows and the second one bought nothing - the buttons sit
            # against the canvas's bottom border now, and the row is as tall as
            # the tallest of the three. Same row class as a unit's below-canvas
            # actions, so the buttons keep the unit's compact tool-group
            # geometry and its place at the right.
            with gr.Row(elem_classes=["controlnet_image_controls",
                                      "cnet-coverage-actions"]):
                gr.HTML(value=_contour_key_html(),
                        elem_classes=["cnet-coverage-key-host"])
                gr.HTML(value=_ramp_html(),
                        elem_classes=["cnet-coverage-scale-host"])
                # The buttons are wrapped in a row of their own so the legend
                # row has EXACTLY THREE grid cells. Gradio auto-wraps
                # consecutive form components in a `.form` container, which a
                # unit's button row relies on - but only when they are
                # consecutive AND first here they are not: with two HTML blocks
                # ahead of them the buttons arrived as three separate children,
                # the 3-column grid wrapped onto two rows, and the key and the
                # ramp overlapped (measured: key 358..521, ramp 471..661).
                # An explicit group does not depend on that behaviour at all.
                with gr.Row(elem_classes=["cnet-coverage-buttons"]):
                    if is_img2img:
                        ToolButton(
                            value=TOSSDOWN_INPUT,
                            elem_id=f"{elem_id_tabname}_coverage_insert_input",
                            elem_classes=["cnet-toolbutton", "cnet-coverage-insert-input"],
                            tooltip="Use the img2img input image as the backdrop",
                        )
                    ToolButton(
                        value=TOSSDOWN_OUTPUT,
                        elem_id=f"{elem_id_tabname}_coverage_insert_output",
                        elem_classes=["cnet-toolbutton", "cnet-coverage-insert-output"],
                        tooltip="Use the current output image as the backdrop",
                    )
                    ToolButton(
                        value=CLEAR,
                        elem_id=f"{elem_id_tabname}_coverage_clear_bg",
                        elem_classes=["cnet-toolbutton", "cnet-coverage-clear-bg"],
                        tooltip="Remove the backdrop",
                    )
                    ToolButton(
                        value=REFRESH,
                        elem_id=f"{elem_id_tabname}_coverage_refresh",
                        elem_classes=["cnet-toolbutton", "cnet-coverage-refresh"],
                        tooltip="Recompute now (the map follows edits on its own;"
                                " this also drops the decoded-mask cache)",
                    )
