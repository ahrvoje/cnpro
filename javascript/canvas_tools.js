/**
 * CNPro canvas TOOL REGISTRY -- one entry per tool, and nothing else.
 *
 * WHAT THIS FILE IS FOR
 * ---------------------
 * Adding, removing or exchanging a canvas tool used to mean editing a 187-line
 * verbatim HTML blob, a hand-maintained id list, a DEFERRED map and a stylesheet
 * -- four places, three of them easy to forget, and every one of them silent
 * when forgotten (see ARCHITECTURE.md section 8).
 *
 * Now a tool is ONE OBJECT in the TOOLS array below. Everything downstream is
 * derived from it:
 *
 *   canvas_nodes.js  renders the button, the menu and the overlays from it, and
 *                    derives the id contract (which ids exist, which may stay
 *                    hidden) from the SAME object -- so the contract cannot
 *                    drift from the markup, because there is only one source.
 *   style.css        styles by CLASS only, never by tool id, so a new tool is
 *                    styled correctly the moment it is declared.
 *
 * Delete a tool: delete its entry. Reorder the toolbar: reorder the array.
 * Swap an implementation: change `button.icon` / `menu`, wire the same ids.
 *
 * WHAT THIS FILE MUST NOT DO
 * --------------------------
 * No behaviour. No DOM. No listeners. This is data that describes chrome;
 * canvas_extra.js owns what the chrome does, and finds it by id. That split is
 * what lets the registry be rewritten without touching 3000 lines of tool logic.
 *
 * THE ID CONTRACT
 * ---------------
 * Every `id` here is a BASE id. The renderer suffixes it with `_<uuid>` per
 * canvas, and canvas_extra.js looks it up as `<base>_<uuid>`. So an id in this
 * file is a promise to canvas_extra.js: rename one here and you must rename its
 * `el('<base>_')` call there. The audit in canvas_nodes.js reports any id that
 * is declared here and missing from the DOM, but it cannot know about an id
 * canvas_extra.js wants and nobody declares -- keep them in step by hand, and
 * let tests/toolbar_dom_js.js tell you when you did not.
 */
(function () {
    "use strict";

    // ---------------------------------------------------------------- helpers
    //
    // Row builders. They exist so the uniform parts of a menu (a labelled
    // slider, a picker row, a reset button) are declared as DATA rather than as
    // hand-written markup repeated 15 times -- which is how the black/white
    // sliders ended up in a box no button ever opened.
    //
    // `note` is rendered as an HTML comment, so the reasoning that used to live
    // in the template survives the move into data.

    /** label + slider, the shape of nearly every control in every menu.
     *
     *  `labelMax` is THE WIDEST TEXT THIS LABEL CAN EVER HOLD -- the same
     *  prefix canvas_extra.js writes, with the value at its most digits
     *  ("mask opacity 100", "rotate -45.0°", "thickness 0.05"). It is not
     *  decoration: canvas_nodes.js renders it into a zero-height CSS sizer and
     *  the label column is EXACTLY that wide, per menu, instead of the one
     *  100px column every menu used to pay for (see style.css, THE LABEL
     *  COLUMN). Declare it wrong and the column is wrong, so
     *  tests/test_toolbar_dom.py reads the format strings back out of
     *  canvas_extra.js and fails when a prefix here stops matching the one
     *  that is actually written at runtime.
     */
    function range(opts) {
        return {
            kind: 'range',
            id: opts.id,
            labelId: opts.labelId,
            label: opts.label,
            labelMax: opts.labelMax,
            min: opts.min,
            max: opts.max,
            value: opts.value,
            step: opts.step,
            cls: opts.cls || '',
            rowCls: opts.rowCls || '',
            note: opts.note || '',
        };
    }

    /** a colour-picker button + its swatch + its RGB readout.
     *  `rgbMax` is the readout's widest text, for the same reason a range row
     *  has `labelMax`: the box is sized from it and never breathes as pixels
     *  are picked. Three channels at three digits is the ceiling. */
    function pick(opts) {
        return {
            kind: 'pick',
            id: opts.id,
            title: opts.title,
            svg: opts.svg,
            swatchId: opts.swatchId,
            rgbId: opts.rgbId,
            rgbText: opts.rgbText || '&mdash;',
            rgbMax: opts.rgbMax || '255 255 255',
        };
    }

    /** the red ↺ that returns one tool to its defaults. */
    function reset(opts) {
        return {kind: 'button', id: opts.id, title: opts.title, text: '↺',
                cls: 'forge-tool-reset'};
    }

    /** an ordinary button inside a menu. */
    function button(opts) {
        return {kind: 'button', id: opts.id, title: opts.title, text: opts.text,
                cls: opts.cls || ''};
    }

    /** markup the builders cannot express (the pen's colour picker, the layer
     *  list). Every id inside MUST be listed in `ids` or the audit is blind to
     *  it -- that is the price of an escape hatch, and it is checked at load. */
    function raw(html, ids) {
        return {kind: 'raw', html: html, ids: ids || []};
    }

    /** Rows that must stay TOGETHER as one block instead of flowing
     *  individually into the menu's wrap.
     *
     *  A menu is a wrapping row of items, which is right when every item is the
     *  same shape. The pen menu is not: it has a 104px colour picker two rows
     *  tall, and letting its three sliders wrap around that put them on three
     *  different left edges, one of them underneath the picker. Grouped, they
     *  are ONE item - a column that sits beside the picker - and the sliders
     *  line up with each other again. */
    function group(cls, rows) {
        return {kind: 'group', cls: cls, rows: rows};
    }

    const PICK_SVG = (fill, stroke) =>
        '<svg class="forge-pick-icon" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6" ' +
        'fill="' + fill + '" stroke="' + stroke + '" stroke-width="2"/></svg>';

    // ------------------------------------------------------------------ tools
    //
    // Toolbar order IS array order. `control` picks the legacy CSS class the
    // reveal sweep keys off ('adjust' -> .forge-adjust-control, 'wmask' ->
    // .forge-wmask-control); `gap` starts a new visual group.

    const TOOLS = [
        {
            id: 'layers',
            button: {
                id: 'layersButton', icon: '⧉', control: 'adjust',
                title: 'Layers menu: compose multiple images on one stage - add an image as a layer, click to select, drag to move, wheel to scale, reorder and delete in any order, and fade one with the opacity field at the right of its row; every tool below (edges included) operates on the flattened composite',
            },
            menu: {
                id: 'layersBox', cls: 'forge-layers-box',
                note: 'layers menu: the stack is armed while this menu is open - click selects the topmost layer under the pointer, drag moves the active layer, wheel scales it around the pointer. Rows run topmost-first; the pen/eraser draw into the active layer. The rightmost field of a row is that layer\'s opacity in percent (100 = opaque), applied when the stack is flattened.',
                rows: [
                    raw('<div class="forge-cc-row">\n' +
                        '    <button id="layerAddButton_forge_mixin" class="forge-btn forge-no-select" title="Add an empty layer on top and make it active - fill it any way an image ever gets in: drag &amp; drop, paste, the open button, the ControlNet insert buttons, or just paint on it">＋</button>\n' +
                        '    <span class="forge-toolbar-label">add layer</span>\n' +
                        '</div>\n' +
                        '<div id="layerList_forge_mixin" class="forge-layer-list"></div>',
                        ['layerAddButton', 'layerList']),
                ],
            },
            overlays: [
                {id: 'layersOverlay', tag: 'canvas', cls: 'forge-layers-overlay',
                 attrs: 'width="1" height="1"',
                 note: 'active-layer outline + drag/scale ghost while the layers tool is open'},
            ],
        },

        {
            id: 'pen',
            button: {
                id: 'penButton', icon: '✎', control: 'adjust',
                title: 'Pen menu: draw straight into the ACTIVE LAYER (part of the composite, not a mask) - color square + hue, brush size, opacity, feathering, eraser toggle; drawing is armed while the menu is open',
            },
            menu: {
                id: 'penBox', cls: 'forge-pen-box',
                note: 'pen menu: photoshop-style color selector (saturation/value square + hue strip) and the brush sliders. Drawing is armed while this menu is open; strokes are baked into the image.',
                rows: [
                    // no id on the picker wrapper: nothing looks it up, the CSS
                    // positions it by class, and an id nobody resolves is the
                    // dead-chrome smell the DOM test now fails on.
                    raw('<div class="forge-pen-picker">\n' +
                        '    <canvas id="penSV_forge_mixin" class="forge-pen-sv" width="128" height="84"></canvas>\n' +
                        '    <div id="penSVCursor_forge_mixin" class="forge-pen-sv-cursor"></div>\n' +
                        '</div>\n' +
                        '<div class="forge-pen-side">\n' +
                        '    <input type="range" id="penHue_forge_mixin" class="forge-pen-hue" min="0" max="360" value="0">\n' +
                        '    <div class="forge-cc-row">\n' +
                        '        <button id="penPickButton_forge_mixin" class="forge-btn forge-no-select" title="Pick the brush color from the image (or hold Alt and click while drawing); one pick, then back to drawing"><svg class="forge-pick-icon" viewBox="0 0 16 16"><path d="M10.5 1.8a2 2 0 0 1 2.8 2.8L11.8 6l-1.6-1.6zM9.1 5.5 10.7 7l-5.4 5.4-2.2.6.6-2.2z" fill="currentColor"/></svg></button>\n' +
                        '        <span id="penSwatch_forge_mixin" class="forge-cc-swatch"></span>\n' +
                        // data-max: the hex readout is SEVEN characters forever,
                        // so it has no business reserving room for "255 255 255"
                        // the way the pick rows must (see pick()/style.css).
                        '        <span id="penHex_forge_mixin" class="forge-cc-rgb" data-max="#000000">#000000</span>\n' +
                        '    </div>\n' +
                        '</div>',
                        ['penSV', 'penSVCursor', 'penHue', 'penPickButton',
                         'penSwatch', 'penHex']),
                    // one block, so the three of them sit beside the picker on
                    // one left edge instead of wrapping around it
                    group('forge-row-column', [
                        range({id: 'penSize', labelId: 'penSizeLabel', label: 'brush 12 px',
                               labelMax: 'brush 256 px',
                               min: 0, max: 100, value: 45,
                               note: 'value is an exponent: diameter = 2^(v/12.5) image px, 1 .. 256'}),
                        range({id: 'penOpacity', labelId: 'penOpacityLabel', label: 'opacity 100',
                               labelMax: 'opacity 100',
                               min: 1, max: 100, value: 100}),
                        range({id: 'penFeather', labelId: 'penFeatherLabel', label: 'feathering 0',
                               labelMax: 'feathering 100',
                               min: 0, max: 100, value: 0,
                               note: '0 = hard round brush, 100 = the whole radius is falloff'}),
                    ]),
                    button({id: 'penEraserButton', text: '⌫',
                            title: 'Eraser: strokes cut the ACTIVE LAYER to transparency (lower layers show through); brush size, opacity and feathering apply as erase strength'}),
                    button({id: 'penUndoButton', text: '↶',
                            title: 'Undo the last pen/eraser stroke on the active layer'}),
                    reset({id: 'penClearButton',
                           title: 'Remove every pen/eraser stroke of the active layer'}),
                ],
            },
            overlays: [
                {id: 'penOverlay', tag: 'canvas', cls: 'forge-pen-overlay',
                 attrs: 'width="1" height="1"',
                 note: 'live preview of the stroke in progress; on release the stroke is baked into the image itself and this canvas is cleared'},
                {id: 'penCursor', tag: 'div', cls: 'forge-pen-cursor'},
            ],
        },

        {
            id: 'flip',
            button: {id: 'flipButton', icon: '⇄', control: 'adjust', gap: true,
                     title: 'Flip horizontally'},
        },

        {
            id: 'rotate',
            button: {id: 'rotateButton', icon: '∠', control: 'adjust',
                     title: 'Fine rotation slider'},
            menu: {
                id: 'rotateBox',
                rows: [
                    range({id: 'rotationSlider', labelId: 'rotationLabel', label: 'rotate 0.0°',
                           labelMax: 'rotate -45.0°',
                           min: -45, max: 45, value: 0, step: 0.1}),
                    reset({id: 'rotateResetButton', title: 'Reset rotation'}),
                ],
            },
        },

        {
            id: 'crop',
            button: {id: 'cropButton', icon: '✂', control: 'adjust',
                     title: 'Crop: drag the border handles, toggle off to apply (non-destructive)'},
            overlays: [
                {id: 'cropOverlay', tag: 'div', cls: 'forge-crop-overlay',
                 note: 'photoshop-style crop overlay: shaded cut-off areas + draggable handles',
                 html:
                    '<div id="cropShadeN_forge_mixin" class="forge-crop-shade"></div>\n' +
                    '<div id="cropShadeS_forge_mixin" class="forge-crop-shade"></div>\n' +
                    '<div id="cropShadeW_forge_mixin" class="forge-crop-shade"></div>\n' +
                    '<div id="cropShadeE_forge_mixin" class="forge-crop-shade"></div>\n' +
                    '<div id="cropBox_forge_mixin" class="forge-crop-box">\n' +
                    '    <div class="forge-crop-thirds forge-crop-thirds-h" style="top: 33.333%;"></div>\n' +
                    '    <div class="forge-crop-thirds forge-crop-thirds-h" style="top: 66.667%;"></div>\n' +
                    '    <div class="forge-crop-thirds forge-crop-thirds-v" style="left: 33.333%;"></div>\n' +
                    '    <div class="forge-crop-thirds forge-crop-thirds-v" style="left: 66.667%;"></div>\n' +
                    '    <div class="forge-crop-move" data-handle="move"></div>\n' +
                    '    <div class="forge-crop-edge forge-crop-n" data-handle="n"></div>\n' +
                    '    <div class="forge-crop-edge forge-crop-s" data-handle="s"></div>\n' +
                    '    <div class="forge-crop-edge forge-crop-w" data-handle="w"></div>\n' +
                    '    <div class="forge-crop-edge forge-crop-e" data-handle="e"></div>\n' +
                    '    <div class="forge-crop-corner forge-crop-nw" data-handle="nw"></div>\n' +
                    '    <div class="forge-crop-corner forge-crop-ne" data-handle="ne"></div>\n' +
                    '    <div class="forge-crop-corner forge-crop-sw" data-handle="sw"></div>\n' +
                    '    <div class="forge-crop-corner forge-crop-se" data-handle="se"></div>\n' +
                    '</div>',
                 ids: ['cropShadeN', 'cropShadeS', 'cropShadeW', 'cropShadeE', 'cropBox']},
            ],
        },

        {
            id: 'color',
            button: {
                id: 'colorButton', control: 'adjust', icon: PICK_SVG('none', 'currentColor'),
                title: 'Levels and colour: black/white point sliders, and black/white/gray point pickers',
            },
            // THE BLACK/WHITE SLIDERS LIVE HERE ON PURPOSE.
            //
            // They used to sit in a second box, `levelsBox`, which had its own
            // id, its own markup, wired `input` handlers in canvas_extra.js --
            // and NO BUTTON ANYWHERE THAT OPENED IT. Two working sliders,
            // unreachable in the shipped UI, in the fork and in CNPro alike.
            // Nothing raised, because `display: none` forever is not an error.
            //
            // They belong to the same tool as the pickers (both set the black
            // and white points; the pickers just sample them off the image), so
            // one tool now owns one menu. `levelsBox` is gone -- an id that no
            // longer exists cannot be silently empty.
            menu: {
                id: 'colorBox',
                rows: [
                    range({id: 'blackPoint', labelId: 'blackLabel', label: 'black 0',
                           labelMax: 'black 254',
                           min: 0, max: 254, value: 0}),
                    range({id: 'whitePoint', labelId: 'whiteLabel', label: 'white 255',
                           labelMax: 'white 255',
                           min: 1, max: 255, value: 255}),
                    pick({id: 'pickBlackButton', title: 'Pick black point on image',
                          svg: PICK_SVG('#000000', '#ffffff'),
                          swatchId: 'blackSwatch', rgbId: 'blackRgb'}),
                    pick({id: 'pickWhiteButton', title: 'Pick white point on image',
                          svg: PICK_SVG('#ffffff', '#000000'),
                          swatchId: 'whiteSwatch', rgbId: 'whiteRgb'}),
                    pick({id: 'pickGrayButton',
                          title: 'Gray area picker: paint one or more neutral regions, toggle off to neutralize color cast (brightness unaffected); click outside the image to cancel',
                          svg: PICK_SVG('#9e9e9e', '#ffffff'),
                          swatchId: 'graySwatch', rgbId: 'grayRgb'}),
                    reset({id: 'pointsResetButton', title: 'Reset black/white/gray points'}),
                ],
            },
            overlays: [
                {id: 'grayMaskOverlay', tag: 'canvas', cls: 'forge-graymask-overlay',
                 attrs: 'width="1" height="1"',
                 note: 'gray-area picker: painted mask regions overlay'},
                {id: 'pickReticle', tag: 'div', cls: 'forge-pick-reticle',
                 note: 'pick-pixel reticle: open center so the target pixel stays visible',
                 html:
                    '<div class="forge-reticle-arm forge-reticle-n"></div>\n' +
                    '<div class="forge-reticle-arm forge-reticle-s"></div>\n' +
                    '<div class="forge-reticle-arm forge-reticle-w"></div>\n' +
                    '<div class="forge-reticle-arm forge-reticle-e"></div>'},
            ],
        },

        {
            id: 'gamma',
            button: {id: 'gammaButton', icon: 'γ', control: 'adjust',
                     title: 'Gamma slider (wide, fine log scale; double-click slider to reset)'},
            menu: {
                id: 'gammaBox',
                rows: [
                    // step 5 on a 4644-wide range is 929 notches end to end, and
                    // the wheel steps by exactly that -- which is why this no
                    // longer needs a wider track than any other slider (it used
                    // to carry `min-width: 380px`, and that stretched the whole
                    // toolbar past the image it sits on).
                    // ±2320, NOT ±2322. A range input snaps to the step grid
                    // measured from `min`, and 4644 is not a multiple of 5 -- so
                    // with ±2322 the highest reachable value was 2318 and the
                    // slider could never touch its own stated maximum. Nothing
                    // reported it, because a slider that stops 4 units early
                    // looks exactly like a slider. ±2320 is 928 whole steps, ends
                    // included, and 0 (= gamma 1.0) stays exactly on the grid.
                    // Costs 0.0003 of gamma range at each end.
                    range({id: 'gammaSlider', labelId: 'gammaLabel', label: 'gamma 1.000',
                           labelMax: 'gamma 4.987',
                           min: -2320, max: 2320, value: 0, step: 5,
                           note: 'value is log2(gamma) * 1000: symmetric around 0 (= gamma 1.0), range 0.2 .. 5.0; span is a whole number of steps so both ends are reachable'}),
                    reset({id: 'gammaResetButton', title: 'Reset gamma'}),
                ],
            },
        },

        {
            id: 'grayscale',
            button: {id: 'grayscaleButton', icon: '◫', control: 'adjust',
                     title: 'Convert to grayscale (non-destructive toggle)'},
        },

        {
            id: 'invert',
            button: {id: 'invertButton', icon: '◙', control: 'adjust',
                     title: 'Invert colors (non-destructive toggle; applied last, so it also inverts the edges mask)'},
        },

        {
            id: 'edges',
            button: {id: 'edgesButton', icon: '△', control: 'adjust',
                     title: 'Edges mask menu: black edges over a white mask, line thickness follows edge prominence; the effect is active while any opacity is above 0 (closing the menu never deactivates it)'},
            menu: {
                id: 'edgesBox',
                rows: [
                    range({id: 'edgeOpacity', labelId: 'edgeOpacityLabel', label: 'edge opacity 0',
                           labelMax: 'edge opacity 100',
                           min: 0, max: 100, value: 0,
                           note: '0 = edges fully transparent (original colors show through), 100 = opaque black edges; both opacities at 0 = edges effect inactive'}),
                    range({id: 'maskOpacity', labelId: 'maskOpacityLabel', label: 'mask opacity 0',
                           labelMax: 'mask opacity 100',
                           min: 0, max: 100, value: 0,
                           note: '0 = edge-free areas fully transparent (original image shows through), 100 = opaque white mask'}),
                    range({id: 'edgeSensitivity', labelId: 'edgeSensLabel', label: 'sensitivity 50',
                           labelMax: 'sensitivity 100',
                           min: 1, max: 100, value: 50}),
                    range({id: 'edgeThickness', labelId: 'edgeThickLabel', label: 'thickness 2',
                           labelMax: 'thickness 0.05',
                           min: 0, max: 7, value: 5, step: 1,
                           note: 'value is an index into the nonlinear 1-2-5 stop series 0.05 .. 10 (canvas_extra.js THICKNESS_STOPS); the label prints the STOP, so its widest text is the four-character 0.05'}),
                    range({id: 'edgeFeather', labelId: 'edgeFeatherLabel', label: 'feathering 0',
                           labelMax: 'feathering 100',
                           min: 0, max: 100, value: 0,
                           note: 'Topology-preserving thinning: broad contours are eaten from both sides towards a precise centerline (0 = original thickness, 100 = one-pixel lines); details that are already one pixel wide are preserved'}),
                    reset({id: 'edgesResetButton',
                           title: 'Reset the menu to defaults (opacities 0 = effect off, sensitivity 50, thickness 2, feathering 0)'}),
                ],
            },
        },

        // ---- Topaz: the only DEFERRED tools. They stay hidden until the server
        // says tpai.exe exists, which is the one legitimate reason for a
        // declared control to be invisible after attach. Everything else that is
        // hidden is a bug and the audit says so.
        {
            id: 'topazMpx',
            button: {id: 'topazMpxButton', icon: '1M', gap: true,
                     title: 'Topaz resample to ~1 Mpx',
                     deferred: 'revealed only when the server reports tpai.exe available'},
        },
        {
            id: 'topazHq',
            button: {id: 'topazHqButton', icon: 'HQ',
                     title: 'Topaz enhance (same size)',
                     deferred: 'revealed only when the server reports tpai.exe available'},
        },
        {
            id: 'topazDenoise',
            button: {id: 'topazDenoiseButton', icon: 'DN',
                     title: 'Topaz denoise',
                     deferred: 'revealed only when the server reports tpai.exe available'},
        },

        // ---- weight masks: CNPro's own feature, not the fork's canvas.
        //
        // TWO SURFACES, TWO QUESTIONS. On an INPUT canvas there is one slot, G,
        // and it answers "which part of this input is worth reading" - a shape,
        // plus one number (the painted value, reduced to this input's scalar
        // share). On the OUTPUT canvas there are four, and they are the four
        // profiles' spatial half: G for the main profile (with depth/drift,
        // which shape main), C/M/F for the band profiles, with the editor's
        // band selector deciding which run (weight_mask.js liveSlotKeys,
        // external_code.masks_in_force). Hence the glyph colours: each slot
        // wears the colour of the plot line whose profile it belongs to.
        //
        // The bands are UNET DEPTH, which is why they are output-side: depth is
        // a property of where control is injected, and an input canvas may not
        // be spatially related to the output at all (an IP-Adapter reference is
        // not). G is the only slot both surfaces share, so it is the only one
        // scoped to both.
        {
            id: 'wmaskGlobal',
            button: {id: 'wmaskButton', icon: 'G', control: 'wmask', gap: true,
                     scope: '.cnet-input-image-group, .cnet-output-mask-group',
                     cls: 'forge-wmask-slot forge-wmask-slot-global',
                     title: 'Global weight mask (G). On an INPUT: which part of this input is read at all - the painted region gates what the control model (or CLIP) is shown of it, and the painted VALUE becomes this input\'s share of the unit - one weight per input, whatever shape you paint it in (paint two and they are averaged, with a warning). On the OUTPUT mask: where the MAIN profile\'s control lands, applied while the main (or depth) selector is pressed in the profile editor. Pick weight and brush size below, click outside the image to exit'},
            menu: {
                id: 'wmaskBox',
                rows: [
                    range({id: 'wmaskWeight', labelId: 'wmaskWeightLabel', label: 'weight 1.00',
                           labelMax: 'weight 1.00',
                           min: 0, max: 1, value: 1, step: 0.01, cls: 'forge-wmask-range'}),
                    range({id: 'wmaskBrush', labelId: 'wmaskBrushLabel', label: 'brush 50',
                           labelMax: 'brush 100',
                           min: 1, max: 100, value: 50,
                           note: 'relative size: 100 = brush diameter of 25% of the image diagonal'}),
                    button({id: 'wmaskInvertButton', text: 'i',
                            title: 'Invert the active weight mask (unpainted = 0, so it becomes weight 1); press again to undo'}),
                    reset({id: 'wmaskClearButton', title: 'Clear the active weight mask'}),
                ],
            },
        },
        {
            id: 'wmaskCoarse',
            button: {id: 'wmaskCoarseButton', icon: 'C', control: 'wmask',
                     scope: '.cnet-output-mask-group',
                     cls: 'forge-wmask-slot forge-wmask-slot-coarse',
                     title: 'Coarse-band output mask (C): where the composition layers (deepest injections + middle block) steer the OUTPUT. Belongs to the COARSE band profile - the C/M/F masks are applied while a band selector is pressed in the profile editor, and the G mask instead while main/depth is. Among them, a band without its own mask contributes ZERO control'},
        },
        {
            id: 'wmaskMid',
            button: {id: 'wmaskMidButton', icon: 'M', control: 'wmask',
                     scope: '.cnet-output-mask-group',
                     cls: 'forge-wmask-slot forge-wmask-slot-mid',
                     title: 'Mid-band output mask (M): where the form layers steer the OUTPUT. Belongs to the MID band profile - the C/M/F masks are applied while a band selector is pressed in the profile editor, and the G mask instead while main/depth is. Among them, a band without its own mask contributes ZERO control'},
        },
        {
            id: 'wmaskFine',
            button: {id: 'wmaskFineButton', icon: 'F', control: 'wmask',
                     scope: '.cnet-output-mask-group',
                     cls: 'forge-wmask-slot forge-wmask-slot-fine',
                     title: 'Fine-band output mask (F): where the texture layers (shallowest injections) steer the OUTPUT. Belongs to the FINE band profile - the C/M/F masks are applied while a band selector is pressed in the profile editor, and the G mask instead while main/depth is. Among them, a band without its own mask contributes ZERO control'},
        },
    ];

    // ------------------------------------------------------------- derivation
    //
    // Everything below is COMPUTED from TOOLS. Nothing here is a second list to
    // keep in sync -- that was the whole disease.

    /** every row of a menu, with the contents of any group() SPLICED IN.
     *
     *  Everything downstream (the id contract, the label sizers, the audit)
     *  asks for the rows and must see the same set whether a row was grouped
     *  for layout reasons or not - grouping is presentation, and presentation
     *  must not be able to hide a control from the checks. */
    function flatRows(menu) {
        const out = [];
        for (const row of (menu && menu.rows) || []) {
            if (row.kind === 'group') out.push(...(row.rows || []));
            else out.push(row);
        }
        return out;
    }

    /** every base id this registry is responsible for, in declaration order. */
    function allIds() {
        const ids = [];
        for (const tool of TOOLS) {
            if (tool.button) ids.push(tool.button.id);
            if (tool.menu) {
                ids.push(tool.menu.id);
                for (const row of flatRows(tool.menu)) {
                    if (row.kind === 'range') ids.push(row.labelId, row.id);
                    else if (row.kind === 'pick') ids.push(row.id, row.swatchId, row.rgbId);
                    else if (row.kind === 'button') ids.push(row.id);
                    else if (row.kind === 'raw') ids.push(...row.ids);
                }
            }
            for (const ov of tool.overlays || []) {
                ids.push(ov.id);
                if (ov.ids) ids.push(...ov.ids);
            }
        }
        return ids;
    }

    /** the toolbar buttons only -- what the reveal sweep and the audit walk. */
    function buttonIds() {
        return TOOLS.filter((t) => t.button).map((t) => t.button.id);
    }

    /** {buttonId: why it may legitimately stay hidden after attach}. */
    function deferred() {
        const out = {};
        for (const tool of TOOLS) {
            if (tool.button && tool.button.deferred) out[tool.button.id] = tool.button.deferred;
        }
        return out;
    }

    /** {buttonId: ancestor selector it is scoped to}. A scoped button is only
     *  revealed on a canvas inside a matching container - the weight-mask
     *  slots are wired by weight_mask.js on CNPro's input canvases alone, and
     *  revealing them on the host's own img2img/inpaint canvases produced
     *  injected, styled, permanently INERT chrome there (the exact
     *  visible-but-dead shape rule 8c names). Registry data, not a check in a
     *  second file, so reveal and audit cannot disagree about the scope. */
    function scopes() {
        const out = {};
        for (const tool of TOOLS) {
            if (tool.button && tool.button.scope) out[tool.button.id] = tool.button.scope;
        }
        return out;
    }

    window.cnproCanvasTools = {
        TOOLS: TOOLS,
        allIds: allIds,
        buttonIds: buttonIds,
        deferred: deferred,
        scopes: scopes,
        flatRows: flatRows,
    };
})();
