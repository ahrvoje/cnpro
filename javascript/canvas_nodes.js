/**
 * CNPro canvas tool nodes -- renders javascript/canvas_tools.js into the DOM.
 *
 * WHY INJECT INSTEAD OF EDITING canvas.html
 * -----------------------------------------
 * The original fork added this chrome directly to
 * modules_forge/forge_canvas/canvas.html. An extension cannot edit a host file,
 * and rebinding the host's `canvas_html` global would change the markup for
 * EVERY ForgeCanvas in the app, including the host's own img2img canvases.
 * Injecting per-container at attach time keeps the blast radius to canvases
 * CNPro attaches to, and survives the host changing its own template.
 *
 * WHAT CHANGED, AND WHY IT MATTERS
 * --------------------------------
 * This file used to be GENERATED: a 187-line verbatim copy of the fork's markup,
 * a hand-written OWNED_IDS array, a hand-written DEFERRED map, and a regex that
 * parsed ids back out of the blob it had just been handed. Four representations
 * of one fact. Every bug this module has ever had was two of them disagreeing:
 *
 *   * a button carried .forge-wmask-control while the reveal swept
 *     .forge-adjust-control                    -> feature absent from the toolbar
 *   * the id regex sliced 14 characters off a 13-character suffix
 *                                              -> the ENTIRE toolbar vanished
 *   * `levelsBox` had markup, ids and wired handlers, and no button that opened
 *     it                                       -> two working sliders unreachable
 *
 * There is now ONE representation: the TOOLS array in canvas_tools.js. Markup is
 * rendered from it, the id contract is derived from it, and DEFERRED is read off
 * it. Two things cannot disagree when there is only one of them.
 *
 * LOAD ORDER IS DELIBERATELY IRRELEVANT
 * -------------------------------------
 * Extension scripts load in filename order, and "canvas_nodes" sorts BEFORE
 * "canvas_tools". So this file must not touch the registry at load time -- it
 * resolves it lazily, on first use, by which point every script has run. Same
 * discipline as canvas_adapter.js: define now, resolve later. Renaming either
 * file can therefore never break the pair.
 */
(function () {
    "use strict";

    // The per-canvas suffix placeholder. `inject()` swaps it for the real uuid,
    // exactly once, over the whole rendered string.
    const MIXIN = 'forge_mixin';

    // -------------------------------------------------------------- registry
    //
    // Resolved on first use, never at load. A missing registry is fatal to the
    // toolbar, so it is reported once, by name, rather than producing an empty
    // toolbar with no explanation -- which is the failure mode this whole module
    // is built around avoiding.
    let registryWarned = false;
    function tools() {
        const reg = window.cnproCanvasTools;
        if (!reg || !Array.isArray(reg.TOOLS)) {
            if (!registryWarned) {
                registryWarned = true;
                console.error('[cnpro] javascript/canvas_tools.js did not load - the tool ' +
                              'registry is empty, so NO tool buttons, menus or overlays ' +
                              'will be injected into any canvas.');
            }
            return null;
        }
        return reg;
    }

    // ---------------------------------------------------------------- render
    //
    // Attribute values are escaped; icons and `raw` rows are not, because they
    // are markup by definition (an SVG icon, the pen's colour picker). That
    // split is the only place trust is granted, and it is granted to this
    // repository's own registry file, never to anything a user can supply.

    function esc(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function id(base) {
        return base + '_' + MIXIN;
    }

    function note(text, indent) {
        return text ? indent + '<!-- ' + text + ' -->\n' : '';
    }

    // 'adjust' and 'wmask' are the two legacy reveal classes. They are kept
    // because style.css and canvas_extra.js's class sweep both key off them;
    // the registry says which group a tool is in, and this is the only place
    // that knows the class names.
    const CONTROL_CLASS = {
        adjust: 'forge-adjust-control',
        wmask: 'forge-wmask-control',
    };

    function renderButton(b) {
        const cls = ['forge-btn', 'forge-no-select'];
        if (b.control) cls.push(CONTROL_CLASS[b.control]);
        if (b.gap) cls.push('forge-adjust-gap');
        // extra classes a tool wants for STYLING (the weight-mask slots colour
        // their glyph to match their profile's plot line). A class, never an
        // id: a tool added to the registry has to be styled by what it IS.
        if (b.cls) cls.push(b.cls);
        // Injected hidden, then revealed by revealToolbar(). The template stays
        // clean for any canvas that gets markup but no JS wiring.
        return '                <button id="' + id(b.id) + '" class="' + cls.join(' ') +
               '" title="' + esc(b.title) + '" style="display: none;">' + b.icon + '</button>\n';
    }

    // ------------------------------------------------------ the label sizers
    //
    // THE LABEL COLUMN IS AS WIDE AS THE WIDEST TEXT THE MENU CAN SHOW, AND NO
    // WIDER. It used to be one 100px column for every menu in the toolbar,
    // sized for "mask opacity 100" -- so the gamma menu, whose longest label is
    // "gamma 4.987", paid 34px of empty column for a label it does not have,
    // on a strip of chrome that sits on top of the user's image.
    //
    // It cannot simply be `width: auto` either: every one of these labels ends
    // in a LIVE VALUE ("black 0" -> "black 255"), so an auto column breathes as
    // the user drags, slides the slider sideways under the cursor and, at the
    // wrap boundary, bounces a row onto the next line mid-gesture. That is the
    // bug the fixed 100px was introduced to fix.
    //
    // So: fixed, but fixed to the RIGHT number. Each row declares `labelMax`,
    // the widest text it can ever hold, and every label in a menu is rendered
    // with the whole menu's set of them in `data-max`. style.css paints that
    // into a zero-height, invisible ::before, and the browser -- which is the
    // only thing that knows the font actually in use -- sizes the box to the
    // widest line. Exact, no measurement, no font-loading race, and no drift
    // from a number typed into a stylesheet.
    //
    // ONE WIDTH PER MENU, not per row. The rows of a menu wrap into lines, and
    // equal-width rows are what makes those lines read as a grid; per-row
    // widths save a few more pixels and leave every line ragged.
    //
    // Joined with '\n' and rendered under `white-space: pre`, so each candidate
    // is its own line inside a box that has no height.
    const SIZER_SEP = '\n';

    function sizerAttr(strings) {
        const seen = [];
        for (const s of strings) {
            if (s && seen.indexOf(s) === -1) seen.push(s);
        }
        if (!seen.length) return '';
        // The attribute really does carry newlines; &#10; is how they survive
        // being written into markup.
        return ' data-max="' + esc(seen.join(SIZER_SEP)).split('\n').join('&#10;') + '"';
    }

    /** the menu's rows with any group() spliced in, so nothing downstream has
     *  to know that grouping exists. Falls back to the raw list when the
     *  registry is older than this helper. */
    function rowsOf(m) {
        const reg = tools();
        if (reg && typeof reg.flatRows === 'function') return reg.flatRows(m);
        return (m && m.rows) || [];
    }

    /** the widest label texts a menu's range rows can produce, in row order. */
    function menuLabelMaxes(m) {
        return rowsOf(m)
            .filter((row) => row.kind === 'range')
            .map((row) => row.labelMax || row.label);
    }

    function renderRange(r, sizer) {
        const cls = ['forge-toolbar-range'];
        if (r.cls) cls.push(r.cls);
        const rowCls = ['forge-range-row'];
        if (r.rowCls) rowCls.push(r.rowCls);
        let out = '                <div class="' + rowCls.join(' ') + '">\n';
        // data-label-max is this row's OWN widest text and data-max the whole
        // menu's set. Keeping both means a row injected later (weight_mask.js's
        // feather slider) can join the column by declaring its own and calling
        // syncLabelSizers -- the union is recomputed from the rows present,
        // never from a list kept somewhere else.
        out += '                    <div id="' + id(r.labelId) +
               '" class="forge-toolbar-label" data-label-max="' + esc(r.labelMax || r.label) + '"' +
               sizer + '>' + esc(r.label) + '</div>\n';
        out += note(r.note, '                    ');
        out += '                    <input type="range" id="' + id(r.id) + '" class="' +
               cls.join(' ') + '" min="' + r.min + '" max="' + r.max + '" value="' + r.value + '"' +
               (r.step != null ? ' step="' + r.step + '"' : '') + '>\n';
        out += '                </div>\n';
        return out;
    }

    function renderPick(p) {
        return '                <div class="forge-cc-row">\n' +
               '                    <button id="' + id(p.id) +
               '" class="forge-btn forge-no-select" title="' + esc(p.title) + '">' + p.svg + '</button>\n' +
               '                    <span id="' + id(p.swatchId) + '" class="forge-cc-swatch"></span>\n' +
               '                    <span id="' + id(p.rgbId) + '" class="forge-cc-rgb"' +
               sizerAttr([p.rgbMax]) + '>' + p.rgbText + '</span>\n' +
               '                </div>\n';
    }

    function renderMenuButton(b) {
        const cls = ['forge-btn', 'forge-no-select'];
        if (b.cls) cls.push(b.cls);
        return '                <button id="' + id(b.id) + '" class="' + cls.join(' ') +
               '" title="' + esc(b.title) + '">' + b.text + '</button>\n';
    }

    function indent(html, pad) {
        return html.split('\n').map((line) => (line ? pad + line : line)).join('\n') + '\n';
    }

    function renderMenu(m) {
        const cls = ['forge-toolbar-box-c'];
        if (m.cls) cls.push(m.cls);
        let out = note(m.note, '            ');
        out += '            <div class="' + cls.join(' ') + '" id="' + id(m.id) +
               '" style="display: none;">\n';
        const sizer = sizerAttr(menuLabelMaxes(m));
        out += renderRows(m.rows || [], sizer, '                ');
        out += '            </div>\n';
        return out;
    }

    /** rows of a menu, or of a group inside one (groups nest exactly once). */
    function renderRows(rows, sizer, pad) {
        let out = '';
        for (const row of rows) {
            if (row.kind === 'range') out += renderRange(row, sizer);
            else if (row.kind === 'pick') out += renderPick(row);
            else if (row.kind === 'button') out += renderMenuButton(row);
            else if (row.kind === 'raw') out += indent(row.html, pad);
            else if (row.kind === 'group') {
                out += pad + '<div class="forge-row-group ' + esc(row.cls || '') + '">\n' +
                       renderRows(row.rows || [], sizer, pad + '    ') +
                       pad + '</div>\n';
            }
        }
        return out;
    }

    function renderOverlay(o) {
        const tag = o.tag || 'div';
        let out = note(o.note, '        ');
        out += '        <' + tag + ' id="' + id(o.id) + '" class="' + o.cls + '"' +
               (o.attrs ? ' ' + o.attrs : '') + '>';
        if (o.html) out += '\n' + indent(o.html, '            ') + '        ';
        out += '</' + tag + '>\n';
        return out;
    }

    // ------------------------------------------------------------ id contract
    //
    // Derived, never listed. There is no second array to fall out of step with
    // the markup, because the markup and these ids come from the same objects.

    function toolbarIds() {
        const reg = tools();
        return reg ? reg.buttonIds() : [];
    }

    function ownedIds() {
        const reg = tools();
        return reg ? reg.allIds() : [];
    }

    function deferredMap() {
        const reg = tools();
        return reg ? reg.deferred() : {};
    }

    function scopeMap() {
        const reg = tools();
        return (reg && reg.scopes) ? reg.scopes() : {};
    }

    // A scoped button (registry `scope`) belongs only to canvases inside a
    // matching container; everywhere else it legitimately stays hidden - the
    // same standing as DEFERRED, and read off the same registry so the reveal
    // and the audit cannot disagree about it.
    function outOfScope(node, base, scopes) {
        const scope = scopes[base];
        return !!(scope && node && node.closest && !node.closest(scope));
    }

    // ---------------------------------------------------------------- inject

    function inject(uuid) {
        selfCheck(); // once per page; see the self-check note below
        const reg = tools();
        if (!reg) return false;

        const toolbar = document.getElementById('toolbar_' + uuid);
        const container = document.getElementById('imageContainer_' + uuid);
        if (!toolbar || !container) return false;
        if (container.dataset.cnproNodes === '1') return true;

        // BUTTONS ARE WRAPPED PER GROUP, and the group is the unit that wraps.
        //
        // The row is `flex-wrap: wrap` and, on a canvas too narrow for it, it
        // does wrap - but a flat row breaks wherever the arithmetic happens to
        // land, which put "1M" alone at the end of one line and "HQ DN" at the
        // start of the next. The groups the registry already declares (`gap:
        // true` starts one) are the natural break points, so each becomes a
        // flex ITEM and the browser breaks between them.
        //
        // Nesting is safe for everything downstream: canvas_extra.js and the
        // reveal sweep both find buttons by ID, never by position in box-a, and
        // the group gap stays where it was (margin-left on the group's first
        // button) so the specificity rule that protects it is untouched.
        const groups = [];
        let menus = '';
        let overlays = '';
        for (const tool of reg.TOOLS) {
            if (tool.button) {
                if (!groups.length || tool.button.gap) groups.push([]);
                groups[groups.length - 1].push(renderButton(tool.button));
            }
            if (tool.menu) menus += renderMenu(tool.menu);
            for (const ov of tool.overlays || []) overlays += renderOverlay(ov);
        }
        const buttons = groups.map((g) =>
            '            <div class="forge-btn-group">\n' + g.join('') +
            '            </div>\n').join('');

        const sub = (html) => html.split(MIXIN).join(uuid);

        const boxA = toolbar.querySelector('.forge-toolbar-box-a');
        if (boxA) boxA.insertAdjacentHTML('beforeend', sub(buttons));
        toolbar.insertAdjacentHTML('beforeend', sub(menus));
        container.insertAdjacentHTML('beforeend', sub(overlays));

        // Marks the canvas as CNPro-owned. style.css scopes ALL of its canvas
        // rules to this attribute, so the host's own canvases -- img2img, inpaint
        // -- keep the host's geometry untouched. Set after injection, so a
        // half-injected canvas is never styled as if it were complete.
        container.dataset.cnproNodes = '1';
        return true;
    }

    // -------------------------------------------------------- THE WIDTH CONTRACT
    //
    //   A tool menu is exactly as wide as the usable canvas, and its rows are
    //   elastic: they pack as many per line as the width allows and stretch to
    //   fill it.
    //
    // Two facts fix this, and neither is negotiable:
    //
    //   1. `.forge-image-container` is `overflow: hidden`. A menu wider than the
    //      canvas is CLIPPED, not shown. So the canvas width is a hard ceiling —
    //      not a preference, a physical limit.
    //   2. Anything narrower wastes space that cannot be recovered anywhere
    //      else, and the cost is paid in HEIGHT: at 190px per row a 520px canvas
    //      fits two rows per line, so the five-row edges menu needs three lines.
    //
    // Ceiling and floor are therefore the same number, and the contract is a
    // single value: **usable width = the canvas, minus the toolbar's own
    // horizontal padding**. There is nothing to choose.
    //
    // What was wrong before, in order:
    //
    //   * `min-width: 100%` — a percentage against an indefinite containing
    //     block, which degrades to min-content in gradio: one row per line.
    //   * width pinned to the BUTTON ROW — correct-looking, and it caps the menu
    //     at the buttons' width even when the canvas is far wider.
    //   * `width: max-content` between a measured floor and ceiling — the floor
    //     was the button row again, so on a narrow canvas the gamma slider and
    //     its reset still could not share a line.
    //
    // Each of those made the width depend on something that is not the question.
    // The question is only ever "how much room is there".
    //
    // How many rows fit per line is then a consequence, owned by ONE number in
    // the stylesheet (`--cnpro-row-basis`) and not by any of this.
    //
    // THIS WAS TRIED IN PURE CSS AND IT DOES NOT SURVIVE CONTACT WITH GRADIO.
    // The idiom is `width: 0` (contribute nothing to the toolbar's intrinsic
    // width) plus `min-width: 100%` (stretch back to it). It measures correctly
    // in a bare page -- 615px against a 615px button row -- and in the real app
    // the percentage resolves against an indefinite containing block, falls back
    // to `min-width: auto`, and a flex container's automatic minimum is its
    // MIN-CONTENT: the width of one `label | slider` row, about 200px. Every
    // menu then stacks one row per line, which is exactly the layout it was
    // written to prevent.
    //
    // A percentage that silently degrades to min-content is not a mechanism, it
    // is a coin flip that depends on ancestors this module cannot see. So the
    // width is MEASURED and published as a custom property. `--cnpro-menu-w` has
    // no ambiguous resolution and no fallback path: it is either the button
    // row's width in px, or absent, in which case the CSS default of `100%`
    // applies and behaves no worse than before.
    //
    // scrollWidth, not the bounding rect: if the button row is itself being
    // squeezed, its rect is the squeezed width while scrollWidth is the extent
    // the buttons actually occupy -- and the menus should match what is on
    // screen, not what the box claims.
    //
    // MEASURED IN PIXELS AND WRITTEN INLINE, never as a percentage and never as
    // an intrinsic keyword. Both of those have failed here: `min-width: 100%`
    // resolves against an indefinite containing block in gradio, degrades to
    // `min-width: auto`, and a flex container's automatic minimum is its
    // MIN-CONTENT — one row per line. That could not be reproduced outside the
    // running app (not with the static toolbar, not with the scribble row
    // hidden, not with the measurement disabled), so the repair deliberately
    // does not rely on knowing why: a px value has one unambiguous resolution
    // and nothing to degrade to, and an inline declaration outranks every
    // stylesheet rule short of !important.
    function syncMenuWidth(uuid) {
        const toolbar = document.getElementById('toolbar_' + uuid);
        const container = document.getElementById('imageContainer_' + uuid);
        if (!toolbar || !container) return 0;
        const view = toolbar.ownerDocument.defaultView;
        if (!view || !view.getComputedStyle) return 0;

        // The canvas, minus the toolbar's own horizontal padding. Padding is
        // READ, not assumed: it is the host's (`.forge-toolbar { padding: 6px
        // 10px }`) and a host that changes it would otherwise push every menu
        // that many pixels into the clip.
        const cs = view.getComputedStyle(toolbar);
        const pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
        const width = Math.max(0, Math.floor(container.clientWidth - pad));

        // A collapsed accordion measures 0. Publishing that would collapse every
        // menu the moment the panel is closed, and it would stay collapsed after
        // reopening, because nothing re-measures on its own. Leave the width
        // alone and let the observer set it when there is something to measure.
        if (width <= 0) return 0;

        toolbar.style.setProperty('--cnpro-menu-w', width + 'px');
        // THE TOOLBAR ITSELF IS CAPPED AT THE CANVAS, and this is not the same
        // statement as the menus being. The button row is `flex-wrap: wrap` and
        // the toolbar is `width: max-content`, which grants the row whatever it
        // asks for - so the wrap never fires and a row wider than the canvas is
        // simply CLIPPED by .forge-image-container's overflow: hidden. Measured
        // in the running app: a 546px row on a 437px canvas, with the last five
        // buttons (G/C/M/F included) outside the clip - present in the DOM, with
        // a healthy bounding box, and unreachable. Capping the toolbar is what
        // makes the wrap that was declared all along actually happen.
        //
        // Inline, in px, for the same reason the menu width is: a percentage
        // resolves against an indefinite containing block in gradio.
        toolbar.style.maxWidth = width + 'px';
        toolbar.querySelectorAll('.forge-toolbar-box-c').forEach((box) => {
            box.style.width = width + 'px';
            // Any leftovers from the previous contract would still be in force:
            // an inline min-width of the old button-row measurement pins a menu
            // exactly as effectively as an inline width did.
            box.style.minWidth = '';
            box.style.maxWidth = '';
        });
        return width;
    }

    // Recompute the label sizers of every menu under `root` from the rows that
    // are THERE, and give every label in a menu the same set.
    //
    // The renderer already does this for the rows it renders. This exists for
    // the rows it does not: weight_mask.js injects the feather slider into the
    // weight-mask menu at attach time, because the painter owns it. Such a row
    // sets its own `data-label-max` and calls this, and the column widens to
    // include it -- rather than the injector having to know what the other rows
    // in that menu are, which is the kind of second list this registry exists
    // to abolish.
    //
    // Idempotent: it reads only `data-label-max` (a per-row constant) and
    // writes only `data-max`, so calling it again after the values have changed
    // cannot ratchet the column.
    function syncLabelSizers(root) {
        const scope = root || document;
        if (!scope.querySelectorAll) return 0;
        let menus = 0;
        const boxes = scope.classList && scope.classList.contains('forge-toolbar-box-c')
            ? [scope] : [...scope.querySelectorAll('.forge-toolbar-box-c')];
        boxes.forEach((box) => {
            const labels = [...box.querySelectorAll('.forge-range-row .forge-toolbar-label')];
            if (!labels.length) return;
            const candidates = labels.map((l) => l.dataset.labelMax || l.textContent || '');
            const seen = [];
            candidates.forEach((s) => {
                if (s && seen.indexOf(s) === -1) seen.push(s);
            });
            if (!seen.length) return;
            const value = seen.join(SIZER_SEP);
            labels.forEach((l) => { l.dataset.max = value; });
            menus++;
        });
        return menus;
    }

    // Re-measure whenever the CANVAS changes size: the accordion opens, the
    // panel is resized, the browser window changes, Maximize is pressed.
    //
    // The container is observed, not the button row. Under the previous contract
    // the width came from the buttons, so that is what was watched; now it comes
    // from the canvas, and watching the old thing would have left every menu at
    // whatever width the canvas happened to have at attach time — stale in
    // exactly the case (a collapsed accordion measuring 0) the retry exists for.
    //
    // Installed once per toolbar; ResizeObserver fires on the initial observe,
    // so this also covers the first real layout.
    function watchMenuWidth(uuid) {
        const toolbar = document.getElementById('toolbar_' + uuid);
        const container = document.getElementById('imageContainer_' + uuid);
        if (!toolbar || !container || toolbar.dataset.cnproWidthWatch === '1') return;
        toolbar.dataset.cnproWidthWatch = '1';
        syncMenuWidth(uuid);
        if (typeof ResizeObserver !== 'function') return; // measured once; no live updates
        try {
            new ResizeObserver(() => syncMenuWidth(uuid)).observe(container);
        } catch (e) {
            console.warn('[cnpro] menu width observer failed; menus keep their ' +
                         'width from attach time', e);
        }
    }

    // ------------------------------------------------------- wheel on sliders
    //
    // One notch = one step, on every slider in every tool menu. The tracks are
    // 100px, so dragging gives roughly two units per pixel on a 0..255 range;
    // the wheel is what makes them precise, and it is why the track can be short.
    //
    // THE STEP IS THE SLIDER'S OWN. Read from the `step` attribute the registry
    // already declares, defaulting to 1. So 0..100 and 0..255 move by 1 (what
    // was asked), rotation by its 0.1, gamma by its 5, the weight mask by its
    // 0.01 — each slider stays consistent with what dragging it does, and a new
    // tool needs no wheel code of its own.
    //
    // TWO THINGS MAKE THIS NON-TRIVIAL, both about who else hears the event:
    //
    //   * The host zooms the canvas on ANY wheel over the container
    //     (canvas.js: `container.addEventListener("wheel", ...)` with an
    //     unconditional preventDefault), and the toolbar is INSIDE that
    //     container. Without stopPropagation, adjusting a slider would also
    //     zoom the image behind it. CNPro's own layers-tool wheel handler
    //     already guards this way (`if (toolbar.contains(e.target)) return`),
    //     so the precedent exists; this is the other half of it.
    //   * The listener therefore goes on the TOOLBAR, not the document: it must
    //     run before the container's, and bubbling gets that for free.
    //
    // Delegated, so it also covers sliders injected later by weight_mask.js
    // (the feather row) without that module knowing anything about it.
    // `passive: false` because a passive listener may not preventDefault.
    function stepOf(input) {
        const raw = (input.step || '').trim();
        if (!raw || raw === 'any') return 1;
        const n = parseFloat(raw);
        return Number.isFinite(n) && n > 0 ? n : 1;
    }

    // Snap to the step grid and round off binary-float dust: 0.1 * 3 is
    // 0.30000000000000004, and an input whose value is that string shows
    // "0.30000000000000004" in any label built from it.
    function quantize(value, step, min) {
        const decimals = (String(step).split('.')[1] || '').length;
        const snapped = min + Math.round((value - min) / step) * step;
        return decimals ? Number(snapped.toFixed(decimals)) : Math.round(snapped);
    }

    function wireWheel(uuid) {
        const toolbar = document.getElementById('toolbar_' + uuid);
        if (!toolbar || toolbar.dataset.cnproWheel === '1') return;
        toolbar.dataset.cnproWheel = '1';
        toolbar.addEventListener('wheel', function (e) {
            // Number inputs are in scope too - the layer list's opacity
            // spinners are one per layer, and a wheel that misses them lands on
            // the container underneath and zooms the picture instead. A browser
            // does scroll a focused number input natively, but only while it
            // HAS focus, which is not the affordance the sliders here give.
            const input = e.target && e.target.closest &&
                          e.target.closest('input[type="range"], input[type="number"]');
            // Menus only. The host's own scribble sliders in box-b are its
            // business, and it already has key+wheel shortcuts for them.
            if (!input || !input.closest('.forge-toolbar-box-c')) {
                // A wheel over the LAYER LIST between its inputs is a scroll:
                // the list overflows past a few layers, and the browser's own
                // scrolling does the work - it just must not ALSO reach the
                // host's zoom handler on the container. So: stop, don't
                // preventDefault. A list short enough not to scroll keeps the
                // fall-through every other menu surface has.
                const list = e.target && e.target.closest &&
                             e.target.closest('.forge-layer-list');
                if (list && list.scrollHeight > list.clientHeight) e.stopPropagation();
                return;
            }
            if (input.disabled) return;

            e.preventDefault();
            e.stopPropagation(); // ...or the canvas zooms behind the menu

            const step = stepOf(input);
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);
            const lo = Number.isFinite(min) ? min : 0;
            const hi = Number.isFinite(max) ? max : 100;
            const dir = e.deltaY < 0 ? 1 : -1; // wheel up = increase
            const next = Math.min(hi, Math.max(lo,
                quantize(parseFloat(input.value) + dir * step, step, lo)));
            if (next === parseFloat(input.value)) return;

            input.value = String(next);
            // Both events, in this order, because the wiring is split: most
            // handlers listen on `input` (live redraw) and a few on `change`
            // (weight_mask.js's feather commits there). Dispatching only one
            // leaves whichever half listens to the other silently dead.
            input.dispatchEvent(new Event('input', {bubbles: true}));
            input.dispatchEvent(new Event('change', {bubbles: true}));
        }, {passive: false});
    }

    // ---------------------------------------------------------------- reveal
    //
    // Reveal every injected toolbar button that is not deliberately deferred.
    //
    // THE DEFAULT IS INVERTED ON PURPOSE. Visibility is not opt-in through a CSS
    // class kept in sync by hand across two files -- that is exactly how the four
    // weight-mask buttons stayed invisible for months. Every button in the
    // registry is revealed unless its descriptor states a `deferred` reason, so
    // adding a tool makes it VISIBLE by default. A wrongly-visible button is
    // noticed in seconds; a wrongly-hidden one is invisible by definition.
    //
    // Idempotent; safe to call more than once per container.
    function revealToolbar(uuid) {
        const deferred = deferredMap();
        const scopes = scopeMap();
        let shown = 0;
        toolbarIds().forEach((base) => {
            if (Object.prototype.hasOwnProperty.call(deferred, base)) return;
            const node = document.getElementById(base + '_' + uuid);
            if (node) {
                // scoped buttons stay hidden off their home containers (the
                // weight-mask slots on the host's own img2img/inpaint
                // canvases: injected there, but wired by nothing)
                if (outOfScope(node, base, scopes)) return;
                node.style.display = '';
                shown++;
            }
        });
        // Revealing changes how wide the button row is, so the menus have to be
        // told. Doing it here rather than only in attach() means the Topaz probe
        // -- which reveals three more buttons long after attach -- is covered by
        // the same call it already makes.
        watchMenuWidth(uuid);
        syncMenuWidth(uuid);
        wireWheel(uuid);
        return shown;
    }

    // ----------------------------------------------------------------- audit
    //
    // Compare the live DOM against the registry. Returns [] when everything
    // declared is present and in its expected state.
    //
    // VISIBILITY IS TESTED ON THE ELEMENT'S OWN COMPUTED DISPLAY, not on
    // `offsetParent`. The first version used `offsetParent === null`, reasoning
    // that it catches an element hidden by ANY ancestor. It does -- and that is
    // precisely why it is wrong here: CNPro's canvas lives inside a gradio
    // accordion that is CLOSED by default, so at attach time every control has a
    // null offsetParent and the audit reported all of them broken on every
    // attach. A check that fires when nothing is wrong is worse than no check:
    // it trains you to ignore it, and the one real failure arrives buried in a
    // wall of false ones.
    //
    // getComputedStyle sees CSS rules (including `display: none !important`,
    // which an inline `style.display = ''` cannot override -- style.css does
    // exactly that to hide the tool chrome inside `.cnet-output-mask-group`)
    // while staying blind to a merely-collapsed ancestor. That is the question
    // worth asking: "is THIS control hidden", not "is it on screen right now".
    function isHidden(node) {
        if (node.style.display === 'none') return true;
        const view = node.ownerDocument && node.ownerDocument.defaultView;
        if (!view || !view.getComputedStyle) return false;
        try {
            return view.getComputedStyle(node).display === 'none';
        } catch (e) {
            return false;
        }
    }

    function audit(uuid) {
        const problems = [];
        if (!tools()) {
            problems.push('canvas_tools.js is not loaded - nothing was injected');
            return problems;
        }

        // The Output-mask canvas is DESIGNED to have no tool chrome: style.css
        // hides the adjustment and weight-mask controls inside
        // .cnet-output-mask-group on purpose. Auditing visibility there produced
        // 13 console errors per output-mask canvas, on every attach, for every
        // unit and both tabs - measured live: ten canvases' worth of false
        // alarms burying any real one.
        //
        // Presence is still checked (a missing NODE is a bug anywhere); only the
        // visibility half is skipped, because in this context invisible is the
        // correct answer.
        const container = document.getElementById('imageContainer_' + uuid);
        const chromeSuppressed = !!(container && container.closest &&
                                    container.closest('.cnet-output-mask-group'));

        const deferred = deferredMap();
        const buttons = toolbarIds();

        // 1. every id the registry declares must be in the DOM. This covers menu
        //    rows and overlays too, not just buttons - a slider that never got
        //    injected is as dead as a button that did not.
        ownedIds().forEach((base) => {
            if (!document.getElementById(base + '_' + uuid)) {
                problems.push(base + ': declared in the registry but absent from the DOM');
            }
        });

        // 2. every non-deferred BUTTON must be visible. Scoped buttons
        //    (registry `scope`) are exempt off their home container for the
        //    same reason the output-mask chrome is: there, hidden is the
        //    correct answer, and auditing it produced false alarms on every
        //    host canvas. Presence was still checked above.
        if (!chromeSuppressed) {
            const scopes = scopeMap();
            buttons.forEach((base) => {
                if (Object.prototype.hasOwnProperty.call(deferred, base)) return;
                const node = document.getElementById(base + '_' + uuid);
                if (node && outOfScope(node, base, scopes)) return;
                if (node && isHidden(node)) {
                    problems.push(base + ': injected and wired but not visible in the toolbar');
                }
            });
        }

        // 3. a deferred reason for a button that no longer exists is a stale
        //    exemption - it would silently excuse nothing, or worse, the wrong
        //    thing after a rename.
        Object.keys(deferred).forEach((base) => {
            if (buttons.indexOf(base) === -1) {
                problems.push(base + ': marked deferred but is not a toolbar button');
            }
        });

        return problems;
    }

    // ------------------------------------------------------------ self-check
    //
    // Runs on first use rather than at load (the registry is not there yet at
    // load - see the load-order note at the top). Costs microseconds, runs once
    // per page, and turns a malformed registry into one named console error
    // instead of a blank toolbar with no explanation.
    //
    // It checks the two things the renderer cannot recover from: an empty
    // registry, and DUPLICATE ids. Duplicates matter more than they look --
    // getElementById returns the first match, so a duplicated id means
    // canvas_extra.js wires one node and the user clicks the other, which
    // presents as "this button does nothing" and is near-impossible to find by
    // reading either file.
    let selfChecked = null;
    function selfCheck() {
        if (selfChecked) return selfChecked;
        const problems = [];
        const reg = tools();
        if (!reg) {
            problems.push('canvas_tools.js did not load - the registry is empty');
            return (selfChecked = problems);
        }
        const ids = reg.allIds();
        if (!ids.length) {
            problems.push('the registry declares no ids at all - the toolbar would be blank');
        }
        const seen = Object.create(null);
        ids.forEach((base) => {
            if (!base) {
                problems.push('a descriptor declares an empty id');
                return;
            }
            if (seen[base]) {
                problems.push('duplicate id "' + base + '" - getElementById returns the FIRST ' +
                              'match, so one of the two nodes is wired and the other is dead');
            }
            seen[base] = true;
        });
        reg.TOOLS.forEach((tool, i) => {
            if (!tool.id) problems.push('TOOLS[' + i + '] has no tool id');
            if (!tool.button && !tool.menu && !(tool.overlays || []).length) {
                problems.push('tool "' + (tool.id || i) + '" declares nothing at all');
            }
            // A range row with no labelMax would size its column from the
            // INITIAL label, which is the value at its narrowest -- so the
            // column would then breathe on the first drag, which is the whole
            // bug the sizer exists to prevent. Undeclared must not be able to
            // pass for declared (ARCHITECTURE.md section 8).
            for (const row of rowsOf(tool.menu)) {
                if (row.kind === 'range' && !row.labelMax) {
                    problems.push('range "' + row.id + '" declares no labelMax - its label ' +
                                  'column would be sized from the starting value and would ' +
                                  'breathe as the slider moves');
                }
            }
        });
        if (problems.length) {
            console.error('[cnpro] canvas tool registry self-check FAILED - the toolbar will ' +
                          'be wrong:\n  ' + problems.join('\n  '));
        }
        return (selfChecked = problems);
    }

    window.cnproCanvasNodes = {
        inject: inject,
        revealToolbar: revealToolbar,
        audit: audit,
        selfCheck: selfCheck,
        syncMenuWidth: syncMenuWidth,
        syncLabelSizers: syncLabelSizers,
        wireWheel: wireWheel,
        // Derived views, exposed as FUNCTIONS rather than arrays: the registry
        // is not loaded when this module is, so there is nothing to snapshot at
        // load time. Callers that want a list ask for one.
        ownedIds: ownedIds,
        toolbarIds: toolbarIds,
        deferred: deferredMap,
    };
})();
