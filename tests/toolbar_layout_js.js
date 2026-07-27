// Playwright harness for tests/test_toolbar_layout.py: MEASURE the toolbar in a
// real layout engine, with the real stylesheets, without needing the webui.
//
// WHY THIS EXISTS, AFTER THE OTHER TOOLBAR TESTS
// ----------------------------------------------
// Every earlier layer is blind to geometry:
//
//   test_toolbar_contract.py  runs the renderer on a DOM stub -> ids and reveal
//   test_toolbar_dom.py       jsdom + the real template       -> nodes and display
//   test_style_sheet.py       parses style.css as text        -> rules exist
//
// None of them has a layout engine, so none can answer the question the user
// actually asked: are the buttons SQUARE and ALIGNED, and do the menus fit the
// toolbar? Those are pixel facts. "The rule is in the file" and "the box is
// 24x24" are different claims, and only the second one is the feature.
//
// test_toolbar_live.py does use a real browser -- against a RUNNING webui, which
// makes it a diagnostic rather than something you can run in a loop. This
// harness assembles the same three real inputs by hand:
//
//     modules_forge/forge_canvas/canvas.html   the host's template
//     modules_forge/forge_canvas/canvas.css    the host's stylesheet
//     extensions/.../style.css                 CNPro's, loaded AFTER, as in the app
//
// plus the two real injectors, and measures the result. Hermetic, ~2 seconds, no
// server.
//
// THE STYLESHEET ORDER IS PART OF THE TEST. Extension CSS is injected at the end
// of <body>, after the host's canvas.css in <head>. Loading them the other way
// round would let CNPro's rules win for the wrong reason and hide a specificity
// bug that would bite in the real app.
//
// stdout: JSON.

const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const EXTENSION = path.join(HERE, '..');
const WEBUI = path.join(EXTENSION, '..', '..');
const CANVAS_DIR = path.join(WEBUI, 'modules_forge', 'forge_canvas');

function loadPlaywright() {
    const extra = process.env.CNPRO_TEST_NODE_PATH;
    if (extra) {
        const Module = require('module');
        Module.globalPaths.push(path.join(extra, 'node_modules'));
        process.env.NODE_PATH = (process.env.NODE_PATH ? process.env.NODE_PATH + path.delimiter : '') +
            path.join(extra, 'node_modules');
        Module._initPaths();
    }
    return require('playwright');
}

const UUID = 'lay0ut';

(async () => {
    const out = {console: []};
    let chromium;
    try {
        ({chromium} = loadPlaywright());
    } catch (exc) {
        out.unavailable = 'playwright is not installed: ' + exc.message.split('\n')[0];
        process.stdout.write(JSON.stringify(out));
        return;
    }

    const template = fs.readFileSync(path.join(CANVAS_DIR, 'canvas.html'), 'utf8')
        .split('forge_mixin').join(UUID);
    const hostCss = fs.readFileSync(path.join(CANVAS_DIR, 'canvas.css'), 'utf8');
    const cnproCss = fs.readFileSync(path.join(EXTENSION, 'style.css'), 'utf8');
    const nodesJs = fs.readFileSync(path.join(EXTENSION, 'javascript', 'canvas_nodes.js'), 'utf8');
    const toolsJs = fs.readFileSync(path.join(EXTENSION, 'javascript', 'canvas_tools.js'), 'utf8');

    // .forge-toolbar is `opacity: 0` until hovered and the container is 512px
    // tall; both are the host's doing and neither affects box geometry, but the
    // toolbar must be laid out, so give the page a real viewport.
    const page = await (await chromium.launch()).newPage({viewport: {width: 1280, height: 900}});
    page.on('console', (m) => out.console.push(m.type() + ': ' + m.text()));
    page.on('pageerror', (e) => out.console.push('pageerror: ' + e.message));

    await page.setContent(
        '<!doctype html><html><head><style>' + hostCss + '</style></head><body>' +
        // gradio wraps the canvas template in .prose typography, and .prose is
        // what used to push the last button out of line. Reproduce the two rules
        // that matter rather than pretending the widget lives in a bare page.
        '<style>.prose :where(*):not(:last-child){margin-bottom:1em}' +
        '.prose :where(button){margin-bottom:0.5em}</style>' +
        '<div class="prose" style="width: 900px">' + template + '</div>' +
        '<style>' + cnproCss + '</style>' +
        '</body></html>', {waitUntil: 'load'});

    // Renderer first, registry second: filename order, as the browser loads them.
    await page.addScriptTag({content: nodesJs});
    await page.addScriptTag({content: toolsJs});

    out.inject = await page.evaluate((uuid) => {
        const ok = window.cnproCanvasNodes.inject(uuid);
        const shown = window.cnproCanvasNodes.revealToolbar(uuid);
        return {ok: ok, shown: shown, audit: window.cnproCanvasNodes.audit(uuid)};
    }, UUID);

    // ---- the measurements -----------------------------------------------------
    out.geometry = await page.evaluate((uuid) => {
        const container = document.getElementById('imageContainer_' + uuid);
        const toolbar = document.getElementById('toolbar_' + uuid);
        const boxA = toolbar.querySelector('.forge-toolbar-box-a');

        const rect = (n) => {
            const r = n.getBoundingClientRect();
            return {x: +r.x.toFixed(2), y: +r.y.toFixed(2),
                    w: +r.width.toFixed(2), h: +r.height.toFixed(2)};
        };

        const buttons = [...boxA.querySelectorAll('.forge-btn')].map((b) => ({
            id: b.id.replace('_' + uuid, ''),
            visible: getComputedStyle(b).display !== 'none',
            groupStart: b.classList.contains('forge-adjust-gap'),
            marginLeft: getComputedStyle(b).marginLeft,
            ...rect(b),
        })).filter((b) => b.visible);

        // Menus are measured OPEN. A menu is display:none until its tool is
        // clicked, and a hidden element has no box -- measuring it closed would
        // report 0x0 for everything and call it a pass.
        const menus = [];
        document.querySelectorAll('#toolbar_' + uuid + ' .forge-toolbar-box-c').forEach((box) => {
            const prev = box.style.display;
            box.style.display = '';
            const r = rect(box);
            // Rows the MENU lays out. A row inside a group() is stacked on
            // purpose (the pen menu's three sliders form one column beside the
            // picker), so counting it against the one-row-per-line check would
            // report a deliberate layout as the collapse bug. The group itself
            // is one item of the menu and is counted as such.
            const rows = [...box.querySelectorAll('.forge-range-row')]
                .filter((n) => !n.closest('.forge-row-column'))
                .map(rect);
            const groupedRows = box.querySelectorAll(
                '.forge-row-column .forge-range-row').length;
            // How many distinct lines the rows occupy. A menu whose rows each
            // get their own line IS the vertical stack, even when every
            // individual row is correctly laid out `label | slider`.
            //
            // CLUSTERED, not `new Set(round(y))`. Rows on the SAME flex line can
            // differ in height -- the weight-mask slider has a taller thumb --
            // and `align-items: center` then gives them different tops. Counting
            // exact tops reported the two-row wmask menu as two lines when it is
            // visibly one, i.e. the check cried wolf on correct output.
            const countLines = (boxes) => {
                const ys = boxes.map((x) => x.y).sort((a, b) => a - b);
                let n = 0;
                let top = -1e9;
                for (const t of ys) {
                    if (t - top > 6) { n++; top = t; }
                }
                return n;
            };
            const lineCount = countLines(rows);
            // EVERY direct child, not just the slider rows. Counting rows alone
            // cannot see a reset button pushed onto a line of its own -- which
            // is the single most-reported symptom here, and it was invisible to
            // this harness while the row count stayed at 1.
            const childLineCount = countLines([...box.children].map(rect));
            const widestRow = rows.length ? Math.max(...rows.map((x) => x.w)) : 0;
            menus.push({
                id: box.id.replace('_' + uuid, ''),
                ...r,
                rowCount: rows.length,
                groupedRows: groupedRows,
                lineCount: lineCount,
                childLineCount: childLineCount,
                widestRow: widestRow,
                // could two rows sit side by side at this width? Only then is
                // "one row per line" evidence of anything.
                fitsTwoPerLine: rows.length >= 2 && (widestRow * 2 + 10) <= r.w,
                // a row that is TALLER than one line means the label is stacked
                // above the slider again
                tallestRow: rows.length ? Math.max(...rows.map((x) => x.h)) : 0,
                overflowsContainer: r.w > container.getBoundingClientRect().width + 0.5,
            });
            box.style.display = prev;
        });

        return {
            container: rect(container),
            toolbar: rect(toolbar),
            boxA: rect(boxA),
            boxAScrollWidth: boxA.scrollWidth,
            // what the contract says a menu must be: the canvas minus the
            // toolbar's own horizontal padding
            usableWidth: (() => {
                const cs = getComputedStyle(toolbar);
                const pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
                return Math.floor(container.clientWidth - pad);
            })(),
            // the buttons' own extent, which is what syncMenuWidth publishes and
            // the only measure that cannot ratchet when a wide menu stretches
            // the row it is compared against
            buttonRowExtent: (() => {
                const left = boxA.getBoundingClientRect().left;
                let right = 0;
                boxA.querySelectorAll('.forge-btn').forEach((b) => {
                    if (getComputedStyle(b).display === 'none') return;
                    right = Math.max(right, b.getBoundingClientRect().right);
                });
                return Math.ceil(right - left);
            })(),
            menuWidthVar: getComputedStyle(toolbar).getPropertyValue('--cnpro-menu-w').trim(),
            buttons: buttons,
            menus: menus,
            sliderWidths: (() => {
                const out = [];
                document.querySelectorAll('#toolbar_' + uuid + ' .forge-toolbar-box-c')
                    .forEach((box) => {
                        const prev = box.style.display;
                        box.style.display = '';
                        box.querySelectorAll('input[type="range"]').forEach((i) => {
                            out.push({id: i.id.replace('_' + uuid, ''),
                                      w: +i.getBoundingClientRect().width.toFixed(2)});
                        });
                        box.style.display = prev;
                    });
                return out;
            })(),
        };
    }, UUID);

    // ---- the label column -----------------------------------------------------
    //
    // The column is sized from `data-max`, the set of widest texts the menu's
    // rows can produce. Two things have to be true and neither is visible in the
    // source:
    //
    //   * the column is EXACTLY the widest of them -- if it is wider, the menu
    //     is paying for a label it does not have (the old one-size-for-the-whole-
    //     toolbar column); if it is narrower, the widest value gets an ellipsis;
    //   * it does NOT MOVE when the label is rewritten. Every label ends in a
    //     live value, so a column that sizes to the current text re-flows the
    //     menu mid-drag. Measured by actually writing each candidate into the
    //     label and re-measuring, which is the only way to see it.
    out.labels = await page.evaluate((uuid) => {
        const menus = [];
        document.querySelectorAll('#toolbar_' + uuid + ' .forge-toolbar-box-c').forEach((box) => {
            const prev = box.style.display;
            box.style.display = '';
            const labels = [...box.querySelectorAll('.forge-range-row .forge-toolbar-label')];
            if (!labels.length) { box.style.display = prev; return; }

            // An independent measurement of each candidate IN THE LABEL'S OWN
            // FONT. The probe is a CLONE of a real label rather than a fresh
            // span: a span inherits the row's 16px while the label's own rule
            // sets 12px, so a hand-built probe reports every candidate 33% too
            // wide and this whole check becomes noise. Cloning takes the font
            // from the same cascade the real thing resolves, whatever it is.
            // data-max is dropped so the clone has no sizer of its own, and the
            // inline styles beat the stylesheet's fallback width.
            const probe = labels[0].cloneNode(false);
            probe.removeAttribute('id');
            probe.removeAttribute('data-max');
            probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre;' +
                                  'display:block;width:auto;max-width:none;flex:none;' +
                                  'overflow:visible';
            labels[0].parentNode.appendChild(probe);
            const measure = (text) => {
                probe.textContent = text;
                return +probe.getBoundingClientRect().width.toFixed(2);
            };

            const candidates = [];
            labels.forEach((l) => {
                (l.dataset.max || '').split('\n').forEach((c) => {
                    if (c && candidates.indexOf(c) === -1) candidates.push(c);
                });
            });
            const widest = candidates.length ? Math.max(...candidates.map(measure)) : 0;

            const rows = labels.map((l) => {
                const w = () => +l.getBoundingClientRect().width.toFixed(2);
                const before = w();
                const original = l.textContent;
                // every candidate in turn, including this row's own longest
                const seen = [];
                (l.dataset.max || '').split('\n').forEach((c) => {
                    if (!c) return;
                    l.textContent = c;
                    seen.push(w());
                });
                l.textContent = '';
                const empty = w();
                l.textContent = original;
                return {
                    id: l.id.replace('_' + uuid, ''),
                    own: l.dataset.labelMax || null,
                    hasSizer: !!l.dataset.max,
                    width: before,
                    widthsWhileWriting: seen,
                    widthWhenEmpty: empty,
                    // the live text must fit: no ellipsis at any declared value
                    clipped: seen.some((_, i) => {
                        l.textContent = (l.dataset.max || '').split('\n')[i] || '';
                        const over = l.scrollWidth > l.clientWidth + 0.5;
                        l.textContent = original;
                        return over;
                    }),
                };
            });
            probe.remove();

            menus.push({
                id: box.id.replace('_' + uuid, ''),
                candidates: candidates,
                widestCandidate: widest,
                columnWidth: labels[0].getBoundingClientRect().width,
                rows: rows,
            });
            box.style.display = prev;
        });
        return menus;
    }, UUID);

    // ---- a row injected AFTER the render --------------------------------------
    //
    // weight_mask.js builds the feather row itself and inserts it into the
    // weight-mask menu at attach time, so that row is not in the registry and
    // gets no sizer from the renderer. It joins the column by declaring its own
    // `data-label-max` and calling syncLabelSizers. If that path is broken the
    // menu looks fine here and the feather label is the one that clips in the
    // running app, which is exactly the class of miss this suite exists for.
    //
    // Reproduced with a row wider than anything the menu declares, so a column
    // that did not widen is unmistakable.
    out.lateRow = await page.evaluate((uuid) => {
        const box = document.getElementById('wmaskBox_' + uuid);
        if (!box) return {error: 'wmaskBox is missing'};
        const prev = box.style.display;
        box.style.display = '';
        const widthsBefore = [...box.querySelectorAll('.forge-range-row .forge-toolbar-label')]
            .map((l) => +l.getBoundingClientRect().width.toFixed(2));

        const row = document.createElement('div');
        row.className = 'forge-range-row';
        const label = document.createElement('div');
        label.className = 'forge-toolbar-label';
        label.textContent = 'feather 0';
        label.dataset.labelMax = 'feather 100 xxxxxx';   // deliberately the widest
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'forge-toolbar-range';
        row.appendChild(label);
        row.appendChild(slider);
        box.appendChild(row);

        const synced = window.cnproCanvasNodes.syncLabelSizers(box);
        const labels = [...box.querySelectorAll('.forge-range-row .forge-toolbar-label')];
        const widthsAfter = labels.map((l) => +l.getBoundingClientRect().width.toFixed(2));
        const sizers = labels.map((l) => l.dataset.max || '');

        // idempotent: calling it again must not ratchet the column
        window.cnproCanvasNodes.syncLabelSizers(box);
        const widthsTwice = labels.map((l) => +l.getBoundingClientRect().width.toFixed(2));

        row.remove();
        window.cnproCanvasNodes.syncLabelSizers(box);
        const widthsAfterRemoval = [...box.querySelectorAll('.forge-range-row .forge-toolbar-label')]
            .map((l) => +l.getBoundingClientRect().width.toFixed(2));
        box.style.display = prev;
        return {
            synced: synced,
            widthsBefore: widthsBefore,
            widthsAfter: widthsAfter,
            widthsTwice: widthsTwice,
            widthsAfterRemoval: widthsAfterRemoval,
            allSizersEqual: sizers.every((s) => s === sizers[0]),
            sizerIncludesLateRow: sizers.every((s) => s.indexOf('feather 100 xxxxxx') !== -1),
        };
    }, UUID);

    // ---- A CANVAS TOO NARROW FOR THE BUTTON ROW --------------------------------
    //
    // The measurement above runs at 900px, where everything fits and nothing is
    // proved. A ControlNet unit in the real two-column layout gives the canvas
    // about 437px, and the button row is 546px. That is not "overflow": the
    // canvas is `overflow: hidden`, so the last five buttons - G/C/M/F among
    // them - are GONE. Unclickable, unhoverable, and invisible to every check
    // this suite had, because a clipped element still reports a perfectly
    // healthy bounding box. It shipped, and it was found by screenshotting the
    // running app.
    //
    // So the narrow case is measured, and the question asked is the one that
    // matters: is every button INSIDE the canvas rectangle.
    out.narrow = await page.evaluate((uuid) => {
        const container = document.getElementById('imageContainer_' + uuid);
        const toolbar = document.getElementById('toolbar_' + uuid);
        const boxA = toolbar.querySelector('.forge-toolbar-box-a');
        const host = container.closest('.prose') || container.parentElement;

        const measureAt = (width) => {
            host.style.width = width + 'px';
            // re-measure exactly as the ResizeObserver would
            window.cnproCanvasNodes.syncMenuWidth(uuid);
            const cr = container.getBoundingClientRect();
            const buttons = [...boxA.querySelectorAll('.forge-btn')]
                .filter((b) => getComputedStyle(b).display !== 'none');
            const outside = buttons.filter((b) => {
                const r = b.getBoundingClientRect();
                return r.right > cr.right + 0.5 || r.left < cr.left - 0.5 ||
                       r.bottom > cr.bottom + 0.5;
            }).map((b) => b.id.replace('_' + uuid, ''));
            const lines = new Set(buttons.map(
                (b) => Math.round(b.getBoundingClientRect().y))).size;
            const menus = [];
            toolbar.querySelectorAll('.forge-toolbar-box-c').forEach((box) => {
                const prev = box.style.display;
                box.style.display = '';
                const r = box.getBoundingClientRect();
                menus.push({
                    id: box.id.replace('_' + uuid, ''),
                    w: +r.width.toFixed(1),
                    outside: r.right > cr.right + 0.5 || r.left < cr.left - 0.5,
                });
                box.style.display = prev;
            });
            return {
                width: width,
                canvasW: +cr.width.toFixed(1),
                toolbarW: +toolbar.getBoundingClientRect().width.toFixed(1),
                rowW: +boxA.getBoundingClientRect().width.toFixed(1),
                rowScrollW: boxA.scrollWidth,
                containerScrollW: container.scrollWidth,
                containerClientW: container.clientWidth,
                buttons: buttons.length,
                buttonLines: lines,
                outsideTheClip: outside,
                menus: menus,
            };
        };

        const results = [900, 500, 437, 320].map(measureAt);
        host.style.width = '900px';
        window.cnproCanvasNodes.syncMenuWidth(uuid);
        return results;
    }, UUID);

    // ---- the wheel contract ---------------------------------------------------
    //
    // Driven for real: a synthetic `wheel` event on each slider, then read the
    // value back. Source inspection cannot tell you that the step was applied,
    // that it clamped at the ends, or -- the one that actually bites -- that the
    // event did not also reach the host's container listener and zoom the canvas.
    out.wheel = await page.evaluate((uuid) => {
        const results = [];
        const container = document.getElementById('imageContainer_' + uuid);

        // Stand in for the host's canvas zoom: canvas.js attaches exactly this,
        // unconditionally, on the container the toolbar lives inside.
        let zoomed = 0;
        container.addEventListener('wheel', () => { zoomed++; });

        document.querySelectorAll('#toolbar_' + uuid + ' .forge-toolbar-box-c').forEach((box) => {
            box.style.display = '';
        });

        document.querySelectorAll('#toolbar_' + uuid + ' .forge-toolbar-box-c input[type="range"]')
            .forEach((input) => {
                const step = parseFloat(input.step) || 1;
                const min = parseFloat(input.min);
                const max = parseFloat(input.max);
                const start = (min + max) / 2;
                input.value = String(start);
                const from = parseFloat(input.value);

                let inputEvents = 0;
                input.addEventListener('input', () => { inputEvents++; });

                const fire = (deltaY) => input.dispatchEvent(
                    new WheelEvent('wheel', {deltaY: deltaY, bubbles: true, cancelable: true}));

                fire(-100);                       // wheel up
                const afterUp = parseFloat(input.value);
                fire(100);                        // wheel down
                const afterDown = parseFloat(input.value);

                // clamping: slam to the top, then try to go further
                input.value = String(max);
                fire(-100);
                const atMax = parseFloat(input.value);

                results.push({
                    id: input.id.replace('_' + uuid, ''),
                    step: step, min: min, max: max,
                    up: +(afterUp - from).toFixed(6),
                    down: +(afterDown - afterUp).toFixed(6),
                    clampedAtMax: atMax === max,
                    // a value that stringifies with float dust would render as
                    // "0.30000000000000004" in the label built from it
                    clean: String(afterUp).length <= 12,
                    inputEvents: inputEvents,
                });
            });

        document.querySelectorAll('#toolbar_' + uuid + ' .forge-toolbar-box-c').forEach((box) => {
            box.style.display = 'none';
        });
        return {sliders: results, zoomEventsLeaked: zoomed};
    }, UUID);

    await page.context().browser().close();
    process.stdout.write(JSON.stringify(out));
})().catch((exc) => {
    process.stdout.write(JSON.stringify({fatal: String((exc && exc.stack) || exc)}));
});
