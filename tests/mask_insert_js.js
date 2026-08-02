// A PAINTED WEIGHT MASK SURVIVES THE IMAGE UNDER IT BEING REPLACED - measured
// on the real ForgeCanvas, with the real painter, through the real insert path.
//
// WHY A BROWSER, AND WHY THIS DEEP
// --------------------------------
// The bug this pins was reported twice about the same action (the below-canvas
// ⤵I / ⤵O buttons wiping a mask that had just been painted) and fixed once in
// between. The fix was correct and the mask still died, because THREE places
// answered "the image changed - does the paint die?" independently and only one
// of them was touched. `tests/test_mask_clear_reasons.py` is the static half and
// forbids that shape in the source; this is the half that drives the feature.
//
// Nothing below a real browser can run it. It needs: a real ForgeCanvas (the
// painter attaches to `imageContainer_<uuid>` / `image_<uuid>` and wraps the
// widget's own upload path), real image decoding (`naturalWidth` is what the
// dims watchdog compares and what `rescaleMask` draws onto), real pointer events
// (the painter intercepts them in the capture phase), and real timers (the 500 ms
// coherence tick). jsdom has none of the four.
//
// THE NEGATIVE CONTROL IS NOT OPTIONAL. "The mask survived" also describes code
// that has stopped clearing anything at all, so the last case removes the image
// entirely and requires the mask to be GONE. Without it this file would pass on
// a painter with its clear path deleted.
//
// stdout: JSON.
const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const EXT = path.dirname(HERE);
// two levels up when this lives in <webui>/extensions/<name>; a standalone clone
// is elsewhere and that guess resolves to a directory that does not exist, which
// used to surface as an ENOENT stack from inside a harness - a crash where the
// honest answer is "the host is not here". CNPRO_WEBUI_DIR overrides it.
const ROOT = process.env.CNPRO_WEBUI_DIR || path.dirname(path.dirname(EXT));
const FC = path.join(ROOT, 'modules_forge', 'forge_canvas');

const UUID = 'uuid_insert0';
const read = (p) => fs.readFileSync(p, 'utf8');

function loadPlaywright() {
    const extra = process.env.CNPRO_TEST_NODE_PATH;
    if (extra) {
        const Module = require('module');
        Module.globalPaths.push(path.join(extra, 'node_modules'));
        process.env.NODE_PATH = (process.env.NODE_PATH ? process.env.NODE_PATH + path.delimiter : '')
            + path.join(extra, 'node_modules');
        Module._initPaths();
    }
    return require('playwright');
}

/** The host ships either the de-obfuscated `canvas.js` (ForgeNeo) or the
 *  minified `canvas.min.js` (lllyasviel Forge). Both define ForgeCanvas and
 *  canvas_adapter.js normalizes the naming difference, which is the whole point
 *  of that file - so either is a valid host to test against. */
function hostCanvasJs() {
    for (const name of ['canvas.js', 'canvas.min.js']) {
        const p = path.join(FC, name);
        if (fs.existsSync(p)) return {name, src: read(p)};
    }
    return null;
}

function buildPage(canvasJs) {
    const tpl = read(path.join(FC, 'canvas.html')).split('forge_mixin').join(UUID);
    const canvasCss = read(path.join(FC, 'canvas.css'));
    const extCss = read(path.join(EXT, 'style.css'));
    const files = fs.readdirSync(path.join(EXT, 'javascript'))
        .filter((f) => f.endsWith('.js')).sort();          // page load order
    const cnpro = files.map((f) => read(path.join(EXT, 'javascript', f)));
    const states = ['global', 'coarse', 'mid', 'fine']
        .map((b) => `<div class="cnet-wmask-0-${b}-state"><label><textarea></textarea></label></div>`)
        .join('');

    return `<!doctype html><html><head><meta charset="utf-8">
<style>${canvasCss}</style><style>${extCss}</style></head><body>
<div id="controlnet" class="input-accordion"><div class="cnet-unit-tab">
  <!-- .cnet-image-tabs IS REQUIRED, and it needs a definite height.
       Required because active_canvas.js skips the whole shared 500 ms tick
       unless a .cnet-image-tabs or .cnet-coverage-panel is laid out, and that
       tick drives the painter's coherence watchdog - without it this harness
       measures a painter with half its machinery switched off.
       Definite height because style.css answers that class with a
       height:100% !important on .forge-container, and a percentage against an
       auto-height parent collapses the container to ZERO. The widget then still
       reports a 512x512 image and the toolbar still attaches; it just has no
       box, so overflow:hidden swallows every pointer event and a stroke paints
       nothing. Both were measured while writing this file. -->
  <div class="cnet-image-tabs" style="height: 560px">
    <div class="cnet-input-image-group cnet-input-slot-0">${tpl}</div>
  </div>
  <div id="${UUID}" class="logical_image_foreground"><label><textarea></textarea></label></div>
  <div id="${UUID}" class="logical_image_background"><label><textarea></textarea></label></div>
  ${states}
</div></div>
<script>
  var gradio_config = {version: "4.40.0"};
  function updateInput(t) { t.dispatchEvent(new Event("input", {bubbles: true})); }
  function gradioApp() { return document; }
  window.__uiUpdate = [];
  function onUiUpdate(f) { window.__uiUpdate.push(f); try { f(); } catch (e) {} }
  function onAfterUiUpdate(f) { window.__uiUpdate.push(f); }
  function onUiLoaded(f) { try { f(); } catch (e) {} }
  window.__pump = function () {
      for (const f of window.__uiUpdate) { try { f(); } catch (e) {} }
  };
  window.__logs = [];
  for (const k of ["warn", "error"]) {
      const orig = console[k].bind(console);
      console[k] = (...a) => { window.__logs.push(k + ": " + a.map(String).join(" ")); orig(...a); };
  }
<\/script>
<script>${canvasJs}<\/script>
${cnpro.map((s) => '<script>' + s + '<\/script>').join('\n')}
</body></html>`;
}

// Solid-colour images, made in the page so they are genuine PNGs through a
// genuine decode. Different SIZES matter (the dims watchdog) and so does a
// different image at the SAME size - that second case is the branch the earlier
// fix left broken, and it is invisible to any dimension check.
const MAKE_IMAGE = `(w, h, color) => {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    ctx.fillStyle = color; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, w >> 2, h >> 2);
    return c.toDataURL('image/png');
}`;

(async () => {
    const out = {console: [], steps: []};
    let chromium;
    try {
        ({chromium} = loadPlaywright());
    } catch (exc) {
        out.unavailable = 'playwright is not installed: ' + exc.message.split('\n')[0];
        process.stdout.write(JSON.stringify(out));
        return;
    }
    if (!fs.existsSync(path.join(FC, 'canvas.html'))) {
        out.unavailable = "the host's canvas.html is not at " + FC + " - this checkout "
            + "is not inside a webui tree. Set CNPRO_WEBUI_DIR to the webui root.";
        process.stdout.write(JSON.stringify(out));
        return;
    }
    const host = hostCanvasJs();
    if (!host) {
        out.unavailable = "neither canvas.js nor canvas.min.js is at " + FC;
        process.stdout.write(JSON.stringify(out));
        return;
    }
    out.host = host.name;

    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({viewport: {width: 1100, height: 900}});
        page.on('console', (m) => {
            if (m.type() === 'error') out.console.push('error: ' + m.text());
        });
        page.on('pageerror', (e) => out.console.push('pageerror: ' + e.message));
        await page.setContent(buildPage(host.src), {waitUntil: 'load'});

        // The template alone is inert - the host's canvas.py emits an inline
        // `new ForgeCanvas(...)` next to it, and without that there is no widget
        // for forgeCanvasPush to find and nothing for the painter to attach to.
        // Same construction as tests/canvas_parity_js.js, for the same reason.
        const attached = await page.evaluate(([uuid]) => {
            const fc = new ForgeCanvas(uuid, false, true, false, 300,
                '#000000', false, 25, false, false, 100, false, 0, false);
            window.__fc = fc;
            (window.__uiUpdate || []).forEach((f) => { try { f(); } catch (e) {} });
            return !!fc;
        }, [UUID]);
        if (!attached) {
            out.fatal = 'ForgeCanvas could not be constructed';
            process.stdout.write(JSON.stringify(out));
            return;
        }
        await page.waitForTimeout(500);
        await page.evaluate(() => window.__pump());
        await page.waitForTimeout(300);
        out.painterAttached = await page.evaluate(([uuid]) => {
            const btn = document.getElementById('wmaskButton_' + uuid);
            return !!(btn && btn.offsetParent !== null);
        }, [UUID]);

        const put = async (w, h, color) => {
            await page.evaluate(([mk, uuid, w, h, color]) => {
                // THE INSERT PATH ITSELF. insert_image.js's ⤵I / ⤵O do exactly
                // this and nothing else: read a source <img>, normalize it to a
                // PNG data url, hand it to forgeCanvasPush.
                window.forgeCanvasPush(uuid, eval('(' + mk + ')')(w, h, color));
            }, [MAKE_IMAGE, UUID, w, h, color]);
            await page.waitForFunction(([uuid, w]) => {
                const img = document.getElementById('image_' + uuid);
                return img && img.naturalWidth === w;
            }, [UUID, w], {timeout: 8000});
            // let the 500 ms coherence tick see the new image at least twice
            await page.waitForTimeout(1300);
        };

        const snapshot = async (label) => {
            const s = await page.evaluate(([uuid]) => {
                const ta = document.querySelector('.cnet-wmask-0-global-state textarea');
                const img = document.getElementById('image_' + uuid);
                const btn = document.getElementById('wmaskButton_' + uuid);
                const value = ta ? ta.value : null;
                return {
                    hasValue: !!(value && value.startsWith('data:image')),
                    valueLen: value ? value.length : 0,
                    marked: !!(btn && btn.classList.contains('forge-tool-modified')),
                    imageDims: img ? img.naturalWidth + 'x' + img.naturalHeight : null,
                    value: value || '',
                };
            }, [UUID]);
            // decode the exported channel so its GEOMETRY can be compared with
            // the image it is supposed to be registered against
            if (s.hasValue) {
                s.maskDims = await page.evaluate((v) => new Promise((res) => {
                    const i = new Image();
                    i.onload = () => res(i.naturalWidth + 'x' + i.naturalHeight);
                    i.onerror = () => res(null);
                    i.src = v;
                }), s.value);
                s.painted = await page.evaluate((v) => new Promise((res) => {
                    const i = new Image();
                    i.onload = () => {
                        const c = document.createElement('canvas');
                        c.width = i.naturalWidth; c.height = i.naturalHeight;
                        const cx = c.getContext('2d');
                        cx.drawImage(i, 0, 0);
                        const d = cx.getImageData(0, 0, c.width, c.height).data;
                        let n = 0;
                        for (let p = 3; p < d.length; p += 4) if (d[p] > 0) n++;
                        res(n / (c.width * c.height));
                    };
                    i.onerror = () => res(-1);
                    i.src = v;
                }), s.value);
            }
            delete s.value;
            s.step = label;
            out.steps.push(s);
            return s;
        };

        // ---- 1. an image, then a painted mask on it
        await put(512, 512, '#3060c0');
        await page.evaluate(([uuid]) => {
            document.getElementById('wmaskButton_' + uuid).click();
        }, [UUID]);
        await page.waitForTimeout(200);
        const rect = await page.evaluate(([uuid]) => {
            const img = document.getElementById('image_' + uuid);
            const cont = document.getElementById('imageContainer_' + uuid);
            const btn = document.getElementById('wmaskButton_' + uuid);
            const r = img.getBoundingClientRect();
            const c = cont.getBoundingClientRect();
            return {
                x: r.left, y: r.top, w: r.width, h: r.height,
                container: {x: c.left, y: c.top, w: c.width, h: c.height},
                armed: !!(btn && btn.classList.contains('forge-tool-active')),
                btnClasses: btn ? btn.className : null,
                picking: cont.classList.contains('forge-picking'),
            };
        }, [UUID]);
        out.arming = rect;
        // a stroke across the middle third, in the image's own box
        await page.mouse.move(rect.x + rect.w * 0.30, rect.y + rect.h * 0.5);
        await page.mouse.down();
        await page.mouse.move(rect.x + rect.w * 0.45, rect.y + rect.h * 0.5, {steps: 6});
        await page.mouse.move(rect.x + rect.w * 0.62, rect.y + rect.h * 0.5, {steps: 6});
        await page.mouse.up();
        await page.waitForTimeout(900);          // debounced export
        out.painted = await snapshot('after painting');

        // ---- 2. THE REPORTED ACTION, at a different size
        await put(640, 384, '#20a060');
        out.afterResize = await snapshot('after inserting 640x384');

        // ---- 3. THE BRANCH THE EARLIER FIX MISSED: a different image at the
        //         SAME size. No dimension moves, so only the upload-sequence
        //         path sees it at all - which is precisely where the surviving
        //         `clearMask` was.
        await put(640, 384, '#c05020');
        out.afterSameSize = await snapshot('after inserting a second 640x384');

        // ---- 4. NEGATIVE CONTROL. No image at all must still end the mask, or
        //         "it survived" is a statement about code that cannot clear.
        await page.evaluate(([uuid]) => {
            document.getElementById('removeButton_' + uuid).click();
        }, [UUID]);
        await page.waitForTimeout(1300);
        out.afterRemove = await snapshot('after removing the image');

        out.logs = await page.evaluate(() => window.__logs);
    } catch (exc) {
        out.fatal = String(exc && exc.stack || exc);
    } finally {
        await browser.close();
    }
    process.stdout.write(JSON.stringify(out));
})().catch((e) => {
    process.stdout.write(JSON.stringify({fatal: String(e && e.stack || e)}));
});
