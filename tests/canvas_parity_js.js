// Harness for tests/test_canvas_parity.py -- asserts the CANVAS / CONTROL
// PARITY CONTRACT stated at the top of javascript/canvas_extra.js:
//
//     the raster the control receives is the flattened canvas the user sees.
//
// Both sides are read from the page and decoded, then compared PIXEL BY PIXEL:
//
//   displayed  =  image_<uuid>.src            (what is on screen)
//   control    =  the logical_image_background textarea  (what gradio holds,
//                                                         i.e. what generates)
//
// WHY A REAL BROWSER
// ------------------
// The contract is about pixels. jsdom has no canvas, no image decoder and no
// compositor, so it cannot compare the two sides at all - it can only check
// that both are non-empty, which is what "looks fine" already did. This runs
// the host's real canvas.html/canvas.js plus every file in javascript/ in
// headless Chrome, so `toDataURL`, `drawImage`, blend modes and the <img>
// read-back inside updateBackgroundImageData are the genuine ones.
//
// No webui instance is involved: the page is assembled from the two real
// sources plus a stand-in for the gradio DOM (the ids and classes the modules
// look up). That keeps it hermetic and a few seconds long.
//
// Requires puppeteer-core and a Chrome/Edge binary; neither is vendored. The
// python wrapper SKIPS LOUDLY when they are missing rather than passing quietly.
//
// stdout: JSON.

const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const EXT = path.dirname(HERE);
const ROOT = path.dirname(path.dirname(EXT));               // .../sd-webui-forge-neo
const FC = path.join(ROOT, 'modules_forge', 'forge_canvas');

const CHROME_CANDIDATES = [
    process.env.CNPRO_CHROME,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
].filter(Boolean);

const read = (p) => fs.readFileSync(p, 'utf8');
const UUID = 'uuid_parity0';

function buildPage() {
    const tpl = read(path.join(FC, 'canvas.html')).replace(/forge_mixin/g, UUID);
    const canvasJs = read(path.join(FC, 'canvas.js'));
    const canvasCss = read(path.join(FC, 'canvas.css'));
    const extCss = read(path.join(EXT, 'style.css'));
    const files = fs.readdirSync(path.join(EXT, 'javascript'))
        .filter((f) => f.endsWith('.js')).sort();          // page load order
    const cnpro = files.map((f) => read(path.join(EXT, 'javascript', f)));
    const wmask = ['global', 'coarse', 'mid', 'fine']
        .map((b) => `<div class="cnet-wmask-0-${b}-state"><label><textarea></textarea></label></div>`).join('');

    return {files, html: `
<!doctype html><html><head><meta charset="utf-8">
<style>${canvasCss}</style><style>${extCss}</style></head><body>
<div id="controlnet" class="input-accordion"><div class="cnet-unit-tab">
  <div class="cnet-input-image-group cnet-input-slot-0">${tpl}</div>
  <div id="${UUID}" class="logical_image_foreground"><label><textarea></textarea></label></div>
  <div id="${UUID}" class="logical_image_background"><label><textarea></textarea></label></div>
  ${wmask}
</div></div>
<script>
  var gradio_config = {version: "4.40.0"};
  function updateInput(t) { t.dispatchEvent(new Event("input", {bubbles: true})); }
  function gradioApp() { return document; }
  window.__uiUpdate = [];
  function onUiUpdate(f) { window.__uiUpdate.push(f); }
  function onAfterUiUpdate(f) { window.__uiUpdate.push(f); }
  function onUiLoaded(f) { try { f(); } catch (e) {} }
  window.__logs = [];
  for (const k of ["warn", "error"]) {
      const o = console[k].bind(console);
      console[k] = (...a) => { window.__logs.push(k + ": " + a.map(String).join(" ")); o(...a); };
  }
<\/script>
<script>${canvasJs}<\/script>
${cnpro.map((s) => '<script>' + s + '<\/script>').join('\n')}
</body></html>`};
}

// Each case leaves the canvas in a state and the contract must hold in it.
const CASES = [
    'single-layer',
    'single-layer-after-drop',
    'global-gamma',
    'global-grayscale-invert',
    'geometry-flip-rotate',
    'crop-committed',
    'crop-tool-open',
    'two-layers',
    'layer-moved',
    'layer-scaled',
    'layer-rotated',
    'layer-flipped',
    'layer-blend-lighten',
    'layer-partly-offstage',
    'per-layer-gamma',
    'per-layer-invert',
    'pen-strokes',
    'strokes-on-empty-layer',
    'drop-onto-stack',
    'drop-onto-occluded-active-layer',
    'layer-deleted',
    'stack-then-global-adjust',
];

(async () => {
    const out = {cases: {}, logs: []};
    let puppeteer;
    try {
        puppeteer = require('puppeteer-core');
    } catch (e) {
        out.skip = 'puppeteer-core is not installed. npm install --no-save --prefix <dir> puppeteer-core';
        process.stdout.write(JSON.stringify(out));
        return;
    }
    const exe = CHROME_CANDIDATES.find((p) => { try { return fs.existsSync(p); } catch (e) { return false; } });
    if (!exe) {
        out.skip = 'no Chrome/Edge binary found; set CNPRO_CHROME to one';
        process.stdout.write(JSON.stringify(out));
        return;
    }
    out.chrome = exe;

    const {html, files} = buildPage();
    out.modules = files;

    let browser;
    try {
        browser = await puppeteer.launch({executablePath: exe, headless: 'new', args: ['--no-sandbox']});
    } catch (e) {
        out.skip = 'could not launch ' + exe + ': ' + (e && e.message);
        process.stdout.write(JSON.stringify(out));
        return;
    }

    for (const kase of CASES) {
        const page = await browser.newPage();
        await page.setViewport({width: 1400, height: 1000});
        const errs = [];
        page.on('pageerror', (e) => errs.push(e.message));
        try {
            await page.setContent(html, {waitUntil: 'load'});
            out.cases[kase] = await page.evaluate(runCase, UUID, kase);
        } catch (e) {
            out.cases[kase] = {error: String(e && e.message || e)};
        }
        if (errs.length) out.cases[kase].pageErrors = errs;
        await page.close();
    }
    await browser.close();
    process.stdout.write(JSON.stringify(out));
})().catch((e) => {
    process.stdout.write(JSON.stringify({fatal: String(e && e.stack || e)}));
});

// ---------------------------------------------------------------- in-page
async function runCase(UUID, kase) {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const el = (p) => document.getElementById(p + UUID);
    const bgTa = () => [...document.querySelectorAll('.logical_image_background')]
        .find((e) => e.id === UUID).querySelector('textarea');

    // deliberately busy content: flat colour hides compositing and rounding bugs
    const mk = (w, h, seed) => {
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        const x = c.getContext('2d');
        for (let i = 0; i < 120; i++) {
            x.fillStyle = 'rgb(' + (i * seed % 256) + ',' + (i * 7 + seed) % 256 + ',' + (i * 13 + seed * 3) % 256 + ')';
            x.fillRect((i * 31 + seed) % w, (i * 17 + seed) % h, 1 + (i % 23), 1 + (i % 19));
        }
        x.fillStyle = '#000'; x.fillRect(0, 0, 12, 12);
        x.fillStyle = '#fff'; x.fillRect(w - 12, h - 12, 12, 12);
        return c.toDataURL('image/png');
    };
    const decode = (url) => new Promise((ok, no) => {
        const i = new Image(); i.onload = () => ok(i); i.onerror = () => no(new Error('undecodable')); i.src = url;
    });
    const pixels = async (url) => {
        if (!url) return null;
        const img = await decode(url);
        const c = document.createElement('canvas');
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        const x = c.getContext('2d', {willReadFrequently: true});
        x.drawImage(img, 0, 0);
        return {w: c.width, h: c.height, data: x.getImageData(0, 0, c.width, c.height).data};
    };
    const drop = async (url) => {
        const dt = new DataTransfer();
        dt.items.add(new File([await (await fetch(url)).blob()], 'x.png', {type: 'image/png'}));
        el('container_').dispatchEvent(new DragEvent('drop',
            {bubbles: true, cancelable: true, dataTransfer: dt}));
        await sleep(1500);
    };
    const stroke = async (x0, y0, x1, y1) => {
        const cont = el('imageContainer_'), r = cont.getBoundingClientRect();
        const o = (x, y) => ({bubbles: true, cancelable: true, clientX: r.left + x, clientY: r.top + y,
            pointerId: 1, buttons: 1, isPrimary: true, pointerType: 'mouse'});
        cont.dispatchEvent(new PointerEvent('pointerdown', o(x0, y0)));
        cont.dispatchEvent(new PointerEvent('pointermove', o(x1, y1)));
        window.dispatchEvent(new PointerEvent('pointerup', o(x1, y1)));
        await sleep(700);
    };
    // always drive the pipeline through a real control, never by hand
    const nudge = async (value) => {
        const g = el('gammaSlider_');
        g.value = String(value);
        g.dispatchEvent(new Event('input', {bubbles: true}));
        await sleep(1000);
    };

    const fc = new ForgeCanvas(UUID, false, true, false, 300,
        '#000000', false, 25, false, false, 100, false, 0, false);
    (window.__uiUpdate || []).forEach((f) => { try { f(); } catch (e) {} });
    await sleep(400);
    const st = fc.__adjust;
    if (!st) return {error: 'CNPro did not attach to the canvas'};

    fc.loadImage(mk(400, 300, 5));
    await sleep(900);

    const wantsStack = /two-layers|layer-|per-layer|drop-onto|stack-then/.test(kase);
    if (wantsStack || kase === 'strokes-on-empty-layer') {
        el('layersButton_').click();
        await sleep(400);
        if (kase === 'strokes-on-empty-layer') {
            el('layerAddButton_').click();
            await sleep(400);
        } else {
            st.addLayerFromDataUrl(mk(220, 160, 41));
            await sleep(900);
        }
    }
    const top = st.layers[st.layers.length - 1];

    switch (kase) {
        case 'single-layer-after-drop': await drop(mk(320, 240, 77)); break;
        case 'global-gamma':            await nudge(155); break;
        case 'global-grayscale-invert': el('grayscaleButton_').click(); await sleep(500);
                                        el('invertButton_').click(); await sleep(800); break;
        case 'geometry-flip-rotate': {
            el('flipButton_').click(); await sleep(500);
            const r = el('rotationSlider_');
            r.value = '17'; r.dispatchEvent(new Event('input', {bubbles: true}));
            await sleep(1000); break;
        }
        case 'crop-committed':
        case 'crop-tool-open': {
            el('cropButton_').click(); await sleep(400);
            st.crop = {x: 0.12, y: 0.08, w: 0.65, h: 0.7};
            if (st.updateCropOverlay) st.updateCropOverlay();
            await nudge(101);
            if (kase === 'crop-committed') { el('cropButton_').click(); await sleep(1000); }
            break;
        }
        case 'layer-moved':          top.x = 120; top.y = 85; await nudge(102); break;
        case 'layer-scaled':         top.scale = 1.6; await nudge(102); break;
        case 'layer-rotated':        top.rotate = 27; await nudge(102); break;
        case 'layer-flipped':        top.flipH = true; await nudge(102); break;
        case 'layer-blend-lighten':  top.blend = 'lighten'; await nudge(102); break;
        case 'layer-partly-offstage': top.x = -90; top.y = 230; await nudge(102); break;
        case 'per-layer-gamma':      top.gamma = 1.9; top.canvas = null; top.canvasKey = null;
                                     await nudge(102); break;
        case 'per-layer-invert':     top.invert = true; top.canvas = null; top.canvasKey = null;
                                     await nudge(102); break;
        case 'pen-strokes':          el('penButton_').click(); await sleep(300);
                                     await stroke(50, 50, 150, 110);
                                     el('penButton_').click(); await sleep(700); break;
        case 'strokes-on-empty-layer': el('penButton_').click(); await sleep(300);
                                     await stroke(60, 60, 170, 130);
                                     el('penButton_').click(); await sleep(700); break;
        case 'drop-onto-stack':      st.activeLayer = st.layers.length - 1;
                                     await drop(mk(320, 240, 91)); break;
        case 'drop-onto-occluded-active-layer':
                                     st.activeLayer = 0;
                                     el('layersButton_').click(); await sleep(500);
                                     await drop(mk(320, 240, 91)); break;
        case 'layer-deleted':        st.layers.splice(0, 1); st.activeLayer = 0;
                                     await nudge(102); break;
        case 'stack-then-global-adjust': await nudge(168); break;
        case 'two-layers':           await nudge(102); break;
        default:                     await nudge(100); break;
    }
    await sleep(900);

    const res = {case: kase, layers: st.layers.length, mode: st.mode || null};
    let displayed, control;
    try {
        displayed = await pixels(el('image_').src);
        control = await pixels(bgTa().value);
    } catch (e) {
        return Object.assign(res, {error: 'could not decode a side: ' + e.message});
    }
    if (!displayed || !control) {
        return Object.assign(res, {
            error: (!displayed ? 'the canvas shows nothing' : 'the control channel is empty'),
        });
    }

    res.displayed = displayed.w + 'x' + displayed.h;
    res.control = control.w + 'x' + control.h;

    // The crop tool being open is the contract's one stated exception: the
    // display is the full frame under edit, gradio holds the committed crop.
    // Assert the exception itself instead of skipping the case.
    if (st.mode === 'crop') {
        const g = st.crop || {w: 1, h: 1};
        const expW = Math.max(1, Math.round(displayed.w * g.w));
        const expH = Math.max(1, Math.round(displayed.h * g.h));
        res.exception = 'crop tool open: control must hold the committed crop';
        res.expectedControl = expW + 'x' + expH;
        res.match = Math.abs(control.w - expW) <= 1 && Math.abs(control.h - expH) <= 1;
        res.diffPixels = res.match ? 0 : -1;
        res.maxDiff = res.match ? 0 : -1;
        return res;
    }

    if (displayed.w !== control.w || displayed.h !== control.h) {
        res.match = false;
        res.reason = 'dimensions differ';
        return res;
    }
    let diffPixels = 0, maxDiff = 0, firstAt = null;
    for (let i = 0; i < displayed.data.length; i += 4) {
        let d = 0;
        for (let c = 0; c < 4; c++) {
            const delta = Math.abs(displayed.data[i + c] - control.data[i + c]);
            if (delta > d) d = delta;
        }
        if (d) {
            diffPixels++;
            if (d > maxDiff) maxDiff = d;
            if (firstAt === null) {
                const p = i / 4;
                firstAt = (p % displayed.w) + ',' + Math.floor(p / displayed.w);
            }
        }
    }
    res.diffPixels = diffPixels;
    res.maxDiff = maxDiff;
    res.firstDiffAt = firstAt;
    res.totalPixels = displayed.w * displayed.h;
    // PNG both sides, no resampling anywhere: this is exact or it is broken.
    res.match = diffPixels === 0;
    return res;
}
