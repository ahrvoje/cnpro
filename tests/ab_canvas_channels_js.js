// Node + jsdom harness for tests/test_ab_canvas_channels.py: the browser half
// of the A/B panel's Canvas layer row (javascript/cnpro_ab.js).
//
// The layer stack of a canvas exists only in the page, so a Canvas layer row
// depends on three channels between python and the browser: the inventory the
// row's dropdowns are built from, the request/reply that renders a duel side's
// composite, and the set channel Set/Reset write through. None of it has a
// python-side test that can see the page, so this one drives the page: a DOM
// with two canvases (a unit Input tab and the host's img2img canvas), stubs for
// the two canvas_extra.js hooks, and the panel's channel elements - then a
// request is painted into the request channel exactly as the tick does, and
// the reply is read out of the reply textbox exactly as gradio would.
//
// What has to hold, and why each would be a silent miss without this:
//   * a canvas is keyed by WHERE IT LIVES (u<unit>.in<slot> / img2img) and by
//     TAB - a txt2img request for the img2img canvas must be refused, not
//     served from the other tab's canvas;
//   * the request's overrides reach forgeCanvasComposite as {index: alpha}
//     for the right uuid, and the reply carries the request's seq;
//   * a layer the canvas does not have is an ERROR reply - not a composite of
//     whatever the canvas does have, which would generate on the wrong image
//     while looking like success;
//   * a set message reaches forgeCanvasSetLayerOpacity with the right layer.
//
// stdout: JSON, see `out` at the bottom.

const fs = require('fs');
const path = require('path');
const Module = require('module');

const EXTENSION = path.join(__dirname, '..');
const AB_JS = path.join(EXTENSION, 'javascript', 'cnpro_ab.js');

function loadJsdom() {
    const extra = process.env.CNPRO_TEST_NODE_PATH;
    if (extra) {
        Module.globalPaths.push(path.join(extra, 'node_modules'));
        process.env.NODE_PATH = (process.env.NODE_PATH ? process.env.NODE_PATH + path.delimiter : '') +
            path.join(extra, 'node_modules');
        Module._initPaths();
    }
    return require('jsdom');
}

let JSDOM;
try {
    JSDOM = loadJsdom().JSDOM;
} catch (exc) {
    process.stdout.write(JSON.stringify({unavailable: 'jsdom is not installed: ' + exc.message}));
    process.exit(0);
}

const out = {logs: [], composites: [], sets: [], replies: [], inventories: []};

const dom = new JSDOM('<!doctype html><html><body>'
    // a unit's Input 2 canvas on txt2img, and the host's img2img canvas
    + '<div id="txt2img_controlnet_ControlNet-0_input_image_1"><div><div id="imageContainer_uuid_unit"></div></div></div>'
    + '<div id="img2img_image"><div id="imageContainer_uuid_host"></div></div>'
    // the panel's channels, both tabs
    + ['txt2img', 'img2img'].map(tab =>
        '<div id="cnpro_ab_' + tab + '_duel"></div>'
        + '<div id="cnpro_ab_' + tab + '_canvases"><textarea></textarea></div>'
        + '<div id="cnpro_ab_' + tab + '_canvas_reply"><textarea></textarea></div>'
        + '<div id="cnpro_ab_' + tab + '_canvas_request"><div></div></div>'
        + '<div id="cnpro_ab_' + tab + '_canvas_set"><div></div></div>').join('')
    + '</body></html>', {runScripts: 'outside-only'});
const window = dom.window;
// jsdom has no layout, so offsetParent is always null and the panel would
// read as folded away; the inventory is published only for a visible panel.
Object.defineProperty(window.HTMLElement.prototype, 'offsetParent', {
    get() { return this.parentElement; },
});

const LAYERS = {
    uuid_unit: [{w: 1024, h: 768, opacity: 1}, {w: 512, h: 512, opacity: 0.4}],
    uuid_host: [{w: 640, h: 640, opacity: 1}],
};
window.forgeCanvasDebugLayers = (uuid) => LAYERS[uuid] ? {layers: LAYERS[uuid]} : null;
window.forgeCanvasComposite = (uuid, opacities, callback) => {
    out.composites.push({uuid: uuid, opacities: opacities});
    callback('data:image/png;base64,' + uuid);
    return true;
};
window.forgeCanvasSetLayerOpacity = (uuid, index, alpha) => {
    out.sets.push({uuid: uuid, index: index, alpha: alpha});
    return !!(LAYERS[uuid] && LAYERS[uuid][index]);
};
// the host's helpers the file relies on
window.updateInput = (target) => {
    const id = target.parentElement.id;
    if (id.endsWith('_canvas_reply')) out.replies.push({tab: id.split('_')[2], text: target.value});
    if (id.endsWith('_canvases')) out.inventories.push({tab: id.split('_')[2], text: target.value});
};
const uiUpdates = [];
window.onUiUpdate = (fn) => uiUpdates.push(fn);
window.console = {
    log: (...a) => out.logs.push('log: ' + a.join(' ')),
    warn: (...a) => out.logs.push('warn: ' + a.join(' ')),
    error: (...a) => out.logs.push('error: ' + a.join(' ')),
};

try {
    window.eval(fs.readFileSync(AB_JS, 'utf8'));
} catch (exc) {
    out.loadError = String((exc && exc.stack) || exc);
    process.stdout.write(JSON.stringify(out));
    process.exit(0);
}
uiUpdates.forEach(fn => fn());   // the panel is mounted: bind the observers

function paint(tab, channel, message) {
    const body = document(tab, channel);
    const pre = window.document.createElement('pre');
    pre.className = 'cnpro-ab-channel-body';
    pre.textContent = JSON.stringify(message);
    body.textContent = '';
    body.appendChild(pre);
}
function document(tab, channel) {
    return window.document.getElementById('cnpro_ab_' + tab + '_' + channel).firstChild;
}
const tick = () => new Promise(resolve => setTimeout(resolve, 20));

(async () => {
    // 1. a txt2img request for the unit's Input 2, layer 2 at 50%
    paint('txt2img', 'canvas_request', {seq: 7, overrides: {'u0.in2': {'1': 0.5}}});
    await tick();
    // 2. the same seq painted again is not served twice (a page reload
    //    repaints the request; python drops a duplicate anyway, but the
    //    composite should not be rendered again)
    paint('txt2img', 'canvas_request', {seq: 7, overrides: {'u0.in2': {'1': 0.5}}});
    await tick();
    // 3. a layer the canvas does not have
    paint('txt2img', 'canvas_request', {seq: 8, overrides: {'u0.in2': {'5': 0.5}}});
    await tick();
    // 4. the img2img canvas asked for from the txt2img tab: not this tab's
    paint('txt2img', 'canvas_request', {seq: 9, overrides: {'img2img': {'0': 0.25}}});
    await tick();
    // 5. ...and from its own tab
    paint('img2img', 'canvas_request', {seq: 10, overrides: {'img2img': {'0': 0.25}}});
    await tick();
    // 6. Set writes into the live canvas
    paint('txt2img', 'canvas_set', {seq: 1, layers: {'u0.in2': {'1': 0.5}}});
    await tick();
    process.stdout.write(JSON.stringify(out));
    // the file's inventory poll (setInterval) would keep node alive forever
    process.exit(0);
})();
