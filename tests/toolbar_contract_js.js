// Node harness for tests/test_toolbar_contract.py: EXECUTE the renderer and the
// registry, and report what they actually do, instead of reading them as text.
//
// WHY THIS EXISTS
// ---------------
// The first version of the toolbar contract derived its ids with
// `.match(...).map(s => s.slice(4, -14))`. `_forge_mixin"` is thirteen
// characters. Every id lost its last letter, every getElementById returned null,
// and the entire toolbar vanished.
//
// The python test passed the whole time. It checked that TOOLBAR_IDS existed and
// was derived from the markup by regex -- both true -- and never once checked
// that the derivation produced CORRECT ids. Reading code as text can verify that
// a mechanism is present; only running it can verify the mechanism works.
//
// WHAT THE CROSS-CHECK IS NOW
// ---------------------------
// There is no longer a markup blob to parse ids back out of; markup is RENDERED
// from javascript/canvas_tools.js. So the independent derivation runs the other
// way: render, then parse the ids out of the rendered HTML, and compare that
// against the ids the registry reports directly. Same source, two routes --
// a renderer that drops, duplicates or mangles a control disagrees with the list
// that claims it exists, at test time, by name.
//
// stdout: JSON, see the keys at the bottom.

const fs = require('fs');
const path = require('path');

const JS = path.join(__dirname, '..', 'javascript');
const NODES_JS = path.join(JS, 'canvas_nodes.js');
const TOOLS_JS = path.join(JS, 'canvas_tools.js');

// ---- capture whatever the modules log ---------------------------------------
const selfCheckErrors = [];
global.console = Object.assign(Object.create(console), {
    error: (...a) => selfCheckErrors.push(a.join(' ')),
    warn: (...a) => selfCheckErrors.push(a.join(' ')),
});

// ---- a DOM just real enough --------------------------------------------------
// Two modes. `captureDom` records the HTML inject() emits without parsing it;
// `buildDom` fakes the post-injection node set so reveal/audit can be exercised
// against controlled failures. Neither needs a real HTML parser -- the real-DOM
// coverage lives in tests/toolbar_dom_js.js (jsdom), and this file exists to
// exercise the failure modes that a healthy DOM cannot produce.
let NODES = {};
global.document = {
    getElementById: (id) => (Object.prototype.hasOwnProperty.call(NODES, id) ? NODES[id] : null),
};
global.window = {};

const UUID = 'testuuid';

function fakeNode(key) {
    return {
        id: key,
        style: {display: 'none'},
        dataset: {},
        // The stub models a canvas inside CNPro's INPUT group - the home
        // container of the registry's `scope`d buttons (the weight-mask
        // slots), so the reveal/audit contract is exercised where those
        // buttons belong. Scenario `outOfScope` flips this to null to model
        // the host's own canvases.
        closest: (sel) => (sel === '.cnet-input-image-group' ? {className: 'cnet-input-image-group'} : null),
        // audit() asks getComputedStyle via ownerDocument.defaultView; leaving
        // that undefined makes isHidden() fall back to the inline style, which
        // is exactly what this harness controls.
        ownerDocument: null,
    };
}

function buildDom(uuid, ids) {
    NODES = {};
    ids.forEach((id) => {
        const key = id + '_' + uuid;
        NODES[key] = fakeNode(key);
    });
    // audit() looks the container up to decide whether chrome is suppressed
    NODES['imageContainer_' + uuid] = fakeNode('imageContainer_' + uuid);
}

// ---- load the modules under test ---------------------------------------------
// Renderer first, registry second: filename order, i.e. the order the browser
// uses. Loading them the convenient way round would hide a reintroduced
// load-time dependency on the registry.
require(NODES_JS);
require(TOOLS_JS);
const api = global.window.cnproCanvasNodes;
const registry = global.window.cnproCanvasTools;
if (!api || !registry) {
    process.stdout.write(JSON.stringify({
        fatal: (!api ? 'canvas_nodes.js did not export window.cnproCanvasNodes. ' : '') +
               (!registry ? 'canvas_tools.js did not export window.cnproCanvasTools.' : ''),
    }));
    process.exit(0);
}

const toolbarIds = api.toolbarIds();
const ownedIds = api.ownedIds();
const deferred = Object.keys(api.deferred());

// ---- the independent derivation: render, then read the ids back out ----------
const emitted = [];
NODES = {};
NODES['toolbar_' + UUID] = Object.assign(fakeNode('toolbar_' + UUID), {
    querySelector: (sel) => (sel === '.forge-toolbar-box-a'
        ? Object.assign(fakeNode('boxA'), {
            insertAdjacentHTML: (where, html) => emitted.push(html),
        })
        : null),
    insertAdjacentHTML: (where, html) => emitted.push(html),
});
NODES['imageContainer_' + UUID] = Object.assign(fakeNode('imageContainer_' + UUID), {
    insertAdjacentHTML: (where, html) => emitted.push(html),
});

let injectError = null;
let injected = null;
try {
    injected = api.inject(UUID);
} catch (exc) {
    injectError = String((exc && exc.stack) || exc);
}
const renderedHtml = emitted.join('\n');
const renderedIds = [];
renderedHtml.replace(new RegExp('id="([A-Za-z0-9_]+)_' + UUID + '"', 'g'), (whole, id) => {
    renderedIds.push(id);
    return whole;
});

// Every attribute the CSS and the reveal sweep key off must survive rendering.
// A tool that loses .forge-adjust-control is injected, wired and invisible --
// the original bug, in one class name.
const renderedClasses = {};
renderedHtml.replace(
    new RegExp('id="([A-Za-z0-9_]+)_' + UUID + '"\\s+class="([^"]*)"', 'g'),
    (whole, id, cls) => {
        renderedClasses[id] = cls.split(/\s+/);
        return whole;
    });
// buttons render id before class; menus render class before id. Catch both.
renderedHtml.replace(
    new RegExp('class="([^"]*)"\\s+id="([A-Za-z0-9_]+)_' + UUID + '"', 'g'),
    (whole, cls, id) => {
        renderedClasses[id] = cls.split(/\s+/);
        return whole;
    });

const scenarios = {};

// Every lookup below goes through this. A derivation bug makes the id list name
// nodes that were never created (`layersButto`), and indexing NODES directly
// would throw -- turning a precise diagnosis into a stack trace from the test
// harness. The harness must survive a broken module and REPORT it; a test that
// crashes tells you something is wrong but not what, which is most of the way
// back to the silent miss it exists to prevent.
const nodeFor = (id) => NODES[id + '_' + UUID] || null;
const isVisible = (id) => {
    const n = nodeFor(id);
    return n ? n.style.display !== 'none' : false;
};

// 1. a clean attach: build the DOM, reveal, audit
buildDom(UUID, ownedIds);
const shown = api.revealToolbar(UUID);
scenarios.clean = {
    shown: shown,
    visible: toolbarIds.filter(isVisible),
    hidden: toolbarIds.filter((id) => !isVisible(id)),
    // ids the module asked for that were never in the DOM at all: the direct
    // fingerprint of a mangled id list
    unresolved: toolbarIds.filter((id) => nodeFor(id) === null),
    audit: api.audit(UUID),
};

// 2. a control that the reveal missed (the exact shape of the mask-button bug)
buildDom(UUID, ownedIds);
api.revealToolbar(UUID);
const victim = toolbarIds.filter((id) => deferred.indexOf(id) === -1 && nodeFor(id))[0] || null;
if (victim) {
    nodeFor(victim).style.display = 'none';
}
scenarios.oneHidden = {victim: victim, audit: victim ? api.audit(UUID) : []};

// 3. a control the registry declares but the DOM does not have (a renamed id)
buildDom(UUID, ownedIds);
api.revealToolbar(UUID);
if (victim) {
    delete NODES[victim + '_' + UUID];
}
scenarios.oneMissing = {victim: victim, audit: victim ? api.audit(UUID) : []};

// 4. a MENU node missing -- not a button. The audit used to walk buttons only,
//    so a slider that never landed was invisible to it, which is how a menu can
//    open empty and nothing says why.
buildDom(UUID, ownedIds);
api.revealToolbar(UUID);
const sliderVictim = ownedIds.filter((id) => toolbarIds.indexOf(id) === -1)[0] || null;
if (sliderVictim) {
    delete NODES[sliderVictim + '_' + UUID];
}
scenarios.menuNodeMissing = {
    victim: sliderVictim,
    audit: sliderVictim ? api.audit(UUID) : [],
};

// 5. nothing injected at all
buildDom(UUID, []);
scenarios.nothingInjected = {shown: api.revealToolbar(UUID), audit: api.audit(UUID)};

// 6. the same toolbar on a canvas OUTSIDE CNPro's input group (the host's own
//    img2img/inpaint canvases): the registry's `scope`d buttons must stay
//    hidden - revealing them there produced visible-but-inert chrome, rule
//    8c's exact shape - and the audit must NOT call that a failure (a check
//    that cries wolf trains everyone to ignore the real one).
buildDom(UUID, ownedIds);
Object.keys(NODES).forEach((key) => { NODES[key].closest = () => null; });
scenarios.outOfScope = {
    shown: api.revealToolbar(UUID),
    hidden: toolbarIds.filter((id) => !isVisible(id)),
    audit: api.audit(UUID),
};

process.stdout.write(JSON.stringify({
    toolbarIds: toolbarIds,
    ownedIds: ownedIds,
    renderedIds: renderedIds,
    renderedClasses: renderedClasses,
    deferred: deferred,
    scoped: Object.keys(registry.scopes ? registry.scopes() : {}),
    inject: {returned: injected, error: injectError},
    selfCheck: api.selfCheck(),
    selfCheckErrors: selfCheckErrors,
    toolCount: registry.TOOLS.length,
    scenarios: scenarios,
}));
