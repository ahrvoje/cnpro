// Node harness for tests/test_toolbar_dom.py: run canvas_nodes.js against the
// HOST'S REAL canvas.html in a real DOM (jsdom), and report what actually ends
// up on screen.
//
// WHY THIS EXISTS
// ---------------
// Two toolbar regressions shipped in a row. Both were "the buttons are simply
// not there", both were silent, and both got past a test suite that was
// checking the wrong layer:
//
//   * tests/test_toolbar_contract.py reads the JS as TEXT. It proved a contract
//     existed. It could not prove the contract's ids were correct.
//   * tests/toolbar_contract_js.js EXECUTES the module, but against a DOM stub
//     that I hand-populated from OWNED_IDS. It proves reveal/audit logic works
//     GIVEN that the nodes exist -- and assumes the thing most likely to be
//     broken.
//
// Neither ever ran `inject()` against the host's actual markup. So nothing in
// the suite could answer the only question that matters: after CNPro attaches to
// a real ForgeCanvas, is each control present in the DOM and visible?
//
// This harness answers exactly that, from the two real inputs:
//   modules_forge/forge_canvas/canvas.html   (the host's template)
//   javascript/canvas_nodes.js               (CNPro's injection)
//
// stdin:  (nothing)
// stdout: JSON, see the keys assembled at the bottom.

const fs = require('fs');
const path = require('path');
const Module = require('module');

const HERE = __dirname;
const EXTENSION = path.join(HERE, '..');
// Two levels up when this lives in <webui>/extensions/<name>; a standalone
// clone is elsewhere and the guess then points at a directory that does not
// exist. CNPRO_WEBUI_DIR overrides it, and a missing template is reported as
// unavailable below rather than thrown as an ENOENT stack.
const WEBUI = process.env.CNPRO_WEBUI_DIR || path.join(EXTENSION, '..', '..');
const CANVAS_HTML = path.join(WEBUI, 'modules_forge', 'forge_canvas', 'canvas.html');
const NODES_JS = path.join(EXTENSION, 'javascript', 'canvas_nodes.js');
const TOOLS_JS = path.join(EXTENSION, 'javascript', 'canvas_tools.js');
const EXTRA_JS = path.join(EXTENSION, 'javascript', 'canvas_extra.js');
const WMASK_JS = path.join(EXTENSION, 'javascript', 'weight_mask.js');

// jsdom is a TEST-ONLY dependency, installed outside the repo. Resolve it from
// wherever the caller put it rather than requiring it to be vendored.
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

if (!fs.existsSync(CANVAS_HTML)) {
    process.stdout.write(JSON.stringify({unavailable:
        "the host's canvas.html is not at " + CANVAS_HTML + " - this checkout " +
        "is not inside a webui tree. Set CNPRO_WEBUI_DIR to the webui root."}));
    process.exit(0);
}

const {JSDOM} = loadJsdom();

const UUID = 'testuuid';
const OUT_UUID = 'testoutput';
const rawTemplate = fs.readFileSync(CANVAS_HTML, 'utf8');
const template = rawTemplate.split('forge_mixin').join(UUID);

// TWO CANVASES, WRAPPED THE WAY PRODUCTION WRAPS THEM
// (lib_cnpro/controlnet_ui/controlnet_ui_group.py): the input canvas inside
// .cnet-input-image-group, the output-mask canvas inside
// .cnet-output-mask-group. The registry scopes the weight-mask slots to these
// groups (G to both, C/M/F to the output group only, since the GCMF move
// 2026-08-02), so a bare, unwrapped template puts EVERY scoped button
// out-of-scope and can verify nothing about them. Most checks run on the
// input canvas (UUID); the output one exists for the per-surface reveal.
const dom = new JSDOM('<!doctype html><html><body>'
    + '<div class="cnet-input-image-group">' + template + '</div>'
    + '<div class="cnet-output-mask-group">'
    + rawTemplate.split('forge_mixin').join(OUT_UUID) + '</div>'
    + '</body></html>');
const {window} = dom;

// Both modules are browser IIFEs assigning onto window.
const logs = [];
global.window = window;
global.document = window.document;
global.console = Object.assign(Object.create(console), {
    error: (...a) => logs.push('error: ' + a.join(' ')),
    warn: (...a) => logs.push('warn: ' + a.join(' ')),
});

// LOADED IN THE ORDER THE BROWSER LOADS THEM -- filename order, which puts the
// renderer BEFORE the registry it renders. That is the exact hazard
// canvas_nodes.js's lazy resolution exists to absorb, so the test must reproduce
// it rather than tidy it away: loading the registry first here would prove
// nothing about the page.
let loadError = null;
try {
    require(NODES_JS);
    require(TOOLS_JS);
} catch (exc) {
    loadError = String(exc && exc.stack || exc);
}

const api = global.window.cnproCanvasNodes;
const out = {loadError: loadError, logs: logs};

if (!api) {
    out.fatal = 'canvas_nodes.js did not export window.cnproCanvasNodes';
    process.stdout.write(JSON.stringify(out));
    process.exit(0);
}

// ---- 1. does the host template have the anchors inject() needs? -------------
out.anchors = {
    toolbar: !!document.getElementById('toolbar_' + UUID),
    imageContainer: !!document.getElementById('imageContainer_' + UUID),
    boxA: !!(document.getElementById('toolbar_' + UUID) || {querySelector: () => null})
        .querySelector('.forge-toolbar-box-a'),
};

// ---- 2. inject, for real ----------------------------------------------------
let injectError = null;
let injected = null;
try {
    injected = api.inject(UUID);
} catch (exc) {
    injectError = String(exc && exc.stack || exc);
}
out.inject = {returned: injected, error: injectError};

// ---- 3. did every declared node actually land in the DOM? -------------------
const ownedIds = api.ownedIds();
out.ownedCount = ownedIds.length;
out.missingAfterInject = ownedIds.filter(
    (id) => !document.getElementById(id + '_' + UUID));

// ---- 4. the nodes attach() requires before it will do ANYTHING --------------
// canvas_extra.js::attach gates on a REQUIRED map of 23 nodes and returns false
// if ONE is absent -- no toolbar, no Topaz probe, no listeners, for that canvas.
//
// The list is READ OUT OF canvas_extra.js, not copied here. A hand-copied gate
// silently stops covering whatever was added to the real one, which is the same
// declared-here/honoured-there drift this whole suite exists to catch. Grep is
// the right tool for it: this is a POLICY question ("which ids does the gate
// name"), not a behavioural one.
const extraSrc = fs.readFileSync(EXTRA_JS, 'utf8');
const gateBlock = extraSrc.match(/const REQUIRED = \{([\s\S]*?)\};/);
const ATTACH_GATE = [];
if (gateBlock) {
    const re = /'([A-Za-z0-9_]+)':/g;
    let m;
    while ((m = re.exec(gateBlock[1]))) ATTACH_GATE.push(m[1]);
}
out.attachGateParsed = ATTACH_GATE.length;
out.attachGate = {};
ATTACH_GATE.forEach((name) => {
    // the gate names ids as canvas_extra.js's el() sees them, minus the uuid
    out.attachGate[name] = !!document.getElementById(name + '_' + UUID);
});

// ---- 5. reveal, then measure what a user would see --------------------------
let revealError = null;
let shown = null;
try {
    shown = api.revealToolbar(UUID);
} catch (exc) {
    revealError = String(exc && exc.stack || exc);
}

const toolbarIds = api.toolbarIds();
const visible = [];
const hidden = [];
toolbarIds.forEach((id) => {
    const n = document.getElementById(id + '_' + UUID);
    if (!n) return;
    (n.style.display === 'none' ? hidden : visible).push(id);
});

out.reveal = {shown: shown, error: revealError, visible: visible, hidden: hidden};
out.audit = api.audit(UUID);

// ---- 5b. the OUTPUT-MASK canvas: inject + reveal + audit on the second
// surface. All four mask slots are in scope there; the audit suppresses its
// visibility half inside .cnet-output-mask-group (style.css hides the chrome
// there on purpose - jsdom loads no stylesheet, so only the audit's own
// suppression can keep it quiet).
let outputError = null;
let outputShown = null;
try {
    api.inject(OUT_UUID);
    outputShown = api.revealToolbar(OUT_UUID);
} catch (exc) {
    outputError = String(exc && exc.stack || exc);
}
const outputVisible = [];
const outputHidden = [];
toolbarIds.forEach((id) => {
    const n = document.getElementById(id + '_' + OUT_UUID);
    if (!n) return;
    (n.style.display === 'none' ? outputHidden : outputVisible).push(id);
});
out.outputReveal = {shown: outputShown, error: outputError,
                    visible: outputVisible, hidden: outputHidden};
out.outputAudit = api.audit(OUT_UUID);

out.deferred = Object.keys(api.deferred());
out.toolbarIds = toolbarIds;
// the registry's {buttonId: selector} scope map, so the python side derives
// the per-surface expectations instead of restating them
const registry = global.window.cnproCanvasTools;
out.scopesMap = registry && registry.scopes ? registry.scopes() : {};
out.selfCheck = api.selfCheck();

// ---- 6. injecting twice must be a no-op, not a duplication ------------------
// Duplicates are checked over EVERY owned id, not just the buttons: a duplicated
// slider id is just as fatal (getElementById wires the first, the user drags the
// second) and is exactly the kind of thing a second inject would produce.
api.inject(UUID);
out.duplicateIds = ownedIds.filter(
    (id) => document.querySelectorAll('#' + id + '_' + UUID).length > 1);

// ---- 7. supply and demand must match, in BOTH directions --------------------
//
// audit() only ever walks the registry and asks "is this in the DOM". Nothing
// asked the reverse -- "does the wiring reach for an id nobody declares" -- and
// that gap is exactly how `levelsBox` shipped: markup, ids and wired `input`
// handlers on both sides, no button that opened it, two working sliders
// unreachable, in the fork AND here, silently, for as long as it existed.
//
// DEMAND: every `el('someId_')` in the two files that wire this chrome.
// SUPPLY: the registry, plus the host's own template, plus the nodes
//         weight_mask.js builds for itself at runtime (it owns an eraser and a
//         feather row that the registry deliberately does not declare - a
//         second injector is allowed, an undeclared lookup is not).
const wmaskSrc = fs.readFileSync(WMASK_JS, 'utf8');

const hostIds = new Set();
template.replace(new RegExp('id="([A-Za-z0-9_]+)_' + UUID + '"', 'g'), (whole, id) => {
    hostIds.add(id);
    return whole;
});

const dynamicIds = new Set();
wmaskSrc.replace(/\.id = '([A-Za-z0-9_]+)_' \+ uuid/g, (whole, id) => {
    dynamicIds.add(id);
    return whole;
});

// Demand is any `'someId_'` string literal, not just `el('someId_')`: ids also
// reach the DOM through lookup TABLES (weight_mask.js resolves its four band
// buttons out of a BAND_BUTTONS map). Matching only the direct call form left
// those four looking unused and would have "found" a bug that was not there --
// the check has to see every way an id is spelled, or it lies in both
// directions.
const wanted = new Set();
for (const src of [extraSrc, wmaskSrc]) {
    src.replace(/'([A-Za-z][A-Za-z0-9]*)_'/g, (whole, id) => {
        wanted.add(id);
        return whole;
    });
}

const declared = new Set(ownedIds);
out.wantedButUndeclared = [...wanted].filter(
    (id) => !declared.has(id) && !hostIds.has(id) && !dynamicIds.has(id));
out.declaredButUnused = [...declared].filter((id) => !wanted.has(id));
out.dynamicIds = [...dynamicIds];

// ---- 8. the label sizers must describe the labels that are ACTUALLY written --
//
// The menu's label column is sized from `labelMax` in canvas_tools.js, while the
// text is written by canvas_extra.js. Two files, one fact - the shape every bug
// in this module has had. Nothing in the browser can catch the drift either: a
// pixel test that writes the DECLARED strings into the label and measures them
// is asking the declaration about itself, and passes happily while the running
// app shows something else. So the declaration is compared against the SOURCE
// that writes it.
//
// Two halves, because a label is `prefix + value`:
//   PREFIX  extracted verbatim. Renaming "sensitivity" to "edge sensitivity"
//           and forgetting the registry is caught here.
//   VALUE   how many characters it can grow to. `x.toFixed(3)` is always
//           1+1+3 characters wide at minimum; a bare slider read spans the
//           range the registry itself declares. Where the printed value is
//           neither (a lookup table, a formula), that is REPORTED rather than
//           quietly skipped - see valueCheck: 'opaque'.
const labelWrites = [];
for (const [file, src] of [['canvas_extra.js', extraSrc], ['weight_mask.js', wmaskSrc]]) {
    // variable -> base id, so `rotLabel` is understood to be `rotationLabel`
    const bound = {};
    src.replace(/(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*el\('([A-Za-z0-9_]+)_'\)/g,
        (whole, name, id) => { bound[name] = id; return whole; });
    // ...and the ones a module builds itself (weight_mask's feather row)
    src.replace(/([A-Za-z0-9_]+)\.id\s*=\s*'([A-Za-z0-9_]+)_'\s*\+\s*uuid/g,
        (whole, name, id) => { bound[name] = id; return whole; });

    src.replace(/([A-Za-z0-9_]+)\.textContent\s*=\s*'([^']*)'\s*\+\s*([^;\n]+);/g,
        (whole, name, prefix, expr) => {
            if (!bound[name]) return whole;
            labelWrites.push({file: file, id: bound[name], prefix: prefix, expr: expr.trim()});
            return whole;
        });
}
// one entry per label id: the same label is rewritten from several places (the
// live handler and the sync-everything pass), and they agree by construction
const byId = {};
labelWrites.forEach((w) => {
    (byId[w.id] = byId[w.id] || []).push(w);
});
out.labelWrites = Object.keys(byId).map((id) => {
    const writes = byId[id];
    const prefixes = [...new Set(writes.map((w) => w.prefix))];
    const exprs = [...new Set(writes.map((w) => w.expr))];
    // trailing literal, e.g. `+ ' px'` on the brush size and `+ '°'` on rotation
    const suffixes = [...new Set(exprs.map((e) => {
        const m = e.match(/\+\s*'([^']*)'\s*$/);
        return m ? m[1] : '';
    }))];
    const fixed = exprs.map((e) => {
        const m = e.match(/\.toFixed\((\d+)\)/);
        return m ? Number(m[1]) : null;
    });
    return {
        id: id,
        files: [...new Set(writes.map((w) => w.file))],
        prefixes: prefixes,
        exprs: exprs,
        suffixes: suffixes,
        // decimals, when every write agrees on a toFixed
        toFixed: fixed.every((f) => f !== null && f === fixed[0]) ? fixed[0] : null,
        // a bare read of state (`T.black`, `st.penOpacity`) prints the slider's
        // own integer value; anything with a call in it does not
        rawValue: exprs.every((e) => /^[A-Za-z_$][A-Za-z0-9_$.]*$/.test(e)),
    };
});

// what the registry says, in the form the comparison needs
const reg = global.window.cnproCanvasTools;
const rowSpecs = [];
(reg ? reg.TOOLS : []).forEach((tool) => {
    if (!tool.menu) return;
    // flatRows, NOT menu.rows: a row nested in a group() for layout reasons is
    // still a row, and walking the raw list silently dropped the pen menu's
    // three sliders out of this check the moment they were grouped. Presentation
    // must not be able to shrink coverage.
    const rows = reg && typeof reg.flatRows === 'function'
        ? reg.flatRows(tool.menu) : (tool.menu.rows || []);
    for (const row of rows) {
        if (row.kind !== 'range') continue;
        rowSpecs.push({
            tool: tool.id, menu: tool.menu.id, id: row.id, labelId: row.labelId,
            label: row.label, labelMax: row.labelMax,
            min: row.min, max: row.max, step: row.step == null ? 1 : row.step,
        });
    }
});
out.rowSpecs = rowSpecs;
// ...and how many labels actually carry a sizer in the DOM. If the walk above
// ever stops seeing some rows again, these two numbers part company and the
// test says so, instead of quietly checking fewer things. Counted on the
// INPUT canvas only - the document holds a second (output-mask) canvas whose
// sizers would double the number.
out.sizerLabelsInDom = document.querySelectorAll(
    '.cnet-input-image-group .forge-range-row .forge-toolbar-label[data-label-max]').length;

process.stdout.write(JSON.stringify(out));
