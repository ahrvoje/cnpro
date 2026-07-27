// Node harness for tests/test_profile_parity.py: parse and evaluate profile
// strings with the EDITOR's implementation, so the python twin can be
// compared against it.
//
// weight_profile.js is a browser IIFE; it exports its class and skips every
// DOM registration when the webui globals are absent (see the headless-load
// guard at its end). The instance is built with Object.create rather than the
// constructor because the constructor wires a canvas - the grammar half under
// test touches none of that.
//
// stdin:  {"cases": ["0@1;1@0.5|2", ...], "samples": 21}
// stdout: {"results": [{"scaleLo":0,"scaleHi":2,"main":[...],"bands":{...},
//                       "depth":[...]|null}, ...]}
// Values are EFFECTIVE (scale-mapped), sampled on an even grid over [0, 1].
// Main and the bands share the step plot's range; the depth curve is a plot of
// its own and is sampled on the range its '#D' segment carries.

global.window = { devicePixelRatio: 1 };
global.document = { documentElement: {} };
global.getComputedStyle = function () {
    return { getPropertyValue: function () { return ''; } };
};

const { WeightProfileEditor } = require('../javascript/weight_profile.js');

function bareEditor(isBalance) {
    const ed = Object.create(WeightProfileEditor.prototype);
    ed.isBalance = !!isBalance;
    return ed;
}

/** Effective values of one parsed profile object, sampled over [0, 1]. */
function sample(ed, profile, lo, hi, samples) {
    // evaluate() reads the working copy, so load this profile into it
    ed.points = profile.points;
    ed.cosOn = !!profile.cosOn;
    ed.cosN = profile.cosN || 0;
    ed.cosPhase = profile.cosPhase || 0;
    ed.gamma = profile.gamma || 1;
    const out = [];
    for (let i = 0; i < samples; i++) {
        const x = samples === 1 ? 0 : i / (samples - 1);
        out.push(lo + ed.evaluate(x) * (hi - lo));
    }
    return out;
}

let raw = '';
process.stdin.on('data', (chunk) => { raw += chunk; });
process.stdin.on('end', () => {
    const input = JSON.parse(raw);
    const samples = input.samples || 21;
    const results = input.cases.map((text) => {
        const ed = bareEditor(input.isBalance);
        const packed = ed.parsePacked(text);
        if (!packed.main) return { main: null };
        const lo = packed.scaleLo;
        const hi = packed.scaleHi;
        const bands = {};
        for (const key of Object.keys(packed.bands)) {
            bands[key] = sample(ed, packed.bands[key], lo, hi, samples);
        }
        return {
            scaleLo: lo,
            scaleHi: hi,
            main: sample(ed, packed.main, lo, hi, samples),
            bands: bands,
            depth: packed.depth
                ? sample(ed, packed.depth, packed.depthLo, packed.depthHi, samples)
                : null,
            selected: packed.selected,
        };
    });
    process.stdout.write(JSON.stringify({ results: results }));
});
