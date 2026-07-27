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
// A case may be a plain string or {"text":..., "phaseIndex":k, "phaseCount":n}:
// the multi-phase families divide the wave between n Inputs, so a case that
// exercises them has to say which Input it is asking about. Counting Inputs
// walks the unit's DOM in the browser, which there is none of here, so the
// count is injected rather than discovered.
//
// stdin:  {"cases": ["0@1;1@0.5|2", ...], "samples": 21}
// stdout: {"results": [{"scaleLo":0,"scaleHi":2,"main":[...],"bands":{...},
//                       "depth":[...]|null,"drift":[...]|null}, ...]}
// Values are EFFECTIVE (scale-mapped), sampled on an even grid over [0, 1].
// Main and the bands share the step plot's range; the depth curve and the drift
// curve are each a plot of their own and are sampled on the range their '#D' /
// '#S' segment carries.

global.window = { devicePixelRatio: 1 };
global.document = { documentElement: {} };
global.getComputedStyle = function () {
    return { getPropertyValue: function () { return ''; } };
};

const { WeightProfileEditor, driftedDepth } = require('../javascript/weight_profile.js');

function bareEditor(isBalance) {
    const ed = Object.create(WeightProfileEditor.prototype);
    ed.isBalance = !!isBalance;
    return ed;
}

/** One profile's EFFECTIVE value at a single x (the sample() loop's body). */
function valueAt(ed, profile, lo, hi, x) {
    ed.points = profile.points;
    ed.cosOn = !!profile.cosOn;
    ed.cosN = profile.cosN || 0;
    ed.cosPhase = profile.cosPhase || 0;
    ed.gamma = profile.gamma || 1;
    ed.phaseFamily = profile.phaseFamily || null;
    ed.kappa = profile.kappa || 0;
    ed._phaseCount = 2;
    return lo + ed.gammaAt(ed.envelopeAt(x) * ed.waveFactor(x, 0)) * (hi - lo);
}

/**
 * The COMPOSITE the drift exists to produce: the depth curve's multiplier for a
 * layer at `depth`, at step `x`, with the drift moving where depth is read.
 *
 * This is what actually runs, and it is the one thing the per-curve comparisons
 * cannot see: two sides can agree on both curves and still disagree on the sign
 * of the shift, on the clamp, or on which axis the shift is measured in. Twin of
 * cnpro_core.weight_profile.depth_multiplier.
 */
function depthMultiplier(ed, packed, depth, x) {
    if (!packed.depth) return 1;
    const shift = packed.drift
        ? valueAt(ed, packed.drift, packed.driftLo, packed.driftHi, x)
        : 0;
    return valueAt(ed, packed.depth, packed.depthLo, packed.depthHi,
                   driftedDepth(depth, shift));
}

/** Effective values of one parsed profile object, sampled over [0, 1]. */
function sample(ed, profile, lo, hi, samples, phaseIndex, phaseCount) {
    // evaluate() reads the working copy, so load this profile into it
    ed.points = profile.points;
    ed.cosOn = !!profile.cosOn;
    ed.cosN = profile.cosN || 0;
    ed.cosPhase = profile.cosPhase || 0;
    ed.gamma = profile.gamma || 1;
    ed.phaseFamily = profile.phaseFamily || null;
    ed.kappa = profile.kappa || 0;
    // _phaseCount is normally resolved once per frame by draw(); there is no
    // canvas here, so stand in for it (waveFactor reads it through
    // phaseCount(), which floors at 2)
    ed._phaseCount = phaseCount || 2;
    const index = phaseIndex || 0;
    const out = [];
    for (let i = 0; i < samples; i++) {
        const x = samples === 1 ? 0 : i / (samples - 1);
        // evaluate() is waveFactor(x, 0); ask for the requested Input's share
        const v = ed.gammaAt(ed.envelopeAt(x) * ed.waveFactor(x, index));
        out.push(lo + v * (hi - lo));
    }
    return out;
}

let raw = '';
process.stdin.on('data', (chunk) => { raw += chunk; });
process.stdin.on('end', () => {
    const input = JSON.parse(raw);
    const samples = input.samples || 21;
    const results = input.cases.map((entry) => {
        const spec = typeof entry === 'string' ? { text: entry } : entry;
        const ed = bareEditor(input.isBalance);
        const packed = ed.parsePacked(spec.text);
        if (!packed.main) return { main: null };
        const lo = packed.scaleLo;
        const hi = packed.scaleHi;
        const bands = {};
        // only the MAIN profile can be multi-phase (python inspects no other
        // segment), so the bands, the depth curve and the drift curve are always
        // sampled as Input 1 of a non-splitting profile
        for (const key of Object.keys(packed.bands)) {
            bands[key] = sample(ed, packed.bands[key], lo, hi, samples, 0, 0);
        }
        return {
            scaleLo: lo,
            scaleHi: hi,
            main: sample(ed, packed.main, lo, hi, samples,
                         spec.phaseIndex, spec.phaseCount),
            bands: bands,
            depth: packed.depth
                ? sample(ed, packed.depth, packed.depthLo, packed.depthHi, samples, 0, 0)
                : null,
            drift: packed.drift
                ? sample(ed, packed.drift, packed.driftLo, packed.driftHi, samples, 0, 0)
                : null,
            // depth-under-drift on a (depth, step) grid, flattened row-major
            // over the same `samples` grid in both coordinates. Null when there
            // is no depth curve, which is also when the drift cannot do
            // anything - the python twin returns its neutral 1.0 there and the
            // comparison would be vacuous.
            depthField: packed.depth
                ? (() => {
                    const out = [];
                    for (let i = 0; i < samples; i++) {
                        const d = samples === 1 ? 0 : i / (samples - 1);
                        for (let j = 0; j < samples; j++) {
                            const x = samples === 1 ? 0 : j / (samples - 1);
                            out.push(depthMultiplier(ed, packed, d, x));
                        }
                    }
                    return out;
                })()
                : null,
            selected: packed.selected,
        };
    });
    process.stdout.write(JSON.stringify({ results: results }));
});
