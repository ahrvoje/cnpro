// Node harness for tests/test_mask_profile_coupling.py.
//
// Two independent implementations decide which weight-mask slots a generation
// uses, and they have to give the same answer for every selector:
//
//   javascript/weight_mask.js   liveSlotKeys()  -> what the toolbar SHOWS
//   lib_cnpro/external_code.py  masks_in_force() -> what is APPLIED
//
// A disagreement is invisible: the toolbar dims a slot the backend still uses,
// or leaves a slot bright that the backend drops. So this harness produces the
// JS half - including the profile STRING the editor writes for each selector,
// which is the only thing the python half ever sees - and the python test
// compares them.
//
// stdout: JSON {bands: [...], js: {band: [slot, ...]}, strings: {band: "..."},
//               published: {...}}

global.window = {devicePixelRatio: 1};
global.document = {documentElement: {}};
global.getComputedStyle = function () {
    return {getPropertyValue: function () { return ''; }};
};

const wmask = require('../javascript/weight_mask.js');
const {WeightProfileEditor} = require('../javascript/weight_profile.js');

// Every selector the editor can be in, including the two that are not bands.
const SELECTORS = ['main', 'coarse', 'mid', 'fine', 'depth'];

// ---- 1. what weight_mask.js says is live, per selector ---------------------
const js = {};
for (const band of SELECTORS) js[band] = wmask.liveSlotKeys(band);

// A selector nobody has heard of must not silently mean "bands". Reading a
// stale/garbage attribute is a real path (the unit root is written by another
// module), and defaulting to the band set there would switch the masks over
// with nothing pressed.
js['(unknown)'] = wmask.liveSlotKeys('something-else');
js['(empty)'] = wmask.liveSlotKeys('');

// ---- 2. how the painter reads the published selector ------------------------
const published = {
    missingUnit: wmask.selectedProfileBand(null),
    noDataset: wmask.selectedProfileBand({}),
    empty: wmask.selectedProfileBand({dataset: {}}),
    set: wmask.selectedProfileBand({dataset: {cnetProfileBand: 'mid'}}),
};

// ---- 3. the STRING the editor writes for each selector ----------------------
//
// This is the whole channel between the two implementations: python never sees
// the editor, only this string. Built through the real serialize(), with a
// non-neutral curve in every slot so nothing is omitted for being a default.
function editorFor(band) {
    const ed = Object.create(WeightProfileEditor.prototype);
    ed.isBalance = false;
    ed.scaleLo = 0;
    ed.scaleHi = 1;
    ed.depthLo = 0;
    ed.depthHi = 2;
    const curve = (y0, y1) => ({
        points: [{x: 0, y: y0}, {x: 1, y: y1}],
        cosOn: false, cosN: 1, cosPhase: 0, multiPhase: false, gamma: 1,
    });
    ed.store = {
        main: curve(1, 0.5),
        coarse: curve(0.9, 0.2),
        mid: curve(0.8, 0.3),
        fine: curve(0.7, 0.4),
        depth: curve(0.6, 0.1),
    };
    ed.band = band;
    // serialize() re-snapshots the working copy into store[band] first, so the
    // working copy has to be the selected profile - exactly as in the app.
    const P = ed.store[band];
    ed.points = P.points;
    ed.cosOn = P.cosOn;
    ed.cosN = P.cosN;
    ed.cosPhase = P.cosPhase;
    ed.multiPhase = P.multiPhase;
    ed.gamma = P.gamma;
    return ed;
}

const strings = {};
for (const band of SELECTORS) {
    strings[band] = editorFor(band).serialize();
}

// ...and the string an editor writes when every curve is left at its default,
// which is the common case and must still carry the selector.
function neutralString(band) {
    const ed = Object.create(WeightProfileEditor.prototype);
    ed.isBalance = false;
    ed.scaleLo = 0;
    ed.scaleHi = 1;
    ed.depthLo = 0;
    ed.depthHi = 2;
    const flat = (y) => ({
        points: [{x: 0, y: y}, {x: 1, y: y}],
        cosOn: false, cosN: 1, cosPhase: 0, multiPhase: false, gamma: 1,
    });
    ed.store = {
        main: flat(1), coarse: flat(1), mid: flat(1), fine: flat(1), depth: flat(0.5),
    };
    ed.band = band;
    const P = ed.store[band];
    ed.points = P.points;
    ed.cosOn = P.cosOn;
    ed.cosN = P.cosN;
    ed.cosPhase = P.cosPhase;
    ed.multiPhase = P.multiPhase;
    ed.gamma = P.gamma;
    return ed.serialize();
}

const neutral = {};
for (const band of SELECTORS) neutral[band] = neutralString(band);

// ---- 4. the round trip: string back to a selector ---------------------------
// parsePacked is what restores the mode on reload. If it does not return the
// selector serialize() wrote, the mode (and with it the live mask slots) is
// lost on every page load.
const roundTrip = {};
for (const band of SELECTORS) {
    const ed = Object.create(WeightProfileEditor.prototype);
    ed.isBalance = false;
    roundTrip[band] = ed.parsePacked(strings[band]).selected;
}

process.stdout.write(JSON.stringify({
    selectors: SELECTORS,
    js: js,
    published: published,
    strings: strings,
    neutral: neutral,
    roundTrip: roundTrip,
}));
