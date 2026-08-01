// Node harness for tests/test_profile_scale_grid.py: what the two range selects
// of the profile editor OFFER, per axis, plus what the parser does with a range
// that is not on the grid.
//
// The editor is loaded headless exactly as tests/profile_parity_js.js does it -
// the class is exported and every DOM registration is skipped when the webui
// globals are absent. scaleGrid() reads only `this.band` / `this.isBalance`, so
// a bare instance answers it.
//
// stdin:  {"axes": ["main", "coarse", "depth", ...], "balance": true|false,
//          "parses": ["0@1;1@0|0~2", ...]}
// stdout: {"grids": {axis: [values]}, "balance": [values],
//          "parses": [{"lo":.., "hi":..}, ...]}

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

let raw = '';
process.stdin.on('data', (chunk) => { raw += chunk; });
process.stdin.on('end', () => {
    const input = JSON.parse(raw);
    const ed = bareEditor(false);
    const grids = {};
    for (const axis of input.axes || []) {
        ed.band = axis;
        grids[axis] = ed.scaleGrid();
    }
    const balanceEd = bareEditor(true);
    balanceEd.band = 'main';

    const parses = (input.parses || []).map((text) => {
        const p = bareEditor(false).parsePacked(text);
        return p && p.main ? {lo: p.scaleLo, hi: p.scaleHi} : null;
    });

    process.stdout.write(JSON.stringify({
        grids: grids,
        balance: balanceEd.scaleGrid(),
        parses: parses,
    }));
});
