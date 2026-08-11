// Behaviour harness for the Edge tool's featherMask().  It executes the
// function extracted from the production file, so these checks cannot pass on
// a second implementation maintained by the test.

const fs = require('fs');
const path = require('path');

function productionFunction(name) {
    const source = fs.readFileSync(
        path.join(__dirname, '..', 'javascript', 'canvas_extra.js'), 'utf8');
    const start = source.indexOf('function ' + name + '(');
    if (start < 0) throw new Error(name + ' is absent from canvas_extra.js');
    const body = source.indexOf('{', start);
    let depth = 0;
    for (let i = body; i < source.length; i++) {
        if (source[i] === '{') depth++;
        else if (source[i] === '}' && --depth === 0) {
            return Function('return (' + source.slice(start, i + 1) + ')')();
        }
    }
    throw new Error(name + ' has no closing brace');
}

const featherMask = productionFunction('featherMask');
const failures = [];
const check = (ok, message) => { if (!ok) failures.push(message); };
const count = (a) => a.reduce((n, v) => n + (v > 0), 0);
const mass = (a) => a.reduce((n, v) => n + v, 0);
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

function blank(w, h) { return new Uint8Array(w * h); }
function put(mask, w, x, y, value = 255) { mask[y * w + x] = value; }
function rect(mask, w, x0, y0, x1, y1, value = 255) {
    for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) put(mask, w, x, y, value);
    }
}
function run(source, w, h, feather) {
    const out = new Uint8Array(source);
    featherMask(out, w, h, feather);
    return out;
}
function components(mask, w, h) {
    const seen = new Uint8Array(mask.length);
    const queue = new Int32Array(mask.length);
    let result = 0;
    for (let start = 0; start < mask.length; start++) {
        if (!mask[start] || seen[start]) continue;
        result++;
        let head = 0, tail = 0;
        queue[tail++] = start;
        seen[start] = 1;
        while (head < tail) {
            const i = queue[head++], x = i % w, y = Math.floor(i / w);
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if (!dx && !dy) continue;
                    const xx = x + dx, yy = y + dy;
                    if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
                    const j = yy * w + xx;
                    if (mask[j] && !seen[j]) { seen[j] = 1; queue[tail++] = j; }
                }
            }
        }
    }
    return result;
}
function blocks2x2(mask, w, h) {
    let n = 0;
    for (let y = 0; y < h - 1; y++) {
        for (let x = 0; x < w - 1; x++) {
            const i = y * w + x;
            if (mask[i] && mask[i + 1] && mask[i + w] && mask[i + w + 1]) n++;
        }
    }
    return n;
}

const w = 41, h = 31;

// Zero is a strict no-op, including fractional coverage.
const arbitrary = blank(w, h);
rect(arbitrary, w, 5, 8, 30, 20, 173);
put(arbitrary, w, 2, 2, 19);
check(same(run(arbitrary, w, h, 0), arbitrary), 'feather 0 changed the mask');

// Already-pedantic detail is protected, not interpreted as material to erase.
const details = blank(w, h);
for (let x = 3; x <= 36; x++) put(details, w, x, 4, 211);
for (let i = 0; i < 13; i++) put(details, w, 4 + i, 9 + i, 97);
for (let y = 8; y <= 25; y++) put(details, w, 31, y, 149);
for (let x = 26; x <= 36; x++) put(details, w, x, 16, 149);
check(same(run(details, w, h, 100), details),
      'feather 100 erased or weakened existing one-pixel detail');

// A thick contour becomes a connected, one-pixel centreline instead of
// disappearing.  Ignore the rounded end shape; topology and thickness are the
// contract.
const bar = blank(w, h);
rect(bar, w, 4, 11, 36, 19);
const barThin = run(bar, w, h, 100);
check(count(barThin) > 20, 'the thick bar was erased instead of thinned');
check(count(barThin) < count(bar) * 0.35, 'the thick bar did not become a fine line');
check(components(barThin, w, h) === 1, 'thinning broke the thick bar apart');
check(blocks2x2(barThin, w, h) === 0, 'the thinned bar still contains a 2x2 thick block');

// Closed contours keep their hole and their connectivity.
const ring = blank(w, h);
rect(ring, w, 6, 5, 34, 25);
rect(ring, w, 11, 10, 29, 20, 0);
const ringThin = run(ring, w, h, 100);
check(count(ringThin) > 40, 'the thick ring was erased instead of thinned');
check(count(ringThin) < count(ring) * 0.5, 'the thick ring stayed broad');
check(components(ringThin, w, h) === 1, 'thinning opened or split the ring');
check(ringThin[15 * w + 20] === 0, 'thinning filled the ring hole');
check(blocks2x2(ringThin, w, h) === 0, 'the thinned ring still contains a 2x2 thick block');

// Slider travel is monotone.  The final centreline must be present with its
// original coverage at every earlier setting.
const amounts = [0, 25, 50, 75, 100];
const series = amounts.map((f) => run(bar, w, h, f));
for (let k = 1; k < series.length; k++) {
    check(mass(series[k]) <= mass(series[k - 1]),
          'raising feather from ' + amounts[k - 1] + ' to ' + amounts[k] + ' restored edge mass');
    check(series[k].every((v, i) => v <= series[k - 1][i]),
          'raising feather from ' + amounts[k - 1] + ' to ' + amounts[k] + ' restored a pixel');
    let centrelineChanged = false;
    for (let i = 0; i < bar.length; i++) {
        if (barThin[i] && series[k][i] !== bar[i]) centrelineChanged = true;
    }
    check(!centrelineChanged,
          'a protected centreline pixel weakened at feather ' + amounts[k]);
}

process.stdout.write(JSON.stringify({
    failures,
    metrics: {
        detailPixels: count(details),
        detailPixelsAt100: count(run(details, w, h, 100)),
        barPixels: count(bar),
        barPixelsAt100: count(barThin),
        ringPixels: count(ring),
        ringPixelsAt100: count(ringThin),
        barMass: series.map(mass),
    },
}));
