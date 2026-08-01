// Node harness for tests/test_coverage_map.py: the coverage panel's PURE math,
// with no DOM anywhere near it.
//
// javascript/coverage_map.js exports its geometry, aggregation, contour tracer
// and colour ramp before it touches the page, and returns early when the webui
// globals are absent (the same headless-load guard weight_mask.js uses). So
// requiring it here runs the arithmetic under test and registers nothing.
//
// stdin:  {"fits": [{"sw":..,"sh":..,"w":..,"h":..,"mode":".."}, ...],
//          "medians": [[border samples], ...],
//          "aggregates": [{"npx":n, "contributions":[{"mask":[..]|null,
//                                                     "w":[..]}, ...]}, ...],
//          "contour": {"field":[..], "w":n, "h":n, "level":x},
//          "levels": [maxValue, ...]}
// stdout: the same keys, answered.

const cov = require('../javascript/coverage_map.js');

let raw = '';
process.stdin.on('data', (chunk) => { raw += chunk; });
process.stdin.on('end', () => {
    const input = JSON.parse(raw);

    const fits = (input.fits || []).map(
        (f) => cov.fitRect(f.sw, f.sh, f.w, f.h, f.mode));

    const medians = (input.medians || []).map((values) => cov.borderMedian(values));

    const aggregates = (input.aggregates || []).map((spec) => {
        const contributions = spec.contributions.map((c) => ({
            mask: c.mask ? Uint8Array.from(c.mask) : null,
            // every contribution carries one weight per sampled step; the
            // python side sends the full STEPS-long array so the two agree on
            // what a "mean over the schedule" divides by
            w: Float32Array.from(c.w),
        }));
        const out = cov.aggregate(contributions, spec.npx);
        return {mean: Array.from(out.mean), peak: Array.from(out.peak)};
    });

    let contour = null;
    if (input.contour) {
        const spec = input.contour;
        const segments = [];
        cov.traceContour(Float32Array.from(spec.field), spec.w, spec.h, spec.level,
            (x0, y0, x1, y1) => segments.push([x0, y0, x1, y1]));
        contour = segments;
    }

    const levels = (input.levels || []).map(
        (max) => cov.levelsFor(max).map((l) => ({value: l.value, color: l.color})));

    const ramp = (v) => [Math.round(255 * v), 0, Math.round(255 * (1 - v))];
    const colors = (input.colors || []).map((v) => cov.colorFor(v, ramp));

    process.stdout.write(JSON.stringify({
        steps: cov.STEPS,
        fits: fits,
        medians: medians,
        aggregates: aggregates,
        contour: contour,
        levels: levels,
        colors: colors,
    }));
});
