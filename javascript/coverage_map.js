// CNPro OUTPUT WEIGHT COVERAGE -- one static picture of what every enabled unit
// together will do to the output, drawn in OUTPUT geometry.
//
// WHY IT EXISTS
// -------------
// A unit's spatial half (its painted weight masks) is only ever seen on the
// canvas it was painted on: an INPUT canvas, at that input's resolution, under
// that unit's resize mode, one unit at a time. What actually reaches the image
// is the SUM of all of them after each has been mapped onto the output
// rectangle. Those two pictures differ by exactly the things that are worth
// seeing and cannot be seen anywhere else:
//
//   * a region no unit covers (nothing steers it, and nothing says so);
//   * a region several units cover at full strength (weights add - two units
//     at 1 make 2, which is where control starts fighting the sampler);
//   * a region that only LOOKS covered, because that unit's resize mode crops
//     or letterboxes the paint out of the frame.
//
// So this panel aggregates the same numbers the generation will use, in the
// same geometry, and colours them in the same hue ramp the masks are painted
// in (violet 0 ... red 1, weight_mask.js weightToRgb). Above 1 the ramp has
// nowhere left to go, so oversaturation is carried by the CONTOURS: 0, 0.25,
// 0.5 and 0.75 are quiet lines, 1 is orange, and every 0.25 above 1 is red.
//
// WHAT IS MODELLED, AND WHAT IS NOT
// ---------------------------------
// Modelled: enabled units; every Input that will run (active_canvas.js
// cnetLiveInputs - the same set the generation fans out over, so the multi-
// phase split lands on the same count); the weight profile with its waves,
// mid controls, response exponent and range; band mode (the C/M/F profiles
// times the main profile, each with its own mask, absent bands zero); the
// depth curve as its mean multiplier; the G / C/M/F weight masks; the output
// mask; the unit's resize mode geometry; the output width/height.
// NOT modelled: the unit's own Use-Mask scribble, the balance profile (it
// moves control between cond and uncond, not across the frame), the drift
// (it moves WHERE the depth curve is read, and the mean over depth is what
// this uses), and the preprocessor - a hint that is black everywhere still
// counts as "covered" here, because coverage is about WEIGHT, not content.
// The status line says which units were skipped and why, rather than letting
// an empty map read as "nothing is wrong".
//
// STATIC BY DESIGN: no tools, no painting, nothing here writes to any unit.
// The only interactive part is the BACKDROP - drop an image on it, or insert
// the img2img input / the current output raster - which is context to read the
// map against and never leaves the browser.
(function () {
    'use strict';

    // Profile samples over the relative step range. The map reduces the whole
    // schedule to one number per pixel, so this is only as fine as that
    // reduction needs: 17 samples resolve a wave of up to 4 oscillations well
    // enough for a mean, and the peak pass costs one full-frame pass each.
    const STEPS = 17;
    // Depth samples for the mean per-layer multiplier (same reason).
    const DEPTH_SAMPLES = 9;
    // Contours are traced on a REDUCED grid: at output resolution a feathered
    // mask edge produces a hairball of one-pixel wiggles that reads as noise,
    // and the lines exist to show shape. Coordinates are scaled back up.
    const CONTOUR_MAX_EDGE = 384;
    // Compute budget. The canvas always MATCHES the output resolution (that is
    // the whole point - this is the raster the control will act on), but the
    // arithmetic behind it is capped and the status line says so when it bites.
    const MAX_COMPUTE_PX = 4.2e6;

    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

    // ------------------------------------------------------------ pure math
    //
    // Everything in this block is DOM-free and exported for tests/coverage_js.js.

    /**
     * Where a source raster of sw x sh lands inside a w x h output under one of
     * the three resize modes - the geometry twin of utils.crop_and_resize_image
     * (lib_cnpro/utils.py), which is what actually maps a weight mask onto the
     * generated frame. Same rounding (np.round -> Math.round), same centring
     * (integer halves), same choice of k.
     *
     * Returns {dx, dy, dw, dh}: draw the source at that rectangle of the output
     * canvas. For "Crop and Resize" the rectangle is LARGER than the output and
     * partly off-canvas, which is the crop; for "Resize and Fill" it is smaller
     * and the remainder is the fill.
     */
    function fitRect(sw, sh, w, h, mode) {
        if (mode === 'Just Resize') return {dx: 0, dy: 0, dw: w, dh: h};
        const k0 = h / sh;
        const k1 = w / sw;
        const k = mode === 'Resize and Fill' ? Math.min(k0, k1) : Math.max(k0, k1);
        const dw = Math.round(sw * k);
        const dh = Math.round(sh * k);
        // python pads/crops with `max(0, (a - b) // 2)`, i.e. floor of the
        // half - not a round, and never negative
        if (mode === 'Resize and Fill') {
            return {dx: Math.max(0, Math.floor((w - dw) / 2)),
                    dy: Math.max(0, Math.floor((h - dh) / 2)), dw: dw, dh: dh};
        }
        return {dx: -Math.max(0, Math.floor((dw - w) / 2)),
                dy: -Math.max(0, Math.floor((dh - h) / 2)), dw: dw, dh: dh};
    }

    /**
     * Fold a list of contributions into the two fields the panel can show.
     *
     * A contribution is {mask, w}: `mask` a Uint8Array of npx (0..255, the
     * spatial weight) or null for "the whole frame", and `w` the per-step
     * weight, STEPS long. Contributions ADD - which is exactly what the units
     * do, and what makes the map able to exceed 1.
     *
     *   mean - the average over the sampling schedule: how much this pixel is
     *          steered in total. Two units that take turns read as 1, not 2.
     *   peak - the largest simultaneous sum: how hard the pixel is pulled at
     *          the worst step. This is the oversaturation question, and it is
     *          why the two are computed together rather than one being derived
     *          from the other (max of a sum is not the sum of maxima).
     */
    function aggregate(contributions, npx) {
        const mean = new Float32Array(npx);
        const peak = new Float32Array(npx);
        const scratch = new Float32Array(npx);
        for (const c of contributions) {
            let sum = 0;
            for (let t = 0; t < c.w.length; t++) sum += c.w[t];
            const avg = sum / (c.w.length || 1);
            if (!avg) continue;
            addScaled(mean, c.mask, avg, npx);
        }
        for (let t = 0; t < STEPS; t++) {
            let any = false;
            for (const c of contributions) {
                if (c.w[t]) {
                    any = true;
                    break;
                }
            }
            if (!any) continue;
            scratch.fill(0);
            for (const c of contributions) {
                if (c.w[t]) addScaled(scratch, c.mask, c.w[t], npx);
            }
            for (let i = 0; i < npx; i++) {
                if (scratch[i] > peak[i]) peak[i] = scratch[i];
            }
        }
        return {mean: mean, peak: peak};
    }

    function addScaled(target, mask, scale, npx) {
        if (!mask) {
            for (let i = 0; i < npx; i++) target[i] += scale;
            return;
        }
        const k = scale / 255;
        for (let i = 0; i < npx; i++) {
            const m = mask[i];
            if (m) target[i] += m * k;
        }
    }

    /**
     * The value "Resize and Fill" pads with: np.median of the border samples,
     * truncated to a byte - `np.median(borders, axis=0).astype(dtype)` in
     * utils.crop_and_resize_image.
     *
     * NOT "the middle element". A border has 2w + 2h samples, always even, and
     * numpy AVERAGES the two middle values there; `.astype` then truncates.
     * Taking the upper middle instead filled the letterbox of a half-painted
     * mask with 255 where the generation fills it with 127 - the map claimed
     * control over two bands the source does not reach, and said 62% of the
     * frame was covered by paint that covers 37%.
     */
    function borderMedian(values) {
        if (!values.length) return 0;
        const sorted = Array.prototype.slice.call(values).sort((a, b) => a - b);
        const mid = sorted.length >> 1;
        return sorted.length % 2
            ? sorted[mid]
            : Math.trunc((sorted[mid - 1] + sorted[mid]) / 2);
    }

    /** Box-average `field` down to at most `maxEdge` on its long side, so the
     *  contour tracer sees shape rather than feather noise. Returns the field
     *  itself when it is already small enough. */
    function reduceField(field, w, h, maxEdge) {
        const factor = Math.ceil(Math.max(w, h) / maxEdge);
        if (factor <= 1) return {field: field, w: w, h: h, factor: 1};
        const rw = Math.max(1, Math.floor(w / factor));
        const rh = Math.max(1, Math.floor(h / factor));
        const out = new Float32Array(rw * rh);
        for (let y = 0; y < rh; y++) {
            const y0 = y * factor;
            const y1 = Math.min(h, y0 + factor);
            for (let x = 0; x < rw; x++) {
                const x0 = x * factor;
                const x1 = Math.min(w, x0 + factor);
                let sum = 0;
                let n = 0;
                for (let yy = y0; yy < y1; yy++) {
                    const row = yy * w;
                    for (let xx = x0; xx < x1; xx++) {
                        sum += field[row + xx];
                        n++;
                    }
                }
                out[y * rw + x] = n ? sum / n : 0;
            }
        }
        return {field: out, w: rw, h: rh, factor: factor};
    }

    /**
     * Marching squares at one level. `emit(x0, y0, x1, y1)` receives every
     * segment in GRID coordinates (cell centres), so the caller scales.
     * Saddle cases are split the same way in both directions - which of the two
     * a saddle takes is arbitrary and invisible at contour widths.
     */
    function traceContour(field, w, h, level, emit) {
        const at = (x, y) => field[y * w + x];
        // linear interpolation along a cell edge: where the level crosses
        const ix = (v0, v1) => (level - v0) / ((v1 - v0) || 1e-9);
        for (let y = 0; y < h - 1; y++) {
            for (let x = 0; x < w - 1; x++) {
                const a = at(x, y), b = at(x + 1, y), c = at(x + 1, y + 1), d = at(x, y + 1);
                const code = (a > level ? 1 : 0) | (b > level ? 2 : 0)
                    | (c > level ? 4 : 0) | (d > level ? 8 : 0);
                if (code === 0 || code === 15) continue;
                // crossing points on the four edges (top, right, bottom, left)
                const top = {x: x + ix(a, b), y: y};
                const right = {x: x + 1, y: y + ix(b, c)};
                const bottom = {x: x + ix(d, c), y: y + 1};
                const left = {x: x, y: y + ix(a, d)};
                const line = (p, q) => emit(p.x, p.y, q.x, q.y);
                switch (code) {
                    case 1: case 14: line(left, top); break;
                    case 2: case 13: line(top, right); break;
                    case 3: case 12: line(left, right); break;
                    case 4: case 11: line(right, bottom); break;
                    case 5: line(left, top); line(right, bottom); break;
                    case 6: case 9: line(top, bottom); break;
                    case 7: case 8: line(left, bottom); break;
                    case 10: line(top, right); line(left, bottom); break;
                    default: break;
                }
            }
        }
    }

    /** The contour levels of a field, given its maximum: the four fixed ones,
     *  the orange 1, and one red line per 0.25 of oversaturation above it. */
    function levelsFor(maxValue) {
        const out = [
            // not 0: a level exactly at the field's floor has no crossing at
            // all. This is the edge of "any control reaches here", which is
            // the question the 0 contour is asked.
            {value: 1e-3, color: 'rgba(255, 255, 255, 0.75)', width: 1, label: '0'},
            {value: 0.25, color: 'rgba(20, 20, 20, 0.55)', width: 1, label: '0.25'},
            {value: 0.5, color: 'rgba(20, 20, 20, 0.7)', width: 1.2, label: '0.5'},
            {value: 0.75, color: 'rgba(20, 20, 20, 0.85)', width: 1.4, label: '0.75'},
            {value: 1, color: '#ff9800', width: 2.2, label: '1'},
        ];
        for (let v = 1.25; v <= Math.max(maxValue, 0) + 1e-6 && v <= 8; v += 0.25) {
            out.push({value: v, color: '#ff1744', width: 2, label: '>1'});
        }
        return out;
    }

    /**
     * Weight -> RGB. Up to 1 this IS the mask ramp (violet 0 ... red 1), so a
     * colour means the same thing here and on the canvas it was painted on.
     * Past 1 the ramp is spent, and stopping at red would make 1.0 and 3.0
     * identical; the red DARKENS instead, so oversaturation reads as burnt
     * before a single contour is looked at.
     */
    function colorFor(v, ramp) {
        if (v <= 1) return ramp(clamp(v, 0, 1));
        const over = clamp(v - 1, 0, 1);
        const k = 1 - 0.55 * over;
        const base = ramp(1);
        return [Math.round(base[0] * k), Math.round(base[1] * k), Math.round(base[2] * k)];
    }

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = {
            fitRect: fitRect,
            borderMedian: borderMedian,
            aggregate: aggregate,
            reduceField: reduceField,
            traceContour: traceContour,
            levelsFor: levelsFor,
            colorFor: colorFor,
            STEPS: STEPS,
        };
    }
    if (typeof onUiUpdate !== 'function') {
        // Same guard as weight_mask.js, and said out loud for the same reason:
        // in node this is the point, in a browser it means the panel is a blank
        // canvas with no explanation anywhere.
        if (typeof window !== 'undefined' && window.document) {
            console.error('[cnpro coverage] onUiUpdate is not available, so the coverage '
                + 'panel registered nothing: it will stay empty. This module expects to '
                + 'run inside the webui.');
        }
        return;
    }

    // --------------------------------------------------------------- reading
    //
    // Everything below reads the live DOM. Every rule it needs that another
    // module also needs is imported from that module, never re-derived:
    // cnetLiveInputs (active_canvas.js), the profile grammar
    // (cnproWeightProfile), the mask codec and hue ramp (cnproWeightMask), the
    // insert sources (cnetInsertSources).

    function api(name, what) {
        const value = window[name];
        if (!value) {
            console.warn('[cnpro coverage] window.' + name + ' is missing, so ' + what
                + ' cannot be read; the coverage map will be incomplete.');
        }
        return value;
    }

    /** A bare, canvas-less editor - the parser and evaluator only. Mirrors
     *  tests/profile_parity_js.js: the class wires a canvas in its constructor
     *  and the grammar half needs none of it. */
    function bareEditor() {
        const wp = api('cnproWeightProfile', 'the weight profiles');
        if (!wp) return null;
        const ed = Object.create(wp.WeightProfileEditor.prototype);
        ed.isBalance = false;
        return ed;
    }

    /** Point the bare editor at one parsed curve and hand back its sampler.
     *  `count` is the Input count the wave is divided between and `name` the
     *  slot the curve occupies (waveCountOf gates the fan-out on it). */
    function sampler(ed, profile, count, name) {
        ed.points = profile.points;
        ed.cosOn = !!profile.cosOn;
        ed.cosN = profile.cosN || 0;
        ed.cosPhase = profile.cosPhase || 0;
        ed.gamma = profile.gamma || 1;
        ed.phaseFamily = profile.phaseFamily || null;
        ed.kappa = profile.kappa || 0;
        ed.converge = profile.converge || null;
        ed.band = name || 'main';
        ed._phaseCount = count || 1;
        return (x, index) => ed.gammaAt(ed.envelopeAt(x) * ed.waveFactor(x, index));
    }

    /** Effective (range-mapped) values of one curve over the step range. */
    function sampleCurve(ed, profile, lo, hi, count, index, name) {
        const curve = sampler(ed, profile, count, name);
        const out = new Float32Array(STEPS);
        for (let i = 0; i < STEPS; i++) {
            const x = i / (STEPS - 1);
            out[i] = lo + curve(x, index) * (hi - lo);
        }
        return out;
    }

    /** Mean of the depth curve over the UNet depth axis: the scalar this map
     *  reduces the per-layer multiplier to. A depth curve that halves the deep
     *  layers and leaves the shallow ones alone is 0.75 of a unit here, which
     *  is what "how much does this unit weigh in total" means. */
    function depthScale(ed, packed) {
        if (!packed.depth) return 1;
        const curve = sampler(ed, packed.depth, 1, 'depth');
        let sum = 0;
        for (let i = 0; i < DEPTH_SAMPLES; i++) {
            const d = i / (DEPTH_SAMPLES - 1);
            sum += packed.depthLo + curve(d, 0) * (packed.depthHi - packed.depthLo);
        }
        return sum / DEPTH_SAMPLES;
    }

    // The panel's root elem_id is `<tab prefix>_coverage` (coverage.py), and
    // the tab prefix is `<gen>_controlnet` - so the units row and the
    // width/height sliders are both derivable from it. A hidden element
    // carrying the same two strings in data-* attributes would be one more
    // thing to keep in step, and an element that exists only to be read is the
    // dead-chrome shape the toolbar audit exists to catch.
    function tabPrefix(panel) {
        return String(panel.id || '').replace(/_coverage$/, '');
    }

    function genOf(panel) {
        return tabPrefix(panel).split('_')[0] || 'txt2img';
    }

    function tabRoot(panel) {
        return document.getElementById(tabPrefix(panel) + '_accordions');
    }

    function outputSize(panel) {
        const gen = genOf(panel);
        const read = (name, fallback) => {
            const box = document.querySelector('#' + gen + '_' + name + ' input[type="number"]');
            const v = box ? parseFloat(box.value) : NaN;
            return isFinite(v) && v > 0 ? Math.round(v) : fallback;
        };
        return {w: read('width', 1024), h: read('height', 1024)};
    }

    function textareaValue(root, selector) {
        const node = root && root.querySelector(selector);
        return node ? String(node.value || '') : '';
    }

    /** Cheap identity of a channel's payload: masks are megabytes of base64 and
     *  the panel compares them twice a second. Length plus a tail slice moves
     *  on any real edit (the tail is the PNG's own trailing bytes). */
    function channelKey(value) {
        return value ? value.length + '~' + value.slice(-24) : '';
    }

    // ------------------------------------------------------------ mask decode

    const maskCache = new Map(); // key -> Uint8Array (npx) | null

    /**
     * Decode one painted mask channel and map it into output geometry.
     *
     * The wire format is what weight_mask.js exports and scripts/cnpro.py
     * decodes: grayscale value = weight, alpha = paint coverage, so the weight
     * is value * alpha and unpainted pixels are 0. Legacy chromatic masks
     * (the old rainbow wire format) fall back to the hue decode, exactly as
     * decode_weight_mask does - via the painter's OWN colorToWeight, so the
     * two cannot drift apart.
     *
     * Geometry is the resize mode's (fitRect), including the "Resize and Fill"
     * border colour, which python takes as the median of the source's border
     * pixels BEFORE resampling. For a mask that is almost always 0 (unpainted
     * edges) - but almost always is not always, and a mask painted to the edge
     * fills with what it is painted with.
     */
    function decodeMask(value, mode, w, h) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                try {
                    resolve(mapMask(img, mode, w, h));
                } catch (err) {
                    console.warn('[cnpro coverage] a weight mask could not be read', err);
                    resolve(null);
                }
            };
            img.onerror = () => {
                console.warn('[cnpro coverage] a weight mask channel did not decode; '
                    + 'that unit is drawn without it');
                resolve(null);
            };
            img.src = value;
        });
    }

    function mapMask(img, mode, w, h) {
        const sw = img.naturalWidth;
        const sh = img.naturalHeight;
        if (!sw || !sh) return null;
        // 1. decode to weights at SOURCE resolution
        const src = document.createElement('canvas');
        src.width = sw;
        src.height = sh;
        const sctx = src.getContext('2d');
        sctx.drawImage(img, 0, 0);
        const data = sctx.getImageData(0, 0, sw, sh);
        const px = data.data;
        const wm = api('cnproWeightMask', 'the mask codec');
        const toWeight = wm ? wm.colorToWeight : null;
        let chromatic = false;
        for (let i = 0; i < px.length; i += 4) {
            if (px[i + 3] < 128) continue;
            const hi = Math.max(px[i], px[i + 1], px[i + 2]);
            const lo = Math.min(px[i], px[i + 1], px[i + 2]);
            if (hi - lo > 2) {
                chromatic = true;
                break;
            }
        }
        let painted = false;
        for (let i = 0; i < px.length; i += 4) {
            const alpha = px[i + 3] / 255;
            let v = 0;
            if (alpha > 0) {
                painted = true;
                v = chromatic && toWeight
                    ? toWeight(px[i], px[i + 1], px[i + 2])
                    : px[i] / 255;
                v *= alpha;
            }
            const g = Math.round(clamp(v, 0, 1) * 255);
            px[i] = px[i + 1] = px[i + 2] = g;
            px[i + 3] = 255;
        }
        if (!painted) return null; // nothing painted = no restriction at all
        sctx.putImageData(data, 0, 0);

        // 2. the fill colour "Resize and Fill" pads with, taken from the SOURCE
        //    border BEFORE any resampling, exactly as crop_and_resize_image
        //    takes it (see borderMedian for the half-pixel of arithmetic that
        //    is not optional here)
        let fill = 0;
        if (mode === 'Resize and Fill') {
            const border = [];
            for (let x = 0; x < sw; x++) {
                border.push(px[x * 4], px[((sh - 1) * sw + x) * 4]);
            }
            for (let y = 0; y < sh; y++) {
                border.push(px[(y * sw) * 4], px[(y * sw + sw - 1) * 4]);
            }
            fill = borderMedian(border);
        }

        // 3. into output geometry
        const out = document.createElement('canvas');
        out.width = w;
        out.height = h;
        const octx = out.getContext('2d');
        octx.fillStyle = 'rgb(' + fill + ',' + fill + ',' + fill + ')';
        octx.fillRect(0, 0, w, h);
        const r = fitRect(sw, sh, w, h, mode);
        octx.drawImage(src, r.dx, r.dy, r.dw, r.dh);
        const mapped = octx.getImageData(0, 0, w, h).data;
        const mask = new Uint8Array(w * h);
        for (let i = 0, p = 0; i < mask.length; i++, p += 4) mask[i] = mapped[p];
        return mask;
    }

    function cachedMask(value, mode, w, h) {
        if (!value || !value.startsWith('data:image')) return Promise.resolve(null);
        const key = channelKey(value) + '|' + mode + '|' + w + 'x' + h;
        if (maskCache.has(key)) return Promise.resolve(maskCache.get(key));
        return decodeMask(value, mode, w, h).then((mask) => {
            // one entry per (channel, geometry); the map is swept when it grows
            // past a stack's worth so a long session cannot accumulate frames
            if (maskCache.size > 48) maskCache.clear();
            maskCache.set(key, mask);
            return mask;
        });
    }

    /** Elementwise product of two masks (either may be null = all ones). */
    function foldMask(a, b, npx) {
        if (!a) return b;
        if (!b) return a;
        const out = new Uint8Array(npx);
        for (let i = 0; i < npx; i++) out[i] = (a[i] * b[i]) / 255;
        return out;
    }

    // ---------------------------------------------------------- one unit

    const BANDS = ['coarse', 'mid', 'fine'];

    /**
     * Every contribution one unit makes, resolved to (spatial mask, per-step
     * weight) pairs. Returns {contributions, note} - `note` is what the status
     * line has to say about this unit when it contributes nothing, because a
     * unit that is enabled and invisible on the map is exactly the thing this
     * panel must not pass over in silence.
     */
    function unitContributions(unit, grid, gen, index) {
        const enabled = unit.querySelector('.cnet-unit-enabled input');
        if (!enabled || !enabled.checked) return Promise.resolve({contributions: []});

        // named by POSITION, not by scraping the accordion header: the header
        // text is decorated (active-unit badge, control type) by active_units.js
        const label = 'unit ' + index;

        const ed = bareEditor();
        if (!ed) return Promise.resolve({contributions: []});
        const text = textareaValue(unit, '[id$="_controlnet_weight_profile"] textarea');
        const packed = ed.parsePacked(text);
        if (!packed || !packed.main) {
            return Promise.resolve({contributions: [], note: label + ': no weight profile'});
        }

        const mode = modeOf(unit);
        let live = (window.cnetLiveInputs ? window.cnetLiveInputs(unit) : []).slice();
        if (!live.length) {
            // img2img falls back to the main img2img image when a unit's canvas
            // is empty (scripts/cnpro.py get_input_data), so the unit DOES run -
            // with no masks, since there is no canvas to have painted them on.
            if (gen === 'img2img' && window.cnetInsertSources
                    && window.cnetInsertSources.img2imgSourceImg()) {
                live = [{slot: 0, fallback: true}];
            } else {
                return Promise.resolve({
                    contributions: [],
                    note: label + ': enabled with no input image',
                });
            }
        }

        const lo = packed.scaleLo;
        const hi = packed.scaleHi;
        const count = live.length;
        const bandMode = BANDS.indexOf(packed.selected) !== -1;
        const depth = depthScale(ed, packed);
        const npx = grid.w * grid.h;

        // Per-input, per-slot weights first (pure math, no decoding yet), then
        // the masks they need. Each Input is a control of its own and their
        // residuals are SUMMED, so a unit with two Inputs and no wave pulls
        // twice - which is the first thing this map is for.
        const specs = [];
        live.forEach((input, phase) => {
            const main = sampleCurve(ed, packed.main, lo, hi, count, phase, 'main');
            if (!bandMode) {
                specs.push({
                    slot: input.slot,
                    fallback: !!input.fallback,
                    band: 'global',
                    scale: depth,
                    weights: main,
                });
                return;
            }
            for (const band of BANDS) {
                const w = new Float32Array(STEPS);
                const curve = packed.bands && packed.bands[band]
                    ? sampleCurve(ed, packed.bands[band], lo, hi, count, phase, band)
                    : null;
                for (let t = 0; t < STEPS; t++) {
                    // a band absent from the string is neutral 1, exactly as
                    // external_code.parse_band_profiles leaves it
                    w[t] = main[t] * (curve ? curve[t] : 1);
                }
                specs.push({
                    slot: input.slot,
                    fallback: !!input.fallback,
                    band: band,
                    // the three bands drive DIFFERENT injection layers, so the
                    // unit's overall pull on a pixel is their mean, not their
                    // sum - a coarse-only mask does not make the unit weigh
                    // three times what it weighs
                    scale: depth / BANDS.length,
                    weights: w,
                });
            }
        });

        const outputValue = textareaValue(unit, '.cnet-output-mask-state textarea');
        const jobs = specs.map((spec) => {
            const value = spec.fallback ? '' : textareaValue(
                unit, '.cnet-wmask-' + spec.slot + '-' + spec.band + '-state textarea');
            return cachedMask(value, mode, grid.w, grid.h)
                .then((mask) => ({spec: spec, mask: mask}));
        });

        return Promise.all(jobs)
            .then((resolved) => cachedMask(outputValue, 'Just Resize', grid.w, grid.h)
                .then((outputMask) => ({resolved: resolved, outputMask: outputMask})))
            .then((got) => {
                const contributions = [];
                // restrict-to-painted, per input: with NOTHING painted in the
                // live slots the unit acts on the whole frame; with something
                // painted, an absent band is ZERO (scripts/cnpro.py, "absent
                // bands zeroed") rather than neutral
                const paintedBySlot = new Map();
                for (const item of got.resolved) {
                    if (item.mask) paintedBySlot.set(item.spec.slot, true);
                }
                for (const item of got.resolved) {
                    const painted = paintedBySlot.get(item.spec.slot);
                    if (bandMode && painted && !item.mask) continue; // zeroed band
                    const spatial = foldMask(item.mask, got.outputMask, npx);
                    const w = new Float32Array(STEPS);
                    let any = false;
                    for (let t = 0; t < STEPS; t++) {
                        w[t] = item.spec.weights[t] * item.spec.scale;
                        if (w[t]) any = true;
                    }
                    if (!any) continue;
                    contributions.push({mask: spatial, w: w});
                }
                if (!contributions.length) {
                    return {contributions: [], note: label + ': weight 0 everywhere'};
                }
                return {contributions: contributions};
            });
    }

    function modeOf(unit) {
        const box = unit.querySelector('.controlnet_resize_mode_radio input');
        const value = box ? String(box.value || '').trim() : '';
        return value || 'Resize and Fill';
    }

    // ------------------------------------------------------------- the panel

    const panels = new Map(); // panel element -> state

    function stateOf(panel) {
        let st = panels.get(panel);
        if (!st) {
            st = {
                canvas: panel.querySelector('.cnet-coverage-canvas'),
                status: panel.querySelector('.cnet-coverage-status'),
                hint: panel.querySelector('.cnet-coverage-hint'),
                background: null,      // decoded backdrop Image
                backgroundKey: '',
                key: '',
                token: 0,
            };
            panels.set(panel, st);
        }
        return st;
    }

    // ---- the settings column
    //
    // Ordinary gradio components (coverage.py), read straight off the DOM: they
    // are VIEW state, so nothing is gained by a round trip to python and a
    // round trip is what would make the panel lag behind the control that
    // drives it. Each reader states its own default, because a gradio component
    // that has not rendered yet must not silently mean something else.

    function metricOf(panel) {
        const checked = panel.querySelector(
            '.cnet-coverage-metric input[type="radio"]:checked');
        return checked && checked.value === 'peak' ? 'peak' : 'mean';
    }

    /** How strongly the map paints over a backdrop, 0..1. Only applies when
     *  there IS one - over nothing, a half-transparent map is a dimmer map. */
    function mapAlphaOf(panel) {
        const box = panel.querySelector('.cnet-coverage-alpha input[type="number"]');
        const v = box ? parseFloat(box.value) : NaN;
        return isFinite(v) ? clamp(v / 100, 0, 1) : 0.5;
    }

    /** Everything the map depends on, as one string. Recompute exactly when
     *  this moves - a full run decodes every mask and paints a 1-megapixel
     *  canvas, and the tick that calls it runs twice a second. */
    function panelKey(panel, st) {
        const size = outputSize(panel);
        // the settings column is part of the key as well as having its own
        // change listener: the listener is the fast path, this is the one that
        // cannot be missed (a value written by ui-config.json or a paste fires
        // nothing this module hears)
        const parts = [size.w, size.h, metricOf(panel), mapAlphaOf(panel),
                       st.backgroundKey];
        const root = tabRoot(panel);
        if (root) {
            root.querySelectorAll('.input-accordion').forEach((unit) => {
                const enabled = unit.querySelector('.cnet-unit-enabled input');
                parts.push(enabled && enabled.checked ? '1' : '0');
                if (!enabled || !enabled.checked) return;
                parts.push(modeOf(unit));
                parts.push(textareaValue(unit, '[id$="_controlnet_weight_profile"] textarea'));
                parts.push(channelKey(textareaValue(unit, '.cnet-output-mask-state textarea')));
                const live = window.cnetLiveInputs ? window.cnetLiveInputs(unit) : [];
                for (const input of live) {
                    parts.push('i' + input.slot);
                    for (const band of ['global'].concat(BANDS)) {
                        parts.push(channelKey(textareaValue(
                            unit, '.cnet-wmask-' + input.slot + '-' + band + '-state textarea')));
                    }
                }
            });
        }
        return parts.join('');
    }

    /** The grid the arithmetic runs on. The CANVAS always matches the output
     *  resolution - that is the point of the panel - but a 4K output is 8
     *  million pixels times a mask per unit per band, and the map is a
     *  qualitative picture. Past the budget the grid shrinks, keeping the
     *  aspect, and the status line says the number it actually used: a preview
     *  that silently drops resolution is a preview that lies about its edges. */
    function computeGrid(size) {
        const npx = size.w * size.h;
        if (npx <= MAX_COMPUTE_PX) return {w: size.w, h: size.h, scaled: false};
        const k = Math.sqrt(MAX_COMPUTE_PX / npx);
        return {
            w: Math.max(1, Math.round(size.w * k)),
            h: Math.max(1, Math.round(size.h * k)),
            scaled: true,
        };
    }

    function refresh(panel, force) {
        const st = stateOf(panel);
        if (!st.canvas) return;
        if (!force && !window.cnetVisible(panel)) return;
        const key = panelKey(panel, st);
        if (!force && key === st.key) return;
        st.key = key;
        const token = ++st.token;
        if (st.status) st.status.textContent = 'computing...';

        const size = outputSize(panel);
        const grid = computeGrid(size);
        const gen = genOf(panel);
        const root = tabRoot(panel);
        const units = root ? Array.from(root.querySelectorAll('.input-accordion')) : [];
        Promise.all(units.map((unit, i) => unitContributions(unit, grid, gen, i)))
            .then((results) => {
                if (token !== st.token) return; // superseded by a newer run
                const contributions = [];
                const notes = [];
                for (const r of results) {
                    for (const c of r.contributions) contributions.push(c);
                    if (r.note) notes.push(r.note);
                }
                paint(panel, st, size, grid, contributions, notes);
            })
            .catch((err) => {
                console.error('[cnpro coverage] the map could not be computed', err);
                // ...and forget the key, so the next edit (or the next tick
                // after whatever broke is fixed) tries again instead of the
                // panel staying frozen on a stale picture forever
                st.key = '';
                if (st.status) {
                    st.status.textContent = 'could not be computed - see the browser console';
                }
            });
    }

    function paint(panel, st, size, grid, contributions, notes) {
        const wm = api('cnproWeightMask', 'the hue ramp');
        const ramp = wm ? wm.weightToRgb : ((v) => [255 * v, 0, 255 * (1 - v)]);
        const canvas = st.canvas;
        // The canvas IS the output raster: same dimensions, so a pixel here is
        // a pixel there and the CSS box merely scales it for display.
        if (canvas.width !== size.w || canvas.height !== size.h) {
            canvas.width = size.w;
            canvas.height = size.h;
        }
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, size.w, size.h);

        const cw = grid.w;
        const ch = grid.h;
        const scaled = grid.scaled;
        const metric = metricOf(panel);
        const both = aggregate(contributions, cw * ch);
        const field = metric === 'peak' ? both.peak : both.mean;

        let max = 0;
        let covered = 0;
        let over = 0;
        for (let i = 0; i < field.length; i++) {
            const v = field[i];
            if (v > max) max = v;
            if (v > 1e-3) covered++;
            if (v > 1 + 1e-3) over++;
        }

        // 1. the backdrop, fitted inside the output rectangle (no distortion)
        if (st.background) {
            const bg = st.background;
            const r = fitRect(bg.naturalWidth, bg.naturalHeight, size.w, size.h,
                              'Resize and Fill');
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, size.w, size.h);
            ctx.drawImage(bg, r.dx, r.dy, r.dw, r.dh);
        }

        // 2. the weight map itself
        const layer = document.createElement('canvas');
        layer.width = cw;
        layer.height = ch;
        const lctx = layer.getContext('2d');
        const image = lctx.createImageData(cw, ch);
        const out = image.data;
        // 256-entry ramp lookup: colorFor allocates a small array per call and
        // this runs once per pixel, which at a megapixel is a million of them
        const lut = new Uint8Array(256 * 3);
        for (let i = 0; i < 256; i++) {
            const c = colorFor(i / 255, ramp);
            lut[i * 3] = c[0];
            lut[i * 3 + 1] = c[1];
            lut[i * 3 + 2] = c[2];
        }
        for (let i = 0, p = 0; i < field.length; i++, p += 4) {
            const v = field[i];
            let r, g, b;
            if (v <= 1) {
                const k = (v <= 0 ? 0 : Math.round(v * 255)) * 3;
                r = lut[k];
                g = lut[k + 1];
                b = lut[k + 2];
            } else {
                const c = colorFor(v, ramp);   // the burnt end: far fewer pixels
                r = c[0];
                g = c[1];
                b = c[2];
            }
            out[p] = r;
            out[p + 1] = g;
            out[p + 2] = b;
            out[p + 3] = 255;
        }
        lctx.putImageData(image, 0, 0);
        // over a backdrop the map is translucent (the picture underneath is
        // context and has to stay readable through it) - how translucent is the
        // settings column's slider. Over NOTHING it is opaque: fading it there
        // would only wash the colours out against the panel.
        ctx.globalAlpha = st.background ? mapAlphaOf(panel) : 1;
        ctx.imageSmoothingEnabled = scaled;
        ctx.drawImage(layer, 0, 0, size.w, size.h);
        ctx.globalAlpha = 1;

        // 3. contours. Not optional: they are the only thing that can say
        //    "above 1" at all - the hue ramp has nowhere left to go past red,
        //    so a map without them cannot distinguish full strength from
        //    oversaturation, which is half of what the panel is for.
        const reduced = reduceField(field, cw, ch, CONTOUR_MAX_EDGE);
        const sx = size.w / reduced.w;
        const sy = size.h / reduced.h;
        // line widths are stated for a 512px frame and scale with the output,
        // so a 2048px map does not draw hairlines
        const lineUnit = Math.max(1, Math.min(size.w, size.h) / 512);
        for (const level of levelsFor(max)) {
            const path = new Path2D();
            let any = false;
            traceContour(reduced.field, reduced.w, reduced.h, level.value,
                (x0, y0, x1, y1) => {
                    any = true;
                    path.moveTo((x0 + 0.5) * sx, (y0 + 0.5) * sy);
                    path.lineTo((x1 + 0.5) * sx, (y1 + 0.5) * sy);
                });
            if (!any) continue;
            ctx.strokeStyle = level.color;
            ctx.lineWidth = level.width * lineUnit;
            ctx.lineCap = 'round';
            ctx.stroke(path);
        }

        if (st.hint) st.hint.style.display = st.background ? 'none' : '';
        if (st.status) {
            const pct = (n) => (100 * n / field.length).toFixed(1) + '%';
            const bits = [
                contributions.length + (contributions.length === 1
                    ? ' contribution' : ' contributions'),
                size.w + '×' + size.h,
                metric === 'peak' ? 'peak over steps' : 'mean over steps',
                'max ' + max.toFixed(2),
                pct(field.length - covered) + ' uncovered',
                pct(over) + ' above 1',
            ];
            if (scaled) bits.push('computed at ' + cw + '×' + ch);
            if (notes.length) bits.push(notes.join('; '));
            st.status.textContent = bits.join(' · ');
        }
    }

    // ------------------------------------------------------------- backdrop

    function setBackground(panel, dataUrl, key) {
        const st = stateOf(panel);
        if (!dataUrl) {
            st.background = null;
            st.backgroundKey = '';
            refresh(panel, true);
            return;
        }
        const img = new Image();
        img.onload = () => {
            st.background = img;
            st.backgroundKey = key || channelKey(dataUrl);
            refresh(panel, true);
        };
        img.onerror = () => {
            console.warn('[cnpro coverage] the backdrop image did not decode');
        };
        img.src = dataUrl;
    }

    function insertSource(panel, which) {
        const sources = api('cnetInsertSources', 'the insert buttons');
        if (!sources) return;
        const st = stateOf(panel);
        const gen = panel.dataset.cnetGen || 'txt2img';
        const img = which === 'input'
            ? sources.img2imgSourceImg()
            : sources.outputImg(gen === 'img2img');
        if (!img) {
            // a button that does nothing reads as a broken button. The unit's
            // own ⤵I/⤵O say this by being disabled; these are plain buttons in
            // an HTML block, so they say it in the status line instead.
            if (st.status) {
                st.status.textContent = which === 'input'
                    ? 'no img2img input image to use as a backdrop'
                    : 'no output image yet - generate something first';
            }
            return;
        }
        sources.toPngDataUrl(img, (dataUrl) => {
            if (dataUrl) setBackground(panel, dataUrl);
        });
    }

    function wire(panel) {
        if (panel.dataset.cnetCoverageWired === '1') return;
        panel.dataset.cnetCoverageWired = '1';

        // The settings are gradio components, so they are DELEGATED rather than
        // bound one by one: gradio re-renders the inner input of a component
        // freely, and a listener bound to the node that exists now would be
        // wired to a node that no longer receives the events. The panel row
        // itself outlives that (and panelKey re-reads them on the tick anyway,
        // so a missed event costs half a second, not correctness).
        const options = panel.querySelector('.cnet-coverage-options') || panel;
        let settingsTimer = 0;
        const settingsChanged = () => {
            // a slider drag is a stream of `input` events and a full run is a
            // megapixel of arithmetic; coalesce them into one run per pause
            clearTimeout(settingsTimer);
            settingsTimer = setTimeout(() => refresh(panel, true), 180);
        };
        ['change', 'input'].forEach((type) => {
            options.addEventListener(type, settingsChanged);
        });

        const on = (selector, fn) => {
            const node = panel.querySelector(selector);
            if (node) node.addEventListener('click', fn);
        };
        on('.cnet-coverage-insert-input', () => insertSource(panel, 'input'));
        on('.cnet-coverage-insert-output', () => insertSource(panel, 'output'));
        on('.cnet-coverage-clear-bg', () => setBackground(panel, null));
        on('.cnet-coverage-refresh', () => {
            maskCache.clear();
            refresh(panel, true);
        });

        // drag & drop a backdrop straight onto the map
        const stage = panel.querySelector('.cnet-coverage-stage') || panel;
        ['dragenter', 'dragover'].forEach((type) => {
            stage.addEventListener(type, (e) => {
                e.preventDefault();
                e.stopPropagation();
                stage.classList.add('cnet-coverage-drop');
            });
        });
        ['dragleave', 'dragend'].forEach((type) => {
            stage.addEventListener(type, () => stage.classList.remove('cnet-coverage-drop'));
        });
        stage.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            stage.classList.remove('cnet-coverage-drop');
            const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (file && /^image\//.test(file.type)) {
                const reader = new FileReader();
                reader.onload = () => setBackground(panel, String(reader.result));
                reader.readAsDataURL(file);
                return;
            }
            // a drag from another part of the page (a gallery thumbnail, an
            // Input canvas) carries a url rather than a file
            const url = e.dataTransfer && (e.dataTransfer.getData('text/uri-list')
                || e.dataTransfer.getData('text/plain'));
            if (!url) return;
            const sources = api('cnetInsertSources', 'the backdrop drop');
            if (!sources) return;
            sources.toPngDataUrl({src: url}, (dataUrl) => {
                if (dataUrl) setBackground(panel, dataUrl);
            });
        });

        // enabling a unit / editing a profile is a DOM event away, but painting
        // a mask is a value-only textarea write nobody hears: the shared tick is
        // the guaranteed channel (one string compare per unit when idle)
        refresh(panel, true);
    }

    onUiUpdate(() => {
        gradioApp().querySelectorAll('.cnet-coverage-panel').forEach(wire);
    });

    window.cnetRegisterTick(() => {
        for (const panel of panels.keys()) {
            if (!panel.isConnected) {
                panels.delete(panel);
                continue;
            }
            refresh(panel, false);
        }
    });
})();
