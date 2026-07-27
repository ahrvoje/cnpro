// Forge Canvas - adjustment controls (layers / horizontal flip / black-white point / gamma / crop / fine rotation / pen+eraser)
// Companion to canvas.min.js. The originally uploaded content is kept and every
// adjustment is re-rendered from it, so all edits are non-destructive until a
// new image is uploaded or the widget is cleared. Attaches to every
// upload-enabled ForgeCanvas widget (ControlNet single image and mask,
// img2img/inpaint canvases, ...); preview-only widgets (no_upload) are skipped.
//
// UI principle: the toolbar shows action buttons only; each button reveals its
// own control when toggled (layer list, color correction menu, gamma slider,
// rotation slider, crop handles, pick-pixel reticle). The black/white/gray
// point pickers live inside the color correction menu together with swatch +
// RGB readouts of the picked points and their shared reset; the legacy
// black/white slider box stays in the template but has no toggle anymore.
//
// LAYERS are the very start of the pipeline. The canvas holds a STAGE (its
// dimensions come from the first upload and outlive it) and an ordered list
// of layers, each with its own bitmap, position, scale and stroke list.
// Every layer stays continuously editable: selectable (layer menu or click),
// movable (drag), scalable (wheel), reorderable and deletable, in any order,
// at any time. The pen draws INTO the active layer and the eraser removes
// from it (to transparency, so lower layers show through); both live in
// layer-local pixels, so paint follows its layer through move/scale.
// The composite of all layers is what every downstream tool consumes:
// rotation, flip, levels, gamma, grayscale, the EDGES tool, invert and crop
// all operate on the totality of the composed content - a single image still
// is just the degenerate one-layer case, which keeps the historical behavior
// of every tool unchanged. Only the flattened composite ever reaches gradio;
// layer structure is session-scoped editing state (a reload keeps the
// flattened image, not the layers).
// RASTER INFLOW: the + button creates an EMPTY logical layer (no file
// dialog); while the layer tool is open - or any multi-layer stack exists -
// every way an image can arrive (drag & drop, paste, the core open button,
// gradio pushes like the ControlNet insert buttons) fills the ACTIVE layer.
// Only a single-layer canvas with the tool closed keeps the classic
// replace-everything upload.
// LAYER TARGETING: while the layer tool is engaged (showLayers), every
// adjustment control - flip, b/w/gray point pickers, gamma, grayscale,
// invert, edges, fine rotation - reads and writes the ACTIVE LAYER's own
// state (its color pipeline runs on the layer raster, flip/rotation are part
// of its placement); with the tool closed the same controls act on the whole
// canvas as one merged raster, exactly as before. Targeting survives
// picker/pen/crop sub-modes and hands the pointer back to the layer tool when
// they exit. Crop and the Topaz tools always act on the flattened whole.

(function () {
    'use strict';

    if (window.__forgeCanvasAdjustmentsInstalled) return;
    window.__forgeCanvasAdjustmentsInstalled = true;

    // gradio mounts the head scripts dynamically, i.e. async: this file and
    // canvas.min.js can execute in either order, so everything is wrapped in
    // install() which runs once the core ForgeCanvas class has appeared.
    function install() {

    const Orig = ForgeCanvas;
    const proto = Orig.prototype;

    // CNPro: hosts rename these (ForgeNeo: uploadBase64 -> loadImage,
    // on_img_upload -> updateBackgroundImageData). One normalization layer owns
    // every such difference; everything below speaks the canonical names only.
    // See javascript/canvas_adapter.js for why this is a call and not a poller.
    if (!window.cnproCanvasApi) {
        console.warn('[cnpro] canvas_adapter.js did not load - adjustment controls disabled.');
        return;
    }
    if (!window.cnproCanvasApi.normalize(proto).ok) {
        return; // normalize() already reported which methods are missing
    }

    // ---------------------------------------------------------------- helpers

    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    const FULL_CROP = () => ({x: 0, y: 0, w: 1, h: 1});

    // a fresh layer at identity transform; x/y are the layer's top-left in
    // stage pixels, strokes are pen/eraser polylines in LAYER-LOCAL pixels.
    // blend: 'normal' composites over (photos), 'lighten' takes the per-pixel
    // max - the union semantics bright-on-black control maps (canny, pose,
    // depth) need, since their opaque black background would otherwise
    // occlude everything underneath.
    const newLayer = (src) => ({
        src: src, img: null, w: 0, h: 0,
        x: 0, y: 0, scale: 1,
        blend: 'normal',
        strokes: [], strokeKey: 0,
        // per-layer adjustments, same names as the global (whole-canvas) state
        // in st so every control can write to either one through target():
        // color pipeline runs on the layer's own raster, flip/rotate are part
        // of the layer's placement transform
        flipH: false, rotate: 0,
        black: 0, white: 255, gamma: 1,
        grayGains: null, pickedBlack: null, pickedWhite: null, grayPicked: null,
        grayscale: false, invert: false,
        edgeSensitivity: 50, edgeOpacity: 0, maskOpacity: 0,
        edgeThickness: 2, edgeFeather: 0,
        baseCanvas: null, baseKey: null,
        canvas: null, canvasKey: null,
    });

    // everything that invalidates a layer's adjusted raster (not its placement)
    function layerAdjKey(l) {
        const g = l.grayGains;
        return [l.black, l.white, l.gamma, g ? g.r.toFixed(4) + ',' + g.g.toFixed(4) + ',' + g.b.toFixed(4) : '1',
            l.grayscale, l.invert, l.edgeSensitivity, l.edgeOpacity, l.maskOpacity,
            l.edgeThickness, l.edgeFeather].join('|');
    }

    const layerColorAdjusted = (l) => l.black !== 0 || l.white !== 255 || l.gamma !== 1
        || !!l.grayGains || l.grayscale || l.invert || edgesActive(l);

    // ---- layer placement transform: local (base raster) px <-> stage px.
    // Must mirror the composite draw exactly: scale about the layer's center,
    // then rotate, then horizontal flip (the same operation order the global
    // geometry canvas uses), then translate to the layer's position.
    function layerLocalToStage(L, p) {
        const cx = L.x + L.w * L.scale / 2, cy = L.y + L.h * L.scale / 2;
        const dx = (p.x - L.w / 2) * L.scale, dy = (p.y - L.h / 2) * L.scale;
        const rad = (L.rotate || 0) * Math.PI / 180;
        const cos = Math.cos(rad), sin = Math.sin(rad);
        let gx = dx * cos - dy * sin;
        const gy = dx * sin + dy * cos;
        if (L.flipH) gx = -gx;
        return {x: cx + gx, y: cy + gy};
    }

    function stageToLayerLocal(L, p) {
        const cx = L.x + L.w * L.scale / 2, cy = L.y + L.h * L.scale / 2;
        let dx = p.x - cx;
        const dy = p.y - cy;
        if (L.flipH) dx = -dx;
        const rad = (L.rotate || 0) * Math.PI / 180;
        const cos = Math.cos(rad), sin = Math.sin(rad);
        const ux = dx * cos + dy * sin;
        const uy = -dx * sin + dy * cos;
        const s = Math.max(1e-6, L.scale);
        return {x: ux / s + L.w / 2, y: uy / s + L.h / 2};
    }

    // geom-canvas px -> stage px: the inverse of the global rotate/flip that
    // ensureGeomCanvas applies (the pickers work in geom space, the layers in
    // stage space)
    function geomToStage(st, gx, gy, geomW, geomH) {
        const rad = st.rotate * Math.PI / 180;
        const cos = Math.cos(rad), sin = Math.sin(rad);
        let dx = gx - geomW / 2;
        const dy = gy - geomH / 2;
        if (st.flipH) dx = -dx;
        return {
            x: dx * cos + dy * sin + st.stageW / 2,
            y: -dx * sin + dy * cos + st.stageH / 2,
        };
    }

    // src may be null: an EMPTY layer, a transparent stage-sized raster the
    // user creates first and fills later (by drop/paste/open/gradio push while
    // the layer tool is open) or just paints on. The stack's "original" is the
    // bottom-most raster layer's src (its format is what gets announced); a
    // rasterless stack still counts as content while any stroke exists - pen
    // work on empty layers must survive deletion of the raster layers.
    function stackOriginal(st) {
        for (const l of st.layers) if (l.src) return l.src;
        for (const l of st.layers) if (l.strokes.length) return 'data:image/png;base64,';
        return null;
    }

    // edge thickness slider stops: nonlinear 1-2-5 series so sub-pixel line
    // weights are reachable (the slider value is an index into this list)
    const THICKNESS_STOPS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];

    const thicknessIndex = (t) => {
        let best = 0;
        for (let i = 1; i < THICKNESS_STOPS.length; i++) {
            if (Math.abs(THICKNESS_STOPS[i] - t) < Math.abs(THICKNESS_STOPS[best] - t)) best = i;
        }
        return best;
    };

    // pen brush size: the slider is an exponent, diameter = 2^(v/12.5) image
    // pixels (1 .. 256), so the whole useful range is reachable with usable
    // resolution at both ends
    const brushDiameter = (v) => Math.pow(2, v / 12.5);

    // hsv (h in degrees, s/v in 0..1) -> '#rrggbb'; the pen color selector is
    // a saturation/value square plus a hue strip, exactly like photoshop's
    function hsvToHex(h, s, v) {
        const f = (n) => {
            const k = (n + h / 60) % 6;
            const c = v - v * s * Math.max(0, Math.min(k, 4 - k, 1));
            return Math.round(c * 255).toString(16).padStart(2, '0');
        };
        return '#' + f(5) + f(3) + f(1);
    }

    // inverse of hsvToHex, for the pen's color picker
    function rgbToHsv(r, g, b) {
        r /= 255; g /= 255; b /= 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
        let h = 0;
        if (d > 0) {
            if (max === r) h = 60 * (((g - b) / d) % 6);
            else if (max === g) h = 60 * ((b - r) / d + 2);
            else h = 60 * ((r - g) / d + 4);
            if (h < 0) h += 360;
        }
        return {h: h, s: max > 0 ? d / max : 0, v: max};
    }

    // menu-button principle: a tool button only toggles its menu; the effect
    // itself is active purely by its settings - for edges, any opacity > 0
    const edgesActive = (st) => st.edgeOpacity > 0 || st.maskOpacity > 0;

    // a full-frame crop rect counts as no crop
    const cropApplied = (st) => !!st.crop
        && !(st.crop.x < 0.0005 && st.crop.y < 0.0005 && st.crop.w > 0.999 && st.crop.h > 0.999);

    // decode one layer's bitmap (cached per src)
    function decodeLayerImg(layer, callback) {
        if (layer.img && layer.img.__forgeSrc === layer.src) {
            callback(layer.img);
            return;
        }
        const src = layer.src;
        const img = new Image();
        img.onload = () => {
            if (layer.src !== src) return; // the layer was replaced meanwhile
            img.__forgeSrc = src;
            layer.img = img;
            layer.w = img.naturalWidth;
            layer.h = img.naturalHeight;
            callback(img);
        };
        // a data url that fails to decode would otherwise leave every
        // adjustment silently dead - at least say so in the console
        img.onerror = () => console.warn('[forge canvas] could not decode a stored layer image; adjustments are inactive for it');
        img.src = src;
    }

    // decode every layer, establish the stage dimensions (first upload's
    // size, kept until the content is replaced or cleared), then run the
    // callback. All pipeline entry points funnel through here.
    function ensureSource(st, callback) {
        if (!st.layers.length) return;
        let pending = 0;
        let fired = false;
        const done = () => {
            if (fired) return;
            for (const l of st.layers) {
                if (l.src && (!l.img || l.img.__forgeSrc !== l.src)) return;
            }
            fired = true;
            if (!st.stageW) {
                st.stageW = st.layers[0].w;
                st.stageH = st.layers[0].h;
            }
            callback();
        };
        for (const l of st.layers) {
            if (l.src && (!l.img || l.img.__forgeSrc !== l.src)) {
                pending++;
                decodeLayerImg(l, done);
            }
        }
        if (!pending) done();
    }

    // one layer's own raster: bitmap + its stroke list replayed in order.
    // Paint composites normally; ERASE strokes cut to transparency
    // (destination-out), so they remove image and earlier paint alike and
    // lower layers show through the hole. This BASE is what the per-layer
    // color pipeline and the layer-targeted pickers read - sampling it keeps
    // repeated picks from compounding, exactly like the global geom canvas
    // does for whole-canvas picks.
    function ensureLayerBaseCanvas(layer) {
        const key = (layer.src ? layer.src.length : 0) + '|' + layer.strokeKey;
        if (layer.baseCanvas && layer.baseKey === key) return layer.baseCanvas;
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, layer.w);
        canvas.height = Math.max(1, layer.h);
        const ctx = canvas.getContext('2d');
        if (layer.img) ctx.drawImage(layer.img, 0, 0); // empty layer: transparent raster, strokes only
        drawStrokes(ctx, layer.strokes);
        layer.baseCanvas = canvas;
        layer.baseKey = key;
        return canvas;
    }

    // base + the layer's OWN color pipeline (levels/gamma/gray balance,
    // grayscale/edges/invert) - same code, same order as the global pipeline,
    // just scoped to one layer. Dimensions never change here; the layer's
    // flip/rotate are applied by the composite draw, non-destructively.
    function ensureLayerCanvas(layer) {
        const base = ensureLayerBaseCanvas(layer);
        if (!layerColorAdjusted(layer)) return base;
        const key = layer.baseKey + '|' + layerAdjKey(layer);
        if (layer.canvas && layer.canvasKey === key) return layer.canvas;
        layer.canvas = applyExtraEffects(
            applyLevels(base, layer.black, layer.white, layer.gamma, layer.grayGains), layer);
        layer.canvasKey = key;
        return layer.canvas;
    }

    function compositeKeyOf(st) {
        return st.stageW + 'x' + st.stageH + '||' + st.layers.map(
            (l) => (l.src ? l.src.length : 0) + ',' + l.x.toFixed(1) + ',' + l.y.toFixed(1)
                + ',' + l.scale.toFixed(4) + ',' + l.strokeKey + ',' + l.blend
                + ',' + l.flipH + ',' + (+l.rotate).toFixed(2) + ',' + layerAdjKey(l)
        ).join('|');
    }

    // the flattened stage: every layer drawn at its transform, in z-order.
    // THIS is the "image" the whole historical pipeline runs on.
    function ensureCompositeCanvas(st) {
        const key = compositeKeyOf(st);
        if (st.compositeCanvas && st.compositeKey === key) return st.compositeCanvas;
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, st.stageW);
        canvas.height = Math.max(1, st.stageH);
        const ctx = canvas.getContext('2d');
        for (const l of st.layers) {
            const lc = ensureLayerCanvas(l);
            ctx.globalCompositeOperation = l.blend === 'lighten' ? 'lighten' : 'source-over';
            // per-layer flip/rotate about the layer's center - the transform
            // layerLocalToStage/stageToLayerLocal must mirror exactly
            const cw = l.w * l.scale, ch = l.h * l.scale;
            ctx.save();
            ctx.translate(l.x + cw / 2, l.y + ch / 2);
            if (l.flipH) ctx.scale(-1, 1);
            ctx.rotate((l.rotate || 0) * Math.PI / 180);
            ctx.drawImage(lc, -cw / 2, -ch / 2, cw, ch);
            ctx.restore();
        }
        ctx.globalCompositeOperation = 'source-over';
        st.compositeCanvas = canvas;
        st.compositeKey = key;
        return canvas;
    }

    // rotation + horizontal flip only (colors untouched) - cached, also used
    // for point picking. Runs on the layer COMPOSITE, so everything drawn,
    // erased or placed is part of the raster exactly as if it had arrived
    // with an uploaded file: levels, gamma, grayscale, the edges mask and
    // invert all process it, and the point pickers can sample it.
    function ensureGeomCanvas(st) {
        const comp = ensureCompositeCanvas(st);
        const key = st.compositeKey + '|' + st.rotate + '|' + st.flipH;
        if (st.geomCanvas && st.geomKey === key) return st.geomCanvas;
        const rad = st.rotate * Math.PI / 180;
        const cos = Math.abs(Math.cos(rad)), sin = Math.abs(Math.sin(rad));
        const w = Math.max(1, Math.round(comp.width * cos + comp.height * sin));
        const h = Math.max(1, Math.round(comp.width * sin + comp.height * cos));
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.translate(w / 2, h / 2);
        if (st.flipH) ctx.scale(-1, 1);
        ctx.rotate(rad);
        ctx.drawImage(comp, -comp.width / 2, -comp.height / 2);
        st.geomCanvas = canvas;
        st.geomKey = key;
        return canvas;
    }

    // color pipeline: gray-point balance (per-channel gains), then black/white point + gamma; alpha untouched
    function applyLevels(geom, black, white, gamma, gains) {
        const canvas = document.createElement('canvas');
        canvas.width = geom.width;
        canvas.height = geom.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(geom, 0, 0);
        const g = gains || {r: 1, g: 1, b: 1};
        if (black === 0 && white === 255 && gamma === 1 && g.r === 1 && g.g === 1 && g.b === 1) return canvas;
        const range = Math.max(1, white - black);
        const exp = 1 / gamma;
        const buildLut = (gain) => {
            const lut = new Uint8ClampedArray(256);
            for (let i = 0; i < 256; i++) {
                const balanced = clamp(i * gain, 0, 255);
                lut[i] = Math.round(Math.pow(clamp((balanced - black) / range, 0, 1), exp) * 255);
            }
            return lut;
        };
        const lutR = buildLut(g.r), lutG = buildLut(g.g), lutB = buildLut(g.b);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const px = data.data;
        for (let i = 0; i < px.length; i += 4) {
            px[i] = lutR[px[i]];
            px[i + 1] = lutG[px[i + 1]];
            px[i + 2] = lutB[px[i + 2]];
        }
        ctx.putImageData(data, 0, 0);
        return canvas;
    }

    // Edges-mask feathering: the white mask diffuses into the edges from
    // their boundary. The slider value is the share of the original edge
    // (non-white) pixels whitened away (0 = untouched, 100 = all edges
    // eaten): pixel depths (chamfer 3-4 distance to the nearest edge-free
    // pixel) are ranked and the budget eats the shallowest pixels first;
    // beyond the front an exponential partial bleach exp(-(depth-front)/band)
    // remains - the equilibrium profile of diffusion against an absorbing
    // front. The band grows with the front depth (floor of ~1.5px), which is
    // the single-parameter middle ground between a sharp erosion cutoff and
    // homogeneous whitening.
    function featherMask(mask, w, h, feather) {
        const n = w * h;
        let total = 0;
        const INF = 0x3fffffff;
        const dist = new Int32Array(n); // chamfer units: 3 = one pixel step
        for (let i = 0; i < n; i++) {
            if (mask[i] > 0) {
                dist[i] = INF;
                total++;
            }
        }
        if (!total) return;
        if (feather >= 100) {
            mask.fill(0);
            return;
        }
        // two-pass chamfer distance transform; out-of-image stays untraversed,
        // so border-hugging edges are eaten from the image interior only
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = y * w + x;
                let d = dist[i];
                if (d === 0) continue;
                if (x > 0 && dist[i - 1] + 3 < d) d = dist[i - 1] + 3;
                if (y > 0) {
                    const j = i - w;
                    if (dist[j] + 3 < d) d = dist[j] + 3;
                    if (x > 0 && dist[j - 1] + 4 < d) d = dist[j - 1] + 4;
                    if (x < w - 1 && dist[j + 1] + 4 < d) d = dist[j + 1] + 4;
                }
                dist[i] = d;
            }
        }
        let dmax = 0;
        for (let y = h - 1; y >= 0; y--) {
            for (let x = w - 1; x >= 0; x--) {
                const i = y * w + x;
                let d = dist[i];
                if (d === 0) continue;
                if (x < w - 1 && dist[i + 1] + 3 < d) d = dist[i + 1] + 3;
                if (y < h - 1) {
                    const j = i + w;
                    if (dist[j] + 3 < d) d = dist[j] + 3;
                    if (x < w - 1 && dist[j + 1] + 4 < d) d = dist[j + 1] + 4;
                    if (x > 0 && dist[j - 1] + 4 < d) d = dist[j - 1] + 4;
                }
                dist[i] = d;
                if (d > dmax) dmax = d;
            }
        }
        // Depth histogram -> the front shell theta where the whitened-pixel
        // budget (feather% of all edge pixels) runs out. Depths are integer
        // chamfer shells and one shell can hold most of a thin line's pixels,
        // so a whole-shell cutoff would make the slider jump: shells below
        // theta are erased outright and the theta shell absorbs the
        // fractional remainder of the budget as a uniform partial bleach
        // (shellFrac) - exact in whitened mass, monotone in the slider.
        const histo = new Uint32Array(dmax + 1);
        for (let i = 0; i < n; i++) {
            if (mask[i] > 0) histo[dist[i]]++;
        }
        const target = total * feather / 100; // fractional pixel budget
        let theta = dmax;
        let thetaPrev = 0; // deepest fully erased shell (0 = none yet)
        let shellFrac = 1;
        for (let d = 1, acc = 0; d <= dmax; d++) {
            if (!histo[d]) continue;
            if (acc + histo[d] >= target) {
                theta = d;
                shellFrac = (target - acc) / histo[d];
                break;
            }
            acc += histo[d];
            thetaPrev = d;
        }
        // The diffusive tail beyond the front. Every term is globally
        // monotone in the slider - amplitude is the slider fraction itself
        // and the anchor psi is the CONTINUOUS front position (interpolated
        // across the straddled shell); anchoring or scaling the tail by the
        // per-shell values instead sawtooths whenever the front crosses a
        // shell (whitening would locally RETREAT while the slider grows).
        const psi = thetaPrev + (theta - thetaPrev) * shellFrac;
        const amp = feather / 100;
        const band = Math.max(4.5, 0.5 * psi); // chamfer units, >= 1.5px
        for (let i = 0; i < n; i++) {
            if (!mask[i]) continue;
            const d = dist[i];
            const wFront = d < theta ? 1 : (d === theta ? shellFrac : 0);
            const wTail = amp * Math.exp(-Math.max(0, d - psi) / band);
            const whiteness = Math.min(1, Math.max(wFront, wTail));
            mask[i] = Math.round(mask[i] * (1 - whiteness));
        }
    }

    // post color-pipeline effects, in place on the freshly built leveled canvas:
    // grayscale -> edges mask -> invert (invert last so it can flip the mask)
    function applyExtraEffects(canvas, st) {
        if (!st.grayscale && !st.invert && !edgesActive(st)) return canvas;
        const w = canvas.width, h = canvas.height;
        const ctx = canvas.getContext('2d');
        const image = ctx.getImageData(0, 0, w, h);
        const px = image.data;

        if (st.grayscale) {
            for (let i = 0; i < px.length; i += 4) {
                const luma = Math.round(0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]);
                px[i] = px[i + 1] = px[i + 2] = luma;
            }
        }

        if (edgesActive(st)) {
            // Sobel gradient magnitude on luminance. Edge strength is the
            // magnitude on a robust scale (99th percentile, so one extreme
            // edge doesn't flatten everything else) boosted by the
            // sensitivity gain. A fixed internal threshold culls edges below
            // a strength, hiding small details while keeping feature-defining
            // lines. Every kept edge pixel is stamped as a disc: a
            // thickness-driven base radius plus a strength-proportional part,
            // so the thickness slider visibly affects ALL edges while strong
            // edges stay thicker. The result is composited over the image:
            // edge pixels blend towards black with edgeOpacity, edge-free
            // pixels towards white with maskOpacity - at 0 the original
            // colors show through, at 100/100 it is a black-on-white edge map.
            const lum = new Float32Array(w * h);
            for (let i = 0, j = 0; j < lum.length; i += 4, j++) {
                lum[j] = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
            }
            const mag = new Float32Array(w * h);
            let magMax = 1e-6;
            for (let y = 1; y < h - 1; y++) {
                for (let x = 1; x < w - 1; x++) {
                    const i = y * w + x;
                    const gx = lum[i - w + 1] + 2 * lum[i + 1] + lum[i + w + 1]
                             - lum[i - w - 1] - 2 * lum[i - 1] - lum[i + w - 1];
                    const gy = lum[i + w - 1] + 2 * lum[i + w] + lum[i + w + 1]
                             - lum[i - w - 1] - 2 * lum[i - w] - lum[i - w + 1];
                    const m = Math.sqrt(gx * gx + gy * gy);
                    mag[i] = m;
                    if (m > magMax) magMax = m;
                }
            }
            const bins = new Uint32Array(1024);
            let nonzero = 0;
            for (let i = 0; i < mag.length; i++) {
                if (mag[i] > 0) {
                    bins[Math.min(1023, (mag[i] / magMax * 1023) | 0)]++;
                    nonzero++;
                }
            }
            let scale = magMax;
            for (let b = 0, acc = 0; b < 1024; b++) {
                acc += bins[b];
                if (acc >= nonzero * 0.99) {
                    scale = Math.max(1e-6, (b + 1) / 1023 * magMax);
                    break;
                }
            }
            const gain = st.edgeSensitivity / 50;   // 50 = neutral
            const cut = 0.2;                        // internal threshold (was a slider)
            const rMax = st.edgeThickness * Math.min(w, h) / 1024; // thickness relative to image size
            // coverage map, 255 = fully covered edge pixel. Radii below half
            // a pixel cannot shrink the stamped disc any further, so instead
            // the trace stays one pixel wide and its COVERAGE drops linearly
            // with the line diameter (alpha = 2r, continuous at r = 0.5).
            // This is what keeps sub-1 thickness values visibly linear: a
            // half-as-thick line renders half as strong instead of clamping
            // at the same 1px dot.
            const mask = new Uint8Array(w * h);
            for (let y = 1; y < h - 1; y++) {
                for (let x = 1; x < w - 1; x++) {
                    let s = mag[y * w + x] / scale * gain;
                    if (s <= cut || s <= 0) continue;
                    if (s > 1) s = 1;
                    const r = rMax * (0.35 + 0.65 * s);
                    if (r < 0.5) {
                        const v = Math.round(Math.min(1, 2 * r) * 255);
                        const idx = y * w + x;
                        if (v > mask[idx]) mask[idx] = v;
                        continue;
                    }
                    const ir = Math.ceil(r);
                    const r2 = r * r;
                    for (let dy = -ir; dy <= ir; dy++) {
                        const yy = y + dy;
                        if (yy < 0 || yy >= h) continue;
                        for (let dx = -ir; dx <= ir; dx++) {
                            const xx = x + dx;
                            if (xx < 0 || xx >= w) continue;
                            if (dx * dx + dy * dy <= r2) mask[yy * w + xx] = 255;
                        }
                    }
                }
            }
            // the white mask eats into the stamped edges before compositing
            if (st.edgeFeather > 0) featherMask(mask, w, h, st.edgeFeather);
            // composite: white mask layer on the edge-free share of a pixel,
            // black edge layer on the covered share - fractional coverage
            // blends both smoothly
            const eo = st.edgeOpacity / 100;
            const mo = st.maskOpacity / 100;
            for (let i = 0, j = 0; j < mask.length; i += 4, j++) {
                const c = mask[j] / 255;
                const whiteA = mo * (1 - c);
                const edgeA = eo * c;
                px[i] = Math.round((px[i] * (1 - whiteA) + 255 * whiteA) * (1 - edgeA));
                px[i + 1] = Math.round((px[i + 1] * (1 - whiteA) + 255 * whiteA) * (1 - edgeA));
                px[i + 2] = Math.round((px[i + 2] * (1 - whiteA) + 255 * whiteA) * (1 - edgeA));
            }
        }

        if (st.invert) {
            for (let i = 0; i < px.length; i += 4) {
                px[i] = 255 - px[i];
                px[i + 1] = 255 - px[i + 1];
                px[i + 2] = 255 - px[i + 2];
            }
        }

        ctx.putImageData(image, 0, 0);
        return canvas;
    }

    function cropFromCanvas(src, crop) {
        if (!crop) return src;
        const cx = Math.round(crop.x * src.width);
        const cy = Math.round(crop.y * src.height);
        const cw = Math.max(1, Math.round(crop.w * src.width));
        const ch = Math.max(1, Math.round(crop.h * src.height));
        const canvas = document.createElement('canvas');
        canvas.width = cw;
        canvas.height = ch;
        canvas.getContext('2d').drawImage(src, cx, cy, cw, ch, 0, 0, cw, ch);
        return canvas;
    }

    // announce what is loaded (original format + dimensions, before any
    // adjustments) on the container, for listeners like the ControlNet
    // raster-info line; the displayed src becomes a re-encoded PNG as soon as
    // an adjustment renders, so only this state knows the uploaded format.
    // When a crop is applied, the current cropped result dimensions ride
    // along (cropWidth/cropHeight) - re-announced on every crop action.
    function announceImageInfo(st) {
        const container = document.getElementById('imageContainer_' + st.uuid);
        if (!container) return;
        const send = (detail) => {
            const value = detail ? JSON.stringify(detail) : '';
            const seq = container.dataset.forgeUploadSeq || '';
            // identical info is normally an echo and stays silent - UNLESS the
            // upload counter moved: a NEW image replacing the old at identical
            // dimensions/format produces byte-identical info, and listeners
            // (the ControlNet weight-mask painter clears stale paint on it)
            // must still hear the event
            if (container.dataset.forgeImageInfo === value
                && (container.dataset.forgeImageInfoSeq || '') === seq) return;
            container.dataset.forgeImageInfo = value;
            container.dataset.forgeImageInfoSeq = seq;
            container.dispatchEvent(new CustomEvent('forge-image-info', {bubbles: true}));
        };
        if (!st.original || !st.layers.length) {
            send(null);
            return;
        }
        const src = st.original;
        const mime = /^data:image\/([a-z0-9.+-]+)/i.exec(src);
        ensureSource(st, () => {
            if (st.original !== src) return; // a newer image arrived meanwhile
            // the stage is what gradio receives, so it is what gets announced;
            // format is the bottom layer's upload format
            const detail = {
                format: mime ? mime[1] : null,
                width: st.stageW,
                height: st.stageH,
            };
            if (st.layers.length > 1) detail.layers = st.layers.length;
            if (cropApplied(st)) {
                // crop applies in the rotated/flipped geometry space; same
                // rounding as ensureGeomCanvas + cropFromCanvas, so this is
                // exactly what gradio receives
                const rad = st.rotate * Math.PI / 180;
                const cos = Math.abs(Math.cos(rad));
                const sin = Math.abs(Math.sin(rad));
                const gw = Math.max(1, Math.round(st.stageW * cos + st.stageH * sin));
                const gh = Math.max(1, Math.round(st.stageW * sin + st.stageH * cos));
                detail.cropWidth = Math.max(1, Math.round(st.crop.w * gw));
                detail.cropHeight = Math.max(1, Math.round(st.crop.h * gh));
            }
            send(detail);
        });
    }

    // ---------------------------------------------------------------------
    // CANVAS / CONTROL PARITY CONTRACT
    //
    //   The raster the control receives is the flattened canvas the user is
    //   looking at. Same pixels, same dimensions, always.
    //
    // The whole layer stack exists to be composited, and the composite is the
    // ONLY thing that may leave this file: layers, per-layer blend/placement/
    // adjustments, strokes, the global colour pipeline, geometry and the crop
    // are all folded into one raster by `ensureLeveledCanvas` + `cropFromCanvas`,
    // and both the display and the gradio channel are produced from THAT canvas
    // (`renderAdjusted` hands `display` to the widget; the widget's own
    // `updateBackgroundImageData` re-encodes the very <img> it was given). A
    // canvas the user can see but the control never got - or the reverse - is a
    // bug in this file, not a tuning question.
    //
    // The one deliberate divergence: while the CROP TOOL IS OPEN the display
    // shows the full frame with handles and shading, because that is the thing
    // being edited. gradio still holds the committed (cropped) result, which is
    // what the contract means by "the canvas" - the crop is committed the moment
    // it is dragged, not when the tool is closed. Everywhere else, display and
    // committed value are the same canvas object.
    //
    // tests/test_canvas_parity.py asserts this in a real browser, decoding both
    // sides and comparing every pixel, across layers, blends, per-layer and
    // global adjustments, strokes, crop and drops. Keep it passing; it is the
    // only thing standing between "looks right" and "is what gets generated".
    // ---------------------------------------------------------------------

    // push the final (cropped) result to gradio without touching the display
    function syncCroppedToGradio(st) {
        const bind = st.fc.background_gradio_bind;
        if (!bind || !st.leveledCanvas || typeof bind.set_value !== 'function') return;
        const url = cropFromCanvas(st.leveledCanvas, st.crop).toDataURL('image/png');
        rememberSynced(st, url); // a re-emission of this value is an echo, not an upload
        bind.set_value(url);
    }

    // ---- pen strokes
    //
    // A stroke is a polyline plus its brush style, stored in original-image
    // pixels. Replaying the list is what produces the paint layer, which makes
    // stroke-level undo free and keeps the paint at full image resolution no
    // matter how it was drawn (zoomed in, on a cropped display, ...).
    //
    // One stroke = ONE canvas operation: a single path stroked once. That is
    // what makes the opacity slider mean what it says - stamping discs along
    // the path would accumulate alpha wherever consecutive stamps overlap and
    // a half-transparent stroke would come out nearly opaque. Feathering is
    // the canvas blur filter on that same single operation, so the falloff is
    // uniform along the whole stroke instead of scalloped per stamp.
    function drawStrokes(ctx, strokes) {
        for (const s of strokes) {
            if (!s.pts.length) continue;
            ctx.save();
            // the eraser is a pen stroke that cuts to transparency instead of
            // painting: alpha is erase strength, feathering works unchanged
            ctx.globalCompositeOperation = s.erase ? 'destination-out' : 'source-over';
            ctx.globalAlpha = s.alpha;
            ctx.filter = s.blur >= 0.25 ? 'blur(' + s.blur.toFixed(2) + 'px)' : 'none';
            ctx.strokeStyle = ctx.fillStyle = s.color;
            ctx.lineCap = ctx.lineJoin = 'round';
            ctx.lineWidth = Math.max(0.1, 2 * s.radius);
            ctx.beginPath();
            if (s.pts.length < 2) {
                ctx.arc(s.pts[0].x, s.pts[0].y, Math.max(0.05, s.radius), 0, Math.PI * 2);
                ctx.fill();
            } else {
                ctx.moveTo(s.pts[0].x, s.pts[0].y);
                for (let i = 1; i < s.pts.length; i++) ctx.lineTo(s.pts[i].x, s.pts[i].y);
                ctx.stroke();
            }
            ctx.restore();
        }
    }

    // rebuild (or reuse) the cached fully color-processed canvas for the
    // current adjustment state; shared by renderAdjusted and the Topaz tool
    function ensureLeveledCanvas(st) {
        const geom = ensureGeomCanvas(st);
        const gains = st.grayGains;
        const gainsKey = gains ? gains.r.toFixed(4) + ',' + gains.g.toFixed(4) + ',' + gains.b.toFixed(4) : '1';
        const leveledKey = st.geomKey + '|' + st.black + '|' + st.white + '|' + st.gamma + '|' + gainsKey
            + '|' + st.grayscale + '|' + st.invert + '|' + st.edgeSensitivity
            + '|' + st.edgeOpacity + '|' + st.maskOpacity + '|' + st.edgeThickness
            + '|' + st.edgeFeather; // layers + strokes ride along in st.geomKey
        if (!st.leveledCanvas || st.leveledKey !== leveledKey) {
            st.leveledCanvas = applyExtraEffects(applyLevels(geom, st.black, st.white, st.gamma, gains), st);
            st.leveledKey = leveledKey;
        }
        return st.leveledCanvas;
    }

    function renderAdjusted(st) {
        if (!st.original) return;
        ensureSource(st, () => {
            if (!st.original) return;
            ensureLeveledCanvas(st);
            const cropMode = st.mode === 'crop';
            // crop-edit mode displays the full image (handles + gray shades on top);
            // otherwise the display is the cropped result itself
            const display = cropMode ? st.leveledCanvas : cropFromCanvas(st.leveledCanvas, st.crop);
            const fc = st.fc;
            const drawingCanvas = document.getElementById('drawingCanvas_' + st.uuid);
            const sameDims = drawingCanvas && drawingCanvas.width === display.width && drawingCanvas.height === display.height;
            if (st.pendingGeomClear) {
                st.pendingGeomClear = false;
                // geometry changed but canvas keeps its size (e.g. flip): scribbles no longer match, clear them
                if (sameDims) {
                    drawingCanvas.getContext('2d').clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
                    fc.saveState();
                }
            }
            if (sameDims) st.preserveViewOnce = true; // keep user zoom/pan for same-size updates
            const displayUrl = display.toDataURL('image/png');
            // outside crop mode this lands in gradio via the core sync: a
            // later re-emission of it (generation finished) is an echo
            rememberSynced(st, displayUrl);
            if (cropMode) {
                // gradio must keep the cropped result, not the crop-edit full
                // display. Suppression is VALUE-COUNTED (crop-edit display url
                // -> number of its uploads still in flight) because the core's
                // uploads complete asynchronously and out of order: a one-shot
                // flag could be consumed by the wrong upload, letting the full
                // image overwrite the cropped result in gradio. Each landing
                // upload consumes one count (patched on_img_upload), so a later
                // byte-identical legitimate render is NOT suppressed - an
                // unbounded blocklist kept eating such renders and left gradio
                // holding a stale image.
                st.suppressSyncUrls.set(displayUrl, (st.suppressSyncUrls.get(displayUrl) || 0) + 1);
                syncCroppedToGradio(st);
            }
            st.displayCropped = !cropMode && !!st.crop;
            st.applying = true;
            try {
                fc.uploadBase64(displayUrl);
            } finally {
                st.applying = false;
            }
            announceImageInfo(st); // keeps the cropped-dimensions readout current
        });
    }

    function scheduleRender(st, geometryChanged) {
        if (!st.original) return;
        if (geometryChanged) st.pendingGeomClear = true;
        clearTimeout(st.renderTimer);
        st.renderTimer = setTimeout(() => renderAdjusted(st), 120);
    }

    // ---- Topaz upscale availability: asked once per page load, shared by all
    // widgets; the toolbar button only appears when the server found tpai.exe
    // (modules_forge/forge_canvas/topaz_upscale.py)
    let topazStatusPromise = null;
    function fetchTopazStatus() {
        if (!topazStatusPromise) {
            // failure and legit unavailability both keep the buttons hidden, so
            // say WHY in the console - a silent miss here cost a debug session
            topazStatusPromise = fetch('./forge-canvas/topaz/status')
                .then((r) => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
                .then((status) => {
                    if (!status.available) console.info('[forge canvas] Topaz tools hidden: server reports tpai.exe not found');
                    return status;
                })
                .catch((err) => {
                    console.info('[forge canvas] Topaz tools hidden: status check failed -', err && err.message ? err.message : err);
                    return { available: false };
                });
        }
        return topazStatusPromise;
    }

    // ---- gradio ECHO detection.
    // The core polls the gradio textarea and calls uploadBase64 whenever its
    // value changes - including when a finished generation makes gradio
    // re-emit the value it holds, which is exactly what WE rendered into it
    // (the flattened composite, possibly re-encoded or turned into a file
    // URL on the round trip). Treating that echo as a fresh upload destroyed
    // the editing model mid-session (user report 2026-07-23: layer stack
    // flattened, edge settings baked in and dead). An inflow is an echo iff
    // it is byte-equal to something we handed gradio (fast path) or
    // pixel-equal to the current pipeline output (slow path - catches
    // re-encodes and file URLs). Echoes are swallowed whole: the display
    // already shows those pixels and the model (layers, adjustments, edge
    // settings) lives on.
    function rememberSynced(st, url) {
        if (typeof url !== 'string' || !url) return;
        st.syncedUrls.add(url);
        while (st.syncedUrls.size > 16) {
            st.syncedUrls.delete(st.syncedUrls.values().next().value);
        }
    }

    function pixelEqual(imgA, canvasB) {
        const aw = imgA.naturalWidth || imgA.width, ah = imgA.naturalHeight || imgA.height;
        if (!aw || !canvasB || aw !== canvasB.width || ah !== canvasB.height) return false;
        const sample = (src) => {
            const c = document.createElement('canvas');
            c.width = Math.min(64, aw);
            c.height = Math.min(64, ah);
            const x = c.getContext('2d');
            x.drawImage(src, 0, 0, c.width, c.height);
            return x.getImageData(0, 0, c.width, c.height).data;
        };
        const da = sample(imgA), db = sample(canvasB);
        let sum = 0;
        for (let i = 0; i < da.length; i++) sum += Math.abs(da[i] - db[i]);
        // mean channel difference of re-encode noise (JPEG chroma shift on
        // saturated content reaches several units). This tolerance is only
        // ever applied to values that CAME FROM gradio (see uploadBase64):
        // used as a general "is this the same picture" test it silently ate
        // new images - two shots of the same scene are well inside it.
        return sum / da.length < 8;
    }

    // The textarea gradio holds this canvas's background in. The bind polls it
    // and calls uploadBase64 with exactly its content, so it is what tells an
    // echo (gradio handing our own output back) apart from user content.
    // Foreground and background share the elem_id, hence the class filter.
    function gradioHeldValue(st) {
        if (!st.bgTextarea || !st.bgTextarea.isConnected) {
            const block = [...document.querySelectorAll('.logical_image_background')]
                .find((e) => e.id === st.uuid);
            st.bgTextarea = block ? block.querySelector('textarea') : null;
        }
        return st.bgTextarea ? st.bgTextarea.value : null;
    }

    function echoTest(st, url, callback) {
        if (st.syncedUrls.has(url)) {
            callback(true);
            return;
        }
        const img = new Image();
        img.onload = () => {
            if (!st.layers.length) {
                callback(false);
                return;
            }
            ensureSource(st, () => {
                let ref = null;
                try {
                    // what gradio should currently hold: the leveled output
                    // with the crop applied
                    ref = cropFromCanvas(ensureLeveledCanvas(st), st.crop);
                } catch (e) {}
                callback(!!ref && pixelEqual(img, ref));
            });
        };
        img.onerror = () => callback(false); // undecodable: let the normal path deal with it
        img.src = url;
    }

    // ------------------------------------------------------ prototype patches

    const origUploadBase64 = proto.uploadBase64;

    // a genuinely new image (user upload / paste / drop / gradio push outside
    // the layer tool): REPLACES the whole content - it becomes the single
    // bottom layer and redefines the stage. Adding on top of existing content
    // goes through the layer tool / forgeCanvasAddLayer instead.
    function replaceAll(fc, st, b64) {
        st.original = (typeof b64 === 'string' && b64.startsWith('data:image')) ? b64 : null;
        st.layers = st.original ? [newLayer(st.original)] : [];
        st.activeLayer = st.layers.length ? 0 : -1;
        st.stageW = 0;
        st.stageH = 0;
        st.compositeCanvas = null;
        st.compositeKey = null;
        st.geomCanvas = null;
        st.geomKey = null;
        st.leveledCanvas = null;
        st.leveledKey = null;
        st.pendingGeomClear = false;
        st.preserveViewOnce = false;
        st.suppressSyncUrls.clear();
        st.syncedUrls.clear();
        // gradio now holds this exact string: a later re-emission of it must
        // not re-reset the model (covers the no-adjustments-yet echo)
        rememberSynced(st, st.original);
        st.displayCropped = false;
        clearTimeout(st.renderTimer);
        st.resetValues();
        if (st.original) {
            // upload sequence: bumped ONLY here - for genuinely new content
            // (uploads, drops, pastes, forgeCanvasPush, Topaz results), never
            // for adjustment echoes or layer edits. It is what lets listeners
            // distinguish a NEW same-size image from an ADJUSTED one, which
            // dimensions alone cannot (see the weight-mask painter).
            const container = document.getElementById('imageContainer_' + st.uuid);
            if (container) {
                container.dataset.forgeUploadSeq =
                    String((+container.dataset.forgeUploadSeq || 0) + 1);
            }
        }
        announceImageInfo(st);
        return origUploadBase64.call(fc, b64);
    }

    // while the layer tool is open - or a multi-layer composition exists at
    // all (switching to the pen mid-composition must not turn a drop into a
    // stack wipe) - EVERY raster inflow: drop, paste, the core open button,
    // gradio pushes (ControlNet get-input / get-output), sets the ACTIVE
    // layer's raster; the + button creates an empty logical layer precisely
    // so any of these methods can fill it. The raw inflow never reaches
    // gradio: the re-rendered composite does, exactly like any other layer
    // edit. A single-layer canvas with the tool closed keeps the classic
    // replace semantics.
    function applyInflow(fc, st, b64) {
        if ((st.showLayers || st.layers.length > 1) && st.layers.length && st.setActiveLayerRaster
            && typeof b64 === 'string' && b64.startsWith('data:image')) {
            st.setActiveLayerRaster(b64);
            return;
        }
        replaceAll(fc, st, b64);
    }

    proto.uploadBase64 = function (b64) {
        const st = this.__adjust;
        if (!st || st.applying) return origUploadBase64.call(this, b64);
        if (st.forceReplace) {
            // Topaz results ARE the flattened stack: never a layer, never an echo
            st.forceReplace = false;
            return replaceAll(this, st, b64);
        }
        if (!st.layers.length || typeof b64 !== 'string') {
            return replaceAll(this, st, b64); // empty canvas: nothing an echo could destroy
        }
        // an EMPTY active layer is armed to be filled: the user just created
        // it, so an arriving raster is their fill - even one byte-identical
        // to existing content (duplicating a layer is legitimate). Intent
        // beats the echo gate here.
        const active = st.layers[Math.min(Math.max(st.activeLayer, 0), st.layers.length - 1)];
        if ((st.showLayers || st.layers.length > 1) && active && !active.src
            && b64.startsWith('data:image') && st.setActiveLayerRaster) {
            st.setActiveLayerRaster(b64);
            return;
        }
        // Content exists. PROVENANCE, not similarity, decides whether this is
        // an echo: the gradio bind polls its textarea and calls us with
        // exactly what it holds, so an inflow equal to the held value is our
        // own content coming back. Everything else - drop, paste, the core
        // open button, forgeCanvasPush - is the user handing us a new image
        // and must be applied verbatim, however much it happens to resemble
        // what is already on the canvas.
        //
        // The pixel test used to run on EVERY inflow, and its re-encode
        // tolerance then read "looks similar" as "is the same": dropping a
        // new raster over an old one was silently discarded (measured: a
        // 110x150 patch changed on a 640x480 image = swallowed; the canvas
        // kept the old picture and gradio never heard about the new one, so
        // generations kept using the old input until the canvas was cleared
        // first). Only gradio-sourced values reach it now, which is the only
        // place a re-encode can come from.
        const held = gradioHeldValue(st);
        if (held === null || b64 !== held) {
            applyInflow(this, st, b64);
            return;
        }
        // gradio-sourced: still async, because a re-encoded echo has to be
        // decoded to be recognized; a newer inflow supersedes an undecided one
        st.inflowSeq = (st.inflowSeq || 0) + 1;
        const seq = st.inflowSeq;
        echoTest(st, b64, (isEcho) => {
            if (seq !== st.inflowSeq || !st.layers.length) return; // superseded / cleared meanwhile
            if (isEcho) return; // our own output bounced back: display is already right
            applyInflow(this, st, b64);
        });
    };

    const origRemoveImage = proto.removeImage;
    proto.removeImage = function () {
        const st = this.__adjust;
        if (st) {
            st.original = null;
            st.layers = [];
            st.activeLayer = -1;
            st.stageW = 0;
            st.stageH = 0;
            st.compositeCanvas = null;
            st.compositeKey = null;
            st.geomCanvas = null;
            st.geomKey = null;
            st.leveledCanvas = null;
            st.leveledKey = null;
            st.suppressSyncUrls.clear();
            st.syncedUrls.clear();
            st.displayCropped = false;
            clearTimeout(st.renderTimer);
            st.resetValues();
            announceImageInfo(st);
        }
        return origRemoveImage.call(this);
    };

    const origAdjustPosition = proto.adjustInitialPositionAndScale;
    proto.adjustInitialPositionAndScale = function () {
        const st = this.__adjust;
        if (st && st.preserveViewOnce) {
            st.preserveViewOnce = false;
            return;
        }
        return origAdjustPosition.call(this);
    };

    const origOnImgUpload = proto.on_img_upload;
    proto.on_img_upload = function () {
        const st = this.__adjust;
        // never let a crop-edit full-image display reach gradio, no matter in
        // which order the async uploads complete: each suppressed url carries
        // a pending-upload count and every match consumes one, so suppression
        // ends exactly when the last display upload has landed
        if (st && st.suppressSyncUrls) {
            const pending = st.suppressSyncUrls.get(this.img);
            if (pending) {
                if (pending > 1) st.suppressSyncUrls.set(this.img, pending - 1);
                else st.suppressSyncUrls.delete(this.img);
                return;
            }
            // self-healing: an ordinary upload landing outside crop mode means
            // the display pipeline has moved on; any counts left behind (an
            // upload the core never reported back) could only mis-suppress a
            // future identical render
            if (st.suppressSyncUrls.size && st.mode !== 'crop') {
                st.suppressSyncUrls.clear();
            }
        }
        return origOnImgUpload.call(this);
    };

    // crop handles must follow the image through pan/zoom/resize
    const origDrawImage = proto.drawImage;
    proto.drawImage = function () {
        const result = origDrawImage.call(this);
        const st = this.__adjust;
        if (st && st.mode === 'crop' && st.updateCropOverlay) st.updateCropOverlay();
        if (st && st.mode === 'pick-gray' && st.updateGrayOverlay) st.updateGrayOverlay();
        if (st && st.mode === 'pen' && st.updatePenOverlay) st.updatePenOverlay();
        if (st && (st.mode === 'layers' || st.showLayers) && st.updateLayersOverlay) st.updateLayersOverlay();
        return result;
    };

    // ------------------------------------------------------- per-widget setup

    function attach(fc, uuid) {
        const el = (id) => document.getElementById(id + uuid);

        // CNPro: the tool chrome below lived in the host's canvas.html in the
        // original fork. An extension cannot edit that file, so the same markup
        // is injected here, per container, before anything looks it up.
        // Idempotent and a no-op once done (see javascript/canvas_nodes.js).
        if (window.cnproCanvasNodes) {
            window.cnproCanvasNodes.inject(uuid);
        } else {
            console.warn('[cnpro] canvas_nodes.js missing - tool controls unavailable.');
        }

        const container = el('imageContainer_');
        const toolbar = el('toolbar_');
        const imgEl = el('image_');
        const drawingCanvas = el('drawingCanvas_');
        const indicator = el('scribbleIndicator_');
        const flipBtn = el('flipButton_');
        const colorBtn = el('colorButton_');
        const colorBox = el('colorBox_');
        const pickBlackBtn = el('pickBlackButton_');
        const pickWhiteBtn = el('pickWhiteButton_');
        const pickGrayBtn = el('pickGrayButton_');
        const blackSwatch = el('blackSwatch_');
        const blackRgb = el('blackRgb_');
        const whiteSwatch = el('whiteSwatch_');
        const whiteRgb = el('whiteRgb_');
        const graySwatch = el('graySwatch_');
        const grayRgb = el('grayRgb_');
        const gammaBtn = el('gammaButton_');
        const grayscaleBtn = el('grayscaleButton_');
        const invertBtn = el('invertButton_');
        const edgesBtn = el('edgesButton_');
        const cropBtn = el('cropButton_');
        const rotateBtn = el('rotateButton_');
        const pointsResetBtn = el('pointsResetButton_');
        const gammaResetBtn = el('gammaResetButton_');
        const edgesResetBtn = el('edgesResetButton_');
        const rotateResetBtn = el('rotateResetButton_');
        const gammaBox = el('gammaBox_');
        const edgesBox = el('edgesBox_');
        const rotateBox = el('rotateBox_');
        const edgeOpacitySlider = el('edgeOpacity_');
        const maskOpacitySlider = el('maskOpacity_');
        const edgeSensSlider = el('edgeSensitivity_');
        const edgeThickSlider = el('edgeThickness_');
        const edgeFeatherSlider = el('edgeFeather_');
        const edgeOpacityLabel = el('edgeOpacityLabel_');
        const maskOpacityLabel = el('maskOpacityLabel_');
        const edgeSensLabel = el('edgeSensLabel_');
        const edgeThickLabel = el('edgeThickLabel_');
        const edgeFeatherLabel = el('edgeFeatherLabel_');
        const blackSlider = el('blackPoint_');
        const whiteSlider = el('whitePoint_');
        const gammaSlider = el('gammaSlider_');
        const rotSlider = el('rotationSlider_');
        const blackLabel = el('blackLabel_');
        const whiteLabel = el('whiteLabel_');
        const gammaLabel = el('gammaLabel_');
        const rotLabel = el('rotationLabel_');
        const reticle = el('pickReticle_');
        const grayOverlay = el('grayMaskOverlay_');
        const penBtn = el('penButton_');
        const penBox = el('penBox_');
        const penSV = el('penSV_');
        const penSVCursor = el('penSVCursor_');
        const penHue = el('penHue_');
        const penPickBtn = el('penPickButton_');
        const penSwatch = el('penSwatch_');
        const penHex = el('penHex_');
        const penSizeSlider = el('penSize_');
        const penOpacitySlider = el('penOpacity_');
        const penFeatherSlider = el('penFeather_');
        const penSizeLabel = el('penSizeLabel_');
        const penOpacityLabel = el('penOpacityLabel_');
        const penFeatherLabel = el('penFeatherLabel_');
        const penUndoBtn = el('penUndoButton_');
        const penClearBtn = el('penClearButton_');
        const penEraserBtn = el('penEraserButton_');
        const penOverlay = el('penOverlay_');
        const penCursor = el('penCursor_');
        const layersBtn = el('layersButton_');
        const layersBox = el('layersBox_');
        const layerAddBtn = el('layerAddButton_');
        const layerList = el('layerList_');
        const layersOverlay = el('layersOverlay_');
        const overlay = el('cropOverlay_');
        const cropBox = el('cropBox_');
        const shadeN = el('cropShadeN_');
        const shadeS = el('cropShadeS_');
        const shadeW = el('cropShadeW_');
        const shadeE = el('cropShadeE_');

        // ---- the attach gate: all-or-nothing, so SAY SO when it bites
        //
        // These were three bare `return false` lines. Twenty-three required nodes,
        // and if any ONE was absent attach abandoned the whole widget - no
        // toolbar, no Topaz probe, no listeners - and said nothing. The
        // MutationObserver below then re-ran attach forever, silently, because
        // the attached flag is only set on success.
        //
        // That is the single most consequential silent failure in this file: it
        // is indistinguishable on screen from "the buttons are hidden", and it is
        // a completely different problem with a completely different fix. Name
        // the missing nodes, once per container, and point at the injector.
        const REQUIRED = {
            'imageContainer': container, 'toolbar': toolbar, 'image': imgEl,
            'flipButton': flipBtn, 'colorButton': colorBtn, 'colorBox': colorBox,
            'pointsResetButton': pointsResetBtn, 'cropOverlay': overlay,
            'cropBox': cropBox, 'pickReticle': reticle,
            'penButton': penBtn, 'penBox': penBox, 'penSV': penSV, 'penHue': penHue,
            'penOverlay': penOverlay, 'penCursor': penCursor, 'penPickButton': penPickBtn,
            'penEraserButton': penEraserBtn, 'layersButton': layersBtn,
            'layersBox': layersBox, 'layerAddButton': layerAddBtn,
            'layerList': layerList, 'layersOverlay': layersOverlay,
        };
        const missingNodes = Object.keys(REQUIRED).filter((k) => !REQUIRED[k]);
        if (missingNodes.length) {
            // Reported once per container: attach is retried by a MutationObserver
            // on every DOM change, and an unthrottled error here would bury the
            // console under thousands of identical lines.
            if (container && container.dataset.cnproAttachReported !== '1') {
                container.dataset.cnproAttachReported = '1';
                console.error(
                    '[cnpro] canvas attach ABORTED for ' + uuid + ': ' +
                    missingNodes.length + ' of ' + Object.keys(REQUIRED).length +
                    ' required nodes are missing from the DOM:\n  ' +
                    missingNodes.join(', ') +
                    '\nNothing is wired for this canvas - no tool buttons, no Topaz, ' +
                    'no weight masks. These nodes come from javascript/canvas_nodes.js' +
                    ' (inject); if they are absent, injection did not run or the ' +
                    'host template changed its anchors (.forge-toolbar-box-a, ' +
                    'imageContainer_<uuid>, toolbar_<uuid>).');
            }
            return false;
        }

        const st = fc.__adjust = {
            fc: fc,
            uuid: uuid,
            original: null,
            // the layer stack, bottom first; the STAGE (composite dimensions)
            // is fixed by the first upload and outlives layer edits
            layers: [],
            activeLayer: -1,
            // layer targeting: while the layers menu is open every adjustment
            // control (flip, points, gamma, grayscale, invert, edges,
            // rotation) reads and writes the ACTIVE LAYER's own state; closed,
            // they act on the whole canvas as one merged raster. Kept separate
            // from st.mode so picker/pen/crop sub-modes don't drop the target.
            showLayers: false,
            forceReplace: false, // next inflow replaces everything (Topaz results)
            stageW: 0,
            stageH: 0,
            compositeCanvas: null,
            compositeKey: null,
            layerDrag: null,     // {x0, y0, startX, startY, k} while a layer is dragged
            geomCanvas: null,
            geomKey: null,
            leveledCanvas: null,
            leveledKey: null,
            flipH: false,
            black: 0,
            white: 255,
            gamma: 1,
            grayGains: null,     // per-channel white-balance gains from the gray-area analysis
            grayMask: null,      // offscreen mask canvas in geom resolution (multi-region, alpha = selected)
            grayStroke: null,    // last painted point {x, y} in geom pixels (connects drag segments)
            grayPainting: false, // a mask stroke is being dragged right now
            pickedBlack: null,   // original RGB of the picked black point (menu readout)
            pickedWhite: null,   // original RGB of the picked white point
            grayPicked: null,    // statistical RGB surrogate of the gray selection
            grayscale: false,
            invert: false,
            // the edges effect is active while any opacity > 0 (edgesActive);
            // the toolbar button only toggles the menu (showEdges)
            edgeSensitivity: 50,
            edgeOpacity: 0,      // 0 = edges transparent (original shows) .. 100 = opaque black edges
            maskOpacity: 0,      // 0 = edge-free areas transparent .. 100 = opaque white mask
            edgeThickness: 2,    // one of THICKNESS_STOPS
            edgeFeather: 0,      // % of edge pixels eaten by the white mask (featherMask)
            rotate: 0,
            // pen: strokes live on the ACTIVE LAYER in layer-local pixels
            // (see commitPenStroke); penErase flips the pen into the eraser
            penErase: false,
            penHue: 0,
            penSat: 0,
            penVal: 0,           // black by default: the useful sketch color
            penSize: 45,         // slider value, see brushDiameter()
            penOpacity: 100,
            penFeather: 0,
            penStroke: null,     // stroke in progress {pts in original px, color, alpha, radius, blur}
            penSVDrag: false,    // dragging inside the saturation/value square
            penPicking: false,   // eyedropper armed (one pick, then back to drawing)
            crop: null,          // normalized rect in rotated/flipped image space
            mode: null,          // 'crop' | 'pick-black' | 'pick-white' | 'pick-gray' | 'pen' | 'layers'
            showColor: false,
            showGamma: false,
            showEdges: false,
            showRotate: false,
            cropDrag: null,
            applying: false,
            preserveViewOnce: false,
            pendingGeomClear: false,
            suppressSyncUrls: new Map(), // crop-edit display url -> pending upload count that must not sync to gradio
            syncedUrls: new Set(), // everything we handed gradio: inbound matches are echoes, not uploads
            inflowSeq: 0,
            displayCropped: false,
            renderTimer: null,
        };

        function setActive(btn, on) {
            btn.classList.toggle('forge-btn-active', !!on);
        }

        // Red inner border on a tool icon while its adjustment modifies the
        // input. Only for tools whose button doesn't already show it: toggles
        // (flip, grayscale, invert, edges) carry the yellow active state, and
        // their own key also resets them, so they get no extra marker.
        function setModified(btn, on) {
            btn.classList.toggle('forge-tool-modified', !!on);
        }

        function syncModified() {
            // markers track the current target (layer or whole canvas), so
            // they always say what the visible controls would reset
            const T = target();
            setModified(pickBlackBtn, T.black !== 0);
            setModified(pickWhiteBtn, T.white !== 255);
            setModified(pickGrayBtn, !!T.grayGains);
            // the color correction tool is marked when any of its picks apply
            setModified(colorBtn, T.black !== 0 || T.white !== 255 || !!T.grayGains);
            setModified(gammaBtn, T.gamma !== 1);
            setModified(edgesBtn, edgesActive(T));
            setModified(cropBtn, cropApplied(st));
            setModified(rotateBtn, T.rotate !== 0);
            setModified(penBtn, st.layers.some((l) => l.strokes.length > 0));
            setModified(layersBtn, st.layers.length > 1);
        }

        // ---- pen color selector (photoshop layout: S/V square + hue strip)

        const penColor = () => hsvToHex(st.penHue, st.penSat, st.penVal);

        function drawPenSV() {
            const ctx = penSV.getContext('2d');
            const w = penSV.width, h = penSV.height;
            ctx.fillStyle = hsvToHex(st.penHue, 1, 1);
            ctx.fillRect(0, 0, w, h);
            const white = ctx.createLinearGradient(0, 0, w, 0);
            white.addColorStop(0, 'rgba(255,255,255,1)');
            white.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.fillStyle = white;
            ctx.fillRect(0, 0, w, h);
            const black = ctx.createLinearGradient(0, 0, 0, h);
            black.addColorStop(0, 'rgba(0,0,0,0)');
            black.addColorStop(1, 'rgba(0,0,0,1)');
            ctx.fillStyle = black;
            ctx.fillRect(0, 0, w, h);
        }

        function syncPenColor() {
            const color = penColor();
            penSVCursor.style.left = (st.penSat * penSV.clientWidth) + 'px';
            penSVCursor.style.top = ((1 - st.penVal) * penSV.clientHeight) + 'px';
            penSVCursor.style.background = color;
            penSwatch.style.background = color;
            penHex.textContent = color;
            penCursor.style.borderColor = st.penVal > 0.6 ? '#000000' : '#ffffff';
        }

        function pickPenSV(e) {
            const r = penSV.getBoundingClientRect();
            st.penSat = clamp((e.clientX - r.left) / r.width, 0, 1);
            st.penVal = clamp(1 - (e.clientY - r.top) / r.height, 0, 1);
            syncPenColor();
        }

        // swatch + numeric RGB readout next to a color correction picker
        function syncPickReadout(swatch, label, c) {
            swatch.style.background = c ? 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')' : 'transparent';
            label.textContent = c ? c.r + ' ' + c.g + ' ' + c.b : '—';
        }

        function syncUI() {
            // every adjustment control shows the state of what it would act
            // on: the active layer while the layer tool is engaged, else the
            // whole canvas - switching target re-syncs all sliders/readouts
            const T = target();
            blackSlider.value = T.black;
            blackLabel.textContent = 'black ' + T.black;
            whiteSlider.value = T.white;
            whiteLabel.textContent = 'white ' + T.white;
            gammaSlider.value = Math.round(Math.log2(T.gamma) * 1000);
            gammaLabel.textContent = 'gamma ' + T.gamma.toFixed(3);
            rotSlider.value = T.rotate;
            rotLabel.textContent = 'rotate ' + (+T.rotate).toFixed(1) + '°';
            edgeOpacitySlider.value = T.edgeOpacity;
            edgeOpacityLabel.textContent = 'edge opacity ' + T.edgeOpacity;
            maskOpacitySlider.value = T.maskOpacity;
            maskOpacityLabel.textContent = 'mask opacity ' + T.maskOpacity;
            edgeSensSlider.value = T.edgeSensitivity;
            edgeSensLabel.textContent = 'sensitivity ' + T.edgeSensitivity;
            edgeThickSlider.value = thicknessIndex(T.edgeThickness);
            edgeThickLabel.textContent = 'thickness ' + T.edgeThickness;
            edgeFeatherSlider.value = T.edgeFeather;
            edgeFeatherLabel.textContent = 'feathering ' + T.edgeFeather;
            syncPickReadout(blackSwatch, blackRgb, T.pickedBlack);
            syncPickReadout(whiteSwatch, whiteRgb, T.pickedWhite);
            syncPickReadout(graySwatch, grayRgb, T.grayPicked);
            penSizeSlider.value = st.penSize;
            penSizeLabel.textContent = 'brush ' + Math.round(brushDiameter(st.penSize)) + ' px';
            penOpacitySlider.value = st.penOpacity;
            penOpacityLabel.textContent = 'opacity ' + st.penOpacity;
            penFeatherSlider.value = st.penFeather;
            penFeatherLabel.textContent = 'feathering ' + st.penFeather;
            penHue.value = st.penHue;
            // the pen menu IS the tool: it is open exactly while drawing is
            // armed. Its display must be settled BEFORE the color selector is
            // synced - the cursor position is measured off the square, and a
            // hidden square measures 0.
            penBox.style.display = st.mode === 'pen' ? '' : 'none';
            drawPenSV();
            syncPenColor();
            setActive(penPickBtn, st.penPicking);
            setActive(penBtn, st.mode === 'pen');
            setActive(penEraserBtn, st.penErase);
            // layer targeting (showLayers) outlives sub-tools, but the menu
            // BOX yields the screen space while a picker/pen/crop is front-most
            // (two open boxes would overlap on a small canvas); the yellow
            // layers button and the active-layer outline keep showing the
            // targeting the whole time
            layersBox.style.display = (st.showLayers && st.mode === 'layers') ? '' : 'none';
            setActive(layersBtn, st.showLayers);
            rebuildLayerList();
            updateLayersOverlay();
            colorBox.style.display = st.showColor ? '' : 'none';
            gammaBox.style.display = st.showGamma ? '' : 'none';
            edgesBox.style.display = st.showEdges ? '' : 'none';
            rotateBox.style.display = st.showRotate ? '' : 'none';
            setActive(flipBtn, T.flipH);
            setActive(colorBtn, st.showColor);
            setActive(gammaBtn, st.showGamma);
            setActive(grayscaleBtn, T.grayscale);
            setActive(invertBtn, T.invert);
            setActive(edgesBtn, st.showEdges);
            setActive(rotateBtn, st.showRotate);
            setActive(cropBtn, st.mode === 'crop');
            setActive(pickBlackBtn, st.mode === 'pick-black');
            setActive(pickWhiteBtn, st.mode === 'pick-white');
            setActive(pickGrayBtn, st.mode === 'pick-gray');
            syncModified();
        }

        function setPickCursor(on) {
            // css class with !important beats the inline grab/crosshair cursors
            // that the core canvas code keeps setting on the image and canvas
            container.classList.toggle('forge-picking', !!on);
        }

        function exitMode() {
            const wasCrop = st.mode === 'crop';
            st.mode = null;
            st.cropDrag = null;
            st.grayMask = null;
            st.grayStroke = null;
            st.grayPainting = false;
            st.penStroke = null;
            st.penPicking = false;
            st.layerDrag = null;
            cropBox.classList.remove('forge-cropping');
            overlay.style.display = 'none';
            reticle.style.display = 'none';
            grayOverlay.style.display = 'none';
            penOverlay.style.display = 'none';
            penCursor.style.display = 'none';
            layersOverlay.style.display = 'none';
            setPickCursor(false);
            // layer targeting survives sub-tools: with the layers menu open,
            // an exited picker/pen/crop hands the canvas back to the layer
            // tool instead of dropping to no mode (closing the menu itself
            // clears showLayers first, so a real exit stays a real exit)
            if (st.showLayers && st.original) st.mode = 'layers';
            syncUI();
            // leaving crop-edit by any route (another tool, outside click, ...)
            // must switch the display back from the full image to the cropped
            // result, otherwise the crop looks still active
            if (wasCrop) scheduleRender(st, false);
        }

        // single-active-tool rule across scripts: activating any canvas tool
        // dispatches this event on the container; every tool owner listens and
        // stands down when somebody else claims the canvas (the weight mask
        // tool in the ControlNet extension follows the same protocol)
        function claimTool() {
            container.dispatchEvent(new CustomEvent('forge-canvas-tool', {detail: {owner: 'adjust'}}));
        }

        container.addEventListener('forge-canvas-tool', function (e) {
            if ((st.mode || st.showLayers) && (!e.detail || e.detail.owner !== 'adjust')) {
                st.showLayers = false; // a foreign tool takes the canvas: stand down fully
                exitMode();
            }
        });

        st.resetValues = function () {
            st.flipH = false;
            st.black = 0;
            st.white = 255;
            st.gamma = 1;
            st.grayGains = null;
            st.pickedBlack = null;
            st.pickedWhite = null;
            st.grayPicked = null;
            st.grayscale = false;
            st.invert = false;
            st.edgeSensitivity = 50;
            st.edgeOpacity = 0;
            st.maskOpacity = 0;
            st.edgeThickness = 2;
            st.edgeFeather = 0;
            st.rotate = 0;
            st.crop = null;
            st.compositeCanvas = null;
            st.compositeKey = null;
            st.geomCanvas = null;
            st.geomKey = null;
            st.leveledCanvas = null;
            st.leveledKey = null;
            st.layerDrag = null;
            st.showLayers = false; // fresh content = fresh (whole-canvas) targeting
            rebuildLayerList();
            exitMode();
        };

        // ---- crop overlay geometry (all in container coordinates)

        function updateCropOverlay() {
            if (st.mode !== 'crop' || !st.crop) {
                overlay.style.display = 'none';
                return;
            }
            const r = imgEl.getBoundingClientRect();
            const c = container.getBoundingClientRect();
            if (!r.width || !r.height) {
                overlay.style.display = 'none';
                return;
            }
            const ix = r.left - c.left, iy = r.top - c.top;
            const bx = ix + st.crop.x * r.width;
            const by = iy + st.crop.y * r.height;
            const bw = Math.max(0, st.crop.w * r.width);
            const bh = Math.max(0, st.crop.h * r.height);

            const place = (n, left, top, width, height) => {
                n.style.left = left + 'px';
                n.style.top = top + 'px';
                n.style.width = Math.max(0, width) + 'px';
                n.style.height = Math.max(0, height) + 'px';
            };
            place(shadeN, ix, iy, r.width, by - iy);
            place(shadeS, ix, by + bh, r.width, iy + r.height - (by + bh));
            place(shadeW, ix, by, bx - ix, bh);
            place(shadeE, bx + bw, by, ix + r.width - (bx + bw), bh);
            place(cropBox, bx, by, bw, bh);
            overlay.style.display = 'block';
        }
        st.updateCropOverlay = updateCropOverlay;

        function applyCropDrag(e) {
            const d = st.cropDrag;
            if (!d) return;
            const du = (e.clientX - d.x0) / d.imgW;
            const dv = (e.clientY - d.y0) / d.imgH;
            const minW = 8 / d.imgW, minH = 8 / d.imgH;
            let x = d.start.x, y = d.start.y, w = d.start.w, h = d.start.h;
            if (d.handle === 'move') {
                x = clamp(d.start.x + du, 0, 1 - w);
                y = clamp(d.start.y + dv, 0, 1 - h);
            } else {
                if (d.handle.includes('w')) {
                    const nx = clamp(d.start.x + du, 0, d.start.x + d.start.w - minW);
                    w = d.start.w + (d.start.x - nx);
                    x = nx;
                }
                if (d.handle.includes('e')) {
                    w = clamp(d.start.w + du, minW, 1 - d.start.x);
                }
                if (d.handle.includes('n')) {
                    const ny = clamp(d.start.y + dv, 0, d.start.y + d.start.h - minH);
                    h = d.start.h + (d.start.y - ny);
                    y = ny;
                }
                if (d.handle.includes('s')) {
                    h = clamp(d.start.h + dv, minH, 1 - d.start.y);
                }
            }
            st.crop = {x: x, y: y, w: w, h: h};
            updateCropOverlay();
            syncModified(); // the red border / reset button track the rect live while dragging
        }

        cropBox.querySelectorAll('[data-handle]').forEach((handle) => {
            handle.addEventListener('pointerdown', function (e) {
                if (e.button !== 0 || st.mode !== 'crop' || !st.crop) return;
                e.preventDefault();
                e.stopPropagation();
                const r = imgEl.getBoundingClientRect();
                if (!r.width || !r.height) return;
                st.cropDrag = {
                    handle: this.dataset.handle,
                    x0: e.clientX,
                    y0: e.clientY,
                    imgW: r.width,
                    imgH: r.height,
                    start: Object.assign({}, st.crop),
                };
                cropBox.classList.add('forge-cropping'); // rule-of-thirds guides while held
                try {
                    this.setPointerCapture(e.pointerId);
                } catch (err) {}
            });
            handle.addEventListener('pointermove', function (e) {
                if (!st.cropDrag) return;
                e.preventDefault();
                e.stopPropagation();
                applyCropDrag(e);
            });
            handle.addEventListener('pointerup', function (e) {
                if (!st.cropDrag) return;
                e.preventDefault();
                e.stopPropagation();
                applyCropDrag(e);
                st.cropDrag = null;
                cropBox.classList.remove('forge-cropping');
                try {
                    this.releasePointerCapture(e.pointerId);
                } catch (err) {}
                syncCroppedToGradio(st); // generation always uses the current crop, even mid-edit
                announceImageInfo(st);   // ... and so does the cropped-dimensions readout
            });
            handle.addEventListener('pointercancel', function () {
                st.cropDrag = null;
                cropBox.classList.remove('forge-cropping');
            });
        });

        // ---- pick-pixel reticle (open center so the target pixel stays visible)

        function moveReticle(e) {
            const c = container.getBoundingClientRect();
            reticle.style.left = (e.clientX - c.left) + 'px';
            reticle.style.top = (e.clientY - c.top) + 'px';
            reticle.style.display = 'block';
        }

        function handlePick(e) {
            // single-pixel black/white point picks
            const r = imgEl.getBoundingClientRect();
            const pickType = st.mode;
            const wasCroppedDisplay = st.displayCropped;
            if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) {
                exitMode(); // clicked outside the image: cancel
                return;
            }
            const u = (e.clientX - r.left) / r.width;
            const v = (e.clientY - r.top) / r.height;
            const T = target(); // capture: the async decode below must not retarget
            // stay in pick mode: allow consecutive picks until the user exits
            // (toggle the tool button again or click outside the image)
            ensureSource(st, () => {
                // the source may still have been decoding when the user
                // clicked; don't apply a pick that outlived its mode
                if (st.mode !== pickType) return;
                const geom = ensureGeomCanvas(st);
                const crop = wasCroppedDisplay ? (st.crop || FULL_CROP()) : FULL_CROP();
                const px = clamp(Math.floor((crop.x + u * crop.w) * geom.width), 0, geom.width - 1);
                const py = clamp(Math.floor((crop.y + v * crop.h) * geom.height), 0, geom.height - 1);
                // sample the pre-color-corrected source so repeated picks
                // don't compound: the geom canvas for the whole canvas, the
                // layer's own base raster for a layer-targeted pick (the geom
                // composite already carries the layer's adjustments)
                let d;
                if (T === st) {
                    d = geom.getContext('2d').getImageData(px, py, 1, 1).data;
                } else {
                    const p = stageToLayerLocal(T, geomToStage(st, px + 0.5, py + 0.5, geom.width, geom.height));
                    const base = ensureLayerBaseCanvas(T);
                    const lx = Math.floor(p.x), ly = Math.floor(p.y);
                    if (lx < 0 || ly < 0 || lx >= base.width || ly >= base.height) return; // outside the layer
                    d = base.getContext('2d').getImageData(lx, ly, 1, 1).data;
                }
                if (d[3] < 8) return; // transparent pixel: nothing to sample
                // black/white points act after gray balance, so measure the balanced value
                const g = T.grayGains || {r: 1, g: 1, b: 1};
                const luma = Math.round(
                    0.299 * clamp(d[0] * g.r, 0, 255) +
                    0.587 * clamp(d[1] * g.g, 0, 255) +
                    0.114 * clamp(d[2] * g.b, 0, 255));
                if (pickType === 'pick-black') {
                    T.black = clamp(luma, 0, T.white - 1);
                    T.pickedBlack = {r: d[0], g: d[1], b: d[2]}; // original colors, for the menu readout
                } else {
                    T.white = clamp(luma, T.black + 1, 255);
                    T.pickedWhite = {r: d[0], g: d[1], b: d[2]};
                }
                syncUI();
                scheduleRender(st, false);
            });
        }

        // ---- gray AREA picker: paint mask regions, then neutralize the hue bias
        // Multi-region selection: every drag stroke adds to the mask; toggling
        // the tool button off runs the analysis, clicking outside cancels.

        // screen position -> geom pixel position + brush radius in geom pixels
        function grayMaskPos(e) {
            const r = imgEl.getBoundingClientRect();
            const geom = st.geomCanvas;
            if (!geom || !r.width || !r.height) return null;
            const crop = st.displayCropped ? (st.crop || FULL_CROP()) : FULL_CROP();
            const u = clamp((e.clientX - r.left) / r.width, 0, 1);
            const v = clamp((e.clientY - r.top) / r.height, 0, 1);
            return {
                x: (crop.x + u * crop.w) * geom.width,
                y: (crop.y + v * crop.h) * geom.height,
                // ~12 screen px brush converted to geom resolution
                radius: Math.max(1, 12 * (crop.w * geom.width) / r.width),
            };
        }

        function paintGrayStroke(e, isStart) {
            ensureSource(st, () => {
                if (st.mode !== 'pick-gray') return;
                const geom = ensureGeomCanvas(st);
                if (!st.grayMask || st.grayMask.width !== geom.width || st.grayMask.height !== geom.height) {
                    // first stroke, or geometry changed mid-selection: start a fresh mask
                    const m = document.createElement('canvas');
                    m.width = geom.width;
                    m.height = geom.height;
                    st.grayMask = m;
                    st.grayStroke = null;
                }
                const p = grayMaskPos(e);
                if (!p) return;
                const ctx = st.grayMask.getContext('2d');
                ctx.fillStyle = ctx.strokeStyle = '#3aa0ff';
                ctx.lineCap = ctx.lineJoin = 'round';
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fill();
                if (!isStart && st.grayStroke) {
                    ctx.lineWidth = p.radius * 2;
                    ctx.beginPath();
                    ctx.moveTo(st.grayStroke.x, st.grayStroke.y);
                    ctx.lineTo(p.x, p.y);
                    ctx.stroke();
                }
                st.grayStroke = {x: p.x, y: p.y};
                updateGrayOverlay();
            });
        }

        function updateGrayOverlay() {
            if (st.mode !== 'pick-gray' || !st.grayMask) {
                grayOverlay.style.display = 'none';
                return;
            }
            const r = imgEl.getBoundingClientRect();
            const c = container.getBoundingClientRect();
            if (!r.width || !r.height) {
                grayOverlay.style.display = 'none';
                return;
            }
            grayOverlay.style.left = (r.left - c.left) + 'px';
            grayOverlay.style.top = (r.top - c.top) + 'px';
            grayOverlay.style.width = r.width + 'px';
            grayOverlay.style.height = r.height + 'px';
            const w = Math.max(1, Math.round(r.width));
            const h = Math.max(1, Math.round(r.height));
            if (grayOverlay.width !== w || grayOverlay.height !== h) {
                grayOverlay.width = w;
                grayOverlay.height = h;
            }
            const ctx = grayOverlay.getContext('2d');
            ctx.clearRect(0, 0, w, h);
            ctx.globalAlpha = 0.4;
            const crop = st.displayCropped ? (st.crop || FULL_CROP()) : FULL_CROP();
            const mask = st.grayMask;
            ctx.drawImage(mask,
                crop.x * mask.width, crop.y * mask.height,
                Math.max(1, crop.w * mask.width), Math.max(1, crop.h * mask.height),
                0, 0, w, h);
            grayOverlay.style.display = 'block';
        }
        st.updateGrayOverlay = updateGrayOverlay;

        // Least-squares hue-bias neutralization over all masked pixels: for
        // each channel C find the gain g_C minimizing sum_i (g_C*C_i - L_i)^2
        // with L_i the pixel luma, i.e. g_C = sum(L*C) / sum(C*C) - the
        // algebraically optimal per-channel (diagonal) transform pulling the
        // selection to neutral gray while statistically keeping its luminance.
        // Pixels are always read from the geometry-only canvas (original input
        // colors, never the color-corrected display), so results don't compound.
        function applyGrayArea() {
            const mask = st.grayMask;
            if (!mask || !st.original) {
                exitMode();
                return;
            }
            const T = target(); // capture: exitMode below must not retarget
            ensureSource(st, () => {
                const geom = ensureGeomCanvas(st);
                if (mask.width !== geom.width || mask.height !== geom.height) {
                    exitMode();
                    return;
                }
                const md = mask.getContext('2d').getImageData(0, 0, mask.width, mask.height).data;
                // for a layer-targeted pick the mask (painted in geom space)
                // is projected onto the layer and its own PRE-adjustment base
                // is sampled - same no-compounding rule as the point pickers
                const forLayer = T !== st;
                const base = forLayer ? ensureLayerBaseCanvas(T) : null;
                const gd = (forLayer ? base : geom).getContext('2d')
                    .getImageData(0, 0, forLayer ? base.width : geom.width, forLayer ? base.height : geom.height).data;
                const srcW = forLayer ? base.width : geom.width;
                const srcH = forLayer ? base.height : geom.height;
                let sLR = 0, sLG = 0, sLB = 0, sRR = 0, sGG = 0, sBB = 0, count = 0;
                let sR = 0, sG = 0, sB = 0; // plain means: statistical RGB surrogate for the readout
                for (let i = 0; i < md.length; i += 4) {
                    if (md[i + 3] < 128) continue; // unmasked
                    let j = i;
                    if (forLayer) {
                        const gx = (i / 4) % mask.width, gy = Math.floor((i / 4) / mask.width);
                        const p = stageToLayerLocal(T, geomToStage(st, gx + 0.5, gy + 0.5, mask.width, mask.height));
                        const lx = Math.floor(p.x), ly = Math.floor(p.y);
                        if (lx < 0 || ly < 0 || lx >= srcW || ly >= srcH) continue; // outside the layer
                        j = (ly * srcW + lx) * 4;
                    }
                    if (gd[j + 3] < 8) continue; // transparent
                    const R = gd[j], G = gd[j + 1], B = gd[j + 2];
                    if (R >= 254 || G >= 254 || B >= 254) continue; // clipped channel: cast info destroyed
                    const L = 0.299 * R + 0.587 * G + 0.114 * B;
                    if (L < 4) continue; // near-black: gain estimate is ill-conditioned
                    sLR += L * R; sRR += R * R;
                    sLG += L * G; sGG += G * G;
                    sLB += L * B; sBB += B * B;
                    sR += R; sG += G; sB += B;
                    count++;
                }
                exitMode();
                if (!count) return; // nothing usable selected: behave like cancel
                T.grayGains = {
                    r: clamp(sRR > 0 ? sLR / sRR : 1, 0.2, 5),
                    g: clamp(sGG > 0 ? sLG / sGG : 1, 0.2, 5),
                    b: clamp(sBB > 0 ? sLB / sBB : 1, 0.2, 5),
                };
                T.grayPicked = {
                    r: Math.round(sR / count),
                    g: Math.round(sG / count),
                    b: Math.round(sB / count),
                };
                syncUI();
                scheduleRender(st, false);
            });
        }

        // ---- pen: draw into the ACTIVE LAYER (eraser: cut into it)
        //
        // Points are recorded in stage pixels right away, so a pan or zoom in
        // the middle of a stroke cannot bend it: the live preview is
        // re-projected from those points on every redraw instead of being kept
        // in screen coordinates. On commit the stroke is converted into the
        // active layer's local pixels, so it follows the layer through every
        // later move/scale.

        // mapping between STAGE pixels and the displayed image, valid for the
        // current pan/zoom/crop/geometry; also used by the layer tool
        function penView() {
            const r = imgEl.getBoundingClientRect();
            const geom = st.geomCanvas;
            if (!geom || !st.stageW || !r.width || !r.height) return null;
            const crop = st.displayCropped ? (st.crop || FULL_CROP()) : FULL_CROP();
            const rad = st.rotate * Math.PI / 180;
            const cos = Math.cos(rad), sin = Math.sin(rad);
            const nw = st.stageW, nh = st.stageH;
            return {
                rect: r,
                // screen px per stage px (rotation preserves scale)
                k: r.width / (crop.w * geom.width),
                toScreen: function (p) {
                    const dx = p.x - nw / 2, dy = p.y - nh / 2;
                    let gx = dx * cos - dy * sin;
                    const gy = dx * sin + dy * cos;
                    if (st.flipH) gx = -gx;
                    gx += geom.width / 2;
                    return {
                        x: ((gx / geom.width) - crop.x) / crop.w * r.width,
                        y: (((gy + geom.height / 2) / geom.height) - crop.y) / crop.h * r.height,
                    };
                },
                toOriginal: function (clientX, clientY) {
                    const u = (clientX - r.left) / r.width;
                    const v = (clientY - r.top) / r.height;
                    let dx = (crop.x + u * crop.w) * geom.width - geom.width / 2;
                    const dy = (crop.y + v * crop.h) * geom.height - geom.height / 2;
                    if (st.flipH) dx = -dx;
                    // inverse rotation
                    return {
                        x: dx * cos + dy * sin + nw / 2,
                        y: -dx * sin + dy * cos + nh / 2,
                    };
                },
            };
        }

        // the layer pen/eraser strokes land on; the stack is never empty
        // while content exists, and a stale index clamps to the top layer
        function activeLayerObj() {
            if (!st.layers.length) return null;
            if (st.activeLayer < 0 || st.activeLayer >= st.layers.length) {
                st.activeLayer = st.layers.length - 1;
            }
            return st.layers[st.activeLayer];
        }

        // what the adjustment controls act on: the active layer while the
        // layer tool is engaged, otherwise the whole canvas (st). Both carry
        // identically named fields, so handlers just write through target().
        function target() {
            return (st.showLayers && activeLayerObj()) || st;
        }

        function penStyle() {
            const d = brushDiameter(st.penSize);
            const f = st.penFeather / 100;
            return {
                color: penColor(),
                alpha: st.penOpacity / 100,
                // feathering splits the radius: the core shrinks by up to half
                // while an equally large blur grows outwards, so the visible
                // brush keeps roughly its diameter and only gets softer
                radius: Math.max(0.05, d / 2 * (1 - 0.5 * f)),
                blur: d / 2 * 0.5 * f,
            };
        }

        // overlay covering the displayed image, carrying the stroke in progress
        function updatePenOverlay() {
            if (st.mode !== 'pen' || !st.penStroke) {
                penOverlay.style.display = 'none';
                return;
            }
            const view = penView();
            if (!view) {
                penOverlay.style.display = 'none';
                return;
            }
            const c = container.getBoundingClientRect();
            const r = view.rect;
            penOverlay.style.left = (r.left - c.left) + 'px';
            penOverlay.style.top = (r.top - c.top) + 'px';
            penOverlay.style.width = r.width + 'px';
            penOverlay.style.height = r.height + 'px';
            const w = Math.max(1, Math.round(r.width));
            const h = Math.max(1, Math.round(r.height));
            if (penOverlay.width !== w || penOverlay.height !== h) {
                penOverlay.width = w;
                penOverlay.height = h;
            }
            const ctx = penOverlay.getContext('2d');
            ctx.clearRect(0, 0, w, h);
            const s = st.penStroke;
            // a destination-out stroke would cut the transparent overlay into
            // nothing visible, so the eraser previews as a translucent red
            // trace instead - the cut happens on commit, in the layer raster
            drawStrokes(ctx, [{
                pts: s.pts.map(view.toScreen),
                color: s.erase ? '#ff4d4d' : s.color,
                alpha: s.erase ? Math.min(0.5, s.alpha) : s.alpha,
                radius: s.radius * view.k,
                blur: s.blur * view.k,
            }]);
            penOverlay.style.display = 'block';
        }
        st.updatePenOverlay = updatePenOverlay;

        // brush outline at the pointer, in the brush's real (zoom-aware) size
        function movePenCursor(e) {
            const view = penView();
            if (!view) {
                penCursor.style.display = 'none';
                return;
            }
            const c = container.getBoundingClientRect();
            const style = penStyle();
            const d = Math.max(4, (2 * style.radius + 2 * style.blur) * view.k);
            penCursor.style.width = d + 'px';
            penCursor.style.height = d + 'px';
            penCursor.style.left = (e.clientX - c.left - d / 2) + 'px';
            penCursor.style.top = (e.clientY - c.top - d / 2) + 'px';
            penCursor.style.display = 'block';
        }

        function penPoint(e) {
            const view = penView();
            return view ? view.toOriginal(e.clientX, e.clientY) : null;
        }

        // eyedropper: takes the color the user actually SEES - the fully
        // processed canvas (paint included, since strokes are part of the
        // raster now), not the untouched original the b/w point pickers read
        function pickPenColor(e) {
            const source = st.leveledCanvas || st.geomCanvas;
            const r = imgEl.getBoundingClientRect();
            if (!source || !r.width || !r.height) return false;
            if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return false;
            const crop = st.displayCropped ? (st.crop || FULL_CROP()) : FULL_CROP();
            const u = (e.clientX - r.left) / r.width;
            const v = (e.clientY - r.top) / r.height;
            const px = clamp(Math.floor((crop.x + u * crop.w) * source.width), 0, source.width - 1);
            const py = clamp(Math.floor((crop.y + v * crop.h) * source.height), 0, source.height - 1);
            const d = source.getContext('2d').getImageData(px, py, 1, 1).data;
            if (d[3] < 8) return false; // transparent: nothing to take
            const hsv = rgbToHsv(d[0], d[1], d[2]);
            // a gray pick carries no hue: keep the hue strip where it is, so
            // the square's cursor lands on the same column the user left it
            if (hsv.s > 0.001) st.penHue = hsv.h;
            st.penSat = hsv.s;
            st.penVal = hsv.v;
            return true;
        }

        function startPenStroke(e) {
            const p = penPoint(e);
            if (!p) return;
            st.penStroke = Object.assign(penStyle(), {pts: [p], erase: st.penErase});
            updatePenOverlay();
        }

        function extendPenStroke(e) {
            if (!st.penStroke) return;
            const p = penPoint(e);
            if (!p) return;
            const pts = st.penStroke.pts;
            const last = pts[pts.length - 1];
            // drop sub-pixel jitter: it only inflates the point list
            if (Math.abs(p.x - last.x) < 0.4 && Math.abs(p.y - last.y) < 0.4) return;
            pts.push(p);
            updatePenOverlay();
        }

        // the stroke becomes part of the ACTIVE LAYER: converted from stage
        // pixels into the layer's local pixels (so it follows the layer
        // through later moves/scales), appended to that layer's stroke list,
        // which invalidates the render caches and re-renders (and re-uploads)
        // the composite with the stroke baked in
        function commitPenStroke() {
            const stroke = st.penStroke;
            st.penStroke = null;
            penOverlay.style.display = 'none';
            if (!stroke || !stroke.pts.length) return;
            const L = activeLayerObj();
            if (!L || !L.w) return;
            const s = Math.max(1e-6, L.scale);
            L.strokes.push({
                // full inverse placement (position/scale/flip/rotation), so
                // paint lands where the cursor was even on a transformed layer
                pts: stroke.pts.map((p) => stageToLayerLocal(L, p)),
                color: stroke.color,
                alpha: stroke.alpha,
                radius: stroke.radius / s,
                blur: stroke.blur / s,
                erase: stroke.erase,
            });
            L.strokeKey++;
            syncModified();
            scheduleRender(st, false);
        }

        function repaintFromStrokes(layer) {
            layer.strokeKey++;
            syncModified();
            scheduleRender(st, false);
        }

        // ---- toolbar buttons

        // Reveal the action buttons (hidden in the template so widgets without JS
        // wiring stay clean).
        //
        // Driven by canvas_nodes.js's contract, NOT by a CSS class. This line
        // used to be `querySelectorAll('.forge-adjust-control')`, and the four
        // weight-mask buttons carry `.forge-wmask-control` instead - so they were
        // injected, wired, and never shown. A missing class produces an empty
        // NodeList, which is not an error; the feature just silently vanished.
        // The contract reveals every injected button except an explicit DEFERRED
        // few, so the default for anything new is VISIBLE. See canvas_nodes.js.
        // ALWAYS reveal by class first, then let the contract reveal on top.
        //
        // Not redundancy for its own sake. attach() is ~1800 lines in one
        // function with no error boundary: anything that throws here aborts
        // everything BELOW it, including the Topaz probe ~70 lines down, and
        // gradio swallows the exception - the user sees a toolbar with pieces
        // missing and no message. The class sweep needs nothing from
        // canvas_nodes.js, so it survives that module being stale, missing, or
        // broken; the contract sweep then adds anything the classes do not
        // cover. Both are idempotent.
        //
        // `display = ''` clears the inline override and defers to CSS - it is
        // NOT "make visible". A rule like style.css's
        // `.cnet-output-mask-group .forge-adjust-control { display: none
        // !important }` still wins, which is deliberate: the output-mask canvas
        // is meant to have no tool chrome.
        try {
            container.querySelectorAll('.forge-adjust-control, .forge-wmask-control')
                .forEach((n) => { n.style.display = ''; });
        } catch (e) {
            console.error('[cnpro] class-based toolbar reveal failed', e);
        }
        try {
            if (window.cnproCanvasNodes && window.cnproCanvasNodes.revealToolbar) {
                window.cnproCanvasNodes.revealToolbar(uuid);
            }
        } catch (e) {
            console.error('[cnpro] contract toolbar reveal failed - falling back to ' +
                          'the class sweep above', e);
        }

        // ---- Topaz tools (1-click): bake the current adjustments + crop, send
        // the result to the local tpai.exe through the server, and load the
        // processed image back as a brand-new image (adjustments reset by the
        // patched uploadBase64, exactly like a fresh upload). The buttons are
        // revealed only when the server reports tpai.exe available.

        // current effective image (adjustments + crop baked) - the same thing
        // gradio holds; direct src fallback covers images pushed before the
        // adjustment layer attached
        function currentImageDataUrl(callback) {
            if (st.original) {
                ensureSource(st, () => {
                    callback(cropFromCanvas(ensureLeveledCanvas(st), st.crop).toDataURL('image/png'));
                });
            } else if (imgEl.src && imgEl.src.startsWith('data:image')) {
                callback(imgEl.src);
            } else {
                callback(null);
            }
        }

        function wireTopazTool(btn, tool) {
            if (!btn) return;
            btn.addEventListener('click', function () {
                if (btn.classList.contains('forge-btn-busy')) return;
                st.showLayers = false; // the result is the flattened stack: layer targeting ends here
                exitMode(); // a live crop/pick session would fight the image swap
                currentImageDataUrl((dataUrl) => {
                    if (!dataUrl) return; // nothing to process
                    btn.classList.add('forge-btn-busy');
                    fetch('./forge-canvas/topaz/process', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: dataUrl, tool: tool }),
                    })
                        .then(async (r) => {
                            const data = await r.json().catch(() => ({}));
                            if (!r.ok || !data.image) {
                                throw new Error(data.error || ('HTTP ' + r.status));
                            }
                            // the processed image already IS the whole (baked)
                            // content - it must never be routed into a layer
                            st.forceReplace = true;
                            fc.uploadBase64(data.image);
                        })
                        .catch((err) => {
                            console.warn('[forge canvas] Topaz ' + tool + ' failed:', err);
                            btn.classList.add('forge-btn-error');
                            setTimeout(() => btn.classList.remove('forge-btn-error'), 1600);
                        })
                        .finally(() => btn.classList.remove('forge-btn-busy'));
                });
            });
        }

        const topazHqBtn = el('topazHqButton_');
        const topazDenoiseBtn = el('topazDenoiseButton_');
        const topazMpxBtn = el('topazMpxButton_');
        wireTopazTool(topazHqBtn, 'hq');
        wireTopazTool(topazDenoiseBtn, 'denoise');
        wireTopazTool(topazMpxBtn, 'mpx1');
        if (topazHqBtn || topazDenoiseBtn || topazMpxBtn) {
            fetchTopazStatus().then((status) => {
                if (!status.available) return;
                if (topazHqBtn) {
                    topazHqBtn.title = 'Topaz enhance at the SAME dimensions' +
                        ' (Photo AI CLI upscaler at scale 1: minor denoise 100, minor deblur 1, fix compression 90);' +
                        ' replaces the image with the enhanced result, adjustments and crop are baked in first';
                    topazHqBtn.style.display = '';
                }
                if (topazDenoiseBtn) {
                    topazDenoiseBtn.title = 'Topaz denoise (Photo AI CLI, model-chosen automatic strength);' +
                        ' replaces the image with the denoised result, adjustments and crop are baked in first';
                    topazDenoiseBtn.style.display = '';
                }
                if (topazMpxBtn) {
                    topazMpxBtn.title = 'Resample to ~1 Mpx (1024×1024 px) keeping the aspect ratio, with both' +
                        ' dimensions divisible by 16; Topaz Photo AI does the resampling (up or down),' +
                        ' adjustments, pen strokes and crop are baked in first';
                    topazMpxBtn.style.display = '';
                }
            });
        }

        function togglePickMode(name) {
            if (!st.original) return;
            const on = st.mode !== name;
            exitMode();
            if (on) {
                st.mode = name;
                setPickCursor(true);
                if (indicator) indicator.style.display = 'none';
                // warm the geometry cache so mask painting starts synchronously
                if (name === 'pick-gray') ensureSource(st, () => ensureGeomCanvas(st));
                syncUI();
                claimTool();
            }
        }

        flipBtn.addEventListener('click', function () {
            if (!st.original) return;
            const T = target();
            T.flipH = !T.flipH;
            // mirror the crop rect together with the image, so the flip applies
            // to the cropped content instead of the crop window sliding away
            // (whole-canvas flips only - a layer flip happens inside the stage)
            if (T === st && st.crop) {
                st.crop = Object.assign({}, st.crop, {
                    x: clamp(1 - st.crop.x - st.crop.w, 0, 1),
                });
            }
            syncUI();
            scheduleRender(st, T === st);
        });

        colorBtn.addEventListener('click', function () {
            st.showColor = !st.showColor;
            syncUI();
        });

        pickBlackBtn.addEventListener('click', function () {
            togglePickMode('pick-black');
        });

        pickWhiteBtn.addEventListener('click', function () {
            togglePickMode('pick-white');
        });

        pickGrayBtn.addEventListener('click', function () {
            if (st.mode === 'pick-gray') {
                applyGrayArea(); // toggle off = analyze the selection and apply
            } else {
                togglePickMode('pick-gray');
            }
        });

        gammaBtn.addEventListener('click', function () {
            st.showGamma = !st.showGamma;
            syncUI();
        });

        grayscaleBtn.addEventListener('click', function () {
            if (!st.original) return;
            const T = target();
            T.grayscale = !T.grayscale;
            syncUI();
            scheduleRender(st, false);
        });

        invertBtn.addEventListener('click', function () {
            if (!st.original) return;
            const T = target();
            T.invert = !T.invert;
            syncUI();
            scheduleRender(st, false);
        });

        // menu toggle only: the effect stays exactly as its settings say
        edgesBtn.addEventListener('click', function () {
            st.showEdges = !st.showEdges;
            syncUI();
        });

        edgeSensSlider.addEventListener('input', function () {
            const T = target();
            T.edgeSensitivity = Math.round(+this.value);
            edgeSensLabel.textContent = 'sensitivity ' + T.edgeSensitivity;
            scheduleRender(st, false);
        });

        // the opacities decide whether the effect is active at all, so they
        // also drive the red modified border on the edges button
        edgeOpacitySlider.addEventListener('input', function () {
            const T = target();
            T.edgeOpacity = Math.round(+this.value);
            edgeOpacityLabel.textContent = 'edge opacity ' + T.edgeOpacity;
            syncModified();
            scheduleRender(st, false);
        });

        maskOpacitySlider.addEventListener('input', function () {
            const T = target();
            T.maskOpacity = Math.round(+this.value);
            maskOpacityLabel.textContent = 'mask opacity ' + T.maskOpacity;
            syncModified();
            scheduleRender(st, false);
        });

        edgeThickSlider.addEventListener('input', function () {
            const T = target();
            T.edgeThickness = THICKNESS_STOPS[clamp(Math.round(+this.value), 0, THICKNESS_STOPS.length - 1)];
            edgeThickLabel.textContent = 'thickness ' + T.edgeThickness;
            scheduleRender(st, false);
        });

        edgeFeatherSlider.addEventListener('input', function () {
            const T = target();
            T.edgeFeather = clamp(Math.round(+this.value), 0, 100);
            edgeFeatherLabel.textContent = 'feathering ' + T.edgeFeather;
            scheduleRender(st, false);
        });

        rotateBtn.addEventListener('click', function () {
            st.showRotate = !st.showRotate;
            syncUI();
        });

        cropBtn.addEventListener('click', function () {
            if (!st.original) return;
            if (st.mode === 'crop') {
                // apply: keep the crop, hide handles, display the cropped result
                if (st.crop && st.crop.x < 0.0005 && st.crop.y < 0.0005 && st.crop.w > 0.999 && st.crop.h > 0.999) {
                    st.crop = null; // full-frame crop means no crop
                }
                exitMode();
                scheduleRender(st, false);
            } else {
                exitMode();
                st.mode = 'crop';
                if (!st.crop) st.crop = FULL_CROP();
                if (indicator) indicator.style.display = 'none';
                syncUI();
                scheduleRender(st, false); // switches the display to the full image; overlay follows via drawImage
                claimTool();
            }
        });

        // ---- pen controls

        penBtn.addEventListener('click', function () {
            if (st.mode === 'pen') {
                exitMode();
                return;
            }
            if (!st.original) return;
            exitMode();
            st.mode = 'pen';
            setPickCursor(true); // the brush ring replaces the OS cursor
            if (indicator) indicator.style.display = 'none';
            // warm the geometry cache so the first stroke maps synchronously
            ensureSource(st, () => {
                if (st.mode === 'pen') ensureGeomCanvas(st);
            });
            syncUI();
            claimTool();
        });

        penHue.addEventListener('input', function () {
            st.penHue = clamp(+this.value, 0, 360);
            drawPenSV();
            syncPenColor();
        });

        penSV.addEventListener('pointerdown', function (e) {
            if (e.button !== 0) return;
            e.preventDefault();
            st.penSVDrag = true;
            try {
                this.setPointerCapture(e.pointerId);
            } catch (err) {}
            pickPenSV(e);
        });

        penSV.addEventListener('pointermove', function (e) {
            if (st.penSVDrag) pickPenSV(e);
        });

        penSV.addEventListener('pointerup', function (e) {
            st.penSVDrag = false;
            try {
                this.releasePointerCapture(e.pointerId);
            } catch (err) {}
        });

        penPickBtn.addEventListener('click', function () {
            st.penPicking = !st.penPicking;
            syncUI();
        });

        penSizeSlider.addEventListener('input', function () {
            st.penSize = clamp(Math.round(+this.value), 0, 100);
            penSizeLabel.textContent = 'brush ' + Math.round(brushDiameter(st.penSize)) + ' px';
        });

        penOpacitySlider.addEventListener('input', function () {
            st.penOpacity = clamp(Math.round(+this.value), 1, 100);
            penOpacityLabel.textContent = 'opacity ' + st.penOpacity;
        });

        penFeatherSlider.addEventListener('input', function () {
            st.penFeather = clamp(Math.round(+this.value), 0, 100);
            penFeatherLabel.textContent = 'feathering ' + st.penFeather;
        });

        penUndoBtn.addEventListener('click', function () {
            const L = activeLayerObj();
            if (!L || !L.strokes.length) return;
            L.strokes.pop();
            repaintFromStrokes(L);
        });

        penClearBtn.addEventListener('click', function () {
            const L = activeLayerObj();
            if (!L || !L.strokes.length) return;
            L.strokes = [];
            repaintFromStrokes(L);
        });

        penEraserBtn.addEventListener('click', function () {
            st.penErase = !st.penErase;
            syncUI();
        });

        // ---- layers: continuously editable stack upstream of everything
        //
        // The layers menu IS the tool (like the pen): while it is open, click
        // selects the topmost layer under the pointer, drag moves the active
        // layer, wheel scales it around the pointer. All transforms live in
        // stage pixels; penView() supplies the exact screen<->stage mapping,
        // so selection and dragging stay correct under pan/zoom/crop/rotation.

        // the four stage-space corners of a layer, mapped to screen space
        function layerScreenQuad(view, L) {
            // local corners through the full placement transform: the quad
            // (outline, ghost, hit test) follows the layer's flip/rotation
            return [
                view.toScreen(layerLocalToStage(L, {x: 0, y: 0})),
                view.toScreen(layerLocalToStage(L, {x: L.w, y: 0})),
                view.toScreen(layerLocalToStage(L, {x: L.w, y: L.h})),
                view.toScreen(layerLocalToStage(L, {x: 0, y: L.h})),
            ];
        }

        function updateLayersOverlay() {
            // visible while layer targeting is on, even mid picker/pen use -
            // the outline always shows what the controls act on
            const L = (st.mode === 'layers' || st.showLayers) ? activeLayerObj() : null;
            const view = L ? penView() : null;
            if (!L || !view || !L.w) {
                layersOverlay.style.display = 'none';
                return;
            }
            const c = container.getBoundingClientRect();
            const r = view.rect;
            layersOverlay.style.left = (r.left - c.left) + 'px';
            layersOverlay.style.top = (r.top - c.top) + 'px';
            layersOverlay.style.width = r.width + 'px';
            layersOverlay.style.height = r.height + 'px';
            const w = Math.max(1, Math.round(r.width));
            const h = Math.max(1, Math.round(r.height));
            if (layersOverlay.width !== w || layersOverlay.height !== h) {
                layersOverlay.width = w;
                layersOverlay.height = h;
            }
            const ctx = layersOverlay.getContext('2d');
            ctx.clearRect(0, 0, w, h);
            const q = layerScreenQuad(view, L);
            // ghost: while the layer is being dragged/scaled the full pipeline
            // render lags behind (it only runs debounced / on release), so the
            // layer bitmap itself is drawn translucently at its live position.
            // The affine transform is built from three mapped corners, which
            // reproduces rotation and flip without knowing about them.
            const ghost = st.layerDrag || performance.now() < (st.ghostUntil || 0);
            if (ghost && (L.img || L.strokes.length)) {
                const lc = ensureLayerCanvas(L);
                ctx.save();
                ctx.globalAlpha = 0.5;
                ctx.setTransform(
                    (q[1].x - q[0].x) / L.w, (q[1].y - q[0].y) / L.w,
                    (q[3].x - q[0].x) / L.h, (q[3].y - q[0].y) / L.h,
                    q[0].x, q[0].y);
                ctx.drawImage(lc, 0, 0);
                ctx.restore();
            }
            ctx.save();
            ctx.strokeStyle = '#3aa0ff';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.moveTo(q[0].x, q[0].y);
            for (let i = 1; i < 4; i++) ctx.lineTo(q[i].x, q[i].y);
            ctx.closePath();
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#3aa0ff';
            ctx.font = '11px sans-serif';
            ctx.fillText('L' + (st.activeLayer + 1) + '  ' + Math.round(L.scale * 100) + '%',
                Math.min(q[0].x, q[1].x) + 4, Math.min(q[0].y, q[3].y) - 5);
            ctx.restore();
            layersOverlay.style.display = 'block';
        }
        st.updateLayersOverlay = updateLayersOverlay;

        function rebuildLayerList() {
            if (!layerList) return;
            layerList.textContent = '';
            // rows run topmost-first: the visual stacking order, top of the
            // pile at the top of the list
            for (let i = st.layers.length - 1; i >= 0; i--) {
                const L = st.layers[i];
                const idx = i;
                const row = document.createElement('div');
                row.className = 'forge-layer-row' + (idx === st.activeLayer ? ' forge-layer-row-active' : '');
                row.title = 'Active layer: click to select. The pen and eraser draw into the active layer; drag moves it, wheel scales it.';
                const label = document.createElement('span');
                label.className = 'forge-layer-label';
                label.textContent = 'L' + (idx + 1)
                    + (L.src ? (L.w ? ' ' + L.w + '×' + L.h : '') : ' empty')
                    + ' @' + Math.round(L.scale * 100) + '%'
                    + (L.blend === 'lighten' ? ' max' : '')
                    + (L.strokes.length ? ' ✎' + L.strokes.length : '')
                    // the layer carries its own adjustments (flip/rotation/color)
                    + (L.flipH || L.rotate || layerColorAdjusted(L) ? ' ±' : '');
                row.appendChild(label);
                const mkBtn = (txt, title, disabled, fn) => {
                    const b = document.createElement('button');
                    b.className = 'forge-btn forge-no-select forge-layer-btn';
                    b.textContent = txt;
                    b.title = title;
                    b.disabled = !!disabled;
                    b.addEventListener('click', (e) => {
                        e.stopPropagation();
                        fn();
                    });
                    row.appendChild(b);
                };
                mkBtn('◍', L.blend === 'lighten'
                    ? 'Blend: lighten (per-pixel max) - the black background of a control map stays transparent to what lies below; click for normal'
                    : 'Blend: normal (opaque over) - click for lighten (per-pixel max), the union mode for bright-on-black control maps (canny/pose/depth)',
                    false, () => {
                    L.blend = L.blend === 'lighten' ? 'normal' : 'lighten';
                    syncUI();
                    scheduleRender(st, false);
                });
                mkBtn('▲', 'Raise the layer one step', idx === st.layers.length - 1, () => {
                    const t = st.layers[idx];
                    st.layers[idx] = st.layers[idx + 1];
                    st.layers[idx + 1] = t;
                    if (st.activeLayer === idx) st.activeLayer = idx + 1;
                    else if (st.activeLayer === idx + 1) st.activeLayer = idx;
                    st.original = stackOriginal(st);
                    syncUI();
                    scheduleRender(st, false);
                });
                mkBtn('▼', 'Lower the layer one step', idx === 0, () => {
                    const t = st.layers[idx];
                    st.layers[idx] = st.layers[idx - 1];
                    st.layers[idx - 1] = t;
                    if (st.activeLayer === idx) st.activeLayer = idx - 1;
                    else if (st.activeLayer === idx - 1) st.activeLayer = idx;
                    st.original = stackOriginal(st);
                    syncUI();
                    scheduleRender(st, false);
                });
                mkBtn('✕', 'Delete the layer (deleting the last one clears the canvas)', false, () => {
                    st.layers.splice(idx, 1);
                    // a stack with neither raster nor strokes left is an empty
                    // canvas, not a pile of blank layers
                    if (!st.layers.length || !stackOriginal(st)) {
                        fc.removeImage();
                        return;
                    }
                    if (st.activeLayer >= st.layers.length) st.activeLayer = st.layers.length - 1;
                    st.original = stackOriginal(st);
                    syncUI();
                    scheduleRender(st, false);
                });
                row.addEventListener('click', () => {
                    st.activeLayer = idx;
                    syncUI();
                    updateLayersOverlay();
                });
                layerList.appendChild(row);
            }
        }

        function enterLayersMode() {
            exitMode();
            st.showLayers = true; // adjustment controls now target the active layer
            st.mode = 'layers';
            if (indicator) indicator.style.display = 'none';
            ensureSource(st, () => {
                if (st.mode === 'layers') {
                    ensureGeomCanvas(st);
                    updateLayersOverlay();
                }
            });
            syncUI();
            claimTool();
        }

        // add an image on TOP of the existing content as a new layer (the
        // ordinary upload path replaces instead - adding is always explicit).
        // On an empty canvas the image simply becomes the stage.
        function addLayer(dataUrl) {
            if (typeof dataUrl !== 'string' || !dataUrl.startsWith('data:image')) return false;
            if (!st.layers.length) {
                fc.uploadBase64(dataUrl);
                return true;
            }
            const layer = newLayer(dataUrl);
            st.layers.push(layer);
            st.activeLayer = st.layers.length - 1;
            ensureSource(st, () => {
                // fit inside the stage, centered - never larger than the stage
                const fit = Math.min(1, st.stageW / Math.max(1, layer.w), st.stageH / Math.max(1, layer.h));
                layer.scale = fit;
                layer.x = (st.stageW - layer.w * fit) / 2;
                layer.y = (st.stageH - layer.h * fit) / 2;
                if (st.mode !== 'layers') enterLayersMode();
                syncUI();
                scheduleRender(st, false);
            });
            return true;
        }
        st.addLayerFromDataUrl = addLayer;

        // replace the ACTIVE layer's raster (the patched uploadBase64 routes
        // every inflow here while the layer tool is open). The layer keeps its
        // z-position and blend; its strokes go with the old raster. The new
        // raster is fitted inside the stage and centered, exactly like an
        // explicitly added layer.
        function setActiveLayerRaster(dataUrl) {
            const L = activeLayerObj();
            if (!L) return;
            L.src = dataUrl;
            L.img = null;
            L.w = 0;
            L.h = 0;
            L.baseCanvas = null;
            L.baseKey = null;
            L.canvas = null;
            L.canvasKey = null;
            L.strokes = [];
            L.strokeKey++;
            st.original = stackOriginal(st);
            ensureSource(st, () => {
                if (L.src !== dataUrl) return; // replaced again meanwhile
                const fit = Math.min(1, st.stageW / Math.max(1, L.w), st.stageH / Math.max(1, L.h));
                L.scale = fit;
                L.x = (st.stageW - L.w * fit) / 2;
                L.y = (st.stageH - L.h * fit) / 2;
                syncUI();
                rebuildLayerList();
                updateLayersOverlay();
                scheduleRender(st, false);
            });
        }
        st.setActiveLayerRaster = setActiveLayerRaster;

        layersBtn.addEventListener('click', function () {
            if (st.showLayers) {
                st.showLayers = false; // closing the menu returns to whole-canvas targeting
                exitMode();
                return;
            }
            if (!st.original) return;
            enterLayersMode();
        });

        // + creates an EMPTY logical layer on top (no file dialog): a
        // transparent stage-sized raster, immediately active, waiting for its
        // content - drop / paste / open button / gradio push all fill the
        // active layer while this tool is open, and the pen paints on it as-is
        layerAddBtn.addEventListener('click', function () {
            if (!st.layers.length) return; // the first upload defines the stage
            ensureSource(st, () => {
                const L = newLayer(null);
                L.w = st.stageW;
                L.h = st.stageH;
                st.layers.push(L);
                st.activeLayer = st.layers.length - 1;
                syncUI();
                rebuildLayerList();
                updateLayersOverlay();
            });
        });

        function layerDragMove(e) {
            const d = st.layerDrag;
            const L = activeLayerObj();
            const view = penView();
            if (!d || !L || !view) return;
            // screen delta -> stage delta through the exact inverse mapping,
            // so dragging tracks the pointer under rotation and flip too
            const p = view.toOriginal(e.clientX, e.clientY);
            L.x = d.startX + (p.x - d.px);
            L.y = d.startY + (p.y - d.py);
            updateLayersOverlay();
        }

        container.addEventListener('wheel', function (e) {
            if (st.mode !== 'layers' || !st.original) return;
            if (toolbar.contains(e.target)) return;
            const view = penView();
            const L = activeLayerObj();
            if (!view || !L) return;
            const r = view.rect;
            if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;
            e.preventDefault();
            e.stopPropagation();
            const f = e.deltaY < 0 ? 1.1 : 1 / 1.1;
            const next = clamp(L.scale * f, 0.02, 50);
            const applied = next / L.scale;
            if (applied === 1) return;
            // scale around the pointer: the stage point under it stays fixed
            const pivot = view.toOriginal(e.clientX, e.clientY);
            L.x = pivot.x - (pivot.x - L.x) * applied;
            L.y = pivot.y - (pivot.y - L.y) * applied;
            L.scale = next;
            st.ghostUntil = performance.now() + 350;
            updateLayersOverlay();
            rebuildLayerList();
            scheduleRender(st, false);
        }, {capture: true, passive: false});

        // ---- reset buttons: b/w/g points reset lives in the toolbar (the
        //      pickers have no menu of their own), the others sit inside
        //      their tool's menu; every reset re-renders from the original

        pointsResetBtn.addEventListener('click', function () {
            const T = target();
            if (T.black === 0 && T.white === 255 && !T.grayGains) return;
            T.black = 0;
            T.white = 255;
            T.grayGains = null;
            T.pickedBlack = null;
            T.pickedWhite = null;
            T.grayPicked = null;
            syncUI();
            scheduleRender(st, false);
        });

        gammaResetBtn.addEventListener('click', function () {
            const T = target();
            if (T.gamma === 1) return;
            T.gamma = 1;
            syncUI();
            scheduleRender(st, false);
        });

        // back to defaults; opacities 0 also deactivate the effect
        edgesResetBtn.addEventListener('click', function () {
            const T = target();
            T.edgeSensitivity = 50;
            T.edgeOpacity = 0;
            T.maskOpacity = 0;
            T.edgeThickness = 2;
            T.edgeFeather = 0;
            syncUI();
            scheduleRender(st, false);
        });

        rotateResetBtn.addEventListener('click', function () {
            const T = target();
            if (T.rotate === 0) return;
            T.rotate = 0;
            syncUI();
            scheduleRender(st, T === st);
        });

        // ---- sliders (revealed by their toggle buttons)

        blackSlider.addEventListener('input', function () {
            const T = target();
            let v = Math.round(+this.value);
            if (v >= T.white) {
                v = T.white - 1;
                this.value = v;
            }
            T.black = v;
            blackLabel.textContent = 'black ' + v;
            syncModified();
            scheduleRender(st, false);
        });

        whiteSlider.addEventListener('input', function () {
            const T = target();
            let v = Math.round(+this.value);
            if (v <= T.black) {
                v = T.black + 1;
                this.value = v;
            }
            T.white = v;
            whiteLabel.textContent = 'white ' + v;
            syncModified();
            scheduleRender(st, false);
        });

        gammaSlider.addEventListener('input', function () {
            const T = target();
            // slider is log2(gamma) * 1000: fine, symmetric control around gamma 1.0
            T.gamma = clamp(Math.pow(2, (+this.value) / 1000), 0.2, 5);
            gammaLabel.textContent = 'gamma ' + T.gamma.toFixed(3);
            syncModified();
            scheduleRender(st, false);
        });

        gammaSlider.addEventListener('dblclick', function () {
            const T = target();
            if (T.gamma === 1) return;
            T.gamma = 1;
            syncUI();
            scheduleRender(st, false);
        });

        rotSlider.addEventListener('input', function () {
            const T = target();
            T.rotate = clamp(+this.value, -45, 45);
            rotLabel.textContent = 'rotate ' + T.rotate.toFixed(1) + '°';
            syncModified();
            scheduleRender(st, T === st);
        });

        rotSlider.addEventListener('dblclick', function () {
            const T = target();
            if (T.rotate === 0) return;
            T.rotate = 0;
            syncUI();
            scheduleRender(st, T === st);
        });

        // ---- capture-phase interception while a pick/crop mode is active

        container.addEventListener('pointerdown', function (e) {
            if (!st.mode || !st.original) return;
            if (toolbar.contains(e.target)) return;
            if (st.mode === 'crop') {
                // crop handles manage their own drags; block scribbling underneath the shades
                if (!overlay.contains(e.target) && e.button === 0) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                return;
            }
            if (e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            if (st.mode === 'pen') {
                // outside the image there is nothing to paint on; stay armed
                const r = imgEl.getBoundingClientRect();
                if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;
                // eyedropper: armed by its button, or held Alt (photoshop habit)
                if (st.penPicking || e.altKey) {
                    pickPenColor(e);
                    st.penPicking = false; // one pick, then the brush is back
                    syncUI();
                    return;
                }
                try {
                    container.setPointerCapture(e.pointerId);
                } catch (err) {}
                startPenStroke(e);
                return;
            }
            if (st.mode === 'layers') {
                const r = imgEl.getBoundingClientRect();
                if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) {
                    st.showLayers = false; // clicked outside the image: leave the tool
                    exitMode();
                    return;
                }
                const view = penView();
                if (!view) return;
                const p = view.toOriginal(e.clientX, e.clientY);
                // topmost layer under the pointer wins; empty space keeps the
                // current selection (a control-map layer is mostly transparent,
                // so the hit test uses the layer's bounding box, not pixels)
                for (let i = st.layers.length - 1; i >= 0; i--) {
                    const L = st.layers[i];
                    // an empty layer with no strokes has nothing to see or
                    // drag; its stage-sized bbox would eat every click
                    if (!L.w || (!L.src && !L.strokes.length)) continue;
                    // bbox test in the layer's LOCAL space, so it follows the
                    // layer's flip/rotation exactly
                    const q = stageToLayerLocal(L, p);
                    if (q.x >= 0 && q.x <= L.w && q.y >= 0 && q.y <= L.h) {
                        st.activeLayer = i;
                        st.layerDrag = {px: p.x, py: p.y, startX: L.x, startY: L.y};
                        try {
                            container.setPointerCapture(e.pointerId);
                        } catch (err) {}
                        syncUI(); // the adjustment controls retarget to the new active layer
                        break;
                    }
                }
                return;
            }
            if (st.mode === 'pick-gray') {
                const r = imgEl.getBoundingClientRect();
                if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) {
                    exitMode(); // clicked outside the image: cancel the selection
                    return;
                }
                st.grayPainting = true;
                st.grayStroke = null;
                try {
                    container.setPointerCapture(e.pointerId);
                } catch (err) {}
                paintGrayStroke(e, true);
                return;
            }
            handlePick(e);
        }, true);

        container.addEventListener('pointermove', function (e) {
            if (!st.mode) return;
            if (toolbar.contains(e.target)) {
                reticle.style.display = 'none';
                penCursor.style.display = 'none';
                return;
            }
            if (st.mode === 'crop') {
                if (!overlay.contains(e.target)) e.stopPropagation();
                return;
            }
            e.stopPropagation();
            if (st.mode === 'pen') {
                if (st.penPicking || e.altKey) {
                    // eyedropper armed: the reticle's open center shows the
                    // exact pixel that will be taken
                    penCursor.style.display = 'none';
                    moveReticle(e);
                    return;
                }
                reticle.style.display = 'none';
                // the pen shows its brush outline instead of the pick reticle
                movePenCursor(e);
                if (st.penStroke && (e.buttons & 1)) extendPenStroke(e);
                return;
            }
            if (st.mode === 'layers') {
                reticle.style.display = 'none';
                penCursor.style.display = 'none';
                if (st.layerDrag && (e.buttons & 1)) layerDragMove(e);
                return;
            }
            moveReticle(e);
            if (st.mode === 'pick-gray' && st.grayPainting && (e.buttons & 1)) {
                paintGrayStroke(e, false);
            }
        }, true);

        container.addEventListener('pointerup', function (e) {
            if (st.layerDrag) {
                layerDragMove(e);
                st.layerDrag = null;
                try {
                    container.releasePointerCapture(e.pointerId);
                } catch (err) {}
                e.stopPropagation();
                updateLayersOverlay();
                rebuildLayerList();
                scheduleRender(st, false); // the moved composite reaches gradio on release
                return;
            }
            if (st.penStroke) {
                extendPenStroke(e);
                commitPenStroke();
                try {
                    container.releasePointerCapture(e.pointerId);
                } catch (err) {}
                e.stopPropagation();
                return;
            }
            if (!st.grayPainting) return;
            st.grayPainting = false;
            st.grayStroke = null;
            try {
                container.releasePointerCapture(e.pointerId);
            } catch (err) {}
            e.stopPropagation();
        }, true);

        container.addEventListener('pointercancel', function () {
            st.grayPainting = false;
            st.grayStroke = null;
            if (st.penStroke) commitPenStroke(); // keep what was drawn so far
            if (st.layerDrag) {
                st.layerDrag = null;
                scheduleRender(st, false); // keep the move made so far
            }
        }, true);

        container.addEventListener('pointerleave', function () {
            reticle.style.display = 'none';
            penCursor.style.display = 'none';
        });

        syncUI();

        // ---- post-attach audit: turn a silent miss into a visible error
        //
        // Everything above wires ~90 controls by id. Any one of them can go
        // missing -- a regenerated template renames it, a reveal rule stops
        // matching it, a stylesheet hides it -- and NONE of those raise. The
        // weight-mask buttons were absent from the toolbar for exactly this
        // reason and nothing said so.
        //
        // So the last thing attach() does is check its own work against the
        // contract. Deliberately console.error rather than throw: a broken
        // toolbar is bad, but a thrown exception here would abort the rest of
        // the attach and take the whole widget with it, turning a missing
        // button into a dead canvas. Loud, attributable, non-fatal.
        try {
            if (window.cnproCanvasNodes && window.cnproCanvasNodes.audit) {
                const problems = window.cnproCanvasNodes.audit(uuid);
                if (problems.length) {
                    console.error(
                        '[cnpro] toolbar contract violated for canvas ' + uuid + ' -- ' +
                        problems.length + ' control(s) declared but not usable:\n  ' +
                        problems.join('\n  ') +
                        '\nEach control is declared by one entry in ' +
                        'javascript/canvas_tools.js and rendered by ' +
                        'javascript/canvas_nodes.js.');
                    container.dataset.cnproToolbarProblems = String(problems.length);
                } else if (container.dataset.cnproToolbarProblems) {
                    delete container.dataset.cnproToolbarProblems;
                }
            }
        } catch (e) {
            // The audit exists to report problems. It must never BE one: this is
            // the last statement before attach() succeeds, and a throw here would
            // turn a diagnostic into the outage it was added to detect.
            console.error('[cnpro] toolbar audit itself failed', e);
        }
        return true;
    }

    // ------------------------------------------------------------ class hook

    // Attaching only once from the constructor is not enough: at construction
    // time the widget's gr.HTML block may not be mounted yet (gradio fires the
    // load events while the app is still rendering), and gradio may later
    // re-render the block, replacing all nodes with fresh ones from the
    // original HTML (adjust buttons hidden again, listeners gone). So every
    // instance registers here and a MutationObserver re-runs attach() whenever
    // its container is present but not yet attached. The 'attached' flag lives
    // on the container element itself, so a re-created DOM automatically
    // invalidates it and triggers a re-attach (all listeners are bound to the
    // widget's own nodes, so nothing duplicates).
    const instances = [];

    function ensureAttached() {
        for (const inst of instances) {
            const container = document.getElementById('imageContainer_' + inst.uuid);
            if (!container || container.__forgeAdjustAttached) continue;
            try {
                if (attach(inst.fc, inst.uuid)) container.__forgeAdjustAttached = true;
            } catch (e) {
                container.__forgeAdjustAttached = true; // a hard failure would repeat on every DOM change
                console.error('[forge-canvas-adjust] failed to attach adjustment controls:', e);
            }
        }
    }

    // public push API: lets companion scripts (e.g. the ControlNet insert
    // buttons in extensions-builtin/sd_forge_controlnet/javascript/
    // insert_image.js) load an image into a specific canvas exactly as if the
    // user uploaded it (adjustments reset, gradio synced). uuid is the
    // 'uuid_<hex>' part of the widget's element ids.
    window.forgeCanvasPush = function (uuid, dataUrl) {
        for (const inst of instances) {
            if (inst.uuid === uuid) {
                inst.fc.uploadBase64(dataUrl);
                return true;
            }
        }
        return false;
    };

    // companion to forgeCanvasPush: add the image ON TOP of the existing
    // content as a new, immediately editable layer (push replaces). On an
    // empty canvas both do the same thing.
    window.forgeCanvasAddLayer = function (uuid, dataUrl) {
        for (const inst of instances) {
            if (inst.uuid === uuid) {
                const st = inst.fc.__adjust;
                return !!(st && st.addLayerFromDataUrl && st.addLayerFromDataUrl(dataUrl));
            }
        }
        return false;
    };

    // read-only snapshot of a widget's layer state - diagnosis helper for the
    // established Playwright/jsdom test flows (state itself is closure-scoped)
    window.forgeCanvasDebugLayers = function (uuid) {
        for (const inst of instances) {
            if (inst.uuid === uuid) {
                const st = inst.fc.__adjust;
                if (!st) return null;
                return {
                    stage: [st.stageW, st.stageH],
                    active: st.activeLayer,
                    mode: st.mode,
                    layerTarget: !!st.showLayers,
                    layers: st.layers.map((l) => ({
                        raster: !!l.src,
                        w: l.w, h: l.h,
                        x: Math.round(l.x), y: Math.round(l.y),
                        scale: +l.scale.toFixed(3),
                        blend: l.blend,
                        strokes: l.strokes.length,
                        erases: l.strokes.filter((s) => s.erase).length,
                        flip: l.flipH, rotate: +(+l.rotate).toFixed(2),
                        black: l.black, white: l.white, gamma: +l.gamma.toFixed(3),
                        grayscale: l.grayscale, invert: l.invert,
                        edges: edgesActive(l),
                    })),
                };
            }
        }
        return null;
    };

    let ensurePending = null;
    function scheduleEnsure() {
        if (ensurePending) return;
        ensurePending = setTimeout(function () {
            ensurePending = null;
            ensureAttached();
        }, 100);
    }

    function startObserver() {
        new MutationObserver(scheduleEnsure).observe(document.body, {childList: true, subtree: true});
    }
    // this script is loaded in <head>, so document.body may not exist yet
    if (document.body) startObserver();
    else document.addEventListener('DOMContentLoaded', startObserver);

    ForgeCanvas = class ForgeCanvas extends Orig {
        constructor(...args) {
            super(...args);
            if (!args[1]) { // args[1] == no_upload: preview-only widgets get no adjustment controls
                instances.push({fc: this, uuid: args[0]});
                ensureAttached();
            }
        }
    };

    } // end install()

    if (typeof ForgeCanvas === 'function') {
        install();
    } else {
        // core script still loading: poll until its class binding shows up
        // (well before gradio's load events construct the canvas instances)
        const waiter = setInterval(function () {
            if (typeof ForgeCanvas !== 'function') return;
            clearInterval(waiter);
            install();
        }, 50);
    }
})();
