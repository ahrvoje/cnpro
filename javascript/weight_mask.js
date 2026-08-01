// ControlNet weight masks as ForgeCanvas toolbar tools: paint hue-coded
// control strength directly over the ControlNet input image (same canvas used
// for the main ControlNet input). Weight v in [0, 1] maps to hue
// (1 - v) * 270 (red = 1 down to violet = 0), a 270 degree span so the range
// ends far from its start and painted values stay visually unambiguous.
//
// Four mask slots share one brush: the global mask (G) and three per-band
// layer masks (C coarse = composition layers, M mid = form layers, F fine =
// texture layers). Each slot paints into its own offscreen canvas and travels
// to python as a PNG data url through its own hidden LogicalImage textbox.
//
// Every canvas of a unit gets the painter (see slotDefsFor): each Input tab
// with its own four channels - inputs are independent controls, so masking one
// restricts only what that one contributes - and the Output mask tab with a
// single slot. The difference is registration, not encoding: masks on an input
// canvas line up with THAT CONTROL INPUT (and therefore also gate what the
// control model gets to see of it), while the output mask lines up with the
// GENERATED image and only restricts where the control injection lands.
//
// WHICH SLOTS ARE LIVE IS THE PROFILE EDITOR'S BAND SELECTOR. The four slots
// are the four weight profiles' spatial half: G belongs to the MAIN profile,
// C/M/F to the coarse/mid/fine band profiles. So main mode - and depth, which
// multiplies main, and drift, which moves where depth is read - runs on G and
// leaves C/M/F dormant, and a band selection runs on C/M/F and leaves G
// dormant. One switch, not two - see
// external_code.masks_in_force for why the masks having their own precedence
// was a bug rather than a feature.
//
// Restrict-to-painted semantics (python side, scripts/cnpro.py) then applies
// WITHIN the live set: with nothing painted there the full image controls as
// usual; a painted global mask governs all layers; among the bands, any
// painted band weights its own layers and every absent band counts as ZERO -
// control never escapes the painted regions.
//
// The mask itself is stored at full opacity in image resolution; the rainbow
// hue is DISPLAY-ONLY - on the wire the mask travels as grayscale (pixel
// value = weight, alpha marks painted pixels; see exportMask / importState),
// which compresses better and lets the python side (decode_weight_mask) read
// the value channel directly, with a hue fallback for legacy chromatic masks.
// The on-image overlays are displayed at 25% opacity (75% transparent) so the
// underlying image stays clearly visible under the hue tint. An overlay is
// shown only while its own tool is active; while inactive, a non-empty mask
// is marked by the red border on its tool button. The tool menu holds the
// eraser (e, injected here), invert (i) and clear (reset) buttons for the
// active slot's mask, plus the feather slider (injected here) that softens
// the exported mask edges with a Gaussian falloff.
//
// Toolbar wiring mirrors modules_forge/forge_canvas/canvas_extra.js: the tool
// buttons live in the canvas template (canvas.html, wmask* ids) and are only
// revealed on ControlNet input canvases; while a tool is active, capture
// phase listeners on the container intercept pointer events so the core
// scribble/pan handlers underneath stay quiet.
(function () {
    'use strict';

    const HUE_SPAN = 270;
    const OVERLAY_OPACITY = 0.25; // painted mask shows 75% transparent

    const BAND_BUTTONS = {
        global: 'wmaskButton_',
        coarse: 'wmaskCoarseButton_',
        mid: 'wmaskMidButton_',
        fine: 'wmaskFineButton_',
    };

    // ------------------------------------------- masks follow the profile
    //
    // THE FOUR MASK SLOTS ARE THE FOUR PROFILES' SPATIAL HALF. G belongs to the
    // MAIN weight profile and C/M/F to the coarse/mid/fine band profiles, so
    // the editor's band selector decides which slots a generation uses: main
    // runs on G, a band selection runs on C/M/F. Depth is not a band - it
    // MULTIPLIES the main profile rather than replacing it - so it runs on G
    // too.
    //
    // This mirrors external_code.masks_in_force on the python side, which is
    // what actually applies them; here it only decides what the toolbar SHOWS.
    // Both read the same selector (this from the unit root, python from the
    // '#B<band>' marker in the profile string weight_profile.js writes), so
    // there is one decision with two readers rather than two decisions.
    const PROFILE_BANDS = ['coarse', 'mid', 'fine'];

    function liveSlotKeys(band) {
        return PROFILE_BANDS.indexOf(band) === -1 ? ['global'] : PROFILE_BANDS.slice();
    }

    /** the pressed profile selector of the unit this canvas belongs to. */
    function selectedProfileBand(unit) {
        return (unit && unit.dataset && unit.dataset.cnetProfileBand) || 'main';
    }

    // The painter serves every canvas of a unit: each Input tab (four slots,
    // input-registered) and the Output mask tab (one slot, output-registered -
    // the backdrop image there is only something to draw on and is never read).
    // Each input canvas owns its OWN four hidden channels, keyed by the slot
    // index its group carries (cnet-input-slot-<n> -> cnet-wmask-<n>-<band>),
    // because every input is a separate control: masking input 3 restricts what
    // input 3 contributes and leaves the others alone.
    function slotDefsFor(container) {
        const group = container.closest('.cnet-output-mask-group');
        if (group) {
            return [{ key: 'global', button: BAND_BUTTONS.global, state: '.cnet-output-mask-state' }];
        }
        const input = container.closest('.cnet-input-image-group');
        if (!input) return null;
        const match = /(?:^|\s)cnet-input-slot-(\d+)(?:\s|$)/.exec(input.className);
        if (!match) return null;
        const slot = match[1];
        return Object.keys(BAND_BUTTONS).map((key) => ({
            key: key,
            button: BAND_BUTTONS[key],
            state: `.cnet-wmask-${slot}-${key}-state`,
        }));
    }

    function weightToRgb(v) {
        const h = (1 - Math.min(Math.max(v, 0), 1)) * HUE_SPAN;
        const f = (n) => {
            const k = (n + h / 30) % 12;
            const c = 0.5 - 0.5 * Math.max(-1, Math.min(k - 3, 9 - k, 1));
            return Math.round(255 * c);
        };
        return [f(0), f(8), f(4)];
    }

    function weightToColor(v) {
        return "#" + weightToRgb(v).map((c) => c.toString(16).padStart(2, "0")).join("");
    }

    // Inverse of weightToRgb, mirroring the python decode (hue only, via the
    // same 270 degree span - scripts/controlnet.py decode_weight_mask). Fully
    // desaturated pixels have hue 0 there too, so they read back as weight 1.
    function colorToWeight(r, g, b) {
        const max = Math.max(r, g, b);
        const d = max - Math.min(r, g, b);
        let h = 0;
        if (d !== 0) {
            if (max === r) h = ((g - b) / d) % 6;
            else if (max === g) h = (b - r) / d + 2;
            else h = (r - g) / d + 4;
            h *= 60;
            if (h < 0) h += 360;
        }
        return Math.min(Math.max(1 - h / HUE_SPAN, 0), 1);
    }

    function notifyGradio(textarea) {
        if (typeof updateInput === "function") {
            updateInput(textarea);
        } else {
            textarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
    }

    function setup(container) {
        // cleared per attempt, so the diagnostic below can never report a
        // previous container's missing nodes as this one's
        setup.lastMissing = null;
        const slotDefs = slotDefsFor(container);
        if (!slotDefs) return false;
        // The OUTPUT mask is orthogonal to the profiles (MAINTENANCE invariant
        // 6): it multiplies the finished result and belongs to no band, so it
        // is always live and must never be dimmed by the band selector.
        const isOutputMask = !!container.closest('.cnet-output-mask-group');

        const uuid = container.id.replace('imageContainer_', '');
        const el = (id) => document.getElementById(id + uuid);

        const unit = container.closest('.input-accordion');
        const imgEl = el('image_');
        const toolbar = el('toolbar_');
        const box = el('wmaskBox_');
        const weightSlider = el('wmaskWeight_');
        const weightLabel = el('wmaskWeightLabel_');
        const brushSlider = el('wmaskBrush_');
        const brushLabel = el('wmaskBrushLabel_');
        const invertButton = el('wmaskInvertButton_');
        const clearButton = el('wmaskClearButton_');
        // Name what is missing. "setup returned false" and "this canvas has no
        // weight-mask channels" are different problems that look identical from
        // outside: the buttons are visible either way (canvas_nodes.js reveals
        // them) and simply do nothing when clicked.
        const REQUIRED = {
            'image': imgEl, 'toolbar': toolbar, 'wmaskBox': box,
            'wmaskWeight': weightSlider, 'wmaskBrush': brushSlider,
            'wmaskClearButton': clearButton,
        };
        const missing = Object.keys(REQUIRED).filter((k) => !REQUIRED[k]);
        if (missing.length) {
            setup.lastMissing = missing;
            return false;
        }

        const slots = [];
        for (const slotDef of slotDefs) {
            const button = el(slotDef.button);
            const textarea = unit && unit.querySelector(slotDef.state + ' textarea');
            if (!button || !textarea) continue;
            slots.push({
                key: slotDef.key,
                button: button,
                textarea: textarea,
                overlay: null,
                mask: null,        // offscreen canvas, image natural resolution, full-alpha strokes
                hasPaint: false,
                last: null,        // previous stroke point in mask pixels
                stroke: 0,         // bumped on every paint, part of the overlay redraw key
                overlayKey: '',
                dims: '',          // imgEl natural dims the mask was painted for
                preInvert: null,   // pre-invert snapshot, so invert toggles exactly
                bounds: null,      // painted bbox in mask px (upper bound, see growBounds)
                state: 'idle',     // idle | painting | pending (see setSlotState)
                _syncTimer: 0,     // debounced export handle; 0 = nothing owed
            });
        }
        if (!slots.length || slots[0].key !== 'global') {
            setup.lastMissing = ['the global weight-mask button and/or its state textarea'];
            return false;
        }
        setup.lastMissing = null;

        const st = {
            active: null,      // slot key of the armed tool, or null
            weight: 1,
            brush: 50,   // relative: 100 = brush diameter of 25% of the image diagonal
            eraser: false,     // strokes remove paint instead of painting
            feather: 0,  // relative: 100 = blur radius of 2% of the image diagonal
            painting: false,
            raf: 0,
        };

        for (const slot of slots) {
            slot.button.style.display = '';
            const overlay = document.createElement('canvas');
            overlay.className = 'forge-wmask-overlay';
            overlay.width = 1;
            overlay.height = 1;
            overlay.style.cssText =
                'position:absolute;top:0;left:0;pointer-events:none;display:none;opacity:' + OVERLAY_OPACITY + ';';
            container.insertBefore(overlay, toolbar);
            slot.overlay = overlay;
        }

        // Brush circle following the cursor: shows the true footprint of the
        // image-relative brush at the current zoom, border in the current
        // brush hue with a dark contrast ring so it stays visible on any image.
        const indicator = document.createElement('div');
        indicator.className = 'forge-wmask-indicator';
        indicator.style.cssText =
            'position:absolute;pointer-events:none;display:none;border-radius:50%;' +
            'border:2px solid #ff0000;' +
            'box-shadow:0 0 0 1px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(0,0,0,0.6);';
        container.insertBefore(indicator, toolbar);

        function activeSlot() {
            for (const slot of slots) {
                if (slot.key === st.active) return slot;
            }
            return null;
        }

        function updateIndicator(e) {
            if (!st.active || toolbar.contains(e.target)) {
                indicator.style.display = 'none';
                return;
            }
            const c = container.getBoundingClientRect();
            const r = imgEl.getBoundingClientRect();
            if (!imgEl.naturalWidth || !r.width) {
                indicator.style.display = 'none';
                return;
            }
            const d = brushRadiusImagePx() * 2 * (r.width / imgEl.naturalWidth);
            indicator.style.left = (e.clientX - c.left - d / 2) + 'px';
            indicator.style.top = (e.clientY - c.top - d / 2) + 'px';
            indicator.style.width = d + 'px';
            indicator.style.height = d + 'px';
            // eraser mode reads as "no color": dashed neutral ring
            indicator.style.borderColor = st.eraser ? '#bbbbbb' : weightToColor(st.weight);
            indicator.style.borderStyle = st.eraser ? 'dashed' : 'solid';
            indicator.style.display = 'block';
        }

        function setActive(key, opts) {
            st.active = key;
            for (const slot of slots) {
                slot.button.classList.toggle('forge-btn-active', slot.key === key);
            }
            box.style.display = key ? '' : 'none';
            if (key) {
                // single-active-tool rule: claim the canvas FIRST so the
                // previous owner's stand-down (which may clear forge-picking)
                // runs before our own class write - the reverse order let the
                // outgoing tool strip the class we had just set
                container.dispatchEvent(new CustomEvent('forge-canvas-tool', {detail: {owner: 'wmask'}}));
                container.classList.add('forge-picking');
                // fresh arm = brush, not a days-old sticky eraser: the mode
                // is invisible while the box is closed, so it must not
                // survive disarm/rearm
                if (st.eraser) {
                    st.eraser = false;
                    const eraser = el('wmaskEraserButton_');
                    if (eraser) eraser.classList.remove('forge-btn-active');
                }
            } else if (!opts || !opts.keepPicking) {
                // keepPicking: a FOREIGN tool claimed the canvas and owns the
                // class now - removing it here would clobber the new owner
                container.classList.remove('forge-picking');
            }
            if (!key) indicator.style.display = 'none';
            ensureLoop();
        }

        function currentDims() {
            return imgEl.naturalWidth + 'x' + imgEl.naturalHeight;
        }

        // ---- slot state machine
        //
        // Coherence between painter and server (MAINTENANCE invariant 15) used
        // to be spelled as a conjunction of ad-hoc flags - `st.painting`,
        // `slot._syncTimer` doubling as "an export is owed", `hasPaint`
        // doubling as "should export". One missing term in that conjunction
        // silently ate every finished stroke on every canvas. The rule is now
        // one explicit state per slot:
        //   idle     - what the server holds is what the painter holds
        //   painting - a stroke is in progress
        //   pending  - the stroke is finished, its export has not run yet
        // and ONLY 'idle' allows the server-cleared watchdog to drop paint.
        function setSlotState(slot, state) {
            slot.state = state;
        }

        function clearMask(slot, pushToGradio) {
            slot.mask = null;
            slot.hasPaint = false;
            slot.last = null;
            slot.preInvert = null;
            slot.bounds = null;
            slot._wireImg = null; // a cleared import must not be re-fitted
            slot.stroke++;
            if (slot._syncTimer) {
                clearTimeout(slot._syncTimer);
                slot._syncTimer = 0;
            }
            setSlotState(slot, 'idle');
            if (pushToGradio && slot.textarea.value) {
                slot.textarea.value = '';
                notifyGradio(slot.textarea);
            }
            updateOverlay(slot);
        }

        function ensureMask(slot) {
            const w = imgEl.naturalWidth;
            const h = imgEl.naturalHeight;
            if (!w || !h) return null;
            const dims = w + 'x' + h;
            if (!slot.mask || slot.dims !== dims) {
                const m = document.createElement('canvas');
                m.width = w;
                m.height = h;
                slot.mask = m;
                slot.dims = dims;
                slot.hasPaint = false;
                slot.last = null;
                slot.bounds = null;
            }
            return slot.mask;
        }

        // Painted bounding box in mask pixels, grown by every stroke. Only an
        // upper bound (erasing never shrinks it), which is all the full-alpha
        // rescan at eraser stroke end needs to stay off the whole image.
        function growBounds(slot, x, y, radius) {
            const b = slot.bounds;
            const x0 = x - radius, y0 = y - radius, x1 = x + radius, y1 = y + radius;
            if (!b) {
                slot.bounds = { x0: x0, y0: y0, x1: x1, y1: y1 };
                return;
            }
            if (x0 < b.x0) b.x0 = x0;
            if (y0 < b.y0) b.y0 = y0;
            if (x1 > b.x1) b.x1 = x1;
            if (y1 > b.y1) b.y1 = y1;
        }

        // brush radius in image pixels: slider value 100 = a brush diameter of
        // 25% of the image diagonal, so strokes scale with the target image
        // instead of the screen
        function brushRadiusImagePx() {
            const diag = Math.hypot(imgEl.naturalWidth, imgEl.naturalHeight);
            return Math.max(1, (st.brush / 100) * 0.25 * diag / 2);
        }

        // screen position -> mask pixel position + brush radius in mask pixels
        function maskPos(e) {
            const r = imgEl.getBoundingClientRect();
            if (!r.width || !r.height || !imgEl.naturalWidth) return null;
            return {
                x: (e.clientX - r.left) * (imgEl.naturalWidth / r.width),
                y: (e.clientY - r.top) * (imgEl.naturalHeight / r.height),
                radius: brushRadiusImagePx(),
            };
        }

        function paintStroke(e, isStart) {
            const slot = activeSlot();
            if (!slot) return;
            const mask = ensureMask(slot);
            const p = mask && maskPos(e);
            if (!p) return;
            const ctx = mask.getContext('2d');
            ctx.save();
            if (st.eraser) {
                // remove paint back to UNPAINTED (alpha 0) - not the same as
                // painting weight 0, which is an explicit zero to the
                // restrict-to-painted rule
                ctx.globalCompositeOperation = 'destination-out';
                ctx.fillStyle = ctx.strokeStyle = '#000000';
            } else {
                ctx.fillStyle = ctx.strokeStyle = weightToColor(st.weight);
            }
            ctx.lineCap = ctx.lineJoin = 'round';
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fill();
            if (!isStart && slot.last) {
                ctx.lineWidth = p.radius * 2;
                ctx.beginPath();
                ctx.moveTo(slot.last.x, slot.last.y);
                ctx.lineTo(p.x, p.y);
                ctx.stroke();
            }
            ctx.restore();
            if (slot.last) growBounds(slot, slot.last.x, slot.last.y, p.radius);
            growBounds(slot, p.x, p.y, p.radius);
            slot.last = { x: p.x, y: p.y };
            // an eraser stroke may just as well have EMPTIED the mask; the
            // real state is recomputed at stroke end (maskHasPaint)
            if (!st.eraser) slot.hasPaint = true;
            slot.preInvert = null; // painting ends the invert toggle pair
            slot.stroke++;
            setSlotState(slot, 'painting');
        }

        // Alpha scan, restricted to the region strokes have actually touched:
        // only run at eraser stroke ends, where "did the stroke empty the
        // mask" cannot be known any cheaper. Scanning the full image cost a
        // getImageData of the whole (possibly 4K) mask on every eraser stroke.
        function maskHasPaint(slot) {
            if (!slot.mask) return false;
            const w = slot.mask.width;
            const h = slot.mask.height;
            const b = slot.bounds;
            const x0 = Math.max(0, Math.floor(b ? b.x0 : 0));
            const y0 = Math.max(0, Math.floor(b ? b.y0 : 0));
            const x1 = Math.min(w, Math.ceil(b ? b.x1 : w));
            const y1 = Math.min(h, Math.ceil(b ? b.y1 : h));
            if (x1 <= x0 || y1 <= y0) return false;
            const px = slot.mask.getContext('2d')
                .getImageData(x0, y0, x1 - x0, y1 - y0).data;
            for (let i = 3; i < px.length; i += 4) {
                if (px[i] >= 128) return true;
            }
            return false;
        }

        // Invert the active mask. Unpainted pixels count as weight 0 both here
        // and on the python side, so inverting paints the WHOLE image: painted
        // pixels become 1-v, unpainted ones become weight 1. The pre-invert
        // mask is kept so pressing the button again restores it exactly -
        // needed because "nothing painted" and "everything painted with 0" are
        // different things to the restrict-to-painted rule, even though both
        // decode to weight 0.
        function invertMask(slot) {
            // this exports immediately; a debounced export still owed would
            // otherwise fire afterwards and re-upload the pre-invert canvas
            if (slot._syncTimer) {
                clearTimeout(slot._syncTimer);
                slot._syncTimer = 0;
            }
            if (slot.preInvert) {
                const snapshot = slot.preInvert;
                slot.mask = snapshot.mask;
                slot.dims = snapshot.dims;
                slot.hasPaint = snapshot.hasPaint;
                slot.preInvert = null;
                slot.last = null;
                slot.stroke++;
                syncState(slot);
                return;
            }
            const w = imgEl.naturalWidth;
            const h = imgEl.naturalHeight;
            if (!w || !h) return;
            slot.preInvert = { mask: slot.mask, dims: slot.dims, hasPaint: slot.hasPaint };

            const inverted = document.createElement('canvas');
            inverted.width = w;
            inverted.height = h;
            const ctx = inverted.getContext('2d');
            if (slot.hasPaint && slot.mask) {
                ctx.drawImage(slot.mask, 0, 0, slot.mask.width, slot.mask.height, 0, 0, w, h);
                const data = ctx.getImageData(0, 0, w, h);
                const px = data.data;
                for (let i = 0; i < px.length; i += 4) {
                    const v = px[i + 3] >= 128 ? colorToWeight(px[i], px[i + 1], px[i + 2]) : 0;
                    const c = weightToRgb(1 - v);
                    px[i] = c[0];
                    px[i + 1] = c[1];
                    px[i + 2] = c[2];
                    px[i + 3] = 255;
                }
                ctx.putImageData(data, 0, 0);
            } else {
                ctx.fillStyle = weightToColor(1);
                ctx.fillRect(0, 0, w, h);
            }
            slot.mask = inverted;
            slot.dims = w + 'x' + h;
            slot.hasPaint = true;
            slot.last = null;
            slot.stroke++;
            syncState(slot);
        }

        // blur radius in image pixels: slider value 100 = 2% of the diagonal
        function featherRadiusImagePx(mask) {
            return (st.feather / 100) * 0.02 * Math.hypot(mask.width, mask.height);
        }

        // Wire format: grayscale (pixel value = weight, alpha marks painted
        // pixels). The rainbow the painter works in is display-only; grays
        // compress far better and the python side reads the value channel
        // directly (decode_weight_mask keeps a hue fallback for legacy
        // chromatic masks). Feather, when set, softens the exported edges
        // with a Gaussian falloff - correct only in this value space, blurring
        // the rainbow would mix hues into unrelated weights.
        function exportMask(slot, scale) {
            const out = document.createElement('canvas');
            out.width = slot.mask.width;
            out.height = slot.mask.height;
            const ctx = out.getContext('2d');
            ctx.drawImage(slot.mask, 0, 0);
            const data = ctx.getImageData(0, 0, out.width, out.height);
            const px = data.data;
            for (let i = 0; i < px.length; i += 4) {
                if (px[i + 3] === 0) continue;
                const v = Math.round(colorToWeight(px[i], px[i + 1], px[i + 2]) * 255);
                px[i] = px[i + 1] = px[i + 2] = v;
            }
            ctx.putImageData(data, 0, 0);
            // oversize fallback: downscale AFTER the gray conversion (gray
            // interpolates as valid weights; scaling the hue canvas would
            // blend hues into unrelated weights)
            let result = out;
            if (scale && scale < 1) {
                const scaled = document.createElement('canvas');
                scaled.width = Math.max(1, Math.round(out.width * scale));
                scaled.height = Math.max(1, Math.round(out.height * scale));
                scaled.getContext('2d').drawImage(
                    out, 0, 0, out.width, out.height, 0, 0, scaled.width, scaled.height);
                result = scaled;
            }
            const blurPx = featherRadiusImagePx(result);
            if (blurPx >= 0.5) {
                const blurred = document.createElement('canvas');
                blurred.width = result.width;
                blurred.height = result.height;
                const bctx = blurred.getContext('2d');
                bctx.filter = 'blur(' + blurPx.toFixed(1) + 'px)';
                bctx.drawImage(result, 0, 0);
                return blurred;
            }
            return result;
        }

        function syncState(slot) {
            let value = '';
            if (slot.hasPaint && slot.mask) {
                // gradio transports this through its queue; oversized payloads
                // can exceed server limits and silently never arrive, which
                // would look like a mask that does nothing. Warning alone did
                // not prevent that - downscale until it fits (the python side
                // resizes the mask to generation dims anyway, so a capped
                // export resolution costs nothing visible).
                const LIMIT = 8 * 1024 * 1024;
                let scale = 1;
                value = exportMask(slot, scale).toDataURL('image/png');
                while (value.length > LIMIT && scale > 0.3) {
                    scale *= 0.7;
                    value = exportMask(slot, scale).toDataURL('image/png');
                }
                if (value.length > LIMIT) {
                    console.warn('[controlnet wmask] mask payload is still '
                        + (value.length / 1024 / 1024).toFixed(1)
                        + ' MB after downscaling; if the mask has no effect, the upload may have hit a server size limit.');
                } else if (scale < 1) {
                    console.info('[controlnet wmask] mask export downscaled to '
                        + Math.round(scale * 100) + '% to stay under the transport size limit.');
                }
            }
            slot.textarea.value = value;
            notifyGradio(slot.textarea);
            setSlotState(slot, 'idle');
        }

        // Stroke-end export (full-res readback + gray conversion + PNG encode
        // + base64 upload) is by far the heaviest thing painting does, and it
        // ran once per STROKE - a rapid series of dabs paid it per dab. It is
        // debounced instead, and the slot stays in the 'pending' state for the
        // whole wait so the server-cleared watchdog cannot mistake the gap for
        // a real clear.
        //
        // The wait is bounded by flushPendingSyncs(): gradio serializes its
        // queue in click order, so as long as the export is STARTED before the
        // Generate click is queued, the unit State is written first. Anything
        // that ends painting - disarming the tool, switching canvas, clicking
        // Generate - flushes.
        const SYNC_DEBOUNCE_MS = 250;

        function scheduleSync(slot) {
            if (slot._syncTimer) clearTimeout(slot._syncTimer);
            setSlotState(slot, 'pending');
            slot._syncTimer = setTimeout(function () {
                slot._syncTimer = 0;
                syncState(slot);
            }, SYNC_DEBOUNCE_MS);
        }

        function flushSlot(slot) {
            if (!slot._syncTimer) return;
            clearTimeout(slot._syncTimer);
            slot._syncTimer = 0;
            syncState(slot);
        }

        function flushSlots() {
            for (const slot of slots) flushSlot(slot);
        }

        registerWhileConnected(painterFlushes, container, flushSlots);

        // Session restore / gradio push: show a mask that arrived from python.
        //
        // The wire mask can be SMALLER than the image (syncState downscales
        // oversized exports to fit the transport limit), so the import FITS
        // the gray data to the image's own dimensions before recoloring -
        // gray interpolates as valid weights, the same rationale the
        // downscale gives on the way out. Stamping slot.dims from the wire
        // image instead made the 500 ms watchdog read every restored
        // downscaled mask as "image replaced with different dimensions"
        // within half a second of it appearing, and clearMask(slot, true)
        // then destroyed the painter copy AND pushed '' to gradio - wiping
        // the server value the restore had just delivered. THE INVARIANT:
        // an import can never write to the server.
        function importState(slot) {
            const value = slot.textarea.value;
            if (!value || !value.startsWith('data:image') || slot.hasPaint) return;
            // remembered so the late-import watch never retries a value that
            // failed to decode (and never re-imports one it already took)
            slot._lastImportTried = value;
            const img = new Image();
            img.onload = () => {
                // superseded while decoding (a newer server push landed, or
                // a stroke started): let the watch import the NEWER value
                // instead - last VALUE wins, not last decode. Same guard as
                // decodeLayerImg in canvas_extra.js.
                if (slot.textarea.value !== value || slot.hasPaint) return;
                slot._wireImg = img;
                installImportedMask(slot);
                ensureLoop();
            };
            img.src = value;
        }

        // Build the painter mask from the imported wire image: at the INPUT
        // image's dimensions when it is loaded (the gray is scaled BEFORE
        // the recolor - hues do not interpolate, gray does), at wire
        // dimensions when it is not (the watchdog re-fits once the image
        // arrives, and without an image the mask is inert anyway). Once
        // fitted the wire copy is dropped, so a LATER dims change is a real
        // image replacement and clears normally.
        function installImportedMask(slot) {
            const img = slot._wireImg;
            if (!img) return;
            const fit = imgEl.naturalWidth > 0;
            const w = fit ? imgEl.naturalWidth : img.width;
            const h = fit ? imgEl.naturalHeight : img.height;
            const m = document.createElement('canvas');
            m.width = w;
            m.height = h;
            const ctx = m.getContext('2d');
            ctx.drawImage(img, 0, 0, w, h);
            // the wire format is grayscale (value = weight) but the
            // painter works in display hues: recolor painted pixels on
            // the way in. Legacy rainbow masks (old sessions) are
            // chromatic and pass through unchanged.
            try {
                const data = ctx.getImageData(0, 0, w, h);
                const px = data.data;
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
                if (!chromatic) {
                    for (let i = 0; i < px.length; i += 4) {
                        if (px[i + 3] === 0) continue;
                        const c = weightToRgb(px[i] / 255);
                        px[i] = c[0];
                        px[i + 1] = c[1];
                        px[i + 2] = c[2];
                    }
                    ctx.putImageData(data, 0, 0);
                }
            } catch (err) {}
            slot.mask = m;
            slot.dims = fit ? currentDims() : (img.width + 'x' + img.height);
            slot.hasPaint = true;
            // an imported mask can hold paint anywhere: no stroke history
            // to bound it, so the eraser rescan starts from the full image
            slot.bounds = { x0: 0, y0: 0, x1: w, y1: h };
            slot.stroke++;
            // what the server holds IS what the painter now holds
            setSlotState(slot, 'idle');
            if (fit) slot._wireImg = null;
        }

        // The overlays follow the displayed image through pan/zoom/resize; a
        // cheap rAF loop runs only while there is something to track.
        function updateOverlay(slot) {
            const overlay = slot.overlay;
            // display writes are same-value-guarded: this runs every rAF tick
            // and unconditional style writes feed onUiUpdate mutation loops.
            // The overlay is visible only while its own tool is active: with
            // the tool off, the red border on the button is the sole marker
            // of a non-empty mask and the image stays unobstructed.
            if (!slot.hasPaint || !slot.mask || st.active !== slot.key || !imgEl.naturalWidth) {
                if (overlay.style.display !== 'none') overlay.style.display = 'none';
                slot.overlayKey = '';
                return;
            }
            const r = imgEl.getBoundingClientRect();
            const c = container.getBoundingClientRect();
            if (!r.width || !r.height) {
                if (overlay.style.display !== 'none') overlay.style.display = 'none';
                slot.overlayKey = '';
                return;
            }
            const key = [r.left - c.left, r.top - c.top, r.width, r.height, slot.stroke].join('|');
            if (key === slot.overlayKey) return;
            slot.overlayKey = key;
            overlay.style.left = (r.left - c.left) + 'px';
            overlay.style.top = (r.top - c.top) + 'px';
            overlay.style.width = r.width + 'px';
            overlay.style.height = r.height + 'px';
            const w = Math.max(1, Math.round(r.width));
            const h = Math.max(1, Math.round(r.height));
            if (overlay.width !== w || overlay.height !== h) {
                overlay.width = w;
                overlay.height = h;
            }
            const ctx = overlay.getContext('2d');
            ctx.clearRect(0, 0, w, h);
            ctx.drawImage(slot.mask, 0, 0, slot.mask.width, slot.mask.height, 0, 0, w, h);
            overlay.style.display = 'block';
        }

        // Coherence checks, formerly run from the rAF loop. They watch for
        // things that happen at human speed (an image replaced, a channel
        // cleared server-side), so the shared 500 ms tick is the right clock -
        // and it lets the rAF loop stop entirely when no tool is armed.
        function watchSlots() {
            for (const slot of slots) {
                // image replaced with different dimensions: the paint no
                // longer matches what the backend will receive. Only checked
                // against a LOADED image - while naturalWidth is 0 the mask
                // is kept, because a mask restored via importState may
                // arrive before the input image finishes loading (clearing
                // here would wipe the restored state), and without an image
                // the mask is inert anyway.
                if (slot.hasPaint && imgEl.naturalWidth && slot.dims !== currentDims()) {
                    if (slot._wireImg) {
                        // not a replacement: the import installed at wire
                        // dimensions before the image finished loading. Fit
                        // it to the image now (drops _wireImg), instead of
                        // destroying the restored mask AND the server value
                        // behind it.
                        installImportedMask(slot);
                    } else {
                        clearMask(slot, true);
                    }
                }
                // channel cleared server-side (close tab, unit reset): drop
                // the local paint too, otherwise the next stroke-end would
                // push the stale mask right back.
                //
                // "paint but empty channel" is ALSO the normal state while an
                // export is owed, which is why this asks the slot's STATE
                // rather than trying to enumerate the ways a stroke can be in
                // flight: while that distinction lived in a conjunction of
                // flags, one missing term made this rule eat every stroke that
                // had just finished (the painter wrote the mask, the watchdog
                // cleared it, the export then wrote hasPaint=false).
                if (slot.state === 'idle' && slot.hasPaint && !slot.textarea.value) {
                    clearMask(slot, false);
                }
                // red border marks buttons whose mask holds paint
                // (classList.toggle with force is a no-op when unchanged)
                slot.button.classList.toggle('forge-tool-modified', slot.hasPaint);
            }
            reflectProfileBand();
        }

        // Say in the TOOLTIP which slots the selected profile does not use.
        // Without this the coupling is invisible: a painted C mask keeps its
        // red "has paint" border while main mode is selected, the overlay
        // still draws the strokes, and the generation ignores all of it. The
        // python side logs the same fact at generation time
        // (report_masks_not_in_force); this is the half that arrives BEFORE
        // the user spends a minute painting.
        //
        // TOOLTIP ONLY - the button itself is left fully enabled, and looks
        // it. The slots used to be dimmed to opacity 0.35 as well, which reads
        // as disabled and turned a hint into an obstacle: it pushed the
        // workflow into "activate the band profiles first" before a C/M/F mask
        // could be started at all. Painting a band mask while main is selected
        // is legitimate preparation - the mask is kept, and pressing the
        // matching profile selector puts it in force. All four slots are
        // armable and paintable at all times; which ones RUN is still the
        // selector's decision alone (liveSlotKeys / masks_in_force), unchanged.
        let lastBand = null;
        function reflectProfileBand() {
            const band = isOutputMask ? null : selectedProfileBand(unit);
            if (band === lastBand) return;   // per 500ms tick; usually a no-op
            lastBand = band;
            const live = band === null ? null : liveSlotKeys(band);
            for (const slot of slots) {
                const isLive = live === null || live.indexOf(slot.key) !== -1;
                if (slot.baseTitle === undefined) slot.baseTitle = slot.button.title;
                slot.button.title = isLive ? slot.baseTitle
                    : slot.baseTitle + '\n\nNOT IN USE: the ' +
                      // the same test liveSlotKeys makes, not a second list of
                      // the non-band selectors: every selector that is not
                      // coarse/mid/fine runs MAIN (depth and drift shape the
                      // main profile rather than replacing it), and spelling
                      // them out here is how 'drift' would have kept naming
                      // itself in a sentence that says G is what runs
                      (PROFILE_BANDS.indexOf(band) === -1 ? 'main' : band) +
                      ' weight profile is selected, and the mask slots follow ' +
                      'that selector (' +
                      (live.indexOf('global') !== -1 ? 'G' : 'C/M/F') +
                      ' is what runs). Paint here anyway if you like - it is ' +
                      'kept, and selecting the matching profile puts it in force.';
            }
        }

        registerWhileConnected(painterWatches, container, watchSlots);
        // ...and once now, so the tooltips describe the right selector on the
        // first frame the toolbar is visible rather than up to 500 ms later.
        // The tick keeps it in step afterwards; the editor and the painter
        // attach in either order, so neither can push to the other.
        reflectProfileBand();

        // The rAF loop exists ONLY to keep the overlay glued to the displayed
        // image while a tool is armed (pan / zoom / resize move it with no
        // event to hook). It used to keep spinning for the page lifetime as
        // soon as any mask held paint, on every one of the widget's canvases.
        function ensureLoop() {
            if (st.raf || !st.active) return;
            const tick = () => {
                st.raf = 0;
                for (const slot of slots) updateOverlay(slot);
                if (st.active) st.raf = requestAnimationFrame(tick);
            };
            st.raf = requestAnimationFrame(tick);
        }

        // ---- toolbar controls

        for (const slot of slots) {
            slot.button.addEventListener('click', function () {
                if (st.active === slot.key) {
                    setActive(null);
                    return;
                }
                if (!imgEl.naturalWidth) {
                    // a silent no-op read as "broken button": flash the deny
                    // state to say "load an image first"
                    slot.button.classList.remove('cnet-flash-deny');
                    void slot.button.offsetWidth; // restart the animation
                    slot.button.classList.add('cnet-flash-deny');
                    return;
                }
                setActive(slot.key);
            });
        }

        // Eraser toggle and feather slider are injected here rather than added
        // to canvas.html: they are painter concerns, and runtime injection into
        // the toolbar is the pattern an addon build needs anyway (the template
        // is core-side). The eraser removes paint back to UNPAINTED (alpha 0),
        // which the weight-0 brush cannot do - painted 0 is an explicit zero
        // under the restrict-to-painted rule, unpainted is not.
        const eraserButton = document.createElement('button');
        eraserButton.id = 'wmaskEraserButton_' + uuid;
        eraserButton.className = 'forge-btn forge-no-select forge-wmask-eraser';
        eraserButton.title = 'Eraser: strokes remove paint (back to unpainted) instead of painting; click to toggle';
        eraserButton.textContent = 'e';
        box.insertBefore(eraserButton, invertButton || clearButton);
        eraserButton.addEventListener('click', function () {
            st.eraser = !st.eraser;
            eraserButton.classList.toggle('forge-btn-active', st.eraser);
        });

        const featherRow = document.createElement('div');
        featherRow.className = 'forge-range-row';
        const featherLabel = document.createElement('div');
        featherLabel.id = 'wmaskFeatherLabel_' + uuid;
        featherLabel.className = 'forge-toolbar-label';
        featherLabel.textContent = 'feather 0';
        // The widest text this label can hold, in the same form the registry
        // declares for the rows it renders (canvas_tools.js `labelMax`). The
        // menu's label column is sized from the union of them, so a row that
        // does not declare one would be the one row that gets clipped.
        featherLabel.dataset.labelMax = 'feather 100';
        const featherSlider = document.createElement('input');
        featherSlider.type = 'range';
        featherSlider.id = 'wmaskFeather_' + uuid;
        featherSlider.className = 'forge-toolbar-range';
        featherSlider.min = '0';
        featherSlider.max = '100';
        featherSlider.value = '0';
        featherSlider.title = 'Feather: soften the exported mask edges with a Gaussian falloff '
            + '(100 = blur radius of 2% of the image diagonal); the overlay keeps showing the hard strokes';
        featherRow.appendChild(featherLabel);
        featherRow.appendChild(featherSlider);
        box.insertBefore(featherRow, eraserButton);
        // ...and now that the row is in the menu, recompute that menu's label
        // column so it includes this one. Guarded: the painter must still work
        // on a page where canvas_nodes.js failed to load (the labels then keep
        // the stylesheet's fallback width, which is wide, not broken).
        try {
            if (window.cnproCanvasNodes && window.cnproCanvasNodes.syncLabelSizers) {
                window.cnproCanvasNodes.syncLabelSizers(box);
            }
        } catch (err) {
            console.warn('[cnpro] weight mask: label sizer sync failed', err);
        }
        featherSlider.addEventListener('input', function () {
            st.feather = Math.round(+this.value);
            featherLabel.textContent = 'feather ' + st.feather;
        });
        // re-export on release only - exporting on every input tick would run
        // the full-resolution conversion dozens of times per drag
        featherSlider.addEventListener('change', function () {
            const slot = activeSlot();
            if (slot && slot.hasPaint) syncState(slot);
        });

        if (invertButton) {
            invertButton.addEventListener('click', function () {
                const slot = activeSlot();
                if (slot) invertMask(slot);
            });
        }

        clearButton.addEventListener('click', function () {
            const slot = activeSlot();
            if (slot) clearMask(slot, true);
        });

        weightSlider.addEventListener('input', function () {
            st.weight = Math.min(Math.max(+this.value, 0), 1);
            weightLabel.textContent = 'weight ' + st.weight.toFixed(2);
        });

        brushSlider.addEventListener('input', function () {
            st.brush = Math.round(+this.value);
            brushLabel.textContent = 'brush ' + st.brush;
        });

        // ---- capture-phase interception while a tool is active
        // (same pattern as canvas_extra.js pick modes: block the core
        // scribble/pan handlers without touching toolbar interactions)

        container.addEventListener('pointerdown', function (e) {
            if (!st.active) return;
            if (toolbar.contains(e.target)) return;
            if (e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            const r = imgEl.getBoundingClientRect();
            if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) {
                setActive(null); // clicked outside the image: exit the tool
                return;
            }
            st.painting = true;
            const slot = activeSlot();
            if (slot) slot.last = null;
            try {
                container.setPointerCapture(e.pointerId);
            } catch (err) {}
            updateIndicator(e);
            paintStroke(e, true);
        }, true);

        container.addEventListener('pointermove', function (e) {
            if (!st.active) return;
            updateIndicator(e);
            if (toolbar.contains(e.target)) return;
            e.stopPropagation();
            if (st.painting && (e.buttons & 1)) paintStroke(e, false);
        }, true);

        container.addEventListener('pointerleave', function () {
            indicator.style.display = 'none';
        });

        function finishStroke(slot) {
            slot.last = null;
            if (st.eraser) {
                slot.hasPaint = maskHasPaint(slot);
                // a fully erased mask still holds sub-threshold antialiased
                // fringes in the offscreen canvas; drop them, or the next
                // brush stroke resurrects them into overlay and export
                if (!slot.hasPaint) clearMask(slot, false);
            }
            scheduleSync(slot);
        }

        container.addEventListener('pointerup', function (e) {
            if (!st.painting) return;
            st.painting = false;
            const slot = activeSlot();
            try {
                container.releasePointerCapture(e.pointerId);
            } catch (err) {}
            e.stopPropagation();
            if (slot) finishStroke(slot);
        }, true);

        container.addEventListener('pointercancel', function () {
            if (!st.painting) return;
            st.painting = false;
            const slot = activeSlot();
            if (slot) finishStroke(slot);
        }, true);

        // another tool (crop, point picker, ...) claimed the canvas; it owns
        // the forge-picking class now, so stand down without touching it
        container.addEventListener('forge-canvas-tool', function (e) {
            if (st.active && (!e.detail || e.detail.owner !== 'wmask')) {
                setActive(null, { keepPicking: true });
            }
        });

        // The canvas announced "no image" (clear button, close tab, unit
        // reset): drop every slot's paint. This is what makes clear + reload
        // safe - the dimension check in the rAF loop cannot tell a NEW
        // same-size image from an ADJUSTED one (levels/crop re-uploads, where
        // paint must survive), but only real clears pass through the empty
        // state. Transition-guarded: at startup a restored mask can arrive
        // BEFORE the image loads, and the initial empty announcement must not
        // wipe it. The upload-sequence counter (canvas_extra.js, bumped only
        // for GENUINE new content - uploads, drops, forgeCanvasPush - never
        // for adjustment echoes) closes the remaining hole: a new image
        // REPLACING the old at identical dimensions, which neither the dims
        // check nor the empty transition can see.
        let hadImage = !!container.dataset.forgeImageInfo;
        let lastSeq = container.dataset.forgeUploadSeq || '';
        container.addEventListener('forge-image-info', function () {
            const seq = container.dataset.forgeUploadSeq || '';
            const seqChanged = seq !== lastSeq;
            lastSeq = seq;
            if (container.dataset.forgeImageInfo) {
                if (hadImage && seqChanged) {
                    // same-size replacement: stale paint must not gate the
                    // new image (different-size ones are caught by the rAF
                    // dims check as well - double clearing is harmless)
                    for (const slot of slots) {
                        if (slot.hasPaint || slot.textarea.value) clearMask(slot, true);
                    }
                }
                hadImage = true;
                return;
            }
            if (!hadImage) return;
            hadImage = false;
            for (const slot of slots) {
                if (slot.hasPaint || slot.textarea.value) clearMask(slot, true);
            }
        });

        for (const slot of slots) importState(slot);

        // registered for the module-level watch: a server->client mask push
        // AFTER setup is a value-only textarea write that no observer and no
        // event sees - without this poll it would stay invisible and the next
        // stroke would overwrite it
        registerWhileConnected(painters, container, function watchLateImports() {
            for (const slot of slots) {
                const value = slot.textarea.value;
                if (!slot.hasPaint && value && value.startsWith('data:image')
                        && slot._lastImportTried !== value) {
                    importState(slot);
                }
            }
        });
        return true;
    }

    // Headless load (node, no webui globals): the slot/profile coupling above
    // is the python side's twin (external_code.masks_in_force) and the two must
    // agree, so tests/test_mask_profile_coupling.py runs BOTH rather than
    // reading one and trusting the other. Exporting the rule and guarding the
    // DOM registrations is all that takes - the same arrangement
    // weight_profile.js uses for the profile grammar. Keep both, or the
    // coupling test silently stops running.
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = {
            liveSlotKeys: liveSlotKeys,
            selectedProfileBand: selectedProfileBand,
            weightToRgb: weightToRgb,
            colorToWeight: colorToWeight,
        };
    }
    // The same four in the browser, for coverage_map.js: it decodes the very
    // masks this file exports (wire gray, legacy chromatic) and paints its map
    // in the same hue ramp. Sharing the codec is what makes the coverage map
    // read the same values the generation will - and the hue ramp shared is
    // what makes "red here" mean the same thing in both places.
    if (typeof window !== 'undefined') {
        window.cnproWeightMask = {
            liveSlotKeys: liveSlotKeys,
            selectedProfileBand: selectedProfileBand,
            weightToRgb: weightToRgb,
            colorToWeight: colorToWeight,
            HUE_SPAN: HUE_SPAN,
        };
    }
    if (typeof onUiUpdate !== 'function') {
        // In node that is the point. In a BROWSER it means the painter is about
        // to do nothing at all, for every canvas, with no other symptom - which
        // is precisely the shape of miss this module has been bitten by before.
        // Say so; a guard that can only be silent is half a guard.
        if (typeof window !== 'undefined' && window.document) {
            console.error('[controlnet wmask] onUiUpdate is not available, so the ' +
                          'weight-mask painter registered nothing: every mask button ' +
                          'will be visible and inert. This module expects to run ' +
                          'inside the webui.');
        }
        return;
    }

    // late server->client mask imports (per canvas), coherence checks, and
    // pending-export flushes: all registered per canvas at setup time
    const painters = [];
    const painterWatches = [];
    const painterFlushes = [];

    // A gradio re-render replaces the container, the node-level __cnetWmaskInit
    // latch dies with it, and setup() registers a FRESH set of closures - so
    // the old ones (each holding full-resolution mask canvases and the
    // detached subtree) used to stay in these arrays forever, running on every
    // 500 ms tick and every Generate click, unbounded in re-render count.
    // Every registration is therefore bound to its container: once the
    // container leaves the DOM, the entry removes itself on its next call.
    function registerWhileConnected(list, node, fn) {
        const entry = function () {
            if (!node.isConnected) {
                const i = list.indexOf(entry);
                if (i >= 0) list.splice(i, 1);
                return;
            }
            fn();
        };
        list.push(entry);
    }

    // ------------------------------------------------------------------------
    // WHY THIS RETRIES, AND WHY IT USED NOT TO
    // ------------------------------------------------------------------------
    // This used to latch `__cnetWmaskInit = true` BEFORE calling setup, so a
    // failed setup was permanent. The stated reason was sound at the time:
    //
    //     "the template nodes render together with the container, so a retry
    //      can never succeed later - it would only re-scan forever"
    //
    // That was true when the toolbar lived in the host's canvas.html. It stopped
    // being true the moment CNPro started INJECTING the toolbar at attach time:
    // the nodes now arrive strictly after the container, from a different
    // module, on a different trigger. Whichever observer fires first wins, and
    // when this one won, every weight-mask button was injected, revealed, styled
    // and permanently dead -- clicks did nothing, and the one console warning
    // scrolled past on page load.
    //
    // A comment that is load-bearing for a `return` is only as good as the
    // premise it names. This one outlived its premise by a whole refactor.
    //
    // Two changes: inject first so the race cannot happen at all, and latch only
    // on SUCCESS so a genuinely late DOM still recovers. The failure path is a
    // handful of getElementById calls, so retrying costs nothing measurable.
    const MAX_QUIET_ATTEMPTS = 12; // ~a dozen ui updates before we say something
    onUiUpdate(() => {
        gradioApp().querySelectorAll(
            '.cnet-input-image-group [id^="imageContainer_"], .cnet-output-mask-group [id^="imageContainer_"]'
        ).forEach((container) => {
            if (container.__cnetWmaskInit) return;

            // The toolbar nodes come from canvas_nodes.js, not from the host
            // template. Injecting here too (idempotent, a no-op once done)
            // removes the ordering dependency instead of racing it.
            if (window.cnproCanvasNodes) {
                try {
                    window.cnproCanvasNodes.inject(container.id.replace('imageContainer_', ''));
                } catch (e) {
                    console.error('[controlnet wmask] node injection threw', e);
                }
            }

            if (setup(container)) {
                container.__cnetWmaskInit = true;
                return;
            }

            const tries = (container.__cnetWmaskTries || 0) + 1;
            container.__cnetWmaskTries = tries;
            if (tries === MAX_QUIET_ATTEMPTS) {
                // Said ONCE, naming the missing nodes. Not latched: if the DOM
                // arrives later than this the next update still picks it up.
                console.warn('[controlnet wmask] painter setup has failed ' + tries +
                    ' times for ' + container.id + ' - the weight mask buttons are ' +
                    'visible but inert. Missing: ' +
                    ((setup.lastMissing || ['(unknown - no slot definitions; is this ' +
                      'canvas inside .cnet-input-image-group / .cnet-output-mask-group?)'])
                        .join(', ')) +
                    '. These nodes come from javascript/canvas_tools.js via ' +
                    'canvas_nodes.js::inject.');
            }
        });
    });

    // shared 500 ms tick (active_canvas.js): late imports plus the painter/
    // server coherence checks that used to ride the per-canvas rAF loop
    window.cnetRegisterTick(() => {
        for (const watch of painters) watch();
        for (const watch of painterWatches) watch();
    });

    // A debounced export must not still be waiting when the generation request
    // is queued: the heavy mask channels reach the unit State only through
    // their .change round trip, and gradio serializes its queue in click
    // order - so starting the export here, in the CAPTURE phase, puts the
    // State write ahead of the Generate call. Also flushed when the page is
    // hidden (tab switch / close), where timers stop being reliable.
    document.addEventListener('click', function (e) {
        const target = e.target;
        if (!target || !target.closest) return;
        if (!target.closest('#txt2img_generate, #img2img_generate, .cnet-toolbutton')) return;
        for (const flush of painterFlushes) flush();
    }, true);

    document.addEventListener('visibilitychange', function () {
        if (document.visibilityState === 'hidden') {
            for (const flush of painterFlushes) flush();
        }
    });
})();
