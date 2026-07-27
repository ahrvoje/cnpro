// Raster info of the ControlNet input image - source format and pixel
// dimensions of whatever is loaded into the input canvas, shown as the line
// under the canvas ("<FMT>, <W> x <H> px (<CW> x <CH>)"; the line's host <p>
// carries no other text since the [invert] hint moved into the Preprocessor
// label). The bracketed pair is the current cropped result and appears only
// while a crop is applied (kept current by canvas_extra.js after every crop
// action).
//
// The info arrives via the 'forge-image-info' events + dataset that
// modules_forge/forge_canvas/canvas_extra.js maintains on the image container
// (the displayed src is a re-encoded PNG as soon as any adjustment renders,
// so only the canvas adjustment state knows the truly uploaded format). A
// direct read of the displayed image is the fallback for canvases populated
// before the adjustment layer attached (e.g. session restore).
(function () {
    'use strict';

    function formatLabel(format) {
        if (!format) return '';
        const f = format.toLowerCase();
        if (f === 'svg+xml') return 'SVG';
        if (f === 'x-icon' || f === 'vnd.microsoft.icon') return 'ICO';
        return f.toUpperCase();
    }

    function infoText(detail) {
        if (!detail || !detail.width || !detail.height) return '';
        const fmt = formatLabel(detail.format);
        let text = (fmt ? fmt + ', ' : '') + detail.width + ' × ' + detail.height + ' px';
        if (detail.cropWidth && detail.cropHeight) {
            text += ' (' + detail.cropWidth + ' × ' + detail.cropHeight + ')';
        }
        return text;
    }

    function updateUnit(container) {
        // input canvases and the Output-mask canvas write the line (the line
        // used to describe the last-viewed INPUT while the Output mask tab
        // showed an unrelated backdrop); mask/preview canvases stay silent
        const inputGroup = container.closest('.cnet-input-image-group');
        const outputGroup = inputGroup ? null : container.closest('.cnet-output-mask-group');
        if (!inputGroup && !outputGroup) return;
        // A unit has one input canvas per Input tab but only ONE hint line, so
        // exactly one canvas may own it: the one whose tab is open (visibility
        // rule centralized in active_canvas.js). Letting every input canvas
        // write the shared line makes the filled and the empty ones overwrite
        // each other on every DOM mutation, which is the endless mutation ->
        // onUiUpdate -> mutation loop described below.
        if (!window.cnetVisible(container)) return;
        const unit = container.closest('.input-accordion');
        const hint = unit && unit.querySelector('.controlnet_invert_warning p');
        if (!hint) return;
        let detail = null;
        const stored = container.dataset.forgeImageInfo;
        if (stored) {
            try {
                detail = JSON.parse(stored);
            } catch (e) {}
        } else if (stored === undefined) {
            // never announced: the image was pushed before canvas_extra.js
            // attached, its displayed src still is exactly what was loaded
            const img = container.querySelector('.forge-image');
            if (img && img.src && img.src.startsWith('data:image') && img.naturalWidth) {
                const m = /^data:image\/([a-z0-9.+-]+)/i.exec(img.src);
                detail = {format: m ? m[1] : null, width: img.naturalWidth, height: img.naturalHeight};
            }
        }
        // this runs from onUiUpdate, i.e. on every DOM mutation: it MUST stay
        // a pure read unless the text really changed, because writing
        // textContent is itself a mutation and an unconditional write feeds an
        // endless mutation -> update -> mutation loop that stalls the page
        let text = infoText(detail);
        // the Output-mask image is a throwaway drawing backdrop, and the
        // painted mask maps onto the OUTPUT rectangle by plain resize - say
        // both, or the line reads like a control-input description
        if (text && outputGroup) text = 'backdrop: ' + text + ' (mask maps onto output)';
        let label = hint.querySelector('.cnet-image-raster-info');
        if (!label) {
            if (!text) return;
            label = document.createElement('b');
            label.className = 'cnet-image-raster-info';
            hint.insertBefore(label, hint.firstChild);
        }
        if (label.textContent !== text) label.textContent = text;
    }

    document.addEventListener('forge-image-info', function (e) {
        if (e.target && e.target.closest) updateUnit(e.target);
    });

    onUiUpdate(() => {
        gradioApp().querySelectorAll(
            '.cnet-input-image-group [id^="imageContainer_"], .cnet-output-mask-group [id^="imageContainer_"]'
        ).forEach(updateUnit);
    });
})();
