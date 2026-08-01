// Insert-image buttons of the below-canvas row (⤵I / ⤵O): push the img2img
// source image or the current output image into the canvas of whichever tab
// the unit currently shows (input image / output mask backdrop).
// Fully client-side - the pixels never round-trip through the server: the
// click reads the source <img>, normalizes it to a PNG data url and hands it
// to window.forgeCanvasPush (modules_forge/forge_canvas/canvas_extra.js),
// which uploads it into the target ForgeCanvas exactly like a user upload.
//
// The buttons are rendered disabled (interactive=False) and are enabled here
// whenever their source is available:
//   ⤵I - only on img2img units, and only while the img2img source canvas
//        holds an image (txt2img units keep it permanently disabled);
//   ⤵O - while the unit's own tab has a generation result in its gallery
//        (the selected gallery image wins, else the first one).
(function () {
    'use strict';

    // the img2img source canvas image, or null while it is empty
    function img2imgSourceImg() {
        const img = gradioApp().querySelector('#img2img_image [id^="imageContainer_"] .forge-image');
        return (img && img.src && img.src.startsWith('data:image')) ? img : null;
    }

    // the current output image of the given tab's result gallery, or null
    function outputImg(isImg2img) {
        const gallery = gradioApp().querySelector(isImg2img ? '#img2img_gallery' : '#txt2img_gallery');
        if (!gallery) return null;
        const img = gallery.querySelector('.thumbnail-item.selected img')
            || gallery.querySelector('.thumbnail-item img');
        return (img && img.src) ? img : null;
    }

    // gradio's LogicalImage sync accepts only 'data:image/png;base64,...', so
    // anything else (gallery file urls, jpeg data urls) is re-encoded via a
    // scratch canvas at natural resolution
    function toPngDataUrl(srcImg, callback) {
        const src = srcImg.src;
        if (src.startsWith('data:image/png;base64,')) {
            callback(src);
            return;
        }
        const img = new Image();
        img.onload = function () {
            const c = document.createElement('canvas');
            c.width = img.naturalWidth;
            c.height = img.naturalHeight;
            c.getContext('2d').drawImage(img, 0, 0);
            try {
                callback(c.toDataURL('image/png'));
            } catch (e) {
                console.warn('[controlnet insert] could not encode the source image:', e);
                callback(null);
            }
        };
        img.onerror = function () {
            console.warn('[controlnet insert] could not load the source image:', src);
            callback(null);
        };
        img.src = src;
    }

    function pushToUnit(btn, srcImg) {
        if (!srcImg) return;
        const unit = btn.closest('.input-accordion');
        // active-tab resolution lives in active_canvas.js (shared with
        // image_info.js and the python-generated button-row _js)
        const container = unit && window.cnetActiveCanvasContainer(unit);
        if (!container) return;
        const uuid = container.id.slice('imageContainer_'.length);
        toPngDataUrl(srcImg, function (dataUrl) {
            if (!dataUrl) return;
            if (!window.forgeCanvasPush || !window.forgeCanvasPush(uuid, dataUrl)) {
                console.warn('[controlnet insert] no ForgeCanvas registered for', uuid);
            }
        });
    }

    // The three source helpers, for anything else that offers "insert the
    // input / output raster". The coverage panel does, for its backdrop, and a
    // second copy of "which img is the current output" would be a second answer
    // the day gradio changes its gallery markup.
    window.cnetInsertSources = {
        img2imgSourceImg: img2imgSourceImg,
        outputImg: outputImg,
        toPngDataUrl: toPngDataUrl,
    };

    // one delegated listener instead of per-button wiring: gradio re-renders
    // survive for free (the buttons carry no server-side click handler)
    document.addEventListener('click', function (e) {
        if (!e.target || !e.target.closest) return;
        const btn = e.target.closest('.cnet-insert-input-image, .cnet-insert-output-image');
        if (!btn || btn.disabled) return;
        if (btn.classList.contains('cnet-insert-input-image')) {
            pushToUnit(btn, img2imgSourceImg());
        } else {
            pushToUnit(btn, outputImg(btn.id.startsWith('img2img')));
        }
    });

    // disabled-state sync. Runs from onUiUpdate (every DOM mutation batch), so
    // it must stay a pure read unless a state really flips - btn.disabled
    // writes are themselves mutations and unconditional writes would loop.
    function syncDisabled() {
        const app = gradioApp();
        const haveSource = !!img2imgSourceImg();
        app.querySelectorAll('.cnet-insert-input-image').forEach(function (btn) {
            const want = !(btn.id.startsWith('img2img') && haveSource);
            if (btn.disabled !== want) btn.disabled = want;
        });
        const haveOut = {true: !!outputImg(true), false: !!outputImg(false)};
        app.querySelectorAll('.cnet-insert-output-image').forEach(function (btn) {
            const want = !haveOut[btn.id.startsWith('img2img')];
            if (btn.disabled !== want) btn.disabled = want;
        });
    }

    onUiLoaded(syncDisabled);
    onUiUpdate(syncDisabled);
    // src-attribute swaps on the img2img canvas are not childList mutations,
    // so onUiUpdate can miss them - but canvas_extra.js announces every image
    // change (upload/clear/adjust) with this bubbling event
    document.addEventListener('forge-image-info', syncDisabled);
})();
