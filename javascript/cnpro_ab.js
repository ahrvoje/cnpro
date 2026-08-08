/* CNPro A/B - the duel images open full size, and the LoRA picker survives
 * being used.
 *
 * THE PROBLEM
 * -----------
 * A LoRA row's textbox lists the candidates for one LoRA slot, so the picker
 * beside it is used four or ten times in a row, not once. Gradio's dropdown is
 * built for the opposite: picking an option sets the value, blurs the filter
 * input and closes the option list, and the python handler that appends the
 * line then clears the picker, which re-renders the component. Per LoRA that
 * costs the user re-opening the list, re-typing the filter they had already
 * typed, and re-finding their place in it. The list also scrolls itself to the
 * selected index whenever it reopens, so even the scroll position is not
 * theirs to keep.
 *
 * WHAT THIS DOES INSTEAD
 * ----------------------
 * Suppresses the selection at the DOM level and does the insert here. Gradio's
 * dropdown selects on MOUSEDOWN on the option list (`ul.options`), not on
 * click - so a capture-phase mousedown listener sees it first:
 *
 *   preventDefault()   the default action of mousedown is moving focus. Killing
 *                      it keeps the filter input focused, and the list closes on
 *                      the input's blur - so this alone is what keeps it open.
 *   stopPropagation()  during CAPTURE, so the event never reaches gradio's own
 *                      listener on the ul. Nothing in the component's state
 *                      changes: not the value, not the filter, not the scroll.
 *
 * The result is that the list stays exactly as the user left it and each click
 * adds one line. The filter keeps working because it was never touched - it is
 * gradio's own, still filtering, still holding what was typed into it.
 *
 * The python side (scripts/CNPro_AB.py, `add_lora`) still has the round-trip
 * version and is the fallback for this file not being loaded - a browser cache
 * away. The two cannot double-insert: this one only inserts when it has
 * swallowed the event that would have triggered the other.
 *
 * THE ONE DIFFERENCE FROM javascript/cnpro_xy.js, which does the same thing for
 * the X/Y panel and carries the same explanation: X/Y inserts
 * `<lora:name:1>`, because there the weight is part of the grid cell. DNA
 * inserts the BARE NAME, because the weight is the thing the search
 * determines - a weight typed here would be a number the user guessed, which
 * is exactly what the row exists to stop them having to do.
 */
(function () {
    "use strict";

    /* Both tabs build the panel and each has MAX_ROWS rows, so twice that
       many pickers exist in one document. Matched whole, not by prefix: the
       id of the textbox these write into is derived from it below, and a
       near-miss would write into a component that is not there. */
    const PICKER_ID = /^cnpro_ab_(txt2img|img2img)_r\d+_lora_pick$/;

    function textareaFor(picker) {
        const box = document.getElementById(
            picker.id.replace(/_lora_pick$/, "_loras"));
        return box ? box.querySelector("textarea") : null;
    }

    function appendLine(textarea, entry) {
        const body = textarea.value.replace(/\s+$/, "");
        textarea.value = body ? body + "\n" + entry : entry;
        /* Through updateInput, so gradio's frontend state follows - the value
           the script reads at generation time is the one gradio holds, not the
           one in the DOM. Writing .value alone leaves the two disagreeing and
           the added LoRAs simply are not searched over. */
        if (typeof updateInput === "function") {
            updateInput(textarea);
        } else {
            textarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
        /* The box is two lines tall and is filled from the top; without this
           the line just added is out of sight from the third one on. */
        textarea.scrollTop = textarea.scrollHeight;
    }

    /* Delegated from the document and registered at parse time rather than in
       onUiLoaded: the option list is created and destroyed every time the
       dropdown opens, so there is no element here worth binding to, and
       nothing to wait for. */
    document.addEventListener("mousedown", function (event) {
        const target = event.target;
        if (!target || typeof target.closest !== "function") return;

        const item = target.closest("li.item");
        if (!item) return;

        const picker = item.closest("[id$='_lora_pick']");
        if (!picker || !PICKER_ID.test(picker.id)) return;

        const textarea = textareaFor(picker);
        /* aria-label carries the option's LABEL, which for this dropdown is
           the LoRA name itself - the choices are plain strings, so label and
           value are the same string. Read from the attribute rather than from
           the li's text, which also contains the selected-marker glyph. */
        const name = item.getAttribute("aria-label");
        /* BOTH resolved before anything is suppressed. Swallowing the event
           and then finding nothing to insert would be the one outcome worse
           than either path: a dropdown where clicking an option does nothing
           at all, silently, with the python fallback shut out too. */
        if (!textarea || !name) return;

        event.preventDefault();
        event.stopPropagation();
        appendLine(textarea, name);
    }, true);

    /* ----------------------------------------------------------------------
     * The duel images, full size on click.
     *
     * Grading is a judgement about detail, and detail is exactly what a 360px
     * pane does not show - so both images have to open full size.
     *
     * WHY NOT THE HOST'S LIGHTBOX, which would have been the obvious answer
     * and was the first one tried: it is bound to the output GALLERY, not to
     * whatever was clicked. `updateOnBackgroundChange()` runs on every UI
     * update, and while the modal is open it overwrites the modal image with
     * the gallery's currently selected one. A panel that polls - which this
     * one does, several times a second, for the whole search - therefore has
     * its duel image replaced by the last generated image within a tick of
     * opening it. Measured, and reported as "maximizing A shows the last
     * image instead".
     *
     * So this is a private overlay. It is deliberately small: fit to the
     * window, click the image to toggle 1:1 (which is the whole point - a
     * duel is decided on detail), click the backdrop or press Escape to
     * close, and arrow keys to flip between A and B without closing.
     * ---------------------------------------------------------------------- */

    const DUEL_IMAGE_SELECTOR =
        "[id^='cnpro_ab_'][id$='_image_a'] img, [id^='cnpro_ab_'][id$='_image_b'] img";

    let overlay = null;
    let overlayImage = null;
    let zoomed = false;

    function ensureOverlay() {
        if (overlay) return overlay;
        overlay = document.createElement("div");
        overlay.className = "cnpro-ab-modal";
        overlayImage = document.createElement("img");
        overlayImage.className = "cnpro-ab-modal-image";
        overlay.appendChild(overlayImage);
        /* On the BACKDROP, so a click on the image itself does not close it -
           that click is the zoom toggle. */
        overlay.addEventListener("click", function (event) {
            if (event.target === overlay) close();
        });
        overlayImage.addEventListener("click", function (event) {
            event.stopPropagation();
            setZoom(!zoomed);
        });
        document.body.appendChild(overlay);
        return overlay;
    }

    function setZoom(on) {
        zoomed = on;
        overlay.classList.toggle("cnpro-ab-modal-zoom", zoomed);
    }

    function open(src) {
        ensureOverlay();
        overlayImage.src = src;
        setZoom(false);
        overlay.classList.add("cnpro-ab-modal-open");
    }

    function close() {
        if (overlay) overlay.classList.remove("cnpro-ab-modal-open");
    }

    function sideImages() {
        return Array.from(document.querySelectorAll(DUEL_IMAGE_SELECTOR))
            .filter(image => image.offsetParent);
    }

    /* Delegated, not bound per <img>: the image element is REPLACED every time
       a duel is painted, so anything bound to the element that was there at
       load time is bound to an element the user can no longer click. */
    document.addEventListener("click", function (event) {
        const target = event.target;
        if (!target || target.tagName !== "IMG") return;
        if (!target.matches(DUEL_IMAGE_SELECTOR)) return;
        event.preventDefault();
        event.stopPropagation();
        open(target.src);
    }, true);

    /* ----------------------------------------------------------------------
     * Let the output column be narrower than the host allows.
     *
     * The splitter between the settings column and the results column stops
     * at 320px on either side. That is a sensible default for a page whose
     * left column is a form - and the wrong one here, because during a search
     * the left column holds two images being compared and the right column
     * holds a gallery nobody is looking at. The width has to be able to go
     * where the attention is.
     *
     * The host keeps those limits as PROPERTIES ON THE SPLITTER ELEMENT
     * (`parent.minLeftColWidth` / `minRightColWidth`, read live on every
     * mousemove - resizeHandle.js), not as constants baked into the drag. So
     * relaxing one is a property write on an element, with no patched
     * function and no edited host file. Only the RIGHT limit is touched: the
     * left column is the one this panel lives in, and shrinking that below
     * the host's floor helps nobody.
     * ---------------------------------------------------------------------- */

    const MIN_RIGHT_COLUMN = 120;

    function relaxColumnMinimum() {
        for (const handle of document.querySelectorAll(".resize-handle")) {
            const parent = handle.parentElement;
            if (!parent) continue;
            /* Absent until the host has set this splitter up, and 0 on the
               ones whose right column is already unconstrained - neither is
               ours to change. */
            if (typeof parent.minRightColWidth !== "number") continue;
            if (parent.minRightColWidth > MIN_RIGHT_COLUMN) {
                parent.minRightColWidth = MIN_RIGHT_COLUMN;
            }
        }
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(relaxColumnMinimum);
    }

    document.addEventListener("keydown", function (event) {
        if (!overlay || !overlay.classList.contains("cnpro-ab-modal-open")) return;
        if (event.key === "Escape") {
            close();
        } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            /* A and B, without going back to the panel - which is how a close
               look at one side turns into a comparison. */
            const images = sideImages();
            const at = images.findIndex(image => image.src === overlayImage.src);
            if (images.length > 1) {
                const step = event.key === "ArrowRight" ? 1 : -1;
                const next = (at + step + images.length) % images.length;
                overlayImage.src = images[next].src;
            }
        } else {
            return;
        }
        event.preventDefault();
        event.stopPropagation();
    }, true);
})();
