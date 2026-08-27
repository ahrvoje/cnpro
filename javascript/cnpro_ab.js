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

/* ----------------------------------------------------------------------
 * The "Canvas layer" row - the browser's half.
 *
 * The layer stack of a canvas exists ONLY here (canvas_extra.js): gradio
 * receives the flattened composite and nothing else, so python can neither
 * list a canvas' layers nor re-composite one at another opacity. Three
 * channels, all through hidden components of the A/B panel, so the python
 * side keeps its one-blocking-run()-plus-a-poll shape (CNPro_AB.py, HOW IT
 * RUNS):
 *
 *   inventory   JS -> python   `_canvases` textbox. Which canvases of this
 *                              tab hold an image, and each one's layers with
 *                              their current opacity - what the row's two
 *                              dropdowns offer. Polled once a second while
 *                              the panel is on screen, written only when it
 *                              CHANGES (a textbox write is a gradio round
 *                              trip and a re-render).
 *   request     python -> JS   `_canvas_request` HTML. A duel side about to
 *   reply       JS -> python   render asks for the composite of each named
 *                              canvas with the named layers at the named
 *                              opacities; the answer goes back through the
 *                              `_canvas_reply` textbox, with the request's
 *                              sequence number so a stale answer is ignored.
 *                              The composite is rendered OFF-SCREEN by
 *                              forgeCanvasComposite - the live canvas does
 *                              not change, which is the "nothing is written
 *                              back to the UI while searching" rule.
 *   set         python -> JS   `_canvas_set` HTML. Set / Set GOOD / Reset
 *                              write a layer's opacity INTO the live canvas
 *                              - the one path that is meant to change it.
 *
 * The two python -> JS channels are gr.HTML, not textboxes: a value-only
 * textbox write makes no DOM mutation and no event (MAINTENANCE invariant
 * 18), while an HTML update replaces child nodes, which a MutationObserver
 * sees at once - no second poll for a request that arrives at most once per
 * generation.
 *
 * A canvas is named by WHERE IT LIVES, not by its uuid: the uuid is minted
 * per page load, and the row's choice has to survive a reload, an Export
 * and a recipe. `u0.in1` is unit 0's Input 1 canvas (1-based, as the tabs
 * are labelled); `img2img` is the host's own img2img canvas - the init
 * image. The python side derives the label from the same key (_canvas_label)
 * and applies the composite by the same key (_apply_composites).
 * ---------------------------------------------------------------------- */
(function () {
    "use strict";

    const TABS = ["txt2img", "img2img"];
    const UNIT_CANVAS = /^(txt2img|img2img)_controlnet_ControlNet-(\d+)_input_image(?:_(\d+))?$/;
    /* The host's own canvases this panel can vary, by block id: only the
       img2img init image. The sketch/inpaint canvases composite their
       foreground into the init image on the host side, which a background
       composite from here would silently drop. */
    const HOST_CANVAS = {img2img_image: "img2img"};

    function keyOf(container) {
        /* Walk up to the gr.HTML block that ForgeCanvas built, whose elem_id
           says which unit input - or which host canvas - this is. */
        for (let el = container.parentElement; el; el = el.parentElement) {
            if (!el.id) continue;
            const unit = UNIT_CANVAS.exec(el.id);
            if (unit) {
                const slot = unit[3] ? parseInt(unit[3], 10) : 0;
                return {tab: unit[1], key: "u" + unit[2] + ".in" + (slot + 1)};
            }
            if (HOST_CANVAS[el.id]) return {tab: "img2img", key: HOST_CANVAS[el.id]};
        }
        return null;
    }

    /* Every canvas of `tab` that holds an image: key -> uuid, with its
       layers. Rebuilt on every use - a canvas can appear (a new Input tab)
       or empty at any time, and the walk is a few hundred nodes. */
    function inventory(tab) {
        const found = [];
        if (typeof window.forgeCanvasDebugLayers !== "function") return found;
        for (const container of document.querySelectorAll("[id^='imageContainer_uuid_']")) {
            const where = keyOf(container);
            if (!where || where.tab !== tab) continue;
            const uuid = container.id.slice("imageContainer_".length);
            const snap = window.forgeCanvasDebugLayers(uuid);
            if (!snap || !snap.layers.length) continue;
            found.push({
                key: where.key, uuid: uuid,
                layers: snap.layers.map(l => ({
                    size: l.w && l.h ? l.w + "×" + l.h : "empty",
                    opacity: Math.round(l.opacity * 100),
                })),
            });
        }
        found.sort((a, b) => a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
        return found;
    }

    function textareaOf(id) {
        const box = document.getElementById(id);
        return box ? box.querySelector("textarea") : null;
    }

    function send(id, value) {
        const textarea = textareaOf(id);
        if (!textarea) return false;
        textarea.value = value;
        /* Through updateInput, so the value reaches gradio and not only the
           DOM - see appendLine above. */
        if (typeof updateInput === "function") {
            updateInput(textarea);
        } else {
            textarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
        return true;
    }

    function panelVisible(tab) {
        const duel = document.getElementById("cnpro_ab_" + tab + "_duel");
        return !!(duel && duel.offsetParent);
    }

    /* ---- inventory: once a second, only what changed ------------------ */

    const lastSent = {};

    function publishInventory(tab) {
        if (!panelVisible(tab)) return;
        const list = inventory(tab).map(c => ({key: c.key, layers: c.layers}));
        const text = JSON.stringify(list);
        if (text === lastSent[tab]) return;
        if (send("cnpro_ab_" + tab + "_canvases", text)) lastSent[tab] = text;
    }

    setInterval(function () {
        for (const tab of TABS) {
            try {
                publishInventory(tab);
            } catch (err) {
                console.warn("[cnpro A/B] canvas inventory failed", err);
            }
        }
    }, 1000);

    /* ---- the python -> JS channels: a gr.HTML each, observed ----------- */

    function channelText(element) {
        const body = element.querySelector(".cnpro-ab-channel-body");
        return body ? body.textContent : "";
    }

    function watch(id, handler) {
        const element = document.getElementById(id);
        if (!element || element.dataset.cnproAbWatched === "1") return;
        element.dataset.cnproAbWatched = "1";
        let last = null;
        const look = function () {
            const text = channelText(element);
            if (!text || text === last) return;
            last = text;
            let message;
            try {
                message = JSON.parse(text);
            } catch (err) {
                console.warn("[cnpro A/B] unreadable channel message in " + id, err);
                return;
            }
            try {
                handler(message);
            } catch (err) {
                console.warn("[cnpro A/B] channel handler failed for " + id, err);
            }
        };
        new MutationObserver(look).observe(
            element, { childList: true, subtree: true, characterData: true });
        /* A page reloaded mid-search already holds the request. */
        look();
    }

    /* A render request: {seq, overrides: {key: {index: alpha}}} -> a reply
       {seq, images: {key: dataUrl}} or {seq, error}. Every named canvas has
       to answer, or the generation would run on the live composite for the
       one that did not - which is the silent miss the error exists to
       prevent. */
    function serveRequest(tab, message) {
        const overrides = message.overrides || {};
        const known = {};
        for (const canvas of inventory(tab)) known[canvas.key] = canvas;
        const images = {};
        const keys = Object.keys(overrides);
        let pending = keys.length;
        let error = null;
        const finish = function () {
            const reply = error ? {seq: message.seq, error: error}
                                : {seq: message.seq, images: images};
            send("cnpro_ab_" + tab + "_canvas_reply", JSON.stringify(reply));
        };
        if (!pending) {
            finish();
            return;
        }
        for (const key of keys) {
            const canvas = known[key];
            const layers = overrides[key] || {};
            const bad = Object.keys(layers).find(
                i => !canvas || !(parseInt(i, 10) < canvas.layers.length));
            if (!canvas || bad !== undefined) {
                error = !canvas
                    ? key + " holds no image any more"
                    : key + " has " + canvas.layers.length + " layer(s), not a layer "
                      + (parseInt(bad, 10) + 1);
                if (--pending === 0) finish();
                continue;
            }
            window.forgeCanvasComposite(canvas.uuid, layers, function (url) {
                if (url) images[key] = url;
                else error = error || (key + " rendered nothing");
                if (--pending === 0) finish();
            });
        }
    }

    /* Set / Set GOOD / Reset: {seq, layers: {key: {index: alpha}}}. Applied
       to the LIVE canvas, and the inventory is republished at once so the
       row's dropdown and the next Reset see the new value. */
    function serveSet(tab, message) {
        const known = {};
        for (const canvas of inventory(tab)) known[canvas.key] = canvas;
        const missed = [];
        for (const key of Object.keys(message.layers || {})) {
            const layers = message.layers[key];
            for (const index of Object.keys(layers)) {
                const canvas = known[key];
                if (!canvas || !window.forgeCanvasSetLayerOpacity(
                        canvas.uuid, parseInt(index, 10), layers[index])) {
                    missed.push(key + " layer " + (parseInt(index, 10) + 1));
                }
            }
        }
        if (missed.length) {
            console.warn("[cnpro A/B] Set could not reach: " + missed.join(", "));
        }
        lastSent[tab] = null;
        publishInventory(tab);
    }

    function bind() {
        for (const tab of TABS) {
            watch("cnpro_ab_" + tab + "_canvas_request", m => serveRequest(tab, m));
            watch("cnpro_ab_" + tab + "_canvas_set", m => serveSet(tab, m));
        }
    }

    if (typeof onUiUpdate === "function") {
        /* The panel's blocks may not be mounted at load time, and gradio
           can rebuild them; `watch` binds once per element and skips what
           it has already bound. */
        onUiUpdate(bind);
    } else {
        document.addEventListener("DOMContentLoaded", bind);
    }
})();
