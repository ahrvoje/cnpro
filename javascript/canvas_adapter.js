/**
 * CNPro canvas host adapter -- the JS twin of cnpro_host/adapter.py.
 *
 * The image-canvas widget is the one host API that has actually broken across
 * the Forge family, so it gets its own normalization layer instead of being
 * spread through the painter and adjustment code.
 *
 * WHAT CHANGED, CONCRETELY
 * -----------------------
 * lllyasviel Forge shipped `canvas.min.js` (obfuscated) whose ForgeCanvas
 * exposed `uploadBase64(b64)` and `on_img_upload()`. ForgeNeo ships a
 * de-obfuscated `canvas.js` that renamed both:
 *
 *     uploadBase64(b64)   ->  loadImage(base64)
 *     on_img_upload()     ->  updateBackgroundImageData()
 *
 * Everything else CNPro depends on survived unchanged: `this.img`,
 * `background_gradio_bind`, `drawImage`, `saveState`, `removeImage`,
 * `adjustInitialPositionAndScale`, the constructor argument order, and the
 * `imageContainer_<uuid>` / `image_<uuid>` / `toolbar_<uuid>` id scheme.
 *
 * WHICH DIRECTION THE ALIAS MUST POINT
 * ------------------------------------
 * This is the whole reason the file is subtle, and getting it backwards is
 * silent and total.
 *
 * The obvious alias is "give CNPro the name it wants":
 *
 *     proto.uploadBase64 = function (b64) { return this.loadImage(b64); };
 *
 * That satisfies the method-presence check, and every tool is dead anyway.
 * canvas_extra.js works by WRAPPING these methods -- it replaces
 * `proto.uploadBase64` so that every image entering the widget is captured as
 * `st.original`, the source every adjustment renders from. But the host never
 * calls `uploadBase64`. It calls `this.loadImage(...)` -- from the gradio bind,
 * from the file picker, from paste, from removeImage. With the alias pointing
 * CNPro -> host, those inflows walk straight past the wrapper, `st.original`
 * stays null, and `renderAdjusted()` returns on its first line. Every menu
 * still opens, every slider still moves, and NOTHING happens to the image.
 * (Observed exactly that: crop, rotate, flip, layers and the pickers all inert
 * while the toolbar looked perfectly healthy.)
 *
 * So the alias points the other way. The canonical name RECEIVES the host's
 * implementation, and the host's own name becomes the delegate:
 *
 *     proto.uploadBase64 = <host's loadImage implementation>
 *     proto.loadImage    = function (...a) { return this.uploadBase64(...a); }
 *
 * Now `loadImage` is a call site like any other, canvas_extra.js's wrapper sits
 * on the single canonical implementation, and every inflow -- whichever name it
 * arrives under -- passes through it. The rule generalises:
 *
 *   ALIAS THE NAME THE HOST CALLS, NOT THE NAME WE CALL. A wrapper is only
 *   worth anything on the method the host actually invokes.
 *
 * WHY A PURE FUNCTION AND NOT A POLLER
 * ------------------------------------
 * `class ForgeCanvas {}` at the top level of a classic script creates a global
 * LEXICAL binding, not a property on `window` -- so it cannot be intercepted
 * with defineProperty, and a second poller would race canvas_extra.js's own
 * bootstrap. canvas_extra.js bails permanently (with a console warning) the
 * first time it finds a method missing, so losing that race would silently
 * disable every adjustment tool.
 *
 * Instead this file only DEFINES a function. canvas_extra.js calls it as the
 * first statement of its install(), by which point ForgeCanvas certainly
 * exists. Load order is filename order and "canvas_adapter" sorts before
 * "canvas_extra", so the function is always defined in time. No race exists.
 *
 * ADDING A HOST
 * -------------
 * Append a row to ALIASES. Aliases are only installed when the canonical name
 * is missing AND the replacement is present, so listing a mapping that does not
 * apply to the current host is harmless.
 */
(function () {
    "use strict";

    // canonical name (what CNPro's code calls)  <-  host's name for it
    const ALIASES = [
        // ForgeNeo / Forge Classic neo
        ["uploadBase64", "loadImage"],
        ["on_img_upload", "updateBackgroundImageData"],
    ];

    // Methods CNPro genuinely cannot work without, checked after aliasing so
    // the diagnostic names the CANONICAL method rather than the host's.
    const REQUIRED = [
        "uploadBase64",
        "removeImage",
        "adjustInitialPositionAndScale",
        "saveState",
        "drawImage",
        "on_img_upload",
    ];

    /**
     * Give `proto` the canonical CNPro canvas API, whatever the host calls it,
     * AND route the host's own name through it (see the direction note above).
     *
     * Idempotent: once `proto[canonical]` exists, a second call is a no-op, so
     * the delegate can never be wrapped around itself.
     *
     * @returns {{ok: boolean, missing: string[], aliased: string[]}}
     */
    function normalize(proto) {
        const aliased = [];
        for (const [canonical, hostName] of ALIASES) {
            if (typeof proto[canonical] === "function") continue;
            const impl = proto[hostName];
            if (typeof impl !== "function") continue;

            // The canonical name BECOMES the implementation, so canvas_extra.js's
            // wrapper lands on the one function everything ends up executing...
            proto[canonical] = impl;
            proto[canonical].cnproCanonicalFor = hostName;

            // ...and the host's own name becomes an ordinary call site that goes
            // through it. `this[canonical]` is resolved at call time, never
            // captured, so it picks up canvas_extra.js's wrapper even though the
            // wrapper is installed after this runs.
            proto[hostName] = function (...args) {
                return this[canonical](...args);
            };
            proto[hostName].cnproDelegatesTo = canonical;

            aliased.push(hostName + " -> " + canonical);
        }

        const missing = REQUIRED.filter((n) => typeof proto[n] !== "function");
        if (aliased.length) {
            console.log("[cnpro] canvas API normalized, host calls routed through " +
                        "CNPro: " + aliased.join(", "));
        }
        if (missing.length) {
            console.warn(
                "[cnpro] unsupported canvas host - missing: " + missing.join(", ") +
                ". Adjustment tools and the weight-mask painter are disabled. " +
                "Add the host's names to ALIASES in javascript/canvas_adapter.js."
            );
        }
        return {ok: missing.length === 0, missing: missing, aliased: aliased};
    }

    window.cnproCanvasApi = {normalize: normalize, ALIASES: ALIASES};
})();
