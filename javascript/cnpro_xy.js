/* CNPro X/Y - the LoRA picker survives being used, and the tool buttons line up.
 *
 * THE PROBLEM
 * -----------
 * The "add LoRA" dropdown of the X/Y panel exists to fill a multi-line textbox,
 * one LoRA per grid cell - so it is used four or ten times in a row, not once.
 * Gradio's dropdown is built for the opposite: picking an option sets the
 * value, blurs the filter input and closes the option list, and the python
 * handler that appends the line then clears the picker, which re-renders the
 * component. Per LoRA that costs the user re-opening the list, re-typing the
 * filter they had already typed, and re-finding their place in it. The list
 * also scrolls itself to the selected index whenever it reopens, so even the
 * scroll position is not theirs to keep.
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
 * The python side (scripts/CNPro_XY.py, `add_lora`) still has the round-trip
 * version and is the fallback for this file not being loaded - a browser cache
 * away. The two cannot double-insert: this one only inserts when it has
 * swallowed the event that would have triggered the other.
 */
(function () {
    "use strict";

    /* MIRRORS LORA_DEFAULT_WEIGHT in scripts/CNPro_XY.py. Both paths insert the
       same line, so a user cannot tell which one served them - which is the
       point of a fallback. */
    const DEFAULT_WEIGHT = 1;

    /* Both tabs build the panel and each has two axes, so four pickers exist in
       one document. Matched whole, not by prefix: the id of the LoRA textbox
       these write into is derived from it below, and a near-miss would write
       into a component that is not there. */
    const PICKER_ID = /^cnpro_xy_(txt2img|img2img)_[xy]_lora_pick$/;

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
           the added LoRAs simply do not run. */
        if (typeof updateInput === "function") {
            updateInput(textarea);
        } else {
            textarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
        /* The box is three lines tall and is being filled from the top; without
           this the line just added is out of sight from the fourth one on. */
        textarea.scrollTop = textarea.scrollHeight;
    }

    /* Delegated from the document and registered at parse time rather than in
       onUiLoaded: the option list is created and destroyed every time the
       dropdown opens, so there is no element here that is worth binding to, and
       nothing to wait for. */
    document.addEventListener("mousedown", function (event) {
        const target = event.target;
        if (!target || typeof target.closest !== "function") return;

        const item = target.closest("li.item");
        if (!item) return;

        const picker = item.closest("[id$='_lora_pick']");
        if (!picker || !PICKER_ID.test(picker.id)) return;

        const textarea = textareaFor(picker);
        /* aria-label carries the option's LABEL, which for this dropdown is the
           LoRA name itself - the choices are plain strings, so label and value
           are the same string. Read from the attribute rather than from the
           li's text, which also contains the selected-marker glyph. */
        const name = item.getAttribute("aria-label");
        /* BOTH resolved before anything is suppressed. Swallowing the event and
           then finding nothing to insert would be the one outcome worse than
           either path: a dropdown where clicking an option does nothing at all,
           silently, with the python fallback shut out too. */
        if (!textarea || !name) return;

        event.preventDefault();
        event.stopPropagation();
        appendLine(textarea, "<lora:" + name + ":" + DEFAULT_WEIGHT + ">");
    }, true);

    /* ----------------------------------------------------------------------
     * The refresh and dice buttons, level with the control they belong to.
     *
     * style.css top-anchors them and pushes them down one label line, which is
     * as far as CSS can go here - because the size of that header is NOT the
     * same for every control gradio renders. Measured on 4.40, from the top of
     * the block to the top of its input box:
     *
     *     single-select Dropdown   27.6px   (label + its 8px margin-bottom)
     *     multiselect Dropdown     19.6px   (label alone - the margin is gone)
     *     Textbox                  19.6px
     *
     * The buttons stand beside a DIFFERENT one of those in each mode: the
     * multiselect model list in Model mode, the single-select LoRA picker in
     * LoRA mode. One constant is therefore wrong by 8px in one of the two,
     * whichever is chosen - so the offset is measured off the actual neighbour
     * instead, the same way the tool menus take their width from a measured
     * --cnpro-menu-w rather than from a percentage that looks right on paper.
     *
     * The stylesheet keeps a static one-label-line value as the no-JS case:
     * being 8px out beats being at the bottom of a three-line textbox.
     * ---------------------------------------------------------------------- */

    /* The labelled control this button sits after. Hidden siblings are skipped
       because they take no space in the row - in LoRA mode the model list is
       still in the DOM, just display:none, and it is not what the user sees the
       button next to. */
    function anchorFor(button) {
        for (let el = button.previousElementSibling; el; el = el.previousElementSibling) {
            if (!el.offsetParent) continue;
            /* .container > .wrap is the box a Dropdown draws, textarea/input is
               the box everything else draws. Listed in that order for reading;
               querySelector returns whichever comes first in the DOM, which for
               a dropdown is the .wrap - its <input> is nested inside it. */
            const input = el.querySelector(".container > .wrap, textarea, input");
            if (!input) continue;   // e.g. the other tool button
            return input.getBoundingClientRect().top - el.getBoundingClientRect().top;
        }
        return null;
    }

    function alignTools() {
        for (const button of document.querySelectorAll("button.cnpro-xy-tool")) {
            if (!button.offsetParent) continue;
            const offset = anchorFor(button);
            if (offset === null || offset < 0) continue;
            const want = Math.round(offset) + "px";
            /* Compared before writing: onUiUpdate runs on every gradio DOM
               batch, and an unconditional write would dirty layout each time.
               important, because the stylesheet's own value carries it too - it
               has to outrank gradio's `button { margin: 0 }` reset. */
            if (button.style.marginTop !== want) {
                button.style.setProperty("margin-top", want, "important");
            }
        }
    }

    if (typeof onUiUpdate === "function") {
        /* onUiUpdate rather than onUiLoaded alone: the row re-lays out whenever
           the axis target changes, which is exactly when the neighbour changes
           too. A1111's observer watches childList only, so writing a style
           attribute here cannot feed itself. */
        onUiUpdate(alignTools);
    }
    if (typeof onUiLoaded === "function") {
        onUiLoaded(() => {
            alignTools();
            window.addEventListener("resize", alignTools);
        });
    }
})();
