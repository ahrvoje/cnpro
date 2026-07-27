/* Prompt retention P/N mirror coupling.
 *
 * The retention slider is ONE logical knob shown on both prompt tabs: the P
 * tab holds the canonical component (the unit field the server reads), the N
 * tab a display mirror (elem id suffix `_n`, not a unit field). Coupling is
 * done HERE, client-side, instead of gradio .change/.input round-trips - the
 * server echo arrived mid-drag and pinned the mirror to the stale canonical
 * value, so dragging the N slider never took. Both inner inputs (range +
 * number) of the counterpart are written and re-dispatched through
 * updateInput(), so gradio's frontend state - and therefore the value the
 * unit collects at generation time - follows whichever tab was used.
 *
 * Recursion is settled by value equality: the echo write-back finds the
 * counterpart already at the target value and stops.
 */
(function () {
    "use strict";

    function couple(a, b) {
        const link = (src, dst) => {
            src.addEventListener("input", (e) => {
                if (!(e.target instanceof HTMLInputElement)) return;
                const value = e.target.value;
                dst.querySelectorAll("input").forEach((input) => {
                    if (input.value !== value) {
                        input.value = value;
                        updateInput(input);
                    }
                });
            });
        };
        link(a, b);
        link(b, a);
    }

    onUiLoaded(() => {
        document
            .querySelectorAll("[id$='_controlnet_unit_prompt_retention']")
            .forEach((canonical) => {
                const mirror = document.getElementById(canonical.id + "_n");
                if (mirror) couple(canonical, mirror);
            });
    });
})();
