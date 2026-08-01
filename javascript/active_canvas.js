// Single source of the "which canvas is the open tab" rule (MAINTENANCE.md
// invariant 14): gradio hides the inactive tab panels with display:none, so
// nothing inside them is laid out and offsetParent is null. Every consumer -
// insert_image.js, image_info.js and the inline _js the python side generates
// for the below-canvas button row (controlnet_ui_group.forward_to_active_canvas)
// - resolves tab visibility through these helpers, so if gradio ever changes
// its hiding strategy (content-visibility, display:contents wrappers, ...)
// this file is the one place to fix.
//
// Loaded before its consumers: webui loads extension scripts in filename
// order and 'active_canvas.js' sorts first among these.
(function () {
    'use strict';

    window.cnetVisible = function (el) {
        return !!el && el.offsetParent !== null;
    };

    // The canvas container of the tab currently open in a ControlNet unit
    // (Input tabs and the Output mask tab), or null when NO canvas is laid
    // out. No first-container fallback: it was justified as "the unit is
    // collapsed", but these buttons are unreachable in a collapsed unit - the
    // only reachable nothing-visible state is a P/N prompt tab, where the
    // fallback silently pushed images into Input 1's HIDDEN canvas. Every
    // consumer already treats null as "no target".
    window.cnetActiveCanvasContainer = function (unit) {
        if (!unit) return null;
        const containers = unit.querySelectorAll(
            '.cnet-input-image-group [id^="imageContainer_"], .cnet-output-mask-group [id^="imageContainer_"]');
        for (const container of containers) {
            if (window.cnetVisible(container)) return container;
        }
        return null;
    };

    // The Input slots of a unit that a generation would actually USE, in slot
    // order: an Input tab holding a decoded image whose mute checkbox is on.
    //
    // ONE RULE, THREE READERS. The generation fans a unit out over
    // len(get_input_data(...)) inputs; weight_profile.js draws one wave per
    // Input from the same count (a preview counting something else is a preview
    // of a different run), and coverage_map.js needs the same set PLUS which
    // slot each one is, to find that input's weight-mask channels
    // (cnet-wmask-<slot>-<band>-state). Written out three times it would drift
    // the first time muting or a fourth Input changed anything.
    //
    // The mute state is the hidden gradio checkbox `_input_enabled_<n>` that
    // belongs to tab panel `_input_tab_<n>` (the same id mapping tab_marks.js
    // uses) - NOT the tab being open: a closed tab's image still runs.
    window.cnetLiveInputs = function (unit) {
        const out = [];
        if (!unit) return out;
        unit.querySelectorAll("[id*='_input_image'] img.forge-image").forEach((img) => {
            if (!img.src || img.naturalWidth <= 0) return;
            const panel = img.closest("[id*='_input_tab_']");
            let slot = 0;
            if (panel) {
                const state = document.getElementById(
                    panel.id.replace('_input_tab_', '_input_enabled_'));
                const check = state && state.querySelector("input[type='checkbox']");
                if (check && !check.checked) return;
                const match = /_input_tab_(\d+)$/.exec(panel.id);
                if (match) slot = parseInt(match[1], 10);
            }
            out.push({
                slot: slot,
                img: img,
                group: img.closest('.cnet-input-image-group'),
            });
        });
        return out;
    };

    // Click a toolbar button of whichever canvas is the OPEN tab. The
    // below-canvas button row (clear / load) is wired to this from python
    // (controlnet_ui_group.forward_to_active_canvas), which passes only the
    // canvas root ids and the button id prefix - the rule itself belongs
    // here, next to cnetVisible.
    window.cnetForwardToActiveCanvas = function (rootIds, buttonPrefix) {
        for (const id of rootIds || []) {
            const root = document.querySelector('#' + CSS.escape(id));
            if (!root || !window.cnetVisible(root)) continue;
            const button = root.querySelector('[id^="' + buttonPrefix + '"]');
            if (button) {
                button.click();
                return;
            }
        }
    };

    // ---- shared 500 ms tick
    //
    // Value-only textbox writes from the server produce no DOM mutation and no
    // event, so several modules here need a poll (MAINTENANCE invariant 18).
    // They share ONE timer instead of running three, and the whole tick is
    // skipped while no unit body is laid out: with the ControlNet accordion
    // closed - the common case - the old pollers still walked every unit of
    // both tabs twice a second forever. Nothing is lost by skipping: every
    // registered tick is idempotent and re-runs from onUiUpdate when the
    // accordion opens (opening it is itself a DOM mutation).
    const ticks = [];
    window.cnetRegisterTick = function (fn) {
        ticks.push(fn);
    };

    // The coverage panel counts as well as the unit bodies: it lives ABOVE the
    // units and is readable with every unit collapsed, which is the state
    // `.cnet-image-tabs` alone reports as "nothing on screen, skip the tick".
    // Its own poller would be a second timer for the same 500 ms.
    function anyUnitVisible() {
        for (const node of document.querySelectorAll('.cnet-image-tabs, .cnet-coverage-panel')) {
            if (window.cnetVisible(node)) return true;
        }
        return false;
    }

    setInterval(function () {
        if (!ticks.length || !anyUnitVisible()) return;
        for (const fn of ticks) {
            try {
                fn();
            } catch (err) {
                console.warn('[controlnet] tick failed', err);
            }
        }
    }, 500);
})();
