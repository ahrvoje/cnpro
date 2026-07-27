// Red border on the canvas tab labels of a ControlNet unit whose canvas
// actually holds an image. A unit can carry several Input tabs plus the Output
// mask tab and only one of them is laid out at a time, so without a mark on
// the strip there is no way to tell which of the hidden tabs still contain
// something that will be preprocessed on Generate.
//
// Tab button ids come from gradio: a tab rendered with elem_id=X gets its
// strip button as "X-button", so the panel -> label mapping is exact and does
// not depend on the order of the (partly hidden) buttons in the strip.
//
// "Holds an image" is read from the canvas adjustment layer's announcement
// (modules_forge/forge_canvas/canvas_extra.js keeps dataset.forgeImageInfo:
// JSON while an image is loaded, "" once it is removed), with the displayed
// <img> as the fallback for canvases populated before that layer attached.
(function () {
    'use strict';

    function isFilled(group) {
        const container = group.querySelector('[id^="imageContainer_"]');
        if (container) {
            const stored = container.dataset.forgeImageInfo;
            if (stored) return true;
            if (stored === '') return false; // announced empty
        }
        const img = group.querySelector('.forge-image');
        return !!(img && img.naturalWidth > 0);
    }

    // Visual order of the Input tabs, mirrored from the hidden input_order
    // channel (X_input_order, slot digits e.g. "10234", "" = natural; written
    // by the move-left button / "+" handlers server-side). Applied as CSS
    // order on the strip buttons - the DOM order never changes, so the
    // positional right-align selectors in style.css and gradio's own keyed
    // rendering stay untouched; non-input strip buttons sit at a higher
    // static order (style.css) and keep trailing. The parse is the same
    // fail-open rule as python's input_order_permutation: valid unique
    // digits first, missing slots appended in natural order.
    //
    // Slots are enumerated from the tab PANELS, never from the strip buttons:
    // gradio renders NO button for a closed (hidden) tab, so walking buttons
    // stops at the first closed slot and every slot behind it drops out of the
    // permutation - it then keeps whatever order it was last given, or none at
    // all when gradio re-created the element (selecting a tab swaps it for a
    // different node), which parks it behind + / Output mask / P / N via the
    // static order in style.css. The panels exist for every slot, open or
    // closed, so both sides model the same full permutation and project the
    // open slots identically - that is what makes closing a tab leave the
    // surviving ones exactly where they were, and keeps the strip agreeing
    // with the order get_input_data feeds the backend in.
    function applyStripOrder(app) {
        app.querySelectorAll('.cnet-input-order-state').forEach((block) => {
            const textarea = block.querySelector('textarea');
            if (!textarea || !block.id.endsWith('_input_order')) return;
            const prefix = block.id.slice(0, -'_input_order'.length);
            const buttons = [];  // one entry per slot; null while the tab is closed
            for (let slot = 0; ; slot++) {
                if (!app.querySelector('#' + CSS.escape(prefix + '_input_tab_' + slot))) break;
                buttons.push(app.querySelector(
                    '#' + CSS.escape(prefix + '_input_tab_' + slot + '-button')));
            }
            if (!buttons.length) return;
            const perm = [];
            for (const ch of textarea.value) {
                const slot = ch.charCodeAt(0) - 48;
                if (slot >= 0 && slot < buttons.length && !perm.includes(slot)) perm.push(slot);
            }
            for (let slot = 0; slot < buttons.length; slot++) {
                if (!perm.includes(slot)) perm.push(slot);
            }
            perm.forEach((slot, pos) => {
                const button = buttons[slot];
                if (!button) return;  // closed slot: no strip button to place
                const order = String(pos + 1);
                // style writes only when the value really moves: this runs
                // from onUiUpdate and an unconditional write is itself a
                // mutation feeding an endless update loop
                if (button.style.order !== order) button.style.order = order;
            });
        });
    }

    function update() {
        const app = gradioApp();
        applyStripOrder(app);
        app.querySelectorAll('.cnet-input-image-group, .cnet-output-mask-group').forEach((group) => {
            const panel = group.closest('.tabitem');
            if (!panel || !panel.id) return;
            const button = app.querySelector('#' + CSS.escape(panel.id + '-button'));
            if (!button) return;
            // this runs from onUiUpdate (every DOM mutation): only write when
            // the state really changed, an unconditional class write is itself
            // a mutation and would feed an endless update loop
            const filled = isFilled(group);
            if (button.classList.contains('cnet-tab-filled') !== filled) {
                button.classList.toggle('cnet-tab-filled', filled);
            }
            // Per-slot mute lives IN the tab title: a native checkbox before
            // the label text, injected here because gradio cannot render
            // components inside the tab strip. The gradio checkbox it mirrors
            // is a hidden state channel; the ids share their suffix:
            // X_input_tab_<n> <-> X_input_enabled_<n>. Since every open Input
            // tab shows its own checkbox, any slot can be muted straight from
            // the strip, without switching to it first.
            let enabledInput = null;
            if (panel.id.includes('_input_tab_')) {
                const stateSel = '#' + CSS.escape(panel.id.replace('_input_tab_', '_input_enabled_'))
                    + ' input[type="checkbox"]';
                enabledInput = app.querySelector(stateSel);
                let mark = button.querySelector('.cnet-tab-mute');
                if (!mark && enabledInput) {
                    // injected once (guarded - this runs from onUiUpdate and
                    // an unconditional insert would feed a mutation loop);
                    // gradio re-renders of the strip wipe it, the next update
                    // pass re-injects and re-syncs from the state channel
                    mark = document.createElement('input');
                    mark.type = 'checkbox';
                    mark.className = 'cnet-tab-mute';
                    mark.title = 'Use this input: unchecking mutes it '
                        + '(image and masks are kept but skipped at generation)';
                    mark.addEventListener('click', function (e) {
                        // toggle the slot WITHOUT selecting the tab: the click
                        // must not bubble to gradio's tab button handler
                        e.stopPropagation();
                        const state = gradioApp().querySelector(stateSel);
                        if (state) state.click();
                    });
                    button.insertBefore(mark, button.firstChild);
                }
                // the state channel is authoritative; property write only when
                // it differs (checked is a property, not a DOM mutation, but
                // stay consistent with the write-guard discipline here)
                if (mark && enabledInput && mark.checked !== enabledInput.checked) {
                    mark.checked = enabledInput.checked;
                }
            }
            const muted = !!enabledInput && !enabledInput.checked;
            if (button.classList.contains('cnet-tab-muted') !== muted) {
                button.classList.toggle('cnet-tab-muted', muted);
            }
        });
        // P / N prompt tabs: same filled marker when their textbox holds
        // text - the prompts are otherwise invisible from the strip, exactly
        // like the images behind the other tabs
        app.querySelectorAll('.controlnet_unit_prompt textarea').forEach((textarea) => {
            const panel = textarea.closest('.tabitem');
            if (!panel || !panel.id) return;
            const button = app.querySelector('#' + CSS.escape(panel.id + '-button'));
            if (!button) return;
            const filled = textarea.value.trim() !== '';
            if (button.classList.contains('cnet-tab-filled') !== filled) {
                button.classList.toggle('cnet-tab-filled', filled);
            }
        });
    }

    document.addEventListener('forge-image-info', update);
    // checkbox flips re-render little; make sure the strip reacts immediately
    document.addEventListener('change', function (e) {
        const t = e.target;
        if (t && t.closest && t.closest('.cnet-input-enabled')) update();
    });
    // typing in a prompt tab: keep its filled mark live (value edits are not
    // DOM mutations, so onUiUpdate alone would lag until the next re-render)
    document.addEventListener('input', function (e) {
        const t = e.target;
        if (t && t.closest && t.closest('.controlnet_unit_prompt')) update();
    });
    // src swaps on a canvas are not childList mutations, so onUiUpdate alone
    // can miss a freshly decoded image
    document.addEventListener('load', function (e) {
        const t = e.target;
        if (t && t.classList && t.classList.contains('forge-image')) update();
    }, true);

    onUiUpdate(update);

    // the input_order channel is written server-side (move-left / "+"): a
    // value-only textarea write produces no DOM mutation and no event, so a
    // watch tick is the guaranteed channel (invariant 18 pattern; update()
    // is fully write-guarded and idle ticks mutate nothing). The timer is
    // shared by every module here and skipped while no unit is laid out
    // (active_canvas.js).
    window.cnetRegisterTick(update);
})();
