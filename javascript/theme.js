// LIGHT-THEME ADAPTATION - one boolean, published on <html>.
//
// Every colour in this extension was picked against Forge's DARK theme, and a
// few of them are WHITE: the main profile's line on the plot, and the main
// profile's selector bar under the presets. On the default LIGHT theme those
// are white on white - not "a bit low-contrast", GONE, and with them the
// ability to see what the main profile is doing at all. Same story for the
// yellow mid band (#fdd835 on white is a 1.4:1 contrast ratio, i.e. nothing)
// and for the step separators, which the plot already special-cased.
//
// The detection lands as `data-cnpro-theme="light" | "dark"` on the ROOT
// element, and everything else reads it from there:
//
//   * style.css overrides the affected variables under
//     `:root[data-cnpro-theme="light"]`, right below the block that declares
//     the dark ones - so a light variant is declared next to the value it
//     replaces and the two cannot drift apart;
//   * javascript/weight_profile.js paints the plot on a <canvas> and so
//     cannot be styled at all: it resolves those SAME variables through
//     getComputedStyle. One source of the colours for both, exactly as the
//     `--cnet-band-*` block already was.
//
// WHY THE ROOT ELEMENT AND NOT body. Gradio puts its own `dark` class on the
// element hosting the app (body when standalone), and a CSS variable declared
// on body is invisible to `getComputedStyle(document.documentElement)` - the
// plot would keep drawing dark-theme colours while the CSS around it had
// already switched. The attribute goes where both parties can see it.
//
// DARK IS THE DEFAULT AND STAYS BIT-IDENTICAL. Light is only ever entered on
// a POSITIVE detection - gradio's `dark` class absent AND the app's own
// background measured light. Everything ambiguous (no container yet, a
// transparent background, a colour that does not parse) is dark, which is
// what this extension already did before the attribute existed.
(function () {
    'use strict';

    const ATTR = 'data-cnpro-theme';
    const listeners = [];
    let current = 'dark';

    // Gradio marks dark mode with a `dark` class on the element that HOSTS the
    // app: document.body when standalone (the Forge case), the app element's
    // parent when embedded. It flips this class live - it listens to
    // prefers-color-scheme itself - so the class is authoritative wherever it
    // is present, and this is why the attribute cannot be computed once at
    // startup and forgotten.
    function hasDarkClass() {
        if (document.documentElement.classList.contains('dark')) return true;
        if (document.body && document.body.classList.contains('dark')) return true;
        const app = document.querySelector('gradio-app');
        return !!(app && app.parentElement && app.parentElement.classList.contains('dark'));
    }

    // Perceived brightness (0..255) of the first OPAQUE background at or above
    // the app container, or null when nothing up the chain paints one.
    //
    // Walking up is what makes this independent of WHICH element carries the
    // fill, and there are three candidates in a running Forge: gradio writes
    // `background: var(--body-background-fill)` onto the app element, the theme
    // writes one onto body, and A1111's ui_gradio_extensions injects a third
    // onto <html>. Reading `backgroundColor` off the computed style rather
    // than a theme variable also means the browser has already resolved
    // whatever `var()` / named colour / palette reference was written there.
    function backgroundBrightness() {
        let el = document.querySelector('.gradio-container') || document.body;
        while (el) {
            const m = getComputedStyle(el).backgroundColor.match(
                /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:[,/\s]+([\d.]+%?))?\s*\)/);
            if (m) {
                const raw = m[4];
                const alpha = raw === undefined ? 1
                    : (raw.endsWith('%') ? parseFloat(raw) / 100 : Number(raw));
                // a translucent fill is not what the user ends up looking at;
                // keep climbing until something actually covers the page
                if (alpha > 0.5) {
                    return 0.299 * Number(m[1]) + 0.587 * Number(m[2]) + 0.114 * Number(m[3]);
                }
            }
            el = el.parentElement;
        }
        return null;
    }

    function detect() {
        if (hasDarkClass()) return 'dark';
        const brightness = backgroundBrightness();
        return brightness !== null && brightness >= 128 ? 'light' : 'dark';
    }

    function apply() {
        const next = detect();
        if (next === current && document.documentElement.getAttribute(ATTR) === next) return;
        current = next;
        // setting the attribute mutates documentElement, which the observer
        // below watches - the early return above is what stops that from
        // becoming a loop
        document.documentElement.setAttribute(ATTR, next);
        for (const fn of listeners) {
            try {
                fn(next);
            } catch (err) {
                console.warn('[controlnet theme] listener failed', err);
            }
        }
    }

    window.cnproTheme = {
        isDark: function () {
            return current !== 'light';
        },
        // fn(theme) runs on every flip AND once on registration, so a consumer
        // never has to know whether it registered before or after the first
        // detection - which depends on script load order and is not worth
        // reasoning about at every call site.
        onChange: function (fn) {
            listeners.push(fn);
            try {
                fn(current);
            } catch (err) {
                console.warn('[controlnet theme] listener failed', err);
            }
        },
        refresh: apply,
    };

    function start() {
        apply();
        // `class` catches the theme toggle, `style` the inline background
        // gradio writes onto the app element - either one can be what changed.
        const observer = new MutationObserver(apply);
        for (const el of [document.documentElement, document.body]) {
            if (el) observer.observe(el, { attributes: true, attributeFilter: ['class', 'style'] });
        }
    }

    if (document.body) {
        start();
    } else {
        document.addEventListener('DOMContentLoaded', start, { once: true });
    }
    // the container the brightness probe wants does not exist until the UI is
    // built, and before it does detect() answers "dark" from the fallback
    if (typeof onUiLoaded === 'function') onUiLoaded(apply);
})();
