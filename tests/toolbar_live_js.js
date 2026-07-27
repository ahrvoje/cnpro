// Playwright harness for tests/test_toolbar_live.py: drive the RUNNING webui in
// a real Chromium and report what a user can actually see.
//
// WHY THIS EXISTS, AFTER THREE OTHER TOOLBAR TESTS ALREADY DID
// ------------------------------------------------------------
// Each earlier layer missed something the next one caught:
//
//   test_toolbar_contract.py   reads the JS as text  -> proved a contract existed,
//                                                       not that its ids were right
//   toolbar_contract_js.js     runs it on a DOM stub -> proved reveal/audit logic,
//                                                       on nodes the TEST created
//   test_toolbar_dom.py        jsdom + real template -> proved injection works,
//                                                       but loads NO CSS and has
//                                                       no layout engine
//
// Only a real browser has the cascade and a layout engine, and both matter here:
//
//   * `.forge-toolbar { opacity: 0 }` - the toolbar is invisible until hovered.
//     Nothing below a real browser can tell you that, and a screenshot taken
//     without hovering looks exactly like a broken toolbar.
//   * `.cnet-output-mask-group ... { display: none !important }` hides thirteen
//     controls on purpose. jsdom loaded no stylesheet, so it could not see this,
//     and the audit reported all thirteen as broken on every output-mask canvas.
//   * a control can have `display: inline-block` and still be 0x0 because its
//     container is collapsed. "Declared visible" and "has pixels" differ.
//
// Requires the webui to be running. Skips (loudly) when it is not - this is a
// diagnostic against a live system, not a hermetic unit test.
//
// stdout: JSON.

const {chromium} = require('playwright');

// Port 7860 is the USER'S. Agents must not attach to it, drive it, or restart
// it - see AGENTS.md section 0. The default here is an agent-owned instance on
// 7870, and 7860 is refused even if someone passes it explicitly, because the
// failure mode is invisible: the test PASSES against the user's session while
// quietly generating images in it and competing for their GPU.
const DEFAULT_URL = 'http://127.0.0.1:7870/';
const URL = process.env.CNPRO_URL || DEFAULT_URL;
const TIMEOUT = parseInt(process.env.CNPRO_TIMEOUT || '120000', 10);

(async () => {
    const out = {url: URL, console: []};
    if (/:7860(\/|$)/.test(URL)) {
        out.skip = 'refusing to use port 7860 - that port belongs to the user ' +
                   '(AGENTS.md section 0). Start your own instance on another ' +
                   'port, e.g. --port 7870, and set CNPRO_URL.';
        process.stdout.write(JSON.stringify(out));
        return;
    }
    let browser;
    try {
        browser = await chromium.launch();
    } catch (e) {
        out.skip = 'could not launch chromium: ' + (e && e.message);
        process.stdout.write(JSON.stringify(out));
        return;
    }
    const page = await browser.newPage({viewport: {width: 1700, height: 1300}});
    page.on('console', (m) => {
        const t = m.text();
        if (/cnpro/i.test(t)) out.console.push(m.type() + ': ' + t);
    });
    page.on('pageerror', (e) => out.console.push('pageerror: ' + (e && e.message)));

    try {
        await page.goto(URL, {waitUntil: 'domcontentloaded', timeout: TIMEOUT});
        await page.waitForSelector('#controlnet', {timeout: TIMEOUT});
    } catch (e) {
        out.skip = 'the webui is not reachable at ' + URL + ': ' + (e && e.message);
        await browser.close();
        process.stdout.write(JSON.stringify(out));
        return;
    }

    // open the ControlNet accordion so the canvas is laid out at all
    await page.evaluate(() => {
        const a = document.querySelector('#controlnet .label-wrap');
        if (a) a.click();
    });
    await page.waitForTimeout(5000);

    out.module = await page.evaluate(() => {
        const a = window.cnproCanvasNodes;
        if (!a) return {present: false};
        // Derived views are FUNCTIONS now, not arrays: the renderer loads before
        // the registry (filename order), so there is nothing to snapshot at load
        // time. Guarded, because a harness that throws here reports nothing at
        // all about a page that may be perfectly healthy.
        const ids = typeof a.toolbarIds === 'function' ? a.toolbarIds() : [];
        const def = typeof a.deferred === 'function' ? a.deferred() : {};
        return {
            present: true,
            registryLoaded: !!window.cnproCanvasTools,
            toolbarIds: ids,
            deferred: Object.keys(def),
            // the shipped-bug fingerprints, read off the LIVE objects
            idsTruncated: ids.some((id) => /Butto$/.test(id)),
            auditUsesComputedStyle: !!(a.audit && a.audit.toString().indexOf('isHidden') !== -1),
        };
    });

    const canvasId = await page.evaluate(() => {
        const c = Array.from(document.querySelectorAll('[id^="imageContainer_"]'))
            .find((x) => x.getBoundingClientRect().width > 50);
        return c ? c.id : null;
    });
    if (!canvasId) {
        out.skip = 'no laid-out CNPro canvas on the page (is the ControlNet accordion present?)';
        await browser.close();
        process.stdout.write(JSON.stringify(out));
        return;
    }
    out.canvasId = canvasId;

    // HOVER: the toolbar is opacity 0 until the pointer is over the canvas.
    await page.hover('#' + canvasId);
    await page.waitForTimeout(1200);   // the 0.3s opacity transition, generously

    out.toolbar = await page.evaluate((cid) => {
        const uuid = cid.replace('imageContainer_', '');
        const tb = document.getElementById('toolbar_' + uuid);
        const res = {
            opacity: tb ? getComputedStyle(tb).opacity : null,
            visible: [],       // has pixels on screen
            zeroSized: [],     // in the DOM, laid out, but 0x0
            missing: [],       // not in the DOM at all
            auditNow: [],
        };
        const api = window.cnproCanvasNodes;
        const ids = (api && typeof api.toolbarIds === 'function') ? api.toolbarIds() : [];
        ids.forEach((id) => {
            const n = document.getElementById(id + '_' + uuid);
            if (!n) { res.missing.push(id); return; }
            const r = n.getBoundingClientRect();
            const cs = getComputedStyle(n);
            const seen = cs.display !== 'none' && cs.visibility !== 'hidden' &&
                         r.width > 0 && r.height > 0;
            (seen ? res.visible : res.zeroSized).push(id);
        });
        if (api && api.audit) res.auditNow = api.audit(uuid);
        return res;
    }, canvasId);

    // Every output-mask canvas, where 13 controls are hidden BY DESIGN: the audit
    // must be quiet there, or it buries real failures under false ones.
    out.outputMask = await page.evaluate(() => {
        const api = window.cnproCanvasNodes;
        const groups = Array.from(document.querySelectorAll('.cnet-output-mask-group'));
        const res = {count: 0, auditProblems: 0, sample: []};
        groups.forEach((g) => {
            const c = g.querySelector('[id^="imageContainer_"]');
            if (!c) return;
            res.count++;
            const uuid = c.id.replace('imageContainer_', '');
            const p = api && api.audit ? api.audit(uuid) : [];
            res.auditProblems += p.length;
            if (p.length && res.sample.length < 3) res.sample.push({uuid: uuid.slice(0, 12), problems: p});
        });
        return res;
    });

    // The PYTHON-side defaults, as the running server actually rendered them.
    // JS is served from disk per request, so a JS fix is live after a browser
    // refresh - but scripts/ and lib_cnpro/ are imported once at startup. That
    // asymmetry cost a round: the toolbar JS was current while the Resize Mode
    // dropdown still showed a value whose source line no longer existed. Reading
    // these lets the test say "restart the server" instead of "the default is
    // wrong", which are different problems with different fixes.
    out.defaults = await page.evaluate(() => {
        const first = (sel, pick) => {
            const el = document.querySelector(sel);
            return el ? pick(el) : null;
        };
        return {
            resizeMode: first('[id*="controlnet_resize_mode_radio"]', (el) => {
                const i = el.querySelector('input');
                return i ? i.value : (el.textContent || '').trim().slice(0, 40);
            }),
            processorRes: first('[id*="preprocessor_resolution_slider"]', (el) => {
                const i = el.querySelector('input[type="number"], input[type="range"]');
                return i ? i.value : null;
            }),
        };
    });

    try {
        await page.locator('#' + canvasId.replace('imageContainer_', 'toolbar_')).screenshot(
            {path: process.env.CNPRO_SHOT || 'live_toolbar.png'});
    } catch (e) { out.screenshotError = String(e); }

    await browser.close();
    process.stdout.write(JSON.stringify(out));
})().catch((e) => {
    process.stdout.write(JSON.stringify({fatal: String(e && e.stack || e)}));
});
