// Playwright harness for tests/test_theme_live.py: load the RUNNING webui once
// per theme and report what the profile editor is actually painted in.
//
// WHY IT HAS TO BE PIXELS
// -----------------------
// The bug this guards was "the main profile line is white on a white page".
// Every cheaper check passes on that: the element exists, the canvas is the
// right size, `draw()` ran, the colour is a valid CSS colour. The only question
// that fails is "is what was painted different from what it was painted on",
// and answering it needs a canvas read-back plus the theme's real background.
//
// It also guards the opposite direction, which is the easier mistake to make:
// the DARK theme must not move. That is asserted against the values the app was
// MEASURED to paint before light support existed, not against the values the
// source appeared to select - the two differed, see test_theme_live.py.
//
// Themes are driven through ?__theme=, gradio's own switch, so the `dark` class
// is set by gradio exactly as it is for a user.
//
// Port 7860 is the user's and is refused (AGENTS.md section 0).
//
// stdout: JSON.

const {chromium} = require('playwright');

const DEFAULT_URL = 'http://127.0.0.1:7870/';
const URL = process.env.CNPRO_URL || DEFAULT_URL;
const TIMEOUT = parseInt(process.env.CNPRO_TIMEOUT || '180000', 10);

// WCAG relative luminance, so the numbers here mean the same thing as the
// ratios written into style.css's comments.
function lum(c) {
    const f = (v) => {
        v /= 255;
        return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
}
function contrast(a, b) {
    const x = Math.max(lum(a), lum(b));
    const y = Math.min(lum(a), lum(b));
    return (x + 0.05) / (y + 0.05);
}
function toRgb(value) {
    if (!value) return null;
    const hex = value.trim().match(/^#([0-9a-f]{6})$/i);
    if (hex) {
        return [0, 2, 4].map((i) => parseInt(hex[1].substr(i, 2), 16));
    }
    const m = value.match(/[\d.]+/g);
    return m && m.length >= 3 ? m.slice(0, 3).map(Number) : null;
}

async function measure(browser, theme, out) {
    const ctx = await browser.newContext({viewport: {width: 1700, height: 1400}});
    const page = await ctx.newPage();
    page.on('pageerror', (e) => out.console.push('pageerror: ' + (e && e.message)));
    page.on('console', (m) => {
        if (m.type() === 'error') out.console.push('error: ' + m.text());
    });

    await page.goto(URL + '?__theme=' + theme, {waitUntil: 'domcontentloaded', timeout: TIMEOUT});
    await page.waitForSelector('#controlnet', {timeout: TIMEOUT});
    // the unit body is not laid out until the accordion is open, and a canvas
    // with no layout draws nothing at all (draw() early-returns at zero size)
    await page.evaluate(() => {
        const a = document.querySelector('#controlnet .label-wrap');
        if (a) a.click();
    });
    await page.waitForTimeout(6000);

    const data = await page.evaluate(() => {
        const root = getComputedStyle(document.documentElement);
        const canvas = document.querySelector('.cnet-weight-profile-canvas');
        const editor = canvas && canvas.closest('.cnet-weight-profile');
        // The surface the plot is painted ON. NOT document.body: gradio leaves
        // that white on both themes and paints the real fill onto the app
        // element inside it - the trap this whole test exists because of.
        let panel = null;
        for (let el = editor; el; el = el.parentElement) {
            const c = getComputedStyle(el).backgroundColor;
            const m = c.match(/^rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/);
            if (m && (m[4] === undefined || Number(m[4]) > 0.5)) { panel = c; break; }
        }
        // every fully opaque colour painted on the plot, most-used first
        let top = [];
        if (canvas && canvas.width) {
            const img = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
            const counts = new Map();
            for (let i = 0; i < img.length; i += 4) {
                if (img[i + 3] < 250) continue;
                const k = img[i] + ',' + img[i + 1] + ',' + img[i + 2];
                counts.set(k, (counts.get(k) || 0) + 1);
            }
            top = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 8);
        }
        return {
            bodyDark: document.body.classList.contains('dark'),
            bodyBackground: getComputedStyle(document.body).backgroundColor,
            attr: document.documentElement.getAttribute('data-cnpro-theme'),
            mainLine: root.getPropertyValue('--cnet-main-line').trim(),
            bandMid: root.getPropertyValue('--cnet-band-mid').trim(),
            bandCoarse: root.getPropertyValue('--cnet-band-coarse').trim(),
            bandFine: root.getPropertyValue('--cnet-band-fine').trim(),
            stepDot: root.getPropertyValue('--cnet-plot-step-dot').trim(),
            panel: panel,
            canvasSize: canvas ? [canvas.width, canvas.height] : null,
            topColors: top,
            hasEditor: !!editor,
        };
    });

    // The main SELECTOR bar, read as a pixel. Its colour arrives through an
    // inline `--band-color: var(--cnet-main-line)` written by python, which is a
    // different path from the plot's and has broken independently before.
    // Element screenshot, not a page clip: the page is taller than the viewport.
    const barEl = await page.$('.cnet-profile-band[data-band="main"]');
    if (barEl) {
        const shot = await barEl.screenshot();
        data.barPixel = await page.evaluate(async (b64) => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + b64;
            await img.decode();
            const cv = document.createElement('canvas');
            cv.width = img.width;
            cv.height = img.height;
            const g = cv.getContext('2d');
            g.drawImage(img, 0, 0);
            const d = g.getImageData(Math.floor(img.width / 2), Math.floor(img.height / 2), 1, 1).data;
            return [d[0], d[1], d[2]];
        }, shot.toString('base64'));
    }

    if (process.env.CNPRO_SHOT_DIR) {
        const ed = await page.$('.cnet-weight-profile');
        if (ed) await ed.screenshot({path: process.env.CNPRO_SHOT_DIR + '/theme_' + theme + '.png'});
    }
    await ctx.close();
    return data;
}

(async () => {
    const out = {url: URL, console: [], themes: {}};
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
    try {
        for (const theme of ['dark', 'light']) {
            out.themes[theme] = await measure(browser, theme, out);
        }
    } catch (e) {
        if (!Object.keys(out.themes).length) {
            out.skip = 'the webui is not reachable at ' + URL + ': ' + (e && e.message);
        } else {
            out.fatal = String(e && e.message);
        }
        await browser.close();
        process.stdout.write(JSON.stringify(out));
        return;
    }
    await browser.close();

    // derived measurements, computed here so the python side stays declarative
    for (const theme of Object.keys(out.themes)) {
        const d = out.themes[theme];
        const panel = toRgb(d.panel) || [0, 0, 0];
        const main = toRgb(d.mainLine);
        d.panelRgb = panel;
        d.mainRgb = main;
        d.mainOnPanel = main ? contrast(main, panel) : null;
        d.barOnPanel = d.barPixel ? contrast(d.barPixel, panel) : null;
        d.dotOnMain = (main && toRgb(d.stepDot)) ? contrast(toRgb(d.stepDot), main) : null;
        d.midOnPanel = toRgb(d.bandMid) ? contrast(toRgb(d.bandMid), panel) : null;
        d.coarseOnPanel = toRgb(d.bandCoarse) ? contrast(toRgb(d.bandCoarse), panel) : null;
        d.fineOnPanel = toRgb(d.bandFine) ? contrast(toRgb(d.bandFine), panel) : null;
        // is the main line's colour ACTUALLY on the canvas? (tolerance for the
        // canvas's own antialiasing at the curve edges)
        d.mainPixelCount = 0;
        if (main) {
            for (const [key, n] of d.topColors) {
                const v = key.split(',').map(Number);
                if (v.every((x, i) => Math.abs(x - main[i]) <= 6)) d.mainPixelCount += n;
            }
        }
    }
    process.stdout.write(JSON.stringify(out));
})().catch((e) => {
    process.stdout.write(JSON.stringify({fatal: String(e && e.stack || e)}));
});
