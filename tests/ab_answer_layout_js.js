'use strict';

const fs = require('fs');
const path = require('path');

function loadPlaywright() {
    const root = process.env.CNPRO_TEST_NODE_PATH || '';
    try {
        return require(root ? path.join(root, 'node_modules', 'playwright')
                            : 'playwright');
    } catch (exc) {
        return {unavailable: 'playwright is not installed: ' + exc.message};
    }
}

(async () => {
    const loaded = loadPlaywright();
    if (loaded.unavailable) {
        process.stdout.write(JSON.stringify(loaded));
        return;
    }
    const url = process.argv[2];
    const css = fs.readFileSync(process.argv[3], 'utf8');
    const hostCssPath = process.argv[4] || '';
    const hostCss = hostCssPath ? fs.readFileSync(hostCssPath, 'utf8') : '';
    const browser = await loaded.chromium.launch({headless: true});
    const page = await browser.newPage({viewport: {width: 1200, height: 300}});
    const cases = [];

    await page.goto(url, {waitUntil: 'networkidle'});
    await page.waitForSelector('#cnpro_ab_layout_config textarea');
    // Forge appends extension CSS at the end of body.  Do the same; Blocks(css=)
    // parses and scopes the sheet, which is not the production cascade.
    await page.evaluate(([hostSheet, sheet]) => {
        if (hostSheet) {
            const hostStyle = document.createElement('style');
            hostStyle.dataset.cnproHostHarness = '1';
            hostStyle.textContent = hostSheet;
            document.body.appendChild(hostStyle);
        }
        const style = document.createElement('style');
        style.dataset.cnproHarness = '1';
        style.textContent = sheet;
        document.body.appendChild(style);
    }, [hostCss, css]);

    for (const width of [1149, 620, 380]) {
        await page.setViewportSize({width: width, height: 300});
        const measured = await page.evaluate((viewport) => {
            const configuration = document.getElementById(
                'cnpro_ab_layout_config');
            const answer = document.querySelector('.cnpro-ab-answer');
            const label = configuration.querySelector('label');
            const textarea = configuration.querySelector('textarea');
            const actions = document.querySelector('.cnpro-ab-set-actions');
            const configurationStyle = getComputedStyle(configuration);
            const configurationRect = configuration.getBoundingClientRect();
            const actionsRect = actions.getBoundingClientRect();
            const textareaRect = textarea.getBoundingClientRect();
            const buttons = [...actions.querySelectorAll('button')].map(
                (button) => {
                    const text = document.createRange();
                    text.selectNodeContents(button);
                    const lines = new Set([...text.getClientRects()].map(
                        (rect) => Math.round(rect.top))).size;
                    return {
                        label: button.textContent.trim(),
                        lineCount: lines,
                        whiteSpace: getComputedStyle(button).whiteSpace,
                    };
                });
            return {
                viewport: viewport,
                rowFlexWrap: getComputedStyle(answer).flexWrap,
                sameLine: Math.abs(configurationRect.top - actionsRect.top) < 2,
                configurationHeight: +configurationRect.height.toFixed(1),
                configurationContentHeight: +(configurationRect.height -
                    parseFloat(configurationStyle.paddingTop) -
                    parseFloat(configurationStyle.paddingBottom) -
                    parseFloat(configurationStyle.borderTopWidth) -
                    parseFloat(configurationStyle.borderBottomWidth)).toFixed(1),
                configurationInnerHeight: +label.getBoundingClientRect()
                    .height.toFixed(1),
                configurationFieldBottomGap: +(label.getBoundingClientRect().bottom -
                    textareaRect.bottom).toFixed(1),
                textareaHeight: +textareaRect.height.toFixed(1),
                actionsHeight: +actionsRect.height.toFixed(1),
                buttons: buttons,
            };
        }, width);
        measured.resized = await page.evaluate(() => {
            const configuration = document.getElementById(
                'cnpro_ab_layout_config');
            const textarea = configuration.querySelector('textarea');
            const actions = document.querySelector('.cnpro-ab-set-actions');
            textarea.style.height = (textarea.getBoundingClientRect().height +
                                     80) + 'px';
            const configurationRect = configuration.getBoundingClientRect();
            const actionsRect = actions.getBoundingClientRect();
            const textareaRect = textarea.getBoundingClientRect();
            const result = {
                sameLine: Math.abs(configurationRect.top - actionsRect.top) < 2,
                configurationHeight: +configurationRect.height.toFixed(1),
                actionsHeight: +actionsRect.height.toFixed(1),
                textareaHeight: +textareaRect.height.toFixed(1),
            };
            textarea.style.height = '';
            return result;
        });
        cases.push(measured);
    }

    const shot = process.env.CNPRO_AB_SHOT;
    if (shot) {
        const shotWidth = parseInt(process.env.CNPRO_AB_SHOT_WIDTH || '1149', 10);
        await page.setViewportSize({width: shotWidth, height: 300});
        await page.screenshot({path: shot, fullPage: true});
    }
    await browser.close();
    process.stdout.write(JSON.stringify({cases: cases, hostStyles: !!hostCss}));
})().catch((exc) => {
    process.stdout.write(JSON.stringify({fatal: String(exc && exc.stack || exc)}));
});
