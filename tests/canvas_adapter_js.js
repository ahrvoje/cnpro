// Node harness for tests/test_canvas_adapter.py: does an image the HOST loads
// actually reach CNPro's wrapper?
//
// WHY THIS EXISTS
// ---------------
// Every canvas tool was inert on ForgeNeo -- crop, rotate, flip, layers, the
// pickers. The toolbar rendered, the menus opened, the sliders moved, and the
// image never changed. Nothing threw, nothing logged, and every existing test
// passed, because every existing test asked about the DOM and this was not a
// DOM problem.
//
// The cause was one line pointing the wrong way. ForgeNeo renamed
// `uploadBase64` to `loadImage`, so the adapter defined:
//
//     proto.uploadBase64 = function (b64) { return this.loadImage(b64); };
//
// which satisfies "the canonical method exists" and is useless, because
// canvas_extra.js works by WRAPPING that method and the host never calls it --
// the host calls `loadImage`, from the gradio bind, the file picker, paste and
// removeImage. So every inflow bypassed the wrapper, `st.original` stayed null,
// and `renderAdjusted()` returned on its first line, forever.
//
// The fix is directional: the canonical name takes the host's IMPLEMENTATION,
// and the host's name becomes a delegate that calls it. This harness pins that
// direction by simulating the real sequence:
//
//     normalize(proto)                 <- canvas_adapter.js
//     wrap proto.uploadBase64          <- what canvas_extra.js does
//     proto.loadImage(...)             <- what the HOST does
//
// and asserting the wrapper ran. A method-presence check cannot: presence was
// true the entire time the feature was dead.
//
// stdin:  (nothing)
// stdout: JSON, see the keys at the bottom.

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const EXTENSION = path.join(__dirname, '..');
const ADAPTER_JS = path.join(EXTENSION, 'javascript', 'canvas_adapter.js');

const out = {logs: []};
const sandbox = {
    window: {},
    console: {
        log: (...a) => out.logs.push('log: ' + a.join(' ')),
        warn: (...a) => out.logs.push('warn: ' + a.join(' ')),
        error: (...a) => out.logs.push('error: ' + a.join(' ')),
    },
};
sandbox.globalThis = sandbox;

try {
    vm.runInNewContext(fs.readFileSync(ADAPTER_JS, 'utf8'), sandbox, {filename: ADAPTER_JS});
} catch (exc) {
    out.loadError = String((exc && exc.stack) || exc);
    process.stdout.write(JSON.stringify(out));
    process.exit(0);
}

const api = sandbox.window.cnproCanvasApi;
if (!api || typeof api.normalize !== 'function') {
    out.fatal = 'canvas_adapter.js did not export window.cnproCanvasApi.normalize';
    process.stdout.write(JSON.stringify(out));
    process.exit(0);
}

// ---------------------------------------------------------------- the fixture
//
// A stand-in for ForgeNeo's ForgeCanvas.prototype: the renamed pair, plus the
// four methods that survived the rename. `loadImage` calls
// `updateBackgroundImageData` internally, exactly as the real one does at
// canvas.js:625 -- that inner call has to reach CNPro's wrapper too, because the
// crop tool's gradio suppression hangs off it.
function forgeNeoProto() {
    const trace = [];
    const proto = {
        trace: trace,
        loadImage(b64) {
            trace.push('host.loadImage(' + b64 + ')');
            this.updateBackgroundImageData();
        },
        updateBackgroundImageData() {
            trace.push('host.updateBackgroundImageData()');
        },
        removeImage() {},
        adjustInitialPositionAndScale() {},
        saveState() {},
        drawImage() {},
    };
    return proto;
}

// A host that already speaks CNPro's names: normalize must leave it completely
// alone, or an alias would wrap a method that needs no wrapping.
function classicForgeProto() {
    const trace = [];
    return {
        trace: trace,
        uploadBase64(b64) { trace.push('host.uploadBase64(' + b64 + ')'); },
        on_img_upload() { trace.push('host.on_img_upload()'); },
        removeImage() {},
        adjustInitialPositionAndScale() {},
        saveState() {},
        drawImage() {},
    };
}

// ------------------------------------------------- 1. ForgeNeo: the real case
{
    const proto = forgeNeoProto();
    const result = api.normalize(proto);
    out.neo = {ok: result.ok, missing: result.missing, aliased: result.aliased};

    // canvas_extra.js's patch, reproduced in miniature.
    const wrapped = [];
    const origUpload = proto.uploadBase64;
    proto.uploadBase64 = function (b64) {
        wrapped.push('cnpro.uploadBase64(' + b64 + ')');
        return origUpload.call(this, b64);
    };
    const origOnUpload = proto.on_img_upload;
    proto.on_img_upload = function () {
        wrapped.push('cnpro.on_img_upload()');
        return origOnUpload.call(this);
    };

    // THE HOST'S OWN CALL. This is the line the old adapter walked past.
    const fc = Object.create(proto);
    fc.loadImage('data:image/png;base64,AAAA');

    // SNAPSHOT, not the live array. Reporting the reference and then clearing it
    // for the next phase made phase 1 look empty and failed the whole suite for
    // a bug in the harness -- the sort of thing that gets a real failure
    // dismissed as "the test is broken".
    out.neo.wrapped = wrapped.slice();
    out.neo.trace = proto.trace.slice();

    // ...and CNPro's own call must still work, without recursing.
    wrapped.length = 0;
    proto.trace.length = 0;
    fc.uploadBase64('data:image/png;base64,BBBB');
    out.neo.wrappedFromCanonical = wrapped.slice();
    out.neo.traceFromCanonical = proto.trace.slice();
}

// --------------------------------------- 2. classic Forge: nothing to rewrite
{
    const proto = classicForgeProto();
    const before = proto.uploadBase64;
    const result = api.normalize(proto);
    out.classic = {
        ok: result.ok,
        aliased: result.aliased,
        untouched: proto.uploadBase64 === before,
    };
}

// ------------------------------------------------ 3. normalize is idempotent
//
// It runs once per page today, but "today" is not a guarantee, and the failure
// mode is a delegate wrapped around itself -- infinite recursion on the first
// image, i.e. a hung tab rather than a message.
{
    const proto = forgeNeoProto();
    api.normalize(proto);
    const afterFirst = proto.uploadBase64;
    api.normalize(proto);
    out.idempotent = {
        sameImplementation: proto.uploadBase64 === afterFirst,
        stillDelegates: proto.loadImage.cnproDelegatesTo === 'uploadBase64',
    };
    let recursed = false;
    try {
        Object.create(proto).loadImage('data:image/png;base64,CCCC');
    } catch (exc) {
        recursed = /Maximum call stack/i.test(String(exc && exc.message));
    }
    out.idempotent.recursed = recursed;
}

// --------------------------------- 4. an unsupported host must SAY it is one
{
    const result = api.normalize({drawImage() {}});
    out.unsupported = {ok: result.ok, missing: result.missing};
}

process.stdout.write(JSON.stringify(out));
