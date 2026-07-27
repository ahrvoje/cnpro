"""The host's own image loads must reach CNPro's wrapper.

This is the test that would have caught the outage where every canvas tool did
nothing on ForgeNeo. See tests/canvas_adapter_js.js for the full story; the
short version is that `canvas_adapter.js` aliased CNPro's canonical name TO the
host's method instead of the other way round, so canvas_extra.js's wrapper sat
on a method the host never calls.

Presence checks could not see it -- `typeof proto.uploadBase64 === 'function'`
was true throughout. Only executing a host-shaped call and watching where it
lands can, which is what the node harness does.

Needs node. Exit code 0 = pass or skip; 1 = fail.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def main():
    if not shutil.which("node"):
        print("SKIPPED - node is not on PATH.")
        print("  The canvas adapter's call routing has NOT been verified.")
        return 0

    proc = subprocess.run(["node", os.path.join(HERE, "canvas_adapter_js.js")],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print("FAIL (1)")
        print("  - node harness failed:\n%s" % proc.stderr.strip())
        return 1
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        print("FAIL (1)")
        print("  - harness produced no JSON:\n%s\n%s"
              % (proc.stdout[:400], proc.stderr[:400]))
        return 1

    if data.get("loadError"):
        fail("canvas_adapter.js threw at load:\n%s" % data["loadError"])
    if data.get("fatal"):
        fail(data["fatal"])
    if FAILURES:
        return report()

    neo = data["neo"]

    # 1. the host must be recognised at all
    if not neo["ok"]:
        fail("normalize() reports ForgeNeo unsupported, missing %r" % neo["missing"])

    # 2. THE ONE THAT MATTERS. The host called loadImage(); CNPro's wrapper must
    #    have run. If this list is empty the adapter is pointing the wrong way
    #    and every adjustment tool is dead while looking perfectly healthy.
    if "cnpro.uploadBase64(data:image/png;base64,AAAA)" not in neo["wrapped"]:
        fail("the host's own loadImage() did NOT reach CNPro's uploadBase64 "
             "wrapper (wrapper calls seen: %r).\nThis is the exact failure that "
             "made crop, rotate, flip, layers and the pickers inert: st.original "
             "is never set, so renderAdjusted() returns immediately and no tool "
             "can change a pixel." % neo["wrapped"])

    # 3. the inner updateBackgroundImageData() call has to be wrapped too -- the
    #    crop tool suppresses gradio syncs from there, and an unwrapped one lets
    #    the uncropped image overwrite the cropped result.
    if "cnpro.on_img_upload()" not in neo["wrapped"]:
        fail("the host's internal updateBackgroundImageData() did not reach "
             "CNPro's on_img_upload wrapper (%r) - crop results will be "
             "overwritten in gradio by the crop-edit display" % neo["wrapped"])

    # 4. the host implementation must still actually run, once
    if neo["trace"].count("host.loadImage(data:image/png;base64,AAAA)") != 1:
        fail("the host's implementation ran %d times, expected exactly 1: %r"
             % (neo["trace"].count("host.loadImage(data:image/png;base64,AAAA)"),
                neo["trace"]))

    # 5. calling the canonical name directly must work identically
    if "cnpro.uploadBase64(data:image/png;base64,BBBB)" not in neo["wrappedFromCanonical"]:
        fail("calling uploadBase64() directly did not reach the wrapper: %r"
             % neo["wrappedFromCanonical"])

    # 6. a host that already speaks CNPro's names must be left alone
    classic = data["classic"]
    if classic["aliased"]:
        fail("normalize() rewrote a host that already has the canonical names: %r"
             % classic["aliased"])
    if not classic["untouched"]:
        fail("normalize() replaced uploadBase64 on a host that already had it")
    if not classic["ok"]:
        fail("normalize() rejected classic Forge, which it must support")

    # 7. idempotence. A delegate wrapped around itself is infinite recursion --
    #    a hung tab, not a message.
    idem = data["idempotent"]
    if not idem["sameImplementation"]:
        fail("a second normalize() replaced the implementation - the alias is "
             "being re-applied on top of itself")
    if not idem["stillDelegates"]:
        fail("after a second normalize() the host name no longer delegates to "
             "the canonical one")
    if idem["recursed"]:
        fail("normalize() run twice produces infinite recursion on the first "
             "image load")

    # 8. an unsupported host must be reported, not limped along
    if data["unsupported"]["ok"]:
        fail("normalize() called a host with no canvas API supported")
    if len(data["unsupported"]["missing"]) < 5:
        fail("an unsupported host was reported as missing only %r"
             % data["unsupported"]["missing"])

    return report(data)


def report(data=None):
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - host-originated loadImage() reaches CNPro's wrapper (%s), the "
          "inner updateBackgroundImageData() does too, classic Forge is left "
          "untouched, normalize is idempotent and an unsupported host is "
          "reported" % ", ".join(data["neo"]["aliased"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
