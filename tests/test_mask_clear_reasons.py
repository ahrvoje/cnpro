"""Painted weight may only be destroyed for a DECLARED reason, and replacing the
image under it is not one.

THE BUG THIS EXISTS FOR
-----------------------
Insert a reference into a canvas (the below-canvas ⤵I / ⤵O buttons) and the
weight mask painted on it disappeared - silently, with no undo, and on the
Output-mask canvas that is the ordinary order of work: insert the result you
want to steer, THEN paint where the control may act.

It was fixed once and stayed broken, which is the interesting part. THREE places
in `weight_mask.js` independently answered "the image changed - does the paint
die?":

  1. the 500 ms dims watchdog       (image replaced at a different size)
  2. the `forge-image-info` handler (image replaced at the SAME size, detected
                                     through the upload-sequence counter)
  3. the same handler's empty branch (no image at all - a real clear)

The fix taught (1) to carry the paint and never touched (2), two hundred lines
away, which went on calling `clearMask(slot, true)` for every genuine new image -
and `forgeCanvasPush`, the insert path, is genuine new content by definition. The
diff looked completely correct. The mask still died.

That is MAINTENANCE invariant 29's shape exactly - declared in one place,
honoured in another - so the fix is not another patched branch. Destroying paint
now requires NAMING a reason from `CLEAR_REASONS`, every image-change site is
funnelled through `onImageReplaced`, and this file reads both out of the source
so the two cannot drift.

WHAT IS PINNED HERE
-------------------
1. Every `clearMask` call names a reason, and every named reason is declared.
2. Every declared reason is actually used - a dead entry is a licence nobody
   revoked.
3. NO declared reason describes an image changing. This is the contract itself:
   the list is the only door, and this check is the lock on it.
4. The `forge-image-info` handler contains no `clearMask` outside its
   no-image-at-all branch. That is bug (2), by name.
5. The dims watchdog does not clear either - it calls the funnel.
6. `onImageReplaced` may reach exactly one clear, the inconsistent-slot repair.
7. The carry mechanism (`rescaleMask`) still exists and is still wired to the
   funnel.

Needs nothing. Exit code 0 = pass.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
SOURCE = os.path.join(EXTENSION, "javascript", "weight_mask.js")

# Words that describe the picture underneath changing rather than the mask
# ending. A reason matching any of them would re-open the exact hole this file
# was written to close, so it is refused by NAME - a future author cannot smuggle
# the old behaviour back in as 'image-replaced' or 'new-upload'.
FORBIDDEN_IN_REASON = (
    "replac", "insert", "upload", "new-image", "newimage", "swap",
    "dims", "dimension", "resize", "geometry", "push", "drop", "paste",
)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def strip_comments(js):
    """Block and line comments out, string literals preserved.

    Written character by character rather than with a regex because this file's
    whole job is reading call sites out of source: a regex that eats a `//`
    inside a string would silently drop the code after it, and dropping code is
    how a checker reports "no problems found" about a file it never read.
    """
    out = []
    i = 0
    n = len(js)
    quote = None
    while i < n:
        ch = js[i]
        if quote:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(js[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and js[i + 1] == "/":
            while i < n and js[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and js[i + 1] == "*":
            i += 2
            while i + 1 < n and not (js[i] == "*" and js[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def block_of(code, header):
    """The braced body of `header`, by brace counting from its first `{`.

    Not a regex: these bodies contain nested braces, object literals and
    functions, and `\\{([^}]*)\\}` stops at the first inner `}` - which here
    would mean checking the first three lines of a handler and declaring the
    rest clean.
    """
    start = code.find(header)
    if start < 0:
        return None
    brace = code.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for i in range(brace, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[brace:i + 1]
    return None


def main():
    if not os.path.exists(SOURCE):
        print("FAIL - %s is missing" % SOURCE)
        return 1
    code = strip_comments(read(SOURCE))

    # ---- 1. the declaration
    decl = block_of(code, "const CLEAR_REASONS")
    if not decl:
        fail("CLEAR_REASONS is gone from weight_mask.js. It is the only list of "
             "ways a painted mask may legitimately end; without it every call "
             "site is back to deciding for itself, which is how inserting an "
             "image came to destroy the mask painted on it - twice.")
        return report()
    declared = set(re.findall(r"'([a-z0-9-]+)'\s*:", decl))
    if not declared:
        fail("CLEAR_REASONS parses as empty - the reason check below would pass "
             "vacuously, which is worse than not running it:\n%s" % decl[:300])
        return report()

    # ---- 2. no reason may describe the image changing. THE CONTRACT.
    for reason in sorted(declared):
        for word in FORBIDDEN_IN_REASON:
            if word in reason:
                fail("CLEAR_REASONS declares %r, which names an IMAGE CHANGE "
                     "(matched %r). A new picture under the paint is not a "
                     "reason to destroy the paint - the mask is registered with "
                     "the frame, and python resizes it onto the generation "
                     "dimensions anyway. Route that site through "
                     "onImageReplaced, which carries the mask instead."
                     % (reason, word))

    # ---- 3. every call names a declared reason
    calls = re.findall(r"clearMask\(([^;]*?)\)\s*;", code)
    if len(calls) < 4:
        fail("only %d clearMask call(s) found - this checker has stopped "
             "reading the file it is supposed to be checking (the parser or the "
             "call shape changed)" % len(calls))
    used = set()
    for args in calls:
        parts = [p.strip() for p in args.split(",")]
        if len(parts) != 3:
            fail("clearMask(%s) does not pass a reason. Destroying painted "
                 "weight requires naming one of %s - see CLEAR_REASONS for why "
                 "this is enforced rather than documented."
                 % (args.strip()[:60], sorted(declared)))
            continue
        literal = re.fullmatch(r"'([a-z0-9-]+)'", parts[2])
        if not literal:
            fail("clearMask(%s) passes a computed reason. It must be a literal, "
                 "or neither this check nor a reader can tell what the call is "
                 "claiming." % args.strip()[:60])
            continue
        reason = literal.group(1)
        used.add(reason)
        if reason not in declared:
            fail("clearMask(...) names the undeclared reason %r. Declared: %s"
                 % (reason, sorted(declared)))

    # ---- 4. no dead licences
    for reason in sorted(declared - used):
        fail("CLEAR_REASONS declares %r and nothing uses it. A standing licence "
             "to destroy paint that no call site needs is one a future edit will "
             "reach for - drop it." % reason)

    # ---- 5. THE BUG, BY NAME: the image-info handler may only clear when there
    #        is NO IMAGE AT ALL.
    handler = block_of(code, "container.addEventListener('forge-image-info'")
    if handler is None:
        fail("the forge-image-info handler is gone from the painter - the canvas "
             "going empty (clear button, tab close, unit reset) is the one image "
             "event that must still end a mask")
    else:
        for args in re.findall(r"clearMask\(([^;]*?)\)\s*;", handler):
            reason = args.split(",")[-1].strip().strip("'")
            if reason != "canvas-emptied":
                fail("the forge-image-info handler clears with reason %r. THIS "
                     "IS THE ORIGINAL BUG: that handler fires for every genuine "
                     "new image - which is exactly what the ⤵I / ⤵O insert "
                     "buttons produce - and clearing there destroyed the mask "
                     "the user had just painted. Only 'canvas-emptied' (no "
                     "image at all) may clear from here; a replacement belongs "
                     "in onImageReplaced." % reason)
        if "onImageReplaced" not in handler:
            fail("the forge-image-info handler no longer carries the mask "
                 "through onImageReplaced. Silence there is not neutral: a "
                 "replaced image leaves the painter's mask sized for the "
                 "previous geometry until the next watchdog tick.")

    # ---- 6. the funnel itself
    funnel = block_of(code, "function onImageReplaced")
    if funnel is None:
        fail("onImageReplaced is gone. It is the single funnel every "
             "image-change site goes through; without it those sites are back "
             "to deciding independently, which is the failure mode this whole "
             "file exists for.")
    else:
        reasons = [a.split(",")[-1].strip().strip("'")
                   for a in re.findall(r"clearMask\(([^;]*?)\)\s*;", funnel)]
        for reason in reasons:
            if reason != "nothing-to-carry":
                fail("onImageReplaced clears with reason %r. The funnel may only "
                     "repair a slot that claims paint with no canvas behind it; "
                     "anything else means a replaced image can destroy a mask "
                     "again." % reason)
        if "rescaleMask" not in funnel:
            fail("onImageReplaced no longer calls rescaleMask - nothing carries "
                 "the paint onto the new geometry, so a resized insert loses it "
                 "even though no clear is visible in the diff")

    # ---- 7. the watchdog delegates rather than deciding
    watchdog = block_of(code, "function watchSlots")
    if watchdog is None:
        fail("watchSlots is gone - nothing notices an image replaced at a "
             "different size")
    else:
        if "onImageReplaced" not in watchdog:
            fail("the dims watchdog no longer routes through onImageReplaced. "
                 "It is the second of the three sites that used to answer 'does "
                 "the paint die?' on its own, and the first one to be patched "
                 "while the others stayed broken.")
        for args in re.findall(r"clearMask\(([^;]*?)\)\s*;", watchdog):
            reason = args.split(",")[-1].strip().strip("'")
            if reason != "server-cleared":
                fail("the dims watchdog clears with reason %r. Its only "
                     "legitimate clear is the server-cleared channel; a "
                     "replacement must go through onImageReplaced." % reason)

    # ---- 8. the refusal is enforced at runtime, not just here.
    #
    #        The MEMBERSHIP TEST is what is required, not the token: the guard's
    #        own diagnostic names CLEAR_REASONS in a string, so `if (false)`
    #        around it left the word in the body and this check passed on a
    #        disarmed guard (caught by mutating it - a checker that cannot fail
    #        is the thing this suite is about).
    guard = block_of(code, "function clearMask")
    if guard is None:
        fail("clearMask is gone from weight_mask.js")
    elif not re.search(r"CLEAR_REASONS\s*,\s*reason\s*\)", guard):
        fail("clearMask no longer TESTS its reason against CLEAR_REASONS "
             "(expected a `hasOwnProperty.call(CLEAR_REASONS, reason)` guard). "
             "This file is a static check and cannot see a reason built at "
             "runtime; the guard is what makes the list real in the browser.")

    return report()


def report():
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - painted weight can only be destroyed for a declared reason, no "
          "declared reason is an image change, and every image-change site is "
          "funnelled through onImageReplaced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
