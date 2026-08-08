"""style.css must parse, and must actually cover the chrome it is paired with.

THE BUG THIS EXISTS FOR
-----------------------
The canvas tool-chrome section of style.css was produced by a LINE-level diff of
the author's fork of `modules_forge/forge_canvas/canvas.css` against the host's
copy: keep the lines the host does not have, drop the rest. That drops SELECTORS
and CLOSING BRACES whenever a rule is partly shared, and it left the file like
this:

        display: inline-flex;      <- body of `.forge-toolbar .forge-btn`, whose
        align-items: center;          selector line was "shared" and deleted
        height: 22px;
    .forge-toolbar-box-c {

Two rules lost their selector, one lost its opening, a stray `}` was left behind
by a dropped `.forge-upload-hint`, and the file ended with 198 `{` against 199
`}`. Browsers do not report this; they discard from the first orphan to the next
thing that parses and carry on. What the user saw was every tool menu rendering
as unstyled stacked blocks -- because `.forge-toolbar-box-c { display: flex }`
was inside the discarded span and never applied.

No test could have caught it, because no test read the stylesheet at all.

WHAT IS PINNED HERE
-------------------
1. The braces balance, no comment is left open or closed twice, and no
   declaration sits outside a rule.
2. Every class the tool registry renders is styled somewhere.
3. The geometry rules stay scoped to CNPro's own canvases, so enabling the
   extension cannot restyle the host's img2img canvas.

Needs nothing. Exit code 0 = pass.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)

FAILURES = []
SKIPS = []


def fail(msg):
    FAILURES.append(msg)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def strip_comments(css):
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def main():
    css_path = os.path.join(EXTENSION, "style.css")
    css = strip_comments(read(css_path))

    # 1. braces balance
    opens, closes = css.count("{"), css.count("}")
    if opens != closes:
        fail("style.css has %d '{' and %d '}' - it is malformed, and a browser "
             "will silently discard everything from the imbalance to the next "
             "rule it can parse" % (opens, closes))

    # 1b. no stray comment delimiter. A `*/` still standing after the comments
    #     have been stripped is a comment that was closed EARLY, and the prose
    #     after it is not a comment at all - it is a selector as far as the
    #     parser is concerned, and it swallows the next real rule as its
    #     block. Two of these sat in the A/B section (the row-label wrap's
    #     comment, closed twice mid-paragraph) and cost that rule entirely:
    #     braces still balanced, no orphaned declaration, nothing to see.
    #     Counted on the RAW file for the opposite failure - a comment never
    #     closed eats every rule after it.
    raw = read(css_path)
    if raw.count("/*") != raw.count("*/"):
        fail("style.css has %d '/*' against %d '*/' - a comment is unclosed, "
             "and everything from it to the next '*/' (or the end of the "
             "file) is discarded" % (raw.count("/*"), raw.count("*/")))
    for match in re.finditer(r"\*/", css):
        line_no = css.count("\n", 0, match.start()) + 1
        line = css.split("\n")[line_no - 1].strip()
        fail("style.css line %d closes a comment that is not open: %r. The "
             "text before it reads as a selector, and the next rule's block "
             "becomes its body - the rule is silently lost."
             % (line_no, line[:60]))

    # 2. no declaration outside a rule. This is what an orphaned rule body looks
    #    like, and it is the exact shape the line-diff produced.
    depth = 0
    line_no = 0
    for raw_line in css.split("\n"):
        line_no += 1
        line = raw_line.strip()
        before = depth
        depth += raw_line.count("{") - raw_line.count("}")
        if before <= 0 and depth <= 0 and line:
            # at top level: only selectors, at-rules and blank lines are legal.
            # A `prop: value;` here is an orphaned declaration.
            if re.match(r"^[a-z-]+\s*:\s*[^;]+;\s*$", line) and not line.startswith("--"):
                fail("style.css line %d is a declaration at top level, outside any "
                     "rule: %r. Its selector was dropped - everything from here to "
                     "the next parseable rule is discarded by the browser."
                     % (line_no, line[:60]))
        if depth < 0:
            fail("style.css line %d closes a rule that was never opened: %r"
                 % (line_no, line[:60]))
            depth = 0

    # 3. every class the registry renders must be styled. A rendered class with
    #    no rule is chrome that looks broken and blames nothing.
    tools_js = read(os.path.join(EXTENSION, "javascript", "canvas_tools.js"))
    nodes_js = read(os.path.join(EXTENSION, "javascript", "canvas_nodes.js"))
    rendered_classes = set()
    for src in (tools_js, nodes_js):
        for chunk in re.findall(r"'(forge-[a-z0-9-]+)'", src):
            rendered_classes.add(chunk)
        for attr in re.findall(r'class="([^"]*)"', src):
            for cls in attr.split():
                if cls.startswith("forge-") and "forge_mixin" not in cls:
                    rendered_classes.add(cls)

    # The host half of the stylesheet may legitimately be absent (the extension
    # checked out on its own, a different host layout). Missing it must DEGRADE
    # this one check and say so -- not abort the whole file and take the brace
    # and scoping checks down with it. A test that crashes diagnoses nothing.
    host_path = os.path.join(EXTENSION, "..", "..", "modules_forge",
                             "forge_canvas", "canvas.css")
    try:
        host_css = read(host_path)
    except OSError:
        host_css = ""
        SKIPS.append("the host's canvas.css was not found at %s - classes it "
                     "styles cannot be recognised, so the unstyled-class check "
                     "was skipped" % host_path)
    styled = set(re.findall(r"\.([a-zA-Z][a-zA-Z0-9_-]*)", css + host_css))

    # A class may earn its keep as a JS SELECTOR rather than as styling --
    # `.forge-wmask-control` exists purely so canvas_extra.js's fallback sweep can
    # find those four buttons, and has no appearance of its own. Treating that as
    # dead would push someone to delete it, and deleting it is precisely the bug
    # this whole suite was built around.
    wiring = read(os.path.join(EXTENSION, "javascript", "canvas_extra.js")) + \
        read(os.path.join(EXTENSION, "javascript", "weight_mask.js"))
    used_as_selector = set(re.findall(r"\.(forge-[a-z0-9-]+)", wiring))

    unstyled = sorted(c for c in rendered_classes
                      if c not in styled and c not in used_as_selector)
    if unstyled and host_css:
        fail("the registry renders %d class(es) that nothing styles and nothing "
             "selects - dead chrome that will look broken and blame nobody: %r"
             % (len(unstyled), unstyled))

    # 4. geometry stays scoped. CNPro must not restyle the host's own canvases:
    #    "do not touch the host" is the extension's first design rule, and a bare
    #    `.forge-toolbar .forge-btn { width: 24px }` would resize every button in
    #    every ForgeCanvas in the app, img2img included.
    scoped_marker = '[data-cnpro-nodes="1"]'
    for rule in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        selector, body = rule[0].strip(), rule[1]
        if not re.search(r"\.forge-(toolbar|btn|adjust-gap|layers-box|pen-box|layer-list)",
                         selector):
            continue
        if not re.search(r"(width|height|display|margin|padding|flex|grid|place-items)\s*:",
                         body):
            continue
        if scoped_marker in selector or ".forge-picking" in selector:
            continue
        fail("unscoped geometry rule %r changes layout for EVERY ForgeCanvas in "
             "the app, including the host's own. Scope it to %s."
             % (selector[:70], scoped_marker))

    # 5. the mechanism the menus depend on.
    #
    #    Menu width is MEASURED (canvas_nodes.js publishes --cnpro-menu-w) rather
    #    than derived from a percentage. The percentage version -- `width: 0` +
    #    `min-width: 100%` -- measured correctly in a bare page and collapsed to
    #    min-content in gradio, where the containing block is indefinite and the
    #    percentage degrades to `min-width: auto`. So this check FORBIDS the old
    #    idiom as well as requiring the new one: it looks like a tidy CSS-only
    #    solution and it is a coin flip.
    box_c = re.search(r"\.forge-image-container\[data-cnpro-nodes=\"1\"\] "
                      r"\.forge-toolbar-box-c\s*\{([^}]*)\}", css)
    if not box_c:
        fail("no scoped rule for .forge-toolbar-box-c - the tool menus have no "
             "layout at all, which is how they rendered as stacked blocks")
    else:
        body = box_c.group(1)
        if "width: var(--cnpro-menu-w" not in body:
            fail("the .forge-toolbar-box-c rule no longer takes its width from "
                 "the measured --cnpro-menu-w. A percentage or an intrinsic "
                 "keyword here resolves against an indefinite containing block "
                 "in gradio and silently becomes min-content - one row per "
                 "line:\n%s" % body.strip()[:200])
        if "min-width: 100%" in body:
            fail("the .forge-toolbar-box-c rule is back to `min-width: 100%%`. "
                 "That degrades to `min-width: auto` (= min-content) in the real "
                 "app, which is the vertical-stack bug")
        if "flex" not in body:
            fail("the .forge-toolbar-box-c rule is not a flex container - menu "
                 "rows will stack vertically again")

    # 5b. ROWS MUST BE ELASTIC AND MUST NOT SET A min-width.
    #
    #     A flex container's min-content is the largest of its items' minimums,
    #     so a min-width on a ROW becomes the MENU's min-content. That is not a
    #     theory: `.forge-range-row-wide { min-width: 260px }` made the gamma
    #     menu 260px wide AND made its 22px reset button unable to share the line
    #     - narrow and spilling, one cause, two symptoms, at every canvas size.
    row_rule = re.search(r"\.forge-toolbar-box-c \.forge-range-row\s*\{([^}]*)\}", css)
    if not row_rule:
        fail("no rule for .forge-toolbar-box-c .forge-range-row - menu rows have "
             "no layout")
    elif "flex: 0 1 auto" not in row_rule.group(1):
        fail("menu rows no longer size to their content. The row pitch is "
             "label + gap + slider, and that is what decides rows-per-line:\n%s"
             % row_rule.group(1).strip()[:200])

    # The slider track is capped. Precision comes from the wheel, so a longer
    # track buys nothing and costs rows-per-line.
    track = re.search(r"--cnpro-range-w:\s*(\d+)px", css)
    if not track:
        fail("--cnpro-range-w is gone - menu sliders have no width cap")
    elif int(track.group(1)) > 100:
        fail("--cnpro-range-w is %spx; menu sliders are capped at 100px"
             % track.group(1))
    for name, rule in re.findall(
            r"\.forge-toolbar-box-c (\.forge-range-row[a-z-]*)\s*\{([^}]*)\}", css):
        # the VALUE is parsed, not pattern-matched around: `min-width:\s*(?!0)`
        # reads as "not zero" and matches `min-width: 0` anyway, because \s* can
        # match nothing and the lookahead then sees the space
        for value in re.findall(r"min-width\s*:\s*([^;]+);", rule):
            value = value.strip()
            if value not in ("0", "0px", "auto"):
                fail("%s sets min-width: %s. That floor becomes the whole menu's "
                     "min-content, which is exactly how the gamma menu ended up "
                     "too narrow to fit its own reset button" % (name, value))

    # ...and the JS half must still exist and be called. A CSS variable nobody
    # sets falls back to 100%, i.e. straight back to the broken behaviour, in
    # complete silence.
    nodes_js = read(os.path.join(EXTENSION, "javascript", "canvas_nodes.js"))
    if "--cnpro-menu-w" not in nodes_js:
        fail("canvas_nodes.js never sets --cnpro-menu-w, so the menus fall back "
             "to the percentage that does not work")
    if "ResizeObserver" not in nodes_js:
        fail("canvas_nodes.js does not re-measure the button row - the Topaz "
             "probe reveals three more buttons after attach and every menu would "
             "keep the stale width")
    if not re.search(r"box\.style\.width\s*=", nodes_js):
        fail("canvas_nodes.js no longer writes the menu width as an INLINE "
             "style. The collapse could not be reproduced outside the running "
             "app, so the inline write is what makes the fix independent of that "
             "diagnosis - a custom property can still be beaten by a stylesheet")
    # The width comes from the CANVAS, because the canvas is what clips it
    # (.forge-image-container is overflow: hidden). Measuring the button row
    # instead caps every menu at the buttons' width even when the canvas is far
    # wider, which is what forced the edges menu onto three lines.
    # The wheel handler is the other half of the 100px track: without it, a
    # short slider is just a less precise slider.
    if "wireWheel" not in nodes_js:
        fail("canvas_nodes.js has no wheel handler - menu sliders are capped at "
             "100px on the assumption that the wheel supplies the precision")
    if "stopPropagation" not in nodes_js:
        fail("the wheel handler does not stopPropagation. The host zooms the "
             "canvas on ANY wheel over the container, and the toolbar sits "
             "inside it - adjusting a slider would zoom the image behind it")
    if "clientWidth" not in nodes_js:
        fail("canvas_nodes.js does not measure the container's clientWidth - the "
             "menu width contract is 'the usable canvas', and anything else "
             "either overflows the clip or wastes width")

    return report()


def report():
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    for s in SKIPS:
        print("PARTIAL SKIP -", s)
    print("ok - style.css parses, every rendered class is styled, canvas geometry "
          "is scoped to CNPro's own canvases, and the menu-width mechanism is intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
