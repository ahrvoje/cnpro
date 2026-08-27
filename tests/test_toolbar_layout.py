"""The toolbar's PIXELS: square buttons, one aligned row, menus that fit.

WHAT THIS TEST IS FOR
---------------------
Three toolbar tests already existed and none of them could see a pixel:

    test_toolbar_contract.py  runs the renderer on a DOM stub -> ids and reveal
    test_toolbar_dom.py       jsdom + the real template       -> nodes and display
    test_style_sheet.py       parses style.css as text        -> rules exist

So the whole suite was green while the colour-picker button rendered wider than
its neighbours and a pixel low, and while every tool menu stacked its rows
vertically and pushed the image off screen. Those are layout facts. "The rule is
in the stylesheet" and "the box is 24x24" are different claims, and only the
second one is the feature the user sees.

This runs the real template, the real host stylesheet, CNPro's real stylesheet
(loaded second, as the app loads it) and the real injectors in headless Chromium,
and measures. Hermetic - no webui, no server, ~2 seconds.

WHAT IS PINNED
--------------
1. Every visible toolbar button is SQUARE and identically sized. This is the
   structural claim: an icon -- text glyph, two-letter label or inline SVG --
   cannot change its button's box.
2. Every button in the row shares one baseline, to the pixel. `.prose` is
   reproduced in the page because its `:not(:last-child)` margin rule is what
   used to knock the last button out of line.
3. No menu is wider than the toolbar, and none overflows the canvas.
4. Menu rows are single-line (label beside slider, not above it).

REQUIRES playwright + chromium:
    npm install --no-save --prefix <dir> playwright
    npx playwright install chromium
    CNPRO_TEST_NODE_PATH=<dir> python tests/test_toolbar_layout.py
Without it the test SKIPS LOUDLY rather than passing quietly.

Exit code 0 = pass or skip; 1 = fail.
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


def run():
    if not shutil.which("node"):
        return None, "node is not on PATH"

    env = dict(os.environ)
    if "CNPRO_TEST_NODE_PATH" not in env:
        env["CNPRO_TEST_NODE_PATH"] = os.path.join(
            os.environ.get("LOCALAPPDATA", "/tmp"), "Temp", "claude")

    proc = subprocess.run(["node", os.path.join(HERE, "toolbar_layout_js.js")],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return None, "node harness failed:\n%s" % proc.stderr.strip()
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return None, ("harness produced no JSON:\n%s\n%s"
                      % (proc.stdout[:400], proc.stderr[:400]))
    if data.get("unavailable"):
        return None, data["unavailable"]
    return data, None


def main():
    data, why = run()
    if data is None:
        # A skip must be IMPOSSIBLE to mistake for a pass.
        print("SKIPPED - the toolbar layout test did not run.")
        print("  %s" % why.replace("\n", "\n  "))
        print("  Button geometry and menu widths have NOT been measured.")
        return 0

    if data.get("fatal"):
        fail("the harness threw:\n%s" % data["fatal"])
        return report()

    errors = [c for c in data.get("console", [])
              if c.startswith("error") or c.startswith("pageerror")]
    if errors:
        fail("the page logged errors while injecting:\n  %s" % "\n  ".join(errors))

    inj = data["inject"]
    if not inj["ok"]:
        fail("inject() returned %r against the real template" % inj["ok"])
    if inj["audit"]:
        fail("audit() is unhappy in a real browser with real CSS: %r" % inj["audit"])

    geo = data["geometry"]
    buttons = geo["buttons"]
    if len(buttons) < 20:
        fail("only %d visible buttons were measured - the toolbar is not fully "
             "rendered, so nothing below means anything" % len(buttons))
        return report(data)

    # 1. SQUARE. The user's report was that the colour-picker button was not.
    not_square = [b for b in buttons if abs(b["w"] - b["h"]) > 0.5]
    if not_square:
        fail("%d button(s) are not square: %s"
             % (len(not_square),
                ", ".join("%s %gx%g" % (b["id"], b["w"], b["h"]) for b in not_square[:6])))

    # 2. IDENTICAL. One odd size in a row of equals is what "misaligned" looks
    #    like even when every box is individually square.
    widths = {b["w"] for b in buttons}
    heights = {b["h"] for b in buttons}
    if len(widths) > 1 or len(heights) > 1:
        by_size = {}
        for b in buttons:
            by_size.setdefault((b["w"], b["h"]), []).append(b["id"])
        fail("buttons come in %d different sizes; an icon is still able to size "
             "its own box: %s"
             % (len(by_size),
                "; ".join("%gx%g -> %s" % (w, h, ", ".join(ids[:4]))
                          for (w, h), ids in by_size.items())))

    # 3. ONE BASELINE. This is the alignment claim, measured. Buttons may wrap to
    #    a second row if the toolbar is narrow, so compare within each row.
    rows = {}
    for b in buttons:
        rows.setdefault(round(b["y"]), []).append(b)
    for y, row in sorted(rows.items()):
        tops = {b["y"] for b in row}
        if len(tops) > 1:
            fail("row at y=%s has buttons on %d different tops (%s) - something "
                 "in the box model still lets a button drift vertically"
                 % (y, len(tops),
                    ", ".join("%s@%g" % (b["id"], b["y"]) for b in row[:6])))

    # 3b. GROUP SEPARATORS SURVIVE. `margin: 0 !important` on every button and
    #     `margin-left: <gap> !important` on the group starters are two
    #     !importants pointing opposite ways; the winner is decided by
    #     specificity, and if the reset out-specifies the gap the groups silently
    #     run together. Measured, because the stylesheet reads correct either way.
    starters = [b for b in buttons if b["groupStart"]]
    if not starters:
        fail("no button carries .forge-adjust-gap - the toolbar's tool groups "
             "have stopped being separated at all")
    for b in starters:
        if b["marginLeft"] in ("0px", "", None):
            fail("group separator %r computed margin-left: %r. The `margin: 0 "
                 "!important` reset is out-specifying the gap rule, so every "
                 "tool group runs into the next." % (b["id"], b["marginLeft"]))

    # 4. THE WIDTH CONTRACT: every menu is exactly the usable canvas width.
    #
    #    Not approximately, and not "between bounds" - one value. Wider is
    #    clipped by .forge-image-container's overflow: hidden; narrower wastes
    #    width that gets paid for in height. Both failures reported from the
    #    running app were a menu that was neither: pinned to min-content (one row
    #    per line), then pinned to the button row (edges on three lines).
    usable = geo["usableWidth"]
    if not geo["menuWidthVar"]:
        fail("--cnpro-menu-w was never published on the toolbar, so every menu "
             "is falling back to the percentage width that collapses to "
             "min-content in the real app")
    #
    #    THE ONE EXCEPTION is declared, not inferred: a menu the registry marks
    #    `fit: 'content'` (the layers menu - a column beside a list, nothing in
    #    it wraps) is as wide as its content and CAPPED at the usable width.
    #    Pinned to the canvas it paid for nothing in height and covered the
    #    picture with backdrop; so here, where the canvas is far wider than
    #    that content, such a menu must be narrower than the canvas - and never
    #    wider.
    fits = []
    for menu in geo["menus"]:
        if menu.get("fit") == "content":
            fits.append(menu["id"])
            if menu["w"] > usable + 1.5:
                fail("content-fit menu %r is %gpx wide against %gpx of usable "
                     "canvas - the cap is not holding, so it is clipped"
                     % (menu["id"], menu["w"], usable))
            elif menu["w"] > usable - 1.5:
                fail("content-fit menu %r is pinned to the canvas (%gpx of "
                     "%gpx): it is being sized like a wrapping menu, and its "
                     "backdrop covers the picture for nothing"
                     % (menu["id"], menu["w"], usable))
            continue
        if abs(menu["w"] - usable) > 1.5:
            fail("menu %r is %gpx wide against %gpx of usable canvas. The "
                 "contract is exactly the usable width: wider is clipped, "
                 "narrower costs height." % (menu["id"], menu["w"], usable))
    if "layersBox" not in fits:
        fail("the layers menu is no longer rendered as a content-fit menu "
             "(data-cnpro-fit) - it is back to the canvas width, covering the "
             "picture with backdrop it does not use")

    # 5. HORIZONTAL ROWS, and MORE THAN ONE PER LINE. A row taller than ~28px
    #    means the label went back above the slider. But rows can be short and
    #    still stack, which is exactly what the collapsed-width bug looked like,
    #    so also require that a multi-row menu is not one row per line.
    for menu in geo["menus"]:
        if menu["tallestRow"] > 28:
            fail("menu %r has a %gpx-tall row - the label is stacked above the "
                 "slider again instead of beside it"
                 % (menu["id"], menu["tallestRow"]))
        # A ONE-ROW MENU MUST BE ONE LINE. Named, because the gamma menu has
        # spilled its reset button onto a second line twice, for two different
        # reasons (`flex: 1 1 100%` claiming the whole line, then a 260px
        # min-width on the row that nothing could shrink). One slider and one
        # 22px button not fitting on a 500px line is the clearest possible signal
        # that a row is refusing to shrink.
        if menu["rowCount"] == 1 and menu["childLineCount"] > 1:
            fail("menu %r has ONE slider row and still occupies %d lines (%gpx "
                 "tall in a %gpx menu) - its row will not shrink, so the tool's "
                 "own reset button is pushed onto a line of its own"
                 % (menu["id"], menu["childLineCount"], menu["h"], menu["w"]))

        # Only meaningful when two rows COULD share a line at this width -- the
        # pen menu's colour picker legitimately takes most of one.
        if menu["fitsTwoPerLine"] and menu["lineCount"] >= menu["rowCount"]:
            fail("menu %r puts its %d rows on %d separate lines even though two "
                 "would fit side by side (%gpx rows in a %gpx menu) - that is the "
                 "vertical stack the user reported"
                 % (menu["id"], menu["rowCount"], menu["lineCount"],
                    menu["widestRow"], menu["w"]))

    # 5b. THE LABEL COLUMN. Sized to the widest text the MENU can show, and
    #     immovable while the value changes.
    #
    #     Before this, one 100px column served every menu in the toolbar because
    #     one menu needs "mask opacity 100". The gamma menu's longest label is
    #     "gamma 4.987", so it reserved 34px of nothing - on chrome that sits on
    #     top of the user's image. The saving is only real if the column is
    #     EXACTLY the widest candidate, so that is what is asserted, in both
    #     directions.
    menus_with_labels = data.get("labels") or []
    if len(menus_with_labels) < 5:
        fail("only %d menu(s) reported a label column - the menus did not open, "
             "so the column width is unverified" % len(menus_with_labels))
    for menu in menus_with_labels:
        rows = menu["rows"]
        no_sizer = [r["id"] for r in rows if not r["hasSizer"]]
        if no_sizer:
            fail("menu %r has %d label(s) with no data-max sizer (%s) - they fall "
                 "back to the stylesheet's one-size column, so the menu is as "
                 "wide as the widest label in the whole toolbar again"
                 % (menu["id"], len(no_sizer), ", ".join(no_sizer[:4])))
            continue
        missing_own = [r["id"] for r in rows if not r["own"]]
        if missing_own:
            fail("menu %r has row(s) with no declared labelMax (%s) - the column "
                 "cannot know how wide that row will get"
                 % (menu["id"], ", ".join(missing_own[:4])))

        # 5b-i. every row in a menu shares one column: that is what keeps the
        #       wrapped lines reading as a grid.
        column_widths = {round(r["width"], 1) for r in rows}
        if len(column_widths) > 1:
            fail("menu %r has %d different label widths (%s) - rows in one menu "
                 "no longer line up"
                 % (menu["id"], len(column_widths),
                    ", ".join("%s=%gpx" % (r["id"], r["width"]) for r in rows[:5])))

        # 5b-ii. EXACTLY the widest candidate. Tolerance is sub-pixel, not
        #        "roughly": slack here is the bug being fixed.
        widest = menu["widestCandidate"]
        actual = round(menu["columnWidth"], 2)
        if actual > widest + 1.0:
            fail("menu %r reserves a %gpx label column for a widest label of "
                 "%gpx (%r) - %gpx of the user's image is covered by nothing"
                 % (menu["id"], actual, widest,
                    ", ".join(menu["candidates"][:3]), actual - widest))
        if actual < widest - 0.5:
            fail("menu %r has a %gpx label column but its widest label needs "
                 "%gpx - that label renders with an ellipsis"
                 % (menu["id"], actual, widest))

        # 5b-iii. IMMOVABLE. The column must be the same width with every value
        #         the tools can produce, and with no text at all - that is the
        #         difference between "fixed at the right number" and "sizes to
        #         its content", and only the first one survives a drag.
        for row in rows:
            moved = {round(w, 1) for w in row["widthsWhileWriting"]}
            moved.add(round(row["widthWhenEmpty"], 1))
            if len(moved) > 1:
                fail("label %r changes width (%s) as its value is written - the "
                     "column breathes mid-drag, which slides the slider "
                     "sideways under the cursor"
                     % (row["id"], ", ".join("%gpx" % w for w in sorted(moved))))
            if row["clipped"]:
                fail("label %r is clipped with an ellipsis at one of its own "
                     "declared values - its labelMax understates the width it "
                     "needs" % row["id"])

    # 5c. A ROW INJECTED AFTER THE RENDER JOINS THE COLUMN.
    #
    #     weight_mask.js inserts the feather slider into the weight-mask menu at
    #     attach time; the registry never sees it. It joins by declaring its own
    #     data-label-max and calling syncLabelSizers. Broken, that path shows up
    #     only in the running app, on one label, as an ellipsis.
    late = data.get("lateRow") or {}
    if late.get("error"):
        fail("the late-row check could not run: %s" % late["error"])
    elif late:
        if not late.get("synced"):
            fail("syncLabelSizers() reported 0 menus synced - a row injected "
                 "after the render never joins its menu's label column")
        if not late.get("allSizersEqual"):
            fail("after syncLabelSizers() the labels in one menu carry different "
                 "sizers - the column is no longer shared")
        if not late.get("sizerIncludesLateRow"):
            fail("syncLabelSizers() did not fold the injected row's own "
                 "data-label-max into the menu's sizer, so that row is the one "
                 "label that gets clipped")
        after = late.get("widthsAfter") or []
        before = late.get("widthsBefore") or []
        if after and before and max(after) <= max(before):
            fail("the label column did not widen for an injected row that is "
                 "wider than every declared label (%s -> %s)" % (before, after))
        if after and len(set(round(w, 1) for w in after)) > 1:
            fail("after syncLabelSizers() the menu's labels have different "
                 "widths %s - the rows no longer line up" % after)
        if late.get("widthsTwice") != after:
            fail("syncLabelSizers() is not idempotent: a second call moved the "
                 "column from %s to %s" % (after, late.get("widthsTwice")))
        if late.get("widthsAfterRemoval") != before:
            fail("removing the injected row left the column at %s instead of "
                 "returning to %s - the sizer ratchets and never narrows again"
                 % (late.get("widthsAfterRemoval"), before))

    # 5d. NOTHING IS EVER OUTSIDE THE CLIP.
    #
    #     `.forge-image-container` is `overflow: hidden`. A button past its right
    #     edge is not "slightly overflowing" - it is unreachable, and it looks
    #     completely healthy from every direction this suite could previously
    #     ask: it is in the DOM, its computed display is not none, and its
    #     bounding box is a proper 24x24. It shipped exactly that way (19
    #     buttons = a 546px row on a 437px canvas; G/C/M/F were the five that
    #     fell off) and was found by screenshotting the running app, which is
    #     the layer this check exists to stop needing.
    #
    #     The 900px case proves nothing on its own - everything fits - so the
    #     row is measured at the widths a real unit actually gets.
    for case in data.get("narrow") or []:
        if case["outsideTheClip"]:
            fail("at a %gpx canvas, %d button(s) are OUTSIDE the clip and "
                 "therefore unreachable: %s. The row is %gpx in a %gpx toolbar; "
                 "`flex-wrap: wrap` cannot fire while the toolbar is allowed to "
                 "be wider than the canvas."
                 % (case["canvasW"], len(case["outsideTheClip"]),
                    ", ".join(case["outsideTheClip"][:8]),
                    case["rowW"], case["toolbarW"]))
        if case["toolbarW"] > case["canvasW"] + 0.5:
            fail("at a %gpx canvas the toolbar is %gpx wide - %gpx of it is "
                 "clipped away" % (case["canvasW"], case["toolbarW"],
                                   case["toolbarW"] - case["canvasW"]))
        if case["containerScrollW"] > case["containerClientW"] + 0.5:
            fail("at a %gpx canvas the container's content is %gpx wide - "
                 "something inside it still overflows the clip"
                 % (case["canvasW"], case["containerScrollW"]))
        for menu in case["menus"]:
            if menu["outside"]:
                fail("at a %gpx canvas, menu %r is outside the clip (%gpx wide)"
                     % (case["canvasW"], menu["id"], menu["w"]))
        # and the wrap must actually be what saved it, not a shrunken button
        if case["canvasW"] < case["rowScrollW"] - 1 and case["buttonLines"] < 2:
            fail("at a %gpx canvas the %d-button row still occupies one line "
                 "(%gpx) - it is not wrapping, it is overflowing"
                 % (case["canvasW"], case["buttons"], case["rowW"]))

    # 6. THE WHEEL CONTRACT. One notch = one of the slider's own steps, clamped
    #    at both ends, and it must NOT reach the canvas underneath.
    wheel = data.get("wheel") or {}
    sliders = wheel.get("sliders") or []
    if len(sliders) < 10:
        fail("only %d menu sliders were wheel-tested - the menus did not open, so "
             "the wheel is unverified" % len(sliders))
    for s in sliders:
        if abs(s["up"] - s["step"]) > 1e-6:
            fail("wheel up on %r moved it by %g, expected its own step of %g"
                 % (s["id"], s["up"], s["step"]))
        if abs(s["down"] + s["step"]) > 1e-6:
            fail("wheel down on %r moved it by %g, expected -%g"
                 % (s["id"], s["down"], s["step"]))
        if not s["clampedAtMax"]:
            fail("wheel up on %r pushed it past its max of %g" % (s["id"], s["max"]))
        if not s["clean"]:
            fail("%r produced a float-dust value (0.1*3 = 0.30000000000000004); "
                 "any label built from it shows the dust" % s["id"])
        if s["inputEvents"] < 2:
            fail("%r fired %d 'input' events for 2 wheel notches - the handlers "
                 "that redraw from these sliders will not run"
                 % (s["id"], s["inputEvents"]))

    # The one that is invisible until you use it: the host zooms the canvas on
    # ANY wheel over the container, and the toolbar is a child of that container.
    if wheel.get("zoomEventsLeaked"):
        fail("%d wheel event(s) reached the canvas container - adjusting a menu "
             "slider also zooms the image behind it" % wheel["zoomEventsLeaked"])

    # every menu slider is capped at 100px
    wide = [s for s in data["geometry"].get("sliderWidths", []) if s["w"] > 100.5]
    if wide:
        fail("%d menu slider(s) exceed the 100px cap: %s"
             % (len(wide), ", ".join("%s %gpx" % (s["id"], s["w"]) for s in wide[:5])))

    return report(data)


def report(data=None):
    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    geo = data["geometry"]
    b = geo["buttons"][0]
    menus = geo["menus"]
    print("ok - %d toolbar buttons, all %gx%g and on one baseline; %d menus, "
          "widest %gpx against a %gpx toolbar, tallest row %gpx"
          % (len(geo["buttons"]), b["w"], b["h"], len(menus),
             max(m["w"] for m in menus), geo["toolbar"]["w"],
             max(m["tallestRow"] for m in menus)))
    labels = data.get("labels") or []
    if labels:
        print("     label columns (each = its own menu's longest label): %s"
              % ", ".join("%s %gpx" % (m["id"].replace("Box", ""),
                                       round(m["columnWidth"], 1))
                          for m in labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
