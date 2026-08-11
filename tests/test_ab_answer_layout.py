"""The A/B answer row's pixel contract.

The Configuration component and its action column are one control.  They must
stay on one horizontal row; the text box must initially reach at least as far
down as the Set / Set GOOD / RESET stack; and none of those wide button labels
may wrap.  That height is a floor, not a ceiling: resizing the textarea must
make Configuration taller without stretching the buttons with it.

This test builds the real Gradio components, injects ``style.css`` the same way
Forge does (a raw style element after Gradio has built the page), and measures
their rendered boxes in Chromium at wide and narrow panel widths.  A CSS rule
being present is not proof that Gradio's more-specific component rules let it
win, so this contract is deliberately browser-only.

REQUIRES gradio + node + playwright + chromium.  Without any one of them the
test SKIPS LOUDLY rather than passing quietly.

Exit code 0 = pass or skip; 1 = fail.
"""
import json
import os
import shutil
import socket
import subprocess


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def skip(reason):
    print("SKIPPED - the A/B answer-row layout test did not run.")
    print("  %s" % reason.replace("\n", "\n  "))
    print("  Configuration height and button labels have NOT been measured.")
    return 0


def main():
    with open(os.path.join(ROOT, "scripts", "CNPro_AB.py"),
              encoding="utf-8") as source_file:
        source = source_file.read()
    answer = source.split("# --- the answer", 1)[-1].split(
        "# Set's own report", 1)[0]
    missing = [name for name in ("cnpro-ab-answer",
                                 "cnpro-ab-configuration",
                                 "cnpro-ab-set-actions")
               if name not in answer]
    if missing:
        print("FAIL - the production answer row no longer declares the layout "
              "contract class(es): %s" % ", ".join(missing))
        return 1
    if not shutil.which("node"):
        return skip("node is not on PATH")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    try:
        import gradio as gr
    except Exception as exc:
        return skip("gradio is unavailable: %s" % exc)

    eid = "cnpro_ab_layout"
    with gr.Blocks() as demo:
        with gr.Row(variant="compact", elem_classes=["cnpro-ab-answer"]):
            gr.Textbox(
                label="Configuration", value="", lines=1, max_lines=3,
                scale=9,
                placeholder="the search prints its recommendation here - or "
                            "paste one back in",
                elem_id=f"{eid}_config",
                elem_classes=["cnpro-ab-configuration"])
            with gr.Column(scale=2, min_width=180,
                           elem_classes=["cnpro-ab-set-actions"]):
                with gr.Row(variant="compact"):
                    gr.Button(value="Set", variant="primary", scale=1,
                              min_width=80, elem_id=f"{eid}_set")
                    gr.Button(value="Set GOOD", scale=1, min_width=80,
                              elem_id=f"{eid}_set_good")
                gr.Button(value="RESET", scale=1, min_width=160,
                          elem_id=f"{eid}_reset")

    env = dict(os.environ)
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    webui = env.get("CNPRO_WEBUI_DIR", "")
    host_css = os.path.join(webui, "style.css") if webui else ""
    if not os.path.isfile(host_css):
        host_css = ""
    port = free_port()
    try:
        demo.launch(server_name="127.0.0.1", server_port=port,
                    prevent_thread_lock=True, quiet=True, show_error=True)
        proc = subprocess.run(
            ["node", os.path.join(HERE, "ab_answer_layout_js.js"),
             "http://127.0.0.1:%d/" % port,
             os.path.join(ROOT, "style.css"), host_css],
            capture_output=True, text=True, env=env)
    except Exception as exc:
        return skip("the private Gradio harness could not start: %s" % exc)
    finally:
        demo.close()

    if proc.returncode != 0:
        return skip("node harness failed:\n%s" % proc.stderr.strip())
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return skip("harness produced no JSON:\n%s\n%s"
                    % (proc.stdout[:400], proc.stderr[:400]))
    if data.get("unavailable"):
        return skip(data["unavailable"])
    if data.get("fatal"):
        print("FAIL\n  browser harness threw:\n  %s"
              % data["fatal"].replace("\n", "\n  "))
        return 1

    failures = []
    cases = data.get("cases") or []
    if len(cases) != 3:
        failures.append("the browser measured %d viewport(s), expected 3"
                        % len(cases))
    for case in cases:
        if case["rowFlexWrap"] != "nowrap":
            failures.append(
                "at %dpx the answer row computes flex-wrap: %s, not nowrap"
                % (case["viewport"], case["rowFlexWrap"]))
        if not case["sameLine"]:
            failures.append(
                "at %dpx Configuration and its buttons are no longer one row"
                % case["viewport"])
        if case["configurationHeight"] + 0.5 < case["actionsHeight"]:
            failures.append(
                "at %dpx the Configuration component is %.1fpx high but the "
                "button stack is %.1fpx" % (
                    case["viewport"], case["configurationHeight"],
                    case["actionsHeight"]))
        if case["configurationInnerHeight"] + 0.5 < \
                case["configurationContentHeight"]:
            failures.append(
                "at %dpx Gradio gives Configuration %.1fpx of content height "
                "but its visible label/textarea occupies only %.1fpx" % (
                    case["viewport"], case["configurationContentHeight"],
                    case["configurationInnerHeight"]))
        if case["configurationFieldBottomGap"] > 0.5:
            failures.append(
                "at %dpx the Configuration textarea ends %.1fpx above the "
                "height its component owns" % (
                    case["viewport"], case["configurationFieldBottomGap"]))
        resized = case["resized"]
        if not resized["sameLine"]:
            failures.append(
                "at %dpx resizing Configuration moved the buttons off its row"
                % case["viewport"])
        if resized["textareaHeight"] < case["textareaHeight"] + 79:
            failures.append(
                "at %dpx Chrome asked the textarea to grow by 80px, but it "
                "grew by only %.1fpx" % (
                    case["viewport"], resized["textareaHeight"] -
                    case["textareaHeight"]))
        if resized["configurationHeight"] <= resized["actionsHeight"] + 0.5:
            failures.append(
                "at %dpx the resized Configuration cannot grow above its "
                "button-height floor" % case["viewport"])
        if abs(resized["actionsHeight"] - case["actionsHeight"]) > 0.5:
            failures.append(
                "at %dpx resizing Configuration also changed the action stack "
                "from %.1fpx to %.1fpx" % (
                    case["viewport"], case["actionsHeight"],
                    resized["actionsHeight"]))
        if len(case["buttons"]) != 3:
            failures.append("at %dpx the browser found %d action buttons, "
                            "expected 3" % (
                                case["viewport"], len(case["buttons"])))
        for button in case["buttons"]:
            if button["whiteSpace"] != "nowrap" or button["lineCount"] != 1:
                failures.append(
                    "at %dpx %s is not protected from wrapping: %d rendered "
                    "line(s), white-space: %s" % (
                        case["viewport"], button["label"],
                        button["lineCount"], button["whiteSpace"]))

    if failures:
        if os.environ.get("CNPRO_AB_DEBUG"):
            print(json.dumps(data, indent=2))
        print("FAIL (%d)" % len(failures))
        for failure in failures:
            print("  - %s" % failure)
        return 1

    print("PASS - the answer stays one row, Configuration has the action-stack "
          "height only as a floor and remains resizable, and all 3 wide button "
          "labels stay on one line at %s%s."
          % (", ".join("%dpx" % case["viewport"]
                       for case in cases),
             " with the host stylesheet" if data.get("hostStyles") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
