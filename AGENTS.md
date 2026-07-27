# Working on CNPro

Read `ARCHITECTURE.md` for the layer split and `MAINTENANCE.md` for the
invariants. This file is about **how to verify work here**, and it exists because
verification is where this codebase has actually gone wrong.

---

## 0. Port 7860 is the user's. Never kill it.

**The Forge instance on port 7860 belongs to the user. Do not attach to it, drive
it, generate on it, or — above all — kill or restart it.** You may freely start
and stop instances *you* started, on other ports. 7870 by convention:

```bash
python launch.py --port 7870 --api --ckpt-dir ... --controlnet-dir ...
export CNPRO_URL=http://127.0.0.1:7870/
```

This is section 0 because it was violated: an earlier run killed the user's
process and restarted it three times, then generated images on it. None of that
was necessary — nothing about the verification needed *their* instance.

Two properties make it worse than it sounds:

* **The tests generate images.** `test_generate_live.py` runs three 1024×1024
  jobs. On the user's instance that competes for their GPU and VRAM mid-session,
  and can evict the model they had loaded.
* **The failure is invisible.** Pointed at 7860 the tests *pass*. Nothing warns
  you; the user just sees their session behaving oddly.

So the rule is enforced in code, not only here: `toolbar_live_js.js` and
`test_generate_live.py` default to 7870 and **refuse 7860 even when it is passed
explicitly**, skipping loudly. If you genuinely need to inspect the user's
running instance, read it (HTTP GET, or Playwright with no clicks) — never
restart it, never submit work to it, and ask first.

Also: `webui-user.bat` is the user's launch config. Do not edit it. Pass what you
need via `COMMANDLINE_ARGS` or arguments to your own instance.

### You cannot identify an instance by its command line

**Several Forge processes can be running, and they look identical.** Measured on
this machine:

```
  PID ParentPID  CommandLine
34928     42468  "…\venv\Scripts\Python.exe" launch.py
20760     34928  "…\venv\Scripts\Python.exe" launch.py
40872     41568  "…\venv\Scripts\Python.exe" launch.py
44728     40872  "…\venv\Scripts\Python.exe" launch.py
```

Four processes, one command line, no `--port` anywhere — because the port arrives
through the **`COMMANDLINE_ARGS` environment variable**, which does not appear in
the command line. Also note the parent/child pairs: `launch.py` re-spawns itself,
so each instance is *two* processes and only the child holds the socket.

So **this is a footgun and must never be used**:

```powershell
# WRONG - kills the user's instance along with yours
Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
  Where-Object { $_.CommandLine -like '*launch.py*' } | Stop-Process -Force
```

### Identify by port, or better, remember your own PID

Preferred — capture the PID when you start it:

```powershell
$p = Start-Process -FilePath "…\venv\Scripts\Python.exe" -ArgumentList "launch.py" `
       -WorkingDirectory "D:\store\sd-webui-forge-neo" -PassThru
$myPid = $p.Id      # kill only this later, plus its child
```

Fallback — resolve the socket owner, and assert it is not the user's:

```powershell
$mine = (Get-NetTCPConnection -LocalPort 7870 -State Listen).OwningProcess
$theirs = (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue).OwningProcess
if ($mine -and $mine -ne $theirs) {
    $parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$mine").ParentProcessId
    Stop-Process -Id $mine -Force
    Stop-Process -Id $parent -Force -ErrorAction SilentlyContinue   # the launcher
}
```

Then verify 7860 is **still listening** before you report anything. If you are
unsure which instance is which, stop and ask — leaving a stray instance running
costs some RAM; killing the user's costs them their session.

**Clean up your own instances when you are done.** A Z-Image instance holds
several GB of VRAM and will compete with the user's next generation. Stop it by
PID (above), and say in your summary whether anything of yours is still running
and on which port.

---

## 1. Verify in a real browser. Use Playwright.

**For anything touching the UI — non-trivial changes, regressions, and anything
that has already broken once — a real browser run is required before you say it
works.**

Three toolbar outages shipped in a row here. Every one was silent, every one got
past a green test suite, and each was caught only by the *next* layer of
verification:

| Layer | What it proves | What it still misses |
|---|---|---|
| reading the source | a mechanism is present | whether it is correct |
| `node` + a DOM stub you populate | the logic works on nodes you created | whether the real nodes exist |
| `node` + jsdom + the real template | injection works | **no CSS, no layout engine** |
| **Playwright + the running app** | what the user sees | — |

Concretely, three facts about this UI are invisible to everything above
Playwright, and all three caused wrong conclusions during the investigation:

* `.forge-toolbar { opacity: 0 }` — **the toolbar is invisible until hovered.**
  A screenshot without a hover looks exactly like a broken toolbar.
* `.cnet-output-mask-group … { display: none !important }` hides 13 controls
  **on purpose**. jsdom loads no stylesheet, so a checker written against it
  reported 13 false failures per output-mask canvas.
* an element can be `display: inline-block` and still measure 0×0 because an
  ancestor is collapsed. *Declared visible* and *has pixels* are different
  questions, and only the second one is the user's.

### Running it

```bash
SP=<scratch dir>
npm install --no-save --prefix "$SP" playwright jsdom
npx --yes playwright install chromium

export CNPRO_TEST_NODE_PATH="$SP"
export CNPRO_URL=http://127.0.0.1:7870/     # YOUR instance, not the user's 7860
python tests/test_toolbar_live.py           # what the user sees
python tests/test_toolbar_dom.py            # injection vs the real canvas.html
python tests/test_generate_live.py          # that control steers a real image
```

Both **skip loudly** when their dependency is absent. A skip is not a pass — if
you see `SKIPPED`, nothing was verified, and saying otherwise is the failure mode
this whole file is about.

### Take the screenshot and look at it

`tests/toolbar_live_js.js` writes `CNPRO_SHOT`. Open it. Measuring that 17
elements have non-zero bounding boxes is not the same as seeing the toolbar, and
the difference has mattered here more than once.

---

## 1b. `ui-config.json` silently overrides code defaults

A1111 records component defaults into `ui-config.json` **keyed by the component's
LABEL**, not its elem_id - `txt2img/Resize Mode/value`, nothing containing
"controlnet". Once written it wins over the code, permanently.

This cost a full restart cycle: the defaults were correct in source, the test
suite passed, and the UI kept showing the old values. Grepping that file for
"controlnet" finds nothing and looks reassuring.

When a default "does not apply", check in this order:
1. `ui-config.json` for the component's **label** (`Resize Mode`,
   `Preprocessor resolution`);
2. whether the server has been restarted (Python is loaded once at startup).

`test_toolbar_live.py` now distinguishes these two automatically and names which
one it is. Beware label collisions when editing that file: CNPro's dropdown is
`Resize Mode`; img2img's own is `Resize mode`.

## 2. JS is served from disk, per request

Changing a `javascript/*.js` file does **not** need a WebUI restart — the server
reads it per request. It **does** need a browser hard-refresh
(<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>R</kbd>); the browser caches it happily,
and a user testing a fix against a cached broken file will report that your fix
did nothing.

Python changes (`scripts/`, `lib_cnpro/`, `cnpro_host/`, `cnpro_core/`) do need a
restart.

---

## 3. Do not report success from a pipeline's exit code

```bash
some_command | tail -5 ; echo "exit=$?"     # WRONG: that is tail's status
```

This bit twice in one session — once masking a truncated 3 GB download that had
run the disk out of space, once masking a failing test. Capture first:

```bash
OUT=$(some_command 2>&1); RC=$?
```

---

## 4. The silent-miss rule

`ARCHITECTURE.md` §8 is the long version and is worth reading in full. The short
version, for anything you add:

1. **Never let "absent" be a default.** Reveal everything except an explicit
   deny-list. A wrongly-visible control is noticed in seconds; a wrongly-hidden
   one is invisible by definition.
2. **Never collapse distinct failures into one outcome.** "Unsupported file",
   "loader is broken", and "deliberately refused" are three different problems
   for the user and used to be one message.
3. **Make "I did not consider this" impossible to express.** Capability flags are
   declared explicitly per patcher, including the `False`s — inheriting a default
   and choosing it look identical in source.
4. **Test what runs, against real inputs.** A stub you populated yourself only
   tests the half you already understood.

Diagnostics must also be unable to break what they diagnose. `attach()` is ~1800
lines in one function with no error boundary: anything that throws kills
everything below it, and the audit is the *last* statement. Every diagnostic
there is wrapped in `try`/`catch` for that reason.

---

## 5. Prove a fix by breaking it again

Before claiming a regression is fixed, re-introduce the bug and confirm the test
fails — then restore. Every toolbar test in `tests/` has been checked this way,
and the checks that looked most convincing were the ones that turned out to pass
on broken code.

A harness must **survive a broken module and report it**. An early version threw
`TypeError` from inside the harness when handed mangled ids: it caught the bug
and diagnosed nothing, which is most of the way back to a silent miss.

---

## 6. Test suite

```bash
for t in tests/test_*.py; do python "$t"; done
```

| File | Needs | Covers |
|---|---|---|
| `test_profile_parity.py` | node | editor JS ≡ python profile evaluation |
| `test_mask_profile_coupling.py` | node (half) | the mask slots follow the profile selector — toolbar and backend compared against each other |
| `test_residual_layout.py` | torch | UNet weighting unchanged; token-layout mask invariants |
| `test_zimage_config.py` | — | Z-Image config sniff; v2.x and merged files refused |
| `test_zimage_module_tree.py` | torch + host | all 136 checkpoint tensors match |
| `test_zimage_injection.py` | torch + host | hooks hit the right blocks; removal is bit-exact |
| `test_defaults.py` | — | "Resize and Fill" + resolution 1024 at every decision point |
| `test_patcher_contract.py` | torch + host | flags declared; registry keeps declined/broken/refused apart |
| `test_control_merge_parity.py` | torch + host | `control_merge` ≡ the pre-subclass copy, 16 configs; the host's weighting surface stays unset; loader returns CNPro's classes |
| `test_canvas_adapter.py` | node | a host-originated `loadImage()` reaches CNPro's wrapper — the alias direction |
| `test_style_sheet.py` | — | style.css parses; every rendered class is styled; canvas geometry stays scoped |
| `test_toolbar_contract.py` | node | executes the renderer + registry: ids, classes, reveal, audit |
| `test_toolbar_dom.py` | node + jsdom | injects into the host's real canvas.html; ids match the wiring both ways |
| `test_toolbar_layout.py` | node + playwright | **measures pixels**: square buttons, one baseline, menus fit the toolbar |
| `test_toolbar_live.py` | node + playwright + running app | **what the user sees** |
| `test_canvas_parity.py` | node + puppeteer-core + Chrome | **that the control gets the canvas** — decodes both sides, every pixel, 22 states |
| `test_unit_state_writes.py` | — | nothing rebuilds the unit State from a snapshot (lost-update guard) |
| `test_generate_live.py` | running app `--api` + models | **that a ControlNet steers a real image** |

`test_generate_live.py` establishes the BASE MODEL FAMILY before it generates
anything, and restores it afterwards. A Z-Image ControlNet on an SDXL base is
refused by CNPro (correctly, loudly) and the host then finishes the generation
without control — so the output equals the baseline and the old version of this
file reported "the ControlNet is not injecting" three times about working code.
It also skips, loudly, when the plain no-control baseline fails: that generation
carries no ControlNet at all, so it cannot be evidence about CNPro. A Z-Image
checkpoint additionally needs its text encoder among the UI's additional
modules; a headless instance has none, and the skip says so.

### 6a. The two that guard "what you see is what you generate"

An edit is only real once it reaches the control, and that path crosses two
boundaries where the value can be produced twice and disagree:

* **canvas → gradio channel.** `test_canvas_parity.py` decodes the displayed
  `<img>` and the `logical_image_background` value and compares every channel of
  every pixel, across layers (moved, scaled, rotated, flipped, lighten-blended,
  partly off-stage, per-layer gamma/invert), global adjustments, geometry, crop,
  strokes and drops. No tolerance — both sides are PNG with no resampling
  between them. The crop tool being open is the contract's one stated exception
  and is asserted, not skipped. The contract itself is written at the top of
  `javascript/canvas_extra.js`.
* **channel → unit State.** `test_unit_state_writes.py` forbids any handler
  writing the unit `gr.State` by rebuilding it. `dataclasses.replace(current,
  image=x)` looks like a field write and is a whole-unit write carrying ~60
  stale fields; two of those overlapping revert each other. One raster landing
  on a canvas moves two channels in the same tick, so they *do* overlap.

Both were written after a report that a dropped image never reached the
control. Neither can be satisfied by looking at the UI, which is the point:
the canvas showed the new picture in every broken case.

---

## 7. `docs/` is for humans. Serialization lives here.

The pages under `docs/` (GitHub Pages, served from that folder) are worked
examples for users. **They show the widget, never the profile string.** A reader
setting this up has a UI in front of them, not an API; a screenshot of the unit
with the curve drawn is the instruction, and `0@1;0.75@0.5;1@0` is noise to them.
The strings belong in this file instead, because an agent reproducing a page
*does* drive it through the API.

### The grammar, as the pages' examples use it

```
main            x@y;x@y;...            y normalized, optional |lo~hi range suffix
band segments   #C<profile> #M<profile> #F<profile>      coarse / mid / fine
band MODE       #B<band>               presence = the unit runs on the bands
```

Every step-domain segment carries the same range suffix, and the suffix is what
makes a band exceed 1: on `|0~2` a drawn `0.31` **is a multiplier of 0.62**. Half
the number you mean, every time — the single easiest thing to get wrong when
writing these by hand.

Example 1 (`docs/example_1.html`) is exactly:

```
unit 0  canny   0@1;0.75@0.5;1@0
unit 1  ipadapt 0@0.5;1@0.5|0~2#Bmid#C0@0.31;1@0.31|0~2#M0@0.75;1@0.75|0~2#F0@0.6;1@0.6|0~2
                -> coarse 0.62, mid 1.5, fine 1.2
```

### Screenshotting the UI for a page

The unit panels in `docs/` were captured by driving a browser over CDP against a
**private** instance (see section 0 — not the user's 7860). Two things make it
easy:

* the whole configuration can be pushed in through the extension's own infotext
  paste path, so the panel provably matches the picture; and
* **the profile editors poll their hidden textbox**, so writing a profile string
  into the textbox and waiting a tick is enough to redraw the plot — no event
  plumbing, no clicking on the canvas. That is how the three coarse-band variants
  in example 1 were shot from one page load.

Infotext does *not* carry the unit images; upload those separately
(`DOM.setFileInputFiles` on the input inside `#..._input_image`).
