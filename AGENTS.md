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

**Look for an existing chromium before downloading one.** Playwright keeps its
browsers in a machine-wide cache that outlives any one session, so on a machine
where these tests have been run before the ~700 MB download is very likely
already satisfied. Check first:

```bash
ls "$LOCALAPPDATA/ms-playwright"        # Windows
ls ~/.cache/ms-playwright               # Linux
ls ~/Library/Caches/ms-playwright       # macOS
```

A populated cache looks like `chromium-<rev>/`, `chromium_headless_shell-<rev>/`,
and possibly `firefox-*`, `webkit-*`, `ffmpeg-*`. If `chromium-<rev>` is there,
skip `playwright install` — confirm with a two-line `chromium.launch()` rather
than assuming either way.

The **node package** is the per-scratch-dir half, and it installs in seconds once
the browser download is satisfied. Earlier sessions may have left copies under
the agent scratch root (on Windows,
`%LOCALAPPDATA%\Temp\claude\<project>\<session>\scratchpad\node_modules\`); if
one is there, point `NODE_PATH` at it and skip the install too.

```bash
SP=<scratch dir>
npm install --no-save --prefix "$SP" playwright jsdom   # skip if reusing an old $SP
npx --yes playwright install chromium                   # skip if the cache above is populated

export CNPRO_TEST_NODE_PATH="$SP"
export CNPRO_URL=http://127.0.0.1:7870/     # YOUR instance, not the user's 7860
python tests/test_toolbar_live.py           # what the user sees
python tests/test_toolbar_dom.py            # injection vs the real canvas.html
python tests/test_generate_live.py          # that control steers a real image
```

Both **skip loudly** when their dependency is absent. A skip is not a pass — if
you see `SKIPPED`, nothing was verified, and saying otherwise is the failure mode
this whole file is about.

### Measuring CSS without the app: the standalone gradio harness

For a **layout** question you do not need Forge at all — a few dozen lines that
build the same components in a bare `gr.Blocks` on a spare port, driven by the
playwright above, answers it in seconds instead of a restart cycle. It found
four separate causes of one misaligned label that four rounds of reading the CSS
did not. Two things make it faithful, and both were learned by getting them
wrong:

* **Inject `style.css` as a raw `<style>` appended to `document.body`** — that is
  precisely what Forge does (`modules/ui_gradio_extensions.py:css_html` emits a
  `<link>` before `</body>`). Do **not** pass it to `gr.Blocks(css=…)`: gradio
  *parses* that string, which silently **drops at-rules it does not handle
  (`@container` vanished entirely)** and **scopes what remains, lending it
  specificity Forge never gives it**. The `css=` harness is therefore *more
  permissive* than production and will pass rules that fail in the real app.
* **Expect gradio's own svelte-scoped rules to outrank yours.** A single
  `.cnpro-…` class loses to gradio's two-class Row selectors: `align-items`,
  `gap` and `flex-wrap` were all being silently overridden. Anything that must
  hold needs `!important` — and the only way to know which is to read
  `getComputedStyle`, not the stylesheet.

Measure numbers, not vibes: element centres against each other (`dy` between a
label and its row's buttons should be `0`), and computed values for the property
you think you set. Then screenshot it and look — the numbers said `dy=0` while
the labels were painted straight over the scale.

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

Three of them read the host's `canvas.html` / `canvas.css` and find the webui two
levels up, which is right when this lives in `<webui>/extensions/<name>` and
wrong for a standalone clone (it resolves to `C:\modules_forge\…`). Set
**`CNPRO_WEBUI_DIR=<webui root>`** there, or `test_toolbar_dom`,
`test_toolbar_layout` and `test_canvas_parity` skip — loudly, saying exactly
that, rather than throwing ENOENT from inside the harness as they used to.

| File | Needs | Covers |
|---|---|---|
| `test_profile_parity.py` | node | editor JS ≡ python profile evaluation |
| `test_partition_of_unity.py` | node (half) | the n Inputs' shares sum to exactly 1 — every family, count, wave and convergence, on both sides |
| `test_mask_profile_coupling.py` | node (half) | the OUTPUT mask slots follow the profile selector — toolbar and backend compared against each other. **Stale as of 2026-08-02**: still asserts the pre-move input-side layout |
| `test_profile_scale_grid.py` | node | the range selects offer what the AXIS means: weights [0, 1], depth/drift -1..2, balance quarters — and an off-grid range in a saved string is still read as written |
| `test_profile_point.py` | — | the "Profile point" channel's arithmetic (`profile_scale.offset_profile_point`): offsets are in the plot's AXIS units (`|lo~hi`), offset 0 returns the profile byte-identical (the legend's "0 is the current profile" contract), tokens (M/C/P/A/G) are never counted or touched — only the chosen drawn point moves, negative indices count from the end, a clamp against the range and an index that names no point both WARN instead of guessing, an absent line offsets from its neutral flat, and parse_offsets accepts the negatives parse_factors still refuses |
| `test_ab_search.py` | numpy | **the A/B search converges — and converges on the WORST configuration when every grade is reversed**, so a flipped comparison sign cannot pass; a 5 is an observation rather than a refusal; no duel is ever two configurations too alike to tell apart (measured at 19–25% before the guard, in the one-LoRA-row spaces where it bites); the dislike row means what it says — a grade on the bottom row pushes both sides below par while still ordering them, a normal grade carries no verdict about the pair, and a dislike redirects the next duel to exploration (streak cleared by the first normal grade), so bad regions are avoided rather than only ranked; the SIMILARITY row learns the separation metric a comparison cannot — with no labels every weight is exactly the hand-written prior (the property that makes the feature safe to ship: an unused row must change nothing), the fit separates a visually inert dimension from a vivid one while leaving unlabelled dimensions untouched, a "similar" verdict also records a tie on the utility side (two images that look the same are worth the same — a stronger claim than the loose grade beside it), and the censoring defence holds end to end: the probe is silent while the metric sits at its prior, produces exactly the duel the learned metric refuses and the prior allows once a weight drops under the gate, restores it within a handful of full-weight answers (discounted, ten of them moved a weight 0.134→0.170 — a recovery path of hundreds of duels, i.e. none), and goes quiet again afterwards; against a two-peak taste the frontier holds BOTH peaks (visibly separated) and `portfolio()` draws good and bad samples from the right ends; no pair is re-asked past the repeat cap; the seed duels cover the categorical levels rather than landing where uniform chance puts them; a decisive win by a written-off configuration arms an immediate follow-up duel (a 10 is "look closer", not "done" — and not one comparison to average away); an "interesting" mark is acquisition, not evidence — the posterior stays bit-identical while the donor's coordinates surface as hybrid candidates inside good bases; the exported session HTML round-trips verbatim (`</script>`-laden prompts included), keeps its human-readable half, and a replayed solver holds the same posterior as the original (state from different rows is refused, not reindexed); the recipe the STOP button prints reconstructs the WHOLE configuration it names — prompt, NEGATIVE prompt, sampler/scheduler/steps/CFG/size (absent knobs stay absent), weights and seed included, and every named token is on the host capture list so Export can read it and Set can apply it; the three coin-fired tactics actually fire at their mean rates (bluff ~1/7, void ~1/12, rival ~1/14 — a rate wired to 0 is invisible to every other test), a void probe lands both sides ≥ VOID_SEPARATION from everything shown and returns None in a space with no void left, and a rival duel draws its two sides from two DIFFERENT attractors' basins; GOOD/N-GOOD outlive the search — a live loop QUEUES presses in arrival order (served after any grade that landed in the same instant, never instead of it), a request carries the COUNT it was made with (a 12-sample collage must not arrive as one sample), an idle session refuses the request so the panel's stage-and-self-click-Generate path runs instead, a stale staged request is dropped rather than turning a later unrelated Generate into a sample render, and the free-text N box is clamped into range on every reading (`""`/`"twelve"`/`-3`/`100000`) because the press has already happened by the time it is parsed; the capacity estimate MEANS something — it never falls below the keepers the frontier has already proved, it separates a taste with one sharp optimum from one where three dimensions are free (a number that does not is measuring the search's length or the space's size, not the good region), and after ONE graded duel it does not claim a slice of the space (measured at 56–155 of 1260 before the quality floor was made to clear the prior — `k(x,x)=1`, so a threshold at or under `Φ(QUALITY_MARGIN·max(NOISES))=0.816` passes everything unexplored); `population(n)` agrees with it — never more than asked, every pair at least MIN_DUEL_SEPARATION apart (a collage of near-duplicates is the failure that costs a GPU-hour and looks like success), and a different set on the second press; nothing ends a search irrecoverably — start() keeps the retained solver state (clearing it let an ungraded interrupted run destroy the previous search's taste), a RESUMED start keeps the Tried record and the recommendation too (the observations being resumed are the very duels those lines describe, and wiping them left a record that began mid-session under a first line claiming to resume 51 observations) while a FRESH start still clears all of it, the disk mirror of the solver state round-trips, and the retained-state lookup prefers staged import > session > disk mirror; a duel waits for its grade indefinitely — there is no idle timeout and no setting for one (a pair on screen has already cost two generations, and a clock only ever means walking away from the desk costs the answer); the tick's phase CSS wears the inference progress on the rendering side's interesting toggle (A holds a FULL bar while B renders), fades the grade-here nudge in exactly while a grade is awaited (never while rendering), clamps progress, and emits nothing at idle; a Profile-points row's offsets form an ORDERED dimension, its listed point indices move as a GROUP (one offset, every listed knot — the interval edit), its recipe names the rewritten weight_profile (omitting it reproduces the un-offset image), offset 0 puts the drawn profile in the recipe verbatim, and the trace says which offset each duel chose; reuse is an economy, not a rut — with the host's cache probe wired the share of already-rendered sides measurably rises (0.17→0.36 mean over the bench seeds; a reused side is ~20x cheaper than a generation) while the SAME convergence bar holds, the swap only ever trades away sampled-utility differences under one judgement's noise, at the reuse streak cap every duel is bit-identical to a cost-blind twin's (the realistic shown-points probe, not the everything-cached one — that variant passes on broken code and caught this test's own first draft), and a probe that throws reads as "nothing cached"; the trend model proposes and the posterior disposes — no nominations before SURROGATE_MIN_DUELS, nominees fed separately-proven winners are enriched for both AND contain the never-shown winning combination (the cross-combination guess the GP's product kernel cannot make), and a conditional weight's trend is learned per pick so one LoRA's weight lessons cannot drag another's |
| `test_coverage_map.py` | node (+ cv2 for half) | the coverage panel predicts the run: its fit geometry and letterbox colour against the real `crop_and_resize_mask`, its mean/peak aggregation, and the orange 1 / red >1 contract |
| `test_residual_layout.py` | torch | UNet weighting unchanged; token-layout mask invariants |
| `test_zimage_config.py` | — | Z-Image config sniff; v2.x and merged files refused |
| `test_zimage_module_tree.py` | torch + host | all 136 checkpoint tensors match |
| `test_zimage_injection.py` | torch + host | hooks hit the right blocks; removal is bit-exact |
| `test_defaults.py` | — | "Resize and Fill" + resolution 1024 at every decision point |
| `test_patcher_contract.py` | torch + host | flags declared; registry keeps declined/broken/refused apart |
| `test_control_merge_parity.py` | torch + host | `control_merge` ≡ the pre-subclass copy, 16 configs; the host's weighting surface stays unset; loader returns CNPro's classes |
| `test_unit_config_not_fatal.py` | — | a half-configured unit skips ITSELF (missing model / image), the other units still run, the skip reaches the infotext — and a genuine failure still raises |
| `test_mask_clear_reasons.py` | — | painted weight dies only for a DECLARED reason, and no declared reason is an image changing (reads the source; the static half of invariant 15e) |
| `test_input_mask_share.py` | numpy + cv2 | **a mask painted with one weight reduces to that weight and warns about nothing** — every shape, feather and rescale, through the real decode; two painted weights still say so; the pre-fix reduction is run alongside and has to fail |
| `test_mask_survives_insert.py` | node + playwright + host template | **that inserting an image does not destroy the mask painted on it** — at a new size, at the same size, and that removing the image still does |
| `test_canvas_adapter.py` | node | a host-originated `loadImage()` reaches CNPro's wrapper — the alias direction |
| `test_style_sheet.py` | — | style.css parses; every rendered class is styled; canvas geometry stays scoped |
| `test_toolbar_contract.py` | node | executes the renderer + registry: ids, classes, reveal, audit |
| `test_toolbar_dom.py` | node + jsdom | injects into the host's real canvas.html; ids match the wiring both ways |
| `test_toolbar_layout.py` | node + playwright | **measures pixels**: square buttons, one baseline, menus fit the toolbar |
| `test_toolbar_live.py` | node + playwright + running app | **what the user sees** |
| `test_theme_live.py` | node + playwright + running app | **that both themes paint a visible profile**: reads the plot canvas back and measures the main line, its selector bar, the step dot and the band colours against the surface they are on, in dark AND light — and pins dark to what it was measured to paint |
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
wave tokens     C<osc>@<phase>         cosine mode; P / PF / PV<κ> = multi-phase
                A<at>@<e>              convergence onto the flat share
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
