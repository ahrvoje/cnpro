# CNPro — architecture

CNPro is the ControlNet unit widget (weight/balance/band/depth profiles, rainbow
weight masks, multi-input units, per-unit prompts) packaged as a self-contained
extension.

Two design goals drive every decision here:

1. **Do not touch the host.** No host file is edited, no module is
   monkey-patched, no global registry is mutated. Enabling CNPro cannot change
   behaviour for anyone who is not using CNPro.
2. **Survive the host and the model architecture changing.** Both are expected
   to change; neither should cost more than an adapter.

---

## 1. The three layers

The codebase is split by **lifespan**, not by subject matter. Each layer has a
different expected rate of change, and the dependency arrow points strictly
downward.

```
    lib_cnpro/   scripts/   javascript/      L3  UI shell
          |                                      rewritten per UI framework
          v
    cnpro_host/                              L2  host bindings
          |                                      rewritten per host
          v
    cnpro_core/                              L1  the engine
                                                 survives everything
```

### L1 — `cnpro_core/` (host-agnostic)

Profile evaluation, LUT construction, balance factors, and the depth/band
mapping. **Hard rule: no imports at all.** Not the host, not torch, not gradio,
not CNPro's own upper layers.

This is enforceable and enforced — the module currently has *zero* import
statements and runs under a bare interpreter:

```
python -c "import sys; sys.path.insert(0,'.'); \
  from cnpro_core.weight_profile import evaluate_weight_profile; \
  print(evaluate_weight_profile([(0,0),(0.5,1),(1,0)], 0.25))"
```

If you ever need something host-shaped in here, you need it in L2 instead.

### L2 — `cnpro_host/` (the only code that may import the host)

* `adapter.py` — the host boundary. Ten names, documented in place.
* `host_forge_neo.py` — the Forge-family implementation. Thin lookups only.
* `registry.py` — CNPro's own ordered control-model list and loader.
* `patchers/` — one module per injection mechanism, plus the injection engines
  (`*_impl.py`), plus three shared surfaces:
  * `weighting.py` — the profile engine. Pure index math over lists of residual
    tensors; knows nothing about UNets, DiTs or model families.
  * `residual_layout.py` — the three things `weighting.py` cannot know: a
    residual's rank, how a painted mask projects onto it, and what depth site
    *i* sits at. One layout per residual geometry, not per model.
  * `zimage_config.py` — checkpoint→config for Z-Image. Zero imports, so it is
    testable against the released file's shapes with no torch and no download.

### L3 — `lib_cnpro/`, `scripts/`, `javascript/` (UI shell)

The widget itself, carried over unchanged. It never imports the host; a grep for
`^from modules_forge|^from backend` across L1 and L3 returns nothing, and that
is the invariant to preserve.

---

## 2. Why CNPro owns its patchers instead of patching the host

Forge exposes `modules_forge.shared.supported_control_models` (a plain list) and
`try_load_supported_control_model` (first match wins). Inserting CNPro's
patchers at the head of that list would work — and would silently change which
classes the **host's own** ControlNet builds, for every user.

So CNPro keeps a private registry (`cnpro_host/registry.py`) and builds its own
`ControlNet` / `ControlLora` / `T2IAdapter`, which subclass the host's classes
of the same names. The profile-aware code is reached by ordinary subclassing.

Recognizing a file is **delegated** to the host's own loader
(`modules_forge.supported_controlnet.ControlNetPatcher.try_build_from_state_dict`);
only the resulting control object is re-wrapped in CNPro's class, reusing the
loaded modules. A second copy of the state-dict sniffing could only drift out of
agreement with the host's about which models load — it already had once, when
neo's `cldm.ControlNet` grew a required `hint_width` argument. Delegating also
shares the host's process-lifetime model cache instead of loading the same
weights twice.

`ControlLora` and `T2IAdapter` go further and inherit from *two* bases —
CNPro's (for the profiles) and the host's same-named class (for the LoRA build
in `pre_run`, the T2I forward pass, `ControlLoraOps`). The MRO puts CNPro's
first, so `ControlLora.pre_run` runs the host's build *and* our profile lookups,
and `T2IAdapter` needs no `get_control` of its own: `ControlBase.get_control`
applies the weight profile and defers to the host's forward pass, which ends by
calling `self.control_merge` — ours.

Three consequences of all this are load-bearing and easy to undo by accident.
`tests/test_control_merge_parity.py` pins all three:

* **The host's weighting surface must stay unset.** `positive_advanced_weighting`
  and friends are inherited, and the host's weighting pass runs inside
  `super().control_merge()`. It is inert only because CNPro's equivalent inputs
  are deliberately named differently (`cond_layer_weights`, `region_masks`, …).
  Give one of them the host's name and every weight is applied twice, silently.
* **`ControlBase.__init__` calls `HostControlBase.__init__` by name, not
  `super()`.** In the two diamond classes `super()` resolves to
  `HostControlLora` / `HostT2IAdapter`, whose constructors demand their own
  arguments and raise. Those two set their own fields themselves.
* **Method resolution has to land on the right side.** `ControlLora.get_models`
  must be ours (the host's `ControlNet` version appends a
  `control_model_wrapped` a LoRA has no equivalent of), while
  `inference_memory_requirements` must be the host's (it counts the LoRA
  weights). Either one silently resolving the other way still generates images.

One stated behaviour note: for `T2IAdapter` the profile is now read at the
pre-modifier timestep, because the gate runs before the host's
`controlnet_conditioning_modifiers` loop rather than after it. Nothing in this
codebase registers such a modifier — the host only exposes the setter — so the
orders cannot currently differ.

Consequences:

* enabling CNPro cannot alter host behaviour by a single class;
* a new control-model family is one module plus one list entry;
* nothing is order-dependent at import time.

**Vendored, not patched.** `patchers/controlnet_impl.py`,
`ipadapter_impl.py` and `lllite_impl.py` are CNPro's copies of the injection
paths. Vendoring is deliberately preferred to runtime patching: it does not
depend on the host's internal structure at call time, so it fails at *import*
with a clear error rather than at *sampling* with wrong pixels.

---

## 3. Porting to ComfyUI

**What ports unchanged:** all of L1, plus `javascript/weight_profile.js` and
`javascript/weight_mask.js` — both are dependency-free vanilla and touch the
host through a handful of hooks, not through imports.

**What is rewritten:** L2 (one adapter module + patchers against ComfyUI's
ControlNet objects).

**What does not port:** the Gradio UI shell and ForgeCanvas integration. A
ComfyUI adapter returns `None` from `canvas_widget()`; the UI layer is simply
not built, and the engine runs headless off node inputs.

The seam that makes this work: ComfyUI's ControlNet applies residuals through
the same `{'input': [...], 'middle': [...], 'output': [...]}` dict shape this
engine already weights. `compute_controlnet_weighting` is index math over those
lists — it does not know or care which application produced them.

Practical order for a port: (1) copy `cnpro_core/`, (2) write
`cnpro_host/host_comfyui.py`, (3) write a patcher that wraps ComfyUI's
`ControlNet.get_control`, (4) expose the profile string as a node widget. The
grammar and the parity corpus come along untouched, which means the curves a
user drew in Forge evaluate identically in ComfyUI.

---

## 4. Supporting DiT models (Flux / Qwen / Z-Image)

**Status: Z-Image is implemented** (`patchers/zimage.py`), both generations:
**v1 Union** (6 sites, 16-channel) and **v2.1 Union / lite / Tile / SAM**
(15 or 3 sites, 33-channel, plus a second injection stage). Verified end to end
by generating against a line-drawing control (`tests/test_generate_live.py`).
What follows was written as a prediction before that work; it is kept because it
held, with the corrections noted inline.

The widget was built against UNet ControlNets. DiT support divides cleanly, and
the division is already reflected in the code.

### Ports unchanged

* **The step axis.** Flow-matching models still have a sigma schedule and a
  percent→timestep map. `build_weight_profile_lookup` is schedule-agnostic.
* **Cosine, multi-phase (cosine / Fejér / von Mises), wave convergence,
  response exponent, scale range** — pure shape features on the step axis.
* **The depth-drift curve.** It only reparameterizes the depth curve's argument,
  so it inherits whatever the depth axis means on the host. On a DiT it is
  arguably the more honest of the two controls: "the control migrates from
  abstract blocks to concrete ones as sampling proceeds" says nothing about
  resolution, which is the claim the coarse/mid/fine labels cannot support
  there.
* **Weight masks.** DiT residuals are per-token and tokens are a 2D latent grid,
  so a spatial mask scales them exactly as it scales pixels.

### Ports mechanically, with changed meaning

* **The depth curve.** `depth_fraction_of_residual` / `depth_fraction_of_unet_block`
  return a normalized 0..1 depth. For a DiT, `n` is the block count
  (Flux: 19 double + 38 single) instead of the residual count. The algorithm is
  unchanged; only `n` moves.
* **Band curves (coarse/mid/fine).** These are a *quantization* of the depth
  axis, and the code treats them that way — `band_of` is derived from the same
  mapping, never hand-tabulated. **But the semantics change**: in a UNet the
  bands are resolution tiers (composition/form/texture) because the encoder
  downsamples. A DiT runs every block at one token resolution, so depth means
  abstraction, not spatial frequency. The curve applies; the *labels* would be
  lying. Treat "coarse/mid/fine" as a UNet-only presentation of the depth axis
  and expose depth directly on DiT hosts.

### Does not port

* **The balance profile.** Flux-dev/Krea/schnell are guidance-*distilled* and
  Z-Image-Turbo is few-step distilled: CFG is 1 and there is no uncond row to
  bias. `supports_balance_profile` must read **unavailable**, not neutral — a
  flat 0.5 on a CFG-free sampler silently means nothing. Qwen-Image, which has
  true CFG, keeps it.
* **In-context editing models** (Flux Kontext, Qwen-Image-Edit, FLUX.2-Klein)
  condition on reference-image *tokens*, not per-layer residuals. There is
  nothing to schedule; this is a different feature category, not a port.

### Adding a DiT ControlNet

1. Add `cnpro_host/patchers/<family>.py` with a patcher declaring its
   `supports_*` flags honestly.
2. Add one entry to `registry.py::_types()`.
3. Resolve the depth of each injection site — **do not** add the family to a UI
   whitelist. The UI reads capability flags only to warn; it never gates on
   model type.

Nothing in `cnpro_core/` should need to change. If it does, the change probably
belongs in the new patcher.

### What Z-Image actually cost (and what the prediction missed)

`cnpro_core/` gained two functions and lost nothing:
`depth_fraction_of_ordered_site` and `band_of_depth_fraction`. Both are pure, the
module still has **zero import statements**, and every existing UNet mapping is
byte-identical (pinned by `tests/test_residual_layout.py`).

Three things the section above did not anticipate:

* **The residual rank changes.** `compute_controlnet_weighting` unpacked
  `B, C, H, W` and broadcast weights as `w[:, None, None, None]`. DiT residuals
  are `[B, tokens, dim]`. This is why `ResidualLayout` exists — it is the
  smallest surface that absorbs rank, mask projection and depth addressing at
  once, and it is what Flux/Qwen/Krea 2 reuse rather than re-derive.
* **A spatial mask has to mean something for caption tokens.** The joint sequence
  is `[caption][image][padding]` and only the middle span is spatial. Caption
  residuals steer the whole image through attention, so leaving them unmasked
  breaks restrict-to-painted; zeroing them makes a small painted region behave
  differently *in kind* from a large one. `TokenResidualLayout` gives them the
  mask's spatial **mean**, which is the only choice that keeps both engine
  invariants exact (all-ones is a true no-op, all-zeros is fully off).
* **This host's DiT silently ignores `control`.** `backend/nn/lumina.py::NextDiT`
  and `krea.py::SingleStreamDiT` have no `control` parameter — but they do have
  `**kwargs`, so the host computes residuals, passes them in, and they vanish.
  No exception, no warning, just a ControlNet that appears to work. CNPro
  therefore injects with `torch.nn.Module` forward pre-hooks on the live model
  instance, installed at patch time and removed after sampling.
  `backend/nn/flux.py` and `qwen.py` *do* consume `control`, so those two need
  no hooks at all — only a layout and a loader.

### The two Z-Image generations

| | v1 Union | v2.1 Union |
|---|---|---|
| conditioning | 16 ch (control latent) | 33 ch = control 16 + mask 1 + known pixels 16 |
| control layers | 6 at `[0,5,…,25]` | 15 at `[0,2,…,28]` (lite: 3 at `[0,10,20]`) |
| refiner blocks | plain - refine only | **control blocks** - also emit hints |
| extra stage | none | hints into the base `noise_refiner[0,1]` |
| CNPro depth bands | 2/2/2 | 5/5/5 |

Two things worth knowing before touching this:

* **The 33 channels are zero-padded for plain control**, which is not a shortcut:
  upstream's own non-inpaint pipeline does exactly that, and it is semantically
  exact (mask 0 = "nothing is already decided", known-pixels 0 = "nothing to
  show you"). This is the verified-good path and the default.
* **The hole (inpaint) conditioning is plumbed but DOES NOT WORK**, and is off
  behind `CNPRO_ZIMAGE_INPAINT=1`. It reads the host's own img2img context
  (`p.image_mask` + `p.init_images[0]`, honouring `inpainting_mask_invert`) - so
  no new UI and no invented convention - and the channels demonstrably reach the
  model. But on a real inpaint the masked region comes back **solid black**: the
  model reproduces the blacked-out void instead of filling it. That happens with
  BOTH mask polarities, so it is not the sign error the code was written to
  avoid. `_inpaint_enabled()` carries the leads worth trying next; the strongest
  is that the base model gets no inpainting conditioning of its own on this host,
  so the ControlNet is the only thing told "there is a hole" and it draws one.
* **v2.0 is refused by filename.** Its tensors are byte-identical to 2.1 and only
  the forward pass differs; upstream calls 2.0's wiring a bug that "caused double
  forward pass and slow inference". The filename is used ONLY to refuse - never
  to decide how to run something. A wrong refusal rejects a working file; a wrong
  guess produces plausible wrong images forever.

The same code path covers **SAM** (`neuralvfx/Z-Image-SAM-ControlNet`, the
`comfy-ui-patch/` file - the repo's other file is a full model merged with a
control tower and is refused as such) and **Tile**.

**Krea 2 is still not a port.** It has no per-layer residuals: its control is a
Control-LoRA plus a VAE-encoded latent concatenated onto the input projection
(16→32 channels). The step axis could scale that latent; depth, band, per-layer
masks and balance have nothing to act on. It belongs in a small separate
extension, not behind this widget — two thirds of the controls would be dead.

---

## 5. The canvas seam

The image-canvas widget is the one host API that has actually broken across the
Forge family, so it has its own normalization layer.

* `javascript/canvas_adapter.js` — maps host method names to CNPro's canonical
  ones. ForgeNeo renamed `uploadBase64`→`loadImage` and
  `on_img_upload`→`updateBackgroundImageData`. It is a pure function, called by
  `canvas_extra.js`, **not** a poller — `class X {}` in a classic script creates
  a *lexical* binding, so `window.ForgeCanvas` is never set and a second poller
  would race the adjustment layer's bootstrap.

  **The alias direction is the whole trick.** CNPro works by *wrapping* these
  methods, so the alias must point at the name the **host** calls, not the name
  CNPro calls: the canonical name receives the host's implementation, and the
  host's own name becomes a delegate into it. Pointed the other way — the
  obvious way — every method-presence check passes and every canvas tool is
  inert, because the host only ever calls `loadImage` and the wrapper sits on
  `uploadBase64`. See section 8c.

* `javascript/canvas_tools.js` — **the tool registry**: one object per tool
  (button, menu rows, overlays, deferred reason). Adding, removing or exchanging
  a tool is one array entry. It is data only — no DOM, no behaviour.

* `javascript/canvas_nodes.js` — the renderer. Builds the markup from the
  registry, and derives the id contract (which ids exist, which may stay hidden)
  from the *same* objects, so the two cannot drift. It resolves the registry
  **lazily**, because extension scripts load in filename order and
  `canvas_nodes` sorts before `canvas_tools`.

* `style.css` carries the fork's `canvas.css` additions. Extension stylesheets
  are injected at end of `<body>`, after the host's `canvas.css` in `<head>`, so
  equal-specificity rules here win without `!important`. Every canvas geometry
  rule is scoped to `[data-cnpro-nodes="1"]` — the marker the renderer sets on a
  container it injected into — so the host's own canvases are untouched and the
  rules out-specify `canvas.css` without `!important`.

### 5a. Button geometry is structural, not corrected

The toolbar's buttons are a fixed square (`--cnpro-btn`), `flex: 0 0 auto`,
`padding: 0`, `margin: 0 !important`, and centre their content with
`display: grid; place-items: center`. Each property closes off one way a button
could be moved by its own content: an icon cannot size the box, a glyph's font
metrics cannot shift it off the baseline, and gradio's `.prose` bottom margin —
which it zeroes on `:last-child`, so whichever button was last sat higher than
the rest — cannot reach it. A new tool can ship text, emoji or inline SVG and is
square and aligned without anyone remembering to check.

Measured, not asserted: `test_toolbar_layout.py` runs the real template, both
real stylesheets and the real injectors in headless Chromium and reports the
boxes. On the pre-fix stylesheet it finds **15 different button sizes** (the
colour-picker button being 27×19 among 24×24 neighbours — the "light point
button is misaligned" report, in numbers); after, 22 buttons all 24×24 on one
baseline.

The one deliberate exception is `margin-left` on group separators, and it is a
worked example of why the measurement matters: `margin: 0 !important` and
`margin-left: <gap> !important` are two `!important`s pointing opposite ways, so
the winner is decided by *specificity*. The gap rule has to match
`.forge-btn.forge-adjust-gap` inside the toolbar to out-specify the reset;
written the obvious way it loses and the tool groups silently run together.

### 5b. The width contract

> **A tool menu is exactly as wide as the usable canvas, and its rows are
> elastic: they pack as many per line as fit, then stretch to fill the line.**

Two facts fix this, and neither is negotiable:

1. `.forge-image-container` is `overflow: hidden`. A menu wider than the canvas
   is **clipped**, not shown. The canvas width is a physical ceiling.
2. Anything narrower wastes width that cannot be recovered anywhere else, and
   the cost is paid in **height** — at 190px per row a 520px canvas fits two
   rows per line, so the five-row edges menu needs three.

Ceiling and floor are the same number, so there is nothing to choose.
`canvas_nodes.js::syncMenuWidth` measures `container.clientWidth` minus the
toolbar's own horizontal padding (read, not assumed) and writes it to every menu
as an inline `width`, re-run by a `ResizeObserver` on the container. How many
rows land per line is then a *consequence*, owned by one number in the
stylesheet — `--cnpro-row-basis` — and by nothing else.

**Three earlier versions, and why each was wrong.** Every one of them made the
width depend on something that is not the question. The question is only ever
"how much room is there".

| Version | Failure |
|---|---|
| `width: 0` + `min-width: 100%` | the percentage resolves against an indefinite containing block in gradio, degrades to `min-width: auto` = **min-content**: one row per line |
| pinned to the **button row** | caps the menu at the buttons' width even when the canvas is far wider |
| `width: max-content` between a measured floor and ceiling | floor was the button row again, so on a narrow canvas the gamma slider and its reset still could not share a line |

The measurement is written **inline**, not only as a custom property: the
min-content collapse could not be reproduced outside the running app (not with
the static toolbar, not with the scribble row hidden, not with the measurement
disabled), so the repair deliberately does not depend on knowing why. An inline
declaration outranks every stylesheet rule short of `!important`.

**A `min-width` on a row is banned, and this is the sharpest lesson in the
section.** A flex container's min-content is the largest of its items' minimums,
so a floor on a *row* silently becomes the floor of the *menu*.
`.forge-range-row-wide { min-width: 260px }` therefore made the gamma menu 260px
wide **and** left nothing for its 22px reset button — narrow and spilling, two
symptoms reported separately, one cause, unfittable at any canvas size. Rows are
`flex: 1 1 var(--cnpro-row-basis); min-width: 0`, and `test_style_sheet.py`
fails on any non-zero row minimum.

Menu rows are `label | slider` on one line. The host's `.forge-range-row` is a
*column*, which is right for the big scribble sliders and wrong for a tool menu:
stacked, the five edge-mask rows measured 54px each and pushed the image out of
view; side by side they are 20px.

**The track is a fixed 100px and precision comes from the WHEEL.** Every slider
in every menu steps by its own declared `step` per notch — 1 for the 0..100 and
0..255 ranges, 0.1 for rotation, 5 for gamma, 0.01 for the weight mask — so each
slider stays consistent with what dragging it does and a new tool needs no wheel
code. That is what makes a short track affordable: it is not a less precise
slider, it is a more precise one that costs less width. The row pitch
(78 + 6 + 100 = 184px) is then what decides rows per line.

Two things about the wheel are easy to get wrong and are pinned by tests:

* **The host zooms the canvas on ANY wheel over the container**, and the toolbar
  is a child of that container, so the handler must `stopPropagation` or every
  slider adjustment also zooms the image behind it. The listener therefore lives
  on the *toolbar*, where bubbling guarantees it runs first.
* **Both `input` and `change` must be dispatched.** The wiring is split — most
  handlers redraw on `input`, weight_mask.js's feather commits on `change` — so
  firing one leaves the other half silently dead.

The wheel work also turned up a slider that could never reach its own maximum:
gamma was `min=-2322 max=2322 step=5`, and a range input snaps to the step grid
from `min`, so with a span of 4644 (not a multiple of 5) the highest reachable
value was 2318. Nothing had ever reported it, because a slider that stops four
units early looks exactly like a slider. It is ±2320 now — 928 whole steps, both
ends reachable, 0 still exactly on the grid.

The one menu that stays vertical is the **layer list** — one layer per row. The
rows are a z-order stack, topmost first; laid out left-to-right they read as an
unordered set and the ▲/▼ buttons move a layer "up" while it travels sideways.
Compactness is not worth making the model unreadable.

---

## 6. Known deviations from the original fork

| Item | Status |
|---|---|
| `xyz_grid_support.py` | parked as `.deferred` — axis names still point at the old extension |
| Topaz canvas tools | not shipped; the availability probe 404s and the buttons stay hidden, which is the designed behaviour |
| Host's builtin ControlNet | must be disabled — CNPro reuses its `elem_id` scheme deliberately, to keep ui-config keys and JS selectors identical |
| Z-Image Fun-ControlNet v2.0 / 2.1 / 2.1-lite | recognised and **refused**. Its 33-channel conditioning is inpainting-shaped (latent + masked latent + mask) and has no UI here; worse, v2.0 and v2.1 ship identical key sets and differ only in whether the refiner is applied as intended, so the file cannot say which it is |
| Z-Image per-unit prompt | `supports_unit_prompt = False`. The control tower does attend over caption tokens, but it takes them from the base model's already-refined sequence — re-refining a *different* caption is a second feature, not a flag |

---

## 7. Invariants worth re-reading before changing anything

`MAINTENANCE.md` (shipped alongside this file) documents the widget's own
invariants — the profile grammar contract, the painter/server coherence state
machine, the gradio echo gate, the static-layout rule. Those are about the
widget's internals and remain authoritative. This file is only about the
packaging around them.

The two added here:

* **L1 imports nothing.** Verified by running it under a bare interpreter.
* **L1 and L3 never import the host.** Verified by grep; if that grep ever
  returns a line, the adapter is missing a name and should grow one.

And two more added by the Z-Image work:

* **The UI never branches on patcher type.** `scripts/cnpro.py` imports no
  concrete patcher at all any more. Mask routing used to ask
  `isinstance(params.model, ControlNetPatcher)`, which silently sent every
  *future* residual patcher down the attention path; it is now the
  `masks_via_advanced_weighting` flag. If you find yourself reaching for a class
  name in L3, the missing thing is a flag on `patchers/base.py`.
* **A patcher must fail at patch time, not at sampling time.** Both new refusal
  paths (`Injector.install` checking the model really is a NextDiT,
  `config_from_state_dict` refusing v2.x rather than guessing) exist because the
  alternative is an image that looks plausible and is wrong. Prefer a readable
  exception over a best effort, every time.

## 8. The silent-miss rule

Three separate bugs in this extension have had one shape. Worth naming, because
the shape is what recurs — the individual bugs never look alike.

| Declared here | Honoured there | Drift produced |
|---|---|---|
| `canvas_nodes.js` injects mask buttons with `.forge-wmask-control` | `canvas_extra.js` revealed `.forge-adjust-control` | the entire weight-mask UI absent from the toolbar |
| `controlnet.py` called `memory_management.get_computation_dtype()` | no such function exists | every SD1.5/SDXL ControlNet reported as "unsupported file" |
| `base.py` defaults every capability flag to `False` | a patcher forgets to declare one | that feature silently stops working for that patcher |
| `lllite.py` passes `depth_profile=` to `load_lllite` | `load_lllite` never declared the keyword | `TypeError` on **every** ControlLLLite generation (found 2026-07-27) |

In the first three, **nothing raised.** An empty `querySelectorAll` is not an
error, a swallowed `AttributeError` is not an error, and an inherited `False` is
not an error. Each failure was invisible until someone happened to look for a
feature and notice it was gone.

The fourth row is the same shape with the opposite volume, and it is worth
keeping for that: a declaration honoured nowhere raised on the very first call,
loudly, with a perfect message — and still survived, because ControlLLLite is
rare enough that nobody ran it. Silence is what usually hides these; obscurity
does the job just as well. The seam that mattered is that `lllite.py` calls
`load_lllite`, not the `load_control_net_lllite_patch` underneath it, so the
profile had been added to the function that reads it and not to the one that is
called. Both new profiles are now named in both.

The common cause is not carelessness. It is a **default that means "absent"**,
combined with **no check that the declaration was made deliberately**. Both parts
are needed; either alone is safe.

So, three rules:

1. **Never let "absent" be the default of a declaration.** The toolbar now
   reveals every injected control *except* an explicit `DEFERRED` few, so a new
   button is visible by default. A wrongly-visible button is noticed in seconds;
   a wrongly-hidden one is invisible by definition.
2. **Never collapse distinct failures into one outcome.** `load_control_model`
   keeps *declined* (returned `None` — normal, silent), *broken* (raised — a
   CNPro bug, named and re-raised), and *refused* (raised with
   `cnpro_recognised` — a deliberate rejection, propagated with its own message)
   apart. Those are three different problems for the user and used to be one.
3. **Make "I did not consider this" impossible to express.** Every patcher must
   state every capability flag in its own class body, including the `False`s.
   Inheriting a default and deciding on a default look identical in the source;
   only one of them is a decision.

Each rule has a test that fails on the pre-fix code
(`test_toolbar_contract.py`, `test_patcher_contract.py`). When you add a
declaration-honoured-elsewhere pair, add the check with it — the cost of the
check is minutes and the cost of the drift is however long nobody looks.

### 8a. A fourth instance, caused by the fix for the first

The contract that fixed the mask buttons derived its ids like this:

```js
.match(/id="[A-Za-z0-9_]+_forge_mixin"/g).map((s) => s.slice(4, -14))
```

`_forge_mixin"` is **thirteen** characters. Every id lost its last letter,
`layersButton` became `layersButto`, every `getElementById` returned `null`, and
the **entire toolbar** disappeared — a strictly worse outcome than the bug being
fixed.

`test_toolbar_contract.py` passed throughout. It verified that `TOOLBAR_IDS`
existed and was derived from the markup by regex. Both were true. It never
checked that the derivation produced *correct* ids, because it read the
JavaScript **as text** and never ran a line of it — while `node` was on PATH the
whole time and another suite in the same directory was already using it.

So a fourth rule, and the sharpest one:

4. **Test what runs, not what is written.** Source inspection can prove a
   mechanism is *present*; only execution proves it *works*. Anything with real
   logic — parsing, index arithmetic, DOM resolution — gets a harness that
   executes it (`tests/toolbar_contract_js.js`). Grep-level checks are fine for
   *policy* ("this call site exists", "this selector is not used"), and actively
   misleading for *behaviour*.

Two more habits came out of it, both cheap:

* **Never slice fixed offsets off a regex match.** Use a capture group. A capture
  group cannot be off by one; a hand-counted offset can, and was.
* **Cross-check derived data against an independently produced list.** `OWNED_IDS`
  is written by the generator, `TOOLBAR_IDS` is parsed at runtime — same source,
  different means, so comparing them catches a mangling parser at *script load*,
  in one console line, instead of as a blank toolbar with no explanation.
  `canvas_nodes.js` does this in `selfCheck()`.

A test harness must also **survive a broken module and report it**. The first
version of the node harness indexed the DOM stub directly, so a mangled id list
threw `TypeError` from the harness — catching the bug but diagnosing nothing.
It now resolves every lookup defensively; a test that crashes is most of the way
back to the silent miss it exists to prevent.

### 8b. Executing the code was still not enough

Rule 4 said "test what runs". The harness added for it (`toolbar_contract_js.js`)
executes the module against a DOM stub — **that the test itself populates from
`OWNED_IDS`**. It proves reveal and audit work *given that the nodes exist*, i.e.
it assumes the thing most likely to be broken. Nothing ran `inject()` against the
host's real template.

`test_toolbar_dom.py` does, in jsdom, from the two real inputs
(`modules_forge/forge_canvas/canvas.html` + `javascript/canvas_nodes.js`). It
immediately found a third bug: `audit()` tested visibility with
`offsetParent === null`, which is null for any element in a collapsed container —
and CNPro's canvas lives in a gradio accordion that is **closed by default**. The
audit would have reported all fourteen controls broken on every attach. A check
that cries wolf is worse than no check; it trains everyone to ignore the one real
failure. It now uses `getComputedStyle(node).display`, which sees CSS (including
`display: none !important`) but is blind to a merely-collapsed ancestor.

So rule 4 sharpens to: **execute the real code against the real inputs.** A stub
you populated yourself only tests the half you already understood.

Two structural fixes came out of the same round, and they matter more than any
individual bug:

* **`attach()` had three bare `return false` lines** covering 23 required nodes.
  One missing node abandoned the entire widget — no toolbar, no Topaz probe, no
  listeners — silently, while a MutationObserver retried forever. It now names
  exactly which nodes are missing, once per container. "Attach bailed" and "one
  control is hidden" look identical on screen and are completely different
  problems.
* **Diagnostics must not be able to break what they diagnose.** The reveal and
  the audit are each wrapped in `try`/`catch`, and the class-based sweep runs
  unconditionally before the contract sweep. `attach()` is ~1800 lines in one
  function with no error boundary: anything that throws kills everything below
  it, and the audit is the *last* statement — so an exception anywhere above
  disables the very check meant to catch it.

Also: `node.style.display = ''` is **not** "make visible". It clears the inline
override and defers to CSS; a rule like style.css's
`.cnet-output-mask-group .forge-adjust-control { display: none !important }`
still wins, deliberately.

### 8c. Three more, and the sharpest rule yet

The user's report was: "crop does not work at all, light points open defective
layout and none of controls work, layers don't work, mirror does not work." One
sentence, three unrelated causes, none of which any test could see.

| Declared here | Honoured there | Drift produced |
|---|---|---|
| `canvas_extra.js` wraps `proto.uploadBase64` | the host only ever calls `loadImage` | **every canvas tool inert** — menus open, sliders move, no pixel changes |
| the fork's `canvas.css` rules | extracted by a **line**-diff, which dropped shared selector and brace lines | orphaned rule bodies, 198 `{` vs 199 `}`, every tool menu unstyled |
| `levelsBox` markup + wired `input` handlers | no button anywhere opened it | two working sliders unreachable, in the fork and here |
| `weight_mask.js` latched setup failure because "the template nodes render together with the container" | the toolbar is **injected** at attach time, by another module, after the container | G/C/M/F injected, revealed, styled — and permanently inert |

The first is the interesting one. The adapter *did* check that every method it
needed existed, and every check passed — the whole time the feature was dead.
`typeof proto.uploadBase64 === 'function'` was true; it was true *because the
adapter had just defined it*, pointing the wrong way. So:

5. **A wrapper is worth nothing on a method the host does not call.** When you
   normalise a renamed API, alias in the direction that puts your code on the
   *call path*, and test it by making the call the way the **host** does. Every
   presence check in the world cannot distinguish "wrapped" from "wrapped
   something nobody uses"; one executed call can, and
   `tests/test_canvas_adapter.py` is four lines of exactly that.

And two more, both about tests rather than code:

6. **Check supply against demand in BOTH directions.** The audit walked the
   registry asking "is this in the DOM". Nothing asked the reverse — "does the
   wiring reach for an id nobody declares" — and `levelsBox` lived in that gap
   for as long as it existed. `toolbar_dom_js.js` now compares the two id sets
   each way; the first run of that check found the dead id it was written to
   find, plus one more.

7. **If the claim is about pixels, measure pixels.** Three toolbar tests were
   green while the buttons came in fifteen different sizes and every menu row was
   54px tall. They read the JS, ran it on a stub, and parsed the CSS as text —
   all useful, none of them able to see a box. `test_toolbar_layout.py` puts the
   real template, both real stylesheets and the real injectors in headless
   Chromium and reads `getBoundingClientRect`. It is hermetic and takes two
   seconds, which is the point: "only a running app can tell you" was never true
   here, it was just never tried.

8. **A comment can outlive its premise, and a `return` that rests on one goes
   with it.** `weight_mask.js` set `__cnetWmaskInit = true` *before* calling
   setup, so a failed setup was permanent, and said exactly why: "the template
   nodes render together with the container, so a retry can never succeed
   later". That was true — until CNPro started injecting the toolbar at attach
   time, from a different module, on a different trigger. The premise died in
   that refactor and the `return` it justified did not. Whichever observer won
   the race decided whether the weight-mask tools worked at all, and when this
   one won they were injected, revealed, styled and dead, with one console
   warning that scrolled past on load. When you move *where* something comes
   from, grep for the code that assumed the old answer — the compiler cannot,
   and the comment will still read as if it were correct.

9. **When you cannot reproduce it, stop needing to be right.** The menu-width
   collapse was reported from the running app and did not reproduce in that same
   headless harness under any of four hypotheses (static toolbar, hidden scribble
   row, measurement disabled, plain page). So the fix is not "the correct CSS for
   the cause I deduced" — it is a measured, **inline** width that outranks
   whatever the cause turns out to be. Prefer the repair whose correctness does
   not depend on a diagnosis you could not confirm, and say plainly which one you
   shipped. See section 5b.

### Test suites

| File | Needs | Pins |
|---|---|---|
| `test_profile_parity.py` | node | editor JS ≡ python profile evaluation |
| `test_residual_layout.py` | torch | UNet weighting unchanged; both mask invariants |
| `test_zimage_config.py` | nothing | config sniff ≡ released checkpoint; v2.x refused |
| `test_zimage_module_tree.py` | torch + host | all 136 tensors match, names and shapes |
| `test_zimage_injection.py` | torch + host | hooks hit the right blocks; removal is bit-exact |
| `test_canvas_adapter.py` | node | **the alias direction**: a host-originated `loadImage()` reaches CNPro's wrapper; classic Forge untouched; normalize idempotent |
| `test_style_sheet.py` | nothing | style.css parses (no orphaned bodies, balanced braces); every rendered class is styled or selected; canvas geometry scoped to CNPro's canvases |
| `test_toolbar_contract.py` | node | **executes** the renderer + registry: rendered ids match the registry id for id, classes survive, reveal works, audit detects each failure mode |
| `test_toolbar_dom.py` | node + jsdom | **injects into the host's real canvas.html**: every node lands, attach gate satisfied (parsed from `canvas_extra.js`, not restated), controls visible, audit quiet, ids match the wiring **both ways**. Skips loudly without jsdom (`npm install --no-save --prefix <dir> jsdom`, `CNPRO_TEST_NODE_PATH=<dir>`) |
| `test_toolbar_layout.py` | node + playwright | **measures pixels** in headless Chromium with both real stylesheets: buttons square and identical, one baseline per row, group gaps survive, menus fit the toolbar, rows single-line. Hermetic — no webui |
| `test_patcher_contract.py` | torch + host | every patcher declares every flag; registry keeps declined/broken/refused apart |
| `test_control_merge_parity.py` | torch + host | `control_merge` ≡ the pre-subclass implementation across 16 configurations, chained and not; the host's inherited weighting surface is never set; each host control class transplants to CNPro's |
