# ControlNet Pro

The most expressive ControlNet extension for **Forge Neo**.

A control usually reaches you as a weight slider and a start/end range: one
number and one interval, for a mechanism that acts across *space*, across
*sampling steps*, and across *network depth*. CNPro opens all three, with editors
built for them — a curve editor for the step axis, a paint tool for the spatial
one, a second plot for depth. Then it lets you stack units, give each one its own
prompt, and mask each one to its own region of the output.

Where the leverage comes from: a ControlNet does not hand the model one nudge, it
injects a *residual at every layer it is wired into*, at every sampling step, and
those residuals simply **add** — which is why several controls compose in the
first place. Those injection sites are ordered by depth: the deep ones decide
composition and pose, the shallow ones decide texture and surface. A single
weight collapses that whole field into one scalar. **A profile is a curve over it
instead** — one over steps, one over depth, one painted over the image — and the
unit runs their product. The field was always there; CNPro is the instrument
built to play it.

![The CNPro unit](resources/ui-unit.png)

---

## The three axes

|  |  |
|---|---|
| **Space** — paint control strength straight onto the image. Rainbow hue is the weight, red 1 → violet 0; brush size, feathering and an eraser are in the same menu. Four independent mask slots per input: one global, three per depth band. Anything painted means the control never escapes the paint. | ![weight mask](resources/ui-weight-mask.png) |
| **Steps** — a curve editor. Click to add points, drag to move, double-click to delete; drag a segment's green midpoint to bend it into a parabola. Presets down the left: step, cosine, multi-phase. A vertical response slider bends the whole curve toward high or low values, and the range selects reach down to **−1**, where the control actively pushes away. The dotted verticals are your real step cells; the dot on each curve is the value that step reads. | ![the weight editor in multi-phase](resources/ui-profile-multiphase.png) |
| **Depth** — its own plot, its own axis: X is the injection layer from fine to coarse, Y a per-layer multiplier. Draw it and the unit runs your step curve *times* your depth curve — a time curve and a depth curve, separable, both yours. | ![depth](resources/ui-profile-depth.png) |

Prefer buckets to a curve? The same depth axis is quantized into three **band
profiles** — coarse / mid / fine, one step curve each, drawn together on one
plot, selected by the coloured buttons under the presets.

![bands](resources/ui-profile-bands.png)

---

## Multi-phase amalgamation

**This is the feature to steal.** A unit takes up to five **Input** tabs. Each
one is preprocessed on its own and patched in as its own control on the same
UNet — a unit is no longer "one image, one control", it is a small ensemble.

Run that ensemble flat and every reference pulls on every step. They compete, and
the model resolves the fight by *stitching*: each donor arrives as a patch, with
an edge where it meets the next one.

**Multi-phase** hands them the schedule instead. Turn on the cosine preset, press
multi-phase, and each input runs the same wave shifted a little further along, so
they take turns steering — input 1 opens on the first step where composition is
decided, the rest come in behind it. The pad sets how many waves cross the
schedule; the editor draws a ghost curve per loaded input so you can see the
hand-off before you generate. Their pulls still sum to exactly the envelope you
drew, so adding a fifth reference does not quietly make the unit weaker.

The result is not a blend of images. It is one thing that looks like it grew that
way.

### Two animals

![amalgam](resources/example-amalgam.jpg)

Same seed, same two references, same strength. On the left the flat run: an owl
bib grafted onto a fox, seam and all. On the right, multi-phase — the plumage and
the fur are the *same coat*.

### Four animals

Owl, fox, chameleon, mandrill. Feathers, fur, scales, bare skin — nothing in
common, which is the point.

![chimera](resources/example-chimera.jpg)

The flat mean gives you a face with a feather cap sitting on it. Every
multi-phase run gives you an animal: scaled cheeks that flow into fur, a mandrill
muzzle that belongs to the head it is on, the chameleon's casque as part of the
skull rather than a hat. One wave is the strangest and the most convincing; four
waves interleaves fastest and comes out smoothest.

### Four working adults, and the depth curve on top

Same machinery on people, and now with the second axis engaged. A chef, a
carpenter, a doctor and a teacher go in. All four outputs run the *same* four
references through the *same* one-wave interleaving — the only thing that moves
along the bottom row is the depth curve on the purple plot.

![professions](resources/example-professions.jpg)

Worth knowing before you try it: **four faces average into one plausible person
no matter how you schedule them.** People converge where animals diverge —
the model has a much stronger prior for "one human face" than for "one creature".
What the depth curve still buys you is everything *around* that face: which
reference's build, cut and cloth survives. Deep layers leading brings the
doctor's collar and the carpenter's apron into one silhouette; shallow layers
leading brings their fabrics and patterns on a lighter frame; the middle of the
stack trades the collar and the cut.

Reorder the inputs with the ← button under the canvas: input 1 gets the
unshifted wave, so which reference opens the composition is a creative decision,
not bookkeeping. Un-tick an input's checkbox in its tab title to A/B it without
clearing anything.

<sub>Animals: IP-Adapter Plus SDXL, plot range 0 → 2. People: IP-Adapter Plus
SDXL, plot range 0 → 1.4, 896×1152. All: RealVisXL V5.0 Lightning, 8 steps,
DPM++ SDE Karras, CFG 2. Every reference was generated by the same model — no
external assets, no real people.</sub>

---

## Units stack

One unit is not the story. **Units combine**, and a unit is only a *kind* of
control — geometry, depth, identity, style. Each one carries its own hint, its
own model, its own profiles, its own masks and its own prompt, and CNPro gives
you the last piece the stack was missing: **an output mask per unit**, so a
control can be told not just how hard to pull but *where in the picture it is
allowed to*.

### Different kinds of unit: canny + IP-Adapter

![multi-unit](resources/example-multiunit.jpg)

Canny alone holds the geometry and nothing else — grey daylight, no aurora.
IP-Adapter alone brings the whole look and loses the lighthouse. Together you get
both, but the aurora bleeds down into the rock and the sea until the whole frame
is teal. Paint an output mask over the sky on unit 2 and the look stays where you
put it: aurora above, natural rock and water below, structure crisp throughout.

That is the loop worth internalising — **stack for capability, mask for
placement.**

### One object per unit: an apple and a pear

Three units. Unit 1 is a canny hint of a table with two fruits on it — that is
the geometry, and nothing else. Units 2 and 3 are IP-Adapters, one holding a
green apple, one holding a red pear, each masked to its own fruit.

![apple and pear](resources/example-applepear.jpg)

Canny alone puts two fruits in the right shapes and picks their colours at
random. Add both references with no masks and they have nowhere to land, so a
ghost fruit smears across the whole frame. Paint one output mask per fruit and
each reference goes exactly where you put it — green on the left, red on the
right, both still in the silhouettes the hint drew.

The last panel adds the fourth ingredient. Canny is a **real ControlNet**, so it
has its own cross-attention and CNPro gives it its own **P** and **N**
textboxes — here, *"a bright green apple on the left and a deep red pear on the
right"*. That text is encoded with the model's own text encoder and fed to the
control branch *in place of* the main prompt: the UNet still sees what you typed
at the top, but this control now reads its edges through different words.
Embedding strength, effect scale and a retention ramp set how hard it argues.

**Object-level control without a second pass, an inpaint, or a mask on the
latent.** Same hint, same seed, same main prompt in all four.

<sub>Canny weight 0.85, IP-Adapters 1.1, unit prompt at embedding strength 2.2,
effect scale 1.8, retention 1.0. Push those past roughly 2.6 / 2.0 and the
control branch stops producing a photograph — the useful band is narrow and worth
sweeping.</sub>

---

## One hint, three axes

One canny hint of a calm lighthouse. One prompt asking for a violent storm. The
only thing that changes between the panels is *how* the hint is scheduled.

![axes](resources/example-axes.jpg)

- **flat weight** — the classic. The hint wins everywhere, including the sea, so
  the storm you asked for never really arrives.
- **step curve** — drag the curve to zero at 40%. The hint sets the composition
  and then lets go. Same layout, real weather, real rock.
- **spatial** — paint the headland at full weight, leave the rest of the canvas
  unpainted. That structure is held exactly; the sea is free to break.
- **depth curve** — press the purple selector, keep the fine layers up and pull
  the coarse ones to zero. The hint's *surface* lands everywhere while the camera
  and the framing go wherever they like.

---

## The canvas is a tool in its own right

The image canvas is not an upload box with a clear button. It is a small
compositor, and it runs *before* the preprocessor, so anything you do here is
what the control model sees.

![canvas tools](resources/ui-canvas-tools.jpg)

- **Layers** — stack images on one stage. Drop, paste or push an image in as a
  new layer, click to select, drag to move, wheel to scale around the pointer,
  reorder and delete in any order. Every tool below operates on the flattened
  composite.
- **Pen** — draw straight into the active layer with a full colour picker,
  eyedropper, brush size, opacity and feathering, plus an eraser that cuts to
  transparency so lower layers show through. Per-stroke undo.
- **Levels and light points** — black and white point sliders, and three
  pickers: click a black point, a white point, or paint over neutral areas and
  let it kill the colour cast without touching brightness.
- **Gamma** — one wide slider on a fine logarithmic scale, double-click to
  reset. Enough range to rescue a hint that is nearly black.
- **Edges** — turn any photograph into a black-lines-on-white hint right on the
  canvas: edge opacity, mask opacity, sensitivity, line thickness and feathering,
  all live. Line weight follows edge prominence, so it degrades gracefully
  instead of turning to noise. Pair it with the invert toggle when a model wants
  the other polarity.
- **Crop, rotate, flip, grayscale, invert** — all non-destructive toggles. Crop
  drags border handles and applies when you toggle it off; nothing is thrown
  away until you say so.

Every slider takes the mouse wheel, at its own step size. The whole toolbar is
one row of squares that fades in when you point at the canvas, so none of this
costs you screen space while you are not using it.

---

## Everything else in the box

- **Balance profile** — a per-step curve for how much of the control lands on the
  conditional side versus the prompt side. Replaces the three-way Control Mode
  radio with something you can actually shape.
- **Per-input mute** — a checkbox in each Input tab's title. A/B a reference
  without clearing its canvas or its masks.
- **Detected-map cache** — preprocessor outputs are reused across Generate
  clicks when nothing that feeds them changed.
- **Round trips** — profiles, masks and settings survive infotext, PNG Info and
  the API, and unknown pieces are skipped rather than dropped, so an old image
  still pastes into a newer build.

---

## Install

```bash
cd <your-forge-neo>/extensions
git clone <repo-url> forge-neo-cnpro
```

Then **disable the built-in `sd_forge_controlnet`** in Settings → Extensions and
restart. This is not optional: CNPro deliberately reuses the built-in's element
ids so your `ui-config.json` keys and any JS selectors keep working, and the two
cannot coexist.

Requires `opencv-python`. The OpenPose editor additionally needs the
`forge_legacy_preprocessors` extension; everything else runs without it.

---

## Model support

| Type | Step | Bands | Depth | Weight masks | Output mask | Balance | Unit prompt |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ControlNet / ControlLora | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| T2I-Adapter | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| IP-Adapter (incl. InstantID) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| ControlLLLite | ✓ | ✓ | ✓ | — | — | — | — |
| Z-Image Fun-ControlNet Union v1 / v2.1 / lite / Tile / SAM | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |

Capabilities are declared per model family, **including the noes** — an
unsupported control warns you, it never silently does nothing. Only true
ControlNets have a text input, so the unit prompt is theirs alone; LLLite has no
mask route at all; and Z-Image is CFG-distilled, so balance reads *unavailable*
rather than neutral, because a balanced setting on a CFG-free sampler would mean
nothing.

---

## Honest limitations

- **Krea 2 is not supported and probably never will be here.** Its control is a
  Control-LoRA plus a concatenated latent — no per-layer residuals, so depth,
  bands and per-layer masks would all be dead controls.
- **Z-Image Fun-ControlNet v2.0 is refused by filename.** Its tensors are
  byte-identical to 2.1 and only the forward pass differs, so the file cannot say
  which it is. A wrong guess produces plausible wrong images forever.
- **Z-Image inpaint conditioning is plumbed but does not work** (behind
  `CNPRO_ZIMAGE_INPAINT=1`): the masked region comes back solid black.
- **A unit prompt re-conditions the control branch; it does not paint objects.**
  It changes how that control reads its hint. Pushed too far it stops being a
  photograph, and the usable range is narrower than the sliders suggest.

---

## License

GPL-3.0.

<small>

Descended from and building on work by others, with thanks:

- **ControlNet for A1111 / Forge** — Mikubill and contributors (GPL-3.0). The unit widget, the preprocessor plumbing and the API surface descend from it.
- **Stable Diffusion WebUI** — AUTOMATIC1111 (AGPL-3.0). The host, and the ForgeCanvas widget these canvas tools extend.
- **ComfyUI_IPAdapter_plus** — cubiq (GPL-3.0), by way of Forge's IP-Adapter extension. The IP-Adapter / InstantID injection path.
- **ControlNet-LLLite** — kohya-ss, by way of Forge. The LLLite injection path.
- **Fooocus** — lllyasviel (GPL-3.0). The inpaint patcher and its licensing notes.
- **Z-Image / Fun-ControlNet-Union** — Tongyi Lab and alibaba-pai (Apache-2.0). Checkpoint layout and config; the engine is re-expressed against this host's tensors.
- **"High Quality Edge Thinning using Pure Python"** — Lvmin Zhang, Stanford, 2023. Cited as the author asks.
- **stable-diffusion-ps-pea** and **sd-webui-openpose-editor** — huchenlei. The Photopea and pose-editor bridges.
- **Pytorch-Contrast-Adaptive-Sharpening** — Jamy-L. The CAS pass in the IP-Adapter path.
- **base64ArrayBuffer** — Jon Leighton, 2011 (MIT), and a base64→Blob helper by gauravmehla.

</small>
