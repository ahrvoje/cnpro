# ControlNet widget — maintenance notes

State consolidation of the local ControlNet widget rework (weight/balance
profiles, rainbow weight masks, canvas-integrated UI), written as preparation
for packaging the feature set as a public Forge extension ("addon"). This is
NOT the addon itself; it maps what exists, what must hold true, and how the
pieces would move.

## 1. Feature set

- **Control Weight Profile** — replaces the Control Weight slider + Timestep
  Range slider with a piecewise-linear curve editor (X = relative sampling
  step, Y = strength). Serialized as `x@y;x@y;...`, optional `Mx@y` parabola
  mid-control tokens, optional `C<osc>@<phase>` cosine token, optional
  `P` / `PF` / `PV<kappa>` multi-phase family token, optional
  `G<e>` response-exponent token, optional
  `|hi` / `|lo~hi` scale suffix mapping the normalized curve onto [-1, 2]
  (negative = repulsive control). Presets: step + the oscillatory ladder,
  driven by a
  square two-parameter pad sized to the plot height. The INVERT toggle was
  removed 2026-07-25 and the step preset generalized to cover it: the pad's y
  is now height AND direction — above the pad's middle the raised part of the
  step is on the RIGHT (0 then h, the old behaviour), below it on the LEFT
  (h then 0), with the height being the distance from the middle, so the
  middle row is flat 0 from either side and the pad stays continuous across
  it. At full height the two halves are exact vertical mirrors, which is what
  invert was overwhelmingly used for. Invert was also a hidden MODE — it
  silently mirrored every later preset application, state the plot could not
  show — so none of it is worth resurrecting; the per-profile `inverted` flag
  is gone from the store too (it was never serialized). A dashed centre line
  appears on the pad while step holds focus (`cnet-profile-pad-split`),
  because crossing it is the gesture that replaced the button. The
  scale range is two selects right of the plot labelled "range" (top =
  maximum, bottom = minimum), replacing the old
  two-handle gutter slider. **The stops follow what the axis MEANS**
  (2026-08-01, `scaleGrid` in weight_profile.js): a WEIGHT axis — the main
  profile and the three band profiles, which share the step plot — offers
  `1, 0.75, 0.5, 0.25, 0.1, 0.05, 0` and cannot
  express anything outside [0, 1] at all, denser at the low end where the
  difference between 0.05 and 0.1 is a doubling. A weight is a share of the
  control, so above 1 is a unit pulling harder than the whole of itself, and
  stacking that across units is exactly what the coverage panel draws as
  oversaturation. The MULTIPLIER and SHIFT axes (depth, drift) keep -1..2 in
  steps of 0.25, because a multiplier's neutral is 1 and a [0, 1] cap would put
  "leave these layers alone" at the very top of the plot. Balance keeps its own
  [0, 1] quarters. A string carrying an off-grid weight range (the `|0~2` band
  ranges of `docs/example_1.html`, say) is still PARSED and DISPLAYED as
  written — `ensureScaleOption` adds the value to the select — because
  clamping on load would silently halve every weight in a saved profile; it is
  only unreachable from the picker. `tests/test_profile_scale_grid.py` pins
  both halves. The range is the PLOT's, not the selected
  curve's (2026-07-25) — where a "plot" is one COORDINATE SYSTEM, and the
  editor has THREE of them (2026-07-25, second pass; a third added
  2026-07-27): the STEP plot (X = relative step, Y = weight) carries the main
  profile and all three band profiles, which are drawn together and therefore
  share one axis, so every step-domain segment carries the same `|lo~hi`
  suffix; the DEPTH plot (X = UNet depth, Y = per-layer multiplier) carries the
  depth curve alone and has its own range, and its `#D` segment carries its own
  suffix; the DRIFT plot (X = relative step again, Y = a shift along the depth
  axis) carries the drift curve alone, with its own range on its own `#S`
  suffix — and it is the one plot no multiplier axis could have hosted, because
  its neutral is 0 rather than 1. Per-
  curve limits inside ONE plot made the three band lines mean different
  things at the same height — that is what the rule prevents, and it never
  covered depth: nothing is ever drawn beside the depth curve, its Y is a
  multiplier rather than a weight, and coupling the two axes meant an
  ordinary weight range of 0..0.8 could not even express the depth curve's
  neutral 1 (it clamped to the top and shipped as a silent global ×0.8),
  while the default 0..1 let a depth curve only ever attenuate. The selects
  show the range of the axis ON SCREEN and never reach across; `rangeFor` /
  `activeRange` in weight_profile.js are the single decision point, and
  `loadBand` is what makes the pair follow the selector. Depth's default is
  0..2 so the neutral multiplier 1 sits mid-plot, drawing up boosting a layer
  and down attenuating it. A step-domain string whose segments disagree
  (hand-written, or written before this) is folded onto the range COVERING
  all of them and every curve is re-normalized into it by effective value, so
  migrating changes no weight, only the shared axis; the `#D` segment is
  excluded from that fold and adopted verbatim, which is also what makes the
  split invisible to strings written before it (the shared suffix they carry
  on `#D` IS the axis that curve was drawn on). Python needed no change at
  all — `parse_weight_profile` always read the suffix per segment. Bands still sitting on their
  untouched flat default ride along when the limits move (their neutral is
  the MULTIPLIER 1, so the flat line stays wherever 1 falls on the axis —
  top at [0, 1], middle at [0, 2]); a drawn curve keeps its shape and takes
  the new weights, which is what the range is for. Click cost follows destructiveness: step replaces
  the profile so it needs a confirming second click, the oscillatory ladder only
  reinterprets the drawn curve so every rung of it acts immediately. The plot's X axis carries just
  the label "Steps", in the same bottom-margin band ticks would use (values
  were redundant on the fixed relative 0..1 range); the Y limits are what the
  range selects say, and a 0..1 scale on the axis would contradict them (the
  drawn 0..1 is normalized, not the weight).
  SAMPLING GRID — the axis carries two cues, both driven by the tab's main
  Steps slider (live, via `attachStepsWatch`) and both on ONE x grid: dotted
  verticals at the step boundaries (N cells for N steps) and, since
  2026-07-25, a small value DOT on every drawn curve at the x that step reads
  its value from, `k/N`, the START of its cell. Same positions on purpose —
  every dot lands on a separator instead of forming a second, competing grid,
  which is why the separators only needed a dash more alpha rather than
  heavier styling. Multi-phase draws a dot series per sibling wave, the case
  the feature exists for (the waves cross, and each Input takes a different
  value at the same step); non-selected band underlays stay bare. Each dot rides a translucent background halo (a softened
  version of the point markers' trick) or it would be a bulge on its own line,
  and the halo radius is what sets `STEP_DOT_MIN_SPACING`. A dot's y comes
  from the same function that drew its line — never a second approximation, so
  it cannot sit off the curve. WEIGHT is calibrated, not incidental: the first
  cut (r 1.8 on an opaque r 2.8 halo, in the curve's own color) cut the curves
  into beaded strings, made a 4-wave plot read as noise and swallowed the
  green mid handle. Final is r 1.15 on a 0.6-alpha halo in ONE muted neutral
  (`stepDot`) — a same-colored dot on the main line shows only as its
  halo ring, while a separated tone reads on both ends of the plot's contrast
  range (under a full-strength curve, over the 0.4-alpha multi-phase underlays)
  and never competes with the band colors. WHICH side of the line that tone
  sits on is the theme's (invariant 33): darker than the white line on dark,
  lighter than the near-black one on light, same separation either way. Curves first, samples on a closer look. Both cues drop out in depth mode (X is the injection layer there,
  not the step) and once the marks would collide. Where the scheduler really
  places each step inside the timestep range is deliberately NOT modeled: the
  editor's X is the relative step range, uniform cells. (Verified headless
  against `draw()` itself — scratchpad check_step_dots.js, 18 checks incl.
  dot-on-curve for every phase-shifted sibling, halo pairing, and both
  collision guards.)
  In cosine mode the drawn curve is the ENVELOPE and the effective
  profile is `envelope(x) * (0.5 + 0.5*cos(2*pi*n*x - phase))` — phase
  subtracted so the pad point moving right shifts the wave right. The
  oscillation count of a profile string with **no** `C` token defaults to
  `COS_DEFAULT_OSC`, not 0 (editor `parse`, 2026-07-30): the buttons that switch
  the wave on do not invent a count, so a 0 there made them write `C0@0` — a
  wave of zero oscillations, a flat factor of 1 — and the click looked like it
  had done nothing. An explicit `C0@0` still parses as zero, since that sets
  `cosOn` with it. Expanded into
  a dense polyline by `parse_weight_profile` exactly like the parabola mids —
  so the string (and the infotext) stays short and the evaluation math never
  learns about cosines. The multi-phase families replace that factor with this
  Input's share of the same wave and are expanded the same way, so the
  evaluation math never learns about them either.
- **Response exponent** (2026-07-24) — vertical slider filling the free
  middle of the range column (between the two selects), per profile (the band
  selection routes it — unlike the range selects it sits between, which are
  per plot), serialized `G<e>`. Bends the
  NORMALIZED [0, 1] profile with `y -> y^e` AFTER mids and wave, BEFORE the
  scale mapping — deliberately independent of the range limits around it.
  Log-mapped travel: top = x^0.1 (10th root, bias to high values), center
  tick = x^1 (linear, neutral, snapped within 1e-3 and not serialized),
  bottom = x^10 (bias to low values). Double-click resets to linear (the
  reset lands after any thumb-jump the clicks cause, so it wins). The plot draws the BENT curve solid
  and, whenever e != 1 (same rule as cosine mode), the raw editable polyline
  as the dashed guide — the power of a piecewise-linear curve is not
  piecewise-linear, so the solid line uses the dense wave sampler either
  way; band overlays and multi-phase ghost waves bend through the same
  `gammaAt`/`profileValueAt`. Python: `_apply_profile_gamma` resamples
  densely exactly like `_apply_profile_wave` (editor `gammaAt` is the JS
  face of the same math — keep in sync). Native vertical range input via
  `writing-mode: vertical-lr` (min at top), center tick = wrapper `::before`.
- **Oscillatory ladder** (2026-07-27, replacing the separate cosine and
  multi-phase toggles of 2026-07-24) — ONE cyclic button, `data-preset="osc"`,
  stepping `off → cosine → multi-cosine → multi-Fejér → multi-von Mises → off`
  (`OSC_LADDER` in weight_profile.js). The two toggles it replaces were a
  single ladder wearing two hats: multi force-enabled cosine on the way in and
  cosine cleared multi on the way out, so "multi-phase without a wave" was
  never reachable. Do not split them again to add a family — extend the
  ladder. The multi rungs are weight-editor only (the balance editor has no
  Inputs to distribute over, same gate as the band selectors); there the
  ladder is just `off → cosine → off`, which is what the button's
  `data-osc-multi` attribute selects. One icon per rung lives in the button
  and `data-osc-state` picks it (style.css): the wave and the three-wave
  glyphs are the old two buttons' icons, Fejér and von Mises are the letters
  **F** and **M** — their lobes differ from the cosine's by shape alone and no
  18px icon carries that. `off` deliberately shows the plain wave, unlit.
  Serialized as a `P` token in the MAIN profile segment: bare `P` cosine,
  `PF` Fejér, `PV<kappa>` von Mises. **A bare `P` must keep meaning the cosine
  forever** — it is what every profile string written before the families
  carries, and its reproduction from infotext depends on it.
  Semantics: with n
  Input tabs holding images, input k (0-based) runs the SAME envelope with
  the wave shifted by `k * 2*pi/n` — input 1 exactly as drawn, each
  next shifted right — so the inputs take turns steering across the steps
  (oscillatory amalgamation). The family decides HOW they divide that one wave.
  **THE PARTITION IS STRUCTURAL, NOT PER-FAMILY** (2026-07-30). A weight is the
  family's raw lobe (`_phase_kernel` / `phaseKernel`) over the SUM of the n
  shifted copies of it (`_phase_weight` / `phaseWeight`), so the n weights sum
  to exactly 1 at every θ *because of the division*, not because each family
  happens to. Adding a family means adding a lobe and nothing else; there is no
  correction anywhere downstream that a new one could be missing. The lobes:
  - **cosine** `0.5 + 0.5*cos(u)`, whose shifts sum to n/2, so the weight is
    `(1 + cos(u))/n`. An input still holds ~65% of its peak at the neighbouring
    node, so the hand-over is soft and softens further with more inputs — and
    the peak itself is only 2/n, which is what a soft n-way split leaves.
  - **Fejér** `(1/n²)·[sin(n·u/2)/sin(u/2)]²`, already summing to 1 (the
    division is the identity) and CARDINAL — 1 at its own node, exactly 0 at
    every other one, so the inputs hand over cleanly instead of all leaking at
    once. No parameter: the sharpness is fixed by the input count. At n=2 it IS
    the cosine factor, so `P → PF` changes nothing on a two-input unit and the
    families only part company from three inputs up. Removable singularity at
    u≡0 (the ratio → n, the weight → 1); the `|sin(u/2)| < 1e-9` guard must
    match on both sides.
  - **von Mises** `exp(κ·cos(u))`, i.e. a softmax after the division, κ being
    the hand-over sharpness: 0 = every input equally on (plain averaging), 10 =
    near-hard switching. `cos - 1` in the exponent (never positive, cancels in
    the division) keeps `exp` in [0, 1] so a large κ cannot overflow to NaN —
    again on both sides.
  Until 2026-07-30 the cosine was the exception: its raw wave was used as the
  weight and a **2/n** was applied by hand in `scripts/cnpro.py`. The numbers
  are unchanged (`(1+cos)/n` is the same product), but the correction is gone
  from the call site, and with it the way a family could be added without one.
  A plain 1/n there had made the cosine unit silently HALF as strong as without
  multi-phase — "half the steps missed" on an IP-Adapter face unit with 5
  inputs (2026-07-24). Note the identity is exact on the NORMALIZED profile: a
  scale range with `lo != 0` maps each input affinely, so the sum carries an
  extra `(n-1)*lo`, and a response exponent `G≠1` bends each input's curve
  separately. Both are pre-existing and the same for every family.
  **A count of 1 is the family's KERNEL, not a partition** (2026-07-30,
  replacing "always takes the cosine path"): one lobe over itself would be a
  flat 1, i.e. the drawn wave silently gone, so `_phase_weight` returns the
  un-normalized lobe there — cosine and Fejér give the plain wave (Fejér falls
  back to its order-2 lobe, which IS the cosine), von Mises the `exp(κ·cos)`
  pulse whose width κ still sets. So a single-Input unit on `PV6` now runs a
  narrow pulse instead of a plain cosine, which is the shape the rung promises.
  The editor draws exactly that: `multiPhaseCount()` floors at **1**, not 2,
  and one Input means ONE line on the plot. It no longer previews a split that
  will not happen.
  Editor draws the sibling waves as thin
  underlays (n from counting loaded `img.forge-image` under the unit's
  `_input_image` containers; MUTED inputs are
  excluded — the id mapping of tab_marks.js resolves each image's
  `_input_enabled_<n>` checkbox — because the generation fan-out counts
  `len(get_input_data(...))`, which drops them, and the preview must count
  what runs. The 500 ms watch tick re-checks the count and redraws on change,
  so the preview follows uploads / clears / mutes live; nothing the editor
  otherwise observes moves when an Input tab gains or loses an image).
  ONLY THE MAIN PROFILE FANS OUT — python reads the marker off the main segment
  alone and parses every other segment with a count of 1 — so the editor's
  `waveCountOf(P, name)` returns the Input count for `main` and 1 for a band /
  depth / drift curve put on a family rung by the same ladder.
  Fan-out happens
  in `process_unit_before_every_sampling`: each pass of the per-input loop
  sets `params.model.weight_profile` to
  `parse_weight_profile(string, phase_index=k, phase_count=n)`, with NO
  `phase_offset` — a marked profile derives its own shift from the ordinal,
  because a partition's weights are defined over the whole set of inputs rather
  than one at a time, and passing both would apply the shift twice. The
  `phase_offset` parameter survives for unmarked cosine profiles only.
  The `P...` token itself is
  tolerated and skipped by the parse, so plain parses keep returning input 1's
  profile — `weight_profile_phase_family` is the caller-side switch (and
  returns `(family, kappa)`; `weight_profile_is_multiphase` is its boolean
  face, kept for callers that only need to know THAT a profile fans out).
  The
  coarse start/end gate is opened to 0..1 in multi-phase mode: it was derived
  from input 1's wave support and would chop sibling phases near the edges;
  every patcher gates per step through its profile LUT anyway. Single input /
  band mode: marker rides along, does nothing.
  `tests/test_partition_of_unity.py` pins the sum at 1e-12 over every family,
  Input count, wave and convergence, on BOTH sides.
- **Wave convergence** (2026-07-30) — a toggle button directly under the
  oscillatory one, serialized `A<at>@<e>` in the same segment. The n waves stop
  being a fixed split and slide onto the FLAT share 1/n of the envelope,
  reaching it at step `at` (pad x) and holding it for the rest of the range,
  with `e` (pad y, log-mapped over [0.1, 10] exactly like the response slider,
  middle row = linear 1) the dynamics of the approach. Early schedule: the
  inputs take turns. Late schedule: they average, which is what the last steps
  usually want — a reference switched off at step 25 cannot help settle the
  surface every reference has to share.
  **The total is safe by construction**: `_converged_weight` /
  `convergedWeight` is a CONVEX combination of two things that each already sum
  to 1 over the inputs — `(1-s)*partition + s*flat` — so the sum is
  `(1-s)*1 + s*n*(1/n) = 1` for any s, and the position and dynamics appear
  nowhere except inside s. Do not "optimize" that into a scaled deviation: a
  uniform factor on `(1/n - w)` keeps the SUM at 1 while overshooting the
  constant, which is why the test pins the plateau (`= 1/n` exactly past `at`)
  as well as the sum.
  It is a parameter OF THE WAVE, like phase and oscillation count: it survives
  the oscillatory ladder, is neither applied nor serialized while the wave is
  off, and switching it on with the wave off switches the wave on with it
  (converging nothing is not a state the user can see). With ONE Input the flat
  share is the whole envelope — A/n with n=1 is A — so the single wave simply
  fades into the envelope, no special case anywhere.
- **Control Balance Profile** — same editor and serialization; replaces the
  Control Mode radio (Balanced / My prompt / ControlNet) with a per-step
  cond/uncond balance curve (0.5 balanced, 1 control-only-on-cond, 0
  prompt-side-only).
- **Band weight profiles** — the weight editor's presets column ends in four
  thin band buttons: main (white) and coarse/mid/fine (red/yellow/blue).
  Exactly one is pressed (radio). Main shows/edits the classic whole-unit
  profile; a band button shows all three C/M/F lines (letters riding the
  lines at staggered x, since untouched bands sit flat at 1). In band mode
  the POINTS of all three lines are directly grabbable (add/move via drag,
  delete via double-click; non-selected lines draw small hollow markers, the
  selected band wins overlap ties) — the band selection only routes the
  PRESET-side actions: step/cosine/pad, response exponent, and where
  new points and margin/mid edits land. NOT the scale range: that is the
  plot's and moves all four profiles at once. Buttons are solid full-color line icons
  (theme button reset kills background-image → the line is declared
  !important); the active one carries the same --primary-400 border as the
  other preset toggles. No area fill below profile lines anywhere (several
  lines share the plot). Band profiles are per-step MULTIPLIERS of that band's
  injection layers (the same coarse/mid/fine mapping as the band masks;
  backend `band_of` is the single source). Neutral = flat 1.0 with no wave:
  not serialized, not forwarded. Serialization appends '#C..#M..#F..'
  segments to the main profile string (each segment full profile grammar —
  mids, cosine, and the scale suffix, which is the SAME on every segment
  because the limits belong to the plot), so infotext/API ride the existing
  `weight_profile` field; `parse_weight_profile` strips segments (split on
  '#') keeping every legacy consumer band-agnostic, `parse_band_profiles`
  reads them. Backend: LUTs built in `ControlBase.pre_run`
  (`band_profile_lookup`), applied per layer in
  `compute_controlnet_weighting` (`final_weight *= band value at sigma`) —
  they multiply painted band masks and equally work with no masks at all.
  Main and bands are EXCLUSIVE and the SELECTOR is the switch (2026-07-24):
  whichever band button is pressed — main, or one of C/M/F — is what runs,
  which is also exactly what the plot is showing (main mode draws the white
  main curve alone, band mode the three band lines alone). In band mode
  `scripts/controlnet.py` forwards no `weight_profile` and a neutral strength
  1 (start/end 0..1) so the two never multiply, and a band left flat at 1
  simply does nothing to its own layers. The mode rides the profile string as
  a trailing `#B<band>` segment (absent = main; `band_mode_active`), because
  it is behaviour and has to survive a reload / infotext round trip — the
  editor restores the pressed button from it. Both profile parsers skip
  segments whose prefix they do not know, so the marker needed no grammar
  change. Both curves stay serialized in either mode, so switching back finds
  the other one as it was left.
  An earlier version (2026-07-23, REVERTED) instead derived the mode from
  "is any band non-neutral" and dimmed the main curve with a warning label.
  Do not reintroduce that: the user's model is that the selector picks the
  active profile, and an implicit criterion made the widget change behaviour
  on its own while the plot showed something else.
  NOT scaled by the multi-input 1/N share (only the unit-level strength is).
  Supported by EVERY control model type the widget offers (2026-07-23):
  ControlNet / ControlLora / T2I scale their residuals per band in
  `compute_controlnet_weighting`; IP-Adapter (incl. InstantID) and
  ControlLLLite inject through one attention patch / one module PER UNET
  BLOCK, so each site resolves its own band at install time with
  `weight_profile.band_of_unet_block` and carries that band's LUT
  (IP-Adapter: `band_lut` per `CrossAttentionPatch`, looked up by sigma;
  LLLite: `band_profile` per `LLLiteModule`, on its step counter). The
  earlier "IP-Adapter has no banded injection layers" gate was WRONG - the
  attention blocks carry the same depth the residuals do, which is exactly
  what per-layer IP-Adapter weighting (linear / style-vs-composition tricks)
  already exploits elsewhere. `supports_band_profiles` remains on the base
  class as False so a patcher with a single whole-UNet hook (Fooocus inpaint)
  still warns rather than silently ignoring a drawn profile.
- **Depth profile** (2026-07-25) — second selector button (purple line,
  `--cnet-depth-line`), serialized as a `#D<profile>` segment. X is normalized
  UNET DEPTH (0 = shallowest injection layer = fine/texture, 1 = deepest =
  coarse/composition), Y a per-layer MULTIPLIER, neutral flat 1 like a band's
  (neutral is not serialized, not forwarded). It is NOT a fourth band: a band
  carries its own step curve and therefore REPLACES the main profile, while
  the depth curve MULTIPLIES it — the unit runs on `main(step) x depth(layer)`,
  the product of a time curve and a depth curve, separable unless a drift is
  drawn (next bullet). Bands and depth are alternative quantizations of the
  same 2D field and stay mutually exclusive: pressing a band selector runs the
  bands (`#B<band>` marker), pressing main, depth or drift runs main x depth
  (no marker — all three ARE main mode, and the `#D` / `#S` segments alone say
  whether they do anything). A per-bucket curve multiplied by a per-depth curve
  would count depth twice and no drawn value could be read literally any more;
  do not "improve" it into a combination. Layer depth comes from
  `depth_fraction_of_residual` / `depth_fraction_of_unet_block`, the
  un-quantized twins of `band_of` / `band_of_unet_block` (same flip
  conventions, opposite per group — see those docstrings). Applied in
  `compute_controlnet_weighting` (residuals), at IP-Adapter patch install
  (`depth_lut` per site) and per LLLite module. It is its OWN PLOT and carries
  its own Y range (see the weight-profile bullet above) — different semantics,
  different axis, and nothing is ever drawn next to it.
- **Depth-drift profile** (2026-07-27) — third selector button (green line,
  `--cnet-drift-line`), serialized as an `#S<profile>` segment. X is the
  relative sampling step like the main profile's; Y is a SHIFT along the depth
  axis, so its NEUTRAL IS 0, not the multiplier 1, and its plot defaults to
  [-1, 1] with the neutral line in the middle. The unit runs

      main(step) x depth(layer - drift(step))

  and that is the whole of it, defined once in
  `cnpro_core.weight_profile.drifted_depth` / `depth_multiplier` and mirrored
  in the editor's `driftedDepth`. Positive shift reads the depth curve further
  left and therefore moves what it draws toward the DEEP (coarse) end, so a
  DESCENDING drift sweeps the control from composition to texture as sampling
  proceeds. Clamped, not wrapped: the depth axis has two ends, not a period.

  **Why it exists.** `main(step) x depth(layer)` is rank-1 — separable — so the
  depth shape it expresses is frozen in time. That is the one thing the three
  band profiles CAN say and it could not, at the price of quantizing depth into
  three buckets. The drift is the missing degree of freedom, and it is a
  reparameterization rather than a fourth multiplier, which is exactly why it
  cannot count depth twice the way a band-times-depth product would.

  It is inert without a depth curve (shifting a flat curve is the identity).
  That falls out of the arithmetic, so nothing guards it — but `scripts/cnpro.py`
  does NOT forward it in that case and says why in a warning, because "drift is
  set" and "drift does something" must not be two states that look the same.

  Applied per step: `weighting.py` resolves the shift ONCE per step
  (`_drift_shift`, from a `drift_sigmas`/`drift_values` LUT built in `pre_run`)
  and `_site_factor` reads the depth curve at the shifted position. The two
  injectors that used to precompute one depth SCALAR per site now build a
  per-site LUT over steps instead (`build_depth_profile_lookup`) — constant
  when no drift is set, deliberately not special-cased into a second path.
  It is its OWN PLOT again, for the range reason above and one more: no
  multiplier axis can express a neutral of 0.
  Main mode draws NOTHING of it (2026-07-25, user decision). It briefly drew
  two thin dashed purple guides at `main(step) x min(depth)` / `x max(depth)`,
  the honest envelope of the product. They were dropped: they carried only the
  two EXTREMES of the depth curve — two scalars rendered as two full scaled
  copies of the main line — said nothing about which layer sits where, and had
  to be clamped to the plot's Y limits, so a product leaving the axis was drawn
  as a flat line along the top border, i.e. as a value it did not have. Cost of
  the removal, accepted knowingly: main mode now shows `main(step)` while
  `main(step) x depth(layer)` runs, and nothing on that plot says the depth
  curve is live — the one place this widget knowingly departs from invariant
  21. Do not re-add an inferred cue (a badge, a dimmed curve) without asking;
  that pattern was rejected once already for the bands.
- **Weight masks** — rainbow-hue paint over the input image encoding local
  control strength (hue (1-v)*270°, red=1 → violet=0; alpha marks painted).
  The hue is DISPLAY-ONLY: on the wire the mask travels as grayscale (pixel
  value = weight, ALPHA = paint coverage), `decode_weight_mask` reads the
  value channel and MULTIPLIES it by alpha/255 (2026-07-23) — canvas blur
  runs on premultiplied alpha, so the FEATHER falloff lives entirely in the
  alpha channel; the original binarize-at-128 decode threw the ramp away and
  every feathered edge came out hard. Unpainted (alpha 0) stays weight 0; a
  hue fallback covers legacy chromatic masks. `decode_weight_mask_parts`
  returns the two channels unmultiplied for the one reader that must not see
  them combined — the scalar share (invariant 5).
  ONE slot per input (`weight_mask[_<n>]`, 5 channels for 5 inputs) plus FOUR
  output slots (`output_mask[_coarse|_mid|_fine]`) — see invariant 6 for why
  the bands are output-side. Each input is an independent control, so its mask
  restricts only its own contribution and gates only its own hint;
  `weight_mask.js` reads the owning slot from the canvas group's
  `cnet-input-slot-<n>` class.
  Restrict-to-painted semantics: any painted mask means control never
  escapes the paint; absent bands count as zero when band masks are in use.
  A knowledge gate blanks the hint where weight is 0; on an embedding
  preprocessor the gate is BINARY and fills with the CLIP mean colour rather
  than black, and the mask's painted VALUE becomes that input's scalar share
  instead of a per-pixel weight (there is no per-region weight for CLIP to
  apply — it emits one token set for the frame).
  The tool box also holds an ERASER toggle (strokes remove paint back to
  UNPAINTED — not the same as painting 0, which is an explicit zero to the
  restrict rule) and a FEATHER slider (Gaussian-blurs the exported grayscale
  mask, 100 = 2% of the image diagonal; the overlay keeps showing the hard
  strokes) — both injected into the toolbar by `weight_mask.js` at attach
  time, not part of canvas.html.
- **Output mask** — same painter, own "Output mask" tab (right-aligned in
  the strip, next to "Input image"), one slot. Painted over a
  throwaway backdrop image that is never read: the mask is registered with
  the GENERATED image, not with the control input, so it only scales the
  injection per output region and never feeds the knowledge gate. Mapped
  with a plain resize (not the unit's resize mode, which maps *source*
  geometry) and multiplied into whatever injection mask the unit already
  carries. Nothing painted = control applies to the whole output.
  Reaches per-layer patchers via `region_masks` and
  attention-level ones via the `mask` argument of
  `process_before_every_sampling` (IP-Adapter `attn_mask`); patchers with
  neither declare `supports_output_mask = False` and get a warning.
- **Output weight coverage** (2026-08-01) — a collapsed accordion directly
  under the CNPro title holding one STATIC canvas: every enabled unit's
  spatial half, aggregated in OUTPUT geometry, in the mask hue ramp (violet 0
  … red 1, `weightToRgb`). Contours at 0 / 0.25 / 0.5 / 0.75, ORANGE at 1 and
  RED at every 0.25 above it, and past 1 the red darkens as well — the ramp
  itself has nowhere left to go there. Two metrics, because they answer
  different questions and neither derives from the other (max of a sum ≠ sum of
  maxima): **mean** over the sampling schedule (total influence — two units
  taking turns read as 1) and **peak** (the largest simultaneous sum — the same
  two read as 2). A backdrop can be dropped on it, or taken from the img2img
  input / the current output, and the map then draws over it at the opacity the
  slider says.
  **LAYOUT: it is built like a unit, because it is read like one** — settings in
  a column on the LEFT (the unit's own `controlnet_main_options_column`, so the
  two line up and share every width rule), the picture on the RIGHT, and the
  action buttons bottom-RIGHT under the canvas in a `controlnet_image_controls`
  row, the same `cnet-toolbutton` group with the same ⤵I / ⤵O glyphs the unit's
  insert buttons use, where every other below-canvas button group in CNPro
  sits. The status line lives under the CONTROLS instead, at the foot of the
  settings column: it is what the panel says, the eye is already in that column,
  and under the canvas it competed with the buttons for one line. It is pinned
  to the bottom (`margin-top: auto`) under a hairline, so it sits level with the
  legend line beside it and the column reads as a block and a footer rather than
  as controls followed by a gap. The settings column also drops the unit's
  `min-height: var(--cnet-unit-body-h)`: that floor stops a unit from becoming
  shorter than its canvas, and here it made the column TALLER than the map
  beside it, leaving a band of dead space across the bottom of the panel
  (measured: a 312px column against a ~277px canvas column).
  THE LINE UNDER THE CANVAS CARRIES ALL THREE: contour key left, weight ramp
  centred, buttons right — a `1fr auto 1fr` GRID, because that puts the ramp on
  the ROW's centre, which is the canvas's centre (the canvas is centred in the
  same column); flex `space-between` would only centre it between its
  neighbours. The key wraps rather than overflows when the panel is too narrow
  for all three (it used to run under the ramp: key 358..521 against a ramp
  starting at 471). The buttons are wrapped in a row of their OWN so the grid
  has exactly three cells: gradio's automatic `.form` grouping of consecutive
  form components — which a unit's button row relies on — did not apply with
  two HTML blocks ahead of them, and three loose children made the 3-column
  grid wrap onto two rows. Contour key before ramp: the contours are the lines
  drawn ON the picture, the ramp is the background the eye already knows from
  the masks. The legend and the status line are 12px, not the 10px they started
  at — the status is the panel's only quantitative answer, and two pixels off
  the one thing that has to be read is not a saving.
  The settings are ordinary gradio components (Radio `Coverage metric`, Slider
  `Map opacity over backdrop`) rather than hand-rolled HTML, read straight off
  the DOM by coverage_map.js — they are VIEW state, and a round trip to python
  is exactly what would make the panel lag behind the control driving it.
  Contours are NOT optional: the hue ramp has nowhere to go past red, so they
  are the only thing that can distinguish full strength from oversaturation,
  which is half of what the panel is for.
  Labels are deliberately distinctive because ui-config.json keys defaults by
  LABEL per tab (invariant 1b). ONE unit rule must not be inherited: the canvas
  column's fixed `--cnet-unit-body-h` height, which in a column-direction
  flexbox WRAPS whatever does not fit into a second column — measured as the
  action row appearing beside the map, 10px past the panel's right edge, at the
  top instead of the bottom. `height: auto !important` on
  `.cnet-coverage-canvas-column` is that fix and only that.
  The panel root's `elem_id` is `<tab prefix>_coverage`, which is where the
  script recovers the units row (`<tab prefix>_accordions`) and the tab from -
  no hidden data-* carrier element, because an element that exists only to hold
  two strings is the dead chrome the audit hunts for.
  MAXIMIZE AND THE HOVER READOUT (2026-08-02) — the panel stays STATIC (nothing
  here writes to a unit); both of these are about LOOKING at the map.
  *Maximize* is the host's own ⛶ / ➖ pair, same glyphs, same titles, same
  top-left corner of the frame (`canvas.html` `.forge-toolbar-box-a`), because
  this canvas is read like the unit canvases and must be enlarged like them.
  Two things are NOT optional in its CSS. It is the **stage** that goes
  `position: fixed`, not the panel — the settings column and the legend say
  nothing that needs the viewport — and therefore the toggle is a plain
  `<button>` inside the stage rather than a ToolButton in the action row: a
  full-viewport overlay covers that row, and an exit control the overlay hides
  is not an exit control (Escape is wired for the same reason, on ONE document
  listener rather than per panel). And lifting `max-height: 240px` is not
  enough on its own — `max-*` only ever shrinks a replaced element, so the map
  sat at its intrinsic size in the middle of the screen (measured 1024×512 on a
  1280×720 viewport); `width/height: 100%` is what makes it grow and
  `object-fit: contain` is what stops it stretching. No repaint is involved
  either way: the canvas already IS the output raster and CSS was the only
  thing scaling it.
  *The readout* is the value under the pointer, following it. The ramp is a
  good picture and a bad number — 0.9, 1.0 and 1.4 are the same red to the eye,
  the contours only mark quarter steps, and past 1 `colorFor` spreads a whole
  range over darker reds, so the picture cannot answer "what is THIS pixel"
  even in principle. It therefore reads the FIELD (kept on the panel state by
  `paint`, at the grid resolution it was aggregated on), never the canvas: the
  canvas holds a colour that is not invertible above 1, with contours stroked
  on top of it, over a backdrop, at the opacity slider's alpha.
  `valueAt` does CONTAIN-MATH, and this is the part that looks like an
  over-complication and is not: `object-fit: contain` letterboxes the raster
  inside the element box on whichever axis has slack, so scaling the element's
  rect reports values for pixels that are not on the picture. Maximized, that
  slack is 36px top and bottom on a 2:1 map in a 720px viewport. That case is
  the one discriminating probe in the browser harness — the naive mapping
  passes every other check in it.
  The readout HIDES the drop hint while it is up (`~` sibling rule in
  style.css, which is why the readout comes first in the stage markup, and why
  that rule must stay AFTER the `:hover` one it ties with on specificity).
  Deliberate: both are centred pills on the same picture, the hint is
  discoverability and the readout is an answer.
  Chrome in `lib_cnpro/controlnet_ui/coverage.py`; all arithmetic in
  `javascript/coverage_map.js`, in the BROWSER — the values it needs (enables,
  profile strings, painted mask channels, resize modes, width/height) reach the
  server only on Generate, so a server-side preview would show the last run
  rather than the pending one. It re-uses, never re-derives: `cnetLiveInputs`
  for which Inputs run, `cnproWeightProfile` for the grammar, `cnproWeightMask`
  for the mask codec and ramp, `cnetInsertSources` for the two insert buttons.
  ONE CHANNEL AT A TIME (2026-08-02): a `.cnet-coverage-channel` radio picks
  cross-attention (IP-Adapter; the DEFAULT) or residual (real ControlNet /
  ControlLora / T2I / Z-Image). Adding a residual weight to an attention weight
  is a category error — they land in different places in the UNet and never sum
  with each other — so the channels are never combined. A unit's mechanism
  comes from `.cnet-model-type-state` (`classify_controlnet_type`, rewritten on
  model.change), failing OPEN to residual on unknown/none per invariant 16;
  ControlLLLite is neither (it modulates attention PROJECTIONS and has no mask
  route at all) and is named in the status line rather than folded into a
  channel it does not belong to. Units on the other channel appear as a note,
  never as silence.
  MODELLED: enabled units, every live Input with its 1/N share (python gives
  each input a share of the unit and the shares sum to 1, so nothing on a
  channel can exceed 1 by itself — a reading above 1 IS cross-unit stacking,
  and the status line names the units that make it), the profile with waves /
  mids / response exponent / range, band mode (main × band, absent bands zero,
  the three averaged since they drive different layers), the depth curve as its
  mean multiplier, the output G and C/M/F masks, the input G mask on the
  residual channel only (it is not registered with the output on the attention
  channel), the resize-mode geometry. NOT modelled, and said so in the panel's tooltip:
  the unit Use-Mask, the balance profile, the drift, and the preprocessor —
  coverage is about WEIGHT, not content. The status line reports contributions,
  output size, metric, max, % uncovered, % above 1, any compute-grid cap, and
  names every enabled unit that contributes nothing (no input image, no
  profile, weight 0) rather than letting an empty map read as "nothing wrong".
  Geometry and pad colour are pinned against the real `crop_and_resize_mask` by
  `tests/test_coverage_map.py` — the pad is `np.median` of the source border,
  which on the always-even border count AVERAGES the two middle samples: taking
  the middle element instead filled a half-painted mask's letterbox with 255
  where the run fills it with 127, measured in a browser as 62% of the frame
  covered by paint that covers 37%.
- **Batch modes removed** — the Batch Folder / Batch Upload tabs, the
  `InputMode` enum and the `batch_*` unit fields are gone. They mapped N
  inputs onto the *generation batch* axis (image k conditions output k) and
  silently discarded images 2..N at batch size 1; multi-unit stacking covers
  the "several controls on one image" case instead.
- **Multiple inputs per unit** — up to `external_code.MAX_INPUT_IMAGES` (5)
  Input tabs (strip label just "In" — the injected mute checkbox already
  widens each tab), opened one at a time by the "+" tab and closed by the ✕
  button. Every input that holds an image is preprocessed on its own and
  patched in as its own control on the same unet, so the patcher chain sums
  their residuals into one contribution from the unit (`control_merge`) -
  the same mechanism that already summed separate units, applied inside one.
  The combination is a MEAN, not a sum: neither `control_merge` nor the
  IP-Adapter attention patch normalizes, so each input's strength (and
  weight-profile y values) is divided by N up front. The unit weight
  therefore keeps meaning "how hard this unit pulls" no matter how many
  Input tabs are open; sum semantics = weight x N.
  Everything else on the unit (model, preprocessor, profiles, masks) is
  shared by all of them; cost is N× control-model forwards per step.
  Gradio cannot create components after page build, so all 5 slots are
  pre-rendered and hidden; the open set lives in the `input_slots_open`
  gr.State and closing a tab clears its canvas, its four weight-mask
  channels and its mute checkbox, which is what actually removes it from the
  backend (the backend keys off "has an image", never off tab state, so tab
  visibility never has to be mirrored).
  Each Input tab has a mute checkbox IN ITS TAB TITLE, before the label text
  (`image[_<n>]_enabled` unit fields): unchecking MUTES the slot — image and
  masks stay, the slot drops out of `get_input_data` exactly like an empty
  one — so inputs can be A/B-ed non-destructively, from the strip, without
  switching tabs. Muting EVERY populated input makes the unit inert
  (2026-07-23: `get_input_data` returns an empty list and the unit is
  skipped — it must NOT fall through to the img2img-source fallback, which
  would silently re-target it; an enabled unit with no image at all on
  txt2img now raises a clean ValueError instead of the bare HWC3 assert).
  Gradio cannot render components inside the tab strip, so
  the gradio checkboxes are HIDDEN state channels and `tab_marks.js` injects
  a native checkbox into each strip button (id suffix mapping:
  `_input_tab_<n>` ↔ `_input_enabled_<n>`), syncing checked state from the
  channel (authoritative) and forwarding clicks to it with stopPropagation
  so the tab is not selected by the toggle. `tab_marks.js` also grays the
  muted tab's label.
  Per-slot SHARE numbers were briefly added next to the mute marks
  (2026-07-23) and removed the same day: a unit has ONE preprocessor and
  ONE model, so intra-unit inputs are same-modality, and uneven per-input
  weighting is already the WEIGHT MASKS' job — a flat-painted global mask
  at value v scales exactly that input's contribution (spatially, which a
  scalar cannot). Do not re-add a scalar duplicate of that path.
- **Input tab order** (2026-07-24) — the below-canvas ← button (right of ✕)
  moves the open Input tab one visual place left. The order is
  generation-relevant ("input 1" = leftmost tab; multi-phase gives it the
  unshifted wave), so it is a unit field: `input_order`, slot digits in
  visual order ("" = natural), lenient FAIL-OPEN parse
  (`ControlNetUnit.input_order_permutation` — garbage degrades to natural
  order, never loses inputs). NOTHING moves between slots: the strip
  re-sorts via CSS `order` on the tab buttons (tab_marks.js mirrors the
  channel; style.css gives every strip button static order 99 so + / Output
  mask / P / N keep trailing, ties = DOM order), and `get_input_data` sorts
  its slot list by the permutation — canvas content, masks and mute state
  stay with their slot, the moved tab stays selected, everything downstream
  (multi-phase phase k, control list) follows the sorted list. Close falls
  back to the visual-left neighbor; "+" appends the (re)opened slot to the
  visual END of the strip (a closed slot's digit parks wherever it is —
  only the visible projection is contractual). Not in infotext (pure
  session state, same exclusion as mute/masks). A content-swap variant
  (moving images/masks between slots) was considered and REJECTED:
  server-pushed images fight the echo gate (invariant 22 — a similar
  same-size image reads as a re-encoded echo and is swallowed), the upload
  seq-bump would clear the traveling masks (invariant 15d), and canvas
  layer stacks would flatten. Order changes are a string swap; nothing
  destructive can happen.
- **Below-canvas button row follows the open tab** — clear / load / ⤵I / ⤵O
  resolve the target canvas client-side (the inactive gradio tab panel is
  display:none, so `offsetParent === null` identifies it); ⤴ / ⤴1M / 📝 read
  the `active_canvas` gr.State kept in sync by the tab `select` events. 📝
  now creates a blank canvas straight at the main Width x Height (the "Open
  New Canvas" accordion with its own size sliders is gone).
- **Mask invert (i)** — toolbar button before the clear button, on both
  canvases. Unpainted counts as weight 0, so inverting paints the whole
  image (painted -> 1-v, unpainted -> 1); the pre-invert mask is snapshotted
  so a second press restores it exactly, and any stroke ends the pair.
- **Per-unit prompt** — two-row textbox at the bottom of the selector column
  (`unit_prompt` unit field): text encoded with the model's own text encoder
  (same pipeline as the main prompt, so attention syntax/embeddings work) and
  fed to the control branch's cross-attention IN PLACE of the main positive
  prompt, on the COND rows only — uncond rows keep the sampled negative
  context (`cond_mark` row blend in `ControlNet.get_control`, token counts
  reconciled by lcm-repeat like `ConditionCrossAttn.concat`). The asymmetry
  is essential: cond/uncond rows share latent+timestep+hint, so same-text
  residuals cancel out of the CFG contrast and the prompt visibly does
  nothing (the v1 both-halves implementation failed exactly this way).
  The main unet still sees the main prompt; pooled `y` conds stay
  sampled-prompt-side. Only true ControlNets
  (ControlNet / ControlLora — UNet-encoder copies with own cross-attention)
  consume it; T2I-Adapter / IP-Adapter / ControlLLLite have no text input and
  a set prompt on them logs a warning (`supports_unit_prompt`, a property on
  ControlNetPatcher because that patcher also wraps T2I adapters). Encoded
  once per run, cached on params for the hires pass; transport attr
  `unit_prompt_cond` is reset unconditionally every run because patcher
  instances are cached. The encode SNAPSHOTS AND RESTORES the torch / cuda /
  numpy RNG around itself (2026-07-24): it runs between `set_numpy_seed` and
  the sampling loop, and whether it runs at all depends on the cache, so
  whatever randomness the text encoder touched made the SAME seed and
  settings produce a different image depending on what the previous run had
  done — with a unit prompt set two identical runs never reproduced each
  other, and clearing the prompt again did not restore the original result.
  That noise swamped the prompt's own contribution, which is what made the
  feature look dead. Measured with the RNG held still: identical runs are
  byte-identical, and swapping the prompt changes the control residuals by
  ~3% of their magnitude (mean |diff| 0.0226 against 0.774; scratchpad
  unit_context_tensor.py). It RE-CONDITIONS the control branch — it does not
  inject subjects into the image, so main-prompt-like behaviour is not what
  it can deliver.
  STRENGTH SLIDERS (2026-07-24): each prompt tab carries two sliders under
  its textbox, both in [-1, 3] with neutral 1, per side (4 unit fields:
  `unit_prompt_emb_strength` / `unit_prompt_delta_scale` and the
  `unit_negative_prompt_*` pair) — added because the plain swap moves the
  control residuals by only ~3%, too subtle to steer with.
  - EMBEDDING STRENGTH: context = sampled + s·(unit − sampled), applied in
    `ControlNet.get_control` after the lcm token-repeat (so it works across
    77/154-token chunk mismatches). 1 = the classic swap and is returned as
    the unit tensor itself, NOT computed via the lerp (fp16 a+(b−a) is not
    bit-exactly b, and the default path must stay bit-identical); 0 = side
    off; >1 extrapolates past the text; <0 pushes away from it.
  - EFFECT SCALE (delta): residual = base + s·(with_text − base) where
    `base` is the residual with the SAMPLED context — isolates the text's
    exact contribution and scales only that, hint strength untouched.
    0 and 1 are resolved STRUCTURALLY (side dropped / plain swap, single
    forward); any other value runs the control model twice per step
    (`run_control_model` with both contexts, per-row combine via cond_mark:
    cond rows scale by the P delta, uncond rows by the N delta — a row whose
    side is inert has with_text == base, so its scale multiplies a zero).
    The extension logs the double-pass engagement so the slowdown is
    explained. A side counts as INERT when |emb| ≤ 1e-4 or |delta| ≤ 1e-4.
  DEFAULTS ≠ NEUTRAL (2026-07-25): all four sliders DEFAULT to 1.6, not to
  the neutral 1 — the plain swap is too subtle to steer with, so a fresh unit
  starts where the text actually shows. Nothing is spent until a prompt is
  typed (both are read only when a unit context exists), but once one is, the
  default effect scale 1.6 ≠ 1 means the double pass is ON by design. Changing
  a default takes TWO edits: the `ControlNetUnit` dataclass in
  `external_code.py` AND the stored `<tab>/<label>/value` keys in
  `ui-config.json` — the latter is applied over the component's value at UI
  build, so editing only the dataclass leaves the widget showing the old
  number. Labels are the ui-config keys, so the P/N retention mirror shares
  one entry (intended).
  Both compose: emb shapes the context (nonlinear through the model), delta
  then scales that effect linearly in residual space — verified
  emb2+delta0.5 == base + 0.5·(emb2_result − base) exactly (scratchpad
  test_prompt_strength.py, 22 checks incl. forward counts). Sliders share
  the textboxes' model-type gate and, like the prompts, stay OUT of
  infotext.
  PROMPT RETENTION (2026-07-24): ONE global slider (`unit_prompt_retention`,
  [0, 3], neutral 0, DEFAULT 0.75 — off neutral for the same reason as the
  strengths above; shown on both tabs, applies to both sides) added because
  the unit prompt's influence concentrates in the first steps: composition
  is decided at high noise, and the cond/uncond contrast of the control
  residuals (the only thing CFG amplifies) shrinks as the latent converges,
  so the text's effect decays regardless of the sliders above. Retention r
  multiplies BOTH per-step delta scales by `1 + r · progress` (progress =
  relative sampling position, from an identity-profile lookup built in
  `ControlBase.pre_run` via the shared `build_weight_profile_lookup`), so
  the text's isolated delta ramps up toward the end of sampling. Applied in
  `ControlNet.get_control` BEFORE the active/!=1 checks, so the base-pass
  isolation engages by itself exactly where the ramped scale leaves 1: the
  first step (progress 0, ramp ×1) stays bit-identical to retention off,
  every later step costs the second control forward (logged with the same
  double-pass note as effect scale). Deliberately a single knob, not a
  profile: the profile editor stays hint-strength-only. It shows on BOTH
  tabs: the P slider is the CANONICAL unit field, the N tab holds a mirror
  (`unit_prompt_retention_n`, not a unit field). Coupling is CLIENT-SIDE
  (javascript/prompt_retention.js): both inner inputs of the counterpart are
  written and re-dispatched via `updateInput()`, recursion settled by value
  equality. A gradio round-trip (P→N `.change` / N→P `.input`) was tried
  first and FAILED: the server echo arrived mid-drag and pinned the mirror
  to the stale canonical value, so dragging the N slider never took.
  Identical labels are deliberate (shared ui-config default — correct for a
  mirror).
  Layout: each tab's .form is a 2-column GRID (style.css), auto-placed from
  DOM order — textbox spans both columns (row 1, minmax(0,1fr) so it
  shrinks, ~107px), the emb/effect pair sits side by side (row 2), the
  retention slider spans both columns below (`_slider_full` class). Grid,
  NOT flex: gradio's `.form` ships `flex-wrap: wrap`, which in a
  fixed-height flex column spilled an overflowing slider into a second
  column beside the textbox. Panel height stays 230/static on both tabs
  (invariant 19; measured, scratchpad check_retention_layout.py — also
  verifies the sync both ways). Layout: the P/N tab .form is a fixed-height flex column — the
  TEXTBOX is the flexible member and shrinks (~127px), sliders keep natural
  height, unit body height stays 312.4 (static layout invariant 19).
  NOT in infotext (free text breaks the `,`/`:`-free
  format — same exclusion as masks); API callers pass `unit_prompt`.
  UI GATING (2026-07-23): the textbox is enabled only when the selected
  model's TYPE supports a unit prompt — disabled (never hidden) otherwise,
  with the placeholder naming which types do ("<Type> models have no text
  input - the unit prompt is read by ControlNet / ControlLora models"), so
  the hint doubles as education/discoverability. Model type is classified
  WITHOUT loading: `global_state.classify_controlnet_type` reads only the
  safetensors HEADER key names (cached per path+mtime; key signatures mirror
  the loaders' try_build_from_state_dict checks: 'lllite' substring →
  LLLite, 'lora_controlnet' → ControlLora, cond-embedding/hint-block/
  'control_model.' → ControlNet, 'ip_adapter./image_proj.' → IP-Adapter,
  'adapter./body.' → T2I). Legacy pickle files fall back to filename
  heuristics and otherwise 'unknown', which FAILS OPEN (classic .pth
  ControlNets must stay usable). Wired by `register_unit_prompt_support`
  (model.change → gr.update(interactive/placeholder)); initial state from
  the default unit's model ('None' → disabled + "no model" hint). Typed text
  is never cleared on disable — switching back re-enables it intact.
  Layout (since 2026-07-23, third iteration): the prompts live on their own
  "P" and "N" tabs at the far right of the canvas TAB STRIP (after Output
  mask), so they occupy no permanent space; each tab holds one textbox that
  FILLS the tab panel, the same footprint a canvas gets (`lines=5` only sets
  the initial rows - the height comes from the static body height below, and
  the box scrolls instead of growing). tab_marks.js gives P/N the same red filled marker
  the image tabs get when their text is non-empty. The strip order is now
  [In x5][+][Output mask][P][N] — the positional right-align selectors in
  style.css moved from :last-child/:nth-last-child(2) to
  :nth-last-child(3)/(4); update them again if a tab is ever added. The
  role is told by the enabled placeholder ("positive - seen only by the
  control model" / "negative - pushes this control away";
  `unit_prompt_state(model, role)`). The textboxes carry UNIQUE labels with
  show_label=False: an empty label collides in ui-config.json (the
  label-keyed "txt2img//value" entry — a stale "(presets)" value from
  another unlabeled textbox got injected into the negative box that way).
  Selecting P/N sets active_canvas to "prompt" (no canvas: dims buttons
  no-op, close button inert).
  A NEGATIVE unit prompt (2026-07-23, `unit_negative_prompt` field, one-row
  textbox under the positive one) replaces the sampled negative context on
  the UNCOND rows the same way ("push this control's semantics away from X");
  either or both may be set, unset sides keep the sampled context
  (`unit_uncond_context` through the same transport/encode-cache path,
  three-way blend in `ControlNet.get_control`). Same model gating, same
  infotext exclusion.
- **Model-type gating beyond the unit prompt** (2026-07-23) — the
  `classify_controlnet_type` result also shows a warning line inside the
  Balance accordion for kinds whose patcher ignores balance
  (`BALANCE_SUPPORT_KINDS` in controlnet_ui_group.py — lllite is the only
  current offender). It briefly disabled the BAND buttons too
  (`BAND_SUPPORT_KINDS` in weight_profile.js, removed the same day): that
  gate was based on a wrong premise, every offered model type honors band
  profiles now, and the editor no longer reads the model type at all.
- **Band exclusivity truth cues** — REMOVED (2026-07-24, user decision). The
  dot badges on non-neutral band buttons and the dimmed/dashed main curve
  with its "bands active - main profile ignored" label existed only because
  the mode was inferred from band neutrality; with the selector as the switch
  the plot already shows exactly what runs, and the cues were noise about a
  rule that no longer exists. Band neutrality itself stays (it decides which
  band segments are worth serializing / forwarding) and still compares
  EFFECTIVE (scale-mapped) values with the same 5e-4 epsilon as python
  `band_points_are_neutral`.
- **Dormant weight-mask dimming** — REMOVED (2026-07-31, user decision). The
  `.cnet-wmask-dormant` opacity rules (0.35, 0.75 on hover) and the class that
  drove them are gone; the slot buttons now look and behave identically
  whatever the profile selector says. A slot the selected profile does not use
  is still armable, paintable and kept, which it always was — but at 0.35 it
  read as DISABLED, so the toolbar was telling users to activate the band
  profiles before they could start a C/M/F mask, a constraint invariant 6 does
  not impose. The coupling is unchanged and still stated twice (the slot
  tooltip in `weight_mask.js::reflectProfileBand`, and
  `report_masks_not_in_force` at generation time). Do not re-add a disabled
  look without asking — invariant 21.
- **Per-step ws listing** — REMOVED (2026-07-23, user decision). It had
  shipped unreachable (inside the display:none hidden-options column), was
  briefly revived during the review round, and was then judged unneeded:
  checkbox, list HTML, `attachWsPreview`/`updateWsList`/`scaledValue`, the
  Steps-slider delegation and all `cnet-profile-ws-*` CSS are gone. Do not
  resurrect without a sigma-accurate step mapping.
- **Detected-map cache** (2026-07-23) — `cached_preprocessor_call` in
  scripts/controlnet.py memoizes preprocessor outputs across Generate
  clicks, keyed module + resolution + sliders + image/mask sha1 (LRU 8,
  ndarray outputs only — embedding preprocessors return dicts holding model
  refs and are never retained; 'shuffle' modules are never cached).
- **Weight masks on IP-Adapter** (2026-07-23, REVERSED 2026-08-02) — the
  painted INPUT masks used to be folded into the `mask` argument for patchers
  that restrict via it (IP-Adapter attn_mask), so they shaped those patchers'
  output region too. That route is REMOVED: an input mask on an embedding
  preprocessor has no spatial correspondence to the output, and sending it
  through the unit's resize mode made the reference's aspect ratio decide where
  in the frame its style landed. The output masks (G/C/M/F) are now the only
  spatial source for that argument; the input mask gates the encoder's view and
  supplies the input's scalar share. See invariant 6.
- **Reference squaring for embedding preprocessors** (2026-08-02) — CLIP's own
  processor center-crops to a square unconditionally, so Resize Mode was inert
  on this path and a 1:2 reference silently lost half its content.
  `utils.fit_square_for_clip` squares it FIRST, under the unit's mode: Resize
  and Fill pads to `max(h, w)` with `CLIP_IMAGE_MEAN_RGB` (123,117,104 — the
  processor's own image mean, the only fill that normalizes to ~0 sigma; white
  is +1.9 and black −1.8, both strong content the "plus" adapters' 257 patch
  tokens will encode), Crop and Resize center-crops to `min(h, w)`, Just Resize
  squashes. Sides are chosen so `k == 1`: a pad or a crop, never a resample
  before an encoder that resamples anyway. The mask-gated region takes the same
  neutral fill for the same reason. Recognizing an embedding preprocessor reads
  the host's `gate_input_by_weight_mask` flag OR
  `classify_controlnet_type(unit.model) == 'ipadapter'` — the flag alone left
  this silently inert wherever the host had not set it.
- **Mask previews in the results gallery are opt-in** (2026-07-23,
  `controlnet_mask_preview_in_results`, default off) — they multiplied the
  gallery (global + bands + output, base + hires, every run) with no
  reachable off switch. The detected map itself still attaches.
- **Static unit body height** (2026-07-23) — the body of an open unit
  (selector column | canvas tabs) is one fixed-height block, `style.css`
  `--cnet-unit-body-h` = the selector column at its full count of four rows
  (4 x 67.6px + 3 seams = 312.4px). Before, the row took the height of
  whichever side was taller, so a canvas tab made it 383px and a P/N prompt
  tab 312px and every tab click resized the unit and everything under it. Now
  the canvas column is pinned to the same height and distributes it (tab strip
  natural, panel = the rest, button row natural), so the below-canvas buttons
  always bottom-align with the last selector, on every tab and at any
  selector count. The canvas FILLS its panel (~230px) instead of setting the
  height with `ForgeCanvas(height=300)`, which the CSS overrides — resize via
  the variable, not the python argument. The tab strip is forced to one
  non-wrapping line for the same reason: every line it wrapped came out of the
  canvas below it.
- **Widget layout rework** — selector column left of the canvas, hidden
  legacy checkboxes, toolbar diet on the input canvas (clear/load forwarded
  to hidden core buttons), 1Mpx send-dimensions button, insert-image buttons
  (⤵I img2img source / ⤵O current output, client-side pixel push via
  `window.forgeCanvasPush`), raster-info line, image
  adjustments via ForgeCanvas (see
  `modules_forge/forge_canvas/canvas_extra.js`, separate subsystem).

## 2. File map

Extension-side (this directory):
- `lib_controlnet/external_code.py` — `ControlNetUnit` fields
  (`weight_profile`, `balance_profile`, `weight_mask[,_coarse,_mid,_fine]`,
  `output_mask`),
  string layer: `parse_weight_profile` / `parse_band_profiles` /
  `band_points_are_neutral` / `serialize_weight_profile` /
  `weight_profile_support` / `weight_profile_from_scalars`, API base64
  decode of masks.
- `lib_controlnet/infotext.py` — profiles ride in infotext; legacy infotexts
  (Weight / Guidance Start–End / Control Mode) are converted to profiles on
  paste. A flat-0.5 balance profile (the UI default) is a no-op and is
  excluded on write (`external_code.balance_points_are_neutral`, the same
  test that keeps it away from the patchers). Masks are images and
  intentionally NOT in infotext.
- `lib_controlnet/controlnet_ui/controlnet_ui_group.py` — layout, hidden
  state channels (profile textboxes, mask `LogicalImage`s), `unit_fields`
  mapping validated against the dataclass at startup.
- `scripts/controlnet.py` — mask decode (`decode_weight_mask`), knowledge
  gate, per-band mask routing, profile parsing into `params.model`,
  balance-unsupported warning.
- `lib_cnpro/controlnet_ui/coverage.py` — the coverage panel's LAYOUT and
  chrome, and nothing else (the map is computed in the browser): a unit-shaped
  Row of a settings Column and a canvas Column, three ordinary gradio controls,
  four `ToolButton`s and the canvas markup. The hue ramp of its legend is
  emitted as `hsl()` stops because hsl IS the formula weight_mask.js paints
  with, so the legend cannot drift from the picture.
- `javascript/active_canvas.js` — `window.cnetVisible` /
  `window.cnetActiveCanvasContainer`: the ONE implementation of the
  "offsetParent null = hidden tab" rule (invariant 14); consumed by
  insert_image.js, image_info.js and the inline `_js` of the button row.
  Also `window.cnetLiveInputs(unit)` — the ONE implementation of "which Input
  slots will actually run" (image decoded + mute checkbox on), read by
  weight_profile.js `multiPhaseCount` and by coverage_map.js, which needs the
  slot numbers too. And the shared 500 ms tick, whose gate counts a visible
  coverage panel as well as a visible unit body — the panel sits above the
  units and is readable with every one of them collapsed.
  Loads first by filename order.
- `javascript/coverage_map.js` — the coverage map: fit geometry (`fitRect`,
  `borderMedian` — the twins of `crop_and_resize_image`), mask decode into
  output geometry, `aggregate` (mean and peak), marching-squares contours and
  the ramp. The pure half is DOM-free and exported for
  `tests/coverage_map_js.js`; the DOM half registers nothing when the webui
  globals are absent.
- `javascript/weight_profile.js` — the curve editor (both profiles). Exports
  the class on `window.cnproWeightProfile` in the browser as well as through
  `module.exports`, so the coverage panel evaluates profiles with the editor's
  own parser instead of a second one.
- `javascript/weight_mask.js` — the painter as canvas toolbar tools;
  `CANVAS_DEFS` binds it to two canvases per unit (input image: four slots;
  Output mask tab: one slot). Injects the eraser button and feather slider
  into the toolbar at attach time; exports masks as grayscale (exportMask)
  and recolors grayscale masks to display hues on import (importState).
  Publishes the codec and the hue ramp on `window.cnproWeightMask` for
  coverage_map.js — the map decodes the very masks this file writes.
- `javascript/image_info.js` — raster-info line (consumes
  `forge-image-info` events from canvas_extra.js).
- `javascript/insert_image.js` — ⤵I/⤵O insert buttons: delegated click
  handlers (no server round-trip; re-encode to PNG data url, then
  `window.forgeCanvasPush(uuid, dataUrl)` from canvas_extra.js) plus the
  disabled-state sync (onUiUpdate + `forge-image-info`; buttons render
  `interactive=False` and are enabled only while their source exists —
  ⤵I never enables on txt2img units). The three source helpers are published
  as `window.cnetInsertSources` for the coverage panel's backdrop buttons.
- `javascript/active_units.js` — dropdown-based Control Type tracking (diff
  vs upstream radio version).
- `javascript/tab_marks.js` — strip-injected tab chrome: filled/muted marks,
  the mute checkboxes in the tab titles, and the visual tab order
  (`applyStripOrder` from the `input_order` channel).
- `javascript/theme.js` — the ONE light/dark decision, published as
  `data-cnpro-theme` on `<html>` (invariant 33). Written on the root element
  and not on body so that `getComputedStyle(document.documentElement)` — the
  plot's only way to read a colour — sees the same override the cascade does.
  Enters light only on a POSITIVE detection (gradio's `dark` class absent AND
  the app's background measured light), so anything it cannot read stays dark.
- `style.css` — widget layout + editor/painter styling.
- `tests/test_profile_parity.py` + `tests/profile_parity_js.js` — the
  python/editor grammar equality check (invariant 2). Standalone scripts, no
  test framework: run with the forge python, needs `node`.
- `tests/test_partition_of_unity.py` — the multi-phase contract: the n Inputs'
  shares sum to exactly 1 (1e-12) over every family, count, wave and
  convergence, and land exactly on 1/n past the convergence position. Drives
  the same node harness for the editor half, so both sides are held to it.

Transport / lifetime rules worth knowing before touching the patcher side:
`ControlModelPatcher.reset_run_state()` (modules_forge/supported_controlnet.py)
returns every per-run attribute to its default and is called BOTH at the start
of a unit's patch pass and after sampling. Patcher instances are cached for
the process lifetime (`cached_controlnet_loader` is an lru_cache) and shared by
two units on the same model file, so this is what keeps a stale mask or prompt
embedding from riding into the next run - and what stops the last run's mask
tensors (a [1,1,H,W] float per band) and encoded prompts from staying resident
between generations. Add new per-run attributes there, not just in `__init__`.

Core-side (would need monkey-patching or shipping-with the addon):
- `backend/patcher/weight_profile.py` — NEW, self-contained profile math:
  `evaluate_weight_profile`, `build_weight_profile_lookup`,
  `lookup_weight_profile_strength`, `balance_factors`. Single source; every
  consumer imports from here.
- `backend/patcher/controlnet.py` — profile/balance transport on
  `ControlBase` (`pre_run` builds sigma LUTs, `get_control` sets per-step
  strength, |strength| < 1e-4 skips the control model — an EPSILON, not
  exact zero, because scale-ranged profiles rarely interpolate to 0.0
  exactly), `compute_controlnet_weighting` applies balance + per-band masks
  (`resolve_band_mask`; `layer_mask_for` caches the repeated/cast/resized
  mask per layer shape for the whole run on the per-pass cnet copy),
  `apply_controlnet_advanced(weight_profile=, balance_profile=,
  unit_context=)`; `ControlBase.unit_context` + the context swap in
  `ControlNet.get_control` (per-unit prompt).
- `modules_forge/supported_controlnet.py` — `ControlModelPatcher`
  transport attrs (incl. `unit_prompt_cond`) + `supports_balance_profile` /
  `supports_output_mask` / `supports_unit_prompt` flags;
  `ControlNetPatcher` forwards into `apply_controlnet_advanced`.
- `modules_forge/supported_preprocessor.py` — `gate_input_by_weight_mask`
  flag (True on CLIP-vision-type preprocessors).
- `extensions-builtin/sd_forge_ipadapter/` — profile + balance in
  `CrossAttentionPatch` (sigma LUTs; `IPAdapterPlus.py`), flags in
  `forge_ipadapter.py`.
- `extensions-builtin/sd_forge_controlllite/` — weight profile in
  `LLLiteModule.forward` (step-counter domain, see limitations).
- `modules_forge/forge_canvas/` — canvas adjustments + wmask toolbar nodes
  (`canvas_extra.js`, `canvas.html`, `canvas.css`); documented separately in
  the project memory / agent notes. Since 2026-07 `canvas_extra.js` opens the
  pipeline with a LAYER STACK (stage = first upload's dims, layers each with
  own bitmap/transform/stroke list, continuously editable: click-select,
  drag-move, wheel-scale, reorder, delete, per-layer normal/lighten blend —
  lighten is the union mode for bright-on-black control maps — and per-layer
  OPACITY, `Opacity ◂ nnn ▸` at the right of each layer row, composed at
  flatten time via `globalAlpha` so the layer fades as ONE picture and the
  value stays freely reversible. The steppers are BESIDE the number, not the
  browser's own up/down spin buttons: those stack two 6px glyphs inside the
  field, and in a field narrow enough for a layer row they overlap the third
  digit, so "100" read as "10". Sideways they cost width the row can give —
  the label prints the layer SCALE only when it is not 100%, which is what paid
  for them — instead of height the digits cannot. A stepper click is 10, the
  wheel over the number is 1 (canvas_nodes.js `wireWheel` covers number inputs
  as well as sliders, and it must: an unhandled wheel there zooms the canvas
  behind the menu). `layerAlpha()` is the one reader of the field and defaults
  a missing one to 1 — `undefined` on `globalAlpha` draws nothing at all), a
  per-layer
  pen + ERASER (destination-out strokes in layer-local px), and only then
  rotation/levels/edges/invert/crop on the flattened composite — so the
  edges tool always operates on the totality of composed content. Layer
  structure is session-scoped; gradio only ever receives the flattened
  composite (upload/push still REPLACE; adding on top is explicit).
  `canvas_extra.js` also exports `window.forgeCanvasPush(uuid, dataUrl)`
  (programmatic upload into a registered canvas) — consumed by
  `javascript/insert_image.js` — plus `window.forgeCanvasAddLayer(uuid,
  dataUrl)` (add on top as a new editable layer) and
  `window.forgeCanvasDebugLayers(uuid)` (read-only state snapshot for
  Playwright/jsdom diagnosis); an addon build must ship or re-create these
  hooks.
  **CANVAS CLIPBOARD** (2026-07-30) — `Ctrl+C` over a canvas copies its
  composite, and a right-click opens a two-item menu, `Copy image` /
  `Paste image`. Both halves live at module scope in `canvas_extra.js`, not in
  `attach()`: one document-level listener each, finding the container by
  selector, so a canvas that appears later needs no registration.
  Why a CNPro menu rather than the browser's: over the picture the topmost
  element is the **transparent scribble canvas** (`elementFromPoint`, measured),
  so the native Copy image would copy an empty layer, and the native menu has no
  Paste to offer outside a text field — nor any way for a page to route one into
  a canvas. The host makes it moot anyway by answering `contextmenu` on the
  container with `preventDefault()`; an extension can neither edit that file nor
  unregister the handler, so ours is registered in the CAPTURE phase on
  `document`, which runs first whatever the host did. A right-click on an
  `input`/`textarea`/`contenteditable` inside the widget is passed through: the
  native cut/copy/paste is the right menu there.
  The menu is appended to `.gradio-container` (the image container is
  `overflow: hidden` and smaller than the menu; the gradio root also resolves
  the theme variables) and closes on click-away, Escape, scroll, resize or blur.
  What is copied is the DISPLAYED composite, which `test_canvas_parity.py`
  already pins to the value the control receives — verified pixel-identical end
  to end (1496×490, 2.9M channels, zero differing). Paste goes to
  `fc.uploadBase64`, the same inflow as a drop or the host's own Ctrl+V.
  Reading the clipboard is a PERMISSION where writing is not, so paste keeps
  `lastCopiedDataUrl` — the last image copied out of a canvas this session — and
  falls back to it when the read is refused or unavailable, saying in the
  console which source it used. Moving a picture between two Input tabs must not
  depend on a permission dialog. An empty clipboard is NOT routed to that
  fallback: it is a different outcome and gets its own message.
  The container flashes green/red for 450ms (`.cnpro-copied` /
  `.cnpro-copy-failed`) because a clipboard write is otherwise invisible.
  Also hosts the 1-click Topaz tools
  (`cnpro_host/optional/topaz.py` + `1M`/`HQ`/`DN` toolbar buttons): Photo AI
  `tpai.exe` CLI behind `/forge-canvas/topaz/process`, buttons
  availability-gated at runtime — general canvas tools, not
  ControlNet-specific, optional for any addon. The gate is a shared prober
  (`onTopazAvailable` in canvas_extra.js), not a one-shot fetch: the status
  route registers from `on_app_started` AFTER the page is already served, so
  the first ask can 404 by startup timing alone. "Could not ask" is retried
  with backoff and never cached as "unavailable"; a definitive server "no" is
  re-asked when the tab regains visibility, matching the server's per-call
  `find_tpai()` so the tool can appear without a restart.

NOT part of the feature (personal fork preferences, exclude from any addon):
`modules/ui.py` (1024 defaults, batch rows hidden, seed row move,
PnginfoImage), `modules/extras.py` (PNG-info format line),
`modules/processing_scripts/sampler.py` (XL/DMD presets),
`modules_forge/main_entry.py` (preset defaults), root `style.css` (sampler
presets + seed row), `styles_integrated.csv`, `webui-user.bat`, the
`processor_res` 512→1024 and Resize-Mode defaults. `modules_forge/
unet_patcher.py` (a dead reference copy of old-Forge code) was moved to
`scripts/model_chain_bckp/` on 2026-07-23.

## 3. Data flow

```
editor canvas (weight_profile.js)     painter (weight_mask.js)
        │ 'x@y;…|lo~hi'                       │ PNG data-url per slot
        ▼                                     ▼
hidden gr.Textbox  ─────────────  hidden LogicalImage x4
        └──────────── unit_fields (order == dataclass) ───────────┐
                                                                  ▼
                                                   ControlNetUnit (gr.State)
                                                                  ▼
scripts/controlnet.py: parse_weight_profile → points; decode_weight_mask_parts
→ (value, coverage) → their product = float mask, the pair = input_shares
(invariant 5); knowledge gate; params.model.{weight_profile,balance_profile,
region_masks(tensor|band-dict),strength/start/end derived}
                                                                  ▼
patcher type:  ControlNetPatcher → apply_controlnet_advanced → ControlBase
               (pre_run sigma LUT; get_control per-step strength;
                compute_controlnet_weighting balance + band masks)
               IPAdapterPatcher   → apply_ipadapter LUTs → CrossAttentionPatch
               ControlLLLite      → LLLiteModule.forward (step counter)
```

Derived legacy scalars (`strength` = max(profile, 0), `start/end` = profile
support) keep patchers that only understand constant weight behaving sanely.

## 4. Invariants — keep these true

1. `unit_fields` in `controlnet_ui_group.py` must list components in
   `ControlNetUnit` dataclass field order — checked at startup with a hard
   RuntimeError (not assert: `python -O` strips asserts, and this check is
   all that stands between a mismatch and silently shifted unit values);
   extend BOTH when adding a unit field (gradio applies them positionally).
   The trailing dataclass fields without components are pinned to exactly
   `['save_detected_map']` in the same check — a field appended without a
   component fails loudly instead of silently defaulting. Raising
   `MAX_INPUT_IMAGES` means adding `image_<n>` / `image_<n>_fg` /
   `image_<n>_enabled` to the dataclass; the ui side derives its entries
   from `image_canvases` / `input_enabled_checks`.
10. Weight masks are per input slot: the painter derives its four hidden
   channels from the canvas group's `cnet-input-slot-<n>` class
   (`cnet-wmask-<n>-<band>-state`). Two canvases must never share a channel —
   that is what makes masking input 3 affect only input 3.
15. Painter and server must never disagree about a mask. Four rules keep
   them coherent: (a) the canvas-EMPTY transition of `forge-image-info`
   clears every slot (clear button / close tab / unit reset) — only
   real clears pass through the empty state; the handler is
   transition-guarded because at startup a restored mask can arrive before
   the image loads. (b) A server-side clear of a mask channel (close tab
   resets canvas + 4 channels + mute checkbox) is detected in the
   rAF loop as "hasPaint but empty textarea AND no export owed
   (`!slot._syncTimer`)" and drops the local paint —
   else the next stroke-end would push the stale mask right back. The
   "no export owed" half is NOT optional and a guard on `st.painting` alone
   does not cover it: the stroke-end export is deferred one macrotask and
   pointerup clears `st.painting` BEFORE scheduling it, so "paint but empty
   channel" is also the normal state for one frame after every stroke. While
   that term was missing (2026-07-23 to -24) a frame boundary landing in the
   gap made the watchdog eat the stroke that had just finished: the painter
   wrote the mask, the watchdog cleared it, and syncState then exported
   hasPaint=false. Painting looked completely dead on every canvas, input and
   output alike, with no error anywhere. (c) The
   eraser recomputes hasPaint at stroke end (an eraser stroke may have
   emptied the mask; a fully emptied mask canvas is also dropped, so
   sub-threshold antialiased fringes cannot resurrect with the next
   stroke). (d) A late server→client mask push (value-only textarea write no
   observer sees) is picked up by a 500 ms watch that calls importState.

   **(e) 2026-08-02 — A NEW PICTURE UNDER THE PAINT IS NOT A REASON TO DESTROY
   THE PAINT, and this is now structural rather than documented.** The mask is
   registered with the FRAME, not with the pixels beneath it (python resizes it
   onto the generation dimensions regardless), so a replaced image — the ⤵I /
   ⤵O insert buttons, a drop, a paste, an upload — carries its mask onto the
   new geometry instead of taking it down. Painting a mask and THEN inserting
   the reference it belongs to is the ordinary order of work on the
   Output-mask canvas, so the old behaviour destroyed the work with the very
   action that set the job up.

   THIS CLAUSE USED TO SAY THE OPPOSITE, and that is the lesson. It described
   an UPLOAD SEQUENCE (`dataset.forgeUploadSeq`) that cleared every slot on a
   new image at identical dimensions. When the contract was inverted, the dims
   watchdog was taught to carry the paint and this branch — two hundred lines
   away in the `forge-image-info` handler — was not, so an insert still wiped
   the mask while the diff looked correct. THREE places independently answered
   "the image changed, does the paint die?"; patching one is not a fix.
   Invariant 29's shape exactly.

   So the answer is given ONCE:
   - `CLEAR_REASONS` (module scope, `weight_mask.js`) is the exhaustive list of
     ways painted weight may legitimately end. Every entry is the USER or the
     SERVER saying the paint itself is gone. None is "the picture changed", and
     none may ever be.
   - `clearMask(slot, push, reason)` REFUSES an undeclared reason: the paint
     survives (the safe direction — a mask that outlives its welcome is visible
     and one click from ✕; a wrongly destroyed one is silent and
     unrecoverable) and the console names the call site.
   - Every image-change site funnels through `onImageReplaced`, which cannot
     clear except to repair a slot claiming paint with no canvas behind it.
     It runs from the dims watchdog, from `forge-image-info`, AND from the
     `<img>`'s own `load` — the announcement fires while the element still
     reports the PREVIOUS `naturalWidth`, so `load` is the first moment the new
     geometry is knowable. Without that third call the carry still happened, on
     the next 500 ms tick, and "works because something else polls" is how this
     contract was broken twice.
   - `tests/test_mask_clear_reasons.py` reads the declaration and every call
     site out of the source and fails if they disagree, if a new reason names
     an image change, or if `clearMask` reappears in the image-info handler.
     `tests/test_mask_survives_insert.py` drives the feature on a real
     ForgeCanvas — paint, insert at a new size, insert at the SAME size (the
     branch the first fix missed, invisible to any dimension check), then
     remove the image and require the mask to be GONE. That last case is not
     optional: without it the file passes on a painter that has simply stopped
     clearing.

   The upload sequence itself stays, for its other job: `announceImageInfo`
   dispatches even byte-identical info when the seq moved, or a same-size
   replacement would be silent to every listener.
11. `params.control_cond` / `control_mask` are LISTS (one entry per input
   image), never batch-stacked — folding inputs onto the batch axis is what
   the removed Batch modes did, and it means something else entirely.
12. Per-field events must NOT rebuild the whole unit. Each component patches
   its own field into the `unit` gr.State (`dataclasses.replace`), with the
   State passed as an input — gradio keeps State server-side, so only the
   changed value is uploaded. The old blanket wiring
   (`fn=UiControlNetUnit, inputs=list(unit_args)`) re-serialized every image
   and every mask on every brush stroke: measured at 3.84x the mask's own
   size with a single input loaded, and growing with each further input.
   (`dummy_gradio_update_trigger` — the old full-rebuild escape hatch — was
   REMOVED 2026-07-23: nothing ever wrote to it, so the path was dead code.)
   The submit click merges LIGHT fields (everything that is not a
   LogicalImage), read live from their components, into the State
   (`merge_light_fields`): light fields stay component-sourced because
   ui-config.json writes initial values without firing events; heavy
   channels come from the State because they are only ever written through
   event-firing paths (user uploads, painter strokes, server pushes —
   gradio 4 fires `.change` for programmatic updates too). Consequence:
   anything that mutates a LogicalImage MUST do it through a path that
   fires its events, or Generate will not see the change.
   Multi-component cascades must not RELY on the echo events: the 8 clears
   of close-tab are read-modify-write races on the State, so
   `close_active_slot` patches the unit State itself in the same handler
   (one atomic `dataclasses.replace`) and the echoes are merely redundant.
   The dimension buttons (⤴ / ⤴1M) read input images from the State for the
   same reason send-side: passing LogicalImages as inputs re-uploaded every
   canvas on each click.
13. The unit must be able to SHRINK, not just grow. Two independent floors
   caused clipping and BOTH are needed (verified: panel 1200->700->1200->560,
   unit and every row now track it, 0 elements outside):
   a) root `style.css:1188` sets `min-width: fit-content !important` on
      `div.accordions > div.input-accordion` — on a flex item that means
      "never shrink below your widest content", so a unit kept the widest
      layout it ever had. `!important` alone does NOT beat it: that selector
      is (0,2,2), so the override has to be more specific (this file uses
      (0,4,2), scoped to open units so collapsed titles keep fit-content).
   b) root `style.css:1201` gives an OPEN unit `flex-flow: column`, which
      makes width the CROSS axis — so `align-items: center` (set here for the
      collapsed title row) sized the content to max-content instead of
      stretching it to the unit, and min-width could not help because
      min-width governs the MAIN axis. Open units need `align-items: stretch`.
   When a layout override looks ineffective, read the COMPUTED value before
   changing anything else — a matching selector is not an applied selector.
14. Anything UNIT-level in the DOM must have exactly ONE writer among the
   per-tab canvases, and that writer is the canvas of the open tab
   (`offsetParent !== null`). `image_info.js` writes the shared hint line and
   is the live example: once several input canvases matched
   `.cnet-input-image-group`, the filled one and the empty ones overwrote
   that one `<p>` in turn, and since each write is a DOM mutation that
   retriggers `onUiUpdate`, the page locked up. Same trap applies to any
   future per-canvas script touching unit-level nodes. The visibility rule
   itself is implemented ONCE, in `javascript/active_canvas.js`
   (`window.cnetVisible` / `window.cnetActiveCanvasContainer`) — never test
   `offsetParent` inline again; if gradio changes its tab-hiding strategy
   that file is the single fix point.
19. The unit body height is STATIC and both columns answer to it. Nothing in
   the canvas column may set its own height: the tab panel is the only
   flexible box and everything in it (canvas, prompt textbox) fills what the
   panel leaves. A component that insists on a pixel height - a wrapping tab
   strip, a taller button row, a canvas with its own `height=` - does not make
   the unit taller any more, it eats the canvas, which is why the strip is
   nowrap and the canvas is `height: 100% !important`. The `!important` also
   has to survive Maximize: that state is `position: fixed`, so 100% resolves
   against the viewport and matches the inline 100vh the canvas asks for.
   Verified by playwright measurement (scratchpad verify_heights.py pattern:
   equal row height on all five tabs, button-row bottom == last selector
   bottom, canvas height constant across panel widths).
   Two traps in the same spirit, both found by measuring (2026-07-30):
   * **`.prose button` adds `margin-bottom: 4px` to every button in a gradio
     HTML component**, at a higher specificity than a single class - so
     `.cnet-profile-preset { margin: 0 }` lost, and the profile editor's
     presets column silently carried 4px under each of its 9 buttons. 32px of
     invisible margin, which is what made the column outgrow the 200px plot the
     moment a third preset button was added. Anything laid out inside a
     `gradio-html` block needs `margin: 0 !important`, and the check is
     `getComputedStyle(el).margin`, not the rule you wrote.
   * **A gradio Row whose every child is `visible=False` still costs its own
     `margin-top` plus the unit column's `gap`** - ~20px of empty strip that
     belongs to nothing on screen. `.controlnet_row:not(:has(> :not(.hidden)))
     { display: none }` collapses exactly those and nothing else; `:has()` is
     live, so the row returns by itself when gradio drops the `hidden` class
     (the Hires-Fix row does this whenever hires fix is enabled - verified both
     ways in a browser). Do not replace it with a static `display: none` on a
     named row: the space is real when its component is shown.
2. Profile string format is parsed in exactly two places that must agree:
   `external_code.parse_weight_profile` (python) and
   `WeightProfileEditor.parse` (weight_profile.js). ENFORCED since 2026-07-25
   by `tests/test_profile_parity.py`: it feeds a corpus of strings (plain,
   scale ranges, mids, cosine, gamma, band and depth segments) through both
   implementations and compares densely sampled EFFECTIVE values — exact
   agreement except for the two documented approximations (mid flattening,
   cosine/gamma resampling), which get an explicit 5e-3 tolerance. Run it
   after touching either side; it needs `node` and loads weight_profile.js
   headless, which is what the module.exports + `typeof onUiUpdate` guard at
   the end of that file exist for. Do not remove them. Same for evaluation:
   `weight_profile.evaluate_weight_profile` ↔ editor `evaluate()`; parabola
   flattening `samples=24` == `MID_CURVE_STEPS`; the wave factor
   `_wave_factor` ↔ editor `waveFactorOf()`, both
   `0.5 + 0.5*cos(2*pi*n*x - phase)` with n capped at 4, and both routing
   EVERY case through `_phase_weight` ↔ `phaseWeight` (the unmarked single wave
   is the cosine lobe at a count of 1, so there is no second formula to keep in
   sync) plus `_converged_weight` ↔ `convergedWeight`. Parity cases
   may carry `(index, count)` to pin which Input's share is being compared —
   the editor discovers that count by walking the unit's DOM, of which the
   headless harness has none, so it is injected. The editor keeps the
   envelope in `envelopeAt()` and only `evaluate()` applies the wave — the
   drawn/serialized points are always the envelope. Alignment points fixed
   2026-07-23 and to be preserved: serialization precision is 4 decimals on
   BOTH sides (js `fmt` toFixed(4) == py `round(x,4):g`); two mid tokens in
   one segment resolve FIRST-wins on both sides; non-finite numbers
   (nan/inf) are REJECTED on both sides (`_finite` in external_code — NaN
   used to sail through into per-step strengths and silently corrupt the
   image); neutrality compares effective (scale-mapped) values with epsilon
   5e-4 on both sides, through ONE helper per side that takes the neutral
   VALUE as a parameter (`profile_points_are_neutral` ↔ `bandNeutral`) — 1 for
   the multiplier curves, 0 for the drift shift, 0.5 for balance, because a
   second copy of the test is how one side omits a segment the other draws;
   the drift coupling `depth(layer - drift(step))` is `depth_multiplier` ↔ the
   harness's `depthMultiplier`, and the parity test compares the COMPOSITE on a
   (depth, step) grid as well as the two curves — agreeing on both curves and
   still disagreeing on the sign of the shift is the failure that check exists
   for, and it does catch it; the python envelope is
   `evaluate_weight_profile` imported from backend.patcher.weight_profile —
   never re-implemented inline.
3. The format must never contain `,` or `:` (infotext survival) — that is
   why `x@y;…`, `~` and `|` were chosen.
4. Balance semantics live ONLY in `weight_profile.balance_factors`; never
   re-derive the min(2b,1) formula inline.
5. The mask WIRE format is grayscale (value = weight, alpha = paint
   coverage, multiplied into the weight at decode — that multiplication IS
   the feather: the canvas blur puts the whole falloff into alpha, and a
   binarized alpha discards it); `decode_weight_mask` decides by chroma
   (achromatic = grayscale) and keeps
   the hue decode only as a LEGACY fallback for old sessions / API callers.

   **A REDUCTION MUST TAKE THE TWO CHANNELS APART** (2026-08-04).
   `decode_weight_mask_parts` returns them unmultiplied and
   `decode_weight_mask` is its product; every SPATIAL reader (gate, mask
   tensor, coverage panel) wants the product, and everything that reduces the
   paint to one number — today `external_code.input_mask_share` — must take
   the parts. The canvas anti-aliases every fill it draws and cannot be asked
   not to, so the product sweeps from ~0 up to the painted value along the
   edge of every stroke: reducing it read a mask painted with ONE weight as a
   gradient (the user was warned their values had been averaged, about paint
   that held a single value) and returned a share BELOW what was painted, by
   the stroke's perimeter-to-area ratio — 0.98 for a blob, 0.86 for a thin
   line, **0.65 for a feathered disc painted at 1.0**. Two inputs painted at
   the same weight then split the unit unevenly, purely by stroke shape.
   Coverage belongs in such a reduction as a WEIGHT and nowhere else, and the
   value is only trustworthy where coverage is at least half (8-bit
   premultiplied storage: at one alpha step, paint at 0.5 reads back as 1.0 —
   which is where that function's noise floor comes from, derived, not a
   constant). `tests/test_input_mask_share.py` pins it, including that the
   generation path still passes the parts.
   The hue span 270° therefore still exists in two places — `weight_mask.js`
   (`HUE_SPAN`, display + import recolor) and `scripts/controlnet.py`
   (`WEIGHT_MASK_HUE_SPAN`, legacy decode) — change together, but new code
   paths must speak grayscale. Feather is applied to the grayscale export
   only; blurring the display rainbow would mix hues into unrelated weights.
   Oversized exports (> 8 MB data-url) DOWNSCALE the grayscale (never the
   hue canvas — gray interpolates as valid weights, hues do not) until they
   fit; the python side resizes to generation dims anyway.
6. **THE OUTPUT MASK SLOTS FOLLOW THE PROFILE SELECTOR.** Rewritten
   2026-08-02: the four slots moved from the INPUT canvases to the Output mask
   tab, and an input carries G alone.

   TWO SURFACES, TWO QUESTIONS.
   - An **input** mask answers "which part of THIS INPUT is worth reading". It
     is a SHAPE plus one NUMBER: the shape gates what the model (or CLIP) is
     shown of it, and the painted value reduces to that input's scalar share of
     the unit (`external_code.input_mask_share`, normalized so the shares sum
     to 1). It has no coarse/mid/fine variant, so there is one slot.
   - The **output** masks are the four weight profiles' spatial half — G for
     the MAIN profile, C/M/F for the coarse/mid/fine band profiles — so the
     editor's band selector picks which of them a generation uses, and nothing
     else does: main (and depth, which multiplies main rather than replacing
     it, and drift, which moves where depth is read) runs on G with C/M/F
     dormant; a band selection runs on C/M/F with G dormant.

   WHY OUTPUT-SIDE. A band addresses UNet DEPTH, a property of where control is
   injected, so "which part of the frame does the coarse band steer" is only
   well posed in output geometry. An input canvas need not be spatially related
   to the output at all: an IP-Adapter reference is not, and routing its paint
   into an attn_mask meant the REFERENCE'S ASPECT RATIO decided which part of
   the frame its style reached (it rode `crop_and_resize_mask` under the unit's
   resize mode, a transform that maps SOURCE onto output). That route is gone;
   the output masks are the only spatial source for patchers that restrict via
   the mask argument.

   One decision, two readers: `external_code.masks_in_force` (applied, python)
   and `weight_mask.js::liveSlotKeys` (shown, JS), which
   `tests/test_mask_profile_coupling.py` runs against each other over the
   strings the editor really writes.

   THE SLOT SET IS DECLARED IN FOUR PLACES AND THEY MUST AGREE — this bit on
   the move itself (2026-08-02): the registry `scope` (`canvas_tools.js`), the
   `display: none !important` id rules in `style.css`, the class sweep in
   `canvas_extra.js::attach`, and `weight_mask.js::slotDefsFor`. Updating only
   the registry left C/M/F invisible on the output canvas with no error
   anywhere — invariant 29's exact shape. A test that renders both container
   kinds and asserts the visible button set is the guard.

   Both readers decide by asking whether the selector IS a band, never by
   listing the ones that are not — `liveSlotKeys` always did; its dormant-slot
   tooltip did not, and named `main`/`depth` explicitly until the drift arrived
   to be left out of the sentence. The test's selector list is likewise taken
   from `weight_profile.js::SELECTOR_ORDER` rather than written out, so a
   seventh curve is covered the moment it exists: a hand-kept list omits
   whichever selector is newest, which is the one most likely to be wrong.

   The masks used to carry their own precedence — "a painted global mask
   governs everything, otherwise any painted band does" — which is a SECOND
   switch for one decision, and the two could disagree: with a band selector
   pressed and an old global mask still painted, the unit ran band profiles
   through a global mask and nothing on screen said so. Same shape as
   invariant 21: masking with a mask the user is not looking at.

   Restrict-to-painted still applies WITHIN the live set: absent band = zero
   control for its layers (`resolve_band_mask` returns None ⇒ layer zeroed).
   Paint in a dormant slot is KEPT, never discarded — the slot's tooltip says
   why it is not in force, and `report_masks_not_in_force` warns at generation
   time if such paint exists, because silently ignoring it while the button
   still shows its painted-mask border is the worst option available.

   THE COUPLING IS NOT A GATE. All four output slots stay armable and
   paintable at all times and are styled identically whatever the selector
   says — the dormant dimming was removed 2026-07-31 (see below). Painting a
   band mask under the main profile is legitimate preparation; requiring a
   profile switch before the brush works is a constraint the rule never needed.

   One thing about the output masks has NOT changed: they are output-side
   ONLY. None of them ever calls `apply_knowledge_gate` — the control model
   keeps seeing the whole hint and only the injection is restricted — and they
   must stay folded in AFTER `preprocessor.process_before_every_sampling`
   (the inpaint preprocessors read the same `mask` argument as their hole
   definition).
7. `suppressSyncUrls` in canvas_extra.js is a COUNTED map consumed per
   landing upload — do not turn it back into a grow-only blocklist (stale
   gradio image bug) or a one-shot flag (crop sync race, see memory).
22. The gradio ECHO GATE decides by PROVENANCE, never by how the picture
   looks. An echo is an inflow equal to the value the bound textarea holds
   (`gradioHeldValue`) — the bind polls that textarea and calls uploadBase64
   with exactly its content, so equality identifies gradio handing our own
   output back. Every other inflow (drop, paste, the core open button,
   forgeCanvasPush) is user intent and is applied verbatim. The pixel
   comparison (`pixelEqual`, mean channel diff < 8) exists ONLY to recognize a
   re-encoded echo and must stay behind the provenance test: applied to all
   inflows it read "looks similar" as "is the same" and silently discarded
   genuinely new images — dropping a new raster over an old one did nothing
   at all (canvas kept the old picture, gradio never heard about it,
   generations kept the old input) unless the canvas was cleared first,
   because a cleared canvas takes the no-layers path that skips the gate.
   Two shots of the same scene are well inside that tolerance, which is
   exactly what control inputs are (2026-07-24; verified: a 110x150 patch
   changed on a 640x480 image used to be swallowed).
8. New unit fields that must survive infotext go into
   `ControlNetUnit.infotext_fields()` AND get a same-named gradio component
   on the ui group (paste wiring is name-based).
9. `gr.Dropdown.update`/`gr.Slider.update` work only because
   `modules/gradio_extensions.py:157` restores `.update` — fine to use
   inside this fork, but an addon must not assume it on stock gradio.
16. `BALANCE_SUPPORT_KINDS` (controlnet_ui_group.py) is keyed by
   `classify_controlnet_type` strings and must fail OPEN on
   'unknown'/'none'/'' — a model whose type cannot be read from the
   safetensors header must keep every feature, never lose one. The hidden
   `.cnet-model-type-state` textbox is the transport (written on
   model.change). Band profiles used to have a second such set and it was
   removed: a UI gate is the wrong tool for "this patcher cannot do X"
   unless the patcher genuinely cannot, and the way to find out is to check
   where the model injects, not which loader built it.
21. What the profile plot shows IS what runs. The band selector picks the
   active profile (main or bands) for both the drawing and the backend, and
   the mode travels with the curves in one string (`#B<band>`). Anything that
   would make the widget compute weights from a curve the user is not looking
   at — an implicit "is it non-default" test, a silent fallback, a mode that
   resets on reload — is a bug in this widget's contract, not a feature.
   ONE deliberate exception, taken by the user 2026-07-25: a non-neutral DEPTH
   curve multiplies what main mode is showing, and main mode does not indicate
   it. The envelope guides that used to (see the Depth profile bullet) cost
   more than they told. Everything else about depth still obeys the rule — it
   is a selectable plot with its own axis, and the `#D` segment is the whole
   of its state.
20. Band profiles are per-depth strength and EVERY patcher resolves depth the
   same way. Residual patchers use `band_of` (residual index), attention
   patchers `band_of_unet_block` (UNet block address); both live next to the
   profile math and both mean coarse = deepest. A new patcher type adds
   itself by resolving the band of each of its injection sites and setting
   `supports_band_profiles` — never by widening a UI whitelist.

   A site's depth multiplier is NOT a scalar. It reads as one whenever no
   drift is drawn, and both attention injectors used to precompute it that way
   — which is precisely what had to be undone when the drift arrived, because
   the depth curve is then read at a moving position. Every injector now goes
   through `cnpro_core.depth_multiplier` (LLLite, which already has the step
   percent) or `build_depth_profile_lookup` (IP-Adapter, which has a sigma),
   with no scalar fast path: a per-site weight computed two different ways is
   a per-site weight that will eventually be computed two different ways.
17. Strip-injected controls mirror hidden gradio channels 1:1 by id suffix:
   `_input_tab_<n>` ↔ `_input_enabled_<n>` (mute checkbox). The channel is
   authoritative and clicks forward to it with stopPropagation.
23. Reordering Input tabs NEVER moves DOM nodes. The strip's visual order is
   CSS `order` computed from the `input_order` channel (tab_marks.js, write-
   guarded, on the module's 500 ms tick — the channel is written server-side
   as a value-only update, invariant 18 applies); gradio's keyed rendering
   and the positional :nth-last-child right-align selectors both depend on
   stable DOM order. The lenient permutation parse lives in exactly two
   places that must agree: `external_code.input_order_permutation` (python)
   and `applyStripOrder` (tab_marks.js). Agreeing means over ALL
   MAX_INPUT_IMAGES slots, not over the open ones: gradio renders no strip
   button at all for a hidden tab, so `applyStripOrder` enumerates slots by
   TAB PANEL (`X_input_tab_<n>`, present whether the tab is open or not) and
   places only the buttons that exist. Walking buttons instead (until
   2026-07-25) truncated the permutation at the first CLOSED slot, so any
   open set that was not a contiguous 0..k prefix — which is what closing a
   tab in the middle produces — left the slots behind the gap with a stale
   `order` or none at all, i.e. the static 99 of style.css, which parks a
   tab behind + / Output mask / P / N. Symptom: closing the active tab
   silently re-shuffled the surviving ones and the strip stopped agreeing
   with the order `get_input_data` feeds the backend in (the strip said
   input 1 was one slot, generation used another). Input buttons also carry
   a LOW default order in style.css, because gradio swaps a strip button for
   a fresh node when it becomes / stops being the selected tab and that node
   has no inline order until the next update pass.
18. A 500 ms watch tick is the ONLY guaranteed channel for value-only textbox
   writes (gradio programmatic updates produce no childList mutations, so
   onUiUpdate/onAfterUiUpdate never see them). maybeReload (editor: adopt /
   reset-to-default on empty / warn-once on invalid), late mask imports and
   the painter/server coherence checks, devicePixelRatio changes and the
   model-type gate all ride it. Do not remove it in favor of observers - that
   is the bug it fixes. Since 2026-07-25 there is exactly ONE timer, in
   active_canvas.js (`window.cnetRegisterTick`), and it is skipped while no
   unit body is laid out: three independent pollers were walking every unit of
   both tabs twice a second even with the ControlNet accordion closed. Skipping
   is safe because every registered tick is idempotent and re-runs from
   onUiUpdate when the accordion opens. Register there; never add a fourth
   setInterval.
24. The painter's coherence rules key off an explicit per-slot STATE
   (`idle` / `painting` / `pending`), not a conjunction of flags. Only `idle`
   lets the server-cleared watchdog drop local paint; a finished stroke goes
   to `pending` synchronously and stays there until its (debounced) export
   runs. This is the shape invariant 15b's bug had: while "an export is owed"
   was implied by a timer handle and "mid-stroke" by a separate global, one
   missing term made the watchdog eat every stroke that had just finished.
   The export is debounced 250 ms and FLUSHED before the generation request
   (capture-phase click listener on the Generate buttons): gradio serializes
   its queue in click order, so the mask's State write only lands first if the
   export is STARTED before the click is queued.

25. The UI branches on CAPABILITY FLAGS, never on patcher class. `scripts/cnpro.py`
   imports no concrete patcher at all. Mask routing used to ask
   `isinstance(params.model, ControlNetPatcher)` to decide whether per-band masks
   go through `region_masks` or through the `mask` argument; that made
   a question about ROUTING into a question about a CLASS, so the first new
   residual patcher (Z-Image) silently fell into the attention branch and would
   have received a single union mask where it should have had one per band. It is
   now `masks_via_advanced_weighting` on `patchers/base.py`. If you need a class
   name in L3, the real answer is a new flag.
26. A patcher fails at PATCH time, never at sampling time. Two mechanisms enforce
   this and both must stay:
   - `zimage_impl.Injector.install()` verifies the live module really is the
     NextDiT it was written against (`layers`, `context_refiner`, `x_pad_token`,
     `pad_tokens_multiple == 32`, injection sites in range) and raises naming
     `backend/nn/lumina.py` if not;
   - `zimage_config.config_from_state_dict()` refuses checkpoint layouts it
     cannot resolve unambiguously (v2.0 vs v2.1 have identical key sets) instead
     of picking one.
   The failure this buys protection from is not a crash, it is a plausible wrong
   image, which costs far more to notice.
27. Z-Image injection hooks MUST be removed after sampling
   (`process_after_every_sampling` → `Injector.uninstall`). They live on the live
   diffusion-model instance, which is shared across `UnetPatcher.clone()` calls
   and survives the run; a leaked injector holds the control chain and its hint
   tensors, and changes behaviour for a subsequent generation that does not use
   CNPro at all — breaking design rule 1 for the whole session.
   `tests/test_zimage_injection.py` pins bit-exact restoration.
28. The residual layout is rebuilt only when the GEOMETRY changes, not per
   forward. `layer_mask_for` keys its cache on the layout's identity, so a fresh
   layout object each step re-runs the mask resize for every injection site on
   every step of every pass and grows the cache without bound. The guard is
   `ZImageControlNet._layout_key`.

29. THE SILENT-MISS RULE. Whenever something is DECLARED in one place and
   HONOURED in another, the drift between them must produce an error. Three bugs
   here had exactly that shape and none of them raised: the weight-mask buttons
   were injected with `.forge-wmask-control` and revealed by a selector naming
   `.forge-adjust-control` (empty NodeList, no error, whole feature missing from
   the toolbar); `controlnet.py` called a `memory_management` function that does
   not exist and the registry's blanket `except Exception` turned it into "this
   file is not a ControlNet"; and a capability flag left undeclared inherits
   `False`, which is indistinguishable from having decided `False`.
   Three counter-measures, each with a test that fails on the pre-fix code:
   - the toolbar reveals every injected control EXCEPT an explicit `DEFERRED`
     list, so "absent" is never the default (`canvas_nodes.js`, and
     `audit()` re-checks the live DOM after attach);
   - `registry.load_control_model` keeps *declined* / *broken* / *refused* as
     three outcomes, never one;
   - every patcher must state every capability flag in its own class body,
     including the `False`s (`tests/test_patcher_contract.py`).
   Read ARCHITECTURE.md section 8 before adding a fourth declaration site.
30b. THE TOOLBAR IS NEVER WIDER THAN THE CANVAS, and this is not a nicety.
   `.forge-image-container` is `overflow: hidden`, so a button past its right
   edge is not "overflowing" — it is unreachable, and it looks perfectly healthy
   from every direction: in the DOM, computed display not `none`, a proper 24×24
   bounding box. `flex-wrap: wrap` has always been on the button row and could
   never fire, because `width: max-content` on the toolbar granted the row
   exactly the width it asked for. Measured in the running app: 19 buttons =
   546px on a 437px canvas, with G/C/M/F among the five that fell off the end.
   The cap is `max-width` = the measured usable width, written inline by
   `syncMenuWidth` (a percentage degrades in gradio — same reason as the menus),
   plus `grid-template-columns: minmax(0, max-content)` so the column can
   actually take it. Buttons wrap BY GROUP (`.forge-btn-group`, one per `gap:
   true` run) so a break lands between tool groups rather than mid-group.
   `tests/test_toolbar_layout.py` measures it at 900/500/437/320px and fails on
   anything outside the clip.

30. THE CANVAS TOOLS ARE A REGISTRY. `canvas_tools.js` holds one object per tool
   (button, menu rows, overlays, deferred reason); `canvas_nodes.js` renders the
   markup from it AND derives the id contract from the same objects. Add, remove
   or exchange a tool by editing one array entry — never by hand-writing markup
   or an id list somewhere.

   Rows may be nested in a `group()` for layout (the pen menu's three sliders are
   one column beside the picker). Grouping is PRESENTATION and must never shrink
   what the checks see: everything that walks rows goes through
   `flatRows()`, and `tests/test_toolbar_dom.py` compares the number of rows it
   walked against the number of sizers in the DOM, because the first version of
   the grouping silently dropped three rows out of the label-sizer check.

   Every `range()` row declares `labelMax`, the widest text it can hold. The menu
   sizes its label column from the set of them (a zero-height CSS sizer, so the
   browser measures the real font) — one width per menu, fixed, no measurement in
   JS. Declared in the registry and written by `canvas_extra.js`, so the drift
   raises: `test_toolbar_dom.py` reads the format strings back out of the source
   and checks prefix AND value width against the slider's own range.

   This replaced a generated markup blob plus a hand-written `OWNED_IDS`, a
   hand-written `DEFERRED` and a regex that parsed ids back out of the blob:
   four representations of one fact, and every bug the module ever had was two of
   them disagreeing (a button whose class the reveal did not sweep; an id regex
   off by one character that removed the entire toolbar; `levelsBox`, which had
   markup, ids and wired handlers and no button that opened it). Two things
   cannot disagree when there is only one of them.

   Rules that come with it:
   - **Styling is by CLASS, never by tool id.** A tool added to the registry is
     styled correctly the moment it is declared.
   - **A `deferred` reason is the only way a control may be invisible after
     attach**, and it must say why. Everything else visible by default; the audit
     reports anything hidden that has no stated reason.
   - **Every id in a `raw()` escape hatch must be listed in its `ids` array**, or
     the audit is blind to it.
   - `tests/test_toolbar_dom.py` compares the registry's ids against the ids
     `canvas_extra.js` and `weight_mask.js` actually resolve, **in both
     directions** — a declared-but-unwired id fails just as loudly as a
     wired-but-undeclared one.

31. Z-IMAGE v2.1: THREE THINGS THAT LOOK RIGHT AND ARE NOT. All three shipped
   and all three produced "the ControlNet is weak" rather than a failure, which is
   why they are listed rather than merely fixed:
   - **Hints go AFTER their block.** `unified = layer(...)` then
     `unified = unified + samples[layer_idx]` (diffusers transformer_z_image.py).
     A forward PRE-hook puts every hint one block early. v1 tolerates it (6 sites,
     5 blocks apart); v2.1 has 15 sites 2 blocks apart, so it is a ~50%
     positional error.
   - **The refiner hints are NOT for the base model.** `transformer_z_image.py`
     runs its `noise_refiner` with no hint application whatsoever. The hints are
     consumed inside the CONTROLNET, which re-embeds and re-refines its own copy
     of the latent; that private result becomes the `unified` argument of
     `control_layers[0]`. Injecting them into the base model corrupts the sampled
     latent AND starves control_layers[0] of its trained input. The private re-run
     costs 2 of 32 blocks per step; pay it.
   - **Re-running the base refiner re-enters the hook.** Calling those blocks
     through `__call__` fires CNPro's own `noise_refiner[0]` pre-hook again and
     recurses to the stack limit (measured: 194 nested calls). Guarded by
     `Injector._in_refine`.
32. THE HINT CACHE MUST STORE THE PADDED LATENT. `_hint_for` encodes once and
   caches. Padding 16 -> `control_in_dim` on the encode path only meant step 1
   got 33 channels and every later step got the cached 16, `embed_control` raised
   inside a try/except, the refiner stage was skipped, and the unit injected
   NOTHING for steps 2..N. The user's report was exactly "control is held only on
   step 1". v1 could never show it, because for v1 the padding is a no-op - so a
   v1-only test proves nothing here. `tests/test_zimage_injection.py` now calls the
   hint path four times for both 16- and 33-channel configs.
33. THE THEME IS ONE ATTRIBUTE, AND IT IS NEVER MEASURED FROM `document.body`.
   `javascript/theme.js` writes `data-cnpro-theme="light" | "dark"` on the ROOT
   element; style.css overrides the affected variables under
   `:root[data-cnpro-theme="light"]` and `weight_profile.js` reads those same
   variables (the plot is a canvas — it cannot be styled, only painted). Three
   things this pins:
   - **Root, not body.** Gradio's own `dark` class lives on body, and a
     variable declared there is invisible to
     `getComputedStyle(document.documentElement)`. Splitting the two would let
     the CSS switch while the plot kept painting the other theme's colours.
   - **Detect light POSITIVELY, default dark.** The class check comes first and
     the background measurement only breaks ties, so an unreadable or
     not-yet-built page renders exactly what it rendered before this existed.
   - **`document.body.backgroundColor` IS NOT THE THEME.** Gradio leaves body
     white on BOTH themes and paints the real fill onto the app element inside
     it. `colors()` used to branch on that probe and therefore took the light
     arm always: the dark theme was rendering the "light" step separators and
     step dots, and had been for its whole life. Measured on the running app,
     dark mode: 926 px of the light-arm separator tone on the plot, 0 of the
     dark arm's. Those two values are now pinned to what was actually shipping
     (`--cnet-plot-step-dot`, and one literal `stepLine`), because "correct
     detection" would otherwise have restyled a finished theme as a side
     effect. Anything that claims to know the theme from a colour must probe
     the app container, not body — and must be checked in pixels, in both
     themes, before it is believed.

## 5. Known limitations / open fragilities (accepted, not bugs)

- **Z-Image v2.1 hole/inpaint conditioning**: implemented, tested at the unit
  level (mask polarity, nearest sampling, pixel-space blanking, zeros fallback),
  and **not working end to end** - the masked region returns solid black with
  either polarity. Gated off by `CNPRO_ZIMAGE_INPAINT`; plain structural control
  is unaffected and is the default. See ARCHITECTURE.md section 4 and
  `zimage_impl._inpaint_enabled` for the investigation leads.
- **Z-Image**: numerical parity against diffusers/ComfyUI has NOT been measured.
  Structure is pinned hard (all 136 tensors match by name and shape; hooks land
  on the configured blocks; hint shapes match the base sequence), and every
  convention that could drift is taken from the base model at run time rather
  than recomputed — RoPE frequencies, the adaLN vector, the refined caption span,
  the pad token. But "same architecture, same weights, same inputs" is not the
  same claim as "same pixels". The remaining risk sits in three places, in
  descending order: the control-refiner span (upstream refines the control image
  tokens before the caption is joined; here that span is `freqs_cis[:, prefix:]`),
  the patch feature ordering, and bf16 accumulation order. Generate one image
  against a ComfyUI reference before trusting it for real work.
- **Z-Image balance profile**: unavailable, not neutral. Z-Image Turbo is
  few-step distilled and samples at CFG 1, so there is no uncond row to bias.
  The UI warns; it does not silently apply a curve that would do nothing.
- **Z-Image band labels**: `coarse` / `mid` / `fine` are a UNet-era name for what
  is, on a DiT, an ABSTRACTION axis rather than a resolution one — every block
  runs at one token resolution. The curves are meaningful and monotone in depth;
  the three words are not literally true. Kept rather than renamed so the UI,
  infotext keys and saved profiles stay identical across model families, which
  was the point. Logged once per run.
- **ControlLLLite**: profile evaluated on an internal step counter that
  assumes one forward per sampling step (batch-split cond/uncond would
  desync it); balance profile not implemented (a warning is logged since the
  supports flag landed). Band and depth profiles ride the same step counter,
  so they inherit the same caveat. NEGATIVE profile values are honored since
  2026-07-25 (the `multiplier <= 0` short-circuit became a magnitude test):
  the module's output is added to the block output, so a negative multiplier
  subtracts it, exactly like a negative ControlNet strength. The same fix
  landed on IP-Adapter, where the injection is `out_ip * weight` under
  weight_type "original" (the only type Forge uses) — a `weight <= 0` skip
  was silently discarding every profile the editor's negative scale range can
  draw. Under "linear" / "channel penalty" the weight scales K/V BEFORE the
  softmax, where a sign flip is NOT a clean subtraction: gate it there if
  those are ever exposed.
- **Bands on attention patchers are per-BLOCK, not per-resolution**: the
  three-way split is thirds of each group's block ids, which lands exactly
  on SD1.x resolution tiers but cannot on SDXL (attention there exists at
  two resolutions and the down/up sides hold different block counts, so one
  tier is always cut — input 7 reads mid while input 8 reads coarse). Depth
  order is still monotone, which is what the bands claim.
- **Per-band attn_masks on IP-Adapter** (implemented 2026-07-25): a painted
  band mask now reaches each attention site as ITS OWN band's mask instead of
  the union of all painted bands — `attn_mask` may be a dict band -> tensor
  and `apply_ipadapter` resolves it per site with `band_of_unet_block`, the
  same call that already picked the band LUT. Restrict-to-painted holds per
  band: a band with no mask contributes nothing, expressed by NOT patching its
  sites at all (correct weight, and cheaper than injecting through a zero
  mask). `patch_kwargs["number"]` must keep counting skipped sites — it
  indexes the adapter's to_k/to_v layers.
- **IP-Adapter masks**: since 2026-07-23 the painted input weight masks
  (global or band-union) ARE routed into the `mask` argument (attn_mask)
  for non-ControlNetPatcher patchers that support it, alongside the input
  gate — they now restrict IP-Adapter's output region too. Its
  `region_masks` remains set-but-unread (the attn_mask route is
  the effective one).
- **ControlLLLite**: no mask of any kind is honored (`mask` argument
  unused); an output mask on such a unit logs the unsupported warning.
- **T2I adapters / plain patchers**: profiles honored via ControlBase;
  patchers outside `apply_controlnet_advanced` (reference-type) see only the
  derived constant strength.
- Weight masks travel as base64 PNG through hidden textboxes on every
  stroke-end — grayscale since the wire-format change (compresses several
  times better than the old rainbow); exports above 8 MB are DOWNSCALED
  until they fit (2026-07-23; the stroke-end export is also deferred one
  macrotask so the pointerup can paint first), and a rAF loop runs while
  any mask holds paint (cheap by design, but permanent).
- Weight masks are not restored from PNG infotext (images don't fit
  infotext); API callers must send RGBA (non-RGBA now logs a warning).
- canvas_extra.js depends on preserved names of the obfuscated
  canvas.min.js (`uploadBase64`, `on_img_upload`, `this.img`,
  `background_gradio_bind.set_value`, id scheme, ctor arg order); only the
  prototype methods are existence-checked. Re-minification with property
  mangling fails silently — extend the guard before shipping publicly.
- Crop rectangle is not remapped when rotation changes the geometry aspect
  (the crop window drifts over different content after rotating).
- Full-resolution main-thread image ops (Sobel disc stamping, gray-area
  analysis) can jank on 4K+ images; caches retain full-res canvases per
  widget for the page lifetime.
- A throw inside canvas `attach()` leaves the container marked attached
  (controls silently missing until gradio re-renders that widget).
- Legacy infotexts with Weight > 1 are preserved via the `|hi` scale suffix
  since `weight_profile_from_scalars` gained scale support; values > 2 clamp
  to the editor range.
- **A step range whose top is below 1 cannot express a neutral BAND.** A
  band's neutral is the multiplier 1, `neutralY` clamps it onto the axis, so
  on e.g. 0..0.75 the three untouched band lines sit at the top and start
  evaluating to 0.75 — `bandNeutral` then reads them as drawn and serializes
  three phantom segments that attenuate their layers in band mode. Weight
  ranges below 1 are ordinary, so this is reachable. It is at least VISIBLE
  (band mode draws those lines and the selects state the limits), which is why
  it was left alone when the depth curve — where the same failure was
  invisible, main mode not drawing it — was given its own axis on 2026-07-25.
  Pinned as current behaviour in scratchpad `check_depth_range.js`. The real
  fix is the same one depth got (a band's Y is a multiplier, not a weight, so
  it wants its own axis), but that would uncouple main from the bands, which
  is a design decision the user has explicitly made the other way.

## 6. Addon packaging — verdict and plan

**Feasible.** No fundamental blocker; two structural decisions dominate:

1. **The ControlNet UI itself**: this directory is a heavily reworked
   builtin. A public addon should ship as a *fork of the whole
   sd_forge_controlnet extension* (users disable the builtin one), not as
   patches against it. Everything in this directory then moves verbatim.
2. **Core hooks**: the addon applies runtime monkey-patches at import:
   - ship `weight_profile.py` inside the addon (it is dependency-free);
   - patch `backend.patcher.controlnet`: wrap `apply_controlnet_advanced`,
     replace `compute_controlnet_weighting`, patch
     `ControlBase.pre_run/cleanup/copy_to` + `ControlNet.get_control` +
     `T2IAdapter.get_control` (all attribute assignments on import — the
     current diff is exactly this surface);
   - set transport attrs on `ControlModelPatcher` dynamically; consumers
     already `getattr(..., None)` everywhere;
   - tag preprocessor instances (`gate_input_by_weight_mask`,
     `show_control_mode`) after registration instead of editing base
     classes;
   - IPAdapter/LLLite: wrap `IPAdapterApply.apply_ipadapter` and
     `load_control_net_lllite_patch` (single call sites). Deepest patch =
     `CrossAttentionPatch`; cleanest long-term fix is upstreaming a
     per-sigma weight callback.
   - canvas layer: `canvas_extra.js` + CSS load fine from an extension
     `javascript/`/`style.css` (proven by the three CN scripts already
     loading that way); the `canvas.html` template additions (adjustment
     boxes, wmask buttons) must become runtime DOM injection into
     `toolbar_<uuid>` at attach time — the only real rework item.
3. Version pinning: the addon must state the Forge commit range it patches
   (obfuscated canvas.min.js contract + backend class shapes).

## 7. Verification habits (from development history)

- Python/HTML changes need a server restart; JS/CSS only Ctrl+F5
  (canvas.html + script list are frozen at server start).
- `D:\store\forge\system\python\python.exe` is the interpreter; use it for
  `py_compile` and for import-level smoke tests (stub `modules.shared`).
- Playwright (`channel:'msedge'`, headless) against the live server and
  jsdom sims of canvas.html+scripts are both established diagnosis tools
  (see project memory for the required stubs).
- After touching profile code, re-run the parse/serialize edge cases
  (empty/malformed/`|lo~hi`/mid tokens/legacy conversion round-trip).
