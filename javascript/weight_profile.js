/**
 * Control Weight Profile editor.
 *
 * Replaces the Control Weight slider + Timestep Range slider with a single
 * piecewise-linear profile: X axis is relative sampling step [0, 1], Y axis is
 * control strength [0, 1].
 *
 * - Click on empty plot area: add a point (and drag it while held).
 * - Click near an existing point: pick it up and move it.
 * - Double-click a point: delete it (the last remaining point cannot be deleted).
 * - Every segment carries a small green midpoint handle. By default it is
 *   passive: it sits on the literal linear midpoint and just follows the two
 *   main vertices. Dragging it makes it an active control: the segment turns
 *   into the parabola (second-order) through the two vertices and the handle,
 *   clamped to the [0, 1] plot range. Double-clicking an active handle turns
 *   it back into the passive linear midpoint.
 * - Click/drag on a margin pins the corresponding coordinate to its extreme:
 *   left margin edits Y of the profile start (x = 0), right margin edits Y of
 *   the profile end (x = 1), bottom margin places a point at y = 0 at the
 *   clicked X, top margin places a point at y = 1.
 * - Presets in a column left of the plot - step (button), the oscillatory
 *   ladder and the convergence toggle - plus a square parameter pad next to
 *   them carrying TWO parameters of whichever preset holds focus.
 *   Click rule follows how destructive the preset is:
 *     step REPLACES the drawn profile, so the first click only aims the pad at
 *       it and a confirming second click rebuilds the profile;
 *     cos only decides whether the drawn polyline is read as an envelope, so
 *       it toggles (and draws the wave) on the very first click;
 *     converge modulates that wave and touches no drawn point, so it toggles
 *       on the first click too.
 *   Moving the pad point always rewrites the focused preset's parameters.
 *     step: pad x = jump position; pad y = height AND direction. Above the
 *       pad's middle the raised part is on the RIGHT (0 then h), below it on
 *       the LEFT (h then 0); the height is the distance from the middle, so
 *       the middle row is flat 0 from either side and the pad is continuous
 *       across it. At full height the two halves are exact vertical mirrors,
 *       which is why there is no invert toggle any more - the pad reaches
 *       both directions on its own, with no hidden mode to remember. The
 *       degenerate ends stay two points, not four: on the upper half full
 *       left is a constant at h (the jump already happened at 0) and full
 *       right a flat 0 (it never happens); on the lower half they swap.
 *     cos:  pad x = phase over [0, 2pi], pad y = oscillations over [0, 4].
 *       The drawn polyline becomes the ENVELOPE of the wave and stays
 *       editable point by point; the wave itself pulses between 0 and that
 *       envelope, so it never flips sign.
 *     converge: pad x = the step by which the waves have settled onto the flat
 *       share of the envelope (they hold it for the rest of the range), pad y =
 *       the dynamics of the approach, the middle row being linear. The shares
 *       still sum to 1 at every step, so the unit's total pull is the drawn
 *       envelope whether the waves are converged or not.
 *
 * Two range selects sit right of the plot - the upper one is the range top,
 * the lower one the range bottom - mapping the normalized profile [0, 1] onto
 * [lo, hi]. The profile shape stays as drawn; only the weights used for
 * calculation are remapped.
 *
 * THE GRID DEPENDS ON WHAT THE AXIS MEANS, and the two cases are not the same
 * quantity. A WEIGHT axis (the main profile and the three band profiles) is a
 * share of the control and lives in [0, 1]: its selects offer
 * WEIGHT_SCALE_GRID and cannot express a value below 0 or above 1 at all. A
 * MULTIPLIER axis (depth) or a SHIFT axis (drift) keeps the old -1 .. 2 grid in
 * steps of 0.25, because a multiplier's neutral is 1 - a [0, 1] cap there would
 * make "leave these layers alone" the top of the plot and everything else a
 * reduction. A legacy string that carries an out-of-grid weight range (the
 * '|0~2' band ranges of docs/example_1.html, say) is still PARSED and still
 * DISPLAYED - ensureScaleOption adds the value to the select - because
 * rewriting a saved profile's range silently halves every weight in it. It is
 * only unreachable from the picker: once such a range is changed it cannot come
 * back above 1.
 *
 * The range belongs to the PLOT, not to a curve - and a "plot" is one
 * COORDINATE SYSTEM, of which this editor has three. The step plot (X = relative
 * sampling step, Y = weight) carries the main white curve and all three band
 * lines: they are drawn together, so per-curve limits would make them mean
 * different things at the same height, and they share one range. The depth
 * plot (X = UNet depth, Y = per-layer multiplier) carries the depth curve
 * ALONE and has its own range: nothing is ever drawn beside it, its Y is a
 * multiplier rather than a weight, and coupling the two axes made a weight
 * range of e.g. 0..0.8 unable to even express the depth curve's neutral 1.
 * The drift plot (X = relative sampling step again, Y = a SHIFT along the depth
 * axis) carries the drift curve alone for the same reasons and one more: its
 * neutral is 0, not the multiplier 1, so it is the only curve here that a
 * multiplier axis could not express at all.
 * The selects always show the range of the axis on screen. Serialization
 * follows the same split: every step-domain segment carries the same
 * '|lo~hi' suffix, the '#D' and '#S' segments carry their own (python parses
 * each segment on its own anyway, so it needs no notion of which axis it was
 * on).
 *
 * The profile is serialized as 'x@y;x@y;...' (with a '|hi' or '|lo~hi'
 * suffix when the range is not [0, 1]) into a hidden gradio textbox
 * (.cnet-weight-profile-state) read by the python side. An active segment
 * mid control adds an 'Mx@y' token between its two vertices' tokens; passive
 * midpoints are derived state and are not serialized. Cosine mode adds one
 * 'C<oscillations>@<phase>' token: the stored points stay the envelope and
 * parse_weight_profile expands the wave into a dense polyline, so infotext
 * stays short and every consumer keeps receiving plain points. The same editor is
 * instantiated for the Control Balance Profile (its block carries the
 * cnet-balance-profile-editor class: neutral is flat 0.5, the scale grid is
 * limited to [0, 1] and there are no band buttons).
 */
(function () {
    const SCALE_MIN = -1;
    const SCALE_MAX = 2;
    const SCALE_STEP = 0.25;
    // The WEIGHT axis grid (main + the three band profiles), high -> low, which
    // is the order the selects list. Not a regular step series on purpose: a
    // weight is a share of the control, so the low end - where the difference
    // between 0.05 and 0.1 is a doubling - is worth more stops than the high
    // end, where 0.7 and 0.8 are barely apart. 0 and 1 are the hard limits of
    // the axis and both are on the grid.
    const WEIGHT_SCALE_GRID = [1, 0.8, 0.7, 0.6, 0.5, 0.35, 0.25, 0.2, 0.15, 0.1, 0.05, 0];
    const MARGIN = { left: 16, right: 16, top: 22, bottom: 24 };
    // step separators disappear below this cell width - denser dotted lines
    // read as haze, not as cells
    const STEP_LINE_MIN_SPACING = 4;
    // Per-step value dots ON the curve (the separators say where a step is,
    // the dots say what it gets). They are a READING AID, not a series of
    // markers: a fraction of the mid handle's radius so they can never be
    // mistaken for something grabbable (or drown it - with N steps x N waves
    // there are far more of them than there are handles), and quieter than the
    // lines they ride, so a glance sees the curves and only a closer look
    // resolves the samples. Each sits on a translucent background halo that
    // mutes the line under it instead of cutting it: enough separation to tell
    // dots apart where the multi-phase waves cross, without punching the
    // curves full of holes. Dots vanish a little earlier than the separators
    // do, since touching halos would chew up the shape they exist to sample.
    const STEP_DOT_RADIUS = 1.15;
    const STEP_DOT_HALO_RADIUS = 2.1;
    const STEP_DOT_HALO_ALPHA = 0.6;   // relative to the dot's own alpha
    const STEP_DOT_MIN_SPACING = 5;
    // The SELECTED profile's own samples get the opposite treatment: a bead in
    // the curve's own color, WIDER than its 2px line and with no halo at all.
    // A muted dot that is barely wider than the line only darkens it, and a
    // line pitted with dark notches is painful to follow - the mark has to add
    // thickness, not take it away. Affordable here because it is one series of
    // N, where the sibling waves are (waves - 1) x N and must stay context.
    const MAIN_DOT_RADIUS = 2.1;
    const MAIN_DOT_ALPHA = 1;
    // sibling (multi-phase) dots, well down from the selected profile's: they
    // are context, and there are (waves - 1) times as many of them
    const PHASE_DOT_ALPHA = 0.45;
    const GRAB_RADIUS = 12;
    const POINT_RADIUS = 5;
    // segment mid controls (parabola handles): smaller than main vertices and
    // with a tighter grab radius, so they steal as few clicks as possible
    const MID_POINT_RADIUS = 3.5;
    const MID_GRAB_RADIUS = 9;
    const MID_MIN_T = 0.05;    // handle x stays strictly inside its segment
    const MID_MAX_T = 0.95;
    const MID_MIN_DX = 0.02;   // no handle on (nearly) vertical segments
    const MID_COLOR = "#4caf50";
    const MID_CURVE_STEPS = 24; // parabola sub-sampling for drawing

    // Injection-band profiles (weight editor only): besides the MAIN profile
    // (white button/line - per-step strength of the whole unit) the editor
    // holds one profile per coarse/mid/fine band (red/yellow/blue), per-step
    // strengths of those bands' injection layers - the same bands the
    // C/M/F weight masks address. Main and bands are EXCLUSIVE and the band
    // SELECTOR is the switch (MAINTENANCE.md invariant 21): whichever button
    // is pressed is what the plot draws and what the backend runs.
    // Serialized as '#C...#M...#F...' segments appended to the main profile
    // string; neutral (flat 1) bands are omitted, so the classic
    // single-profile format stays the common case. A band's neutral is the
    // MULTIPLIER 1, so on a plot whose range is not [0, 1] its flat line sits
    // wherever 1 falls on the shared axis, not at the top.
    const BAND_ORDER = ["coarse", "mid", "fine"];
    const BAND_PREFIX = { coarse: "C", mid: "M", fine: "F" };
    // single source of the curve colors: CSS variables in style.css (the
    // selector bars use the same ones); resolved lazily because this module
    // evaluates before stylesheets are guaranteed applied.
    //
    // They are also THEME-DEPENDENT (javascript/theme.js writes
    // `data-cnpro-theme` on <html>, style.css overrides the affected ones
    // under it), which is why the resolution is a cache with an invalidator
    // rather than a one-shot: read from document.documentElement, so the
    // root-scoped override is what is seen, and dropped again when the theme
    // flips under a running UI.
    const BAND_COLOR_FALLBACK = { coarse: "#e53935", mid: "#fdd835", fine: "#1e88e5" };
    let BAND_COLORS = null;
    function resolveBandColors() {
        if (BAND_COLORS) return;
        const styles = getComputedStyle(document.documentElement);
        BAND_COLORS = {};
        for (const b of BAND_ORDER) {
            const v = styles.getPropertyValue(`--cnet-band-${b}`).trim();
            BAND_COLORS[b] = v || BAND_COLOR_FALLBACK[b];
        }
        DEPTH_COLOR = styles.getPropertyValue("--cnet-depth-line").trim() || DEPTH_COLOR_FALLBACK;
        DRIFT_COLOR = styles.getPropertyValue("--cnet-drift-line").trim() || DRIFT_COLOR_FALLBACK;
        MAIN_COLOR = styles.getPropertyValue("--cnet-main-line").trim() || MAIN_COLOR_FALLBACK;
    }
    function invalidateCurveColors() {
        BAND_COLORS = null;
    }
    const BAND_LABELS = { coarse: "C", mid: "M", fine: "F" };
    const BAND_LABEL_X = { coarse: 0.08, mid: 0.5, fine: 0.92 };
    // The main profile's line is achromatic - it is the one curve that is not
    // a band, and every other line here carries a hue - but WHICH achromatic
    // is the theme's call: white on dark, near-black on the light theme, where
    // white was simply not drawn as far as the user could tell.
    const MAIN_COLOR_FALLBACK = "#ffffff";
    let MAIN_COLOR = MAIN_COLOR_FALLBACK;
    const BAND_CURVE_SAMPLES = 96;

    // Depth profile (weight editor only): the SAME depth axis the three bands
    // quantize, un-quantized. X is normalized UNet depth (0 = shallowest =
    // fine/texture, 1 = deepest = coarse/composition), Y a per-layer
    // MULTIPLIER, so its neutral is flat 1 exactly like a band's.
    //
    // It is NOT a fourth band: a band carries its own step curve and therefore
    // REPLACES the main profile, while this one has no step dimension and
    // multiplies it - the unit runs on main(step) * depth(layer). Bands and
    // depth are alternative shapes of the same 2D field and stay mutually
    // exclusive (a per-bucket curve times a per-depth curve would count depth
    // twice, and no drawn value could be read literally any more): pressing a
    // band selector runs the bands, pressing main or depth runs main x depth.
    // Serialized as a '#D<profile>' segment; neutral is omitted like a band.
    //
    // Its Y RANGE is its own, decoupled from the step plot's (see the header):
    // different semantics, different axis, and nothing is ever drawn next to
    // it. Default 0..2 puts the neutral multiplier 1 in the middle of the
    // plot, so drawing up boosts a layer and drawing down attenuates it - on
    // the step plot's default 0..1 a depth curve could only ever attenuate.
    const DEPTH_KEY = "depth";
    const DEPTH_PREFIX = "D";
    const DEPTH_COLOR_FALLBACK = "#9c4dff";
    const DEPTH_RANGE_DEFAULT = { lo: 0, hi: 2 };
    let DEPTH_COLOR = null;

    // Depth-DRIFT profile (weight editor only): the third plot, and the only
    // one that couples the other two. X is the relative sampling step like the
    // main profile's; Y is a SHIFT along the depth axis, so the unit runs
    //
    //     main(step) x depth(layer - drift(step))
    //
    // Positive shift reads the depth curve further left and therefore moves
    // whatever it draws toward the DEEP (coarse) end, so a descending drift
    // sweeps the control from composition to texture as sampling proceeds.
    // The definition is cnpro_core.weight_profile.drifted_depth - mirrored in
    // driftedDepth() below, and pinned by tests/test_profile_parity.py.
    //
    // Why it exists: main x depth is a rank-1 (separable) field, so the depth
    // shape it expresses is frozen in time. The three band profiles CAN vary
    // depth over steps, at the price of quantizing depth into three buckets.
    // The drift is the missing degree of freedom rather than a fourth curve on
    // an existing axis, which is why it gets its own plot and why it multiplies
    // nothing: it only moves the depth curve's argument, so it cannot count
    // depth twice the way a band-times-depth product would.
    //
    // Its NEUTRAL IS 0, not 1 - it is a shift, not a multiplier - and its plot
    // defaults to [-1, 1] so the neutral line sits in the middle and either
    // direction is reachable. That is the one thing every neutrality helper
    // below had to stop assuming.
    //
    // A drift with no depth curve to move does nothing: that falls out of the
    // arithmetic (shifting a flat curve is the identity), so the editor keeps
    // the two independent and scripts/cnpro.py is where the user is TOLD.
    // Serialized as an '#S<profile>' segment; neutral is omitted like a band's.
    const DRIFT_KEY = "drift";
    const DRIFT_PREFIX = "S";
    const DRIFT_COLOR_FALLBACK = "#00c853";
    const DRIFT_RANGE_DEFAULT = { lo: -1, hi: 1 };
    let DRIFT_COLOR = null;

    /**
     * Every selector the editor can be in, in the order its buttons appear:
     * the three that ASSEMBLE into one field first, then the three that each
     * REPLACE the main profile. COMPOSED from the constants above rather than
     * typed out, so it cannot name a curve the editor does not have or miss one
     * it does. Exported for tests/mask_profile_js.js, which has to exercise the
     * mask coupling on every selector - and a hand-written list there would
     * quietly skip whichever selector was added last, i.e. exactly the one most
     * likely to have got the coupling wrong.
     */
    const SELECTOR_ORDER = ["main", DEPTH_KEY, DRIFT_KEY].concat(BAND_ORDER);

    /** Neutral EFFECTIVE value of a stored curve: 0 for the drift shift, 1 for
     *  every multiplier curve (the bands and the depth curve). The main profile
     *  is never asked - it is not omitted from the string and has no neutral to
     *  re-anchor to. Single source, so a segment cannot be dropped by one
     *  neutrality test and drawn by another. */
    function neutralValueOf(name) {
        return name === DRIFT_KEY ? 0 : 1;
    }

    /** The depth position a depth curve is READ AT under a shift - the JS twin
     *  of cnpro_core.weight_profile.drifted_depth, clamped the same way (the
     *  depth axis has two ends, not a period, so a wrapped shift would teleport
     *  the deepest layer's multiplier onto the shallowest). */
    function driftedDepth(depth, shift) {
        return clamp01(depth - shift);
    }

    function defaultBandProfile(lo, hi, neutral) {
        // flat neutral: the curve leaves its layers alone. Where that line sits
        // is a position on the axis it is drawn on - the multiplier 1 is the top
        // of the step plot's default [0, 1] and the middle of the depth plot's
        // [0, 2]; the drift's 0 is the middle of its own [-1, 1].
        const y = neutralY(lo, hi, neutral);
        return {
            points: [{ x: 0, y: y }, { x: 1, y: y }],
            cosOn: false, cosN: COS_DEFAULT_OSC, cosPhase: 0,
            phaseFamily: null, kappa: 0, converge: null, gamma: 1,
        };
    }

    const TAU = Math.PI * 2;
    const GAMMA_MAX = 10;         // response slider ends: x^10 ... x^(1/10)
    const COS_MAX_OSC = 4;        // pad top = 4 oscillations over the step range
    const COS_DEFAULT_OSC = 1;
    const KAPPA_MAX = 10;         // pad right in von Mises: near-hard switching
    const KAPPA_DEFAULT = 3;
    // Convergence pad: y is the dynamics exponent, log-mapped exactly like the
    // response slider so both directions get equal travel - the pad's middle
    // row is the linear 1, the bottom the tenth root (leave early, arrive
    // patiently) and the top the tenth power (the other way round).
    const CONVERGE_EXP_MAX = 10;
    // arrives exactly at the end of the step range, linearly: the whole plot
    // shows the approach, which is what the button is for
    const CONVERGE_DEFAULT = { at: 1, exp: 1 };

    /**
     * The oscillatory button's ladder, in click order. `null` family = the
     * plain single wave; the rest split that same wave between the Inputs.
     * The balance editor stops after "cos" (it has no Inputs to split over),
     * which is what the button's data-osc-multi attribute says.
     *
     * This is ONE control because "multi-phase without a wave" was never a
     * reachable state: the old multi toggle force-enabled cosine on the way in
     * and cosine cleared multi on the way out, i.e. two booleans encoding a
     * single ladder. Mirrors PHASE_FAMILY_* in lib_cnpro/external_code.py.
     */
    const OSC_LADDER = [
        { state: "off", cosOn: false, family: null },
        { state: "cos", cosOn: true, family: null },
        { state: "multi", cosOn: true, family: "cos" },
        { state: "fejer", cosOn: true, family: "fejer" },
        { state: "mises", cosOn: true, family: "mises" },
    ];
    const PAD_MAX_SHARE = 0.4;  // pad never eats more than this of the free width
    const PAD_MIN_SIDE = 46;
    const COS_CURVE_SAMPLES = 480; // dense enough for 4 oscillations across the plot

    function clamp01(v) {
        return Math.min(Math.max(v, 0), 1);
    }

    /**
     * The Fejer kernel of order `count` at circular offset u - the JS face
     * of _fejer_weight in lib_cnpro/external_code.py, and the singularity
     * guard has to match it: sin(N*u/2)/sin(u/2) -> N as u -> 0, so the
     * weight -> 1.
     */
    function fejerWeight(u, count) {
        const s = Math.sin(0.5 * u);
        if (Math.abs(s) < 1e-9) return 1;
        const ratio = Math.sin(0.5 * count * u) / s;
        return (ratio * ratio) / (count * count);
    }

    /**
     * The family's raw lobe at circular offset u: 1 at u = 0, never negative.
     * Mirrors _phase_kernel (external_code.py) - see there for what each family
     * means. The partition is made by dividing by the sum of the shifted
     * copies, so summing to 1 is a property of that division and not of any
     * single family.
     */
    function phaseKernel(u, count, family, kappa) {
        if (family === "fejer") return fejerWeight(u, Math.max(count, 2));
        if (family === "mises") return Math.exp(kappa * (Math.cos(u) - 1));
        return 0.5 + 0.5 * Math.cos(u);
    }

    /**
     * Share of the envelope that Input `index` of `count` takes at wave angle
     * `theta`: its own lobe over the sum of all of them, so the count weights
     * sum to EXACTLY 1 at every theta in every family. Mirrors _phase_weight
     * (external_code.py).
     *
     * `count` 1 is the degenerate case and is deliberately NOT normalized (a
     * lobe over itself is a flat 1, i.e. the drawn wave would vanish): it
     * returns the kernel itself - one Input, one wave, not a partition of
     * anything.
     */
    function phaseWeight(theta, index, count, family, kappa) {
        const own = phaseKernel(theta - (TAU * index) / count, count, family, kappa);
        if (count <= 1) return own;
        let total = 0;
        for (let j = 0; j < count; j++) {
            total += phaseKernel(theta - (TAU * j) / count, count, family, kappa);
        }
        return total > 0 ? own / total : 1 / count;
    }

    /** How far the waves have converged onto the flat share at step x: 0 = the
     *  wave as drawn, 1 = the flat share, reached at `at` and held after.
     *  Twin of _converge_blend (external_code.py). */
    function convergeBlend(x, at, exponent) {
        if (at <= 0) return 1;
        return Math.pow(clamp01(x / at), exponent);
    }

    /**
     * One Input's share pulled `blend` of the way toward the flat 1/count.
     *
     * THE SUM IS SAFE BY CONSTRUCTION - a convex combination of two things
     * that each already sum to 1 over the Inputs, so the total stays exactly 1
     * for any blend. Neither the convergence position nor its dynamics appears
     * anywhere but inside `blend`, which is why no setting of them can make the
     * Inputs together pull more or less than the drawn envelope. Twin of
     * _converged_weight (external_code.py).
     */
    function convergedWeight(weight, count, blend) {
        if (blend <= 0) return weight;
        return weight + blend * (1 / count - weight);
    }

    /** Normalized y whose EFFECTIVE value is `neutral` (default the multiplier
     *  1) - where a neutral line sits on the axis it is drawn on. A zero-width
     *  range expresses a single value and cannot place anything in particular;
     *  keep the top. */
    function neutralY(lo, hi, neutral) {
        const n = neutral === undefined ? 1 : neutral;
        const span = hi - lo;
        if (Math.abs(span) < 1e-9) return 1;
        return clamp01((n - lo) / span);
    }

    /** Same effective value, expressed on another range. */
    function renormalize(y, from, to) {
        const span = to.hi - to.lo;
        if (Math.abs(span) < 1e-9) return clamp01(y);
        return clamp01((from.lo + y * (from.hi - from.lo) - to.lo) / span);
    }

    /** Move a profile's points (mid controls included - their y is normalized
     *  too) onto another range IN PLACE, keeping what they evaluate to. Used
     *  once per incoming string: the limits are the plot's, so a string that
     *  carries a different suffix per segment has to be folded onto one axis
     *  without changing a single weight. */
    function renormalizePoints(points, from, to) {
        if (from.lo === to.lo && from.hi === to.hi) return points;
        for (const p of points) {
            p.y = renormalize(p.y, from, to);
            if (p.mid) p.mid.y = renormalize(p.mid.y, from, to);
        }
        return points;
    }

    function fmt(v) {
        // 4 decimals = python's round(x, 4):g, so a python-authored string
        // (infotext conversion, API) survives the first editor edit unchanged
        return String(+v.toFixed(4));
    }

    function notifyGradio(textarea) {
        if (typeof updateInput === "function") {
            updateInput(textarea);
        } else {
            textarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
    }

    class WeightProfileEditor {
        constructor(container, textarea) {
            resolveBandColors();
            this.container = container;
            this.textarea = textarea;
            // the balance editor shares this class but has different neutral
            // semantics (flat 0.5) and a [0, 1] meaningful scale range
            this.isBalance = !!container.closest(".cnet-balance-profile-editor");
            this.unitRoot = container.closest(".input-accordion");
            this.canvas = container.querySelector(".cnet-weight-profile-canvas");
            this.canvas.tabIndex = 0;  // keyboard access (arrows / Delete)
            this.ctx = this.canvas.getContext("2d");
            const packed = this.parsePacked(textarea.value);
            const parsed = packed.main || this.defaultMainProfile(packed.scaleLo, packed.scaleHi);
            this.points = parsed.points;
            // ONE range per plot (see parsePacked), and this editor holds two
            // plots: the STEP axis below is the Y axis of the main curve and
            // the three band lines (they share one coordinate system), the
            // DEPTH axis further down belongs to the depth curve alone. The
            // two selects right of the canvas always edit the axis on screen.
            this.scaleLo = packed.scaleLo;
            this.scaleHi = packed.scaleHi;
            this.depthLo = packed.depthLo;
            this.depthHi = packed.depthHi;
            this.driftLo = packed.driftLo;
            this.driftHi = packed.driftHi;
            // cosine mode: the drawn polyline becomes the envelope of a wave
            // with cosN oscillations over the step range and phase cosPhase
            this.cosOn = parsed.cosOn;
            this.cosN = parsed.cosN || COS_DEFAULT_OSC;
            this.cosPhase = parsed.cosPhase;
            // multi-phase: the wave is replicated per Input, each shifted by
            // 2pi/n. null = plain single wave; 'cos' / 'fejer' / 'mises' pick
            // how the Inputs divide it ('P' / 'PF' / 'PV<kappa>'). kappa is
            // the von Mises hand-over sharpness and is meaningless otherwise.
            this.phaseFamily = parsed.phaseFamily || null;
            this.kappa = parsed.kappa || 0;
            // convergence: null = off, {at, exp} = the waves slide onto the
            // flat share 1/n of the envelope, arriving at step `at` with
            // dynamics `exp` and holding it after ('A<at>@<exp>')
            this.converge = parsed.converge || null;
            // response exponent (vertical slider in the range column): bends
            // the normalized [0, 1] profile with y -> y^gamma BEFORE the
            // scale mapping; 1 = linear (neutral), serialized as 'G<e>'
            this.gamma = parsed.gamma || 1;
            // the four selectable profiles: the working copy above IS
            // store[band] (points shared by reference; scalar fields are
            // re-snapshotted before anything reads the store)
            this.band = "main";
            this.store = { main: null };
            for (const b of BAND_ORDER) {
                this.store[b] = packed.bands[b] || defaultBandProfile(this.scaleLo, this.scaleHi);
            }
            // depth curve: same store, same neutral, but it is not a band -
            // see the DEPTH_KEY notes at the top. Its flat default sits on the
            // DEPTH axis, which is not the one the bands ride.
            this.store[DEPTH_KEY] = packed.depth
                || defaultBandProfile(this.depthLo, this.depthHi);
            // drift curve: a third plot again, and the only one whose neutral is
            // 0 rather than the multiplier 1 (see DRIFT_KEY)
            this.store[DRIFT_KEY] = packed.drift
                || defaultBandProfile(this.driftLo, this.driftHi, 0);
            this.snapshot();
            this.lastSerialized = textarea.value;
            this.dragPoint = null;
            this.dragMid = null;   // left vertex of the segment whose mid control is held
            this.dragPointerId = null;  // only this pointer moves the drag (multitouch)
            this.selPoint = null;  // last touched point, target of keyboard nudges
            this.attachEvents();
            this.attachKeyboard();
            this.attachPresets();
            this.attachScaleSelects();
            this.attachGammaSlider();
            this.attachBands();
            this.attachStepsWatch();
            // restore the mode the string was saved in (which selector is
            // pressed decides which profile drives the weights AND which
            // weight-mask slots are live - see publishMode)
            if (!this.isBalance && packed.selected !== "main") this.selectBand(packed.selected);
            this.publishMode();
            new ResizeObserver(() => this.resize()).observe(this.canvas);
            this.resize();
        }

        /**
         * Default main profile: flat at the EFFECTIVE value that does nothing -
         * the multiplier 1 for the weight editor, 0.5 for the balance one.
         *
         * The range is an ARGUMENT rather than `this.scaleLo/Hi` for two
         * reasons. The constructor calls this before those fields exist (they
         * are assigned from the same parse a line later), so reading them here
         * would silently produce NaN in exactly the path that handles a missing
         * profile. And "flat 1" is a position that depends on the axis: on the
         * default [0, 1] it is the top of the plot, on a [0, 2] axis the top is
         * the multiplier 2 and the neutral line sits in the middle. Callers
         * with the default axis therefore get the same profile they always did.
         */
        defaultMainProfile(lo, hi) {
            const y = neutralY(lo, hi, this.isBalance ? 0.5 : 1);
            return {
                points: [{ x: 0, y: y }, { x: 1, y: y }],
                cosOn: false, cosN: COS_DEFAULT_OSC, cosPhase: 0,
                phaseFamily: null, kappa: 0, converge: null, gamma: 1,
            };
        }

        /**
         * THE default of any slot, on the axis that slot is drawn on. One
         * function, because "reset this curve" and "this unit has no profile"
         * have to mean the same thing - the empty-value path in maybeReload
         * builds the same four objects, and two definitions of neutral that
         * drift apart is how a reset stops being a reset.
         *
         * Note what it does NOT reset: the plot's Y RANGE. That belongs to the
         * plot and is shared by the main profile and all three bands (see
         * attachScaleSelects), so clearing one curve must not move the other
         * three - the neutral is placed on whatever axis is currently up
         * instead. The empty-value path CAN reset the range, because there it
         * is clearing all of them at once.
         */
        defaultProfileFor(name) {
            const range = this.rangeFor(name);
            if (name === "main") return this.defaultMainProfile(range.lo, range.hi);
            return defaultBandProfile(range.lo, range.hi, neutralValueOf(name));
        }

        /**
         * Split a packed 'main#Cband#Mband#Fband#Ddepth' string into its
         * profiles, plus the range of each of the two plots they live on.
         *
         * The STEP-domain segments (main + the three bands) share one axis, so
         * a string whose suffixes disagree - hand-written, produced by python,
         * or saved before the limits became a property of the plot - is folded
         * onto the range COVERING all of them and every curve re-normalized
         * into it. That conversion is by effective value, so nothing that runs
         * changes; only the axis those curves share does. Once the editor has
         * written the string back they all carry the same suffix and this is
         * the identity.
         *
         * The '#D' and '#S' segments take NO part in that fold: depth and drift
         * are each their own plot with their own axis (see DEPTH_KEY and
         * DRIFT_KEY above), so their suffixes are adopted verbatim - which is
         * also what makes the split invisible to strings written before it,
         * since the shared suffix they carry on '#D' IS the axis that curve was
         * drawn on. Absent segment = that plot's default.
         */
        parsePacked(text) {
            const segments = (text || "").split("#");
            const raw = {};
            let selected = "main";
            for (const segment of segments.slice(1)) {
                if (segment[0] === "B") {
                    // mode marker (see serialize): which selector is pressed
                    const named = BAND_ORDER.find((b) => BAND_PREFIX[b] === segment[1]);
                    selected = named || "coarse";
                    continue;
                }
                const key = segment[0] === DEPTH_PREFIX
                    ? DEPTH_KEY
                    : (segment[0] === DRIFT_PREFIX
                        ? DRIFT_KEY
                        : BAND_ORDER.find((b) => BAND_PREFIX[b] === segment[0]));
                if (!key) continue;
                const p = this.parse(segment.slice(1));
                // FIRST segment wins on a duplicate key - the same rule as
                // python's parse_depth_profile (next(...)), so a hand-edited
                // string with two '#D' segments draws the one that runs.
                if (p && !(key in raw)) raw[key] = p;
            }
            const main = this.parse(segments[0]);
            if (!main) {
                return { main: null, bands: {}, depth: null, drift: null,
                         selected: selected,
                         scaleLo: 0, scaleHi: 1,
                         depthLo: DEPTH_RANGE_DEFAULT.lo, depthHi: DEPTH_RANGE_DEFAULT.hi,
                         driftLo: DRIFT_RANGE_DEFAULT.lo, driftHi: DRIFT_RANGE_DEFAULT.hi };
            }
            // every field of the grammar, main or not: a band CAN be put on a
            // family rung or given a convergence (the ladder acts on whichever
            // curve is selected) and serializeProfile writes both out, so
            // dropping them here would lose them on the next reload - and make
            // the editor draw a band that python parses differently
            const toProfile = (p) => ({
                points: p.points,
                cosOn: p.cosOn, cosN: p.cosN, cosPhase: p.cosPhase,
                phaseFamily: p.phaseFamily || null, kappa: p.kappa || 0,
                converge: p.converge || null,
                gamma: p.gamma,
            });
            // step axis: covers main + the band segments; never the depth or
            // drift ones, each of which is a plot of its own
            const ownPlot = (key) => key === DEPTH_KEY || key === DRIFT_KEY;
            let lo = main.scaleLo;
            let hi = main.scaleHi;
            for (const key in raw) {
                if (ownPlot(key)) continue;
                lo = Math.min(lo, raw[key].scaleLo);
                hi = Math.max(hi, raw[key].scaleHi);
            }
            const to = { lo: lo, hi: hi };
            renormalizePoints(main.points, { lo: main.scaleLo, hi: main.scaleHi }, to);
            const bands = {};
            for (const key in raw) {
                if (ownPlot(key)) continue;
                const p = raw[key];
                renormalizePoints(p.points, { lo: p.scaleLo, hi: p.scaleHi }, to);
                bands[key] = toProfile(p);
            }
            // depth and drift axes: whatever their own segments say, untouched
            const rawDepth = raw[DEPTH_KEY];
            const rawDrift = raw[DRIFT_KEY];
            return {
                main: main, bands: bands,
                depth: rawDepth ? toProfile(rawDepth) : null,
                drift: rawDrift ? toProfile(rawDrift) : null,
                selected: selected,
                scaleLo: lo, scaleHi: hi,
                depthLo: rawDepth ? rawDepth.scaleLo : DEPTH_RANGE_DEFAULT.lo,
                depthHi: rawDepth ? rawDepth.scaleHi : DEPTH_RANGE_DEFAULT.hi,
                driftLo: rawDrift ? rawDrift.scaleLo : DRIFT_RANGE_DEFAULT.lo,
                driftHi: rawDrift ? rawDrift.scaleHi : DRIFT_RANGE_DEFAULT.hi,
            };
        }

        /** The Y axis a stored profile lives on: the depth and drift curves each
         *  have their own, everything else rides the step plot's. */
        rangeFor(name) {
            if (name === DEPTH_KEY) return { lo: this.depthLo, hi: this.depthHi };
            if (name === DRIFT_KEY) return { lo: this.driftLo, hi: this.driftHi };
            return { lo: this.scaleLo, hi: this.scaleHi };
        }

        /** The axis currently on screen - what the two range selects edit. */
        activeRange() {
            return this.rangeFor(this.band);
        }

        /**
         * True for the three selectors that all run MAIN mode: main itself, the
         * depth curve and the drift curve. They are one group because none of
         * them REPLACES the main profile - depth multiplies it, drift moves
         * where depth is read - so the unit runs main(step) x depth(...) with
         * whichever of the three is pressed. The C/M/F selectors are the other
         * group: each is a whole alternative to the main profile.
         *
         * That distinction decides three separate things (the '#B' mode marker,
         * which curves are drawn and grabbable, and which weight-mask slots are
         * live), and it used to be spelled out as `band !== "main" && band !==
         * DEPTH_KEY` at each of them. Adding a third main-mode selector to a
         * repeated condition is exactly how one call site keeps the old
         * two-member group - see ARCHITECTURE.md section 8.
         */
        mainModeSelector(name) {
            return name === "main" || name === DEPTH_KEY || name === DRIFT_KEY;
        }

        /** The line color of one stored curve. Single lookup, so the plot line,
         *  the drag readout and the button (which takes the same CSS variable
         *  through --band-color) cannot end up disagreeing about which curve is
         *  which color. */
        curveColorOf(name) {
            if (name === "main") return MAIN_COLOR;
            if (name === DEPTH_KEY) return DEPTH_COLOR;
            if (name === DRIFT_KEY) return DRIFT_COLOR;
            return BAND_COLORS[name];
        }

        /** Refresh the store slot of the selected profile from the working copy.
         *  The range is NOT part of a slot - it belongs to the plot. */
        snapshot() {
            this.store[this.band] = {
                points: this.points,
                cosOn: this.cosOn, cosN: this.cosN, cosPhase: this.cosPhase,
                phaseFamily: this.phaseFamily || null, kappa: this.kappa || 0,
                converge: this.converge || null,
                gamma: this.gamma || 1,
            };
        }

        /**
         * Put one curve back to its default: flat neutral, no wave, no
         * convergence, no response exponent, no parabola handles.
         *
         * Reached by double-clicking that profile's selector bar, and the WHOLE
         * profile goes - not just its points. A curve whose points were cleared
         * while an oscillation, a multi-phase family or a bent response stayed
         * lit would still be drawn as a wave, and the buttons above it would
         * still be pressed: a "reset" that leaves the control panel armed is the
         * worst of both. defaultProfileFor is the single answer to what neutral
         * is, shared with the empty-value path in maybeReload.
         *
         * Reloading is not optional when the target is the SELECTED curve: the
         * working copy holds `points` by reference and the wave fields by
         * value, so replacing the store entry alone would leave the editor
         * drawing the old arrays. loadBand re-points both and refreshes the
         * preset column with them.
         */
        resetProfile(name) {
            this.store[name] = this.defaultProfileFor(name);
            this.selPoint = null;
            if (name === this.band) this.loadBand(name);
            this.draw();
            this.sync();
        }

        /** Load a stored profile into the working copy (the editable one). */
        loadBand(name) {
            const P = this.store[name];
            this.band = name;
            this.points = P.points;
            this.cosOn = P.cosOn;
            this.cosN = P.cosN;
            this.cosPhase = P.cosPhase;
            this.phaseFamily = P.phaseFamily || null;
            this.kappa = P.kappa || 0;
            this.converge = P.converge || null;
            this.gamma = P.gamma || 1;
            this.syncPadFromWave();
            this.setOscButtonState();
            this.setConvergeButtonState();
            this.updatePad();
            this.updateScaleSelects();
            this.updateGammaSlider();
        }

        /**
         * Band selector: six thin bars at the presets column bottom, in two
         * groups. Radio semantics - exactly one pressed.
         *
         *   main | depth | drift    the three that ASSEMBLE into one unit: the
         *                           unit runs main(step) x depth(layer -
         *                           drift(step)), so pressing any of them runs
         *                           the same thing and only changes which factor
         *                           is on screen (mainModeSelector).
         *   coarse | mid | fine     the three that REPLACE the main profile:
         *                           each is a whole per-step curve for its third
         *                           of the depth axis.
         *
         * The separator between the groups is the whole of that distinction on
         * screen, which is why it is markup (a real element in the row) rather
         * than a margin: the row's reading order is the model.
         *
         * A band button shows all three C/M/F lines and routes every edit
         * (points, presets, pad, invert) to the selected band. The range selects
         * are the axis of the PLOT, not of the selected curve: in main / band
         * mode they move the main profile and all three bands together, in depth
         * or drift mode that one curve alone (see the header - the three are
         * different coordinate systems and never share a range).
         */
        attachBands() {
            this.bandButtons = {};
            this.container.querySelectorAll(".cnet-profile-band").forEach((button) => {
                this.bandButtons[button.dataset.band] = button;
                button.setAttribute("aria-pressed", String(button.dataset.band === this.band));
                button.addEventListener("click", (e) => {
                    e.preventDefault();
                    this.selectBand(button.dataset.band);
                });
                // Double-click = clear that profile. The button IS its line (a
                // solid bar in the line's colour), so this reads as "wipe this
                // curve" and needs no target of its own on the plot - where the
                // three double-click gestures are already taken by the point and
                // the mid handle. Same idiom as the response slider's
                // double-click reset. The first of the two clicks selects the
                // profile, which is what makes the wipe visible: whatever you
                // clear, you are looking at.
                button.addEventListener("dblclick", (e) => {
                    e.preventDefault();
                    this.resetProfile(button.dataset.band);
                });
            });
        }

        selectBand(name) {
            if (!(name in this.store) || name === this.band) return;
            this.snapshot();
            this.loadBand(name);
            for (const key in this.bandButtons) {
                const pressed = key === name;
                this.bandButtons[key].classList.toggle("cnet-profile-band-active", pressed);
                this.bandButtons[key].setAttribute("aria-pressed", String(pressed));
            }
            this.draw();
            // the selection is the mode, not just an edit target: push it so
            // the backend switches with the button
            this.sync();
            this.publishMode();
        }

        /**
         * Publish the pressed selector on the unit root, for the parts of the
         * widget that are not this editor.
         *
         * THE WEIGHT MASKS FOLLOW THIS SELECTOR. The four mask slots are the
         * four profiles' spatial half - G belongs to the main profile, C/M/F to
         * the band profiles - so which slots are live is decided here and
         * nowhere else (python reads the same thing out of the profile string
         * as '#B<band>'; see external_code.masks_in_force). weight_mask.js
         * reads this attribute to dim the slots the selection does not use,
         * which is the only warning a user gets before painting into a mask
         * that will not be applied.
         *
         * The RAW selector is published, depth and drift included: the reader
         * decides what they mean for it (for masks both mean main, because they
         * shape main rather than replacing it). Publishing a pre-interpreted
         * value would put that rule in the writer, where the next reader cannot
         * see it.
         */
        publishMode() {
            if (this.isBalance || !this.unitRoot) return;
            if (this.unitRoot.dataset.cnetProfileBand !== this.band) {
                this.unitRoot.dataset.cnetProfileBand = this.band;
            }
        }

        /** Whole-token number, python-strict. parseFloat prefix-parses
         *  ("1,5" -> 1, "0.5x" -> 0.5) what python's float() refuses, so the
         *  editor adopted and drew curves the generation side replaces with
         *  the constant-weight fallback - the plot showing a profile that
         *  will not run. Invalid on both sides beats valid on one. */
        strictFloat(s) {
            const t = String(s).trim();
            if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t)) return NaN;
            return parseFloat(t);
        }

        parse(text) {
            if (!text || !text.trim()) return null;
            const parts = text.split("|");
            let scaleLo = 0;
            let scaleHi = 1;
            if (parts.length > 1) {
                // scale-range suffix: '|hi' (legacy, lo = 0) or '|lo~hi'
                const clampScale = (v) => Math.min(Math.max(v, SCALE_MIN), SCALE_MAX);
                const bounds = parts[1].split("~");
                if (bounds.length > 1) {
                    scaleLo = this.strictFloat(bounds[0]);
                    scaleHi = this.strictFloat(bounds[1]);
                } else {
                    scaleHi = this.strictFloat(bounds[0]);
                }
                if (!isFinite(scaleLo) || !isFinite(scaleHi)) return null;
                scaleLo = clampScale(scaleLo);
                scaleHi = clampScale(scaleHi);
                if (scaleLo > scaleHi) {
                    const swap = scaleLo;
                    scaleLo = scaleHi;
                    scaleHi = swap;
                }
            }
            const points = [];
            const mids = [];
            let cosOn = false;
            // A string with no 'C' token carries no oscillation count, and the
            // default has to be the same one a brand new profile gets: the
            // buttons that switch the wave ON (the ladder, the convergence
            // toggle) do not invent one, so a 0 here made them produce
            // 'C0@0' - a wave with zero oscillations, i.e. a flat factor of 1.
            // The click then appeared to do nothing at all. An EXPLICIT 'C0@0'
            // still parses as zero, because that sets cosOn with it.
            let cosN = COS_DEFAULT_OSC;
            let cosPhase = 0;
            let phaseFamily = null;
            let kappa = 0;
            let converge = null;
            let gamma = 1;
            for (const pair of parts[0].split(";")) {
                let token = pair.trim();
                if (!token) continue;
                // 'P...' = multi-phase marker: with several Inputs each one
                // runs this wave shifted by 2pi/n (see the oscillatory
                // button's ladder). The suffix picks the family - bare 'P'
                // cosine, 'PF' Fejer, 'PV<kappa>' von Mises - and an
                // unrecognized one degrades to the cosine rather than failing
                // the parse, matching _parse_phase_family in python.
                if (token[0] === "P") {
                    const body = token.slice(1);
                    if (body[0] === "F" || body[0] === "f") {
                        phaseFamily = "fejer";
                    } else if (body[0] === "V" || body[0] === "v") {
                        const k = parseFloat(body.slice(1));
                        phaseFamily = "mises";
                        kappa = isFinite(k) ? Math.min(Math.max(k, 0), KAPPA_MAX) : 0;
                    } else {
                        phaseFamily = "cos";
                    }
                    continue;
                }
                // 'C<oscillations>@<phase>' = cosine mode; the drawn points are
                // the envelope of that wave (see waveFactor / evaluate)
                if (token[0] === "C") {
                    const co = token.slice(1).split("@");
                    if (co.length !== 2) return null;
                    const n = this.strictFloat(co[0]);
                    const ph = this.strictFloat(co[1]);
                    if (!isFinite(n) || !isFinite(ph)) return null;
                    cosOn = true;
                    cosN = Math.min(Math.max(n, 0), COS_MAX_OSC);
                    cosPhase = ((ph % TAU) + TAU) % TAU;
                    continue;
                }
                // 'A<at>@<exponent>' = convergence: the waves slide onto the
                // flat share 1/n of the envelope, arriving at step `at` and
                // holding it after (see convergeBlend / convergedWeight)
                if (token[0] === "A") {
                    const cv = token.slice(1).split("@");
                    if (cv.length !== 2) return null;
                    const at = this.strictFloat(cv[0]);
                    const exp = this.strictFloat(cv[1]);
                    if (!isFinite(at) || !isFinite(exp)) return null;
                    converge = {
                        at: clamp01(at),
                        exp: Math.min(Math.max(exp, 1 / CONVERGE_EXP_MAX), CONVERGE_EXP_MAX),
                    };
                    continue;
                }
                // 'G<exponent>' = response exponent: the normalized profile is
                // bent with y -> y^e BEFORE the scale mapping (see gammaAt)
                if (token[0] === "G") {
                    const g = this.strictFloat(token.slice(1));
                    if (!isFinite(g)) return null;
                    gamma = Math.min(Math.max(g, 1 / GAMMA_MAX), GAMMA_MAX);
                    continue;
                }
                // 'Mx@y' = active mid control of the segment containing x
                const isMid = token[0] === "M";
                if (isMid) token = token.slice(1);
                const xy = token.split("@");
                if (xy.length !== 2) return null;
                const x = this.strictFloat(xy[0]);
                const y = this.strictFloat(xy[1]);
                if (!isFinite(x) || !isFinite(y)) return null;
                (isMid ? mids : points).push({ x: clamp01(x), y: clamp01(y) });
            }
            if (!points.length) return null;
            points.sort((a, b) => a.x - b.x);
            // attach each mid control to the segment strictly containing its x
            // (stored relative to the segment, so it follows the vertices);
            // orphaned mids are dropped. FIRST mid in token order wins when a
            // segment carries several - the same rule as python's
            // _flatten_profile_mids (next(...)), so hand-edited strings parse
            // identically on both sides.
            for (const m of mids) {
                for (let i = 0; i + 1 < points.length; i++) {
                    const dx = points[i + 1].x - points[i].x;
                    if (dx > 0 && m.x > points[i].x && m.x < points[i + 1].x) {
                        if (!points[i].mid) points[i].mid = { t: (m.x - points[i].x) / dx, y: m.y };
                        break;
                    }
                }
            }
            return {
                points: points, scaleLo: scaleLo, scaleHi: scaleHi,
                cosOn: cosOn, cosN: cosN, cosPhase: cosPhase,
                phaseFamily: phaseFamily, kappa: kappa, converge: converge,
                gamma: gamma,
            };
        }

        serializeProfile(P, range) {
            const tokens = [];
            for (let i = 0; i < P.points.length; i++) {
                const p = P.points[i];
                tokens.push(`${fmt(p.x)}@${fmt(p.y)}`);
                const next = P.points[i + 1];
                if (p.mid && next && next.x - p.x > 0) {
                    tokens.push(`M${fmt(p.x + p.mid.t * (next.x - p.x))}@${fmt(p.mid.y)}`);
                }
            }
            if (P.cosOn) tokens.push(`C${fmt(P.cosN)}@${fmt(P.cosPhase)}`);
            // the family marker only means something ON a wave, and the ladder
            // cannot reach a family without one - but a hand-written string
            // can, and writing 'PF' with no 'C' back out would keep a marker
            // alive that nothing acts on
            if (P.cosOn && P.phaseFamily === "cos") tokens.push("P");
            if (P.cosOn && P.phaseFamily === "fejer") tokens.push("PF");
            if (P.cosOn && P.phaseFamily === "mises") tokens.push(`PV${fmt(P.kappa || 0)}`);
            // convergence needs a wave for the same reason the family marker
            // does: there is nothing to converge without one
            if (P.cosOn && P.converge) {
                tokens.push(`A${fmt(P.converge.at)}@${fmt(P.converge.exp)}`);
            }
            if (Math.abs((P.gamma || 1) - 1) > 1e-4) tokens.push(`G${fmt(P.gamma)}`);
            const base = tokens.join(";");
            // the range is the PLOT's, so every segment of one plot gets the
            // same suffix: python parses each one on its own and all of them
            // have to land on the weights that plot is showing. The depth
            // segment is a plot of its own and is passed its own range.
            const lo = range.lo;
            const hi = range.hi;
            if (lo === 0 && hi === 1) return base;
            if (lo === 0) return base + "|" + fmt(hi);
            return base + "|" + fmt(lo) + "~" + fmt(hi);
        }

        /** Flat at its neutral value with no wave = a no-op: not worth
         *  serializing. Compares EFFECTIVE (scale-mapped) values with the same
         *  5e-4 epsilon as python's profile_points_are_neutral - a band nudged
         *  to 0.9997 must be neutral on both sides, or the editor would show
         *  band mode engaged while the backend runs the main profile. The
         *  mapping is the range of the profile's own plot, so where a neutral
         *  line sits depends on those limits (the multiplier 1 is the top at
         *  [0, 1] and the middle at [0, 2]) - which is why the caller says which
         *  axis P is on. `neutral` defaults to the multiplier 1; the drift curve
         *  is the one caller that passes 0, because it is a shift. */
        bandNeutral(P, range, neutral) {
            const n = neutral === undefined ? 1 : neutral;
            const value = (y) => range.lo + y * (range.hi - range.lo);
            // The response exponent is part of what the plot draws (evaluate()
            // bends the normalized value through gammaAt), so a flat curve is
            // only a no-op when the bend cannot move it: gamma ~ 1, or every
            // point sitting at a fixed point of v^g (normalized 0 or 1).
            // Without this term a depth curve left flat at neutral (normalized
            // 0.5 on its [0, 2] axis) with a bent response DREW a global x0.5
            // attenuation while serializing nothing - the plot showed one
            // curve, python ran another. Epsilons: 1e-4 matches gammaAt's own
            // identity test, 5e-4 the value test below.
            const g = P.gamma || 1;
            const gammaInert = Math.abs(g - 1) <= 1e-4
                || P.points.every((p) => p.y <= 1e-4 || p.y >= 1 - 1e-4);
            return !P.cosOn
                && gammaInert
                && P.points.every((p) => Math.abs(value(p.y) - n) <= 5e-4 && !p.mid);
        }

        serialize() {
            this.snapshot();
            const stepRange = this.rangeFor("main");
            const depthRange = this.rangeFor(DEPTH_KEY);
            let out = this.serializeProfile(this.store.main, stepRange);
            // the balance plot has no band buttons: its band slots are never
            // reachable, so they must never reach the string either (they ride
            // the shared range and a non-[0,1] balance range would push them
            // off their neutral value and serialize three phantom segments)
            if (this.isBalance) return out;
            for (const b of BAND_ORDER) {
                const P = this.store[b];
                if (!this.bandNeutral(P, stepRange)) {
                    out += "#" + BAND_PREFIX[b] + this.serializeProfile(P, stepRange);
                }
            }
            const depth = this.store[DEPTH_KEY];
            if (depth && !this.bandNeutral(depth, depthRange)) {
                out += "#" + DEPTH_PREFIX + this.serializeProfile(depth, depthRange);
            }
            // the drift rides the STEP axis in x but its own axis in y, and its
            // neutral is 0 rather than the multiplier 1 (see DRIFT_KEY)
            const driftRange = this.rangeFor(DRIFT_KEY);
            const drift = this.store[DRIFT_KEY];
            if (drift && !this.bandNeutral(drift, driftRange, 0)) {
                out += "#" + DRIFT_PREFIX + this.serializeProfile(drift, driftRange);
            }
            // Which selector is pressed IS the mode, and the mode is behaviour,
            // so it has to survive a reload: '#B<band>' says "bands drive the
            // weights, this one is being edited", no marker says "main drives".
            // The DEPTH and DRIFT selectors write no marker - neither replaces
            // the main profile, they shape it, so main mode is what runs
            // whichever of the three is pressed and the '#D' / '#S' segments
            // alone say whether they are doing anything. Every curve stays
            // serialized in all modes, so switching back finds the others
            // exactly as they were left.
            if (!this.mainModeSelector(this.band)) {
                out += "#B" + BAND_PREFIX[this.band];
            }
            return out;
        }

        sync() {
            const serialized = this.serialize();
            if (serialized !== this.lastSerialized) {
                this.lastSerialized = serialized;
                this.textarea.value = serialized;
                notifyGradio(this.textarea);
            }
        }

        /**
         * Evaluate the profile at relative position x in [0, 1]. The python
         * side matches this up to the documented approximations:
         * parse_weight_profile (external_code.py) flattens parabola segments
         * into a clamped 24-chord polyline (this evaluates the exact
         * parabola; divergence < ~1e-3), and evaluate_weight_profile
         * (cnpro_core/weight_profile.py) stays piecewise-linear.
         */
        evaluate(x) {
            return this.gammaAt(this.envelopeAt(x) * this.waveFactor(x, 0));
        }

        /**
         * THE wave multiplier, for any stored profile and any Input - the JS
         * face of _wave_factor (external_code.py) and the only place either
         * side of the editor computes one.
         *
         * `name` is the profile's slot, because only the MAIN profile ever
         * fans out: python reads the 'P' marker off the main segment alone
         * (weight_profile_phase_family) and parses every other segment with a
         * count of 1, so a band drawn on a family rung has to preview as the
         * single kernel it will actually run, not as a split.
         */
        waveFactorOf(P, name, x, index) {
            if (!P.cosOn) return 1;
            const theta = TAU * (P.cosN || 0) * x - (P.cosPhase || 0);
            const count = this.waveCountOf(P, name);
            // family null = no marker: the plain single wave, which is the
            // cosine kernel at a count of 1 - same call, no second formula
            const w = phaseWeight(theta, index, count,
                                  P.phaseFamily || "cos", P.kappa || 0);
            if (!P.converge) return w;
            return convergedWeight(
                w, count, convergeBlend(x, P.converge.at, P.converge.exp));
        }

        /**
         * Waves a stored profile is drawn as: the Input count for the main
         * profile on a family rung, 1 for everything else (see waveFactorOf).
         * A count of 1 is a real state now, not a floor - a unit holding one
         * Input draws and runs ONE wave, the family's kernel.
         */
        waveCountOf(P, name) {
            // The balance editor's working slot is also called "main", but
            // python always evaluates a balance profile with a count of 1
            // (only the weight profile fans out over Inputs) - so a
            // hand-written 'P' in a balance string must not make the preview
            // divide the wave by the Input count. Preview what runs.
            if (this.isBalance) return 1;
            // main AND the three bands fan out: a band is a step curve over one
            // third of the depth axis, and python partitions it per Input just
            // like the main curve (external_code.parse_band_profiles takes
            // phase_index/phase_count). DEPTH and DRIFT are excluded - depth has
            // no step axis to partition, and the drift's y is a shift, not a
            // weight, so a partition of unity means nothing on it.
            const fansOut = name === "main" || BAND_ORDER.indexOf(name) !== -1;
            return (fansOut && P.cosOn && P.phaseFamily)
                ? this.phaseCount() : 1;
        }

        /**
         * Response exponent: bends the NORMALIZED [0, 1] value with y -> y^g
         * before the scale mapping (toScreen applies the range afterwards, so
         * the bend is range-independent, matching python's
         * _apply_profile_gamma). 1 = linear = no-op, bit-exactly.
         */
        gammaAt(v) {
            const g = this.gamma || 1;
            if (Math.abs(g - 1) <= 1e-4) return v;
            return Math.pow(Math.min(Math.max(v, 0), 1), g);
        }

        /**
         * The wave multiplier Input `index` runs on the WORKING COPY, matching
         * _apply_profile_wave in lib_cnpro/external_code.py: the drawn profile
         * is the ENVELOPE of the wave, and the wave pulses between 0 and that
         * envelope (it never flips sign, so the scale range keeps its meaning).
         * The phase is SUBTRACTED so that dragging the pad point right shifts
         * the wave right, which is what the gesture reads as.
         *
         * The working copy carries the same field names as a stored profile,
         * so this is waveFactorOf on `this` - one implementation, and the
         * selected curve cannot end up drawn by different math from the rest.
         */
        waveFactor(x, index) {
            return this.waveFactorOf(this, this.band, x, index);
        }

        /**
         * Input count the phase families divide the wave between. Resolved
         * once per frame by draw() (multiPhaseCount walks the DOM) and cached
         * here, because this runs once per curve sample - hundreds per frame -
         * and counting the Inputs walks the unit's DOM.
         *
         * Floors at 1, which is not a fallback but the honest degenerate case:
         * a unit with a single Input (or none loaded yet) fans out nowhere, and
         * both sides then run the family's kernel as one wave.
         */
        phaseCount() {
            return this._phaseCount || 1;
        }

        /**
         * Effective value of the sibling wave a further Input runs in
         * multi-phase mode: same envelope, same response exponent, that
         * Input's share of the wave. Its preview line and its per-step dots
         * both go through here, so the dots cannot land off the line they
         * sample.
         */
        phaseValueAt(x, index) {
            return this.gammaAt(this.envelopeAt(x) * this.waveFactor(x, index));
        }

        /** The drawn/editable polyline itself (the envelope in cosine mode). */
        envelopeAt(x) {
            return this.envelopeOf(this.points, x);
        }

        /** Same, for an arbitrary stored profile's points (band overlays). */
        envelopeOf(pts, x) {
            if (x <= pts[0].x) return pts[0].y;
            for (let i = 1; i < pts.length; i++) {
                if (x <= pts[i].x) {
                    const p0 = pts[i - 1];
                    const p1 = pts[i];
                    const dx = p1.x - p0.x;
                    if (dx <= 0) return p1.y;
                    if (p0.mid) {
                        // parabola (Lagrange quadratic) through the two
                        // vertices and the active mid control, clamped to the
                        // plot range so the scale mapping stays valid
                        const xm = p0.x + p0.mid.t * dx;
                        const ym = p0.mid.y;
                        const l0 = ((x - xm) * (x - p1.x)) / ((p0.x - xm) * (p0.x - p1.x));
                        const lm = ((x - p0.x) * (x - p1.x)) / ((xm - p0.x) * (xm - p1.x));
                        const l1 = ((x - p0.x) * (x - xm)) / ((p1.x - p0.x) * (p1.x - xm));
                        return clamp01(p0.y * l0 + ym * lm + p1.y * l1);
                    }
                    return p0.y + ((p1.y - p0.y) * (x - p0.x)) / dx;
                }
            }
            return pts[pts.length - 1].y;
        }

        /** Pick up changes made from the python side (e.g. infotext paste). */
        maybeReload() {
            const value = this.textarea.value;
            if (value === this.lastSerialized) return;
            const packed = this.parsePacked(value);
            this.lastSerialized = value;
            if (packed.main) {
                // all three ranges come with the string: the step one covers
                // every step-domain segment in it (parsePacked folds divergent
                // suffixes onto one axis), the depth and drift ones are the '#D'
                // and '#S' segments' own
                this.scaleLo = packed.scaleLo;
                this.scaleHi = packed.scaleHi;
                this.depthLo = packed.depthLo;
                this.depthHi = packed.depthHi;
                this.driftLo = packed.driftLo;
                this.driftHi = packed.driftHi;
                this.store.main = {
                    points: packed.main.points,
                    cosOn: packed.main.cosOn, cosN: packed.main.cosN,
                    cosPhase: packed.main.cosPhase,
                    phaseFamily: packed.main.phaseFamily, kappa: packed.main.kappa,
                    converge: packed.main.converge,
                    gamma: packed.main.gamma,
                };
                for (const b of BAND_ORDER) {
                    this.store[b] = packed.bands[b]
                        || defaultBandProfile(this.scaleLo, this.scaleHi);
                }
                this.store[DEPTH_KEY] = packed.depth
                    || defaultBandProfile(this.depthLo, this.depthHi);
                this.store[DRIFT_KEY] = packed.drift
                    || defaultBandProfile(this.driftLo, this.driftHi, 0);
                this.loadBand(this.band); // refresh the working copy in place
                // the incoming string also carries the mode (which profile
                // drives the weights), so follow it
                if (!this.isBalance && packed.selected !== this.band) {
                    this.selectBand(packed.selected);
                }
                this.publishMode();
                this.draw();            } else if (!value.trim()) {
                // empty = neutral (e.g. an infotext paste of a unit that
                // omitted the profile): reset to the default drawing instead
                // of keeping the stale curve - otherwise the display would
                // disagree with the unit and the next edit would resurrect
                // the old curve into it
                // the limits are part of what runs, so they reset with the
                // curve - a default flat profile left on a [0, 2] axis would
                // be weight 2, not the neutral the empty value asks for
                this.scaleLo = 0;
                this.scaleHi = 1;
                this.depthLo = DEPTH_RANGE_DEFAULT.lo;
                this.depthHi = DEPTH_RANGE_DEFAULT.hi;
                this.driftLo = DRIFT_RANGE_DEFAULT.lo;
                this.driftHi = DRIFT_RANGE_DEFAULT.hi;
                this.store.main = this.defaultMainProfile(this.scaleLo, this.scaleHi);
                for (const b of BAND_ORDER) {
                    this.store[b] = defaultBandProfile(this.scaleLo, this.scaleHi);
                }
                this.store[DEPTH_KEY] = defaultBandProfile(this.depthLo, this.depthHi);
                this.store[DRIFT_KEY] = defaultBandProfile(this.driftLo, this.driftHi, 0);
                // ...and the MODE resets with them. An empty value means "this
                // unit has no profile", which is main mode - keeping a band
                // selector pressed would leave the string about to be written
                // back carrying '#B<band>' that the value it came from does not
                // have. That was cosmetic while the marker only chose a curve;
                // it is not any more, because the marker also chooses which
                // weight-mask slots are live (publishMode), so a stale band
                // selection would silently switch off a painted G mask.
                // ADOPTION, not an edit - two rules, both load-bearing:
                // 1. loadBand runs FIRST. selectBand starts with snapshot(),
                //    which writes the WORKING COPY back into its store slot;
                //    called before loadBand refreshed it, that resurrected the
                //    pre-reset curve into the freshly reset slot, and sync()
                //    then pushed it to the server - a "cleared" depth curve
                //    silently multiplying every following generation.
                // 2. No sync()/selectBand at all: this value came FROM the
                //    server, and adopting a server-authored value must not
                //    write anything back (writing an explicit-neutral string
                //    would turn the unit's empty field non-empty). The band
                //    buttons and the published mode are updated by hand
                //    instead.
                this.selPoint = null;
                this.loadBand(this.isBalance ? this.band : "main");
                if (!this.isBalance && this.bandButtons) {
                    for (const key in this.bandButtons) {
                        const pressed = key === "main";
                        this.bandButtons[key].classList.toggle("cnet-profile-band-active", pressed);
                        this.bandButtons[key].setAttribute("aria-pressed", String(pressed));
                    }
                }
                this.publishMode();
                this.draw();            } else if (this._warnedInvalid !== value) {
                // invalid non-empty: keep drawing the old profile, but say so
                // once - the generation side will warn about it too
                this._warnedInvalid = value;
                console.warn("[controlnet profile] ignoring invalid profile value:", value);
            }
        }

        /**
         * Presets: 'step' (button) and 'osc' (the oscillatory ladder - see
         * OSC_LADDER and cycleOsc). The square pad next to them carries the two
         * parameters of whichever preset currently holds focus.
         *
         * Focus rule: clicking a preset that is NOT focused only binds the pad
         * to it - the existing profile is kept, nothing is regenerated.
         * Clicking a preset that IS already focused regenerates the profile
         * from the pad. Moving the pad point always rewrites the profile of the
         * focused preset. 'osc' sits outside that rule: every one of its rungs
         * is non-destructive, so it acts on the first click and takes focus as
         * a side effect of having a wave to parameterize.
         *
         * There is no invert toggle: the step preset spans both directions
         * through the pad's vertical halves (see applyPadPreset), which is
         * what invert was overwhelmingly used for, and a mode that silently
         * mirrored every future preset was state the plot could not show.
         */
        attachPresets() {
            this.activePreset = null;
            this.presetButtons = {};
            // pad position per preset, so switching back restores its parameters
            this.padPos = {
                step: { x: 0.5, y: 1 },
                osc: { x: 0, y: 0 },
                converge: { x: CONVERGE_DEFAULT.at, y: 0.5 },
            };
            this.syncPadFromWave();
            this.pad = this.container.querySelector(".cnet-profile-preset-pad");
            this.padDot = this.container.querySelector(".cnet-profile-pad-dot");

            this.container.querySelectorAll(".cnet-profile-preset").forEach((button) => {
                this.presetButtons[button.dataset.preset] = button;
                button.addEventListener("click", (e) => {
                    e.preventDefault();
                    const name = button.dataset.preset;
                    if (name === "osc") {
                        // Non-destructive at every rung: the drawn polyline is
                        // kept throughout (the wave only makes it the
                        // envelope), so there is nothing to protect the user
                        // from and the click acts immediately - no second
                        // confirming click, unlike step below.
                        this.cycleOsc();
                        return;
                    }
                    if (name === "converge") {
                        // same reasoning: it modulates the wave and touches no
                        // drawn point, so it toggles on the first click
                        this.toggleConverge();
                        return;
                    }
                    // Step is DESTRUCTIVE - it replaces the drawn profile - so
                    // it takes a confirming second click: the first one only
                    // aims the pad at it.
                    if (this.activePreset !== name) {
                        this.setActivePreset(name);
                        this.drawAndSync();
                        return;
                    }
                    this.applyPadPreset();
                });
            });

            if (this.pad) {
                // during a drag only the DRAWING updates (defer = true);
                // serialization + the ws listing wait for pointerup, exactly
                // like canvas point drags - a pad sweep must not serialize at
                // pointer rate
                const padTo = (e, defer) => {
                    const r = this.pad.getBoundingClientRect();
                    if (!r.width || !r.height) return;
                    const pos = this.padPos[this.activePreset];
                    if (!pos) return;
                    pos.x = clamp01((e.clientX - r.left) / r.width);
                    // pad y is measured upwards, like the plot
                    pos.y = clamp01(1 - (e.clientY - r.top) / r.height);
                    this.applyPadPreset(defer);
                };
                this.pad.addEventListener("pointerdown", (e) => {
                    if (!this.activePreset || this.padDragging) return;
                    e.preventDefault();
                    this.pad.setPointerCapture(e.pointerId);
                    this.padDragging = true;
                    this.padPointerId = e.pointerId;
                    padTo(e, true);
                });
                this.pad.addEventListener("pointermove", (e) => {
                    if (this.padDragging && e.pointerId === this.padPointerId) padTo(e, true);
                });
                const endDrag = (e) => {
                    if (!this.padDragging || e.pointerId !== this.padPointerId) return;
                    this.padDragging = false;
                    this.padPointerId = null;
                    try {
                        this.pad.releasePointerCapture(e.pointerId);
                    } catch (err) {}
                    this.sync();                };
                this.pad.addEventListener("pointerup", endDrag);
                this.pad.addEventListener("pointercancel", endDrag);
            }
            this.updatePad();
        }

        /**
         * Scale range as two selects right of the plot (top = range maximum,
         * bottom = range minimum), replacing the old two-handle gutter slider.
         * Picking a bottom above the current top (or vice versa) pushes the
         * other one along, so the range can never invert.
         *
         * The pair is the PLOT's Y axis, not a property of the selected curve -
         * but this editor has two plots, so it is the axis CURRENTLY ON SCREEN
         * (see the header): in main / band mode it moves the main profile and
         * all three band lines together, in depth mode it moves the depth
         * curve alone. It never reaches across that boundary, which is the
         * point of the split - a weight range is not a multiplier range. That
         * split is also why the GRID follows the axis (scaleGrid): weights are
         * [0, 1], multipliers and shifts keep -1 .. 2 in steps of 0.25.
         * Drawn curves keep their shape and are remapped (that is what the
         * range is for); a curve still sitting on its untouched flat default
         * rides along instead, so "this one does nothing" survives a change of
         * limits.
         */
        attachScaleSelects() {
            this.scaleHiSelect = this.container.querySelector(".cnet-profile-scale-hi");
            this.scaleLoSelect = this.container.querySelector(".cnet-profile-scale-lo");
            if (!this.scaleHiSelect || !this.scaleLoSelect) return;
            this.fillScaleOptions();

            const onChange = (which) => {
                const value = parseFloat(
                    (which === "hi" ? this.scaleHiSelect : this.scaleLoSelect).value);
                if (!isFinite(value)) return;
                // only the axis on screen moves, so only the curves ON it are
                // candidates for re-anchoring. Which ones are untouched has to
                // be read BEFORE it moves - neutrality is an effective value,
                // and the old range is the one they were drawn on
                this.snapshot();
                const range = this.activeRange();
                // the depth and the drift plot each carry ONE curve; the step
                // plot carries the main profile and all three bands
                const ownPlot = this.band === DEPTH_KEY || this.band === DRIFT_KEY;
                const onAxis = this.isBalance
                    ? []
                    : (ownPlot ? [this.band] : BAND_ORDER);
                const untouched = onAxis.filter(
                    (b) => this.store[b]
                        && this.bandNeutral(this.store[b], range, neutralValueOf(b)));
                const lo = which === "lo" ? value : Math.min(range.lo, value);
                const hi = which === "hi" ? value : Math.max(range.hi, value);
                if (hi <= lo) {
                    // a zero-span axis would keep drawing SHAPE that python
                    // maps to the constant lo (every y lands on the same
                    // value) - the plot and the run disagreeing by design.
                    // Refuse, and re-sync the select to the value in force.
                    this.updateScaleSelects();
                    return;
                }
                if (this.band === DEPTH_KEY) {
                    this.depthLo = lo;
                    this.depthHi = hi;
                } else if (this.band === DRIFT_KEY) {
                    this.driftLo = lo;
                    this.driftHi = hi;
                } else {
                    this.scaleLo = lo;
                    this.scaleHi = hi;
                }
                this.reanchorNeutral(untouched, this.activeRange());
                this.updateScaleSelects();
                this.draw();
                this.sync();            };
            this.scaleHiSelect.addEventListener("change", () => onChange("hi"));
            this.scaleLoSelect.addEventListener("change", () => onChange("lo"));
            this.updateScaleSelects();
        }

        /**
         * The stops the two selects offer for the axis CURRENTLY ON SCREEN.
         *
         * A weight and a multiplier are different quantities and get different
         * grids. Main and the three bands are WEIGHTS - a share of the control -
         * and are capped at [0, 1]: nothing above 1 (a unit cannot pull harder
         * than the whole of itself, and stacking that across units is what the
         * coverage panel calls oversaturation) and nothing below 0. Depth and
         * drift keep the -1 .. 2 / 0.25 grid: depth is a per-layer MULTIPLIER
         * whose neutral is 1, so a [0, 1] cap would put "leave these layers
         * alone" at the top of the plot, and drift is a SHIFT whose useful half
         * is negative. Balance is clamped to [0, 1] by balance_factors.
         */
        scaleGrid() {
            if (this.isBalance || !(this.band === DEPTH_KEY || this.band === DRIFT_KEY)) {
                if (this.isBalance) {
                    const out = [];
                    for (let v = 1; v >= -1e-9; v -= SCALE_STEP) {
                        out.push(Math.round(v / SCALE_STEP) * SCALE_STEP);
                    }
                    return out;
                }
                return WEIGHT_SCALE_GRID.slice();
            }
            const out = [];
            for (let v = SCALE_MAX; v >= SCALE_MIN - 1e-9; v -= SCALE_STEP) {
                out.push(Math.round(v / SCALE_STEP) * SCALE_STEP);
            }
            return out;
        }

        /** (Re)fill both selects from the grid of the axis on screen. A no-op
         *  when the grid has not changed, so switching between main and the
         *  bands - which share one axis - does not rebuild anything. */
        fillScaleOptions() {
            if (!this.scaleHiSelect || !this.scaleLoSelect) return;
            const grid = this.scaleGrid();
            const key = grid.join(",");
            if (this.scaleGridKey === key) return;
            this.scaleGridKey = key;
            for (const select of [this.scaleHiSelect, this.scaleLoSelect]) {
                select.innerHTML = "";
                for (const v of grid) {
                    const option = document.createElement("option");
                    option.value = String(v);
                    option.textContent = fmt(v);
                    select.appendChild(option);
                }
            }
        }

        /**
         * Put the flat line of untouched curves (bands, or the depth or drift
         * curve) back where their NEUTRAL value sits on their axis, after its
         * limits moved. One that was left alone must keep doing nothing -
         * unlike a DRAWN curve, which keeps its shape and takes the new
         * weights, the whole point of the range. Points are shared by
         * reference with the working copy, so mutating them in place covers
         * the selected curve too.
         *
         * The neutral is read PER CURVE, not per plot: the drift's is 0 and
         * every multiplier curve's is 1, so a single y computed for the whole
         * list would park a drift wherever the multiplier 1 happens to fall on
         * [-1, 1] - a shift of +1, i.e. the exact opposite of doing nothing.
         */
        reanchorNeutral(names, range) {
            if (!names || !names.length) return;
            for (const b of names) {
                const y = neutralY(range.lo, range.hi, neutralValueOf(b));
                for (const p of this.store[b].points) {
                    p.y = y;
                    if (p.mid) p.mid.y = y;
                }
            }
        }

        /**
         * Response exponent slider: vertical, between the two range selects.
         * Slider value v in [-100, 100] maps to exponent 10^(v/100) - log so
         * both tenfold directions get equal travel: top -100 = x^0.1 (bias to
         * high values), middle 0 = x^1 (linear, the default; marked by the
         * center tick), bottom 100 = x^10 (bias to low values). Bends the
         * NORMALIZED profile only - the range mapping is applied after, so
         * the slider is fully independent of the limits above and below it.
         * Drags redraw only; serialization waits for 'change' (pointer up),
         * like the pad.
         */
        attachGammaSlider() {
            this.gammaSlider = this.container.querySelector(".cnet-profile-gamma");
            if (!this.gammaSlider) return;
            const apply = (serialize) => {
                const v = parseFloat(this.gammaSlider.value);
                if (!isFinite(v)) return;
                this.gamma = Math.pow(10, v / 100);
                if (Math.abs(this.gamma - 1) <= 1e-3) this.gamma = 1; // snap the tick
                this.draw();
                if (serialize) this.sync();
            };
            this.gammaSlider.addEventListener("input", () => apply(false));
            this.gammaSlider.addEventListener("change", () => apply(true));
            // double-click = back to linear (the center tick). The two clicks
            // may first jump the thumb (a track click seeks on range inputs)
            // and serialize that value, but this reset lands last and wins.
            this.gammaSlider.addEventListener("dblclick", () => {
                this.gamma = 1;
                this.updateGammaSlider();
                this.draw();
                this.sync();
            });
            this.updateGammaSlider();
        }

        updateGammaSlider() {
            if (!this.gammaSlider) return;
            this.gammaSlider.value = String(Math.round(100 * Math.log10(this.gamma || 1)));
        }

        /** Insert a (sorted) option for an off-grid value, so e.g. a legacy
         *  '|1.35' scale DISPLAYS as 1.35 instead of a blank select. */
        ensureScaleOption(select, v) {
            const s = String(v);
            const options = Array.from(select.options);
            if (options.some((o) => o.value === s)) return;
            const option = document.createElement("option");
            option.value = s;
            option.textContent = fmt(v);
            // grid is ordered high -> low; keep the custom value in order
            const before = options.find((o) => parseFloat(o.value) < v);
            select.insertBefore(option, before || null);
        }

        /** Show the range of the axis on screen (loadBand calls this, so the
         *  pair follows the selector into and out of depth mode - grid
         *  included, since the weight axis and the multiplier axis do not
         *  offer the same stops). */
        updateScaleSelects() {
            if (!this.scaleHiSelect || !this.scaleLoSelect) return;
            this.fillScaleOptions();
            const range = this.activeRange();
            this.ensureScaleOption(this.scaleHiSelect, range.hi);
            this.ensureScaleOption(this.scaleLoSelect, range.lo);
            this.scaleHiSelect.value = String(range.hi);
            this.scaleLoSelect.value = String(range.lo);
        }

        setActivePreset(name) {
            this.activePreset = name;
            for (const key in this.presetButtons) {
                // these two highlight their STATE (the rung / whether the
                // convergence runs), not which of them the pad is aimed at
                if (key === "osc" || key === "converge") continue;
                this.presetButtons[key].classList.toggle("cnet-profile-preset-active", key === name);
            }
            if (this.pad) {
                this.pad.classList.toggle("cnet-profile-pad-armed", !!name);
                this.pad.title = {
                    step: "Step parameters: x = jump position, y = height AND direction - "
                        + "above the middle the raised part is on the right, below it on the "
                        + "left (the mirrored step); distance from the middle is the height, "
                        + "the middle row itself is flat 0",
                    // Y is the oscillation count on every rung of the ladder;
                    // only X changes hands, and it does so because von Mises
                    // is the one family with a shape parameter of its own.
                    // Phase is the cheaper of the two to give up there: with
                    // the Inputs spread over the full 2*pi it only decides
                    // WHICH of them leads, and it keeps whatever value the
                    // other rungs left on it.
                    osc: this.phaseFamily === "mises"
                        ? "von Mises parameters: x = hand-over sharpness κ (0 = every input "
                          + "equally on ... 10 = near-hard switching), y = oscillations (0 ... 4)"
                        : "Wave parameters: x = phase (0 ... 2π), y = oscillations (0 ... 4)",
                    converge: "Convergence parameters: x = the step by which the waves have "
                        + "reached the flat share of the envelope (and hold it after), "
                        + "y = the dynamics of getting there - the middle row is linear, "
                        + "below it the approach leaves early and arrives patiently, above "
                        + "it the other way round",
                }[name] || "Preset parameters (click a preset first)";
                // the halves only mean something for the step preset, so the
                // centre line is shown only while it holds focus
                this.pad.classList.toggle("cnet-profile-pad-split", name === "step");
            }
            this.updatePad();
        }

        /**
         * Advance the oscillatory button one rung along OSC_LADDER.
         *
         * The wave PARAMETERS survive the whole cycle: phase and oscillation
         * count live in the 'C' token and are never rewritten here, so passing
         * through von Mises (whose pad X is kappa, not phase) and back out
         * returns the exact curve you left. That is also why the pad follows
         * the wave rather than the other way round - see syncPadFromWave.
         *
         * A state that is not on the ladder (a hand-written 'PF' in the
         * balance editor, which has no Inputs to split over) resolves to rung
         * 0, so one click normalizes it to off rather than leaving it stuck.
         */
        cycleOsc() {
            const button = this.presetButtons.osc;
            const ladder = (button && button.dataset.oscMulti)
                ? OSC_LADDER : OSC_LADDER.slice(0, 2);
            const at = ladder.findIndex((rung) => rung.cosOn === !!this.cosOn
                && (rung.family || null) === (this.phaseFamily || null));
            const next = ladder[(at + 1) % ladder.length];
            this.cosOn = next.cosOn;
            this.phaseFamily = next.family;
            // kappa is only ever read in the von Mises rung, so it is seeded
            // on the way in rather than carried around: 0 would land on the
            // pad's far left and show every Input equally on, which reads as
            // "the family does nothing"
            if (next.family === "mises" && !this.kappa) this.kappa = KAPPA_DEFAULT;
            this.setActivePreset(this.cosOn ? "osc" : null);
            this.syncPadFromWave();
            this.drawAndSync();
        }

        /**
         * Turn the convergence on or off.
         *
         * On, the n waves stop being a fixed partition and slide onto the flat
         * share 1/n of the envelope - reaching it at the pad's x and holding it
         * for the rest of the range. The n shares still sum to 1 at every step
         * (convergedWeight is a convex combination of two things that do), so
         * the unit's total pull is the drawn envelope throughout, converged or
         * not. With one Input the share IS the envelope, so the single wave
         * simply fades into it.
         *
         * It is a parameter OF THE WAVE, like the phase and the oscillation
         * count: it survives the oscillatory ladder and is not applied (nor
         * serialized) while the wave is off. Switching it on with no wave
         * therefore turns the wave on too, since "converging nothing" is not a
         * state the user can see - the ladder stays in charge of WHICH wave.
         */
        toggleConverge() {
            if (this.converge) {
                this.converge = null;
                this.setActivePreset(null);
            } else {
                this.cosOn = true;
                this.converge = { at: CONVERGE_DEFAULT.at, exp: CONVERGE_DEFAULT.exp };
                this.syncPadFromWave();
                this.setActivePreset("converge");
            }
            this.drawAndSync();
        }

        /**
         * Point the pad at the wave's current parameters.
         *
         * The wave is the source of truth, not the pad. The pad's X changes
         * MEANING between rungs - phase everywhere except von Mises, where it
         * is kappa - so a position carried across a rung change would be read
         * back as the wrong quantity and silently move the curve. The
         * convergence pad is refreshed here too: it is a wave parameter and has
         * to follow the same source, or a band switch would leave it showing
         * the previous curve's numbers.
         */
        syncPadFromWave() {
            if (!this.padPos) return;
            this.padPos.osc.x = this.phaseFamily === "mises"
                ? clamp01((this.kappa || 0) / KAPPA_MAX)
                : clamp01(this.cosPhase / TAU);
            this.padPos.osc.y = clamp01(this.cosN / COS_MAX_OSC);
            const converge = this.converge || CONVERGE_DEFAULT;
            this.padPos.converge.x = clamp01(converge.at);
            // exponent = CONVERGE_EXP_MAX^(2y - 1), inverted: log-mapped like
            // the response slider, so 1 (linear) is exactly the middle row
            this.padPos.converge.y = clamp01(
                0.5 + Math.log(converge.exp) / (2 * Math.log(CONVERGE_EXP_MAX)));
        }

        /** Move the visible pad dot to the focused preset's parameters. */
        updatePad() {
            if (!this.pad || !this.padDot) return;
            const pos = this.padPos[this.activePreset];
            if (!pos) {
                this.padDot.style.display = "none";
                return;
            }
            this.padDot.style.display = "block";
            this.padDot.style.left = (pos.x * 100) + "%";
            this.padDot.style.top = ((1 - pos.y) * 100) + "%";
        }

        /** Visual refresh only - no serialization (mid-drag path). */
        drawOnly() {
            this.setOscButtonState();
            this.setConvergeButtonState();
            this.updatePad();
            this.draw();
        }

        drawAndSync() {
            this.drawOnly();
            this.sync();        }

        /** Put the oscillatory button on the rung the state actually is on:
         *  data-osc-state picks the icon (style.css), the active class lights
         *  it for every rung that has a wave. */
        setOscButtonState() {
            const button = this.presetButtons.osc;
            if (!button) return;
            const rung = OSC_LADDER.find((r) => r.cosOn === !!this.cosOn
                && (r.family || null) === (this.phaseFamily || null));
            button.dataset.oscState = rung ? rung.state : "off";
            button.classList.toggle("cnet-profile-preset-active", !!this.cosOn);
        }

        /** Light the convergence button when a convergence is actually RUNNING.
         *  Stored-but-idle is a real state - it survives the ladder's off rung
         *  like every other wave parameter - and a lit button there would
         *  promise something the flat curve on screen is not doing. */
        setConvergeButtonState() {
            const button = this.presetButtons.converge;
            if (!button) return;
            button.classList.toggle("cnet-profile-preset-active",
                                    !!(this.converge && this.cosOn));
        }

        /** Waves the multi-phase preview draws: one per Input tab holding an
         *  image, at least one. A unit with a single Input (or none loaded
         *  yet) draws exactly ONE wave, because that is what it will run - the
         *  family's kernel, no split. Muted
         *  inputs (tab-title checkbox off) are skipped: the generation
         *  fan-out counts len(get_input_data(...)), which drops them, and the
         *  preview must count the same thing that runs. The watch tick
         *  re-checks this count and redraws when it moves, so the preview
         *  follows uploads / clears / mutes without any edit to the curve. */
        multiPhaseCount() {
            // the rule itself lives in active_canvas.js (cnetLiveInputs), so
            // the preview, the coverage panel and anything else that has to
            // know "which Inputs run" cannot answer it differently
            const live = typeof window !== "undefined" && window.cnetLiveInputs
                ? window.cnetLiveInputs(this.unitRoot) : [];
            return Math.max(live.length, 1);
        }

        /**
         * Rebuild the focused preset's profile from the pad point.
         *
         * step: pad x is the jump position, pad y the step height. Full left
         *   means the jump already happened at 0, i.e. a constant at y; full
         *   right means the jump never happens, i.e. a flat 0 (height is then
         *   irrelevant). Both degenerate cases are two points, not four.
         * osc: pad y is the number of oscillations over [0, 4] on every rung;
         *   pad x is the phase over [0, 2π], except in the von Mises rung
         *   where it is the hand-over sharpness κ over [0, 10] (that family is
         *   the only one with a shape parameter of its own, and phase is the
         *   cheaper of the two to give up - see setActivePreset). The envelope
         *   (the drawn polyline) is NOT regenerated - the wave just re-rides it.
         */
        applyPadPreset(defer) {
            const pos = this.padPos[this.activePreset];
            if (!pos) return;

            const finish = () => (defer ? this.drawOnly() : this.drawAndSync());

            if (this.activePreset === "converge") {
                // x = where the waves have arrived at the flat share, y = the
                // dynamics of the approach, log-mapped so both directions get
                // equal travel and the middle row is exactly linear
                const exp = Math.pow(CONVERGE_EXP_MAX, 2 * clamp01(pos.y) - 1);
                this.cosOn = true;
                this.converge = {
                    at: clamp01(pos.x),
                    exp: Math.abs(exp - 1) <= 1e-3 ? 1 : exp,   // snap the middle
                };
                finish();
                return;
            }

            if (this.activePreset === "osc") {
                this.cosOn = true;
                if (this.phaseFamily === "mises") {
                    this.kappa = clamp01(pos.x) * KAPPA_MAX;
                } else {
                    this.cosPhase = clamp01(pos.x) * TAU;
                }
                this.cosN = clamp01(pos.y) * COS_MAX_OSC;
                finish();
                return;
            }

            // Step, generalized over the pad's vertical CENTRE: distance from
            // the centre is the step height, the SIDE is its direction.
            //   above centre -> the raised part is on the RIGHT (0 then h)
            //   below centre -> the raised part is on the LEFT  (h then 0)
            // At full height the two are exact vertical mirrors of each other,
            // which is what the invert toggle used to be for - the pad now
            // reaches both without a separate mode to remember. The centre row
            // is height 0 (flat) from either side, so the pad is continuous
            // across it.
            const at = clamp01(pos.x);
            const raisedRight = pos.y >= 0.5;
            const height = clamp01(Math.abs(pos.y - 0.5) * 2);
            if (raisedRight) {
                // full left = the jump already happened at 0 (constant h),
                // full right = it never happens (flat 0)
                if (at === 0) this.points = [{ x: 0, y: height }, { x: 1, y: height }];
                else if (at === 1) this.points = [{ x: 0, y: 0 }, { x: 1, y: 0 }];
                else {
                    this.points = [
                        { x: 0, y: 0 }, { x: at, y: 0 },
                        { x: at, y: height }, { x: 1, y: height },
                    ];
                }
            } else {
                // mirrored: full left = the drop already happened (flat 0),
                // full right = it never happens (constant h)
                if (at === 0) this.points = [{ x: 0, y: 0 }, { x: 1, y: 0 }];
                else if (at === 1) this.points = [{ x: 0, y: height }, { x: 1, y: height }];
                else {
                    this.points = [
                        { x: 0, y: height }, { x: at, y: height },
                        { x: at, y: 0 }, { x: 1, y: 0 },
                    ];
                }
            }
            finish();
        }

        /** Canvas CSS size, cached until the ResizeObserver invalidates it:
         *  toScreen runs per curve sample per frame (hundreds of calls), and
         *  an uncached getBoundingClientRect there was the dominant per-frame
         *  cost. Only the SIZE is cached - it changes exclusively through
         *  resize() - while mousePos keeps reading the live rect (left/top
         *  shift on scroll without any resize). */
        cssSize() {
            if (!this._cssSize) {
                const rect = this.canvas.getBoundingClientRect();
                this._cssSize = { w: rect.width, h: rect.height };
            }
            return this._cssSize;
        }

        plotRect() {
            if (!this._plotRect) {
                const { w, h } = this.cssSize();
                this._plotRect = {
                    left: MARGIN.left,
                    top: MARGIN.top,
                    w: Math.max(w - MARGIN.left - MARGIN.right, 1),
                    h: Math.max(h - MARGIN.top - MARGIN.bottom, 1),
                };
            }
            return this._plotRect;
        }

        mousePos(e) {
            const rect = this.canvas.getBoundingClientRect();
            return { mx: e.clientX - rect.left, my: e.clientY - rect.top };
        }

        /**
         * Convert mouse position to profile coordinates. Clamping implements
         * the margin behavior: clicks in a margin pin the coordinate to 0 or 1.
         */
        toPlot(pos) {
            const r = this.plotRect();
            return {
                x: clamp01((pos.mx - r.left) / r.w),
                y: clamp01(1 - (pos.my - r.top) / r.h),
            };
        }

        toScreen(p) {
            const r = this.plotRect();
            return { sx: r.left + p.x * r.w, sy: r.top + (1 - p.y) * r.h };
        }

        /**
         * Everything the pointer may grab: in main mode just the main
         * profile; in band mode the points of ALL THREE band lines are
         * directly editable - the band buttons only pick which profile the
         * presets, pad and invert act on. The selected band is iterated first
         * and wins distance ties. Depth and drift mode each show one curve and
         * only that curve is grabbable (the bands are not on those plots - they
         * are the alternative to them, not companions of them).
         */
        editableBands() {
            if (this.mainModeSelector(this.band)) {
                return [{ band: this.band, pts: this.points }];
            }
            const out = [{ band: this.band, pts: this.points }];
            for (const b of BAND_ORDER) {
                if (b !== this.band) out.push({ band: b, pts: this.store[b].points });
            }
            return out;
        }

        findPoint(pos) {
            let best = null;
            let bestDist = GRAB_RADIUS + 1e-9;
            for (const { band, pts } of this.editableBands()) {
                for (const p of pts) {
                    const { sx, sy } = this.toScreen(p);
                    const dist = Math.hypot(pos.mx - sx, pos.my - sy);
                    if (dist < bestDist) {
                        bestDist = dist;
                        best = { point: p, pts: pts, band: band };
                    }
                }
            }
            return best;
        }

        /**
         * Mid control handle of the segment starting at points[i]: the active
         * control position when set, otherwise the passive linear midpoint.
         * Null for degenerate (vertical) segments.
         */
        midHandle(i) {
            const p0 = this.points[i];
            const p1 = this.points[i + 1];
            if (!p1) return null;
            const dx = p1.x - p0.x;
            const active = !!p0.mid;
            // tiny segments get no PASSIVE handle (it would sit on the
            // vertices and steal their clicks) - but an ACTIVE mid control
            // must stay visible and grabbable no matter how far the vertices
            // were dragged together, or it becomes an invisible, unremovable
            // bend that still evaluates and serializes
            if (dx <= 0 || (!active && dx <= MID_MIN_DX)) return null;
            return {
                p0: p0,
                p1: p1,
                active: active,
                x: active ? p0.x + p0.mid.t * dx : p0.x + dx / 2,
                y: active ? p0.mid.y : (p0.y + p1.y) / 2,
            };
        }

        findMid(pos) {
            let best = null;
            let bestDist = MID_GRAB_RADIUS;
            for (let i = 0; i + 1 < this.points.length; i++) {
                const m = this.midHandle(i);
                if (!m) continue;
                const { sx, sy } = this.toScreen(m);
                const dist = Math.hypot(pos.mx - sx, pos.my - sy);
                if (dist <= bestDist) {
                    bestDist = dist;
                    best = m;
                }
            }
            return best;
        }

        /**
         * A click on a margin edits the profile extreme instead of stacking a
         * new point on the border: left/right margins pick the point sitting
         * on that border (x = 0 / x = 1) and set its Y to the clicked level;
         * top/bottom margins pick a border point (y = 1 / y = 0) near the
         * clicked X and slide it there. Returns null if there is no point to
         * pick up, in which case a new one is created by the caller.
         */
        findMarginPoint(pos) {
            const r = this.plotRect();
            const target = this.toPlot(pos);
            if (pos.mx < r.left || pos.mx > r.left + r.w) {
                const xEdge = pos.mx < r.left ? 0 : 1;
                const point = this.points.find((p) => p.x === xEdge);
                if (point) point.y = target.y;
                return point || null;
            }
            if (pos.my < r.top || pos.my > r.top + r.h) {
                const yEdge = pos.my < r.top ? 1 : 0;
                const point = this.points.find(
                    (p) => p.y === yEdge && Math.abs(this.toScreen(p).sx - pos.mx) <= GRAB_RADIUS
                );
                if (point) point.x = target.x;
                return point || null;
            }
            return null;
        }

        sortPoints() {
            this.points.sort((a, b) => a.x - b.x);
        }

        attachEvents() {
            this.canvas.addEventListener("pointerdown", (e) => {
                if (e.button !== 0) return;
                // one drag at a time: a second touch must not hijack the
                // point the first finger is moving
                if (this.dragPoint || this.dragMid) return;
                const pos = this.mousePos(e);
                const hit = this.findPoint(pos);
                let point = hit && hit.point;
                // the owning point list: a grabbed point edits ITS band; new
                // points, margin edits and mid controls go to the selected one
                this.dragPts = hit ? hit.pts : this.points;
                this.dragColor = hit && hit.band !== "main"
                    ? this.curveColorOf(hit.band)
                    : null;
                if (!point) {
                    // grabbing a (passive or active) segment midpoint starts a
                    // mid-control drag: a passive one becomes active on pickup
                    const m = this.findMid(pos);
                    if (m) {
                        if (!m.active) m.p0.mid = { t: 0.5, y: (m.p0.y + m.p1.y) / 2 };
                        this.dragMid = m.p0;
                        this.dragPointerId = e.pointerId;
                        this.canvas.setPointerCapture(e.pointerId);
                        this.draw();
                        e.preventDefault();
                        return;
                    }
                    point = this.findMarginPoint(pos);
                }
                if (!point) {
                    point = this.toPlot(pos);
                    this.points.push(point);
                    this.dragPts = this.points;
                }
                this.dragPts.sort((a, b) => a.x - b.x);
                this.dragPoint = point;
                this.selPoint = point;  // keyboard nudges target the last touched point
                this.dragPointerId = e.pointerId;
                this.canvas.setPointerCapture(e.pointerId);
                this.draw();
                e.preventDefault();
            });

            this.canvas.addEventListener("pointermove", (e) => {
                if ((this.dragPoint || this.dragMid) && e.pointerId !== this.dragPointerId) return;
                const pos = this.mousePos(e);
                if (this.dragMid) {
                    const i = this.points.indexOf(this.dragMid);
                    const p1 = this.points[i + 1];
                    if (i >= 0 && p1 && p1.x - this.dragMid.x > 0) {
                        const target = this.toPlot(pos);
                        const t = (target.x - this.dragMid.x) / (p1.x - this.dragMid.x);
                        this.dragMid.mid = {
                            t: Math.min(Math.max(t, MID_MIN_T), MID_MAX_T),
                            y: target.y,
                        };
                        this.draw();
                    }
                    return;
                }
                if (this.dragPoint) {
                    const target = this.toPlot(pos);
                    this.dragPoint.x = target.x;
                    this.dragPoint.y = target.y;
                    (this.dragPts || this.points).sort((a, b) => a.x - b.x);
                    this.draw();
                } else {
                    // (the scale-gutter hover test lived here until the gutter
                    // slider was replaced by the range selects; the call
                    // outlived the method and threw on every hover, killing
                    // this cursor logic below it)
                    this.canvas.style.cursor = this.findPoint(pos) || this.findMid(pos) ? "grab" : "crosshair";
                }
            });

            const endDrag = (e) => {
                if (!this.dragPoint && !this.dragMid) return;
                if (e && e.pointerId !== this.dragPointerId) return;
                this.dragPoint = null;
                this.dragMid = null;
                this.dragPts = null;
                this.dragColor = null;
                this.dragPointerId = null;
                this.draw();
                this.sync();            };
            this.canvas.addEventListener("pointerup", endDrag);
            this.canvas.addEventListener("pointercancel", endDrag);

            this.canvas.addEventListener("dblclick", (e) => {
                const pos = this.mousePos(e);
                const hit = this.findPoint(pos);
                if (hit && hit.pts.length > 1) {
                    hit.pts.splice(hit.pts.indexOf(hit.point), 1);
                    if (this.selPoint === hit.point) this.selPoint = null;
                    this.draw();
                    this.sync();                } else if (!hit) {
                    // double-click on an active mid control: back to the
                    // passive linear midpoint (segment becomes linear again)
                    const m = this.findMid(pos);
                    if (m && m.active) {
                        delete m.p0.mid;
                        this.draw();
                        this.sync();                    }
                }
                e.preventDefault();
            });
        }

        /**
         * Minimal keyboard access (the editor was pointer-only): the canvas is
         * focusable, arrow keys nudge the last-touched point (Shift = finer),
         * Delete/Backspace removes it. Everything syncs on release like a
         * drag would.
         */
        attachKeyboard() {
            this.canvas.addEventListener("keydown", (e) => {
                const p = this.selPoint;
                if (!p) return;
                // the point may belong to any editable line (band mode)
                const owner = this.editableBands()
                    .map((entry) => entry.pts)
                    .find((pts) => pts.includes(p));
                if (!owner) return;
                const step = e.shiftKey ? 0.002 : 0.02;
                let handled = true;
                switch (e.key) {
                    case "ArrowLeft": p.x = clamp01(p.x - step); break;
                    case "ArrowRight": p.x = clamp01(p.x + step); break;
                    case "ArrowUp": p.y = clamp01(p.y + step); break;
                    case "ArrowDown": p.y = clamp01(p.y - step); break;
                    case "Delete":
                    case "Backspace":
                        if (owner.length > 1) {
                            owner.splice(owner.indexOf(p), 1);
                            this.selPoint = null;
                        }
                        break;
                    default: handled = false;
                }
                if (!handled) return;
                e.preventDefault();
                owner.sort((a, b) => a.x - b.x);
                this.draw();
                this.sync();            });
        }

        resize() {
            this._cssSize = null;
            this._plotRect = null;
            const { w, h } = this.cssSize();
            if (w <= 0 || h <= 0) return;
            const dpr = window.devicePixelRatio || 1;
            this._lastDpr = dpr;
            this.canvas.width = Math.round(w * dpr);
            this.canvas.height = Math.round(h * dpr);
            this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            // the parameter pad is a square whose side matches the PLOT height
            // (canvas height minus the axis margins), so it lines up with the
            // drawing area rather than with the whole widget
            if (this.pad) {
                // Square, ideally as wide as the plot is high - but never so
                // wide that the plot or the range column gets squeezed out of
                // the parent. The body width is the budget: presets column,
                // range column and gaps are reserved first, and the pad may
                // take at most PAD_MAX_SHARE of what is left.
                const body = this.pad.parentElement;
                // clear the previous px width first, so a stale (wider) value
                // cannot inflate the measurement and latch the row open when
                // the panel is dragged narrower
                this.pad.style.width = "0px";
                const bodyW = body ? body.clientWidth : 0;
                const reserved =
                    (this.container.querySelector(".cnet-weight-profile-presets")?.offsetWidth || 0) +
                    (this.container.querySelector(".cnet-profile-scale")?.offsetWidth || 0) + 24;
                const budget = Math.max(bodyW - reserved, 0);
                const side = Math.max(Math.min(h - MARGIN.top - MARGIN.bottom,
                                               budget * PAD_MAX_SHARE), PAD_MIN_SIDE);
                this.pad.style.width = side + "px";
                this.pad.style.height = side + "px";
            }
            this.draw();
        }

        /** Sampling step count from the owning tab's main Steps slider
         *  (0 = not resolvable; hires/other passes are out of scope). */
        stepsCount() {
            let input = this._stepsInput;
            if (!input || !input.isConnected) {
                const tabRoot = this.container.closest("#tab_txt2img, #tab_img2img");
                if (!tabRoot) return 0;
                input = gradioApp().querySelector(`#${tabRoot.id.slice(4)}_steps input[type='number']`);
                this._stepsInput = input;
            }
            const n = input ? Math.round(parseFloat(input.value)) : 0;
            return Number.isFinite(n) && n > 1 ? n : 0;
        }

        /** The step separators must track the Steps slider live, so both of
         *  its inputs (number box + range) redraw this plot. Server-side
         *  writes to the slider fire no DOM events; those catch up on the
         *  next redraw of any kind, which is acceptable staleness. */
        attachStepsWatch() {
            const tabRoot = this.container.closest("#tab_txt2img, #tab_img2img");
            if (!tabRoot) return;
            const redraw = () => this.draw();
            for (const input of gradioApp().querySelectorAll(`#${tabRoot.id.slice(4)}_steps input`)) {
                input.addEventListener("input", redraw);
            }
        }

        colors() {
            const styles = getComputedStyle(this.canvas);
            const accent = styles.getPropertyValue("--primary-400").trim() || "#4b7ecb";
            const background = getComputedStyle(document.body).backgroundColor || "#000";
            // Step separators. ONE tone, deliberately, and the dotted duty
            // cycle already mutes them - the dot alpha sits above the solid
            // quarter grid's to end up quieter overall, not invisible.
            //
            // This used to be a ternary: white on dark themes, this gray on
            // light, chosen by measuring `document.body`'s background right
            // here. The probe never worked. Gradio leaves body WHITE on both
            // themes and paints the real fill onto the app element inside it,
            // so the measurement read 255 and the light arm won always -
            // verified on the running app in dark mode: 926 px of this tone on
            // the plot, none of the white one. The gray is therefore what the
            // dark theme has always looked like, and it carries on the light
            // one too (0.38 alpha over white is a legible 194 gray), so there
            // is nothing left for a branch to decide. Do not reinstate one
            // without measuring the plot in both themes first.
            return {
                accent: accent,
                background: background,
                text: "rgba(127, 127, 127, 0.9)",
                grid: "rgba(127, 127, 127, 0.16)",
                stepLine: "rgba(96, 96, 96, 0.38)",
                // Value dots take ONE muted tone instead of their curve's
                // color: a same-colored dot on the main line would show only as
                // its halo ring, while one muted tone reads across the plot's
                // whole contrast range - clearly under a full-strength curve,
                // still over the 0.4-alpha multi-phase underlays it also has to
                // mark. Neutral on purpose, so it never competes with the band
                // colors for meaning. On the 2px main line the SEPARATION is
                // the mark, which is the one thing that has to survive a theme:
                // the value is a CSS variable (style.css) so it can be flipped
                // to the far side of the line when the line itself is, and the
                // two are decided together instead of in two files. The
                // fallback below is the dark theme's, matching that variable -
                // MAINTENANCE.md's tuning story describes the OTHER tone this
                // used to select on paper, which the broken probe next to
                // stepLine meant it never actually painted.
                stepDot: styles.getPropertyValue("--cnet-plot-step-dot").trim()
                    || "rgb(56, 62, 80)",
                border: "rgba(127, 127, 127, 0.4)",
                marginFill: "rgba(127, 127, 127, 0.07)",
            };
        }

        /** Effective (normalized) value of a stored profile object at x - the
         *  standalone twin of evaluate(), for curves that are NOT the working
         *  copy (band overlays, the depth line). Same wave math through
         *  waveFactorOf, so the two paths cannot disagree about the same
         *  profile object; the slot NAME is what decides whether it fans out. */
        profileValueAt(P, name, x) {
            let v = this.envelopeOf(P.points, x) * this.waveFactorOf(P, name, x, 0);
            const g = P.gamma || 1;
            if (Math.abs(g - 1) > 1e-4) v = Math.pow(Math.min(Math.max(v, 0), 1), g);
            return v;
        }

        /** Thin overlay line of a non-selected band profile. Its points are
         *  directly grabbable (band selection only routes the presets), so
         *  they are drawn as small hollow markers. */
        drawBandOverlay(P, name, color) {
            const ctx = this.ctx;
            const n = P.cosOn ? COS_CURVE_SAMPLES : BAND_CURVE_SAMPLES;
            ctx.save();
            ctx.globalAlpha = 0.6;
            ctx.beginPath();
            for (let i = 0; i <= n; i++) {
                const x = i / n;
                const s = this.toScreen({ x: x, y: this.profileValueAt(P, name, x) });
                if (i === 0) ctx.moveTo(s.sx, s.sy);
                else ctx.lineTo(s.sx, s.sy);
            }
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.globalAlpha = 0.9;
            const bg = this.colors().background;
            for (const p of P.points) {
                const { sx, sy } = this.toScreen(p);
                const held = p === this.dragPoint;
                ctx.beginPath();
                ctx.arc(sx, sy, held ? MID_POINT_RADIUS + 1 : MID_POINT_RADIUS, 0, Math.PI * 2);
                ctx.fillStyle = held ? color : bg;
                ctx.fill();
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
            ctx.restore();
        }

        /** The current effective profile object of band b (working copy when
         *  selected, store otherwise). No range: that is the plot's, shared. */
        storedProfileOf(b) {
            if (b !== this.band) return this.store[b];
            return {
                points: this.points,
                cosOn: this.cosOn, cosN: this.cosN, cosPhase: this.cosPhase,
                phaseFamily: this.phaseFamily || null, kappa: this.kappa || 0,
                converge: this.converge || null,
                gamma: this.gamma || 1,
            };
        }

        /** The current shape of band b: the working copy when selected. */
        bandProfileFor(b) {
            return this.storedProfileOf(b);
        }

        /**
         * One small filled dot per sampling step, sitting ON a curve at the x
         * that step reads its value from - so the plot states the values that
         * run instead of leaving them to be eyeballed between separators.
         *
         * X follows the SAME uniform-cell model as the separators (N cells for
         * N steps, `steps` from the tab's Steps slider): step k takes its value
         * at the START of its cell, k/N. Every dot therefore lands on a
         * separator - the k=0 one on the left border - so the two cues
         * reinforce each other instead of forming two competing x grids, which
         * is why the lines only needed a dash more brightness. Where the
         * scheduler really places each step inside the timestep range stays out
         * of scope here exactly as it is for the separators.
         *
         * Y always comes from the very function that drew the line
         * (`valueAt`), never from a second approximation, so a dot cannot drift
         * off its curve. The one exception is sub-pixel: a parabola segment's
         * LINE is drawn as MID_CURVE_STEPS chords while the value is the exact
         * parabola (< ~1e-3 normalized, the same tolerance the python side is
         * documented to keep).
         *
         * Each dot is laid on a translucent background halo first, a softened
         * version of what the point markers do: a dot flush against its own
         * line is only a bulge, and where the multi-phase waves cross it would
         * belong to either series. A part-opaque halo mutes the line under the
         * dot just enough to separate the two, where the opaque one first
         * tried here cut the curves into beaded strings and turned a 4-wave
         * plot into visual noise (it also swallowed the green mid handle).
         * Same reason the dots are small and take one muted tone (`stepDot`)
         * rather than their curve's color: curves first, samples on a closer
         * look. The halo radius is what sets STEP_DOT_MIN_SPACING - touching
         * halos would chew the curve up.
         *
         * Cost: two paths and two fills for the whole curve, N arcs each - an
         * order of magnitude under the COS_CURVE_SAMPLES-point polylines this
         * same frame already strokes. The x/y of every dot is computed once and
         * reused by both passes, and the caller drops the whole thing once the
         * dots would collide.
         */
        drawStepDots(steps, valueAt, color, alpha, haloColor, radius) {
            const ctx = this.ctx;
            const r = this.plotRect();
            ctx.save();
            // flat [sx, sy, ...] scratch: no per-dot object for the second pass
            const xy = this._dotScratch && this._dotScratch.length >= steps * 2
                ? this._dotScratch
                : (this._dotScratch = new Float64Array(Math.max(steps, 64) * 2));
            for (let k = 0; k < steps; k++) {
                const x = k / steps;
                xy[k * 2] = r.left + x * r.w;
                xy[k * 2 + 1] = r.top + (1 - valueAt(x)) * r.h;
            }
            // haloColor null = no halo pass: a bead in its line's own color
            // needs no separation from it, and a ring would be the very notch
            // this variant exists to avoid
            const passes = [{ radius: radius, fill: color, alpha: alpha }];
            if (haloColor) {
                passes.unshift({ radius: STEP_DOT_HALO_RADIUS, fill: haloColor,
                                 alpha: alpha * STEP_DOT_HALO_ALPHA });
            }
            for (const pass of passes) {
                ctx.globalAlpha = pass.alpha;
                ctx.fillStyle = pass.fill;
                ctx.beginPath();
                for (let k = 0; k < steps; k++) {
                    const sx = xy[k * 2];
                    const sy = xy[k * 2 + 1];
                    // moveTo before each arc: without it canvas connects the
                    // previous dot to this one with a chord of the shared path
                    ctx.moveTo(sx + pass.radius, sy);
                    ctx.arc(sx, sy, pass.radius, 0, TAU);
                }
                ctx.fill();
            }
            ctx.restore();
        }

        /** Dense screen-space samples of the modulated wave (cosine mode). */
        cosineCurve() {
            const out = [];
            for (let i = 0; i <= COS_CURVE_SAMPLES; i++) {
                const x = i / COS_CURVE_SAMPLES;
                out.push(this.toScreen({ x: x, y: this.evaluate(x) }));
            }
            return out;
        }

        draw() {
            const { w, h } = this.cssSize();
            if (w <= 0 || h <= 0) return;
            const ctx = this.ctx;
            const r = this.plotRect();

            // Input count FIRST: under a multi-phase family the main curve is
            // itself input 1's share of the wave, so waveFactor needs the
            // count before anything samples evaluate(). Counting walks the
            // unit's DOM, hence once per frame and cached on the instance
            // (waveFactor runs per curve sample - hundreds of times).
            this._phaseCount = (this.cosOn && this.phaseFamily)
                ? this.multiPhaseCount() : 1;
            this._lastPhaseCount = this._phaseCount;
            const c = this.colors();

            ctx.clearRect(0, 0, w, h);

            // Margins backdrop; the plot area is kept transparent.
            ctx.fillStyle = c.marginFill;
            ctx.fillRect(0, 0, w, h);
            ctx.clearRect(r.left, r.top, r.w, r.h);

            // Grid.
            ctx.strokeStyle = c.grid;
            ctx.lineWidth = 1;
            for (const q of [0.25, 0.5, 0.75]) {
                ctx.beginPath();
                ctx.moveTo(r.left + q * r.w, r.top);
                ctx.lineTo(r.left + q * r.w, r.top + r.h);
                ctx.moveTo(r.left, r.top + q * r.h);
                ctx.lineTo(r.left + r.w, r.top + q * r.h);
                ctx.stroke();
            }
            // Step separators: one dotted vertical per boundary between
            // adjacent sampling steps (N cells for N steps), tracking the
            // tab's main Steps slider. X is the relative step range, so the
            // cells are uniform - where each step actually lands within the
            // timestep range is the scheduler's business, not the editor's.
            // Depth mode is excluded: there X is the injection layer, not the
            // step, so both the separators and the per-step dots below would
            // be marking something that axis does not measure. DRIFT mode is
            // NOT excluded - its x IS the step axis (only its y is a depth
            // shift), so the separators mark exactly what they always do.
            const steps = this.band === DEPTH_KEY ? 0 : this.stepsCount();
            if (steps > 1 && r.w / steps >= STEP_LINE_MIN_SPACING) {
                ctx.save();
                ctx.strokeStyle = c.stepLine;
                ctx.setLineDash([1, 3]);
                ctx.beginPath();
                for (let k = 1; k < steps; k++) {
                    const sx = r.left + (k / steps) * r.w;
                    ctx.moveTo(sx, r.top);
                    ctx.lineTo(sx, r.top + r.h);
                }
                ctx.stroke();
                ctx.restore();
            }

            ctx.strokeStyle = c.border;
            ctx.strokeRect(r.left, r.top, r.w, r.h);

            // Single X axis label. Tick values were redundant - X is always
            // the fixed relative 0..1 step range - and the Y limits are
            // stated by the range selects (a 0..1 scale there would be
            // ambiguous: the drawn 0..1 is normalized, not the final weight).
            // Drawn in the same bottom-margin band the ticks used, so the
            // label costs no extra space.
            ctx.fillStyle = c.text;
            ctx.font = "12px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            // depth mode retitles the axis: same plot, different X meaning
            // (0 = shallowest injection layer, 1 = deepest), and the arrow
            // says which way it runs. Drift mode keeps the STEP axis - the
            // depth direction is what its Y says, so the label names that
            // instead (up = toward coarse, matching the depth plot's own
            // fine -> coarse direction).
            ctx.fillText(
                this.band === DEPTH_KEY ? "Depth   fine → coarse"
                    : (this.band === DRIFT_KEY ? "Steps   ↑ coarse  ↓ fine" : "Steps"),
                r.left + 0.5 * r.w, r.top + r.h + 7);

            // the selected profile draws full-strength below in its color:
            // white for main, purple for depth, green for drift, the band color
            // otherwise. In band mode the two NON-selected band lines go first,
            // as thin underlays; depth and drift mode each show their curve
            // alone (the bands are their alternative, not their companions).
            const curveColor = this.curveColorOf(this.band);
            if (!this.mainModeSelector(this.band)) {
                for (const b of BAND_ORDER) {
                    if (b !== this.band) this.drawBandOverlay(this.store[b], b, BAND_COLORS[b]);
                }
            }
            // A drawn depth curve is NOT echoed onto the main plot (2026-07-25,
            // user decision). Two dashed guides at main(step) x min/max(depth)
            // used to mark the span the layers cover; they carried only the two
            // extremes of the depth curve, said nothing about which layer sits
            // where, and had to be clamped to the plot's Y limits - drawing a
            // product that leaves the axis as a flat line along the border. The
            // depth plot states depth; the step plot states steps.

            // Profile curve, extended horizontally to both borders; segments
            // with an active mid control are sampled as parabola arcs.
            const extended = [{ x: 0, y: this.points[0].y }];
            for (let i = 0; i < this.points.length; i++) {
                extended.push(this.points[i]);
                const p1 = this.points[i + 1];
                if (p1 && this.points[i].mid && p1.x - this.points[i].x > 0) {
                    for (let k = 1; k < MID_CURVE_STEPS; k++) {
                        const x = this.points[i].x + ((p1.x - this.points[i].x) * k) / MID_CURVE_STEPS;
                        extended.push({ x: x, y: this.envelopeAt(x) });
                    }
                }
            }
            extended.push({ x: 1, y: this.points[this.points.length - 1].y });
            const screen = extended.map((p) => this.toScreen(p));

            // In cosine mode the drawn curve is the wave, the envelope only a
            // dashed guide. Same split with a non-linear response exponent:
            // the solid line is the EFFECTIVE (bent) curve, the editable raw
            // polyline stays as the dashed guide - the power of a piecewise-
            // linear function is curved, so the effective line needs the
            // dense sampling either way. No area fill below the line - the
            // plot stays clean with several lines on it.
            const gammaOn = Math.abs((this.gamma || 1) - 1) > 1e-4;
            const waveScreen = (this.cosOn || gammaOn) ? this.cosineCurve() : screen;

            if (this.cosOn || gammaOn) {
                // envelope as a dashed guide: it stays the editable polyline
                ctx.save();
                ctx.setLineDash([4, 3]);
                ctx.globalAlpha = 0.55;
                ctx.beginPath();
                for (let i = 0; i < screen.length; i++) {
                    if (i === 0) ctx.moveTo(screen[i].sx, screen[i].sy);
                    else ctx.lineTo(screen[i].sx, screen[i].sy);
                }
                ctx.strokeStyle = curveColor;
                ctx.lineWidth = 1.5;
                ctx.stroke();
                ctx.restore();
            }

            // multi-phase: the sibling waves each further Input runs, phase
            // shifted by 2pi/n, as thin underlays below the main wave (the
            // main wave IS input 1's profile - and under a partition family it
            // is input 1's SHARE of the wave, which is why the count above had
            // to be resolved before waveScreen was sampled). With ONE Input
            // there is no sibling and none is drawn: the loops below are empty
            // at a count of 1, so the degenerate case needs no special case -
            // the single line already shows the kernel that runs.
            const phaseCount = this.waveCountOf(this, this.band);
            if (phaseCount > 1) {
                ctx.save();
                ctx.globalAlpha = 0.4;
                for (let k = 1; k < phaseCount; k++) {
                    ctx.beginPath();
                    for (let i = 0; i <= COS_CURVE_SAMPLES; i++) {
                        const x = i / COS_CURVE_SAMPLES;
                        const s = this.toScreen({ x: x, y: this.phaseValueAt(x, k) });
                        if (i === 0) ctx.moveTo(s.sx, s.sy);
                        else ctx.lineTo(s.sx, s.sy);
                    }
                    ctx.strokeStyle = curveColor;
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
                ctx.restore();
            }

            ctx.beginPath();
            for (let i = 0; i < waveScreen.length; i++) {
                if (i === 0) ctx.moveTo(waveScreen[i].sx, waveScreen[i].sy);
                else ctx.lineTo(waveScreen[i].sx, waveScreen[i].sy);
            }
            ctx.strokeStyle = curveColor;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Per-step value dots, on top of every line that RUNS: the drawn
            // profile plus, in multi-phase, one series per sibling Input - that
            // is the case where reading values off the plot is hardest, since
            // the waves cross and each Input takes a different value at the
            // same step. Non-selected band underlays stay bare on purpose (they
            // carry their own grabbable markers). Drawn after the
            // curves so the dots sit on the lines, before the handles so a
            // handle is never hidden by one.
            if (steps > 1 && r.w / steps >= STEP_DOT_MIN_SPACING) {
                for (let k = 1; k < phaseCount; k++) {
                    this.drawStepDots(steps, (x) => this.phaseValueAt(x, k),
                                      c.stepDot, PHASE_DOT_ALPHA, c.background,
                                      STEP_DOT_RADIUS);
                }
                this.drawStepDots(steps, (x) => this.evaluate(x), curveColor,
                                  MAIN_DOT_ALPHA, null, MAIN_DOT_RADIUS);
            }

            // Segment mid controls: small green handles under the main
            // vertices; passive linear midpoints are hollow, active parabola
            // controls are filled.
            for (let i = 0; i + 1 < this.points.length; i++) {
                const m = this.midHandle(i);
                if (!m) continue;
                const { sx, sy } = this.toScreen(m);
                const held = m.p0 === this.dragMid;
                ctx.beginPath();
                ctx.arc(sx, sy, held ? MID_POINT_RADIUS + 1 : MID_POINT_RADIUS, 0, Math.PI * 2);
                ctx.fillStyle = m.active ? MID_COLOR : c.background;
                ctx.fill();
                ctx.strokeStyle = MID_COLOR;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Points.
            for (const p of this.points) {
                const { sx, sy } = this.toScreen(p);
                const active = p === this.dragPoint;
                ctx.beginPath();
                ctx.arc(sx, sy, active ? POINT_RADIUS + 1 : POINT_RADIUS, 0, Math.PI * 2);
                ctx.fillStyle = active ? curveColor : c.background;
                ctx.fill();
                ctx.strokeStyle = curveColor;
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // Coordinates readout for the point or mid control being dragged.
            const heldMid = this.dragMid && this.dragMid.mid
                ? this.midHandle(this.points.indexOf(this.dragMid))
                : null;
            const readout = heldMid || this.dragPoint;
            if (readout) {
                const { sx, sy } = this.toScreen(readout);
                const label = `${readout.x.toFixed(2)} , ${readout.y.toFixed(2)}`;
                ctx.font = "11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "bottom";
                const tx = Math.min(Math.max(sx, r.left + 24), r.left + r.w - 24);
                const ty = sy - 10 < r.top + 12 ? sy + 22 : sy - 10;
                ctx.fillStyle = heldMid ? MID_COLOR : (this.dragColor || curveColor);
                ctx.fillText(label, tx, ty);
            }

            // Band mode: C / M / F letters riding their lines (staggered x
            // positions, since untouched bands all sit flat at 1.0). The three
            // main-mode plots each draw one curve and need no letters.
            if (!this.mainModeSelector(this.band)) {
                ctx.font = "bold 11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "bottom";
                for (const b of BAND_ORDER) {
                    const lx = BAND_LABEL_X[b];
                    const s = this.toScreen({ x: lx, y: this.profileValueAt(this.bandProfileFor(b), b, lx) });
                    ctx.fillStyle = BAND_COLORS[b];
                    ctx.fillText(BAND_LABELS[b], s.sx, Math.max(s.sy - 3, r.top + 11));
                }
            }
        }
    }

    const editors = [];
    const initialized = new WeakSet();

    // Headless load (node, no webui globals): the grammar half of this file is
    // the python side's twin and is covered by tests/test_profile_parity.py,
    // which needs the class without any of the DOM wiring below. Exporting it
    // and guarding the registrations is all that takes - keep both, or the
    // parity test silently stops running.
    // driftedDepth rides along because it is the DEFINITION of the drift, not a
    // drawing detail: the parity test compares the composite
    // depth(layer - drift(step)) against python's depth_multiplier, and a test
    // that recomputed the shift itself would agree with whatever it recomputed.
    if (typeof module !== "undefined" && module.exports) {
        module.exports = {
            WeightProfileEditor: WeightProfileEditor,
            driftedDepth: driftedDepth,
            SELECTOR_ORDER: SELECTOR_ORDER,
        };
    }
    // ...and the same three in the BROWSER, for the coverage panel: it has to
    // turn a profile string into per-step weights, and the one thing it must
    // not do is parse the grammar a second time. A second parser would agree
    // with this one right up to the first wave, mid control or convergence
    // token - and then quietly draw a coverage map of a profile nobody is
    // running. Read-only use: the panel never constructs an editor.
    if (typeof window !== "undefined") {
        window.cnproWeightProfile = {
            WeightProfileEditor: WeightProfileEditor,
            driftedDepth: driftedDepth,
            SELECTOR_ORDER: SELECTOR_ORDER,
        };
    }
    if (typeof onUiUpdate !== "function") return;

    onUiUpdate(() => {
        gradioApp().querySelectorAll(".cnet-weight-profile-editor").forEach((block) => {
            if (initialized.has(block)) return;
            const container = block.querySelector(".cnet-weight-profile");
            const row = block.closest(".controlnet_weight_steps");
            const textarea = row && row.querySelector(".cnet-weight-profile-state textarea");
            if (!container || !textarea) return;
            // latched BEFORE the constructor on purpose (a throwing ctor must
            // not retry-storm every ui update), but a throw is at least SAID -
            // the old silent blacklisting left the widget dead with no clue
            initialized.add(block);
            try {
                editors.push(new WeightProfileEditor(container, textarea));
            } catch (err) {
                console.warn("[controlnet profile] editor init failed for", block, err);
            }
        });
    });

    // Redraw editors when their hidden textbox is updated from the python side
    // (infotext paste, UI state restore, ...). onAfterUiUpdate rides a
    // childList-only MutationObserver, so a VALUE-only textbox write produces
    // no callback at all - pastes only worked because they happen to churn
    // other DOM. The interval below is the guaranteed channel (one string
    // compare per editor per tick when idle); the observer path stays as the
    // fast lane. The same tick re-checks devicePixelRatio (a monitor change
    // resizes nothing, so the ResizeObserver never fires), the model-type
    // gate of the band buttons, and the multi-phase input count (uploading /
    // clearing / muting an Input changes how many sibling waves run, but
    // touches nothing the editor observes).
    const reloadAll = () => editors.forEach((editor) => editor.maybeReload());
    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(reloadAll);
    } else {
        onUiUpdate(reloadAll);
    }

    // A theme flip changes the curve colors (javascript/theme.js), and a plot
    // is a canvas: nothing about it re-renders on its own when a CSS variable
    // changes. Drop the resolved cache and repaint, or the main line keeps the
    // colour of the theme the page was loaded in - which on a switch INTO the
    // light theme is white on white, exactly the failure this exists to fix.
    if (window.cnproTheme) {
        window.cnproTheme.onChange(() => {
            invalidateCurveColors();
            // No editors yet means this is onChange's registration call, which
            // lands while this module is still evaluating - the exact moment
            // the resolution is lazy to avoid (no stylesheet guaranteed). Drop
            // the cache and let the constructor do it, as before.
            if (!editors.length) return;
            // With editors live it MUST be eager: curveColorOf() reads
            // BAND_COLORS directly and only the constructor ever fills it, so a
            // null cache here is a null dereference on the next draw.
            resolveBandColors();
            for (const editor of editors) editor.draw();
        });
    }
    // shared tick (active_canvas.js): one timer for every module here, and
    // skipped entirely while no unit body is laid out
    window.cnetRegisterTick(() => {
        for (const editor of editors) {
            editor.maybeReload();
            if (editor._lastDpr && (window.devicePixelRatio || 1) !== editor._lastDpr) {
                editor.resize();
            }
            // multi-phase preview follows the number of active inputs; the
            // count is written back here too (not only in draw) so a hidden
            // canvas (draw early-returns at zero size) does not retry-storm
            if (editor.cosOn && editor.phaseFamily) {
                const n = editor.multiPhaseCount();
                if (n !== editor._lastPhaseCount) {
                    editor._lastPhaseCount = n;
                    editor.draw();
                }
            }
        }
    });
})();
