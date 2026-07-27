"""The Z-Image injection hooks must hit the right blocks and leave no trace.

This is the part of the port that cannot be checked by reading. CNPro adds its
residuals with `torch.nn.Module` forward pre-hooks on the host's own
`backend/nn/lumina.py::NextDiT`, because that class ignores the `control`
argument the host passes it (it has `**kwargs`, so the residuals would be
silently dropped - a ControlNet that appears to work and does nothing).

Hooks are powerful and easy to get subtly wrong, so this exercises the REAL
NextDiT class - built small, but the same code path the 6B model runs - and
pins the four things that must hold:

  1. the hint lands on the blocks in `control_layers_places`, and on no others;
  2. `prefix_len` really is where the image span starts, so a spatial mask lands
     on image tokens rather than on the caption;
  3. removing the injector restores BIT-IDENTICAL behaviour - "enabling CNPro
     cannot change anything for anyone not using CNPro" is the extension's
     first design rule, and a leaked hook breaks it for the whole session;
  4. install() REFUSES a model that does not look like NextDiT, instead of
     injecting into whatever it found.

Run:  <webui python> extensions/forge-neo-cnpro/tests/test_zimage_injection.py
Needs torch and the host on sys.path. Runs on CPU in a second or two.
Exit code 0 = pass.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
WEBUI = os.path.dirname(os.path.dirname(EXTENSION))
sys.path.insert(0, EXTENSION)
sys.path.insert(0, WEBUI)
sys.path.insert(0, os.path.join(WEBUI, "modules_forge", "packages"))

import torch  # noqa: E402

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


# A NextDiT small enough to run on CPU. Two constraints fix these numbers:
#   * NextDiT asserts dim // n_heads == sum(axes_dims);
#   * dim must be >= 256. The host builds `t_embedder` with output_size=256 when
#     z_image_modulation is on, but each block's adaLN takes min(dim, 256) - so
#     below 256 the host's own model does not typecheck against itself. Worth
#     knowing before trying to debug a "CNPro" shape error at dim=32.
DIM = 256
N_HEADS = 8
HEAD_DIM = DIM // N_HEADS          # 32
AXES = [8, 12, 12]                 # sums to 32
N_LAYERS = 6
N_REFINER = 2
CAP_DIM = 8
PATCH = 2
IN_CH = 16


def init_weights(model, seed):
    """Give every parameter a finite value.

    `using_forge_operations` builds layers with UNINITIALISED storage - Forge
    never initialises weights it is about to overwrite from a state dict, which
    is a real speedup and a real trap: a model built that way and run without
    loading anything produces NaN, and every comparison against NaN is False.
    (`torch.equal(nan, nan)` is False, so a bit-identity test would "fail" for a
    reason that has nothing to do with what it is testing.)
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or "_norm" in name or name.endswith("scale"):
                p.copy_(torch.ones_like(p))          # norms: identity-ish
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
        for name, b in model.named_buffers():
            if b.is_floating_point():
                b.copy_(torch.zeros_like(b))
    return model


def build_base():
    # Must be built under using_forge_operations: the host's own layers carry
    # `weight_function` / manual-cast plumbing that its forward assumes, and a
    # plainly-constructed NextDiT raises inside its first attention.
    from backend.nn.lumina import NextDiT
    from backend.operations import using_forge_operations
    torch.manual_seed(0)
    with using_forge_operations(dtype=torch.float32, manual_cast_enabled=False):
        model = NextDiT(
            patch_size=PATCH, in_channels=IN_CH, dim=DIM, n_layers=N_LAYERS,
            n_refiner_layers=N_REFINER, n_heads=N_HEADS, n_kv_heads=N_HEADS,
            multiple_of=8, ffn_dim_multiplier=8.0 / 3.0, norm_eps=1e-5, qk_norm=True,
            cap_feat_dim=CAP_DIM, axes_dims=AXES, axes_lens=[1536, 512, 512],
            rope_theta=256.0, z_image_modulation=True, time_scale=1000.0,
            pad_tokens_multiple=32,
        )
    return init_weights(model.to(torch.float32), 0).eval()


def build_control(n_control_layers):
    from backend.operations import using_forge_operations
    from cnpro_host.patchers.zimage_impl import ZImageControlNetModel
    torch.manual_seed(1)
    with using_forge_operations(dtype=torch.float32, manual_cast_enabled=False):
        model = ZImageControlNetModel(
            dim=DIM, heads=N_HEADS, hidden_dim=DIM * 2, control_in_dim=IN_CH,
            patch_size=PATCH, n_control_layers=n_control_layers,
            n_refiner_layers=N_REFINER)
    return init_weights(model.to(torch.float32), 1).eval()


class StubControl:
    """Stands in for the ZImageControlNet chain: records what it was asked for."""

    def __init__(self, hints_for, record):
        self.hints_for = hints_for
        self.record = record

    def run_inside_forward(self, sequence, prefix_len, freqs_cis, adaln_input,
                           x_pad_token, places, block_count, transformer_options):
        self.record["sequence_shape"] = tuple(sequence.shape)
        self.record["prefix_len"] = prefix_len
        self.record["places"] = list(places)
        self.record["block_count"] = block_count
        self.record["calls"] = self.record.get("calls", 0) + 1
        return {p: torch.full_like(sequence, float(p + 1)) for p in self.hints_for}


def run_base(model, x, ctx):
    with torch.no_grad():
        return model(x, torch.tensor([0.5]), ctx, attention_mask=None)


def test_hooks_hit_the_right_blocks():
    from cnpro_host.patchers.zimage_impl import Injector

    model = build_base()
    x = torch.randn(1, IN_CH, 8, 8)
    ctx = torch.randn(1, 7, CAP_DIM)

    baseline = run_base(model, x, ctx)

    places = [0, 2, 4]
    record = {}
    seen = []

    inj = Injector.install(model, places)

    # Watch what each base block actually receives. Registered AFTER the injector
    # on purpose: forward pre-hooks fire in registration order, so an observer
    # attached first would see the tensor BEFORE the injection and report that
    # nothing happened. (Learned the hard way; the ordering is the test.)
    # Watch block OUTPUTS, because that is where the hint is added now
    # (upstream: `unified = layer(...); unified = unified + samples[layer_idx]`).
    # Registered AFTER the injector so this observer runs after the injecting
    # hook - forward hooks fire in registration order, and an observer attached
    # first would report that nothing happened.
    def observe(m, args, output, i):
        seen.append((i, output.clone()))

    handles = [
        blk.register_forward_hook(lambda m, a, o, i=i: observe(m, a, o, i))
        for i, blk in enumerate(model.layers)
    ]
    # Exactly the host's transport: sampling_function.py sets
    # `c["control"] = control.get_control(...)` and then calls
    # `model.apply_model(input_x, timestep_, **c)`, so the chain arrives as a
    # `control=` kwarg that NextDiT itself ignores.
    with torch.no_grad():
        model(x, torch.tensor([0.5]), ctx, attention_mask=None,
              control=StubControl(places, record))

    for h in handles:
        h.remove()

    if record.get("calls") != 1:
        fail("control tower should run exactly once per forward, ran %r times"
             % record.get("calls"))
    if record.get("block_count") != N_LAYERS:
        fail("block_count %r != %d" % (record.get("block_count"), N_LAYERS))
    if record.get("places") != places:
        fail("places %r != %r" % (record.get("places"), places))

    # prefix_len must equal the padded caption length, and the sequence must be
    # caption-then-image (this host's order).
    seq_len = record["sequence_shape"][1]
    prefix = record["prefix_len"]
    cap_padded = 7 + (-7 % 32)
    if prefix != cap_padded:
        fail("prefix_len %r != padded caption length %r" % (prefix, cap_padded))
    img_tokens = (8 // PATCH) * (8 // PATCH)
    img_padded = img_tokens + (-img_tokens % 32)
    if seq_len != cap_padded + img_padded:
        fail("sequence %r != caption %r + image %r" % (seq_len, cap_padded, img_padded))

    # StubControl returns a CONSTANT (place + 1) for each site, so the effect is
    # directly measurable: run again with no chain and diff what each block saw.
    with_hint = {i: t for i, t in seen}
    seen.clear()
    handles = [
        blk.register_forward_hook(lambda m, a, o, i=i: observe(m, a, o, i))
        for i, blk in enumerate(model.layers)
    ]
    with torch.no_grad():
        model(x, torch.tensor([0.5]), ctx, attention_mask=None)
    for h in handles:
        h.remove()
    without = {i: t for i, t in seen}

    for i in range(N_LAYERS):
        if i not in with_hint or i not in without:
            fail("no observation for block %d" % i)
            continue
        delta = (with_hint[i] - without[i])
        if i == 0:
            # block 0 is the first site: its OUTPUT differs by exactly the hint
            if not torch.allclose(delta, torch.full_like(delta, 1.0), atol=1e-5):
                fail("block 0's output should differ by exactly the constant hint "
                     "1.0, got min %r max %r"
                     % (delta.min().item(), delta.max().item()))
        elif i == 1:
            # block 1 is NOT a site; its output still moves because block 0's did
            pass
        elif i in places:
            if delta.abs().max().item() < 1e-6:
                fail("block %d is an injection site but its input did not change" % i)
        else:
            pass

    # The decisive check that no site is missed or invented: ask for hints at ONE
    # place and confirm only that block's input jumps by the constant.
    seen.clear()
    Injector.uninstall(model)
    inj2 = Injector.install(model, [4])
    handles = [
        blk.register_forward_hook(lambda m, a, o, i=i: observe(m, a, o, i))
        for i, blk in enumerate(model.layers)
    ]
    with torch.no_grad():
        model(x, torch.tensor([0.5]), ctx, attention_mask=None,
              control=StubControl([4], {}))
    for h in handles:
        h.remove()
    only4 = {i: t for i, t in seen}
    for i in range(N_LAYERS):
        delta = (only4[i] - without[i]).abs().max().item()
        if i < 4 and delta > 1e-6:
            fail("injecting only at block 4 changed block %d's output (delta %r)" % (i, delta))
        if i == 4 and delta < 1e-6:
            fail("injecting at block 4 did not change block 4's output")
    Injector.uninstall(model)
    del inj2

    if not FAILURES:
        # re-run with no injector and confirm the base result is untouched
        Injector.uninstall(model)
        after = run_base(model, x, ctx)
        if not torch.equal(baseline, after):
            fail("removing the injector did not restore the base model exactly "
                 "(max dev %r)" % (baseline - after).abs().max().item())
        if hasattr(model, Injector.ATTR):
            fail("uninstall left the injector attribute on the model")
        if inj.handles:
            fail("uninstall left %d hook handles alive" % len(inj.handles))


def test_hint_actually_changes_the_output():
    from cnpro_host.patchers.zimage_impl import Injector

    model = build_base()
    x = torch.randn(1, IN_CH, 8, 8)
    ctx = torch.randn(1, 7, CAP_DIM)
    baseline = run_base(model, x, ctx)

    Injector.install(model, [0, 2, 4])
    with torch.no_grad():
        injected = model(x, torch.tensor([0.5]), ctx, attention_mask=None,
                         control=StubControl([0, 2, 4], {}))
    if torch.equal(baseline, injected):
        fail("injecting a non-zero hint did not change the output at all - the "
             "hooks are not reaching the residual stream")

    # An empty hint dict must be a no-op: this is the path taken when the weight
    # profile evaluates to (near) zero and the control tower is skipped.
    with torch.no_grad():
        empty = model(x, torch.tensor([0.5]), ctx, attention_mask=None,
                      control=StubControl([], {}))
    if not torch.equal(baseline, empty):
        fail("an empty hint dict changed the output; the zero-strength skip path "
             "is not free")

    # No control at all, with hooks still installed, must also be free - two
    # units where only one is active is the ordinary multi-unit case.
    with torch.no_grad():
        none_at_all = model(x, torch.tensor([0.5]), ctx, attention_mask=None)
    if not torch.equal(baseline, none_at_all):
        fail("installed hooks changed the output when no control was passed")
    Injector.uninstall(model)


def test_control_tower_runs():
    """The real tower, not a stub: shapes must line up with the base sequence."""
    from cnpro_host.patchers.zimage_impl import Injector

    model = build_base()
    control = build_control(3)
    x = torch.randn(1, IN_CH, 8, 8)
    ctx = torch.randn(1, 7, CAP_DIM)
    hint_latent = torch.randn(1, IN_CH, 8, 8)

    captured = {}

    class RealChain:
        def run_inside_forward(self, sequence, prefix_len, freqs_cis, adaln_input,
                               x_pad_token, places, block_count, transformer_options):
            hints = control(hint_latent, sequence, prefix_len, freqs_cis,
                            adaln_input, x_pad_token, transformer_options)
            captured["shapes"] = [tuple(h.shape) for h in hints]
            captured["sequence"] = tuple(sequence.shape)
            return dict(zip(places, hints))

    Injector.install(model, [0, 2, 4])
    with torch.no_grad():
        model(x, torch.tensor([0.5]), ctx, attention_mask=None, control=RealChain())
    Injector.uninstall(model)

    if "shapes" not in captured:
        fail("the control tower never ran")
        return
    for s in captured["shapes"]:
        if s != captured["sequence"]:
            fail("hint %r does not match the base sequence %r - it cannot be "
                 "added to the residual stream" % (s, captured["sequence"]))


def test_install_refuses_a_foreign_model():
    from cnpro_host.patchers.zimage_impl import Injector, UnsupportedZImageControlNet

    class NotADiT(torch.nn.Module):
        pass

    try:
        Injector.install(NotADiT(), [0])
    except UnsupportedZImageControlNet as exc:
        if "lumina" not in str(exc).lower() and "layers" not in str(exc).lower():
            fail("refusal message should name what it expected: %r" % str(exc))
    else:
        fail("install() accepted a model with no `layers` - it must refuse")

    model = build_base()
    try:
        Injector.install(model, [0, 99])
    except UnsupportedZImageControlNet:
        pass
    else:
        Injector.uninstall(model)
        fail("install() accepted an injection site past the end of the model")


def test_v2_two_stage():
    """v2.1 injects TWICE: into noise_refiner[0,1], then into the block stack.

    The refiner stage is the only genuinely new mechanism in v2, and it is easy
    to get subtly wrong in a way that still produces an image: it must run at the
    boundary of the base model's FIRST refiner block (where `x` is the raw
    embedded tokens, before any refinement) and its hints must land on the
    OUTPUT of each refiner block, not the input.
    """
    from backend.operations import using_forge_operations
    from cnpro_host.patchers.zimage_impl import Injector, ZImageControlNetModel

    model = build_base()
    torch.manual_seed(2)
    with using_forge_operations(dtype=torch.float32, manual_cast_enabled=False):
        tower = ZImageControlNetModel(
            dim=DIM, heads=N_HEADS, hidden_dim=DIM * 2, control_in_dim=33,
            patch_size=PATCH, n_control_layers=4, n_refiner_layers=N_REFINER,
            variant='v2', refiner_is_control=True)
    tower = init_weights(tower.to(torch.float32), 2).eval()

    x = torch.randn(1, IN_CH, 8, 8)
    ctx = torch.randn(1, 7, CAP_DIM)
    hint33 = torch.randn(1, 33, 8, 8)
    baseline = run_base(model, x, ctx)

    seen = {}

    class V2Chain:
        def refine_inside_forward(self, x_unrefined, image_freqs, adaln, x_pad_token,
                                  refiner_places, base_noise_refiner, opts):
            seen['refine_called'] = seen.get('refine_called', 0) + 1
            seen['x_unrefined_shape'] = tuple(x_unrefined.shape)
            seen['refiner_places'] = list(refiner_places)
            seen['got_base_refiner'] = len(base_noise_refiner)
            tokens = tower.embed_control(hint33, x_unrefined.shape[1], x_pad_token)
            hints, stream = tower.refine_v2(tokens, x_unrefined, image_freqs, adaln, opts)
            self.stream = stream
            seen['n_refiner_hints'] = len(hints)
            seen['refiner_hint_shape'] = tuple(hints[0].shape)
            # the tower's PRIVATE refined latent: base blocks re-run, hints added
            x_ctrl = x_unrefined
            for i, blk in enumerate(base_noise_refiner):
                x_ctrl = blk(x_ctrl, None, image_freqs, adaln, transformer_options=opts)
                if i < len(hints) and hints[i] is not None:
                    x_ctrl = x_ctrl + hints[i]
            self.unified_x = x_ctrl
            seen['unified_x_shape'] = tuple(x_ctrl.shape)

        def run_inside_forward(self, sequence, prefix_len, freqs_cis, adaln,
                               x_pad_token, places, block_count, opts):
            seen['emit_called'] = seen.get('emit_called', 0) + 1
            seen['sequence_shape'] = tuple(sequence.shape)
            unified = torch.cat((sequence[:, :prefix_len], self.unified_x), dim=1)
            hints = tower.emit(self.stream, sequence, prefix_len, freqs_cis, adaln,
                               opts, unified=unified)
            seen['main_hint_shape'] = tuple(hints[0].shape)
            return dict(zip(places, hints))

    places = [0, 2, 4]
    refiner_places = [0, 1]
    Injector.install(model, places, refiner_places)
    with torch.no_grad():
        out = model(x, torch.tensor([0.5]), ctx, attention_mask=None, control=V2Chain())
    Injector.uninstall(model)

    if seen.get('refine_called') != 1:
        fail("the refiner stage ran %r times, expected exactly 1" % seen.get('refine_called'))
    if seen.get('emit_called') != 1:
        fail("the main stage ran %r times, expected exactly 1" % seen.get('emit_called'))
    if seen.get('refiner_places') != refiner_places:
        fail("refiner_places %r != %r" % (seen.get('refiner_places'), refiner_places))
    if seen.get('n_refiner_hints') != N_REFINER:
        fail("expected %d refiner hints, got %r" % (N_REFINER, seen.get('n_refiner_hints')))

    # stage one sees the IMAGE span only (no caption), stage two the joint sequence
    img_tokens = (8 // PATCH) * (8 // PATCH)
    img_padded = img_tokens + (-img_tokens % 32)
    if seen.get('x_unrefined_shape', (0, 0))[1] != img_padded:
        fail("the refiner stage got a %r sequence; it must see the image span "
             "only (%d tokens), not the joint sequence"
             % (seen.get('x_unrefined_shape'), img_padded))
    if seen.get('refiner_hint_shape') != seen.get('x_unrefined_shape'):
        fail("refiner hint %r does not match what it is added to %r"
             % (seen.get('refiner_hint_shape'), seen.get('x_unrefined_shape')))
    if seen.get('main_hint_shape') != seen.get('sequence_shape'):
        fail("main hint %r does not match the joint sequence %r"
             % (seen.get('main_hint_shape'), seen.get('sequence_shape')))

    if torch.equal(baseline, out):
        fail("the v2 two-stage injection did not change the output at all")

    if seen.get('got_base_refiner') != N_REFINER:
        fail("the refiner stage was not handed the base model's noise_refiner "
             "(got %r) - it cannot build the tower's private latent without it"
             % seen.get('got_base_refiner'))
    if seen.get('unified_x_shape') != seen.get('x_unrefined_shape'):
        fail("the tower's private latent %r does not match the image span %r"
             % (seen.get('unified_x_shape'), seen.get('x_unrefined_shape')))

    # The refiner hints must NOT reach the base model's latent. Upstream's
    # transformer runs noise_refiner with no hint application at all; the hints
    # belong to the tower's private copy. If a hook still added them to the base
    # blocks, running with a chain that injects NOTHING at `places` would still
    # change the output - so that is the check.
    class RefineOnly(V2Chain):
        def run_inside_forward(self, *a, **k):
            return {}

    Injector.install(model, places, refiner_places)
    with torch.no_grad():
        refine_only = model(x, torch.tensor([0.5]), ctx, attention_mask=None,
                            control=RefineOnly())
    Injector.uninstall(model)
    if not torch.equal(baseline, refine_only):
        fail("the refiner stage altered the base model's latent (max dev %r). Those "
             "hints belong to the control tower's private copy only - "
             "transformer_z_image.py applies none of them to the transformer."
             % (baseline - refine_only).abs().max().item())


def test_hint_channels_are_stable_across_steps():
    """EVERY step must get the tower's channel count, not just the first.

    This is the bug the user reported as "control holds only on step 1". The hint
    is encoded once and cached; the cache-hit path returned the RAW 16-channel
    latent while the miss path returned the padded 33-channel one. Step 1 worked,
    steps 2..N raised inside embed_control, the refiner stage was skipped, and the
    unit injected nothing - silently, because the exception was caught and logged.

    v1 could never show it: padding 16 -> 16 is a no-op. So the test has to run a
    v2-shaped config, and it has to call the hint path MORE THAN ONCE.
    """
    from cnpro_host.patchers.zimage_impl import ZImageControlNet

    class Stub(ZImageControlNet):
        def __init__(self, want):
            self.config = {'control_in_dim': want, 'patch_size': PATCH}
            self.cond_hint_original = torch.rand(1, 3, 64, 64)
            self.cond_hint_latent = None
            self.latent_scale = 8
            # a fake VAE: 3-channel pixels -> 16-channel latent at /8
            self.encode_hint = lambda px: torch.randn(
                px.shape[0], 16, px.shape[-2] // 8, px.shape[-1] // 8)

    for want in (16, 33):
        s = Stub(want)
        x_noisy = torch.zeros(1, 16, 8, 8)
        got = [tuple(s._hint_for(x_noisy).shape) for _ in range(4)]
        if len(set(got)) != 1:
            fail("control_in_dim=%d: the hint changes shape across steps %r - the "
                 "cache and the encode path disagree, and every step after the "
                 "first will fail inside embed_control" % (want, got))
        if got[0][1] != want:
            fail("control_in_dim=%d: hint has %d channels, want %d"
                 % (want, got[0][1], want))


def test_hole_channel_polarity():
    """Channel 16 must be 1 where pixels are KEPT, and the hole blacked out.

    The single most dangerous line in the v2 path. Upstream builds it as
    ``interpolate(1 - mask, ..., mode="nearest")`` and blacks the hole out in
    PIXEL space before the VAE. Both are inverted relative to how a UI mask
    reads, so a sign error here inpaints the complement of what the user painted
    and still produces a completely plausible image - the exact failure no
    eyeball test catches.
    """
    from cnpro_host.patchers.zimage_impl import ZImageControlNet

    H = W = 64
    LH = LW = H // 8

    class Stub(ZImageControlNet):
        def __init__(self):
            self.config = {'control_in_dim': 33, 'patch_size': PATCH}
            self.latent_scale = 8
            self._logged_hole = True
            # top half painted = the region to REGENERATE
            m = torch.zeros(1, 1, H, W)
            m[:, :, : H // 2, :] = 1.0
            self.hole_mask = m
            self.known_pixels = torch.ones(1, 3, H, W)   # all-white known image
            self.seen = {}
            def enc(px):
                # record what the VAE was handed, then fake a latent from it
                self.seen['mean_top'] = float(px[:, :, : px.shape[-2] // 2, :].mean())
                self.seen['mean_bottom'] = float(px[:, :, px.shape[-2] // 2:, :].mean())
                return torch.zeros(1, 16, px.shape[-2] // 8, px.shape[-1] // 8)
            self.encode_hint = enc

    s = Stub()
    extra = s._hole_channels((LH, LW), torch.float32, torch.device('cpu'))
    if extra is None:
        fail("_hole_channels returned None with a mask and a known image present")
        return
    if tuple(extra.shape) != (1, 17, LH, LW):
        fail("hole channels have shape %r, want (1,17,%d,%d)" % (tuple(extra.shape), LH, LW))
        return

    keep = extra[:, 0]
    top = float(keep[:, : LH // 2, :].mean())
    bottom = float(keep[:, LH // 2:, :].mean())
    if not (top < 0.01):
        fail("the PAINTED half (to regenerate) must be 0 in channel 16, got %.3f. "
             "Channel 16 is 1 = KEEP - inverted from the UI mask." % top)
    if not (bottom > 0.99):
        fail("the UNPAINTED half (keep) must be 1 in channel 16, got %.3f" % bottom)

    # mask must be NEAREST-sampled: only exact 0s and 1s, no grey halo
    vals = torch.unique(keep)
    if not torch.all((vals < 1e-6) | (vals > 1 - 1e-6)):
        fail("channel 16 has intermediate values %r - it was not nearest-sampled, "
             "and the model reads a grey halo as 'half known'" % vals.tolist()[:6])

    # the known image must have been BLACKED OUT in the hole before encoding
    if s.seen.get('mean_top', 1.0) > 0.01:
        fail("the hole was not blacked out before the VAE (top half mean %.3f); "
             "upstream zeroes it in pixel space" % s.seen.get('mean_top'))
    if s.seen.get('mean_bottom', 0.0) < 0.99:
        fail("the KEPT half was altered before encoding (mean %.3f, want ~1.0)"
             % s.seen.get('mean_bottom'))


def test_no_inpaint_context_is_zeros():
    """Without an inpaint context the extra channels must be exactly zero."""
    from cnpro_host.patchers.zimage_impl import ZImageControlNet

    class Stub(ZImageControlNet):
        def __init__(self):
            self.config = {'control_in_dim': 33, 'patch_size': PATCH}
            self.latent_scale = 8
            self._logged_hole = True
            self.hole_mask = None
            self.known_pixels = None
            self.encode_hint = None

    got = Stub()._pad_channels(torch.randn(1, 16, 8, 8))
    if tuple(got.shape) != (1, 33, 8, 8):
        fail("plain control should still widen to 33, got %r" % (tuple(got.shape),))
    elif got[:, 16:].abs().max().item() != 0.0:
        fail("channels 16..32 must be ZERO without an inpaint context "
             "(mask 0 = nothing decided, known pixels 0 = nothing to show)")


def test_channel_padding():
    """A 16-channel latent must widen to whatever the tower declares."""
    from cnpro_host.patchers.zimage_impl import ZImageControlNet

    class Stub(ZImageControlNet):
        def __init__(self, want):
            self.config = {'control_in_dim': want}

    lat = torch.randn(1, 16, 8, 8)
    got = Stub(33)._pad_channels(lat)
    if tuple(got.shape) != (1, 33, 8, 8):
        fail("padding to 33 gave %r" % (tuple(got.shape),))
    if not torch.equal(got[:, :16], lat):
        fail("padding altered the real control channels")
    if got[:, 16:].abs().max().item() != 0.0:
        fail("the padded channels are not zero (mask 0 = 'nothing decided', "
             "known-pixels 0 = 'nothing to show')")
    if tuple(Stub(16)._pad_channels(lat).shape) != (1, 16, 8, 8):
        fail("a v1 tower must get its 16 channels unchanged")


def main():
    for fn in (test_hooks_hit_the_right_blocks, test_hint_actually_changes_the_output,
               test_control_tower_runs, test_install_refuses_a_foreign_model,
               test_v2_two_stage, test_channel_padding,
               test_hint_channels_are_stable_across_steps,
               test_hole_channel_polarity, test_no_inpaint_context_is_zeros):
        try:
            fn()
        except Exception as exc:
            import traceback
            fail("%s raised %s: %s\n%s" % (fn.__name__, type(exc).__name__, exc,
                                           traceback.format_exc()))

    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - hooks inject at the configured blocks, the control tower's hints "
          "match the base sequence, and removal restores the model bit-exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
