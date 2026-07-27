"""The weighting engine must behave identically on UNet and token residuals.

`cnpro_host/patchers/weighting.py` was extracted from the ControlNet engine and
parameterised by a `ResidualLayout` so the DiT patchers could share it. Two
things have to be true for that to be safe, and neither is obvious by reading:

  1. NOTHING CHANGED FOR UNETS. The UNet layout must reproduce the historical
     `band_of` / `depth_fraction_of_residual` mapping exactly, for every group
     and every layer count - a one-layer drift in the band split silently
     re-weights every existing SD1.x/SDXL profile.

  2. THE TOKEN LAYOUT KEEPS THE TWO MASK INVARIANTS the UI promises: an all-ones
     mask is EXACTLY a no-op, and an all-zeros mask turns the injection fully
     off. Those are what make "paint where the control applies" mean what it
     says, and the token layout has to buy them while also deciding what a
     spatial mask means for non-spatial caption tokens.

Run:  <webui python> extensions/forge-neo-cnpro/tests/test_residual_layout.py
Needs torch. Does NOT need the host (neither module imports backend.*).
Exit code 0 = pass.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
sys.path.insert(0, EXTENSION)

import torch  # noqa: E402

from cnpro_core.weight_profile import (  # noqa: E402
    band_of_depth_fraction,
    depth_fraction_of_ordered_site,
    depth_fraction_of_residual,
)
from cnpro_host.patchers.residual_layout import (  # noqa: E402
    UNET_LAYOUT,
    TokenResidualLayout,
)
from cnpro_host.patchers.weighting import compute_controlnet_weighting  # noqa: E402

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def historical_band_of(k, i, n):
    """The mapping as it was written before the layout split. Frozen on purpose."""
    if k == 'middle':
        return 'coarse'
    idx = min(i * 3 // max(n, 1), 2)
    if k == 'input':
        idx = 2 - idx
    return ('fine', 'mid', 'coarse')[idx]


class FakeCnet:
    """The attribute surface compute_controlnet_weighting reads."""

    def __init__(self, batch=2, **kw):
        self.transformer_options = {
            'cond_or_uncond': [0, 1],
            'sigmas': torch.tensor([1.0]),
            # 0 = cond row, 1 = uncond row
            'cond_mark': torch.tensor([0.0] * (batch // 2) + [1.0] * (batch - batch // 2)),
        }
        self.cond_layer_weights = None
        self.uncond_layer_weights = None
        self.frame_weights = None
        self.sigma_weight_fn = None
        self.region_masks = None
        self.balance_sigmas = None
        self.balance_values = None
        self.band_profile_lookup = None
        self.depth_profile = None
        self.residual_layout = None
        self._layer_mask_cache = None
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# 1. the UNet layout must not have moved
# ---------------------------------------------------------------------------

def test_unet_layout_unchanged():
    for n in range(1, 20):
        for k in ('input', 'middle', 'output'):
            for i in range(n):
                want = historical_band_of(k, i, n)
                got = UNET_LAYOUT.band(k, i, n)
                if got != want:
                    fail("UNET_LAYOUT.band(%r,%d,%d) = %r, historically %r" % (k, i, n, got, want))
                wd = depth_fraction_of_residual(k, i, n)
                gd = UNET_LAYOUT.depth_fraction(k, i, n)
                if abs(gd - wd) > 1e-12:
                    fail("UNET_LAYOUT.depth_fraction(%r,%d,%d) = %r, want %r" % (k, i, n, gd, wd))


# ---------------------------------------------------------------------------
# 2. the token layout's depth axis
# ---------------------------------------------------------------------------

def test_token_depth_axis():
    places = [0, 5, 10, 15, 20, 25]
    layout = TokenResidualLayout((8, 8), prefix_len=32, places=places, block_count=30)

    fracs = [layout.depth_fraction('block', i, len(places)) for i in range(len(places))]
    if fracs != sorted(fracs):
        fail("token depth is not monotone in block order: %r" % (fracs,))
    if abs(fracs[0]) > 1e-12:
        fail("first injection site should sit at depth 0, got %r" % fracs[0])
    if fracs[-1] >= 1.0:
        fail("deepest injection is at block 25 of 30 - depth must be < 1, got %r" % fracs[-1])

    # band and depth are the same quantity here, so they cannot disagree
    for i, f in enumerate(fracs):
        want = band_of_depth_fraction(f)
        got = layout.band('block', i, len(places))
        if got != want:
            fail("band/depth disagree at site %d: band=%r depth=%r" % (i, got, f))

    # the released model's six sites split 2/2/2 across the three bands
    bands = [layout.band('block', i, 6) for i in range(6)]
    if bands != ['fine', 'fine', 'mid', 'mid', 'coarse', 'coarse']:
        fail("unexpected band split for the released 6-site model: %r" % (bands,))

    # shallow must be 'fine' and deep must be 'coarse', not the reverse
    if band_of_depth_fraction(0.0) != 'fine' or band_of_depth_fraction(0.99) != 'coarse':
        fail("depth->band direction is inverted")


# ---------------------------------------------------------------------------
# 3. the two mask invariants, on the token layout
# ---------------------------------------------------------------------------

def token_signal(batch=2, prefix=32, h=4, w=4, pad=0, dim=8):
    seq = prefix + h * w + pad
    return torch.randn(batch, seq, dim)


def test_token_mask_invariants():
    h, w, prefix, pad = 4, 4, 32, 16
    layout = TokenResidualLayout((h, w), prefix_len=prefix, places=[0], block_count=30)
    signal = token_signal(prefix=prefix, h=h, w=w, pad=pad)

    ones = torch.ones(1, 1, 64, 64)
    proj = layout.project_mask(signal, ones)
    if not torch.allclose(proj, torch.ones_like(proj)):
        fail("all-ones mask is not a no-op: min=%r max=%r" % (proj.min().item(), proj.max().item()))

    zeros = torch.zeros(1, 1, 64, 64)
    proj = layout.project_mask(signal, zeros)
    if proj.abs().max().item() > 1e-6:
        fail("all-zeros mask does not zero the injection: max=%r" % proj.abs().max().item())

    # a half-painted mask must land on the IMAGE span, in row-major order, and
    # give caption + padding tokens the mask's mean (see project_mask's docstring)
    half = torch.zeros(1, 1, h, w)
    half[:, :, : h // 2, :] = 1.0
    proj = layout.project_mask(signal, half)
    image = proj[0, prefix:prefix + h * w, 0].reshape(h, w)
    if not torch.allclose(image, half[0, 0]):
        fail("image span does not carry the painted mask:\n%r" % image)
    outside = proj[0, :prefix, 0]
    if not torch.allclose(outside, torch.full_like(outside, 0.5)):
        fail("caption tokens should take the mask mean (0.5 here), got %r" % outside[:3])
    padding = proj[0, prefix + h * w:, 0]
    if not torch.allclose(padding, torch.full_like(padding, 0.5)):
        fail("padding tokens should take the mask mean, got %r" % padding[:3])


# ---------------------------------------------------------------------------
# 4. the engine end to end, both layouts
# ---------------------------------------------------------------------------

def test_engine_token_path():
    h, w, prefix = 4, 4, 8
    places = [0, 5, 10, 15, 20, 25]
    layout = TokenResidualLayout((h, w), prefix_len=prefix, places=places, block_count=30)

    def fresh():
        return {'block': [torch.ones(2, prefix + h * w, 8) for _ in places]}

    # no profiles configured -> untouched
    cnet = FakeCnet(residual_layout=layout)
    before = fresh()
    after = compute_controlnet_weighting({'block': [t.clone() for t in before['block']]}, cnet)
    for a, b in zip(after['block'], before['block']):
        if not torch.allclose(a, b):
            fail("engine modified residuals with no profile configured")

    # depth profile: 0 at the shallowest site, 1 at the deepest
    cnet = FakeCnet(residual_layout=layout, depth_profile=[(0.0, 0.0), (1.0, 1.0)])
    out = compute_controlnet_weighting(fresh(), cnet)['block']
    scales = [t.mean().item() for t in out]
    if scales != sorted(scales):
        fail("depth profile is not monotone across injection sites: %r" % scales)
    if abs(scales[0]) > 1e-6:
        fail("depth profile 0 at depth 0 should zero the shallowest site, got %r" % scales[0])

    # band profile: only 'coarse' is non-zero -> only the two deepest sites survive
    cnet = FakeCnet(residual_layout=layout, band_profile_lookup={
        'coarse': ([1.0, 0.0], [1.0, 1.0]),
        'mid': ([1.0, 0.0], [0.0, 0.0]),
        'fine': ([1.0, 0.0], [0.0, 0.0]),
    })
    out = compute_controlnet_weighting(fresh(), cnet)['block']
    live = [i for i, t in enumerate(out) if t.abs().max().item() > 1e-6]
    if live != [4, 5]:
        fail("coarse-only band profile should leave sites [4, 5] live, got %r" % live)

    # all-ones mask must be a no-op through the WHOLE engine, not just the layout
    cnet = FakeCnet(residual_layout=layout, region_masks=torch.ones(1, 1, 32, 32))
    out = compute_controlnet_weighting(fresh(), cnet)['block']
    for i, t in enumerate(out):
        if not torch.allclose(t, torch.ones_like(t), atol=1e-6):
            fail("all-ones mask changed site %d (max dev %r)" % (i, (t - 1).abs().max().item()))

    # a band without its own mask injects nothing (restrict-to-painted)
    cnet = FakeCnet(residual_layout=layout,
                    region_masks={'coarse': torch.ones(1, 1, 32, 32)})
    out = compute_controlnet_weighting(fresh(), cnet)['block']
    live = [i for i, t in enumerate(out) if t.abs().max().item() > 1e-6]
    if live != [4, 5]:
        fail("only the coarse band is masked, so only sites [4,5] should inject, got %r" % live)


def test_engine_unet_path_still_4d():
    def fresh():
        return {'input': [torch.ones(2, 4, 8, 8) for _ in range(4)],
                'middle': [torch.ones(2, 4, 8, 8)],
                'output': [torch.ones(2, 4, 8, 8) for _ in range(4)]}

    cnet = FakeCnet(depth_profile=[(0.0, 0.0), (1.0, 1.0)])
    out = compute_controlnet_weighting(fresh(), cnet)
    if out['middle'][0].mean().item() != 1.0:
        fail("middle block sits at depth 1 and must keep full weight, got %r"
             % out['middle'][0].mean().item())
    # 'input' residuals are stored back-to-front, so index 0 is the DEEPEST
    if out['input'][0].mean().item() <= out['input'][-1].mean().item():
        fail("UNet 'input' depth direction flipped")

    cnet = FakeCnet(region_masks=torch.ones(1, 1, 16, 16))
    out = compute_controlnet_weighting(fresh(), cnet)
    for k, v in out.items():
        for i, t in enumerate(v):
            if not torch.allclose(t, torch.ones_like(t), atol=1e-6):
                fail("all-ones mask changed UNet %s[%d]" % (k, i))


def main():
    torch.manual_seed(0)
    for fn in (test_unet_layout_unchanged, test_token_depth_axis,
               test_token_mask_invariants, test_engine_token_path,
               test_engine_unet_path_still_4d):
        try:
            fn()
        except Exception as exc:  # a crash is a failure, and must name itself
            import traceback
            fail("%s raised %s: %s\n%s" % (fn.__name__, type(exc).__name__, exc,
                                           traceback.format_exc()))

    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - UNet weighting is bit-identical to before the layout split, and "
          "the token layout holds both mask invariants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
