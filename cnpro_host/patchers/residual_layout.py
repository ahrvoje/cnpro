"""Residual layouts: what shape a family's control residuals have, where each
one sits on the model's depth axis, and how a painted mask lands on it.

WHY THIS EXISTS
---------------
The weighting engine (``weighting.py``) is index math over lists of residual
tensors. Everything it does -- per-step strength, per-band strength, per-depth
multiplier, cond/uncond balance, spatial masking -- is family-agnostic EXCEPT
three things that are not:

  1. the tensor's rank (UNet: ``[B, C, H, W]``; DiT: ``[B, tokens, dim]``);
  2. how a painted ``[B, 1, H, W]`` mask is projected onto it;
  3. what "depth" means for injection site *i*.

Those three are exactly what a layout answers, and they are the ONLY thing a
new model family has to supply to reuse the whole engine. Adding Flux, Qwen or
Krea 2 support means picking a layout here (or adding one), not touching
``weighting.py`` and certainly not touching ``cnpro_core``.

THE CONTRACT
------------
``depth_fraction(key, index, count)``  -> 0..1, 0 = shallowest, 1 = deepest
``band(key, index, count)``            -> 'coarse' | 'mid' | 'fine'
``project_mask(signal, mask, cache)``  -> tensor broadcastable against `signal`
``broadcast_weight(signal, weight)``   -> per-batch-row weight, same

Two invariants every layout must hold, because the UI promises them:

  * an all-ones mask is EXACTLY a no-op;
  * an all-zeros mask zeroes the injection completely.

Both are checked by ``tests/test_residual_layout.py``.
"""

from __future__ import annotations

import torch

from cnpro_core.weight_profile import (
    band_of_depth_fraction,
    depth_fraction_of_ordered_site,
    depth_fraction_of_residual,
)


def _cached(cache, key, build):
    """Memoize `build()` under `key` in `cache`, tolerating cache=None.

    Masks are constant while sampling, so every projection below only needs to
    run once per (mask, target geometry) pair rather than once per layer per
    step per cond/uncond pass. The cache is owned by the per-run control object,
    so it cannot leak across generations.

    Mask tensors are used as dict keys BY IDENTITY (torch tensors hash by id,
    and holding the key keeps the tensor alive, so a freed mask's address can
    never be recycled into a false hit for a different mask).
    """
    if cache is None:
        return build()
    hit = cache.get(key)
    if hit is None:
        hit = cache[key] = build()
    return hit


class UNetResidualLayout:
    """Residuals emitted per skip connection: ``[B, C, H, W]``.

    Depth and band come from ``depth_fraction_of_residual`` / the ``band_of``
    ordinal quantizer, which encode the U's shape: an ascending 'output' index
    comes back UP the U while 'input' residuals are stored back-to-front, and
    'middle' is always the bottom. That asymmetry is UNet-specific and is the
    reason this is a layout rather than a hardcoded rule.
    """

    #: Residual tensors are 4D; a 2D painted mask maps onto (H, W) directly.
    rank = 4

    def depth_fraction(self, key, index, count):
        return depth_fraction_of_residual(key, index, count)

    def band(self, key, index, count):
        # The historical ordinal quantizer, kept verbatim: it is what every
        # existing SD1.x/SDXL profile was drawn against. See
        # cnpro_core.weight_profile.band_of_depth_fraction for why the two
        # quantizers are deliberately NOT unified.
        if key == 'middle':
            return 'coarse'
        idx = min(index * 3 // max(count, 1), 2)
        if key == 'input':
            idx = 2 - idx
        return ('fine', 'mid', 'coarse')[idx]

    def project_mask(self, signal, mask, cache=None):
        H, W = signal.shape[2], signal.shape[3]
        key = (id(self), mask, signal.shape[0], H, W, signal.dtype, signal.device)

        def build():
            m = mask
            if m.shape[0] != 1:
                k = int(signal.shape[0] // m.shape[0])
                if signal.shape[0] == k * m.shape[0]:
                    m = m.repeat(k, 1, 1, 1)
            # antialias: at the deep sites this is an 8..64x downscale, and
            # plain bilinear samples a 2x2 neighbourhood per output pixel - a
            # stroke a few pixels wide could contribute NOTHING to the 8x8
            # coarse-band masks, so restrict-to-painted silently under-delivered
            # exactly where composition is decided. With antialias each output
            # pixel is the stroke's true coverage fraction. Both engine
            # invariants survive exactly (all-ones stays all-ones, all-zeros
            # stays all-zeros); test_residual_layout.py's pins moved with this,
            # deliberately. Projected in float32 because the antialiased kernel
            # is not implemented for every half dtype/device pair, then cast to
            # the signal's dtype - one cast, same as before.
            return torch.nn.functional.interpolate(
                m.to(device=signal.device, dtype=torch.float32), size=(H, W),
                mode='bilinear', antialias=True).to(signal.dtype)

        return _cached(cache, key, build)

    def broadcast_weight(self, signal, weight):
        return weight[:, None, None, None]


class TokenResidualLayout:
    """Residuals emitted per transformer block: ``[B, sequence, dim]``.

    The sequence is not an image. On this host's Z-Image implementation it is
    ``[caption tokens][image tokens][image padding]``, and only the middle span
    has a spatial meaning, so a painted mask has to be projected onto that span
    and given a defensible value everywhere else (see `project_mask`).

    Depth is the injection block's position in the model's block stack, NOT its
    ordinal among the injection sites -- see
    ``cnpro_core.weight_profile.depth_fraction_of_ordered_site``. Band is that
    same fraction quantized, so the two agree by construction.

    Constructed per sampling run (the token grid depends on the resolution), and
    cheap to build: it holds four ints and a list.
    """

    #: Residual tensors are 3D; a 2D painted mask has to be flattened into the
    #: image span of the sequence.
    rank = 3

    def __init__(self, token_grid, prefix_len, places, block_count):
        #: (h_tokens, w_tokens) of the image span, row-major -- the order
        #: `patchify_and_embed` flattens the latent in.
        self.token_grid = tuple(token_grid)
        #: how many sequence positions precede the image span (caption tokens).
        self.prefix_len = int(prefix_len)
        #: block index in the BASE model that each injection site targets.
        self.places = list(places)
        #: the base model's total block count -- the denominator of the depth axis.
        self.block_count = int(block_count)

    # --- depth axis --------------------------------------------------------

    def _fraction(self, index):
        if 0 <= index < len(self.places):
            return depth_fraction_of_ordered_site(self.places[index], self.block_count)
        return 0.5

    def depth_fraction(self, key, index, count):
        return self._fraction(index)

    def band(self, key, index, count):
        return band_of_depth_fraction(self._fraction(index))

    # --- spatial masking ---------------------------------------------------

    def project_mask(self, signal, mask, cache=None):
        """A painted ``[B, 1, H, W]`` mask as a ``[B, sequence, 1]`` multiplier.

        Image tokens take the mask resized to the token grid. Caption tokens and
        padding tokens take the mask's spatial MEAN.

        The mean is a deliberate choice, not a fallback. Caption-token residuals
        are not spatial -- they steer the whole image through attention -- so
        leaving them at 1.0 would let control escape the painted region entirely
        and quietly break the restrict-to-painted promise the UI makes. Zeroing
        them instead would make a small painted region behave differently in kind
        from a large one. The mean is continuous between those, and it is the only
        choice that keeps BOTH engine invariants exact: an all-ones mask has mean
        1 (a true no-op) and an all-zeros mask has mean 0 (fully off).
        """
        seq = signal.shape[1]
        h, w = self.token_grid
        key = (id(self), mask, signal.shape[0], seq, signal.dtype, signal.device)

        def build():
            m = mask
            if m.shape[0] != 1:
                k = int(signal.shape[0] // m.shape[0])
                if signal.shape[0] == k * m.shape[0]:
                    m = m.repeat(k, 1, 1, 1)
            # antialias for the same reason as UNetResidualLayout.project_mask:
            # a token grid is a 16x+ downscale of the painted mask, and without
            # it a thin stroke can vanish entirely. Everything in float32 (the
            # antialiased kernel is not implemented for every half dtype, and
            # the spatial mean is more exact there), one cast at the end -
            # invariants (all-ones/all-zeros) survive exactly either way.
            m = m.to(device=signal.device, dtype=torch.float32)
            grid = torch.nn.functional.interpolate(
                m, size=(h, w), mode='bilinear', antialias=True)
            flat = grid.flatten(2).movedim(1, 2)                     # [B, h*w, 1]
            outside = m.mean(dim=(1, 2, 3)).reshape(-1, 1, 1)        # [B, 1, 1]
            out = outside.expand(flat.shape[0], seq, 1).clone()
            start = self.prefix_len
            end = min(start + flat.shape[1], seq)
            if end > start:
                out[:, start:end, :] = flat[:, : end - start, :]
            return out.to(signal.dtype)

        return _cached(cache, key, build)

    def broadcast_weight(self, signal, weight):
        return weight[:, None, None]


#: The default for every UNet-shaped patcher. Stateless, so one shared instance.
UNET_LAYOUT = UNetResidualLayout()
