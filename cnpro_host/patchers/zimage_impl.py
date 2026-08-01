"""Z-Image (Fun-ControlNet-Union) engine - CNPro vendored copy (L2, host-bound).

Provenance: ``diffusers.models.controlnets.controlnet_z_image``
(``ZImageControlNetModel`` / ``ZImageControlTransformerBlock``), re-expressed
against this host's tensors. The upstream config lives in VideoX-Fun's
``config/z_image/z_image_control.yaml``:

    control_layers_places: [0, 5, 10, 15, 20, 25]
    control_in_dim: 16

Weights: ``alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union`` (single 3.1 GB
safetensors, 136 tensors, Apache-2.0). The parameter NAMES below are the
checkpoint's own (``attention.to_q`` / ``norm_q`` / ``to_out.0``, i.e. diffusers
spelling) rather than this host's NextDiT spelling (``attention.qkv`` /
``q_norm`` / ``out``), so the file loads with no key remapping at all. That is
deliberate: a remap table is a second place to be wrong, and a wrong remap loads
silently and produces plausible-but-incorrect images.


WHY THIS INJECTS THROUGH MODULE HOOKS
-------------------------------------
Every other CNPro patcher returns residuals from ``get_control`` and lets the
host hand them to the model as ``control=``. That does not work here, and the
way it fails is the dangerous kind:

``backend/nn/lumina.py::NextDiT.forward`` has no ``control`` parameter. It has
``**kwargs``. So the host computes residuals, passes them in, and the DiT
SILENTLY DISCARDS them -- no exception, no warning, just a ControlNet that does
nothing while appearing to work. (``backend/nn/flux.py`` and ``qwen.py`` do
consume ``control``; ``lumina.py`` and ``krea.py`` do not.)

So the residuals are added by ``torch.nn.Module`` forward pre-hooks on the
DiT's own blocks. Hooks are installed on the live module INSTANCE and removed
after sampling -- no host file is edited, no host class is patched, and a run
without CNPro is bit-identical to one where CNPro was never imported.

``install()`` verifies the module actually looks like the NextDiT it expects and
raises if not, so a host refactor fails LOUDLY at patch time rather than quietly
at sampling time. That is the same trade the vendored ControlNet engine makes.


WHY THE CONTROL BRANCH RUNS INSIDE THE FORWARD
----------------------------------------------
The upstream ControlNet shares eight modules with the transformer (``t_embedder``,
``cap_embedder``, ``rope_embedder``, ``noise_refiner``, ``context_refiner``,
``x_pad_token``, ``cap_pad_token``, ``all_x_embedder``) and re-runs the refiners
itself, so diffusers pays for them twice per step. Running from inside the
forward, at the boundary of ``layers[0]``, the refined sequence, the RoPE
frequencies and the adaLN vector are all just... there, as hook arguments. They
are taken from the base model rather than recomputed, which is both cheaper and
strictly more correct: there is no second copy of the timestep convention to
drift out of sync (the host applies ``1 - timesteps`` before ``t_embedder``,
diffusers does not).


TOKEN ORDER
-----------
Upstream builds the joint sequence as ``[image][caption]``; this host builds it
as ``[caption][image]`` (lumina.py: ``cat((cap_feats, x))``). Both are correct --
attention is permutation-equivariant and the RoPE ids carry the real positions --
but the residual for token *j* must land on token *j* OF THE HOST'S ORDER. Rather
than permute upstream's output, everything here is computed in the host's order
from the host's own tensors, so no reordering step exists to get wrong. The one
consequence is that the image span starts at ``prefix_len`` (the padded caption
length), which is what ``TokenResidualLayout`` is told.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from backend import memory_management
from backend.misc import image_resize
from backend.attention import attention_function
from backend.operations import using_forge_operations
from backend.patcher.base import ModelPatcher
# The SAME padding the host applies to its own latent before patchify
# (backend/nn/lumina.py::NextDiT.forward). Imported rather than re-derived so the
# control grid cannot disagree with the base grid about what an odd dimension
# rounds to -- see _hint_for.
from backend.utils import pad_to_patch_size

from ..adapter import computation_dtype as host_computation_dtype
from .controlnet_impl import ControlBase, broadcast_image_to
from .residual_layout import TokenResidualLayout
from .weighting import compute_controlnet_weighting
# Config sniffing and the injection-site table live next door in `zimage_config`
# (zero imports, so they are testable against the real checkpoint's shapes with
# no torch, no host and no 3.1 GB download). Re-exported here because callers
# reasonably expect the engine module to own them.
from .zimage_config import (  # noqa: F401  (re-exports)
    ADALN_EMBED_DIM,
    UnsupportedZImageControlNet,
    config_from_state_dict,
    places_for,
    refiner_places_for,
)

logger = logging.getLogger("CNPro")


def _resolve_apply_rope():
    """The RoPE helper the base model's own JointAttention uses, wherever it lives.

    The control branch MUST apply position encoding exactly the way the branch it
    steers does, or the residuals land on rotated-differently tokens and the model
    degrades in a way that looks like a bad ControlNet rather than a bug. So this
    resolves the host's helper instead of vendoring a copy - a vendored RoPE
    cannot be wrong on the day it is written and cannot stay right afterwards.

    It has moved once already: Forge Neo lifted `apply_rope` out of
    `backend.nn.flux` into the comfy-kitchen dispatcher (`backend.quant_ops.ck`),
    which is what `backend/nn/lumina.py` - the DiT this patcher hooks - now calls.
    Both spellings take `(xq, xk, freqs_cis)` and return `(q, k)`, so the newest
    home is tried first and the older one is the fallback. A host that has
    neither gets a diagnosis naming both, not an AttributeError three frames deep.
    """
    try:
        from backend.quant_ops import ck
        return ck.apply_rope
    except (ImportError, AttributeError):
        pass
    try:
        from backend.nn.flux import apply_rope as legacy
        return legacy
    except (ImportError, AttributeError):
        pass
    raise ImportError(
        "CNPro Z-Image: this host exposes no RoPE helper where CNPro knows to "
        "look - neither backend.quant_ops.ck.apply_rope (Forge Neo) nor "
        "backend.nn.flux.apply_rope (older Forge). The Z-Image ControlNet needs "
        "the host's own helper to stay in step with the base DiT, so it is "
        "unavailable on this host; other control model families are unaffected. "
        "Check what backend/nn/lumina.py calls and add that spelling here.")


#: Resolved once at import. See _resolve_apply_rope for why it is not vendored.
apply_rope = _resolve_apply_rope()


def _inpaint_enabled():
    """Is v2.1's hole conditioning switched on? Default NO, and deliberately so.

    THE HOLE PATH IS PLUMBED BUT DOES NOT WORK. Measured against the real 2.1
    checkpoint on a real img2img inpaint: the mask and known-pixel channels reach
    the model (verified by log and by a 95-107/255 change inside the hole versus
    no ControlNet), but the model REPRODUCES THE BLACKED-OUT VOID instead of
    filling it - the masked region comes back solid black. That happens with BOTH
    mask polarities, so it is not the sign error the code was written to avoid.

    Leads for whoever picks this up, in the order worth trying:
      * the base model gets no inpainting conditioning of its own here. Upstream's
        inpaint PIPELINE also seeds `latents` from the init image and re-applies
        the mask every step; the host's img2img does its own compositing instead, so
        the ControlNet may be the only thing being told "there is a hole" and it
        answers by drawing one.
      * `denoising_strength` / `inpaint_full_res` interact with the host's own
        masked blending; try full-res inpainting and < 1.0 strength.
      * check the VAE normalisation of the known-pixel half against upstream's
        `(latents - shift_factor) * scaling_factor`, which is not obviously the
        same as this host's `process_in`.

    Off by default so the verified-good behaviour (zero padding = plain
    structural control, which measurably works) is what users get. Set
    CNPRO_ZIMAGE_INPAINT=1 to experiment.
    """
    import os
    return os.environ.get("CNPRO_ZIMAGE_INPAINT", "").strip() in ("1", "true", "yes")

#: Sequence padding granularity. Upstream calls it SEQ_MULTI_OF; this host calls
#: it `pad_tokens_multiple` and sets it to the same 32 for Z-Image
#: (huggingface_guess/detection.py). Asserted against the live model in install().
SEQ_MULTI_OF = 32


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """SwiGLU, in the checkpoint's w1/w2/w3 spelling."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Self-attention over the joint sequence, with separate q/k/v projections.

    The base model fuses these into one `qkv` Linear; this checkpoint ships them
    split, so they stay split. `attention_function` is the host's dispatcher, so
    CNPro automatically follows whatever backend (sage/flash/sdpa/xformers) the
    user configured for the base model.
    """

    def __init__(self, dim, heads, qk_norm=True):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        if qk_norm:
            self.norm_q = nn.RMSNorm(self.head_dim, elementwise_affine=True)
            self.norm_k = nn.RMSNorm(self.head_dim, elementwise_affine=True)
        else:
            self.norm_q = self.norm_k = nn.Identity()

    def forward(self, x, freqs_cis, transformer_options={}):
        b, s, _ = x.shape
        q = self.to_q(x).view(b, s, self.heads, self.head_dim)
        k = self.to_k(x).view(b, s, self.heads, self.head_dim)
        v = self.to_v(x).view(b, s, self.heads, self.head_dim)
        q = self.norm_q(q)
        k = self.norm_k(k)
        # Same RoPE helper the base model's JointAttention uses on its non-fused
        # path, fed the base model's own freqs_cis - so the control branch and the
        # branch it steers cannot disagree about position encoding.
        q, k = apply_rope(q, k, freqs_cis)
        out = attention_function(
            q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2), self.heads,
            None, skip_reshape=True, transformer_options=transformer_options)
        return self.to_out[0](out)


class ZImageBlock(nn.Module):
    """One Z-Image transformer block, optionally with the control projections.

    `before_proj` exists only on control_layers[0] (it is what lifts the control
    latent into the residual stream); `after_proj` exists on every control layer
    and is the zero-initialised readout whose output becomes the hint. Blocks
    built with `control=False` are plain refiner blocks and have neither.
    """

    def __init__(self, dim, heads, hidden_dim, norm_eps=1e-5, qk_norm=True,
                 control=False, first=False):
        super().__init__()
        self.attention = Attention(dim, heads, qk_norm)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        # `min(dim, 256)` is upstream's rule, not a constant 256: the released
        # model has dim 3840 so the two coincide, but writing the rule rather
        # than its value is what lets a small model be built for tests - and it
        # is what upstream would do to a future smaller Z-Image.
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))
        self.is_control = control
        if control:
            self.after_proj = nn.Linear(dim, dim)
            if first:
                self.before_proj = nn.Linear(dim, dim)

    def _body(self, c, freqs_cis, adaln_input, transformer_options):
        scale_msa, gate_msa, scale_mlp, gate_mlp = \
            self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        attn = self.attention(self.attention_norm1(c) * scale_msa, freqs_cis, transformer_options)
        c = c + gate_msa * self.attention_norm2(attn)
        c = c + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(c) * scale_mlp))
        return c

    def forward(self, c, freqs_cis, adaln_input, transformer_options={}, x=None):
        # control_layers[0] seeds the control stream from the base model's own
        # sequence: `before_proj(c) + x`. before_proj is zero-initialised in the
        # original training, so an untrained ControlNet starts as a no-op.
        if x is not None:
            c = self.before_proj(c) + x
        c = self._body(c, freqs_cis, adaln_input, transformer_options)
        if self.is_control:
            return c, self.after_proj(c)
        return c, None


class ZImageControlNetModel(nn.Module):
    """The Fun-ControlNet-Union tower: 2 refiner blocks + N control blocks.

    Holds no copy of the base model. Everything it needs from the transformer
    (embedded caption, refined sequence, RoPE frequencies, adaLN vector, pad
    token) is passed in by the hook that calls it.
    """

    def __init__(self, dim, heads, hidden_dim, control_in_dim, patch_size,
                 n_control_layers, n_refiner_layers, norm_eps=1e-5, qk_norm=True,
                 variant="v1", refiner_is_control=False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.control_in_dim = control_in_dim
        self.variant = variant
        self.refiner_is_control = refiner_is_control
        self.control_all_x_embedder = nn.ModuleDict({
            f"{patch_size}-1": nn.Linear(patch_size * patch_size * control_in_dim, dim, bias=True),
        })
        # v1: plain transformer blocks that only refine the control tokens.
        # v2: control blocks - they refine AND emit hints into the BASE model's
        #     own noise_refiner, which v1 never touches. Block 0 carries
        #     before_proj in both stages (it is the one that seeds the stream).
        self.control_noise_refiner = nn.ModuleList([
            ZImageBlock(dim, heads, hidden_dim, norm_eps, qk_norm,
                        control=refiner_is_control,
                        first=(refiner_is_control and i == 0))
            for i in range(n_refiner_layers)
        ])
        self.control_layers = nn.ModuleList([
            ZImageBlock(dim, heads, hidden_dim, norm_eps, qk_norm, control=True, first=(i == 0))
            for i in range(n_control_layers)
        ])

    @property
    def embedder(self):
        return self.control_all_x_embedder[f"{self.patch_size}-1"]

    # --- shared -----------------------------------------------------------

    def embed_control(self, control_latent, image_len, x_pad_token):
        """Patchify + embed the control latent into `image_len` sequence tokens.

        Patchified exactly as NextDiT.patchify_and_embed does, so token i of the
        control stream is token i of the image span. (Upstream's own patchify
        produces the same (pH, pW, C) feature order; verified against diffusers
        ZImageControlNetModel._patchify_image with f_patch_size=1.)
        """
        p = self.patch_size
        b, c, h, w = control_latent.shape
        h_t, w_t = h // p, w // p
        tokens = control_latent.view(b, c, h_t, p, w_t, p) \
            .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        tokens = self.embedder(tokens.to(self.embedder.weight.dtype))

        if tokens.shape[1] > image_len:
            tokens = tokens[:, :image_len]
        if tokens.shape[1] < image_len:
            # The base model padded its image span to SEQ_MULTI_OF with
            # x_pad_token; the control stream is padded with the same token so
            # the two streams stay aligned position for position.
            pad = x_pad_token.to(device=tokens.device, dtype=tokens.dtype)
            pad = pad.reshape(1, 1, -1).expand(tokens.shape[0], image_len - tokens.shape[1], -1)
            tokens = torch.cat((tokens, pad), dim=1)
        return tokens

    # --- v2 stage 1: the refiner ------------------------------------------

    def refine_v2(self, tokens, x_unrefined, image_freqs, adaln_input, transformer_options={}):
        """Run the v2 control refiner. Returns (hints, stream).

        `x_unrefined` is the base model's image tokens straight out of
        `x_embedder`, BEFORE its own noise_refiner has touched them -- which is
        exactly what arrives at a pre-hook on ``noise_refiner[0]``. Upstream
        seeds the control stream from it (``before_proj(c) + x``) and then hands
        one hint to each base refiner block.

        The hint list is positional: hints[i] is added AFTER base
        ``noise_refiner[i]`` (see refiner_places_for).
        """
        hints = []
        c = tokens
        for i, block in enumerate(self.control_noise_refiner):
            c, hint = block(c, image_freqs, adaln_input, transformer_options,
                            x=x_unrefined if i == 0 else None)
            hints.append(hint)
        return hints, c

    # --- v1 stage 1: the refiner, unchanged -------------------------------

    def refine_v1(self, tokens, image_freqs, adaln_input, transformer_options={}):
        """v1's refiner: plain blocks, no hints, no base-model input.

        Kept as its own method rather than folded into refine_v2 with flags. The
        two are genuinely different passes - v1's blocks take no `x` and produce
        no hints - and v1 is the only variant with a verified module tree, so it
        stays on a code path that a v2 change cannot reach.
        """
        for block in self.control_noise_refiner:
            tokens, _ = block(tokens, image_freqs, adaln_input, transformer_options)
        return tokens

    # --- stage 2: the main control layers (both variants) -----------------

    def emit(self, stream, sequence, prefix_len, freqs_cis, adaln_input,
             transformer_options={}, unified=None):
        """Run control_layers over the joint sequence. Returns one hint each.

        Joined in the HOST's order: caption first, then the control image
        tokens. The caption half is taken from the base model's already-refined
        sequence, which is what upstream does too (it reuses cap_feats after
        context_refiner rather than re-embedding).

        `unified` is what control_layers[0] adds to in ``before_proj(c) + x``.
        For v1 that IS the base model's sequence. For v2 it is the TOWER'S OWN
        refined sequence - which differs, because the refiner hints were added to
        it and never to the base model's (see refine_inside_forward). Defaulting
        to `sequence` keeps v1 exactly as it was.
        """
        if int(stream.shape[0]) != int(sequence.shape[0]):
            # get_control aligns the hint to the sampling batch; if that ever
            # regresses, fail with the actual numbers rather than letting
            # torch.cat raise its anonymous size error from inside a forward
            # hook on the first sampling step.
            raise RuntimeError(
                f"CNPro Z-Image: control stream batch {int(stream.shape[0])} does "
                f"not match the model sequence batch {int(sequence.shape[0])}; "
                f"the hint was not broadcast to the sampling batch.")
        control = torch.cat((sequence[:, :prefix_len], stream), dim=1)
        seed = sequence if unified is None else unified
        hints = []
        for i, block in enumerate(self.control_layers):
            control, hint = block(control, freqs_cis, adaln_input, transformer_options,
                                  x=seed if i == 0 else None)
            hints.append(hint)
        return hints

    def forward(self, control_latent, sequence, prefix_len, freqs_cis, adaln_input,
                x_pad_token, transformer_options={}):
        """v1 convenience: the whole tower in one call, at the layers[0] boundary.

        v1 needs nothing from the base model before its own blocks run, so its
        entire pass fits at a single hook point. v2 cannot use this - its refiner
        stage has to run earlier, against the UNREFINED image tokens, and emit
        hints into the base model's noise_refiner on the way (see
        Injector._on_noise_refiner).

        Returns one hint per control layer, each shaped like `sequence`.
        """
        seq_len = sequence.shape[1]
        image_len = seq_len - prefix_len
        tokens = self.embed_control(control_latent, image_len, x_pad_token)

        # The image span's own frequencies: upstream refines the control tokens
        # before the caption is concatenated, and the pre-concat span here is
        # [prefix_len:].
        image_freqs = freqs_cis[:, prefix_len:] if freqs_cis.shape[1] == seq_len else freqs_cis
        tokens = self.refine_v1(tokens, image_freqs, adaln_input, transformer_options)
        return self.emit(tokens, sequence, prefix_len, freqs_cis, adaln_input,
                         transformer_options)


# ---------------------------------------------------------------------------
# state-dict -> config
# ---------------------------------------------------------------------------



def build_control_model(sd, ckpt_path):
    """Instantiate and load the tower, under the host's cast/quantisation ops."""
    config = config_from_state_dict(sd, ckpt_path)
    unet_dtype = memory_management.unet_dtype()
    load_device = memory_management.get_torch_device()
    computation_dtype = host_computation_dtype() or unet_dtype

    with using_forge_operations(dtype=unet_dtype,
                                manual_cast_enabled=computation_dtype != unet_dtype):
        model = ZImageControlNetModel(**config).to(dtype=unet_dtype)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise UnsupportedZImageControlNet(
            f"{len(missing)} tensors missing from {ckpt_path} (first: {missing[0]}). "
            f"Refusing to run a partially-loaded ControlNet.")
    if unexpected:
        logger.debug("CNPro Z-Image ControlNet: %d unexpected keys ignored", len(unexpected))
    return model, config, load_device, computation_dtype


# ---------------------------------------------------------------------------
# the control object
# ---------------------------------------------------------------------------

class ZImageControlNet(ControlBase):
    """CNPro's control object for Z-Image.

    Reuses ControlBase wholesale -- profile lookups, timestep gating, chaining,
    cleanup -- and overrides only the two things that differ: `get_control`
    prepares the hint instead of producing residuals (they cannot exist until the
    forward is running), and `run_inside_forward` produces and weights them.
    """

    #: single injection group: unlike a UNet there is no input/middle/output
    #: split, just a list of blocks. The layout is what gives its entries a depth.
    GROUP = 'block'

    def __init__(self, control_model, config, load_device=None, manual_cast_dtype=None):
        super().__init__()
        self.control_model = control_model
        self.config = config
        self.load_device = load_device
        self.manual_cast_dtype = manual_cast_dtype
        self.control_model_wrapped = ModelPatcher(
            control_model, load_device=load_device,
            offload_device=memory_management.unet_offload_device())
        #: the VAE-encoded hint at the CURRENT pass's resolution, built lazily by
        #: `_hint_for`. Z-Image's ControlNet conditions on a LATENT, not on pixels
        #: - it has no learned hint encoder of its own the way `input_hint_block`
        #: is one - so the VAE stands in for that encoder.
        self.cond_hint_latent = None
        #: bound by the patcher to the engine's `encode_first_stage`, so the hint
        #: can be re-encoded when the resolution changes (hires fix) without the
        #: control object needing to know what a `process` is.
        self.encode_hint = None
        #: inpaint context for v2's hole channels, set per run by the patcher.
        #: hole_mask: [1,1,H,W] with 1 = REGENERATE (UI convention);
        #: known_pixels: [1,3,H,W] in [0,1], the image outside the hole.
        #: Both None -> channels 16..32 are zeros (plain structural control).
        self.hole_mask = None
        self.known_pixels = None
        #: VAE downscale factor; 8 for Z-Image's Flux-format latents. Corrected
        #: from the first encode if that assumption is ever wrong.
        self.latent_scale = 8
        #: set per step by get_control, consumed by run_inside_forward
        self.pending = None
        #: built by run_inside_forward once the token grid is known, and reused
        #: until the geometry changes (see the `_layout_key` guard there)
        self.residual_layout = None
        self._layout_key = None
        self._logged_hole = False

    # --- host lifecycle ----------------------------------------------------

    def copy(self):
        c = ZImageControlNet(self.control_model, self.config,
                             self.load_device, self.manual_cast_dtype)
        c.control_model_wrapped = self.control_model_wrapped
        c.encode_hint = self.encode_hint
        c.latent_scale = self.latent_scale
        c.hole_mask = self.hole_mask
        c.known_pixels = self.known_pixels
        self.copy_to(c)
        return c

    def get_models(self):
        return super().get_models() + [self.control_model_wrapped]

    def inference_memory_requirements(self, dtype):
        return (memory_management.module_size(self.control_model)
                + super().inference_memory_requirements(dtype))

    def cleanup(self):
        # The hint latent is [B,16,h,w] and the layout's mask cache holds one
        # [B,seq,1] tensor per painted mask: both are per-run and both must go,
        # because patcher instances are cached for the process lifetime.
        self.pending = None
        self.cond_hint_latent = None
        self.encode_hint = None
        # v2's stage-one products are full [B, image_len, dim] activations in
        # the compute dtype (~30 MB apiece at 1024x1024) and were the one
        # per-run tensor pair this method missed: they sat in VRAM between
        # generations for as long as the run's control copy stayed referenced.
        self._stream = None
        self._unified_x = None
        # inpaint context for v2's hole channels, set per run by the patcher.
        # hole_mask: [1,1,H,W] with 1 = REGENERATE (UI convention);
        # known_pixels: [1,3,H,W] in [0,1], the image outside the hole.
        # Both None -> channels 16..32 are zeros (plain structural control).
        self.hole_mask = None
        self.known_pixels = None
        self.residual_layout = None
        self._layout_key = None
        super().cleanup()

    # --- per step ----------------------------------------------------------

    def get_control(self, x_noisy, t, cond, batched_number):
        """Prepare this unit's hint for the forward that is about to run.

        Returns the chain head so the host has something to put in ``c['control']``
        and so ``run_inside_forward`` can walk every unit; the tensors themselves
        cannot be produced here (see the module docstring).

        Safe because the host calls this immediately before exactly one
        ``apply_model`` on the same thread -- sampling_function.py sets
        ``c['control'] = control.get_control(...)`` and then calls
        ``model.apply_model(input_x, timestep_, **c)`` with nothing in between.
        """
        for modifier in self.transformer_options.get('controlnet_conditioning_modifiers', []):
            x_noisy, t, cond, batched_number = modifier(self, x_noisy, t, cond, batched_number)

        if self.previous_controlnet is not None:
            self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        self.pending = None

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return self

        strength = self.strength
        if self.weight_profile_sigmas is not None:
            strength = self.current_profile_strength(t)
            if abs(strength) < 1e-4:
                # (near-)zero skips the control tower entirely, same epsilon and
                # same reasoning as the UNet ControlNet: profiles mapped through
                # a scale range rarely interpolate to exactly 0.0.
                return self

        hint = self._hint_for(x_noisy)
        if hint is None:
            return self
        if x_noisy.shape[0] != hint.shape[0]:
            hint = broadcast_image_to(hint, x_noisy.shape[0], batched_number)
            if hint.shape[0] == 1 and x_noisy.shape[0] > 1:
                # broadcast_image_to returns batch-1 tensors unchanged, which is
                # right for the UNet path - the residual ADD broadcasts
                # implicitly - and wrong here: this path CONCATENATES the control
                # stream onto the batched sequence (emit), and torch.cat cannot
                # broadcast a batch. Left alone, Batch size >= 2 died on step 1
                # inside the forward hook. emit() guards the invariant loudly.
                hint = hint.repeat(x_noisy.shape[0], 1, 1, 1)

        self.pending = (hint, float(strength))
        return self

    def _hint_for(self, x_noisy):
        """The control latent at this pass's resolution, encoded once and cached.

        Resizing happens in PIXEL space, before the VAE, using the same
        `adaptive_resize(..., 'nearest-exact', 'center')` the UNet ControlNet
        engine uses. Resizing the LATENT instead would be far cheaper and is
        wrong for exactly the models this is for: a canny or depth hint is a thin
        high-frequency structure, and bilinear interpolation of its latent smears
        the edges the ControlNet exists to follow. 'nearest-exact' is likewise not
        an arbitrary choice - it is what keeps a 1px canny line 1px.

        Re-encodes when the target resolution changes, which in practice means
        once per pass: txt2img and its hires-fix pass are two different sizes.
        """
        if self.cond_hint_original is None:
            return None

        want = (int(x_noisy.shape[-2]), int(x_noisy.shape[-1]))
        # The host pads its latent to a patch multiple before patchify
        # (NextDiT.forward: pad_to_patch_size), so on an odd latent dimension
        # the base image grid is CEIL(dim/patch). The hint must land on that
        # padded grid: floored, every control-token row came up one token short
        # and the stream drifted one column per row against the RoPE positions
        # it is rotated with - a silent smear that read as "the ControlNet is
        # weak at this resolution". The hint is encoded at the UNPADDED size
        # (content rows align 1:1 with the base image) and then padded with the
        # host's own function, so the pad rows carry the same circular
        # semantics as the base model's.
        patch = int(self.config.get('patch_size', 2)) or 2
        want_pad = (want[0] + (-want[0]) % patch, want[1] + (-want[1]) % patch)
        cached = self.cond_hint_latent
        if cached is not None and tuple(cached.shape[-2:]) == want_pad:
            return cached.to(device=x_noisy.device)

        encode = getattr(self, 'encode_hint', None)
        if encode is None:
            # No encoder bound (the patcher always binds one; this is the "some
            # future caller forgot" path). Fall back to latent interpolation of
            # whatever was pre-encoded rather than dropping control silently.
            if cached is None:
                return None
            logger.warning("CNPro Z-Image: no VAE encoder bound; falling back to "
                           "latent-space resize, which softens hint edges.")
            return torch.nn.functional.interpolate(
                cached.float().to(x_noisy.device), size=want_pad,
                mode='bilinear', align_corners=False).to(cached.dtype)

        pixels = self.cond_hint_original
        scale = self.latent_scale or 8
        resized = image_resize.adaptive_resize(
            pixels, want[1] * scale, want[0] * scale, 'nearest-exact', 'center')
        latent = encode(resized)

        if tuple(latent.shape[-2:]) != want:
            # The VAE's downscale factor is not what was assumed. Correct it once,
            # remember it, and re-encode - rather than silently interpolating from
            # then on.
            got_h = int(latent.shape[-2])
            if got_h > 0 and (want[0] * scale) % got_h == 0:
                self.latent_scale = (want[0] * scale) // got_h
                logger.info("CNPro Z-Image: VAE downscale factor is %d, not %d; "
                            "re-encoding the hint.", self.latent_scale, scale)
                resized = image_resize.adaptive_resize(
                    pixels, want[1] * self.latent_scale, want[0] * self.latent_scale,
                    'nearest-exact', 'center')
                latent = encode(resized)
            if tuple(latent.shape[-2:]) != want:
                logger.warning(
                    "CNPro Z-Image: encoded hint is %s but the latent is %s; "
                    "falling back to a latent-space resize.",
                    tuple(latent.shape[-2:]), want)
                latent = torch.nn.functional.interpolate(
                    latent.float(), size=want, mode='bilinear', align_corners=False)

        # Cache the PADDED latent, not the raw one. Padding on the way out of this
        # branch only is how v2.1 lost control after the first step: step 1 missed
        # the cache and got 33 channels, every later step HIT the cache and got 16,
        # `embed_control` raised (132x3840 against 4096x64), the refiner stage was
        # skipped, and the unit silently injected nothing for steps 2..N. v1 was
        # immune because padding 16 -> 16 is a no-op, which is exactly why the bug
        # looked like "2.1 is weak" rather than "2.1 is broken".
        # `_pad_channels` is idempotent, so caching the padded form is safe.
        # Spatial padding comes LAST, on the channel-complete tensor, so v2's
        # hole channels get the same circular pad rows as the control latent.
        self.cond_hint_latent = pad_to_patch_size(
            self._pad_channels(latent), (patch, patch))
        return self.cond_hint_latent.to(device=x_noisy.device)

    def _hole_channels(self, want_hw, dtype, device):
        """Channels 16..32 for v2: the inpaint mask and the known pixels.

        Returns None when there is no inpaint context, and the caller zero-pads
        instead - which is what upstream's non-inpaint pipeline does and means
        "nothing is already decided".

        Built to mirror ``pipeline_z_image_controlnet_inpaint.py`` exactly:

            mask_condition = interpolate(1 - mask, latent_size, mode="nearest")
            init_image     = init_image * (mask < 0.5)      # BLACK OUT the hole
            init_latent    = vae.encode(init_image)
            control_image  = cat([control, mask_condition, init_latent], dim=1)

        Three details are load-bearing and each is a way to be silently wrong:

        * **Polarity is INVERTED.** ``1 - mask``, so in the tensor 1 = KEEP and
          0 = REGENERATE - the opposite of every UI convention, where the painted
          region is the hole. Getting it backwards inpaints the complement and
          still produces a plausible image.
        * **The hole is blacked out in PIXEL space, before the VAE.** Not a latent
          patch: the model was trained seeing a real black void pushed through
          the encoder.
        * **The mask is downsampled with NEAREST.** Bilinear would produce a grey
          halo, which the model reads as "half-known".
        """
        mask = getattr(self, 'hole_mask', None)
        known = getattr(self, 'known_pixels', None)
        if mask is None or known is None:
            return None

        h, w = want_hw
        # known pixels: black out the hole at FULL resolution, then encode
        px = known.to(dtype=torch.float32)
        m_px = torch.nn.functional.interpolate(
            mask.to(dtype=torch.float32), size=px.shape[-2:], mode='nearest')
        blanked = px * (m_px < 0.5).to(px)

        encode = getattr(self, 'encode_hint', None)
        if encode is None:
            return None
        scale = self.latent_scale or 8
        resized = image_resize.adaptive_resize(
            blanked, w * scale, h * scale, 'nearest-exact', 'center')
        known_latent = encode(resized)
        if tuple(known_latent.shape[-2:]) != (h, w):
            known_latent = torch.nn.functional.interpolate(
                known_latent.float(), size=(h, w), mode='bilinear', align_corners=False)

        # the mask channel: inverted, nearest, at latent resolution
        keep = torch.nn.functional.interpolate(
            (1.0 - mask.to(dtype=torch.float32)), size=(h, w), mode='nearest')

        out = torch.cat((keep.to(known_latent), known_latent), dim=1)
        return out.to(dtype=dtype, device=device)

    def _pad_channels(self, latent):
        """Widen a 16-channel control latent to whatever the tower expects.

        v2 takes 33 channels: ``[control latent (16)][mask (1)][known pixels (16)]``.
        For plain structural control the last 17 are ZERO, and that is not a
        shortcut - it is what upstream's own non-inpaint pipeline does
        (pipeline_z_image_controlnet.py pads with `torch.zeros`), and it is
        semantically exact: mask 0 = "nothing is already decided", known-pixels 0
        = "and there is nothing to show you". Which is text-to-image with
        structural control.

        The mask channel is 1 = KEEP, 0 = REGENERATE - inverted relative to how
        masks are usually written down. Anything that later fills these channels
        for real inpainting has to pin that polarity with a test; getting it
        backwards inpaints the complement of the region and still looks plausible.
        """
        want = int(self.config.get('control_in_dim', 16))
        have = int(latent.shape[1])
        if have >= want:
            return latent[:, :want]

        extra = None
        if want - have == 17 and _inpaint_enabled():
            # the inpaint shape: 1 mask channel + 16 known-pixel channels
            try:
                extra = self._hole_channels(tuple(latent.shape[-2:]),
                                            latent.dtype, latent.device)
            except Exception:
                logger.exception("CNPro Z-Image v2: could not build the hole "
                                 "channels; falling back to zeros (plain control)")
                extra = None
        if extra is not None and extra.shape[1] == want - have:
            if not self._logged_hole:
                self._logged_hole = True
                logger.info("CNPro Z-Image v2: inpaint context wired - channels "
                            "16..32 carry the mask (1=keep) and the known pixels.")
            return torch.cat((latent, extra.to(latent)), dim=1)

        pad = torch.zeros(latent.shape[0], want - have, *latent.shape[2:],
                          device=latent.device, dtype=latent.dtype)
        return torch.cat((latent, pad), dim=1)

    def refine_inside_forward(self, x_unrefined, image_freqs, adaln_input,
                              x_pad_token, refiner_places, base_noise_refiner,
                              transformer_options):
        """v2 stage one, at the boundary of the base model's noise_refiner[0].

        WHOSE LATENT THE REFINER HINTS MODIFY
        -------------------------------------
        Not the base model's. This is the subtlety that made 2.1 hold the control
        far more weakly than v1 until it was read properly:

        ``transformer_z_image.py`` runs its ``noise_refiner`` with NO hint
        application - the transformer never receives
        ``noise_refiner_block_samples`` at all. Those hints are consumed inside
        the CONTROLNET (``controlnet_z_image.py``), which re-embeds and re-refines
        its own copy of the noisy latent, adding a hint after each refiner block.
        That private result is what becomes ``unified`` for the control_layers
        stage - i.e. the ``x`` in ``before_proj(c) + x``.

        So the hints steer what the control tower SEES, and never the denoised
        latent directly. Injecting them into the base model (which is what CNPro
        did first) both corrupts the sampled latent and starves control_layers[0]
        of the input it was trained with.

        Upstream pays for the refiners twice per step and so does this: the two
        blocks are re-run here on a private copy. Two blocks out of thirty-two is
        a few percent, and the alternative is a different model.

        Returns nothing to inject; it stashes `_stream` (the control stream) and
        `_unified_x` (the tower's private refined image tokens) for
        `run_inside_forward` to finish with.
        """
        prev = self.previous_controlnet
        if prev is not None and hasattr(prev, 'refine_inside_forward'):
            prev.refine_inside_forward(x_unrefined, image_freqs, adaln_input,
                                       x_pad_token, refiner_places,
                                       base_noise_refiner, transformer_options)

        self._stream = None
        self._unified_x = None
        if self.pending is None or self.control_model.variant != 'v2':
            return

        hint_latent, strength = self.pending
        tokens = self.control_model.embed_control(
            hint_latent.to(x_unrefined.dtype), x_unrefined.shape[1], x_pad_token)
        hints, stream = self.control_model.refine_v2(
            tokens, x_unrefined, image_freqs, adaln_input, transformer_options)
        self._stream = stream

        # Weighted by STRENGTH ONLY - no per-layer curve. The depth axis addresses
        # the main control_layers; these two sites sit before the block stack and
        # have no position on it, so a depth or band profile would be inventing a
        # meaning the model does not have. The step curve still applies, because
        # `strength` already carries it (see get_control).
        by_place = {}
        # Same rule as run_inside_forward: zip against THIS unit's refiner
        # table, not the injector's union (see own_places there).
        own_refiner = list(getattr(self, 'refiner_places', None) or refiner_places)
        for place, hint in zip(own_refiner, hints):
            if hint is not None:
                by_place[place] = hint.to(x_unrefined.dtype) * strength

        # The tower's OWN refined image tokens: the base refiner blocks re-run on
        # a private copy, with each hint added after its block. `base_noise_refiner`
        # is the base model's module list - shared weights, separate activations,
        # exactly as upstream shares them.
        x_ctrl = x_unrefined
        for idx, block in enumerate(base_noise_refiner):
            x_ctrl = block(x_ctrl, None, image_freqs, adaln_input,
                           transformer_options=transformer_options)
            hint = by_place.get(idx)
            if hint is not None and hint.shape == x_ctrl.shape:
                x_ctrl = x_ctrl + hint
        self._unified_x = x_ctrl

    def run_inside_forward(self, sequence, prefix_len, freqs_cis, adaln_input,
                           x_pad_token, places, block_count, transformer_options):
        """Produce this unit's weighted hints, summed with any chained units.

        Returns ``{place: tensor}`` in the base model's block numbering, or an
        empty dict when nothing is active this step.
        """
        out = {}
        if self.previous_controlnet is not None and hasattr(self.previous_controlnet, 'run_inside_forward'):
            out = self.previous_controlnet.run_inside_forward(
                sequence, prefix_len, freqs_cis, adaln_input, x_pad_token,
                places, block_count, transformer_options)

        if self.pending is None:
            return out

        hint_latent, strength = self.pending
        h_t = hint_latent.shape[-2] // self.config['patch_size']
        w_t = hint_latent.shape[-1] // self.config['patch_size']

        # This unit's OWN site table, stamped by the patcher at patch time. The
        # `places` argument is the injector's table - the UNION across every
        # chained unit, because its hooks must cover them all. Zipping THIS
        # unit's hints against that union sent a v1 unit's six hints to blocks
        # [0,2,4,...] whenever a unit with a different table installed after
        # it: no shape error, no warning, control simply steered from the wrong
        # depths (and the depth/band axis moved with it). The argument remains
        # as a fallback so a single-unit chain built without the stamp still
        # works.
        own_places = list(getattr(self, 'places', None) or places)

        # Rebuild the layout only when the geometry actually changes (a hires-fix
        # pass, or a prompt whose padded length moved). Rebuilding it every step
        # would be correct but quietly quadratic: `layer_mask_for` keys its cache
        # on the layout's identity, so a fresh object each forward means the mask
        # resize re-runs for every injection site on every step of every pass,
        # AND the cache grows without bound for the length of the run.
        key = ((h_t, w_t), prefix_len, tuple(own_places), block_count)
        if getattr(self, '_layout_key', None) != key:
            self._layout_key = key
            self.residual_layout = TokenResidualLayout(
                token_grid=(h_t, w_t), prefix_len=prefix_len,
                places=own_places, block_count=block_count)
            self._layer_mask_cache = None

        if self.control_model.variant == 'v2':
            # Stage two. The stream was produced by refine_inside_forward at the
            # noise_refiner boundary earlier in this same forward; if that did not
            # run (no v2 refiner hook installed, or the model shape surprised us)
            # there is nothing coherent to emit, and inventing a stream from the
            # unrefined tokens would be a different model, not a fallback.
            stream = getattr(self, '_stream', None)
            if stream is None:
                logger.warning(
                    "CNPro Z-Image v2: the refiner stage did not run this step, so "
                    "the control stream is missing; skipping this unit's injection.")
                return out
            # The tower's private refined sequence, built in stage one. Its
            # caption half comes from the base model (shared context_refiner),
            # its image half is the tower's own.
            x_ctrl = getattr(self, '_unified_x', None)
            unified = None
            if x_ctrl is not None and x_ctrl.shape[1] == sequence.shape[1] - prefix_len:
                unified = torch.cat((sequence[:, :prefix_len], x_ctrl), dim=1)
            hints = self.control_model.emit(
                stream, sequence, prefix_len, freqs_cis, adaln_input,
                transformer_options, unified=unified)
        else:
            hints = self.control_model(
                hint_latent.to(sequence.dtype), sequence, prefix_len, freqs_cis,
                adaln_input, x_pad_token, transformer_options)

        control = {self.GROUP: [h.to(sequence.dtype) * strength for h in hints]}
        control = compute_controlnet_weighting(control, self)

        for place, hint in zip(own_places, control[self.GROUP]):
            if hint is None:
                continue
            out[place] = hint if place not in out else out[place] + hint
        return out


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------

class Injector:
    """Forward hooks that carry the control residuals into a live NextDiT.

    Owned by the diffusion model instance for the duration of a run and removed
    afterwards. One injector serves every CNPro unit: the units chain through
    `previous_controlnet` and `run_inside_forward` walks the chain, so N units
    cost N control towers but one set of hooks.
    """

    ATTR = "_cnpro_zimage_injector"

    def __init__(self, model, places, refiner_places=()):
        self.model = model
        self.places = list(places)
        #: base `noise_refiner` block indices that take v2's refiner-stage hints.
        #: Empty for v1, which has no such stage.
        self.refiner_places = list(refiner_places)
        self.block_count = len(model.layers)
        self.handles = []
        #: per-forward state, all reset by the NextDiT pre-hook
        self.chain = None
        self.prefix_len = None
        self.hints = None
        self.refiner_hints = None
        #: re-entrancy guard. The v2 refiner stage RE-RUNS the base model's own
        #: noise_refiner blocks on a private copy, which calls them through
        #: __call__ and therefore fires this very pre-hook again. Without the
        #: guard that recurses until the stack dies (measured: 194 nested calls
        #: before RecursionError). A flag rather than calling .forward() directly,
        #: so any hook the HOST put on those blocks still runs.
        self._in_refine = False
        #: consecutive forwards seen with no CNPro chain - the self-heal
        #: counter (see _on_model).
        self._orphan_forwards = 0

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def install(cls, model, places, refiner_places=()):
        """Attach to `model`, or extend the injector already attached to it.

        The compatibility check is deliberately strict and deliberately EARLY: a
        host refactor that renames `layers` or changes the block signature must
        stop CNPro here, with a message naming the file, rather than let sampling
        proceed against assumptions that no longer hold. It also runs BEFORE the
        previous injector is removed, so a refused second install cannot leave
        the first unit's hooks torn down.
        """
        places = sorted(set(places))
        refiner_places = sorted(set(refiner_places))
        existing = getattr(model, cls.ATTR, None)
        if existing is not None:
            # Chained units may use DIFFERENT site tables (v1 Union + v2.1 Tile
            # in two units). The injector hooks the UNION of every table - a
            # hooked block no unit feeds is a no-op - and each unit zips its
            # hints against its OWN table inside run_inside_forward. Blindly
            # replacing on mismatch (the previous behaviour) left whichever
            # unit installed LAST deciding the table for the whole chain, and
            # the other unit's hints landed on the wrong blocks with no error.
            places = sorted(set(places) | set(existing.places))
            refiner_places = sorted(set(refiner_places) | set(existing.refiner_places))
            if places == existing.places and refiner_places == existing.refiner_places:
                return existing

        for attr in ("layers", "context_refiner", "x_pad_token"):
            if not hasattr(model, attr):
                raise UnsupportedZImageControlNet(
                    f"the loaded diffusion model has no `{attr}`; CNPro's Z-Image "
                    f"injection is written against backend/nn/lumina.py::NextDiT "
                    f"and that file appears to have changed.")
        if getattr(model, "pad_tokens_multiple", None) != SEQ_MULTI_OF:
            raise UnsupportedZImageControlNet(
                f"expected pad_tokens_multiple={SEQ_MULTI_OF}, got "
                f"{getattr(model, 'pad_tokens_multiple', None)}; the control "
                f"stream would not stay aligned with the image span.")
        if not places:
            raise UnsupportedZImageControlNet(
                "no injection sites resolved for this ControlNet")
        if max(places) >= len(model.layers):
            raise UnsupportedZImageControlNet(
                f"ControlNet injects at block {max(places)} but the loaded model "
                f"has {len(model.layers)} blocks - this ControlNet is for a "
                f"different Z-Image variant.")

        if refiner_places:
            if not hasattr(model, "noise_refiner"):
                raise UnsupportedZImageControlNet(
                    "this ControlNet injects into the model's noise_refiner, but "
                    "the loaded diffusion model has none (backend/nn/lumina.py::"
                    "NextDiT appears to have changed).")
            if max(refiner_places) >= len(model.noise_refiner):
                raise UnsupportedZImageControlNet(
                    f"ControlNet injects at noise_refiner[{max(refiner_places)}] "
                    f"but the model has {len(model.noise_refiner)} refiner blocks.")

        # Only now that the union table validated does the old injector come
        # off: a refused install must leave the previously installed units
        # working, not silently uninjected.
        if existing is not None:
            existing.remove()
        inj = cls(model, places, refiner_places)
        inj._attach()
        setattr(model, cls.ATTR, inj)
        return inj

    def _attach(self):
        self.handles.append(self.model.register_forward_pre_hook(
            self._on_model, with_kwargs=True))
        self.handles.append(self.model.context_refiner[-1].register_forward_hook(
            self._on_context))
        if self.refiner_places:
            # Stage one runs at the FIRST base refiner block, where `x` is still
            # the raw embedded image tokens - which is exactly what upstream seeds
            # the v2 control stream from.
            #
            # A PRE-hook only. There are deliberately NO hooks on the refiner
            # blocks' outputs: the refiner hints do not belong to the base model's
            # latent at all (transformer_z_image.py runs noise_refiner with no hint
            # application). They are added to the TOWER's private copy inside
            # refine_inside_forward. Adding them here as well would corrupt the
            # sampled latent and double-count the hint.
            self.handles.append(self.model.noise_refiner[0].register_forward_pre_hook(
                self._on_noise_refiner, with_kwargs=True))
        # One PRE-hook on the first site to build the hints (it needs the block's
        # input args), then a FORWARD hook per site to add each hint to that
        # block's OUTPUT - which is where upstream adds it.
        self.handles.append(self.model.layers[self.places[0]].register_forward_pre_hook(
            self._on_first_block, with_kwargs=True))
        for order, place in enumerate(self.places):
            self.handles.append(self.model.layers[place].register_forward_hook(
                self._make_block_hook(order)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.chain = self.hints = self.prefix_len = self.refiner_hints = None
        if getattr(self.model, self.ATTR, None) is self:
            try:
                delattr(self.model, self.ATTR)
            except AttributeError:
                pass

    @classmethod
    def uninstall(cls, model):
        inj = getattr(model, cls.ATTR, None) if model is not None else None
        if inj is not None:
            inj.remove()

    # -- hooks -------------------------------------------------------------

    def _on_model(self, module, args, kwargs):
        """Pick the control chain out of the arguments NextDiT is about to ignore.

        The host DOES pass `control=` down to the diffusion model; NextDiT simply
        swallows it in **kwargs. Reading it here means the transport is the host's
        own, with no side channel: whatever `get_control` returned for THIS batch
        is what arrives here, so cond/uncond batching and multi-unit chaining
        cannot desynchronise.
        """
        control = kwargs.get("control")
        self.refiner_hints = None
        if not hasattr(control, "run_inside_forward"):
            # Self-heal. A forward with no CNPro chain is a generation CNPro is
            # not steering, so the hooks have no business staying installed.
            # The normal removal path is process_after_every_sampling; this one
            # exists for the run that DIED mid-sampling (OOM, a raise inside a
            # hook) and never reached it - without it, the leaked hooks held
            # the dead run's chain, hints and towers for the rest of the
            # session. Two consecutive chainless forwards, not one, so a stray
            # auxiliary forward inside a controlled run cannot trip it (the
            # hooks are inert either way - this is hygiene, not behaviour).
            # torch materialises the hook list before iterating, so removing
            # from inside the hook is safe; the guard is for versions where
            # that changes.
            self._orphan_forwards += 1
            if self._orphan_forwards >= 2:
                try:
                    self.remove()
                except Exception:
                    logger.exception(
                        "CNPro Z-Image: failed to self-remove leaked hooks")
            self.chain = None
            self.hints = None
            self.prefix_len = None
            return None
        self._orphan_forwards = 0
        # Duck-typed on the protocol, not on ZImageControlNet, on purpose (the
        # hasattr above). The injector's contract is "give me something with
        # run_inside_forward and I will place what it returns" - which is the
        # whole of what a Flux, Qwen or Krea 2 control chain would need from it
        # too, and what lets this be tested without constructing a 3.1 GB
        # model. A UNet ControlNet's plain dict does not have the attribute, so
        # it is ignored rather than misread.
        self.chain = control
        self.hints = None
        self.prefix_len = None
        return None

    def _on_context(self, module, args, output):
        """Record the padded caption length: the image span starts right after it.

        Taken from the tensor rather than computed from the latent shape on
        purpose - the caption length depends on the prompt, on the padding
        multiple and on whether the host pads at all, and every one of those is
        the host's business, not CNPro's.
        """
        self.prefix_len = int(output.shape[1])
        return None

    def _on_noise_refiner(self, module, args, kwargs):
        """v2 stage one: seed the control stream from the UNREFINED image tokens.

        `noise_refiner[0]` receives `x` straight out of `x_embedder`, which is
        the tensor upstream's ``before_proj(c) + x`` adds to. It also receives the
        IMAGE-ONLY RoPE frequencies (the host passes
        ``freqs_cis[:, cap_len:]`` here), so both inputs the refiner needs arrive
        as hook arguments and neither has to be reconstructed.

        Signature, from backend/nn/lumina.py::NextDiT.patchify_and_embed:
            layer(x, padded_img_mask, freqs_cis[:, cap_len:], t, transformer_options=...)
        """
        if self._in_refine:
            return None
        self.refiner_hints = None
        if self.chain is None or not self.refiner_places:
            return None
        if not hasattr(self.chain, 'refine_inside_forward'):
            return None

        x = args[0] if args else kwargs.get('x')
        freqs = args[2] if len(args) > 2 else kwargs.get('freqs_cis')
        adaln = args[3] if len(args) > 3 else kwargs.get('adaln_input')
        options = kwargs.get('transformer_options', {}) or {}
        if x is None or freqs is None or adaln is None:
            logger.warning("CNPro Z-Image v2: unexpected noise_refiner signature; "
                           "the refiner stage is skipped and control will be "
                           "weaker than intended.")
            return None
        self._in_refine = True
        try:
            self.chain.refine_inside_forward(
                x, freqs, adaln, self.model.x_pad_token, self.refiner_places,
                self.model.noise_refiner, options)
        except Exception:
            logger.exception("CNPro Z-Image v2: the refiner stage failed")
        finally:
            self._in_refine = False
        return None


    def _on_first_block(self, module, args, kwargs):
        """Compute every unit's hints, once, before the block stack runs.

        A PRE-hook purely because the tower needs the block's INPUT arguments -
        the joint sequence, the RoPE frequencies and the adaLN vector. It injects
        nothing; `_make_block_hook` does that, after each block.
        """
        if self.chain is None:
            return None
        if self.hints is None:
            self.hints = self._compute(args, kwargs)
        return None

    def _make_block_hook(self, order):
        """Add this site's hint to the OUTPUT of block `places[order]`.

        AFTER the block, not before. Upstream is explicit about it
        (diffusers transformer_z_image.py):

            unified = layer(unified, ...)
            if layer_idx in controlnet_block_samples:
                unified = unified + controlnet_block_samples[layer_idx]

        This was a forward PRE-hook until it was measured, which put every hint
        one block early. v1 tolerated it - six sites five blocks apart - but v2.1
        has fifteen sites only two blocks apart, so a one-block shift is a ~50%
        positional error, and 2.1 held the control far more weakly than v1 while
        looking superficially plausible. Measured control adherence (edge energy
        on the drawn lines vs off them) went 3.34 -> see AGENTS/ARCHITECTURE.
        """
        def hook(module, args, output):
            if self.hints is None:
                return None
            hint = self.hints.get(self.places[order])
            if hint is None:
                return None
            if hint.shape != output.shape:
                logger.warning(
                    "CNPro Z-Image: hint %s does not match block output %s at block "
                    "%d; skipping this injection site.",
                    tuple(hint.shape), tuple(output.shape), self.places[order])
                return None
            return output + hint.to(output)
        return hook

    def _compute(self, args, kwargs):
        """Run every chained control tower once, at the first injection site.

        `places` starts at 0 upstream (`assert 0 in control_layers_places`), so
        this runs before any block has consumed a hint. The base model's own
        refined sequence, RoPE frequencies and adaLN vector arrive as the hook's
        arguments -- they are never recomputed, which is what keeps the control
        branch exactly in step with the branch it steers.
        """
        sequence = args[0] if args else kwargs.get("x")
        freqs_cis = args[2] if len(args) > 2 else kwargs.get("freqs_cis")
        adaln = args[3] if len(args) > 3 else kwargs.get("adaln_input")
        options = kwargs.get("transformer_options", {}) or {}
        if sequence is None or freqs_cis is None or adaln is None:
            logger.warning("CNPro Z-Image: unexpected block signature; control skipped.")
            return {}
        if self.prefix_len is None:
            logger.warning("CNPro Z-Image: caption length unknown; control skipped.")
            return {}
        return self.chain.run_inside_forward(
            sequence, self.prefix_len, freqs_cis, adaln, self.model.x_pad_token,
            self.places, self.block_count, options)
