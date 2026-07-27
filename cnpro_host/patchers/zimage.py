"""CNPro's Z-Image ControlNet patcher (Fun-ControlNet-Union).

Same shape as ``controlnet.py``: sniff a state dict, build CNPro's own control
object, and let the profile machinery reach it by ordinary composition. Two
things differ, and both are properties of the MODEL, not choices:

  * the hint is a LATENT, not pixels. UNet ControlNets learn their own hint
    encoder (``input_hint_block``); Z-Image's takes the VAE's latent directly,
    so the preprocessor's output has to be encoded before it can be used. That
    happens once per run in ``process_before_every_sampling``, not once per step.

  * the residuals are injected by module hooks rather than by returning them,
    because this host's DiT ignores its ``control`` argument. See
    ``zimage_impl`` for why that failure mode is worth this much machinery.

CAPABILITY FLAGS -- what is honestly true here
----------------------------------------------
balance    NO.  Z-Image Turbo is few-step distilled and runs at CFG 1: there is
                no uncond row to bias, so a balance curve would be a control
                that does nothing. Reported as unavailable rather than neutral.
band/depth YES. Six injection sites at blocks [0,5,10,15,20,25] of 30, each with
                a real position on the model's depth axis. The BAND LABELS mean
                something different than on a UNet - see below.
mask       YES. Tokens are a 2D latent grid, so a painted mask projects onto them.
prompt     YES. The control tower attends over the caption tokens, so a per-unit
                prompt genuinely re-conditions it... but CNPro does not wire it
                up yet: the unit-prompt path swaps ``c_crossattn`` on a control
                model that takes its caption from the base model's already-refined
                sequence, and re-refining a different caption is a second feature,
                not a flag. Declared False so the UI warns instead of lying.

BAND SEMANTICS: on a UNet, coarse/mid/fine are RESOLUTION tiers because the
encoder downsamples - coarse really is composition and fine really is texture. A
DiT runs every block at one token resolution, so this axis measures ABSTRACTION
instead. The curves work and stay monotone in depth; the three labels are a
UNet-era name for the axis. Kept rather than renamed so the UI, the infotext keys
and every saved profile stay identical across model families - which is the point
of the exercise. Logged once per run so the meaning is never a surprise.
"""

from __future__ import annotations

import logging

import torch

from ..adapter import diffusion_model as host_diffusion_model
from ..adapter import model_family
from . import zimage_impl as impl
from .base import CNProModelPatcher

logger = logging.getLogger("CNPro")


class ZImageControlNetPatcher(CNProModelPatcher):
    # See the module docstring for why each of these is what it is.
    supports_balance_profile = False
    supports_output_mask = True
    supports_band_profiles = True
    supports_unit_prompt = False
    # Per-layer masks arrive through `region_masks` (the weighting
    # engine applies them per injection site), not through the `mask` argument.
    masks_via_advanced_weighting = True

    #: Architectures this patcher can inject into. The UI never gates on model
    #: type; this gates on the MODEL LOADED, which is a different question and
    #: the only one that can be answered correctly.
    families = frozenset({"zimage"})

    def __init__(self, model_patcher):
        super().__init__(model_patcher)

    # --- loading -----------------------------------------------------------

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        """Recognise a Z-Image ControlNet, or return None to try the next type.

        `control_layers.0.after_proj.weight` is the discriminator: it exists in
        no other control family CNPro loads, so the sniff cannot false-positive
        on a UNet ControlNet, a ControlLora, a T2I adapter or an IP-Adapter.
        """
        if "control_layers.0.after_proj.weight" not in state_dict:
            return None
        if "control_all_x_embedder.2-1.weight" not in state_dict:
            return None

        model, config, load_device, cast_dtype = impl.build_control_model(state_dict, ckpt_path)
        control = impl.ZImageControlNet(model, config, load_device=load_device,
                                        manual_cast_dtype=cast_dtype)
        logger.info(
            "CNPro: loaded Z-Image ControlNet %s (%d control layers, %d %s refiner "
            "blocks, dim %d, %d-channel conditioning)",
            config["variant"], config["n_control_layers"], config["n_refiner_layers"],
            "control" if config["refiner_is_control"] else "plain",
            config["dim"], config["control_in_dim"])
        if config["variant"] == "v2":
            logger.info(
                "CNPro: v2.1 conditioning is 33-channel [control latent 16 | mask 1 "
                "| known pixels 16]; CNPro fills the last 17 with zeros, which is "
                "what upstream's non-inpaint pipeline does and means 'nothing is "
                "already decided'. Inpainting through the ControlNet is not wired.")
        return ZImageControlNetPatcher(control)

    # --- per run -----------------------------------------------------------

    def process_before_every_sampling(self, process, cond, mask, *args, control_type=None, **kwargs):
        family = model_family(process)
        if family not in self.families and family != "unknown":
            # Refuse loudly. The alternative is a shape mismatch several layers
            # deep into someone else's model, which reads as a CNPro crash.
            raise RuntimeError(
                f"CNPro: this is a Z-Image ControlNet but the loaded checkpoint is "
                f"'{family}'. Z-Image ControlNets only work on Z-Image models; "
                f"SD1.5/SDXL ControlNets only work on SD1.5/SDXL. Nothing transfers "
                f"between UNet and DiT control - they are different mechanisms.")

        model = host_diffusion_model(process)
        if model is None:
            raise RuntimeError("CNPro: cannot reach the diffusion model to inject Z-Image control.")

        if not isinstance(cond, torch.Tensor):
            raise RuntimeError("CNPro: Z-Image control needs an image hint.")

        control = self.model_patcher.copy()
        control.set_cond_hint(cond, self.strength, (self.start_percent, self.end_percent))
        control.set_control_type(control_type)
        # Bind the encoder rather than encoding now: the target resolution is not
        # known until sampling starts, and txt2img + hires fix are two different
        # ones. `_hint_for` resizes in pixel space and re-encodes per pass.
        control.encode_hint = lambda pixels, p=process: self._encode_hint(p, pixels)

        # v2.1's hole channels, from the HOST's own inpaint context.
        #
        # Deliberately not from CNPro's painted weight masks, and not from the
        # unit's Use-Mask: those mean "where does control apply", which is a
        # different question from "which pixels are already decided". img2img's
        # `image_mask` + `init_images` already mean exactly the second one, with
        # semantics the host defines and the user already understands, so there is
        # no new convention to invent and nothing extra to paint.
        #
        # Absent (txt2img, or img2img without a mask) -> channels 16..32 stay
        # zero, which is plain structural control.
        if self.model_patcher.config.get('control_in_dim') == 33:
            control.hole_mask, control.known_pixels = self._inpaint_context(process)

        control.cond_layer_weights = self.cond_layer_weights
        control.uncond_layer_weights = self.uncond_layer_weights
        control.frame_weights = self.frame_weights
        control.sigma_weight_fn = self.sigma_weight_fn
        control.region_masks = self.region_masks
        control.weight_profile = self.weight_profile
        control.depth_profile = self.depth_profile
        control.drift_profile = self.drift_profile
        if self.band_weight_profiles is not None:
            assert all(b in ('coarse', 'mid', 'fine') for b in self.band_weight_profiles)
            control.band_weight_profiles = self.band_weight_profiles

        if self.balance_profile:
            logger.warning(
                "CNPro: balance profile ignored on Z-Image - the model is "
                "guidance-distilled (CFG 1), so there is no uncond row to bias.")
        if self.band_weight_profiles or self.depth_profile:
            logger.info(
                "CNPro: Z-Image depth axis = block position in the 30-block stack "
                "(injection at %s). Note that coarse/mid/fine mean ABSTRACTION here, "
                "not resolution - a DiT runs every block at one token resolution.",
                impl.places_for(self.model_patcher.config['n_control_layers'],
                                len(getattr(model, 'layers', []))))

        cfg = self.model_patcher.config
        places = impl.places_for(cfg['n_control_layers'], len(model.layers))
        # v2 also injects into the base model's own noise_refiner, before the
        # block stack; v1 has no such stage and gets an empty list.
        refiner_places = (impl.refiner_places_for(cfg['n_refiner_layers'],
                                                  len(getattr(model, 'noise_refiner', [])))
                          if cfg.get('variant') == 'v2' else [])
        impl.Injector.install(model, places, refiner_places)

        unet = process.sd_model.forge_objects.unet.clone()
        unet.add_patched_controlnet(control)
        process.sd_model.forge_objects.unet = unet

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        # Hooks must not outlive the run: they hold a reference to the control
        # chain (and through it to the hint tensors), and a stale injector on a
        # model the user then samples WITHOUT CNPro would be exactly the
        # "enabling CNPro changed behaviour for someone else" failure the whole
        # extension is built to avoid.
        try:
            impl.Injector.uninstall(host_diffusion_model(process))
        except Exception:
            logger.exception("CNPro: failed to remove Z-Image injection hooks")
        super().process_after_every_sampling(process, params, *args, **kwargs)

    # --- hint --------------------------------------------------------------

    @staticmethod
    def _inpaint_context(process):
        """(hole_mask, known_pixels) from img2img, or (None, None).

        `hole_mask` is [1,1,H,W] with **1 = REGENERATE**, i.e. the UI's own
        convention: white paint marks what you want replaced. The inversion the
        model wants (1 = keep) happens once, in `_hole_channels`, so there is
        exactly one place that can have the polarity wrong and it is tested.

        `p.inpainting_mask_invert` is honoured because the host honours it: the
        user can flip the meaning of their own mask in the img2img UI, and a
        ControlNet that ignored that flag would inpaint the complement of what
        the rest of the pipeline does.
        """
        try:
            import numpy as np
            from PIL import Image
        except Exception:
            return None, None

        mask_img = getattr(process, "image_mask", None)
        init = getattr(process, "init_images", None) or [None]
        init_img = init[0] if init else None
        if mask_img is None or init_img is None:
            return None, None

        try:
            if getattr(process, "inpainting_mask_invert", 0):
                mask_img = Image.eval(mask_img, lambda v: 255 - v)

            m = np.asarray(mask_img.convert("L"), dtype=np.float32) / 255.0
            k = np.asarray(init_img.convert("RGB"), dtype=np.float32) / 255.0
            if m.shape[:2] != k.shape[:2]:
                mask_img = mask_img.convert("L").resize(init_img.size, Image.NEAREST)
                m = np.asarray(mask_img, dtype=np.float32) / 255.0

            hole = torch.from_numpy(m)[None, None]          # [1,1,H,W], 1 = repaint
            known = torch.from_numpy(k).permute(2, 0, 1)[None]   # [1,3,H,W] in [0,1]
            logger.info("CNPro Z-Image v2: inpaint context found (%dx%d, %.1f%% of "
                        "the image marked for regeneration).",
                        hole.shape[-1], hole.shape[-2], 100.0 * float(m.mean()))
            return hole, known
        except Exception:
            logger.exception("CNPro Z-Image v2: could not read the inpaint context; "
                             "falling back to plain structural control")
            return None, None

    @staticmethod
    def _encode_hint(process, pixels):
        """VAE-encode a control image into Z-Image's latent space.

        Done through the engine's own ``encode_first_stage`` rather than by
        calling the VAE directly: that method carries the model's normalisation
        (``process_in``, and the [-1,1] -> [0,1] shift the Z-Image engine applies
        before the VAE). A control latent that is off by that transform still
        produces an image - just a subtly wrong one, which is the hardest kind of
        bug to notice.

        `pixels` is [B,3,H,W] in [0,1], the host's convention throughout this
        extension (``numpy_to_pytorch`` divides by 255); ``encode_first_stage``
        wants [-1,1], hence the shift.

        Called from `_hint_for`, once per sampling PASS and after the pixel-space
        resize - never per step.
        """
        with torch.no_grad():
            return process.sd_model.encode_first_stage(
                pixels.to(dtype=torch.float32) * 2.0 - 1.0)
