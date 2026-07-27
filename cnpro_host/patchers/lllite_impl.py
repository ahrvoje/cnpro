"""ControlLLLite injection with per-step, per-band and per-depth weighting.

CNPro vendored copy (L2, host-bound). Provenance: the author's Forge fork.
Vendored, not monkey-patched: CNPro constructs these itself so no host module
is mutated and non-CNPro users see unchanged behaviour. Host-agnostic profile
math lives in cnpro_core/weight_profile.py - keep it there, that module is what
a ComfyUI port reuses unchanged.
"""
import math
import torch

from cnpro_core.weight_profile import (evaluate_weight_profile, band_of_unet_block,
                                            depth_fraction_of_unet_block)


def extra_options_to_module_prefix(extra_options):
    # extra_options = {'transformer_index': 2, 'block_index': 8, 'original_shape': [2, 4, 128, 128], 'block': ('input', 7), 'n_heads': 20, 'dim_head': 64}

    # block is: [('input', 4), ('input', 5), ('input', 7), ('input', 8), ('middle', 0),
    #   ('output', 0), ('output', 1), ('output', 2), ('output', 3), ('output', 4), ('output', 5)]
    # transformer_index is: [0, 1, 2, 3, 4, 5, 6, 7, 8], for each block
    # block_index is: 0-1 or 0-9, depends on the block
    # input 7 and 8, middle has 10 blocks

    # make module name from extra_options
    block = extra_options["block"]
    block_index = extra_options["block_index"]
    if block[0] == "input":
        module_pfx = f"lllite_unet_input_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    elif block[0] == "middle":
        module_pfx = f"lllite_unet_middle_block_1_transformer_blocks_{block_index}"
    elif block[0] == "output":
        module_pfx = f"lllite_unet_output_blocks_{block[1]}_1_transformer_blocks_{block_index}"
    else:
        raise Exception("invalid block name")
    return module_pfx


def lllite_block_of(module_name):
    """('input'|'middle'|'output', block id) of an LLLite module, or None.

    Module names are built from the UNet address by
    extra_options_to_module_prefix above ('lllite_unet_input_blocks_4_1_...'),
    so the block - and with it the band - is readable straight off the name.
    """
    if "_middle_block_" in module_name:
        return "middle", 0
    for group in ("input", "output"):
        token = f"_{group}_blocks_"
        if token in module_name:
            head = module_name.split(token, 1)[1].split("_", 1)[0]
            if head.isdigit():
                return group, int(head)
    return None


def lllite_module_bands(module_names):
    """Band name per module, resolved from each module's UNet block.

    The thirds are taken over the blocks THIS checkpoint actually patches
    (band_of_unet_block), so a LLLite model that only covers part of the UNet
    still gets a full coarse/mid/fine spread over what it does cover.
    """
    sites = {name: lllite_block_of(name) for name in module_names}
    ids_per_group = {}
    for site in sites.values():
        if site is not None and site[0] != "middle":
            ids_per_group.setdefault(site[0], set()).add(site[1])

    bands = {}
    for name, site in sites.items():
        if site is None:
            # unknown address: 'mid' rather than unscaled - in band mode the
            # bands ARE the strength, so a module left out would keep running
            # at full weight while every other layer follows its profile
            print(f"ControlLLLite: unrecognized module name {name!r}, banding it as 'mid'")
            bands[name] = "mid"
            continue
        group, block_id = site
        bands[name] = band_of_unet_block(group, block_id, ids_per_group.get(group, {block_id}))
    return bands


def lllite_module_depths(module_names):
    """Normalized UNet depth (0 = fine, 1 = coarse) per module.

    The un-quantized twin of lllite_module_bands, over the same block set, so
    a depth profile and a band profile place the same module identically.
    """
    sites = {name: lllite_block_of(name) for name in module_names}
    ids_per_group = {}
    for site in sites.values():
        if site is not None and site[0] != "middle":
            ids_per_group.setdefault(site[0], set()).add(site[1])

    depths = {}
    for name, site in sites.items():
        if site is None:
            depths[name] = 0.5  # unknown address: mid depth, same as banding
            continue
        group, block_id = site
        depths[name] = depth_fraction_of_unet_block(
            group, block_id, ids_per_group.get(group, {block_id}))
    return depths


def load_control_net_lllite_patch(ctrl_sd, cond_image, multiplier, num_steps, start_percent, end_percent, model_dtype, weight_profile=None, band_profiles=None, depth_profile=None):
    # calculate start and end step
    start_step = math.floor(num_steps * start_percent) if start_percent > 0 else 0
    end_step = math.floor(num_steps * end_percent) if end_percent > 0 else num_steps

    # split each weights for each module
    module_weights = {}
    for key, value in ctrl_sd.items():
        fragments = key.split(".")
        module_name = fragments[0]
        weight_name = ".".join(fragments[1:])

        if module_name not in module_weights:
            module_weights[module_name] = {}
        module_weights[module_name][weight_name] = value

    # per-band step profiles: each module is scaled by the profile of the band
    # its own UNet block falls in (one LLLite module per attention projection,
    # so 'per band' is as expressible here as it is for ControlNet residuals)
    module_bands = lllite_module_bands(module_weights.keys()) if band_profiles else {}
    # depth profile: one step-invariant multiplier per module, resolved from
    # the same block mapping the bands use
    module_depth_weights = {}
    if depth_profile:
        module_depth_weights = {
            name: evaluate_weight_profile(depth_profile, depth)
            for name, depth in lllite_module_depths(module_weights.keys()).items()
        }

    # load each module
    modules = {}
    for module_name, weights in module_weights.items():
        # ここの自動判定を何とかしたい
        if "conditioning1.4.weight" in weights:
            depth = 3
        elif weights["conditioning1.2.weight"].shape[-1] == 4:
            depth = 2
        else:
            depth = 1

        module = LLLiteModule(
            name=module_name,
            is_conv2d=weights["down.0.weight"].ndim == 4,
            in_dim=weights["down.0.weight"].shape[1],
            depth=depth,
            cond_emb_dim=weights["conditioning1.0.weight"].shape[0] * 2,
            mlp_dim=weights["down.0.weight"].shape[0],
            multiplier=multiplier,
            num_steps=num_steps,
            start_step=start_step,
            end_step=end_step,
            dtype=model_dtype,
            weight_profile=weight_profile,
            band_profile=(band_profiles or {}).get(module_bands.get(module_name)),
            depth_weight=module_depth_weights.get(module_name, 1.0)
        )
        info = module.load_state_dict(weights)
        modules[module_name] = module
        if len(modules) == 1:
            module.is_first = True

    print(f"{len(modules)} modules")

    # cond imageをセットする
    cond_image = cond_image.permute(0, 3, 1, 2)  # b,h,w,3 -> b,3,h,w
    cond_image = cond_image * 2.0 - 1.0  # 0-1 -> -1-+1

    for module in modules.values():
        module.set_cond_image(cond_image)

    class control_net_lllite_patch:
        def __init__(self, modules):
            self.modules = modules

        def __call__(self, q, k, v, extra_options):
            module_pfx = extra_options_to_module_prefix(extra_options)

            is_attn1 = q.shape[-1] == k.shape[-1]  # self attention
            if is_attn1:
                module_pfx = module_pfx + "_attn1"
            else:
                module_pfx = module_pfx + "_attn2"

            module_pfx_to_q = module_pfx + "_to_q"
            module_pfx_to_k = module_pfx + "_to_k"
            module_pfx_to_v = module_pfx + "_to_v"

            if module_pfx_to_q in self.modules:
                q = q + self.modules[module_pfx_to_q](q)
            if module_pfx_to_k in self.modules:
                k = k + self.modules[module_pfx_to_k](k)
            if module_pfx_to_v in self.modules:
                v = v + self.modules[module_pfx_to_v](v)

            return q, k, v

        def to(self, device):
            for d in self.modules.keys():
                self.modules[d] = self.modules[d].to(device)
            return self

    return control_net_lllite_patch(modules)


class LLLiteModule(torch.nn.Module):
    def __init__(
        self,
        name: str,
        is_conv2d: bool,
        in_dim: int,
        depth: int,
        cond_emb_dim: int,
        mlp_dim: int,
        multiplier: int,
        num_steps: int,
        start_step: int,
        end_step: int,
        dtype: torch.dtype = torch.float32,
        weight_profile=None,
        band_profile=None,
        depth_weight=1.0
    ):
        super().__init__()
        self.name = name
        self.is_conv2d = is_conv2d
        self.multiplier = multiplier
        self.num_steps = num_steps
        self.start_step = start_step
        self.end_step = end_step
        self.weight_profile = weight_profile
        # profile of the band this module's UNet block belongs to, or None
        self.band_profile = band_profile
        # depth profile evaluated at this module's UNet depth: step-invariant,
        # so it is a plain number rather than a curve
        self.depth_weight = depth_weight
        self.is_first = False

        modules = []
        modules.append(torch.nn.Conv2d(3, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))  # to latent (from VAE) size*2
        if depth == 1:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))
        elif depth == 2:
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=4, stride=4, padding=0))
        elif depth == 3:
            # kernel size 8は大きすぎるので、4にする / kernel size 8 is too large, so set it to 4
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim // 2, kernel_size=4, stride=4, padding=0))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Conv2d(cond_emb_dim // 2, cond_emb_dim, kernel_size=2, stride=2, padding=0))

        self.conditioning1 = torch.nn.Sequential(*modules)

        if self.is_conv2d:
            self.down = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim + cond_emb_dim, mlp_dim, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Conv2d(mlp_dim, in_dim, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.down = torch.nn.Sequential(
                torch.nn.Linear(in_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.mid = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim + cond_emb_dim, mlp_dim),
                torch.nn.ReLU(inplace=True),
            )
            self.up = torch.nn.Sequential(
                torch.nn.Linear(mlp_dim, in_dim),
            )

        self.depth = depth
        self.cond_image = None
        self.cond_emb = None
        self.current_step = 0
        self.to(dtype=dtype)

    # @torch.inference_mode()
    def set_cond_image(self, cond_image):
        # print("set_cond_image", self.name)
        self.cond_image = cond_image
        self.cond_emb = None
        self.current_step = 0

    def forward(self, x):
        multiplier = self.multiplier
        if self.num_steps > 0 and (self.weight_profile is not None or self.band_profile is not None):
            # per-step strength from the weight profile, x axis = relative step
            step_percent = self.current_step / max(self.num_steps - 1, 1)
            if self.weight_profile is not None:
                multiplier = evaluate_weight_profile(self.weight_profile, step_percent)
            if self.band_profile is not None:
                # in band mode the extension sends no main profile and
                # strength 1, so this is the per-step strength of this layer
                multiplier = multiplier * evaluate_weight_profile(self.band_profile, step_percent)
        if self.depth_weight != 1.0:
            # depth profile at this module's UNet depth (step-invariant):
            # multiplies whatever per-step strength is in force above
            multiplier = multiplier * self.depth_weight
        if self.num_steps > 0:
            if self.current_step < self.start_step:
                self.current_step += 1
                return torch.zeros_like(x)
            elif self.current_step >= self.end_step:
                if self.is_first and self.current_step == self.end_step:
                    print(f"end LLLite: step {self.current_step}")
                self.current_step += 1
                if self.current_step >= self.num_steps:
                    self.current_step = 0  # reset
                return torch.zeros_like(x)
            else:
                if self.is_first and self.current_step == self.start_step:
                    print(f"start LLLite: step {self.current_step}")
                self.current_step += 1
                if self.current_step >= self.num_steps:
                    self.current_step = 0  # reset

        # magnitude test, not `<= 0`: a NEGATIVE multiplier is meaningful - the
        # module's output is added to the block output, so a negative value
        # subtracts it (repulsive control, exactly what a negative ControlNet
        # strength does). Only a (near-)zero multiplier is worth short-cutting.
        if abs(multiplier) < 1e-4:
            return torch.zeros_like(x)

        if self.cond_emb is None:
            # print(f"cond_emb is None, {self.name}")
            cx = self.conditioning1(self.cond_image.to(x.device, dtype=x.dtype))
            if not self.is_conv2d:
                # reshape / b,c,h,w -> b,h*w,c
                n, c, h, w = cx.shape
                cx = cx.view(n, c, h * w).permute(0, 2, 1)
            self.cond_emb = cx

        cx = self.cond_emb
        # print(f"forward {self.name}, {cx.shape}, {x.shape}")

        # uncond/condでxはバッチサイズが2倍
        if x.shape[0] != cx.shape[0]:
            if self.is_conv2d:
                cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1, 1)
            else:
                # print("x.shape[0] != cx.shape[0]", x.shape[0], cx.shape[0])
                cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        cx = torch.cat([cx, self.down(x)], dim=1 if self.is_conv2d else 2)
        cx = self.mid(cx)
        cx = self.up(cx)
        return cx * multiplier


class LLLiteLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": None,
                "cond_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lllite"
    CATEGORY = "loaders"

    def load_lllite(self, model, state_dict, cond_image, strength, steps, start_percent, end_percent, weight_profile=None,
                    band_profiles=None):
        # cond_image is b,h,w,3, 0-1

        model_lllite = model.clone()
        patch = load_control_net_lllite_patch(state_dict, cond_image, strength, steps, start_percent, end_percent,
                                              model.model.diffusion_model.computation_dtype, weight_profile=weight_profile,
                                              band_profiles=band_profiles)
        if patch is not None:
            model_lllite.set_model_attn1_patch(patch)
            model_lllite.set_model_attn2_patch(patch)

        return (model_lllite,)
