"""The Z-Image config sniff must match the real checkpoint, and refuse the rest.

`cnpro_host/patchers/zimage_config.py` derives the entire model config from the
checkpoint's tensor shapes, because the release
(``alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union``) ships one safetensors file
and no config.json. If that arithmetic is wrong, the ControlNet either fails to
load or - much worse - loads into a differently-shaped tower and produces
plausible but incorrect images.

This file pins it against the ACTUAL shapes of the released v1 checkpoint (read
from its safetensors header, reproduced below - 136 tensors, no weights needed),
and pins the refusals for the v2.x variants, which CNPro deliberately does not
run. See zimage_config.config_from_state_dict for why.

Run:  <webui python> extensions/forge-neo-cnpro/tests/test_zimage_config.py
Needs nothing: zimage_config imports nothing at all.
Exit code 0 = pass.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
sys.path.insert(0, EXTENSION)

from cnpro_host.patchers.zimage_config import (  # noqa: E402
    UnsupportedZImageControlNet,
    config_from_state_dict,
    places_for,
    refiner_places_for,
)


class Shape:
    """The only thing config_from_state_dict touches on a tensor."""

    def __init__(self, *dims):
        self.shape = dims


def v1_union_state_dict():
    """The released v1 'Union' checkpoint, as shapes.

    Transcribed from the safetensors header of
    ``Z-Image-Turbo-Fun-Controlnet-Union.safetensors``:
    136 tensors, all BF16, `control_layers` 0..5 and `control_noise_refiner` 0..1.
    Only layer 0 carries `before_proj`; the refiner blocks carry no projections
    at all, which is what identifies this as v1.
    """
    sd = {
        "control_all_x_embedder.2-1.weight": Shape(3840, 64),
        "control_all_x_embedder.2-1.bias": Shape(3840),
    }

    def block(prefix, control, first):
        sd[prefix + "adaLN_modulation.0.weight"] = Shape(15360, 256)
        sd[prefix + "adaLN_modulation.0.bias"] = Shape(15360)
        sd[prefix + "attention.norm_k.weight"] = Shape(128)
        sd[prefix + "attention.norm_q.weight"] = Shape(128)
        sd[prefix + "attention.to_k.weight"] = Shape(3840, 3840)
        sd[prefix + "attention.to_q.weight"] = Shape(3840, 3840)
        sd[prefix + "attention.to_v.weight"] = Shape(3840, 3840)
        sd[prefix + "attention.to_out.0.weight"] = Shape(3840, 3840)
        sd[prefix + "attention_norm1.weight"] = Shape(3840)
        sd[prefix + "attention_norm2.weight"] = Shape(3840)
        sd[prefix + "feed_forward.w1.weight"] = Shape(10240, 3840)
        sd[prefix + "feed_forward.w2.weight"] = Shape(3840, 10240)
        sd[prefix + "feed_forward.w3.weight"] = Shape(10240, 3840)
        sd[prefix + "ffn_norm1.weight"] = Shape(3840)
        sd[prefix + "ffn_norm2.weight"] = Shape(3840)
        if control:
            sd[prefix + "after_proj.weight"] = Shape(3840, 3840)
            sd[prefix + "after_proj.bias"] = Shape(3840)
            if first:
                sd[prefix + "before_proj.weight"] = Shape(3840, 3840)
                sd[prefix + "before_proj.bias"] = Shape(3840)

    for i in range(6):
        block("control_layers.%d." % i, control=True, first=(i == 0))
    for i in range(2):
        block("control_noise_refiner.%d." % i, control=False, first=False)
    return sd


def v2_state_dict(n_control=15):
    """The released v2.1 shape, transcribed from its safetensors header.

    295 tensors for the 15-layer Union, 91 for the 3-layer lite. In both:
    control_all_x_embedder is [3840, 132] (= 33 channels x 2 x 2), control_layers.0
    carries before_proj + after_proj and the rest after_proj only, and -- the
    property that separates v2 from v1 -- control_noise_refiner.0 ALSO carries
    before_proj + after_proj, i.e. the refiner blocks are control blocks that emit
    hints into the base model's own noise_refiner.
    """
    sd = {
        "control_all_x_embedder.2-1.weight": Shape(3840, 132),
        "control_all_x_embedder.2-1.bias": Shape(3840),
    }

    def block(prefix, first):
        sd[prefix + "adaLN_modulation.0.weight"] = Shape(15360, 256)
        sd[prefix + "adaLN_modulation.0.bias"] = Shape(15360)
        sd[prefix + "attention.norm_k.weight"] = Shape(128)
        sd[prefix + "attention.norm_q.weight"] = Shape(128)
        sd[prefix + "attention.to_k.weight"] = Shape(3840, 3840)
        sd[prefix + "attention.to_q.weight"] = Shape(3840, 3840)
        sd[prefix + "attention.to_v.weight"] = Shape(3840, 3840)
        sd[prefix + "attention.to_out.0.weight"] = Shape(3840, 3840)
        sd[prefix + "attention_norm1.weight"] = Shape(3840)
        sd[prefix + "attention_norm2.weight"] = Shape(3840)
        sd[prefix + "feed_forward.w1.weight"] = Shape(10240, 3840)
        sd[prefix + "feed_forward.w2.weight"] = Shape(3840, 10240)
        sd[prefix + "feed_forward.w3.weight"] = Shape(10240, 3840)
        sd[prefix + "ffn_norm1.weight"] = Shape(3840)
        sd[prefix + "ffn_norm2.weight"] = Shape(3840)
        sd[prefix + "after_proj.weight"] = Shape(3840, 3840)
        sd[prefix + "after_proj.bias"] = Shape(3840)
        if first:
            sd[prefix + "before_proj.weight"] = Shape(3840, 3840)
            sd[prefix + "before_proj.bias"] = Shape(3840)

    for i in range(n_control):
        block("control_layers.%d." % i, first=(i == 0))
    for i in range(2):
        block("control_noise_refiner.%d." % i, first=(i == 0))
    return sd


FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append("%s: got %r, want %r" % (label, got, want))


def expect_refusal(label, sd, must_mention):
    path = ""
    if isinstance(sd, tuple):
        sd, path = sd
    try:
        config_from_state_dict(sd, path)
    except UnsupportedZImageControlNet as exc:
        if must_mention.lower() not in str(exc).lower():
            FAILURES.append("%s: message %r does not mention %r" % (label, str(exc), must_mention))
        return
    FAILURES.append("%s: expected UnsupportedZImageControlNet, got a config" % label)


def main():
    # --- the released v1 checkpoint ---------------------------------------
    cfg = config_from_state_dict(v1_union_state_dict())
    check("v1 dim", cfg["dim"], 3840)
    check("v1 heads", cfg["heads"], 30)              # 3840 / head_dim 128
    check("v1 hidden_dim", cfg["hidden_dim"], 10240)
    check("v1 control_in_dim", cfg["control_in_dim"], 16)
    check("v1 patch_size", cfg["patch_size"], 2)
    check("v1 n_control_layers", cfg["n_control_layers"], 6)
    check("v1 n_refiner_layers", cfg["n_refiner_layers"], 2)

    # heads must match the BASE model's n_heads=30 (Tongyi-MAI/Z-Image-Turbo
    # transformer/config.json), or the control branch attends differently from
    # the branch it steers.
    check("v1 heads == base n_heads", cfg["heads"], 30)

    # --- injection sites --------------------------------------------------
    # The three published VideoX-Fun configs, verbatim.
    check("v1 places", places_for(6, 30), [0, 5, 10, 15, 20, 25])
    check("2.1-lite places", places_for(3, 30), [0, 10, 20])
    check("2.x places", places_for(15, 30), list(range(0, 30, 2)))
    # upstream asserts 0 is always present (control layer 0 seeds the stream)
    for n in range(1, 31):
        if 0 not in places_for(n, 30):
            FAILURES.append("places_for(%d, 30) does not contain block 0" % n)
    # never index past the model
    for n in (1, 3, 6, 15, 29, 30, 40):
        p = places_for(n, 30)
        if p and max(p) >= 30:
            FAILURES.append("places_for(%d, 30) = %r indexes past the model" % (n, p))
    check("degenerate places", places_for(0, 30), [])

    check("v1 variant", cfg["variant"], "v1")
    check("v1 refiner is plain", cfg["refiner_is_control"], False)

    # --- the released v2.1 checkpoint --------------------------------------
    cfg2 = config_from_state_dict(v2_state_dict(15), "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors")
    check("v2 variant", cfg2["variant"], "v2")
    check("v2 control_in_dim", cfg2["control_in_dim"], 33)
    check("v2 n_control_layers", cfg2["n_control_layers"], 15)
    check("v2 n_refiner_layers", cfg2["n_refiner_layers"], 2)
    check("v2 refiner is control", cfg2["refiner_is_control"], True)
    check("v2 dim", cfg2["dim"], 3840)
    check("v2 heads", cfg2["heads"], 30)

    lite = config_from_state_dict(v2_state_dict(3), "Union-2.1-lite-2602-8steps.safetensors")
    check("lite n_control_layers", lite["n_control_layers"], 3)
    check("lite variant", lite["variant"], "v2")

    # --- v2 injection sites ------------------------------------------------
    check("v2 places", places_for(15, 30), list(range(0, 30, 2)))
    check("v2 refiner places", refiner_places_for(2, 2), [0, 1])
    check("refiner places clamp", refiner_places_for(2, 1), [0])
    check("refiner places none", refiner_places_for(0, 2), [])

    # --- refusals ---------------------------------------------------------
    # 2.0 has BYTE-IDENTICAL tensors to 2.1 and a different forward pass, so the
    # filename is the only discriminator - used ONLY to refuse, never to decide
    # how to run something.
    expect_refusal("v2.0 by filename",
                   (v2_state_dict(15), "Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors"),
                   "2.0")
    # ...and the same tensors under the 2.1 name must LOAD, or the refusal is
    # just a broken sniff wearing a message.
    try:
        config_from_state_dict(v2_state_dict(15), "Union-2.1-8steps.safetensors")
    except UnsupportedZImageControlNet as exc:
        FAILURES.append("the 2.1 filename was refused too: %s" % exc)

    # a shape that is neither generation
    mixed = v2_state_dict(15)
    mixed["control_all_x_embedder.2-1.weight"] = Shape(3840, 64)   # 16ch + projected refiners
    expect_refusal("mixed generation", mixed, "neither")

    sd = v1_union_state_dict()
    del sd["control_all_x_embedder.2-1.weight"]
    expect_refusal("no embedder", sd, "control_all_x_embedder")

    sd = v1_union_state_dict()
    del sd["control_layers.0.before_proj.weight"]
    expect_refusal("no before_proj", sd, "before_proj")

    sd = v1_union_state_dict()
    sd["control_layers.0.adaLN_modulation.0.weight"] = Shape(15360, 1024)
    expect_refusal("wrong adaLN width", sd, "adaln")

    sd = v1_union_state_dict()
    sd["control_layers.0.adaLN_modulation.0.weight"] = Shape(7680, 256)
    expect_refusal("wrong adaLN fan-out", sd, "4 * dim")

    sd = v1_union_state_dict()
    del sd["control_layers.0.attention.norm_q.weight"]
    expect_refusal("no qk-norm", sd, "norm_q")

    # An SDXL ControlNet must not be mistaken for one of these.
    expect_refusal("sdxl controlnet", {"zero_convs.0.0.weight": Shape(320, 320, 1, 1)},
                   "control_all_x_embedder")

    # A MERGED checkpoint (full transformer + control tower in one file, e.g.
    # neuralvfx/Z-Image-SAM-ControlNet at 816 tensors) must be reported AS a
    # merged file, not as "v2.x". Both statements are true of it; only one tells
    # the user what to do. Distinct failures, distinct messages.
    merged = v2_state_dict(15)
    merged["layers.0.attention.to_q.weight"] = Shape(3840, 3840)
    merged["noise_refiner.0.ffn_norm1.weight"] = Shape(3840)
    merged["cap_embedder.1.weight"] = Shape(3840, 2560)
    expect_refusal("merged full model + control", merged, "full Z-Image transformer")

    # ...and the merged check must not fire on a plain ControlNet: every key of
    # the control tower is `control_`-prefixed, and `control_layers.` must not be
    # mistaken for the base model's `layers.`
    try:
        config_from_state_dict(v1_union_state_dict())
    except UnsupportedZImageControlNet as exc:
        if "full Z-Image transformer" in str(exc):
            FAILURES.append("the merged-file check false-positives on control_layers.*")

    if FAILURES:
        print("FAIL (%d)" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ok - config sniff matches the released v1 AND v2.1 checkpoints "
          "(incl. lite), refuses v2.0 by name, merged files and truncated variants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
