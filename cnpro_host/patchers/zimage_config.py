"""Z-Image ControlNet: checkpoint -> config, and injection sites.

Split out of ``zimage_impl`` deliberately. This module has ZERO imports -- it
reads ``.shape`` off whatever the caller hands it and returns plain dicts and
lists -- which means:

  * it is unit-testable against the real checkpoint's SHAPES without downloading
    3.1 GB of weights, without torch, and without a host (see
    ``tests/test_zimage_config.py``);
  * when the v2.x variants get implemented, the part that has to change is 60
    lines of arithmetic with tests around it, not the engine.

It is not in ``cnpro_core`` because it is knowledge about one model family, and
cnpro_core is the part that is supposed to outlive every model family.


WHY EVERYTHING IS DERIVED
-------------------------
``alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union`` ships ONE file:
``Z-Image-Turbo-Fun-Controlnet-Union.safetensors``. No config.json, no
model_index.json. So every structural number below is read off the tensors, and
the two that cannot be (the injection sites) are reconstructed from the
published VideoX-Fun configs and documented as such.
"""

#: adaLN input width. `min(dim, 256)` upstream; the host's TimestepEmbedder emits
#: 256 when z_image_modulation is on. Checked against the checkpoint, because a
#: mismatch here means the model is not the Z-Image modulation variant at all.
ADALN_EMBED_DIM = 256

#: The only patch layout released so far: 2x2 spatial patches, 1 frame.
PATCH_SIZE = 2
F_PATCH_SIZE = 1
EMBEDDER_KEY = "control_all_x_embedder.%d-%d.weight" % (PATCH_SIZE, F_PATCH_SIZE)


class UnsupportedZImageControlNet(Exception):
    """A Z-Image ControlNet CNPro recognises but will not run.

    Raised instead of guessing. Every message says what was found and why that
    is not enough, because the alternative failure mode -- a ControlNet that
    loads and produces subtly wrong images -- costs the user far more time.
    """

    #: Tells `registry.load_control_model` this is a DELIBERATE refusal, not a
    #: loader that broke: the file was recognised and rejected on purpose, so the
    #: registry raises it immediately instead of trying the remaining patchers
    #: and then reporting the far vaguer "not a control model CNPro recognises".
    #: A plain attribute rather than a shared base class, so this module can keep
    #: its zero-imports property (which is what makes it testable standalone).
    cnpro_recognised = True


def _indices(keys, prefix):
    out = set()
    for k in keys:
        if k.startswith(prefix):
            rest = k[len(prefix):].split(".", 1)[0]
            if rest.isdigit():
                out.add(int(rest))
    return out


def config_from_state_dict(sd, ckpt_path=""):
    """Derive the model config from the checkpoint alone, or raise.

    The released variants differ observably:

      v1 "Union"    6 control layers, control_in_dim 16, refiner blocks WITHOUT
                    the control projections (they are plain transformer blocks).
      v2.0 / v2.1  15 control layers, control_in_dim 33 (latent + masked latent
                    + mask), refiner blocks WITH the control projections.
      v2.1-lite     3 control layers, otherwise as v2.x.

    v2.x is refused rather than approximated, for two independent reasons:

      1. its 33-channel conditioning is an inpainting-shaped input (image latent,
         masked-image latent, and the mask) that CNPro has no UI to produce;
      2. v2.0 and v2.1 have BYTE-IDENTICAL key sets and differ only in whether
         the refiner is applied the way the authors intended
         (`add_control_noise_refiner_correctly`). The checkpoint cannot say which
         one it is, so any implementation would be right half the time and
         silently wrong the other half.
    """
    keys = set(sd)
    emb = sd.get(EMBEDDER_KEY)
    if emb is None:
        raise UnsupportedZImageControlNet(
            "no %s: patch size %d / frame patch %d is the only Z-Image control "
            "layout released so far" % (EMBEDDER_KEY, PATCH_SIZE, F_PATCH_SIZE))

    dim = int(emb.shape[0])
    control_in_dim = int(emb.shape[1]) // (PATCH_SIZE * PATCH_SIZE * F_PATCH_SIZE)

    n_control = len(_indices(keys, "control_layers."))
    n_refiner = len(_indices(keys, "control_noise_refiner."))
    if n_control == 0:
        raise UnsupportedZImageControlNet("no control_layers.* tensors")

    # A MERGED checkpoint: the whole Z-Image transformer plus a control tower in
    # one file (neuralvfx/Z-Image-SAM-ControlNet is 816 tensors of exactly this).
    # Checked BEFORE the v2.x test because a merged file is v2.x-SHAPED, and
    # answering "this looks like v2.x" would be true about the control tower and
    # useless about the actual problem - which is that CNPro loads a ControlNet
    # to sit alongside the checkpoint the user selected, not a second copy of the
    # model. Collapsing two different failures into one message is the exact
    # thing ARCHITECTURE.md section 8 exists to prevent, so it must not happen in
    # the code that section points at.
    #
    # `layers.` / `noise_refiner.` / `cap_embedder.` name the BASE model only;
    # the control tower's own keys are all `control_`-prefixed, so there is no
    # overlap to false-positive on.
    base_keys = [p for p in ("layers.", "noise_refiner.", "context_refiner.", "cap_embedder.")
                 if any(k.startswith(p) for k in keys)]
    if base_keys:
        raise UnsupportedZImageControlNet(
            "this file contains a full Z-Image transformer as well as a control "
            "tower (%d tensors, base-model keys: %s). CNPro loads a STANDALONE "
            "ControlNet that runs beside the checkpoint you selected; a merged "
            "file would need its base half stripped, and which half should win "
            "is the user's call, not a guess CNPro should make."
            % (len(keys), ", ".join(base_keys)))

    # --- which generation is this? -----------------------------------------
    # Two observable properties separate them, and they always agree:
    #
    #   v1   control_in_dim 16, control_noise_refiner blocks are PLAIN
    #        (no projections) - they only refine the control tokens.
    #   v2.x control_in_dim 33, control_noise_refiner blocks are CONTROL blocks
    #        (block 0 has before_proj, all have after_proj) - they refine AND
    #        emit hints into the base model's own noise_refiner.
    #
    # Verified against the released headers: v1 = 136 tensors, 2.1 = 295,
    # 2.1-lite = 91, and in every v2 file control_noise_refiner.0 carries
    # before_proj + after_proj while .1 carries after_proj only.
    refiner_is_control = any(
        k.startswith("control_noise_refiner.") and ".after_proj." in k for k in keys)

    if control_in_dim == 16 and not refiner_is_control:
        variant = "v1"
    elif control_in_dim == 33 and refiner_is_control:
        variant = "v2"
    else:
        raise UnsupportedZImageControlNet(
            "unrecognised Z-Image ControlNet generation: control_in_dim=%d with "
            "%s refiners. CNPro knows v1 (16 / plain) and v2.x (33 / projected); "
            "this is neither, so its forward pass is unknown."
            % (control_in_dim, "projected" if refiner_is_control else "plain"))

    # v2.0 vs v2.1 cannot be told apart from the tensors - identical key sets.
    # They differ only in WHICH module list acts as the refiner, and 2.0's choice
    # is a bug the authors fixed: their own release notes say it "used
    # control_layers instead of control_noise_refiner, which caused double
    # forward pass and slow inference". So 2.0 is both wrong and slower, and it
    # is superseded by a file sitting in the same repo.
    #
    # The FILENAME is the only signal, and it is used only to REFUSE, never to
    # decide how to run something. Refusing on a filename can at worst reject a
    # file that would have worked; guessing the forward pass from one would at
    # worst produce plausible wrong images, silently.
    name = str(ckpt_path or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    if variant == "v2" and ("2.0" in name or "2_0" in name or "v20" in name):
        raise UnsupportedZImageControlNet(
            "this filename says Fun-ControlNet v2.0 (%r). CNPro implements v2.1, "
            "which is the same architecture with the refiner bug fixed - upstream "
            "describes 2.0 as running a double forward pass and being slower for "
            "it. 2.0 and 2.1 have identical tensors, so CNPro cannot verify which "
            "this really is; use the 2.1 file from the same repository." % name)

    ff = sd.get("control_layers.0.feed_forward.w1.weight")
    if ff is None:
        raise UnsupportedZImageControlNet("no control_layers.0.feed_forward.w1.weight")

    qn = sd.get("control_layers.0.attention.norm_q.weight")
    if qn is None:
        raise UnsupportedZImageControlNet(
            "no control_layers.0.attention.norm_q.weight; CNPro reads the head "
            "dimension from it, and a model without qk-norm is a different variant")
    head_dim = int(qn.shape[0])
    if head_dim <= 0 or dim % head_dim:
        raise UnsupportedZImageControlNet(
            "dim %d is not a multiple of head dim %d" % (dim, head_dim))

    adaln = sd.get("control_layers.0.adaLN_modulation.0.weight")
    want_adaln = min(dim, ADALN_EMBED_DIM)
    if adaln is None or int(adaln.shape[1]) != want_adaln:
        raise UnsupportedZImageControlNet(
            "adaLN input width %s != min(dim, %d) = %d; this is not a "
            "z_image_modulation model"
            % (None if adaln is None else adaln.shape[1], ADALN_EMBED_DIM, want_adaln))
    if int(adaln.shape[0]) != 4 * dim:
        raise UnsupportedZImageControlNet(
            "adaLN output width %d != 4 * dim (%d); the block does not use the "
            "four-way scale/gate modulation this implementation applies"
            % (adaln.shape[0], 4 * dim))

    if sd.get("control_layers.0.before_proj.weight") is None:
        raise UnsupportedZImageControlNet(
            "control_layers.0 has no before_proj; upstream asserts that control "
            "layer 0 is the one that seeds the control stream from the base "
            "model's sequence, and nothing else can do that job")

    return dict(dim=dim, heads=dim // head_dim, hidden_dim=int(ff.shape[0]),
                control_in_dim=control_in_dim, patch_size=PATCH_SIZE,
                n_control_layers=n_control, n_refiner_layers=n_refiner,
                # 'v1' | 'v2' -- selects the forward pass, see zimage_impl.
                variant=variant,
                # v2 only: the refiner blocks carry the control projections and
                # emit a second set of hints into the BASE model's noise_refiner.
                refiner_is_control=refiner_is_control)


def refiner_places_for(n_refiner_layers, base_refiner_count):
    """Which base-model `noise_refiner` blocks the refiner-stage hints land on.

    Upstream ships this as ``control_refiner_layers_places``, and every published
    v2 config sets it to ``[0, 1]`` for two refiner blocks
    (config/z_image/z_image_control_2.{0,1}*.yaml). The general rule is the same
    one ``places_for`` uses -- one hint per block, in order -- which for the only
    shipped shape (2 control refiners, 2 base refiners) is the identity.

    Clamped to the base model's actual refiner count so a mismatch drops the
    extra hints rather than indexing off the end.
    """
    if n_refiner_layers <= 0 or base_refiner_count <= 0:
        return []
    return list(range(min(n_refiner_layers, base_refiner_count)))


def places_for(n_control_layers, n_blocks):
    """Which base-model blocks each control layer injects at.

    Upstream ships these as CONFIG, not as weights, so they are reconstructed
    from the published VideoX-Fun configs (config/z_image/*.yaml):

        6 layers  -> [0, 5, 10, 15, 20, 25]   z_image_control.yaml, 30 blocks
        3 layers  -> [0, 10, 20]              z_image_control_2.1_lite.yaml
        15 layers -> [0, 2, 4, ..., 28]       z_image_control_2.x.yaml

    All three are exactly ``range(0, n_blocks, n_blocks // n_layers)``, so that
    rule is used rather than a lookup table: an unseen layer count then follows
    the family's own convention instead of failing.

    Upstream asserts ``0 in control_layers_places`` (layer 0 seeds the stream);
    that holds here by construction, and is re-checked by the config sniff.
    """
    if n_control_layers <= 0 or n_blocks <= 0:
        return []
    stride = max(1, n_blocks // n_control_layers)
    return [i * stride for i in range(n_control_layers) if i * stride < n_blocks]
