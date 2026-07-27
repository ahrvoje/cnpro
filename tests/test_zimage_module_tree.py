"""CNPro's Z-Image ControlNet must have EXACTLY the released checkpoint's tensors.

`zimage_impl.build_control_model` calls `load_state_dict(..., strict=False)` and
then refuses on missing keys - but "missing" only catches parameters the module
tree HAS and the file lacks. The dangerous direction is the other one: a module
tree that silently omits a projection loads cleanly, reports nothing, and steers
the image with a ControlNet that is quietly missing a layer.

So this compares both directions against the real key set. The checkpoint is
reproduced from the safetensors header of
``alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union.safetensors``
(136 tensors) - names only, no weights, nothing to download.

Run:  <webui python> extensions/forge-neo-cnpro/tests/test_zimage_module_tree.py
Needs torch AND the host on sys.path (the vendored blocks use the host's own
attention dispatcher, deliberately - see zimage_impl.Attention).
Exit code 0 = pass.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
WEBUI = os.path.dirname(os.path.dirname(EXTENSION))
sys.path.insert(0, EXTENSION)
sys.path.insert(0, WEBUI)
# The host vendors several packages (gguf, huggingface_guess, comfy) and puts
# this directory on sys.path during its own startup; importing backend.* outside
# a webui run needs the same.
sys.path.insert(0, os.path.join(WEBUI, "modules_forge", "packages"))

sys.path.insert(0, HERE)
from test_zimage_config import v1_union_state_dict, v2_state_dict  # noqa: E402

from cnpro_host.patchers.zimage_config import config_from_state_dict  # noqa: E402


def main():
    rc = 0
    for label, shapes, path in (
            ("v1 Union", v1_union_state_dict(), "Union.safetensors"),
            ("v2.1 Union", v2_state_dict(15), "Union-2.1.safetensors"),
            ("v2.1 lite", v2_state_dict(3), "Union-2.1-lite.safetensors")):
        rc |= check_tree(label, shapes, path)
    return rc


def check_tree(label, shapes, path):
    from cnpro_host.patchers.zimage_impl import ZImageControlNetModel

    config = config_from_state_dict(shapes, path)
    model = ZImageControlNetModel(**config)

    want = set(shapes)
    got = set(model.state_dict())

    missing = sorted(want - got)     # in the file, not in the module tree
    extra = sorted(got - want)       # in the module tree, not in the file

    problems = []
    if missing:
        problems.append(
            "%d checkpoint tensors have nowhere to load into (the module tree is "
            "INCOMPLETE - this is the silent-wrong-image case):\n    %s"
            % (len(missing), "\n    ".join(missing[:12])))
    if extra:
        problems.append(
            "%d module parameters are absent from the checkpoint (they would keep "
            "their random init):\n    %s"
            % (len(extra), "\n    ".join(extra[:12])))

    # shapes must match too - a same-named parameter of the wrong size is a
    # load-time error at best and a transposed projection at worst
    for k in sorted(want & got):
        w = tuple(shapes[k].shape)
        g = tuple(model.state_dict()[k].shape)
        if w != g:
            problems.append("shape mismatch %s: checkpoint %r, module %r" % (k, w, g))

    if problems:
        print("FAIL (%s)" % label)
        for p in problems:
            print("  -", p)
        return 1

    print("ok - %-11s module tree matches all %d tensors, names and shapes"
          % (label, len(want)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
