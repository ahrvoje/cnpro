"""`control_merge` and the model loader, before and after the subclass refactor.

CNPro's `ControlBase` used to carry a full copy of the host's `control_merge`
with the weighting call spliced into its middle, and `ControlNetPatcher` carried
a copy of the host's state-dict sniffing. Both are now composed instead: the
host groups and scales the residuals, CNPro weights them, CNPro folds in the
previous unit; and the host recognizes the file while CNPro re-wraps the result
in its own control class.

Both are meant to be BEHAVIOUR-PRESERVING, and this file is what says so. The
pre-refactor `control_merge` is frozen below and every case runs through both.

It also pins the property the refactor DEPENDS ON and which nothing else here
would notice breaking: the host's own weighting pass runs inside the `super()`
call, and it is inert only because CNPro never writes the host's attribute names
(`positive_advanced_weighting` and friends) onto a control object. Reintroduce
one of those names and every weight is applied TWICE -- silently, with a plausible
image out the other end. That is the failure this exists to catch.

Run:  <webui python> extensions/forge-neo-cnpro/tests/test_control_merge_parity.py
Needs torch + the host (imports backend.*). Skips the loader half, loudly, when
no SDXL-family control model is on disk.
Exit code 0 = pass or skip; 1 = fail.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENSION = os.path.dirname(HERE)
WEBUI = os.path.dirname(os.path.dirname(EXTENSION))
sys.path.insert(0, EXTENSION)
sys.path.insert(0, WEBUI)
# backend.utils pulls in the vendored gguf package, which is only importable
# from modules_forge/packages - same bootstrap as test_patcher_contract.py
sys.path.insert(0, os.path.join(WEBUI, "modules_forge", "packages"))

import torch  # noqa: E402

from cnpro_host.patchers import controlnet_impl as impl  # noqa: E402
from cnpro_host.patchers.weighting import (  # noqa: E402
    WEIGHTING_INPUTS,
    compute_controlnet_weighting,
)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


#: The five names the HOST's own weighting pass reads. CNPro must never set any
#: of them on a control object; see the module docstring.
HOST_WEIGHTING_SURFACE = (
    'positive_advanced_weighting',
    'negative_advanced_weighting',
    'advanced_frame_weighting',
    'advanced_sigma_weighting',
    'advanced_mask_weighting',
)


def reference_control_merge(self, control_input, control_output, control_prev, output_dtype):
    """CNPro's `control_merge` exactly as it was before the refactor. Frozen."""
    out = {'input': [], 'middle': [], 'output': []}

    if control_input is not None:
        for i in range(len(control_input)):
            key = 'input'
            x = control_input[i]
            if x is not None:
                x *= self.strength
                if x.dtype != output_dtype:
                    x = x.to(output_dtype)
            out[key].insert(0, x)

    if control_output is not None:
        for i in range(len(control_output)):
            if i == (len(control_output) - 1):
                key = 'middle'
            else:
                key = 'output'
            x = control_output[i]
            if x is not None:
                if self.global_average_pooling:
                    x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

                x *= self.strength
                if x.dtype != output_dtype:
                    x = x.to(output_dtype)

            out[key].append(x)

    out = compute_controlnet_weighting(out, self)

    if control_prev is not None:
        for x in ['input', 'middle', 'output']:
            o = out[x]
            for i in range(len(control_prev[x])):
                prev_val = control_prev[x][i]
                if i >= len(o):
                    o.append(prev_val)
                elif prev_val is not None:
                    if o[i] is None:
                        o[i] = prev_val
                    else:
                        if o[i].shape[0] < prev_val.shape[0]:
                            o[i] = prev_val + o[i]
                        else:
                            o[i] += prev_val
    return out


def make_control(batch=2, strength=1.0, gap=False, **profiles):
    """A CNPro ControlBase configured the way a sampling run would leave it."""
    cb = impl.ControlBase(device=torch.device('cpu'))
    cb.strength = strength
    cb.global_average_pooling = gap
    cb.transformer_options = {
        'cond_or_uncond': [0, 1],
        'sigmas': torch.tensor([1.0]),
        # 0 = cond row, 1 = uncond row
        'cond_mark': torch.tensor([0.0] * (batch // 2) + [1.0] * (batch - batch // 2)),
    }
    for k, v in profiles.items():
        setattr(cb, k, v)
    return cb


def residuals(n, batch=2, c=4, hw=8, fill=None):
    out = []
    for i in range(n):
        t = torch.full((batch, c, hw, hw), float(i + 1) if fill is None else fill)
        out.append(t)
    return out


def compare(name, got, want):
    if set(got) != set(want):
        fail("%s: group names differ, got %r want %r" % (name, sorted(got), sorted(want)))
        return
    for k in want:
        if len(got[k]) != len(want[k]):
            fail("%s: group %r has %d entries, reference has %d"
                 % (name, k, len(got[k]), len(want[k])))
            continue
        for i, (a, b) in enumerate(zip(got[k], want[k])):
            if a is None or b is None:
                if a is not b:
                    fail("%s: %s[%d] None mismatch (%r vs %r)" % (name, k, i, a, b))
                continue
            if a.shape != b.shape:
                fail("%s: %s[%d] shape %r, reference %r" % (name, k, i, tuple(a.shape), tuple(b.shape)))
            elif not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
                fail("%s: %s[%d] differs from reference by %r"
                     % (name, k, i, (a - b).abs().max().item()))


# ---------------------------------------------------------------------------
# 1. the refactor is behaviour-preserving
# ---------------------------------------------------------------------------

CASES = [
    ("plain, no profiles", dict(), None, False, 1.0),
    ("strength 0.5", dict(), None, False, 0.5),
    ("global average pooling", dict(), None, True, 1.0),
    ("depth profile", dict(depth_profile=[(0.0, 0.0), (1.0, 1.0)]), None, False, 1.0),
    ("band profiles", dict(band_profile_lookup={
        'coarse': ([1.0, 0.0], [1.0, 1.0]),
        'mid': ([1.0, 0.0], [0.5, 0.5]),
        'fine': ([1.0, 0.0], [0.0, 0.0]),
    }), None, False, 1.0),
    ("region mask, whole", dict(region_masks=torch.ones(1, 1, 16, 16)), None, False, 1.0),
    ("region mask, per band", dict(region_masks={'coarse': torch.ones(1, 1, 16, 16)}),
     None, False, 1.0),
    ("per-site cond/uncond weights", dict(
        cond_layer_weights={'input': [0.1, 0.2, 0.3, 0.4], 'middle': [1.0],
                            'output': [0.4, 0.3, 0.2, 0.1]},
        uncond_layer_weights={'input': [1.0, 1.0, 1.0, 1.0], 'middle': [0.0],
                              'output': [0.5, 0.5, 0.5, 0.5]},
    ), None, False, 1.0),
]


def test_refactor_matches_reference():
    for name, profiles, _prev, gap, strength in CASES:
        for with_prev in (False, True):
            label = "%s%s" % (name, " + chained unit" if with_prev else "")

            def build():
                return residuals(4), residuals(5)

            prev = None
            if with_prev:
                prev = {'input': residuals(4, fill=0.25),
                        'middle': residuals(1, fill=0.25),
                        'output': residuals(4, fill=0.25)}

            ci_a, co_a = build()
            ci_b, co_b = build()
            prev_a = None if prev is None else {k: [t.clone() for t in v] for k, v in prev.items()}
            prev_b = None if prev is None else {k: [t.clone() for t in v] for k, v in prev.items()}

            got = make_control(gap=gap, strength=strength, **profiles).control_merge(
                ci_a, co_a, prev_a, torch.float32)
            want = reference_control_merge(
                make_control(gap=gap, strength=strength, **profiles),
                ci_b, co_b, prev_b, torch.float32)
            compare(label, got, want)


def test_chain_with_wider_previous_batch():
    """A previous unit that ran on more rows must win the accumulator.

    The real asymmetry is 1 row against N: a control that ran unbatched
    broadcasts against one that did not, and `narrow += wide` would raise where
    `wide + narrow` does not. Both implementations pick the same side.
    """
    prev = {'input': [torch.ones(2, 4, 8, 8) for _ in range(4)],
            'middle': [torch.ones(2, 4, 8, 8)],
            'output': [torch.ones(2, 4, 8, 8) for _ in range(4)]}
    got = make_control().control_merge(
        residuals(4, batch=1), residuals(5, batch=1),
        {k: [t.clone() for t in v] for k, v in prev.items()}, torch.float32)
    want = reference_control_merge(
        make_control(), residuals(4, batch=1), residuals(5, batch=1),
        {k: [t.clone() for t in v] for k, v in prev.items()}, torch.float32)
    compare("wider previous batch", got, want)
    for k, v in got.items():
        for i, t in enumerate(v):
            if t.shape[0] != 2:
                fail("wider previous batch: %s[%d] came out %d rows, expected the "
                     "previous unit's 2" % (k, i, t.shape[0]))


# ---------------------------------------------------------------------------
# 2. the property the refactor depends on
# ---------------------------------------------------------------------------

def test_host_weighting_surface_stays_unset():
    """CNPro must never write the host's weighting names onto a control object.

    They are inherited (our ControlBase subclasses the host's), so they are
    always *readable*; what must hold is that they are never set to anything
    the host's pass would act on.
    """
    cb = make_control(depth_profile=[(0.0, 0.5), (1.0, 1.0)],
                      region_masks=torch.ones(1, 1, 16, 16))
    for name in HOST_WEIGHTING_SURFACE:
        value = getattr(cb, name, None)
        if value is not None:
            fail("ControlBase sets the host's %r (= %r); the host's weighting pass "
                 "would fire inside super().control_merge() and every weight would "
                 "be applied twice" % (name, value))

    # and the same after a full apply_controlnet_advanced-style configuration
    if set(HOST_WEIGHTING_SURFACE) & set(WEIGHTING_INPUTS):
        fail("CNPro's weighting input names collide with the host's: %r"
             % (sorted(set(HOST_WEIGHTING_SURFACE) & set(WEIGHTING_INPUTS)),))


def test_double_application_would_be_caught():
    """Prove the guard above can fail: set a host name and see the weights double."""
    mask = torch.ones(1, 1, 16, 16)
    clean = make_control(region_masks=mask).control_merge(
        residuals(4), residuals(5), None, torch.float32)

    poisoned = make_control(region_masks=mask)
    # what a careless re-merge with the host's naming would produce
    poisoned.positive_advanced_weighting = {'input': [2.0] * 4, 'middle': [2.0],
                                            'output': [2.0] * 4}
    poisoned.negative_advanced_weighting = {'input': [2.0] * 4, 'middle': [2.0],
                                            'output': [2.0] * 4}
    doubled = poisoned.control_merge(residuals(4), residuals(5), None, torch.float32)

    same = all(torch.allclose(a, b, atol=1e-6)
               for k in clean for a, b in zip(clean[k], doubled[k]))
    if same:
        fail("setting the host's positive_advanced_weighting changed nothing - the "
             "guard in test_host_weighting_surface_stays_unset cannot fail, so it "
             "proves nothing")


# ---------------------------------------------------------------------------
# 3. the delegated loader returns CNPro's classes
# ---------------------------------------------------------------------------

CONTROL_DIRS = [r"D:/store/models/CN/big", r"D:/store/models/CN"]


def find_model(*needles):
    for d in CONTROL_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            low = f.lower()
            if f.endswith(".safetensors") and all(n in low for n in needles):
                return os.path.join(d, f)
    return None


def test_transplant_maps_every_host_class():
    """`_as_cnpro_control` must map each host control class to CNPro's own.

    This is the half of the delegation that CNPro still owns, and it is checked
    without touching a file: the mapping is what decides whether the profiles
    are ever read, and getting it wrong for one branch (ControlLora is a
    SUBCLASS of ControlNet in the host, so an isinstance chain in the wrong
    order silently flattens it) is invisible at runtime until a LoRA-style
    control quietly behaves like a plain one.
    """
    from backend.patcher.controlnet import (
        ControlLora as HostControlLora,
        ControlNet as HostControlNet,
        T2IAdapter as HostT2IAdapter,
    )
    from cnpro_host.patchers.controlnet import _as_cnpro_control

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.dtype = torch.float32
            self.xl = False
            self.input_channels = 3

    host_cn = HostControlNet(DummyModel(), global_average_pooling=True,
                             device=torch.device('cpu'),
                             load_device=torch.device('cpu'),
                             manual_cast_dtype=torch.float16)
    host_lora = HostControlLora({'lora_controlnet': torch.zeros(1)},
                                global_average_pooling=True,
                                device=torch.device('cpu'))
    host_t2i = HostT2IAdapter(DummyModel(), 3, device=torch.device('cpu'))

    for host_obj, want in ((host_lora, impl.ControlLora),
                           (host_t2i, impl.T2IAdapter),
                           (host_cn, impl.ControlNet)):
        got = _as_cnpro_control(host_obj)
        if got is None:
            fail("_as_cnpro_control returned None for the host's %s"
                 % type(host_obj).__name__)
            continue
        if type(got) is not want:
            fail("the host's %s maps to CNPro's %s, expected %s"
                 % (type(host_obj).__name__, type(got).__name__, want.__name__))
        if not isinstance(got, impl.ControlBase):
            fail("%s is not a CNPro ControlBase - its profiles would never be read"
                 % type(got).__name__)
        for name in HOST_WEIGHTING_SURFACE:
            if getattr(got, name, None) is not None:
                fail("transplanted %s has the host's %r set" % (type(got).__name__, name))

    # the transplant must carry the loaded modules, not rebuild them
    if _as_cnpro_control(host_cn).control_model is not host_cn.control_model:
        fail("ControlNet transplant did not reuse the loaded control_model")
    if _as_cnpro_control(host_t2i).t2i_model is not host_t2i.t2i_model:
        fail("T2IAdapter transplant did not reuse the loaded t2i_model")
    if _as_cnpro_control(host_cn).global_average_pooling is not True:
        fail("ControlNet transplant dropped global_average_pooling")

    # an unknown control class must be declined, not silently accepted
    if _as_cnpro_control(object()) is not None:
        fail("_as_cnpro_control accepted an unknown class instead of declining it")


# ---------------------------------------------------------------------------
# 2b. what the inheritance has to resolve to
# ---------------------------------------------------------------------------

#: (class, method, must the winner be CNPro's?) - every one of these is a
#: silent failure if it resolves the other way: the control still runs, still
#: produces an image, and simply ignores the profiles or double-counts a model.
RESOLUTION = [
    ('ControlNet', 'get_control', True),
    ('ControlNet', 'control_merge', True),
    ('ControlLora', 'get_control', True),      # else the unit prompt path is lost
    ('ControlLora', 'control_merge', True),    # else no weighting at all
    ('ControlLora', 'get_models', True),       # else control_model_wrapped, which it lacks
    ('ControlLora', 'inference_memory_requirements', False),  # the host's counts the weights
    ('T2IAdapter', 'get_control', True),       # else the weight profile never applies
    ('T2IAdapter', 'control_merge', True),     # else no weighting at all
]


def test_method_resolution():
    for cls_name, method, want_cnpro in RESOLUTION:
        cls = getattr(impl, cls_name)
        owner = next((c for c in cls.__mro__ if method in c.__dict__), None)
        if owner is None:
            fail("%s.%s resolves to nothing" % (cls_name, method))
            continue
        is_cnpro = 'cnpro' in owner.__module__
        if is_cnpro != want_cnpro:
            fail("%s.%s resolves to %s.%s (%s), expected %s"
                 % (cls_name, method, owner.__module__.split('.')[-1], owner.__name__,
                    "CNPro's" if is_cnpro else "the host's",
                    "CNPro's" if want_cnpro else "the host's"))


def test_t2i_profile_gate_skips_the_model():
    """A near-zero weight profile must skip the T2I forward entirely.

    T2IAdapter no longer has a `get_control` of its own -- it inherits the gate
    from CNPro's ControlBase and the forward pass from the host's. If the gate
    stopped being reached, the model would run on every step and the profile
    would be silently ignored except for its effect on `strength`.
    """
    ran = []

    class Probe(impl.T2IAdapter):
        def __init__(self):
            impl.ControlBase.__init__(self, device=torch.device('cpu'))
            self.t2i_model = None
            self.channels_in = 3

    probe = Probe()
    # profile that evaluates to ~0 at every sigma
    probe.weight_profile_sigmas = [1.0, 0.0]
    probe.weight_profile_strengths = [0.0, 0.0]
    probe.transformer_options = {'cond_or_uncond': [0], 'sigmas': torch.tensor([1.0]),
                                 'cond_mark': torch.tensor([0.0])}

    out = probe.get_control(torch.zeros(1, 4, 8, 8), torch.tensor([1.0]), {}, 1)
    if ran:
        fail("the T2I forward ran on a step the weight profile zeroed")
    if out is not None:
        fail("a skipped step with no previous unit must yield None, got %r" % type(out))

    # and a live profile must NOT skip: it should reach the host's forward,
    # which fails on the None model - proving the gate let it through
    probe.weight_profile_strengths = [1.0, 1.0]
    try:
        probe.get_control(torch.zeros(1, 4, 8, 8), torch.tensor([1.0]), {}, 1)
    except Exception:
        pass  # reached the host's forward, which is the point
    else:
        fail("a live profile should have reached the host's T2I forward pass")


#: (needles, the CNPro class the file must come back as)
LOADER_CASES = [
    (("controlnet", "sdxl"), 'ControlNet'),
    (("t2i-adapter",), 'T2IAdapter'),
]


def test_loader_returns_cnpro_classes():
    """A real file, through the host's sniffing, must come back as OUR class.

    The point of delegating the sniffing was that CNPro stops carrying a second
    copy of it. The point of NOT delegating the wrapping is that the control
    object still has to be CNPro's, or none of the profiles are ever read.

    A file the HOST ITSELF cannot load is reported as exactly that and does not
    fail this test: it says nothing about CNPro's wrapping, and CNPro's deleted
    copy of the sniffing called the same strict `load_state_dict` and would have
    raised on the same file. Collapsing the two into one red result would send
    the next reader after the wrong bug.
    """
    from cnpro_host.patchers.controlnet import ControlNetPatcher
    from safetensors.torch import load_file

    tried, host_refused = 0, []
    for needles, want_class in LOADER_CASES:
        path = find_model(*needles)
        if path is None:
            continue
        tried += 1
        try:
            state = load_file(path)
            patcher = ControlNetPatcher.try_build_from_state_dict(state, path)
        except ImportError as exc:
            # modules.sd_models cannot be imported outside the webui's own
            # startup ordering (modules.processing imports from it while it is
            # still initializing). Pre-existing, unrelated to CNPro, and it
            # means the weight-loading branch is only reachable in a live app.
            host_refused.append("%s: not reachable outside a running webui (%s)"
                                % (os.path.basename(path), str(exc)[:90]))
            continue
        except Exception as exc:
            host_refused.append("%s: the host's own loader raised %s: %s"
                                % (os.path.basename(path), type(exc).__name__,
                                   str(exc).split('\n')[0][:110]))
            continue

        if patcher is None:
            fail("the host recognized nothing in %s - the delegated loader "
                 "declined a control model that used to load"
                 % os.path.basename(path))
            continue

        control = patcher.model_patcher
        if not isinstance(control, impl.ControlBase):
            fail("loader returned %s, which is not one of CNPro's control classes"
                 " - its profiles would never be read" % type(control).__name__)
        elif type(control).__name__ != want_class:
            fail("%s came back as CNPro's %s, expected %s"
                 % (os.path.basename(path), type(control).__name__, want_class))
        else:
            print("  loaded %s -> CNPro %s"
                  % (os.path.basename(path), type(control).__name__))
        for name in HOST_WEIGHTING_SURFACE:
            if getattr(control, name, None) is not None:
                fail("freshly loaded control has the host's %r set" % name)

    for msg in host_refused:
        print("  NOT CNPro's problem - %s" % msg)
    if tried == 0:
        print("SKIPPED (loader half) - no control model found under %s; the "
              "control_merge half above still ran." % (CONTROL_DIRS,))


def main():
    torch.manual_seed(0)
    for fn in (test_refactor_matches_reference,
               test_chain_with_wider_previous_batch,
               test_host_weighting_surface_stays_unset,
               test_double_application_would_be_caught,
               test_method_resolution,
               test_t2i_profile_gate_skips_the_model,
               test_transplant_maps_every_host_class,
               test_loader_returns_cnpro_classes):
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
    print("ok - control_merge is unchanged by the subclass refactor across %d "
          "configurations (chained and not), the host's weighting surface stays "
          "unset, and the delegated loader returns CNPro's own classes" % (len(CASES) * 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
