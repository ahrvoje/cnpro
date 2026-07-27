"""Every patcher must DECLARE every capability, and the registry must not swallow.

THE BUG CLASS THIS EXISTS FOR
-----------------------------
Three separate failures in this extension have had the same shape: something was
declared in one place, was supposed to be honoured in another, and when the two
drifted apart NOTHING RAISED. The feature just quietly stopped existing.

  1. The four weight-mask toolbar buttons were injected with one CSS class and
     revealed by a selector naming a different one. Empty NodeList, no error,
     feature absent from the UI. (Pinned by tests/test_toolbar_contract.py.)

  2. `patchers/controlnet.py` called `memory_management.get_computation_dtype()`,
     which does not exist. Every SD1.5/SDXL ControlNet load raised AttributeError
     inside the registry's blanket `except Exception: continue`, and the user was
     told "Recognizing Control Model failed" - i.e. "your file is unsupported".

  3. Capability flags default to False on `CNProModelPatcher`. A patcher that
     forgets one does not fail; it silently loses the feature. `Z-Image` needed
     `masks_via_advanced_weighting = True`, and had it defaulted, per-band masks
     would have degraded to a single union mask with no warning at all.

The common cause is a DEFAULT that means "absent" combined with NO CHECK that the
declaration was made deliberately. This file removes the third instance by
requiring every patcher to state every flag in its own class body - not inherit
it - so "I did not think about this" is impossible to express. It also pins the
registry's three-outcome behaviour from (2).

Run:  <webui python> extensions/forge-neo-cnpro/tests/test_patcher_contract.py
Needs torch + the host on sys.path (it imports the real patchers).
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
sys.path.insert(0, os.path.join(WEBUI, "extensions-builtin", "sd_forge_ipadapter"))
sys.path.insert(0, os.path.join(WEBUI, "extensions-builtin", "sd_forge_controlllite"))

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


#: Every capability a patcher must take a position on. Adding one here forces
#: every existing patcher to be revisited, which is the point: a new capability
#: that silently defaults to "no" on four patchers is the bug this prevents.
REQUIRED_FLAGS = (
    "supports_balance_profile",
    "supports_output_mask",
    "supports_unit_prompt",
    "supports_band_profiles",
    "masks_via_advanced_weighting",
)


def test_every_patcher_declares_every_flag():
    from cnpro_host.registry import _types

    patchers = _types()
    if len(patchers) < 4:
        fail("expected at least 4 registered patchers, found %d" % len(patchers))

    for cls in patchers:
        # Walk the MRO up to (but excluding) CNProModelPatcher: a flag declared
        # on an intermediate base is still a deliberate decision, but one that
        # comes from CNProModelPatcher itself is just the default.
        from cnpro_host.patchers.base import CNProModelPatcher

        own = {}
        for klass in cls.__mro__:
            if klass is CNProModelPatcher:
                break
            own.update({k: v for k, v in vars(klass).items() if k in REQUIRED_FLAGS})

        for flag in REQUIRED_FLAGS:
            if flag not in own:
                fail("%s does not declare %s - it would inherit the default "
                     "(False) and lose that feature with no warning. State it "
                     "explicitly, even when the answer is False."
                     % (cls.__name__, flag))
                continue
            value = own[flag]
            # a property is a legitimate declaration (ControlNetPatcher computes
            # supports_unit_prompt from the wrapped model type)
            if not isinstance(value, (bool, property)):
                fail("%s.%s is %r; capability flags must be bool (or a property "
                     "computing one)" % (cls.__name__, flag, value))


def test_flags_are_documented_on_the_base():
    """Every required flag must exist on the base, so the list cannot rot."""
    from cnpro_host.patchers.base import CNProModelPatcher

    for flag in REQUIRED_FLAGS:
        if not hasattr(CNProModelPatcher, flag):
            fail("CNProModelPatcher has no %s, but this test requires it of every "
                 "patcher - the two lists have drifted" % flag)


def test_registry_distinguishes_declined_from_broken():
    """A loader that RAISES must not be reported as 'unsupported file'."""
    import cnpro_host.registry as registry

    class Declines:
        __name__ = "Declines"

        @staticmethod
        def try_build_from_state_dict(sd, path):
            return None

    class Breaks:
        __name__ = "Breaks"

        @staticmethod
        def try_build_from_state_dict(sd, path):
            raise AttributeError("module has no attribute 'get_computation_dtype'")

    class RefusesOnPurpose:
        __name__ = "RefusesOnPurpose"

        @staticmethod
        def try_build_from_state_dict(sd, path):
            exc = ValueError("this is a v2.x file and CNPro will not guess")
            exc.cnpro_recognised = True
            raise exc

    original = registry._types
    fake_sd = {"a": 1}
    registry_utils_patched = []

    class FakeUtils:
        @staticmethod
        def load_torch_file(path, safe_load=True):
            return dict(fake_sd)

    # `from backend import utils` happens inside the function, so patch the module
    import backend
    real_utils = backend.utils
    backend.utils = FakeUtils
    registry_utils_patched.append(True)

    try:
        # all decline -> "not recognised", and it must RAISE, not return None
        registry._types = lambda: [Declines]
        try:
            out = registry.load_control_model("x.safetensors")
            fail("all-decline returned %r instead of raising" % (out,))
        except RuntimeError as exc:
            if "not a control model" not in str(exc):
                fail("all-decline message is wrong: %s" % exc)
        except Exception as exc:
            fail("all-decline raised %s, expected RuntimeError" % type(exc).__name__)

        # one breaks -> must say it is a CNPro bug, and name the loader
        registry._types = lambda: [Breaks, Declines]
        try:
            registry.load_control_model("x.safetensors")
            fail("a broken loader did not surface as an error")
        except RuntimeError as exc:
            msg = str(exc)
            if "BUG IN CNPRO" not in msg:
                fail("broken-loader message does not distinguish a bug from an "
                     "unsupported file: %s" % msg)
            if "Breaks" not in msg or "get_computation_dtype" not in msg:
                fail("broken-loader message does not name the loader and cause: %s" % msg)
        except Exception as exc:
            fail("broken loader raised %s, expected RuntimeError" % type(exc).__name__)

        # a deliberate refusal must propagate AS ITSELF, immediately
        registry._types = lambda: [RefusesOnPurpose, Declines]
        try:
            registry.load_control_model("x.safetensors")
            fail("a deliberate refusal was swallowed")
        except ValueError as exc:
            if "will not guess" not in str(exc):
                fail("refusal lost its message: %s" % exc)
        except Exception as exc:
            fail("deliberate refusal surfaced as %s, losing its own explanation "
                 "(%s)" % (type(exc).__name__, exc))
    finally:
        registry._types = original
        backend.utils = real_utils


def main():
    for fn in (test_flags_are_documented_on_the_base,
               test_every_patcher_declares_every_flag,
               test_registry_distinguishes_declined_from_broken):
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
    print("ok - every patcher declares all %d capabilities explicitly, and the "
          "registry keeps declined / broken / refused apart" % len(REQUIRED_FLAGS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
