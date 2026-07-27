"""Optional features -- present if the environment happens to support them.

DESIGN RULE FOR THIS PACKAGE
----------------------------
Nothing in here may be required by anything outside it. Every feature is
registered through `register_all()` inside its own try/except, reports its
outcome once at import, and CNPro carries on regardless. A feature failing here
must never degrade the widget, and must never raise into the host's startup.

That is stricter than "wrap it in a try" -- it also means:

  * no module outside `optional/` may import from `optional/`;
  * an optional feature may not add a unit field, a UI component the layout
    depends on, or an entry in any dataclass;
  * the client side must degrade on its own. The Topaz toolbar buttons, for
    instance, are hidden unless the status endpoint answers `available: true`,
    so an unregistered backend is indistinguishable from an uninstalled tool.

Why this rule: availability here is RARE and environmental (a licensed desktop
app on one machine), so the common case is "absent". Absent has to be the
boring, silent, zero-cost path.

ADDING AN OPTIONAL FEATURE
--------------------------
Write a module here exposing `register_routes(demo, app)` (or any callable), add
one entry to `_FEATURES`, and document what makes it optional.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("CNPro")

# name -> (module attribute path, human description)
_FEATURES = [
    ("topaz", "Topaz Photo AI canvas tools (needs a local licensed tpai.exe)"),
]


def serving_routes(app, register, label):
    """Run `register(app)` and make the routes it adds actually reachable.

    THE BUG THIS FIXES (found on ForgeNeo, 2026-07-26): routes registered from
    an `on_app_started` callback are appended to the router AFTER the UI
    framework has mounted its own catch-all. The catch-all matches first, so
    every request to the new route gets a plain 404 -- identical to what you see
    if registration never happened at all. It cost a full debugging round here
    precisely because the two failure modes are indistinguishable from outside.

    Promoting the new routes to the front of the router makes "registered" mean
    "served". Returns the number of routes added.

    Any extension route added this late needs this; it is not Topaz-specific,
    which is why it lives here as a helper rather than inline.
    """
    before = len(getattr(app, "routes", ()))
    register(app)
    added = len(app.routes) - before
    if added > 0:
        new = app.routes[-added:]
        del app.routes[-added:]
        app.routes[0:0] = new
    logger.info("CNPro: %s -> %d route(s) registered and promoted", label, added)
    return added


def register_all(demo, app):
    """Register every optional feature that can be registered. Never raises."""
    for name, description in _FEATURES:
        try:
            mod = __import__(f"cnpro_host.optional.{name}", fromlist=[name])
            serving_routes(app, lambda a: mod.register_routes(demo, a), f"optional/{name}")
        except Exception as exc:
            # INFO, not WARNING: absence is the expected case, not a fault.
            logger.info(
                "CNPro optional feature '%s' not available (%s) - %s",
                name,
                type(exc).__name__,
                description,
            )


def register_settings(shared):
    """Register optional features' settings. Never raises.

    Kept separate from route registration because settings are read at UI build
    time while routes are attached after the app exists.
    """
    try:
        shared.opts.add_option(
            "cnpro_topaz_scale",
            shared.OptionInfo(
                2,
                "CNPro: Topaz upscale factor",
                section=("cnpro", "CNPro"),
            ).info("used by the canvas Topaz buttons; takes effect immediately"),
        )
    except Exception as exc:
        logger.info("CNPro: optional settings not registered (%s)", type(exc).__name__)
