"""Render resources/partition-of-unity.png (+ -dark) for the README.

Run:  python resources/partition-of-unity.py     (needs matplotlib)

3 columns (the three multi-phase families) x 2 rows (two and five Inputs).

The CURVES are the raw family weights - exactly what the weight editor draws,
so the picture and the tool have the same layout. The editor deliberately shows
no share factor: it draws the shape of each Input's wave, and the share is
applied later, per generation, in scripts/cnpro.py. The SUM line is therefore
the other quantity - what the unit actually pulls once those shares are in -
which is flat at the envelope for every family. Keeping both on one plot is the
point of the figure: the inputs redistribute the envelope and never amplify it.

The weights come from lib_cnpro.external_code so the picture cannot drift from
the code it documents.
"""
import math
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _external_code():
    m = types.ModuleType("modules"); m.__path__ = []; sys.modules["modules"] = m
    sh = types.ModuleType("modules.shared")
    sh.opts = types.SimpleNamespace(data={})
    sh.cmd_opts = types.SimpleNamespace(cnpro_loglevel="INFO")
    sys.modules["modules.shared"] = sh; m.shared = sh
    ap = types.ModuleType("modules.api"); ap.__path__ = []
    am = types.ModuleType("modules.api.api"); ap.api = am
    sys.modules["modules.api"] = ap; sys.modules["modules.api.api"] = am
    from lib_cnpro import external_code
    return external_code


ec = _external_code()

KAPPA = 3.0
OSC = 1.0                      # one hand-over cycle across the step range
COLUMNS = [
    ("Cosine", ec.PHASE_FAMILY_COSINE, 0.0),
    ("Fej\u00e9r", ec.PHASE_FAMILY_FEJER, 0.0),
    ("von Mises   \u03ba = 3", ec.PHASE_FAMILY_MISES, KAPPA),
]
ROWS = [2, 5]

THEMES = {
    "": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e",
             grid="#dedddb", series="#2a78d6", total="#eb6834", sib=0.45),
    "-dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7",
                  grid="#3a3a38", series="#3987e5", total="#d95926", sib=0.6),
}


def weight_curve(family, kappa, index, count, xs):
    """One Input's wave, as the weight editor draws it - NO share factor.

    The editor's `evaluate` is envelope * waveFactor and stops there; the share
    is a generation-time correction (scripts/cnpro.py), not part of the shape.
    Scaling here instead would squash the cosine column to a 2/n-tall smear and
    the docs would stop looking like the tool.
    """
    # phase puts input k's peak at (k + 0.5) / count, so no lobe is cut by the
    # plot edge - a presentation choice, nothing the code does
    phase = math.pi / count
    out = []
    for x in xs:
        theta = 2.0 * math.pi * OSC * x - phase
        out.append(ec._phase_weight(theta, index, count, family, kappa))
    return out


def share_of(family, count):
    """The generation-time share, inverting the family's summing constant."""
    return 2.0 / count if family == ec.PHASE_FAMILY_COSINE else 1.0


def render(suffix, t):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.3), sharex=True, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    xs = [i / 400 for i in range(401)]

    for row, count in enumerate(ROWS):
        for col, (title, family, kappa) in enumerate(COLUMNS):
            ax = axes[row][col]
            ax.set_facecolor(t["surface"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(t["grid"])
                ax.spines[side].set_linewidth(1)
            ax.grid(True, color=t["grid"], linewidth=0.8, alpha=0.7)
            ax.set_axisbelow(True)

            curves = [weight_curve(family, kappa, k, count, xs) for k in range(count)]
            share = share_of(family, count)
            # siblings first, thin - the eye should land on input 1 and the sum
            for k in range(1, count):
                ax.plot(xs, curves[k], color=t["series"], linewidth=1.2, alpha=t["sib"])
            ax.plot(xs, curves[0], color=t["series"], linewidth=2.0)
            # the summed PULL: the curves above are shapes, this is what the
            # unit delivers once the share factor is in
            ax.plot(xs, [sum(c[i] for c in curves) * share for i in range(len(xs))],
                    color=t["total"], linewidth=2.0, linestyle=(0, (5, 3)))

            # the two top-left panels are identical on purpose (Fejer IS the
            # cosine at two inputs); say so, or it reads as a rendering bug
            if row == 0 and family == ec.PHASE_FAMILY_FEJER:
                ax.annotate("identical to the cosine", xy=(0.5, 0.09),
                            xycoords="axes fraction", ha="center",
                            color=t["muted"], fontsize=9, style="italic")

            if row == 0:
                ax.set_title(title, color=t["ink"], fontsize=12,
                             fontweight="600", pad=10)
            if col == 0:
                ax.set_ylabel(f"{count} inputs", color=t["ink"], fontsize=11,
                              fontweight="600", labelpad=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.18)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(colors=t["muted"], labelsize=9, length=0)
            if row == 1:
                ax.set_xlabel("relative sampling step", color=t["muted"], fontsize=9)

    handles = [
        Line2D([], [], color=t["series"], linewidth=2.0, label="input 1 \u2014 as the editor draws it"),
        Line2D([], [], color=t["series"], linewidth=1.2, alpha=t["sib"], label="inputs 2\u2026n \u2014 the same wave, shifted"),
        Line2D([], [], color=t["total"], linewidth=2.0, linestyle=(0, (5, 3)), label="their summed pull \u2014 the envelope, intact"),
    ]
    legend = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                        fontsize=10, bbox_to_anchor=(0.5, -0.005))
    for text in legend.get_texts():
        text.set_color(t["muted"])

    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = os.path.join(REPO, "resources", f"partition-of-unity{suffix}.png")
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", out)


for suffix, theme in THEMES.items():
    render(suffix, theme)
