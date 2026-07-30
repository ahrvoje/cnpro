"""Render resources/partition-of-unity.png (+ -dark) for the README.

Run:  python resources/partition-of-unity.py     (needs matplotlib)

3 columns (the three multi-phase families) x 3 rows (two Inputs, five Inputs,
and five Inputs with the convergence switched on).

The CURVES are the family weights - exactly what the weight editor draws and
exactly what each Input runs, since the weights are normalized in
_phase_weight itself and nothing downstream corrects them any more. The SUM
line is then the point of the figure: it is flat at the envelope in every
panel, because that is what a partition of unity means. The inputs
redistribute the envelope and never amplify it.

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
# (row label, Input count, convergence). The third row is the same five-way
# split as the second with the convergence button on: the waves leave their
# partition and slide onto the flat 1/5 share, arriving at CONVERGE_AT. The sum
# line stays where it is, which is the whole point of drawing it here.
CONVERGE = (0.6, 1.0)          # arrives at 60% of the schedule, linearly
ROWS = [
    ("2 inputs", 2, None),
    ("5 inputs", 5, None),
    ("5 inputs\nconverging", 5, CONVERGE),
]

THEMES = {
    "": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e",
             grid="#dedddb", series="#2a78d6", total="#eb6834", sib=0.45),
    "-dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7",
                  grid="#3a3a38", series="#3987e5", total="#d95926", sib=0.6),
}


def weight_curve(family, kappa, index, count, xs, converge=None):
    """One Input's wave - what the editor draws AND what that Input runs.

    There is no share factor left to leave out: every family's weights are its
    own lobe over the sum of all of them, so they already are the shares. The
    cosine column is therefore 2/n tall rather than full height, which is the
    honest picture of a five-way split.
    """
    # phase puts input k's peak at (k + 0.5) / count, so no lobe is cut by the
    # plot edge - a presentation choice, nothing the code does
    phase = math.pi / count
    out = []
    for x in xs:
        theta = 2.0 * math.pi * OSC * x - phase
        out.append(ec._wave_factor(theta, x, index, count, family, kappa, converge))
    return out


def render(suffix, t):
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(10.5, 7.6),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    xs = [i / 400 for i in range(401)]

    for row, (row_label, count, converge) in enumerate(ROWS):
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

            curves = [weight_curve(family, kappa, k, count, xs, converge)
                      for k in range(count)]
            # siblings first, thin - the eye should land on input 1 and the sum
            for k in range(1, count):
                ax.plot(xs, curves[k], color=t["series"], linewidth=1.2, alpha=t["sib"])
            ax.plot(xs, curves[0], color=t["series"], linewidth=2.0)
            # the summed PULL: flat at 1 by construction, no correction applied
            ax.plot(xs, [sum(c[i] for c in curves) for i in range(len(xs))],
                    color=t["total"], linewidth=2.0, linestyle=(0, (5, 3)))

            # the two top-left panels are identical on purpose (Fejer IS the
            # cosine at two inputs); say so, or it reads as a rendering bug
            if row == 0 and family == ec.PHASE_FAMILY_FEJER:
                ax.annotate("identical to the cosine", xy=(0.5, 0.09),
                            xycoords="axes fraction", ha="center",
                            color=t["muted"], fontsize=9, style="italic")

            # the convergence row says where the waves arrive, since the answer
            # is a setting rather than a property of the family
            if converge is not None:
                ax.axvline(converge[0], color=t["muted"], linewidth=1,
                           linestyle=(0, (2, 3)), alpha=0.8)
                if col == 0:
                    ax.annotate("flat A/n from here", ha="center",
                                xy=((converge[0] + 1.0) / 2, 0.62),
                                xycoords=("data", "axes fraction"),
                                color=t["muted"], fontsize=8.5, style="italic")

            if row == 0:
                ax.set_title(title, color=t["ink"], fontsize=12,
                             fontweight="600", pad=10)
            if col == 0:
                ax.set_ylabel(row_label, color=t["ink"], fontsize=11,
                              fontweight="600", labelpad=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.18)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(colors=t["muted"], labelsize=9, length=0)
            if row == len(ROWS) - 1:
                ax.set_xlabel("relative sampling step", color=t["muted"], fontsize=9)

    handles = [
        Line2D([], [], color=t["series"], linewidth=2.0, label="input 1 \u2014 its share of the wave"),
        Line2D([], [], color=t["series"], linewidth=1.2, alpha=t["sib"], label="inputs 2\u2026n \u2014 the same share, shifted"),
        Line2D([], [], color=t["total"], linewidth=2.0, linestyle=(0, (5, 3)), label="their sum \u2014 the envelope, intact"),
    ]
    legend = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                        fontsize=10, bbox_to_anchor=(0.5, -0.005))
    for text in legend.get_texts():
        text.set_color(t["muted"])

    fig.tight_layout(rect=(0, 0.052, 1, 1))
    out = os.path.join(REPO, "resources", f"partition-of-unity{suffix}.png")
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", out)


for suffix, theme in THEMES.items():
    render(suffix, theme)
