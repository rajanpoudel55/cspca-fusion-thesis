# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nbstyle import nbwrap, NB_BLUE, NB_ORANGE, NB_GREEN, NB_RED, NB_PURPLE, NB_GREY
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from ch4_data import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "ch56"); os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
NAVY, BLUE, TEAL, TERRA, PURPLE, GREY, GOLD = "#3A4756", "#5B7A94", "#80A09B", "#B69080", "#8C819A", "#6B7280", "#B8A57C"
LIGHT, GREEN, ROSE, SAND, LILAC = "#EEF1F4", "#EEF2EF", "#F4EEEC", "#F4F1EA", "#F1EFF3"

def save(fig, name): fig.savefig(os.path.join(OUT, name), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08); plt.close(fig)
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)
def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=7, bold=False, lw=1.1, ls="-", tc="#111827"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, zorder=4, linespacing=1.25)
def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, linewidth=lw, color=color, zorder=2, shrinkA=0, shrinkB=0, linestyle=ls))
def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off"); return fig, ax

# 1. information ladder for Random Forest (like-for-like)
def ladder():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1.9, 1.1]})
    ax = axs[0]; lab = ["Imaging\nonly", "Clinical\nonly", "Fully\nautomated", "PI-RADS-\ninformed", "Biopsy-pathway\n(not pre-biopsy)"]; val = [0.613, 0.724, 0.765, 0.884, 0.888]; col = [NB_GREEN, NB_BLUE, NB_PURPLE, NB_RED, NB_ORANGE]
    b = ax.bar(lab, val, color=col, width=0.62, zorder=3); b[4].set_hatch("//"); b[4].set_edgecolor("white")
    for r, v in zip(b, val): ax.text(r.get_x() + r.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(0.5, color=NB_GREY, ls="--", lw=1, zorder=2); ax.set_ylim(0.4, 0.95); ax.set_ylabel("ROC-AUC (Random Forest)", fontsize=8); ax.grid(axis="y", alpha=0.3, zorder=0); clean(ax)
    ax.tick_params(axis="x", labelsize=6.6)
    ax = axs[1]; lab2 = ["Fully\nautomated", "PI-RADS-\ninformed", "Biopsy-\npathway"]; v2 = [0.400, 0.694, 0.882]
    b = ax.bar(lab2, v2, color=[NB_PURPLE, NB_RED, NB_ORANGE], width=0.6, zorder=3); b[2].set_hatch("//"); b[2].set_edgecolor("white")
    for r, v in zip(b, v2): ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Recall (Random Forest)", fontsize=8); ax.grid(axis="y", alpha=0.3, zorder=0); clean(ax); ax.tick_params(axis="x", labelsize=6.6)
    fig.tight_layout(); save(fig, "fig_ladder.png")

# 2. trade-off: this study (derived) against published studies
def tradeoff():
    lit = {"Guo et al.": (57.4, 11.1, (-6, 6), "right"), "Kamecki et al.": (50.7, 3.0, (7, -11), "left"), "Esteban et al. (XGBoost)": (41.77, 10.0, (0, -13), "center")}
    d42 = {k: derived(v) for k, v in T42.items()}; d46 = {k: derived(v) for k, v in T46.items()}
    def pt(dd, k): d = dd[k]; return 100 * d["spec"], 100 * d["fn"] / P
    expert = {"Stacking": ((74.9, 8.2), (7, 3), "left"), "RF": ((85.1, 30.6), (7, 3), "left"), "XGBoost": ((72.6, 8.2), (-8, -11), "right"), "GB": ((87.9, 37.6), (7, 3), "left")}
    fused = {"Stacking": (pt(d42, "Stacking Ensemble"), (7, 3), "left"), "XGBoost": (pt(d42, "Tuned XGBoost"), (-8, -4), "right")}
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for lab, (x, y, off, ha) in lit.items():
        ax.scatter([x], [y], s=60, color="#9CA3AF", marker="s", zorder=3, edgecolor="white")
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=6.6, color=NB_GREY, ha=ha)
    for lab, ((x, y), off, ha) in expert.items():
        ax.scatter([x], [y], s=70, color=NB_RED, marker="D", zorder=4, edgecolor="white")
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=7, color="#8c564b", ha=ha)
    for lab, ((x, y), off, ha) in fused.items():
        ax.scatter([x], [y], s=70, color=NB_BLUE, marker="o", zorder=4, edgecolor="white")
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=7, color=NB_BLUE, ha=ha)
    ax.set_xlim(27, 95); ax.set_ylim(-2, 58); ax.set_xlabel("Unnecessary biopsies avoided (% of patients without csPCa)", fontsize=8.5); ax.set_ylabel("csPCa missed (% of csPCa)", fontsize=8.5)
    ax.grid(alpha=0.3, zorder=0); clean(ax)
    h = [matplotlib.lines.Line2D([0], [0], marker="s", ls="", color="#9CA3AF", label="Published studies", markersize=6),
         matplotlib.lines.Line2D([0], [0], marker="D", ls="", color=NB_RED, label="This study, PI-RADS-informed (pre-biopsy)", markersize=6),
         matplotlib.lines.Line2D([0], [0], marker="o", ls="", color=NB_BLUE, label="This study, fully automated\u2020", markersize=6)]
    ax.legend(handles=h, fontsize=6.8, frameon=True, loc="upper left")
    save(fig, "fig_tradeoff56.png")

# 3. evidence tiers and where the thesis sits
def tiers():
    fig, ax = canvas(7.4, 3.9, 100, 54)
    steps = [("1. Internal", "Single hold-out\npartition: 300 studies\n(about 85 csPCa)", "Done", TEAL, "#E6EEEC"),
             ("2. Cross-site", "Leave-one-centre-out\nthree Dutch centres", "Done", TEAL, "#E6EEEC"),
             ("3. External", "One independent cohort\nlocked model, n = 782", "Done (one cohort)", GOLD, SAND),
             ("4. Prospective", "Silent or live use\nin routine care", "Not done", TERRA, ROSE)]
    for i, (t, a, s, ec, fc) in enumerate(steps):
        x = 2 + i * 24.5; h = 24 + i * 6; top = 3 + h
        ax.add_patch(Rectangle((x, 3), 22.5, h, facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2))
        ax.text(x + 11.25, top - 2.2, t, ha="center", va="top", fontsize=8.2, fontweight="bold", color=NAVY, zorder=3)
        ax.text(x + 11.25, top - 8.5, a, ha="center", va="top", fontsize=6.6, color="#111827", zorder=3, linespacing=1.25)
        ax.add_patch(FancyBboxPatch((x + 2, 5), 18.5, 5.4, boxstyle="round,pad=0,rounding_size=0.8", facecolor="white", edgecolor=ec, linewidth=1.2, zorder=3))
        ax.text(x + 11.25, 7.7, s, ha="center", va="center", fontsize=7, fontweight="bold", color=ec, zorder=4)
    arrow(ax, 4, 51, 96, 51, lw=1.4); ax.text(50, 52.6, "Increasing strength of evidence that a model will generalise", ha="center", fontsize=7.6, fontweight="bold", color=NAVY)
    save(fig, "fig_tiers56.png")

# 4. information available when the biopsy decision is made
def info():
    fig, ax = canvas(7.6, 4.0, 100, 60)
    stages = [("Clinical\nassessment", "age, PSA,\nPSA density, volume"), ("MRI acquired\nand reported", "images,\nPI-RADS score"), ("Biopsy decision", "biopsy or not?"), ("Biopsy and\nhistopathology", "how the diagnosis\nwas obtained")]
    xs = [2, 27, 52, 77]
    for (t, s_), x in zip(stages, xs):
        fc = ROSE if "Biopsy and" in t else (SAND if "decision" in t else LIGHT); ec = TERRA if "Biopsy and" in t else (GOLD if "decision" in t else BLUE)
        box(ax, x, 44, 21, 13, t, fc=fc, ec=ec, fs=7.4, bold=True)
        ax.text(x + 10.5, 41.5, s_, ha="center", va="top", fontsize=6.4, color=GREY, linespacing=1.2)
    for i in range(3): arrow(ax, xs[i] + 21, 50.5, xs[i + 1], 50.5)
    rows = [(27, "Automated pipeline: clinical + imaging features", 2, 48, LIGHT, BLUE, "available before the biopsy decision"),
            (16.5, "+ PI-RADS", 2, 48, LILAC, PURPLE, "available before the biopsy decision")]
    for y, lab, x0, x1, fc, ec, note in rows:
        ax.add_patch(Rectangle((x0, y - 3.4), x1 - x0, 6.8, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2))
        ax.text(x0 + 1.2, y, lab, ha="left", va="center", fontsize=6.6, fontweight="bold", color="#111827", zorder=3)
        ax.text(x1 + 2, y, note, ha="left", va="center", fontsize=6.4, color=ec)
    y = 6
    ax.add_patch(Rectangle((2, y - 3.4), 61, 6.8, facecolor=LILAC, edgecolor=PURPLE, linewidth=1.2, zorder=2))
    ax.add_patch(Rectangle((63, y - 3.4), 35, 6.8, facecolor=ROSE, edgecolor=TERRA, linewidth=1.2, zorder=2))
    ax.text(3.2, y, "+ PI-RADS + biopsy pathway", ha="left", va="center", fontsize=6.6, fontweight="bold", color="#111827", zorder=3)
    ax.text(80.5, y, "known only after the decision", ha="center", va="center", fontsize=6.4, color=TERRA, fontweight="bold", zorder=3)
    save(fig, "fig_info.png")

# 5. deployment pathway
def deploy():
    fig, ax = canvas(7.6, 3.6, 100, 46)
    items = [("1. Confirm\ninputs", "Same variables\nand definitions\nlocally?"), ("2. Recalibrate", "Adjust predicted\nrisk to local\nprevalence"), ("3. Choose\nthreshold", "Set from the cost\nof a missed\ncancer"),
             ("4. Silent\nprospective test", "Run alongside\nroutine care\nwithout acting"), ("5. Monitor", "Drift, calibration\nand subgroup\nperformance"), ("6. Decision\nsupport", "Radiologist and\nclinician stay\nin the loop")]
    w, h, gx = 14, 19, 2.4
    for i, (t, s_) in enumerate(items):
        x = 2 + i * (w + gx); fc = [LIGHT, GREEN, SAND, LILAC, "#E6EEEC", ROSE][i]; ec = [BLUE, TEAL, GOLD, PURPLE, TEAL, TERRA][i]
        box(ax, x, 16, w, h, "", fc=fc, ec=ec)
        ax.text(x + w / 2, 16 + h - 1.6, t, ha="center", va="top", fontsize=6.8, fontweight="bold", linespacing=1.15)
        ax.text(x + w / 2, 16 + 6.6, s_, ha="center", va="center", fontsize=5.7, color=GREY, linespacing=1.2)
        if i < 5: arrow(ax, x + w, 25.5, x + w + gx, 25.5)
    ax.text(50, 8.5, "Steps 1 to 3 use data from the deployment site. Step 4 is the prospective test that this thesis did not perform.", ha="center", fontsize=6.4, color=GREY, style="italic")
    ax.text(50, 42, "Requirements for responsible local use of a locked prediction model", ha="center", fontsize=7.8, fontweight="bold", color=NAVY)
    save(fig, "fig_deploy.png")

# 6. research roadmap
def roadmap():
    fig, ax = canvas(7.6, 4.4, 100, 62)
    cols = [("Near term (within a year)", BLUE, LIGHT, ["Evaluate the locked models on\nfurther independent cohorts\n(PLCO, FastMRI, IMPROD)", "Add resampling-based confidence\nintervals and repeat the key\ncomparisons", "Test the pre-biopsy model\n(PI-RADS, no pathway) on a\nsecond independent cohort", "Test recalibration to the\ndeployment population and\nadd SHAP explanations"]),
            ("Medium term (1 to 3 years)", PURPLE, LILAC, ["Fine-tuned or 3D networks and\nlesion segmentation on\nlarger datasets", "Ordinal outcome across\nISUP grade groups", "Subgroup and fairness analysis\nin more diverse populations", "Federated training across\ninstitutions"]),
            ("Long term (3+ years)", TERRA, ROSE, ["Silent prospective validation\nin a clinical pathway", "Impact study of biopsy\ndecisions and outcomes", "Extension to histopathology\nwith foundation models"])]
    w = 31; gx = 2.5
    for i, (t, ec, fc, its) in enumerate(cols):
        x = 1 + i * (w + gx)
        ax.add_patch(Rectangle((x, 3), w, 56, facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=1))
        ax.text(x + w / 2, 55.5, t, ha="center", va="center", fontsize=7.4, fontweight="bold", color=ec)
        for j, s_ in enumerate(its): box(ax, x + 1.6, 42 - j * 12.6, w - 3.2, 10.4, s_, fc="white", ec=ec, fs=5.9)
        if i < 2: arrow(ax, x + w, 30, x + w + gx, 30, color=GREY)
    save(fig, "fig_roadmap.png")

for _n in ['ladder', 'tradeoff']: globals()[_n] = nbwrap(globals()[_n])

if __name__ == "__main__":
    ladder(); tradeoff(); tiers(); info(); deploy(); roadmap()
    print(sorted(os.listdir(OUT)))
