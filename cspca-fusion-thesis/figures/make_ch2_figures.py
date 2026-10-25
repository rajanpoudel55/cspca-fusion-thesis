# -*- coding: utf-8 -*-
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nbstyle import nbwrap, NB_BLUE, NB_ORANGE, NB_GREEN, NB_RED, NB_PURPLE, NB_GREY
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "ch2")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"

NAVY, BLUE, TEAL, TERRA, PURPLE, GREY = "#3A4756", "#5B7A94", "#80A09B", "#B69080", "#8C819A", "#6B7280"
LIGHT, LIGHT2, GREEN = "#EEF1F4", "#F5F7FA", "#EEF2EF"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=7.5, bold=False, tc="#111827", lw=1.1, rs=0.05):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=4, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, lw=1.2, color=NAVY, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color, zorder=2, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# ---------------------------------------------------------------- timeline
def timeline():
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.set_xlim(2007.5, 2027.2); ax.set_ylim(-3.3, 3.3); ax.axis("off")
    ax.plot([2008, 2027], [0, 0], color="black", lw=1.6, zorder=1)
    ev = [  # year, label, level (+up/-down), colour
        (2009, "ERSPC and PLCO\nscreening trials", 1.5, NB_ORANGE),
        (2015, "PI-RADS v2", -1.5, NB_BLUE),
        (2017, "PROMIS;\nERSPC/PLCO\nreconciled", 1.5, NB_GREEN),
        (2018, "PRECISION", -2.5, NB_GREEN),
        (2019, "MRI-FIRST;\nPI-RADS v2.1", 2.55, NB_GREEN),
        (2020, "Combined-biopsy\ntrial", -1.3, NB_GREEN),
        (2022, "PI-CAI public\ndataset", 1.4, NB_PURPLE),
        (2024, "PI-CAI confirmatory\nstudy; TRIPOD+AI;\nEAU guidelines", -2.55, NB_PURPLE),
        (2025, "VISION\nmeta-analysis", 1.5, NB_GREEN),
        (2026, "Domain\ngeneralisation\nstudy", -1.35, NB_PURPLE),
    ]
    for yr, lab, lv, col in ev:
        ax.plot([yr, yr], [0, lv * 0.78], color=col, lw=1.0, zorder=1)
        ax.scatter([yr], [0], s=34, color=col, zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(yr, lv * 0.84, f"{lab}", ha="center", va="bottom" if lv > 0 else "top",
                fontsize=6.9, color="black", linespacing=1.2)
        ax.text(yr, -0.34 if lv > 0 else 0.34, str(yr), ha="center", va="top" if lv > 0 else "bottom",
                fontsize=6.9, color=NB_GREY, fontweight="bold")
    handles = [matplotlib.lines.Line2D([0], [0], marker="o", ls="", color=c, label=l, markersize=5)
               for c, l in [(NB_ORANGE, "Screening"), (NB_BLUE, "Imaging standard"), (NB_GREEN, "MRI diagnostic evidence"),
                            (NB_PURPLE, "AI, datasets and reporting")]]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=True, ncol=4, bbox_to_anchor=(0.0, -0.04))
    save(fig, "fig_timeline.png")


# ---------------------------------------------------------------- pathways
def pathways():
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    rows = [
        (47, "Traditional pathway", ["Raised PSA or\nabnormal DRE", "Systematic\nTRUS biopsy", "Histopathology"], "#F3F4F6", GREY),
        (26, "MRI-based pathway", ["Raised PSA", "Pre-biopsy MRI\n(PI-RADS score)", "Targeted \u00b1\nsystematic biopsy", "Histopathology"], LIGHT, BLUE),
        (4, "Machine learning-assisted pathway (this thesis)", ["Clinical variables\n(age, PSA, PSAD,\nvolume)", "MRI features and\nPI-RADS score", "Model estimate of\ncsPCa risk", "Biopsy decision\nsupport"], GREEN, TEAL),
    ]
    for y, title, items, fc, ec in rows:
        ax.text(1, y + 11.2, title, fontsize=8, fontweight="bold", color=ec)
        n = len(items); w = 20 if n >= 4 else 24; gap = 4.5
        x = 1
        for i, t in enumerate(items):
            box(ax, x, y, w, 9.5, t, fc=fc, ec=ec, fs=7)
            if i < n - 1:
                arrow(ax, x + w, y + 4.75, x + w + gap, y + 4.75, color=ec)
            x += w + gap
    ax.text(26, 22.6, "A negative MRI may allow biopsy to be avoided (PROMIS: up to 27% of men)", fontsize=6.8, color=BLUE, va="center", style="italic")
    ax.text(2, -2.8, "TRUS: transrectal ultrasound; DRE: digital rectal examination; PSAD: PSA density.", fontsize=6.4, color=GREY)
    save(fig, "fig_pathways.png")


# ---------------------------------------------------------------- PROMIS
def promis():
    import numpy as np
    labels = ["Sensitivity", "Specificity", "Positive\npredictive value", "Negative\npredictive value"]
    mri = [93, 41, 51, 89]; trus = [48, 96, 90, 74]
    x = np.arange(4); w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    b1 = ax.bar(x - w / 2, mri, w, color=NB_BLUE, label="Multiparametric MRI", zorder=3)
    b2 = ax.bar(x + w / 2, trus, w, color=NB_ORANGE, label="TRUS-guided biopsy", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 108); ax.set_ylabel("Percent", fontsize=8.5); ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=True, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_promis.png")


# ---------------------------------------------------------------- fusion strategies
def fusion():
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    ax.set_xlim(0, 120); ax.set_ylim(0, 50); ax.axis("off")
    def panel(x0, title, mode):
        ax.text(x0 + 17, 47, title, ha="center", fontsize=8.2, fontweight="bold", color=NAVY)
        if mode == "early":
            box(ax, x0, 31, 15, 7, "Clinical\nfeatures", fc=LIGHT, ec=BLUE, fs=6.8)
            box(ax, x0 + 19, 31, 15, 7, "Imaging\nfeatures", fc=LIGHT, ec=TEAL, fs=6.8)
            arrow(ax, x0 + 7.5, 31, x0 + 14, 25); arrow(ax, x0 + 26.5, 31, x0 + 20, 25)
            box(ax, x0 + 8, 18, 18, 7, "Concatenate", fc=GREEN, ec=TEAL, fs=7)
            arrow(ax, x0 + 17, 18, x0 + 17, 13)
            box(ax, x0 + 6, 5.5, 22, 7.5, "Single\nclassifier", fc="#F1EFF3", ec=PURPLE, fs=7)
        elif mode == "inter":
            box(ax, x0, 31, 15, 7, "Clinical\nbranch", fc=LIGHT, ec=BLUE, fs=6.8)
            box(ax, x0 + 19, 31, 15, 7, "Imaging\nbranch", fc=LIGHT, ec=TEAL, fs=6.8)
            arrow(ax, x0 + 7.5, 31, x0 + 14, 25); arrow(ax, x0 + 26.5, 31, x0 + 20, 25)
            box(ax, x0 + 5, 18, 24, 7, "Merged internal\nrepresentation", fc=GREEN, ec=TEAL, fs=6.8)
            arrow(ax, x0 + 17, 18, x0 + 17, 13)
            box(ax, x0 + 6, 5.5, 22, 7.5, "Shared output\nlayers", fc="#F1EFF3", ec=PURPLE, fs=7)
        else:
            box(ax, x0, 31, 15, 7, "Clinical\nfeatures", fc=LIGHT, ec=BLUE, fs=6.8)
            box(ax, x0 + 19, 31, 15, 7, "Imaging\nfeatures", fc=LIGHT, ec=TEAL, fs=6.8)
            arrow(ax, x0 + 7.5, 31, x0 + 7.5, 26); arrow(ax, x0 + 26.5, 31, x0 + 26.5, 26)
            box(ax, x0, 19, 15, 7, "Model A", fc="#F1EFF3", ec=PURPLE, fs=7)
            box(ax, x0 + 19, 19, 15, 7, "Model B", fc="#F1EFF3", ec=PURPLE, fs=7)
            arrow(ax, x0 + 7.5, 19, x0 + 14, 13); arrow(ax, x0 + 26.5, 19, x0 + 20, 13)
            box(ax, x0 + 4, 5.5, 26, 7.5, "Combine predictions\n(average or meta-learner)", fc=GREEN, ec=TEAL, fs=6.5)
    panel(2, "Early (feature-level)", "early")
    panel(43, "Intermediate", "inter")
    panel(84, "Late (decision-level)", "late")
    ax.text(60, 0.8, "Output in each case: estimated probability of clinically significant prostate cancer",
            ha="center", fontsize=6.8, color=GREY, style="italic")
    save(fig, "fig_fusion.png")


# ---------------------------------------------------------------- trade-off scatter
def tradeoff():
    pts = [("Guo et al.\n(derived from counts)", 57.4, 11.1, NB_BLUE, (-6, 9), "right"), ("Kamecki et al.", 50.7, 3.0, NB_GREEN, (6, 6), "left"), ("Esteban et al.\n(XGBoost)", 41.77, 10.0, NB_PURPLE, (-6, 6), "right")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for lab, x, y, c, off, ha in pts:
        ax.scatter([x], [y], s=70, color=c, zorder=3, edgecolor="white", linewidth=1)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=7.5, color="black", ha=ha, linespacing=1.1)
    ax.set_xlim(30, 70); ax.set_ylim(0, 15)
    ax.set_xlabel("Unnecessary biopsies avoided (% of patients without csPCa)", fontsize=8.5); ax.set_ylabel("csPCa missed (% of csPCa)", fontsize=8.5)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.3, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    save(fig, "fig_tradeoff.png")


def auc_plot():
    data = [
        ("Arafa et al. 2024 (n = 528; up to)", 0.97, "internal"),
        ("This thesis, best configuration (internal hold-out)", 0.893, "thesis_int"),
        ("Ji et al. 2024 (n = 1,196; training set)", 0.890, "internal"),
        ("Mo et al. 2022 (n = 315; PHI + mpMRI nomogram)", 0.882, "internal"),
        ("Du et al. 2025 (n = 928; test set)", 0.881, "internal"),
        ("Roest et al. 2024, early fusion (internal)", 0.874, "internal"),
        ("Wo\u017anicki et al. 2020 (n = 191; csPCa vs insignificant)", 0.844, "internal"),
        ("Bacchetti et al. 2025 (n = 538)", 0.822, "internal"),
        ("Fully automated DL model (n = 5,215), external test", 0.86, "external"),
        ("Roest et al. 2024, early fusion (external)", 0.77, "external"),
        ("This thesis, locked model (external)", 0.745, "thesis_ext"),
        ("Domain generalisation, best method (out-of-domain)", 0.678, "ood"),
        ("Domain generalisation, baseline (out-of-domain)", 0.613, "ood"),
    ]
    colours = {"internal": NB_BLUE, "external": NB_GREEN, "ood": NB_PURPLE, "thesis_int": NB_ORANGE, "thesis_ext": NB_ORANGE}
    marks = {"internal": "o", "external": "s", "ood": "^", "thesis_int": "o", "thesis_ext": "s"}
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    n = len(data)
    for i, (lab, v, k) in enumerate(data):
        y = n - 1 - i
        ax.plot([0.5, v], [y, y], color="#D1D5DB", lw=0.8, zorder=1)
        ax.scatter([v], [y], s=48, color=colours[k], marker=marks[k], zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(v + 0.008, y, f"{v:.3f}" if v not in (0.97, 0.86, 0.77) else f"{v:.2f}", va="center", fontsize=7, color="black")
    ax.set_yticks(range(n)); ax.set_yticklabels([d[0] for d in data][::-1], fontsize=7.3)
    ax.set_xlim(0.5, 1.02); ax.set_xlabel("Reported ROC-AUC", fontsize=8.5); ax.tick_params(axis="x", labelsize=8)
    ax.axvspan(0.5, 0.5, color="white")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    h = [matplotlib.lines.Line2D([0], [0], marker="o", ls="", color=NB_BLUE, label="Internal or development data", markersize=6),
         matplotlib.lines.Line2D([0], [0], marker="s", ls="", color=NB_GREEN, label="External validation", markersize=6),
         matplotlib.lines.Line2D([0], [0], marker="^", ls="", color=NB_PURPLE, label="Out-of-domain evaluation", markersize=6),
         matplotlib.lines.Line2D([0], [0], marker="o", ls="", color=NB_ORANGE, label="This thesis", markersize=6)]
    ax.legend(handles=h, fontsize=7, frameon=True, loc="lower right")
    save(fig, "fig_auc.png")


# ---------------------------------------------------------------- validation tiers
def tiers():
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
    steps = [
        ("1. Internal", "Random split or\ncross-validation", "Tests overfitting to\nthe development\nsample", LIGHT, GREY),
        ("2. Cross-site", "A whole centre\nis withheld", "Tests robustness to\nan unseen site in the\nsame dataset", "#E6EBF0", BLUE),
        ("3. External", "Independent cohort,\nmodel locked", "Tests transport to a\ndifferent institution\nand population", "#E6EEEC", TEAL),
        ("4. Prospective", "Applied in\nroutine care", "Tests performance\nand impact in real\nclinical use", GREEN, "#7E9A88"),
    ]
    for i, (t, a, b, fc, ec) in enumerate(steps):
        x = 2 + i * 24.5; h = 27 + i * 7; top = 3 + h
        ax.add_patch(Rectangle((x, 3), 22.5, h, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2))
        ax.text(x + 11.25, top - 2.2, t, ha="center", va="top", fontsize=8.3, fontweight="bold", color=NAVY, zorder=3)
        ax.text(x + 11.25, top - 8.0, a, ha="center", va="top", fontsize=7, color="#111827", zorder=3, linespacing=1.25)
        ax.text(x + 11.25, top - 16.5, b, ha="center", va="top", fontsize=6.6, color=GREY, zorder=3, linespacing=1.25)
    arrow(ax, 4, 58.0, 96, 58.0, lw=1.4, color=NAVY)
    ax.text(50, 59.6, "Increasing strength of evidence that a model will generalise", ha="center", fontsize=7.8, color=NAVY, fontweight="bold")
    save(fig, "fig_tiers.png")


for _n in ['timeline', 'promis', 'tradeoff', 'auc_plot']: globals()[_n] = nbwrap(globals()[_n])

if __name__ == "__main__":
    timeline(); pathways(); promis(); fusion(); tradeoff(); auc_plot(); tiers()
    print("figures:", sorted(os.listdir(OUT)))
