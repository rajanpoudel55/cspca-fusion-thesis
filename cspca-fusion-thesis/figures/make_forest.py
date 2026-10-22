import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
GREY, BLUE, GOLD, PURPLE, TERRA = "#6B7280", "#5B7A94", "#B8A57C", "#8C819A", "#B69080"
rows = [("Rule: PSA density \u2265 0.15", .662, .603, .715, BLUE), ("Logistic regression: PSA density alone", .745, .684, .805, BLUE),
        ("Logistic regression: age, log PSA, volume", .757, .699, .816, BLUE), ("Random Forest: clinical variables", .724, .664, .783, BLUE),
        ("Random Forest: fully automated", .759, .699, .814, PURPLE), ("Stacking: fully automated", .755, .697, .811, PURPLE),
        ("Rule: PI-RADS \u2265 3", .779, .743, .814, GOLD), ("Rule: PI-RADS \u2265 4", .828, .787, .864, GOLD), ("Rule: PI-RADS \u2265 4, or 3 with PSA density \u2265 0.15", .801, .761, .837, GOLD),
        ("PI-RADS score alone", .860, .817, .899, GOLD), ("Logistic regression: PI-RADS + PSA density", .878, .837, .915, GOLD),
        ("Logistic regression: age, log PSA, volume + PI-RADS", .881, .840, .919, GOLD), ("Stacking: PI-RADS-informed (locked)", .884, .845, .917, TERRA)]
fig, ax = plt.subplots(figsize=(7.2, 4.6)); n = len(rows)
for i, (lab, v, lo, hi, c) in enumerate(rows):
    y = n - 1 - i; ax.plot([lo, hi], [y, y], color=c, lw=2.2, zorder=2, solid_capstyle="round"); ax.scatter([v], [y], s=46, color=c, zorder=3, edgecolor="white", linewidth=0.8)
    ax.text(0.935, y, f"{v:.3f} ({lo:.3f}\u2013{hi:.3f})", va="center", fontsize=6.6, color="#111827")
ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7); ax.set_xlim(0.58, 1.07); ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("ROC-AUC on the hold-out partition (95% bootstrap interval)", fontsize=8); ax.grid(axis="x", alpha=0.3, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
h = [Line2D([0], [0], marker="o", ls="", color=c, markersize=6, label=l) for c, l in ((BLUE, "Clinical baselines"), (PURPLE, "Fully automated models"), (GOLD, "PI-RADS baselines"), (TERRA, "PI-RADS-informed model"))]
ax.legend(handles=h, fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=2)
fig.tight_layout(); os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "ch4"), exist_ok=True); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "ch4", "fig_forest.png"), dpi=300); print("saved")
