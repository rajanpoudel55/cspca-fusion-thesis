# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nbstyle import nbwrap, NB_BLUE, NB_ORANGE, NB_GREEN, NB_RED, NB_PURPLE, NB_GREY
from matplotlib.patches import Rectangle
from ch4_data import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "ch4"); os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
NAVY, BLUE, TEAL, TERRA, PURPLE, GREY, GOLD = "#3A4756", "#5B7A94", "#80A09B", "#B69080", "#8C819A", "#6B7280", "#B8A57C"
SHORT = {"Random Forest": "RF", "AdaBoost": "AdaBoost", "Tuned XGBoost": "XGBoost", "XGBoost": "XGBoost", "Gradient Boosting": "GB", "Extra Trees": "ET",
         "SVM": "SVM", "Multi-Layer Perceptron": "MLP", "Linear Discriminant Analysis": "LDA", "Logistic Regression": "LR", "Naive Bayes": "NB",
         "Stacking Ensemble": "Stacking", "k-Nearest Neighbours": "kNN", "Decision Tree": "DT"}

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08); plt.close(fig)
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)

# 1 modality
def modality():
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    lab = ["Imaging only", "Clinical only", "Fused\n(clinical + imaging)"]; val = [0.613, 0.724, 0.765]; col = [NB_GREEN, NB_BLUE, NB_PURPLE]
    b = ax.bar(lab, val, color=col, width=0.55, zorder=3)
    for r, v in zip(b, val): ax.text(r.get_x() + r.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(0.5, color=NB_GREY, ls="--", lw=1, zorder=2); ax.set_xlim(-0.5, 3.1); ax.text(2.4, 0.5, "chance\n(0.5)", fontsize=6.6, color=NB_GREY, ha="left", va="center")
    ax.set_ylim(0.4, 0.82); ax.set_ylabel("ROC-AUC (Random Forest)", fontsize=8); ax.grid(axis="y", alpha=0.3, zorder=0); clean(ax)
    save(fig, "fig_modality.png")

# 2 recall vs AUC
def recall_auc():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    off = {"Random Forest": (5, 6), "AdaBoost": (-6, -10), "Tuned XGBoost": (5, 4), "Gradient Boosting": (4, -9), "Extra Trees": (6, 2), "SVM": (6, 3),
           "Multi-Layer Perceptron": (-4, 8), "Linear Discriminant Analysis": (-7, -11), "Logistic Regression": (-6, 7), "Naive Bayes": (-4, 8),
           "Stacking Ensemble": (-6, 8), "k-Nearest Neighbours": (-4, 8), "Decision Tree": (6, 4)}
    for k, v in T42.items():
        a, p, r, f, u = v; deg = k in ("Extra Trees", "k-Nearest Neighbours")
        ax.scatter([u], [r], s=62, color=NB_ORANGE if deg else NB_BLUE, zorder=3, edgecolor="white", linewidth=0.8)
        dx, dy = off[k]; ax.annotate(SHORT[k], (u, r), xytext=(dx, dy), textcoords="offset points", fontsize=7.2, ha="left" if dx > 0 else "right", color="black")
    ax.axhline(0.5, color="#D1D5DB", lw=0.8, zorder=1)
    ax.set_xlim(0.58, 0.80); ax.set_ylim(-0.02, 0.76)
    ax.set_xlabel("ROC-AUC", fontsize=8.5); ax.set_ylabel("Recall", fontsize=8.5); ax.grid(alpha=0.3, zorder=0); clean(ax)
    ax.annotate("high AUC, recall below 0.05", xy=(0.709, 0.04), xytext=(0.614, 0.20), fontsize=7, color=NB_ORANGE, ha="left",
                arrowprops=dict(arrowstyle="->", color=NB_ORANGE, lw=0.9))
    save(fig, "fig_recall_auc.png")

# 3 heatmaps
def heat(T, name, title_names=None):
    names = sorted(T, key=lambda k: -T[k][4])
    cols = ["Accuracy", "Precision", "Recall", "F1", "Specificity\u2020", "Balanced\naccuracy\u2020", "ROC-AUC"]
    M = []
    for k in names:
        a, p, r, f, u = T[k]; d = derived(T[k]); M.append([a, p, r, f, d["spec"], d["bal"], u])
    M = np.array(M); norm = (M - M.min(0)) / (M.max(0) - M.min(0))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.imshow(norm, cmap="Blues", aspect="auto", vmin=-0.1, vmax=1.25)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.68 else "black")
    ax.set_xticks(range(7)); ax.set_xticklabels(cols, fontsize=7.4); ax.xaxis.tick_top()
    ax.set_yticks(range(len(names))); ax.set_yticklabels([("Stacking" if k == "Stacking Ensemble" else k) for k in names], fontsize=7.4)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True); ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
    save(fig, name)

# 4 SMOTE
def smote():
    names = ["AdaBoost", "Logistic\nRegression", "LDA", "Random\nForest", "kNN"]
    rb = [0.306, 0.647, 0.388, 0.400, 0.035]; rs = [0.659, 0.635, 0.635, 0.365, 0.929]
    ab = [0.758, 0.714, 0.717, 0.765, 0.683]; as_ = [0.716, 0.704, 0.696, 0.758, 0.598]
    x = np.arange(5); w = 0.36
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, b, s, ttl, lo in ((axs[0], rb, rs, "Recall", 0), (axs[1], ab, as_, "ROC-AUC", 0.5)):
        r1 = ax.bar(x - w / 2, b, w, color=NB_BLUE, label="Baseline", zorder=3); r2 = ax.bar(x + w / 2, s, w, color=NB_ORANGE, label="With SMOTE", zorder=3)
        for bars in (r1, r2):
            for r in bars: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012, f"{r.get_height():.3f}", ha="center", fontsize=5.4)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.8); ax.set_title(ttl, fontsize=8.5, fontweight="bold", color="black")
        ax.set_ylim(lo, 1.02 if ttl == "Recall" else 0.85); ax.grid(axis="y", alpha=0.3, zorder=0); clean(ax)
    axs[0].legend(fontsize=7, frameon=True, loc="upper left")
    fig.tight_layout(); save(fig, "fig_smote.png")

# 5 fusion
def fusion():
    names = ["Early fusion\n(Random Forest)", "Late fusion\n(simple average)", "Late fusion\n(learned combiner)", "Intermediate fusion\n(neural network)"]
    acc = [0.730, 0.703, 0.700, 0.687]; rec = [0.400, 0.541, 0.341, 0.647]; auc = [0.765, 0.726, 0.727, 0.733]
    x = np.arange(4); w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for off, v, c, l in ((-w, acc, NB_BLUE, "Accuracy"), (0, rec, NB_ORANGE, "Recall"), (w, auc, NB_GREEN, "ROC-AUC")):
        b = ax.bar(x + off, v, w, color=c, label=l, zorder=3)
        for r in b: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.3f}", ha="center", fontsize=6.4)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.2); ax.set_ylim(0, 0.95); ax.grid(axis="y", alpha=0.3, zorder=0); clean(ax)
    ax.legend(fontsize=7.4, frameon=True, ncol=3, loc="upper right"); save(fig, "fig_fusion_res.png")

# 6 imaging variants
def variants():
    names = ["Baseline: single-slice T2W (fused, RF)", "Multi-slice averaging (5 slices, fused)", "Multiparametric (T2W+ADC+HBV, fused)",
             "End-to-end CNN, frozen backbone", "End-to-end CNN, fine-tuned (last 10 layers)", "End-to-end CNN, frozen + augmentation"]
    acc = [0.730, 0.747, 0.760, 0.689, 0.689, 0.611]; rec = [0.400, 0.424, 0.494, 0.238, 0.333, 0.583]; auc = [0.765, 0.760, 0.770, 0.601, 0.629, 0.656]
    y = np.arange(6)[::-1]; h = 0.26
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    for off, v, c, l in ((h, acc, NB_BLUE, "Accuracy"), (0, rec, NB_ORANGE, "Recall"), (-h, auc, NB_GREEN, "ROC-AUC")):
        b = ax.barh(y + off, v, h, color=c, label=l, zorder=3)
        for r in b: ax.text(r.get_width() + 0.008, r.get_y() + r.get_height() / 2, f"{r.get_width():.3f}", va="center", fontsize=6.2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7); ax.set_xlim(0, 0.95); ax.grid(axis="x", alpha=0.3, zorder=0); clean(ax)
    ax.legend(fontsize=7.2, frameon=True, loc="lower right"); save(fig, "fig_variants.png")

# 7 dumbbell
def dumbbell():
    mp = {"Tuned XGBoost": "XGBoost"}
    order = sorted(T42, key=lambda k: T46[mp.get(k, k)][4])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.3), sharey=True)
    for ax, idx, ttl, xlim in ((axs[0], 4, "ROC-AUC", (0.55, 0.95)), (axs[1], 2, "Recall", (0.0, 1.02))):
        for i, k in enumerate(order):
            a = T42[k][idx]; b = T46[mp.get(k, k)][idx]
            ax.plot([a, b], [i, i], color="#cccccc", lw=2, zorder=1)
            ax.scatter([a], [i], s=40, color=NB_BLUE, zorder=3, label="Fused features" if i == 0 else None)
            ax.scatter([b], [i], s=40, color=NB_ORANGE, zorder=3, label="+ PI-RADS and biopsy pathway" if i == 0 else None)
        ax.set_xlim(*xlim); ax.set_title(ttl, fontsize=8.5, fontweight="bold", color="black"); ax.grid(axis="x", alpha=0.3, zorder=0); clean(ax)
    axs[0].set_yticks(range(13)); axs[0].set_yticklabels([("Stacking" if k == "Stacking Ensemble" else k.replace("Tuned ", "")) for k in order], fontsize=7)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=True, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig_dumbbell.png")

# 8 confusion matrices
def cm_panel(ax, tp, fn, fp, tn, title, sub):
    M = np.array([[tp, fn], [fp, tn]]); tot = M.sum(1, keepdims=True)
    colors = np.array([[1.0, 0.25], [0.45, 0.8]])
    ax.imshow(colors, cmap="Blues", vmin=0, vmax=1.4, aspect="equal")
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.07, f"{M[i, j]}", ha="center", va="center", fontsize=12, fontweight="bold", color="white" if colors[i, j] > 0.7 else "black")
            ax.text(j, i + 0.22, f"{100 * M[i, j] / tot[i, 0]:.0f}% of row", ha="center", va="center", fontsize=6.4, color="white" if colors[i, j] > 0.7 else NB_GREY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Flagged\n(positive)", "Not flagged\n(negative)"], fontsize=6.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["csPCa", "No csPCa"], fontsize=6.8)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title + "\n", fontsize=7.6, fontweight="bold", color="black", pad=22)
    ax.text(0.5, -0.12, sub, transform=ax.transAxes, ha="center", fontsize=6.4, color=NB_GREY)

def confusion3():
    a = derived(T42["Stacking Ensemble"]); b = derived(T46["Stacking Ensemble"]); c = derived(T46["Random Forest"])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.1))
    cm_panel(axs[0], a["tp"], a["fn"], a["fp"], a["tn"], "Stacking, fully\nautomated", "recall 0.682, precision 0.453")
    cm_panel(axs[1], b["tp"], b["fn"], b["fp"], b["tn"], "Stacking,\nbiopsy-pathway", "recall 0.941, precision 0.567")
    cm_panel(axs[2], c["tp"], c["fn"], c["fp"], c["tn"], "Random Forest,\nbiopsy-pathway", "recall 0.882, precision 0.600")
    fig.tight_layout(); save(fig, "fig_confusion3.png")

# 9 LOCO
def loco():
    cent = ["RUMC", "PCNN", "ZGT"]; clin = [0.685, 0.623, 0.707]; img = [0.578, 0.516, 0.629]; fus = [0.732, 0.585, 0.779]
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for off, v, c, l in ((-w, clin, NB_BLUE, "Clinical only"), (0, img, NB_GREEN, "Imaging only"), (w, fus, NB_PURPLE, "Fused")):
        b = ax.bar(x + off, v, w, color=c, label=l, zorder=3)
        for r in b: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, f"{r.get_height():.3f}", ha="center", fontsize=6.6)
    ax.axhline(0.5, color=NB_GREY, ls="--", lw=1, zorder=2); ax.set_xlim(-0.55, 3.0); ax.text(2.52, 0.5, "chance", fontsize=6.6, color=NB_GREY, ha="left", va="center")
    ax.set_xticks(x); ax.set_xticklabels([f"{c} held out" for c in cent], fontsize=8); ax.set_ylim(0.4, 0.82)
    ax.set_ylabel("ROC-AUC", fontsize=8); ax.grid(axis="y", alpha=0.3, zorder=0); clean(ax); ax.legend(fontsize=7.4, frameon=True, ncol=3, loc="upper left")
    save(fig, "fig_loco_res.png")

# 10 thresholds
def thresholds():
    a = counts(.506, .538); b = counts(.318, .659)
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 3.1))
    cm_panel(axs[0], a[0], a[1], a[2], a[3], "Default threshold 0.50", "recall 0.506, precision 0.538")
    cm_panel(axs[1], b[0], b[1], b[2], b[3], "Accuracy-optimised threshold 0.74", "recall 0.318, precision 0.659")
    fig.tight_layout(); save(fig, "fig_thresholds.png")

# 11 external
def external():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]; lab = ["Accuracy", "Precision", "Recall", "ROC-AUC"]; val = [0.669, 0.720, 0.696, 0.745]
    b = ax.bar(lab, val, color=[NB_BLUE, NB_ORANGE, NB_GREEN, NB_PURPLE], width=0.58, zorder=3)
    for r, v in zip(b, val): ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(0.575, color=NB_GREY, ls="--", lw=1, zorder=2); ax.set_xlim(-0.5, 4.6); ax.text(3.42, 0.575, "prevalence\n0.575", fontsize=6.4, color=NB_GREY, ha="left", va="center")
    ax.set_ylim(0, 0.9); ax.grid(axis="y", alpha=0.3, zorder=0); clean(ax); ax.set_title("Reported metrics (n = 782)", fontsize=8.2, fontweight="bold", color="black")
    tp, fn, fp, tn = EXT_C; cm_panel(axs[1], tp, fn, fp, tn, "Classification outcomes", "exact counts, 450 csPCa and 332 without")
    fig.tight_layout(); save(fig, "fig_external.png")

# 12 summary of AUC across programme
def summary():
    rows = [("Imaging-only features (Random Forest)", 0.613, "auto"), ("Clinical-only features (Random Forest)", 0.724, "auto"), ("Fused features (Random Forest)", 0.765, "auto"),
            ("Best fully automated model (tuned XGBoost)", 0.774, "auto"), ("Best PI-RADS-informed model (Stacking)", 0.887, "expert"), ("Best biopsy-pathway model (Stacking; not pre-biopsy)", 0.893, "path"),
            ("Held-out PCNN, fused features", 0.585, "loco"), ("Held-out RUMC, fused features", 0.732, "loco"), ("Held-out ZGT, fused features", 0.779, "loco"),
            ("Locked Stacking model, independent cohort", 0.745, "ext")]
    cols = {"auto": NB_BLUE, "expert": NB_RED, "path": "#8c564b", "loco": NB_PURPLE, "ext": NB_ORANGE}; mk = {"auto": "o", "expert": "D", "path": "X", "loco": "s", "ext": "^"}
    fig, ax = plt.subplots(figsize=(7.0, 4.3)); n = len(rows)
    for i, (lab, v, t) in enumerate(rows):
        y = n - 1 - i; ax.plot([0.5, v], [y, y], color="#D1D5DB", lw=0.8, zorder=1)
        ax.scatter([v], [y], s=52, color=cols[t], marker=mk[t], zorder=3, edgecolor="white", linewidth=0.8); ax.text(v + 0.008, y, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.2); ax.set_xlim(0.5, 0.96)
    ax.set_xlabel("ROC-AUC", fontsize=8.5); ax.grid(axis="x", alpha=0.3, zorder=0); clean(ax)
    h = [matplotlib.lines.Line2D([0], [0], marker=mk[k], ls="", color=cols[k], markersize=6, label=l) for k, l in
         (("auto", "Fully automated, hold-out test"), ("expert", "PI-RADS-informed, hold-out test"), ("path", "Biopsy-pathway, hold-out test (not pre-biopsy)"), ("loco", "Cross-centre (leave-one-centre-out)"), ("ext", "External cohort (locked model)"))]
    ax.legend(handles=h, fontsize=6.8, frameon=True, loc="upper center", bbox_to_anchor=(0.42, -0.14), ncol=2)
    save(fig, "fig_summary_auc.png")

for _n in ['modality', 'recall_auc', 'heat', 'smote', 'fusion', 'variants', 'dumbbell', 'cm_panel', 'confusion3', 'loco', 'thresholds', 'external', 'summary']: globals()[_n] = nbwrap(globals()[_n])

if __name__ == "__main__":
    modality(); recall_auc(); heat(T42, "fig_heat42.png"); heat(T46, "fig_heat46.png"); smote(); fusion(); variants(); dumbbell()
    confusion3(); loco(); thresholds(); external(); summary()
    print(sorted(os.listdir(OUT)))
