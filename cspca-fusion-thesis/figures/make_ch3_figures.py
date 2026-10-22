# -*- coding: utf-8 -*-
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "ch3")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
NAVY, BLUE, TEAL, TERRA, PURPLE, GREY, GOLD = "#3A4756", "#5B7A94", "#80A09B", "#B69080", "#8C819A", "#6B7280", "#B8A57C"
LIGHT, LIGHT2, GREEN, ROSE, SAND = "#EEF1F4", "#F5F7FA", "#EEF2EF", "#F4EEEC", "#F4F1EA"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=7.2, bold=False, tc="#111827", lw=1.1, rs=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}", linewidth=lw,
                                edgecolor=ec, facecolor=fc, zorder=3, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=4, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, lw=1.2, color=NAVY, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, linewidth=lw,
                                 color=color, zorder=2, shrinkA=0, shrinkB=0, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off")
    return fig, ax


# ------------------------------------------------------------ 0. pipeline overview (Figure 3.1)
def pipeline():
    fig, ax = canvas(7.6, 5.5, 100, 78)
    ax.text(50, 76.5, "Clinical and imaging inputs are fused once and pass through one shared development and validation pathway",
            ha="center", va="center", fontsize=7.6, fontweight="bold", color="#111827")
    cx = [13, 38, 63, 88]                                     # four fixed columns; every arrow is vertical
    bw = 22
    box(ax, 2, 66, 96, 6, "Data: PI-CAI training cohort (1,500 studies, 1,476 patients)   |   external cohort: Prostate-MRI-US-Biopsy (782 patients)", fc=LIGHT2, ec=NAVY, fs=6.6)
    box(ax, cx[0] - bw / 2, 49, bw, 12, "Clinical variables\nage, PSA, PSA density,\nprostate volume\n(all configurations)", fc=LIGHT, ec=BLUE, fs=6.4)
    box(ax, cx[1] - bw / 2, 49, bw, 12, "T2-weighted MRI\ncentral axial slice\n(all configurations)", fc=LIGHT, ec=BLUE, fs=6.4)
    box(ax, cx[2] - bw / 2, 49, bw, 12, "PI-RADS score\nradiologist-assigned\n(PI-RADS-informed and\nbiopsy-pathway only)", fc=SAND, ec=GOLD, fs=6.4, ls="--")
    box(ax, cx[3] - bw / 2, 49, bw, 12, "Biopsy pathway\nhistopathology type\n(biopsy-pathway only;\nnot pre-biopsy)", fc=ROSE, ec=TERRA, fs=6.4, ls="--")
    for x in cx: arrow(ax, x, 66, x, 61)
    box(ax, cx[1] - bw / 2, 37, bw, 8, "ResNet50, frozen\nImageNet weights\n2,048 features", fc=LIGHT, ec=BLUE, fs=6.2)
    box(ax, cx[1] - bw / 2, 28, bw, 7, "PCA to 100 components\n(about 71% of the variance)", fc=LIGHT, ec=BLUE, fs=6.2)
    arrow(ax, cx[1], 49, cx[1], 45); arrow(ax, cx[1], 37, cx[1], 35)
    box(ax, 2, 17, 96, 7, "Early (feature-level) fusion: one concatenated feature vector per study", fc=GREEN, ec=TEAL, fs=6.8, bold=True)
    arrow(ax, cx[0], 49, cx[0], 24); arrow(ax, cx[1], 28, cx[1], 24)
    arrow(ax, cx[2], 49, cx[2], 24, ls="--"); arrow(ax, cx[3], 49, cx[3], 24, ls="--")
    box(ax, 2, 7.5, 96, 6, "Model development: thirteen algorithms with grid-search tuning, including a stacking ensemble", fc="#F1EFF3", ec=PURPLE, fs=6.8, bold=True)
    arrow(ax, 50, 17, 50, 13.5)
    vx = [17, 50, 83]
    for x, t, ls_ in zip(vx, ("Internal hold-out\n80/20 split of the studies", "Leave-one-centre-out\nRUMC, ZGT, PCNN", "External validation\nlocked model, 782 patients"), ("-", "-", "--")):
        box(ax, x - 15, 0, 30, 5, t, fc=LIGHT, ec=BLUE, fs=6.4, ls=ls_)
        arrow(ax, x, 7.5, x, 5)
    save(fig, "fig_pipeline.png")


# ------------------------------------------------------------ 1. programme
def programme():
    fig, ax = canvas(7.6, 4.0, 100, 56)
    items = [
        ("1. Modality\ncontribution", "clinical, imaging,\nfused features", "4.2", LIGHT, BLUE),
        ("2. Algorithm\ncomparison", "13 algorithms on\nfused data", "4.3", LIGHT, BLUE),
        ("3. Robustness\nexperiments", "SMOTE, fusion types,\nimaging variants", "4.4 to 4.6", SAND, GOLD),
        ("4. PI-RADS-informed\nand biopsy-pathway", "added separately;\npathway not pre-biopsy", "4.7", SAND, GOLD),
        ("5. Leave-one-\ncentre-out", "three PI-CAI\ncentres", "4.8", "#E6EBF0", BLUE),
        ("6. Threshold\nsensitivity", "default versus\naccuracy-optimised", "4.9", "#E6EBF0", BLUE),
        ("7. External\nvalidation", "locked model,\nindependent cohort", "4.10", GREEN, TEAL),
        ("8. Utility and\ncalibration", "decision and\ncalibration curves", "4.11", GREEN, TEAL),
    ]
    w, h, gx, x0 = 20.5, 19, 5.5, 0.75
    for i, (t, sub, sec, fc, ec) in enumerate(items):
        row, col = divmod(i, 4)
        x = x0 + col * (w + gx); y = 30 if row == 0 else 3
        box(ax, x, y, w, h, "", fc=fc, ec=ec)
        ax.text(x + w / 2, y + h - 2.2, t, fontsize=7.4, fontweight="bold", ha="center", va="top", color="#111827", linespacing=1.2)
        ax.text(x + w / 2, y + 7.6, sub, fontsize=6.3, ha="center", va="center", color=GREY, linespacing=1.2)
        ax.text(x + w / 2, y + 2.2, f"Section {sec}", fontsize=6.4, ha="center", va="center", color=ec, fontweight="bold")
        if col < 3: arrow(ax, x + w, y + h / 2, x + w + gx, y + h / 2, color=ec)
    xr = x0 + 3 * (w + gx) + w / 2; xl = x0 + w / 2
    ax.plot([xr, xr], [30, 25.5], color=GREY, lw=1.1, zorder=1)
    ax.plot([xr, xl], [25.5, 25.5], color=GREY, lw=1.1, zorder=1)
    arrow(ax, xl, 25.5, xl, 22.2, color=GREY)
    ax.text(50, 53.2, "Each stage changes one element of the pipeline and holds the others fixed", ha="center", fontsize=7.6, color=NAVY, fontweight="bold")
    save(fig, "fig_programme.png")


# ------------------------------------------------------------ 2. cohort flow
def cohort():
    fig, ax = canvas(7.6, 5.0, 100, 68)
    cols = [(1, "TCGA-PRAD (GDC portal)", TERRA), (35, "PI-CAI public dataset", BLUE), (69, "Prostate-MRI-US-Biopsy (TCIA)", TEAL)]
    for x, title, ec in cols:
        ax.text(x + 14, 65.5, title, ha="center", fontsize=8, fontweight="bold", color=ec)
    H = 12.5; ys = [46, 29.5, 13]
    spec = [
        [("500 patients\n1,239 clinical rows\n(five GDC files)", ROSE, TERRA, False),
         ("Cross-referenced with TCIA:\nonly 14 patients\nhave MRI", ROSE, TERRA, False),
         ("Insufficient for a\nfused model: not used\nfor modelling", "#F3F4F6", GREY, False)],
        [("1,500 studies\n1,476 patients\nThree Dutch centres", LIGHT, BLUE, False),
         ("Outcome case_csPCa:\n425 positive (28.3%)\n1,075 negative", LIGHT, BLUE, False),
         ("Modelling cohort:\ntraining and\ninternal test", "#E6EBF0", BLUE, True)],
        [("1,150 patients with\nbiopsy pathology\ndata", GREEN, TEAL, False),
         ("Complete clinical\nvariables and a T2W\nMRI series available", GREEN, TEAL, False),
         ("782 patients:\nexternal cohort,\nprevalence 57.5%", "#E6EEEC", TEAL, True)],
    ]
    for ci, (x, _, ec) in enumerate(cols):
        for ri in range(3):
            t, fc, ec2, bold = spec[ci][ri]
            box(ax, x, ys[ri], 28, H, t, fc=fc, ec=ec2, bold=bold, fs=6.8)
            if ri < 2: arrow(ax, x + 14, ys[ri], x + 14, ys[ri + 1] + H, color=ec2)
    arrow(ax, 29, ys[2] + H / 2, 35, ys[2] + H / 2, color=GREY, ls="--")
    ax.text(32, ys[2] + H + 0.8, "imaging\nsource\nchanged", fontsize=5.8, color=GREY, ha="center", va="bottom", linespacing=1.1)
    arrow(ax, 63, ys[2] + H / 2, 69, ys[2] + H / 2, color=GREY, ls="--")
    ax.text(66, ys[2] + H + 0.8, "locked\nmodel\napplied", fontsize=5.8, color=GREY, ha="center", va="bottom", linespacing=1.1)
    ax.text(50, 6.0, "Study counts are as reported in Sections 3.2.1 to 3.2.3", ha="center", fontsize=6.6, color=GREY, style="italic")
    save(fig, "fig_cohort.png")


# ------------------------------------------------------------ 3. partition / leakage-safe workflow
def partition():
    fig, ax = canvas(7.6, 5.0, 100, 72)
    box(ax, 27, 61.5, 46, 10, "PI-CAI modelling cohort: 1,500 studies (425 csPCa)\nLabel-free preprocessing fitted here:\nimputation medians, imaging scaler, PCA", fc=LIGHT, ec=BLUE, bold=True, fs=6.4)
    ax.text(75, 66.8, "study-level stratified\n80/20 split, fixed seed", fontsize=6.4, ha="left", va="center", color=GREY, style="italic", linespacing=1.2)
    arrow(ax, 40, 61.5, 27, 56.2); arrow(ax, 60, 61.5, 66, 56.2)
    # panels
    box(ax, 2, 17, 50, 39, "", fc=LIGHT2, ec=BLUE)
    ax.text(27, 54.2, "Training partition: 1,200 studies\n(about 340 csPCa)", fontsize=7.2, fontweight="bold", ha="center", va="top", color=NAVY, linespacing=1.25)
    box(ax, 5, 36.5, 44, 9.5, "Fit on training data only:\nfeature standardisation", fc=GREEN, ec=TEAL, fs=6.5)
    box(ax, 5, 25.5, 44, 9.5, "Five-fold stratified cross-validation\n(GridSearchCV, ROC-AUC) for tuning", fc=SAND, ec=GOLD, fs=6.5)
    box(ax, 5, 19, 44, 5, "SMOTE (experiments only), then final model fit", fc="#F1EFF3", ec=PURPLE, fs=6.2)
    box(ax, 56, 17, 20, 39, "", fc=LIGHT2, ec=BLUE)
    ax.text(66, 54.2, "Test partition:\n300 studies\n(about 85 csPCa)", fontsize=7.0, fontweight="bold", ha="center", va="top", color=NAVY, linespacing=1.25)
    box(ax, 58, 31, 16, 10, "Transform only\n(no refitting)", fc=GREEN, ec=TEAL, fs=6.3)
    box(ax, 58, 19.5, 16, 9, "Predict and\nevaluate", fc="#E6EBF0", ec=BLUE, fs=6.3)
    arrow(ax, 66, 31, 66, 29.2, color=NAVY)
    box(ax, 80, 17, 19, 39, "", fc=ROSE, ec=TERRA, ls="--")
    ax.text(89.5, 54.2, "External cohort\n(782 patients)", fontsize=7.0, fontweight="bold", ha="center", va="top", color=NAVY, linespacing=1.25)
    box(ax, 82, 31, 15, 10, "Transform with\nlocked scaler\nand PCA", fc=GREEN, ec=TEAL, fs=6.0)
    box(ax, 82, 19.5, 15, 9, "Locked model\npredicts", fc="#E6EBF0", ec=BLUE, fs=6.3)
    arrow(ax, 89.5, 31, 89.5, 29.2, color=NAVY)
    arrow(ax, 52, 40, 56, 40, color=TEAL, ls="--"); arrow(ax, 76, 40, 80, 40, color=GREY, ls="--")
    ax.text(50, 11.2, "Outcome labels of the test partition and the external cohort are never used in fitting.", fontsize=7.2, ha="center", color=NAVY, fontweight="bold")
    ax.text(50, 6.2, "Imputation medians, imaging scaler and PCA were fitted on all 1,500 studies before the split (no labels used);\nthe fused-feature scaler and the models use training data only.", fontsize=6.4, ha="center", color=GREY, style="italic", linespacing=1.3)
    save(fig, "fig_partition.png")


# ------------------------------------------------------------ 4. imaging pipeline
def imaging():
    fig, ax = canvas(7.6, 3.5, 100, 40)
    steps = [
        ("T2-weighted\nMRI volume", "loaded with\nSimpleITK", LIGHT, BLUE),
        ("Central axial\nslice", "one 2D image\nper patient", LIGHT, BLUE),
        ("8-bit scaling,\nresize to\n224 x 224", "grey to three\nchannels", LIGHT, BLUE),
        ("ResNet50\n(frozen,\nImageNet)", "final layer removed,\nglobal average pooling", "#F1EFF3", PURPLE),
        ("2,048-D\nfeature vector", "per patient", "#F1EFF3", PURPLE),
        ("PCA to 100\ncomponents", "fitted on training\ndata only", GREEN, TEAL),
    ]
    w, gx = 13.2, 3.9
    for i, (t, s, fc, ec) in enumerate(steps):
        x = 1 + i * (w + gx)
        box(ax, x, 18, w, 12, t, fc=fc, ec=ec, fs=6.8)
        ax.text(x + w / 2, 14.2, s, fontsize=5.9, ha="center", va="top", color=GREY, linespacing=1.2)
        if i < len(steps) - 1: arrow(ax, x + w, 24, x + w + gx, 24, color=ec)
    ax.text(1, 36.5, "Image preparation", fontsize=7.4, fontweight="bold", color=BLUE)
    ax.plot([1, 2 * (w + gx) + w - gx + 1.5 - 4.5], [35, 35], color=BLUE, lw=0.8)
    ax.text(1 + 3 * (w + gx), 36.5, "Feature extraction", fontsize=7.4, fontweight="bold", color=PURPLE)
    ax.plot([1 + 3 * (w + gx), 1 + 5 * (w + gx) + w - 1], [35, 35], color=PURPLE, lw=0.8)
    ax.text(1 + 5 * (w + gx), 36.5, "Reduction", fontsize=7.4, fontweight="bold", color=TEAL)
    ax.plot([1 + 5 * (w + gx), 1 + 5 * (w + gx) + w], [35, 35], color=TEAL, lw=0.8)
    ax.text(50, 3.2, "Applied identically to the PI-CAI and external T2-weighted series; the external series are transformed with the already-fitted PCA.",
            fontsize=6.2, ha="center", color=GREY, style="italic")
    save(fig, "fig_imaging.png")


# ------------------------------------------------------------ 5. fusion architectures as implemented
def fusion():
    fig, ax = canvas(7.6, 4.4, 120, 60)
    def title(x, t): ax.text(x, 57, t, ha="center", fontsize=7.8, fontweight="bold", color=NAVY)
    # early
    title(19, "Early fusion (primary)")
    box(ax, 2, 42, 15, 8, "Clinical\n(4 features)", fc=LIGHT, ec=BLUE, fs=6.6)
    box(ax, 21, 42, 15, 8, "Imaging PCA\n(100 features)", fc=LIGHT, ec=TEAL, fs=6.6)
    arrow(ax, 9.5, 42, 15, 36); arrow(ax, 28.5, 42, 23, 36)
    box(ax, 8, 28, 22, 8, "Concatenated vector\n(+ PI-RADS, pathway)", fc=GREEN, ec=TEAL, fs=6.1)
    arrow(ax, 19, 28, 19, 22)
    box(ax, 8, 10, 22, 12, "Single classifier\n(13 algorithms)", fc="#F1EFF3", ec=PURPLE, fs=6.6)
    arrow(ax, 19, 10, 19, 5.5); ax.text(19, 3.2, "probability", ha="center", fontsize=6.4, color=GREY)
    # late
    title(60, "Late fusion")
    box(ax, 42, 42, 15, 8, "Clinical\n(4 features)", fc=LIGHT, ec=BLUE, fs=6.6)
    box(ax, 63, 42, 15, 8, "Imaging PCA\n(100 features)", fc=LIGHT, ec=TEAL, fs=6.6)
    arrow(ax, 49.5, 42, 49.5, 36.5); arrow(ax, 70.5, 42, 70.5, 36.5)
    box(ax, 42, 28, 15, 8.5, "Clinical-only\nmodel", fc="#F1EFF3", ec=PURPLE, fs=6.4)
    box(ax, 63, 28, 15, 8.5, "Imaging-only\nmodel", fc="#F1EFF3", ec=PURPLE, fs=6.4)
    arrow(ax, 49.5, 28, 56, 22); arrow(ax, 70.5, 28, 64, 22)
    box(ax, 46, 10, 28, 12, "Combine probabilities:\nsimple average or learned\nlogistic regression", fc=GREEN, ec=TEAL, fs=6.1)
    arrow(ax, 60, 10, 60, 5.5); ax.text(60, 3.2, "probability", ha="center", fontsize=6.4, color=GREY)
    # intermediate
    title(101, "Intermediate fusion")
    box(ax, 84, 42, 15, 8, "Clinical\n(4 features)", fc=LIGHT, ec=BLUE, fs=6.6)
    box(ax, 104, 42, 15, 8, "Imaging PCA\n(100 features)", fc=LIGHT, ec=TEAL, fs=6.6)
    arrow(ax, 91.5, 42, 91.5, 36.5); arrow(ax, 111.5, 42, 111.5, 36.5)
    box(ax, 84, 28, 15, 8.5, "Clinical\nhidden layers", fc="#F1EFF3", ec=PURPLE, fs=6.2)
    box(ax, 104, 28, 15, 8.5, "Imaging\nhidden layers", fc="#F1EFF3", ec=PURPLE, fs=6.2)
    arrow(ax, 91.5, 28, 98, 22); arrow(ax, 111.5, 28, 105, 22)
    box(ax, 88, 10, 28, 12, "Concatenate, then\nshared layers to a\nsingle output unit", fc=GREEN, ec=TEAL, fs=6.1)
    arrow(ax, 102, 10, 102, 5.5); ax.text(102, 3.2, "probability", ha="center", fontsize=6.4, color=GREY)
    save(fig, "fig_fusion3.png")


# ------------------------------------------------------------ 6. stacking
def stacking():
    fig, ax = canvas(7.2, 3.5, 100, 50)
    box(ax, 1, 16, 17, 16, "Fused feature\nvector\n(104 to 105+\nfeatures)", fc=LIGHT, ec=BLUE, fs=6.6)
    ax.text(39, 45.2, "Base learners", fontsize=7.4, fontweight="bold", ha="center", color=PURPLE)
    for n, y in [("Random Forest", 33), ("AdaBoost", 19.5), ("Gradient Boosting", 6)]:
        box(ax, 28, y, 22, 9, n, fc="#F1EFF3", ec=PURPLE, fs=6.9)
    for yy, ty in [(37.5, 29.5), (24, 24), (10.5, 18.5)]:
        arrow(ax, 18, 24, 28, yy, color=GREY)
        arrow(ax, 50, yy, 60, ty, color=PURPLE)
    box(ax, 60, 14.5, 17, 19, "Out-of-fold\npredicted\nprobabilities\nof the three\nlearners", fc=SAND, ec=GOLD, fs=6.3)
    arrow(ax, 77, 24, 82, 24, color=NAVY)
    box(ax, 82, 15, 17, 18, "Logistic\nregression\nmeta-learner", fc=GREEN, ec=TEAL, fs=6.8)
    ax.text(90.5, 12.2, "final csPCa\nprobability", fontsize=6.2, ha="center", va="top", color=GREY, linespacing=1.15)
    save(fig, "fig_stacking.png")


# ------------------------------------------------------------ 7. LOCO design
def loco():
    fig, ax = canvas(7.4, 3.9, 100, 54)
    ax.text(50, 51, "Leave-one-centre-out validation (study counts)", ha="center", fontsize=7.8, fontweight="bold", color=NAVY)
    cent = {"RUMC": 800, "ZGT": 350, "PCNN": 350}
    rounds = [("Round 1", ["ZGT", "PCNN"], "RUMC"), ("Round 2", ["RUMC", "PCNN"], "ZGT"), ("Round 3", ["RUMC", "ZGT"], "PCNN")]
    tot = 1500.0; X0, W = 14, 82
    for i, (r, train, test) in enumerate(rounds):
        y = 36 - i * 13.5
        ax.text(1, y + 4.2, r, fontsize=7.2, fontweight="bold", color=NAVY, va="center")
        x = X0
        for c in ["RUMC", "ZGT", "PCNN"]:
            w = W * cent[c] / tot
            is_test = (c == test)
            box(ax, x, y, w - 0.8, 8.4, f"{c}\n{cent[c]}", fc=ROSE if is_test else LIGHT, ec=TERRA if is_test else BLUE, fs=6.9, bold=is_test)
            x += w
        n_tr = sum(cent[c] for c in train)
        ax.text(X0 + W + 0.5, y + 4.2, f"train {n_tr:,}\ntest {cent[test]:,}", fontsize=6.2, va="center", color=GREY, linespacing=1.15)
    ax.add_patch(Rectangle((X0, 1.2), 4, 3, facecolor=LIGHT, edgecolor=BLUE, lw=1)); ax.text(X0 + 5.2, 2.7, "training centres", fontsize=6.6, va="center")
    ax.add_patch(Rectangle((X0 + 26, 1.2), 4, 3, facecolor=ROSE, edgecolor=TERRA, lw=1)); ax.text(X0 + 31.2, 2.7, "held-out centre (test)", fontsize=6.6, va="center")
    ax.text(X0 + 58, 2.7, "Repeated for clinical-only, imaging-only and fused features", fontsize=6.2, va="center", color=GREY, style="italic")
    save(fig, "fig_loco.png")


# ------------------------------------------------------------ 8. locked-model external validation
def locked():
    fig, ax = canvas(7.6, 4.6, 100, 62)
    box(ax, 1, 9, 44, 47, "", fc=LIGHT2, ec=BLUE)
    ax.text(23, 53.4, "Development phase\n(PI-CAI training data only)", fontsize=7.4, fontweight="bold", ha="center", va="top", color=NAVY, linespacing=1.2)
    L = [("Fit imputation values and\nStandardScaler", GREEN, TEAL), ("Fit PCA on ResNet50 features\n(100 components)", GREEN, TEAL),
         ("Train and tune the\nStacking classifier", "#F1EFF3", PURPLE)]
    for i, (t, fc, ec) in enumerate(L):
        y = 36 - i * 9.5
        box(ax, 4, y, 38, 7.4, t, fc=fc, ec=ec, fs=6.5)
        if i < 2: arrow(ax, 23, y, 23, y - 2.1)
    arrow(ax, 23, 17.5, 23, 15.9)
    box(ax, 4, 10.5, 38, 5.4, "LOCK: scaler + PCA + classifier fixed", fc=SAND, ec=GOLD, fs=6.3, bold=True)
    arrow(ax, 45, 13.2, 53, 13.2, color=NAVY, lw=1.6)
    ax.text(49, 15.6, "locked", fontsize=6.2, ha="center", color=GREY)
    box(ax, 53, 9, 46, 47, "", fc=ROSE, ec=TERRA, ls="--")
    ax.text(76, 53.4, "Evaluation phase\n(external cohort, n = 782)", fontsize=7.4, fontweight="bold", ha="center", va="top", color=NAVY, linespacing=1.2)
    R = [("Harmonise variables: age imputed,\nUCLA score used as PI-RADS", LIGHT, BLUE), ("Transform with the fitted scaler\nand PCA (no refitting)", GREEN, TEAL),
         ("Predict with the locked model\n(default threshold)", "#F1EFF3", PURPLE), ("Accuracy, precision, recall, ROC-AUC,\ndecision and calibration curves", "#E6EBF0", BLUE)]
    for i, (t, fc, ec) in enumerate(R):
        y = 38.2 - i * 8.6
        box(ax, 56, y, 40, 6.9, t, fc=fc, ec=ec, fs=6.3)
        if i < 3: arrow(ax, 76, y, 76, y - 1.7)
    ax.text(50, 4.2, "No retraining, parameter adjustment or threshold recalibration on external data", fontsize=7.0, ha="center", color=TERRA, fontweight="bold")
    save(fig, "fig_locked.png")


# ------------------------------------------------------------ 9. evaluation framework
def evalfw():
    fig, ax = canvas(7.6, 3.9, 100, 52)
    box(ax, 38, 41, 24, 8, "Predicted probability\nfor each patient", fc=LIGHT, ec=BLUE, bold=True, fs=7)
    cols = [
        (1, "Ranking", "ROC-AUC", "Does the model rank patients\nwith csPCa above those\nwithout, at any threshold?", BLUE, LIGHT),
        (26, "Classification", "Accuracy, precision,\nrecall, F1-score", "How many cases are right,\nand which errors occur\nat the chosen threshold?", PURPLE, "#F1EFF3"),
        (51, "Calibration", "Calibration curve", "Do predicted risks match\nobserved frequencies?", TEAL, GREEN),
        (76, "Clinical utility", "Decision curve\n(net benefit)", "Does using the model beat\nbiopsying all or none?", GOLD, SAND),
    ]
    for x, t, m, q, ec, fc in cols:
        arrow(ax, 50, 41, x + 11.5, 33.5, color=ec)
        box(ax, x, 6, 23, 27.5, "", fc=fc, ec=ec)
        ax.text(x + 11.5, 31.6, t, fontsize=7.2, fontweight="bold", ha="center", va="top", color=ec, wrap=True)
        ax.text(x + 11.5, 23.2, m, fontsize=7.0, ha="center", va="top", color="#111827", linespacing=1.2)
        ax.text(x + 11.5, 14.2, q, fontsize=5.9, ha="center", va="top", color=GREY, linespacing=1.2)
    save(fig, "fig_evalfw.png")


if __name__ == "__main__":
    pipeline(); programme(); cohort(); partition(); imaging(); fusion(); stacking(); loco(); locked(); evalfw()
    print(sorted(os.listdir(OUT)))
