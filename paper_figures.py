"""
논문 그림 (Fig 1~6, Supplement) - 결과 폴더(results)의 실험 결과로 그린다.
저장: <IMR>/figures/FigN.tif (500 dpi, LZW) + 미리보기 FigN.jpg
사용 예) python paper_figures.py --root "D:/4. 논문/조성진/mapping Evans blue spots/IMR" fig1 fig2
"""
import warnings

warnings.filterwarnings("ignore")
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from scipy import ndimage as ndi
from skimage import measure

import limb_mapping as LM
import limb_validation as V

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "legend.fontsize": 6.5, "pdf.fonttype": 42})
MM = 1 / 25.4
FULL = 190 * MM
VIEW_EN = {"medial": "Medial (ulnar)", "lateral": "Lateral (radial)", "dorsal": "Dorsal", "palmar": "Palmar"}
VIEW_COL = {"medial": "#1b9e77", "lateral": "#d95f02", "dorsal": "#7570b3", "palmar": "#e7298a"}
METH_COL = {"axis": "#bdbdbd", "similarity": "#969696", "affine": "#636363", "tps_arc": "#2b8cbe",
            "tps_dtw": "#08519c", "affine_oracle": "#fdae6b", "tps_oracle": "#e6550d"}
ROOT = RES = FIG = None
EXAMPLE = ("palmar", "C7")       # 예시 그림에 쓰는 신경·부위 (paper_experiments.EXAMPLE 과 같게)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".tif"), dpi=500, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(FIG, name + ".jpg"), dpi=200, pil_kwargs={"quality": 92})
    plt.close(fig)
    print("saved", name)


def label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def tpl_bg(tpl):
    """영문 그림용 템플릿 배경: 흰 바탕 + 50 px 격자 (글자 없음)"""
    W, H = tpl["canvas"]
    img = np.full((H, W, 3), 255, np.uint8)
    for x in range(0, W, LM.GRID):
        img[:, x] = 225
    for y in range(0, H, LM.GRID):
        img[y, :] = 225
    return img


def hcbar(fig, ax, im, text):
    cax = ax.inset_axes([0.25, -0.1, 0.5, 0.04])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5, length=2, pad=1)
    cb.set_label(text, fontsize=5.5, labelpad=1)
    return cb


def crop_tissue(img_rgb, alpha, pad=10):
    yy, xx = np.nonzero(alpha > 127)
    y0, y1, x0, x1 = max(0, yy.min() - pad), yy.max() + pad, max(0, xx.min() - pad), xx.max() + pad
    return img_rgb[y0:y1, x0:x1], (x0, y0)


def outline(ax, tpl, **kw):
    ax.plot(*LM.outline_xy(tpl), **({"color": "k", "lw": 0.8} | kw))


def placeholder(ax, text):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ls="--", lw=0.8, ec="gray"))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=7, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


# ----------------------------------------------------------------------------
def fig1():
    """원본 사진의 이질성"""
    ds = V.dataset()
    het = pd.read_csv(os.path.join(RES, "hetero.csv"))
    fig = plt.figure(figsize=(FULL, 128 * MM))
    gsA = fig.add_gridspec(4, 4, hspace=0.28, wspace=0.06, left=0.035, right=0.555, top=0.95, bottom=0.04)
    gsR = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.42, left=0.63, right=0.99, top=0.91, bottom=0.14,
                           height_ratios=[1.15, 1])
    # A: 예시 그룹 하나(한 신경·한 부위)의 사진 4장 (동물마다 크기·방향이 다름)
    ex_view, ex_seg = EXAMPLE
    pick = [d for d in ds if d[0] == ex_view and d[1] == ex_seg][:4]
    gsA2 = gsA[:, :].subgridspec(2, 2, hspace=0.18, wspace=0.06)
    for k, (_, sgm, _, p) in enumerate(pick):
        ax = fig.add_subplot(gsA2[k // 2, k % 2])
        rgba = LM.load_rgba(p)
        ax.imshow(LM.on_white(rgba))
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color("#bbbbbb")
        h = het[het.image == os.path.relpath(p, LM.DATA_DIR)].iloc[0]
        ax.set_title(f"Rat {k + 1}  ({rgba.shape[1]}×{rgba.shape[0]} px)", fontsize=6.5, pad=2)
        ax.text(0.02, 0.03, f"L={h.limb_length_px:.0f} px, θ={h.axis_angle_deg:.0f}°", transform=ax.transAxes,
                fontsize=6, color="#333333")
        if k == 0:
            label(ax, "A")
    # B: palmar 윤곽 (원본 좌표, 중심만 맞춤)
    axB = fig.add_subplot(gsR[0, :])
    for view in LM.VIEWS:
        for d in [x for x in ds if x[0] == view]:
            a = LM.analyze(LM.load_rgba(d[3]))
            c = a["contour"]
            c = c - c.mean(0)
            axB.plot(*np.vstack([c, c[:1]]).T, lw=0.5, color=VIEW_COL[view], alpha=0.75)
    axB.set_aspect("equal")
    axB.invert_yaxis()
    axB.set_xlabel("x (px, centered)")
    axB.set_ylabel("y (px, centered)")
    axB.set_title("Raw outlines of all 60 photographs (centroids\nsuperimposed, native pixels)", fontsize=6.5)
    label(axB, "B", x=-0.18)
    # C: 방향 vs 크기
    axC = fig.add_subplot(gsR[1, 0])
    for view in LM.VIEWS:
        h = het[het.view == view]
        axC.scatter(h.axis_angle_deg, h.limb_length_px, s=9, color=VIEW_COL[view], label=VIEW_EN[view],
                    edgecolor="k", linewidth=0.2)
    axC.set_xlabel("Limb axis angle (°)")
    axC.set_ylabel("Tip–base length (px)")
    axC.set_yscale("log")
    axC.set_yticks([200, 500, 1000, 2000], ["200", "500", "1000", "2000"])
    axC.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axC.set_xlim(-185, 185)
    axC.set_xticks([-180, -90, 0, 90, 180])
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=5.5,
               handletextpad=0.2, columnspacing=0.6)
    label(axC, "C", x=-0.3)
    # D: 유사변환 후 남는 모양 차이 (palmar)
    axD = fig.add_subplot(gsR[1, 1])
    tpl = LM.load_template("palmar")
    shapes = [LM.canonical(LM.analyze(LM.load_rgba(d[3]))["points"], False) * LM.TPL_LEN
              for d in ds if d[0] == "palmar"]
    ref = np.array(tpl["points"])
    for s_ in shapes:
        al = LM.procrustes_align(s_, ref)
        axD.plot(*np.vstack([al, al[:1]]).T, lw=0.5, color=VIEW_COL["palmar"], alpha=0.7)
    axD.plot(*np.vstack([ref, ref[:1]]).T, lw=1.2, color="k")
    axD.set_aspect("equal")
    axD.invert_yaxis()
    axD.axis("off")
    axD.set_title("Palmar outlines after\nsimilarity alignment;\nblack = common template", fontsize=6)
    label(axD, "D", x=0.0)
    save(fig, "Fig1")


# ----------------------------------------------------------------------------
def fig2():
    """처리 과정 개요 (예시 사진 한 장)"""
    view, seg = EXAMPLE
    tpl = LM.load_template(view)
    tm = LM.template_mask(tpl)
    ds = [d for d in V.dataset([view])]
    ex = next(d for d in ds if d[1] == seg)
    rgba = LM.load_rgba(ex[3])
    a = LM.analyze(rgba)
    reg = V.register(rgba, tpl, tm, "tps_dtw", analysis=a)
    fig = plt.figure(figsize=(FULL, 112 * MM))
    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.12, left=0.02, right=0.99, top=0.93, bottom=0.04)
    titles = ["Input photograph\n(transparent background)", "Pose-normalized mask, geodesic\ndistance and width profile",
              "Five-segment contour and\nlandmark-based correspondence", "Common template (GPA mean)\nwith reference grid",
              "Thin-plate spline mapping\n(template grid on photograph)", "Registered image in\ntemplate coordinates",
              "Blue index and detected\nEvans blue spots", "Group frequency map\n(C7, n = 4)"]
    axs = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    for i, (ax, t) in enumerate(zip(axs, titles)):
        ax.set_title(t, fontsize=6.8)
        ax.axis("off")
        label(ax, "ABCDEFGH"[i], x=0.04)
    # A
    chk = ((np.indices(rgba.shape[:2]) // 20).sum(0) % 2) * 25 + 225
    bg = np.repeat(chk[..., None], 3, 2).astype(float)
    al = rgba[..., 3:4] / 255
    axs[0].imshow((rgba[..., :3] * al + bg * (1 - al)).astype(np.uint8))
    # B
    m, A = LM.work_mask(rgba)
    dt = ndi.distance_transform_edt(m)
    c0 = np.unravel_index(np.argmax(dt), dt.shape)
    g0 = LM.geodesic(m, c0)
    tip = np.unravel_index(np.argmax(np.where(np.isfinite(g0), g0, -1)), g0.shape)
    g = LM.geodesic(m, tip)
    axs[1].imshow(np.ma.masked_where(~m, g), cmap="viridis")
    axs[1].plot(tip[1], tip[0], "w*", ms=6, mec="k", mew=0.4)
    ss = a["s"]
    for k, col in (("paw", "orange"), ("wrist", "red"), ("root", "white")):
        axs[1].contour(np.isfinite(g) & (g <= ss[k]), levels=[0.5], colors=col, linewidths=0.7)
    ins = axs[1].inset_axes([0.0, -0.32, 1.0, 0.3])
    h = 3
    bins = np.arange(0, ss["max"], h)
    gi = np.floor(np.where(np.isfinite(g), g, -1) / h).astype(int)
    wd = ndi.gaussian_filter1d((np.bincount(gi[gi >= 0], minlength=len(bins))[:len(bins)] / h).astype(float), 2)
    ins.plot(bins / ss["max"], wd, color="k", lw=0.7)
    for k, col in (("paw", "orange"), ("wrist", "red"), ("root", "gray")):
        ins.axvline(ss[k] / ss["max"], color=col, lw=0.8)
        ins.text(ss[k] / ss["max"], wd.max() * 1.02, k, color=col, fontsize=5.5, ha="center", va="bottom")
    ins.set_xlabel("geodesic distance from tip (normalized)", fontsize=5.5, labelpad=1)
    ins.set_ylabel("width", fontsize=5.5, labelpad=1)
    ins.tick_params(labelsize=5, length=2, pad=1)
    # C
    axs[2].imshow(LM.on_white(rgba))
    cut = np.cumsum([0] + LM.SEG)
    cols = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:blue"]
    P = reg["P_img"]
    for i in range(5):
        sg = P[cut[i]:cut[i + 1] + 1] if i < 4 else np.vstack([P[cut[4]:], P[:1]])
        axs[2].plot(sg[:, 0], sg[:, 1], "-", color=cols[i], lw=1.2)
        axs[2].plot(sg[::8, 0], sg[::8, 1], ".", color=cols[i], ms=2)
    for nm, i in tpl["landmarks"].items():
        axs[2].plot(*P[i], "o", ms=3.5, mfc="yellow", mec="k", mew=0.5)
    axs[2].text(0.02, 0.02, "tip → wrist → base (cut edge) → base → wrist → tip", transform=axs[2].transAxes, fontsize=5)
    # D
    tg = tpl_bg(tpl)
    axs[3].imshow(tg)
    for d in ds:
        aa = LM.analyze(LM.load_rgba(d[3]))
        s_ = LM.procrustes_align(LM.canonical(aa["points"], False) * LM.TPL_LEN, np.array(tpl["points"]))
        axs[3].plot(*np.vstack([s_, s_[:1]]).T, lw=0.4, color="#e7298a", alpha=0.5)
    outline(axs[3], tpl, lw=1.2)
    Pt = np.array(tpl["points"])
    for nm, i in tpl["landmarks"].items():
        axs[3].plot(*Pt[i], "o", ms=3.5, mfc="yellow", mec="k", mew=0.5)
    # E: 템플릿 격자를 사진 위로
    axs[4].imshow(LM.on_white(rgba))
    W, H = tpl["canvas"]
    for x in range(0, W + 1, 25):
        q = np.c_[np.full(200, x), np.linspace(0, H, 200)]
        ins_ = tm[np.clip(q[:, 1].astype(int), 0, H - 1), np.clip(q[:, 0].astype(int), 0, W - 1)]
        src = reg["inv"](q)
        src[~ins_] = np.nan
        axs[4].plot(src[:, 0], src[:, 1], color="#08519c", lw=0.35)
    for y in range(0, H + 1, 25):
        q = np.c_[np.linspace(0, W, 300), np.full(300, y)]
        ins_ = tm[np.clip(q[:, 1].astype(int), 0, H - 1), np.clip(q[:, 0].astype(int), 0, W - 1)]
        src = reg["inv"](q)
        src[~ins_] = np.nan
        axs[4].plot(src[:, 0], src[:, 1], color="#08519c", lw=0.35)
    # F
    axs[5].imshow(tg)
    axs[5].imshow(reg["warped"])
    outline(axs[5], tpl, color="r", lw=0.8)
    # G
    bidx = LM.blue_index(reg["warped"])
    thr = LM.auto_threshold(reg["warped"], tm)
    blue = LM.detect_blue(reg["warped"], tm, thr, 6)
    axs[6].imshow(tg)
    im = axs[6].imshow(np.ma.masked_where(~LM.valid_tissue(reg["warped"], tm), bidx), cmap="magma", vmin=-20, vmax=120)
    axs[6].contour(blue, levels=[0.5], colors="cyan", linewidths=0.6)
    outline(axs[6], tpl, lw=0.8)
    hcbar(fig, axs[6], im, "blue index  B − max(R,G)")
    axs[6].text(0.02, 0.97, f"threshold = median + 4·MAD = {thr:.0f}", transform=axs[6].transAxes, fontsize=5.5, va="top")
    # H
    z = np.load(os.path.join(RES, "demo_arrays", f"freq_{view}_{seg}.npz"))
    axs[7].imshow(tg)
    fr = z["freq"]
    im = axs[7].imshow(np.ma.masked_where(fr <= 0, fr), cmap="Blues", vmin=0, vmax=1)
    if fr.max() >= 0.5:
        axs[7].contour(fr >= 0.5, levels=[0.5], colors="red", linewidths=0.7)
    outline(axs[7], tpl, lw=0.8)
    hcbar(fig, axs[7], im, "fraction of animals with a spot")
    save(fig, "Fig2")


# ----------------------------------------------------------------------------
def boxes(ax, data, labels, colors, ylabel, log=False, rot=25):
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6, medianprops=dict(color="k", lw=0.8),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for b, c in zip(bp["boxes"], colors):
        b.set_facecolor(c)
    for i, d in enumerate(data):
        x = np.random.default_rng(i).normal(i + 1, 0.06, len(d))
        ax.scatter(x, d, s=1.5, color="k", alpha=0.35, lw=0)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=rot, ha="right" if rot else "center")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def fig3():
    """템플릿과 대응 방식"""
    ds = V.dataset()
    summ = pd.read_csv(os.path.join(RES, "register_summary.csv"))
    fig = plt.figure(figsize=(FULL, 100 * MM))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25, left=0.04, right=0.99, top=0.93, bottom=0.12,
                          height_ratios=[1, 1.15])
    for i, view in enumerate(LM.VIEWS):
        ax = fig.add_subplot(gs[0, i])
        tpl = LM.load_template(view)
        ax.imshow(tpl_bg(tpl))
        used = set(s_.replace("/", os.sep) for s_ in tpl.get("sources", []))
        for d in ds:
            if d[0] != view:
                continue
            aa = LM.analyze(LM.load_rgba(d[3]))
            s_ = LM.procrustes_align(LM.canonical(aa["points"], False) * LM.TPL_LEN, np.array(tpl["points"]))
            inc = os.path.relpath(d[3], LM.DATA_DIR) in used
            ax.plot(*np.vstack([s_, s_[:1]]).T, lw=0.4, color=VIEW_COL[view] if inc else "#999999",
                    alpha=0.6, ls="-" if inc else ":")
        outline(ax, tpl, lw=1.1)
        Pt = np.array(tpl["points"])
        for nm, k in tpl["landmarks"].items():
            ax.plot(*Pt[k], "o", ms=3, mfc="yellow", mec="k", mew=0.5)
        W, H = tpl["canvas"]
        for c in range(0, W // LM.GRID, 2):
            ax.text(c * LM.GRID + LM.GRID / 2, 8, LM.col_label(c), fontsize=4, ha="center", va="top", color="gray")
        for r in range(2, H // LM.GRID, 2):
            ax.text(4, r * LM.GRID + LM.GRID / 2, str(r + 1), fontsize=4, va="center", color="gray")
        ax.set_title(f"{VIEW_EN[view]}\n(n = {tpl['n_images']}/{sum(d[0] == view for d in ds)}, "
                     f"{'DTW' if tpl.get('correspondence') == 'dtw' else 'arc-length'})", fontsize=6.5)
        ax.axis("off")
        if i == 0:
            label(ax, "A", x=0.03)
    tpl = LM.load_template("palmar")
    W, H = tpl["canvas"]
    for j, (m, t) in enumerate((("tps_arc", "Mean registered palmar image\n(arc-length correspondence)"),
                                ("tps_dtw", "Mean registered palmar image\n(DTW correspondence)"))):
        ax = fig.add_subplot(gs[1, j])
        mean = np.load(os.path.join(RES, "register_arrays", f"mean_palmar_{m}.npy"))
        ax.imshow(mean[int(0.1 * H):int(0.75 * H), int(0.03 * W):int(0.5 * W)])
        ax.set_title(t, fontsize=6.5)
        ax.axis("off")
        label(ax, "BC"[j], x=0.03)
    for j, (col, pcol, yl) in enumerate((("bd_mean_px", "p_bd_vs_arc", "Mean boundary distance (px)"),
                                         ("ncc_mean", "p_ncc_limb_vs_arc", "Inter-animal texture NCC"))):
        ax = fig.add_subplot(gs[1, 2 + j])
        x = np.arange(4)
        for k, m in enumerate(("tps_arc", "tps_dtw")):
            v = [summ[(summ.view == vw) & (summ.method == m)][col].values[0] for vw in LM.VIEWS]
            ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=METH_COL[m], label=V.METHOD_LABEL[m])
        for vi, vw in enumerate(LM.VIEWS):
            r = summ[(summ.view == vw) & (summ.method == "tps_dtw")]
            if pcol in r and np.isfinite(r[pcol].values[0]):
                top = summ[(summ.view == vw) & summ.method.isin(["tps_arc", "tps_dtw"])][col].max()
                ax.text(vi, top * 1.03, sig(r[pcol].values[0]), ha="center", fontsize=6)
        ax.set_xticks(x, [VIEW_EN[v].split(" ")[0] for v in LM.VIEWS], rotation=25, ha="right")
        ax.set_ylabel(yl)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.legend(frameon=False, fontsize=5.5)
        label(ax, "DE"[j], x=-0.15)
    save(fig, "Fig3")


def fig4():
    """정합 정확도"""
    per = pd.read_csv(os.path.join(RES, "register_per_image.csv"))
    pairs = pd.read_csv(os.path.join(RES, "register_pairs.csv"))
    if "same_animal" in pairs:
        pairs = pairs[~pairs.same_animal]                   # 같은 동물을 두 자세로 찍은 사진 쌍 제외
    dfm = pd.read_csv(os.path.join(RES, "deform.csv"))
    pose = pd.read_csv(os.path.join(RES, "pose.csv"))
    M = V.METHODS
    labs = [V.METHOD_LABEL[m] for m in M]
    fig = plt.figure(figsize=(FULL, 130 * MM))
    gs = fig.add_gridspec(3, 5, hspace=0.7, wspace=0.6, left=0.07, right=0.99, top=0.95, bottom=0.1,
                          height_ratios=[0.28, 1, 1])
    view, seg = EXAMPLE
    tpl = LM.load_template(view)
    tm = LM.template_mask(tpl)
    ex = next(d for d in V.dataset([view]) if d[1] == seg)
    rgba = LM.load_rgba(ex[3])
    an = LM.analyze(rgba)
    tg = tpl_bg(tpl)
    for i, m in enumerate(M):
        ax = fig.add_subplot(gs[0, i])
        reg = V.register(rgba, tpl, tm, m, analysis=an)
        ax.imshow(tg)
        ax.imshow(reg["warped"])
        outline(ax, tpl, color="r", lw=0.7)
        ys_, xs_ = np.nonzero(tm)
        ax.set_xlim(xs_.min() - 6, xs_.max() + 6)
        ax.set_ylim(ys_.max() + 6, ys_.min() - 6)
        ax.set_title(f"{V.METHOD_LABEL[m]}\nDice {reg['dice']:.3f}", fontsize=6)
        ax.axis("off")
        if i == 0:
            label(ax, "A", x=0.05)
    ax = fig.add_subplot(gs[1, 0:2])
    boxes(ax, [per[per.method == m].boundary_mean_px.values for m in M], labs, [METH_COL[m] for m in M],
          "Mean boundary distance (px)", log=True)
    label(ax, "B", x=-0.12)
    ax = fig.add_subplot(gs[1, 2:4])
    boxes(ax, [pairs[pairs.method == m].ncc_limb.dropna().values for m in M], labs, [METH_COL[m] for m in M],
          "Inter-animal texture NCC")
    label(ax, "C", x=-0.12)
    ax = fig.add_subplot(gs[1, 4])
    boxes(ax, [pairs[(pairs.method == m) & pairs.same_segment].blue_dice_smooth.dropna().values for m in M],
          ["Axis", "Similarity", "Affine", "TPS-arc", "TPS-DTW"], [METH_COL[m] for m in M],
          "Spot-map Dice\n(same segment)", rot=60)
    label(ax, "D", x=-0.35)
    ax = fig.add_subplot(gs[2, 0:3])
    M2 = M + ["affine_oracle", "tps_oracle"]
    lab2 = labs + ["Affine\n(true corresp.)", "TPS\n(true corresp.)"]
    boxes(ax, [100 * dfm[dfm.method == m].err_median_px.values / LM.TPL_LEN for m in M2], lab2,
          [METH_COL[m] for m in M2], "Simulated TRE (% limb length)", log=True)
    nr = 100 * dfm.groupby(["image", "trial"]).nonrigid_tpl_px.first().median() / LM.TPL_LEN
    ax.axhline(nr, ls="--", lw=0.6, color="gray")
    ax.text(len(M2) + 0.45, nr, "applied\nnon-rigid", fontsize=5, va="center", color="gray")
    label(ax, "E", x=-0.07)
    ax = fig.add_subplot(gs[2, 3])
    cond = [("pose+photo", "Pose +\nphoto"), ("resolution", "Down-\nsample")]
    boxes(ax, [100 * pose[pose.condition == c].coord_err_mean_px.values / LM.TPL_LEN for c, _ in cond],
          [t for _, t in cond], ["#9ecae1", "#9ecae1"], "Coordinate deviation (% L)", rot=0)
    label(ax, "F", x=-0.35)
    ax = fig.add_subplot(gs[2, 4])
    placeholder(ax, "Landmark TRE vs.\ninter-rater variability\n(manual annotation;\npending)")
    label(ax, "G", x=-0.05)
    save(fig, "Fig4")


def fig5():
    """파란 점 검출"""
    sp = pd.read_csv(os.path.join(RES, "spots.csv"))
    real = pd.read_csv(os.path.join(RES, "spots_real.csv"))
    par = pd.read_csv(os.path.join(RES, "params.csv"))
    pose = pd.read_csv(os.path.join(RES, "pose.csv"))
    fig = plt.figure(figsize=(FULL, 112 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.55, left=0.07, right=0.97, top=0.95, bottom=0.1)
    ax = fig.add_subplot(gs[0, 0])
    g = sp[sp.gain == 1.0]
    setc = {"auto_k3": "#9ecae1", "auto_k4": "#08519c", "auto_k6": "#3182bd", "fixed_35": "#fdae6b",
            "fixed_50": "#e6550d"}
    setl = {"auto_k3": "adaptive k=3", "auto_k4": "adaptive k=4", "auto_k6": "adaptive k=6",
            "fixed_35": "fixed 35", "fixed_50": "fixed 50"}
    bins = np.arange(0, 181, 10)
    cen = (bins[:-1] + bins[1:]) / 2
    ax2 = ax.twinx()
    ax2.hist(real.peak_b, bins=bins, color="#d9d9d9", zorder=0)
    ax2.set_ylabel("Detected real spots (n)", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    for k, c in setc.items():
        d = g[g.setting == k]
        idx = np.digitize(d.peak_b, bins) - 1
        rec = [d.detected[idx == i].mean() if (idx == i).sum() >= 20 else np.nan for i in range(len(cen))]
        ax.plot(cen, rec, "-o", ms=2, lw=0.9, color=c, label=setl[k])
    ax.axvline(g[g.setting == "auto_k4"].threshold.median(), color="#08519c", ls=":", lw=0.7)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("Peak blue index of spot")
    ax.set_ylabel("Recall of synthetic spots")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=5, loc="center right")
    label(ax, "A", x=-0.2)
    ax = fig.add_subplot(gs[0, 1])
    pv = g[g.setting == "auto_k4"].pivot_table(index="contrast", columns="sigma_tpl", values="detected")
    im = ax.imshow(pv.values, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="w" if pv.values[i, j] > 0.6 else "k")
    ax.set_xticks(range(pv.shape[1]), [f"{v:g}" for v in pv.columns])
    ax.set_yticks(range(pv.shape[0]), [f"{v:g}" for v in pv.index])
    ax.set_xlabel("Spot size σ (template px)")
    ax.set_ylabel("Peak dye opacity")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("recall (adaptive, k=4)", fontsize=6)
    label(ax, "B", x=-0.2)
    ax = fig.add_subplot(gs[0, 2])
    d = g[(g.setting == "auto_k4") & (g.detected == 1)]
    ss = sorted(d.sigma_tpl.unique())
    boxes(ax, [d[d.sigma_tpl == s_].loc_err_px.values for s_ in ss], [f"{s_:g}" for s_ in ss],
          ["#9ecae1"] * len(ss), "Centroid error (template px)", rot=0)
    ax.set_xlabel("Spot size σ (template px)")
    label(ax, "C", x=-0.2)
    ax = fig.add_subplot(gs[1, 0])
    br = pd.concat([pose[pose.condition == "brightness"],
                    pose[pose.condition == "brightness"].iloc[0:0]])
    gains = sorted(br.gain.unique().tolist() + [1.0])
    for col, c, l in (("dice_auto", "#08519c", "adaptive threshold"), ("dice_fixed", "#e6550d", "fixed threshold")):
        mv, q1, q3 = [], [], []
        for gg in gains:
            v = br[br.gain == gg][col] if gg != 1.0 else pd.Series([1.0])
            mv.append(v.median())
            q1.append(v.quantile(0.25))
            q3.append(v.quantile(0.75))
        ax.errorbar(gains, mv, yerr=[np.subtract(mv, q1), np.subtract(q3, mv)], fmt="-o", ms=2.5, lw=0.9,
                    capsize=1.5, color=c, label=l)
    ax.set_xlabel("Global brightness gain")
    ax.set_ylabel("Spot-mask Dice vs. original")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=5.5, loc="lower center")
    ax.spines[["top", "right"]].set_visible(False)
    label(ax, "D", x=-0.2)
    ax = fig.add_subplot(gs[1, 1])
    pk = par[par.min_area == 6].groupby("k")
    ax.plot(pk.n_spots.mean().index, pk.n_spots.mean().values, "-o", ms=2.5, color="#08519c")
    ax.set_xlabel("k (threshold = median + k·MAD)")
    ax.set_ylabel("Detected spots per image", color="#08519c")
    ax3 = ax.twinx()
    ax3.plot(pk.threshold.mean().index, pk.threshold.mean().values, "-s", ms=2.5, color="#e6550d")
    ax3.set_ylabel("Mean threshold (blue index)", color="#e6550d")
    ax.axvline(4, ls=":", color="gray", lw=0.6)
    label(ax, "E", x=-0.2)
    ax = fig.add_subplot(gs[1, 2])
    placeholder(ax, "Automatic vs. manual spot\nidentification (precision,\nrecall, count ICC;\ntwo blinded raters; pending)")
    label(ax, "F", x=-0.05)
    save(fig, "Fig5")


def fig6():
    """적용 예: 한 그룹(EXAMPLE)의 동물 4마리를 템플릿에 통합 (유사변환 vs TPS)"""
    view, seg = EXAMPLE
    ex = json.load(open(os.path.join(RES, "example.json"), encoding="utf-8"))
    z = np.load(os.path.join(RES, "example_arrays.npz"))
    tpl = LM.load_template(view)
    tm = LM.template_mask(tpl)
    tg = tpl_bg(tpl)
    W, H = tpl["canvas"]
    ys, xs = np.nonzero(tm)
    crop = (max(0, xs.min() - 8), min(W, xs.max() + 8), max(0, ys.min() - 8), min(H, ys.max() + 8))

    def frame(ax):
        ax.set_xlim(crop[0], crop[1])
        ax.set_ylim(crop[3], crop[2])
        ax.axis("off")

    fig = plt.figure(figsize=(FULL, 100 * MM))
    gs = fig.add_gridspec(2, 4, hspace=0.12, wspace=0.05, left=0.02, right=0.99, top=0.95, bottom=0.1,
                          height_ratios=[0.55, 1])
    cols = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3"]
    n = ex["n_images"]
    for k in range(n):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(tg)
        ax.imshow(z["warps_tps"][k])
        ax.contour(z["masks_tps"][k], levels=[0.5], colors="cyan", linewidths=0.5)
        outline(ax, tpl, lw=0.6)
        frame(ax)
        ax.set_title(f"Rat {k + 1}: {len(z[f'dots_tps_{k}'])} spots", fontsize=6.5, color=cols[k])
        if k == 0:
            label(ax, "A", x=0.06)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.imshow(tg)
    outline(ax, tpl, lw=0.8)
    for k in range(n):
        d = z[f"dots_tps_{k}"]
        ax.plot(d[:, 0], d[:, 1], "o", ms=2.2, mfc=cols[k], mec="k", mew=0.2, label=f"Rat {k + 1}")
    W_, H_ = crop[1] - crop[0], crop[3] - crop[2]
    for c in range(0, W // LM.GRID):
        x = c * LM.GRID + LM.GRID / 2
        if crop[0] + 12 <= x <= crop[1] - 12:
            ax.text(x, crop[2] + 2, LM.col_label(c), fontsize=4.5, ha="center", va="top", color="gray")
    for r in range(0, H // LM.GRID):
        y = r * LM.GRID + LM.GRID / 2
        if crop[2] + 12 <= y <= crop[3] - 12:
            ax.text(crop[0] + 2, y, str(r + 1), fontsize=4.5, va="center", color="gray")
    frame(ax)
    ax.legend(loc="lower right", fontsize=5.5, frameon=False, markerscale=1.5, handletextpad=0.2)
    ax.set_title("Spot centroids of all animals in template coordinates (TPS)", fontsize=6.5)
    label(ax, "B", x=0.02, y=1.04)
    im = None
    for j, (key, t) in enumerate((("similarity", "Similarity alignment"), ("tps", "TPS registration (proposed)"))):
        ax = fig.add_subplot(gs[1, 2 + j])
        fr = z[f"freq_{key}"]
        ax.imshow(tg)
        im = ax.imshow(np.ma.masked_where(fr <= 0, fr), cmap="Blues", vmin=0, vmax=1)
        if fr.max() >= 0.5:
            ax.contour(fr >= 0.5, levels=[0.5], colors="red", linewidths=0.6)
        outline(ax, tpl, lw=0.6)
        frame(ax)
        e = ex[key]
        ax.set_title(f"{t}\ncommon area {e['common_area_pct']:.1f}%, pairwise Dice {e['pairwise_dice']:.2f}",
                     fontsize=6.3)
        label(ax, "CD"[j], x=0.04, y=1.18)
    cax = fig.add_axes([0.6, 0.1, 0.3, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Fraction of animals with an Evans blue spot (red line: ≥ 50%)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    save(fig, "Fig6")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6}


def main():
    global ROOT, RES, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("figs", nargs="+")
    a = ap.parse_args()
    ROOT = a.root
    RES = os.path.join(ROOT, "results")
    FIG = os.path.join(ROOT, "figures")
    for f in (FIGS if a.figs == ["all"] else a.figs):
        FIGS[f]()


if __name__ == "__main__":
    main()
