"""
논문용 계산 실험 (평가자 주석이 필요 없는 것들)
================================================================
  hetero    : 원본 사진의 이질성 (크기, 방향, 다리 길이, 잘림)
  register  : 정합 방법 5가지 비교 (윤곽 Dice/경계 거리, 사진 간 질감 NCC, 같은 분절 내 파란 영역 Dice, 시간)
  deform    : 알려진 비강체 변형 + 자세 변화를 준 모의 사진 -> 내부 점의 템플릿 좌표 일치 오차 (모의 TRE)
  pose      : 회전/크기/밝기/색/잡음, 해상도 저하에 대한 불변성 (좌표 일치, 파란 영역 Dice)
  spots     : 크기·대비를 아는 합성 파란 점을 심고 검출 재현율·위치 오차·오검출, 밝기 변화와 임계값 방식 비교
  params    : k(MAD 배수), 최소 크기에 따른 검출 결과 민감도
  demo      : C5~C8 부위별 통합 빈도 지도 (적용 예시, 자동 임계값만 사용)

사용 예)  python paper_experiments.py register --out "D:/.../IMR/results"
          python paper_experiments.py all --out ...
"""
import warnings

warnings.filterwarnings("ignore", message=r".*cryptography\.hazmat\.decrepit.*")
import os
import re
import sys
import csv
import time
import json
import argparse
import itertools
from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
from skimage import measure

import limb_mapping as LM
import limb_validation as V

K_MAD, MIN_AREA = 4.0, 6
SHARD = (0, 1)
TPL, TMASK = {}, {}


def tpl_of(view):
    if view not in TPL:
        TPL[view] = LM.load_template(view)
        TMASK[view] = LM.template_mask(TPL[view])
    return TPL[view], TMASK[view]


def proposed(view):
    return "tps_dtw" if tpl_of(view)[0].get("correspondence") == "dtw" else "tps_arc"


def write(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def limb_length(analysis):
    lm = np.cumsum([0] + LM.SEG[:-1])
    P = analysis["points"]
    return float(np.linalg.norm((P[lm[2]] + P[lm[3]]) / 2 - P[0]))


def detect(warped, tmask, k=K_MAD, min_area=MIN_AREA, thr=None):
    t = LM.auto_threshold(warped, tmask, k) if thr is None else thr
    return LM.detect_blue(warped, tmask, t, min_area), t


def warp_with(reg, rgba, tmask):
    """reg 의 map_x/map_y 로 다른 그림(같은 기하) 다시 그리기"""
    H, W = tmask.shape
    yy, xx = np.nonzero(tmask)
    out = np.zeros((H, W, 4), np.uint8)
    c = [reg["map_y"][yy, xx], reg["map_x"][yy, xx]]
    for ch in range(4):
        out[yy, xx, ch] = np.clip(ndi.map_coordinates(rgba[..., ch].astype(float), c, order=1, cval=0), 0, 255)
    return out


def interior_points(rgba, n, erode_frac, rng, L):
    m = rgba[..., 3] > 127
    m = ndi.binary_erosion(m, iterations=max(1, int(erode_frac * L)))
    yy, xx = np.nonzero(m)
    k = rng.choice(len(xx), min(n, len(xx)), replace=False)
    return np.c_[xx[k], yy[k]].astype(float) + rng.random((len(k), 2)) - 0.5


# ----------------------------------------------------------------------------
def exp_hetero(out):
    rows = []
    for view, seg, folder, p in V.dataset():
        rgba = LM.load_rgba(p)
        a = LM.analyze(rgba)
        lm = np.cumsum([0] + LM.SEG[:-1])
        P = a["points"]
        d = (P[lm[2]] + P[lm[3]]) / 2 - P[0]
        rows.append(dict(view=view, segment=seg, image=os.path.relpath(p, LM.DATA_DIR),
                         width=rgba.shape[1], height=rgba.shape[0], limb_length_px=round(limb_length(a), 1),
                         axis_angle_deg=round(float(np.degrees(np.arctan2(d[1], d[0]))), 1),
                         tissue_area_px=int((rgba[..., 3] > 127).sum()),
                         truncated=round(a["truncated"], 3), landmarks_ok=a["landmarks_ok"]))
        print("hetero", rows[-1]["image"])
    write(os.path.join(out, "hetero.csv"), rows)


# ----------------------------------------------------------------------------
def exp_register(out):
    os.makedirs(os.path.join(out, "register_arrays"), exist_ok=True)
    per, pairs = [], []
    ds = V.dataset()
    for view in LM.VIEWS:
        items = [d for d in ds if d[0] == view]
        tpl, tm = tpl_of(view)
        limb = ndi.binary_erosion(tm, iterations=10)
        hp = defaultdict(list)
        hv = defaultdict(list)
        blues = defaultdict(list)
        means = {m: np.zeros(tm.shape + (3,)) for m in V.METHODS}
        for _, seg, folder, p in items:
            rgba = LM.load_rgba(p)
            a = LM.analyze(rgba)
            for m in V.METHODS:
                t0 = time.time()
                reg = V.register(rgba, tpl, tm, m, analysis=a)
                t_reg = time.time() - t0
                t0 = time.time()
                blue, thr = detect(reg["warped"], tm)
                t_det = time.time() - t0
                bd, hd = V.boundary_distance(reg["warped"], tm)
                per.append(dict(view=view, segment=seg, image=os.path.relpath(p, LM.DATA_DIR), method=m,
                                dice=reg["dice"], boundary_mean_px=bd, boundary_hd95_px=hd,
                                threshold=thr, n_spots=int(measure.label(blue).max()), blue_area_px=int(blue.sum()),
                                t_register_s=round(t_reg, 3), t_detect_s=round(t_det, 3)))
                hp[m].append(V.highpass_gray(reg["warped"]).astype(np.float32))
                hv[m].append(V.texture_valid(reg["warped"], tm))
                blues[m].append(blue)
                means[m] += LM.on_white(reg["warped"])
            print("register", view, os.path.basename(p), flush=True)
        segs = [d[1] for d in items]
        for m in V.METHODS:
            for i, j in itertools.combinations(range(len(items)), 2):
                pairs.append(dict(view=view, method=m, i=i, j=j, same_segment=segs[i] == segs[j],
                                  same_animal_possible=same_animal_possible(segs[i], items[i][2], segs[j], items[j][2]),
                                  ncc_limb=V.ncc(hp[m][i], hp[m][j], limb & hv[m][i] & hv[m][j]),
                                  blue_dice=V.dice(blues[m][i], blues[m][j]),
                                  blue_dice_smooth=V.dice(ndi.gaussian_filter(blues[m][i].astype(float), 6) > 0.15,
                                                          ndi.gaussian_filter(blues[m][j].astype(float), 6) > 0.15)))
            np.save(os.path.join(out, "register_arrays", f"mean_{view}_{m}.npy"),
                    (means[m] / len(items)).astype(np.uint8))
    write(os.path.join(out, "register_per_image.csv"), per)
    write(os.path.join(out, "register_pairs.csv"), pairs)
    summarize_register(out, per, pairs)


def same_animal_possible(seg_i, folder_i, seg_j, folder_j):
    """같은 분절에서 '<폴더>_1', '<폴더>_2' 처럼 같은 동물을 다른 자세로 찍은 폴더 쌍 (예: C8 내측 _1/_2)"""
    base = lambda f: re.sub(r"_\d+$", "", f)
    return seg_i == seg_j and folder_i != folder_j and base(folder_i) == base(folder_j)


def exp_resummarize(out):
    """저장된 CSV로 쌍 비교 요약만 다시 계산 (같은 동물 가능 쌍 표시·제외)"""
    import pandas as pd
    per = pd.read_csv(os.path.join(out, "register_per_image.csv")).to_dict("records")
    pairs = pd.read_csv(os.path.join(out, "register_pairs.csv")).to_dict("records")
    first = V.METHODS[0]
    order = {v: [r["image"] for r in per if r["view"] == v and r["method"] == first] for v in LM.VIEWS}
    for r in pairs:
        a, b = order[r["view"]][int(r["i"])].split(os.sep), order[r["view"]][int(r["j"])].split(os.sep)
        r["same_animal_possible"] = same_animal_possible(a[0], a[1], b[0], b[1])
    write(os.path.join(out, "register_pairs.csv"), pairs)
    summarize_register(out, per, pairs)
    print("pairs excluded (same animal possible):", sum(r["same_animal_possible"] for r in pairs) // len(V.METHODS))


def summarize_register(out, per, pairs):
    pairs = [r for r in pairs if not r.get("same_animal_possible", False)]
    rows = []
    for view in list(LM.VIEWS) + ["all"]:
        prop = {v: proposed(v) for v in LM.VIEWS}
        for m in V.METHODS:
            P = [r for r in per if r["method"] == m and view in (r["view"], "all")]
            Q = [r for r in pairs if r["method"] == m and view in (r["view"], "all")]
            S = [r for r in Q if r["same_segment"]]
            row = dict(view=view, method=m, label=V.METHOD_LABEL[m], n_images=len(P),
                       dice_mean=np.mean([r["dice"] for r in P]), dice_sd=np.std([r["dice"] for r in P], ddof=1),
                       bd_mean_px=np.mean([r["boundary_mean_px"] for r in P]),
                       bd_sd_px=np.std([r["boundary_mean_px"] for r in P], ddof=1),
                       hd95_mean_px=np.mean([r["boundary_hd95_px"] for r in P]),
                       ncc_mean=np.mean([r["ncc_limb"] for r in Q]), ncc_sd=np.std([r["ncc_limb"] for r in Q], ddof=1),
                       blue_dice_same_seg=np.nanmean([r["blue_dice"] for r in S]) if S else np.nan,
                       blue_dice_smooth_same_seg=np.nanmean([r["blue_dice_smooth"] for r in S]) if S else np.nan,
                       blue_dice_smooth_diff_seg=np.nanmean([r["blue_dice_smooth"] for r in Q if not r["same_segment"]]),
                       t_register_s=np.mean([r["t_register_s"] for r in P]),
                       t_detect_s=np.mean([r["t_detect_s"] for r in P]))
            # 제안 방법(TPS arc)과 짝 비교
            for ref in ("tps_arc",):
                if m != ref:
                    a_ = {(r["view"], r["image"]): r["boundary_mean_px"] for r in P}
                    b_ = {(r["view"], r["image"]): r["boundary_mean_px"] for r in per
                          if r["method"] == ref and view in (r["view"], "all")}
                    k = sorted(set(a_) & set(b_))
                    row["p_bd_vs_arc"] = wilcoxon([a_[x] - b_[x] for x in k]).pvalue
                    for col in ("ncc_limb", "blue_dice_smooth"):
                        a_ = {(r["view"], r["i"], r["j"]): r[col] for r in Q}
                        b_ = {(r["view"], r["i"], r["j"]): r[col] for r in pairs
                              if r["method"] == ref and view in (r["view"], "all")}
                        k = [x for x in sorted(set(a_) & set(b_)) if np.isfinite(a_[x] - b_[x])]
                        row[f"p_{col}_vs_arc"] = wilcoxon([a_[x] - b_[x] for x in k]).pvalue
            rows.append(row)
    write(os.path.join(out, "register_summary.csv"), rows)


# ----------------------------------------------------------------------------
def random_deformation(rng, rgba, L, amp, rot, scale, nb=6):
    """y = S(p + u(p)),  u = 가우시안 덩어리 nb개 합 (최대 크기 amp*L)"""
    m = rgba[..., 3] > 127
    yy, xx = np.nonzero(m)
    C = np.c_[xx, yy][rng.choice(len(xx), nb)].astype(float)
    sig = rng.uniform(0.15, 0.3, nb) * L
    vec = rng.normal(0, 1, (nb, 2))

    def u_raw(p):
        d2 = ((p[:, None, :] - C[None]) ** 2).sum(2)
        w = np.exp(-d2 / (2 * sig[None] ** 2))
        return w @ vec
    sub = np.c_[xx, yy][rng.choice(len(xx), min(4000, len(xx)))].astype(float)
    k = amp * L / np.max(np.linalg.norm(u_raw(sub), axis=1))
    u = lambda p: k * u_raw(p)
    th = np.radians(rot)
    R = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ctr = np.array([rgba.shape[1], rgba.shape[0]]) / 2

    def fwd(p):
        return (p + u(p) - ctr) @ R.T

    # 출력 캔버스
    corners = fwd(np.c_[xx, yy][::50].astype(float))
    lo = corners.min(0) - 20
    hi = corners.max(0) + 20
    shift = -lo
    Wd, Hd = int(np.ceil(hi[0] - lo[0])), int(np.ceil(hi[1] - lo[1]))
    F = lambda p: fwd(p) + shift
    Ri = np.linalg.inv(R)
    gy, gx = np.mgrid[0:Hd, 0:Wd]
    Y = np.c_[gx.ravel(), gy.ravel()].astype(float)
    base = (Y - shift) @ Ri.T + ctr
    P = base.copy()
    for _ in range(12):                                         # p = S^-1(y) - u(p) 고정점 반복
        P = base - u(P)
    out = np.zeros((Hd, Wd, 4), np.uint8)
    for ch in range(4):
        out[..., ch] = ndi.map_coordinates(rgba[..., ch].astype(float), [P[:, 1], P[:, 0]],
                                           order=1, cval=0).reshape(Hd, Wd)
    return out, F


def exp_deform(out, trials=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for view, seg, folder, p in V.dataset():
        tpl, tm = tpl_of(view)
        rgba = LM.load_rgba(p)
        a0 = LM.analyze(rgba)
        L = limb_length(a0)
        regs0 = {m: V.register_points(rgba, tpl, tm, m, analysis=a0) for m in V.METHODS}
        for t in range(trials):
            amp = rng.uniform(0.03, 0.08)
            rot, sc = rng.uniform(-180, 180), rng.uniform(0.7, 1.3)
            dimg, F = random_deformation(rng, rgba, L, amp, rot, sc)
            pts = interior_points(rgba, 300, 0.04, rng, L)
            ypts = F(pts)
            a1 = LM.analyze(dimg)
            # 적용한 비강체 성분 크기 (유사변환으로 설명 안 되는 부분, 템플릿 단위)
            Ms = V._fit_similarity(pts, ypts)
            nonrigid = np.linalg.norm(V._apply(Ms, pts) - ypts, axis=1).mean() / (sc * L) * LM.TPL_LEN
            lmi = np.cumsum([0] + LM.SEG[:-1])
            lm_shift = np.linalg.norm(a1["points"][lmi] - F(a0["points"][lmi]), axis=1) / (sc * L) * LM.TPL_LEN
            # 오라클: 변형 사진의 윤곽 대응점을 '정답'(원본 대응점을 변형으로 옮긴 것)으로 준 경우 -> 정합 모델 자체의 오차
            a_or = dict(a1, points=F(a0["points"]))
            runs = [(m, m, a1) for m in V.METHODS] + [("affine_oracle", "affine", a_or), ("tps_oracle", "tps_arc", a_or)]
            for name, m, an in runs:
                r1 = V.register_points(dimg, tpl, tm, m, analysis=an)
                r0 = regs0["tps_arc" if name == "tps_oracle" else m]
                e = np.linalg.norm(V.forward_points(r0, pts) - V.forward_points(r1, ypts), axis=1)
                rows.append(dict(view=view, segment=seg, image=os.path.relpath(p, LM.DATA_DIR), trial=t,
                                 method=name, amp_pctL=100 * amp, rot_deg=rot, scale=sc,
                                 **{f"shift_{n}_px": v for n, v in zip(LM.LM_NAMES, lm_shift)},
                                 nonrigid_tpl_px=nonrigid, err_mean_px=e.mean(), err_median_px=np.median(e),
                                 err_p95_px=np.percentile(e, 95), err_mean_pctL=100 * e.mean() / LM.TPL_LEN,
                                 landmarks_ok=a1["landmarks_ok"]))
            print("deform", view, os.path.basename(p), t,
                  " ".join(f"{r['method']}={r['err_mean_px']:.1f}" for r in rows[-7:]), flush=True)
        write(os.path.join(out, "deform.csv"), rows)
    write(os.path.join(out, "deform.csv"), rows)


# ----------------------------------------------------------------------------
def photometric(img, rng, gain, gamma, wb, noise):
    rgb = img[..., :3].astype(float) / 255
    rgb = np.clip(rgb * gain * wb[None, None, :], 0, 1) ** gamma
    rgb = rgb * 255 + rng.normal(0, noise, rgb.shape)
    out = img.copy()
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def similarity_image(rgba, rot, scale):
    th = np.radians(rot)
    R = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    H, W = rgba.shape[:2]
    ctr = np.array([W, H]) / 2
    cs = (np.array([[0, 0], [W, 0], [0, H], [W, H]]) - ctr) @ R.T
    lo, hi = cs.min(0) - 10, cs.max(0) + 10
    Wd, Hd = int(np.ceil(hi[0] - lo[0])), int(np.ceil(hi[1] - lo[1]))
    F = lambda p: (np.atleast_2d(p) - ctr) @ R.T - lo
    gy, gx = np.mgrid[0:Hd, 0:Wd]
    P = (np.c_[gx.ravel(), gy.ravel()] + lo) @ np.linalg.inv(R).T + ctr
    out = np.zeros((Hd, Wd, 4), np.uint8)
    for ch in range(4):
        out[..., ch] = ndi.map_coordinates(rgba[..., ch].astype(float), [P[:, 1], P[:, 0]],
                                           order=1, cval=0).reshape(Hd, Wd)
    return out, F


def exp_pose(out, trials=3, seed=1):
    rng = np.random.default_rng(seed + 100 * SHARD[0])
    rows = []
    name = f"pose_shard{SHARD[0]}.csv" if SHARD[1] > 1 else "pose.csv"
    for view, seg, folder, p in V.dataset()[SHARD[0]::SHARD[1]]:
        tpl, tm = tpl_of(view)
        m = proposed(view)
        rgba = LM.load_rgba(p)
        a0 = LM.analyze(rgba)
        L = limb_length(a0)
        reg0 = V.register(rgba, tpl, tm, m, analysis=a0)
        b0, thr0 = detect(reg0["warped"], tm)
        pts = interior_points(rgba, 300, 0.04, rng, L)
        q0 = V.forward_points(reg0, pts)
        conds = []
        for t in range(trials):
            conds.append(("pose+photo", dict(rot=rng.uniform(0, 360), scale=rng.uniform(0.5, 1.5),
                                             gain=rng.uniform(0.8, 1.2), gamma=rng.uniform(0.8, 1.25),
                                             wb=rng.uniform(0.92, 1.08, 3), noise=2.0)))
        for f in (0.75, 0.5, 0.35, 0.25):
            conds.append((f"resolution", dict(rot=0.0, scale=f, gain=1.0, gamma=1.0, wb=np.ones(3), noise=0.0)))
        for g in (0.7, 0.85, 1.15, 1.3):
            conds.append(("brightness", dict(rot=0.0, scale=1.0, gain=g, gamma=1.0, wb=np.ones(3), noise=0.0)))
        for kind, c in conds:
            img, F = similarity_image(rgba, c["rot"], c["scale"])
            img = photometric(img, rng, c["gain"], c["gamma"], c["wb"], c["noise"])
            reg1 = V.register(img, tpl, tm, m)
            e = np.linalg.norm(V.forward_points(reg1, F(pts)) - q0, axis=1)
            b_auto, thr1 = detect(reg1["warped"], tm)
            b_fix, _ = detect(reg1["warped"], tm, thr=thr0)
            rows.append(dict(view=view, segment=seg, image=os.path.relpath(p, LM.DATA_DIR), condition=kind,
                             rot=round(c["rot"], 1), scale=round(c["scale"], 3), gain=round(c["gain"], 3),
                             gamma=round(c["gamma"], 3), coord_err_mean_px=e.mean(), coord_err_p95_px=np.percentile(e, 95),
                             thr_ref=thr0, thr_auto=thr1, n_ref=int(measure.label(b0).max()),
                             n_auto=int(measure.label(b_auto).max()), n_fixed=int(measure.label(b_fix).max()),
                             dice_auto=V.dice(b0, b_auto), dice_fixed=V.dice(b0, b_fix)))
        print("pose", view, os.path.basename(p), f"{rows[-1]['coord_err_mean_px']:.2f}", flush=True)
        write(os.path.join(out, name), rows)
    write(os.path.join(out, name), rows)


# ----------------------------------------------------------------------------
CONTRASTS = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
SIGMAS = [1.5, 2.5, 4.0, 6.0]


def dye_color():
    """실제 데이터에서 진한 파란 점 픽셀의 중앙값 색 (합성 점 색으로 사용)"""
    cols = []
    for view, seg, folder, p in V.dataset():
        rgba = LM.load_rgba(p)
        b = LM.blue_index(rgba)
        m = (rgba[..., 3] > 127) & (b > np.percentile(b[rgba[..., 3] > 127], 99.9)) & (b > 50)
        if m.sum() > 5:
            cols.append(np.median(rgba[..., :3][m], axis=0))
    return np.median(cols, axis=0)


def plant(rgba, centers, sigmas, contrasts, color):
    img = rgba[..., :3].astype(float)
    H, W = img.shape[:2]
    a_tot = np.zeros((H, W))
    for (x, y), s, c in zip(centers, sigmas, contrasts):
        R = int(4 * s) + 2
        y0, y1, x0, x1 = max(0, int(y) - R), min(H, int(y) + R + 1), max(0, int(x) - R), min(W, int(x) + R + 1)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        a = c * np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * s ** 2))
        a_tot[y0:y1, x0:x1] = 1 - (1 - a_tot[y0:y1, x0:x1]) * (1 - a)
    img = img * (1 - a_tot[..., None]) + color[None, None, :] * a_tot[..., None]
    out = rgba.copy()
    out[..., :3] = np.clip(img, 0, 255).astype(np.uint8)
    return out


def exp_spots(out, trials=2, per_trial=12, seed=2):
    rng = np.random.default_rng(seed)
    color = dye_color()
    json.dump(dict(dye_rgb=color.tolist()), open(os.path.join(out, "spots_color.json"), "w"))
    print("dye colour", color)
    combos = list(itertools.product(CONTRASTS, SIGMAS))
    rows, fps, real = [], [], []
    ci = 0
    for view, seg, folder, p in V.dataset():
        tpl, tm = tpl_of(view)
        m = proposed(view)
        rgba = LM.load_rgba(p)
        a0 = LM.analyze(rgba)
        L = limb_length(a0)
        reg = V.register(rgba, tpl, tm, m, analysis=a0)
        valid = LM.valid_tissue(reg["warped"], tm)
        ref = {}
        for key, kw in DET_SETTINGS.items():
            ref[key] = detect(reg["warped"], tm, **kw)[0]
        bref = LM.blue_index(reg["warped"])
        for pp in measure.regionprops(measure.label(ref["auto_k4"]), intensity_image=bref):
            r0, c0 = (int(v) for v in pp.centroid)
            real.append(dict(view=view, image=os.path.relpath(p, LM.DATA_DIR), area_px=pp.area,
                             threshold=detect(reg["warped"], tm)[1],
                             peak_b=float(pp.max_intensity), delta_b=float(pp.max_intensity - local_bg(bref, valid, r0, c0))))
        far = ndi.distance_transform_edt(~ref["auto_k4"]) > 15
        cand = valid & far & ndi.binary_erosion(valid, iterations=8)
        yy, xx = np.nonzero(cand)
        s_ratio = L / LM.TPL_LEN                                  # 템플릿 px -> 원본 px (대략)
        for t in range(trials):
            C = []
            for _ in range(2000):
                if len(C) == per_trial:
                    break
                i = rng.integers(len(xx))
                q = np.array([xx[i], yy[i]], float)
                if all(np.linalg.norm(q - c) > 25 for c in C):
                    C.append(q)
            C = np.array(C)
            par = [combos[(ci + k) % len(combos)] for k in range(len(C))]
            ci += len(C)
            co = reg["inv"](C)
            img = plant(rgba, co, [s * s_ratio for _, s in par], [c for c, _ in par], color)
            for gain in (0.8, 1.0, 1.2):
                im2 = photometric(img, rng, gain, 1.0, np.ones(3), 0.0) if gain != 1.0 else img
                w = warp_with(reg, im2, tm)
                bw = LM.blue_index(w)
                pks = [float(bw[int(round(q[1])), int(round(q[0]))]) for q in C]
                dbs = [pk - local_bg(bw, valid, int(round(q[1])), int(round(q[0]))) for pk, q in zip(pks, C)]
                thr_used = {key: detect(w, tm, **kw)[1] for key, kw in DET_SETTINGS.items()}
                for key, kw in DET_SETTINGS.items():
                    if gain != 1.0 and key not in ("auto_k4", "fixed_50"):
                        continue
                    blue = detect(w, tm, **kw)[0]
                    new = blue & ~ndi.binary_dilation(ref[key], iterations=2)
                    lab = measure.label(new)
                    props = {pp.label: pp for pp in measure.regionprops(lab)}
                    used = set()
                    for (c_, s_), q, db, pk in zip(par, C, dbs, pks):
                        r0, c0 = int(round(q[1])), int(round(q[0]))
                        win = lab[max(0, r0 - 3):r0 + 4, max(0, c0 - 3):c0 + 4]
                        labs = [v for v in np.unique(win) if v]
                        hit = labs[0] if labs else 0
                        err = area = np.nan
                        if hit:
                            used.add(hit)
                            cy, cx = props[hit].centroid
                            err = float(np.hypot(cx - q[0], cy - q[1]))
                            area = props[hit].area
                        rows.append(dict(view=view, image=os.path.relpath(p, LM.DATA_DIR), trial=t, gain=gain,
                                         setting=key, contrast=c_, sigma_tpl=s_, delta_b=db, peak_b=pk,
                                         threshold=thr_used[key], detected=int(bool(hit)),
                                         loc_err_px=err, area_px=area, nominal_area_px=np.pi * (1.1774 * s_) ** 2))
                    fps.append(dict(view=view, image=os.path.relpath(p, LM.DATA_DIR), trial=t, gain=gain,
                                    setting=key, n_new_components=int(lab.max()), n_false=int(lab.max()) - len(used)))
        print("spots", view, os.path.basename(p), flush=True)
    write(os.path.join(out, "spots.csv"), rows)
    write(os.path.join(out, "spots_fp.csv"), fps)
    write(os.path.join(out, "spots_real.csv"), real)


def local_bg(b, valid, r, c, rad=15):
    y0, x0 = max(0, r - rad), max(0, c - rad)
    w, v = b[y0:r + rad + 1, x0:c + rad + 1], valid[y0:r + rad + 1, x0:c + rad + 1]
    return float(np.median(w[v])) if v.any() else 0.0


DET_SETTINGS = {"auto_k3": dict(k=3.0), "auto_k4": dict(k=4.0), "auto_k6": dict(k=6.0),
                "fixed_35": dict(thr=35.0), "fixed_50": dict(thr=50.0)}


# ----------------------------------------------------------------------------
def exp_params(out):
    rows = []
    for view, seg, folder, p in V.dataset():
        tpl, tm = tpl_of(view)
        reg = V.register(LM.load_rgba(p), tpl, tm, proposed(view))
        for k in (2, 3, 4, 5, 6, 8):
            for ma in (1, 3, 6, 10, 20):
                if k != 4 and ma != 6:
                    continue
                b, thr = detect(reg["warped"], tm, k=k, min_area=ma)
                rows.append(dict(view=view, segment=seg, image=os.path.relpath(p, LM.DATA_DIR), k=k, min_area=ma,
                                 threshold=thr, n_spots=int(measure.label(b).max()), blue_area_px=int(b.sum())))
        print("params", os.path.basename(p))
    write(os.path.join(out, "params.csv"), rows)


# ----------------------------------------------------------------------------
def exp_demo(out, sigma=6.0, common=0.5):
    os.makedirs(os.path.join(out, "demo_arrays"), exist_ok=True)
    rows, summ = [], []
    groups = defaultdict(list)
    for d in V.dataset():
        groups[(d[0], d[1])].append(d)
    for (view, seg), items in sorted(groups.items()):
        tpl, tm = tpl_of(view)
        masks, warps = [], []
        for _, _, folder, p in items:
            reg = V.register(LM.load_rgba(p), tpl, tm, proposed(view))
            blue, thr = detect(reg["warped"], tm)
            dots = LM.extract_dots(blue, LM.blue_index(reg["warped"]), reg, tpl)
            rows += [dict(view=view, segment=seg, image=os.path.relpath(p, LM.DATA_DIR), threshold=round(thr, 1), **d)
                     for d in dots]
            masks.append(blue)
            warps.append(LM.on_white(reg["warped"]))
        freq = np.mean([ndi.gaussian_filter(b.astype(float), sigma) > 0.15 for b in masks], axis=0)
        freq[~tm] = 0
        np.savez_compressed(os.path.join(out, "demo_arrays", f"freq_{view}_{seg}.npz"), freq=freq,
                            masks=np.array(masks), mean=np.mean(warps, axis=0).astype(np.uint8))
        com = freq >= common
        yy, xx = np.nonzero(com)
        lmk = tpl["landmarks"]
        P = np.array(tpl["points"])
        tip = P[lmk["tip"]]
        Lt = (P[lmk["root_a"]] + P[lmk["root_b"]])[0] / 2 - tip[0]
        summ.append(dict(view=view, segment=seg, n_images=len(items),
                         spots_per_image=np.mean([sum(r["image"] == os.path.relpath(it[3], LM.DATA_DIR) for r in rows)
                                                  for it in items]),
                         common_area_pct_template=100 * com.sum() / tm.sum(),
                         common_centroid_xrel=((xx.mean() - tip[0]) / Lt) if len(xx) else np.nan,
                         common_centroid_yrel=((yy.mean() - tip[1]) / Lt) if len(xx) else np.nan,
                         any_area_pct_template=100 * (freq > 0).sum() / tm.sum()))
        print("demo", view, seg)
    write(os.path.join(out, "demo_dots.csv"), rows)
    write(os.path.join(out, "demo_summary.csv"), summ)


EXAMPLE = ("palmar", "C7")       # 예시 그림(Fig 1A, 2, 4A, 6)에 쓰는 신경·부위


def exp_example(out, sigma=6.0, common=0.5):
    """예시 그룹 하나(EXAMPLE)를 유사변환과 TPS로 각각 통합 -> example_arrays.npz, example.json"""
    view, seg = EXAMPLE
    tpl, tm = tpl_of(view)
    items = [d for d in V.dataset([view]) if d[1] == seg]
    res, arr = {}, {}
    for m in ("similarity", proposed(view)):
        masks, warps, dots_all = [], [], []
        for _, _, _, p in items:
            reg = V.register(LM.load_rgba(p), tpl, tm, m)
            blue, thr = detect(reg["warped"], tm)
            dots = LM.extract_dots(blue, LM.blue_index(reg["warped"]), reg, tpl)
            masks.append(blue)
            warps.append(reg["warped"])
            dots_all.append(np.array([[d["x_tpl"], d["y_tpl"]] for d in dots]).reshape(-1, 2))
        sm = [ndi.gaussian_filter(b.astype(float), sigma) > 0.15 for b in masks]
        freq = np.mean(sm, axis=0)
        freq[~tm] = 0
        key = "tps" if m != "similarity" else "similarity"
        arr[f"freq_{key}"] = freq
        arr[f"masks_{key}"] = np.array(masks)
        arr[f"warps_{key}"] = np.array(warps)
        for k, d in enumerate(dots_all):
            arr[f"dots_{key}_{k}"] = d
        res[key] = dict(method=m, common_area_pct=100 * (freq >= common).sum() / tm.sum(),
                        any_area_pct=100 * (freq > 0).sum() / tm.sum(),
                        pairwise_dice=float(np.mean([V.dice(sm[i], sm[j])
                                                     for i, j in itertools.combinations(range(len(sm)), 2)])),
                        spots_per_image=float(np.mean([len(d) for d in dots_all])))
        print("example", key, res[key])
    res.update(view=view, segment=seg, n_images=len(items), images=[os.path.relpath(d[3], LM.DATA_DIR) for d in items])
    np.savez_compressed(os.path.join(out, "example_arrays.npz"), **arr)
    with open(os.path.join(out, "example.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


EXPS = dict(hetero=exp_hetero, register=exp_register, deform=exp_deform, pose=exp_pose,
            spots=exp_spots, params=exp_params, demo=exp_demo, example=exp_example, resummarize=exp_resummarize)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=list(EXPS) + ["all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: 데이터의 i번째 조각만 (pose 병렬 실행용)")
    a = ap.parse_args()
    global SHARD
    SHARD = tuple(int(v) for v in a.shard.split("/"))
    os.makedirs(a.out, exist_ok=True)
    for name in (EXPS if a.exp == "all" else [a.exp]):
        t0 = time.time()
        EXPS[name](a.out)
        print(f"== {name} 완료 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
