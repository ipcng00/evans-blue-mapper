"""논문 추가 분석 (사진 단위 민감도 분석, 같은 규칙의 반점 일치도, 제안 방식 요약) -> <root>/build/extra.json
사용 예) python paper_extra_analyses.py --root "<논문 폴더>"   (results/ 아래에 paper_experiments.py, validation_eval.py 결과가 있어야 함)
  1) 제안 방식(손바닥면만 DTW, 나머지 arc-length) 성능 (Table 2, Table 4 행)
  2) 사진 단위로 평균한 뒤의 민감도 검정 (쌍·표지점 반복 관측 보정)
  3) 파란 점 검출: 같은 반경(4/6/8 px)·1:1 매칭으로 자동-평가자, 평가자-평가자 비교, 애매한 점 제외 결과,
     사진 단위 부트스트랩 95% CI, 개수 차이
  4) Table 3 해상도 행의 >10% 비율, Table 2 지표별 n
"""
import os, sys, json, csv
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.optimize import linear_sum_assignment
from skimage import measure

sys.stdout.reconfigure(encoding="utf-8")
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--root", required=True, help="results/ 와 build/ 가 있는 논문 폴더")
ROOT = _ap.parse_args().root
HERE = os.path.join(ROOT, "build")
os.makedirs(HERE, exist_ok=True)
RES = os.path.join(ROOT, "results")
import limb_mapping as LM
import limb_validation as V
from validation_annotate_gui import load_annotations

L = 500.0
OUT = {}
rng = np.random.default_rng(20261123)


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ---------------------------------------------------------------- 1) 제안 방식
per = pd.read_csv(os.path.join(RES, "register_per_image.csv"))
pairs = pd.read_csv(os.path.join(RES, "register_pairs.csv"))
prop_of = lambda v: "tps_dtw" if v == "palmar" else "tps_arc"
pp = per[per.apply(lambda r: r["method"] == prop_of(r["view"]), axis=1)]
pq = pairs[pairs.apply(lambda r: r["method"] == prop_of(r["view"]), axis=1)]
pq = pq[~pq.same_animal]
pq_s = pq[pq.same_segment]
OUT["proposed"] = dict(dice_mean=pp.dice.mean(), dice_sd=pp.dice.std(ddof=1),
                       bd_mean=pp.boundary_mean_px.mean(), bd_sd=pp.boundary_mean_px.std(ddof=1),
                       hd95_mean=pp.boundary_hd95_px.mean(),
                       ncc_mean=pq.ncc_limb.mean(), ncc_sd=pq.ncc_limb.std(ddof=1),
                       spot_same=pq_s.blue_dice_smooth.mean(), spot_diff=pq[~pq.same_segment].blue_dice_smooth.mean(),
                       time=pp.t_register_s.mean())
pa = pairs[~pairs.same_animal]
OUT["n_pairs_all"] = int((pa.method == "tps_arc").sum())
OUT["n_pairs_same"] = int(((pa.method == "tps_arc") & pa.same_segment).sum())
OUT["n_pairs_diff"] = int(((pa.method == "tps_arc") & ~pa.same_segment).sum())

# ---------------------------------------------------------------- 2) 사진 단위 민감도
# 쌍 지표: 사진마다 그 사진이 포함된 쌍의 평균 -> 사진 60장(보기별)에 대해 Wilcoxon
def per_photo(df, col):
    rows = []
    for (v, m), g in df.groupby(["view", "method"]):
        idx = sorted(set(g.i) | set(g.j))
        for k in idx:
            s = g[(g.i == k) | (g.j == k)][col]
            if len(s):
                rows.append(dict(view=v, method=m, k=k, val=s.mean()))
    return pd.DataFrame(rows).pivot_table(index=["view", "k"], columns="method", values="val")


sens = {}
for col, sub in (("ncc_limb", pa), ("blue_dice_smooth", pa[pa.same_segment])):
    pv = per_photo(sub, col)
    ps = {m: float(wilcoxon(pv["tps_arc"] - pv[m]).pvalue) for m in ("axis", "similarity", "affine")}
    sens[col] = dict(n=int(len(pv)), p=ps, p_holm=dict(zip(ps, holm(list(ps.values())))))
# 표지점: 사진마다 (표지점 3개 x 평가자 2명) 평균
tp_ = pd.read_csv(os.path.join(RES, "validation", "tre_points.csv")).dropna(subset=["tre_pctL"])
lp = tp_.groupby(["image", "method"]).tre_pctL.mean().unstack()
ps = {m: float(wilcoxon(lp["tps_arc"] - lp[m]).pvalue) for m in ("axis", "similarity", "affine", "tps_dtw")}
sens["landmark"] = dict(n=int(len(lp)), p=ps, p_holm=dict(zip(ps, holm(list(ps.values())))),
                        median={m: float(lp[m].median()) for m in lp.columns})
# 제안 방식의 표지점 편차
tp_["prop"] = tp_.apply(lambda r: r["method"] == prop_of(r["view"]), axis=1)
t_prop = tp_[tp_.prop].tre_pctL
OUT["landmark_proposed"] = dict(mean=float(t_prop.mean()), sd=float(t_prop.std(ddof=1)), median=float(t_prop.median()),
                                q1=float(t_prop.quantile(.25)), q3=float(t_prop.quantile(.75)))
# 주 분석(표지점 위치 단위) P 값의 Holm 보정
key = ["annotator", "image", "landmark"]
w = tp_.pivot_table(index=key, columns="method", values="tre_pctL")
ps = {m: float(wilcoxon(w["tps_arc"] - w[m]).pvalue) for m in ("axis", "similarity", "affine", "tps_dtw")}
sens["landmark_points_holm"] = dict(zip(ps, holm(list(ps.values()))))
OUT["sensitivity"] = sens

# ---------------------------------------------------------------- 3) 파란 점
ann_dir = os.path.join(LM.DATA_DIR, "annotations")
anns = load_annotations(ann_dir)
by_img = defaultdict(list)
for x in anns:
    by_img[x["image"]].append(x)
tpls, tms = {}, {}
rows = []
n_unc = defaultdict(int)
n_all = defaultdict(int)
for k, (img, lst) in enumerate(sorted(by_img.items())):
    view = lst[0]["view"]
    if view not in tpls:
        tpls[view] = LM.load_template(view)
        tms[view] = LM.template_mask(tpls[view])
    tpl, tm = tpls[view], tms[view]
    rgba = LM.load_rgba(os.path.join(LM.DATA_DIR, img))
    reg = V.register(rgba, tpl, tm, prop_of(view))
    thr = LM.auto_threshold(reg["warped"], tm, 4.0)
    blue = LM.detect_blue(reg["warped"], tm, thr, 6)
    lab = measure.label(blue)
    cents = np.array([p.centroid[::-1] for p in measure.regionprops(lab)]).reshape(-1, 2)
    pts = {}
    for x in lst:
        S = x["spots"]
        n_all[x["annotator"]] += len(S)
        n_unc[x["annotator"]] += sum(1 for s in S if s[2])
        P = V.forward_points(reg, np.array([s[:2] for s in S], float).reshape(-1, 2))
        unc = np.array([bool(s[2]) for s in S], bool)
        pts[x["annotator"]] = (P, unc)
    rows.append(dict(image=img, view=view, blue=blue, cents=cents, pts=pts))
    print(f"[{k + 1}/{len(by_img)}] {img}", flush=True)


def one2one(a, b, r):
    """헝가리안 1:1 매칭 (거리 r 이하만). 반환 tp, 매칭 안 된 a, b"""
    if len(a) == 0 or len(b) == 0:
        return 0, len(a), len(b)
    D = np.linalg.norm(a[:, None] - b[None], axis=2)
    C = np.where(D <= r, D, 1e6)
    i, j = linear_sum_assignment(C)
    tp = int((C[i, j] < 1e6).sum())
    return tp, len(a) - tp, len(b) - tp


def counts(rule, r, excl_unc=False):
    """사진별 (tp, fn, fp) : 기준 = 평가자. rule: 'mask' (원래 규칙), 'cent' (성분 중심 1:1), 'rr' (평가자 간)"""
    out = defaultdict(list)
    for row in rows:
        P = {a: (p[~u] if excl_unc else p) for a, (p, u) in row["pts"].items()}
        if rule == "rr":
            tp, fa, fb = one2one(P["R1"], P["R2"], r)
            out["R1 vs R2"].append((tp, fa, fb))
            continue
        for a in ("R1", "R2"):
            if rule == "mask":
                tp, fn, fp, _ = V.match_points_to_mask(P[a], row["blue"], r)
            else:
                tp, fn, fp = one2one(P[a], row["cents"], r)
            out[f"auto vs {a}"].append((tp, fn, fp))
    return out


def prf(lst):
    t = np.array(lst, float).sum(0)
    tp, fn, fp = t
    p, rc = tp / (tp + fp), tp / (tp + fn)
    return p, rc, 2 * p * rc / (p + rc)


def boot_f1(lst, B=2000):
    a = np.array(lst, float)
    f = []
    for _ in range(B):
        s = a[rng.integers(0, len(a), len(a))].sum(0)
        f.append(2 * s[0] / (2 * s[0] + s[1] + s[2]))
    return [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]


spot = {}
for r in (4, 6, 8):
    for rule in ("mask", "cent", "rr"):
        for ex in (False, True):
            for cmp_, lst in counts(rule, r, ex).items():
                p, rc, f1 = prf(lst)
                spot[f"{rule}_r{r}_{'excl' if ex else 'all'}_{cmp_}"] = dict(
                    precision=p, recall=rc, f1=f1, ci=boot_f1(lst))
OUT["spot"] = spot
OUT["uncertain"] = {a: dict(n_uncertain=n_unc[a], n_total=n_all[a]) for a in n_all}
# 개수 차이 (자동 - 평가자) : 원래 규칙의 성분 수
det = pd.read_csv(os.path.join(RES, "validation", "spot_detection.csv"))
OUT["count_diff"] = {a: dict(mean=float((g.n_auto - g.n_rater).mean()), sd=float((g.n_auto - g.n_rater).std(ddof=1)))
                     for a, g in det.groupby("annotator")}
rr = np.array([[len(row["pts"]["R1"][0]), len(row["pts"]["R2"][0])] for row in rows])
OUT["count_diff"]["R2-R1"] = dict(mean=float((rr[:, 1] - rr[:, 0]).mean()), sd=float((rr[:, 1] - rr[:, 0]).std(ddof=1)))

# ---------------------------------------------------------------- 4) Table 3 해상도 행 >10%
p = pd.read_csv(os.path.join(RES, "pose.csv"))
OUT["res_gt10"] = {str(sc): float((100 * g.coord_err_mean_px / L > 10).mean() * 100)
                   for sc, g in p[p.condition == "resolution"].groupby("scale")}
OUT["res_n"] = {str(sc): int(len(g)) for sc, g in p[p.condition == "resolution"].groupby("scale")}

json.dump(OUT, open(os.path.join(HERE, "extra.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
          default=float)
print(json.dumps({k: OUT[k] for k in OUT if k != "spot"}, indent=1, ensure_ascii=False, default=float))
for k, v in OUT["spot"].items():
    print(f"{k:40s} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} CI {v['ci'][0]:.2f}-{v['ci'][1]:.2f}")
