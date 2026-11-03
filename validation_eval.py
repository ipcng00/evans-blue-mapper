"""
검증 평가 스크립트 (평가자 주석 -> 정합 정확도 / 파란 점 검출 정확도 / 평가자 간 일치도)
================================================================
입력: <데이터 폴더>/annotations/<평가자ID>/*.json  (validation_annotate_gui.py 로 만든 주석)
출력 (--out 폴더):
  tre_points.csv      표지점별·사진별·방법별 템플릿 좌표와 TRE
  tre_summary.csv     방법별·부위별 TRE 요약 (+ 제안 방법과의 Wilcoxon 검정)
  interrater_landmarks.csv  평가자 간 표지점 거리 (템플릿 좌표, 원본 좌표)
  spot_detection.csv  사진별 자동 검출 vs 평가자 (TP/FN/FP)
  spot_summary.csv    평가자별·부위별 정밀도/재현율/F1, 개수 ICC
  fig_tre_by_method.png, fig_spot_counts.png

TRE 정의: 같은 표지점을 모든 사진에서 템플릿 좌표로 옮긴 뒤, 각 사진의 점과
         '나머지 사진들 평균 위치' 사이 거리 (leave-one-out 분산). 단위 px 와 %L
         (L = 템플릿의 발끝-기시부 길이 = TPL_LEN px)

사용 예)
  python validation_eval.py
  python validation_eval.py --annotations "D:/.../annotations" --out "D:/.../results/validation"
"""
import warnings

warnings.filterwarnings("ignore", message=r".*cryptography\.hazmat\.decrepit.*")
import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import limb_mapping as LM
import limb_validation as V
from validation_annotate_gui import load_annotations


def proposed_method(tpl):
    return "tps_dtw" if tpl.get("correspondence") == "dtw" else "tps_arc"


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


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=LM.DATA_DIR)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--methods", nargs="+", default=V.METHODS)
    ap.add_argument("--k-mad", type=float, default=4.0)
    ap.add_argument("--min-area", type=int, default=6)
    ap.add_argument("--tol", type=float, default=4.0, help="점-검출영역 매칭 허용 거리 (템플릿 px)")
    ap.add_argument("--tol-rater", type=float, default=8.0, help="평가자 간 파란 점 매칭 허용 거리 (템플릿 px)")
    ap.add_argument("--exclude-uncertain", action="store_true", help="애매한 점(Shift 클릭)은 정답에서 제외")
    a = ap.parse_args()
    ann_dir = a.annotations or os.path.join(a.data, "annotations")
    out = a.out or os.path.join(ann_dir, "_evaluation")
    os.makedirs(out, exist_ok=True)

    anns = load_annotations(ann_dir)
    if not anns:
        print("주석이 없습니다:", ann_dir)
        return
    by_img = defaultdict(list)
    for x in anns:
        by_img[x["image"]].append(x)
    print(f"주석 {len(anns)}개, 사진 {len(by_img)}장, 평가자 {sorted({x['annotator'] for x in anns})}")

    tpls, tmasks = {}, {}
    pts_rows, det_rows = [], []
    rater_lm = defaultdict(dict)          # (image, landmark) -> {rater: (tpl xy, orig xy, L_orig)}
    rater_spots = defaultdict(dict)       # image -> {rater: tpl pts}
    for k, (img, lst) in enumerate(sorted(by_img.items())):
        view = lst[0]["view"]
        if view not in tpls:
            tpls[view] = LM.load_template(view)
            tmasks[view] = LM.template_mask(tpls[view])
        tpl, tm = tpls[view], tmasks[view]
        rgba = LM.load_rgba(os.path.join(a.data, img))
        an = LM.analyze(rgba)
        lmi = np.cumsum([0] + LM.SEG[:-1])
        L_orig = float(np.linalg.norm((an["points"][lmi[2]] + an["points"][lmi[3]]) / 2 - an["points"][0]))
        prop = proposed_method(tpl)
        regs = {m: V.register(rgba, tpl, tm, m, analysis=an) for m in a.methods}
        print(f"[{k + 1}/{len(by_img)}] {img}  ({view})")
        for x in lst:
            r = x["annotator"]
            names = [n for n, p in x["landmarks"].items() if p is not None]
            P = np.array([x["landmarks"][n] for n in names], float).reshape(-1, 2)
            for m in a.methods:
                Q = V.forward_points(regs[m], P)
                for n, p, q in zip(names, P, Q):
                    pts_rows.append(dict(annotator=r, image=img, view=view, segment=x["segment"], landmark=n,
                                         method=m, x_orig=p[0], y_orig=p[1], x_tpl=q[0], y_tpl=q[1]))
                    if m == prop:
                        rater_lm[(img, n)][r] = (q, p, L_orig)
            # 파란 점
            S = [s for s in x["spots"] if not (a.exclude_uncertain and s[2])]
            Sp = V.forward_points(regs[prop], np.array([s[:2] for s in S], float).reshape(-1, 2))
            rater_spots[img][r] = Sp
        reg = regs[prop]
        thr = LM.auto_threshold(reg["warped"], tm, a.k_mad)
        blue = LM.detect_blue(reg["warped"], tm, thr, a.min_area)
        for r, Sp in rater_spots[img].items():
            tp, fn, fp, ncomp = V.match_points_to_mask(Sp, blue, a.tol)
            det_rows.append(dict(annotator=r, image=img, view=view, segment=lst[0]["segment"],
                                 threshold=round(thr, 1), n_rater=len(Sp), n_auto=ncomp, tp=tp, fn=fn, fp=fp))

    # ---------------------------------------------------------------- TRE
    L = LM.TPL_LEN
    groups = defaultdict(list)
    for i, row in enumerate(pts_rows):
        groups[(row["annotator"], row["view"], row["landmark"], row["method"])].append(i)
    for idx in groups.values():
        Q = np.array([[pts_rows[i]["x_tpl"], pts_rows[i]["y_tpl"]] for i in idx])
        for j, i in enumerate(idx):
            if len(idx) < 3:
                pts_rows[i]["tre_px"] = np.nan
                continue
            others = np.delete(Q, j, axis=0).mean(0)
            pts_rows[i]["tre_px"] = float(np.linalg.norm(Q[j] - others))
            pts_rows[i]["tre_pctL"] = 100 * pts_rows[i]["tre_px"] / L
    write(os.path.join(out, "tre_points.csv"), pts_rows)

    summ = []
    views = sorted({r["view"] for r in pts_rows}) + ["all"]
    for v in views:
        sel = [r for r in pts_rows if (v == "all" or r["view"] == v) and np.isfinite(r.get("tre_px", np.nan))]
        if not sel:
            continue
        key = lambda r: (r["annotator"], r["image"], r["landmark"])
        base = {}
        for m in a.methods:
            vals = {key(r): r["tre_px"] for r in sel if r["method"] == m}
            base[m] = vals
        prop_m = "tps_arc" if "tps_arc" in a.methods else a.methods[-1]
        for m in a.methods:
            t = np.array(list(base[m].values()))
            row = dict(view=v, method=m, label=V.METHOD_LABEL.get(m, m), n_points=len(t),
                       tre_mean_px=t.mean(), tre_sd_px=t.std(ddof=1) if len(t) > 1 else np.nan,
                       tre_median_px=np.median(t), tre_mean_pctL=100 * t.mean() / L)
            for ref in ("tps_arc", "tps_dtw"):
                if ref in base and ref != m:
                    ks = sorted(set(base[m]) & set(base[ref]))
                    if len(ks) >= 6:
                        d = np.array([base[m][q] - base[ref][q] for q in ks])
                        row[f"p_vs_{ref}"] = wilcoxon(d).pvalue if np.any(d) else 1.0
            summ.append(row)
    write(os.path.join(out, "tre_summary.csv"), summ)

    # ---------------------------------------------------------------- 평가자 간 표지점
    ir = []
    for (img, n), d in rater_lm.items():
        rs = sorted(d)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                (q1, p1, Lo), (q2, p2, _) = d[rs[i]], d[rs[j]]
                ir.append(dict(image=img, landmark=n, rater_a=rs[i], rater_b=rs[j],
                               dist_tpl_px=float(np.linalg.norm(q1 - q2)),
                               dist_pctL=100 * float(np.linalg.norm(p1 - p2)) / Lo))
    write(os.path.join(out, "interrater_landmarks.csv"), ir)

    # ---------------------------------------------------------------- 파란 점
    write(os.path.join(out, "spot_detection.csv"), det_rows)
    ss = []
    for r in sorted({d["annotator"] for d in det_rows}):
        for v in sorted({d["view"] for d in det_rows}) + ["all"]:
            sel = [d for d in det_rows if d["annotator"] == r and (v == "all" or d["view"] == v)]
            if not sel:
                continue
            tp, fn, fp = (sum(d[k] for d in sel) for k in ("tp", "fn", "fp"))
            P = tp / (tp + fp) if tp + fp else np.nan        # 근사: 성분 기준 FP, 점 기준 TP
            R = tp / (tp + fn) if tp + fn else np.nan
            ss.append(dict(compare=f"auto vs {r}", view=v, n_images=len(sel), tp=tp, fn=fn, fp=fp,
                           precision=P, recall=R, f1=2 * P * R / (P + R) if P + R else np.nan,
                           icc_count=V.icc21([[d["n_auto"], d["n_rater"]] for d in sel])))
    raters = sorted({r for d in rater_spots.values() for r in d})
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            tp = fa = fb = 0
            cnt = []
            for img, d in rater_spots.items():
                if raters[i] in d and raters[j] in d:
                    pairs, ua, ub = V.match_points(d[raters[i]], d[raters[j]], a.tol_rater)
                    tp, fa, fb = tp + len(pairs), fa + ua, fb + ub
                    cnt.append([len(d[raters[i]]), len(d[raters[j]])])
            if cnt:
                ss.append(dict(compare=f"{raters[i]} vs {raters[j]}", view="all", n_images=len(cnt), tp=tp,
                               fn=fa, fp=fb, precision=tp / (tp + fb) if tp + fb else np.nan,
                               recall=tp / (tp + fa) if tp + fa else np.nan,
                               f1=2 * tp / (2 * tp + fa + fb) if tp else np.nan, icc_count=V.icc21(cnt)))
    write(os.path.join(out, "spot_summary.csv"), ss)

    # ---------------------------------------------------------------- 그림
    sel = [r for r in pts_rows if np.isfinite(r.get("tre_px", np.nan))]
    if sel:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = [[100 * r["tre_px"] / L for r in sel if r["method"] == m] for m in a.methods]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(a.methods) + 1), [V.METHOD_LABEL.get(m, m) for m in a.methods], rotation=15)
        if ir:
            ax.axhline(np.median([d["dist_pctL"] for d in ir]), ls="--", color="gray", label="inter-rater (median)")
            ax.legend()
        ax.set_ylabel("Landmark TRE (% of limb length)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_tre_by_method.png"), dpi=200)
    if det_rows:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        x = [d["n_rater"] for d in det_rows]
        y = [d["n_auto"] for d in det_rows]
        ax.scatter(x, y, s=14)
        mx = max(x + y + [1])
        ax.plot([0, mx], [0, mx], "k--", lw=0.8)
        ax.set_xlabel("Spots counted by rater")
        ax.set_ylabel("Spots detected automatically")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_spot_counts.png"), dpi=200)
    print("결과 ->", out)
    for row in summ:
        if row["view"] == "all":
            print(f"  {row['label']:18s} TRE {row['tre_mean_pctL']:.2f} %L (n={row['n_points']})")
    for row in ss:
        print(f"  {row['compare']:14s} {row['view']:8s} P={row['precision']:.2f} R={row['recall']:.2f} "
              f"F1={row['f1']:.2f} ICC={row['icc_count']:.2f}")


if __name__ == "__main__":
    main()
