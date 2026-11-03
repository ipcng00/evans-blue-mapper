"""
쥐 앞다리 피부 사진 -> 공통 템플릿 윤곽선 정합 + 파란 점(염색) 좌표 추출

방법 요약
  1) 배경 제거: PNG 알파 채널(투명 배경)로 다리 마스크를 만든다.
  2) 랜드마크 자동 검출: 다리 내부 측지거리(geodesic distance)로
       발끝(tip) -> 손목(wrist, 양쪽) -> 사지 기시부(root, 양쪽)를 찾는다.
  3) 윤곽선을 랜드마크 사이 5개 구간으로 나누어 같은 개수의 점으로 재샘플링
       (tip→wrist_a→root_a→[근위부 피부]→root_b→wrist_b→tip)
     -> 사진 윤곽점과 템플릿 윤곽점이 1:1 대응된다.
  4) Thin-plate spline(TPS)으로 사진 윤곽선을 템플릿 윤곽선에 맞추고,
     내부는 비율대로 보간해서 다시 그린다. (회전·크기·모양 보정, 좌우 반전은 --flip 으로 수동)
  5) 정합된 그림에서 파란 점을 검출하여 템플릿 좌표(격자 셀 포함)와 원본 좌표를 저장한다.
     결과는 부위 폴더와 같은 위치의 '<부위 폴더명>_registered' 폴더에 저장된다.
  GUI로 하려면: python limb_mapping_gui.py

사용법
  # 1) 부위별 공통 템플릿 만들기 (mapping data/templates 에 저장)
  python limb_mapping.py template          # 자동 생성
  python limb_template_gui.py              # GUI로 만들고 윤곽선 직접 수정

  # 2) 폴더 하나를 템플릿에 정합 + 파란 점 좌표 추출
  python limb_mapping.py register --folder "mapping data/C6/내측(Ulnar side)"
  python limb_mapping.py register --folder "mapping data/C6/손등면" --view dorsal
  # 반대쪽 다리 등 좌우가 뒤집힌 사진은 --flip 으로 미러링
  python limb_mapping.py register --folder "mapping data/C6/내측(Ulnar side)" --flip 그림31.png
"""
import argparse
import csv
import json
import os
import string
import sys
import warnings

# scikit-image -> pooch -> paramiko 가 import 될 때 나오는 무해한 cryptography 경고 숨김 (import 전에 설정해야 함)
warnings.filterwarnings("ignore", message=r".*cryptography\.hazmat\.decrepit.*")

import matplotlib
import matplotlib.figure
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
from skimage import draw, graph, measure, morphology, transform

from limb_i18n import T, fig_name, view_en

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = ["Arial", "DejaVu Sans"]   # 그림 글자는 항상 영어
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "mapping data")
TEMPLATE_DIR = os.path.join(DATA_DIR, "templates")

VIEWS = {  # 영문 부위명: 폴더 이름에 들어있는 한글 키워드
    "medial": "내측",
    "lateral": "외측",
    "dorsal": "손등",
    "palmar": "손바닥",
}
VIEW_KO = {"medial": "내측(Ulnar)", "lateral": "외측(Radial)", "dorsal": "손등면", "palmar": "손바닥면"}

WORK = 500                       # 랜드마크 검출용 해상도(긴 변 px)
SEG = [40, 60, 200, 60, 40]      # tip→wrist_a→root_a→root_b→wrist_b→tip 구간별 점 개수
LM_NAMES = ["tip", "wrist_a", "root_a", "root_b", "wrist_b"]
TPL_LEN = 500                    # 템플릿에서 발끝~기시부 중점 거리(px)
GRID = 50                        # 템플릿 격자 간격(px)
MARGIN = 75
BLUE_MIN = 35                    # 파란색 지수(B - max(R,G))의 최소 임계값
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


# ----------------------------------------------------------------------------
# 기본 도구
# ----------------------------------------------------------------------------
def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def on_white(rgba):
    a = rgba[..., 3:4].astype(float) / 255
    return (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def detect_view(folder):
    """폴더 이름에 들어있는 키워드로 부위 판단 (templates 폴더의 템플릿 키워드 우선, 긴 키워드 우선)"""
    name = os.path.basename(os.path.normpath(folder))
    cands = [(k, v) for v, _, kws in available_views() for k in kws if k]
    cands += [(k, v) for v, k in VIEWS.items()] + [(v, v) for v in VIEWS]   # 한글 키워드 또는 영문 부위명
    for k, v in sorted(cands, key=lambda kv: -len(kv[0])):
        if k in name or k in name.lower():
            return v
    raise ValueError(T(f"폴더 이름에서 부위를 알 수 없습니다: {name}  (부위를 직접 지정하세요)",
                       f"Cannot determine the view from the folder name: {name} (select the view manually)"))


def work_mask(rgba):
    """알파 채널 -> 분석용 다리 마스크(가장 큰 덩어리, 구멍 메움)와 원본->분석 좌표 아핀 A (2x3)
    자세 정규화: 조직의 주축을 가로로 돌리고, 주축 길이가 WORK*0.9 px 가 되게 크기를 맞춘다.
    -> 사진의 회전·크기·여백(캔버스 크기)과 무관하게 같은 조건에서 랜드마크를 찾는다."""
    alpha = rgba[..., 3].astype(float) / 255
    yy, xx = np.nonzero(alpha > 0.5)
    if len(xx) < 10:
        raise ValueError(T("조직(불투명 영역)이 없습니다.", "No tissue (opaque region) found."))
    k = np.linspace(0, len(xx) - 1, min(len(xx), 200000)).astype(int)
    P = np.c_[xx[k], yy[k]].astype(float)
    c = P.mean(0)
    w, v = np.linalg.eigh(np.cov((P - c).T))
    e = v[:, 1]
    t = (P - c) @ e
    if np.mean(t ** 3) < 0:                                   # 방향: 질량이 많은(두꺼운) 쪽이 +x
        e = -e
        t = -t
    n = np.array([-e[1], e[0]])
    u = (P - c) @ n
    s = 0.9 * WORK / (t.max() - t.min())
    pad = 6
    Ww = int(np.ceil((t.max() - t.min()) * s)) + 2 * pad
    Hw = int(np.ceil((u.max() - u.min()) * s)) + 2 * pad
    off = np.array([pad - t.min() * s, pad - u.min() * s])
    A = np.c_[s * np.vstack([e, n]), off - s * np.vstack([e, n]) @ c]   # x_w = A[:, :2] @ p + A[:, 2]
    if s < 1:
        alpha = ndi.gaussian_filter(alpha, 0.5 / s)
    gy, gx = np.mgrid[0:Hw, 0:Ww]
    Ai = np.linalg.inv(A[:, :2])
    src = (np.c_[gx.ravel(), gy.ravel()] - A[:, 2]) @ Ai.T
    m = ndi.map_coordinates(alpha, [src[:, 1], src[:, 0]], order=1, cval=0).reshape(Hw, Ww) > 0.5
    m = morphology.binary_opening(m, morphology.disk(1))
    lab = measure.label(m)
    if lab.max() > 0:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndi.binary_fill_holes(m), A


def work_to_orig(A, pts):
    return (np.asarray(pts, float) - A[:, 2]) @ np.linalg.inv(A[:, :2]).T


def geodesic(mask, start):
    mcp = graph.MCP_Geometric(np.where(mask, 1.0, 1e6))
    d, _ = mcp.find_costs([start])
    d[~mask] = np.inf
    return d


def resample_open(pts, n):
    """열린 폴리라인을 호 길이 기준 n개 점으로 (끝점 제외)"""
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = np.linspace(0, seg[-1], n, endpoint=False)
    return np.c_[np.interp(t, seg, pts[:, 0]), np.interp(t, seg, pts[:, 1])]


def resample_closed(pts, n):
    return resample_open(np.vstack([pts, pts[:1]]), n)


def col_label(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = string.ascii_uppercase[r] + s
    return s


def grid_cell(x, y):
    return f"{col_label(int(x // GRID))}{int(y // GRID) + 1}"


# ----------------------------------------------------------------------------
# 1) 윤곽선 + 랜드마크 분석
# ----------------------------------------------------------------------------
def analyze(rgba, flip_toggle=False):
    """사진 하나의 윤곽선을 랜드마크 구간별로 재샘플링한다.
    반환 points: 원본 픽셀 좌표 (x, y), SEG 순서대로 sum(SEG)개"""
    m, A = work_mask(rgba)
    s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    dt = ndi.distance_transform_edt(m)
    center = np.unravel_index(np.argmax(dt), dt.shape)          # 가장 두꺼운 곳(근위부 피부)
    g0 = geodesic(m, center)
    tip = np.unravel_index(np.argmax(np.where(np.isfinite(g0), g0, -1)), g0.shape)  # 가장 먼 곳 = 발끝
    g = geodesic(m, tip)
    S = np.max(g[np.isfinite(g)])

    # 발끝으로부터의 측지거리 구간별 폭 프로파일
    h = 3
    bins = np.arange(0, S, h)
    gi = np.floor(np.where(np.isfinite(g), g, -1) / h).astype(int)
    widths = np.bincount(gi[gi >= 0], minlength=len(bins))[: len(bins)] / h
    widths = ndi.gaussian_filter1d(widths.astype(float), 2)

    def rng(a, b):
        return (bins >= a) & (bins <= b)

    i_paw = np.flatnonzero(rng(0.05 * S, 0.35 * S))
    s_paw = bins[i_paw[np.argmax(widths[i_paw])]]
    i_wr = np.flatnonzero(rng(s_paw, min(2.0 * s_paw, 0.6 * S)))
    s_wrist = bins[i_wr[np.argmin(widths[i_wr])]] if len(i_wr) else s_paw
    i_fa = rng(s_wrist, min(s_wrist + s_paw, 0.7 * S))
    w_fore = np.median(widths[i_fa]) if i_fa.any() else widths.max() / 3
    cand = np.flatnonzero((bins > s_wrist + 0.5 * s_paw) & (widths > 1.6 * w_fore))
    s_root = bins[cand[0]] if len(cand) else 0.7 * S
    s_root = float(np.clip(s_root, s_wrist + 0.3 * s_paw, 0.85 * S))

    # 윤곽선 (x, y) - 촘촘하게 재샘플링
    cs = measure.find_contours(np.pad(m, 1).astype(float), 0.5)
    c = max(cs, key=len)[:-1] - 1
    dense = resample_closed(c[:, ::-1], 2000)                    # (x, y)
    idx = ndi.distance_transform_edt(~np.isfinite(g), return_distances=False, return_indices=True)
    g_full = g[tuple(idx)]
    r = np.clip(np.round(dense[:, 1]).astype(int), 0, m.shape[0] - 1)
    q = np.clip(np.round(dense[:, 0]).astype(int), 0, m.shape[1] - 1)
    gc = g_full[r, q]
    N = len(dense)
    i_tip = int(np.argmin(gc))
    fwd = gc[(i_tip + np.arange(N)) % N]
    bwd = gc[(i_tip - np.arange(N)) % N]

    def first(arr, v):
        k = np.flatnonzero(arr >= v)
        return int(k[0]) if len(k) else N // 4

    kW_f, kR_f = first(fwd, s_wrist), first(fwd, s_root)
    kW_b, kR_b = first(bwd, s_wrist), first(bwd, s_root)
    P = lambda k: dense[(i_tip + k) % N]
    tipxy = dense[i_tip]
    root_mid = (P(kR_f) + P(-kR_b)) / 2

    # 표준 좌표계: 발끝 원점, 발끝->기시부 중점 = +x (회전만, 좌우 반전 없음)
    e = (root_mid - tipxy) / (np.linalg.norm(root_mid - tipxy) + 1e-9)
    n = np.array([-e[1], e[0]])
    # 같은 쪽 다리를 털 쪽에서 찍었다면 사진끼리 좌우가 뒤집힐 일은 없으므로 기본은 미러링하지 않는다.
    # (회전·크기·모양 차이만 보정) 반대쪽 다리이거나 좌우 반전된 사진만 --flip 으로 지정.
    flip = bool(flip_toggle)
    sgn = -1 if flip else 1

    # side a = 표준 좌표계에서 아래쪽(y>0) 가장자리
    off_f = ((P(kW_f) - tipxy) @ n + (P(kR_f) - tipxy) @ n) * sgn
    off_b = ((P(-kW_b) - tipxy) @ n + (P(-kR_b) - tipxy) @ n) * sgn
    if off_f >= off_b:
        order = (i_tip + np.arange(N)) % N
        kWa, kRa, kWb, kRb = kW_f, kR_f, kW_b, kR_b
    else:
        order = (i_tip - np.arange(N)) % N
        kWa, kRa, kWb, kRb = kW_b, kR_b, kW_f, kR_f
    ring = dense[order]
    cuts = [0, kWa, kRa, N - kRb, N - kWb, N]
    ok = all(cuts[i] < cuts[i + 1] for i in range(5))
    if not ok:  # 랜드마크 순서가 꼬이면 비율로 대체
        cuts = [0, int(0.08 * N), int(0.2 * N), int(0.8 * N), int(0.92 * N), N]
    pts = []
    for i in range(5):
        seg = ring[cuts[i]: cuts[i + 1] + 1] if i < 4 else np.vstack([ring[cuts[4]:], ring[:1]])
        pts.append(resample_open(seg, SEG[i]))
    pts = work_to_orig(A, np.vstack(pts))                      # 원본 픽셀 좌표

    do = work_to_orig(A, dense)                                # 사진 가장자리에서 잘린 윤곽 비율
    tol = 1.5 / s
    border = np.mean((do[:, 0] < tol) | (do[:, 1] < tol) |
                     (do[:, 0] > rgba.shape[1] - 1 - tol) | (do[:, 1] > rgba.shape[0] - 1 - tol))
    return dict(points=pts, flip=bool(flip), landmarks_ok=ok,
                truncated=float(border), scale=s,
                contour=work_to_orig(A, ring), contour_cuts=cuts[:5],      # 촘촘한 윤곽선(원본 좌표)과 랜드마크 위치 (DTW 대응용)
                s=dict(paw=float(s_paw), wrist=float(s_wrist), root=float(s_root), max=float(S)))


def canonical(pts, flip):
    """발끝 원점, 발끝->기시부 중점 = +x, 길이 1, 근위부 피부 위쪽"""
    lm = np.cumsum([0] + SEG[:-1])
    tip = pts[lm[0]]
    rm = (pts[lm[2]] + pts[lm[3]]) / 2
    e = (rm - tip) / np.linalg.norm(rm - tip)
    n = np.array([-e[1], e[0]])
    q = np.c_[(pts - tip) @ e, (pts - tip) @ n] / np.linalg.norm(rm - tip)
    if flip:
        q[:, 1] *= -1
    return q


# ----------------------------------------------------------------------------
# 1-2) 모양 기반 탄력 대응 (DTW) - 템플릿 옵션 "correspondence": "dtw"
#   랜드마크 사이 구간을 길이 비율 대신 윤곽선 굴곡(곡률)이 가장 잘 맞도록 대응시킨다.
#   발가락처럼 모양이 복잡한 부위에 효과적. 근위부(root_a→root_b) 구간은 잘린 피부라 길이 비율 유지.
# ----------------------------------------------------------------------------
DTW_UP = 4                      # 구간별 재샘플 배율
DTW_SEGS = (0, 1, 3, 4)         # DTW를 적용할 구간 (2 = 근위부 제외)
DTW_LAMBDA = 1.0                # 길이 비율에서 벗어나는 것에 대한 벌점
DTW_BAND = 0.25                 # 구간 길이 대비 최대 어긋남


def _curvature(pts, sigma):
    x = ndi.gaussian_filter1d(pts[:, 0], sigma, mode="wrap")
    y = ndi.gaussian_filter1d(pts[:, 1], sigma, mode="wrap")
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)


def _seg_features(dense, cuts, sign=1.0):
    """닫힌 윤곽선 -> 구간별 (재샘플 점, 정규화 곡률 특징)"""
    L = np.linalg.norm((dense[cuts[2]] + dense[cuts[3]]) / 2 - dense[cuts[0]])
    k = _curvature(dense, 8 * len(dense) / 2000) * L * sign
    ring = np.vstack([dense, dense[:1]])
    kr = np.r_[k, k[:1]]
    cc = list(cuts) + [len(dense)]
    out = []
    for i in range(5):
        p, kk = ring[cc[i]:cc[i + 1] + 1], kr[cc[i]:cc[i + 1] + 1]
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        t = np.linspace(0, arc[-1], DTW_UP * SEG[i], endpoint=False)
        out.append((np.c_[np.interp(t, arc, p[:, 0]), np.interp(t, arc, p[:, 1])],
                    np.tanh(np.interp(t, arc, kk) / 20)))
    return out


def template_features(tpl_pts):
    """템플릿 윤곽점(sum(SEG)개) -> 구간별 특징"""
    P = np.asarray(tpl_pts, float)
    cum = np.cumsum([0] + SEG)
    dense, cuts = [], []
    for i in range(5):
        seg = np.vstack([P[cum[i]:cum[i + 1]], P[cum[i + 1] % len(P)][None]])
        cuts.append(sum(len(d) for d in dense))
        dense.append(resample_open(seg, 5 * SEG[i]))
    return _seg_features(np.vstack(dense), cuts)


def _dtw_positions(fa, fb):
    """기울기 1/2~2 로 제한한 DTW. 반환: 템플릿 샘플마다 대응되는 사진 샘플 위치 (실수, 끝점 고정)"""
    n, m = len(fa), len(fb)
    ta, tb = np.linspace(0, 1, n), np.linspace(0, 1, m)
    C = (fa[:, None] - fb[None, :]) ** 2 + DTW_LAMBDA * (ta[:, None] - tb[None, :]) ** 2
    C[np.abs(ta[:, None] - tb[None, :]) > DTW_BAND] = np.inf
    D = np.full((n, m), np.inf)
    B = np.zeros((n, m), np.int8)
    D[0, 0] = C[0, 0]
    for i in range(1, n):
        st = np.full((3, m), np.inf)
        st[0, 1:] = D[i - 1, :-1]                          # (1,1)
        st[1, 2:] = D[i - 1, :-2] + C[i, 1:-1]             # (1,2)
        if i >= 2:
            st[2, 1:] = D[i - 2, :-1] + C[i - 1, 1:]       # (2,1)
        b = np.argmin(st, axis=0)
        D[i] = C[i] + st[b, np.arange(m)]
        B[i] = b
    if not np.isfinite(D[-1, -1]):
        return None
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        b = B[i, j]
        if b == 0:
            i, j = i - 1, j - 1
        elif b == 1:
            path.append((i, j - 1))
            i, j = i - 1, j - 2
        else:
            path.append((i - 1, j))
            i, j = i - 2, j - 1
        if i < 0 or j < 0:
            return None
        path.append((i, j))
    path = np.array(path)
    sums = np.bincount(path[:, 1], weights=path[:, 0], minlength=m)
    cnts = np.bincount(path[:, 1], minlength=m)
    return sums / np.maximum(cnts, 1)


def dtw_points(analysis, tpl_feats):
    """사진 윤곽점(sum(SEG)개, 원본 좌표)을 DTW 대응으로 다시 만든다"""
    segs = _seg_features(analysis["contour"], analysis["contour_cuts"], -1.0 if analysis["flip"] else 1.0)
    out = []
    for i in range(5):
        p, fa = segs[i]
        pos = _dtw_positions(fa, tpl_feats[i][1]) if i in DTW_SEGS else None
        if pos is None:
            pos = np.arange(DTW_UP * SEG[i], dtype=float)
        pos = pos[::DTW_UP]
        pe = np.vstack([p, segs[(i + 1) % 5][0][:1]])
        k = np.arange(len(pe))
        out.append(np.c_[np.interp(pos, k, pe[:, 0]), np.interp(pos, k, pe[:, 1])])
    return np.vstack(out)


def template_shape(analyses, correspondence="arc", iters=3):
    """여러 사진 -> 공통 템플릿 윤곽점. dtw 면 평균 -> DTW 재대응 -> 평균 을 iters 번 반복"""
    tpl_pts = mean_shape(analyses)
    if correspondence == "dtw":
        for _ in range(iters):
            feats = template_features(tpl_pts)
            tpl_pts = mean_shape([dict(points=dtw_points(r, feats), flip=r["flip"]) for r in analyses])
    return tpl_pts


# ----------------------------------------------------------------------------
# 2) 공통 템플릿 만들기
# ----------------------------------------------------------------------------
def procrustes_align(X, Y):
    """X를 Y에 유사변환(회전+크기+이동, 반전 없음)으로 맞춤"""
    mx, my = X.mean(0), Y.mean(0)
    A, B = X - mx, Y - my
    U, S, Vt = np.linalg.svd(A.T @ B)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    sc = (S * np.diag(D)).sum() / (A ** 2).sum()
    return sc * A @ R + my


def collect_view_images(view):
    items = []
    for c in sorted(d for d in os.listdir(DATA_DIR) if d.upper().startswith("C")):
        cdir = os.path.join(DATA_DIR, c)
        if not os.path.isdir(cdir):
            continue
        for d in sorted(os.listdir(cdir)):
            if ((VIEWS[view] in d or view in d.lower()) and not d.endswith("_registered")
                    and os.path.isdir(os.path.join(cdir, d))):
                for f in list_images(os.path.join(cdir, d)):
                    items.append(os.path.join(cdir, d, f))
    return items


def mean_shape(analyses):
    """여러 사진의 윤곽선(analyze 결과) -> 평균 윤곽선 (Generalized Procrustes Analysis)
    반환: 발끝 원점, 발끝->기시부 중점 = +x, 길이 TPL_LEN 인 좌표 (sum(SEG)개 점)"""
    shapes = [canonical(r["points"], r["flip"]) for r in analyses]
    mean = np.mean(shapes, axis=0)
    for _ in range(10):
        al = [procrustes_align(s_, mean) for s_ in shapes]
        new = np.mean(al, axis=0)
        new = new - new.mean(0)
        new /= np.linalg.norm(new)
        if np.linalg.norm(new - mean) < 1e-7:
            break
        mean = new
    mean = np.c_[ndi.gaussian_filter1d(mean[:, 0], 1.5, mode="wrap"),
                 ndi.gaussian_filter1d(mean[:, 1], 1.5, mode="wrap")]
    return canonical(mean, False) * TPL_LEN


def place_on_canvas(pts, extra=()):
    """윤곽선이 여백(MARGIN)을 두고 격자 배수 크기의 캔버스 가운데 오도록 이동.
    반환: (이동된 점, 캔버스 [W, H], 이동량) - extra 좌표 배열들도 같은 양만큼 이동해 함께 반환"""
    x0, y0 = pts.min(0) - MARGIN
    x1, y1 = pts.max(0) + MARGIN
    W = int(np.ceil((x1 - x0) / GRID) * GRID)
    H = int(np.ceil((y1 - y0) / GRID) * GRID)
    off = np.array([x0 - (W - (x1 - x0)) / 2, y0 - (H - (y1 - y0)) / 2])
    return pts - off, [W, H], off, [np.asarray(e) - off for e in extra]


def make_template(view, pts, sources=(), label=None, keywords=None, nodes=None, node_landmarks=None,
                  correspondence="arc"):
    """템플릿 dict 생성 (pts: sum(SEG)개 윤곽점, 랜드마크는 각 구간의 첫 점)
    correspondence: "arc" = 랜드마크 사이 길이 비율 대응, "dtw" = 모양 기반 탄력 대응"""
    extra = [nodes] if nodes is not None else []
    pts, canvas, off, moved = place_on_canvas(np.asarray(pts, float), extra)
    lm = np.cumsum([0] + SEG[:-1])
    tpl = dict(view=view, label=label or VIEW_KO.get(view, view),
               keywords=list(keywords) if keywords else ([VIEWS[view]] if view in VIEWS else []),
               correspondence=correspondence, canvas=canvas, grid=GRID, segments=SEG,
               landmarks={k: int(i) for k, i in zip(LM_NAMES, lm)},
               points=np.round(pts, 2).tolist(), n_images=len(sources), sources=list(sources))
    if nodes is not None:   # 템플릿 제작 GUI에서 다시 편집할 수 있도록 조절점 저장
        tpl["nodes"] = np.round(moved[0], 2).tolist()
        tpl["node_landmarks"] = [int(i) for i in node_landmarks]
    return tpl


def save_template(tpl, analyses=()):
    """templates 폴더에 template_<view>.json / .png / _check.png 저장"""
    view = tpl["view"]
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    with open(os.path.join(TEMPLATE_DIR, f"template_{view}.json"), "w", encoding="utf-8") as f:
        json.dump(tpl, f, ensure_ascii=False, indent=1)
    draw_template(tpl).save(os.path.join(TEMPLATE_DIR, f"template_{view}.png"))
    template_check_figure(tpl, analyses).savefig(
        os.path.join(TEMPLATE_DIR, f"template_{view}_check.png"), dpi=100)


def build_template(view, max_trunc=0.08, correspondence="arc"):
    """명령줄용: C5~C8 폴더에서 같은 부위 사진을 모아 자동 템플릿 생성"""
    paths = collect_view_images(view)
    info = [(p, analyze(load_rgba(p))) for p in paths]
    keep = [(p, r) for p, r in info if r["truncated"] <= max_trunc and r["landmarks_ok"]]
    if len(keep) < 3:
        keep = [(p, r) for p, r in info if r["landmarks_ok"]] or info
    used = [os.path.relpath(p, DATA_DIR) for p, _ in keep]
    tpl = make_template(view, template_shape([r for _, r in keep], correspondence), used,
                        correspondence=correspondence)
    save_template(tpl, [r for _, r in keep])
    print(f"[template] {view}: {len(used)}/{len(paths)}장 사용, 대응={correspondence} "
          f"-> {tpl['canvas'][0]}x{tpl['canvas'][1]}px")
    return tpl


def _font(size):
    for f in ("malgun.ttf", "malgunbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def view_label(view_or_tpl):
    if isinstance(view_or_tpl, dict):
        return view_or_tpl.get("label") or VIEW_KO.get(view_or_tpl["view"], view_or_tpl["view"])
    for v, lab, _ in available_views():
        if v == view_or_tpl:
            return lab
    return VIEW_KO.get(view_or_tpl, view_or_tpl)


def draw_template(tpl, outline=True):
    """격자 + 윤곽선 템플릿 그림 (그림의 픽셀 좌표 = 템플릿 좌표)"""
    W, H = tpl["canvas"]
    G = tpl["grid"]
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f = _font(13)
    for x in range(0, W + 1, G):
        d.line([(x, 0), (x, H)], fill=(205, 205, 205), width=1)
    for y in range(0, H + 1, G):
        d.line([(0, y), (W, y)], fill=(205, 205, 205), width=1)
    for i, x in enumerate(range(0, W, G)):
        d.text((x + G / 2, 3), col_label(i), fill=(120, 120, 120), font=f, anchor="mt")
    for j, y in enumerate(range(0, H, G)):
        d.text((3, y + G / 2), str(j + 1), fill=(120, 120, 120), font=f, anchor="lm")
    if outline:
        pts = [tuple(p) for p in tpl["points"]]
        d.line(pts + [pts[0]], fill=(0, 0, 0), width=3, joint="curve")
        for k, i in tpl["landmarks"].items():
            x, y = tpl["points"][i]
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(220, 30, 30))
            d.text((x + 7, y + 7), k, fill=(220, 30, 30), font=f)
    d.text((W - 6, H - 6), f"{view_en(tpl)} template  (grid {G}px, n={tpl.get('n_images', 0)})",
           fill=(90, 90, 90), font=f, anchor="rd")
    return im


def template_mask(tpl):
    W, H = tpl["canvas"]
    p = np.array(tpl["points"])
    return draw.polygon2mask((H, W), p[:, ::-1])


def template_check_figure(tpl, analyses=()):
    """템플릿을 만든 사진들의 정규화 윤곽선과 공통 템플릿을 겹쳐 그린 Figure"""
    pts = np.array(tpl["points"])
    fig = matplotlib.figure.Figure(figsize=(9, 9 * tpl["canvas"][1] / tpl["canvas"][0]))
    ax = fig.add_subplot(111)
    ax.imshow(draw_template(tpl, outline=False))
    for r in analyses:
        q = procrustes_align(canonical(r["points"], r["flip"]), pts)
        ax.plot(*np.vstack([q, q[:1]]).T, lw=0.8, alpha=0.6)
    ax.plot(*np.vstack([pts, pts[:1]]).T, "k-", lw=2.5)
    ax.set_title(f"{view_en(tpl)}: individual outlines (color) vs common template (black)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def available_views():
    """templates 폴더의 template_*.json -> [(view, 표시 이름, 폴더 키워드 목록)]"""
    out = []
    if os.path.isdir(TEMPLATE_DIR):
        for f in sorted(os.listdir(TEMPLATE_DIR)):
            if f.startswith("template_") and f.endswith(".json"):
                try:
                    with open(os.path.join(TEMPLATE_DIR, f), encoding="utf-8") as fh:
                        t = json.load(fh)
                except (OSError, ValueError):
                    continue
                v = t.get("view") or f[len("template_"):-5]
                kws = t.get("keywords") or ([VIEWS[v]] if v in VIEWS else [])
                out.append((v, t.get("label") or VIEW_KO.get(v, v), kws))
    order = {v: i for i, v in enumerate(VIEWS)}
    return sorted(out, key=lambda x: (order.get(x[0], 99), x[0]))


def load_template(view):
    p = os.path.join(TEMPLATE_DIR, f"template_{view}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(T(f"{p} 가 없습니다. 템플릿 제작 GUI(limb_template_gui.py)로 먼저 만드세요.",
                                  f"{p} not found. Create it first with limb_template_gui.py."))
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# 3) 정합 (사진 -> 템플릿 캔버스)
# ----------------------------------------------------------------------------
def register_image(path, tpl, tmask, flip_toggle=False):
    """사진을 템플릿 캔버스로 다시 그린다.
    map_x/map_y: 템플릿 각 픽셀이 원본 사진의 어느 좌표에서 왔는지 (템플릿 밖은 NaN)"""
    rgba = load_rgba(path)
    r = analyze(rgba, flip_toggle)
    if tpl.get("correspondence") == "dtw":                # 모양 기반 탄력 대응
        r["points"] = dtw_points(r, template_features(tpl["points"]))
    P_img = r["points"]
    P_tpl = np.array(tpl["points"])
    sub = slice(None, None, 2)
    inv = RBFInterpolator(P_tpl[sub], P_img[sub], kernel="thin_plate_spline", degree=1, smoothing=1.0)

    H, W = tmask.shape
    yy, xx = np.nonzero(tmask)
    src = np.vstack([inv(np.c_[xx, yy][i:i + 60000]) for i in range(0, len(xx), 60000)])
    coords = [src[:, 1], src[:, 0]]
    out = np.zeros((H, W, 4), np.uint8)
    for ch in range(4):
        out[yy, xx, ch] = ndi.map_coordinates(rgba[..., ch].astype(float), coords, order=1, cval=0)
    map_x = np.full((H, W), np.nan, np.float32)
    map_y = np.full((H, W), np.nan, np.float32)
    map_x[yy, xx], map_y[yy, xx] = src[:, 0], src[:, 1]
    wtissue = out[..., 3] > 127
    dice = 2 * (wtissue & tmask).sum() / (wtissue.sum() + tmask.sum())
    return dict(rgba=rgba, analysis=r, warped=out, map_x=map_x, map_y=map_y, dice=float(dice))


# ----------------------------------------------------------------------------
# 4) 정합된 그림에서 파란 점 검출
# ----------------------------------------------------------------------------
def blue_index(rgba):
    """파란색 지수 b = B - max(R,G)  (흰/회색 조직 ~0, 파란 염색일수록 큼)"""
    rgb = rgba[..., :3].astype(np.float32)
    return rgb[..., 2] - np.maximum(rgb[..., 0], rgb[..., 1])


def valid_tissue(warped, tmask, erode=3):
    """정합된 그림에서 조직이 있는 곳 (가장자리 털/경계 제외)"""
    return ndi.binary_erosion((warped[..., 3] > 127) & tmask, iterations=erode)


def auto_threshold(warped, tmask, k_mad=4.0):
    """자동 임계값 = 조직 배경의 파란색 지수 중앙값 + k*MAD"""
    v = blue_index(warped)[valid_tissue(warped, tmask)]
    if not len(v):
        return float(BLUE_MIN)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return float(max(BLUE_MIN, med + k_mad * mad))


def detect_blue(warped, tmask, thr, min_area=4, bidx=None, valid=None):
    """임계값 이상인 파란 영역 (정합된 그림 = 템플릿 좌표)"""
    bidx = blue_index(warped) if bidx is None else bidx
    valid = valid_tissue(warped, tmask) if valid is None else valid
    blue = valid & (bidx > thr)
    blue = morphology.binary_opening(blue, morphology.disk(1))
    return morphology.remove_small_objects(blue, max(1, int(min_area)))


def grow_region(bidx, valid, x, y, frac=0.5, radius=25, snap=4):
    """클릭한 곳의 파란 점 영역을 찾는다 (수동 추가용).
    클릭 주변 snap px 안에서 가장 파란 픽셀(peak)을 씨앗으로 잡고,
    주변 배경(bg)과의 사이 값  bg + frac*(peak-bg) 이상이면서 씨앗과 연결된 픽셀을
    반경 radius 안에서 모은다. frac이 작을수록 넓게 잡힌다.
    반환: (영역 마스크 또는 None, 메시지)"""
    H, W = bidx.shape
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < W and 0 <= y < H):
        return None, T("템플릿 밖입니다.", "Outside the template.")
    y0, y1, x0, x1 = max(0, y - snap), min(H, y + snap + 1), max(0, x - snap), min(W, x + snap + 1)
    sub = np.where(valid[y0:y1, x0:x1], bidx[y0:y1, x0:x1], -np.inf)
    if not np.isfinite(sub.max()):
        return None, T("조직이 없는 곳입니다.", "No tissue here.")
    sy, sx = np.unravel_index(np.argmax(sub), sub.shape)
    sy, sx = sy + y0, sx + x0
    peak = bidx[sy, sx]
    R = 2 * radius
    Y0, Y1, X0, X1 = max(0, sy - R), min(H, sy + R + 1), max(0, sx - R), min(W, sx + R + 1)
    wv = valid[Y0:Y1, X0:X1]
    wb = bidx[Y0:Y1, X0:X1]
    bg = np.median(wb[wv])
    if peak - bg < 8:
        return None, T(f"클릭한 곳이 주변보다 충분히 파랗지 않습니다 (차이 {peak - bg:.0f}).",
                       f"The clicked area is not sufficiently bluer than its surroundings (difference {peak - bg:.0f}).")
    level = bg + frac * (peak - bg)
    yy, xx = np.ogrid[Y0:Y1, X0:X1]
    disk_ = (yy - sy) ** 2 + (xx - sx) ** 2 <= radius ** 2
    cand = wv & (wb >= level) & disk_
    lab = measure.label(cand)
    comp = lab == lab[sy - Y0, sx - X0]
    comp = ndi.binary_fill_holes(comp)
    out = np.zeros((H, W), bool)
    out[Y0:Y1, X0:X1] = comp
    return out, T(f"영역 추가: {int(comp.sum())}px (배경 {bg:.0f}, 최고 {peak:.0f}, 기준 {level:.0f})",
                  f"Region added: {int(comp.sum())} px (background {bg:.0f}, peak {peak:.0f}, level {level:.0f})")


def component_at(mask, x, y, snap=4):
    """mask 에서 (x, y) 또는 그 근처(snap px)의 연결 성분 (수동 삭제용)"""
    H, W = mask.shape
    x, y = int(round(x)), int(round(y))
    y0, y1, x0, x1 = max(0, y - snap), min(H, y + snap + 1), max(0, x - snap), min(W, x + snap + 1)
    yy, xx = np.nonzero(mask[y0:y1, x0:x1])
    if not len(yy):
        return None
    k = np.argmin((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2)
    lab = measure.label(mask)
    return lab == lab[yy[k] + y0, xx[k] + x0]


def extract_dots(blue, bidx, reg, tpl, manual=None):
    """파란 영역(연결 성분)별 중심 좌표: 템플릿 좌표 + 원본 사진 좌표 + 격자 셀
    manual: 수동으로 추가한 영역 마스크 (source 열에 auto / manual / auto+manual 표시)"""
    P_tpl = np.array(tpl["points"])
    lm = tpl["landmarks"]
    tip = P_tpl[lm["tip"]]
    rmid = (P_tpl[lm["root_a"]] + P_tpl[lm["root_b"]]) / 2
    Lt = rmid[0] - tip[0]
    dots = []
    for i, p in enumerate(measure.regionprops(measure.label(blue), intensity_image=bidx)):
        yt, xt = p.centroid
        r_, c_ = int(round(yt)), int(round(xt))
        src = "auto"
        if manual is not None:
            m = manual[p.coords[:, 0], p.coords[:, 1]].mean()
            src = "manual" if m > 0.99 else ("auto+manual" if m > 0 else "auto")
        dots.append(dict(dot_id=i + 1, source=src, area_tpl_px=int(p.area),
                         mean_blue_index=round(float(p.mean_intensity), 1),
                         x_tpl=round(xt, 1), y_tpl=round(yt, 1),
                         x_rel=round((xt - tip[0]) / Lt, 4), y_rel=round((yt - tip[1]) / Lt, 4),
                         grid_cell=grid_cell(xt, yt),
                         x_orig=round(float(reg["map_x"][r_, c_]), 1),
                         y_orig=round(float(reg["map_y"][r_, c_]), 1)))
    return dots


DOT_FIELDS = ["image", "threshold", "min_area", "dot_id", "source", "area_tpl_px", "mean_blue_index",
              "x_tpl", "y_tpl", "x_rel", "y_rel", "grid_cell", "x_orig", "y_orig"]


# ----------------------------------------------------------------------------
# 5) 그림 그리기 / 저장
# ----------------------------------------------------------------------------
def outline_xy(tpl):
    p = np.array(tpl["points"])
    return np.vstack([p, p[:1]]).T


def plot_registered(ax, tpl, tgrid, warped, title=None):
    ax.imshow(tgrid)
    ax.imshow(warped)
    ax.plot(*outline_xy(tpl), "r-", lw=1.2)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def plot_bluedots(ax, tpl, tgrid, blue, dots, under=None, title=None, added=None, removed=None):
    """파랑 = 검출 영역, 청록 = 수동 추가 영역, 연분홍 = 수동 삭제한 곳, 빨간 + = 점 중심"""
    ax.imshow(tgrid)
    if under is not None:
        u = under.copy()
        u[..., 3] = (u[..., 3] * 0.45).astype(np.uint8)
        ax.imshow(u)
    ov = np.zeros(blue.shape + (4,))
    if removed is not None:
        ov[removed] = [1, 0.4, 0.4, 0.35]
    ov[blue] = [0.1, 0.3, 1, 0.9]
    if added is not None:
        ov[blue & added] = [0, 0.8, 0.8, 0.95]
    ax.imshow(ov)
    ax.plot(*outline_xy(tpl), "k-", lw=1.2)
    if dots:
        d = np.array([[q["x_tpl"], q["y_tpl"]] for q in dots])
        ax.plot(d[:, 0], d[:, 1], "r+", ms=6, mew=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def plot_original(ax, rgba, analysis=None, tpl=None, title=None):
    ax.imshow(on_white(rgba))
    if analysis is not None:
        cut = np.cumsum([0] + SEG)
        cols = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:blue"]
        P = analysis["points"]
        for i in range(5):
            seg = P[cut[i]:cut[i + 1] + 1] if i < 4 else np.vstack([P[cut[4]:], P[:1]])
            ax.plot(seg[:, 0], seg[:, 1], "-", color=cols[i], lw=1.5)
        for k, i in tpl["landmarks"].items():
            ax.plot(*P[i], "o", ms=5, mec="k", color="yellow")
            ax.annotate(k, P[i], xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def save_bluedots_png(path, tpl, tgrid, blue, dots, title, added=None, removed=None):
    W, H = tpl["canvas"]
    fig = matplotlib.figure.Figure(figsize=(W / 100, H / 100 + 0.4))
    ax = fig.add_subplot(111)
    plot_bluedots(ax, tpl, tgrid, blue, dots, title=title, added=added, removed=removed)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def save_overview(path, tpl, names, warped, dices):
    W, H = tpl["canvas"]
    tgrid = np.asarray(draw_template(tpl, outline=False))
    n = len(names)
    cols = min(4, n)
    rws = int(np.ceil(n / cols))
    fig = matplotlib.figure.Figure(figsize=(4.5 * cols, 4.5 * rws * H / W + 0.4))
    for k, (nm, wimg, dc) in enumerate(zip(names, warped, dices)):
        plot_registered(fig.add_subplot(rws, cols, k + 1), tpl, tgrid, wimg, f"{fig_name(nm)}  Dice={dc:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def save_frequency(out_dir, tpl, names, masks, rows, sigma=6.0, common=0.5):
    """빈도 지도: 각 사진의 파란 영역을 약간 번지게(sigma) 한 뒤 평균 + 사진별 점 중심"""
    W, H = tpl["canvas"]
    tmask = template_mask(tpl)
    tgrid = np.asarray(draw_template(tpl, outline=False))
    n = len(masks)
    freq = np.mean([ndi.gaussian_filter(b.astype(float), sigma) > 0.15 if sigma > 0 else b
                    for b in masks], axis=0)
    freq[~tmask] = np.nan
    np.save(os.path.join(out_dir, "blue_frequency.npy"), freq)
    fig = matplotlib.figure.Figure(figsize=(15, 6.5 * H / W + 0.6))
    ax = fig.subplots(1, 2)
    ax[0].imshow(tgrid)
    im = ax[0].imshow(np.ma.masked_where(~(freq > 0), freq), cmap="Blues", vmin=0, vmax=1, alpha=0.9)
    ax[0].plot(*outline_xy(tpl), "k-", lw=1.5)
    if np.nanmax(freq) >= common:
        ax[0].contour(np.nan_to_num(freq) >= common, levels=[0.5], colors="red", linewidths=1.5)
    fig.colorbar(im, ax=ax[0], fraction=0.03, label="Fraction of images with a blue spot")
    ax[0].set_title(f"Blue-spot frequency map (n={n}; red line: common area >= {int(common * 100)}%)")
    ax[1].imshow(tgrid)
    ax[1].plot(*outline_xy(tpl), "k-", lw=1.5)
    cmap = matplotlib.colormaps["tab10"]
    for i, nm in enumerate(names):
        d = [(q["x_tpl"], q["y_tpl"]) for q in rows if q["image"] == nm]
        if d:
            d = np.array(d)
            ax[1].plot(d[:, 0], d[:, 1], "o", ms=3, color=cmap(i % 10), label=fig_name(nm))
    if rows:
        ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_title("Blue-spot centroids per image")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "_blue_frequency_map.png"), dpi=90)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def registered_dir(folder):
    """부위 폴더와 같은 위치에 '<부위 폴더명>_registered'"""
    return os.path.normpath(os.path.abspath(folder)) + "_registered"


def register_folder(folder, view=None, flip_names=(), sigma=6.0, common=0.5, k_mad=4.0, min_area=4):
    """명령줄용: 폴더 전체를 정합하고 자동 임계값으로 파란 점까지 저장"""
    view = view or detect_view(folder)
    tpl = load_template(view)
    tmask = template_mask(tpl)
    tgrid = np.asarray(draw_template(tpl, outline=False))
    out = registered_dir(folder)
    os.makedirs(out, exist_ok=True)
    names = list_images(folder)
    rows, summ, warped, masks = [], [], [], []
    for nm in names:
        reg = register_image(os.path.join(folder, nm), tpl, tmask, flip_toggle=nm in flip_names)
        stem = os.path.splitext(nm)[0]
        Image.fromarray(reg["warped"]).save(os.path.join(out, f"{stem}_registered.png"))
        thr = auto_threshold(reg["warped"], tmask, k_mad)
        bidx = blue_index(reg["warped"])
        blue = detect_blue(reg["warped"], tmask, thr, min_area, bidx)
        dots = extract_dots(blue, bidx, reg, tpl)
        Image.fromarray((blue * 255).astype(np.uint8)).save(os.path.join(out, f"{stem}_bluemask.png"))
        save_bluedots_png(os.path.join(out, f"{stem}_bluedots.png"), tpl, tgrid, blue, dots,
                          f"{fig_name(nm)}  (threshold {thr:.0f}, {len(dots)} spots)")
        rows += [dict(image=nm, threshold=round(thr, 1), min_area=min_area, **d) for d in dots]
        a = reg["analysis"]
        summ.append(dict(image=nm, flip=a["flip"], landmarks_ok=a["landmarks_ok"],
                         truncated=round(a["truncated"], 3), dice=round(reg["dice"], 3),
                         threshold=round(thr, 1), min_area=min_area, n_dots=len(dots),
                         blue_area_tpl_px=int(blue.sum())))
        warped.append(reg["warped"])
        masks.append(blue)
        print(f"  {nm}: Dice={reg['dice']:.3f}  임계값={thr:.0f}  점 {len(dots)}개")
    write_csv(os.path.join(out, "blue_dots.csv"), rows, DOT_FIELDS)
    write_csv(os.path.join(out, "summary.csv"), summ, list(summ[0].keys()))
    save_overview(os.path.join(out, "_overview_registered.png"), tpl, names, warped, [s_["dice"] for s_ in summ])
    save_frequency(out, tpl, names, masks, rows, sigma, common)
    print(f"[register] {folder} ({view}) -> {out}")
    return out


# ----------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("template", help="부위별 공통 템플릿 생성")
    t.add_argument("--view", choices=list(VIEWS), nargs="*", default=list(VIEWS))
    t.add_argument("--max-trunc", type=float, default=0.08,
                   help="사진 가장자리에서 잘린 윤곽 비율이 이 값보다 크면 템플릿 제작에서 제외")
    t.add_argument("--dtw", action="store_true", help="모양 기반 탄력 대응(DTW) 템플릿으로 만들기")
    r = sp.add_parser("register", help="폴더 사진들을 템플릿에 정합하고 파란 점 좌표 추출")
    r.add_argument("--folder", required=True)
    r.add_argument("--view", help="부위 (templates 폴더의 template_<부위>.json, 생략하면 폴더 이름으로 판단)")
    r.add_argument("--flip", nargs="*", default=[], help="좌우 반전(미러링)해서 정합할 파일 이름")
    r.add_argument("--sigma", type=float, default=6.0, help="빈도 지도용 번짐 정도(px, 0=없음)")
    r.add_argument("--common", type=float, default=0.5, help="공통 영역 기준 비율")
    r.add_argument("--k-mad", type=float, default=4.0,
                   help="파란 점 임계값 = 조직 중앙값 + k*MAD (작을수록 옅은 점까지 검출)")
    a = ap.parse_args()
    if a.cmd == "template":
        for v in a.view:
            build_template(v, a.max_trunc, "dtw" if a.dtw else "arc")
    else:
        register_folder(a.folder, a.view, set(a.flip), a.sigma, a.common, a.k_mad)


if __name__ == "__main__":
    main()
