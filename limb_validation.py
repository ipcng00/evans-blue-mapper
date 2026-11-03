"""
검증·비교 실험용 공통 함수 (논문용)
================================================================
- 비교용 정합 방법 5가지 (같은 윤곽 대응점 사용)
    axis       : 발끝-기시부 축만 맞춤 (회전+크기+이동, 사람이 손으로 맞추는 방식에 해당)
    similarity : 윤곽 전체 Procrustes 유사변환
    affine     : 윤곽 전체 아핀변환
    tps_arc    : 제안 방법 - TPS + 길이 비율 대응
    tps_dtw    : 제안 방법 - TPS + 모양 기반(DTW) 대응
- 원본 좌표 -> 템플릿 좌표 순방향 변환 (주석 점을 템플릿으로 옮길 때)
- 지표: Dice, 경계 거리, 고주파 질감 NCC, ICC(2,1), 점 매칭
"""
import warnings

warnings.filterwarnings("ignore", message=r".*cryptography\.hazmat\.decrepit.*")
import os
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from scipy.spatial import cKDTree
from skimage import measure

import limb_mapping as LM

METHODS = ["axis", "similarity", "affine", "tps_arc", "tps_dtw"]
METHOD_LABEL = {"axis": "Axis alignment", "similarity": "Similarity", "affine": "Affine",
                "tps_arc": "TPS (arc-length)", "tps_dtw": "TPS (DTW)"}


# ----------------------------------------------------------------------------
# 정합
# ----------------------------------------------------------------------------
def _fit_similarity(src, dst):
    """src -> dst 유사변환 (반전 없음). 반환: 2x3 행렬"""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(A.T @ B)
    D = np.eye(2)
    D[1, 1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    sc = (S * np.diag(D)).sum() / (A ** 2).sum()
    M = sc * R.T
    return np.c_[M, md - M @ ms]


def _fit_affine(src, dst):
    X = np.c_[src, np.ones(len(src))]
    sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
    return sol.T


def _invert(M):
    Ai = np.linalg.inv(M[:, :2])
    return np.c_[Ai, -Ai @ M[:, 2]]


def _apply(M, p):
    p = np.atleast_2d(p)
    return p @ M[:, :2].T + M[:, 2]


def correspondences(rgba, tpl, method, flip_toggle=False, analysis=None):
    """사진 윤곽점과 템플릿 윤곽점 (같은 순서)"""
    r = analysis if analysis is not None else LM.analyze(rgba, flip_toggle)
    r = dict(r)
    if method == "tps_dtw":
        r["points"] = LM.dtw_points(r, LM.template_features(tpl["points"]))
    return r, np.asarray(r["points"], float), np.asarray(tpl["points"], float)


def register(rgba, tpl, tmask, method="tps_arc", flip_toggle=False, analysis=None, full_canvas=False):
    """method 로 정합. 반환 dict: warped, map_x, map_y, inv(템플릿->원본 함수), fwd(원본->템플릿 함수), dice
    full_canvas=True 면 템플릿 밖까지 map 계산 (순방향 변환이 경계 근처에서도 되도록)"""
    r, P_img, P_tpl = correspondences(rgba, tpl, method, flip_toggle, analysis)
    if method == "axis":
        lm = np.cumsum([0] + LM.SEG[:-1])
        a = np.array([P_tpl[lm[0]], (P_tpl[lm[2]] + P_tpl[lm[3]]) / 2])
        b = np.array([P_img[lm[0]], (P_img[lm[2]] + P_img[lm[3]]) / 2])
        M = _fit_similarity(a, b)
        Mf = _invert(M)
        inv = lambda q: _apply(M, q)
        fwd = lambda q: _apply(Mf, q)
    elif method in ("similarity", "affine"):
        fit = _fit_similarity if method == "similarity" else _fit_affine
        M = fit(P_tpl, P_img)
        Mf = _invert(M)
        inv = lambda q: _apply(M, q)
        fwd = lambda q: _apply(Mf, q)
    else:
        sub = slice(None, None, 2)
        rbf = RBFInterpolator(P_tpl[sub], P_img[sub], kernel="thin_plate_spline", degree=1, smoothing=1.0)
        inv = lambda q: np.vstack([rbf(np.atleast_2d(q)[i:i + 60000])
                                   for i in range(0, len(np.atleast_2d(q)), 60000)])
        fwd = None
    H, W = tmask.shape
    yy, xx = np.nonzero(np.ones_like(tmask) if full_canvas else tmask)
    src = inv(np.c_[xx, yy].astype(float))
    out = np.zeros((H, W, 4), np.uint8)
    inside = tmask[yy, xx]
    for ch in range(4):
        out[yy[inside], xx[inside], ch] = ndi.map_coordinates(
            rgba[..., ch].astype(float), [src[inside, 1], src[inside, 0]], order=1, cval=0)
    map_x = np.full((H, W), np.nan, np.float32)
    map_y = np.full((H, W), np.nan, np.float32)
    map_x[yy, xx], map_y[yy, xx] = src[:, 0], src[:, 1]
    if fwd is None:
        fwd = _refined(_grid_inverse(xx, yy, src), inv)
    wt = out[..., 3] > 127
    dice = 2 * (wt & tmask).sum() / (wt.sum() + tmask.sum())
    return dict(rgba=rgba, analysis=r, warped=out, map_x=map_x, map_y=map_y, inv=inv, fwd=fwd,
                dice=float(dice), method=method, P_img=P_img, P_tpl=P_tpl)


def register_points(rgba, tpl, tmask, method="tps_arc", flip_toggle=False, analysis=None, step=4):
    """그림은 만들지 않고 좌표 변환(inv, fwd)만 빠르게 (모의 실험용)"""
    r, P_img, P_tpl = correspondences(rgba, tpl, method, flip_toggle, analysis)
    if method in ("axis", "similarity", "affine"):
        return register(rgba, tpl, tmask, method, flip_toggle, r)
    sub = slice(None, None, 2)
    rbf = RBFInterpolator(P_tpl[sub], P_img[sub], kernel="thin_plate_spline", degree=1, smoothing=1.0)
    inv = lambda q: rbf(np.atleast_2d(q))
    H, W = tmask.shape
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    xx, yy = xx.ravel(), yy.ravel()
    src = inv(np.c_[xx, yy].astype(float))
    fwd = _refined(_grid_inverse(xx, yy, src, step=1), inv)
    return dict(analysis=r, inv=inv, fwd=fwd, method=method, P_img=P_img, P_tpl=P_tpl)


def _grid_inverse(xx, yy, src, step=2):
    """템플릿 격자 (xx,yy) -> 원본 src 대응표로부터 원본->템플릿 선형 보간 함수"""
    k = (xx % step == 0) & (yy % step == 0)
    interp = LinearNDInterpolator(src[k], np.c_[xx[k], yy[k]].astype(float))
    tree = cKDTree(src[k])
    tg = np.c_[xx[k], yy[k]].astype(float)

    def fwd(q):
        q = np.atleast_2d(np.asarray(q, float))
        out = interp(q)
        bad = ~np.isfinite(out).all(1)
        if bad.any():                                  # 볼록 껍질 밖: 가장 가까운 격자점
            out[bad] = tg[tree.query(q[bad])[1]]
        return out
    return fwd


def _refined(fwd0, inv, iters=3, h=0.5):
    """격자 보간 초기값을 뉴턴 반복으로 다듬어 inv(fwd(q)) = q 가 되게 한다"""
    def fwd(q):
        q = np.atleast_2d(np.asarray(q, float))
        t = fwd0(q)
        for _ in range(iters):
            f0 = inv(t)
            fx = inv(t + [h, 0])
            fy = inv(t + [0, h])
            J = np.stack([(fx - f0) / h, (fy - f0) / h], axis=2)       # (n, 2, 2): d img / d tpl
            e = q - f0
            ok = np.abs(np.linalg.det(J)) > 1e-6
            t[ok] += np.linalg.solve(J[ok], e[ok][..., None])[..., 0]
        return t
    return fwd


def forward_points(reg, pts):
    """원본 좌표 점들 -> 템플릿 좌표"""
    return reg["fwd"](np.asarray(pts, float)) if len(pts) else np.zeros((0, 2))


# ----------------------------------------------------------------------------
# 지표
# ----------------------------------------------------------------------------
def boundary_distance(warped, tmask):
    """정합된 조직 경계와 템플릿 경계 사이 평균 대칭 거리(px)와 95% Hausdorff"""
    wt = ndi.binary_fill_holes(warped[..., 3] > 127)
    if not wt.any():
        return np.nan, np.nan
    bw = wt ^ ndi.binary_erosion(wt)
    bt = tmask ^ ndi.binary_erosion(tmask)
    dt_t = ndi.distance_transform_edt(~bt)
    dt_w = ndi.distance_transform_edt(~bw)
    d = np.r_[dt_t[bw], dt_w[bt]]
    return float(d.mean()), float(np.percentile(d, 95))


def highpass_gray(warped, s1=2, s2=12):
    """회색조 대역통과(DoG) 질감 - 피부 주름·패드·털 결 같은 내부 구조"""
    g = warped[..., :3].astype(float).mean(2)
    return ndi.gaussian_filter(g, s1) - ndi.gaussian_filter(g, s2)


def texture_valid(warped, tmask, erode=4):
    return ndi.binary_erosion((warped[..., 3] > 127) & tmask, iterations=erode)


def ncc(a, b, m):
    if m.sum() < 500:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum() + 1e-12))


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan


def icc21(Y):
    """ICC(2,1) 이원 무작위, 절대 일치, 단일 측정. Y: (대상 n, 평가자 k)"""
    Y = np.asarray(Y, float)
    Y = Y[np.isfinite(Y).all(1)]
    n, k = Y.shape
    if n < 2 or k < 2:
        return np.nan
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def match_points_to_mask(points, mask, tol=4):
    """점(템플릿 좌표) 목록과 검출 마스크 비교.
    점이 성분 안 또는 tol px 안이면 TP. 반환: tp, fn, fp(점이 하나도 안 닿은 성분 수), 성분 수"""
    lab = measure.label(mask)
    n = int(lab.max())
    if n == 0:
        return 0, len(points), 0, 0
    dist, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
    hit = set()
    tp = 0
    H, W = mask.shape
    for x, y in points:
        r, c = int(round(y)), int(round(x))
        if not (0 <= r < H and 0 <= c < W):
            continue
        if dist[r, c] <= tol:
            tp += 1
            hit.add(int(lab[idx[0][r, c], idx[1][r, c]]))
    return tp, len(points) - tp, n - len(hit), n


def match_points(a, b, tol):
    """두 점 집합 1:1 매칭 (탐욕적, 가까운 순). 반환: 매칭 쌍 거리 목록, 매칭 안 된 a 수, b 수"""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if len(a) == 0 or len(b) == 0 or a.size == 0 or b.size == 0:
        return [], len(a) if a.size else 0, len(b) if b.size else 0
    D = np.linalg.norm(a[:, None] - b[None], axis=2)
    pairs, ua, ub = [], set(range(len(a))), set(range(len(b)))
    for i, j in zip(*np.unravel_index(np.argsort(D, axis=None), D.shape)):
        if D[i, j] > tol:
            break
        if i in ua and j in ub:
            pairs.append(D[i, j])
            ua.discard(i)
            ub.discard(j)
    return pairs, len(ua), len(ub)


# ----------------------------------------------------------------------------
# 데이터 목록
# ----------------------------------------------------------------------------
def dataset(views=None):
    """[(view, 분절(C5..), 폴더, 파일 경로)] - _registered 폴더 제외"""
    out = []
    for v in (views or list(LM.VIEWS)):
        for p in LM.collect_view_images(v):
            seg = os.path.relpath(p, LM.DATA_DIR).split(os.sep)[0]
            out.append((v, seg, os.path.basename(os.path.dirname(p)), p))
    return out
