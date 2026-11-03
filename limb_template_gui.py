"""
공통 템플릿 제작 GUI
언어: 메뉴 [언어 / Language] 또는  python limb_template_gui.py --lang en  (그림 안의 글자는 항상 영어)

실행:  python limb_template_gui.py

1) [사진 폴더 선택] -> 폴더(및 다른 척수 폴더의 같은 부위 폴더) 사진들의 윤곽선을 따서 겹쳐 그리고
   자동 공통 템플릿(평균 윤곽선)을 계산한다.
2) 템플릿 윤곽선 = 조절점(엣지)들을 지나는 매끄러운 닫힌 곡선.
     - 조절점 끌기         : 왼쪽 버튼으로 끌기 ('끌기 영향 반경' > 0 이면 주변 조절점도 함께 부드럽게 이동)
     - 조절점 추가         : 윤곽선 위를 더블클릭
     - 조절점 삭제         : 조절점을 오른쪽 클릭 (빨간 랜드마크 5개는 정합에 필요해서 삭제 불가, 이동만 가능)
     - 되돌리기            : Ctrl+Z
     - 매끄럽게 / 조절점 균등 재배치 / 자동 템플릿으로 되돌리기 버튼
3) [템플릿 저장] -> mapping data/templates 에 template_<부위ID>.json / .png / _check.png 저장
   (정합 GUI의 부위 목록에 자동으로 나타남)
대응 방식: 기본은 랜드마크 사이 길이 비율. '모양 기반 탄력 대응(DTW)'을 켜면 굴곡이 맞도록 대응시켜
   발가락처럼 복잡한 모양에서 정합이 좋아진다 (손바닥면 시험: 내부 구조 일치도 +13%). 설정은 템플릿에 저장되고
   정합 GUI는 템플릿 설정을 자동으로 따른다.
"""
import os
import queue
import re
import threading
import tkinter as tk
import warnings
from tkinter import filedialog, messagebox, ttk

warnings.filterwarnings("ignore", message=r".*cryptography\.hazmat\.decrepit.*")

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import CubicSpline

import limb_mapping as LM
from limb_i18n import T, add_language_menu

NODE_SEG = [4, 6, 16, 6, 4]     # 자동 템플릿에서 구간별 조절점 개수 (tip→wrist_a→root_a→root_b→wrist_b→tip)
N_PER = 24                      # 조절점 사이 곡선 샘플 수
PICK_PX = 10                    # 조절점/윤곽선 선택 허용 거리(화면 px)


# ----------------------------------------------------------------------------
# 조절점 <-> 윤곽선
# ----------------------------------------------------------------------------
def spline_dense(nodes):
    """조절점을 지나는 닫힌 cubic spline. 반환: 촘촘한 곡선 (조절점 k 는 dense[k*N_PER])"""
    P = np.vstack([nodes, nodes[:1]])
    d = np.maximum(np.linalg.norm(np.diff(P, axis=0), axis=1), 1e-6)
    t = np.r_[0, np.cumsum(d)]
    cs = CubicSpline(t, P, bc_type="periodic")
    tt = np.concatenate([np.linspace(t[k], t[k + 1], N_PER, endpoint=False) for k in range(len(nodes))])
    return cs(tt)


def nodes_to_points(nodes, lm_nodes):
    """조절점 곡선 -> 템플릿 윤곽점 (랜드마크 사이 구간을 LM.SEG 개수로 재샘플링)"""
    dense = spline_dense(nodes)
    ring = np.vstack([dense, dense[:1]])
    cuts = [k * N_PER for k in lm_nodes] + [len(nodes) * N_PER]
    return np.vstack([LM.resample_open(ring[cuts[i]:cuts[i + 1] + 1], LM.SEG[i]) for i in range(5)])


def points_to_nodes(pts):
    """템플릿 윤곽점(sum(SEG)개) -> 조절점 + 랜드마크 조절점 번호"""
    cum = np.cumsum([0] + LM.SEG)
    idx, lm = [], []
    for i, n in enumerate(NODE_SEG):
        lm.append(len(idx))
        idx += [cum[i] + int(round(j * LM.SEG[i] / n)) for j in range(n)]
    return np.asarray(pts, float)[idx].copy(), lm


def arc_positions(nodes):
    P = np.vstack([nodes, nodes[:1]])
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.r_[0, np.cumsum(d)[:-1]], d.sum()


class TemplateApp:
    def __init__(self, root):
        self.root = root
        root.title(T("공통 템플릿 제작", "Common template construction"))
        add_language_menu(root)
        root.geometry("1500x930")
        self.folder = None
        self.paths = []
        self.analyses = {}          # 사진 경로 -> analyze() 결과
        self.used = []              # 템플릿 계산에 사용한 사진 경로
        self.auto_pts = None        # 자동(평균) 템플릿 윤곽점
        self.corr_used = "arc"      # 자동 템플릿을 계산한 대응 방식
        self.overlays = []          # 개별 사진 윤곽선 (현재 템플릿 좌표에 맞춘 것)
        self.mean_disp = None
        self.nodes = None
        self.lm_nodes = None
        self.hist = []
        self.drag = None
        self.q = queue.Queue()
        self.busy = False

        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self._build_controls()
        self._build_canvas()
        root.bind_all("<Control-z>", lambda e: self.undo())
        root.after(100, self._poll)

    # ------------------------------------------------------------------ 화면 구성
    def _build_controls(self):
        f = ttk.Frame(self.root, width=430)
        f.grid(row=0, column=0, sticky="ns", padx=6, pady=6)
        f.columnconfigure(1, weight=1)

        s1 = ttk.LabelFrame(f, text=T("1. 참고 사진", "1. Reference photographs"))
        s1.pack(fill="x", pady=3)
        s1.columnconfigure(1, weight=1)
        ttk.Button(s1, text=T("사진 폴더 선택…", "Select photo folder…"), command=self.choose_folder).grid(row=0, column=0, padx=4, pady=3, sticky="w")
        self.lbl_folder = ttk.Label(s1, text=T("(부위 사진 폴더를 선택하세요)", "(select a view folder)"), foreground="gray", wraplength=300)
        self.lbl_folder.grid(row=0, column=1, sticky="w")
        self.var_sib = tk.BooleanVar(value=True)
        ttk.Checkbutton(s1, text=T("다른 상위 폴더(C5~C8 등)의 같은 키워드 폴더도 포함",
                                         "Include folders with the same keyword in sibling groups (C5–C8, etc.)"),
                        variable=self.var_sib).grid(row=1, column=0, columnspan=2, sticky="w", padx=4)
        kf = ttk.Frame(s1)
        kf.grid(row=2, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Label(kf, text=T("폴더 키워드", "Folder keyword")).pack(side="left")
        self.var_kw = tk.StringVar()
        ttk.Entry(kf, textvariable=self.var_kw, width=16).pack(side="left", padx=4)
        ttk.Button(kf, text=T("사진 다시 찾기", "Find photographs"), command=self.find_images).pack(side="left")
        self.pbar = ttk.Progressbar(s1, mode="determinate")
        self.pbar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=3)
        lf = ttk.Frame(s1)
        lf.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=4)
        self.lst = tk.Listbox(lf, selectmode="extended", height=11, width=52, exportselection=False,
                              font=("Malgun Gothic", 9))
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.lst.yview)
        self.lst.config(yscrollcommand=sb.set)
        self.lst.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        ttk.Label(s1, text=T("선택(파랑)한 사진만 템플릿 계산에 사용 (Ctrl/Shift+클릭)",
                                  "Only selected (blue) photographs are used (Ctrl/Shift+click)"), foreground="#444").grid(
            row=5, column=0, columnspan=2, sticky="w", padx=4)
        tf = ttk.Frame(s1)
        tf.grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=3)
        ttk.Label(tf, text=T("잘림 비율 상한(%)", "Max truncation (%)")).pack(side="left")
        self.var_trunc = tk.DoubleVar(value=8)
        ttk.Spinbox(tf, from_=0, to=100, increment=1, textvariable=self.var_trunc, width=5,
                    command=self.auto_select).pack(side="left", padx=3)
        ttk.Button(tf, text=T("자동 선택", "Auto select"), command=self.auto_select).pack(side="left", padx=3)
        self.var_dtw = tk.BooleanVar(value=False)
        ttk.Checkbutton(s1, text=T("모양 기반 탄력 대응(DTW) - 발가락 등 복잡한 모양에 권장",
                                         "Shape-based elastic correspondence (DTW) - for complex outlines such as digits"),
                        variable=self.var_dtw).grid(row=7, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Button(s1, text=T("선택한 사진으로 자동 템플릿 계산", "Compute template from selected photographs"), command=self.compute_auto).grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=4, pady=3)

        s2 = ttk.LabelFrame(f, text=T("2. 윤곽선 편집", "2. Outline editing"))
        s2.pack(fill="x", pady=3)
        ttk.Label(s2, justify="left", foreground="#333", text=T(
            "• 조절점(엣지) 끌기: 왼쪽 버튼 드래그\n"
            "• 조절점 추가: 윤곽선 위 더블클릭\n"
            "• 조절점 삭제: 조절점 오른쪽 클릭 (빨간 랜드마크는 이동만)\n"
            "• 되돌리기: Ctrl+Z    • 확대/이동: 그림 아래 도구막대",
            "• Move a control point: left-button drag\n"
            "• Add a control point: double-click on the outline\n"
            "• Delete a control point: right click (red landmarks can only be moved)\n"
            "• Undo: Ctrl+Z    • Zoom/pan: toolbar below the plot")).pack(anchor="w", padx=4)
        rf = ttk.Frame(s2)
        rf.pack(fill="x", padx=4, pady=2)
        ttk.Label(rf, text=T("끌기 영향 반경(px)", "Drag influence radius (px)")).pack(side="left")
        self.var_rad = tk.DoubleVar(value=0)
        ttk.Scale(rf, from_=0, to=400, variable=self.var_rad,
                  command=lambda e: self.lbl_rad.config(text=f"{self.var_rad.get():.0f}")).pack(
            side="left", fill="x", expand=True, padx=4)
        self.lbl_rad = ttk.Label(rf, text="0", width=4)
        self.lbl_rad.pack(side="left")
        ttk.Label(s2, text=T("  (0 = 조절점 하나만, 크게 = 주변 윤곽까지 부드럽게 함께 이동)",
                              "  (0 = single point; larger = neighboring outline moves smoothly too)"),
                  foreground="gray").pack(anchor="w", padx=4)
        bf = ttk.Frame(s2)
        bf.pack(fill="x", padx=4, pady=3)
        ttk.Button(bf, text=T("매끄럽게", "Smooth"), command=self.smooth).pack(side="left", padx=2)
        ttk.Button(bf, text=T("조절점 균등 재배치", "Redistribute points"), command=self.redistribute).pack(side="left", padx=2)
        ttk.Button(bf, text=T("자동 템플릿으로", "Reset to automatic"), command=self.reset_auto).pack(side="left", padx=2)
        ttk.Button(bf, text=T("되돌리기", "Undo"), command=self.undo).pack(side="left", padx=2)
        cf = ttk.Frame(s2)
        cf.pack(fill="x", padx=4)
        self.var_ov = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, text=T("개별 사진 윤곽선", "Individual outlines"), variable=self.var_ov, command=self.redraw_all).pack(side="left")
        self.var_mean = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, text=T("자동 템플릿(점선)", "Automatic template (dashed)"), variable=self.var_mean, command=self.redraw_all).pack(side="left", padx=8)
        self.var_nodes = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, text=T("조절점", "Control points"), variable=self.var_nodes, command=self.redraw_all).pack(side="left")

        s3 = ttk.LabelFrame(f, text=T("3. 저장", "3. Save"))
        s3.pack(fill="x", pady=3)
        s3.columnconfigure(1, weight=1)
        ttk.Label(s3, text=T("부위 ID (파일 이름)", "View ID (file name)")).grid(row=0, column=0, sticky="w", padx=4)
        self.var_id = tk.StringVar()
        ttk.Entry(s3, textvariable=self.var_id).grid(row=0, column=1, sticky="ew", padx=4, pady=1)
        ttk.Label(s3, text=T("표시 이름", "Display name")).grid(row=1, column=0, sticky="w", padx=4)
        self.var_label = tk.StringVar()
        ttk.Entry(s3, textvariable=self.var_label).grid(row=1, column=1, sticky="ew", padx=4, pady=1)
        ttk.Label(s3, text=T("폴더 키워드 (쉼표 구분)", "Folder keywords (comma-separated)")).grid(row=2, column=0, sticky="w", padx=4)
        ttk.Entry(s3, textvariable=self.var_kw).grid(row=2, column=1, sticky="ew", padx=4, pady=1)
        ttk.Label(s3, text=T("  정합 GUI에서 폴더 이름에 이 키워드가 있으면 이 템플릿을 자동 선택",
                              "  The registration GUI selects this template when a folder name contains a keyword"),
                  foreground="gray").grid(row=3, column=0, columnspan=2, sticky="w", padx=4)
        bf2 = ttk.Frame(s3)
        bf2.grid(row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        ttk.Button(bf2, text=T("템플릿 저장", "Save template"), command=self.save).pack(side="left", padx=2)
        ttk.Button(bf2, text=T("기존 템플릿 불러오기…", "Load existing template…"), command=self.load_existing).pack(side="left", padx=6)

        self.lbl_status = ttk.Label(f, text="", foreground="#1a5fb4", wraplength=410, justify="left")
        self.lbl_status.pack(fill="x", pady=4)

    def _build_canvas(self):
        frm = ttk.LabelFrame(self.root, text=T("개별 윤곽선(색) + 공통 템플릿(검정)   파랑 = 조절점, 빨강 = 랜드마크",
                                              "Individual outlines (color) + common template (black)   blue = control points, red = landmarks"))
        frm.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.fig = Figure(figsize=(9, 7), dpi=90)
        self.ax = self.fig.add_axes([0.04, 0.04, 0.94, 0.92])
        self.cv = FigureCanvasTkAgg(self.fig, master=frm)
        self.toolbar = NavigationToolbar2Tk(self.cv, frm, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.cv.get_tk_widget().pack(fill="both", expand=True)
        self.cv.mpl_connect("button_press_event", self.on_press)
        self.cv.mpl_connect("motion_notify_event", self.on_motion)
        self.cv.mpl_connect("button_release_event", self.on_release)
        self._style_axes()

    def _style_axes(self):
        ax = self.ax
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MultipleLocator(LM.GRID))
        ax.yaxis.set_major_locator(MultipleLocator(LM.GRID))
        ax.grid(True, color="#d8d8d8", lw=0.7)
        ax.tick_params(labelsize=7, colors="#888")
        ax.set_axisbelow(True)

    def status(self, msg):
        self.lbl_status.config(text=msg)

    # ------------------------------------------------------------------ 사진 / 분석
    def choose_folder(self):
        init = LM.DATA_DIR if os.path.isdir(LM.DATA_DIR) else os.getcwd()
        d = filedialog.askdirectory(title=T("템플릿 제작에 참고할 부위 사진 폴더", "Folder of reference photographs for the template"), initialdir=init)
        if not d:
            return
        self.folder = os.path.normpath(d)
        name = os.path.basename(self.folder)
        try:                                   # 이미 아는 부위면 그 정보로 채움
            v = LM.detect_view(self.folder)
            kws = next((k for vv, _, k in LM.available_views() if vv == v), None) or [LM.VIEWS.get(v, v)]
            self.var_id.set(v)
            self.var_label.set(LM.view_label(v))
            self.var_kw.set(", ".join(kws))
        except ValueError:
            kw = re.split(r"[\s(_]", name)[0] or name
            self.var_id.set(kw)
            self.var_label.set(name)
            self.var_kw.set(kw)
        self.lbl_folder.config(text=self.folder, foreground="black")
        self.find_images()

    def keywords(self):
        return [k.strip() for k in self.var_kw.get().split(",") if k.strip()]

    def find_images(self):
        if not self.folder or self.busy:
            return
        folders = [self.folder]
        kws = self.keywords()
        if self.var_sib.get() and kws:
            top = os.path.dirname(os.path.dirname(self.folder))
            for c in sorted(os.listdir(top)):
                cdir = os.path.join(top, c)
                if not os.path.isdir(cdir):
                    continue
                for d in sorted(os.listdir(cdir)):
                    p = os.path.normpath(os.path.join(cdir, d))
                    if (os.path.isdir(p) and not d.endswith("_registered") and p not in folders
                            and any(k in d for k in kws)):
                        folders.append(p)
        self.paths = [os.path.join(d, f) for d in folders for f in LM.list_images(d)]
        if not self.paths:
            messagebox.showwarning(T("사진 없음", "No photographs"), T("그림 파일을 찾지 못했습니다.", "No image files found."))
            return
        todo = [p for p in self.paths if p not in self.analyses]
        self.pbar.config(maximum=max(1, len(todo)), value=0)
        self.busy = True
        self.status(T(f"사진 {len(self.paths)}장 ({len(folders)}개 폴더) 윤곽선 분석 중…",
                      f"Analyzing outlines of {len(self.paths)} photographs ({len(folders)} folders)…"))
        threading.Thread(target=self._analyze_worker, args=(todo,), daemon=True).start()

    def _analyze_worker(self, todo):
        for i, p in enumerate(todo):
            try:
                self.q.put(("ana", p, LM.analyze(LM.load_rgba(p))))
            except Exception as e:
                self.q.put(("err", p, e))
            self.q.put(("prog", i + 1))
        self.q.put(("done",))

    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "ana":
                    self.analyses[m[1]] = m[2]
                elif m[0] == "err":
                    self.status(T(f"분석 실패: {os.path.basename(m[1])}: {m[2]}", f"Analysis failed: {os.path.basename(m[1])}: {m[2]}"))
                elif m[0] == "prog":
                    self.pbar.config(value=m[1])
                elif m[0] == "done":
                    self.busy = False
                    self.paths = [p for p in self.paths if p in self.analyses]
                    self.fill_list()
                    self.auto_select()
                    self.compute_auto(confirm=False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def rel(self, p):
        try:
            r = os.path.relpath(p, LM.DATA_DIR)
            return p if r.startswith("..") else r
        except ValueError:
            return p

    def fill_list(self):
        self.lst.delete(0, "end")
        for p in self.paths:
            r = self.analyses[p]
            self.lst.insert("end", f"{self.rel(p)}   {T('잘림', 'truncated')} {r['truncated'] * 100:.0f}%"
                                   f"{T('   [랜드마크 실패]', '   [landmarks failed]') if not r['landmarks_ok'] else ''}")

    def auto_select(self):
        if not self.paths:
            return
        try:
            mt = float(self.var_trunc.get()) / 100
        except (tk.TclError, ValueError):
            mt = 0.08
        ok = [i for i, p in enumerate(self.paths)
              if self.analyses[p]["landmarks_ok"] and self.analyses[p]["truncated"] <= mt]
        if len(ok) < 3:
            ok = [i for i, p in enumerate(self.paths) if self.analyses[p]["landmarks_ok"]] or list(range(len(self.paths)))
        self.lst.selection_clear(0, "end")
        for i in ok:
            self.lst.selection_set(i)

    # ------------------------------------------------------------------ 템플릿 계산
    def compute_auto(self, confirm=True):
        sel = [self.paths[i] for i in self.lst.curselection()]
        if len(sel) < 2:
            messagebox.showinfo(T("사진 선택", "Select photographs"), T("템플릿 계산에 쓸 사진을 2장 이상 선택하세요.", "Select at least two photographs."))
            return
        if confirm and self.hist and not messagebox.askyesno(T("다시 계산", "Recompute"), T("편집한 윤곽선이 자동 템플릿으로 바뀝니다. 계속할까요?",
                                                                                   "The edited outline will be replaced by the automatic template. Continue?")):
            return
        self.used = sel
        self.corr_used = self.corr()
        self.status(T("자동 템플릿 계산 중…", "Computing automatic template…"))
        self.root.update_idletasks()
        self.auto_pts = LM.template_shape([self.analyses[p] for p in sel], self.corr_used)
        self.nodes, self.lm_nodes = points_to_nodes(self.auto_pts)
        self.hist = []
        self.align_overlays()
        self.redraw_all(reset_view=True)
        self.status(T(f"자동 템플릿 계산 완료 ({len(sel)}장, 대응={self.corr_label(self.corr_used)}). "
                      f"윤곽선을 편집한 뒤 [템플릿 저장]을 누르세요.",
                      f"Automatic template computed ({len(sel)} photographs, correspondence={self.corr_label(self.corr_used)}). "
                      f"Edit the outline and press [Save template]."))

    def corr(self):
        return "dtw" if self.var_dtw.get() else "arc"

    @staticmethod
    def corr_label(c):
        return T("모양 기반(DTW)", "shape-based (DTW)") if c == "dtw" else T("길이 비율", "arc-length")

    def align_overlays(self):
        """개별 사진 윤곽선과 자동 템플릿을 현재 템플릿 좌표에 맞춤"""
        ref = nodes_to_points(self.nodes, self.lm_nodes)
        self.overlays = [LM.procrustes_align(LM.canonical(self.analyses[p]["points"], self.analyses[p]["flip"]), ref)
                         for p in self.used]
        self.mean_disp = LM.procrustes_align(self.auto_pts, ref) if self.auto_pts is not None else None

    # ------------------------------------------------------------------ 그리기
    def redraw_all(self, reset_view=False):
        if self.nodes is None:
            return
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self._style_axes()
        cmap = matplotlib.colormaps["tab20"]
        if self.var_ov.get():
            for i, q in enumerate(self.overlays):
                self.ax.plot(*np.vstack([q, q[:1]]).T, lw=0.8, alpha=0.6, color=cmap(i % 20))
        if self.var_mean.get() and self.mean_disp is not None:
            self.ax.plot(*np.vstack([self.mean_disp, self.mean_disp[:1]]).T, "--", color="#777", lw=1.2)
        self.ln_curve, = self.ax.plot([], [], "k-", lw=2.4)
        self.sc_nodes, = self.ax.plot([], [], "o", ms=6, mfc="#2a6fdb", mec="white", mew=0.8, ls="none")
        self.sc_lm, = self.ax.plot([], [], "o", ms=9, mfc="#e01b24", mec="white", mew=1, ls="none")
        self.txt_lm = [self.ax.text(0, 0, n, color="#e01b24", fontsize=9, fontweight="bold") for n in LM.LM_NAMES]
        self.update_artists()
        if reset_view:
            allp = np.vstack(self.overlays + [self.nodes]) if self.overlays else self.nodes
            lo, hi = allp.min(0) - LM.MARGIN, allp.max(0) + LM.MARGIN
            self.ax.set_xlim(lo[0], hi[0])
            self.ax.set_ylim(hi[1], lo[1])          # 이미지 좌표 (y 아래로)
        else:
            self.ax.set_xlim(xl)
            self.ax.set_ylim(yl)
        self.cv.draw_idle()

    def update_artists(self):
        dense = spline_dense(self.nodes)
        self.ln_curve.set_data(*np.vstack([dense, dense[:1]]).T)
        show = self.var_nodes.get()
        others = np.setdiff1d(np.arange(len(self.nodes)), self.lm_nodes)
        self.sc_nodes.set_data(*(self.nodes[others].T if show else ([], [])))
        L = self.nodes[self.lm_nodes]
        self.sc_lm.set_data(*L.T)
        for t, (x, y) in zip(self.txt_lm, L):
            t.set_position((x + 8, y - 8))
        self.cv.draw_idle()

    # ------------------------------------------------------------------ 마우스 편집
    def _disp(self, xy):
        return self.ax.transData.transform(xy)

    def nearest_node(self, ev):
        d = np.linalg.norm(self._disp(self.nodes) - [ev.x, ev.y], axis=1)
        k = int(np.argmin(d))
        return (k, d[k]) if d[k] <= PICK_PX else (None, None)

    def push(self):
        self.hist.append((self.nodes.copy(), list(self.lm_nodes)))
        del self.hist[:-50]

    def on_press(self, ev):
        if self.nodes is None or ev.inaxes is not self.ax or self.toolbar.mode != "":
            return
        k, _ = self.nearest_node(ev)
        if ev.button == 1 and ev.dblclick:
            self.insert_node(ev)
        elif ev.button == 1 and k is not None:
            self.push()
            t, L = arc_positions(self.nodes)
            dt = np.abs(t - t[k])
            self.drag = dict(k=k, x0=ev.xdata, y0=ev.ydata, start=self.nodes.copy(),
                             dist=np.minimum(dt, L - dt))
        elif ev.button == 3 and k is not None:
            self.delete_node(k)

    def on_motion(self, ev):
        if not self.drag or ev.inaxes is not self.ax or ev.xdata is None:
            return
        d = self.drag
        delta = np.array([ev.xdata - d["x0"], ev.ydata - d["y0"]])
        rad = float(self.var_rad.get())
        if rad > 0:
            w = np.exp(-0.5 * (d["dist"] / (rad / 2)) ** 2)   # 윤곽선을 따라 잰 거리 기준 가우시안
        else:
            w = np.zeros(len(self.nodes))
        w[d["k"]] = 1
        self.nodes = d["start"] + w[:, None] * delta
        self.update_artists()

    def on_release(self, ev):
        if self.drag:
            if np.allclose(self.nodes, self.drag["start"]):
                self.hist.pop()                     # 움직이지 않았으면 기록 취소
            self.drag = None

    def insert_node(self, ev):
        dense = spline_dense(self.nodes)
        d = np.linalg.norm(self._disp(dense) - [ev.x, ev.y], axis=1)
        j = int(np.argmin(d))
        if d[j] > PICK_PX:
            self.status(T("조절점 추가: 윤곽선(검정) 위를 더블클릭하세요.", "To add a point, double-click on the (black) outline."))
            return
        k = j // N_PER                              # 조절점 k 와 k+1 사이
        self.push()
        self.nodes = np.insert(self.nodes, k + 1, dense[j], axis=0)
        self.lm_nodes = [i + 1 if i > k else i for i in self.lm_nodes]
        self.update_artists()
        self.status(T(f"조절점 추가 (총 {len(self.nodes)}개)", f"Point added ({len(self.nodes)} in total)"))

    def delete_node(self, k):
        if k in self.lm_nodes:
            nm_ = LM.LM_NAMES[self.lm_nodes.index(k)]
            self.status(T(f"'{nm_}' 랜드마크는 사진과 템플릿을 대응시키는 기준점이라 삭제할 수 없습니다. 끌어서 위치만 옮기세요.",
                          f"The '{nm_}' landmark anchors the correspondence and cannot be deleted; drag it to move it."))
            return
        if len(self.nodes) <= 8:
            self.status(T("조절점이 너무 적어 더 지울 수 없습니다.", "Too few points left to delete more."))
            return
        self.push()
        self.nodes = np.delete(self.nodes, k, axis=0)
        self.lm_nodes = [i - 1 if i > k else i for i in self.lm_nodes]
        self.update_artists()
        self.status(T(f"조절점 삭제 (총 {len(self.nodes)}개)", f"Point deleted ({len(self.nodes)} in total)"))

    # ------------------------------------------------------------------ 전체 윤곽 도구
    def undo(self):
        if not self.hist:
            self.status(T("되돌릴 편집이 없습니다.", "Nothing to undo."))
            return
        self.nodes, self.lm_nodes = self.hist.pop()
        self.update_artists()

    def smooth(self):
        """랜드마크를 뺀 조절점을 이웃의 중간 쪽으로 조금씩 당김 (라플라시안 평활)"""
        if self.nodes is None:
            return
        self.push()
        n = self.nodes
        mid = (np.roll(n, 1, axis=0) + np.roll(n, -1, axis=0)) / 2
        new = n + 0.5 * (mid - n)
        new[self.lm_nodes] = n[self.lm_nodes]
        self.nodes = new
        self.update_artists()

    def redistribute(self):
        """현재 모양은 유지하고, 랜드마크 사이 구간마다 조절점을 같은 간격으로 다시 배치"""
        if self.nodes is None:
            return
        self.push()
        dense = spline_dense(self.nodes)
        ring = np.vstack([dense, dense[:1]])
        K = len(self.nodes)
        cuts = [k * N_PER for k in self.lm_nodes] + [K * N_PER]
        counts = np.diff(list(self.lm_nodes) + [K])
        new, lm = [], []
        for i in range(5):
            lm.append(len(new))
            new.append(LM.resample_open(ring[cuts[i]:cuts[i + 1] + 1], int(counts[i])))
        self.nodes = np.vstack(new)
        self.lm_nodes = list(np.cumsum([0] + [len(a) for a in new[:-1]]))
        self.update_artists()

    def reset_auto(self):
        if self.auto_pts is None:
            return
        self.push()
        ref = nodes_to_points(self.nodes, self.lm_nodes)
        self.nodes, self.lm_nodes = points_to_nodes(LM.procrustes_align(self.auto_pts, ref))
        self.update_artists()

    # ------------------------------------------------------------------ 저장 / 불러오기
    def save(self):
        if self.nodes is None:
            messagebox.showinfo(T("템플릿 없음", "No template"), T("먼저 사진 폴더를 선택해 템플릿을 만들거나 기존 템플릿을 불러오세요.",
                                                            "Create a template from a photo folder or load an existing one first."))
            return
        vid = self.var_id.get().strip()
        if not vid or re.search(r'[\\/:*?"<>|\s]', vid):
            messagebox.showwarning(T("부위 ID", "View ID"), T("부위 ID를 입력하세요 (공백, \\ / : * ? \" < > | 사용 불가).",
                                                         "Enter a view ID (no spaces or \\ / : * ? \" < > |)."))
            return
        if not self.keywords():
            messagebox.showwarning(T("폴더 키워드", "Folder keywords"), T("폴더 키워드를 하나 이상 입력하세요.", "Enter at least one folder keyword."))
            return
        path = os.path.join(LM.TEMPLATE_DIR, f"template_{vid}.json")
        if os.path.exists(path) and not messagebox.askyesno(T("덮어쓰기", "Overwrite"), T(f"template_{vid} 가 이미 있습니다. 덮어쓸까요?",
                                                                                    f"template_{vid} already exists. Overwrite?")):
            return
        if self.corr() != self.corr_used and not messagebox.askyesno(
                T("대응 방식", "Correspondence"),
                T(f"현재 윤곽선은 '{self.corr_label(self.corr_used)}' 방식으로 계산했는데, "
                  f"'{self.corr_label(self.corr())}' 방식으로 저장하려 합니다.\n"
                  f"[자동 템플릿 계산]을 다시 하는 것을 권장합니다. 그대로 저장할까요?",
                  f"The outline was computed with '{self.corr_label(self.corr_used)}' correspondence but will be "
                  f"saved as '{self.corr_label(self.corr())}'.\nRecomputing the template is recommended. Save anyway?")):
            return
        pts = nodes_to_points(self.nodes, self.lm_nodes)
        tpl = LM.make_template(vid, pts, [self.rel(p) for p in self.used], self.var_label.get().strip() or vid,
                               self.keywords(), self.nodes, self.lm_nodes, self.corr())
        try:
            LM.save_template(tpl, [self.analyses[p] for p in self.used])
        except PermissionError as e:
            messagebox.showerror(T("저장 실패", "Save failed"), str(e))
            return
        self.status(f"{T('저장 완료', 'Saved')}: template_{vid}.json / .png / _check.png  ({tpl['canvas'][0]}×{tpl['canvas'][1]}px)\n"
                    f"→ {LM.TEMPLATE_DIR}\n" + T("정합 GUI를 다시 열거나 부위 목록의 ↻ 를 누르면 나타납니다.",
                                                   "It appears after reopening the registration GUI or pressing ↻ in the view list."))
        if messagebox.askyesno(T("저장 완료", "Saved"), T(f"template_{vid} 저장했습니다.\n저장된 템플릿 그림을 열까요?",
                                                       f"template_{vid} saved.\nOpen the template image?")):
            os.startfile(os.path.join(LM.TEMPLATE_DIR, f"template_{vid}.png"))

    def load_existing(self):
        p = filedialog.askopenfilename(title=T("기존 템플릿 불러오기", "Load existing template"), initialdir=LM.TEMPLATE_DIR,
                                       filetypes=[(T("템플릿", "Template"), "template_*.json")])
        if not p:
            return
        tpl = LM.load_template(os.path.basename(p)[len("template_"):-5])
        if "nodes" in tpl:
            self.nodes, self.lm_nodes = np.array(tpl["nodes"], float), list(tpl["node_landmarks"])
        else:
            self.nodes, self.lm_nodes = points_to_nodes(tpl["points"])
        self.var_id.set(tpl["view"])
        self.var_label.set(LM.view_label(tpl))
        self.var_kw.set(", ".join(tpl.get("keywords") or [LM.VIEWS.get(tpl["view"], tpl["view"])]))
        self.corr_used = tpl.get("correspondence", "arc")
        self.var_dtw.set(self.corr_used == "dtw")
        self.hist = []
        # 사진을 아직 분석 안 했으면 템플릿에 기록된 사진들을 불러와 겹쳐 보기
        srcs = [s if os.path.isabs(s) else os.path.join(LM.DATA_DIR, s) for s in tpl.get("sources", [])]
        srcs = [s for s in srcs if os.path.exists(s)]
        if self.used and not srcs:
            srcs = self.used
        missing = [s for s in srcs if s not in self.analyses]
        if missing:
            self.status(T("템플릿에 사용된 사진 윤곽선 분석 중…", "Analyzing outlines of the template photographs…"))
            self.root.update()
            for s in missing:
                self.analyses[s] = LM.analyze(LM.load_rgba(s))
        self.used = srcs
        if srcs:
            self.paths = list(dict.fromkeys(self.paths + srcs))
            self.fill_list()
            self.lst.selection_clear(0, "end")
            for s in srcs:
                self.lst.selection_set(self.paths.index(s))
            self.auto_pts = LM.template_shape([self.analyses[s] for s in srcs], self.corr_used)
        else:
            self.auto_pts = None
        self.align_overlays()
        self.redraw_all(reset_view=True)
        self.status(T(f"template_{tpl['view']} 불러옴 (조절점 {len(self.nodes)}개). 편집 후 저장하세요.",
                      f"template_{tpl['view']} loaded ({len(self.nodes)} control points). Edit and save."))


def main():
    root = tk.Tk()
    ttk.Style().theme_use("vista" if "vista" in ttk.Style().theme_names() else "clam")
    TemplateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
