"""
쥐 앞다리 사진 정합 + 파란 점 검출 GUI
언어: 메뉴 [언어 / Language] 또는  python limb_mapping_gui.py --lang en  (그림 안의 글자는 항상 영어)

실행:  python limb_mapping_gui.py

화면 (2x2)
  좌상: 폴더 선택, 부위, 정합 실행, 파란 점 임계값(슬라이더) 등 설정
  우상: 원본 사진 (◀ ▶ 로 사진 넘기기)
  좌하: 정합된 사진 + 공통 템플릿(격자)
  우하: 정합된 사진에서 검출한 파란 점 + 공통 템플릿(격자)  -> 임계값을 바꾸면 바로 다시 그림

흐름
  1) 부위 폴더(예: mapping data/C6/내측(Ulnar side)) 선택 -> [정합 실행]
     정합된 그림은 '<부위 폴더>_registered' 폴더에 저장
  2) 임계값을 조절하며 파란 점 확인 -> [다음 ▶] 누르면 현재 사진의 파란 점 그림/좌표 저장 후 다음 사진
  3) 마지막 사진에서 [다음 ▶] -> 저장 + 전체 요약(빈도 지도, 정합 요약) 저장
"""
import os
import queue
import threading
import tkinter as tk
import warnings
from tkinter import filedialog, messagebox, ttk

# 라이브러리 import 시 나오는 무해한 cryptography(paramiko) 경고 숨김
warnings.filterwarnings("ignore", message=r".*cryptography\.hazmat\.decrepit.*")

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image

import limb_mapping as LM
from limb_i18n import T, fig_name, add_language_menu, view_en



class App:
    def __init__(self, root):
        self.root = root
        root.title(T("쥐 앞다리 피부 매핑 - 정합 & 파란 점 검출", "Rat forelimb skin mapping - registration & blue-spot detection"))
        add_language_menu(root)
        root.geometry("1500x950")

        self.folder = None
        self.out_dir = None
        self.view = None
        self.tpl = None
        self.tmask = None
        self.tgrid = None
        self.names = []
        self.regs = {}          # 사진 이름 -> 정합 결과
        self.cache = {}         # 사진 이름 -> (파란색 지수, 조직 영역)  임계값 바꿀 때 재사용
        self.saved = {}         # 사진 이름 -> 저장한 결과 (threshold, min_area, blue, dots)
        self.edits = {}         # 사진 이름 -> 수동 수정 {"add": 마스크, "rem": 마스크, "hist": 되돌리기용}
        self.flips = set()
        self.idx = 0
        self.q = queue.Queue()
        self.busy = False
        self._after = None

        root.columnconfigure(0, weight=1, uniform="c")
        root.columnconfigure(1, weight=1, uniform="c")
        root.rowconfigure(0, weight=1, uniform="r")
        root.rowconfigure(1, weight=1, uniform="r")
        self._build_controls()
        self._build_original()
        self.fig_reg, self.ax_reg, self.cv_reg = self._make_panel(1, 0, T("정합된 사진 + 공통 템플릿", "Registered photograph + common template"))
        self.fig_dot, self.ax_dot, self.cv_dot = self._make_panel(
            1, 1, T("파란 점 + 공통 템플릿   (왼쪽 클릭: 추가 / 오른쪽 클릭: 삭제)",
                  "Blue spots + common template   (left click: add / right click: remove)"))
        self.cv_dot.mpl_connect("button_press_event", self.on_click_dots)
        root.bind_all("<Control-z>", lambda e: self.undo_edit())
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ 화면 구성
    def _make_panel(self, r, c, title, parent=None):
        frm = ttk.LabelFrame(parent or self.root, text=title)
        if parent is None:
            frm.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
        else:
            frm.pack(fill="both", expand=True)
        fig = Figure(figsize=(5, 4), dpi=90)
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.92])
        ax.axis("off")
        cv = FigureCanvasTkAgg(fig, master=frm)
        cv.get_tk_widget().pack(fill="both", expand=True)
        return fig, ax, cv

    def _build_controls(self):
        f = ttk.LabelFrame(self.root, text=T("설정", "Settings"))
        f.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        f.columnconfigure(1, weight=1)
        r = 0
        ttk.Button(f, text=T("폴더 선택…", "Select folder…"), command=self.choose_folder).grid(row=r, column=0, sticky="w", padx=4, pady=3)
        self.lbl_folder = ttk.Label(f, text=T("(부위 폴더를 선택하세요)", "(select a view folder)"), foreground="gray", wraplength=560)
        self.lbl_folder.grid(row=r, column=1, columnspan=3, sticky="w")
        r += 1
        ttk.Label(f, text=T("부위", "View")).grid(row=r, column=0, sticky="w", padx=4)
        self.var_view = tk.StringVar(value="auto")
        self.view_frame = ttk.Frame(f)
        self.view_frame.grid(row=r, column=1, columnspan=3, sticky="w")
        self.refresh_views()
        r += 1
        self.btn_run = ttk.Button(f, text=T("정합 실행", "Run registration"), command=self.run_registration)
        self.btn_run.grid(row=r, column=0, sticky="w", padx=4, pady=3)
        self.pbar = ttk.Progressbar(f, mode="determinate")
        self.pbar.grid(row=r, column=1, columnspan=3, sticky="ew", padx=4)
        r += 1
        self.var_flip = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text=T("현재 사진 좌우 반전(미러링)해서 다시 정합 (반대쪽 다리 등)",
                                         "Mirror the current photograph and register again (e.g. contralateral limb)"),
                        variable=self.var_flip, command=self.toggle_flip).grid(
            row=r, column=0, columnspan=4, sticky="w", padx=4)
        r += 1
        ttk.Separator(f).grid(row=r, column=0, columnspan=4, sticky="ew", pady=5)
        r += 1

        ttk.Label(f, text=T("파란 점 임계값\n(낮을수록 민감)", "Blue-spot threshold\n(lower = more sensitive)")).grid(row=r, column=0, sticky="w", padx=4)
        self.var_thr = tk.DoubleVar(value=40)
        self.scl_thr = ttk.Scale(f, from_=0, to=150, variable=self.var_thr, command=lambda e: self.schedule_redraw())
        self.scl_thr.grid(row=r, column=1, columnspan=2, sticky="ew", padx=4)
        self.lbl_thr = ttk.Label(f, text="40", width=6)
        self.lbl_thr.grid(row=r, column=3, sticky="w")
        r += 1
        ttk.Label(f, text=T("자동 임계값 k", "Adaptive threshold k")).grid(row=r, column=0, sticky="w", padx=4)
        kf = ttk.Frame(f)
        kf.grid(row=r, column=1, columnspan=3, sticky="w")
        self.var_k = tk.DoubleVar(value=4.0)
        ttk.Spinbox(kf, from_=1, to=10, increment=0.5, textvariable=self.var_k, width=6).pack(side="left", padx=2)
        ttk.Button(kf, text=T("자동값 적용", "Apply adaptive"), command=self.apply_auto).pack(side="left", padx=4)
        ttk.Label(kf, text=T("(배경 중앙값 + k×MAD)", "(background median + k×MAD)"), foreground="gray").pack(side="left")
        r += 1
        self.var_auto_next = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text=T("새 사진으로 넘어갈 때 자동 임계값으로 시작 (해제하면 현재 값 유지)",
                                         "Start each new photograph with the adaptive threshold (unchecked = keep current value)"),
                        variable=self.var_auto_next).grid(row=r, column=0, columnspan=4, sticky="w", padx=4)
        r += 1
        ttk.Label(f, text=T("최소 점 크기 (px)", "Minimum spot size (px)")).grid(row=r, column=0, sticky="w", padx=4)
        self.var_minarea = tk.IntVar(value=6)
        ttk.Spinbox(f, from_=1, to=200, increment=1, textvariable=self.var_minarea, width=6,
                    command=self.schedule_redraw).grid(row=r, column=1, sticky="w", padx=4)
        r += 1
        cf = ttk.Frame(f)
        cf.grid(row=r, column=0, columnspan=4, sticky="w", padx=4)
        self.var_under = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, text=T("파란 점 화면에 정합 사진 흐리게 겹쳐 보기", "Show faded registered photograph under spots"), variable=self.var_under,
                        command=self.schedule_redraw).pack(side="left")
        self.var_lm = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, text=T("원본에 윤곽선·랜드마크 표시", "Show outline and landmarks on original"), variable=self.var_lm,
                        command=self.show_original).pack(side="left", padx=12)
        r += 1
        ttk.Separator(f).grid(row=r, column=0, columnspan=4, sticky="ew", pady=5)
        r += 1
        ttk.Label(f, text=T("수동 수정", "Manual correction")).grid(row=r, column=0, sticky="w", padx=4)
        ttk.Label(f, text=T("우하단 그림에서  왼쪽 클릭 = 그 주변 파란 영역 추가,  오른쪽 클릭 = 영역 삭제",
                         "In the bottom-right panel: left click = add the blue region around it, right click = remove a region"),
                  foreground="#444").grid(row=r, column=1, columnspan=3, sticky="w")
        r += 1
        ttk.Label(f, text=T("추가 영역 기준\n(낮을수록 넓게)", "Region-growing level\n(lower = larger)")).grid(row=r, column=0, sticky="w", padx=4)
        self.var_frac = tk.DoubleVar(value=0.5)
        ttk.Scale(f, from_=0.15, to=0.9, variable=self.var_frac,
                  command=lambda e: self.lbl_frac.config(text=f"{self.var_frac.get():.2f}")).grid(
            row=r, column=1, sticky="ew", padx=4)
        mf = ttk.Frame(f)
        mf.grid(row=r, column=2, columnspan=2, sticky="w")
        self.lbl_frac = ttk.Label(mf, text="0.50", width=5)
        self.lbl_frac.pack(side="left")
        ttk.Label(mf, text=T("최대 반경(px)", "Max radius (px)")).pack(side="left", padx=(6, 0))
        self.var_radius = tk.IntVar(value=25)
        ttk.Spinbox(mf, from_=3, to=150, increment=1, textvariable=self.var_radius, width=5).pack(side="left", padx=3)
        r += 1
        bf = ttk.Frame(f)
        bf.grid(row=r, column=1, columnspan=3, sticky="w")
        ttk.Button(bf, text=T("되돌리기 (Ctrl+Z)", "Undo (Ctrl+Z)"), command=self.undo_edit).pack(side="left", padx=2)
        ttk.Button(bf, text=T("이 사진의 수동 수정 모두 지우기", "Clear all manual edits of this photograph"), command=self.clear_edits).pack(side="left", padx=6)
        r += 1
        ttk.Separator(f).grid(row=r, column=0, columnspan=4, sticky="ew", pady=5)
        r += 1
        ttk.Label(f, text=T("요약 지도", "Summary map")).grid(row=r, column=0, sticky="w", padx=4)
        sf = ttk.Frame(f)
        sf.grid(row=r, column=1, columnspan=3, sticky="w")
        ttk.Label(sf, text=T("번짐 sigma(px)", "Smoothing sigma (px)")).pack(side="left")
        self.var_sigma = tk.DoubleVar(value=6.0)
        ttk.Spinbox(sf, from_=0, to=30, increment=1, textvariable=self.var_sigma, width=5).pack(side="left", padx=3)
        ttk.Label(sf, text=T("공통 영역 기준", "Common-area level")).pack(side="left", padx=(10, 0))
        self.var_common = tk.DoubleVar(value=0.5)
        ttk.Spinbox(sf, from_=0.1, to=1.0, increment=0.05, textvariable=self.var_common, width=5).pack(side="left", padx=3)
        ttk.Button(sf, text=T("지금 요약 저장", "Save summary now"), command=self.finalize).pack(side="left", padx=10)
        r += 1
        self.lbl_status = ttk.Label(f, text="", foreground="#1a5fb4", wraplength=640, justify="left")
        self.lbl_status.grid(row=r, column=0, columnspan=4, sticky="w", padx=4, pady=6)

    def _build_original(self):
        frm = ttk.LabelFrame(self.root, text=T("원본 사진", "Original photograph"))
        frm.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        nav = ttk.Frame(frm)
        nav.pack(fill="x")
        self.btn_prev = ttk.Button(nav, text=T("◀ 이전", "◀ Prev"), command=self.prev_image, state="disabled")
        self.btn_prev.pack(side="left", padx=4, pady=2)
        self.var_idx = tk.StringVar(value="0")
        self.spn = ttk.Spinbox(nav, from_=1, to=1, textvariable=self.var_idx, width=5, state="disabled",
                               command=self.jump_image)
        self.spn.pack(side="left")
        self.spn.bind("<Return>", lambda e: self.jump_image())
        self.lbl_idx = ttk.Label(nav, text="/ 0")
        self.lbl_idx.pack(side="left", padx=4)
        self.btn_next = ttk.Button(nav, text=T("저장 후 다음 ▶", "Save & next ▶"), command=self.next_image, state="disabled")
        self.btn_next.pack(side="left", padx=4)
        self.lbl_name = ttk.Label(nav, text="", font=("Malgun Gothic", 10, "bold"))
        self.lbl_name.pack(side="left", padx=10)
        inner = ttk.Frame(frm)
        inner.pack(fill="both", expand=True)
        self.fig_org = Figure(figsize=(5, 4), dpi=90)
        self.ax_org = self.fig_org.add_axes([0.01, 0.01, 0.98, 0.95])
        self.ax_org.axis("off")
        self.cv_org = FigureCanvasTkAgg(self.fig_org, master=inner)
        self.cv_org.get_tk_widget().pack(fill="both", expand=True)

    def status(self, msg):
        self.lbl_status.config(text=msg)

    # ------------------------------------------------------------------ 폴더 / 정합
    def choose_folder(self):
        init = LM.DATA_DIR if os.path.isdir(LM.DATA_DIR) else os.getcwd()
        d = filedialog.askdirectory(title=T("부위 폴더 선택 (그림이 들어있는 폴더)", "Select a view folder (containing the photographs)"), initialdir=init)
        if not d:
            return
        d = os.path.normpath(d)
        names = LM.list_images(d)
        if not names:
            messagebox.showwarning(T("그림 없음", "No images"), T(f"이 폴더에 그림 파일이 없습니다:\n{d}", f"No image files in this folder:\n{d}"))
            return
        self.folder = d
        try:
            v_ = LM.detect_view(d)
            auto = T(LM.view_label(v_), view_en(v_))
        except ValueError:
            auto = T("알 수 없음 → 부위를 직접 선택하세요", "unknown → select the view manually")
        self.lbl_folder.config(text=T(f"{d}\n그림 {len(names)}장, 폴더 이름으로 판단한 부위: {auto}",
                                         f"{d}\n{len(names)} images; view from folder name: {auto}"), foreground="black")
        self.status(T("[정합 실행]을 누르세요.", "Press [Run registration]."))

    def refresh_views(self):
        """templates 폴더에 있는 템플릿으로 부위 목록(옵션 버튼)을 다시 만든다"""
        for w in self.view_frame.winfo_children():
            w.destroy()
        views = LM.available_views()
        choices = [(T("자동 (폴더 이름)", "Auto (folder name)"), "auto")] + [(T(lab, view_en(v)), v) for v, lab, _ in views]
        per_row = 5
        for i, (t, v) in enumerate(choices):
            ttk.Radiobutton(self.view_frame, text=t, value=v, variable=self.var_view).grid(
                row=i // per_row, column=i % per_row, sticky="w", padx=2)
        ttk.Button(self.view_frame, text="↻", width=3, command=self.refresh_views).grid(
            row=0, column=per_row, padx=4)
        if self.var_view.get() not in [v for _, v in choices]:
            self.var_view.set("auto")

    def _resolve_view(self):
        v = self.var_view.get()
        if v != "auto":
            return v
        return LM.detect_view(self.folder)

    def run_registration(self):
        if self.busy:
            return
        if not self.folder:
            messagebox.showinfo(T("폴더", "Folder"), T("먼저 부위 폴더를 선택하세요.", "Select a view folder first."))
            return
        try:
            self.view = self._resolve_view()
            self.tpl = LM.load_template(self.view)
        except Exception as e:
            messagebox.showerror(T("오류", "Error"), str(e))
            return
        self.tmask = LM.template_mask(self.tpl)
        self.tgrid = np.asarray(LM.draw_template(self.tpl, outline=False))
        self.names = LM.list_images(self.folder)
        self.out_dir = LM.registered_dir(self.folder)
        os.makedirs(self.out_dir, exist_ok=True)
        self.regs, self.cache, self.saved, self.edits = {}, {}, {}, {}
        self.flips = set()
        self.pbar.config(maximum=len(self.names), value=0)
        self._set_busy(True)
        threading.Thread(target=self._register_worker, args=(list(self.names),), daemon=True).start()

    def _register_worker(self, names):
        for i, nm in enumerate(names):
            self.q.put(("status", T(f"정합 중… {i + 1}/{len(names)}  {nm}", f"Registering… {i + 1}/{len(names)}  {nm}")))
            try:
                reg = LM.register_image(os.path.join(self.folder, nm), self.tpl, self.tmask, nm in self.flips)
                Image.fromarray(reg["warped"]).save(
                    os.path.join(self.out_dir, f"{os.path.splitext(nm)[0]}_registered.png"))
                self.q.put(("reg", nm, reg))
            except Exception as e:
                self.q.put(("error", nm, e))
            self.q.put(("progress", i + 1))
        self.q.put(("done",))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "status":
                    self.status(msg[1])
                elif kind == "progress":
                    self.pbar.config(value=msg[1])
                elif kind == "reg":
                    self.regs[msg[1]] = msg[2]
                    self.cache.pop(msg[1], None)
                elif kind == "error":
                    self.status(T(f"{msg[1]} 정합 실패: {msg[2]}", f"{msg[1]} registration failed: {msg[2]}"))
                elif kind == "done":
                    self._set_busy(False)
                    self.names = [n for n in self.names if n in self.regs]
                    if not self.names:
                        self.status(T("정합된 사진이 없습니다.", "No photographs were registered."))
                    else:
                        self.spn.config(to=len(self.names))
                        self.lbl_idx.config(text=f"/ {len(self.names)}")
                        dtw = self.tpl.get('correspondence') == 'dtw'
                        self.status(T(f"정합 완료 ({LM.view_label(self.view)}, 대응: "
                                      f"{'모양 기반(DTW)' if dtw else '길이 비율'}) → {self.out_dir}\n"
                                      f"임계값을 조절하고 [저장 후 다음 ▶]을 누르세요.",
                                      f"Registration finished ({view_en(self.view)}, correspondence: "
                                      f"{'shape-based (DTW)' if dtw else 'arc-length'}) → {self.out_dir}\n"
                                      f"Adjust the threshold and press [Save & next ▶]."))
                        self.show(0, fresh=True)
                elif kind == "reflip":
                    self._set_busy(False)
                    nm, reg = msg[1], msg[2]
                    self.regs[nm] = reg
                    self.cache.pop(nm, None)
                    self.saved.pop(nm, None)
                    self.edits.pop(nm, None)          # 정합이 바뀌면 수동 수정 위치도 무효
                    self.status(T(f"{nm} 좌우 반전={reg['analysis']['flip']} 로 다시 정합했습니다 (수동 수정 초기화).",
                                  f"{nm} registered again with mirror={reg['analysis']['flip']} (manual edits cleared)."))
                    self.show(self.idx, fresh=True)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _set_busy(self, b):
        self.busy = b
        st = "disabled" if b else "normal"
        self.btn_run.config(state=st)
        has = bool(self.regs) and not b
        for w in (self.btn_prev, self.btn_next):
            w.config(state="normal" if has else "disabled")
        self.spn.config(state="normal" if has else "disabled")

    def toggle_flip(self):
        if not self.names or self.busy:
            self.var_flip.set(False)
            return
        nm = self.names[self.idx]
        if self.var_flip.get():
            self.flips.add(nm)
        else:
            self.flips.discard(nm)
        self._set_busy(True)
        self.status(T(f"{nm} 다시 정합 중…", f"Registering {nm} again…"))

        def work():
            reg = LM.register_image(os.path.join(self.folder, nm), self.tpl, self.tmask, nm in self.flips)
            Image.fromarray(reg["warped"]).save(os.path.join(self.out_dir, f"{os.path.splitext(nm)[0]}_registered.png"))
            self.q.put(("reflip", nm, reg))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ 표시
    def current(self):
        nm = self.names[self.idx]
        return nm, self.regs[nm]

    def show(self, idx, fresh=False):
        self.idx = idx
        nm, reg = self.current()
        self.var_idx.set(str(idx + 1))
        self.lbl_name.config(text=nm)
        self.var_flip.set(nm in self.flips)
        if nm in self.saved:                       # 이미 저장한 사진이면 그때 값으로
            self.var_thr.set(self.saved[nm]["threshold"])
            self.var_minarea.set(self.saved[nm]["min_area"])
        elif fresh or self.var_auto_next.get():
            self.var_thr.set(round(LM.auto_threshold(reg["warped"], self.tmask, self._k()), 1))
        self.btn_next.config(text=T("저장 후 요약 ✔", "Save & summarize ✔") if idx == len(self.names) - 1
                             else T("저장 후 다음 ▶", "Save & next ▶"))
        self.show_original()
        self.ax_reg.clear()
        LM.plot_registered(self.ax_reg, self.tpl, self.tgrid, reg["warped"],
                           f"{fig_name(nm)}   Dice={reg['dice']:.3f}  (red: common template)")
        self.cv_reg.draw_idle()
        self.redraw_dots()

    def show_original(self):
        if not self.names or not self.regs:
            return
        nm, reg = self.current()
        self.ax_org.clear()
        a = reg["analysis"]
        LM.plot_original(self.ax_org, reg["rgba"], a if self.var_lm.get() else None, self.tpl,
                         f"{fig_name(nm)}  ({reg['rgba'].shape[1]}×{reg['rgba'].shape[0]})"
                         f"{'  [mirrored]' if a['flip'] else ''}{'  [landmark detection failed]' if not a['landmarks_ok'] else ''}")
        self.cv_org.draw_idle()

    def _k(self):
        try:
            return float(self.var_k.get())
        except (tk.TclError, ValueError):
            return 4.0

    def _minarea(self):
        try:
            return max(1, int(self.var_minarea.get()))
        except (tk.TclError, ValueError):
            return 6

    def apply_auto(self):
        if self.names and self.regs:
            nm, reg = self.current()
            self.var_thr.set(round(LM.auto_threshold(reg["warped"], self.tmask, self._k()), 1))
            self.schedule_redraw()

    def schedule_redraw(self):
        self.lbl_thr.config(text=f"{self.var_thr.get():.0f}")
        if self._after:
            self.root.after_cancel(self._after)
        self._after = self.root.after(60, self.redraw_dots)

    def detect_current(self):
        nm, reg = self.current()
        if nm not in self.cache:
            self.cache[nm] = (LM.blue_index(reg["warped"]), LM.valid_tissue(reg["warped"], self.tmask))
        bidx, valid = self.cache[nm]
        thr = float(self.var_thr.get())
        blue = LM.detect_blue(reg["warped"], self.tmask, thr, self._minarea(), bidx, valid)
        ed = self.edits.get(nm)
        add = rem = None
        if ed:                                      # 최종 = (자동 검출 ∪ 수동 추가) − 수동 삭제
            add, rem = ed["add"], ed["rem"]
            blue = (blue | add) & ~rem
        dots = LM.extract_dots(blue, bidx, reg, self.tpl, manual=add)
        return nm, reg, thr, blue, dots, add, rem

    def redraw_dots(self):
        self._after = None
        if not self.names or not self.regs:
            return
        nm, reg, thr, blue, dots, add, rem = self.detect_current()
        self.lbl_thr.config(text=f"{thr:.0f}")
        n_man = sum(d["source"] != "auto" for d in dots)
        self.ax_dot.clear()
        LM.plot_bluedots(self.ax_dot, self.tpl, self.tgrid, blue, dots,
                         under=reg["warped"] if self.var_under.get() else None,
                         title=f"Threshold {thr:.0f}, min {self._minarea()} px  →  {len(dots)} spots"
                               f"{f' ({n_man} manual)' if n_man else ''}"
                               f"{'   (saved)' if nm in self.saved else ''}",
                         added=add, removed=rem)
        self.cv_dot.draw_idle()

    # ------------------------------------------------------------------ 수동 수정 (우하단 클릭)
    def on_click_dots(self, ev):
        if self.busy or not self.names or not self.regs or ev.inaxes is not self.ax_dot or ev.xdata is None:
            return
        nm, reg = self.current()
        if nm not in self.cache:
            self.cache[nm] = (LM.blue_index(reg["warped"]), LM.valid_tissue(reg["warped"], self.tmask))
        bidx, valid = self.cache[nm]
        H, W = self.tmask.shape
        ed = self.edits.setdefault(nm, dict(add=np.zeros((H, W), bool), rem=np.zeros((H, W), bool), hist=[]))
        if ev.button == 1:                          # 왼쪽 클릭: 주변 파란 영역 추가
            try:
                radius = max(3, int(self.var_radius.get()))
            except (tk.TclError, ValueError):
                radius = 25
            region, msg = LM.grow_region(bidx, valid, ev.xdata, ev.ydata, float(self.var_frac.get()), radius)
            if region is None:
                self.status(T(f"추가 안 됨: {msg}", f"Not added: {msg}"))
                return
            self._push_hist(ed)
            ed["add"] |= region
            ed["rem"] &= ~region
            self.status(msg)
        elif ev.button == 3:                        # 오른쪽 클릭: 클릭한 영역 삭제
            *_, blue, _dots, _a, _r = self.detect_current()
            comp = LM.component_at(blue, ev.xdata, ev.ydata)
            if comp is None:
                self.status(T("삭제할 파란 영역이 클릭한 곳 근처에 없습니다.", "No blue region near the clicked point."))
                return
            self._push_hist(ed)
            ed["rem"] |= comp
            ed["add"] &= ~comp
            self.status(T(f"영역 삭제: {int(comp.sum())}px", f"Region removed: {int(comp.sum())} px"))
        else:
            return
        self.redraw_dots()

    def _push_hist(self, ed):
        ed["hist"].append((ed["add"].copy(), ed["rem"].copy()))
        del ed["hist"][:-30]                        # 최근 30번까지만 되돌리기

    def undo_edit(self):
        if not self.names or not self.regs:
            return
        ed = self.edits.get(self.names[self.idx])
        if not ed or not ed["hist"]:
            self.status(T("되돌릴 수동 수정이 없습니다.", "Nothing to undo."))
            return
        ed["add"], ed["rem"] = ed["hist"].pop()
        self.status(T("수동 수정 1단계 되돌림", "Undid one manual edit"))
        self.redraw_dots()

    def clear_edits(self):
        if not self.names or not self.regs:
            return
        nm = self.names[self.idx]
        if self.edits.pop(nm, None) is not None:
            self.status(T(f"{nm} 수동 수정을 모두 지웠습니다.", f"Cleared all manual edits of {nm}."))
            self.redraw_dots()

    # ------------------------------------------------------------------ 저장 / 이동
    def save_current(self):
        nm, reg, thr, blue, dots, add, rem = self.detect_current()
        stem = os.path.splitext(nm)[0]
        n_man = sum(d["source"] != "auto" for d in dots)
        LM.save_bluedots_png(os.path.join(self.out_dir, f"{stem}_bluedots.png"), self.tpl, self.tgrid, blue, dots,
                             f"{fig_name(nm)}  (threshold {thr:.0f}, min {self._minarea()} px, {len(dots)} spots"
                             f"{f', {n_man} manual' if n_man else ''})", added=add, removed=rem)
        Image.fromarray((blue * 255).astype(np.uint8)).save(os.path.join(self.out_dir, f"{stem}_bluemask.png"))
        self.saved[nm] = dict(threshold=round(thr, 1), min_area=self._minarea(), blue=blue, dots=dots)
        self.write_tables()
        return nm

    def write_tables(self):
        """저장한 모든 사진의 점 좌표를 하나의 CSV로 (사진 순서대로, 다시 저장하면 그 사진만 갱신)"""
        rows, summ = [], []
        for nm in self.names:
            if nm not in self.saved:
                continue
            s = self.saved[nm]
            reg = self.regs[nm]
            rows += [dict(image=nm, threshold=s["threshold"], min_area=s["min_area"], **d) for d in s["dots"]]
            a = reg["analysis"]
            summ.append(dict(image=nm, flip=a["flip"], landmarks_ok=a["landmarks_ok"],
                             truncated=round(a["truncated"], 3), dice=round(reg["dice"], 3),
                             threshold=s["threshold"], min_area=s["min_area"], n_dots=len(s["dots"]),
                             n_manual=sum(d["source"] != "auto" for d in s["dots"]),
                             blue_area_tpl_px=int(s["blue"].sum())))
        LM.write_csv(os.path.join(self.out_dir, "blue_dots.csv"), rows, LM.DOT_FIELDS)
        if summ:
            LM.write_csv(os.path.join(self.out_dir, "summary.csv"), summ, list(summ[0].keys()))
        return rows

    def next_image(self):
        if self.busy or not self.names:
            return
        try:
            nm = self.save_current()
        except PermissionError as e:
            messagebox.showerror(T("저장 실패", "Save failed"), T(f"파일이 다른 프로그램(엑셀 등)에서 열려 있는지 확인하세요.\n{e}",
                                                           f"Check whether the file is open in another program (e.g. Excel).\n{e}"))
            return
        if self.idx < len(self.names) - 1:
            self.status(T(f"{nm} 저장 완료 → {self.out_dir}", f"{nm} saved → {self.out_dir}"))
            self.show(self.idx + 1)
        else:
            self.finalize()

    def prev_image(self):
        if self.busy or not self.names or self.idx == 0:
            return
        self.show(self.idx - 1)

    def jump_image(self):
        if self.busy or not self.names:
            return
        try:
            i = int(self.var_idx.get()) - 1
        except ValueError:
            return
        i = min(max(i, 0), len(self.names) - 1)
        if i != self.idx:
            self.show(i)

    def finalize(self):
        if not self.names or not self.regs:
            return
        missing = [n for n in self.names if n not in self.saved]
        if missing:
            ok = messagebox.askyesno(
                T("저장 안 된 사진", "Unsaved photographs"),
                T(f"아직 저장하지 않은 사진이 {len(missing)}장 있습니다:\n{', '.join(missing)}\n\n"
                  f"자동 임계값으로 검출해서 포함할까요? (아니오 = 저장한 사진만으로 요약)",
                  f"{len(missing)} photographs have not been saved yet:\n{', '.join(missing)}\n\n"
                  f"Detect them with the adaptive threshold and include them? (No = summarize saved photographs only)"))
            if ok:
                for nm in missing:
                    reg = self.regs[nm]
                    thr = round(LM.auto_threshold(reg["warped"], self.tmask, self._k()), 1)
                    bidx = LM.blue_index(reg["warped"])
                    blue = LM.detect_blue(reg["warped"], self.tmask, thr, self._minarea(), bidx)
                    ed = self.edits.get(nm)
                    if ed:
                        blue = (blue | ed["add"]) & ~ed["rem"]
                    dots = LM.extract_dots(blue, bidx, reg, self.tpl, manual=ed["add"] if ed else None)
                    stem = os.path.splitext(nm)[0]
                    LM.save_bluedots_png(os.path.join(self.out_dir, f"{stem}_bluedots.png"), self.tpl, self.tgrid,
                                         blue, dots, f"{fig_name(nm)}  (adaptive threshold {thr:.0f}, {len(dots)} spots)")
                    Image.fromarray((blue * 255).astype(np.uint8)).save(
                        os.path.join(self.out_dir, f"{stem}_bluemask.png"))
                    self.saved[nm] = dict(threshold=thr, min_area=self._minarea(), blue=blue, dots=dots)
        names = [n for n in self.names if n in self.saved]
        if not names:
            messagebox.showinfo(T("요약", "Summary"), T("저장된 사진이 없습니다.", "No saved photographs."))
            return
        try:
            rows = self.write_tables()
            LM.save_overview(os.path.join(self.out_dir, "_overview_registered.png"), self.tpl, self.names,
                             [self.regs[n]["warped"] for n in self.names], [self.regs[n]["dice"] for n in self.names])
            LM.save_frequency(self.out_dir, self.tpl, names, [self.saved[n]["blue"] for n in names], rows,
                              float(self.var_sigma.get()), float(self.var_common.get()))
        except PermissionError as e:
            messagebox.showerror(T("저장 실패", "Save failed"), T(f"파일이 다른 프로그램(엑셀 등)에서 열려 있는지 확인하세요.\n{e}",
                                                           f"Check whether the file is open in another program (e.g. Excel).\n{e}"))
            return
        self.status(T(f"요약 저장 완료 ({len(names)}장) → {self.out_dir}", f"Summary saved ({len(names)} images) → {self.out_dir}"))
        if messagebox.askyesno(T("완료", "Done"), T(f"전체 요약을 저장했습니다.\n{self.out_dir}\n\n결과 폴더를 열까요?",
                                                  f"Summary saved.\n{self.out_dir}\n\nOpen the output folder?")):
            os.startfile(self.out_dir)


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    ttk.Style().theme_use("vista" if "vista" in ttk.Style().theme_names() else "clam")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
