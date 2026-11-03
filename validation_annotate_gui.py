"""
검증용 수동 주석 GUI
언어: 메뉴 [언어 / Language] 또는  python validation_annotate_gui.py --lang en  (그림 안의 글자는 항상 영어)
================================================================
평가자가 '원본 사진' 위에 직접 찍는다 (정합 결과를 보지 않은 상태 = 블라인드).
  1) 해부학 표지점: 부위별 목록(validation_landmarks.json)의 점을 차례로 찍는다
  2) 파란 점: 눈으로 보이는 파란 점(Evans blue 누출 지점)의 중심을 하나씩 찍는다

원본에 찍는 이유: 같은 주석을 여러 정합 방법으로 템플릿에 옮겨 공정하게 비교할 수 있고,
정합된(변형된) 그림이나 템플릿 윤곽이 평가자 판단에 영향을 주지 않는다.
평가(validation_eval.py)에서 각 정합 방법으로 템플릿 좌표로 옮겨 비교한다.

조작
  왼쪽 클릭            : 표지점 모드 = 선택한 표지점 찍기 (다음 표지점으로 자동 이동)
                         파란 점 모드 = 파란 점 추가
  Shift + 왼쪽 클릭    : 파란 점 모드에서 '애매한 점'으로 추가 (주황색)
  오른쪽 클릭          : 가장 가까운 점 지우기
  마우스 휠            : 확대/축소 (커서 위치 중심),  툴바의 이동/확대 도구도 사용 가능
  ◀ / ▶ (또는 PageUp / PageDown) : 이전/다음 사진 (자동 저장)
저장: <데이터 폴더>/annotations/<평가자ID>/<분절>__<폴더>__<파일>.json
"""
import warnings

warnings.filterwarnings("ignore", message=r".*cryptography\.hazmat\.decrepit.*")
import os
import sys
import csv
import json
import time
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import limb_mapping as LM
from limb_i18n import T, fig_name, view_en, add_language_menu

HERE = os.path.dirname(os.path.abspath(__file__))
LANDMARK_FILE = os.path.join(HERE, "validation_landmarks.json")
matplotlib.rcParams["font.family"] = ["Arial", "DejaVu Sans"]   # 그림 글자는 항상 영어
matplotlib.rcParams["axes.unicode_minus"] = False

ALL = T("전체", "all")


def load_landmark_defs():
    with open(LANDMARK_FILE, encoding="utf-8") as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def find_images(root):
    """root/C*/<부위 폴더>/사진  ->  [(상대경로, 부위, 분절)]"""
    out = []
    for c in sorted(os.listdir(root)):
        cdir = os.path.join(root, c)
        if not (os.path.isdir(cdir) and c.upper().startswith("C")):
            continue
        for d in sorted(os.listdir(cdir)):
            fd = os.path.join(cdir, d)
            if not os.path.isdir(fd) or d.endswith("_registered"):
                continue
            view = next((v for v, k in LM.VIEWS.items() if k in d or v in d.lower()), None)
            if view is None:
                continue
            for f in LM.list_images(fd):
                out.append((os.path.join(c, d, f), view, c))
    return out


def ann_name(rel):
    return rel.replace("\\", "__").replace("/", "__") + ".json"


class AnnotateApp:
    def __init__(self, root):
        self.root = root
        root.title(T("검증용 수동 주석 (Evans blue mapping validation)", "Manual validation annotation (Evans blue mapping)"))
        add_language_menu(root)
        root.geometry("1400x900")
        self.defs = load_landmark_defs()
        self.data_root = LM.DATA_DIR
        self.items, self.order, self.idx = [], [], 0
        self.ann = None
        self.t_start = time.time()
        self._build()
        root.bind("<Prior>", lambda e: self.go(-1))
        root.bind("<Next>", lambda e: self.go(1))

    # ------------------------------------------------------------------ UI
    def _build(self):
        top = ttk.Frame(self.root)
        top.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Button(top, text=T("데이터 폴더…", "Data folder…"), command=self.choose_root).pack(side="left")
        self.lbl_root = ttk.Label(top, text=self.data_root, width=50)
        self.lbl_root.pack(side="left", padx=4)
        ttk.Label(top, text=T("평가자 ID", "Rater ID")).pack(side="left", padx=(12, 2))
        self.var_rater = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.var_rater, width=10).pack(side="left")
        ttk.Label(top, text=T("부위", "View")).pack(side="left", padx=(12, 2))
        self.var_view = tk.StringVar(value=ALL)
        ttk.Combobox(top, textvariable=self.var_view, width=10, state="readonly",
                     values=[ALL] + list(LM.VIEWS)).pack(side="left")
        self.var_shuffle = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text=T("순서 섞기", "Shuffle order"), variable=self.var_shuffle).pack(side="left", padx=6)
        self.var_hide = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text=T("파일 이름 숨기기(블라인드)", "Hide file names (blind)"), variable=self.var_hide,
                        command=self.redraw).pack(side="left", padx=6)
        ttk.Button(top, text=T("시작 / 이어서 하기", "Start / resume"), command=self.start).pack(side="left", padx=8)
        ttk.Button(top, text=T("CSV 내보내기", "Export CSV"), command=self.export_csv).pack(side="left")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, width=330)
        left.pack(side="left", fill="y", padx=6)
        left.pack_propagate(False)

        self.var_mode = tk.StringVar(value="lm")
        mf = ttk.LabelFrame(left, text=T("주석 종류", "Annotation type"))
        mf.pack(fill="x", pady=4)
        ttk.Radiobutton(mf, text=T("① 해부학 표지점", "① Anatomical landmarks"), variable=self.var_mode, value="lm",
                        command=self.redraw).pack(anchor="w")
        ttk.Radiobutton(mf, text=T("② 파란 점 (Evans blue)", "② Blue spots (Evans blue)"), variable=self.var_mode, value="spot",
                        command=self.redraw).pack(anchor="w")

        lf = ttk.LabelFrame(left, text=T("표지점 (클릭해서 선택)", "Landmarks (click to select)"))
        lf.pack(fill="both", expand=True, pady=4)
        self.lb = tk.Listbox(lf, height=14, exportselection=False, font=("Malgun Gothic", 9))
        self.lb.pack(fill="both", expand=True)
        self.lb.bind("<<ListboxSelect>>", lambda e: self.redraw())
        bf = ttk.Frame(lf)
        bf.pack(fill="x")
        ttk.Button(bf, text=T("보이지 않음", "Not visible"), command=self.mark_invisible).pack(side="left", expand=True, fill="x")
        ttk.Button(bf, text=T("선택 표지점 지우기", "Clear selected"), command=self.clear_lm).pack(side="left", expand=True, fill="x")

        sf = ttk.LabelFrame(left, text=T("파란 점", "Blue spots"))
        sf.pack(fill="x", pady=4)
        self.lbl_spots = ttk.Label(sf, text=T("0개", "0"))
        self.lbl_spots.pack(anchor="w")
        ttk.Button(sf, text=T("파란 점 모두 지우기", "Clear all spots"), command=self.clear_spots).pack(fill="x")
        self.var_done_spot = tk.BooleanVar(value=False)
        ttk.Checkbutton(sf, text=T("이 사진의 파란 점 확인 완료 (0개여도 체크)", "Spots completed (check even if none)"),
                        variable=self.var_done_spot, command=self.touch).pack(anchor="w")

        nf = ttk.Frame(left)
        nf.pack(fill="x", pady=6)
        ttk.Button(nf, text=T("◀ 이전", "◀ Prev"), command=lambda: self.go(-1)).pack(side="left", expand=True, fill="x")
        ttk.Button(nf, text=T("저장 후 다음 ▶", "Save & next ▶"), command=lambda: self.go(1)).pack(side="left", expand=True, fill="x")
        self.lbl_prog = ttk.Label(left, text="", wraplength=310)
        self.lbl_prog.pack(anchor="w")
        ttk.Label(left, wraplength=310, foreground="#555", justify="left", text=T(
            "왼쪽 클릭: 표지점/파란 점 찍기\nShift+클릭: 애매한 파란 점\n오른쪽 클릭: 가까운 점 지우기\n"
            "휠: 확대/축소,  PageUp/PageDown: 이전/다음\n"
            "표지점이 사진에서 안 보이거나 잘렸으면 [보이지 않음]",
            "Left click: place landmark / spot\nShift+click: uncertain spot\nRight click: delete nearest point\n"
            "Wheel: zoom,  PageUp/PageDown: previous/next\n"
            "Use [Not visible] if a landmark is hidden or cut off")).pack(anchor="w", pady=6)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(9, 7))
        self.ax = self.fig.add_axes([0, 0, 1, 0.96])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

    # ------------------------------------------------------------------ 데이터
    def choose_root(self):
        d = filedialog.askdirectory(initialdir=self.data_root, title=T("C5~C8 폴더가 있는 데이터 폴더", "Data folder containing the C5–C8 folders"))
        if d:
            self.data_root = d
            self.lbl_root.config(text=d)

    def rater_dir(self):
        return os.path.join(self.data_root, "annotations", self.var_rater.get().strip())

    def start(self):
        rater = self.var_rater.get().strip()
        if not rater:
            messagebox.showwarning(T("평가자 ID", "Rater ID"), T("평가자 ID를 입력하세요 (예: R1, R2).", "Enter a rater ID (e.g. R1, R2)."))
            return
        items = find_images(self.data_root)
        if self.var_view.get() != ALL:
            items = [it for it in items if it[1] == self.var_view.get()]
        if not items:
            messagebox.showwarning(T("사진 없음", "No photographs"), T("C5~C8/<부위> 폴더에서 사진을 찾지 못했습니다.", "No photographs found in C5–C8/<view> folders."))
            return
        self.items = items
        self.order = list(range(len(items)))
        if self.var_shuffle.get():
            random.Random(rater).shuffle(self.order)       # 평가자마다 다르지만 다시 열면 같은 순서
        os.makedirs(self.rater_dir(), exist_ok=True)
        # 처음 안 한 사진부터
        self.idx = next((k for k, i in enumerate(self.order)
                         if not os.path.exists(os.path.join(self.rater_dir(), ann_name(items[i][0])))), 0)
        self.load()

    def cur(self):
        return self.items[self.order[self.idx]]

    def load(self):
        rel, view, seg = self.cur()
        self.rgba = LM.load_rgba(os.path.join(self.data_root, rel))
        names = self.defs.get(view, [])
        p = os.path.join(self.rater_dir(), ann_name(rel))
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                self.ann = json.load(f)
            for n in names:
                self.ann["landmarks"].setdefault(n, None)
        else:
            self.ann = dict(image=rel.replace("\\", "/"), view=view, segment=seg,
                            annotator=self.var_rater.get().strip(),
                            image_size=[int(self.rgba.shape[1]), int(self.rgba.shape[0])],
                            landmarks={n: None for n in names}, invisible=[], spots=[],
                            spots_done=False, time_spent_s=0.0)
        self.var_done_spot.set(bool(self.ann.get("spots_done")))
        self.t_start = time.time()
        self.fill_list()
        sel = next((k for k, n in enumerate(names)
                    if self.ann["landmarks"].get(n) is None and n not in self.ann["invisible"]), 0)
        self.lb.selection_clear(0, "end")
        if names:
            self.lb.selection_set(sel)
        self.ax.clear()
        self.img_artist = self.ax.imshow(LM.on_white(self.rgba))
        self.ax.axis("off")
        self.redraw(reset=True)

    def save(self):
        if self.ann is None:
            return
        self.ann["time_spent_s"] = round(self.ann.get("time_spent_s", 0) + time.time() - self.t_start, 1)
        self.t_start = time.time()
        self.ann["spots_done"] = bool(self.var_done_spot.get())
        self.ann["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.rater_dir(), ann_name(self.cur()[0])), "w", encoding="utf-8") as f:
            json.dump(self.ann, f, ensure_ascii=False, indent=1)

    def go(self, step):
        if not self.items:
            return
        self.save()
        k = self.idx + step
        if k >= len(self.order):
            n_done = sum(os.path.exists(os.path.join(self.rater_dir(), ann_name(it[0]))) for it in self.items)
            messagebox.showinfo(T("끝", "Finished"), T(f"마지막 사진입니다. 저장된 사진 {n_done}/{len(self.items)}장.\n"
                                                    f"[CSV 내보내기]로 표 파일을 만들 수 있습니다.",
                                                    f"This is the last photograph. Saved {n_done}/{len(self.items)}.\n"
                                                    f"Use [Export CSV] to create the tables."))
            return
        self.idx = max(0, k)
        self.load()

    def touch(self):
        if self.ann is not None:
            self.ann["spots_done"] = bool(self.var_done_spot.get())

    # ------------------------------------------------------------------ 편집
    def names(self):
        return self.defs.get(self.cur()[1], []) if self.items else []

    def sel_name(self):
        s = self.lb.curselection()
        return self.names()[s[0]] if s else None

    def fill_list(self):
        self.lb.delete(0, "end")
        for i, n in enumerate(self.names()):
            st = "✔" if self.ann["landmarks"].get(n) is not None else (T("✖ 안 보임", "✖ not visible") if n in self.ann["invisible"] else "·")
            self.lb.insert("end", f"{i + 1:2d}. [{st}] {n}")

    def advance(self):
        names = self.names()
        s = self.lb.curselection()
        start = s[0] if s else 0
        for k in list(range(start + 1, len(names))) + list(range(0, start + 1)):
            n = names[k]
            if self.ann["landmarks"].get(n) is None and n not in self.ann["invisible"]:
                self.lb.selection_clear(0, "end")
                self.lb.selection_set(k)
                return

    def mark_invisible(self):
        n = self.sel_name()
        if n is None:
            return
        self.ann["landmarks"][n] = None
        if n not in self.ann["invisible"]:
            self.ann["invisible"].append(n)
        self.after_edit(advance=True)

    def clear_lm(self):
        n = self.sel_name()
        if n is None:
            return
        self.ann["landmarks"][n] = None
        if n in self.ann["invisible"]:
            self.ann["invisible"].remove(n)
        self.after_edit()

    def clear_spots(self):
        if self.ann and self.ann["spots"] and messagebox.askyesno(T("확인", "Confirm"), T("이 사진의 파란 점을 모두 지울까요?", "Delete all spots of this photograph?")):
            self.ann["spots"] = []
            self.after_edit()

    def after_edit(self, advance=False):
        s = self.lb.curselection()
        self.fill_list()
        if s:
            self.lb.selection_set(s[0])
        if advance:
            self.advance()
        self.redraw()

    def on_click(self, ev):
        if self.ann is None or ev.inaxes != self.ax or ev.xdata is None or self.toolbar.mode:
            return
        x, y = float(ev.xdata), float(ev.ydata)
        tol = self.pick_tol()
        if self.var_mode.get() == "lm":
            if ev.button == 1:
                n = self.sel_name()
                if n is None:
                    return
                self.ann["landmarks"][n] = [round(x, 2), round(y, 2)]
                if n in self.ann["invisible"]:
                    self.ann["invisible"].remove(n)
                self.after_edit(advance=True)
            elif ev.button == 3:
                pts = [(n, p) for n, p in self.ann["landmarks"].items() if p is not None]
                if pts:
                    d = [np.hypot(p[0] - x, p[1] - y) for _, p in pts]
                    if min(d) < tol:
                        self.ann["landmarks"][pts[int(np.argmin(d))][0]] = None
                        self.after_edit()
        else:
            if ev.button == 1:
                unc = "shift" in (ev.key or "")
                self.ann["spots"].append([round(x, 2), round(y, 2), int(unc)])
                self.var_done_spot.set(True)
                self.after_edit()
            elif ev.button == 3 and self.ann["spots"]:
                d = [np.hypot(p[0] - x, p[1] - y) for p in self.ann["spots"]]
                if min(d) < tol:
                    self.ann["spots"].pop(int(np.argmin(d)))
                    self.after_edit()

    def pick_tol(self):
        x0, x1 = self.ax.get_xlim()
        return abs(x1 - x0) / 60

    def on_scroll(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        f = 0.8 if ev.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = ev.xdata, ev.ydata
        self.ax.set_xlim(cx + (x0 - cx) * f, cx + (x1 - cx) * f)
        self.ax.set_ylim(cy + (y0 - cy) * f, cy + (y1 - cy) * f)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ 그리기
    def redraw(self, reset=False):
        if self.ann is None:
            return
        lim = None if reset else (self.ax.get_xlim(), self.ax.get_ylim())
        for a in list(self.ax.lines) + list(self.ax.texts):
            a.remove()
        mode = self.var_mode.get()
        sel = self.sel_name()
        for i, n in enumerate(self.names()):
            p = self.ann["landmarks"].get(n)
            if p is None:
                continue
            on = mode == "lm"
            self.ax.plot(*p, "o", ms=9 if n == sel else 7, mfc="yellow" if on else "none",
                         mec="red" if n == sel else "k", mew=1.5, alpha=1 if on else 0.5)
            self.ax.text(p[0] + 6, p[1] - 6, str(i + 1), fontsize=9, color="k",
                         alpha=1 if on else 0.5, fontweight="bold")
        for x, y, u in self.ann["spots"]:
            on = mode == "spot"
            self.ax.plot(x, y, "o", ms=8, mfc="none", mec="orange" if u else "red", mew=1.6,
                         alpha=1 if on else 0.4)
        n_s, n_u = len(self.ann['spots']), sum(s[2] for s in self.ann['spots'])
        self.lbl_spots.config(text=T(f"{n_s}개 (애매 {n_u}개)", f"{n_s} ({n_u} uncertain)"))
        rel, view, seg = self.cur()
        name = f"#{self.order[self.idx] + 1:03d}" if self.var_hide.get() else fig_name(rel)
        n_lm = sum(p is not None for p in self.ann["landmarks"].values())
        self.ax.set_title(f"{name}   [{view_en(view)}]   landmarks {n_lm}/{len(self.names())}", fontsize=10)
        self.lbl_prog.config(text=T(f"진행: {self.idx + 1} / {len(self.order)}", f"Progress: {self.idx + 1} / {len(self.order)}"))
        if lim is not None:
            self.ax.set_xlim(lim[0])
            self.ax.set_ylim(lim[1])
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ 내보내기
    def export_csv(self):
        base = os.path.join(self.data_root, "annotations")
        if not os.path.isdir(base):
            messagebox.showwarning(T("없음", "Not found"), T("annotations 폴더가 없습니다.", "No annotations folder."))
            return
        n = export_all(base)
        messagebox.showinfo(T("내보내기", "Export"), T(f"{n}개 주석 파일 -> {base}\\landmarks.csv, spots.csv",
                                                    f"{n} annotation files -> {base}\\landmarks.csv, spots.csv"))


def load_annotations(base):
    """annotations/<평가자>/*.json 전부 -> 목록"""
    out = []
    for r in sorted(os.listdir(base)):
        d = os.path.join(base, r)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    a = json.load(fh)
                a["annotator"] = r
                out.append(a)
    return out


def export_all(base):
    anns = load_annotations(base)
    with open(os.path.join(base, "landmarks.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["annotator", "image", "view", "segment", "landmark", "x", "y", "status"])
        for a in anns:
            for n, p in a["landmarks"].items():
                st = "ok" if p is not None else ("invisible" if n in a.get("invisible", []) else "missing")
                w.writerow([a["annotator"], a["image"], a["view"], a["segment"], n,
                            p[0] if p else "", p[1] if p else "", st])
    with open(os.path.join(base, "spots.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["annotator", "image", "view", "segment", "spot_id", "x", "y", "uncertain", "spots_done"])
        for a in anns:
            for i, (x, y, u) in enumerate(a["spots"]):
                w.writerow([a["annotator"], a["image"], a["view"], a["segment"], i + 1, x, y, u, a.get("spots_done")])
    return len(anns)


def main():
    root = tk.Tk()
    AnnotateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
