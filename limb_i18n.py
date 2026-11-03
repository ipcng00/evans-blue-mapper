"""GUI 언어 설정 (한국어 / English)

- 메뉴·버튼·설명·메시지: 선택한 언어 (T(ko, en))
- 그림 안의 글자(제목, 축 레이블, 범례, 저장되는 그림): 언어와 관계없이 항상 영어
언어 선택 순서: 명령줄 --lang ko|en  >  환경변수 LIMB_LANG  >  저장된 설정(~/.limb_mapping_lang)  >  ko
"""
import os
import re
import sys

_CFG = os.path.join(os.path.expanduser("~"), ".limb_mapping_lang")
VIEW_EN = {"medial": "Medial (ulnar)", "lateral": "Lateral (radial)", "dorsal": "Dorsal", "palmar": "Palmar"}


def _initial():
    for i, a in enumerate(sys.argv):
        if a == "--lang" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--lang="):
            return a.split("=", 1)[1]
    if os.environ.get("LIMB_LANG"):
        return os.environ["LIMB_LANG"]
    try:
        with open(_CFG, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "ko"


LANG = "en" if _initial().lower().startswith("en") else "ko"


def T(ko, en):
    """화면 글자: 현재 언어로"""
    return ko if LANG == "ko" else en


def save_lang(lang):
    try:
        with open(_CFG, "w", encoding="utf-8") as f:
            f.write(lang)
    except OSError:
        pass


def argv_without_lang():
    out, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--lang":
            skip = True
            continue
        if not a.startswith("--lang="):
            out.append(a)
    return out


def restart(lang):
    """설정을 저장하고 같은 프로그램을 새 언어로 다시 실행 (호출한 쪽에서 창을 닫는다)"""
    import subprocess
    save_lang(lang)
    subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0])] + argv_without_lang() + ["--lang", lang])


def add_language_menu(root, on_before_restart=None):
    """창 메뉴에 '언어 / Language' 추가. 선택하면 확인 후 다시 실행한다."""
    import tkinter as tk
    from tkinter import messagebox
    menubar = root.nametowidget(root["menu"]) if root["menu"] else tk.Menu(root)
    m = tk.Menu(menubar, tearoff=0)
    var = tk.StringVar(value=LANG)

    def choose():
        new = var.get()
        if new == LANG:
            return
        if not messagebox.askyesno(T("언어 변경", "Change language"),
                                   T("프로그램을 다시 시작합니다. 저장하지 않은 작업은 사라집니다. 계속할까요?",
                                     "The program will restart. Unsaved work will be lost. Continue?")):
            var.set(LANG)
            return
        if on_before_restart:
            on_before_restart()
        restart(new)
        root.destroy()

    m.add_radiobutton(label="한국어", value="ko", variable=var, command=choose)
    m.add_radiobutton(label="English", value="en", variable=var, command=choose)
    menubar.add_cascade(label="언어 / Language", menu=m)
    root.config(menu=menubar)
    return menubar


def fig_name(name):
    """그림 제목에 넣을 파일 이름: 영어만 남긴다 (예: '그림31.png' -> 'img31.png')"""
    s = name.replace("그림", "img")
    s = re.sub(r"[^\x00-\x7f]+", "", s)
    return s if s.strip(" ._-") else "image"


def view_en(view_or_tpl):
    """그림용 부위 이름 (영어)"""
    if isinstance(view_or_tpl, dict):
        v = view_or_tpl.get("view", "")
        return view_or_tpl.get("label_en") or VIEW_EN.get(v) or fig_name(view_or_tpl.get("label", v))
    return VIEW_EN.get(view_or_tpl, fig_name(view_or_tpl))
