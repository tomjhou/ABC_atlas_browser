#!/usr/bin/env python3
"""
jpx_aligner.py

Adjust the alignment of individual sections in a NeuroInfo serial section
assembler stack (reconstruction.jpx), then export a new JPX that NeuroInfo can
open. NeuroInfo itself can't revise an alignment; this tool can, as often as
needed: every exported JPX remembers its edits and can be reopened here.

Open either a NeuroInfo reconstruction.jpx (with reconstruction.xml and
CaseFile.H5 beside it, used to recover NeuroInfo's alignment) or a JPX that
this tool exported. Sections are re-rendered from their original section JP2
files, so edits never stack up resampling or compression losses.

Mouse (in the image):
  drag                move the current section
  Shift+drag, or right-drag   rotate it about its center
  Ctrl+drag, or middle-drag   pan the view;  wheel: zoom
Keys:
  arrows              nudge 5 um (Shift: 50 um)
  [  ]                rotate +0.1 / -0.1 deg (Shift: 1 deg)
  F                   flip left-right
  Page Up / Page Down previous / next section
  Ctrl+Z              undo;  Home: fit view;  Ctrl+S: export

Brightness controls only change the display; exported pixel values always
come from the original section images.

Requires the libraries neuroinfo_stack.py needs, plus Pillow (installed
automatically if missing).
"""

import copy
import hashlib
import math
import os
import queue
import sys
import threading
import time

import neuroinfo_stack as ns

ns.ensure_packages({"PIL": "pillow"})

import tkinter as tk  # noqa: E402
from tkinter import ttk, filedialog, messagebox  # noqa: E402

import cv2  # noqa: E402
import imagecodecs  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

PREVIEW_MAX_PX = 2500   # longest side of the cached preview of each section
CACHE_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                         "JhouLab", "jpx_aligner_cache")
NUDGE_UM, NUDGE_BIG_UM = 5.0, 50.0
ROT_STEP, ROT_BIG_STEP = 0.1, 1.0


##################################################
#
#    Previews
#
##################################################

def _cache_path(src):
    st = os.stat(src)
    key = f"{os.path.abspath(src).lower()}|{st.st_size}|{st.st_mtime_ns}|{PREVIEW_MAX_PX}"
    return os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".npz")


def preview_is_cached(src):
    try:
        return os.path.isfile(_cache_path(src))
    except OSError:
        return False


def load_preview(src):
    """Downsampled copy of a section image: (image HxWxC uint16, kx, ky) where
    kx, ky are source pixels per preview pixel. Cached on disk."""
    path = _cache_path(src)
    if os.path.isfile(path):
        try:
            with np.load(path) as d:
                return d["img"], float(d["kx"]), float(d["ky"])
        except Exception:
            pass  # rebuild a damaged cache file
    full = imagecodecs.imread(src)
    if full.ndim == 2:
        full = full[..., None]
    h, w = full.shape[:2]
    k = max(1, math.ceil(max(h, w) / PREVIEW_MAX_PX))
    pw, ph = max(1, round(w / k)), max(1, round(h / k))
    img = np.stack([cv2.resize(full[..., c], (pw, ph), interpolation=cv2.INTER_AREA)
                    for c in range(full.shape[2])], -1)
    del full
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + ".partial.npz"
    np.savez(tmp, img=img, kx=w / pw, ky=h / ph)
    os.replace(tmp, path)
    return img, w / pw, h / ph


def human_gb(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


##################################################
#
#    Application
#
##################################################

def shade(color, weight):
    """`color` ('#rrggbb') dimmed toward black: weight 1 = as it is, 0 = black (invisible on the black canvas)."""
    weight = min(max(float(weight), 0.0), 1.0)
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (round(r * weight), round(g * weight), round(b * weight))


class Aligner:
    def __init__(self, path=None):
        self.root = tk.Tk()
        self.root.title("JPX aligner")
        self.root.geometry("1500x950")
        self.stack = None
        self.previews = {}          # section index -> (img, kx, ky)
        self.cur = 0
        self.undo = []              # steps: each a list of (section index, Edit before the change)
        self.drag = None
        self.view = None            # (scale view px per canvas px, ox, oy)
        self.fitted = True          # the view is the fit-to-window one (not panned or zoomed by hand since)
        self.last_size = None       # canvas size at the last redraw, to scale the view when the window is resized
        self.opening = False        # a stack is being opened
        self.loading_previews = False   # section images are still being loaded in the background
        self.levels = []            # per channel [lo, hi] (display only)
        self.levels_set = False
        self.busy = False
        self.msgs = queue.Queue()
        self.stop_event = threading.Event()
        self.photo = None

        self.var_ref = tk.StringVar(self.root, "previous")
        self.var_mode = tk.StringVar(self.root, "blend")
        self.var_opacity = tk.DoubleVar(self.root, 0.5)
        self.var_rot = tk.StringVar(self.root, "0")
        self.var_dx = tk.StringVar(self.root, "0")
        self.var_dy = tk.StringVar(self.root, "0")
        self.var_quality = tk.StringVar(self.root, f"{ns.DEFAULT_QUALITY_DB:g}")
        self.var_rerender = tk.BooleanVar(self.root, True)   # every image is written again, so each JPX file has all of its images
        self.var_crosshair = tk.BooleanVar(self.root, True)
        self.var_follow = tk.BooleanVar(self.root, False)   # an edit also moves the sections after it
        self.var_status = tk.StringVar(self.root, "Open a reconstruction.jpx to start (File > Open).")
        self.chan_vars = []         # per channel (visible BooleanVar, lo DoubleVar, hi DoubleVar)

        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.after(100, self.poll)
        if path:
            self.root.after(200, lambda: self.open_path(path))

    # ---------------------------------------------------------------- layout

    def build(self):
        root = self.root
        menu = tk.Menu(root)
        fm = tk.Menu(menu, tearoff=0)
        fm.add_command(label="Open...", command=self.open_dialog, accelerator="Ctrl+O")
        fm.add_command(label="Export JPX...", command=self.export_dialog, accelerator="Ctrl+S")
        fm.add_separator()
        fm.add_command(label="Exit", command=self.quit)
        menu.add_cascade(label="File", menu=fm)
        em = tk.Menu(menu, tearoff=0)
        em.add_command(label="Undo", command=self.do_undo, accelerator="Ctrl+Z")
        em.add_command(label="Reset this section", command=self.reset_section)
        em.add_command(label="Reset all sections", command=self.reset_all)
        menu.add_cascade(label="Edit", menu=em)
        root.config(menu=menu)

        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, padding=5)
        left.grid(row=0, column=0, sticky="ns")
        ttk.Label(left, text="Sections").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=30, exportselection=False, activestyle="none",
                                  font=("Segoe UI", 10))
        self.listbox.pack(fill=tk.Y, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)
        ttk.Label(left, text="* changed since opened\n~ edited  |  F flipped", foreground="gray30",
                  justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

        self.canvas = tk.Canvas(main, bg="black", highlightthickness=0, cursor="fleur")
        self.canvas.grid(row=0, column=1, sticky="nsew")
        c = self.canvas
        c.bind("<Configure>", self.on_resize)
        c.bind("<ButtonPress-1>", lambda e: self.press(e, "rotate" if e.state & 0x1 else
                                                       "pan" if e.state & 0x4 else "move"))
        c.bind("<ButtonPress-3>", lambda e: self.press(e, "rotate"))
        c.bind("<ButtonPress-2>", lambda e: self.press(e, "pan"))
        for b in ("<B1-Motion>", "<B2-Motion>", "<B3-Motion>"):
            c.bind(b, self.motion)
        for b in ("<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>"):
            c.bind(b, self.release)
        c.bind("<MouseWheel>", self.wheel)

        right = ttk.Frame(main, padding=5)
        right.grid(row=0, column=2, sticky="ns")
        self.build_right(right)

        bottom = ttk.Frame(root, padding=(5, 2))
        bottom.pack(fill=tk.X)
        self.progress = ttk.Progressbar(bottom, length=220, mode="determinate")
        self.progress.pack(side=tk.RIGHT)
        self.button_cancel = ttk.Button(bottom, text="Cancel", command=self.stop_event.set, state="disabled")
        self.button_cancel.pack(side=tk.RIGHT, padx=5)
        ttk.Label(bottom, textvariable=self.var_status, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        for seq, fn in (("<Control-o>", self.open_dialog), ("<Control-s>", self.export_dialog),
                        ("<Control-z>", self.do_undo)):
            root.bind(seq, lambda e, f=fn: f())
        root.bind("<Key>", self.key)

    def build_right(self, right):
        f = ttk.LabelFrame(right, text="Current section", padding=6)
        f.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(f, text="Edits also move the following sections", variable=self.var_follow).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        self.label_section = ttk.Label(f, text="-", font=("Segoe UI", 10, "bold"))
        self.label_section.grid(row=1, column=0, columnspan=4, sticky="w")
        for r, (text, var) in enumerate((("Rotation (deg, clockwise)", self.var_rot), ("Shift x (um)", self.var_dx),
                                         ("Shift y (um)", self.var_dy)), start=2):
            ttk.Label(f, text=text).grid(row=r, column=0, sticky="e", pady=1)
            e = ttk.Entry(f, textvariable=var, width=10)
            e.grid(row=r, column=1, columnspan=2, sticky="w", padx=4)
            e.bind("<Return>", lambda ev: self.apply_entries())
        ttk.Button(f, text="Apply", command=self.apply_entries).grid(row=2, column=3, rowspan=3, sticky="ns")
        bf = ttk.Frame(f)
        bf.grid(row=5, column=0, columnspan=4, pady=(6, 0))
        for text, deg in (("-1°", -1), ("-0.1°", -0.1), ("+0.1°", 0.1), ("+1°", 1)):   # (+ is clockwise)
            ttk.Button(bf, text=text, width=6, command=lambda d=deg: self.rotate_by(-d)).pack(side=tk.LEFT)
        bf2 = ttk.Frame(f)
        bf2.grid(row=6, column=0, columnspan=4, pady=(4, 0))
        ttk.Button(bf2, text="Flip left-right (F)", command=self.flip).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf2, text="Reset", command=self.reset_section).pack(side=tk.LEFT, padx=2)
        self.label_detail = ttk.Label(f, text="", foreground="gray30", justify=tk.LEFT)
        self.label_detail.grid(row=7, column=0, columnspan=4, sticky="w", pady=(6, 0))

        f = ttk.LabelFrame(right, text="Compare with", padding=6)
        f.pack(fill=tk.X, pady=(0, 6))
        for text, val in (("Previous section", "previous"), ("Next section", "next"), ("Nothing", "none")):
            ttk.Radiobutton(f, text=text, value=val, variable=self.var_ref,
                            command=self.redraw).pack(anchor="w")
        ttk.Separator(f).pack(fill=tk.X, pady=4)
        ttk.Radiobutton(f, text="Blend (true colors)", value="blend", variable=self.var_mode,
                        command=self.redraw).pack(anchor="w")
        ttk.Radiobutton(f, text="Overlay (this = magenta, other = cyan)", value="overlay",
                        variable=self.var_mode, command=self.redraw).pack(anchor="w")
        ttk.Label(f, text="Opacity of this section").pack(anchor="w", pady=(4, 0))
        ttk.Scale(f, from_=0, to=1, variable=self.var_opacity,
                  command=lambda v: self.redraw()).pack(fill=tk.X)

        f = ttk.LabelFrame(right, text="View", padding=6)
        f.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(f, text="Crosshair at the centre of the view", variable=self.var_crosshair,
                        command=self.redraw).pack(anchor="w")

        self.chan_frame = ttk.LabelFrame(right, text="Display brightness (view only)", padding=6)
        self.chan_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(self.chan_frame, text="(open a stack)").pack()

        f = ttk.LabelFrame(right, text="Export", padding=6)
        f.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(f, text="Quality (dB, max ~96)").grid(row=0, column=0, sticky="e")   # 16-bit frames: nothing above ~96 dB changes the file
        ttk.Entry(f, textvariable=self.var_quality, width=6).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Checkbutton(f, text="Re-render unchanged sections too", variable=self.var_rerender).grid(
            row=1, column=0, columnspan=2, sticky="w")
        ttk.Button(f, text="Export JPX...", command=self.export_dialog).grid(row=2, column=0, columnspan=2,
                                                                              pady=(6, 0), sticky="ew")

        help_text = ("drag: move   Shift/right-drag: rotate\n"
                     "Ctrl/middle-drag: pan   wheel: zoom\n"
                     "arrows: nudge (image focused)  [ ]: rotate  F: flip\n"
                     "PgUp/PgDn: section   Home: fit view")
        ttk.Label(right, text=help_text, foreground="gray30", justify=tk.LEFT).pack(anchor="w")

    def build_channel_controls(self):
        for w in self.chan_frame.winfo_children():
            w.destroy()
        self.chan_vars = []
        for i, ch in enumerate(self.stack.channel_info):
            lo, hi = self.levels[i]
            vis = tk.BooleanVar(self.root, True)
            vlo = tk.DoubleVar(self.root, lo)
            vhi = tk.DoubleVar(self.root, hi)
            self.chan_vars.append((vis, vlo, vhi))
            hexcol = "#%02x%02x%02x" % tuple(ch.color)
            row = ttk.Frame(self.chan_frame)
            row.pack(fill=tk.X, pady=(4 if i else 0, 0))
            tk.Label(row, bg=hexcol, width=2).pack(side=tk.LEFT)
            ttk.Checkbutton(row, text=ch.name, variable=vis, command=self.redraw).pack(side=tk.LEFT, padx=4)
            for text, var in (("black", vlo), ("white", vhi)):
                r = ttk.Frame(self.chan_frame)
                r.pack(fill=tk.X)
                ttk.Label(r, text=text, width=6).pack(side=tk.LEFT)
                ttk.Scale(r, from_=0, to=65535, variable=var,
                          command=lambda v: self.redraw()).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ---------------------------------------------------------------- busy cursor

    def update_cursor(self):
        """The hourglass over the whole window (the image canvas has a cursor of its own, so it is set too) while a stack is
        opening, its section images are loading or an export is running; the normal cursors otherwise."""
        busy = self.opening or self.loading_previews or self.busy
        self.root.config(cursor="watch" if busy else "")
        self.canvas.config(cursor="watch" if busy else "fleur")

    # ---------------------------------------------------------------- opening

    def confirm_discard(self):
        if self.stack is not None and self.stack.dirty:
            return messagebox.askyesno("Unsaved changes",
                                       "Some sections have changes that haven't been exported.\n"
                                       "Discard them?", parent=self.root)
        return True

    def open_dialog(self):
        if self.busy or not self.confirm_discard():
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Open a reconstruction (NeuroInfo or exported by this tool)",
            filetypes=[("JPX stacks", "*.jpx"), ("All files", "*.*")],
            initialdir=self.stack.folder if self.stack else None)
        if path:
            self.open_path(path)

    def open_path(self, path):
        self.opening = True
        self.update_cursor()
        self.root.update()
        try:
            stack = ns.open_stack(path)
        except Exception as e:
            self.opening = False
            self.update_cursor()
            messagebox.showerror("Can't open", str(e), parent=self.root)
            return
        self.opening = False
        self.update_cursor()
        missing = [s.name for s in stack.sections if not s.source_path]
        if missing:
            messagebox.showwarning(
                "Source images not found",
                "These section images weren't found, so they can't be shown or re-rendered "
                "(their current images in the JPX are kept as they are):\n\n" + "\n".join(missing),
                parent=self.root)
        need = [s for s in stack.sections if s.source_path and not preview_is_cached(s.source_path)]
        online = [s for s in need if ns.is_online_only(s.source_path)]
        if online:
            size = sum(os.path.getsize(s.source_path) for s in online)
            if not messagebox.askyesno(
                    "Download from OneDrive?",
                    f"{len(online)} section image(s) ({human_gb(size)}) are online-only in OneDrive and "
                    "must be downloaded to build previews. Continue?", parent=self.root):
                return
        self.stack = stack
        self.previews = {}
        self.undo = []
        self.cur = 0
        self.view = None
        self.levels = [[0.0, 65535.0] for _ in range(stack.channels)]
        self.levels_set = False
        self.build_channel_controls()
        self.root.title(f"JPX aligner - {os.path.basename(path)}")
        self.refresh_list()
        self.update_controls()
        self.redraw(fit_if_needed=True)
        self.canvas.focus_set()   # so the arrow keys nudge the image right away (click the list to move through sections)
        self.start_preview_loading()

    def start_preview_loading(self):
        stack = self.stack
        todo = [s for s in stack.sections if s.source_path]
        self.progress.config(maximum=max(len(todo), 1), value=0)
        self.loading_previews = bool(todo)
        self.update_cursor()
        # load the current section and its neighbors first
        todo.sort(key=lambda s: abs(s.index - self.cur))

        def work():
            for n, s in enumerate(todo, 1):
                if self.stack is not stack:
                    return
                cached = preview_is_cached(s.source_path)
                self.msgs.put(("status", f"{'Loading' if cached else 'Building'} preview of {s.name} "
                                         f"({n} of {len(todo)})..."))
                try:
                    pv = load_preview(s.source_path)
                except Exception as e:
                    self.msgs.put(("status", f"Couldn't read {s.name}: {e}"))
                    continue
                self.msgs.put(("preview", (stack, s.index, pv, n)))
            self.msgs.put(("loaded", (stack, f"Ready. {len(todo)} sections loaded.")))

        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "status":
                    self.var_status.set(payload)
                elif kind == "loaded":
                    stack, text = payload
                    if stack is self.stack:   # (a stack opened since has its own loading under way)
                        self.loading_previews = False
                        self.update_cursor()
                        self.var_status.set(text)
                elif kind == "preview":
                    stack, idx, pv, n = payload
                    if stack is self.stack:
                        self.previews[idx] = pv
                        self.progress.config(value=n)
                        if not self.levels_set:
                            self.auto_levels(pv[0])
                        self.redraw()
                elif kind == "progress":
                    i, n, text = payload
                    self.progress.config(maximum=n, value=i)
                    self.var_status.set(text)
                elif kind == "export_done":
                    self.export_finished(*payload)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def auto_levels(self, img):
        for c in range(min(img.shape[2], len(self.levels))):
            v = img[..., c][img[..., c] > 0]
            if v.size:
                self.levels[c] = [float(np.percentile(v, 1)), float(np.percentile(v, 99.7))]
                self.chan_vars[c][1].set(self.levels[c][0])
                self.chan_vars[c][2].set(self.levels[c][1])
        self.levels_set = True

    # ---------------------------------------------------------------- list / controls

    def section_label(self, s):
        marks = ("*" if s.dirty else " ") + ("~" if not s.edit.is_identity() else " ") + \
                ("F" if s.edit.flip else " ")
        return f"{marks}  {s.index + 1}. {os.path.splitext(s.name)[0]}"

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        if not self.stack:
            return
        for s in self.stack.sections:
            self.listbox.insert(tk.END, self.section_label(s))
            if not s.source_path:
                self.listbox.itemconfig(tk.END, foreground="gray60")
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.cur)
        self.listbox.see(self.cur)

    def refresh_list_item(self, i):
        sel = self.listbox.curselection()
        self.listbox.delete(i)
        self.listbox.insert(i, self.section_label(self.stack.sections[i]))
        if sel:
            self.listbox.selection_set(sel[0])

    def on_list_select(self, _event=None):
        sel = self.listbox.curselection()
        if sel and self.stack:
            self.select(sel[0])

    def select(self, i):
        if not self.stack:
            return
        i = max(0, min(len(self.stack.sections) - 1, i))
        self.cur = i
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(i)
        self.listbox.see(i)
        self.update_controls()
        self.redraw()

    def update_controls(self):
        if not self.stack:
            return
        s = self.stack.sections[self.cur]
        self.label_section.config(text=f"{self.cur + 1}. {s.name}")
        e = s.edit
        self.var_rot.set(f"{-e.rotation_deg:.3f}")   # shown clockwise-positive; the Edit itself stays counterclockwise-positive
        self.var_dx.set(f"{e.dx_um:.1f}")
        self.var_dy.set(f"{e.dy_um:.1f}")
        base = "mirrored" if np.linalg.det(s.R) < 0 else "not mirrored"
        self.label_detail.config(
            text=f"Edit is relative to NeuroInfo's alignment\n(base: {base}).\n"
                 f"Net: rotation {-s.rotation_deg:+.2f}° (clockwise), {'mirrored' if s.mirrored else 'not mirrored'}"
                 + ("" if s.source_path else "\nSource image not found: can't edit."))

    # ---------------------------------------------------------------- editing

    def editable(self):
        if not self.stack or self.busy:
            return None
        s = self.stack.sections[self.cur]
        return s if s.source_path else None

    def followers(self, s):
        """The sections after `s` that can be edited (their source image was found)."""
        return [f for f in self.stack.sections[s.index + 1:] if f.source_path]

    def push_undo(self, s):
        """Remember the edit of `s` (and, with "also move the following sections" on, of the sections after it, which an edit
        of `s` will move too) as one undo step."""
        step = [(s.index, copy.copy(s.edit))]
        if self.var_follow.get():
            step += [(f.index, copy.copy(f.edit)) for f in self.followers(s)]
        self.undo.append(step)
        if len(self.undo) > 500:
            self.undo.pop(0)

    def start_state(self, s):
        """What the sections look like before an edit of `s` begins: its own and its followers' placements."""
        return {"eff": s.effective(),
                "followers": [(f, f.effective()) for f in self.followers(s)] if self.var_follow.get() else []}

    def edit_for_effective(self, sec, R, t):
        """The Edit that gives section `sec` the placement (R, t): inverse of Section.effective()."""
        E = R @ np.linalg.inv(sec.R)
        flip = bool(np.linalg.det(E) < 0)
        pure = E @ ns.FLIP_X if flip else E
        angle = math.degrees(math.atan2(pure[1, 0], pure[0, 0]))
        c = sec.center
        d = t - c - E @ (sec.t - c)
        return ns.Edit(angle, float(d[0]), float(d[1]), flip)

    def set_edit(self, s, edit, start=None):
        """Give section `s` the edit. With "also move the following sections" on, every later section is moved by the same rigid
        change that took `s` from its placement in `start` (the state before the edit began; taken now if not given) to its
        new one, so the sections after it keep their positions relative to it. With it off only `s` changes; the sections'
        transforms are independent of each other, so nothing needs to be undone on the others."""
        if start is None:
            start = self.start_state(s)
        s.edit = edit
        if not start["followers"]:
            return
        R0, t0 = start["eff"]
        R1, t1 = s.effective()
        Rd = R1 @ np.linalg.inv(R0)       # the change of s's placement, as a map of the stack plane: p -> Rd p + td
        td = t1 - Rd @ t0
        for f, (Rf, tf) in start["followers"]:
            f.edit = self.edit_for_effective(f, Rd @ Rf, Rd @ tf + td)
            self.refresh_list_item(f.index)

    def changed(self, s):
        self.refresh_list_item(s.index)
        self.update_controls()
        self.redraw()

    def apply_entries(self):
        s = self.editable()
        if not s:
            return
        try:
            rot_deg, dx, dy = float(self.var_rot.get()), float(self.var_dx.get()), float(self.var_dy.get())
        except ValueError:
            messagebox.showerror("Not a number", "Rotation and shifts must be numbers.", parent=self.root)
            return
        self.push_undo(s)
        self.set_edit(s, ns.Edit(-rot_deg, dx, dy, s.edit.flip))   # (typed clockwise-positive)
        self.changed(s)

    def rotate_by(self, deg):
        s = self.editable()
        if s:
            self.push_undo(s)
            self.set_edit(s, ns.Edit(s.edit.rotation_deg + deg, s.edit.dx_um, s.edit.dy_um, s.edit.flip))
            self.changed(s)

    def nudge(self, dx, dy):
        s = self.editable()
        if s:
            self.push_undo(s)
            self.set_edit(s, ns.Edit(s.edit.rotation_deg, s.edit.dx_um + dx, s.edit.dy_um + dy, s.edit.flip))
            self.changed(s)

    def flip(self):
        """Mirror left-right as currently seen: F*Rot(a)*F^f = Rot(-a)*F^(f+1)."""
        s = self.editable()
        if s:
            self.push_undo(s)
            self.set_edit(s, ns.Edit(-s.edit.rotation_deg, s.edit.dx_um, s.edit.dy_um, not s.edit.flip))
            self.changed(s)

    def reset_section(self):
        s = self.editable()
        if s and not s.edit.is_identity():
            self.push_undo(s)
            self.set_edit(s, ns.Edit())
            self.changed(s)

    def reset_all(self):
        if not self.stack or self.busy:
            return
        if not messagebox.askyesno("Reset all", "Remove the edits from every section?", parent=self.root):
            return
        for s in self.stack.sections:
            if not s.edit.is_identity():
                self.undo.append([(s.index, copy.copy(s.edit))])
                s.edit = ns.Edit()
        self.refresh_list()
        self.update_controls()
        self.redraw()

    def do_undo(self):
        if not self.stack or self.busy or not self.undo:
            return
        step = self.undo.pop()
        for i, e in step:
            self.stack.sections[i].edit = e
            self.refresh_list_item(i)
        i = step[0][0]
        if i != self.cur:
            self.select(i)
        self.changed(self.stack.sections[i])

    # ---------------------------------------------------------------- mouse / keys

    def press(self, event, mode):
        self.canvas.focus_set()
        s = self.editable() if mode != "pan" else None
        if mode != "pan" and not s:
            mode = "pan"
        self.drag = {"mode": mode, "x": event.x, "y": event.y, "view": self.view}
        if s:
            self.push_undo(s)
            self.drag["edit"] = copy.copy(s.edit)
            self.drag["start"] = self.start_state(s)
            if mode == "rotate":
                cx, cy = self.um_to_view(s.center + [s.edit.dx_um, s.edit.dy_um])
                self.drag["pivot"] = (cx, cy)
                self.drag["a0"] = math.atan2(-(event.y - cy), event.x - cx)

    def motion(self, event):
        d = self.drag
        if not d or not self.view:
            return
        if d["mode"] == "pan":
            s, ox, oy = d["view"]
            self.view = (s, ox + event.x - d["x"], oy + event.y - d["y"])
            self.fitted = False
            self.redraw()
            return
        sec = self.stack.sections[self.cur]
        e0 = d["edit"]
        if d["mode"] == "move":
            scale = self.view[0]
            dx_um = (event.x - d["x"]) / scale * self.stack.pixel_um
            dy_um = -(event.y - d["y"]) / scale * self.stack.pixel_um
            self.set_edit(sec, ns.Edit(e0.rotation_deg, e0.dx_um + dx_um, e0.dy_um + dy_um, e0.flip), d["start"])
        else:
            cx, cy = d["pivot"]
            a = math.atan2(-(event.y - cy), event.x - cx)
            self.set_edit(sec, ns.Edit(e0.rotation_deg + math.degrees(a - d["a0"]), e0.dx_um, e0.dy_um, e0.flip), d["start"])
        self.update_controls()
        self.redraw()

    def release(self, _event):
        d, self.drag = self.drag, None
        if d and "edit" in d:
            s = self.stack.sections[self.cur]
            if s.edit.same_as(d["edit"]):
                self.undo.pop()  # a click without a drag
            self.refresh_list_item(s.index)

    def wheel(self, event):
        if not self.view:
            return
        s, ox, oy = self.view
        f = 1.2 ** (event.delta / 120)
        u, v = (event.x - ox) / s, (event.y - oy) / s
        s2 = min(max(s * f, 0.005), 8.0)
        self.view = (s2, event.x - u * s2, event.y - v * s2)
        self.fitted = False
        self.redraw()

    def key(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry)) or not self.stack:
            return
        k = event.keysym
        # The arrow keys nudge the image only while the focus is on the image itself; anywhere else (the section list, say)
        # they are left to the widget that has the focus, so in the list they just change the selection. The list's own
        # Page Up / Page Down / Home / End do the same there, so they must not also move the section from here.
        if k in ("Left", "Right", "Up", "Down") and event.widget is not self.canvas:
            return
        if event.widget is self.listbox and k in ("Prior", "Next", "Home", "End"):
            return
        shift = bool(event.state & 0x1)
        step = NUDGE_BIG_UM if shift else NUDGE_UM
        rstep = ROT_BIG_STEP if shift else ROT_STEP
        if k == "Left":
            self.nudge(-step, 0)
        elif k == "Right":
            self.nudge(step, 0)
        elif k == "Up":
            self.nudge(0, step)
        elif k == "Down":
            self.nudge(0, -step)
        elif k in ("bracketleft", "braceleft"):
            self.rotate_by(rstep)
        elif k in ("bracketright", "braceright"):
            self.rotate_by(-rstep)
        elif k in ("f", "F"):
            self.flip()
        elif k == "Prior":
            self.select(self.cur - 1)
        elif k == "Next":
            self.select(self.cur + 1)
        elif k == "Home":
            self.view = None
            self.redraw(fit_if_needed=True)
        else:
            return
        return "break"

    # ---------------------------------------------------------------- drawing

    def um_to_view(self, p):
        s, ox, oy = self.view
        px = self.stack.pixel_um
        return ox + p[0] / px * s, oy - p[1] / px * s

    def fit_view(self):
        w, h = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        st = self.stack
        s = 0.96 * min(w / st.canvas_w, h / st.canvas_h)
        self.view = (s, (w - st.canvas_w * s) / 2, (h - st.canvas_h * s) / 2)
        self.fitted = True

    def on_resize(self, event):
        """The window was resized: a fitted view is fitted again; one the user has panned or zoomed is scaled about the
        centre of the canvas, so the same part of the image stays at the centre (where the crosshair is)."""
        w, h = event.width, event.height
        old = self.last_size
        self.last_size = (w, h)
        if self.view is not None and old and old != (w, h) and w > 20 and h > 20:
            if self.fitted:
                self.view = None   # redraw fits it to the new size
            else:
                s, ox, oy = self.view
                ratio = min(w / old[0], h / old[1])
                u, v = (old[0] / 2 - ox) / s, (old[1] / 2 - oy) / s   # the image point at the old centre
                self.view = (s * ratio, w / 2 - u * s * ratio, h / 2 - v * s * ratio)
        self.redraw(fit_if_needed=True)

    def warp_section(self, sec, vw, vh):
        """(image vh x vw x C float32 in raw units, coverage mask) of a section in the view."""
        pv = self.previews.get(sec.index)
        if pv is None:
            return None, None
        img, kx, ky = pv
        s, ox, oy = self.view
        A = np.array([[1 / s, 0, -ox / s], [0, 1 / s, -oy / s], [0, 0, 1]])        # view -> canvas px
        R, t = sec.effective()
        M = np.vstack([ns.canvas_to_source_matrix(R, t, self.stack.pixel_um), [0, 0, 1]])  # -> source px
        P = np.array([[1 / kx, 0, 0.5 / kx - 0.5], [0, 1 / ky, 0.5 / ky - 0.5], [0, 0, 1]])  # -> preview px
        T = (P @ M @ A)[:2]
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        out = np.stack([cv2.warpAffine(img[..., c], T, (vw, vh), flags=flags, borderValue=0)
                        for c in range(img.shape[2])], -1).astype(np.float32)
        mask = cv2.warpAffine(np.ones(img.shape[:2], np.uint8), T, (vw, vh),
                              flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0) > 0
        return out, mask

    def colorize(self, img):
        rgb = np.zeros(img.shape[:2] + (3,), np.float32)
        for c, (vis, vlo, vhi) in enumerate(self.chan_vars):
            if c >= img.shape[2] or not vis.get():
                continue
            lo, hi = vlo.get(), max(vhi.get(), vlo.get() + 1)
            v = np.clip((img[..., c] - lo) / (hi - lo), 0, 1)
            rgb += v[..., None] * (np.array(self.stack.channel_info[c].color, np.float32) / 255)
        return np.clip(rgb, 0, 1)

    def luminance(self, img):
        lum = np.zeros(img.shape[:2], np.float32)
        for c, (vis, vlo, vhi) in enumerate(self.chan_vars):
            if c < img.shape[2] and vis.get():
                lo, hi = vlo.get(), max(vhi.get(), vlo.get() + 1)
                lum = np.maximum(lum, np.clip((img[..., c] - lo) / (hi - lo), 0, 1))
        return lum

    def redraw(self, fit_if_needed=False):
        c = self.canvas
        c.delete("all")
        if not self.stack:
            return
        vw, vh = c.winfo_width(), c.winfo_height()
        if vw < 20 or vh < 20:
            return
        if self.view is None:
            if not fit_if_needed:
                return
            self.fit_view()
        secs = self.stack.sections
        cur = secs[self.cur]
        ref = None
        if self.var_ref.get() == "previous" and self.cur > 0:
            ref = secs[self.cur - 1]
        elif self.var_ref.get() == "next" and self.cur < len(secs) - 1:
            ref = secs[self.cur + 1]

        cur_img, cur_m = self.warp_section(cur, vw, vh)
        ref_img, ref_m = self.warp_section(ref, vw, vh) if ref else (None, None)
        alpha = float(self.var_opacity.get())
        if self.var_mode.get() == "overlay":
            out = np.zeros((vh, vw, 3), np.float32)
            if ref_img is not None:
                out += self.luminance(ref_img)[..., None] * np.array([0, 1, 1], np.float32) * (1 - alpha) * 2
            if cur_img is not None:
                out += self.luminance(cur_img)[..., None] * np.array([1, 0, 1], np.float32) * alpha * 2
        else:
            out = np.zeros((vh, vw, 3), np.float32)
            rc = self.colorize(ref_img) if ref_img is not None else None
            cc = self.colorize(cur_img) if cur_img is not None else None
            if rc is not None and cc is not None:
                both = (cur_m & ref_m)[..., None]
                out = np.where(both, rc * (1 - alpha) + cc * alpha,
                               np.where(cur_m[..., None], cc, rc))
            elif cc is not None:
                out = cc
            elif rc is not None:
                out = rc * (1 - alpha)
        rgb = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        if self.var_crosshair.get():
            self.draw_crosshair(rgb)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        c.create_image(0, 0, image=self.photo, anchor="nw")

        # canvas boundary, outlines and pivot
        st = self.stack
        x0, y0 = self.um_to_view((0, 0))
        x1, y1 = self.um_to_view((st.canvas_w * st.pixel_um, -st.canvas_h * st.pixel_um))
        c.create_rectangle(x0, y0, x1, y1, outline="gray50", dash=(4, 4))
        # The outlines (and the pivot cross) fade with their section's opacity: the canvas has no transparency, so each is drawn
        # in its colour dimmed toward the black background by the same weight the image gets (this section: the opacity
        # slider's value, the other one: its complement; with nothing to compare with, this section is at full strength).
        cur_color = shade("#ffe000", alpha if ref is not None else 1.0)
        if ref is not None:
            self.draw_outline(ref, shade("#00e5ff", 1 - alpha), dash=(5, 3))
        self.draw_outline(cur, cur_color)
        px, py = self.um_to_view(cur.center + [cur.edit.dx_um, cur.edit.dy_um])
        c.create_line(px - 8, py, px + 8, py, fill=cur_color)
        c.create_line(px, py - 8, px, py + 8, fill=cur_color)
        if cur.index not in self.previews and cur.source_path:
            c.create_text(vw / 2, 20, text="loading preview...", fill="white")

    def draw_crosshair(self, rgb):
        """One-pixel dashed vertical and horizontal lines meeting at the centre of the canvas, blended into the picture at
        50% white (the canvas cannot draw see-through lines, so they are put into the image itself)."""
        h, w = rgb.shape[:2]
        cx, cy = w // 2, h // 2
        dash = 3   # pixels on, then as many off
        on_y = (np.arange(h) // dash) % 2 == 0
        on_x = (np.arange(w) // dash) % 2 == 0
        rgb[on_y, cx] = (rgb[on_y, cx] // 2 + 128).astype(np.uint8)
        rgb[cy, on_x] = (rgb[cy, on_x] // 2 + 128).astype(np.uint8)

    def draw_outline(self, sec, color, dash=None):
        pts = sec.stack_points
        coords = []
        for p in np.vstack([pts, pts[:1]]):
            coords.extend(self.um_to_view(p))
        self.canvas.create_line(*coords, fill=color, width=2, dash=dash)

    # ---------------------------------------------------------------- export

    def export_dialog(self):
        if not self.stack or self.busy:
            return
        st = self.stack
        try:
            quality = float(self.var_quality.get())
        except ValueError:
            messagebox.showerror("Quality", "Quality must be a number of dB, e.g. 60.", parent=self.root)
            return
        rerender_all = self.var_rerender.get()
        todo = [s for s in st.sections if rerender_all or s.dirty]
        if not any(s.dirty for s in st.sections):   # (also when "re-render unchanged sections too" would still write every image)
            same = ("every image would be re-rendered, but with the same placement as before" if rerender_all
                    else "the images would be copied unchanged")
            if not messagebox.askyesno("Nothing changed",
                                       "No section has been edited since this file was opened, so an export would be a copy "
                                       f"of it ({same}).\n\nExport anyway?", parent=self.root):
                return
        bad = st.out_of_canvas()
        if bad and not messagebox.askyesno(
                "Sections past the canvas edge",
                "These sections extend past the edge of the canvas, and the parts outside will be cut off:\n\n"
                + "\n".join(f"{s.name}: {o:.0f} um" for s, o in bad) + "\n\nExport anyway?", parent=self.root):
            return
        missing = [s.name for s in todo if not s.source_path]
        if missing:
            messagebox.showerror("Source images missing", "Can't re-render:\n" + "\n".join(missing),
                                 parent=self.root)
            return
        online = [s for s in todo if ns.is_online_only(s.source_path)]
        if online and not messagebox.askyesno(
                "Download from OneDrive?",
                f"{len(online)} section image(s) ({human_gb(sum(os.path.getsize(s.source_path) for s in online))}) "
                "are online-only and will be downloaded to re-render them. Continue?", parent=self.root):
            return
        out = filedialog.asksaveasfilename(
            parent=self.root, title="Export JPX", initialdir=st.folder,
            initialfile=ns.next_free_name(st.folder), defaultextension=".jpx",
            filetypes=[("JPX stacks", "*.jpx")])
        if not out:
            return
        self.busy = True
        self.stop_event.clear()
        self.button_cancel.config(state="normal")
        self.update_cursor()
        self.var_status.set(f"Exporting to {out} ...")

        def progress(i, n, name):
            self.msgs.put(("progress", (i, n, f"Exporting: {name} ({i + 1} of {n})...")))

        def work():
            t0 = time.time()
            try:
                ns.export_stack(st, out, quality_db=quality, rerender_all=rerender_all,
                                log=lambda m: print(m, flush=True), progress=progress,
                                stop_event=self.stop_event)
                self.msgs.put(("export_done", (out, None, time.time() - t0)))
            except ns.UserCanceled:
                self.msgs.put(("export_done", (out, "canceled", 0)))
            except Exception as e:
                self.msgs.put(("export_done", (out, str(e), 0)))

        threading.Thread(target=work, daemon=True).start()

    def export_finished(self, out, error, seconds):
        self.busy = False
        self.button_cancel.config(state="disabled")
        self.update_cursor()
        if error == "canceled":
            self.var_status.set("Export canceled; nothing was written.")
            return
        if error:
            self.var_status.set("Export failed.")
            messagebox.showerror("Export failed", error, parent=self.root)
            return
        self.progress.config(value=self.progress["maximum"])
        self.var_status.set(f"Exported {os.path.basename(out)} in {seconds:.0f} s. "
                            "Now editing the exported file.")
        # Continue from the exported file, so later exports only re-render later changes
        keep_previews, keep_view, cur = self.previews, self.view, self.cur
        try:
            stack = ns.open_stack(out)
        except Exception as e:
            messagebox.showwarning("Exported", f"Exported, but couldn't reopen it: {e}", parent=self.root)
            return
        self.stack = stack
        self.previews, self.view, self.cur = keep_previews, keep_view, cur
        self.undo = []
        self.root.title(f"JPX aligner - {os.path.basename(out)}")
        self.refresh_list()
        self.update_controls()
        self.redraw()

    # ---------------------------------------------------------------- run

    def quit(self):
        if self.busy:
            if not messagebox.askyesno("Export in progress", "An export is running. Cancel it and quit?",
                                       parent=self.root):
                return
            self.stop_event.set()
        elif not self.confirm_discard():
            return
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Aligner(sys.argv[1] if len(sys.argv) > 1 else None).run()
