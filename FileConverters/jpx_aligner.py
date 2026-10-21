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
import json
import math
import os
import queue
import sys
import threading
import time
import weakref
from dataclasses import asdict

import neuroinfo_stack as ns
import section_project as sp
import neuroinfo_dialogs
from section_order import OrderGrid, THUMB_W, THUMB_H

ns.ensure_packages({"PIL": "pillow"})

import tkinter as tk  # noqa: E402
from tkinter import ttk, filedialog, messagebox  # noqa: E402

import cv2  # noqa: E402
import imagecodecs  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

# Brightness histograms of the section pictures (for the Auto buttons of the display levels): the pixel values 1..65535 in LEVEL_BINS bins
# equal in ratio (about 0.5 % wide each), so that the dim end, where the black point lies, is as finely resolved as the bright end.
LEVEL_BINS = 2048
_LEVEL_LN = math.log(65536.0)


def _level_bin_table():
    v = np.arange(65536, dtype=np.float64)
    v[0] = 1.0
    return np.minimum((np.log(v) / _LEVEL_LN * LEVEL_BINS).astype(np.int64), LEVEL_BINS - 1)


LEVEL_BIN_OF = _level_bin_table()


def histogram_percentile(h, q):
    """The q-th percentile (0-100) of the pixel values counted in the histogram h (LEVEL_BINS counts), or None if it is empty."""
    total = float(h.sum())
    if total <= 0:
        return None
    cum = np.cumsum(h)
    target = q / 100.0 * total
    b = min(int(np.searchsorted(cum, target)), LEVEL_BINS - 1)
    before = float(cum[b - 1]) if b else 0.0
    frac = min(max((target - before) / max(float(h[b]), 1.0), 0.0), 1.0)
    lo, hi = math.exp(b / LEVEL_BINS * _LEVEL_LN), math.exp((b + 1) / LEVEL_BINS * _LEVEL_LN)
    return lo + frac * (hi - lo)


PREVIEW_WORKERS = max(2, min(8, (os.cpu_count() or 4) - 2))   # slide scans whose sections' pictures are made at the same time: most of the
# cores, leaving two for the window and the other work, and at most 8 (each decodes a big picture, so memory and the drive limit it too)
PREVIEW_MAX_PX = ns.PREVIEW_MAX_PX   # longest side of the cached preview of each section
THUMB_MAX_PX = 1600     # longest side of the slide thumbnail as it is built (it is scaled to the left pane's width when shown)
CACHE_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                         "JhouLab", "jpx_aligner_cache")
SETTINGS_FILE = os.path.join(os.path.dirname(CACHE_DIR), "jpx_aligner_settings.json")
RECENT_MAX = 10

# The pointer over the image says what a drag would do (Tk's built-in cursor names).
CURSOR_MOVE = "fleur"          # drag moves the section
CURSOR_ROTATE = "exchange"     # Shift held: drag rotates it
CURSOR_PAN = "hand1"           # Ctrl held: drag pans the view
CURSOR_SELECT = "crosshair"    # outline selected, over empty space: drag makes a selection box
CURSOR_GRAB = "hand2"          # over a vertex (or the outline): click selects, drag moves it
CURSOR_ADD = "plus"            # over a line of the outline: double-click adds a vertex
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
    full = ns.read_jp2_image(src)
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


def load_settings():
    """The settings file's contents as a dict ({} if there is none or it cannot be read)."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(**changes):
    """Update the given settings in the file, keeping the others; failures to save are ignored."""
    data = load_settings()
    data.update(changes)
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    except OSError:
        pass


def load_recent():
    """The recently opened stacks (newest first) from the settings file; [] if there is none or it cannot be read."""
    recent = load_settings().get("recent", [])
    return [p for p in recent if isinstance(p, str)] if isinstance(recent, list) else []


def add_recent(path):
    """Put `path` first in the recent list (once, whatever the letter case), keeping RECENT_MAX; failures to save are ignored."""
    key = os.path.normcase(os.path.abspath(path))
    recent = [os.path.abspath(path)] + [p for p in load_recent() if os.path.normcase(os.path.abspath(p)) != key]
    save_settings(recent=recent[:RECENT_MAX])


def clear_recent():
    save_settings(recent=[])


def load_frame_preview(jpx_path, frame_box, index):
    """Downsampled copy of one frame of the JPX itself (the same kind of tuple as load_preview), cached on disk. Used for
    sections whose source image this tool cannot read: the frame is already on the stack canvas, so it is shown as it is."""
    st = os.stat(jpx_path)
    key = f"frame|{os.path.abspath(jpx_path).lower()}|{st.st_size}|{st.st_mtime_ns}|{index}|{PREVIEW_MAX_PX}"
    path = os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".npz")
    if os.path.isfile(path):
        try:
            with np.load(path) as d:
                return d["img"], float(d["kx"]), float(d["ky"])
        except Exception:
            pass  # rebuild a damaged cache file
    with open(jpx_path, "rb") as f:
        full = imagecodecs.jpeg2k_decode(ns.read_payload(f, frame_box))
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


class Tooltip:
    """A small note that appears when the pointer rests on a widget."""

    def __init__(self, widget, text, delay=500):
        self.widget, self.text, self.delay, self.tip, self.job = widget, text, delay, None, None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, event=None):
        self.hide()
        self.job = self.widget.after(self.delay, self.show)

    def show(self):
        self.job = None
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        tk.Label(self.tip, text=self.text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                 wraplength=260, padx=4, pady=2).pack()
        self.tip.update_idletasks()
        x = min(self.widget.winfo_rootx(), self.widget.winfo_screenwidth() - self.tip.winfo_reqwidth() - 4)   # (kept on screen)
        self.tip.wm_geometry(f"+{max(x, 0)}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 4}")

    def hide(self, event=None):
        if self.job:
            self.widget.after_cancel(self.job)
            self.job = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


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
        self.thumbs = {}            # .czi path -> (picture as a PIL image, (x0, y0, k)): the slide scan's thumbnail, for the picture above the list
        self.thumb_photos = {}      # (path, width) -> the PhotoImage of that picture at that width (it follows the left pane's width)
        self.thumb_job = None
        self.frame_previews = set()  # indices of sections whose preview is the JPX's own frame (not rendered from a source image)
        self.cur = 0
        self.undo = []              # steps: each a list of (section index, Edit before the change)
        self.drag = None
        self.view = None            # (scale view px per canvas px, ox, oy)
        self.fitted = True          # the view is the fit-to-window one (not panned or zoomed by hand since)
        self.last_size = None       # canvas size at the last redraw, to scale the view when the window is resized
        self.sel_outline = False    # the current section's outline is selected (its vertices show as squares)
        self.image_selected = False  # what sel_outline was when the picture was last drawn (the opacity depends on it)
        self.sel_vertices = set()   # indices of the selected vertices of that outline (red squares)
        self.odrag = None           # an outline gesture in progress (moving vertices, or a rubber band)
        self.prov_sel = set()       # vertices inside the rubber band while it is being dragged
        self.outline_undo = []      # steps of the Drawing panel's Undo: (section index, its outline before the change)
        self.last_ref = None        # the comparison section and opacity of the last redraw, for the overlay
        self.last_alpha = 0.75
        self.var_draw_info = tk.StringVar(self.root, "")
        self.var_draw_sel = tk.StringVar(self.root, "")   # second, indented line: how many vertices are selected
        self.hover = None           # ('vertex', i) or ('edge', i) under the pointer, for the current outline
        self.mod_shift = self.mod_ctrl = False   # Shift / Ctrl are held (for the cursor)
        self.pointer_in_canvas = False           # the pointer is over the image (for the hint in the status bar)
        self.canvas_cursor_now = None
        self.opening = False        # a stack is being opened
        self.loading_previews = False   # section images are still being loaded in the background
        self.levels = []            # per channel [lo, hi] (display only)
        self.levels_set = False
        self.level_cache = {}       # id(picture) -> (weak reference to it, its brightness histograms), see level_hist
        self._busy = False
        self.calibrating = False    # the .czi sections are being located on their slide scans
        self.locating_done = False  # that has finished (what is still not located then is shown in red)
        self.msgs = queue.Queue()
        self.stop_event = threading.Event()
        self.stop_event_verify = threading.Event()   # (set when a stack is closed or replaced: the position check ends)
        self.photo = None

        self.var_ref = tk.StringVar(self.root, "previous")
        self.var_mode = tk.StringVar(self.root, "blend")
        self.var_opacity = tk.DoubleVar(self.root, 0.75)
        self.var_rot = tk.StringVar(self.root, "0")
        self.var_dx = tk.StringVar(self.root, "0")
        self.var_dy = tk.StringVar(self.root, "0")
        self.var_quality = tk.StringVar(self.root, f"{ns.DEFAULT_QUALITY_DB:g}")
        self.return_to_launcher = False   # set by the start window: closing this window goes back to it, instead of ending the program
        self.back_to_ordering = False     # hidden with "Back to step 3": the start window shows the loader at the ordering tab meanwhile
        self.hidden = False               # (while hidden nothing is written to the project file: the loader owns it then)
        self.verifying = False            # the check of the sections' assumed positions is running
        self.start_verify = lambda: False # (set while a stack loads: starts that check)
        self.check_incomplete = False     # the check ended with some positions still unchecked (it could not run): export asks instead of waiting
        self.reuse_info = None            # what section_project.reuse_order needs to tell whether this window can stand for the project again
        self.timer_ids = {}         # the repeating timers (poll, autosave), to cancel when the window closes
        self.project_path = ""      # the section project the stack came from (its edits are saved into it as you work), if it did
        self.project_obj = None
        self.saved_sig = None       # what stack_signature() was when the project was last written
        self.var_fit = tk.BooleanVar(self.root, True)          # fit the canvas to the sections when exporting a stack made in memory
        self.var_margin = tk.StringVar(self.root, "10")       # (the margin around them, %)
        self.var_spacing = tk.StringVar(self.root, "")        # distance between adjacent sections (um), written into the JPX
        self.var_rerender = tk.BooleanVar(self.root, True)   # every image is written again, so each JPX file has all of its images
        self.var_background = tk.BooleanVar(self.root, bool(load_settings().get("background_loading", True)))
        self.var_advanced = tk.BooleanVar(self.root, bool(load_settings().get("advanced_mode", False)))
        self.var_verify = tk.BooleanVar(self.root, bool(load_settings().get("verify_positions", True)))
        self.var_crosshair = tk.BooleanVar(self.root, True)
        self.var_follow = tk.BooleanVar(self.root, False)   # an edit also moves the sections after it
        self.var_status = tk.StringVar(self.root, "Open a reconstruction.jpx to start (File > Open).")
        self.chan_vars = []         # per channel (visible BooleanVar, lo DoubleVar, hi DoubleVar)

        self.build()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.timer("poll", 100, self.poll)
        self.timer("autosave", 2000, self.autosave_tick)
        if path:
            self.root.after(200, lambda: self.open_path(path))

    # ---------------------------------------------------------------- layout

    def build(self):
        root = self.root
        menu = tk.Menu(root)
        fm = tk.Menu(menu, tearoff=0)
        fm.add_command(label="Open...", command=self.open_dialog, accelerator="Ctrl+O")
        fm.add_command(label="Save project now (also done automatically)", command=lambda: self.save_project_state(force=True))
        self.recent_menu = tk.Menu(fm, tearoff=0)
        fm.add_cascade(label="Open recent", menu=self.recent_menu)
        fm.config(postcommand=self.fill_recent_menu)   # (refreshed each time the File menu opens)
        fm.add_command(label="Export JPX...", command=self.export_dialog, accelerator="Ctrl+S")
        om = tk.Menu(fm, tearoff=0)
        om.add_checkbutton(label="Load images in the background", variable=self.var_background,
                           command=self.background_option_changed)
        om.add_checkbutton(label="Check section positions in the background", variable=self.var_verify,
                           command=lambda: save_settings(verify_positions=bool(self.var_verify.get())))
        om.add_checkbutton(label="Advanced mode", variable=self.var_advanced, command=self.advanced_option_changed)
        fm.add_cascade(label="Options", menu=om)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.quit)
        menu.add_cascade(label="File", menu=fm)
        em = tk.Menu(menu, tearoff=0)
        em.add_command(label="Undo", command=self.do_undo, accelerator="Ctrl+Z")
        em.add_command(label="Reset this section", command=self.reset_section)
        em.add_command(label="Reset all sections", command=self.reset_all)
        menu.add_cascade(label="Edit", menu=em)
        tm = tk.Menu(menu, tearoff=0)
        tm.add_command(label="Analyze project", command=self.analyze_project)
        menu.add_cascade(label="Tools", menu=tm)
        menu.add_command(label="\u25c4 Back to step 3: order sections", command=self.go_back_to_ordering)
        self.back_menu_index = menu.index("end")
        self.menubar = menu
        root.config(menu=menu)

        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # The left pane and the image sit in a paned window: drag the bar between them to give the pane more room for its text
        # or the image more room. The pane keeps the width it has (text changing inside it cannot widen it).
        self.paned = tk.PanedWindow(main, orient=tk.HORIZONTAL, sashwidth=7, sashrelief=tk.RAISED, bd=0,
                                    opaqueresize=True, background="gray70")
        self.paned.grid(row=0, column=0, sticky="nsew")
        left = self.left_pane = ttk.Frame(self.paned, padding=5)
        draw = ttk.LabelFrame(left, text="Outline drawing", padding=6)
        draw.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        ttk.Label(draw, textvariable=self.var_draw_info, justify=tk.LEFT).pack(anchor="w")
        ttk.Label(draw, textvariable=self.var_draw_sel, justify=tk.LEFT).pack(anchor="w", padx=(18, 0))
        self.button_undo_outline = ttk.Button(draw, text="Undo", command=self.undo_outline, state="disabled")
        self.button_undo_outline.pack(anchor="w", pady=(4, 4))
        Tooltip(self.button_undo_outline, "Undo most recent change to section outline")
        ttk.Label(draw, foreground="gray30", justify=tk.LEFT,
                  text="click an outline: select it\nthen: drag a square = move it\n"
                       "drag a box = select squares\ndrag a selection = move the group\n"
                       "double-click a line = add a vertex\nDelete = remove vertex / outline\n"
                       "click elsewhere: deselect").pack(anchor="w")
        ttk.Label(left, text="* changed since opened\n~ edited  |  F flipped", foreground="gray30",
                  justify=tk.LEFT).pack(side=tk.BOTTOM, anchor="w", pady=(4, 0))
        self.var_sections_title = tk.StringVar(self.root, "Sections")   # (shows the loading progress while images load)
        self.label_sections = ttk.Label(left, textvariable=self.var_sections_title)
        self.label_sections.pack(anchor="w")
        # Where the current section lies on the slide (for stacks whose sections come from a slide scan: their names are all
        # alike). Packed above the label when there is a picture to show.
        self.thumb_canvas = tk.Canvas(left, bg="black", highlightthickness=0)
        self.thumb_shown = False
        order = ttk.Frame(left)   # under the list (packed before it, so the list takes what is left)
        order.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
        for text, command, tip in (
                ("Move up", lambda: self.move_section(-1), "Move selected section earlier in the sequence order"),
                ("Move down", lambda: self.move_section(1), "Move selected section later in the sequence order"),
                ("Delete", self.delete_section, "Remove section from sequence"),
                ("Thumbnails...", self.order_dialog, "Put the sections in order by dragging their pictures about in a window")):
            button = ttk.Button(order, text=text, command=command)
            button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            Tooltip(button, tip)
        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)   # (shown only when the list is longer than its window)
        self.list_scroll_shown = False
        self.listbox = tk.Listbox(list_frame, width=30, height=12, exportselection=False, activestyle="none",
                                  font=("Segoe UI", 10), yscrollcommand=self.on_list_scroll)
        self.list_scroll.config(command=self.listbox.yview)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)
        # drag a section to a new place in the order
        self.list_drag = None
        self.listbox.bind("<ButtonPress-1>", self.list_press, add="+")
        self.listbox.bind("<B1-Motion>", self.list_motion)          # (replaces the default, which would select what the pointer passes)
        self.listbox.bind("<ButtonRelease-1>", self.list_release, add="+")

        self.canvas = tk.Canvas(self.paned, bg="black", highlightthickness=0, cursor="fleur")
        self.paned.add(left, minsize=180, stretch="never")
        self.paned.add(self.canvas, minsize=300, stretch="always")
        # fix the pane's width (at least 320 px at 96 dpi: the slide thumbnail takes the width of the pane, so it is not left tiny)
        self.root.after_idle(lambda: self.paned.paneconfigure(
            left, width=max(left.winfo_reqwidth(), round(320 * self.root.winfo_fpixels("1i") / 96))))
        left.bind("<Configure>", self.on_left_resize)
        c = self.canvas
        c.bind("<Configure>", self.on_resize)
        c.bind("<ButtonPress-1>", self.on_left_press)
        c.bind("<Motion>", self.on_hover)
        c.bind("<Leave>", self.on_leave_canvas)
        c.bind("<Double-Button-1>", self.on_double_click)
        c.bind("<Triple-Button-1>", self.on_double_click)
        c.bind("<ButtonPress-3>", self.on_right_press)
        c.bind("<ButtonPress-2>", lambda e: self.press(e, "pan"))
        for b in ("<B1-Motion>", "<B2-Motion>", "<B3-Motion>"):
            c.bind(b, self.motion)
        for b in ("<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>"):
            c.bind(b, self.release)
        c.bind("<MouseWheel>", self.wheel)

        right = ttk.Frame(main, padding=5)
        right.grid(row=0, column=1, sticky="ns")
        self.build_right(right)

        bottom = ttk.Frame(root, padding=(5, 2))
        bottom.pack(fill=tk.X)
        self.progress = ttk.Progressbar(bottom, length=220, mode="determinate")
        self.progress.pack(side=tk.RIGHT)
        self.button_cancel = ttk.Button(bottom, text="Cancel", command=self.stop_event.set, state="disabled")
        self.button_cancel.pack(side=tk.RIGHT, padx=5)
        # The status text, or (instead of it, while there is one) a hint about what is under the pointer on the image.
        self.var_hint = tk.StringVar(self.root, "")
        self.status_label = ttk.Label(bottom, textvariable=self.var_status, anchor="w")
        self.hint_label = ttk.Label(bottom, textvariable=self.var_hint, anchor="w", foreground="#0050a0")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        for seq, fn in (("<Control-o>", self.open_dialog), ("<Control-s>", self.export_dialog),
                        ("<Control-z>", self.do_undo)):
            root.bind(seq, lambda e, f=fn: f())
        root.bind("<Key>", self.key)
        for key in ("Shift_L", "Shift_R", "Control_L", "Control_R"):
            root.bind(f"<KeyPress-{key}>", lambda e: self.modifier_key(e, True), add="+")
            root.bind(f"<KeyRelease-{key}>", lambda e: self.modifier_key(e, False), add="+")
        root.bind("<FocusOut>", lambda e: self.modifier_lost(), add="+")

    def build_right(self, right):
        f = ttk.LabelFrame(right, text="Current section", padding=6)
        f.pack(fill=tk.X, pady=(0, 6))
        # (advanced mode only)
        self.follow_check = ttk.Checkbutton(f, text="Edits also move subsequent sections", variable=self.var_follow)
        self.follow_check.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        self.label_section = ttk.Label(f, text="-", font=("Segoe UI", 10, "bold"))
        self.label_section.grid(row=1, column=0, columnspan=4, sticky="w")
        for r, (text, var) in enumerate((("Rotation (deg, clockwise)", self.var_rot), ("Shift x (um)", self.var_dx),
                                         ("Shift y (um)", self.var_dy)), start=2):
            ttk.Label(f, text=text).grid(row=r, column=0, sticky="e", pady=1)
            e = ttk.Entry(f, textvariable=var, width=10)
            e.grid(row=r, column=1, columnspan=2, sticky="w", padx=4)
            e.bind("<Return>", lambda ev: self.apply_entries())
        ab = ttk.Frame(f)   # Apply and Undo share the height the Apply button had alone
        ab.grid(row=2, column=3, rowspan=3, sticky="ns")
        ttk.Button(ab, text="Apply", command=self.apply_entries).pack(fill=tk.BOTH, expand=True)
        undo = ttk.Button(ab, text="Undo", command=self.do_undo)
        undo.pack(fill=tk.BOTH, expand=True)
        Tooltip(undo, "Undoes the last alignment edit (rotation, shift, flip). Shortcut: Ctrl+Z.\n"
                      "Outline edits have their own Undo, in the Outline drawing panel.")
        bf = ttk.Frame(f)
        bf.grid(row=5, column=0, columnspan=4, pady=(6, 0))
        for text, deg in (("-1°", -1), ("-0.1°", -0.1), ("+0.1°", 0.1), ("+1°", 1)):   # (+ is clockwise)
            ttk.Button(bf, text=text, width=6, command=lambda d=deg: self.rotate_by(-d)).pack(side=tk.LEFT)
        bf2 = ttk.Frame(f)
        bf2.grid(row=6, column=0, columnspan=4, pady=(4, 0))
        ttk.Button(bf2, text="Flip left-right (F)", command=self.flip).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf2, text="Reset", command=self.reset_section).pack(side=tk.LEFT, padx=2)
        # (wrapped, so that a long note cannot widen the panel at the expense of the picture)
        self.label_detail = ttk.Label(f, text="", foreground="gray30", justify=tk.LEFT,
                                      wraplength=round(250 * self.root.winfo_fpixels("1i") / 96))
        self.label_detail.grid(row=7, column=0, columnspan=4, sticky="w", pady=(6, 0))

        f = ttk.LabelFrame(right, text="Compare with", padding=6)
        f.pack(fill=tk.X, pady=(0, 6))
        for text, val in (("Previous section", "previous"), ("Next section", "next"), ("Nothing", "none")):
            ttk.Radiobutton(f, text=text, value=val, variable=self.var_ref,
                            command=self.redraw).pack(anchor="w")
        self.mode_frame = ttk.Frame(f)   # the blending modes: advanced mode only (otherwise the sections are always blended)
        ttk.Separator(self.mode_frame).pack(fill=tk.X, pady=4)
        ttk.Radiobutton(self.mode_frame, text="Blend (true colors)", value="blend", variable=self.var_mode,
                        command=self.redraw).pack(anchor="w")
        ttk.Radiobutton(self.mode_frame, text="Overlay (this = magenta, other = cyan)", value="overlay",
                        variable=self.var_mode, command=self.redraw).pack(anchor="w")
        self.label_opacity = ttk.Label(f, text="Opacity of this section")
        self.label_opacity.pack(anchor="w", pady=(4, 0))
        ttk.Scale(f, from_=0, to=1, variable=self.var_opacity,
                  command=lambda v: self.redraw()).pack(fill=tk.X)

        self.view_frame = f = ttk.LabelFrame(right, text="View", padding=6)   # (advanced mode only)
        ttk.Checkbutton(f, text="Crosshair at the centre of the canvas", variable=self.var_crosshair,
                        command=self.redraw).pack(anchor="w")

        self.chan_frame = ttk.LabelFrame(right, text="Brightness (applies to all sections, affects view only)", padding=6)
        self.chan_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(self.chan_frame, text="(open a stack)").pack()

        f = ttk.LabelFrame(right, text="Export", padding=6)
        f.pack(fill=tk.X, pady=(0, 6))
        # (the quality and re-render choices are for advanced mode only; the defaults are the sensible ones)
        self.export_advanced = [
            ttk.Label(f, text="Quality (dB, max ~96)"),   # 16-bit frames: nothing above ~96 dB changes the file
            ttk.Entry(f, textvariable=self.var_quality, width=6),
            ttk.Checkbutton(f, text="Re-render unchanged sections too", variable=self.var_rerender)]
        self.export_advanced[0].grid(row=0, column=0, sticky="e")
        self.export_advanced[1].grid(row=0, column=1, sticky="w", padx=4)
        self.export_advanced[2].grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(f, text="Distance between sections (\u00b5m)").grid(row=2, column=0, sticky="e", pady=(4, 0))
        ttk.Entry(f, textvariable=self.var_spacing, width=6).grid(row=2, column=1, sticky="w", padx=4, pady=(4, 0))
        self.fit_row = ttk.Frame(f)    # (only for a stack made in memory: one opened from a JPX keeps its canvas)
        ttk.Checkbutton(self.fit_row, text="Fit the canvas to the sections, margin", variable=self.var_fit).pack(side=tk.LEFT)
        ttk.Entry(self.fit_row, textvariable=self.var_margin, width=4).pack(side=tk.LEFT, padx=3)
        ttk.Label(self.fit_row, text="%").pack(side=tk.LEFT)
        self.fit_row.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.fit_row.grid_remove()
        ttk.Button(f, text="Export JPX...", command=self.export_dialog).grid(row=4, column=0, columnspan=2,
                                                                              pady=(6, 0), sticky="ew")

        self.apply_advanced_mode()   # (all the widgets it shows or hides exist now)

        help_text = ("drag: move   Shift/right-drag: rotate\n"
                     "Ctrl/middle-drag: pan   wheel: zoom\n"
                     "arrows: nudge (image focused)  [ ]: rotate  F: flip\n"
                     "PgUp/PgDn: section   Home: fit view")
        ttk.Label(right, text=help_text, foreground="gray30", justify=tk.LEFT).pack(anchor="w")

    def build_channel_controls(self):
        for w in self.chan_frame.winfo_children():
            w.destroy()
        self.chan_vars = []
        self.chan_frame.columnconfigure(2, weight=1)
        # buttons above the sliders that set the black and white points of all the channels at once
        bar = ttk.Frame(self.chan_frame)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        for c in range(3):
            bar.columnconfigure(c, weight=1, uniform="levelbtn")
        for c, (text, how, tip) in enumerate((
                ("Auto1", "current", "Set each channel's black point to the 1st percentile and its white point to the 99.7th percentile of "
                                     "the pixel values (ignoring empty pixels) in the current section only."),
                ("Auto2", "all", "The same, but the percentiles are taken over all the sections whose pictures have loaded."),
                ("Full", "full", "Full scale: set every channel's black point to 0 and its white point to 65535, the whole range of "
                                 "16-bit pixel values. No analysis of the pictures."))):
            b = ttk.Button(bar, text=text, command=lambda how=how: self.set_levels(how))
            b.grid(row=0, column=c, sticky="ew", padx=(0 if c == 0 else 2, 0))
            Tooltip(b, tip)
        for i, ch in enumerate(self.stack.channel_info):
            lo, hi = self.levels[i]
            vis = tk.BooleanVar(self.root, True)
            vlo = tk.DoubleVar(self.root, lo)
            vhi = tk.DoubleVar(self.root, hi)
            self.chan_vars.append((vis, vlo, vhi))
            hexcol = "#%02x%02x%02x" % tuple(ch.color)
            # One grid for all the channels, so that the squares and the sliders line up whatever the names' widths: each channel
            # has two rows, the square (with the channel's name under it) at the left and the two sliders beside it.
            row = self.chan_frame
            row.columnconfigure(2, weight=1)
            sq_col = ttk.Frame(row)
            sq_col.grid(row=1 + 2 * i, column=0, rowspan=2, sticky="nw", padx=(0, 4), pady=(4 if i else 0, 0))
            # The colored square is the show/hide checkbox: while the channel is shown, two thin lines join its opposite corners
            # (black, or white on a very dark color so that they can be seen).
            dark = 0.299 * ch.color[0] + 0.587 * ch.color[1] + 0.114 * ch.color[2] < 60
            size = round(22 * self.root.winfo_fpixels("1i") / 96)
            square = tk.Canvas(sq_col, width=size, height=size, bg=hexcol, cursor="hand2", highlightthickness=1,
                               highlightbackground="gray40")
            square.pack(anchor="w")

            def toggle(event=None, vis=vis):
                vis.set(not vis.get())
                self.redraw()

            def show_x(*_, vis=vis, square=square, size=size, line="white" if dark else "black"):
                square.delete("x")
                if vis.get():
                    m = 1   # (the lines run from corner to corner of the square, inside its 1-pixel border)
                    square.create_line(m, m, size, size, fill=line, tags="x")
                    square.create_line(m, size, size, m, fill=line, tags="x")

            vis.trace_add("write", show_x)
            show_x()
            square.bind("<Button-1>", toggle)
            Tooltip(square, "Show / hide this channel")
            name = ttk.Label(sq_col, text=ch.name, cursor="hand2")
            name.pack(anchor="w")
            name.bind("<Button-1>", toggle)
            for r, (text, var) in enumerate((("black", vlo), ("white", vhi))):
                pad = (4 if i and r == 0 else 0, 0)
                ttk.Label(row, text=text).grid(row=1 + 2 * i + r, column=1, sticky="w", padx=(0, 4), pady=pad)
                ttk.Scale(row, from_=0, to=65535, variable=var,
                          command=lambda v: self.redraw()).grid(row=1 + 2 * i + r, column=2, sticky="ew", pady=pad)

    # ---------------------------------------------------------------- busy cursor

    @property
    def busy(self):
        """Something is running that rules out editing: an export, or (with background loading switched off) the loading of
        the section images."""
        return self._busy or (self.loading_previews and not self.var_background.get())

    @busy.setter
    def busy(self, value):
        self._busy = value

    def apply_advanced_mode(self):
        """Show the features meant for experienced users only in advanced mode (and use the plain ones otherwise)."""
        if self.var_advanced.get():
            self.mode_frame.pack(fill=tk.X, before=self.label_opacity)
            self.view_frame.pack(fill=tk.X, pady=(0, 6), before=self.chan_frame)
            for w in self.export_advanced:
                w.grid()
            self.follow_check.grid()
        else:
            # what is hidden goes back to its default, so nothing unseen keeps affecting the picture or the export
            self.mode_frame.pack_forget()
            self.view_frame.pack_forget()
            for w in self.export_advanced:
                w.grid_remove()
            self.follow_check.grid_remove()
            self.var_follow.set(False)
            self.var_mode.set("blend")
            self.var_crosshair.set(True)
            self.var_quality.set(f"{ns.DEFAULT_QUALITY_DB:g}")
            self.var_rerender.set(True)

    def advanced_option_changed(self):
        save_settings(advanced_mode=bool(self.var_advanced.get()))
        self.apply_advanced_mode()
        if self.stack:
            self.redraw()

    def background_option_changed(self):
        save_settings(background_loading=bool(self.var_background.get()))
        self.update_cursor()
        self.refresh_list()
        if self.stack:
            self.update_controls()
            self.redraw()

    def update_cursor(self):
        """The hourglass over the whole window (the image canvas has a cursor of its own, so it is set too) while a stack is
        opening, its section images are loading or an export is running; the normal cursors otherwise."""
        busy = self.opening or self.busy
        self.root.config(cursor="watch" if busy else "")
        self.canvas_cursor_now = None
        self.canvas.config(cursor="watch" if busy else self.canvas_cursor())
        self.canvas_cursor_now = None if busy else self.canvas.cget("cursor")

    def canvas_cursor(self):
        """The cursor for the image for the current state: Shift and Ctrl show what they change (rotating, panning); with an
        outline selected a drag no longer moves the section, so the cursor is a crosshair (selection box) or, over the
        outline, a hand (a vertex) or a plus (a line, where a double-click adds a vertex)."""
        if self.mod_shift:
            return CURSOR_ROTATE
        if self.mod_ctrl:
            return CURSOR_PAN
        if self.hover is not None:
            return CURSOR_ADD if self.hover[0] == "edge" else CURSOR_GRAB
        if self.sel_outline:
            return CURSOR_SELECT
        return CURSOR_MOVE

    def refresh_canvas_cursor(self):
        """Set the image's cursor to match the state (nothing is done while the hourglass is showing, or when it is already right)."""
        if self.opening or self.busy:
            return
        self.update_hint()
        name = self.canvas_cursor()
        if name != self.canvas_cursor_now:
            self.canvas_cursor_now = name
            self.canvas.config(cursor=name)

    def hint_text(self):
        """What the pointer is over on the image and what can be done there ("" when not over the image or no stack)."""
        if not self.stack or not self.pointer_in_canvas or self.busy:
            return ""
        if self.mod_shift:
            return "Drag to rotate the section"
        if self.mod_ctrl:
            return "Drag to pan the view"
        if self.hover is not None:
            if self.hover[0] == "vertex":
                return ("Drag point to move, DEL key to delete" if self.sel_outline else
                        "Click to select the outline and this point")
            return ("Double-click on line to add point" if self.sel_outline else
                    "Click to select the outline; double-click on line to add point")
        if self.sel_outline:
            return "Drag to select several points; click elsewhere to deselect"
        return ""

    def update_hint(self):
        """Show the hint in the status bar in place of the status text (and bring the status text back when there is none)."""
        text = self.hint_text()
        self.var_hint.set(text)
        if text and not self.hint_label.winfo_ismapped():
            self.status_label.pack_forget()
            self.hint_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        elif not text and self.hint_label.winfo_ismapped():
            self.hint_label.pack_forget()
            self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def modifier_lost(self):
        """The window lost the keyboard focus: any key released elsewhere is not seen, so forget Shift and Ctrl."""
        if self.mod_shift or self.mod_ctrl:
            self.mod_shift = self.mod_ctrl = False
            self.refresh_canvas_cursor()

    def modifier_key(self, event, down):
        """Shift or Ctrl pressed or released (the pointer need not move for the cursor to change)."""
        if event.keysym.startswith("Shift"):
            self.mod_shift = down
        else:
            self.mod_ctrl = down
        self.refresh_canvas_cursor()

    # ---------------------------------------------------------------- opening

    # ---------------------------------------------------------------- the project file

    def stack_signature(self):
        """Something that changes when anything the project file keeps does: each section's edit, outline, pivot and source, in order."""
        st = self.stack
        if st is None:
            return None
        return (len(st.sections), st.z_spacing_um, tuple(
            (s.frame, s.czi_assumed, s.czi_origin, s.name, s.source_path, tuple(asdict(s.edit).values()), len(s.case_points), round(float(s.case_points.sum()), 3),
             round(float(s.pivot_src.sum()), 3), round(float(s.t.sum()), 3)) for s in st.sections))

    def save_project_state(self, force=False):
        """Write the stack's alignment into the project file it came from (nothing if it did not come from one). True if the file
        holds the current state afterwards."""
        if not self.project_path or self.stack is None:
            return True
        sig = self.stack_signature()
        if not force and sig == self.saved_sig:
            return True
        try:
            proj = self.project_obj or sp.Project.load(self.project_path)
            if proj.is_jpx:
                proj.jpx_order = [s.frame for s in self.stack.sections]       # (the order made here is the project's now)
            proj.alignment = ns.stack_json(self.stack, self.project_path)
            proj.alignment_key = proj.fingerprint()
            proj.save(self.project_path)
            self.project_obj = proj
            self.saved_sig = sig
            return True
        except Exception as e:
            self.var_status.set(f"Couldn't save the project file: {e}")
            return False

    def autosave_tick(self):
        if not self.busy and not self.opening and not self.hidden:
            self.save_project_state()
        self.timer("autosave", 2000, self.autosave_tick)

    def timer(self, name, ms, fn):
        self.timer_ids[name] = self.root.after(ms, fn)

    def confirm_discard(self):
        if self.project_path and self.save_project_state():
            return True      # (what was done is in the project file: nothing to lose)
        if self.stack is not None and self.stack.changed:
            return messagebox.askyesno("Unsaved changes",
                                       "Some sections have changes that haven't been exported.\n"
                                       "Discard them?", parent=self.root)
        return True

    def fill_recent_menu(self):
        """The recently opened stacks as menu entries (newest first), then a way to clear the list."""
        menu = self.recent_menu
        menu.delete(0, tk.END)
        recent = load_recent()
        for n, path in enumerate(recent):
            folder = os.path.basename(os.path.dirname(path.rstrip("/\\")))
            label = f"{n + 1}  {os.path.basename(path)}" + (f"   ({folder})" if folder else "")
            menu.add_command(label=label, command=lambda p=path: self.open_recent(p))
        if not recent:
            menu.add_command(label="(no recent files)", state="disabled")
        else:
            menu.add_separator()
            menu.add_command(label="Clear list", command=clear_recent)

    def open_recent(self, path):
        if self.busy or self.opening or not self.confirm_discard():
            return
        if not os.path.exists(path):
            messagebox.showerror("Can't open", f"This file or folder no longer exists:\n{path}", parent=self.root)
            return
        self.open_path(path)

    def open_dialog(self):
        if self.busy or self.opening or not self.confirm_discard():
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="Open a reconstruction (NeuroInfo or exported by this tool)",
            filetypes=[("JPX stacks and section projects", "*.jpx *" + sp.PROJECT_SUFFIX + " *" + sp.OLD_PROJECT_SUFFIX), ("JPX stacks", "*.jpx"),
                       ("Section projects (from the step that finds the sections)", "*" + sp.PROJECT_SUFFIX + " *" + sp.OLD_PROJECT_SUFFIX), ("All files", "*.*")],
            initialdir=self.stack.folder if self.stack else None)
        if path:
            self.open_path(path)

    def locate_file(self, kind, expected, folder, candidates, reason="missing", default=None, notes=None):
        """Ask which NeuroInfo file to use (see neuroinfo_dialogs.locate_file). Returns the path, or None if the user declines."""
        return neuroinfo_dialogs.locate_file(self.root, kind, expected, folder, candidates, reason, default, notes)

    def open_project(self, path):
        """A section project (step 1 of the workflow): its sections become an in-memory stack, rendered from the slide scans."""
        self.opening = True
        self.update_cursor()
        self.root.update()
        try:
            project = sp.Project.load(path)
            missing = [s.path for s in project.slides if not os.path.isfile(s.path)]
            if missing:
                raise FileNotFoundError("These slide scans weren't found:\n\n" + "\n".join(missing[:6]))
            stack = sp.open_jpx_stack(project, locate=self.locate_file) if project.is_jpx else sp.build_stack(project)
            self.pending_project = project
        except Exception as e:
            self.opening = False
            self.update_cursor()
            messagebox.showerror("Can't open", str(e), parent=self.root)
            return
        self.opening = False
        self.update_cursor()
        self.open_seconds = 0.0
        self.show_stack(stack, path)

    def open_path(self, path):
        """Open a JPX (or a section project). A JPX's NeuroInfo files are read in a thread, so that the window stays alive and can say
        what is happening (a big stack takes some seconds to read); a dialog asking for one of those files is made by the window itself."""
        if sp.is_project_path(path):
            return self.open_project(path)
        self.opening = True
        self.update_cursor()
        self.var_status.set("Preparing for load...")
        self.root.update()
        t_open = time.time()

        def ask(*args, **kwargs):
            """locate_file for the thread: the dialog is made, and waited for, by the window's own thread."""
            done, answer = threading.Event(), {}

            def run():
                try:
                    answer["value"] = self.locate_file(*args, **kwargs)
                except BaseException as e:
                    answer["error"] = e
                done.set()

            self.msgs.put(("call", run))
            done.wait()
            if "error" in answer:
                raise answer["error"]
            return answer["value"]

        def work():
            try:
                stack = ns.open_stack(path, locate=ask)
            except BaseException as e:
                self.msgs.put(("opened", (path, None, e, t_open)))
            else:
                self.msgs.put(("opened", (path, stack, None, t_open)))

        threading.Thread(target=work, daemon=True).start()

    def opened(self, path, stack, error, t_open):
        """The NeuroInfo files of a JPX have been read (or not): the questions that can come before the stack is shown."""
        self.open_seconds = time.time() - t_open   # (includes any dialog asking for the NeuroInfo files)
        self.opening = False
        self.update_cursor()
        if error is not None:
            self.var_status.set("")
            messagebox.showerror("Can't open", str(error) or type(error).__name__, parent=self.root)
            return
        missing = [s.name for s in stack.sections if not s.source_path]
        if missing:
            if not messagebox.askokcancel(
                    "Source images not found",
                    "These section images weren't found, so they can't be shown or re-rendered "
                    "(their current images in the JPX are kept as they are):\n\n" + "\n".join(missing)
                    + "\n\nOK opens the stack anyway; Cancel doesn't open it.",
                    icon="warning", parent=self.root):
                self.var_status.set("")
                return
        if stack.source_pixel_um and abs(stack.source_pixel_um / stack.pixel_um - 1) > 0.01:
            answer = messagebox.askyesnocancel(
                    "Pixel size doesn't match the source images",
                    f"This reconstruction is labelled with a pixel size of {stack.pixel_um:g} um, "
                    f"but its source images have {stack.source_pixel_um:g} um (from CaseFile.H5).\n\n"
                    f"Correct the pixel size to {stack.source_pixel_um:g} um?\n\n"
                    "If Yes, the next "
                    "export writes the corrected pixel size into the JPX's metadata. reconstruction.xml is not changed.",
                    icon="warning", parent=self.root)
            if answer is None:   # Cancel: don't open the stack at all
                self.var_status.set("")
                return
            if answer:
                stack.apply_pixel_size(stack.source_pixel_um)
        def other_format(s):
            return bool(s.source_path) and not s.renderable and not s.source_path.lower().endswith(".czi")
        unreadable = sorted({os.path.basename(s.source_path) for s in stack.sections if other_format(s)})
        if unreadable:
            messagebox.showinfo(
                "Source images this tool can't read",
                f"{sum(1 for s in stack.sections if other_format(s))} of the {len(stack.sections)} sections "
                "come from source images that are not JPEG 2000 or Zeiss .czi files:\n\n"
                + "\n".join(unreadable[:6]) + ("\n..." if len(unreadable) > 6 else "")
                + "\n\nFor those sections the picture is the JPX's own frame, their alignment can't be edited here, and "
                "their outlines can. Exporting copies their frames unchanged.", parent=self.root)
        czi_online = sorted({s.source_path for s in stack.sections
                             if s.source_path.lower().endswith(".czi") and ns.is_online_only(s.source_path)})
        if czi_online and not messagebox.askyesno(
                "Download from OneDrive?",
                f"{len(czi_online)} slide scan(s) ({human_gb(sum(os.path.getsize(p) for p in czi_online))}) are online-only in "
                "OneDrive and must be downloaded to show and re-render their sections. Continue?", parent=self.root):
            self.var_status.set("")
            return
        need = [s for s in stack.sections if s.renderable and not preview_is_cached(s.source_path)]
        online = [s for s in need if ns.is_online_only(s.source_path)]
        if online:
            size = sum(os.path.getsize(s.source_path) for s in online)
            if not messagebox.askyesno(
                    "Download from OneDrive?",
                    f"{len(online)} section image(s) ({human_gb(size)}) are online-only in OneDrive and "
                    "must be downloaded to build previews. Continue?", parent=self.root):
                self.var_status.set("")
                return
        self.show_stack(stack, path)

    def show_stack(self, stack, path=None):
        """Make `stack` the one being edited. `path` is the file it came from: a JPX, or a section project (the stack is then in
        memory only, and its first export writes a new JPX). With no path the stack is in memory with nothing behind it."""
        self.stack = stack
        self.previews = {}
        self.frame_previews = set()
        self.locating_done = False
        self.thumbs = {}
        self.thumb_photos = {}
        self.draw_thumbnail()
        self.undo = []
        self.outline_undo = []
        self.sel_outline, self.sel_vertices, self.odrag = False, set(), None
        self.cur = 0
        self.view = None
        self.levels = [[0.0, 65535.0] for _ in range(stack.channels)]
        self.levels_set = False
        self.build_channel_controls()
        self.var_spacing.set(f"{stack.z_spacing_um:g}" if stack.z_spacing_um > 0 else "")
        if stack.jpx_path:
            self.fit_row.grid_remove()
        else:
            self.fit_row.grid()
        self.project_path = os.path.abspath(path) if path and sp.is_project_path(path) else ""
        self.project_obj = getattr(self, "pending_project", None) if self.project_path else None
        self.pending_project = None
        self.saved_sig = self.stack_signature() if (self.project_obj is not None and self.project_obj.alignment) else None
        if stack.jpx_path:
            self.root.title(f"JPX aligner - {os.path.basename(path or stack.jpx_path)}")
        else:
            self.root.title(f"JPX aligner - unsaved project from {len(stack.sections)} sections"
                            + (f" ({os.path.basename(path)})" if path else ""))
        if path:
            add_recent(path)
            if sp.is_project_path(path):
                sp.add_recent_project(path)
        self.refresh_list()
        self.update_controls()
        self.redraw(fit_if_needed=True)
        self.canvas.focus_set()   # so the arrow keys nudge the image right away (click the list to move through sections)
        self.start_preview_loading()

    def start_preview_loading(self):
        stack = self.stack
        todo = list(stack.sections)
        self.progress.config(maximum=max(len(todo), 1), value=0)
        self.loading_previews = bool(todo)
        t_start = time.time()   # (the loading proper: after the questions asked when opening)
        open_seconds = getattr(self, "open_seconds", 0.0)
        first = []
        self.calibrating = any(s.source_path.lower().endswith(".czi") and s.czi_origin is None for s in todo)
        self.locating_done = not self.calibrating
        self.update_cursor()
        self.refresh_list()   # (sections not loaded yet are grey)
        self.update_sections_title()
        # load the current section and its neighbors first
        todo.sort(key=lambda s: abs(s.index - self.cur))

        czi_paths = sorted({s.source_path for s in todo if s.source_path.lower().endswith(".czi")})
        early = getattr(stack, "prelocate", None)         # (the loader may have made some of the pictures already: see start_prelocate)
        thumbs_done = threading.Event()
        if czi_paths:
            max_px = THUMB_MAX_PX   # (large; the picture is scaled to the left pane's width as it is drawn)

            def thumbs():
                ns.lower_thread_priority()
                try:
                    for path in czi_paths:
                        try:
                            made = early["thumbs"].get(path) if early is not None else None
                            img, geometry = made if made is not None else ns.czi_slide_thumbnail(path, max_px)
                        except Exception:
                            continue   # (the picture is only a convenience)
                        self.msgs.put(("thumb", (stack, path, img, geometry)))
                finally:
                    thumbs_done.set()

            threading.Thread(target=thumbs, daemon=True).start()
        else:
            thumbs_done.set()

        # (the previews the loader made ahead of time, while the order was being chosen, are used as they are: see start_prelocate)
        posted = set()
        lock = early["lock"] if early is not None else threading.Lock()
        czi_scenes = early["scenes"] if early is not None else {}

        def post(s, pv, is_frame):
            """Hand a section's picture to the window (once), which shows it and un-greys the section."""
            with lock:
                if s.frame in posted:
                    return
                posted.add(s.frame)
                n = len(posted)
                if not first:
                    first.append(time.time() - t_start)
            try:
                self.level_hist(pv[0])      # (ready for the Auto buttons of the display levels, which then need no time)
            except Exception:
                pass
            self.msgs.put(("preview", (stack, s.frame, pv, n, is_frame)))

        def czi_picture(s):
            key = (s.source_path, ns.czi_scene_of(s, stack))
            with lock:
                pv = czi_scenes.get(key)
            if pv is None:
                pv = ns.czi_preview(s, stack, PREVIEW_MAX_PX)
                with lock:
                    czi_scenes[key] = pv
            return pv

        def located(s):
            """A .czi section was just located on its scan: its picture can be shown now, before the others are done."""
            try:
                post(s, czi_picture(s), False)
            except Exception as e:
                self.msgs.put(("status", f"Couldn't read the scan of section {s.index + 1}: {e}"))

        locate_seconds = [0.0]
        verify_wanted = bool(self.var_verify.get())   # (read here: Tk variables are for the main thread only)
        verify_stop = threading.Event()   # (a stack opened later ends the check of this one)
        self.stop_event_verify.set()
        self.stop_event_verify = verify_stop

        def verify():
            """Check, one by one, the sections that were given their scan's origin against their own frames."""
            ns.lower_thread_priority()
            t_v = time.time()
            n_assumed = sum(1 for s in stack.sections if s.czi_assumed)
            try:
                summary = ns.verify_czi_origins(stack, progress=lambda i, n, text: self.msgs.put(("status", text)),
                                                stop=verify_stop)
            except BaseException as e:
                self.check_incomplete = True
                self.verifying = False
                self.msgs.put(("status", f"Couldn't check the section positions: {e}"))
                return
            self.check_incomplete = any(s.czi_assumed for s in stack.sections)    # (e.g. no rasterio: nothing could be checked)
            self.verifying = False
            parts = [f"{summary['checked']} checked in {time.time() - t_v:.0f} s"]
            if summary["corrected"]:
                parts.append(f"{len(summary['corrected'])} had a position of their own (> 3 px from the scan's) and were moved to it: "
                             + ", ".join(f"{n} ({dx:+.1f}, {dy:+.1f} px)" for n, dx, dy in summary["corrected"][:12]))
            if summary["unverified"]:
                parts.append(f"{len(summary['unverified'])} could not be checked (no good match; kept the scan's position): "
                             + ", ".join(str(n) for n in summary["unverified"][:12]))
            print(f"Checked the section positions of {os.path.basename(stack.jpx_path)}: " + "; ".join(parts), flush=True)
            self.msgs.put(("verified", (stack, summary)))

        def start_verify():
            """Start the check of the unchecked positions (once, and only if there are any); True if one is running."""
            if self.verifying:
                return True
            if self.stack is not stack or not any(s.czi_assumed for s in stack.sections):
                return False
            self.verifying = True
            threading.Thread(target=verify, daemon=True).start()
            return True

        self.start_verify = start_verify

        def work():
            ns.lower_thread_priority()   # (the window stays responsive while the images load)
            # a slide scan can be read by one thread at a time, and locating, checking and loading its sections keeps it busy for minutes: the
            # overview pictures above the list (a few seconds in all) go first, or they would not appear until those are done with the scan
            self.msgs.put(("status", "Preparing for load: reading the slide scans' overview pictures..."))
            thumbs_done.wait(30)
            frame_boxes = None
            pre = getattr(stack, "prelocate", None)
            if pre is not None:
                # the loader (step 3) started locating the sections while the order was being chosen: wait for what is left, showing the
                # sections as they are located
                stack.prelocate = None
                self.msgs.put(("status", "Preparing for load: locating the sections on the slide scans (started while you ordered them)..."))
                seen = 0
                n_czi = sum(1 for s in todo if s.source_path.lower().endswith(".czi"))
                while True:
                    finished = pre["calibrated"].wait(0.3)
                    if self.stack is not stack:
                        return
                    while seen < len(pre["located"]):
                        located(pre["located"][seen])
                        seen += 1
                    if finished:
                        break
                    self.msgs.put(("status", "Preparing for load: locating the sections on the slide scans (started while you ordered them)... "
                                   f"{seen} of {n_czi} located" + ("\n" + pre["text"] if pre["text"] else "")))
                self.msgs.put(("czi_ready", (stack, dict(pre["problems"]))))
            elif any(s.source_path.lower().endswith(".czi") and s.czi_origin is None for s in todo):
                self.msgs.put(("status", "Preparing for load: starting the Zeiss .czi reader (installed first if it is missing)..."))
                t_locate = time.time()
                try:
                    problems = ns.calibrate_stack_czi(stack, progress=lambda i, n, text: self.msgs.put(("status", text)),
                                                      on_located=located)
                except BaseException as e:   # (a failed install of the reader library, for one)
                    problems = {"the .czi scans": str(e) or type(e).__name__}
                locate_seconds[0] = time.time() - t_locate
                self.msgs.put(("czi_ready", (stack, {k: v for k, v in problems.items() if v})))
            if verify_wanted:         # (also for a stack made from a saved alignment: a position still unchecked there is checked now)
                start_verify()
            # the .czi sections' pictures: the slide scans in parallel (the sections of one scan one after the other: a scan is read by one thread at
            # a time), those the loader made ahead of time being used as they are
            czi_todo = [s for s in todo if s.renderable and s.source_path.lower().endswith(".czi") and s.frame not in posted]
            if czi_todo:
                if early is not None:
                    early["stop"].set()                  # (this takes over from the loader's thread)
                stop_now = threading.Event()

                def got(s, pv, err):
                    if self.stack is not stack:
                        stop_now.set()
                        return
                    if err is not None:
                        self.msgs.put(("status", f"Couldn't read section {s.index + 1}: {err}"))
                        return
                    self.msgs.put(("status", f"Loading section {s.index + 1} of {len(stack.sections)}..."))
                    post(s, pv, False)

                ns.render_czi_previews(stack, czi_todo, czi_scenes, lock, on_done=got, stop=stop_now, workers=PREVIEW_WORKERS)
            for s in todo:
                if self.stack is not stack:
                    return
                if s.frame in posted:
                    continue
                is_frame = not s.renderable
                try:
                    if not is_frame and s.source_path.lower().endswith(".czi"):
                        self.msgs.put(("status", f"Loading section {s.index + 1} of {len(stack.sections)}..."))
                        pv = czi_picture(s)
                    elif is_frame:
                        if frame_boxes is None:
                            frame_boxes = [b for b in ns.read_boxes(stack.jpx_path) if b.kind == b"jp2c"]
                        self.msgs.put(("status", f"Loading section {s.index + 1} of {len(stack.sections)}..."))
                        pv = load_frame_preview(stack.jpx_path, frame_boxes[s.frame], s.frame)
                    else:
                        cached = preview_is_cached(s.source_path)
                        self.msgs.put(("status", f"{'Loading' if cached else 'Building (first time only)'} section {s.index + 1} of {len(stack.sections)}..."))
                        pv = load_preview(s.source_path)
                except Exception as e:
                    self.msgs.put(("status", f"Couldn't read section {s.index + 1}: {e}"))
                    continue
                post(s, pv, is_frame)
            total = time.time() - t_start
            n = max(len(posted), 1)
            print(f"Loaded {os.path.basename(stack.jpx_path)}: {len(posted)} of {len(todo)} sections in {open_seconds + total:.1f} s "
                  f"(reading the NeuroInfo files {open_seconds:.1f} s; the first section was ready {first[0] if first else 0:.1f} s "
                  f"after that" + (f"; locating the sections on the slide scans {locate_seconds[0]:.1f} s" if locate_seconds[0] else "")
                  + f"; {total / n:.2f} s per section, not counting dialogs you answered)", flush=True)
            self.msgs.put(("loaded", (stack, f"Ready. {len(posted)} of {len(todo)} sections loaded.")))

        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "call":
                    payload()
                elif kind == "opened":
                    self.opened(*payload)
                elif kind == "status":
                    self.var_status.set(payload)
                    for item in self.canvas.find_withtag("loadtext"):   # (the message in the middle of the picture follows it)
                        self.canvas.itemconfig(item, text=payload)
                elif kind == "loaded":
                    stack, text = payload
                    if stack is self.stack:   # (a stack opened since has its own loading under way)
                        self.loading_previews = False
                        self.update_cursor()
                        self.update_sections_title()
                        self.refresh_list()
                        self.update_controls()
                        self.var_status.set(text)
                elif kind == "thumb":
                    stack, path, img, geometry = payload
                    if stack is self.stack:
                        self.thumbs[path] = (Image.fromarray(img), geometry)
                        self.draw_thumbnail()
                elif kind == "verified":
                    stack, summary = payload
                    if stack is self.stack:
                        n_bad = len(summary["corrected"]) + len(summary["unverified"])
                        self.var_status.set(f"Section positions checked: {summary['checked'] - n_bad} confirmed, "
                                            f"{len(summary['corrected'])} corrected, {len(summary['unverified'])} unchecked.")
                        self.redraw()
                elif kind == "czi_ready":
                    stack, problems = payload
                    if stack is self.stack:
                        self.calibrating = False
                        self.locating_done = True
                        self.refresh_list()
                        self.update_controls()
                        if problems:
                            few = sum(1 for s in stack.sections if s.source_path.lower().endswith(".czi") and not s.renderable)
                            messagebox.showwarning(
                                "Some sections couldn't be located on the slide scan",
                                f"{few} of {len(stack.sections)} sections {'shows' if few == 1 else 'show'} the JPX's own "
                                f"{'frame, so its alignment' if few == 1 else 'frames, so their alignment'} can't be edited "
                                f"({'its outline' if few == 1 else 'their outlines'} can):\n\n"
                                + "\n".join(f"{k}: {v}" for k, v in problems.items()), parent=self.root)
                elif kind == "preview":
                    stack, idx, pv, n, is_frame = payload
                    if stack is self.stack:
                        self.previews[idx] = pv
                        if is_frame:
                            self.frame_previews.add(idx)
                        self.progress.config(value=n)
                        if not self.levels_set:
                            self.auto_levels(pv[0])
                        self.update_sections_title()
                        sec = self.by_frame(idx)
                        if sec is not None:
                            self.refresh_list_item(sec.index)
                            if sec.index == self.cur:
                                self.update_controls()
                        # (redrawing is the costly part: only when the new picture is one of those on view)
                        if sec is not None and abs(sec.index - self.cur) <= 1:
                            self.redraw()
                elif kind == "progress":
                    i, n, text = payload
                    self.progress.config(maximum=n, value=i)
                    self.var_status.set(text)
                    for item in self.canvas.find_withtag("loadtext"):
                        self.canvas.itemconfig(item, text=text)
                elif kind == "export_done":
                    self.export_finished(*payload)
        except queue.Empty:
            pass
        self.timer("poll", 100, self.poll)

    def auto_levels(self, img):
        for c in range(min(img.shape[2], len(self.levels))):
            v = img[..., c][img[..., c] > 0]
            if v.size:
                # (the white point is raised by half, so that the brightest 0.3 % are at two thirds: the first section alone tends to show
                # brighter than the stack as a whole, which the Auto2 button, once everything has loaded, gets right)
                self.levels[c] = [float(np.percentile(v, 1)), min(float(np.percentile(v, 99.7)) * 1.5, 65535.0)]
                self.chan_vars[c][1].set(self.levels[c][0])
                self.chan_vars[c][2].set(self.levels[c][1])
        self.levels_set = True

    def set_levels(self, how):
        """The Auto1 / Auto2 / Full buttons: "current" = percentiles (1 and 99.7) of the current section's picture, "all" = of all
        the pictures loaded so far, "full" = 0 and 65535 with no analysis. Empty pixels (0) don't count. View only."""
        if not self.stack or not self.stack.sections:
            return
        if how == "full":
            pvs = []
        elif how == "current":
            pvs = [self.previews.get(self.stack.sections[self.cur].frame)]
        else:
            pvs = list(self.previews.values())
        pvs = [pv for pv in pvs if pv is not None]
        if how != "full" and not pvs:
            self.var_status.set("The picture isn't loaded yet.")
            return
        for c in range(len(self.levels)):
            if how == "full":
                lo, hi = 0.0, 65535.0
            else:
                hs = [self.level_hist(img)[c] for img, _, _ in pvs if c < img.shape[2]]
                if not hs:
                    continue
                h = np.sum(hs, axis=0)
                lo, hi = histogram_percentile(h, 1), histogram_percentile(h, 99.7)
                if lo is None:
                    continue
            self.levels[c] = [lo, hi]
            self.chan_vars[c][1].set(lo)
            self.chan_vars[c][2].set(hi)
        self.levels_set = True                    # (the automatic setting from the first picture must not override this)
        self.redraw()

    def level_hist(self, img):
        """The brightness histograms of a section picture, one per channel (channels x LEVEL_BINS counts; empty pixels, 0, are left
        out). Made once per picture: the loading thread asks for it as each picture arrives, so that the Auto buttons only have to
        add up what is there."""
        hit = self.level_cache.get(id(img))
        if hit is not None and hit[0]() is img:
            return hit[1]
        h = np.zeros((img.shape[2], LEVEL_BINS), np.float64)
        for c in range(img.shape[2]):
            v = img[..., c]
            if v.dtype != np.uint16:
                v = np.clip(v, 0, 65535).astype(np.uint16)
            counts = np.bincount(v.ravel(), minlength=65536).astype(np.float64)
            counts[0] = 0
            h[c] = np.bincount(LEVEL_BIN_OF, weights=counts, minlength=LEVEL_BINS)
        try:
            self.level_cache[id(img)] = (weakref.ref(img), h)
        except TypeError:
            pass
        if len(self.level_cache) > 4 * max(len(self.previews), 64):       # (forget those of pictures that are gone)
            self.level_cache = {k: v for k, v in self.level_cache.items() if v[0]() is not None}
        return h

    # ---------------------------------------------------------------- list / controls

    def section_label(self, s):
        marks = ("*" if s.changed else " ") + ("~" if not s.edit.is_identity() else " ") + \
                ("F" if s.edit.flip else " ")
        # "(scene N)": the scene of the slide scan that holds the section (as in the scan, counted from 1; it is not a position)
        scene = ns.czi_scene_if_known(s, self.stack) if self.stack is not None else None
        where = f"  (scene {scene + 1})" if scene is not None else ""
        return f"{marks}  {s.index + 1}. {os.path.splitext(s.name)[0]}{where}"

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        if not self.stack:
            return
        for s in self.stack.sections:
            self.listbox.insert(tk.END, self.section_label(s))
            self.style_list_item(s)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.cur)
        self.listbox.see(self.cur)

    def on_left_resize(self, event=None):
        """The left pane was resized (its bar dragged, or the window): the slide thumbnail follows its width."""
        if self.thumbs:
            if self.thumb_job is not None:
                self.root.after_cancel(self.thumb_job)
            self.thumb_job = self.root.after(60, self.draw_thumbnail)

    def draw_thumbnail(self):
        """The slide scan's picture above the section list, with the outlines of the located sections (the current one bright):
        the sections can be told apart by where they lie on the slide."""
        c = self.thumb_canvas
        sec = self.stack.sections[self.cur] if self.stack and self.stack.sections else None
        path = sec.source_path if sec is not None and sec.source_path in self.thumbs else next(iter(self.thumbs), None)
        if not self.stack or path is None:
            if self.thumb_shown:
                c.pack_forget()
                self.thumb_shown = False
            return
        picture, (gx0, gy0, k) = self.thumbs[path]
        # as wide as the left pane allows (less its padding), but not so tall that the list is squeezed out, nor larger than the picture
        pane_w = max(150, self.left_pane.winfo_width() - 14)
        pane_h = self.left_pane.winfo_height()
        width = min(pane_w, picture.width, int(0.5 * pane_h * picture.width / picture.height) if pane_h > 300 else pane_w)
        width = max(width, 100)
        photo = self.thumb_photos.get((path, width))
        if photo is None:
            if len(self.thumb_photos) > 12:
                self.thumb_photos.clear()
            small = picture.resize((width, max(1, round(picture.height * width / picture.width))), Image.LANCZOS)
            photo = self.thumb_photos[(path, width)] = ImageTk.PhotoImage(small, master=self.root)
        k = k * picture.width / width     # (source pixels per pixel of the picture as it is shown)
        if not self.thumb_shown:
            c.pack(before=self.label_sections, anchor="w", pady=(0, 6))
            self.thumb_shown = True
        c.config(width=photo.width(), height=photo.height())
        c.delete("all")
        c.create_image(0, 0, image=photo, anchor="nw")
        for s in self.stack.sections:
            if s.source_path != path or s.czi_origin is None or not len(s.case_points):
                continue
            X = s.case_points[:, 0] / self.stack.pixel_um + s.czi_origin[0]
            Y = -s.case_points[:, 1] / self.stack.pixel_um + s.czi_origin[1]
            pts = np.column_stack([(X - gx0) / k, (Y - gy0) / k])
            here = s is sec
            c.create_polygon(*pts.ravel().tolist(), outline="#ff3030" if here else "#a08000", fill="", width=2 if here else 1)
            if here:
                cx, cy = pts.mean(0)
                c.create_text(cx, cy, text=str(s.index + 1), fill="#ff3030", font=("Segoe UI", 11, "bold"))

    def update_sections_title(self):
        """"Sections", with how many pictures have been loaded while they are loading."""
        text = "Sections"
        if self.stack and self.loading_previews:
            text += f" (loaded {sum(1 for s in self.stack.sections if self.loaded(s))} of {len(self.stack.sections)})"
        self.var_sections_title.set(text)

    def loaded(self, s):
        """The section's picture has been loaded."""
        return s.frame in self.previews

    def on_list_scroll(self, first, last):
        """The list's view changed: move the scroll bar, which is only there when not all of the list fits."""
        self.list_scroll.set(first, last)
        needed = float(first) > 0.0 or float(last) < 1.0
        if needed != self.list_scroll_shown:
            self.list_scroll_shown = needed
            if needed:
                self.list_scroll.pack(side=tk.RIGHT, fill=tk.Y, before=self.listbox)
            else:
                self.list_scroll.pack_forget()

    def style_list_item(self, s):
        """Red: a slide-scan section that could not be located on its scan (its frame is shown as it is). Grey: no source
        image (only its frame can be shown), or its picture has not been loaded yet."""
        failed = self.locating_done and s.source_path.lower().endswith(".czi") and s.czi_origin is None
        grey = not s.source_path or (self.loading_previews and not self.loaded(s))
        self.listbox.itemconfig(s.index, foreground="#c00000" if failed else "gray60" if grey else "")

    def refresh_list_item(self, i):
        sel = self.listbox.curselection()
        self.listbox.delete(i)
        self.listbox.insert(i, self.section_label(self.stack.sections[i]))
        self.style_list_item(self.stack.sections[i])
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
        if i != self.cur:
            self.sel_outline, self.sel_vertices, self.odrag, self.hover = False, set(), None, None
        self.cur = i
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(i)
        self.listbox.see(i)
        self.update_controls()
        self.redraw()

    def update_controls(self):
        if not self.stack:
            return
        self.draw_thumbnail()
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
                 + ("" if s.renderable else
                    "\nSource image not found: can't edit." if not s.source_path else
                    "\nStill loading: can't edit yet." if self.loading_previews else
                    ("\nCouldn't be located on the slide scan" if s.source_path.lower().endswith(".czi")
                     else "\nSource image is not a JPEG 2000 file")
                    + ": the picture is the JPX's own frame; alignment can't be edited, the outline can."))

    # ---------------------------------------------------------------- editing

    def editable(self):
        if not self.stack or self.busy:
            return None
        s = self.stack.sections[self.cur]
        if not s.renderable or not self.loaded(s):
            return None
        if self.calibrating and self.var_follow.get() and any(
                f.source_path.lower().endswith(".czi") and f.czi_origin is None for f in self.stack.sections[s.index + 1:]):
            self.var_status.set("Following sections are still being located: wait, or turn off \"Edits also move subsequent sections\".")
            return None
        return s

    def followers(self, s):
        """The sections after `s` whose alignment can be edited (their source image was found and can be read)."""
        return [f for f in self.stack.sections[s.index + 1:] if f.renderable]

    def push_undo(self, s):
        """Remember the edit of `s` (and, with "Edits also move subsequent sections" on, of the sections after it, which an edit
        of `s` will move too) as one undo step."""
        step = [(s.frame, copy.copy(s.edit))]
        if self.var_follow.get():
            step += [(f.frame, copy.copy(f.edit)) for f in self.followers(s)]
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
        """Give section `s` the edit. With "Edits also move subsequent sections" on, every later section is moved by the same rigid
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
                self.undo.append([(s.frame, copy.copy(s.edit))])
                s.edit = ns.Edit()
        self.refresh_list()
        self.update_controls()
        self.redraw()

    def do_undo(self):
        if not self.stack or self.busy or not self.undo:
            return
        step = self.undo.pop()
        done = []
        for frame, e in step:
            s = self.by_frame(frame)
            if s is not None:
                s.edit = e
                self.refresh_list_item(s.index)
                done.append(s)
        if not done:
            return
        if done[0].index != self.cur:
            self.select(done[0].index)
        self.changed(done[0])

    def move_section(self, delta):
        """Move the current section up (-1) or down (+1) in the stack."""
        if not self.stack or self.busy:
            return
        i, j = self.cur, self.cur + delta
        if not 0 <= j < len(self.stack.sections):
            return
        self.move_to(i, j)

    def move_to(self, i, j):
        """Take section i out of the order and put it back at position j (the others shift), and show it selected."""
        secs = self.stack.sections
        secs.insert(j, secs.pop(i))
        self.renumber()
        self.select(j)
        self.refresh_list()
        self.var_status.set(f"Moved section {i + 1} to position {j + 1}. Export to save the new order.")

    def list_press(self, event):
        """The start of a possible drag of a row of the section list (a plain click selects it, as before)."""
        i = self.listbox.nearest(event.y)
        if self.stack and not self.busy and 0 <= i < len(self.stack.sections):
            self.list_drag = {"from": i, "y0": event.y, "moved": False, "target": i}
        else:
            self.list_drag = None

    def list_motion(self, event):
        d = self.list_drag
        if d is None:
            return "break"
        if not d["moved"]:
            if abs(event.y - d["y0"]) < 5:
                return "break"
            d["moved"] = True
            self.listbox.config(cursor="sb_v_double_arrow")
        h = self.listbox.winfo_height()
        if event.y < 0:                                  # (the pointer is above or below the list: it scrolls)
            self.listbox.yview_scroll(-1, "units")
        elif event.y > h:
            self.listbox.yview_scroll(1, "units")
        t = max(0, min(self.listbox.nearest(event.y), len(self.stack.sections) - 1))
        if t != d["target"]:
            self.listbox.itemconfig(d["target"], background="")
            d["target"] = t
        if t != d["from"]:
            self.listbox.itemconfig(t, background="#cfe3ff")   # (where the section would go)
        return "break"

    def list_release(self, event):
        d, self.list_drag = self.list_drag, None
        if d is None or not d["moved"]:
            return
        self.listbox.config(cursor="")
        self.listbox.itemconfig(d["target"], background="")
        if d["target"] != d["from"] and self.stack and not self.busy:
            self.move_to(d["from"], d["target"])

    def order_dialog(self):
        """A window with a picture of every section, to put them in order by dragging (the grid of step 1's third tab). Apply makes the
        new order; Cancel leaves things as they were. A double-click applies it too, and goes to that section."""
        if not self.stack or self.busy or self.opening or len(self.stack.sections) < 2:
            return
        stack = self.stack
        secs = list(stack.sections)
        by_frame = {sec.frame: sec for sec in secs}
        dlg = tk.Toplevel(self.root)
        dlg.title("Order the sections")
        dlg.transient(self.root)
        dlg.geometry(f"1000x700+{self.root.winfo_rootx() + 40}+{self.root.winfo_rooty() + 40}")
        top = ttk.Frame(dlg, padding=(12, 8))
        top.pack(fill=tk.X)
        text = tk.StringVar(dlg)
        state = {"closed": False}

        def caption(frame, i):
            sec = by_frame[frame]
            scene = ns.czi_scene_if_known(sec, stack)
            return f"{i + 1}  {os.path.splitext(sec.name)[0][-9:]}", f"scene {scene + 1}" if scene is not None else ""

        grid = OrderGrid(dlg, caption, on_double=lambda frame, i: apply(frame),
                         after_draw=lambda: text.set(f"{len(grid.order)} sections. Drag to reorder; double-click applies and goes to it."))
        grid.order = [sec.frame for sec in secs]

        def close():
            state["closed"] = True
            dlg.destroy()

        def apply(go_to=None):
            if stack is not self.stack:
                return close()
            new = [by_frame[f] for f in grid.order]
            changed = [sec.frame for sec in new] != [sec.frame for sec in secs]
            keep = by_frame[go_to] if go_to is not None else secs[self.cur]
            close()
            if changed:
                stack.sections[:] = new
                self.renumber()
                self.cur = keep.index
                self.refresh_list()
                self.select(self.cur)
                self.var_status.set("Reordered the sections. Export to save the new order.")
            elif go_to is not None:
                self.select(keep.index)

        def thumb(frame):
            """The section as the main view shows it (aligned, at the brightness set), fitted into its thumbnail with its outline."""
            sec = by_frame[frame]
            P = sec.stack_points
            if not len(P):
                return None
            px = stack.pixel_um
            X, Y = P[:, 0] / px, -P[:, 1] / px
            lo, hi = np.array([X.min(), Y.min()]), np.array([X.max(), Y.max()])
            pad = 0.08 * float((hi - lo).max()) + 4
            sc = min(THUMB_W / (hi[0] - lo[0] + 2 * pad), THUMB_H / (hi[1] - lo[1] + 2 * pad))
            ox, oy = THUMB_W / 2 - (lo[0] + hi[0]) / 2 * sc, THUMB_H / 2 - (lo[1] + hi[1]) / 2 * sc
            saved, self.view = self.view, (sc, ox, oy)
            try:
                img, _ = self.warp_section(sec, THUMB_W, THUMB_H)
            finally:
                self.view = saved
            if img is None:
                return None
            rgb = (np.clip(self.colorize(img), 0, 1) * 255).astype(np.uint8)
            cv2.polylines(rgb, [np.column_stack([ox + X * sc, oy + Y * sc]).astype(np.int32)], True, (255, 212, 0), 1, cv2.LINE_AA)
            return ImageTk.PhotoImage(Image.fromarray(rgb), master=self.root)

        skip = set()      # (sections with nothing to show)

        def fill():
            """The thumbnails are made a few at a time as the sections' pictures are loaded (they may still be coming in)."""
            if state["closed"]:
                return
            made = 0
            for frame in grid.order:
                if frame in grid.thumbs or frame in skip:
                    continue
                if frame not in self.previews:
                    continue
                try:
                    ph = thumb(frame)
                except Exception:
                    ph = None
                if ph is None:
                    skip.add(frame)
                else:
                    grid.thumbs[frame] = ph
                made += 1
                if made >= 12:
                    break
            if made:
                grid.draw()
            if len(grid.thumbs) + len(skip) < len(by_frame):
                dlg.after(30 if made else 300, fill)

        ttk.Button(top, text="Reverse order", command=grid.reverse).pack(side=tk.RIGHT)
        ttk.Button(top, text="Original order", command=lambda: (grid.order.sort(), grid.draw())).pack(side=tk.RIGHT, padx=8)
        ttk.Label(top, textvariable=text, foreground="gray30").pack(side=tk.LEFT)
        bottom = ttk.Frame(dlg, padding=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(bottom, text="Apply", command=apply).pack(side=tk.RIGHT)
        ttk.Button(bottom, text="Cancel", command=close).pack(side=tk.RIGHT, padx=8)
        grid.frame.pack(fill=tk.BOTH, expand=True)
        dlg.protocol("WM_DELETE_WINDOW", close)
        dlg.bind("<Escape>", lambda e: close())
        dlg.update_idletasks()
        dlg.grab_set()
        grid.draw()
        fill()

    def delete_section(self):
        """Remove the current section from the stack (after asking); the exported JPX will not contain it."""
        if not self.stack or self.busy:
            return
        secs = self.stack.sections
        if len(secs) <= 1:
            messagebox.showinfo("Delete", "A stack needs at least one section.", parent=self.root)
            return
        s = secs[self.cur]
        if not messagebox.askyesno(
                "Delete section", f"Delete section {s.index + 1} from the stack?\n\n"
                "It is left out when you export. The source image and the original JPX are not touched, but the deletion "
                "can't be undone here (reopen the file to get it back).", icon="warning", parent=self.root):
            return
        i = self.cur
        del secs[i]
        self.stack.deleted += 1
        self.previews.pop(s.frame, None)
        self.frame_previews.discard(s.frame)
        self.undo = [step for step in ([(f, e) for f, e in step if f != s.frame] for step in self.undo) if step]
        self.outline_undo = [u for u in self.outline_undo if u[0] != s.frame]
        self.renumber()
        self.update_sections_title()
        self.cur = min(i, len(secs) - 1)
        self.sel_outline, self.sel_vertices, self.odrag, self.hover = False, set(), None, None
        self.refresh_list()
        self.update_controls()
        self.redraw()
        self.var_status.set(f"Deleted section {i + 1}. Export to save the stack without it.")

    def renumber(self):
        for i, sec in enumerate(self.stack.sections):
            sec.index = i

    def by_frame(self, frame):
        """The section with that frame of the opened JPX (None if it was deleted)."""
        return next((s for s in self.stack.sections if s.frame == frame), None)

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
        if self.odrag:
            self.outline_motion(event)
            return
        d = self.drag
        if not d or not self.view:
            return
        if d["mode"] == "pan":
            s, ox, oy = d["view"]
            self.view = self.clamp_view((s, ox + event.x - d["x"], oy + event.y - d["y"]))
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
        self.root.after_idle(self.refresh_canvas_cursor)   # (after the release has been handled)
        if self.odrag:
            self.outline_release(_event)
            return
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
        self.view = self.clamp_view((s2, event.x - u * s2, event.y - v * s2))
        self.fitted = False
        self.redraw()

    def key(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry)) or not self.stack:
            return
        k = event.keysym
        if event.widget is self.canvas and self.sel_outline:   # (with an outline selected on the image)
            if k == "Delete":
                self.delete_selection()
                return "break"
            if k == "Escape":
                self.sel_outline, self.sel_vertices = False, set()
                self.draw_overlay()
                return "break"
            if k in ("Left", "Right", "Up", "Down") and self.sel_vertices:   # the arrows move the selected vertices
                step = NUDGE_BIG_UM if event.state & 0x1 else NUDGE_UM
                self.nudge_vertices({"Left": -step, "Right": step}.get(k, 0), {"Up": step, "Down": -step}.get(k, 0))
                return "break"
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

    def view_bounds(self):
        """The part of the plane the view may show, in canvas pixels (x0, y0, x1, y1): the canvas, and beyond it whatever
        outlines hang over its edge (so that an overhang can be seen, and never blank space that is not needed)."""
        st = self.stack
        x0, y0, x1, y1 = 0.0, 0.0, float(st.canvas_w), float(st.canvas_h)
        for sec in st.sections:
            P = sec.stack_points
            if len(P):
                u, v = P[:, 0] / st.pixel_um, -P[:, 1] / st.pixel_um
                x0, y0 = min(x0, float(u.min())), min(y0, float(v.min()))
                x1, y1 = max(x1, float(u.max())), max(y1, float(v.max()))
        return x0, y0, x1, y1

    def fit_view(self):
        w, h = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        x0, y0, x1, y1 = self.view_bounds()
        s = min(w / (x1 - x0), h / (y1 - y0))
        self.view = (s, (w - (x1 - x0) * s) / 2 - x0 * s, (h - (y1 - y0) * s) / 2 - y0 * s)
        self.fitted = True

    def clamp_view(self, view, w=None, h=None):
        """`view` (scale, ox, oy) kept so that there is never blank space around the picture that could be avoided: it is not
        zoomed out beyond the whole of view_bounds() fitting the window, and an axis along which that is larger than the window
        cannot be panned past its edge (along one that is smaller, it is centred)."""
        w = w or max(self.canvas.winfo_width(), 50)
        h = h or max(self.canvas.winfo_height(), 50)
        x0, y0, x1, y1 = self.view_bounds()
        s, ox, oy = view
        s = max(s, min(w / (x1 - x0), h / (y1 - y0)))
        out = []
        for size, lo, hi, o in ((w, x0, x1, ox), (h, y0, y1, oy)):
            extent = (hi - lo) * s
            out.append((size - extent) / 2 - lo * s if extent <= size else min(-lo * s, max(o, size - hi * s)))
        return (s, out[0], out[1])

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
                self.view = self.clamp_view((s * ratio, w / 2 - u * s * ratio, h / 2 - v * s * ratio), w, h)
        self.redraw(fit_if_needed=True)

    def warp_section(self, sec, vw, vh):
        """(image vh x vw x C float32 in raw units, coverage mask) of a section in the view."""
        pv = self.previews.get(sec.frame)
        if pv is None:
            return None, None
        img, kx, ky = pv
        s, ox, oy = self.view
        A = np.array([[1 / s, 0, -ox / s], [0, 1 / s, -oy / s], [0, 0, 1]])        # view -> canvas px
        if sec.frame in self.frame_previews:   # the JPX's own frame: already on the canvas, so canvas px = frame px
            M = np.eye(3)
        else:
            R, t = sec.effective()
            M = np.vstack([ns.canvas_to_source_matrix(R, t, self.stack.pixel_um), [0, 0, 1]])  # -> source px
            M[:2, 2] += ns.section_source_offset(sec, self.stack)   # (a .czi's preview is of its scene, not of the slide)
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
        alpha = self.effective_opacity()
        self.image_selected = self.sel_outline
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
                # A true cross-fade in which the empty space outside a section counts as black: where this section is absent the
                # other one is only (1 - opacity) strong, so at 90% a larger neighbouring section is mostly hidden, not shown
                # at full strength around this one.
                out = (np.where(ref_m[..., None], rc, 0.0) * (1 - alpha) + np.where(cur_m[..., None], cc, 0.0) * alpha)
            elif cc is not None:
                out = cc
            elif rc is not None:
                out = rc * (1 - alpha)
        rgb = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        if self.var_crosshair.get():
            self.draw_crosshair(rgb)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb), master=self.root)
        c.create_image(0, 0, image=self.photo, anchor="nw")

        self.last_ref, self.last_alpha = ref, alpha
        self.draw_overlay()

    OUTLINE_EDIT_MIN_OPACITY = 0.9

    def effective_opacity(self):
        """The opacity slider's value, except that while an outline is selected for editing it is at least 90% for the time being
        (the neighbouring section then does not clutter the view); the slider itself is not moved."""
        alpha = float(self.var_opacity.get())
        return max(alpha, self.OUTLINE_EDIT_MIN_OPACITY) if self.sel_outline else alpha

    def sync_picture_to_selection(self):
        """Selecting or deselecting an outline changes the opacity the picture is drawn with: redraw it when that matters.
        Returns True if it redrew (which also redrew the overlay)."""
        if self.sel_outline == self.image_selected or not self.stack or not self.view:
            return False
        if float(self.var_opacity.get()) >= self.OUTLINE_EDIT_MIN_OPACITY:
            self.image_selected = self.sel_outline   # (already at least 90%: nothing changes)
            return False
        self.redraw()
        return True

    def draw_overlay(self):
        """Everything drawn over the picture (canvas edge, outlines, pivot cross, vertex squares); can be redrawn alone."""
        if self.sync_picture_to_selection():
            return
        c = self.canvas
        c.delete("overlay")
        if not self.stack or not self.view:
            return
        vw, vh = c.winfo_width(), c.winfo_height()
        cur, ref, alpha = self.stack.sections[self.cur], self.last_ref, self.last_alpha
        st = self.stack
        x0, y0 = self.um_to_view((0, 0))
        x1, y1 = self.um_to_view((st.canvas_w * st.pixel_um, -st.canvas_h * st.pixel_um))
        c.create_rectangle(x0, y0, x1, y1, outline="gray50", dash=(4, 4), tags="overlay")
        # The outlines (and the pivot cross) fade with their section's opacity: the canvas has no transparency, so each is drawn
        # in its colour dimmed toward the black background by the same weight the image gets (this section: the opacity
        # slider's value, the other one: its complement; with nothing to compare with, this section is at full strength).
        cur_color = shade("#ffe000", alpha if ref is not None else 1.0)
        if ref is not None:
            self.draw_outline(ref, shade("#00e5ff", 1 - alpha), dash=(5, 3))
        self.draw_outline(cur, cur_color)
        px, py = self.um_to_view(cur.center + [cur.edit.dx_um, cur.edit.dy_um])
        c.create_line(px - 8, py, px + 8, py, fill=cur_color, tags="overlay")
        c.create_line(px, py - 8, px, py + 8, fill=cur_color, tags="overlay")
        self.draw_vertex_marks()
        if cur.frame not in self.previews and self.loading_previews:
            box = c.create_text(vw / 2, vh / 2, text="Loading preview...", fill="white", font=("Segoe UI", max(10, min(28, int((vw - 80) / 11))), "bold"), tags="overlay")
            x0, y0, x1, y1 = c.bbox(box)
            c.create_rectangle(x0 - 30, y0 - 18, x1 + 30, y1 + 18, fill="#202020", outline="#808080", tags="overlay")
            c.tag_raise(box)
            c.create_text(vw / 2, y1 + 50, text=self.var_status.get(), fill="white", font=("Segoe UI", 14), width=max(vw - 40, 200),
                          justify="center", tags=("overlay", "loadtext"))

    # ---------------------------------------------------------------- outline editing

    HIT_PX = 8   # how close (screen pixels) a click must be to a vertex or an edge of the outline

    def outline_view_points(self, sec):
        """The outline's vertices in canvas pixels, an (n, 2) array."""
        P = sec.stack_points
        if not len(P) or not self.view:
            return np.zeros((0, 2))
        s, ox, oy = self.view
        px = self.stack.pixel_um
        return np.column_stack([ox + P[:, 0] / px * s, oy - P[:, 1] / px * s])

    def view_to_source(self, sec, x, y):
        """A canvas pixel as a point in the section's source coordinates (um)."""
        s, ox, oy = self.view
        px = self.stack.pixel_um
        stack_pt = np.array([(x - ox) / s * px, -(y - oy) / s * px])
        R, t = sec.effective()
        return np.linalg.solve(R, stack_pt - t)

    def stack_delta_to_source(self, sec, dx_um, dy_um):
        R, _ = sec.effective()
        return np.linalg.solve(R, np.array([dx_um, dy_um]))

    def hit_test(self, x, y):
        """('vertex', i) or ('edge', i, (x, y) nearest point on the edge from vertex i to the next) for a click at canvas pixel
        (x, y) near the current section's outline, else None. A vertex wins over an edge."""
        V = self.outline_view_points(self.stack.sections[self.cur])
        n = len(V)
        if not n:
            return None
        d = np.hypot(V[:, 0] - x, V[:, 1] - y)
        i = int(np.argmin(d))
        if d[i] <= self.HIT_PX:
            return ("vertex", i)
        best = None
        for j in range(n if n > 2 else 1):
            a, b = V[j], V[(j + 1) % n]
            ab = b - a
            L2 = float(ab @ ab)
            u = 0.0 if L2 == 0 else min(max(float((np.array([x, y]) - a) @ ab) / L2, 0.0), 1.0)
            q = a + u * ab
            dist = math.hypot(q[0] - x, q[1] - y)
            if dist <= self.HIT_PX - 2 and (best is None or dist < best[0]):
                best = (dist, j, (float(q[0]), float(q[1])))
        return ("edge", best[1], best[2]) if best else None

    def push_outline_undo(self, sec):
        self.outline_undo.append((sec.frame, sec.case_points.copy()))
        if len(self.outline_undo) > 500:
            self.outline_undo.pop(0)

    def undo_outline(self):
        if not self.outline_undo or not self.stack or self.busy:
            return
        frame, points = self.outline_undo.pop()
        sec = self.by_frame(frame)
        if sec is None:
            return
        i = sec.index
        sec.case_points = points
        self.sel_outline, self.sel_vertices, self.odrag = False, set(), None
        if i != self.cur:
            self.select(i)
        self.refresh_list_item(i)
        self.draw_overlay()

    def draw_vertex_marks(self, provisional=None):
        """White squares on the current outline's vertices while it is selected or the pointer is within reach of it (the
        preview of what a click would select); selected vertices are red, the one under the pointer is orange and larger.
        `provisional` replaces the selection while a rubber band is being dragged. Also updates the Drawing panel's text
        (two lines, so the panel never needs to be wider) and its Undo button."""
        self.refresh_canvas_cursor()   # (the selection or the hover changed)
        if self.sync_picture_to_selection():
            return
        c = self.canvas
        c.delete("vmarks")
        sec = self.stack.sections[self.cur] if self.stack else None
        n = len(sec.case_points) if sec is not None else 0
        hover = self.hover if (sec is not None and n and self.hover is not None) else None
        if sec is not None and n and (self.sel_outline or hover is not None):
            selected = self.sel_vertices if provisional is None else provisional
            V = self.outline_view_points(sec)
            if hover is not None and hover[0] == "edge" and hover[1] < n and n > 1:
                a, b = V[hover[1]], V[(hover[1] + 1) % n]
                c.create_line(a[0], a[1], b[0], b[1], fill="white", width=3, tags=("overlay", "vmarks"))
            for i, (x, y) in enumerate(V):
                red = i in selected
                over = hover is not None and hover[0] == "vertex" and hover[1] == i
                half = 6 if over else 4
                c.create_rectangle(x - half, y - half, x + half, y + half,
                                   fill="red" if red else "#ff9900" if over else "white", outline="black",
                                   tags=("overlay", "vmarks"))
        if sec is None:
            info, sel_text = "", ""
        elif not n:
            info, sel_text = "This section has no outline.", ""
        elif not self.sel_outline:
            info, sel_text = f"Outline: {n} vertices", ""
        else:
            sel = len(self.sel_vertices if provisional is None else provisional)
            info, sel_text = f"Outline selected: {n} vertices", (f"{sel} selected" if sel else "")
        self.var_draw_info.set(info)
        self.var_draw_sel.set(sel_text)
        self.button_undo_outline.config(state="normal" if self.outline_undo and self.stack else "disabled")

    def on_hover(self, event):
        """The pointer moved over the image (no button down): note whether it is within reach of the current outline's vertex
        or edge, and redraw the squares when that changes."""
        if self.odrag or self.drag or not self.stack or not self.view:
            return
        self.mod_shift, self.mod_ctrl = bool(event.state & 0x1), bool(event.state & 0x4)
        self.pointer_in_canvas = True
        sec = self.stack.sections[self.cur]
        hit = self.hit_test(event.x, event.y) if len(sec.case_points) else None
        hover = None if hit is None else (hit[0], hit[1])
        if hover != self.hover:
            self.hover = hover
            self.draw_vertex_marks()
        self.refresh_canvas_cursor()

    def on_leave_canvas(self, _event=None):
        self.pointer_in_canvas = False
        if self.hover is not None:
            self.hover = None
            if self.stack and self.view:
                self.draw_vertex_marks()
        self.refresh_canvas_cursor()

    def on_left_press(self, event):
        """A plain left press: with the current outline selected, or on it, this works on the outline; anywhere else it moves
        the section as before (Shift rotates and Ctrl pans, so those never touch the outline)."""
        mode = "rotate" if event.state & 0x1 else "pan" if event.state & 0x4 else "move"
        if not self.stack or not self.view or mode != "move":
            return self.press(event, mode)
        sec = self.stack.sections[self.cur]
        if not len(sec.case_points):
            return self.press(event, mode)
        hit = self.hit_test(event.x, event.y)
        if not self.sel_outline and hit is None:
            return self.press(event, mode)
        self.canvas.focus_set()
        ctx = {"x0": event.x, "y0": event.y, "moved": False, "kind": "none"}
        if hit is not None and hit[0] == "vertex":
            i = hit[1]
            if not self.sel_outline:
                self.sel_outline, self.sel_vertices = True, {i}
            elif i not in self.sel_vertices:
                self.sel_vertices = {i}
            ctx.update(kind="vertex", clicked=i, start=sec.case_points.copy(), idx=sorted(self.sel_vertices))
        elif not self.sel_outline:
            self.sel_outline, self.sel_vertices = True, set()   # (pressed on a line: just select the outline)
        elif hit is None:
            ctx.update(kind="rubber")
            self.prov_sel = set()
        self.odrag = ctx
        self.draw_vertex_marks()

    def on_double_click(self, event):
        """Double-click on a line of the current outline: add a vertex there."""
        if not self.stack or not self.view or event.state & 0x5:
            return self.on_left_press(event)
        sec = self.stack.sections[self.cur]
        hit = self.hit_test(event.x, event.y) if len(sec.case_points) else None
        if hit is None or hit[0] != "edge":
            return self.on_left_press(event)
        j, (qx, qy) = hit[1], hit[2]
        self.push_outline_undo(sec)
        pts = sec.case_points
        sec.case_points = np.insert(pts, j + 1, self.view_to_source(sec, qx, qy), axis=0)
        self.sel_outline, self.sel_vertices, self.odrag = True, {j + 1}, None
        self.canvas.focus_set()
        self.refresh_list_item(sec.index)
        self.draw_overlay()

    def outline_motion(self, event):
        d = self.odrag
        sec = self.stack.sections[self.cur]
        if not d["moved"] and math.hypot(event.x - d["x0"], event.y - d["y0"]) > 4:
            d["moved"] = True
            if d["kind"] == "vertex":
                self.push_outline_undo(sec)
                sec.case_points = d["start"].copy()
        if not d["moved"]:
            return
        if d["kind"] == "vertex":
            s = self.view[0]
            px = self.stack.pixel_um
            delta = self.stack_delta_to_source(sec, (event.x - d["x0"]) / s * px, -(event.y - d["y0"]) / s * px)
            pts = d["start"].copy()
            pts[d["idx"]] += delta
            sec.case_points = pts
            self.draw_overlay()
        elif d["kind"] == "rubber":
            self.canvas.delete("rubber")
            self.canvas.create_rectangle(d["x0"], d["y0"], event.x, event.y, outline="white", dash=(3, 3),
                                         tags=("overlay", "rubber"))
            V = self.outline_view_points(sec)
            lo_x, hi_x = sorted((d["x0"], event.x))
            lo_y, hi_y = sorted((d["y0"], event.y))
            inside = (V[:, 0] >= lo_x) & (V[:, 0] <= hi_x) & (V[:, 1] >= lo_y) & (V[:, 1] <= hi_y)
            self.prov_sel = set(int(i) for i in np.flatnonzero(inside))
            self.draw_vertex_marks(provisional=self.prov_sel)

    def outline_release(self, event):
        d, self.odrag = self.odrag, None
        sec = self.stack.sections[self.cur]
        if d["kind"] == "vertex":
            if not d["moved"]:
                self.sel_vertices = {d["clicked"]}   # a click on a vertex: just that one
            else:
                self.refresh_list_item(sec.index)
        elif d["kind"] == "rubber":
            self.canvas.delete("rubber")
            if d["moved"]:
                self.sel_vertices = set(self.prov_sel)   # the outline stays selected
            else:
                self.sel_outline, self.sel_vertices = False, set()   # a click on empty image: back to moving the section
        self.prov_sel = set()
        self.draw_overlay()

    def on_right_press(self, event):
        """A right press on a selected vertex of the outline opens a menu to delete it (as in NeuroInfo); anywhere else it
        starts a rotation, as before."""
        if self.stack and self.view and self.sel_outline and self.sel_vertices and not self.odrag and not self.drag:
            sec = self.stack.sections[self.cur]
            hit = self.hit_test(event.x, event.y) if len(sec.case_points) else None
            if hit is not None and hit[0] == "vertex" and hit[1] in self.sel_vertices:
                self.canvas.focus_set()
                n = len(self.sel_vertices)
                menu = tk.Menu(self.root, tearoff=0)
                menu.add_command(label=f"Delete {'point' if n == 1 else f'{n} points'} (can also use DEL key)",
                                 command=self.delete_selection)
                try:
                    menu.tk_popup(event.x_root, event.y_root)
                finally:
                    menu.grab_release()
                return "break"
        return self.press(event, "rotate")

    def delete_selection(self):
        """Delete: the selected vertices, or, with only the outline selected, the whole outline (after asking)."""
        sec = self.stack.sections[self.cur]
        n = len(sec.case_points)
        if not (self.sel_outline and n):
            return
        whole, asked = not self.sel_vertices, False
        if self.sel_vertices and n - len(self.sel_vertices) < 3:
            if not messagebox.askyesno(
                    "Delete the outline?", f"Deleting these {len(self.sel_vertices)} vertices would leave fewer than 3, "
                    "which is no longer an outline. Delete the whole outline instead?", icon="warning", parent=self.root):
                return
            whole, asked = True, True   # (that question already was the warning)
        if whole and not asked and not messagebox.askyesno(
                "Delete the whole outline?", f"Delete the entire outline of section {sec.index + 1} ({n} vertices)?"
                "\n\nThe Undo button in the Outline drawing panel can bring it back.", icon="warning", parent=self.root):
            return
        self.push_outline_undo(sec)
        if whole:
            sec.case_points = np.zeros((0, 2))
            self.sel_outline = False
        else:
            keep = [i for i in range(n) if i not in self.sel_vertices]
            sec.case_points = sec.case_points[keep]
        self.sel_vertices = set()
        self.refresh_list_item(sec.index)
        self.draw_overlay()

    def nudge_vertices(self, dx_um, dy_um):
        sec = self.stack.sections[self.cur]
        self.push_outline_undo(sec)
        pts = sec.case_points.copy()
        pts[sorted(self.sel_vertices)] += self.stack_delta_to_source(sec, dx_um, dy_um)
        sec.case_points = pts
        self.refresh_list_item(sec.index)
        self.draw_overlay()

    def draw_crosshair(self, rgb):
        """One-pixel dashed vertical and horizontal lines meeting at the centre of the stack canvas (so they move with the image
        when it is panned or zoomed, and may be out of view), blended into the picture at 50% white (the canvas cannot draw
        see-through lines, so they are put into the image itself)."""
        h, w = rgb.shape[:2]
        s, ox, oy = self.view
        cx, cy = int(round(ox + self.stack.canvas_w / 2 * s)), int(round(oy + self.stack.canvas_h / 2 * s))
        dash = 3   # pixels on, then as many off
        if 0 <= cx < w:
            on_y = (np.arange(h) // dash) % 2 == 0
            rgb[on_y, cx] = (rgb[on_y, cx] // 2 + 128).astype(np.uint8)
        if 0 <= cy < h:
            on_x = (np.arange(w) // dash) % 2 == 0
            rgb[cy, on_x] = (rgb[cy, on_x] // 2 + 128).astype(np.uint8)

    def draw_outline(self, sec, color, dash=None):
        pts = sec.stack_points
        if len(pts) < 2:
            return
        coords = []
        for p in np.vstack([pts, pts[:1]]):
            coords.extend(self.um_to_view(p))
        self.canvas.create_line(*coords, fill=color, width=2, dash=dash, tags="overlay")

    # ---------------------------------------------------------------- export

    def export_dialog(self):
        if not self.stack or self.busy:
            return
        unchecked = [s for s in self.stack.sections if s.czi_assumed]
        if unchecked:
            # A section's position on its scan can be one that was only assumed (the scan's shared origin): the picture exported is rendered
            # from the scan at that position, so an unchecked one could be a few pixels off. The export waits until they are checked.
            running = False if self.check_incomplete else self.start_verify()
            if running:
                messagebox.showinfo(
                    "Section positions are being checked",
                    f"The positions of {len(unchecked)} section(s) on their slide scans haven't been checked against their own frames yet: "
                    "they are only assumed, and an exported picture would be rendered at that position.\n\nThe check is running (see the "
                    "status line); export when it has finished.", parent=self.root)
                return
            if not messagebox.askyesno(
                    "Section positions not checked",
                    f"The positions of {len(unchecked)} section(s) on their slide scans haven't been checked, and can't be checked "
                    "now (the check is off in File > Options, or it couldn't run). An exported picture is rendered from the scan at the "
                    "assumed position, so it could be a few pixels off.\n\nExport anyway?", icon="warning", parent=self.root):
                return
        print("Preparing for JPX export", flush=True)   # (the checks and questions below can take a few seconds to appear)
        st = self.stack
        try:
            quality = float(self.var_quality.get())
        except ValueError:
            messagebox.showerror("Quality", "Quality must be a number of dB, e.g. 60.", parent=self.root)
            return
        try:
            spacing = float(self.var_spacing.get())
        except ValueError:
            spacing = 0.0
        if spacing <= 0 and not st.z_spacing_um:
            messagebox.showerror("Distance between sections", "Enter the distance between adjacent sections, in \u00b5m (as NeuroInfo asks "
                                 "for when it makes a reconstruction).", parent=self.root)
            return
        if spacing > 0:
            st.z_spacing_um = spacing
        exp = st      # (the stack to export: with the canvas fitted to the sections, a copy of it, when that is asked for)
        if not st.jpx_path and self.var_fit.get():
            try:
                margin = float(self.var_margin.get()) / 100
                if not 0 <= margin <= 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Margin", "The margin must be a number of percent between 0 and 100, e.g. 10.", parent=self.root)
                return
            exp, note = ns.fitted_copy(st, margin)
            print(note, flush=True)
        rerender_all = self.var_rerender.get() or not st.jpx_path   # (a stack made from slide scans has no frames to copy)
        todo = [s for s in st.sections if s.renderable and (rerender_all or s.dirty)]   # (others: frames copied as they are)
        if not st.changed:   # (also when "re-render unchanged sections too" would still write every image)
            same = ("every image would be re-rendered, but with the same placement as before" if rerender_all
                    else "the images would be copied unchanged")
            if not messagebox.askyesno("Nothing changed",
                                       "No section has been edited since this file was opened, so an export would be a copy "
                                       f"of it ({same}).\n\nExport anyway?", parent=self.root):
                return
        bad = exp.out_of_canvas()
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
            initialfile=os.path.basename(st.jpx_path) or ns.next_free_name(st.folder),
            defaultextension=".jpx",   # (the dialog asks before overwriting)
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
                ns.export_stack(exp, out, quality_db=quality, rerender_all=rerender_all,
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
        add_recent(out)   # the exported file joins the File > Open recent list (an existing entry just moves to the top)
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
        self.outline_undo = []
        self.sel_outline, self.sel_vertices, self.odrag = False, set(), None
        self.root.title(f"JPX aligner - {os.path.basename(out)}")
        self.refresh_list()
        self.update_controls()
        self.redraw()

    # ---------------------------------------------------------------- run

    # ---------------------------------------------------------------- Tools > Analyze project

    ANALYSIS_COLUMNS = (
        ("#", 45, "e"), ("Slide (source image)", 260, "w"), ("Scene", 60, "e"), ("Position", 190, "w"),
        ("Centroid in the stack (um)", 190, "e"), ("Centroid in the stack (px)", 190, "e"),
        ("Centroid in the section's own coordinates (um)", 290, "e"), ("Centroid on the slide (px)", 190, "e"),
        ("Rotation (deg)", 110, "e"), ("Mirrored", 80, "w"), ("Outline points", 110, "e"))   # (widths at 96 dpi)

    @staticmethod
    def polygon_centroid(P):
        """The centroid of the area of a closed polygon (the mean of the vertices if the area is degenerate)."""
        P = np.asarray(P, float)
        if len(P) < 3:
            return P.mean(0) if len(P) else None
        x, y = P[:, 0], P[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cross = x * y1 - x1 * y
        area = cross.sum() / 2
        if abs(area) < 1e-9:
            return P.mean(0)
        return np.array([((x + x1) * cross).sum() / (6 * area), ((y + y1) * cross).sum() / (6 * area)])

    def analysis_rows(self):
        """One row per section: where it lies, in the stack and (for a slide scan) on the slide."""
        st = self.stack
        rows = []
        for s in st.sections:
            slide = os.path.basename(s.source_path) if s.source_path else "(source image not found)"
            czi = s.source_path.lower().endswith(".czi")
            scene, where, slide_xy = "", "", ""
            if czi:
                if s.czi_origin is None:
                    where = "not located on the scan"
                else:
                    where = "origin shared with its scan" if s.czi_assumed else "located on the scan"
                    try:
                        scene = str(ns.czi_scene_of(s, st) + 1)
                    except Exception:
                        pass
            elif s.source_path:
                where = "own image file"
            c_src = self.polygon_centroid(s.case_points) if len(s.case_points) else None
            c_stack = self.polygon_centroid(s.stack_points) if len(s.case_points) else None
            stack_um = stack_px = src_um = ""
            if c_stack is not None:
                stack_um = f"{c_stack[0]:.1f}, {c_stack[1]:.1f}"
                stack_px = f"{c_stack[0] / st.pixel_um:.0f}, {-c_stack[1] / st.pixel_um:.0f}"
                src_um = f"{c_src[0]:.1f}, {c_src[1]:.1f}"
                if czi and s.czi_origin is not None:
                    slide_xy = f"{c_src[0] / st.pixel_um + s.czi_origin[0]:.0f}, {-c_src[1] / st.pixel_um + s.czi_origin[1]:.0f}"
            rows.append((s.index + 1, slide, scene, where, stack_um, stack_px, src_um, slide_xy,
                         f"{-s.rotation_deg:+.2f}", "yes" if s.mirrored else "no", len(s.case_points)))
        return rows

    def build_outline_tab(self, parent):
        """The second tab of the analysis window: choose a slide at the left; at the right every outline of that slide, numbered,
        drawn at the coordinates the case file stores (no transform of any kind)."""
        st = self.stack
        tab = ttk.Frame(parent)
        slides = {}
        for s in st.sections:
            slides.setdefault(os.path.basename(s.source_path) if s.source_path else "(source image not found)", []).append(s)
        names = ["All sections"] + list(slides)
        has_raw = any(s.raw_points is not None for s in st.sections)
        left = ttk.Frame(tab, padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Slide").pack(anchor="w")
        lb_frame = ttk.Frame(left)
        lb_frame.pack(fill=tk.Y, expand=True)
        lb = tk.Listbox(lb_frame, width=34, exportselection=False, activestyle="none", font=("Segoe UI", 10))
        sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL, command=lb.yview)
        lb.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        for n in names:
            lb.insert(tk.END, n if n == "All sections" else f"{n}  ({len(slides[n])})")
        right = ttk.Frame(tab, padding=6)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        top = ttk.Frame(right)
        top.pack(fill=tk.X)
        y_down = tk.BooleanVar(tab, False)
        ttk.Label(top, text=(f"The outlines as {st.case_file or 'the case file'} stores them (X, Y in um), with no transform." if has_raw else
                             "This file has no case file: the outlines as stored in it (um), with no transform.")).pack(side=tk.LEFT)
        raw_on = tk.BooleanVar(tab, True)
        rec_on = tk.BooleanVar(tab, True)
        ttk.Checkbutton(top, text="y increases downward", variable=y_down).pack(side=tk.RIGHT)
        ttk.Button(top, text="Fit", width=5, command=lambda: (state.update(zoom=None), redraw())).pack(side=tk.RIGHT, padx=8)
        row2 = ttk.Frame(right)
        row2.pack(fill=tk.X, pady=(4, 0))
        ttk.Checkbutton(row2, text="Raw case-file coordinates (solid lines)", variable=raw_on).pack(side=tk.LEFT)
        ttk.Checkbutton(row2, text="Positions found on the slide scan (dashed lines; the scan's pixel grid, as um)",
                        variable=rec_on).pack(side=tk.LEFT, padx=16)
        row3 = ttk.Frame(right)
        row3.pack(fill=tk.X)
        scene_on = tk.BooleanVar(tab, True)
        ttk.Checkbutton(row3, text="Scenes of the scan (numbered rectangles; one slide at a time)",
                        variable=scene_on).pack(side=tk.LEFT)
        stage_on = tk.BooleanVar(tab, False)
        ttk.Checkbutton(row3, text="Scenes at their stage positions (dashed rectangles, from the scan's metadata)",
                        variable=stage_on).pack(side=tk.LEFT, padx=16)
        row4 = ttk.Frame(right)
        row4.pack(fill=tk.X)
        xml_on = tk.BooleanVar(tab, False)
        ttk.Checkbutton(row4, text="Outlines in reconstruction.xml (dash-dot lines; stack coordinates, where the sections pile up)",
                        variable=xml_on).pack(side=tk.LEFT)
        canvas = tk.Canvas(right, bg="white", highlightthickness=1, highlightbackground="gray70")
        canvas.pack(fill=tk.BOTH, expand=True, pady=4)
        readout = tk.StringVar(tab, "")
        ttk.Label(right, textvariable=readout, foreground="gray30").pack(anchor="w")
        palette = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22")
        state = {"items": [], "view": None, "zoom": None}

        def chosen_sections():
            sel = lb.curselection()
            if not sel or sel[0] == 0:
                return list(st.sections), True
            return slides[names[sel[0]]], False

        def raw(s):
            return s.raw_points if s.raw_points is not None else s.case_points

        def nice_step(span, target=8):
            raw_step = span / target
            mag = 10 ** np.floor(np.log10(raw_step))
            for m in (1, 2, 5, 10):
                if raw_step <= m * mag:
                    return m * mag
            return 10 * mag

        DASH = (1, 2)   # the dashes of the dashed outlines (px on, px off): small, so that the line's detail shows
        DASH_XML = (4, 2, 1, 2)   # dash-dot, for the outlines of reconstruction.xml

        def in_xml(sec):
            """The outline as reconstruction.xml has it (the stack's coordinates, um; y up from the canvas's top-left). A stack
            without the XML (one exported by this tool) gets the same outline computed from the alignment it stores."""
            if sec.xml_points is not None and len(sec.xml_points) >= 2:
                return np.asarray(sec.xml_points, float)
            if len(sec.saved_points) >= 2:
                return (sec.R @ np.asarray(sec.saved_points, float).T).T + sec.t
            return None

        def reconstructed(sec):
            """The outline where the search placed it on the slide scan, in the same units as the raw one (um): the scan's own
            pixel grid times the pixel size. None if the section has no position on a scan."""
            if sec.czi_origin is None or not len(sec.saved_points):
                return None
            pxs = st.source_pixel_um or st.pixel_um
            P = np.asarray(sec.saved_points, float)
            return np.column_stack([(P[:, 0] / st.pixel_um + sec.czi_origin[0]) * pxs, (-P[:, 1] / st.pixel_um + sec.czi_origin[1]) * pxs])

        def redraw(_event=None):
            canvas.delete("all")
            secs, everything = chosen_sections()
            sets = []     # (section, number in the slide, raw outline or None, reconstructed outline or None, xml outline or None)
            for k, s in enumerate(secs, 1):
                r = np.asarray(raw(s), float) if raw_on.get() and len(raw(s)) >= 2 else None
                c = reconstructed(s) if rec_on.get() else None
                x = in_xml(s) if xml_on.get() else None
                if r is not None or c is not None or x is not None:
                    sets.append((s, k, r, c, x))
            state["items"] = []
            if not sets:
                canvas.create_text(20, 20, anchor="nw", text="Nothing to show (switch on one of the sets"
                                   + ("; none of these sections has a position on a scan yet)." if rec_on.get() and not raw_on.get() and not xml_on.get() else ")."))
                return
            # the scenes of the scan (a slide is chosen): rectangles in the slide's own grid, in the color of the section they hold
            scene_rects = []     # (scene number from 1, (x0, y0, x1, y1) in um, color, by stage position?)
            sel = lb.curselection()
            if (scene_on.get() or stage_on.get()) and sel and sel[0] > 0:
                try:
                    path = secs[0].source_path
                    if path.lower().endswith(".czi"):
                        pxs = st.source_pixel_um or st.pixel_um
                        hosts = {}
                        for k, s_ in enumerate(secs, 1):
                            if s_.czi_origin is not None:
                                hosts[ns.czi_scene_of(s_, st)] = k
                        colors = lambda j: palette[(hosts[j] - 1) % len(palette)] if j in hosts else "#9a9a9a"
                        if scene_on.get():
                            for j, (bx, by, bw, bh) in enumerate(ns.czi_scenes(path)):
                                scene_rects.append((j + 1, (bx * pxs, by * pxs, (bx + bw) * pxs, (by + bh) * pxs), colors(j), False))
                        if stage_on.get():
                            for j, (cx_, cy_, sw, sh) in enumerate(ns.czi_scene_stage(path)):
                                scene_rects.append((j + 1, (cx_ - sw / 2, cy_ - sh / 2, cx_ + sw / 2, cy_ + sh / 2), colors(j), True))
                except Exception:
                    scene_rects = []
            allp = np.vstack([p for _, _, r, c, x in sets for p in (r, c, x) if p is not None])
            for _, (a0, b0, a1, b1), _c, _stage in scene_rects:
                allp = np.vstack([allp, [[a0, b0], [a1, b1]]])
            fx0, fy0, fx1, fy1 = allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()
            if fx1 - fx0 < 1e-6: fx1 = fx0 + 1
            if fy1 - fy0 < 1e-6: fy1 = fy0 + 1
            pad = 0.03 * max(fx1 - fx0, fy1 - fy0)
            x0, y0, x1, y1 = state["zoom"] or (fx0 - pad, fy0 - pad, fx1 + pad, fy1 + pad)
            w, h = max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 200)
            ml, mr, mt, mb = 70, 20, 36, 40
            scale = min((w - ml - mr) / (x1 - x0), (h - mt - mb) / (y1 - y0))
            ox = ml + ((w - ml - mr) - (x1 - x0) * scale) / 2
            oy = mt + ((h - mt - mb) - (y1 - y0) * scale) / 2

            def to_canvas(x, y):
                cy = oy + ((y - y0) * scale if y_down.get() else (y1 - y) * scale)
                return ox + (x - x0) * scale, cy

            state["view"] = (x0, y0, x1, y1, scale, ox, oy)
            # axes: a grid with the numbers
            sx, sy = nice_step(x1 - x0), nice_step(y1 - y0)
            for gx in np.arange(np.ceil(x0 / sx) * sx, x1 + 1e-9, sx):
                a_, _ = to_canvas(gx, y0)
                if ml <= a_ <= w - mr:
                    canvas.create_line(a_, mt, a_, h - mb, fill="#e6e6e6")
                    canvas.create_text(a_, h - mb + 12, text=f"{gx:g}", fill="gray30", font=("Segoe UI", 8))
            for gy in np.arange(np.ceil(y0 / sy) * sy, y1 + 1e-9, sy):
                _, b_ = to_canvas(x0, gy)
                if mt <= b_ <= h - mb:
                    canvas.create_line(ml, b_, w - mr, b_, fill="#e6e6e6")
                    canvas.create_text(ml - 6, b_, anchor="e", text=f"{gy:g}", fill="gray30", font=("Segoe UI", 8))
            canvas.create_text(w - mr, h - 8, anchor="e", text="X (um)", fill="gray30", font=("Segoe UI", 8))
            canvas.create_text(ml - 6, 8, anchor="ne", text="Y (um)", fill="gray30", font=("Segoe UI", 8))
            for num, (a0, b0, a1, b1), color, by_stage in scene_rects:
                p0, p1 = to_canvas(a0, b0), to_canvas(a1, b1)
                lx, rx_, ty, by_ = min(p0[0], p1[0]), max(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[1], p1[1])
                if by_stage:     # (dashed, labelled at the bottom right, so that it does not run into the pixel box's label)
                    canvas.create_rectangle(p0[0], p0[1], p1[0], p1[1], outline=color, width=1, dash=(5, 3))
                    canvas.create_text(rx_ - 5, by_ - 3, anchor="se", text=f"scene {num} (stage)", fill=color, font=("Segoe UI", 9, "italic"))
                else:
                    canvas.create_rectangle(p0[0], p0[1], p1[0], p1[1], outline=color, width=1)
                    canvas.create_text(lx + 5, ty + 3, anchor="nw", text=f"scene {num}", fill=color, font=("Segoe UI", 9, "italic"))
            # the outlines, numbered 1, 2, ... in the order of the stack (for All sections: the stack's own numbers); the raw ones
            # are solid, the positions found by the search dashed (finely, so that the detail of the line shows), same color
            for s, k, r, c, x in sets:
                color = palette[(k - 1) % len(palette)]
                label = str(s.index + 1) if everything else str(k)
                for p, dash, size, kind in ((r, None, 13, "raw"), (c, DASH, 11, "found on the scan"), (x, DASH_XML, 10, "reconstruction.xml")):
                    if p is None:
                        continue
                    pts = [v for xy in p for v in to_canvas(*xy)]
                    kw = {"dash": dash} if dash else {}
                    canvas.create_polygon(*pts, outline=color, fill="", width=2 if dash is None else 1, **kw)
                    cx, cy = self.polygon_centroid(p)
                    canvas.create_text(*to_canvas(cx, cy), text=label, fill=color, font=("Segoe UI", size, "bold"))
                    state["items"].append((s, k, p, kind))

        def motion(event):
            v = state["view"]
            if not v:
                return
            x0, y0, x1, y1, scale, ox, oy = v
            x = x0 + (event.x - ox) / scale
            y = (y0 + (event.y - oy) / scale) if y_down.get() else (y1 - (event.y - oy) / scale)
            hit = ""
            for s, k, p, kind in state["items"]:
                if cv2.pointPolygonTest(p.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0:
                    hit = f"   | inside the {kind} outline of section {s.index + 1} of the stack"
                    break
            readout.set(f"X = {x:.1f} um, Y = {y:.1f} um{hit}")

        def zoom(event):
            v = state["view"]
            if not v:
                return
            x0, y0, x1, y1, scale, ox, oy = v
            f = 1 / 1.25 if (event.delta > 0 or getattr(event, "num", 0) == 4) else 1.25
            cx = x0 + (event.x - ox) / scale
            cy = (y0 + (event.y - oy) / scale) if y_down.get() else (y1 - (event.y - oy) / scale)
            state["zoom"] = (cx - (cx - x0) * f, cy - (cy - y0) * f, cx + (x1 - cx) * f, cy + (y1 - cy) * f)
            redraw()

        drag = {}

        def press(event):
            drag["at"] = (event.x, event.y, state["view"])

        def move(event):
            if "at" not in drag or not drag["at"][2]:
                return
            sx_, sy_, v = drag["at"]
            x0, y0, x1, y1, scale, ox, oy = v
            dx = (event.x - sx_) / scale
            dy = (event.y - sy_) / scale * (1 if y_down.get() else -1)
            state["zoom"] = (x0 - dx, y0 - dy, x1 - dx, y1 - dy)
            redraw()

        lb.bind("<<ListboxSelect>>", lambda e: (state.update(zoom=None), redraw()))
        canvas.bind("<Configure>", redraw)
        canvas.bind("<Motion>", motion)
        canvas.bind("<MouseWheel>", zoom)
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", move)
        canvas.bind("<Double-Button-1>", lambda e: (state.update(zoom=None), redraw()))
        for var in (y_down, raw_on, rec_on, scene_on, stage_on, xml_on):
            var.trace_add("write", lambda *_: redraw())
        lb.selection_set(1 if len(names) > 1 else 0)
        tab.redraw = redraw
        return tab

    def analyze_project(self):
        """A window with a table of every section (its slide, position and the coordinates of its outline's centroid)."""
        if not self.stack:
            messagebox.showinfo("Analyze project", "Open a reconstruction first.", parent=self.root)
            return
        st = self.stack
        rows = self.analysis_rows()
        win = tk.Toplevel(self.root)
        win.title(f"Project analysis - {os.path.basename(st.jpx_path)}")
        want = round(sum(c[1] for c in self.ANALYSIS_COLUMNS) * self.root.winfo_fpixels("1i") / 96) + 70   # (wide enough for every column)
        win.geometry(f"{min(want, win.winfo_screenwidth() - 80)}x{min(700, win.winfo_screenheight() - 120)}+30+30")
        slides = {}
        for r in rows:
            slides[r[1]] = slides.get(r[1], 0) + 1
        located = sum(1 for r in rows if r[3] in ("located on the scan", "origin shared with its scan"))
        czi_n = sum(1 for r in rows if r[3].startswith(("located", "origin", "not located")))
        summary = (f"{len(rows)} sections | canvas {st.canvas_w} x {st.canvas_h} px at {st.pixel_um:g} um | "
                   f"{len(slides)} slide(s) / source file(s)" + (f" | {located} of {czi_n} slide-scan sections located on their scans"
                                                                if czi_n else ""))
        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True)
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Sections")
        notebook.add(self.build_outline_tab(notebook), text="Outlines by slide (raw coordinates)")
        ttk.Label(tab1, text=summary, padding=(8, 6)).pack(anchor="w")
        frame = ttk.Frame(tab1)
        frame.pack(fill=tk.BOTH, expand=True, padx=8)
        cols = [c[0] for c in self.ANALYSIS_COLUMNS]
        scale = self.root.winfo_fpixels("1i") / 96
        ttk.Style(win).configure("Analysis.Treeview", rowheight=round(22 * scale))
        tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="extended", style="Analysis.Treeview")
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        state = {"col": None, "reverse": False}

        def sort_by(col):
            idx = cols.index(col)

            def key(item):
                v = tree.set(item, col)
                try:
                    return (0, float(v.split(",")[0]), v)
                except ValueError:
                    return (1, 0.0, v)
            items = list(tree.get_children(""))
            state["reverse"] = (not state["reverse"]) if state["col"] == col else False
            state["col"] = col
            for i, item in enumerate(sorted(items, key=key, reverse=state["reverse"])):
                tree.move(item, "", i)

        for name, width, anchor in self.ANALYSIS_COLUMNS:
            tree.heading(name, text=name, command=lambda n=name: sort_by(n))
            tree.column(name, width=round(width * scale), anchor=anchor, stretch=False)
        for r in rows:
            tree.insert("", tk.END, values=r)

        def as_text(sep):
            lines = [sep.join(cols)] + [sep.join(str(v) for v in r) for r in rows]
            return "\n".join(lines) + "\n"

        def copy_all():
            win.clipboard_clear()
            win.clipboard_append(as_text("\t"))

        def save_csv():
            import csv
            path = filedialog.asksaveasfilename(parent=win, title="Save the table", defaultextension=".csv",
                                                initialfile="project_analysis.csv", filetypes=[("CSV files", "*.csv")])
            if path:
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(cols)
                    w.writerows(rows)

        bar = ttk.Frame(tab1, padding=8)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Copy table", command=copy_all).pack(side=tk.LEFT)
        ttk.Button(bar, text="Save as CSV...", command=save_csv).pack(side=tk.LEFT, padx=6)
        ttk.Label(bar, foreground="gray30",
                  text="Click a column heading to sort. The stack's coordinates are measured from the top-left of the canvas (y up in um, "
                       "down in px); \"on the slide\" is in the slide scan's pixels.").pack(side=tk.LEFT, padx=12)
        ttk.Button(bar, text="Close", command=win.destroy).pack(side=tk.RIGHT)
        win.analysis_tree = tree   # (for tests)

    def go_back_to_ordering(self):
        """Back to the section loader (its ordering tab) with this project: its alignment is saved first, and kept (it is of the images, so
        it holds through another order of the sections)."""
        if not self.project_path:
            messagebox.showinfo("Back to step 3", "Only a stack that came from the section loader (a project) can go back to it.", parent=self.root)
            return
        if self.busy or self.opening:
            return
        if not self.save_project_state(force=True):
            return
        self.back_to_ordering = True
        self.hidden = True
        self.root.withdraw()       # (the window is only hidden, with its pictures, edits and view; what is still loading carries on)
        self.root.quit()           # (this ends the window's loop: the start window shows the loader, and Next brings this one back)

    def show_again(self, sections, spacing=0.0):
        """Brought back by the loader's Next: the sections in the order it gave (those still here), its distance between sections, and the
        window as it was left (zoom, brightness, the pictures already loaded)."""
        self.hidden = False
        self.back_to_ordering = False
        self.project_obj = None                  # (the loader has written the project file since: it is read afresh when saving)
        keep = self.stack.sections[self.cur] if self.stack.sections else None
        named = {id(s) for s in sections}
        self.stack.sections[:] = list(sections) + [s for s in self.stack.sections if id(s) not in named]
        self.renumber()
        if spacing > 0:
            self.stack.z_spacing_um = spacing
            self.var_spacing.set(f"{spacing:g}")
        self.cur = keep.index if keep is not None else 0
        self.refresh_list()
        self.update_sections_title()
        self.select(self.cur)
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def shutdown(self):
        """Close the hidden window for good (the loader's window was closed, or the project changed so that a new one is made)."""
        try:
            self.save_project_state(force=True)
        except Exception:
            pass
        self.stop_event_verify.set()
        for after_id in self.timer_ids.values():
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
        self.stack = None
        self.root.destroy()

    def quit(self):
        self.save_project_state()
        if self.busy:
            if not messagebox.askyesno("Export in progress", "An export is running. Cancel it and quit?",
                                       parent=self.root):
                return
            self.stop_event.set()
        elif not self.confirm_discard():
            return
        for after_id in self.timer_ids.values():
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
        self.root.destroy()
        if self.return_to_launcher:
            # back to the start window: what is still loading in the background is told to stop, and the program goes on
            self.stop_event_verify.set()
            self.stack = None
            return
        if not self._busy:
            # Nothing is left to protect (settings are saved as they change, and cache files are written under a temporary name
            # and then renamed), but background loading may still be mid-task: leave at once instead of waiting for it, so that
            # the console closes with the window. (After an export was cancelled, the normal exit lets it clean up.)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Aligner(sys.argv[1] if len(sys.argv) > 1 else None).run()
