"""A Tk version of the atlas browser's interactive UMAP viewer (first version: the UMAP, a grid of the run's brain sections, colouring
by group / single group(s) / gene / imputed gene, and the hover link between the two).

Why it is quicker than the matplotlib window: every plot here is one image on a Tk canvas that is drawn directly from the cells' positions
(a point is a few pixels written into an array), so a recolour, a zoom or a pan costs the same however many artists the figure would
have had, and each section panel is a plain canvas of its own with its own zoom and pan. The hover ring is a canvas item.

    show_umap_window_tk(adata, abc_cache, imputed_state=None, adata_backed=None, rois=None, run_folder=None, ensure_imputed=None)

blocks until the window is closed, like the matplotlib one. Not yet in this version: more than one gene at once, the UMAP variants,
the group (family) highlight on hover, and the exports.
"""
import colorsys
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from scipy.spatial import cKDTree

import abc_core as core

GREY = np.array([205, 205, 205], np.uint8)
LIGHT_GREY = np.array([222, 222, 222], np.uint8)
RING = "#ff1493"
MAX_COLORED_CATEGORIES = 400
LEVEL_NAMES = {"class": "Class", "subclass": "Subclass", "supertype": "Supertype", "cluster": "Cluster", "leiden": "Leiden"}
TILE_PX = 230
HOVER_RADIUS_PX = 9
SCALE_UNITS = {("standard", "log"): "log1p(expr)", ("standard", "linear"): "expr (raw counts)",
               ("imputed", "log"): "log2 (imputed)", ("imputed", "linear"): "linear (imputed)"}


def rank_color(rank):
    """The browser's colour for a category of this rank (0 = commonest): ten hues stepped by the golden angle, in a vivid and a muted set."""
    hue = ((rank % 10) * 0.6180339887498949) % 1.0
    s, v = (0.90, 0.95) if (rank // 10) % 2 == 0 else (0.45, 0.85)
    return np.array([int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, s, v)], np.uint8)


def viridis_rgb(t):
    from matplotlib import colormaps
    return (colormaps["viridis"](np.clip(t, 0, 1))[:, :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------------- the point canvas

class PointCanvas:
    """Points drawn straight onto a Tk canvas: wheel zoom about the pointer, left- or right-drag pan, never out past the whole (`home`)
    view and never off its edges. `layers` are drawn in order; each is {'x','y','rgb','size'}. Markers (the hover ring), rectangles
    and text labels are canvas items given in data coordinates and repositioned after each draw."""

    MAX_ZOOM = 400.0

    def __init__(self, parent, home, flip_y, width=300, height=300):
        self.canvas = tk.Canvas(parent, width=width, height=height, bg="white", highlightthickness=1, highlightbackground="#bbbbbb")
        self.home = home                            # (xmin, xmax, ymin, ymax)
        self.flip_y = flip_y                        # True: data y grows upward on screen (the UMAP); False: downward (a brain section)
        self.layers = []
        self.cx = (home[0] + home[1]) / 2
        self.cy = (home[2] + home[3]) / 2
        self.scale = 1.0
        self.at_fit = True
        self.photo = None
        self.pending = None
        self.item = self.canvas.create_image(0, 0, anchor="nw")
        self.markers = {}                           # name -> [item, x, y] (x None: hidden)
        self.rects = []                             # [item, x0, y0, x1, y1]
        self.labels = []                            # [item, x, y]
        self.on_motion = None                       # fn(cx, cy) ; fn(None, None) on leave
        self.on_view = None                         # fn() after a redraw
        self.pan_from = None
        c = self.canvas
        c.bind("<Configure>", lambda e: self.on_resize())
        c.bind("<MouseWheel>", lambda e: self.zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8))
        c.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.25))
        c.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 0.8))
        for press, motion, release in (("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>"), ("<ButtonPress-3>", "<B3-Motion>", "<ButtonRelease-3>"),
                                       ("<ButtonPress-2>", "<B2-Motion>", "<ButtonRelease-2>")):
            c.bind(press, self.pan_start)
            c.bind(motion, self.pan_move)
            c.bind(release, self.pan_end)
        c.bind("<Motion>", lambda e: self.on_motion and self.on_motion(e.x, e.y), add="+")
        c.bind("<Leave>", lambda e: self.on_motion and self.on_motion(None, None))
        c.bind("<Double-Button-1>", lambda e: self.fit())

    # ---- geometry

    def size(self):
        return max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)

    def fit_scale(self):
        w, h = self.size()
        x0, x1, y0, y1 = self.home
        return min(w / max(x1 - x0, 1e-9), h / max(y1 - y0, 1e-9))

    def fit(self):
        self.at_fit = True
        self.cx, self.cy = (self.home[0] + self.home[1]) / 2, (self.home[2] + self.home[3]) / 2
        self.scale = self.fit_scale()
        self.request_draw()

    def clamp(self):
        w, h = self.size()
        fs = self.fit_scale()
        self.scale = min(max(self.scale, fs), fs * self.MAX_ZOOM)
        x0, x1, y0, y1 = self.home
        hw, hh = w / (2 * self.scale), h / (2 * self.scale)
        self.cx = (x0 + x1) / 2 if 2 * hw >= x1 - x0 else min(max(self.cx, x0 + hw), x1 - hw)
        self.cy = (y0 + y1) / 2 if 2 * hh >= y1 - y0 else min(max(self.cy, y0 + hh), y1 - hh)

    def on_resize(self):
        if self.at_fit:
            self.fit()
        else:
            self.clamp()
            self.request_draw()

    def to_canvas(self, x, y):
        w, h = self.size()
        sy = -(y - self.cy) if self.flip_y else (y - self.cy)
        return w / 2 + (x - self.cx) * self.scale, h / 2 + sy * self.scale

    def to_data(self, px, py):
        w, h = self.size()
        x = self.cx + (px - w / 2) / self.scale
        dy = (py - h / 2) / self.scale
        return x, self.cy + (-dy if self.flip_y else dy)

    def zoom_at(self, px, py, factor):
        x, y = self.to_data(px, py)
        before = self.scale
        fs = self.fit_scale()
        self.scale = min(max(self.scale * factor, fs), fs * self.MAX_ZOOM)
        if self.scale == before:
            return
        self.at_fit = abs(self.scale - fs) < 1e-9 * fs
        w, h = self.size()
        self.cx = x - (px - w / 2) / self.scale
        dy = (py - h / 2) / self.scale
        self.cy = y + dy if self.flip_y else y - dy
        self.clamp()
        self.request_draw()

    def pan_start(self, e):
        self.pan_from = (e.x, e.y, self.cx, self.cy)

    def pan_move(self, e):
        if self.pan_from is None:
            return
        x0, y0, cx, cy = self.pan_from
        self.cx = cx - (e.x - x0) / self.scale
        dy = (e.y - y0) / self.scale
        self.cy = cy + dy if self.flip_y else cy - dy
        self.at_fit = False
        self.clamp()
        self.request_draw()

    def pan_end(self, e):
        self.pan_from = None

    # ---- drawing

    def set_layers(self, layers):
        self.layers = layers
        self.request_draw()

    def request_draw(self):
        if self.pending is None:
            self.pending = self.canvas.after_idle(self.draw)

    def draw(self):
        self.pending = None
        w, h = self.size()
        if w <= 2 or h <= 2:
            return
        if self.at_fit:
            self.scale = self.fit_scale()
            self.cx, self.cy = (self.home[0] + self.home[1]) / 2, (self.home[2] + self.home[3]) / 2
        zoom = self.scale / self.fit_scale()
        img = np.full((h, w, 3), 255, np.uint8)
        hw, hh = w / (2 * self.scale), h / (2 * self.scale)
        for lay in self.layers:
            d = max(1, int(round(lay["size"] * zoom ** 0.4)))
            pad = d
            x, y = lay["x"], lay["y"]
            m = (x >= self.cx - hw - pad / self.scale) & (x <= self.cx + hw + pad / self.scale) & (y >= self.cy - hh - pad / self.scale) & (y <= self.cy + hh + pad / self.scale)
            idx = np.flatnonzero(m)
            if not len(idx):
                continue
            px = ((x[idx] - self.cx) * self.scale + w / 2).astype(np.int32)
            dy = (y[idx] - self.cy) * self.scale
            py = ((-dy if self.flip_y else dy) + h / 2).astype(np.int32)
            rgb = lay["rgb"][idx]
            lo, hi = -(d // 2), d - d // 2
            r2 = (d / 2.0) ** 2
            for oy in range(lo, hi):
                for ox in range(lo, hi):
                    if d >= 3 and ((ox - (lo + hi - 1) / 2.0) ** 2 + (oy - (lo + hi - 1) / 2.0) ** 2) > r2 + 0.25:
                        continue
                    xx, yy = px + ox, py + oy
                    ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                    img[yy[ok], xx[ok]] = rgb[ok]
        self.photo = ImageTk.PhotoImage(Image.fromarray(img), master=self.canvas)
        self.canvas.itemconfig(self.item, image=self.photo)
        self.reposition()
        if self.on_view:
            self.on_view()

    # ---- items in data coordinates

    def add_marker(self, name, radius=8):
        item = self.canvas.create_oval(0, 0, 0, 0, outline=RING, width=2, state="hidden")
        self.markers[name] = [item, None, None, radius]

    def set_marker(self, name, x, y):
        m = self.markers[name]
        m[1], m[2] = x, y
        self.place_marker(m)

    def place_marker(self, m):
        if m[1] is None:
            self.canvas.itemconfig(m[0], state="hidden")
            return
        px, py = self.to_canvas(m[1], m[2])
        r = m[3]
        self.canvas.coords(m[0], px - r, py - r, px + r, py + r)
        self.canvas.itemconfig(m[0], state="normal")
        self.canvas.tag_raise(m[0])

    def add_rect(self, x0, y0, x1, y1):
        item = self.canvas.create_rectangle(0, 0, 0, 0, outline="#ff8c00", width=2, dash=(6, 3))
        self.rects.append([item, x0, y0, x1, y1])

    def clear_labels(self):
        for it in self.labels:
            self.canvas.delete(it[0])
        self.labels = []

    def add_label(self, text, x, y):
        item = self.canvas.create_text(0, 0, text=text, font=("Segoe UI", 8, "bold"), fill="black")
        self.labels.append([item, x, y])

    def reposition(self):
        for m in self.markers.values():
            self.place_marker(m)
        for it, x0, y0, x1, y1 in self.rects:
            a, b = self.to_canvas(x0, y0)
            c, d = self.to_canvas(x1, y1)
            self.canvas.coords(it, a, b, c, d)
            self.canvas.tag_raise(it)
        w, h = self.size()
        for it, x, y in self.labels:
            px, py = self.to_canvas(x, y)
            if 0 <= px <= w and 0 <= py <= h:
                self.canvas.coords(it, px, py)
                self.canvas.itemconfig(it, state="normal")
                self.canvas.tag_raise(it)
            else:
                self.canvas.itemconfig(it, state="hidden")

    def nearest(self, tree, px, py):
        """Index (into the tree's points) of the point within HOVER_RADIUS_PX of the canvas position, or None."""
        x, y = self.to_data(px, py)
        dist, k = tree.query([x, y], distance_upper_bound=HOVER_RADIUS_PX / self.scale)
        return None if not np.isfinite(dist) else int(k)


# ---------------------------------------------------------------------------------------------------------------------- the window

class UmapWindow:
    def __init__(self, adata, abc_cache, imputed_state, adata_backed, rois, run_folder, ensure_imputed, title, parent=None):
        self.adata, self.abc_cache = adata, abc_cache
        self.imputed_state = imputed_state if imputed_state is not None else {"adata": None, "load_thread": None, "load_error": None}
        self.ensure_imputed = ensure_imputed
        self.own_root = parent is None
        self.root = tk.Tk() if parent is None else tk.Toplevel(parent)
        self.root.title(title)
        self.root.geometry("1750x950")
        loading = ttk.Label(self.root, text="Preparing the UMAP viewer...", font=("Segoe UI", 14), padding=40)
        loading.pack()
        self.root.update()
        self.msgs = queue.Queue()
        self.busy = False
        self.mode = "all"                  # "all" | "single" | "gene" | "imputed"
        self.level = "subclass" if "subclass" in adata.obs.columns else "class"
        self.shown_gene = None
        self.shown_dataset = "standard"
        self.values = None                 # the shown gene's value for each cell (as coloured), or None
        self.tree = None
        self.hover_job = None
        self.load_data(adata_backed, rois)
        loading.destroy()
        self.build()
        self.recolor()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda e: self.close())
        self.poll()

    # ---- data

    def load_data(self, adata_backed, rois):
        a = self.adata
        self.coords = np.asarray(a.obsm["X_umap"], dtype=float)[:, :2]
        self.n = len(self.coords)
        self.has_counts = "counts" in a.layers
        sec_col = core.SECTION_COL
        spatial = core.load_section_spatial_coords(a, self.abc_cache) if sec_col in a.obs.columns else None
        self.section_of = a.obs[sec_col].to_numpy() if sec_col in a.obs.columns else None
        ok = spatial is not None and "x" in spatial.columns and "y" in spatial.columns
        self.cx = pd.to_numeric(spatial["x"], errors="coerce").to_numpy() if ok else None
        self.cy = pd.to_numeric(spatial["y"], errors="coerce").to_numpy() if ok else None
        self.sections = core.sorted_sections_descending(a.obs[sec_col].dropna().unique().tolist()) if ok else []
        self.info = {k: (a.obs[k].astype(str).to_numpy() if k in a.obs.columns else None) for k in LEVEL_NAMES}
        self.levels = [k for k in LEVEL_NAMES if self.info[k] is not None]
        # the section panels' background: every cell of each section (from the unfiltered data), when given
        self.bg = None
        if ok and adata_backed is not None:
            try:
                bsec = core.get_section_labels(adata_backed, self.abc_cache)
                bsp = core.load_section_spatial_coords(adata_backed, self.abc_cache)
                if bsp is not None and "x" in bsp.columns:
                    codes, uniques = pd.factorize(bsec.to_numpy())
                    order = np.argsort(codes, kind="stable")
                    starts = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
                    where = {u: order[starts[i]:starts[i + 1]] for i, u in enumerate(uniques)}
                    self.bg = {"x": pd.to_numeric(bsp["x"], errors="coerce").to_numpy(), "y": pd.to_numeric(bsp["y"], errors="coerce").to_numpy(),
                               "where": where, "levels": {k: (bsp[k].astype(str).to_numpy() if k in bsp.columns else None) for k in LEVEL_NAMES}}
            except Exception as e:
                print(f"Could not load the all-cell background for the section panels: {e}", flush=True)
        self.run_idx = {}
        if ok:
            codes, uniques = pd.factorize(self.section_of)
            for i, u in enumerate(uniques):
                if pd.notna(u):
                    self.run_idx[u] = np.flatnonzero(codes == i)
        self.id_of_cell = {}
        self.rois = []
        for r in rois or []:
            try:
                vals = [float(r[k]) for k in ("x_min", "x_max", "y_min", "y_max")]
            except (TypeError, ValueError, KeyError):
                continue
            if all(np.isfinite(vals)):
                self.rois.append((str(r["section"]), *vals))
        self.cache = {}

    def category_codes(self, level):
        """(codes of each run cell, uniques) of a level; and for the all-cell background (codes, uniques) over the same uniques."""
        key = ("codes", level)
        if key not in self.cache:
            codes, uniques = pd.factorize(self.info[level])
            self.cache[key] = (codes, uniques)
        return self.cache[key]

    def bg_codes(self, level):
        key = ("bgcodes", level)
        if key not in self.cache:
            arr = self.bg["levels"].get(level) if self.bg else None
            if arr is None:
                self.cache[key] = None
            else:
                codes, uniques = self.category_codes(level)
                self.cache[key] = pd.Index(uniques).get_indexer(arr)       # (-1: a group none of the run's cells is in)
        return self.cache[key]

    # ---- the window

    def build(self):
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        # the section panels
        left = ttk.Frame(pw)
        self.tile_scroll = tk.Canvas(left, highlightthickness=0, bg="#f4f4f4")
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tile_scroll.yview)
        self.tile_scroll.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tile_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tile_frame = tk.Frame(self.tile_scroll, bg="#f4f4f4")
        self.tile_window = self.tile_scroll.create_window(0, 0, window=self.tile_frame, anchor="nw")
        self.tile_frame.bind("<Configure>", lambda e: self.tile_scroll.configure(scrollregion=self.tile_scroll.bbox("all")))
        self.tile_scroll.bind("<Configure>", lambda e: self.layout_tiles())
        self.tile_scroll.bind("<MouseWheel>", lambda e: self.tile_scroll.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        pw.add(left, weight=2)
        # the UMAP
        mid = ttk.Frame(pw)
        c = self.coords
        padx = (c[:, 0].max() - c[:, 0].min()) * 0.05 or 1.0
        pady = (c[:, 1].max() - c[:, 1].min()) * 0.05 or 1.0
        self.umap = PointCanvas(mid, (c[:, 0].min() - padx, c[:, 0].max() + padx, c[:, 1].min() - pady, c[:, 1].max() + pady), flip_y=True)
        self.umap.canvas.pack(fill=tk.BOTH, expand=True)
        self.umap.add_marker("hover", 9)
        self.umap.on_motion = self.umap_motion
        self.umap.on_view = None
        self.umap_tree = cKDTree(c)
        pw.add(mid, weight=3)
        side = ttk.Frame(pw, padding=10, width=330)
        side.pack_propagate(False)
        self.build_side(side)
        pw.add(side, weight=0)
        self.build_tiles()

    def build_side(self, f):
        ttk.Label(f, text="Color by", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.var_mode = tk.StringVar(f, "all")
        for value, text in (("all", "All groups"), ("single", "Single group(s)"), ("gene", "Gene"), ("imputed", "Imputed Gene")):
            ttk.Radiobutton(f, text=text, value=value, variable=self.var_mode, command=self.on_mode).pack(anchor="w")
        lv = ttk.LabelFrame(f, text="Level", padding=6)
        lv.pack(fill=tk.X, pady=(8, 0))
        self.var_level = tk.StringVar(f, self.level)
        for k, name in enumerate(self.levels):
            ttk.Radiobutton(lv, text=LEVEL_NAMES[name], value=name, variable=self.var_level, command=self.on_level).grid(row=k // 2, column=k % 2, sticky="w", padx=(0, 14))
        self.box = ttk.Frame(f)
        self.box.pack(fill=tk.X, pady=(8, 0))
        # a group number box (Single group(s))
        self.id_frame = ttk.LabelFrame(self.box, text="Group number(s)", padding=6)
        ttk.Label(self.id_frame, text="One or more, separated by commas (Enter):", foreground="gray30", wraplength=300).pack(anchor="w")
        self.var_ids = tk.StringVar(f, "")
        e = ttk.Entry(self.id_frame, textvariable=self.var_ids)
        e.pack(fill=tk.X, pady=(2, 0))
        e.bind("<Return>", lambda ev: self.recolor())
        self.id_entry = e
        # the gene box
        self.gene_frame = ttk.LabelFrame(self.box, text="Gene", padding=6)
        self.var_gene = tk.StringVar(f, "")
        self.gene_entry = ttk.Entry(self.gene_frame, textvariable=self.var_gene)
        self.gene_entry.pack(fill=tk.X)
        self.gene_entry.bind("<KeyRelease>", self.on_gene_key)
        self.gene_entry.bind("<Return>", lambda ev: self.recolor())
        self.gene_entry.bind("<Down>", lambda ev: self.focus_suggestions())
        self.sugg = tk.Listbox(self.gene_frame, height=6, exportselection=False, activestyle="none")
        self.sugg.bind("<ButtonRelease-1>", lambda ev: self.pick_suggestion())
        self.sugg.bind("<Return>", lambda ev: self.pick_suggestion())
        sc = ttk.Frame(self.gene_frame)
        sc.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(sc, text="Scale:").pack(side=tk.LEFT)
        self.var_scale = tk.StringVar(f, "log")
        for value, text in (("linear", "Linear"), ("log", "Log")):
            ttk.Radiobutton(sc, text=text, value=value, variable=self.var_scale, command=self.on_scale).pack(side=tk.LEFT, padx=(8, 0))
        self.cbar = tk.Canvas(self.gene_frame, width=300, height=70, highlightthickness=0)
        self.cbar.pack(fill=tk.X, pady=(6, 0))
        self.cbar_photo = None
        # the legend for Single group(s)
        self.legend = tk.Canvas(f, width=300, height=10, highlightthickness=0)
        hov = ttk.LabelFrame(f, text="Under the pointer", padding=6)
        hov.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))
        self.var_cell = tk.StringVar(f, "Hover over a cell in the UMAP or a section.")
        ttk.Label(hov, textvariable=self.var_cell, wraplength=300, justify=tk.LEFT).pack(anchor="w")
        self.var_status = tk.StringVar(f, "")
        ttk.Label(f, textvariable=self.var_status, foreground="gray30", wraplength=300, justify=tk.LEFT).pack(anchor="w", side=tk.BOTTOM)
        ttk.Label(f, text="wheel: zoom   drag: pan   double-click: whole view", foreground="gray40", wraplength=300).pack(anchor="w", side=tk.BOTTOM, pady=(0, 6))
        self.update_frames()

    def update_frames(self):
        for w in (self.id_frame, self.gene_frame, self.legend):
            w.pack_forget()
        if self.mode == "single":
            self.id_frame.pack(fill=tk.X)
            self.id_entry.focus_set()
        elif self.mode in ("gene", "imputed"):
            self.gene_frame.pack(fill=tk.X)
            self.gene_entry.focus_set()

    # ---- the section panels

    def build_tiles(self):
        self.tiles = []
        if not self.sections:
            return
        boxes = {}
        for s in self.sections:
            idx = self.run_idx[s]
            xs, ys = self.cx[idx], self.cy[idx]
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.any():
                boxes[s] = (xs[ok].min(), xs[ok].max(), ys[ok].min(), ys[ok].max())
        span = max(max(b[1] - b[0], b[3] - b[2]) for b in boxes.values()) * 1.06 or 1.0
        for s in self.sections:
            if s not in boxes:
                continue
            b = boxes[s]
            mx, my = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
            fr = tk.Frame(self.tile_frame, bg="#f4f4f4")
            token = core.sanitize_section_token(s)
            title = tk.StringVar(fr, token)
            tk.Label(fr, textvariable=title, bg="#f4f4f4", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            pc = PointCanvas(fr, (mx - span / 2, mx + span / 2, my - span / 2, my + span / 2), flip_y=False, width=TILE_PX, height=TILE_PX)
            pc.canvas.pack()
            pc.add_marker("hover", 7)
            tile = {"label": s, "frame": fr, "pc": pc, "title": title, "token": token, "tree": None}
            pc.on_motion = lambda px, py, t=tile: self.tile_motion(t, px, py)
            for r in self.rois:
                if r[0] == str(s) or core.sanitize_section_token(r[0]) == token:
                    pc.add_rect(r[1], r[3], r[2], r[4])
            self.tiles.append(tile)
        self.tile_cols = 0
        self.layout_tiles()

    def layout_tiles(self):
        if not getattr(self, "tiles", None):
            return
        w = max(self.tile_scroll.winfo_width(), TILE_PX)
        cols = max(1, w // (TILE_PX + 10))
        if cols == self.tile_cols:
            return
        self.tile_cols = cols
        for k, t in enumerate(self.tiles):
            t["frame"].grid(row=k // cols, column=k % cols, padx=4, pady=3, sticky="nw")

    def section_tile(self, label):
        for t in self.tiles:
            if t["label"] == label:
                return t
        return None

    # ---- colouring

    def on_mode(self):
        if self.busy:
            self.var_mode.set(self.mode)
            return
        mode = self.var_mode.get()
        if mode == "imputed" and self.imputed_state["adata"] is None:
            ok = bool(self.ensure_imputed(self.imputed_state, self.abc_cache)) if self.ensure_imputed else False
            if not ok:
                self.var_mode.set(self.mode)
                return
        self.mode = mode
        self.update_frames()
        self.recolor()

    def on_level(self):
        if self.busy:
            self.var_level.set(self.level)
            return
        self.level = self.var_level.get()
        if self.mode in ("all", "single"):
            self.recolor()

    def on_scale(self):
        if self.mode in ("gene", "imputed") and self.var_gene.get().strip():
            self.recolor()

    def recolor(self):
        self.var_status.set("")
        self.legend.pack_forget()
        if self.mode == "all":
            self.color_all()
        elif self.mode == "single":
            self.color_single()
        else:
            self.color_gene()

    def apply(self, rgb_run, order_run, bg_rgb_fn, labels=None):
        """Show the colouring. `rgb_run`: (n, 3) colour of each run cell; `order_run`: the order to draw them in (lowest first);
        `bg_rgb_fn(section label)` -> (x, y, rgb) of the section panel's background cells, or None."""
        self.rgb_run = rgb_run
        o = order_run
        self.umap.set_layers([{"x": self.coords[o, 0], "y": self.coords[o, 1], "rgb": rgb_run[o], "size": 2}])
        self.umap.clear_labels()
        for text, x, y in labels or []:
            self.umap.add_label(text, x, y)
        self.umap.reposition()
        self.tile_queue = list(self.tiles)
        self.tile_fn = bg_rgb_fn
        self.tile_o = o
        if not getattr(self, "tile_job", None):
            self.tile_job = self.root.after(1, self.tile_step)

    def tile_step(self):
        self.tile_job = None
        if not self.tile_queue:
            return
        t0 = time.perf_counter()
        while self.tile_queue and time.perf_counter() - t0 < 0.03:
            t = self.tile_queue.pop(0)
            idx = self.run_idx[t["label"]]
            layers = []
            bgl = self.tile_fn(t["label"])
            if bgl is not None:
                layers.append({"x": bgl[0], "y": bgl[1], "rgb": bgl[2], "size": 1})
            key = self.rgb_run
            # the run's own cells of this section, in drawing order (lowest first)
            rank = self.rank_run
            sel = idx[np.argsort(rank[idx], kind="stable")]
            layers.append({"x": self.cx[sel], "y": self.cy[sel], "rgb": key[sel], "size": 1})
            t["pc"].set_layers(layers)
        if self.tile_queue:
            self.tile_job = self.root.after(1, self.tile_step)

    def color_all(self):
        level = self.level
        if self.info.get(level) is None:
            self.var_status.set(f"No '{level}' column on this dataset.")
            return
        codes, uniques = self.category_codes(level)
        counts = np.bincount(codes[codes >= 0], minlength=len(uniques))
        ranked = np.argsort(-counts, kind="stable")[:MAX_COLORED_CATEGORIES]
        lut = np.tile(GREY, (len(uniques), 1))
        for r, u in enumerate(ranked):
            lut[u] = rank_color(r)
        rgb = lut[codes]
        self.rank_run = np.zeros(self.n, np.int32)
        bgc = self.bg_codes(level) if self.bg else None

        def bg_fn(label):
            if not self.bg:
                return None
            w = self.bg["where"].get(label)
            if w is None:
                return None
            c = bgc[w] if bgc is not None else np.full(len(w), -1)
            col = np.where((c >= 0)[:, None], lut[np.maximum(c, 0)], LIGHT_GREY)
            return self.bg["x"][w], self.bg["y"][w], col.astype(np.uint8)

        # each coloured group's number at its centre
        cent = pd.DataFrame({"g": codes, "x": self.coords[:, 0], "y": self.coords[:, 1]})
        cent = cent[cent["g"].isin(ranked)].groupby("g").median()
        labels = []
        for g, row in cent.iterrows():
            m = re.match(r"^\s*(\d+)", str(uniques[g]))
            labels.append((m.group(1) if m else str(uniques[g]), row["x"], row["y"]))
        self.apply(rgb, np.arange(self.n), bg_fn, labels)
        self.umap.current = f"all_{level}"
        n_cat = int((counts > 0).sum())
        self.var_status.set(f"Showing all {n_cat} {LEVEL_NAMES[level].lower()}s" + (f"; the {len(ranked)} largest colored, the rest gray." if n_cat > len(ranked) else " (numbers at each center)."))
        self.values = None

    def parse_ids(self):
        ids, bad = [], []
        for tok in re.split(r"[,\s]+", self.var_ids.get().strip()):
            if not tok:
                continue
            try:
                v = int(tok)
            except ValueError:
                bad.append(tok)
                continue
            if v not in ids:
                ids.append(v)
        return ids, bad

    def color_single(self):
        level = self.level
        if self.info.get(level) is None:
            self.var_status.set(f"No '{level}' column on this dataset.")
            return
        ids, bad = self.parse_ids()
        codes, uniques = self.category_codes(level)
        uid = core.extract_leading_numeric_id(pd.Series(uniques)).to_numpy()
        colors = {t: (np.array([255, 0, 0], np.uint8) if len(ids) == 1 else rank_color(i)) for i, t in enumerate(ids)}
        lut = np.tile(GREY, (len(uniques), 1))
        for t, col in colors.items():
            lut[uid == t] = col
        rgb = lut[codes]
        hit = np.zeros(len(uniques), np.int32)
        for i, t in enumerate(ids):
            hit[uid == t] = i + 1
        self.rank_run = hit[codes].astype(np.int32)
        order = np.argsort(self.rank_run, kind="stable")
        bgc = self.bg_codes(level) if self.bg else None

        def bg_fn(label):
            if not self.bg:
                return None
            w = self.bg["where"].get(label)
            if w is None:
                return None
            c = bgc[w] if bgc is not None else np.full(len(w), -1)
            h = np.where(c >= 0, hit[np.maximum(c, 0)], 0)
            col = np.where((h > 0)[:, None], lut[np.maximum(c, 0)], LIGHT_GREY).astype(np.uint8)
            o = np.argsort(h, kind="stable")
            return self.bg["x"][w][o], self.bg["y"][w][o], col[o]

        self.apply(rgb, order, bg_fn)
        counts = {t: int((uid[np.maximum(codes, 0)] == t)[codes >= 0].sum()) for t in ids}
        parts = []
        if ids:
            parts.append("Highlighting " + ", ".join(f"{t} ({counts[t]} cells)" for t in ids if counts[t]))
            if any(not counts[t] for t in ids):
                parts.append("not found: " + ", ".join(str(t) for t in ids if not counts[t]))
        else:
            parts.append(f"Enter one or more {LEVEL_NAMES[level].lower()} numbers.")
        if bad:
            parts.append("ignored: " + ", ".join(bad))
        self.var_status.set("; ".join(parts) + ".")
        self.draw_legend(ids, colors, counts)
        self.values = None

    def draw_legend(self, ids, colors, counts):
        if len(ids) < 2:
            return
        self.legend.pack(fill=tk.X, pady=(6, 0), after=self.box)
        self.legend.delete("all")
        self.legend.config(height=18 * len(ids) + 4)
        for k, t in enumerate(ids):
            col = "#%02x%02x%02x" % tuple(int(v) for v in colors[t])
            self.legend.create_rectangle(4, 4 + 18 * k, 18, 16 + 18 * k, fill=col, outline="")
            self.legend.create_text(26, 3 + 18 * k, anchor="nw", text=f"{t}  ({counts[t]} cells)", font=("Segoe UI", 9))

    # ---- genes

    def symbols(self):
        return core.gene_symbol_list(self.adata, self.imputed_state, self.mode == "imputed")

    def on_gene_key(self, e):
        self.gene_entry.config(foreground="")
        if e.keysym in ("Return", "Up", "Down", "Escape", "Tab"):
            return
        token = self.var_gene.get().strip()
        matches = core.rank_gene_suggestions(self.symbols(), token, 6) if token else []
        self.sugg.delete(0, tk.END)
        for m in matches:
            self.sugg.insert(tk.END, m)
        if matches:
            self.sugg.config(height=len(matches))
            self.sugg.pack(fill=tk.X, after=self.gene_entry)
        else:
            self.sugg.pack_forget()

    def focus_suggestions(self):
        if self.sugg.winfo_ismapped() and self.sugg.size():
            self.sugg.focus_set()
            self.sugg.selection_clear(0, tk.END)
            self.sugg.selection_set(0)
        return "break"

    def pick_suggestion(self):
        sel = self.sugg.curselection()
        if not sel:
            return
        self.var_gene.set(self.sugg.get(sel[0]))
        self.sugg.pack_forget()
        self.gene_entry.focus_set()
        self.recolor()

    def color_gene(self):
        self.sugg.pack_forget()
        name = self.var_gene.get().strip()
        if not name:
            self.cbar.delete("all")
            self.var_status.set("Enter a gene name.")
            self.color_all_quiet()
            return
        imputed = self.mode == "imputed"
        if not imputed and not self.has_counts:
            self.var_status.set("No raw counts on this dataset (layers['counts'] missing).")
            return
        source = self.imputed_state["adata"] if imputed else self.adata
        _cols, resolved, missing = core.resolve_gene_names(source, [name])
        if missing:
            self.gene_entry.config(foreground="red")
            self.var_status.set(f"'{name}' not found in the {'imputed' if imputed else 'standard'} dataset.")
            return
        gene = resolved[0]
        self.var_gene.set(gene)
        scale = self.var_scale.get()
        adata = self.adata

        def work():
            col = core.find_gene_index(source, gene)
            if imputed:
                full = core._get_full_gene_column(source, col)
                pos = source.obs_names.get_indexer(adata.obs_names)
                vals = np.full(adata.n_obs, np.nan)
                found = pos >= 0
                vals[found] = full[pos[found]]
                if scale == "linear":
                    vals = np.exp2(vals) - 1.0
            else:
                expr = adata.layers["counts"][:, col]
                expr = expr.toarray() if hasattr(expr, "toarray") else expr
                vals = np.asarray(expr).ravel().astype(float)
                if scale == "log":
                    vals = np.log1p(vals)
            return vals

        def done(res):
            if isinstance(res, BaseException):
                self.var_status.set(f"Could not read {gene}: {res}")
                return
            self.show_values(gene, "imputed" if imputed else "standard", scale, res)

        self.job(f"Reading {gene}...", work, done)

    def color_all_quiet(self):
        mode = self.mode
        self.mode = "all"
        self.color_all()
        self.mode = mode

    def show_values(self, gene, dataset, scale, vals):
        if (self.mode == "imputed") != (dataset == "imputed") or self.mode not in ("gene", "imputed"):
            return
        fin = np.isfinite(vals)
        vmin, vmax = (float(np.nanmin(vals)), float(np.nanmax(vals))) if fin.any() else (0.0, 1.0)
        t = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        rgb = viridis_rgb(np.where(fin, t, 0))
        rgb[~fin] = GREY
        order = np.argsort(np.where(fin, vals, -np.inf), kind="stable")
        self.rank_run = np.argsort(order, kind="stable").astype(np.int64)       # the rank of each cell: highest expression on top

        def bg_fn(label):
            if not self.bg:
                return None
            w = self.bg["where"].get(label)
            if w is None:
                return None
            return self.bg["x"][w], self.bg["y"][w], np.tile(LIGHT_GREY, (len(w), 1))

        self.apply(rgb, order, bg_fn)
        self.values = (gene, dataset, scale, vals)
        self.draw_colorbar(vmin, vmax, scale, dataset)
        self.var_status.set(f"Showing {gene}{' [imputed]' if dataset == 'imputed' else ''} (cells outside this run's selection are gray).")

    def draw_colorbar(self, vmin, vmax, scale, dataset):
        c = self.cbar
        c.delete("all")
        w = max(c.winfo_width(), 200)
        ramp = viridis_rgb(np.linspace(0, 1, w - 20))[None, :, :]
        img = np.repeat(ramp, 16, axis=0)
        self.cbar_photo = ImageTk.PhotoImage(Image.fromarray(img), master=c)
        c.create_image(10, 8, anchor="nw", image=self.cbar_photo)
        c.create_text(10, 28, anchor="nw", text=f"{vmin:.2f}", font=("Segoe UI", 9))
        c.create_text(w - 10, 28, anchor="ne", text=f"{vmax:.2f}", font=("Segoe UI", 9))
        c.create_text(w / 2, 50, text=SCALE_UNITS[(dataset, scale)], font=("Segoe UI", 9), fill="gray30")

    # ---- background work

    def job(self, text, work, done):
        if self.busy:
            return False
        self.busy = True
        self.var_status.set(text)
        self.root.config(cursor="watch")

        def target():
            try:
                res = work()
            except BaseException as e:
                res = e
            self.msgs.put((done, res))

        threading.Thread(target=target, daemon=True).start()
        return True

    def poll(self):
        try:
            while True:
                done, res = self.msgs.get_nowait()
                self.busy = False
                self.root.config(cursor="")
                done(res)
        except queue.Empty:
            pass
        for t in self.tiles:                  # (the table of the sections' z values is read in the background: it may arrive later)
            if t["title"].get() == t["token"]:
                z = core.section_z_text(t["label"], self.abc_cache, wait=False, short=True)
                if z:
                    t["title"].set(f"{t['token']}   {z}")
        self.poll_id = self.root.after(300, self.poll)

    # ---- hover and the link between the UMAP and the section panels

    def describe(self, i):
        lines = []
        if self.section_of is not None:
            lines.append(f"Section {core.sanitize_section_token(self.section_of[i])}")
        for k in self.levels:
            lines.append(f"{LEVEL_NAMES[k]}: {self.info[k][i]}")
        if self.values is not None:
            gene, dataset, scale, vals = self.values
            lines.append(f"{gene} = {vals[i]:.3g}" if np.isfinite(vals[i]) else f"{gene} = n/a")
        return "\n".join(lines)

    def show_cell(self, i):
        """Ring the cell in the UMAP and in its section panel (only that one), and describe it."""
        if i is None:
            self.umap.set_marker("hover", None, None)
            for t in self.tiles:
                t["pc"].set_marker("hover", None, None)
            self.var_cell.set("Hover over a cell in the UMAP or a section.")
            return
        self.umap.set_marker("hover", *self.coords[i])
        sec = self.section_of[i] if self.section_of is not None else None
        for t in self.tiles:
            if t["label"] == sec and np.isfinite(self.cx[i]):
                t["pc"].set_marker("hover", self.cx[i], self.cy[i])
            else:
                t["pc"].set_marker("hover", None, None)
        self.var_cell.set(self.describe(i))

    def umap_motion(self, px, py):
        if self.hover_job is not None:
            self.root.after_cancel(self.hover_job)
            self.hover_job = None
        if px is None:
            self.show_cell(None)
            return
        i = self.umap.nearest(self.umap_tree, px, py)
        self.show_cell(i)
        if i is not None and self.section_of is not None:
            self.hover_job = self.root.after(250, lambda: self.reveal_tile(self.section_of[i]))

    def reveal_tile(self, label):
        """Scroll the section panels so this section's is in view."""
        self.hover_job = None
        t = self.section_tile(label)
        if t is None:
            return
        self.root.update_idletasks()
        fr = t["frame"]
        top, h = fr.winfo_y(), fr.winfo_height()
        total = max(self.tile_frame.winfo_height(), 1)
        view_top = self.tile_scroll.canvasy(0)
        view_h = self.tile_scroll.winfo_height()
        if top < view_top or top + h > view_top + view_h:
            self.tile_scroll.yview_moveto(max(0.0, (top - 10) / total))

    def tile_motion(self, tile, px, py):
        if px is None:
            self.show_cell(None)
            return
        idx = self.run_idx[tile["label"]]
        if tile["tree"] is None:
            good = np.flatnonzero(np.isfinite(self.cx[idx]) & np.isfinite(self.cy[idx]))
            tile["good"] = idx[good]
            tile["tree"] = cKDTree(np.column_stack([self.cx[idx][good], self.cy[idx][good]])) if len(good) else False
        if tile["tree"] is False:
            return
        k = tile["pc"].nearest(tile["tree"], px, py)
        self.show_cell(None if k is None else int(tile["good"][k]))

    # ---- closing

    def close(self):
        try:
            self.root.after_cancel(self.poll_id)
        except Exception:
            pass
        if self.own_root:
            try:
                self.root.quit()
            except tk.TclError:
                pass
        self.root.destroy()

    def run(self):
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass
        self.root.lift()
        self.root.focus_force()
        if self.own_root:
            self.root.mainloop()
        else:
            self.root.wait_window(self.root)


def show_umap_window_tk(adata, abc_cache, imputed_state=None, adata_backed=None, rois=None, run_folder=None, ensure_imputed=None, title=None,
                        parent=None):
    """Open the viewer and return when it is closed."""
    if "X_umap" not in adata.obsm:
        print("No UMAP embedding available; skipping the interactive UMAP viewer.")
        return
    win = UmapWindow(adata, abc_cache, imputed_state, adata_backed, rois, run_folder, ensure_imputed, title or "Interactive UMAP viewer", parent)
    win.run()
