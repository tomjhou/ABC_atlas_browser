"""Step 1 of the workflow: find the sections on Zeiss .czi slide scans.

Three stages in one window, shown by a strip along the top:
  1. Load slides      - choose the .czi files (or open a saved project);
  2. Detect outlines  - for each slide, a brightness threshold finds the sections; the outlines can then be merged, deleted
                        and drawn by hand;
  3. Order sections   - thumbnails are dragged into their order along the brain.
"Done" saves the project beside the first slide and hands the sections, in memory, to the alignment tool (no JPX is written).

Run it directly (python section_loader.py [slides...]), or from the start window (workflow.py)."""
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import neuroinfo_stack as ns
import section_project as sp

STAGES = ("Load slides", "Detect outlines", "Order sections", "Align")
ACCENT = "#1f5fa8"
THUMB_W, THUMB_H, CAPTION_H, CELL_PAD = 190, 150, 20, 10
SELECT_COLOR, OUTLINE_COLOR, DRAW_COLOR = "#ff3030", "#ffd400", "#00e5ff"
TINT = np.array([0, 190, 255], np.float32)


def human_gb(n):
    return f"{n / 2 ** 30:.1f} GB"


class SectionLoader:
    def __init__(self, on_done=None, slides=(), project=None):
        self.on_done = on_done
        self.root = tk.Tk()
        self.root.title("Find the sections")
        self.root.geometry("1450x900")
        self.project = project or sp.Project()
        self.infos = {}                 # slide path -> sp.SlideInfo
        self.ov = {}                    # slide path -> (overview gray image, (x0, y0, k)), once loaded
        self.ovc = {}                   # slide path -> the same picture in color (RGB)
        self.stage = 0
        self.msgs = queue.Queue()
        self.busy = 0
        self.result = None
        self.order_key = None           # project.polygons_key() when the order was last set
        # ---- the strip of stages
        self.strip = tk.Frame(self.root, bg="#eef1f5")
        self.strip.pack(fill=tk.X)
        self.strip_labels = []
        for i, name in enumerate(STAGES):
            if i:
                tk.Label(self.strip, text="›", bg="#eef1f5", fg="gray50", font=("Segoe UI", 14)).pack(side=tk.LEFT)
            number = str(i + 1) if i < 3 else "→"
            lab = tk.Label(self.strip, text=f"{number}  {name}", padx=14, pady=8, font=("Segoe UI", 11))
            lab.pack(side=tk.LEFT)
            self.strip_labels.append(lab)
        self.var_help = tk.StringVar(self.root, "")
        tk.Label(self.root, textvariable=self.var_help, bg="#eef1f5", fg="gray30", font=("Segoe UI", 10), anchor="w",
                 padx=14).pack(fill=tk.X)
        # ---- body and footer
        self.body = ttk.Frame(self.root)
        self.body.pack(fill=tk.BOTH, expand=True)
        foot = ttk.Frame(self.root, padding=8)
        foot.pack(fill=tk.X)
        self.var_status = tk.StringVar(self.root, "")
        ttk.Label(foot, textvariable=self.var_status, foreground="gray30").pack(side=tk.LEFT)
        self.btn_next = ttk.Button(foot, text="Next", command=self.go_next)
        self.btn_next.pack(side=tk.RIGHT)
        self.btn_back = ttk.Button(foot, text="Back", command=self.go_back)
        self.btn_back.pack(side=tk.RIGHT, padx=6)
        self.frames = [self.build_load(), self.build_detect(), self.build_order()]
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.after(50, self.poll)
        for p in slides:
            self.project.slides.append(sp.Slide(p))
        self.refresh_slide_list()
        if self.project.slides:
            self.read_infos()
        self.show_stage(0)

    # ------------------------------------------------------------------ plumbing

    def run(self):
        self.root.mainloop()
        return self.result

    def quit(self):
        if self.project.slides and not messagebox.askyesno("Quit", "Close without handing the sections to the alignment tool?\n"
                                                           "(What you have done is saved in the project file when you go on.)",
                                                           parent=self.root):
            return
        self.root.destroy()

    def background(self, work, done):
        """Run work() in a thread, then done(result or exception) in the window's thread."""
        self.busy += 1
        self.root.config(cursor="watch")

        def target():
            try:
                res = work()
            except BaseException as e:   # (shown to the user)
                res = e
            self.msgs.put(("done", (done, res)))

        threading.Thread(target=target, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "done":
                    done, res = payload
                    self.busy -= 1
                    if not self.busy:
                        self.root.config(cursor="")
                    done(res)
        except queue.Empty:
            pass
        self.root.after(50, self.poll)

    def show_stage(self, n):
        self.stage = n
        for i, lab in enumerate(self.strip_labels):
            if i == n:
                lab.config(bg=ACCENT, fg="white", font=("Segoe UI", 11, "bold"))
            else:
                lab.config(bg="#eef1f5", fg="black" if i < n else "gray55", font=("Segoe UI", 11))
        for f in self.frames:
            f.pack_forget()
        self.frames[n].pack(fill=tk.BOTH, expand=True)
        self.btn_back.config(state="normal" if n else "disabled")
        self.btn_next.config(text="Done: open in the alignment tool" if n == 2 else "Next")
        self.var_help.set(("Choose the slide scans (.czi) that hold your sections.",
                           "Set the threshold so that each section is tinted and debris is not, then accept the outlines.",
                           "Drag the pictures into the order of the sections along the brain.")[n])
        if n == 1:
            self.enter_detect()
        elif n == 2:
            self.enter_order()

    def go_back(self):
        if self.stage == 1:
            self.leave_detect()
        if self.stage > 0:
            self.show_stage(self.stage - 1)

    def go_next(self):
        if self.busy:
            return
        if self.stage == 0:
            if not self.project.slides:
                messagebox.showinfo("No slides", "Add at least one .czi slide scan first.", parent=self.root)
                return
            self.show_stage(1)
        elif self.stage == 1:
            self.leave_detect()
            bad = [os.path.basename(s.path) for s in self.project.slides if not s.accepted]
            n = sum(len(s.polygons) for s in self.project.slides)
            if not n:
                messagebox.showinfo("No outlines", "No section outlines yet: accept the outlines of at least one slide.", parent=self.root)
                return
            if bad and not messagebox.askyesno(
                    "Slides without outlines",
                    "The outlines of these slides haven't been accepted, so they will have no sections:\n\n" + "\n".join(bad[:8])
                    + "\n\nGo on anyway?", parent=self.root):
                return
            self.show_stage(2)
        elif self.stage == 2:
            self.finish()

    # ------------------------------------------------------------------ stage 1: load slides

    def build_load(self):
        f = ttk.Frame(self.body, padding=20)
        ttk.Label(f, text="Slide scans (.czi)", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(f, text="Add every scan whose sections you want to align. They are taken in the order listed.",
                  foreground="gray30").pack(anchor="w", pady=(0, 8))
        mid = ttk.Frame(f)
        mid.pack(fill=tk.BOTH, expand=True)
        cols = ("name", "scenes", "size", "pixel")
        self.slide_tree = ttk.Treeview(mid, columns=cols, show="headings", selectmode="extended", height=14)
        for c, t, w, a in (("name", "File", 620, "w"), ("scenes", "Scenes", 80, "e"), ("size", "Size", 90, "e"), ("pixel", "Pixel (um)", 90, "e")):
            self.slide_tree.heading(c, text=t)
            self.slide_tree.column(c, width=w, anchor=a)
        sb = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self.slide_tree.yview)
        self.slide_tree.config(yscrollcommand=sb.set)
        self.slide_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        side = ttk.Frame(mid, padding=(12, 0))
        side.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(side, text="Add slides...", command=self.add_slides).pack(fill=tk.X)
        ttk.Button(side, text="Remove", command=self.remove_slides).pack(fill=tk.X, pady=6)
        ttk.Button(side, text="Move up", command=lambda: self.move_slide(-1)).pack(fill=tk.X)
        ttk.Button(side, text="Move down", command=lambda: self.move_slide(1)).pack(fill=tk.X, pady=6)
        ttk.Separator(side).pack(fill=tk.X, pady=10)
        ttk.Button(side, text="Open a saved project...", command=self.open_project).pack(fill=tk.X)
        return f

    def refresh_slide_list(self):
        self.slide_tree.delete(*self.slide_tree.get_children())
        for i, s in enumerate(self.project.slides):
            info = self.infos.get(s.path)
            self.slide_tree.insert("", tk.END, iid=str(i), values=(
                s.path, info.scenes if info else "...", human_gb(info.size_bytes) if info else "",
                f"{info.pixel_um:g}" if info else ""))

    def read_infos(self):
        todo = [s.path for s in self.project.slides if s.path not in self.infos]
        if not todo:
            return
        self.var_status.set("Reading the slides...")

        def work():
            return [(p, sp.slide_info(p)) for p in todo]

        def done(res):
            if isinstance(res, BaseException):
                messagebox.showerror("Can't read the slides", f"{type(res).__name__}: {res}", parent=self.root)
                self.var_status.set("")
                return
            for p, info in res:
                self.infos[p] = info
            self.refresh_slide_list()
            self.var_status.set(f"{len(self.project.slides)} slide(s).")

        self.background(work, done)

    def add_slides(self):
        paths = filedialog.askopenfilenames(parent=self.root, title="Slide scans", filetypes=[("Zeiss .czi", "*.czi"), ("All files", "*.*")])
        known = {s.path for s in self.project.slides}
        for p in paths:
            p = os.path.normpath(p)
            if p not in known:
                self.project.slides.append(sp.Slide(p))
        self.refresh_slide_list()
        self.read_infos()

    def remove_slides(self):
        gone = {int(i) for i in self.slide_tree.selection()}
        if gone:
            self.project.slides = [s for i, s in enumerate(self.project.slides) if i not in gone]
            self.project.order = []
            self.refresh_slide_list()

    def move_slide(self, d):
        sel = [int(i) for i in self.slide_tree.selection()]
        if len(sel) != 1:
            return
        i = sel[0]
        j = i + d
        if 0 <= j < len(self.project.slides):
            sl = self.project.slides
            sl[i], sl[j] = sl[j], sl[i]
            self.project.order = []
            self.refresh_slide_list()
            self.slide_tree.selection_set(str(j))

    def open_project(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open a saved project",
                                          filetypes=[("Section projects", "*" + sp.PROJECT_SUFFIX), ("All files", "*.*")])
        if not path:
            return
        try:
            proj = sp.Project.load(path)
        except Exception as e:
            messagebox.showerror("Can't open", str(e), parent=self.root)
            return
        missing = [s.path for s in proj.slides if not os.path.isfile(s.path)]
        if missing:
            messagebox.showerror("Slides not found", "These slide scans weren't found:\n\n" + "\n".join(missing[:6]), parent=self.root)
            return
        self.project = proj
        self.infos, self.ov = {}, {}
        self.order_key = proj.polygons_key() if proj.order else None
        self.refresh_slide_list()
        self.read_infos()

    # ------------------------------------------------------------------ stage 2: detect outlines

    def build_detect(self):
        f = ttk.Frame(self.body)
        self.cur = 0                       # the slide being worked on
        self.mode = "threshold"            # "threshold" | "edit" | "draw"
        self.polys = {}                    # slide index -> outlines in overview pixels (what the editor works on)
        self.selected = set()              # indices of the selected outlines of the current slide
        self.undo = {}                     # slide index -> list of earlier outline lists
        self.draw_pts = []
        self.view = None                   # (scale, ox, oy): overview pixel -> canvas pixel
        self.mask = None
        self.base_rgb = None
        self.photo = None
        self.pan = None
        self.job = None
        # ---- the picture
        left = ttk.Frame(f)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        nav = ttk.Frame(left, padding=(8, 6))
        nav.pack(fill=tk.X)
        self.btn_prev = ttk.Button(nav, text="◀", width=3, command=lambda: self.goto_slide(self.cur - 1))
        self.btn_prev.pack(side=tk.LEFT)
        self.var_slide = tk.StringVar(f, "")
        ttk.Label(nav, textvariable=self.var_slide, font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=10)
        self.btn_fwd = ttk.Button(nav, text="▶", width=3, command=lambda: self.goto_slide(self.cur + 1))
        self.btn_fwd.pack(side=tk.LEFT)
        ttk.Button(nav, text="Fit", width=5, command=self.fit_view).pack(side=tk.RIGHT)
        self.var_color = tk.BooleanVar(f, True)
        ttk.Checkbutton(nav, text="Show in color", variable=self.var_color, command=self.on_color).pack(side=tk.RIGHT, padx=14)
        ttk.Label(nav, text="Wheel: zoom   Right button: pan", foreground="gray40").pack(side=tk.RIGHT, padx=10)
        self.canvas = tk.Canvas(left, bg="#202020", highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<ButtonPress-3>", self.pan_start)
        self.canvas.bind("<B3-Motion>", self.pan_move)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Shift-Button-1>", lambda e: self.on_click(e, shift=True))
        self.canvas.bind("<Double-Button-1>", self.on_double)
        self.canvas.bind("<Motion>", self.on_motion)
        for key, fn in (("<Key-c>", lambda e: self.close_polygon()), ("<Key-C>", lambda e: self.close_polygon()),
                        ("<Delete>", lambda e: self.delete_selected()), ("<Escape>", lambda e: self.cancel_draw()),
                        ("<Control-z>", lambda e: self.undo_last())):
            self.root.bind(key, fn)
        # ---- the controls
        right = ttk.Frame(f, padding=12, width=330)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        self.panel_threshold = ttk.Frame(right)
        self.panel_edit = ttk.Frame(right)
        p = self.panel_threshold
        ttk.Label(p, text="Find the sections", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(p, text="Pixels brighter than the threshold are tinted blue. Move the slider until each section is tinted "
                          "as a whole and the debris is not.", wraplength=300, foreground="gray30").pack(anchor="w", pady=(2, 12))
        self.var_thr = tk.IntVar(f, 100)
        self.var_thr_text = tk.StringVar(f, "")
        ttk.Label(p, textvariable=self.var_thr_text).pack(anchor="w")
        self.scale_thr = ttk.Scale(p, from_=0, to=255, command=self.on_threshold)
        self.scale_thr.pack(fill=tk.X, pady=(0, 12))
        self.var_min = tk.DoubleVar(f, 0.1)
        self.var_min_text = tk.StringVar(f, "")
        ttk.Label(p, textvariable=self.var_min_text).pack(anchor="w")
        self.scale_min = ttk.Scale(p, from_=0, to=3, command=self.on_min_area)
        self.scale_min.pack(fill=tk.X, pady=(0, 12))
        self.var_smooth_text = tk.StringVar(f, "")
        ttk.Label(p, textvariable=self.var_smooth_text).pack(anchor="w")
        self.scale_smooth = ttk.Scale(p, from_=0, to=20, command=self.on_smooth)
        self.scale_smooth.pack(fill=tk.X, pady=(0, 12))
        self.var_found = tk.StringVar(f, "")
        ttk.Label(p, textvariable=self.var_found, foreground="gray30").pack(anchor="w")
        ttk.Button(p, text="Accept", command=self.accept).pack(fill=tk.X, pady=(18, 0))
        ttk.Label(p, text="The outline of each tinted region is drawn around its outside; holes inside it (ventricles) stay in.",
                  wraplength=300, foreground="gray40").pack(anchor="w", pady=8)
        p = self.panel_edit
        ttk.Label(p, text="Edit the outlines", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.var_edit_help = tk.StringVar(f, "")
        ttk.Label(p, textvariable=self.var_edit_help, wraplength=300, foreground="gray30").pack(anchor="w", pady=(2, 12))
        self.btn_merge = ttk.Button(p, text="Merge selected", command=self.merge_selected)
        self.btn_merge.pack(fill=tk.X)
        self.btn_delete = ttk.Button(p, text="Delete selected", command=self.delete_selected)
        self.btn_delete.pack(fill=tk.X, pady=6)
        self.btn_add = ttk.Button(p, text="Add an outline...", command=self.start_draw)
        self.btn_add.pack(fill=tk.X)
        self.btn_undo = ttk.Button(p, text="Undo", command=self.undo_last)
        self.btn_undo.pack(fill=tk.X, pady=6)
        ttk.Separator(p).pack(fill=tk.X, pady=12)
        ttk.Button(p, text="Change the threshold", command=self.back_to_threshold).pack(fill=tk.X)
        ttk.Label(p, text="Click an outline to select it; shift-click adds to the selection. Merge joins two outlines along "
                          "their closest vertices (the two halves of a section). To draw one, click its points; double-click the "
                          "last point or press C to close it; Esc cancels.", wraplength=300, foreground="gray40").pack(anchor="w", pady=10)
        return f

    # -- entering and leaving the stage

    def enter_detect(self):
        self.read_infos()
        self.cur = min(self.cur, max(len(self.project.slides) - 1, 0))
        self.goto_slide(self.cur, force=True)

    def leave_detect(self):
        self.sync_project()

    def slide(self):
        return self.project.slides[self.cur]

    def goto_slide(self, i, force=False):
        if not self.project.slides or not (0 <= i < len(self.project.slides)):
            return
        if not force and i != self.cur:
            self.sync_project()
        self.cur = i
        s = self.slide()
        self.selected, self.draw_pts = set(), []
        self.btn_prev.config(state="normal" if i else "disabled")
        self.btn_fwd.config(state="normal" if i < len(self.project.slides) - 1 else "disabled")
        self.var_slide.set(f"Slide {i + 1} of {len(self.project.slides)}:  {os.path.basename(s.path)}")
        if s.path in self.ov:
            self.slide_ready()
        else:
            self.base_rgb = None
            self.canvas.delete("all")
            self.canvas.create_text(30, 30, anchor="nw", fill="white", text="Loading the slide picture...", font=("Segoe UI", 12))
            path = s.path
            self.var_status.set("Loading the slide picture...")

            def done(res):
                if isinstance(res, BaseException):
                    messagebox.showerror("Can't read the slide", f"{type(res).__name__}: {res}", parent=self.root)
                    return
                self.ov[path], self.ovc[path] = res[:2], res[2]
                self.var_status.set("")
                if self.stage == 1 and self.slide().path == path:
                    self.slide_ready()

            self.background(lambda: sp.overview(path), done)

    def slide_ready(self):
        s = self.slide()
        img, geo = self.ov[s.path]
        self.set_base()
        if s.threshold < 0:
            s.threshold = sp.auto_threshold(img)
        if self.cur not in self.polys:
            self.polys[self.cur] = [self.to_overview(p) for p in s.polygons]
        self.set_mode("edit" if s.accepted else "threshold")
        self.fit_view()

    def set_base(self):
        """The picture shown: the slide in color, or in gray (the picture the threshold is applied to)."""
        s = self.slide()
        gray = self.ov[s.path][0]
        color = self.ovc.get(s.path)
        self.base_rgb = color if self.var_color.get() and color is not None else np.repeat(gray[:, :, None], 3, axis=2)

    def on_color(self):
        if self.base_rgb is not None and self.slide().path in self.ov:
            self.set_base()
            self.redraw()

    def to_overview(self, poly):
        x0, y0, k = self.ov[self.slide().path][1]
        return (poly - np.array([x0, y0])) / k

    def to_slide(self, poly, path=None):
        x0, y0, k = self.ov[path or self.slide().path][1]
        return poly * k + np.array([x0, y0])

    def sync_project(self):
        """Write the editor's outlines of every slide visited back into the project, and save it."""
        for i, polys in self.polys.items():
            s = self.project.slides[i]
            if s.path in self.ov:
                s.polygons = [self.to_slide(p, s.path) for p in polys]
        if self.project.slides and any(s.polygons for s in self.project.slides):
            try:
                self.project.save(self.project.path or self.project.default_path())
            except OSError as e:
                self.var_status.set(f"Couldn't save the project: {e}")

    # -- modes

    def set_mode(self, mode):
        self.mode = mode
        self.draw_pts = []
        self.panel_threshold.pack_forget()
        self.panel_edit.pack_forget()
        s = self.slide()
        if mode == "threshold":
            self.panel_threshold.pack(fill=tk.X)
            self.scale_thr.set(s.threshold)
            self.scale_min.set(self.min_area_mm2())
            self.scale_smooth.set(s.smooth)
            self.update_labels()
            self.recompute_mask()
        else:
            self.panel_edit.pack(fill=tk.X)
            self.mask = None
        self.canvas.config(cursor="crosshair" if mode == "draw" else "arrow")
        self.update_edit_help()
        self.redraw()

    def px_mm2(self):
        """Area of an overview pixel in mm^2 for the current slide."""
        info = self.infos.get(self.slide().path)
        k = self.ov[self.slide().path][1][2]
        side = k * (info.pixel_um if info else 0.65) / 1000
        return side * side

    def min_area_mm2(self):
        return self.slide().min_area * self.px_mm2()

    def update_labels(self):
        s = self.slide()
        self.var_thr_text.set(f"Brightness threshold: {s.threshold}")
        self.var_min_text.set(f"Ignore regions smaller than {self.min_area_mm2():.2f} mm²")
        um = s.smooth * self.ov[s.path][1][2] * (self.infos[s.path].pixel_um if s.path in self.infos else 0.65)
        self.var_smooth_text.set("Smoothing: none" if not s.smooth else f"Smoothing: {s.smooth} px ({um:.0f} µm), outward")

    def on_threshold(self, v):
        if self.mode != "threshold" or self.base_rgb is None:
            return
        self.slide().threshold = int(round(float(v)))
        self.update_labels()
        self.schedule_mask()

    def on_min_area(self, v):
        if self.mode != "threshold" or self.base_rgb is None:
            return
        self.slide().min_area = max(int(round(float(v) / self.px_mm2())), 1)
        self.update_labels()
        self.schedule_mask()

    def on_smooth(self, v):
        if self.mode != "threshold" or self.base_rgb is None:
            return
        n = int(round(float(v)))
        if n != self.slide().smooth:
            self.slide().smooth = n
            self.update_labels()
            self.schedule_mask()

    def schedule_mask(self):
        if self.job is not None:
            self.root.after_cancel(self.job)
        self.job = self.root.after(25, self.recompute_mask)

    def recompute_mask(self):
        self.job = None
        if self.base_rgb is None or self.mode != "threshold":
            return
        s = self.slide()
        img = self.ov[s.path][0]
        self.mask = sp.detect_mask(img, s.threshold, s.min_area, s.smooth)
        n, _ = cv2.connectedComponents(self.mask, connectivity=8)
        self.var_found.set(f"{n - 1} region(s) found")
        self.redraw()

    def accept(self):
        s = self.slide()
        if self.mask is None:
            self.recompute_mask()
        polys = self.sorted_polys(sp.mask_to_polygons(self.mask))
        self.polys[self.cur] = polys
        self.undo[self.cur] = []
        s.accepted = True
        self.selected = set()
        self.sync_project()
        self.set_mode("edit")

    def sorted_polys(self, polys):
        """The outlines in the order of the scenes of the current slide's .czi (see sp.sort_by_scene)."""
        boxes = ns.czi_scenes(self.slide().path)
        return sp.sort_by_scene(polys, boxes, self.to_slide)[0]

    def back_to_threshold(self):
        if self.polys.get(self.cur) and not messagebox.askyesno(
                "Change the threshold", "Finding the sections again replaces the outlines of this slide, including any you have edited. Go on?",
                parent=self.root):
            return
        self.slide().accepted = False
        self.set_mode("threshold")

    def update_edit_help(self):
        n = len(self.polys.get(self.cur, []))
        sel = len(self.selected)
        if self.mode == "draw":
            self.var_edit_help.set(f"Drawing: {len(self.draw_pts)} point(s). Double-click the last point, or press C, to close the "
                                   "outline; Esc cancels.")
        else:
            self.var_edit_help.set(f"{n} outline(s), numbered in the order of the scenes of the .czi; {sel} selected.")
        for b in (self.btn_merge,):
            b.config(state="normal" if sel >= 2 and self.mode == "edit" else "disabled")
        self.btn_delete.config(state="normal" if sel and self.mode == "edit" else "disabled")
        self.btn_add.config(state="normal" if self.mode == "edit" else "disabled")
        self.btn_undo.config(state="normal" if self.undo.get(self.cur) and self.mode == "edit" else "disabled")

    # -- editing the outlines

    def push_undo(self):
        self.undo.setdefault(self.cur, []).append([p.copy() for p in self.polys[self.cur]])

    def undo_last(self):
        if self.stage != 1 or self.mode != "edit" or not self.undo.get(self.cur):
            return
        self.polys[self.cur] = self.undo[self.cur].pop()
        self.selected = set()
        self.update_edit_help()
        self.redraw()

    def merge_selected(self):
        idx = sorted(self.selected)
        if len(idx) < 2:
            return
        self.push_undo()
        polys = self.polys[self.cur]
        merged = polys[idx[0]]
        for i in idx[1:]:
            merged = sp.merge_polygons(merged, polys[i])
        keep = [p for i, p in enumerate(polys) if i not in self.selected] + [merged]
        self.polys[self.cur] = self.sorted_polys(keep)
        self.selected = {next(i for i, p in enumerate(self.polys[self.cur]) if p is merged)}
        self.update_edit_help()
        self.redraw()

    def delete_selected(self):
        if self.stage != 1 or self.mode != "edit" or not self.selected:
            return
        self.push_undo()
        self.polys[self.cur] = [p for i, p in enumerate(self.polys[self.cur]) if i not in self.selected]
        self.selected = set()
        self.update_edit_help()
        self.redraw()

    def start_draw(self):
        self.mode = "draw"
        self.draw_pts = []
        self.selected = set()
        self.canvas.config(cursor="crosshair")
        self.canvas.focus_set()
        self.update_edit_help()
        self.redraw()

    def cancel_draw(self):
        if self.stage == 1 and self.mode == "draw":
            self.mode = "edit"
            self.draw_pts = []
            self.canvas.config(cursor="arrow")
            self.update_edit_help()
            self.redraw()

    def close_polygon(self):
        if self.stage != 1 or self.mode != "draw":
            return
        if len(self.draw_pts) < 3:
            self.var_status.set("An outline needs at least 3 points.")
            return
        self.push_undo()
        new = np.array(self.draw_pts, float)
        self.polys[self.cur] = self.sorted_polys(self.polys[self.cur] + [new])
        self.selected = {next(i for i, p in enumerate(self.polys[self.cur]) if p is new)}
        self.mode = "edit"
        self.draw_pts = []
        self.canvas.config(cursor="arrow")
        self.update_edit_help()
        self.redraw()

    # -- the picture

    def canvas_size(self):
        return max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)

    def fit_view(self):
        if self.base_rgb is None:
            return
        h, w = self.base_rgb.shape[:2]
        cw, ch = self.canvas_size()
        s = min(cw / w, ch / h) * 0.98
        self.view = (s, (cw - w * s) / 2, (ch - h * s) / 2)
        self.redraw()

    def to_canvas(self, pts):
        s, ox, oy = self.view
        return np.asarray(pts, float) * s + np.array([ox, oy])

    def to_image(self, x, y):
        s, ox, oy = self.view
        return (x - ox) / s, (y - oy) / s

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.stage != 1 or self.base_rgb is None:
            return
        if self.view is None:
            self.fit_view()
            return
        cw, ch = self.canvas_size()
        s, ox, oy = self.view
        img = self.base_rgb
        if self.mode == "threshold" and self.mask is not None:
            tinted = img.astype(np.float32)
            m = self.mask.astype(bool)
            color = bool(self.var_color.get()) and self.slide().path in self.ovc
            a = 0.3 if color else 0.55      # (lighter over the colors, with the edge of the region drawn solid)
            tinted[m] = tinted[m] * (1 - a) + TINT * a
            if color:
                tinted[m & ~cv2.erode(self.mask, np.ones((3, 3), np.uint8)).astype(bool)] = TINT
            img = tinted.astype(np.uint8)
        M = np.array([[s, 0, ox], [0, s, oy]], np.float32)
        out = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self.photo = ImageTk.PhotoImage(Image.fromarray(out))
        c.create_image(0, 0, image=self.photo, anchor="nw")
        if self.mode != "threshold":
            boxes = ns.czi_scenes(self.slide().path)
            for i, p in enumerate(self.polys.get(self.cur, [])):
                pts = self.to_canvas(p)
                sel = i in self.selected
                color = SELECT_COLOR if sel else OUTLINE_COLOR
                c.create_polygon(*pts.ravel(), outline=color, fill="", width=3 if sel else 2)
                cx, cy = self.to_canvas(sp.polygon_centroid(p))
                c.create_text(cx, cy, text=str(i + 1), fill=color, font=("Segoe UI", 16, "bold"))
                sx, sy = self.to_slide(sp.polygon_centroid(p))
                scene = sp.scene_of(boxes, sx, sy) + 1
                if scene != i + 1:     # (the number is the place in the scene order; where it differs, say which scene it is in)
                    c.create_text(cx, cy + 20, text=f"scene {scene}", fill=color, font=("Segoe UI", 9))
        if self.mode == "draw" and self.draw_pts:
            pts = self.to_canvas(self.draw_pts)
            if len(pts) > 1:
                c.create_line(*pts.ravel(), fill=DRAW_COLOR, width=2)
            for x, y in pts:
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=DRAW_COLOR, outline="")

    def on_wheel(self, e):
        if self.base_rgb is None or self.view is None:
            return
        s, ox, oy = self.view
        f = 1.25 if e.delta > 0 else 1 / 1.25
        s2 = min(max(s * f, 0.05), 40)
        f = s2 / s
        self.view = (s2, e.x - (e.x - ox) * f, e.y - (e.y - oy) * f)
        self.redraw()

    def pan_start(self, e):
        self.pan = (e.x, e.y, self.view)

    def pan_move(self, e):
        if self.pan and self.view:
            x, y, (s, ox, oy) = self.pan
            self.view = (s, ox + e.x - x, oy + e.y - y)
            self.redraw()

    def on_click(self, e, shift=False):
        if self.stage != 1 or self.base_rgb is None or self.view is None:
            return
        self.canvas.focus_set()
        x, y = self.to_image(e.x, e.y)
        if self.mode == "draw":
            self.draw_pts.append((x, y))
            self.update_edit_help()
            self.redraw()
        elif self.mode == "edit":
            hit = None
            best = None
            for i, p in enumerate(self.polys.get(self.cur, [])):
                if cv2.pointPolygonTest(p.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0:
                    area = abs(sp.signed_area(p))
                    if best is None or area < best:
                        hit, best = i, area
            if hit is None:
                if not shift:
                    self.selected = set()
            elif shift:
                self.selected ^= {hit}
            else:
                self.selected = {hit}
            self.update_edit_help()
            self.redraw()

    def on_double(self, e):
        if self.mode == "draw":
            self.close_polygon()

    def on_motion(self, e):
        if self.mode == "draw" and self.draw_pts and self.view:
            self.redraw()
            last = self.to_canvas([self.draw_pts[-1]])[0]
            self.canvas.create_line(last[0], last[1], e.x, e.y, fill=DRAW_COLOR, dash=(4, 3))

    # ------------------------------------------------------------------ stage 3: order sections

    def build_order(self):
        f = ttk.Frame(self.body)
        self.order = []                    # [(slide index, polygon index)]
        self.thumbs = {}                   # (slide index, polygon index) -> PhotoImage
        self.drag = None
        top = ttk.Frame(f, padding=(12, 8))
        top.pack(fill=tk.X)
        self.var_order_text = tk.StringVar(f, "")
        ttk.Label(top, textvariable=self.var_order_text, foreground="gray30").pack(side=tk.LEFT)
        ttk.Button(top, text="Reverse order", command=self.reverse_order).pack(side=tk.RIGHT)
        ttk.Button(top, text="Slide by slide", command=self.reset_order).pack(side=tk.RIGHT, padx=8)
        wrap = ttk.Frame(f)
        wrap.pack(fill=tk.BOTH, expand=True)
        self.ocanvas = tk.Canvas(wrap, bg="white", highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.ocanvas.yview)
        self.ocanvas.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.ocanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ocanvas.bind("<Configure>", lambda e: self.draw_order())
        self.ocanvas.bind("<MouseWheel>", lambda e: self.ocanvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.ocanvas.bind("<ButtonPress-1>", self.o_press)
        self.ocanvas.bind("<B1-Motion>", self.o_move)
        self.ocanvas.bind("<ButtonRelease-1>", self.o_release)
        return f

    def enter_order(self):
        proj = self.project
        key = proj.polygons_key()
        if proj.order and self.order_key == key and all(si < len(proj.slides) and pi < len(proj.slides[si].polygons) for si, pi in proj.order):
            self.order = list(proj.order)
        else:
            self.order = proj.default_order()
            self.order_key = key
        self.thumbs = {}
        self.make_thumbs()
        self.draw_order()

    def make_thumbs(self):
        need = {(si, pi) for si, pi in self.order}
        paths = {self.project.slides[si].path for si, _ in need if self.project.slides[si].path not in self.ov}
        if paths:
            self.var_status.set("Loading the slide pictures...")

            def work():
                return {p: sp.overview(p) for p in paths}

            def done(res):
                self.var_status.set("")
                if isinstance(res, BaseException):
                    messagebox.showerror("Can't read a slide", f"{type(res).__name__}: {res}", parent=self.root)
                    return
                for p, r in res.items():
                    self.ov[p], self.ovc[p] = r[:2], r[2]
                self.build_thumbs()
                self.draw_order()

            self.background(work, done)
        else:
            self.build_thumbs()

    def build_thumbs(self):
        for si, pi in self.order:
            if (si, pi) in self.thumbs:
                continue
            s = self.project.slides[si]
            if s.path not in self.ov:
                continue
            img, (x0, y0, k) = self.ov[s.path]
            poly = (s.polygons[pi] - np.array([x0, y0])) / k
            lo, hi = poly.min(0), poly.max(0)
            pad = 0.08 * (hi - lo).max() + 4
            a = np.maximum(np.floor(lo - pad), 0).astype(int)
            b = np.minimum(np.ceil(hi + pad), [img.shape[1], img.shape[0]]).astype(int)
            crop = img[a[1]:b[1], a[0]:b[0]]
            if crop.size == 0:
                continue
            f = min(THUMB_W / crop.shape[1], THUMB_H / crop.shape[0])
            color = self.ovc.get(s.path)
            crop3 = color[a[1]:b[1], a[0]:b[0]] if self.var_color.get() and color is not None else np.repeat(crop[:, :, None], 3, axis=2)
            rgb = cv2.resize(crop3, (max(int(crop.shape[1] * f), 1), max(int(crop.shape[0] * f), 1)), interpolation=cv2.INTER_AREA)
            cv2.polylines(rgb, [((poly - a) * f).astype(np.int32)], True, (255, 212, 0), 1, cv2.LINE_AA)
            self.thumbs[(si, pi)] = ImageTk.PhotoImage(Image.fromarray(rgb))

    def cell_geometry(self):
        w = max(self.ocanvas.winfo_width(), 300)
        cw, chh = THUMB_W + CELL_PAD, THUMB_H + CAPTION_H + CELL_PAD
        cols = max(1, (w - CELL_PAD) // cw)
        return cols, cw, chh

    def cell_origin(self, i):
        cols, cw, chh = self.cell_geometry()
        return CELL_PAD + (i % cols) * cw, CELL_PAD + (i // cols) * chh

    def draw_order(self, gap=None):
        c = self.ocanvas
        c.delete("all")
        if self.stage != 2:
            return
        cols, cw, chh = self.cell_geometry()
        rows = (len(self.order) + cols - 1) // cols
        c.config(scrollregion=(0, 0, 10, CELL_PAD + rows * chh))
        self.var_order_text.set(f"{len(self.order)} section(s). Drag a picture to where it belongs in the series; "
                                "the number is its place in the order.")
        for i, key in enumerate(self.order):
            if self.drag and self.drag["moved"] and i == self.drag["index"]:
                continue
            x, y = self.cell_origin(i)
            c.create_rectangle(x, y, x + THUMB_W, y + THUMB_H, outline="#c0c0c0", fill="#101010")
            ph = self.thumbs.get(key)
            if ph:
                c.create_image(x + THUMB_W / 2, y + THUMB_H / 2, image=ph)
            slide = self.project.slides[key[0]]
            c.create_text(x + 2, y + THUMB_H + 2, anchor="nw", fill="black", font=("Segoe UI", 9),
                          text=f"{i + 1}   {os.path.splitext(os.path.basename(slide.path))[0][-16:]}")
        if self.drag and self.drag["moved"]:
            g = self.drag["slot"]
            x, y = self.cell_origin(g)
            if g == len(self.order) and g % cols == 0 and g:
                x, y = self.cell_origin(g - 1)
                x += cw
            c.create_rectangle(x - 6, y, x - 3, y + THUMB_H, fill=ACCENT, outline="")
            ph = self.thumbs.get(self.order[self.drag["index"]])
            if ph:
                c.create_image(self.drag["x"], self.drag["y"], image=ph)

    def slot_at(self, x, y):
        """Insertion position (0 .. len) for a point on the canvas."""
        cols, cw, chh = self.cell_geometry()
        y = self.ocanvas.canvasy(y)
        col = int(min(max((x - CELL_PAD) // cw, 0), cols - 1))
        row = int(max((y - CELL_PAD) // chh, 0))
        i = row * cols + col
        cx = CELL_PAD + col * cw + THUMB_W / 2
        if x > cx:
            i += 1
        return min(i, len(self.order))

    def o_press(self, e):
        y = self.ocanvas.canvasy(e.y)
        cols, cw, chh = self.cell_geometry()
        col, row = (e.x - CELL_PAD) // cw, (y - CELL_PAD) // chh
        i = int(row * cols + col)
        if 0 <= col < cols and 0 <= i < len(self.order):
            self.drag = {"index": i, "x0": e.x, "y0": y, "x": e.x, "y": y, "moved": False, "slot": i}

    def o_move(self, e):
        d = self.drag
        if not d:
            return
        y = self.ocanvas.canvasy(e.y)
        if not d["moved"] and abs(e.x - d["x0"]) + abs(y - d["y0"]) < 6:
            return
        d["moved"], d["x"], d["y"] = True, e.x, y
        d["slot"] = self.slot_at(e.x, e.y)
        self.draw_order()

    def o_release(self, e):
        d, self.drag = self.drag, None
        if d and d["moved"]:
            item = self.order.pop(d["index"])
            slot = d["slot"] - (1 if d["slot"] > d["index"] else 0)
            self.order.insert(slot, item)
        self.draw_order()

    def reverse_order(self):
        self.order.reverse()
        self.draw_order()

    def reset_order(self):
        self.order = self.project.default_order()
        self.draw_order()

    # ------------------------------------------------------------------ done

    def finish(self):
        proj = self.project
        proj.order = list(self.order)
        self.order_key = proj.polygons_key()
        try:
            proj.save(proj.path or proj.default_path())
        except OSError as e:
            messagebox.showwarning("Project not saved", f"Couldn't save the project file: {e}", parent=self.root)
        self.var_status.set("Preparing the sections...")

        def done(res):
            self.var_status.set("")
            if isinstance(res, BaseException):
                messagebox.showerror("Can't go on", f"{type(res).__name__}: {res}", parent=self.root)
                return
            self.result = res
            self.root.destroy()
            if self.on_done:
                self.on_done(res)

        self.background(lambda: sp.build_stack(proj), done)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    loader = SectionLoader(slides=[os.path.normpath(a) for a in argv if a.lower().endswith(".czi")])
    return loader.run()


if __name__ == "__main__":
    main()
