"""Step 1 of the workflow: find the sections on Zeiss .czi slide scans (several sections each) and .jp2 section images (one each).

Three stages in one window, shown by a strip along the top:
  1. Load slides      - choose the .czi files (or open a saved project);
  2. Detect outlines  - for each slide, a brightness threshold finds the sections; the outlines can then be merged, deleted
                        and drawn by hand;
  3. Order sections   - thumbnails are dragged into their order along the brain.
"Done" saves the project beside the first slide and hands the sections, in memory, to the alignment tool (no JPX is written).
A .jpx stack can be added on its own (not mixed with scans): its sections come with outlines, so the outlines tab is skipped and the
order tab shows the JPX's own pictures of its sections.

Run it directly (python section_loader.py [slides...]), or from the start window (SerialAssembler.py)."""
import hashlib
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import imagecodecs
import numpy as np
from PIL import Image, ImageTk

import neuroinfo_dialogs
import neuroinfo_stack as ns
import section_project as sp
from section_order import OrderGrid, THUMB_W, THUMB_H, CELL_PAD
from order_editor import OrderEditor, placeholder_image
from tooltip import Tooltip

STAGES = ("Load slides", "Detect outlines", "Order sections", "Edit outlines and alignment")
ACCENT = "#1f5fa8"
SELECT_COLOR, OUTLINE_COLOR, DRAW_COLOR = "#ff3030", "#ffd400", "#00e5ff"
TINT = np.array([0, 190, 255], np.float32)
GRID_LABEL = 22          # the height of the line of text under each picture of the grid of all the images


def human_size(n):
    """A file size in the unit that suits it (bytes, KB, MB, GB or TB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


DEFAULT_MIN_AREA_MM2 = 0.25    # the smallest region kept, until the user moves the slider

_jpx_frames = {}
_jpx_frames_lock = threading.Lock()


def jpx_frame_boxes(path):
    """The boxes of the frames of a JPX (read once: for a big file it takes a moment)."""
    with _jpx_frames_lock:
        if path not in _jpx_frames:
            _jpx_frames[path] = [b for b in ns.read_boxes(path) if b.kind == b"jp2c"]
        return _jpx_frames[path]


def jpx_colorize(raw, stack, mode="auto"):
    """The thumbnail picture (RGB uint8) of a section's raw crop (H x W x channels, as the frame holds it), the channels in their colors.
    mode "auto": each channel stretched on its own, from its 1st to its 99.7th percentile (x1.3) of the nonzero pixels, so that every
    section is bright enough to see; "full": the whole range of the file's numbers (0 to 65535 for 16 bits), as it is, so that sections
    can be compared (a faint one then looks faint)."""
    top = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 65535.0
    rgbf = np.zeros(raw.shape[:2] + (3,), np.float32)
    for c in range(raw.shape[2]):
        v = raw[..., c].astype(np.float32)
        if mode == "full":
            lo_v, hi_v = 0.0, top
        else:
            nz = v[v > 0]
            lo_v = float(np.percentile(nz, 1)) if nz.size else 0.0
            hi_v = max(float(np.percentile(nz, 99.7)) * 1.3 if nz.size else 1.0, lo_v + 1.0)
        color = np.array(stack.channel_info[c].color if c < len(stack.channel_info) else (255, 255, 255), np.float32) / 255
        rgbf += np.clip((v - lo_v) / (hi_v - lo_v), 0, 1)[..., None] * color
    return (np.clip(rgbf, 0, 1) * 255).astype(np.uint8)


def jpx_crop(stack, path, frame, section, target_px=256):
    """The raw picture of one section of a JPX, cropped round its outline from the JPX's own frame (the section as NeuroInfo rendered it,
    so no slide scan is needed): (the crop, H x W x channels as the frame holds it; the outline in it; um per picture pixel). Only the reduced
    picture the thumbnail needs is read from the frame (a few percent of it), and the result is kept on disk, so that the next time is
    instant. jpx_colorize makes the thumbnail of it, at whatever brightness is asked for."""
    if getattr(section, "placeholder", False):      # (a blank frame NeuroInfo inserted: nothing to read; the thumbnail says what it is)
        return np.zeros((40, 56, max(stack.channels, 1)), np.uint16), np.zeros((0, 2), np.float32), float(stack.pixel_um)
    st = os.stat(path)
    P = section.stack_points
    outline_hash = hashlib.sha1(np.round(P, 1).tobytes()).hexdigest()[:12]
    key = f"jpxcrop2|{os.path.abspath(path).lower()}|{st.st_size}|{st.st_mtime_ns}|{frame}|{outline_hash}|{target_px}"
    cache = os.path.join(sp.CACHE_DIR, "jt_" + hashlib.sha1(key.encode()).hexdigest() + ".npz")
    if os.path.isfile(cache):
        try:
            with np.load(cache) as d:
                return d["raw"], d["poly"], float(d["um"])
        except Exception:
            pass                                   # (a damaged cache file: made again)
    px = stack.pixel_um
    poly = np.column_stack([P[:, 0] / px, -P[:, 1] / px])         # the outline in the frame's pixels (the frame is the stack's canvas)
    lo, hi = poly.min(0), poly.max(0)
    pad = 0.08 * float((hi - lo).max()) + 4
    a = np.maximum(np.floor(lo - pad), 0).astype(int)
    b = np.minimum(np.ceil(hi + pad), [stack.canvas_w, stack.canvas_h]).astype(int)
    skip = int(max(0, min(math.floor(math.log2(max(int((b - a).max()), 1) / target_px)), 6)))
    frame_obj = ns._Frame(source=(path, jpx_frame_boxes(path)[frame]))
    for k in range(skip, -1, -1):                  # (as reduced as the file allows, up to what the thumbnail needs)
        try:
            try:
                img = imagecodecs.jpeg2k_decode(frame_obj._stream(skip=k), skipres=k)
            except Exception:                              # (the decoder did not take the short stream: the whole layout, as before)
                img = imagecodecs.jpeg2k_decode(frame_obj._stream(skip=k, truncate=False), skipres=k)
            break
        except Exception:
            if k == 0:
                raise
    s = 2 ** k
    if img.ndim == 2:
        img = img[..., None]
    x0, y0 = int(a[0] // s), int(a[1] // s)
    crop = img[y0:-(-int(b[1]) // s), x0:-(-int(b[0]) // s)]
    if crop.size == 0:
        raise ValueError("the outline is outside the frame")
    raw = np.ascontiguousarray(crop)
    poly_c = (poly / s - np.array([x0, y0])).astype(np.float32)
    try:
        tmp = cache + ".partial.npz"
        np.savez(tmp, raw=raw, poly=poly_c, um=px * s)
        os.replace(tmp, cache)
    except OSError:
        pass
    return raw, poly_c, px * s


class SectionLoader:
    def __init__(self, on_done=None, slides=(), project=None):
        self.on_done = on_done
        self.root = tk.Tk()
        self.root.title("Load/detect/order sections")
        self.root.geometry("1450x900")
        self.project = project or sp.Project()
        self.infos = {}                 # slide path -> sp.SlideInfo
        self.ov = {}                    # slide path -> (overview gray image, (x0, y0, k)), once loaded
        self.ovc = {}                   # slide path -> the same picture in color (RGB)
        self.stage = 0
        self.msgs = queue.Queue()
        self.busy = 0
        self.closing = False
        # pictures are loaded in the background, the one wanted now first (two at a time: they are different files)
        self.ov_lock = threading.Lock()
        self.ov_gray_q, self.ov_color_q, self.ov_active = [], [], set()
        self.ov_wake = threading.Event()
        self.waiting_for = None          # the image shown as soon as its picture arrives
        self.pending_min_mm2 = {}        # slide path -> smallest region (mm2) to use when its picture is first shown
        for _ in range(2):
            threading.Thread(target=self.ov_worker, daemon=True).start()
        self.result = None
        self.order_key = None           # project.polygons_key() when the order was last set
        self.jpx_stack, self.jpx_stack_path = None, ""     # a JPX project: its stack once opened (for the ordering tab)
        self.neuro = {}                                    # JPX path -> what neuroinfo_check found (and what the user chose about it)
        self.neuro_presets = {}                            # JPX path -> {"xml", "h5"} found from a reconstruction .xml that was added
        self.jpx_by_frame = {}                             # frame -> its section
        self.jpx_crops, self.jpx_inflight = {}, set()      # frame -> (colour crop, outline in it, um per pixel) or False; frames being made
        self.bright_mode = "auto"                          # Auto / Full scale of a JPX's thumbnails (the alignment tool starts with the same)
        self.jpx_target = 256                              # pixels the crops of a JPX's sections are read at (more for large thumbnails)
        self.stale_thumbs = set()                          # JPX thumbnails shown as they are until their new version (brightness, scale) is made
        self.jpx_pool, self.jpx_tick_id = None, None
        # ---- the strip of stages
        self.strip = tk.Frame(self.root, bg="#eef1f5")
        self.strip.pack(fill=tk.X)
        self.strip_labels = []
        for i, name in enumerate(STAGES):
            if i:
                tk.Label(self.strip, text="›", bg="#eef1f5", fg="gray50", font=("Segoe UI", 14)).pack(side=tk.LEFT)
            number = str(i + 1)
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
        self.poll_id = self.root.after(50, self.poll)
        for p in slides:
            self.project.slides.append(sp.Slide(p))
        self.refresh_slide_list()
        if self.project.slides:
            self.read_infos()
        self.update_title()
        self.show_stage(0)

    # ------------------------------------------------------------------ plumbing

    def run(self):
        self.root.mainloop()
        return self.result

    def close(self):
        """Close the window (its timer cancelled first, so that it can't go off in the next window)."""
        self.closing = True
        self.ov_wake.set()
        try:
            self.root.quit()       # (the loop ends even if another window, the hidden alignment tool, is still alive)
        except tk.TclError:
            pass
        for after_id in (self.poll_id, self.jpx_tick_id):
            try:
                if after_id is not None:
                    self.root.after_cancel(after_id)
            except (tk.TclError, AttributeError):
                pass
        if self.jpx_pool is not None:
            self.jpx_pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def quit(self):
        if self.project.slides and not messagebox.askyesno("Quit", "Close without handing the sections to the alignment tool?\n"
                                                           "Your work is saved in the project file.",
                                                           parent=self.root):
            return
        try:
            self.sync_project()          # (the outlines worked on, and the settings, are written to the project file before the window goes)
        except Exception as e:
            print(f"Couldn't save the project: {e}", flush=True)
        self.close()

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

    def want(self, path, kind="gray", urgent=False):
        """Ask for a slide's picture to be loaded in the background: `kind` "gray" (the one outlines are found on) or "color". Urgent
        ones go to the front; the rest are taken in the order asked."""
        if (kind == "gray" and path in self.ov) or (kind == "color" and path in self.ovc) or sp.is_jpx(path):
            return
        key = (kind, path)
        q = self.ov_gray_q if kind == "gray" else self.ov_color_q
        with self.ov_lock:
            if key in self.ov_active:
                return
            if key in q:
                q.remove(key)
            q.insert(0, key) if urgent else q.append(key)
        self.ov_wake.set()

    def queue_preload(self):
        """Load every slide's picture in the background (grays first, then the colors), without waiting for it to be looked at."""
        for s in self.project.slides:
            self.want(s.path, "gray")
        for s in self.project.slides:
            if not sp.is_jp2(s.path):       # (a .jp2's color comes with its gray)
                self.want(s.path, "color")

    def ov_worker(self):
        ns.lower_thread_priority()
        while not self.closing:
            self.ov_wake.wait()
            while not self.closing:
                with self.ov_lock:
                    q = self.ov_gray_q or self.ov_color_q
                    if not q:
                        self.ov_wake.clear()
                        break
                    kind, path = q.pop(0)
                    self.ov_active.add((kind, path))
                if all(s.path != path for s in self.project.slides):
                    res = None                  # (the slide was taken out of the list)
                else:
                    try:
                        res = sp.overview(path, color=(kind == "color"))
                        if kind == "color":
                            res = res[2]
                    except BaseException as e:
                        res = e
                with self.ov_lock:
                    self.ov_active.discard((kind, path))
                self.msgs.put(("overview", (kind, path, res)))

    def on_overview(self, kind, path, res):
        """A picture arrived from the background loader."""
        if res is None:
            return
        if isinstance(res, BaseException):
            if kind == "gray" and self.waiting_for == path:
                self.waiting_for = None
                self.var_status.set("")
                messagebox.showerror("Can't read the image", f"{os.path.basename(path)}\n\n{type(res).__name__}: {res}", parent=self.root)
            return
        if kind == "gray":
            self.ov[path] = res[:2]
            if res[2] is not None:
                self.ovc[path] = res[2]
        else:
            self.ovc[path] = res
        if self.stage == 1 and self.project.slides and self.slide().path == path:
            if kind == "gray" and self.waiting_for == path:
                self.waiting_for = None
                self.var_status.set("")
                self.slide_ready()
            elif kind == "color" and self.var_color.get() and self.base_rgb is not None:
                self.var_status.set("")
                self.set_base()
                self.redraw()
        elif self.stage == 2 and kind == "gray":
            self.finish_thumbs()
        if kind == "gray" and self.stage == 1 and self.grid_mode:
            self.schedule_grid()

    def poll(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "overview":
                    self.on_overview(*payload)
                elif kind == "grid":
                    version, i, rgb = payload
                    if version == self.grid_version and self.grid_mode:
                        self.grid_rgb[i] = rgb
                        self.redraw()
                elif kind == "call":
                    payload()                                # (a dialog the thread needs, made here)
                elif kind == "jpxcrop":
                    self.on_jpx_crop(*payload)
                elif kind == "done":
                    done, res = payload
                    self.busy -= 1
                    if not self.busy:
                        self.root.config(cursor="")
                    done(res)
        except queue.Empty:
            pass
        self.poll_id = self.root.after(50, self.poll)

    def show_stage(self, n):
        self.stage = n
        for i, lab in enumerate(self.strip_labels):
            if i == n:
                lab.config(bg=ACCENT, fg="white", font=("Segoe UI", 11, "bold"))
            else:
                lab.config(bg="#eef1f5", fg="black" if i < n else "gray55", font=("Segoe UI", 11))
        self.strip_labels[1].config(text="2  Detect outlines" + (" (not needed)" if self.project.is_jpx else ""))
        if self.project.is_jpx and n != 1:
            self.strip_labels[1].config(fg="gray70")       # (a JPX comes with its outlines: step 2 is skipped)
        for f in self.frames:
            f.pack_forget()
        self.frames[n].pack(fill=tk.BOTH, expand=True)
        self.btn_back.config(state="normal" if n else "disabled")
        self.btn_next.config(text="Done: open in the alignment tool" if n == 2 else "Next")
        self.update_next_button()
        self.var_help.set(("Choose the slide scans (.czi) and section images (.jp2) that hold your sections, or one .jpx stack on its own.",
                           "Set the threshold so that each section is tinted and debris is not, then accept the outlines.",
                           "Drag the pictures into the order of the sections along the brain.")[n])
        if n == 1:
            self.enter_detect()
        elif n == 2:
            self.enter_order()

    def go_back(self):
        if self.stage == 1:
            self.leave_detect()
        if self.stage == 2 and self.project.is_jpx:
            self.show_stage(0)                     # (a JPX has no outlines tab)
            return
        if self.stage > 0:
            self.show_stage(self.stage - 1)

    def go_next(self):
        if self.busy:
            return
        ns.foreground_active(4.0)        # (the click is answered first: the background locating and picture-making wait for a moment)
        if self.stage == 0:
            if not self.project.slides:
                messagebox.showinfo("No slides", "Add at least one .czi slide scan or .jp2 section image first.", parent=self.root)
                return
            if self.project.is_jpx:                  # (its sections come with outlines: straight on to the order)
                if not self.save_slide_list():
                    return
                self.show_stage(2)
                return
            unknown = [os.path.basename(s.path) for s in self.project.slides if s.path in self.infos and not self.slide_px(s)]
            if unknown:
                messagebox.showinfo("Pixel size needed", "These files don't say what their pixel size is. Select them and use "
                                    "\"Set pixel size...\":\n\n" + "\n".join(unknown[:8]), parent=self.root)
                return
            if not self.save_slide_list():
                return                               # (no name was given for the project file: stay, and nothing is lost)
            self.show_stage(1)
        elif self.stage == 1:
            self.leave_detect()
            bad = [os.path.basename(s.path) for s in self.project.slides if not s.accepted]
            n = sum(len(s.polygons) for s in self.project.slides)
            if not n:
                messagebox.showinfo("No outlines", "No section outlines yet: accept the outlines of at least one image.", parent=self.root)
                return
            if bad and not messagebox.askyesno(
                    "Images without outlines",
                    "The outlines of these images haven't been accepted, so they will have no sections:\n\n" + "\n".join(bad[:8])
                    + "\n\nGo on anyway?", parent=self.root):
                return
            self.show_stage(2)
        elif self.stage == 2:
            self.finish()

    # ------------------------------------------------------------------ stage 1: load slides

    def build_load(self):
        f = ttk.Frame(self.body, padding=20)
        ttk.Label(f, text="Image files", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(f, text="Add every file whose sections you want to align: a slide scan holds several, a .jp2 holds one. "
                          "They are taken in the order listed.",
                  foreground="gray30").pack(anchor="w", pady=(0, 8))
        mid = ttk.Frame(f)
        mid.pack(fill=tk.BOTH, expand=True)
        cols = ("name", "scenes", "size", "pixel", "neuro")
        self.slide_tree = ttk.Treeview(mid, columns=cols, show="headings", selectmode="extended", height=14)
        for c, t, w, a in (("name", "File", 620, "w"), ("scenes", "Scenes", 90, "e"), ("size", "Size", 110, "e"), ("pixel", "Pixel (um)", 90, "e"),
                           ("neuro", "NeuroInfo files", 170, "w")):
            self.slide_tree.heading(c, text=t)
            self.slide_tree.column(c, width=w, anchor=a)
        sb = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self.slide_tree.yview)
        self.slide_tree.config(yscrollcommand=sb.set)
        self.slide_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        # the list's rows can also be dragged to a new place with the mouse (the buttons at the right do the same a step at a time)
        self.drag_row, self.drag_active = None, False
        self.drop_line = tk.Frame(self.slide_tree, bg=ACCENT, height=2)
        self.slide_tree.bind("<ButtonPress-1>", self.row_press, add="+")
        self.slide_tree.bind("<B1-Motion>", self.row_drag, add="+")
        self.slide_tree.bind("<ButtonRelease-1>", self.row_drop, add="+")
        self.row_menu = tk.Menu(self.root, tearoff=0)
        self.row_menu.add_command(label="Copy path", command=self.copy_paths)
        self.slide_tree.bind("<Button-3>", self.row_menu_open)
        side = ttk.Frame(mid, padding=(12, 0))
        side.pack(side=tk.LEFT, fill=tk.Y)
        self.need_images = []              # the buttons that mean nothing until an image has been added (see refresh_slide_list)
        # the project buttons first, then those for the images in the list
        ttk.Button(side, text="Open a saved project...", command=self.open_project).pack(fill=tk.X)
        ttk.Button(side, text="Open recent projects...", command=self.open_recent_project).pack(fill=tk.X, pady=(6, 0))
        close_button = ttk.Button(side, text="Close project", command=self.close_project)
        close_button.pack(fill=tk.X, pady=(6, 0))
        self.need_images.append(close_button)
        ttk.Separator(side).pack(fill=tk.X, pady=10)
        ttk.Button(side, text="Add images...", command=self.add_slides).pack(fill=tk.X)
        for text, command, pad in (("Remove image", self.remove_slides, 6), ("Move up", lambda: self.move_slide(-1), 0),
                                   ("Move down", lambda: self.move_slide(1), 6), ("Set pixel size...", self.set_pixel_size, 0)):
            button = ttk.Button(side, text=text, command=command)
            button.pack(fill=tk.X, pady=pad)
            self.need_images.append(button)
        return f

    def neuro_state(self, path):
        return getattr(self, "neuro", {}).get(path)

    def neuro_unresolved(self, path):
        """A JPX from NeuroInfo whose .xml / .h5 are not there and the user has not yet said what to do about it."""
        n = self.neuro_state(path)
        return bool(n) and n.get("state") == "missing" and not n.get("kept")

    def neuro_text(self, path):
        """The "NeuroInfo files" column: what a JPX needs beside it (blank for other files)."""
        if not sp.is_jpx(path):
            return ""
        n = self.neuro_state(path)
        if n is None:
            return "" if self.project.alignment else ("checking..." if path not in self.infos else "")
        state = n["state"]
        if state == "own":
            return "not needed"
        if state == "found":
            return "found" + (" (chosen)" if n.get("xml") and n.get("chosen") else "")
        if state == "missing":
            what = " and ".join({"xml": ".xml", "h5": "case file"}[k] for k in n["missing"])
            return f"MISSING: {what}" + (" (kept anyway)" if n.get("kept") else "")
        return "could not check"

    def resolve_missing_neuroinfo(self):
        """For each JPX added whose NeuroInfo files are missing: say so now (not at step 3), and ask what to do."""
        for slide in list(self.project.slides):
            n = self.neuro_state(slide.path)
            while n and n.get("state") == "missing" and not n.get("kept"):
                name = os.path.basename(slide.path)
                what = " and ".join({"xml": f"{os.path.splitext(name)[0]}.xml (the reconstruction)", "h5": "the case file (.h5)"}[k] for k in n["missing"])
                choice = neuroinfo_dialogs.missing_files_dialog(self.root, name, what)
                if choice == "keep":
                    n["kept"] = True
                elif choice == "remove":
                    self.project.slides = [s for s in self.project.slides if s.path != slide.path]
                    self.neuro.pop(slide.path, None)
                    self.infos.pop(slide.path, None)
                    if not self.project.slides:
                        self.reset_project()
                    self.var_status.set(f"{name} was not added.")
                    break
                else:
                    self.locate_neuroinfo(slide.path)
                    n = self.neuro_state(slide.path)
        self.refresh_slide_list()

    def locate_neuroinfo(self, path):
        """The user points at the .xml and the .h5 of a JPX (as many as are missing); what is chosen is used when the JPX is opened."""
        n = self.neuro[path]
        folder = os.path.dirname(os.path.abspath(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        chosen = {}
        for kind in ("xml", "h5"):
            if kind == "xml" and "xml" not in n["missing"]:
                continue
            if kind == "h5" and "h5" not in n["missing"] and "xml" not in n["missing"]:
                continue
            names = sorted(f for f in os.listdir(folder) if f.lower().endswith("." + kind))
            got = neuroinfo_dialogs.locate_file(self.root, kind, f"{stem}.xml" if kind == "xml" else "the case file (.h5)", folder, names,
                                                "ambiguous" if names else "missing", names[0] if names else None, {})
            if not got:
                return                                   # (cancelled: back to the question)
            chosen[kind] = got
        self.neuro[path] = {"state": "found", "chosen": True, **chosen}

    def update_next_button(self):
        """On the first tab, Next is greyed out while the files in the list are still being read (their scenes, sizes and pixel sizes are what the
        next tabs go by)."""
        waiting = self.stage == 0 and not getattr(self, "infos_failed", False) and (
            any(s.path not in self.infos for s in self.project.slides) or any(self.neuro_unresolved(s.path) for s in self.project.slides))
        self.btn_next.state(["disabled"] if waiting else ["!disabled"])

    def refresh_slide_list(self):
        for button in getattr(self, "need_images", ()):          # (greyed while no image is in the list)
            button.state(["!disabled"] if self.project.slides else ["disabled"])
        self.slide_tree.delete(*self.slide_tree.get_children())
        for i, s in enumerate(self.project.slides):
            info = self.infos.get(s.path)
            px = self.slide_px(s)
            self.slide_tree.insert("", tk.END, iid=str(i), values=(
                s.path, (f"{info.scenes} sections" if sp.is_jpx(s.path) else info.scenes) if info else "counting...",
                human_size(info.size_bytes) if info else "determining...",     # (until the file has been read)
                (f"{px:g}" if px else "unknown") if info else "", self.neuro_text(s.path)))
        self.update_next_button()

    def read_infos(self):
        todo = [s.path for s in self.project.slides if s.path not in self.infos]
        self.queue_preload()
        if not todo:
            return
        self.var_status.set("Reading the files...")
        self.infos_failed = False
        self.update_next_button()

        check_neuro = not self.project.alignment       # (a project with its alignment saved opens its JPX without NeuroInfo's files)

        def work():
            out = []
            for p in todo:
                info = sp.slide_info(p)
                out.append((p, info, ns.neuroinfo_check(p) if check_neuro and sp.is_jpx(p) else None))
            return out

        def done(res):
            if isinstance(res, BaseException):
                messagebox.showerror("Can't read the slides", f"{type(res).__name__}: {res}", parent=self.root)
                self.var_status.set("")
                self.infos_failed = True                  # (Next must not stay grey for ever: the list can be corrected)
                self.update_next_button()
                return
            for p, info, neuro in res:
                self.infos[p] = info
                if neuro is not None:
                    self.neuro[p] = neuro
                if p in self.neuro_presets:                # (found from the reconstruction .xml that was added: not asked again)
                    self.neuro[p] = {"state": "found", "chosen": True, **self.neuro_presets[p]}
            self.refresh_slide_list()
            self.queue_preload()
            self.var_status.set(f"{len(self.project.slides)} slide(s).")
            self.resolve_missing_neuroinfo()

        self.background(work, done)

    def slide_px(self, slide):
        """The pixel size (um) of a slide: the one set by the user, else the file's (0 if it doesn't say)."""
        info = self.infos.get(slide.path)
        return slide.pixel_um or (info.pixel_um if info else 0.0)

    def set_pixel_size(self):
        """Give the pixel size of the selected files (a .jp2 may not say; the sections of a stack must all have the same)."""
        sel = [int(i) for i in self.slide_tree.selection()]
        if not sel:
            messagebox.showinfo("Pixel size", "Select the files first.", parent=self.root)
            return
        cur = self.slide_px(self.project.slides[sel[0]])
        value = simpledialog.askfloat("Pixel size", "Pixel size of the selected file(s), in um per pixel:", parent=self.root,
                                      initialvalue=cur or 0.65, minvalue=0.001, maxvalue=1000)
        if value:
            for i in sel:
                self.project.slides[i].pixel_um = float(value)
            self.refresh_slide_list()

    def add_slides(self):
        """Add images. A project's images all lie in the folder its file is in (so that it can be moved or opened on another machine with
        paths mapped differently): the dialog starts there, and files from elsewhere are refused. For a new project the first image
        chosen sets the folder."""
        folder = self.project.folder()
        paths = filedialog.askopenfilenames(
            parent=self.root, title="Slide scans, section images, JPX stacks, or a NeuroInfo reconstruction (.xml)", initialdir=folder or None,
            filetypes=[("Slide scans, section images, JPX stacks and NeuroInfo .xml", "*.czi *.jp2 *.j2k *.j2c *.jpx *.xml"),
                       ("Zeiss .czi", "*.czi"), ("JPEG 2000 section images", "*.jp2 *.j2k *.j2c"), ("JPX stack of sections", "*.jpx"),
                       ("NeuroInfo reconstruction .xml (its JPX and case file are found from it)", "*.xml"), ("All files", "*.*")])
        if not paths:
            return
        if any(p.lower().endswith(".xml") for p in paths):
            jpx = self.jpx_from_reconstruction_xml(paths)
            if not jpx:
                return
            paths = [jpx]                                # (from here on, as if the JPX had been chosen: its XML and case file are known)
        if any(sp.is_jpx(p) for p in paths) or self.project.is_jpx:
            # a JPX is a stack of sections already, with their outlines: a project of its own, not mixed with scans or other stacks
            if self.project.slides or len(paths) > 1:
                messagebox.showinfo("JPX stack", "A JPX stack is a project of its own: it can't be combined with other files. Add it on its own "
                                    "to an empty list (use \"Close project\" first), and add scans only to a project that has none.",
                                    parent=self.root)
                return
        if len(paths) == 1 and sp.is_jpx(paths[0]) and not self.project.slides:
            # a project of this JPX may exist already (saved when it was loaded before, with its order and alignment): open it, or start afresh
            existing = sp.projects_for_jpx(paths[0])
            if existing:
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(existing[0])))
                more = (f"\n\n({len(existing) - 1} other project(s) of this JPX are in the same folder.)" if len(existing) > 1 else "")
                answer = messagebox.askyesnocancel(
                    "A project of this JPX exists",
                    f"{os.path.basename(paths[0])} already has a project here:\n\n{os.path.basename(existing[0])}  (saved {when}){more}\n\n"
                    f"Yes: open that project, with its order and alignment.\n"
                    f"No: start a new project of this JPX (as another file).\nCancel: do not add it.", parent=self.root)
                if answer is None:
                    return
                if answer:
                    self.open_project(existing[0])
                    return
        same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
        folder = folder or os.path.dirname(os.path.abspath(paths[0]))
        refused = [p for p in paths if not same(os.path.dirname(os.path.abspath(p)), folder)]
        if refused:
            messagebox.showinfo(
                "Images from another folder",
                f"A project's images must be in the folder of its project file, so that the project can be moved or opened on another "
                f"computer (its paths are relative). That folder is:\n{folder}\n\nNot added:\n\n" + "\n".join(refused[:8])
                + ("\n..." if len(refused) > 8 else "") + "\n\nTo use them, start a new project: remove all the images here, "
                "then add the others.", parent=self.root)
        known = {os.path.normcase(os.path.abspath(s.path)) for s in self.project.slides}
        for p in paths:
            p = os.path.normpath(p)
            if p in refused or os.path.normcase(os.path.abspath(p)) in known:
                continue
            self.project.slides.append(sp.Slide(p))
        self.refresh_slide_list()
        self.update_title()
        self.read_infos()

    def jpx_from_reconstruction_xml(self, paths):
        """A NeuroInfo reconstruction .xml was chosen: find its JPX and its case file (.h5) beside it, as far as they can be told from the
        files (see ns.find_jpx_for_xml, ns.find_case_file), asking only where that does not settle it. Returns the JPX's path, or None (the
        reason is shown); the .xml and .h5 found are remembered for when the JPX is opened."""
        if len(paths) != 1:
            messagebox.showinfo("NeuroInfo reconstruction file", "A NeuroInfo reconstruction file (.xml) is added on its own: it brings its "
                                "JPX and case file with it. Choose just that one file.", parent=self.root)
            return None
        xml = os.path.normpath(paths[0])
        if self.project.slides:
            messagebox.showinfo("NeuroInfo reconstruction file", "A JPX stack is a project of its own: it can't be combined with other files. "
                                "Add it to an empty list (use \"Close project\" first).", parent=self.root)
            return None
        try:
            info = ns.read_reconstruction_xml(xml)
        except (OSError, ValueError) as e:
            messagebox.showerror("NeuroInfo reconstruction file", f"{os.path.basename(xml)}: {e}", parent=self.root)
            return None
        locate = lambda *args, **kwargs: neuroinfo_dialogs.locate_file(self.root, *args, **kwargs)
        self.var_status.set("Looking for the JPX and the case file...")
        self.root.update_idletasks()
        try:
            jpx = os.path.normpath(ns.find_jpx_for_xml(xml, info["jpx_name"], locate))
            h5 = ns.find_case_file(os.path.dirname(os.path.abspath(xml)), info["guids"], locate)
        except (OSError, ValueError) as e:
            self.var_status.set("")
            messagebox.showinfo("NeuroInfo reconstruction file", str(e), parent=self.root)
            return None
        self.neuro_presets[jpx] = {"xml": xml, "h5": os.path.normpath(h5)}
        self.var_status.set(f"{os.path.basename(xml)}: using {os.path.basename(jpx)} and {os.path.basename(h5)}.")
        return jpx

    def save_slide_list(self):
        """Write the project file (a new project's is named after the first image of the list as it is now). Done when going on from the
        first tab, so that what is done there takes effect only then."""
        if not self.project.slides:
            return True
        try:
            return self.write_project(self.project)
        except OSError as e:
            self.var_status.set(f"Couldn't save the project: {e}")
            return True                              # (an error is not a refusal: going on as before)

    def close_project(self):
        """Close the project and reset the interface: the list of images is emptied and the next image added starts a new project, whose
        file (in the folder of that image) is made when the first tab is left. The project file already saved, if any, is not touched."""
        if not self.project.slides:
            return
        saved = f"\n\nThe project file {os.path.basename(self.project.path)} is kept as it is." if self.project.path else ""
        if not messagebox.askyesno("Close project", f"Close project and reset interface?{saved}", parent=self.root):
            return
        self.sync_project(save=False)
        self.reset_project()
        self.refresh_slide_list()
        self.var_status.set("Project closed: add images to start a new one.")

    def reset_project(self):
        """Back to a new, empty project (when every image has been removed): the next image added starts a project file in its own folder."""
        self.project = sp.Project()
        self.var_numbering.set(self.project.numbering)
        self.jpx_stack, self.jpx_stack_path, self.jpx_crops, self.jpx_inflight = None, "", {}, set()
        self.infos, self.ov, self.ovc = {}, {}, {}
        self.polys, self.undo, self.order, self.order_key = {}, {}, [], None
        self.cur, self.waiting_for, self.pending_min_mm2 = 0, None, {}
        self.update_title()

    def remove_slides(self):
        gone = {int(i) for i in self.slide_tree.selection()}
        if gone:
            self.sync_project(save=False)            # (outlines worked on are written back before the list changes under them)
            self.project.slides = [s for i, s in enumerate(self.project.slides) if i not in gone]
            self.project.order = []
            self.polys, self.undo = {}, {}           # (these are by position in the list: rebuilt from the slides' outlines)
            self.cur = 0
            if not self.project.slides:
                self.reset_project()
            self.refresh_slide_list()

    def row_menu_open(self, e):
        """Right click on a file of the list: its menu (the row under the pointer is selected first, unless it already is)."""
        row = self.slide_tree.identify_row(e.y)
        if not row:
            return
        if row not in self.slide_tree.selection():
            self.slide_tree.selection_set(row)
        self.row_menu.tk_popup(e.x_root, e.y_root)

    def copy_paths(self):
        """The full path of each selected file, one to a line, onto the clipboard."""
        rows = sorted(int(i) for i in self.slide_tree.selection())
        text = "\n".join(os.path.normpath(self.project.slides[i].path) for i in rows if i < len(self.project.slides))
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()          # (so that the clipboard keeps it after this window closes)
            self.var_status.set("Copied the path of 1 file to the clipboard." if len(rows) == 1 else f"Copied the paths of {len(rows)} files to the clipboard.")

    def move_slide(self, d):
        sel = [int(i) for i in self.slide_tree.selection()]
        if len(sel) != 1:
            return
        self.move_slide_to(sel[0], sel[0] + d)

    def move_slide_to(self, i, j):
        """Move the image at position i of the list to position j."""
        sl = self.project.slides
        if i == j or not (0 <= i < len(sl) and 0 <= j < len(sl)):
            return
        self.sync_project(save=False)
        sl.insert(j, sl.pop(i))
        self.project.order = []
        self.polys, self.undo = {}, {}
        self.refresh_slide_list()
        self.slide_tree.selection_set(str(j))

    def row_press(self, e):
        row = self.slide_tree.identify_row(e.y)
        self.drag_row = int(row) if row and self.slide_tree.identify_region(e.x, e.y) == "cell" else None
        self.drag_active = False
        self.drag_start = (e.x, e.y)

    def drop_index(self, y):
        """Where in the list a row dropped at height y goes: 0 .. the number of rows (before the row under the pointer if y is in its
        upper half, after it if in its lower half)."""
        row = self.slide_tree.identify_row(y)
        n = len(self.project.slides)
        if not row:
            return n if y > 0 else 0
        k = int(row)
        box = self.slide_tree.bbox(row)
        if not box:
            return k
        return k + 1 if y > box[1] + box[3] / 2 else k

    def row_drag(self, e):
        if self.drag_row is None:
            return
        if not self.drag_active:
            if abs(e.y - self.drag_start[1]) + abs(e.x - self.drag_start[0]) < 5:
                return
            self.drag_active = True
            self.slide_tree.config(cursor="fleur")
        t = self.drop_index(e.y)
        n = len(self.project.slides)
        box = self.slide_tree.bbox(self.slide_tree.get_children()[-1] if t >= n else str(t))
        if box:
            self.drop_line.place(x=0, y=(box[1] + box[3] if t >= n else box[1]) - 1, relwidth=1)
        if e.y < 8:                                  # (near the edge of the list: it scrolls)
            self.slide_tree.yview_scroll(-1, "units")
        elif e.y > self.slide_tree.winfo_height() - 8:
            self.slide_tree.yview_scroll(1, "units")

    def row_drop(self, e):
        self.drop_line.place_forget()
        self.slide_tree.config(cursor="")
        if self.drag_row is None or not self.drag_active:
            self.drag_row = None
            return
        i, t = self.drag_row, self.drop_index(e.y)
        self.drag_row, self.drag_active = None, False
        self.move_slide_to(i, t - 1 if t > i else t)

    def open_recent_project(self):
        """A list of the recently used projects to choose from."""
        paths = sp.recent_projects()
        if not paths:
            messagebox.showinfo("Recent projects", "No projects have been saved or opened yet.", parent=self.root)
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("Recent projects")
        dlg.transient(self.root)
        dlg.geometry(f"+{self.root.winfo_rootx() + 120}+{self.root.winfo_rooty() + 100}")
        ttk.Label(dlg, text="Choose a project (double-click opens it):", padding=(10, 10, 10, 4)).pack(anchor="w")
        cols = ("name", "sections", "folder", "when")
        tree = ttk.Treeview(dlg, columns=cols, show="headings", selectmode="browse", height=min(max(len(paths), 4), 12))
        for c, t, w, a in (("name", "Project", 280, "w"), ("sections", "Sections", 110, "e"), ("folder", "Folder", 400, "w"), ("when", "Last saved", 190, "w")):
            tree.heading(c, text=t)
            tree.column(c, width=w, anchor=a, stretch=c in ("name", "folder"))   # (a wider window widens only Project and Folder)
        # The window comes up at once, from the names alone: reading each project file for its date and section count can take a while (the files
        # are often on a network drive, and a big project is a big file), so that is filled in, row by row, by a thread.
        for i, p in enumerate(paths):
            tree.insert("", tk.END, iid=str(i), values=(os.path.basename(p), "...", os.path.dirname(p), "..."))
        found = queue.Queue()

        def read_details():
            for i, p in enumerate(paths):
                try:
                    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
                    n = sp.sections_text(p)
                except OSError:
                    when, n = "(missing)", ""
                found.put((i, n, when))

        threading.Thread(target=read_details, daemon=True).start()

        def show_details():
            try:
                while True:
                    i, n, when = found.get_nowait()
                    if tree.exists(str(i)):
                        tree.set(str(i), "sections", n)
                        tree.set(str(i), "when", when)
            except queue.Empty:
                pass
            except tk.TclError:
                return                                   # (the window was closed)
            if dlg.winfo_exists():
                dlg.after(100, show_details)

        dlg.after(50, show_details)
        tree.pack(fill=tk.BOTH, expand=True, padx=10)
        if paths:
            tree.selection_set("0")
            tree.focus("0")
        chosen = []

        def choose(event=None):
            sel = tree.selection()
            if sel:
                chosen.append(paths[int(sel[0])])
                dlg.destroy()

        def forget():
            sel = tree.selection()
            if sel:
                sp.remove_recent_project(paths[int(sel[0])])
                tree.delete(sel[0])

        def delete():
            sel = tree.selection()
            if not sel:
                return
            p = paths[int(sel[0])]
            if not messagebox.askyesno("Delete project", f"Delete the project file\n\n{p}\n\nfrom the disk? Only this file goes: the "
                                       "images are not touched. This can't be undone.", icon="warning", default="no", parent=dlg):
                return
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError as e:
                messagebox.showerror("Can't delete", str(e), parent=dlg)
                return
            sp.remove_recent_project(p)
            tree.delete(sel[0])

        tree.bind("<Double-1>", choose)
        tree.bind("<Return>", choose)
        bar = ttk.Frame(dlg, padding=10)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Remove project from list", command=forget).pack(side=tk.LEFT)
        ttk.Button(bar, text="Delete project file (raw files remain intact)", command=delete).pack(side=tk.LEFT, padx=8)
        ttk.Button(bar, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT)       # (packed first: the rightmost)
        ttk.Button(bar, text="Open", command=choose).pack(side=tk.RIGHT, padx=(0, 8))
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        dlg.grab_set()
        tree.focus_set()
        self.root.wait_window(dlg)
        if chosen:
            if not os.path.isfile(chosen[0]):
                messagebox.showerror("Can't open", f"This project file no longer exists:\n{chosen[0]}", parent=self.root)
                sp.remove_recent_project(chosen[0])
                return
            self.open_project(chosen[0])

    def open_project(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(parent=self.root, title="Open a saved project",
                                              filetypes=[("Section projects", "*" + sp.PROJECT_SUFFIX + " *" + sp.OLD_PROJECT_SUFFIX), ("All files", "*.*")])
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
        sp.add_recent_project(path)
        self.project = proj
        self.jpx_stack, self.jpx_stack_path, self.jpx_crops, self.jpx_inflight = None, "", {}, set()
        self.var_shared.set(bool(proj.shared))
        self.var_numbering.set(proj.numbering)
        self.update_title()
        self.infos, self.ov, self.ovc = {}, {}, {}
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
        self.mask_show = None
        self.base_rgb = None
        self.photo = None
        self.pan = None
        self.job = None
        self.panel_nav = []                # (previous, next) buttons of the panels at the right
        self.grid_buttons = []             # the "Show all images" buttons of the panels
        self.accept_all_buttons = []       # the "Accept detection for all images" buttons (only usable while all the images are shown)
        self.grid_mode = False             # all the images shown on a grid
        self.grid_rgb, self.grid_cells, self.grid_photos, self.grid_job = {}, [], [], None
        self.grid_version = 0
        self.grid_size = None
        # ---- the picture
        left = ttk.Frame(f)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        nav = ttk.Frame(left, padding=(8, 6))
        nav.pack(fill=tk.X)
        # (a grid: the arrows sit against the menu, the checkboxes are pushed to the right by a spacer column, and when the window is
        # narrow the menu gives way (fit_combo), not the checkboxes beside it)
        nav.columnconfigure(3, weight=1)
        self.nav = nav
        self.btn_prev = ttk.Button(nav, text="◀", width=3, command=lambda: self.goto_slide(self.cur - 1))
        self.btn_prev.grid(row=0, column=0)
        self.var_slide = tk.StringVar(f, "")
        # the heading is a menu of the images, to jump to any of them (its text keeps the bold type of the heading it replaces)
        self.combo = ttk.Combobox(nav, textvariable=self.var_slide, state="readonly", font=("Segoe UI", 11, "bold"), width=60)
        self.combo.grid(row=0, column=1, padx=8, sticky="w")
        self.combo.bind("<<ComboboxSelected>>", self.on_pick_image)
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))
        self.btn_fwd = ttk.Button(nav, text="▶", width=3, command=lambda: self.goto_slide(self.cur + 1))
        self.btn_fwd.grid(row=0, column=2)
        self.var_scenes = tk.BooleanVar(f, False)
        ttk.Checkbutton(nav, text="Show scene outlines", variable=self.var_scenes, command=self.redraw).grid(row=0, column=4, padx=(14, 0))
        self.var_color = tk.BooleanVar(f, False)
        ttk.Checkbutton(nav, text="Show sections in color", variable=self.var_color, command=self.on_color).grid(row=0, column=5, padx=14)
        ttk.Button(nav, text="Fit", width=5, command=self.fit_view).grid(row=0, column=6)
        nav.bind("<Configure>", lambda e: self.fit_combo())
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
                        ("<Key-m>", lambda e: self.merge_key()), ("<Key-M>", lambda e: self.merge_key()),
                        ("<Delete>", lambda e: self.delete_selected()), ("<Escape>", lambda e: self.cancel_draw()),
                        ("<Control-z>", lambda e: self.undo_any(e))):
            self.root.bind(key, fn)
        # ---- the controls
        right = ttk.Frame(f, padding=12, width=330)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        left.pack_forget()                 # (packed again after the panel, so that a wide row above the picture can't push it off)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # how the outlines are numbered (for all the images): by scene number, across then down, or down then across
        number_box = ttk.LabelFrame(right, text="Number the outlines", padding=(8, 2, 8, 4))
        number_box.pack(fill=tk.X, pady=(0, 10))
        self.var_numbering = tk.StringVar(f, "rows")
        for value, text in (("rows", "Order by rows"), ("columns", "Order by columns"), ("scene", "Order by scene number")):
            ttk.Radiobutton(number_box, text=text, value=value, variable=self.var_numbering, command=self.on_numbering).pack(anchor="w")
        # both buttons of the usual run in one: accept the detection for all the images, then merge the contours within scenes
        self.btn_full_auto = ttk.Button(right, text="Auto detect outlines (can be edited later)", command=self.full_auto)
        self.btn_full_auto.pack(fill=tk.X, pady=(0, 10))
        self.panel_threshold = ttk.Frame(right)
        self.panel_edit = ttk.Frame(right)
        p = self.panel_threshold
        ttk.Label(p, text="Find the sections", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(p, text="Pixels brighter than the threshold are tinted blue. Move the slider until each section is tinted "
                          "as a whole and the debris is not.", wraplength=300, foreground="gray30").pack(anchor="w", pady=(2, 8))
        self.var_shared = tk.BooleanVar(f, False)
        ttk.Checkbutton(p, text="Same settings for all images", variable=self.var_shared, command=self.on_shared).pack(anchor="w", pady=(0, 12))
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
        self.var_pad_text = tk.StringVar(f, "")
        ttk.Label(p, textvariable=self.var_pad_text).pack(anchor="w")
        self.scale_pad = ttk.Scale(p, from_=0, to=10, command=self.on_pad)
        self.scale_pad.pack(fill=tk.X, pady=(0, 12))
        self.var_found = tk.StringVar(f, "")
        ttk.Label(p, textvariable=self.var_found, foreground="gray30").pack(anchor="w")
        # one button: it accepts the image on show, or, while all the images are shown on the grid, all of them (and says so)
        self.accept_button = ttk.Button(p, text="Accept detection for this image", command=self.accept_clicked)
        self.accept_button.pack(fill=tk.X, pady=(18, 0))
        self.image_buttons(p)
        ttk.Label(p, text="The outline of each tinted region is drawn around its outside; holes inside it (ventricles) stay in.",
                  wraplength=300, foreground="gray40").pack(anchor="w", pady=8)
        p = self.panel_edit
        ttk.Label(p, text="Edit the outlines", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.var_edit_help = tk.StringVar(f, "")
        ttk.Label(p, textvariable=self.var_edit_help, wraplength=300, foreground="gray30").pack(anchor="w", pady=(2, 12))
        self.btn_merge = ttk.Button(p, text="Merge (M)", command=self.merge_selected)
        self.btn_merge.pack(fill=tk.X)
        self.btn_split = ttk.Button(p, text="Split", command=self.split_selected)
        self.btn_split.pack(fill=tk.X, pady=(6, 0))
        self.btn_merge_scenes = ttk.Button(p, text="Merge contours within scenes", command=self.merge_all_within_scenes)
        self.btn_merge_scenes.pack(fill=tk.X, pady=(6, 0))
        self.btn_delete = ttk.Button(p, text="Delete selected", command=self.delete_selected)
        self.btn_delete.pack(fill=tk.X, pady=6)
        self.btn_add = ttk.Button(p, text="Add an outline...", command=self.start_draw)
        self.btn_add.pack(fill=tk.X)
        self.btn_undo = ttk.Button(p, text="Undo", command=self.undo_last)
        self.btn_undo.pack(fill=tk.X, pady=6)
        ttk.Separator(p).pack(fill=tk.X, pady=12)
        ttk.Button(p, text="Change detect parameters", command=self.back_to_threshold).pack(fill=tk.X)
        btn = ttk.Button(p, text="Accept detection for all images", command=self.accept_all)
        btn.pack(fill=tk.X, pady=(6, 0))
        self.accept_all_buttons.append(btn)
        self.image_buttons(p)
        ttk.Label(p, text="Click an outline to select it; shift-click adds to the selection. Merge joins two outlines along "
                          "their closest vertices (the two halves of a section). To draw one, click its points; double-click the "
                          "last point or press C to close it; Esc cancels.", wraplength=300, foreground="gray40").pack(anchor="w", pady=10)
        self.update_slider_states()
        return f

    def image_buttons(self, parent):
        """Previous image / Next image, in the panel at the right (the arrows above the picture do the same)."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(10, 0))
        prev = ttk.Button(row, text="Previous image", command=lambda: self.goto_slide(self.cur - 1))
        prev.pack(side=tk.LEFT, fill=tk.X, expand=True)
        nxt = ttk.Button(row, text="Next image", command=lambda: self.goto_slide(self.cur + 1))
        nxt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.panel_nav.append((prev, nxt))
        grid = ttk.Button(parent, text="Show all images", command=self.toggle_grid)
        grid.pack(fill=tk.X, pady=(6, 0))
        self.grid_buttons.append(grid)

    def on_pick_image(self, event=None):
        i = self.combo.current()
        self.combo.selection_clear()
        self.canvas.focus_set()
        if i >= 0 and i != self.cur:
            self.goto_slide(i)

    def update_nav_buttons(self):
        """Previous / next image (the arrows above the picture, and the buttons in the panel): none at the first or last image, and none
        while all the images are shown on the grid (there is no one image to move from)."""
        last = len(self.project.slides) - 1
        can_prev = not self.grid_mode and self.cur > 0
        can_next = not self.grid_mode and self.cur < last
        self.btn_prev.config(state="normal" if can_prev else "disabled")
        self.btn_fwd.config(state="normal" if can_next else "disabled")
        for prev, nxt in self.panel_nav:
            prev.config(state="normal" if can_prev else "disabled")
            nxt.config(state="normal" if can_next else "disabled")

    def refresh_image_menu(self):
        n = len(self.project.slides)
        names = [f"Image {i + 1} of {n}:  {os.path.basename(s.path)}" for i, s in enumerate(self.project.slides)]
        self.combo.config(values=names)
        for b in self.grid_buttons:             # (a grid of all the images means nothing with just one)
            b.config(state="normal" if n > 1 else "disabled")
        self.combo_chars = max((len(t) for t in names), default=20) + 1      # (wide enough for the longest name)
        self.fit_combo()

    def fit_combo(self):
        """The image menu as wide as its longest name needs, or as the row allows: the arrows stay against it, and the checkboxes and
        the Fit button are never pushed out of view."""
        import tkinter.font as tkfont
        if not hasattr(self, "combo_chars"):
            return
        others = sum(w.winfo_reqwidth() for w in self.nav.grid_slaves(row=0) if w is not self.combo) + 150      # (the gaps, with some to spare)
        avail = self.nav.winfo_width() - others
        char = tkfont.Font(font=("Segoe UI", 11, "bold")).measure("0")
        chars = self.combo_chars if avail <= 1 else min(self.combo_chars, max(int(avail // max(char, 1)), 6))
        self.combo.config(width=min(chars, 80))

    # -- entering and leaving the stage

    def enter_detect(self):
        self.read_infos()
        self.cur = min(self.cur, max(len(self.project.slides) - 1, 0))
        self.goto_slide(self.cur, force=True)
        # With several images, the first time step 2 opens for this list of images it shows them all at once (coming back from step 3 to
        # work on one image does not take that away again).
        signature = tuple(s.path for s in self.project.slides)
        if len(signature) > 1 and signature != getattr(self, "grid_default_for", None):
            self.grid_default_for = signature
            self.set_grid_mode(True)

    def leave_detect(self):
        if self.grid_mode:
            self.set_grid_mode(False)
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
        if self.grid_mode:
            self.set_grid_mode(False)
        self.update_nav_buttons()
        self.refresh_image_menu()
        self.var_slide.set(f"Image {i + 1} of {len(self.project.slides)}:  {os.path.basename(s.path)}")
        if s.path in self.ov:
            self.waiting_for = None
            self.slide_ready()
        else:
            self.base_rgb = None
            self.canvas.delete("all")
            self.canvas.create_text(30, 30, anchor="nw", fill="white", text="Loading the picture...", font=("Segoe UI", 12))
            self.var_status.set("Loading the picture...")
            self.waiting_for = s.path
            self.want(s.path, "gray", urgent=True)

    def slide_ready(self):
        s = self.slide()
        img, geo = self.ov[s.path]
        self.set_base()
        if self.project.shared:                   # (the settings are shared: this image has them too, not its own automatic ones)
            self.apply_settings(s, self.project.shared)
        if s.threshold < 0:
            s.threshold = sp.auto_threshold(img)
        if s.path in self.pending_min_mm2:        # (a size asked for from another image, in mm2: in pixels now that the scale is known)
            s.min_area = max(int(round(self.pending_min_mm2.pop(s.path) / self.px_mm2())), 1)
        if s.min_area < 0:
            s.min_area = max(int(round(DEFAULT_MIN_AREA_MM2 / self.px_mm2())), 1)
        if self.cur not in self.polys:
            self.polys[self.cur] = [self.to_overview(p) for p in s.polygons]
        self.set_mode("edit" if s.accepted else "threshold")
        self.fit_view()

    def set_base(self):
        """The picture shown: the slide in color, or in gray (the picture the threshold is applied to)."""
        s = self.slide()
        gray = self.ov[s.path][0]
        color = self.ovc.get(s.path)
        if self.var_color.get() and color is None:
            self.want(s.path, "color", urgent=True)      # (gray until the color picture is ready)
        self.base_rgb = color if self.var_color.get() and color is not None else np.repeat(gray[:, :, None], 3, axis=2)

    def on_color(self):
        if self.base_rgb is None or self.slide().path not in self.ov:
            return
        path = self.slide().path
        if self.var_color.get() and path not in self.ovc:
            self.var_status.set("Building the color picture...")      # (the gray one stays until it is ready)
            self.want(path, "color", urgent=True)
            return
        self.set_base()
        self.redraw()

    def to_overview(self, poly):
        x0, y0, k = self.ov[self.slide().path][1]
        return (poly - np.array([x0, y0])) / k

    def to_slide(self, poly, path=None):
        x0, y0, k = self.ov[path or self.slide().path][1]
        return poly * k + np.array([x0, y0])

    def update_title(self):
        """The window's title: what this step is for, and the project's file name (or that it is new, not saved yet)."""
        name = os.path.basename(self.project.path) if self.project.path else "new project (not saved yet)"
        self.root.title(f"Load/detect/order sections - {name}")

    def write_project(self, proj):
        """Save the project (to its own file, or, the first time, to a new one whose name the user is asked for) and note it in the list of
        recent projects. False if the user declined to name a new file (nothing is saved then), else True."""
        path = self.save_path()
        if path is None:
            return False
        proj.save(path)
        sp.add_recent_project(path)
        self.update_title()
        return True

    def save_path(self):
        """Where the project is saved: the file it was opened from, else a new file, named in a save dialog that offers the usual name (the
        first image's, or with a number if that is taken by another project, which is left as it is). The file stays in the images' folder
        (a project's paths are relative to it) and its name ends in _project.json, which is how project files are told from other .json
        files. None if the dialog was cancelled."""
        if self.project.path:
            return self.project.path
        default = self.project.new_path()
        folder = os.path.dirname(default)
        initial = os.path.basename(default)
        # said once, just before the first dialog, so that the user knows what the file is for when the dialog comes up
        messagebox.showinfo(
            "Save the project file",
            "You are about to be shown a dialog to choose where to create a new project file.\n\n"
            "This project file will record the images you chose, the outlines found on them, the order of the sections and, "
            "later, their alignment. It is written next to the images and updated as you work, so you can close the program and "
            "come back to where you left off.\n\n"
            "Open it again with \"Open a saved project\" or \"Open recent projects\" in step 1 (for a JPX, adding the same JPX again "
            "offers its project too). The images themselves are not changed or copied.\n\n"
            "The project file must be placed in the images' folder. Its name will end in _project.json.",
            parent=self.root)
        while True:
            chosen = filedialog.asksaveasfilename(
                parent=self.root, title="Save the project file", initialdir=folder, initialfile=initial,
                defaultextension=".json", filetypes=[("Section projects", "*" + sp.PROJECT_SUFFIX), ("All files", "*.*")])
            if not chosen:
                self.var_status.set("The project file was not saved.")
                return None
            chosen = os.path.normpath(chosen)
            name = os.path.basename(chosen)
            if not name.lower().endswith(sp.PROJECT_SUFFIX):
                stem = name[:-5] if name.lower().endswith(".json") else name
                name = stem + sp.PROJECT_SUFFIX
                chosen = os.path.join(os.path.dirname(chosen), name)
            if os.path.normcase(os.path.dirname(os.path.abspath(chosen))) != os.path.normcase(os.path.abspath(folder)):
                messagebox.showinfo("Project file folder", f"The project file has to be in the folder of its images (its paths are relative "
                                    f"to it, so that the project can be moved):\n\n{folder}\n\nYou can change its name.", parent=self.root)
                initial = os.path.basename(chosen)
                continue
            if chosen != default and os.path.exists(chosen) and not messagebox.askyesno(
                    "Replace the project file?", f"{name} already exists and will be replaced.\n\nReplace it?", parent=self.root):
                initial = name
                continue
            self.var_status.set(f"Project file: {name}")
            print(f"Project file: {chosen}", flush=True)
            return chosen

    def sync_project(self, save=True):
        """Write the editor's outlines of every slide visited back into the project, and (unless save is False) save it."""
        for i, polys in self.polys.items():
            s = self.project.slides[i]
            if s.path in self.ov:
                s.polygons = [self.to_slide(p, s.path) for p in polys]
        if save and self.project.slides and any(s.polygons for s in self.project.slides) and self.project.path:
            try:
                self.write_project(self.project)
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
            self.scale_pad.set(s.pad)
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
        k = self.ov[self.slide().path][1][2]
        side = k * (self.slide_px(self.slide()) or 0.65) / 1000
        return side * side

    def min_area_mm2(self):
        return self.slide().min_area * self.px_mm2()

    def update_labels(self):
        s = self.slide()
        self.var_thr_text.set(f"Brightness threshold: {s.threshold}")
        self.var_min_text.set(f"Ignore regions smaller than {self.min_area_mm2():.2f} mm²")
        um = s.smooth * self.ov[s.path][1][2] * (self.slide_px(s) or 0.65)
        pad_um = s.pad * self.ov[s.path][1][2] * (self.slide_px(s) or 0.65)
        self.var_pad_text.set("Padding: none" if not s.pad else f"Padding: {s.pad} px ({pad_um:.0f} \u00b5m)")
        self.var_smooth_text.set("Smoothing: none" if not s.smooth else f"Smoothing: {s.smooth} px ({um:.0f} µm), outward")

    def on_threshold(self, v):
        if self.mode != "threshold" or self.base_rgb is None:
            return
        self.slide().threshold = int(round(float(v)))
        self.update_labels()
        self.sync_shared()
        self.schedule_mask()

    def on_min_area(self, v):
        if self.mode != "threshold" or self.base_rgb is None:
            return
        self.slide().min_area = max(int(round(float(v) / self.px_mm2())), 1)
        self.update_labels()
        self.sync_shared()
        self.schedule_mask()

    def on_smooth(self, v):
        if self.mode != "threshold" or self.base_rgb is None:
            return
        n = int(round(float(v)))
        if n != self.slide().smooth:
            self.slide().smooth = n
            self.update_labels()
            self.sync_shared()
            self.schedule_mask()

    def on_pad(self, v):
        if self.mode != "threshold" or self.base_rgb is None:
            return
        n = int(round(float(v)))
        if n != self.slide().pad:
            self.slide().pad = n
            self.update_labels()
            self.sync_shared()
            self.schedule_mask()

    def schedule_mask(self):
        if self.job is not None:
            self.root.after_cancel(self.job)
        self.job = self.root.after(25, self.recompute_mask)

    def recompute_mask(self):
        self.job = None
        if self.base_rgb is None or self.mode != "threshold" or self.grid_mode:
            return
        s = self.slide()
        img = self.ov[s.path][0]
        self.mask = sp.detect_mask(img, s.threshold, s.min_area, s.smooth, s.pad)
        self.mask_show = sp.fill_holes(self.mask)      # (shown without the holes inside the regions: the outlines go round them)
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

    def apply_settings(self, s, shared):
        """Give image `s` the shared settings: brightness threshold, smoothing, padding, and the smallest region (the same area in mm2,
        whatever the image's scale)."""
        s.threshold, s.smooth, s.pad = shared["threshold"], shared["smooth"], shared["pad"]
        if s.path in self.ov:
            k = self.ov[s.path][1][2]
            side = k * (self.slide_px(s) or 0.65) / 1000
            s.min_area = max(int(round(shared["min_mm2"] / (side * side))), 1)
            self.pending_min_mm2.pop(s.path, None)
        else:
            self.pending_min_mm2[s.path] = shared["min_mm2"]

    def sync_shared(self):
        """While the settings are shared: the image on show sets them for all the others."""
        if not self.project.shared:
            return
        cur = self.slide()
        self.project.shared = dict(threshold=cur.threshold, smooth=cur.smooth, pad=cur.pad, min_mm2=self.min_area_mm2())
        for i, s in enumerate(self.project.slides):
            if i != self.cur:
                self.apply_settings(s, self.project.shared)
        if self.grid_mode:
            self.schedule_grid()

    def on_shared(self):
        """"Same settings for all images": checked, the settings on show become every image's (each image's own are kept); unchecked,
        every image goes back to its own."""
        slides = self.project.slides
        if self.base_rgb is None or not slides:
            self.var_shared.set(bool(self.project.shared))
            return
        if self.var_shared.get():
            for s in slides:
                s.own = dict(threshold=s.threshold, smooth=s.smooth, pad=s.pad, min_area=s.min_area)
            cur = self.slide()
            self.project.shared = dict(threshold=cur.threshold, smooth=cur.smooth, pad=cur.pad, min_mm2=self.min_area_mm2())
            for i, s in enumerate(slides):
                if i != self.cur:
                    self.apply_settings(s, self.project.shared)
            self.var_status.set("The settings on show are now every image's.")
        else:
            for s in slides:
                if s.own:
                    s.threshold, s.smooth, s.pad, s.min_area = (s.own["threshold"], s.own["smooth"], s.own["pad"], s.own["min_area"])
                s.own = None
                self.pending_min_mm2.pop(s.path, None)
            self.project.shared = None
            self.var_status.set("Each image has its own settings again.")
            if self.mode == "threshold":
                self.set_mode("threshold")
        self.update_slider_states()
        if self.grid_mode:
            self.schedule_grid()

    def update_slider_states(self):
        """The sliders work on one image: on the grid of all of them they are greyed out, unless the settings are shared."""
        state = "disabled" if (self.grid_mode and not self.var_shared.get()) else "normal"
        for sc in (self.scale_thr, self.scale_min, self.scale_smooth, self.scale_pad):
            sc.config(state=state)
        for btn in self.accept_all_buttons:       # (the edit view's button: for all the images, usable while they are all shown)
            btn.config(state="normal" if self.grid_mode else "disabled")
        self.accept_button.config(text="Accept detection for all images" if self.grid_mode else "Accept detection for this image")

    def accept_clicked(self):
        """The accept button: the image on show, or all the images when they are shown on the grid."""
        if self.grid_mode:
            self.accept_all()
        else:
            self.accept()

    def accept_all(self, then=None, quiet=False):
        """Accept the detection for every image not accepted yet: each with its own brightness threshold (the automatic one where it
        hasn't been set), and the smoothing, padding and smallest region of the image on show. Images already accepted stay as they are
        (and any can still be edited), so there is no question to answer. `then` is called when it is done (also when there was nothing
        to accept, with `quiet` saying nothing about that)."""
        slides = self.project.slides
        targets = [i for i, s in enumerate(slides) if not s.accepted]
        if not targets:
            if not quiet:
                messagebox.showinfo("Accept all", "Every image has been accepted already.", parent=self.root)
            if then:
                then()
            return
        cur = self.slide()
        smooth, pad, min_mm2 = cur.smooth, cur.pad, self.min_area_mm2()
        numbering = self.project.numbering
        jobs = [(i, slides[i].path, slides[i].threshold, self.slide_px(slides[i]) or 0.65, self.ov.get(slides[i].path)) for i in targets]
        self.var_status.set(f"Finding the sections of {len(targets)} images...")

        def work():
            out = []
            for i, path, thr, px, ov in jobs:
                gray, geo = ov if ov is not None else sp.overview(path)[:2]
                x0, y0, k = geo
                if thr < 0:
                    thr = sp.auto_threshold(gray)
                min_area = max(int(round(min_mm2 / ((k * px / 1000) ** 2))), 1)
                polys = sp.mask_to_polygons(sp.detect_mask(gray, thr, min_area, smooth, pad))
                polys = sp.sort_by_scene(polys, sp.scene_boxes(path), lambda q: q * k + np.array([x0, y0]), numbering)[0]
                out.append((i, thr, min_area, polys, (x0, y0, k)))
            return out

        def done(res):
            self.var_status.set("")
            if isinstance(res, BaseException):
                messagebox.showerror("Can't go on", f"{type(res).__name__}: {res}", parent=self.root)
                return
            for i, thr, min_area, polys, (x0, y0, k) in res:
                s = slides[i]
                s.threshold, s.min_area, s.smooth, s.pad, s.accepted = thr, min_area, smooth, pad, True
                s.polygons = [q * k + np.array([x0, y0]) for q in polys]
                self.polys[i] = polys
                self.undo[i] = []
            self.sync_project()
            if self.cur in targets and self.stage == 1 and self.base_rgb is not None:
                self.selected = set()
                self.set_mode("edit")
            if self.grid_mode:                       # (the grid shows the outlines of the images just accepted, not the blue preview)
                self.start_grid()
            self.var_status.set(f"Accepted the detection for {len(res)} image(s): "
                                f"{sum(len(slides[i].polygons) for i, *_ in res)} outlines in all.")
            if then:
                then()

        self.background(work, done)

    def full_auto(self):
        """The Full auto button: "Accept detection for all images", then "Merge contours within scenes" (for all the images), in that order."""
        if not self.project.slides or self.busy:
            return
        self.accept_all(then=lambda: self.merge_all_within_scenes(everything=True, quiet=True), quiet=True)

    def on_numbering(self):
        """The outlines of every accepted image are numbered anew in the order chosen; the order of the sections (the third tab) starts
        again, since it is by these numbers."""
        mode = self.var_numbering.get()
        if mode == self.project.numbering:
            return
        self.project.numbering = mode
        self.sync_project(save=False)
        for i, s in enumerate(self.project.slides):
            if s.polygons:
                s.polygons = sp.sort_by_scene(s.polygons, sp.scene_boxes(s.path), lambda q: q, mode)[0]
                self.polys.pop(i, None)
                if s.path in self.ov:
                    x0, y0, k = self.ov[s.path][1]
                    self.polys[i] = [(q - np.array([x0, y0])) / k for q in s.polygons]
            self.undo[i] = []
        self.project.order = []
        self.order_key = None
        self.selected = set()
        self.sync_project()
        self.update_edit_help()
        if self.grid_mode:
            self.start_grid()
        self.redraw()

    def sorted_polys(self, polys):
        """The outlines in the order chosen at the top of the panel (see sp.sort_by_scene)."""
        boxes = sp.scene_boxes(self.slide().path)
        return sp.sort_by_scene(polys, boxes, self.to_slide, self.project.numbering)[0]

    def back_to_threshold(self):
        if self.grid_mode:
            # all the images are shown: they all go back to finding the sections (each keeps its brightness threshold and the other settings)
            slides = self.project.slides
            if any(s.accepted or s.polygons for s in slides) and not messagebox.askyesno(
                    "Change detect parameters",
                    f"Finding the sections again replaces the outlines of ALL {len(slides)} images, including any you have edited, and the "
                    "order of the sections made from them. Go on?", parent=self.root):
                return
            for i, s in enumerate(slides):
                s.accepted = False
                s.polygons = []
                self.polys.pop(i, None)
                self.undo[i] = []
            self.project.order = []
            self.order_key = None
            self.selected = set()
            self.sync_project()
            self.set_mode("threshold")
            self.start_grid()                        # (the grid shows the blue preview of each image again)
            self.var_status.set("All the images are back to finding the sections.")
            return
        if self.polys.get(self.cur) and not messagebox.askyesno(
                "Change detect parameters", "Finding the sections again replaces the outlines of this image, including any you have edited. Go on?",
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
            how = {"scene": "in the order of the scenes of the .czi", "rows": "across, then down",
                   "columns": "down, then across"}[self.project.numbering]
            self.var_edit_help.set(f"{n} outline(s), numbered {how}; {sel} selected.")
        for b in (self.btn_merge,):
            b.config(state="normal" if sel >= 2 and self.mode == "edit" else "disabled")
        self.btn_split.config(state="normal" if sel == 1 and self.mode == "edit" else "disabled")
        # (scenes are a .czi's: an image of one section has none to merge within)
        # (on the grid it acts on all the images shown, so it needs just one with scenes; otherwise on the image on show)
        if self.grid_mode:
            usable = any(not sp.is_jp2(s.path) and s.accepted for s in self.project.slides)
        else:
            usable = bool(self.project.slides) and not sp.is_jp2(self.slide().path)
        self.btn_merge_scenes.config(state="normal" if usable else "disabled",
                                     text="Merge contours within scenes (all images)" if self.grid_mode else "Merge contours within scenes")
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
        self.sync_project()
        self.redraw()

    def merge_all_within_scenes(self, everything=False, quiet=False):
        """Merge all the outlines that lie in one scene into a single outline (the fragments of a section: two hemispheres, a piece
        broken off), in the image on show, or in every image while all of them are shown on the grid (or when `everything` is asked: the
        Full auto button). Outlines are counted in the scene their centre is in. With `quiet` no dialog is shown: there is no question about
        what will be merged, and nothing is said when there is nothing to merge (the status line reports what was done)."""
        slides = self.project.slides
        self.sync_project(save=False)
        many = self.grid_mode or everything
        shown = range(len(slides)) if many else [self.cur]
        plan = []
        for i in shown:
            s = slides[i]
            if not s.accepted or len(s.polygons) < 2 or sp.is_jp2(s.path):      # (only a .czi has scenes)
                continue
            boxes = sp.scene_boxes(s.path)
            groups = sp.scene_groups(s.polygons, boxes)          # (the scenes with more than one outline, and which outlines)
            merged, n = sp.merge_within_scenes(s.polygons, boxes, self.project.numbering)
            if n:
                plan.append((i, merged, n, len(s.polygons) - len(merged), groups))
        if not plan:
            if not quiet:
                messagebox.showinfo("Merge contours within scenes", "No scene of " + ("any image shown" if many else "this image")
                                    + " has more than one outline.", parent=self.root)
            return
        total_scenes = sum(n for _, _, n, _, _ in plan)
        total_outlines = sum(d for _, _, _, d, _ in plan)
        # which outlines are involved: "scene 4: outlines 4 and 5", under the image's name when several images are
        numbers = lambda idx: ", ".join(str(k + 1) for k in idx[:-1]) + (" and " if len(idx) > 1 else "") + str(idx[-1] + 1)
        lines = []
        for i, _, _, _, groups in plan:
            if many:
                lines.append(f"{os.path.basename(slides[i].path)}:")
            lines += [("   " if many else "") + f"scene {sc + 1}: outlines {numbers(idx)}" for sc, idx in groups]
        listing = "\n".join(lines[:16]) + (f"\n... and {len(lines) - 16} more line(s)" if len(lines) > 16 else "")
        where = f"In {len(plan)} image(s)," if many else "In this image,"
        undo_note = "Only the image on the panel can be undone afterwards. " if many else "Undo goes back. "
        if not quiet and not messagebox.askyesno(          # (quiet, the Full auto button: no details to read; the status line says what was done)
                "Merge contours within scenes",
                f"{where} {total_scenes} scene(s) have more than one outline. Merging each scene's outlines into one "
                f"takes away {total_outlines} outline(s):\n\n{listing}\n\nFragments are joined along their closest points, so a small piece far "
                "from the section would be joined to it by a thin bridge: delete stray pieces first. " + undo_note + "Go on?",
                parent=self.root):
            return
        for i, merged, n, d, _ in plan:
            s = slides[i]
            if i == self.cur and i in self.polys:
                self.push_undo()                       # (the image on show can be undone)
            s.polygons = merged
            self.polys.pop(i, None)
            if s.path in self.ov:
                x0, y0, k = self.ov[s.path][1]
                self.polys[i] = [(q - np.array([x0, y0])) / k for q in merged]
        self.selected = set()
        self.sync_project()
        self.update_edit_help()
        if self.grid_mode:
            self.start_grid()
        self.redraw()
        self.var_status.set(f"Merged the outlines within {total_scenes} scene(s) of {len(plan)} image(s): {total_outlines} outline(s) fewer.")

    def split_selected(self):
        """Cut the selected outline in two along the chord whose ends are closest together for the way round the outline between them
        (see sp.split_polygon): the pieces of a pinched outline come apart, and a merge is undone."""
        if self.stage != 1 or self.mode != "edit" or len(self.selected) != 1:
            return
        i = next(iter(self.selected))
        parts = sp.split_polygon(self.polys[self.cur][i])
        if parts is None:
            self.var_status.set("This outline is too small to split.")
            return
        self.push_undo()
        keep = [q for k, q in enumerate(self.polys[self.cur]) if k != i] + list(parts)
        self.polys[self.cur] = self.sorted_polys(keep)
        self.selected = {k for k, q in enumerate(self.polys[self.cur]) if any(q is part for part in parts)}
        self.update_edit_help()
        self.sync_project()
        self.redraw()
        self.var_status.set("Split into two outlines (Undo goes back).")

    def merge_key(self):
        """The M key: the same as the Merge button (which is usable only for two or more selected outlines in the editing view)."""
        if self.stage == 1 and self.mode == "edit" and not self.grid_mode and len(self.selected) >= 2:
            self.merge_selected()

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
        self.sync_project()           # (saved at once: an edit is not lost if the window is closed)
        self.redraw()

    def delete_selected(self):
        if self.stage != 1 or self.mode != "edit" or not self.selected:
            return
        self.push_undo()
        self.polys[self.cur] = [p for i, p in enumerate(self.polys[self.cur]) if i not in self.selected]
        self.selected = set()
        self.update_edit_help()
        self.sync_project()
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
        self.sync_project()
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

    # -- all the images on a grid

    def toggle_grid(self):
        self.set_grid_mode(not self.grid_mode)

    def set_grid_mode(self, on):
        self.grid_mode = on
        for b in self.grid_buttons:
            b.config(text="Show one image" if on else "Show all images")
        if on:
            self.sync_project(save=False)             # (outlines worked on are written back, to be drawn here)
            self.selected, self.draw_pts = set(), []
            self.grid_rgb = {}
            for s in self.project.slides:
                self.want(s.path)
            self.canvas.config(cursor="arrow")
        self.update_slider_states()
        self.update_nav_buttons()
        self.update_edit_help()                   # (the merge-within-scenes button follows what is shown)
        if on:
            self.start_grid()
        self.redraw()

    def grid_layout(self):
        """(columns, rows, picture width, picture height) of the grid that gives the pictures the most room in the window."""
        cw, ch = self.canvas_size()
        n = max(len(self.project.slides), 1)
        ars = [self.ov[s.path][0].shape[1] / self.ov[s.path][0].shape[0] for s in self.project.slides if s.path in self.ov]
        ar = float(np.median(ars)) if ars else 1.6
        best = None
        for cols in range(1, n + 1):
            rows = -(-n // cols)
            w, h = cw / cols - 10, ch / rows - 10 - GRID_LABEL
            if w < 20 or h < 20:
                continue
            size = min(w, h * ar)
            if best is None or size > best[0]:
                best = (size, cols, rows, w, h)
        if best is None:
            return n, 1, max(cw / n - 10, 20), max(ch - 10 - GRID_LABEL, 20)
        return best[1:]

    def schedule_grid(self):
        if self.grid_job is not None:
            self.root.after_cancel(self.grid_job)
        self.grid_job = self.root.after(150, self.start_grid)

    def start_grid(self):
        """Make the pictures of the grid in the background (the settings and outlines of each image as they are now)."""
        self.grid_job = None
        if not self.grid_mode:
            return
        self.grid_version += 1
        cols, rows, w, h = self.grid_layout()
        self.grid_size = self.canvas_size()
        color = bool(self.var_color.get())
        jobs = []
        for i, s in enumerate(self.project.slides):
            ov = self.ov.get(s.path)
            if ov is None:
                continue
            gray, (x0, y0, k) = ov
            side = k * (self.slide_px(s) or 0.65) / 1000
            if s.path in self.pending_min_mm2:
                min_area = max(int(round(self.pending_min_mm2[s.path] / (side * side))), 1)
            elif s.min_area >= 0:
                min_area = s.min_area
            else:
                min_area = max(int(round(DEFAULT_MIN_AREA_MM2 / (side * side))), 1)
            polys = None
            if s.accepted:
                polys = self.polys.get(i)
                if polys is None:
                    polys = [(p - np.array([x0, y0])) / k for p in s.polygons]
            jobs.append((i, gray, self.ovc.get(s.path) if color else None, s.threshold if s.threshold >= 0 else None,
                         min_area, s.smooth, s.pad, polys))
        threading.Thread(target=self.grid_worker, args=(self.grid_version, jobs, int(w), int(h)), daemon=True).start()

    def grid_worker(self, version, jobs, tw, th):
        ns.lower_thread_priority()
        for i, gray, rgb_full, thr, min_area, smooth, pad, polys in jobs:
            if version != self.grid_version:
                return
            H, W = gray.shape
            f = min(tw / W, th / H)
            w2, h2 = max(int(W * f), 1), max(int(H * f), 1)
            small = cv2.resize(gray, (w2, h2), interpolation=cv2.INTER_AREA)
            base = cv2.resize(rgb_full, (w2, h2), interpolation=cv2.INTER_AREA) if rgb_full is not None else np.repeat(small[:, :, None], 3, axis=2)
            if polys is None:      # (not accepted: the detection as the sliders have it now, at this size)
                t = thr if thr is not None else sp.auto_threshold(gray)
                sm = max(int(round(smooth * f)), 1) if smooth else 0
                pd = max(int(round(pad * f)), 1) if pad else 0
                mask = sp.fill_holes(sp.detect_mask(small, t, max(int(round(min_area * f * f)), 1), sm, pd)).astype(bool)
                tinted = base.astype(np.float32)
                a = 0.3 if rgb_full is not None else 0.55
                tinted[mask] = tinted[mask] * (1 - a) + TINT * a
                base = tinted.astype(np.uint8)
            else:                  # (accepted: its outlines)
                for n, q in enumerate(polys, 1):
                    pts = np.asarray(q, float) * f
                    cv2.polylines(base, [np.round(pts).astype(np.int32)], True, (255, 212, 0), 1, cv2.LINE_AA)
                    if len(polys) < 2:
                        continue                     # (a lone outline needs no number: a 1 on every image looks odd)
                    # the outline's number, as in the single-image view, at its centre (its size follows the outline's)
                    scale = float(np.clip(np.sqrt(abs(sp.signed_area(pts))) / 70, 0.35, 0.9))
                    thick = 1 if scale < 0.6 else 2
                    (tw_, th_), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                    cx, cy = sp.polygon_centroid(pts)
                    cv2.putText(base, str(n), (int(cx - tw_ / 2), int(cy + th_ / 2)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 212, 0), thick, cv2.LINE_AA)
            self.msgs.put(("grid", (version, i, base)))

    def draw_grid(self):
        c = self.canvas
        slides = self.project.slides
        if not slides:
            return
        size = self.canvas_size()
        if self.grid_size is not None and size != self.grid_size and self.grid_job is None:
            self.schedule_grid()                        # (the window changed size: the pictures are made again to fit)
        cols, rows, w, h = self.grid_layout()
        cw, ch = size
        self.grid_cells, self.grid_photos = [], []
        for i, s in enumerate(slides):
            col, row = i % cols, i // cols
            x0, y0 = col * (cw / cols) + 5, row * (ch / rows) + 5
            rgb = self.grid_rgb.get(i)
            if rgb is not None:
                ph = ImageTk.PhotoImage(Image.fromarray(rgb), master=self.root)
                self.grid_photos.append(ph)
                ix, iy = x0 + (w - rgb.shape[1]) / 2, y0 + (h - rgb.shape[0]) / 2
                c.create_image(ix, iy, image=ph, anchor="nw")
                box = (ix, iy, ix + rgb.shape[1], iy + rgb.shape[0])
            else:
                box = (x0, y0, x0 + w, y0 + h)
                c.create_rectangle(*box, outline="#555555")
                c.create_text((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, text="loading...", fill="#a0a0a0", font=("Segoe UI", 10))
            if i == self.cur:
                c.create_rectangle(box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2, outline=ACCENT, width=3)
            name = os.path.basename(s.path)
            c.create_text(x0 + 2, y0 + h + 3, anchor="nw", fill="white", font=("Segoe UI", 9, "bold" if i == self.cur else "normal"),
                          text=f"{i + 1}  {name if len(name) < 34 else name[:15] + '...' + name[-16:]}" + ("   \u2713" if s.accepted else ""))
            self.grid_cells.append((box[0], box[1], box[2], box[3] + GRID_LABEL, i))

    def fit_scale(self):
        h, w = self.base_rgb.shape[:2]
        cw, ch = self.canvas_size()
        return min(cw / w, ch / h) * 0.98

    def clamp_view(self, view):
        """The view with its zoom no smaller than the starting one (the whole picture in the window), and panned no further than the
        picture's edges: there is nothing to see beyond it."""
        s, ox, oy = view
        s = min(max(s, self.fit_scale()), 40)
        h, w = self.base_rgb.shape[:2]
        cw, ch = self.canvas_size()
        ox = (cw - w * s) / 2 if w * s <= cw else min(0.0, max(cw - w * s, ox))
        oy = (ch - h * s) / 2 if h * s <= ch else min(0.0, max(ch - h * s, oy))
        return (s, ox, oy)

    def to_canvas(self, pts):
        s, ox, oy = self.view
        return np.asarray(pts, float) * s + np.array([ox, oy])

    def to_image(self, x, y):
        s, ox, oy = self.view
        return (x - ox) / s, (y - oy) / s

    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.stage == 1 and self.grid_mode:
            self.draw_grid()
            return
        if self.stage != 1 or self.base_rgb is None:
            return
        if self.view is None:
            self.fit_view()
            return
        cw, ch = self.canvas_size()
        self.view = self.clamp_view(self.view)
        s, ox, oy = self.view
        img = self.base_rgb
        if self.mode == "threshold" and self.mask is not None:
            shown = self.mask_show if self.mask_show is not None else self.mask
            tinted = img.astype(np.float32)
            m = shown.astype(bool)
            color = bool(self.var_color.get()) and self.slide().path in self.ovc
            a = 0.3 if color else 0.55      # (lighter over the colors, with the edge of the region drawn solid)
            tinted[m] = tinted[m] * (1 - a) + TINT * a
            if color:
                tinted[m & ~cv2.erode(shown, np.ones((3, 3), np.uint8)).astype(bool)] = TINT
            img = tinted.astype(np.uint8)
        M = np.array([[s, 0, ox], [0, s, oy]], np.float32)
        out = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self.photo = ImageTk.PhotoImage(Image.fromarray(out), master=self.root)
        c.create_image(0, 0, image=self.photo, anchor="nw")
        c.create_text(8, ch - 6, anchor="sw", fill="#a0a0a0", font=("Segoe UI", 9), text="Wheel: zoom    Right button: pan")
        if self.var_scenes.get() and not sp.is_jp2(self.slide().path):
            # the scenes of the scan, as dashed rectangles (a black line under the white one keeps them visible on bright tissue)
            x0, y0, k = self.ov[self.slide().path][1]
            for n, (bx, by, bw, bh) in enumerate(sp.scene_boxes(self.slide().path), 1):
                a = self.to_canvas([((bx - x0) / k, (by - y0) / k)])[0]
                b = self.to_canvas([((bx + bw - x0) / k, (by + bh - y0) / k)])[0]
                c.create_rectangle(a[0], a[1], b[0], b[1], outline="black", width=3)
                c.create_rectangle(a[0], a[1], b[0], b[1], outline="white", width=1, dash=(5, 4))
                c.create_text(a[0] + 6, a[1] + 4, anchor="nw", text=f"scene {n}", fill="white", font=("Segoe UI", 9, "italic"))
        if self.mode != "threshold":
            boxes = sp.scene_boxes(self.slide().path)
            for i, p in enumerate(self.polys.get(self.cur, [])):
                pts = self.to_canvas(p)
                sel = i in self.selected
                color = SELECT_COLOR if sel else OUTLINE_COLOR
                c.create_polygon(*pts.ravel(), outline=color, fill="", width=3 if sel else 2)
                cx, cy = self.to_canvas(sp.polygon_centroid(p))
                lone = len(self.polys.get(self.cur, [])) < 2      # (an image with just one outline: no number, a 1 on every image looks odd)
                if not lone:
                    c.create_text(cx, cy, text=str(i + 1), fill=color, font=("Segoe UI", 16, "bold"))
                sx, sy = self.to_slide(sp.polygon_centroid(p))
                scene = sp.scene_of(boxes, sx, sy) + 1
                if scene != i + 1:     # (the number is the place in the scene order; where it differs, say which scene it is in)
                    c.create_text(cx, cy if lone else cy + 20, text=f"scene {scene}", fill=color, font=("Segoe UI", 9))
        if self.mode == "draw" and self.draw_pts:
            pts = self.to_canvas(self.draw_pts)
            if len(pts) > 1:
                c.create_line(*pts.ravel(), fill=DRAW_COLOR, width=2)
            for x, y in pts:
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=DRAW_COLOR, outline="")

    def on_wheel(self, e):
        if self.grid_mode or self.base_rgb is None or self.view is None:
            return
        s, ox, oy = self.view
        f = 1.25 if e.delta > 0 else 1 / 1.25
        s2 = min(max(s * f, self.fit_scale()), 40)      # (zooming out stops at the starting view)
        f = s2 / s
        self.view = self.clamp_view((s2, e.x - (e.x - ox) * f, e.y - (e.y - oy) * f))
        self.redraw()

    def pan_start(self, e):
        if self.grid_mode:
            return
        self.pan = (e.x, e.y, self.view)

    def pan_move(self, e):
        if not self.grid_mode and self.pan and self.view:
            x, y, (s, ox, oy) = self.pan
            self.view = self.clamp_view((s, ox + e.x - x, oy + e.y - y))
            self.redraw()

    def on_click(self, e, shift=False):
        if self.stage == 1 and self.grid_mode:
            for x0, y0, x1, y1, i in self.grid_cells:      # (a click on an image opens it)
                if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                    self.goto_slide(i)
                    return
            return
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
        if not self.grid_mode and self.mode == "draw":
            self.close_polygon()

    def on_motion(self, e):
        if not self.grid_mode and self.mode == "draw" and self.draw_pts and self.view:
            self.redraw()
            last = self.to_canvas([self.draw_pts[-1]])[0]
            self.canvas.create_line(last[0], last[1], e.x, e.y, fill=DRAW_COLOR, dash=(4, 3))

    # ------------------------------------------------------------------ stage 3: order sections

    def build_order(self):
        f = ttk.Frame(self.body)
        top = ttk.Frame(f, padding=(12, 8))
        top.pack(fill=tk.X)
        self.var_order_text = tk.StringVar(f, "")
        # the grid and what goes with it (Reverse, Reset, Undo, the sizes, the right-click menu) are the shared editor's: see order_editor.py
        self.editor = OrderEditor(f, self, enabled=lambda: self.stage == 2,
                                  after_draw=lambda: self.var_order_text.set(f"{len(self.grid.order)} section(s). Drag a picture to reorder."))
        self.grid = self.editor.grid
        help_label = ttk.Label(top, textvariable=self.var_order_text, foreground="gray30")
        self.editor.reverse_button(top).pack(side=tk.RIGHT)
        self.var_spacing = tk.StringVar(f, "")
        ttk.Entry(top, textvariable=self.var_spacing, width=7).pack(side=tk.RIGHT, padx=(4, 18))
        ttk.Label(top, text="Distance between adjacent sections (\u00b5m):").pack(side=tk.RIGHT)
        self.editor.reset_button(top).pack(side=tk.RIGHT, padx=8)
        self.editor.undo_button(top).pack(side=tk.RIGHT)
        self.var_fit_thumbs = tk.BooleanVar(f, True)
        ttk.Checkbutton(top, text="Scale sections to fit thumbnails", variable=self.var_fit_thumbs,
                        command=self.on_fit_thumbs).pack(side=tk.RIGHT, padx=8)
        # brightness of a JPX's thumbnails, as the alignment tool's buttons: Auto stretches each picture on its own, Full scale shows the
        # file's numbers as they are (the thumbnails of scans are made from one overview of the slide, with one stretch for all)
        self.var_bright = tk.StringVar(f, "auto")
        bright = ttk.Frame(top)
        bright.pack(side=tk.RIGHT, padx=8)
        ttk.Label(bright, text="Brightness:").pack(side=tk.LEFT, padx=(0, 4))
        self.bright_buttons = []
        for text, value in (("Auto", "auto"), ("Full scale", "full")):
            b = ttk.Radiobutton(bright, text=text, value=value, variable=self.var_bright, style="Toolbutton", command=self.on_bright)
            b.pack(side=tk.LEFT, padx=1)
            self.bright_buttons.append(b)
        self.editor.size_buttons(top).pack(side=tk.RIGHT, padx=8)
        help_label.pack(side=tk.LEFT)      # (packed last: if the window is narrow it is the help text that is cut, not the buttons)
        self.grid.frame.pack(fill=tk.BOTH, expand=True)
        return f

    @property
    def order(self):
        """[(slide index, polygon index)] in the order of the sections (the shared thumbnail grid keeps it)."""
        return self.grid.order

    @order.setter
    def order(self, value):
        self.grid.order = value

    @property
    def thumbs(self):
        """(slide index, polygon index) -> PhotoImage"""
        return self.grid.thumbs

    @thumbs.setter
    def thumbs(self, value):
        self.grid.thumbs = value

    def enter_order(self):
        proj = self.project
        self.editor.clear_history()
        if proj.is_jpx:
            self.enter_order_jpx()
            return
        for b in getattr(self, "bright_buttons", ()):
            b.state(["disabled"])                    # (a scan's thumbnails come from one overview of the slide)
        key = proj.polygons_key()
        if proj.order and self.order_key == key and all(si == -1 or (si < len(proj.slides) and pi < len(proj.slides[si].polygons))
                                                         for si, pi in proj.order):
            self.order = list(proj.order)
        else:
            self.order = proj.default_order()
            self.order_key = key
        self.thumbs = {}
        self.var_spacing.set(f"{proj.z_spacing_um:g}" if proj.z_spacing_um > 0 else "")
        self.make_thumbs()
        self.draw_order()

    def make_thumbs(self):
        need = {(si, pi) for si, pi in self.order if si >= 0}
        paths = {self.project.slides[si].path for si, _ in need if self.project.slides[si].path not in self.ov}
        for p in paths:
            self.want(p, "gray", urgent=True)
        if paths:
            self.var_status.set("Loading the pictures...")
        else:
            self.build_thumbs()

    def finish_thumbs(self):
        """The ordering stage is waiting for pictures: once all are here, the thumbnails are made."""
        if all(self.project.slides[si].path in self.ov for si, _ in self.order if si >= 0):
            self.var_status.set("")
            self.build_thumbs()
            self.draw_order()

    def on_bright(self):
        """Auto / Full scale: the thumbnails of a JPX are made again from the pictures already read. The old ones stay on screen and
        are each replaced as its new version is made (clearing them first would show black cells until they were rebuilt)."""
        self.bright_mode = self.var_bright.get()
        if self.project.is_jpx:
            self.stale_thumbs = set(self.thumbs)

    def aligner_brightness(self):
        """"auto" or "full" for the alignment tool to start with (the Brightness buttons of a JPX's ordering tab), or None for slide
        scans, whose thumbnails have no such setting."""
        return self.bright_mode if self.project.is_jpx else None

    def order_size_changed(self, scale):
        """Small / Medium / Large thumbnails (half, one and two times the usual size): made again at the new size. A JPX's pictures are read
        again only when they must be sharper (large); the thumbnails on screen stay until each is replaced by one of the new size."""
        if self.project.is_jpx:
            target = 512 if scale > 1.0 else 256
            if target != self.jpx_target:
                self.jpx_target = target
                self.jpx_crops = {}                  # (a picture read for the other size is not wanted; one still arriving is ignored)
            self.stale_thumbs = set(self.thumbs)
            self.draw_order()
            return
        self.thumbs = {}
        self.build_thumbs()
        self.draw_order()

    def on_fit_thumbs(self):
        """"Scale sections to fit thumbnails" changed: the thumbnails are made again."""
        if self.project.is_jpx:
            self.stale_thumbs = set(self.thumbs)     # (made again from the pictures already read, a few at a time, as the window asks;
            return                                   # the old ones stay on screen until each is replaced)
        self.thumbs = {}
        self.build_thumbs()
        self.draw_order()

    def build_thumbs(self):
        """The thumbnail of each section: a crop of its slide's picture around its outline. Each is scaled up or down to fill its
        thumbnail (the default), or, unchecked, all are drawn to one scale (the one at which the biggest section just fits), so that
        sizes can be compared; the scale is in um, so that slides of different pixel sizes agree."""
        fit = self.var_fit_thumbs.get()
        items = []                                   # (key, crop box a, b, outline in overview pixels, um per overview pixel)
        for si, pi in self.order:
            if si < 0:
                continue                             # (a placeholder: its thumbnail is made below)
            s = self.project.slides[si]
            if s.path not in self.ov:
                continue
            img, (x0, y0, k) = self.ov[s.path]
            poly = (s.polygons[pi] - np.array([x0, y0])) / k
            lo, hi = poly.min(0), poly.max(0)
            pad = 0.08 * (hi - lo).max() + 4
            a = np.maximum(np.floor(lo - pad), 0).astype(int)
            b = np.minimum(np.ceil(hi + pad), [img.shape[1], img.shape[0]]).astype(int)
            if b[0] <= a[0] or b[1] <= a[1]:
                continue
            items.append(((si, pi), a, b, poly, k * (self.slide_px(s) or 0.65)))
        fits = {key: min(self.grid.tw / (b[0] - a[0]), self.grid.th / (b[1] - a[1])) for key, a, b, _, _ in items}
        # one scale for all, in thumbnail pixels per um: the smallest of those that make each section fit
        common = min((fits[key] / u for key, _, _, _, u in items), default=1.0)
        for key, a, b, poly, u in items:
            if key in self.thumbs:
                continue
            s = self.project.slides[key[0]]
            img = self.ov[s.path][0]
            f = fits[key] if fit else common * u
            crop = img[a[1]:b[1], a[0]:b[0]]
            color = self.ovc.get(s.path)
            crop3 = color[a[1]:b[1], a[0]:b[0]] if self.var_color.get() and color is not None else np.repeat(crop[:, :, None], 3, axis=2)
            rgb = cv2.resize(crop3, (max(int(crop.shape[1] * f), 1), max(int(crop.shape[0] * f), 1)), interpolation=cv2.INTER_AREA)
            cv2.polylines(rgb, [((poly - a) * f).astype(np.int32)], True, (255, 212, 0), 1, cv2.LINE_AA)
            self.thumbs[key] = ImageTk.PhotoImage(Image.fromarray(rgb), master=self.root)
        for key in self.order:                       # (the placeholders: a grey box)
            if key[0] < 0 and key not in self.thumbs:
                self.thumbs[key] = ImageTk.PhotoImage(Image.fromarray(placeholder_image(self.grid.tw, self.grid.th)), master=self.root)

    # ---- the ordering tab of a JPX project: the thumbnails come from the JPX's own frames, only those in view (and a few rows more), a few at a
    # time in the background, and are kept on disk

    def enter_order_jpx(self):
        proj = self.project
        self.thumbs = {}
        self.stale_thumbs = set()
        for b in getattr(self, "bright_buttons", ()):
            b.state(["!disabled"])
        self.var_spacing.set(f"{proj.z_spacing_um:g}" if proj.z_spacing_um > 0 else "")
        if self.jpx_stack is not None and self.jpx_stack_path == proj.jpx_path:
            self.jpx_ready()
            return
        self.order = []
        self.draw_order()
        self.var_status.set("Reading the NeuroInfo files...")
        path, alignment = proj.jpx_path, proj.alignment

        def ask(*args, **kwargs):
            """A NeuroInfo file to choose (see ns.open_stack): the dialog is made, and waited for, by the window's thread."""
            chosen = (self.neuro.get(path) or {}).get(args[0]) if args else None
            if chosen:
                return chosen                            # (the user located it when the JPX was added)
            finished, answer = threading.Event(), {}

            def run():
                try:
                    answer["value"] = neuroinfo_dialogs.locate_file(self.root, *args, **kwargs)
                except BaseException as e:
                    answer["error"] = e
                finished.set()

            self.msgs.put(("call", run))
            finished.wait()
            if "error" in answer:
                raise answer["error"]
            return answer["value"]

        def done(res):
            self.var_status.set("")
            if isinstance(res, BaseException):
                messagebox.showerror("Can't open the JPX", f"{type(res).__name__}: {res}", parent=self.root)
                self.show_stage(0)
                return
            self.jpx_stack, self.jpx_stack_path = res, path
            self.jpx_crops, self.jpx_inflight = {}, set()
            self.start_prelocate(res)
            if self.stage == 2 and self.project.is_jpx:
                self.jpx_ready()

        self.background(lambda: ns.open_stack(path, locate=ask, alignment=alignment), done)

    def start_prelocate(self, stack):
        """Locating the .czi sections on their slide scans is the slow part of opening the alignment tool, and it needs only the stack that
        has just been read: it starts now, while the order is being chosen, so that by the time of Next most of it (often all) is done. The
        alignment tool picks it up from stack.prelocate (waits for what is left, and shows the sections as they are located)."""
        if getattr(stack, "prelocate", None) is not None or getattr(self, "reusing", False):
            return                                   # (coming back from the alignment tool, which keeps its own located stack: nothing to do)
        if not any(s.source_path.lower().endswith(".czi") for s in stack.sections):
            return
        pre = {"done": threading.Event(), "calibrated": threading.Event(), "thumbs_done": threading.Event(), "problems": {}, "located": [],
               "text": "", "thumbs": {}, "scenes": {}, "lock": threading.Lock(), "stop": threading.Event(), "frames": None}
        stack.prelocate = pre

        def work():
            ns.lower_thread_priority()
            needs_locating = any(s.source_path.lower().endswith(".czi") and s.czi_origin is None for s in stack.sections)
            if not needs_locating:
                pre["calibrated"].set()                  # (nothing to locate: the alignment tool need not wait for the overview pictures below,
                                                         # which take a long time for many scans and are only a convenience)
            # the slide scans' overview pictures first (a few seconds each): the alignment tool shows them above its list as they are
            # made, instead of waiting behind the locating, which keeps each scan busy
            for path in sorted({s.source_path for s in stack.sections if s.source_path.lower().endswith(".czi")}):
                if pre["stop"].is_set():
                    break
                try:
                    pre["thumbs"][path] = ns.czi_slide_thumbnail(path, 1600)
                except Exception:
                    pass                                 # (the picture is only a convenience)
            pre["thumbs_done"].set()
            try:
                if needs_locating:
                    problems = ns.calibrate_stack_czi(stack, progress=lambda i, n, text: pre.__setitem__("text", text),
                                                      on_located=pre["located"].append)
                    pre["problems"] = {k: v for k, v in problems.items() if v}
            except BaseException as e:               # (a failed install of the reader library, for one)
                pre["problems"] = {"the .czi scans": str(e) or type(e).__name__}
            pre["calibrated"].set()
            # then the sections' pictures (the scene of each, as the alignment tool shows them), in the order chosen so far: the alignment
            # tool takes them from here instead of reading them from the scans itself
            try:
                by_frame = {s.frame: s for s in stack.sections}
                ordered = [by_frame[f] for f in (pre["frames"] or []) if f in by_frame] or list(stack.sections)
                todo = [s for s in ordered if s.source_path.lower().endswith(".czi") and s.czi_origin is not None]
                ns.render_czi_previews(stack, todo, pre["scenes"], pre["lock"], stop=pre["stop"])      # (the scans in parallel)
            except BaseException:
                pass
            pre["done"].set()

        threading.Thread(target=work, daemon=True).start()

    def jpx_ready(self):
        """The JPX's stack is open: its sections go in the grid (in the project's order, if it has one) and their pictures are asked for."""
        proj, stack = self.project, self.jpx_stack
        for f in proj.jpx_order:       # (placeholders inserted before: new blank frames, which the JPX does not have yet)
            if f >= ns.PLACEHOLDER_FRAME_BASE and not any(s.frame == f for s in stack.sections):
                stack.sections.append(ns.placeholder_section(len(stack.sections), frame=f))
        self.jpx_by_frame = {s.frame: s for s in stack.sections}
        order = [f for f in proj.jpx_order if f in self.jpx_by_frame]
        pre = getattr(stack, "prelocate", None)
        if pre is not None:
            pre["frames"] = list(order) or None      # (the pictures are made ahead in this order: see start_prelocate)
        named = set(order)
        if not proj.jpx_order:        # (a saved order names the sections there are: any it leaves out were deleted)
            order += [s.frame for s in stack.sections if s.frame not in named]
        self.order = [(0, f) for f in order]
        if proj.path and not proj.jpx_sections and self.order:
            # (the project is being opened, and has no count of its sections yet: now that they are known, the list of recent projects can
            # show it. Only this one value is written into the file.)
            proj.jpx_sections = len(self.order)
            try:
                sp.record_jpx_sections(proj.path, proj.jpx_sections)
            except (OSError, ValueError):
                pass
        self.editor.clear_history()
        if proj.z_spacing_um <= 0 and stack.z_spacing_um > 0:
            self.var_spacing.set(f"{stack.z_spacing_um:g}")
        self.draw_order()
        if self.jpx_tick_id is not None:
            self.root.after_cancel(self.jpx_tick_id)
        self.jpx_tick()

    def jpx_tick(self):
        """A few times a second: the thumbnails in view (and a few rows beyond) that aren't made are made, or their pictures asked for."""
        self.jpx_tick_id = None
        if self.closing or self.stage != 2 or not self.project.is_jpx or self.jpx_stack is None:
            return
        canvas = self.grid.canvas
        cols, _cw, row_h = self.grid.cell_geometry()
        first_row = max(int((canvas.canvasy(0) - CELL_PAD) // row_h) - 1, 0)
        last_row = int((canvas.canvasy(max(canvas.winfo_height(), 1)) - CELL_PAD) // row_h) + 2
        made = False
        for key in self.grid.order[first_row * cols:(last_row + 1) * cols]:
            if key in self.thumbs and key not in self.stale_thumbs:
                continue
            crop = self.jpx_crops.get(key[1], 0)
            if crop is False:
                continue                             # (this frame couldn't be read: a blank thumbnail)
            ns.foreground_active()                   # (a thumbnail in view is wanted: the background locating and picture-making wait for it)
            if crop:
                made = self.jpx_make_thumb(key) or made
            elif key[1] not in self.jpx_inflight and len(self.jpx_inflight) < 6:
                self.jpx_submit(key[1])
        if made:
            self.draw_order()
        self.jpx_tick_id = self.root.after(200, self.jpx_tick)

    def jpx_submit(self, frame):
        if self.jpx_pool is None:
            self.jpx_pool = ThreadPoolExecutor(3)
        stack, path, section = self.jpx_stack, self.jpx_stack_path, self.jpx_by_frame[frame]
        self.jpx_inflight.add(frame)
        target = self.jpx_target

        def work():
            ns.lower_thread_priority()
            try:
                data = jpx_crop(stack, path, frame, section, target)
            except Exception as e:
                data = e
            self.msgs.put(("jpxcrop", (stack, frame, data, target)))

        self.jpx_pool.submit(work)

    def on_jpx_crop(self, stack, frame, data, target=None):
        if stack is not self.jpx_stack:
            return                                   # (a picture of a stack that has been closed)
        self.jpx_inflight.discard(frame)
        if target is not None and target != self.jpx_target:
            return                                   # (read for another thumbnail size: it will be asked for again)
        if isinstance(data, BaseException):
            print(f"Couldn't read the picture of section {frame + 1}: {type(data).__name__}: {data}", flush=True)
            self.jpx_crops[frame] = False
            return
        self.jpx_crops[frame] = data
        if not self.var_fit_thumbs.get():
            self.stale_thumbs = set(self.thumbs)     # (to one scale for all: a new picture can change that scale, so they are all made again)

    def jpx_make_thumb(self, key):
        """The thumbnail of a section from its picture, fitted to the thumbnail (or, with "Scale sections to fit thumbnails" off, drawn to the
        scale at which the biggest section just fits, so that sizes compare). True if it was made."""
        crop = self.jpx_crops.get(key[1])
        if not crop:
            return False
        raw, poly, um = crop
        rgb = jpx_colorize(raw, self.jpx_stack, self.var_bright.get())
        h, w = rgb.shape[:2]
        if self.var_fit_thumbs.get():
            f = min(self.grid.tw / w, self.grid.th / h)
        else:
            f = min((min(self.grid.tw / c[0].shape[1], self.grid.th / c[0].shape[0]) / c[2] for c in self.jpx_crops.values() if c and len(c[1])),
                    default=1.0) * um
        img = cv2.resize(rgb, (max(int(w * f), 1), max(int(h * f), 1)), interpolation=cv2.INTER_AREA)
        section = self.jpx_by_frame.get(key[1])
        if section is not None and getattr(section, "placeholder", False):
            img = placeholder_image(self.grid.tw, self.grid.th)                     # (a grey box saying what it is)
        elif len(poly):
            cv2.polylines(img, [(poly * f).astype(np.int32)], True, (255, 212, 0), 1, cv2.LINE_AA)
        self.thumbs[key] = ImageTk.PhotoImage(Image.fromarray(img), master=self.root)
        self.stale_thumbs.discard(key)
        return True

    def draw_order(self):
        self.grid.draw()

    # ---- what the shared editor (order_editor.py) asks of this window

    def order_status(self, text):
        self.var_status.set(text)

    def order_caption(self, key, i):
        if key[0] < 0:                               # (a placeholder in a scan project: key (-1, k))
            return f"{i + 1}   placeholder", "new"
        if self.project.is_jpx:                      # (key: (0, frame): #, the section's place in the JPX as it was)
            sec = self.jpx_by_frame.get(key[1])
            if sec is not None and sec.placeholder:      # (one NeuroInfo made is a frame of the JPX; one made here is new)
                return f"{i + 1}   placeholder", "new" if ns.is_new_placeholder(sec) else f"#{key[1] + 1}"
            return f"{i + 1}   {os.path.splitext(sec.name)[0][-9:] if sec else ''}", f"#{key[1] + 1}"
        slide = self.project.slides[key[0]]
        return f"{i + 1}   {os.path.splitext(os.path.basename(slide.path))[0][-9:]}", f"#{key[1] + 1}"   # (#: its outline's number in tab 2)

    def order_new_placeholder(self, position):
        """A placeholder, a blank section, to go in the order: in a JPX project a section of the stack with a frame number of its own (its blank
        frame is written when the stack is exported); in a project of scans or images a key (-1, number) that the project's order keeps."""
        if self.project.is_jpx:
            stack = self.jpx_stack
            if stack is None:
                return None
            used = [s.frame for s in stack.sections if s.frame >= ns.PLACEHOLDER_FRAME_BASE]
            frame = max(used + [ns.PLACEHOLDER_FRAME_BASE - 1]) + 1
            section = ns.placeholder_section(len(stack.sections), frame=frame)
            stack.sections.append(section)
            self.jpx_by_frame[frame] = section
            return (0, frame)
        used = [k for si, k in self.order if si < 0]
        return (-1, max(used + [-1]) + 1)

    def order_restore(self, keys):
        """Undo is about to put the order `keys` back: a placeholder that Reset order took away of a JPX's stack is made again."""
        if not (self.project.is_jpx and self.jpx_stack is not None):
            return
        for key in keys:
            if key[1] >= ns.PLACEHOLDER_FRAME_BASE and not any(s.frame == key[1] for s in self.jpx_stack.sections):
                self.jpx_stack.sections.append(ns.placeholder_section(len(self.jpx_stack.sections), frame=key[1]))
        self.jpx_by_frame = {s.frame: s for s in self.jpx_stack.sections}

    def undo_any(self, event=None):
        """Ctrl+Z: undoes the last change of the order in step 3, of the outlines in step 2."""
        if event is not None and isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return
        if self.stage == 2:
            self.editor.undo(event)
        else:
            self.undo_last()

    def order_double(self, key, i):
        """Double-click on a thumbnail: back to the outlines tab, at that section's image with its outline selected."""
        if self.busy or self.project.is_jpx or key[0] < 0:         # (a JPX has no outlines tab, and a placeholder no outline)
            return
        si, pi = key
        self.project.order = list(self.order)         # (the order made so far is kept for when the tab is entered again)
        self.cur = si
        self.show_stage(1)
        if self.slide().path in self.ov and pi < len(self.polys.get(si, [])):
            self.selected = {pi}
            self.update_edit_help()
            self.redraw()

    def order_reset_help(self):
        """What Reset order does, for its tool tip (it depends on the kind of project and, for scans, on how step 2 numbers the outlines)."""
        if self.project.is_jpx:
            return ("Puts the sections back in the order of the frames in the JPX file, as it was when the file was opened. "
                    "Any order made by dragging is lost, and so are the placeholders inserted here; deleted sections come back.")
        how = {"rows": "across the image, then down (by rows)", "columns": "down the image, then across (by columns)",
               "scene": "by scene number, the order of the scenes in the .czi"}.get(self.project.numbering, self.project.numbering)
        return ("Puts the sections back in the order they were numbered in step 2: the images in the order of the list in step 1, and "
                f"within each image the outlines in the order chosen under \"Number the outlines\" in step 2, now: {how}. "
                "Any order made by dragging is lost, and so are the placeholders inserted here; deleted sections come back.")

    def order_reset_keys(self):
        """The order Reset puts back."""
        if self.project.is_jpx:
            if self.jpx_stack is not None:        # (placeholders inserted here are not in the JPX: they go)
                self.jpx_stack.sections[:] = [s for s in self.jpx_stack.sections if not ns.is_new_placeholder(s)]
                for k, sec in enumerate(self.jpx_stack.sections):
                    sec.index = k
                self.jpx_by_frame = {sec.frame: sec for sec in self.jpx_stack.sections}
            return [(0, f) for f in sorted(self.jpx_by_frame)]                 # (the JPX's own order)
        return self.project.default_order()

    # ------------------------------------------------------------------ done

    def finish_jpx(self, spacing):
        """Done, for a JPX project: the order and the spacing are saved, and the stack (as opened, in that order) goes to the alignment tool."""
        proj, stack = self.project, self.jpx_stack
        if stack is None:
            return
        proj.z_spacing_um = spacing
        proj.jpx_order = [f for _, f in self.order]
        proj.jpx_sections = len(proj.jpx_order)
        try:
            if not self.write_project(proj):
                return
        except OSError as e:
            messagebox.showwarning("Project not saved", f"Couldn't save the project file: {e}", parent=self.root)
        self.var_status.set("Preparing the sections...")

        def work():
            sp.apply_jpx_order(stack, proj.jpx_order)
            stack.z_spacing_um = spacing
            return stack

        def done(res):
            self.var_status.set("")
            if isinstance(res, BaseException):
                messagebox.showerror("Can't go on", f"{type(res).__name__}: {res}", parent=self.root)
                return
            self.result = res
            self.close()
            if self.on_done:
                self.on_done(res)

        self.background(work, done)

    def resume(self, path, stage=2):
        """Open a saved project and go straight to one of the tabs (the alignment tool's "Back to step 1" comes here, to the order tab)."""
        self.reusing = True
        self.open_project(path)
        if self.project.slides:
            self.show_stage(stage)

    def finish(self):
        proj = self.project
        try:
            spacing = float(self.var_spacing.get())
        except ValueError:
            spacing = 0.0
        if spacing <= 0:
            messagebox.showinfo("Distance between sections", "Enter the distance between adjacent sections, in \u00b5m (the same number "
                                "NeuroInfo asks for when it makes a reconstruction).", parent=self.root)
            return
        if proj.is_jpx:
            self.finish_jpx(spacing)
            return
        proj.z_spacing_um = spacing
        proj.order = list(self.order)
        if proj.alignment and proj.alignment_key != proj.fingerprint():
            # The outlines or their order have changed since the alignment was saved. The alignment is of the images, so it carries over to redrawn outlines and to another order. What can't: sections whose
            # outline was deleted or merged into another (and so has nothing to go with).
            assign, dropped, fresh = sp.match_alignment(proj)
            lost = [i for i in dropped if not ns.Edit(**proj.alignment["sections"][i]["edit"]).is_identity()]
            if lost and not messagebox.askyesno(
                    "Alignment of removed outlines",
                    f"{len(lost)} section(s) with alignment edits (moves, rotations or flips) have no outline now (deleted, or merged "
                    "into another), so their alignment will be dropped:\n\n"
                    + "\n".join(proj.alignment["sections"][i].get("name", "?") for i in lost[:8]) + (
                        "\n..." if len(lost) > 8 else "") + f"\n\nThe other {len(assign)} keep theirs. Go on?", parent=self.root):
                return
        proj.order = list(self.order)
        self.order_key = proj.polygons_key()
        try:
            if not self.write_project(proj):
                return
        except OSError as e:
            messagebox.showwarning("Project not saved", f"Couldn't save the project file: {e}", parent=self.root)
        self.var_status.set("Preparing the sections...")

        def done(res):
            self.var_status.set("")
            if isinstance(res, BaseException):
                messagebox.showerror("Can't go on", f"{type(res).__name__}: {res}", parent=self.root)
                return
            self.result = res
            self.close()
            if self.on_done:
                self.on_done(res)

        self.background(lambda: sp.build_stack(proj), done)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    loader = SectionLoader(slides=[os.path.normpath(a) for a in argv if a.lower().endswith((".czi",) + sp.JP2_SUFFIXES)])
    return loader.run()


if __name__ == "__main__":
    main()
