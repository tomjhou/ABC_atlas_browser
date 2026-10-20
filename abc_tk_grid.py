"""A Tk version of the atlas browser's section grid picker: the picture of every section's thumbnail, on which whole sections are picked by
clicking, and a double-click opens that section's own ROI window (abc_tk_picker.py).

    python abc_tk_grid.py            # on its own, for trying it

The browser calls prompt_section_selection_tk() instead of its matplotlib picker when the launcher's Settings ask for the Tk windows.
The picture and layout of the grid are the ones the matplotlib picker uses (abc_core.generate_section_grid_image).
"""
import json
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image

import abc_core as core
from abc_tk_picker import ZoomView, prompt_subregion_selection_tk

SECTION_Z_LABEL_FONTSIZE = 8          # points: the z label at the top-right of each thumbnail
SELECT_COLOR, ROI_COLOR = "#ff0000", "#ff8c00"
DOUBLE_CLICK_MS = 450                 # a double-click's second press this soon after a toggle undoes that toggle


class ProgressWindow:
    """A small window with a progress bar and a Cancel button, for the waits before a section opens."""

    def __init__(self, parent, title, text):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.resizable(False, False)
        self.cancelled = False
        f = ttk.Frame(self.top, padding=16)
        f.pack()
        self.var = tk.StringVar(f, text)
        ttk.Label(f, textvariable=self.var, wraplength=360, justify=tk.LEFT).pack(anchor="w")
        self.bar = ttk.Progressbar(f, length=360, mode="indeterminate", maximum=1.0)
        self.bar.pack(pady=(10, 10))
        self.bar.start(15)
        ttk.Button(f, text="Cancel", command=self.cancel).pack(anchor="e")
        self.top.protocol("WM_DELETE_WINDOW", self.cancel)
        self.top.update_idletasks()
        self.top.geometry(f"+{parent.winfo_rootx() + 200}+{parent.winfo_rooty() + 200}")

    def cancel(self):
        self.cancelled = True

    def set(self, fraction=None, text=None):
        if fraction is not None:
            self.bar.stop()
            self.bar.config(mode="determinate", value=fraction)
        if text:
            self.var.set(text)

    def close(self):
        try:
            self.top.destroy()
        except tk.TclError:
            pass


class GridWindow:
    def __init__(self, adata, abc_cache, section_series, img, layout, out_folder=None, cell_type_selection="All", imputed_state=None,
                 ensure_imputed=None, root=None):
        self.adata, self.abc_cache, self.section_series = adata, abc_cache, section_series
        self.layout = layout
        self.sections = list(layout.keys())
        self.out_folder, self.cell_type_selection, self.ensure_imputed = out_folder, cell_type_selection, ensure_imputed
        self.imputed_state = imputed_state if imputed_state is not None else {"adata": None, "load_thread": None, "load_error": None}
        self.spatial_cache = {"df": None, "load_thread": None, "load_progress": {"rows": 0, "total": None}, "load_error": None}
        # shared with every single-section window of this session (they write back what they were left on)
        self.session_view_settings = {"mode": "standard", "gene": "Sox14"}
        self.session_color_scale = {"mode": "log"}
        self.selected = set()
        self.rois = []                 # {'section', 'x_min', 'x_max', 'y_min', 'y_max'}
        self.result = None             # ("confirmed", sections, rois) | ("cancelled",) | None while open
        self.last_toggle = None
        self.busy = False
        self.own_root = root is None
        self.root = tk.Tk() if root is None else tk.Toplevel(root)
        self.root.title("Choose sections")
        self.root.geometry("1500x950")

        self.img_h, self.img_w = img.shape[:2]
        boxes = np.array([layout[s]["pixel_bbox"] for s in self.sections], float)     # x0, x1, y0, y1 (y counting up from the bottom)
        self.boxes = boxes

        pane = ttk.Frame(self.root)
        pane.pack(fill=tk.BOTH, expand=True)
        side = ttk.Frame(pane, padding=10, width=330)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)
        self.view = ZoomView(pane, bg="white")
        self.view.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.build_side(side)

        c = self.view.canvas
        self.z_items = {}
        self.z_text = {}               # section -> its z label
        self.sel_items = {}
        self.roi_items = {}            # section -> [border item, roi rect items...]
        for sec in self.sections:
            self.z_items[sec] = c.create_text(0, 0, text="", anchor="se", fill="#444444", font=("Segoe UI", SECTION_Z_LABEL_FONTSIZE), tags="overlay")
        self.view.listeners.append(self.place_overlays)
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<Double-Button-1>", self.on_double)
        c.bind("<Motion>", self.on_motion)
        c.bind("<Leave>", lambda e: self.var_hover.set("Point at a section."))
        self.view.set_image(img)
        self.z_ready = False
        core.start_section_coordinate_load(abc_cache)
        self.root.protocol("WM_DELETE_WINDOW", self.exit)
        self.root.bind("<Escape>", lambda e: None)
        self.poll()

    # ---- the side panel

    def build_side(self, f):
        ttk.Label(f, text="Choose sections", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(f, text="click: pick or unpick a whole section\ndouble-click: open its ROI window\nwheel: zoom   right-drag: pan   Home: all",
                  foreground="gray30", justify=tk.LEFT).pack(anchor="w", pady=(4, 10))
        self.var_summary = tk.StringVar(f, "")
        box = ttk.LabelFrame(f, text="Chosen", padding=8)
        box.pack(fill=tk.X)
        ttk.Label(box, textvariable=self.var_summary, wraplength=285, justify=tk.LEFT).pack(anchor="w")
        hov = ttk.LabelFrame(f, text="Under the pointer", padding=8)
        hov.pack(fill=tk.X, pady=(10, 0))
        self.var_hover = tk.StringVar(f, "Point at a section.")
        ttk.Label(hov, textvariable=self.var_hover, wraplength=285, justify=tk.LEFT).pack(anchor="w")
        self.var_perf = tk.StringVar(f, "")
        ttk.Label(f, textvariable=self.var_perf, foreground="gray50", font=("Segoe UI", 8)).pack(anchor="w", side=tk.BOTTOM)
        self.var_status = tk.StringVar(f, "")
        ttk.Label(f, textvariable=self.var_status, foreground="gray30", wraplength=310, justify=tk.LEFT).pack(anchor="w", side=tk.BOTTOM, pady=(8, 0))
        for text, command in (("Exit", self.exit), ("Clear ROIs", self.clear_rois), ("Load ROIs", self.load_rois), ("Confirm Selection", self.confirm)):
            ttk.Button(f, text=text, command=command).pack(side=tk.BOTTOM, fill=tk.X, pady=3)
        self.update_summary()

    def update_summary(self):
        with_rois = {r["section"] for r in self.rois}
        lines = [f"{len(self.selected - with_rois)} whole section(s) picked", f"{len(self.rois)} ROI(s) on {len(with_rois)} section(s)"]
        if not self.selected and not self.rois:
            lines.append("Nothing chosen: confirming runs the whole brain.")
        self.var_summary.set("\n".join(lines))

    # ---- which section is where

    def section_at(self, cx, cy):
        ix, iy = self.view.to_picture(cx, cy)
        y_up = self.img_h - iy
        b = self.boxes
        hit = np.flatnonzero((b[:, 0] <= ix) & (ix <= b[:, 1]) & (b[:, 2] <= y_up) & (y_up <= b[:, 3]))
        return self.sections[int(hit[0])] if len(hit) else None

    def box_on_canvas(self, section):
        x0, x1, y0, y1 = self.layout[section]["pixel_bbox"]
        (a, b), (c, d) = self.view.to_canvas(x0, self.img_h - y1), self.view.to_canvas(x1, self.img_h - y0)
        return a, b, c, d

    # ---- the marks on the thumbnails

    def place_overlays(self):
        c = self.view.canvas
        cw, ch = self.view.canvas_size()
        for sec in self.sections:
            a, b, x, y = self.box_on_canvas(sec)
            visible = x > 0 and a < cw and y > 0 and b < ch
            item = self.z_items[sec]
            if visible:
                c.coords(item, x, b - 1)
                # a thumbnail too small for the whole label (it would run over the section number above it) gets just the value
                full = self.z_text.get(sec, "")
                c.itemconfig(item, state="normal", text=full if (x - a) >= 140 else full.rsplit(" ", 1)[-1])
            else:
                c.itemconfig(item, state="hidden")
            if sec in self.sel_items:
                c.coords(self.sel_items[sec], a, b, x, y)
            for k, rect in enumerate(self.roi_items.get(sec, [])):
                if k == 0:
                    c.coords(rect, a, b, x, y)
        for sec, items in self.roi_items.items():
            for roi, item in zip([r for r in self.rois if r["section"] == sec], items[1:]):
                p0 = core.map_data_to_pixel(self.layout[sec], roi["x_min"], roi["y_min"])
                p1 = core.map_data_to_pixel(self.layout[sec], roi["x_max"], roi["y_max"])
                (ax, ay), (bx, by) = self.view.to_canvas(p0[0], self.img_h - p0[1]), self.view.to_canvas(p1[0], self.img_h - p1[1])
                c.coords(item, ax, ay, bx, by)
        c.tag_raise("overlay")

    def refresh_marks(self, section):
        """The red outline of a picked section (only while it has no ROI: an ROI takes priority) and the orange marks of its ROIs."""
        c = self.view.canvas
        has_rois = any(r["section"] == section for r in self.rois)
        if section in self.selected and not has_rois:
            if section not in self.sel_items:
                self.sel_items[section] = c.create_rectangle(0, 0, 0, 0, outline=SELECT_COLOR, width=3, tags="overlay")
        elif section in self.sel_items:
            c.delete(self.sel_items.pop(section))
        for item in self.roi_items.pop(section, []):
            c.delete(item)
        mine = [r for r in self.rois if r["section"] == section]
        if mine:
            items = [c.create_rectangle(0, 0, 0, 0, outline=ROI_COLOR, width=3, dash=(8, 4), tags="overlay")]
            items += [c.create_rectangle(0, 0, 0, 0, outline=ROI_COLOR, width=2, tags="overlay") for _ in mine]
            self.roi_items[section] = items
        self.place_overlays()
        self.update_summary()

    # ---- clicks

    def on_press(self, e):
        self.view.canvas.focus_set()
        sec = self.section_at(e.x, e.y)
        if sec is None:
            return
        self.toggle(sec)
        self.last_toggle = (sec, time.perf_counter())

    def toggle(self, sec):
        if sec in self.selected:
            self.selected.discard(sec)
        else:
            self.selected.add(sec)
        self.refresh_marks(sec)

    def on_double(self, e):
        sec = self.section_at(e.x, e.y)
        if sec is None:
            return
        last, self.last_toggle = self.last_toggle, None
        if last is not None and last[0] == sec and time.perf_counter() - last[1] <= DOUBLE_CLICK_MS / 1000:
            self.toggle(sec)             # (the first press of the double-click toggled it: undone)
        self.open_section(sec)

    def on_motion(self, e):
        sec = self.section_at(e.x, e.y)
        if sec is None:
            self.var_hover.set("Point at a section.")
            return
        z = core.section_z_text(sec, self.abc_cache)
        n_rois = sum(1 for r in self.rois if r["section"] == sec)
        parts = [f"Section {core.sanitize_section_token(sec)}" + (f"   {z}" if z else "")]
        if sec in self.selected:
            parts.append("picked (whole section)")
        if n_rois:
            parts.append(f"{n_rois} ROI(s)")
        self.var_hover.set("\n".join(parts))

    # ---- the single-section window

    def wait_for(self, thread_alive, title, text, progress=None):
        """Wait (the window keeps running) for something in the background, with a Cancel. True if it finished."""
        win = ProgressWindow(self.root, title, text)
        try:
            while thread_alive():
                self.root.update()
                if progress is not None:
                    frac = progress()
                    if frac is not None:
                        win.set(frac)
                if win.cancelled:
                    return False
                time.sleep(0.03)
        finally:
            win.close()
        return True

    def open_section(self, section):
        if self.busy:
            return
        self.busy = True
        try:
            cache = self.spatial_cache
            if cache["df"] is None:
                core.ensure_spatial_load_started(cache, self.adata, self.abc_cache)
                thread = cache["load_thread"]
                if thread is not None:
                    prog = cache["load_progress"]
                    ok = self.wait_for(thread.is_alive, "Loading", "Loading the cell metadata (the first time takes a while)...",
                                       progress=lambda: prog["rows"] / prog["total"] if prog.get("total") else None)
                    if not ok:
                        self.var_status.set("Cancelled: the load carries on in the background.")
                        return
                if cache["df"] is None:
                    self.var_status.set(f"Could not load the spatial data: {cache['load_error']}")
                    return
            if core.load_valid_section_roi_cache(section) is None:
                state = {"fraction": 0.0, "error": None}

                def work():
                    try:
                        core.generate_and_cache_section_image(self.adata, self.abc_cache, self.section_series, section, cache["df"], (15, 10),
                                                              progress_callback=lambda f: state.__setitem__("fraction", f))
                    except Exception as e:
                        state["error"] = e

                t = threading.Thread(target=work, daemon=True)
                t.start()
                if not self.wait_for(t.is_alive, "Rendering", f"Drawing section {core.sanitize_section_token(section)} (the first time only)...",
                                     progress=lambda: state["fraction"]):
                    self.var_status.set("Cancelled.")
                    return
                if state["error"] is not None:
                    self.var_status.set(f"Could not render section {section}: {state['error']}")
                    return
            existing = [(i, (r["x_min"], r["x_max"], r["y_min"], r["y_max"])) for i, r in enumerate(self.rois) if r["section"] == section]
            self.root.config(cursor="watch")
            try:
                actions = prompt_subregion_selection_tk(
                    self.adata, self.abc_cache, self.section_series, section, existing_rois=existing, spatial=cache["df"],
                    initial_view={"mode": self.session_view_settings["mode"], "gene": self.session_view_settings["gene"] or None},
                    session_view_settings=self.session_view_settings, cell_type_selection=self.cell_type_selection,
                    session_color_scale=self.session_color_scale, imputed_state=self.imputed_state, ensure_imputed=self.ensure_imputed,
                    parent=self.root)
            finally:
                self.root.config(cursor="")
            self.apply_actions(section, actions)
        finally:
            self.busy = False

    def apply_actions(self, section, actions):
        """What the section window returned: edits and additions first (while the deletions' indices still mean the positions in the list),
        then the deletions, highest first."""
        if not actions:
            return
        deletes = []
        for a in actions:
            if a["action"] == "delete":
                deletes.append(a["index"])
                continue
            x_min, x_max, y_min, y_max = a["bounds"]
            if a["action"] == "edit":
                self.rois[a["index"]].update(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
            else:
                self.rois.append({"section": section, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        for i in sorted(deletes, reverse=True):
            self.rois.pop(i)
        self.refresh_marks(section)

    # ---- the buttons

    def load_rois(self):
        path = filedialog.askopenfilename(parent=self.root, title="Select a previously saved ROI CSV", initialdir=self.out_folder or None,
                                          filetypes=[("ROI CSV files", "roi_coords.csv"), ("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            loaded, whole = core.load_rois_csv(path)
        except Exception as e:
            self.var_status.set(f"Could not load ROIs from {os.path.basename(path)}: {e}")
            return
        known = {(r["section"], r["x_min"], r["x_max"], r["y_min"], r["y_max"]) for r in self.rois}
        touched, added, skipped_unknown, skipped_dup, whole_added = set(), 0, 0, 0, 0
        for roi in loaded:
            if roi["section"] not in self.layout:
                skipped_unknown += 1
                continue
            key = (roi["section"], roi["x_min"], roi["x_max"], roi["y_min"], roi["y_max"])
            if key in known:
                skipped_dup += 1
                continue
            self.rois.append(roi)
            known.add(key)
            touched.add(roi["section"])
            added += 1
        for sec in whole:
            if sec not in self.layout:
                skipped_unknown += 1
            elif sec not in self.selected:
                self.selected.add(sec)
                touched.add(sec)
                whole_added += 1
        for sec in touched:
            self.refresh_marks(sec)
        msg = f"Loaded {added} ROI(s) and {whole_added} whole section(s) across {len(touched)} section(s)."
        if skipped_unknown:
            msg += f" Skipped {skipped_unknown} not in this grid."
        if skipped_dup:
            msg += f" Skipped {skipped_dup} already present."
        self.var_status.set(msg)

    def clear_rois(self):
        if not self.rois and not self.selected:
            self.var_status.set("No ROIs or whole-section picks to clear.")
            return
        if not messagebox.askyesno("Clear all ROIs?", f"Clear {len(self.rois)} ROI(s) and {len(self.selected)} whole-section pick(s)? "
                                   "This cannot be undone.", parent=self.root):
            return
        touched = set(self.selected) | {r["section"] for r in self.rois}
        self.rois.clear()
        self.selected.clear()
        for sec in touched:
            self.refresh_marks(sec)
        self.var_status.set("Cleared.")

    def confirm(self):
        self.result = ("confirmed", [s for s in self.sections if s in self.selected], list(self.rois))
        self.close()

    def exit(self):
        self.result = ("cancelled",)
        self.close()

    # ---- running

    def poll(self):
        if not self.z_ready and core.section_coordinates_ready():
            self.z_text = {sec: core.section_z_text(sec, short=True) for sec in self.sections}
            self.z_ready = True
            self.place_overlays()
        self.var_perf.set(f"view {self.view.scale / max(self.view.fit_scale(), 1e-9):.1f}x   redraw {self.view.last_draw_ms:.0f} ms")
        self.poll_id = self.root.after(300, self.poll)

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
        if self.own_root:
            self.root.mainloop()
        else:
            self.root.wait_window(self.root)
        return self.result


# ---------------------------------------------------------------------------------------------------------------------- the browser's call

def load_grid(adata, abc_cache, section_series):
    """(RGB uint8 picture, layout) of the section grid: the cached ones, made first if there are none."""
    layout = None
    if core.GRID_CACHE_PNG.exists() and core.GRID_CACHE_LAYOUT.exists():
        with open(core.GRID_CACHE_LAYOUT) as f:
            layout = json.load(f)
        if layout and not isinstance(next(iter(layout.values())), dict):
            layout = None                                  # (an older format: made again)
    if layout is None:
        if section_series is None:
            raise RuntimeError("no section metadata available")
        print("No cached section grid image found; generating one (this happens once)...", flush=True)
        _img, layout = core.generate_section_grid_image(adata, abc_cache, section_series)
    img = np.asarray(Image.open(core.GRID_CACHE_PNG).convert("RGB"))
    return img, layout


def prompt_section_selection_tk(adata, abc_cache, section_series, out_folder=None, cell_type_selection="All", imputed_state=None,
                                ensure_imputed=None):
    """The section grid picker. Returns ("confirmed", picked sections, rois, how many sections there are) or ("cancelled",) or
    ("closed",) (the window was closed some other way). The caller turns that into what the pipeline wants, as the matplotlib picker does."""
    img, layout = load_grid(adata, abc_cache, section_series)
    win = GridWindow(adata, abc_cache, section_series, img, layout, out_folder=out_folder, cell_type_selection=cell_type_selection,
                     imputed_state=imputed_state, ensure_imputed=ensure_imputed)
    try:
        win.root.state("zoomed")
    except tk.TclError:
        pass
    win.root.lift()
    win.root.focus_force()
    result = win.run()
    if result is None:
        return ("closed",)
    if result[0] == "confirmed":
        return ("confirmed", result[1], result[2], len(layout))
    return result


# ---------------------------------------------------------------------------------------------------------------------- running it

def main():
    from pathlib import Path
    import anndata
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    core.load_coordinate_settings()
    base = Path(os.environ.get("ABC_ATLAS_CACHE", str(core.SCRIPT_DIR.parent / "abc_atlas_cache")))
    cache = AbcProjectCache.from_local_cache(base)
    adata = anndata.read_h5ad(cache.get_data_path(directory="MERFISH-C57BL6J-638850", file_name="C57BL6J-638850/raw"), backed="r")
    sections = core.get_section_labels(adata, cache)
    result = prompt_section_selection_tk(adata, cache, sections)
    print(result[0], "|", len(result[1]) if len(result) > 1 else "", "section(s) |", len(result[2]) if len(result) > 2 else "", "ROI(s)")


if __name__ == "__main__":
    main()
