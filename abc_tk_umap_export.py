"""The exports and 3D views of the Tk UMAP viewer (abc_tk_umap.py): Save UMAP, Save section maps, Export DEGs, Export expression, Export
coordinates, 3D view and 3D in browser. They are the same outputs as the matplotlib viewer's (the same file contents, and the same file
names as the ones proposed); the statistics, dialogs and 3D viewers are the browser's own functions, which it hands to the window as `host`.

`ExportsMixin` is mixed into UmapWindow: it reads the window's state (what is colored by what, the cells, the view) and writes files. Each
export first shows a Save dialog with the file name it would use (in the run's folder, or the folder of the last export), so that the name
can be changed; cancelling the dialog cancels the export.
"""
import math
import re
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd

import abc_core as core

EXPORT_FONTSIZE = 14
EXPORT_AXES_PX = 6400                         # the UMAP's plotting square in a saved file, at the save DPI
EXPORT_MARGIN_IN = 2.5
EXPORT_LEGEND_MAX_ROWS = 35
UMAP_POINT_SIZE = 10                          # matplotlib's s (an area) of a dot in a saved UMAP
POINT_ALPHA = 0.75


def hexcolor(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def sibling(path, tail, ext):
    """`path`, a file name that ends in `ext`, with `tail` put before the ending. The name may have dots inside it (a UMAP's md0.5_pc100), so
    Path.with_suffix, which takes the text after the last dot for an ending, must not be used on it."""
    return path.with_name(path.name[:-len(ext)] + tail + ext)


def safe_token(text, default="view"):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or default


class ExportsMixin:
    # ---- shared helpers

    def target_dir(self):
        d = Path(self.run_folder) if self.run_folder is not None else Path(self.host["CACHE_DIR"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ask_save_path(self, title, proposed, extension, kind):
        """The file the user picks in a Save dialog, or None if the dialog was cancelled. The dialog offers `proposed` (a file name) in the run's
        folder, or in the folder of the last export of this window; a name typed without the extension gets it. The dialog itself asks before
        replacing a file."""
        folder = getattr(self, "export_dir", None)
        if folder is None or not Path(folder).is_dir():
            folder = self.target_dir()
        chosen = filedialog.asksaveasfilename(
            parent=self.root, title=title, initialdir=str(folder), initialfile=proposed, defaultextension=extension,
            filetypes=[(kind, "*" + extension), ("All files", "*.*")])
        if not chosen:
            return None
        path = Path(chosen)
        if path.suffix.lower() != extension:       # (a name typed with some other ending keeps it, and gets this one after it)
            path = path.with_name(path.name + extension)
        self.export_dir = path.parent
        return path

    def ask_image_format(self, title, svg_note):
        """"png" or "svg", chosen in a small dialog (None if it was cancelled or closed). The last choice made in this window is offered first.
        `svg_note` is added under the SVG choice."""
        import tkinter as tk
        from tkinter import ttk
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.transient(self.root)
        dlg.resizable(False, False)
        var = tk.StringVar(dlg, getattr(self, "export_format", "png"))
        box = ttk.Frame(dlg, padding=16)
        box.pack()
        ttk.Label(box, text="Save as:").pack(anchor="w")
        ttk.Radiobutton(box, text="PNG: a picture", value="png", variable=var).pack(anchor="w", pady=(6, 0))
        ttk.Radiobutton(box, text="SVG: vector graphics, for editing in a drawing program", value="svg", variable=var).pack(anchor="w", pady=(4, 0))
        if svg_note:
            ttk.Label(box, text=svg_note, foreground="gray30", wraplength=320, justify=tk.LEFT).pack(anchor="w", padx=(22, 0))
        result = []

        def finish(value):
            result.append(value)
            dlg.destroy()
        row = ttk.Frame(box)
        row.pack(anchor="e", pady=(14, 0))
        ttk.Button(row, text="OK", command=lambda: finish(var.get())).pack(side=tk.LEFT)
        ttk.Button(row, text="Cancel", command=lambda: finish(None)).pack(side=tk.LEFT, padx=(6, 0))
        dlg.bind("<Return>", lambda e: finish(var.get()))
        dlg.bind("<Escape>", lambda e: finish(None))
        dlg.protocol("WM_DELETE_WINDOW", lambda: finish(None))
        dlg.update_idletasks()
        try:                                        # (in the middle of the window, not in a corner of the screen)
            x = self.root.winfo_rootx() + (self.root.winfo_width() - dlg.winfo_reqwidth()) // 2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - dlg.winfo_reqheight()) // 3
            dlg.geometry(f"+{max(x, 0)}+{max(y, 0)}")
            dlg.grab_set()
        except tk.TclError:
            pass
        self.root.wait_window(dlg)
        choice = result[0] if result else None
        if choice:
            self.export_format = choice
        return choice

    def confirm_replacing(self, paths):
        """True unless some of `paths` (files the Save dialog did not ask about, saved along with the one it did) exist and the user declines
        to replace them."""
        existing = [p for p in paths if Path(p).exists()]
        if not existing:
            return True
        return messagebox.askyesno("Replace files?", "These files exist and would be replaced:\n\n" + "\n".join(str(p) for p in existing)
                                   + "\n\nReplace them?", icon="warning", parent=self.root)

    def view_token_safe(self):
        token = safe_token(getattr(self, "view_token", "view"))
        params_token = self.host["umap_params_token"](self.params) if self.params is not None else ""
        return f"{token}_{params_token}" if params_token else token

    def level_ids(self, level):
        return core.extract_leading_numeric_id(pd.Series(self.info[level])).to_numpy()

    def group_choice(self, action):
        """(level, targets) of the groups highlighted in Single group(s) that have cells in this run, or None (saying why)."""
        sel = getattr(self, "last_ids", None)
        if self.mode != "single" or not sel or not sel["targets"]:
            self.var_status.set(f"Enter at least one group number that exists in this run before {action}.")
            return None
        targets = [t for t in sel["targets"] if sel["counts"].get(t, 0) > 0]
        if not targets:
            self.var_status.set(f"Enter at least one group number that exists in this run before {action}.")
            return None
        return sel["level"], targets

    def background(self, text, work, done):
        """Run `work` (no Tk in it) off the main thread; `done(result)` runs back on it."""
        return self.job(text, work, done)

    # ---- Save UMAP

    def view_limits(self):
        c = self.umap
        w, h = c.size()
        return (c.cx - w / (2 * c.scale), c.cx + w / (2 * c.scale)), (c.cy - h / (2 * c.scale), c.cy + h / (2 * c.scale))

    def build_umap_figure(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.lines import Line2D
        h = self.host
        dpi = h["UMAP_SAVE_DPI"]
        axes_in = EXPORT_AXES_PX / dpi
        total = axes_in + 2 * EXPORT_MARGIN_IN
        fig = Figure(figsize=(total, total))
        FigureCanvasAgg(fig)
        frac, margin = axes_in / total, EXPORT_MARGIN_IN / total
        ax = fig.add_axes([margin, margin, frac, frac])
        ax.set_aspect("equal", adjustable="box")
        xlim, ylim = self.view_limits()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=EXPORT_FONTSIZE)
        ax.set_ylabel("UMAP2", fontsize=EXPORT_FONTSIZE)
        ax.set_title(getattr(self, "view_title", ""), fontsize=EXPORT_FONTSIZE)
        size = UMAP_POINT_SIZE * self.dot_scale() ** 2
        coords = self.coords
        kind = getattr(self, "view_kind", "all")
        entries = None
        if kind == "gene":
            gene, dataset, scale, vals = self.shown_values
            from matplotlib import colormaps
            fin = np.isfinite(vals)
            vmin, vmax = (float(np.nanmin(vals)), float(np.nanmax(vals))) if fin.any() else (0.0, 1.0)
            order = np.argsort(np.where(fin, vals, -np.inf), kind="stable")
            sc = ax.scatter(coords[order, 0], coords[order, 1], c=vals[order], cmap=colormaps["viridis"], vmin=vmin, vmax=vmax, s=size,
                            alpha=POINT_ALPHA, linewidths=0)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label(gene, fontsize=EXPORT_FONTSIZE)
            cb.ax.tick_params(labelsize=EXPORT_FONTSIZE)
        elif kind == "multi":
            ax.set_facecolor(tuple(v / 255 for v in (26, 26, 26)))
            for lay in self.umap.layers:
                ax.scatter(lay["x"], lay["y"], c=lay["rgb"] / 255.0, s=size * (0.25 if lay.get("shape") != "half" and lay["size"] < 4 * self.dot_scale() else 1.0),
                           linewidths=0, alpha=None)
            handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=f"{g} ({name})")
                       for i, (g, color, name) in enumerate(zip(self.multi_genes, ("red", "green", "#4040ff", "red", "green", "#4040ff"), ("red", "green", "blue", "red", "green", "blue")))
                       if i not in self.hidden]
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), bbox_transform=ax.transAxes, fontsize=EXPORT_FONTSIZE, frameon=True)
        else:
            cat = getattr(self, "export_cat", None)
            if cat is None:
                ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=size, alpha=POINT_ALPHA, linewidths=0)
            else:
                rest = cat["rank"] < 0
                ax.scatter(coords[rest, 0], coords[rest, 1], c="lightgray", s=size, alpha=POINT_ALPHA, linewidths=0)
                for r, (color, label, count) in enumerate(cat["entries"]):
                    m = cat["rank"] == r
                    if not m.any():
                        continue
                    marker, is_open = h["category_rank_shape"](r)
                    if is_open:
                        ax.scatter(coords[m, 0], coords[m, 1], facecolors="none", edgecolors=color, marker=marker, s=size, alpha=POINT_ALPHA, linewidths=1.0)
                    else:
                        ax.scatter(coords[m, 0], coords[m, 1], c=color, marker=marker, s=size, alpha=POINT_ALPHA, linewidths=0)
                entries = cat
            by_count = self.var_label_scale.get()                 # (the "Label font size scales with cell count" checkbox: the saved UMAP follows it)
            for text, x, y, color, weight in getattr(self, "label_info", []):
                if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]:
                    ax.text(x, y, text, fontsize=EXPORT_FONTSIZE * 0.6 * (weight if by_count else 1.0), color=color, fontweight="bold", ha="center", va="center", clip_on=True)
            if entries is not None:
                handles = self.legend_handles(entries, sort="id")
                ncol = max(1, math.ceil(len(handles) / EXPORT_LEGEND_MAX_ROWS))
                fig.legend(handles=handles, title=entries["title"], loc="center left", bbox_to_anchor=(1.02, 0.5), bbox_transform=ax.transAxes,
                           fontsize=EXPORT_FONTSIZE * 0.85, title_fontsize=EXPORT_FONTSIZE, ncol=ncol, frameon=True, handlelength=1.0,
                           handleheight=1.0, labelspacing=0.4, columnspacing=1.5)
        return fig

    def legend_handles(self, cat, sort="id"):
        from matplotlib.lines import Line2D
        rows = []
        for r, (color, label, count) in enumerate(cat["entries"]):
            marker, is_open = self.host["category_rank_shape"](r)
            rows.append((color, label, count, marker, is_open))
        if sort == "count":
            rows.sort(key=lambda e: e[2], reverse=True)
        else:
            def key(e):
                m = re.match(r"^(\d+)", e[1])
                return int(m.group(1)) if m else math.inf
            rows.sort(key=key)
        return [Line2D([], [], marker=m, linestyle="none", markersize=9, markerfacecolor="none" if o else c, markeredgecolor=c,
                       markeredgewidth=1.2 if o else 0.5, label=f"{lab}  ({n} cells)") for c, lab, n, m, o in rows]

    def save_umap(self):
        if self.busy:
            return
        target, token, dpi = self.target_dir(), self.view_token_safe(), self.host["UMAP_SAVE_DPI"]
        fmt = self.ask_image_format("Save the UMAP", "")
        if fmt is None:
            self.var_status.set("Save UMAP cancelled.")
            return
        ext = "." + fmt
        proposed = self.host["unique_export_paths"](target, f"umap_{token}", (ext,))[ext].name
        cat = getattr(self, "export_cat", None)
        has_legend = getattr(self, "view_kind", "all") in ("all", "single") and cat is not None
        chosen = self.ask_save_path("Save the UMAP" + (f" (its legend is saved with it, as _legend{ext})" if has_legend else ""), proposed, ext,
                                    fmt.upper() + " image")
        if chosen is None:
            self.var_status.set("Save UMAP cancelled.")
            return
        legend_path = sibling(chosen, "_legend", ext)
        if has_legend and not self.confirm_replacing([legend_path]):
            self.var_status.set("Save UMAP cancelled.")
            return
        self.root.config(cursor="watch")
        self.var_status.set("Saving the UMAP...")
        self.root.update()
        try:
            fig = self.build_umap_figure()
            written = []
            path = chosen
            self.var_status.set(f"Saving {fmt}...")
            self.root.update()
            fig.savefig(path, bbox_inches="tight", dpi=dpi if fmt == "png" else None)
            written.append(path)
            if has_legend:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                handles = self.legend_handles(cat, sort="count")
                lf = Figure()
                FigureCanvasAgg(lf)
                lf.legend(handles=handles, title=cat["title"], loc="center", fontsize=EXPORT_FONTSIZE * 0.85, title_fontsize=EXPORT_FONTSIZE,
                          ncol=max(1, math.ceil(len(handles) / EXPORT_LEGEND_MAX_ROWS)), frameon=True, handlelength=1.0, handleheight=1.0,
                          labelspacing=0.4, columnspacing=1.5)
                lf.savefig(legend_path, bbox_inches="tight", dpi=dpi if fmt == "png" else None)
                written.append(legend_path)
            self.var_status.set(f"Saved {', '.join(p.name for p in written)} to {chosen.parent}.")
        except Exception as e:
            self.var_status.set(f"Could not save the UMAP: {e}")
            print(f"Could not save the UMAP: {e}", flush=True)
        self.root.config(cursor="")

    # ---- Save section maps

    def save_section_maps(self):
        if self.busy or not self.tiles:
            if not self.tiles:
                self.var_status.set("No section panels to save.")
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Rectangle
        h = self.host
        target = self.target_dir()
        fmt = self.ask_image_format("Save the section maps", "The gray cells are pictures inside it, to keep the file small; the colored cells and the "
                                                              "outlines stay vector graphics.")
        if fmt is None:
            self.var_status.set("Save section maps cancelled.")
            return
        ext = "." + fmt
        roi_only = "_ROI_only" if self.var_roi_only.get() else ""        # (the ROI only box above the panels: the name says what the maps show)
        ccf = "_CCF-overlay" if any(t["pc"].outlines for t in self.tiles) else ""        # (and the CCF atlas box)
        proposed = h["unique_export_paths"](target, f"{self.view_token_safe()}_section_maps{roi_only}{ccf}", (ext,))[ext].name
        chosen = self.ask_save_path("Save the section maps", proposed, ext, fmt.upper() + " image")
        if chosen is None:
            self.var_status.set("Save section maps cancelled.")
            return
        self.root.config(cursor="watch")
        self.var_status.set("Saving the section maps...")
        self.root.update()
        try:
            n, cols = len(self.tiles), max(1, self.tile_cols)
            rows = -(-n // cols)
            pw_px, ph_px = self.tiles[0]["pc"].size()
            ppi = 100.0
            pad = 0.08
            fig = Figure(figsize=(cols * (pw_px / ppi + pad), rows * (ph_px / ppi + pad + 0.25)))
            FigureCanvasAgg(fig)
            fw, fh = fig.get_size_inches()
            for k, t in enumerate(self.tiles):
                if t.get("dirty") or not t["pc"].layers:
                    self.build_tile(t)
                pc = t["pc"]
                r, c = divmod(k, cols)
                x0 = c * (pw_px / ppi + pad) / fw
                y0 = 1 - (r + 1) * (ph_px / ppi + pad + 0.25) / fh
                ax = fig.add_axes([x0, y0, (pw_px / ppi) / fw, (ph_px / ppi) / fh])
                ax.set_aspect("equal", adjustable="box")
                w, hh = pc.size()
                xl = (pc.cx - w / (2 * pc.scale), pc.cx + w / (2 * pc.scale))
                yl = (pc.cy - hh / (2 * pc.scale), pc.cy + hh / (2 * pc.scale))
                ax.set_xlim(xl)
                ax.set_ylim(yl[1], yl[0])                            # (a section's y grows downward)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor(tuple(v / 255 for v in pc.bgcolor))
                ax.set_title(t["title"].get(), fontsize=9, loc="left", pad=3)
                pts_per_mm = 72.0 * (pw_px / ppi) / (xl[1] - xl[0])
                from abc_tk_umap import DOT_MM, GREY, LIGHT_GREY, SHAPE_MARKERS
                for lay in pc.layers:
                    m = (lay["x"] >= xl[0]) & (lay["x"] <= xl[1]) & (lay["y"] >= yl[0]) & (lay["y"] <= yl[1])
                    x, y, rgb = lay["x"][m], lay["y"][m], lay["rgb"][m]
                    marks = lay["marks"][m] if lay.get("marks") is not None else None         # (a marker code for each cell, when the Shapes box is ticked)
                    # (in an SVG the gray cells, many thousands of them, are pictures: a vector dot apiece would make the file huge. They are told by
                    # their color, so that the gray ones of a layer that is mostly colored, or the other way round, are found too.)
                    if fmt == "svg" and not lay.get("gray"):
                        gray = (rgb == GREY).all(axis=1) | (rgb == LIGHT_GREY).all(axis=1)
                        parts = ((gray, True), (~gray, False))
                    else:
                        parts = ((np.ones(len(x), bool), fmt == "svg" and bool(lay.get("gray"))),)
                    for sel, raster in parts:
                        if not sel.any():
                            continue
                        for code in (np.unique(marks[sel]) if marks is not None else [0]):
                            sub = sel & (marks == code) if marks is not None else sel
                            marker, is_open = SHAPE_MARKERS[int(code) % 10], int(code) >= 10
                            if is_open:
                                ax.scatter(x[sub], y[sub], facecolors="none", edgecolors=rgb[sub] / 255.0, marker=marker, s=(DOT_MM * pts_per_mm) ** 2,
                                           linewidths=0.5, rasterized=raster)
                            else:
                                ax.scatter(x[sub], y[sub], c=rgb[sub] / 255.0, marker=marker, s=(DOT_MM * pts_per_mm) ** 2, linewidths=0,
                                           rasterized=raster)
                if pc.outlines:                                      # (the CCF atlas box: the regions' outlines, as on the screen)
                    from matplotlib.collections import LineCollection
                    seen = [np.vstack([p, p[:1]]) for p in pc.outlines
                            if p[:, 0].max() >= xl[0] and p[:, 0].min() <= xl[1] and p[:, 1].max() >= yl[0] and p[:, 1].min() <= yl[1]]
                    ink = "#d7d7d7" if sum(pc.bgcolor) < 300 else "#5a5a5a"
                    ax.add_collection(LineCollection(seen, colors=ink, linewidths=0.4))
                for _it, a, b, c2, d in pc.rects:
                    ax.add_patch(Rectangle((a, b), c2 - a, d - b, fill=False, edgecolor="#ff8c00", linestyle=(0, (4, 2)), linewidth=1.2))
                if pc.scalebar:
                    from abc_tk_umap import SCALEBAR_UM
                    view_mm = xl[1] - xl[0]
                    cands = [u for u in SCALEBAR_UM if u <= view_mm * 1000 * 0.22]
                    um = cands[-1] if cands else SCALEBAR_UM[0]
                    ink = "white" if sum(pc.bgcolor) < 300 else "black"
                    bx0 = xl[0] + 0.06 * view_mm
                    by = yl[1] - 0.06 * (yl[1] - yl[0])
                    ax.plot([bx0, bx0 + um / 1000], [by, by], color=ink, linewidth=2.5, solid_capstyle="butt")
                    ax.text(bx0 + um / 2000, by - 0.01 * view_mm, f"{um:g} µm" if um < 1000 else f"{um / 1000:g} mm", color=ink, fontsize=8,
                            ha="center", va="bottom")
            path = chosen
            self.var_status.set(f"Saving {fmt}...")
            self.root.update()
            fig.savefig(path, dpi=h["SECTION_MAPS_SAVE_DPI"])         # (for an SVG the dpi is that of the pictures in it)
            self.var_status.set(f"Saved {path.name} to {chosen.parent}.")
        except Exception as e:
            self.var_status.set(f"Could not save section maps: {e}")
            print(f"Could not save section maps: {e}", flush=True)
        self.root.config(cursor="")

    # ---- Export DEGs / expression

    def imputed_for_export(self):
        """The imputed dataset (loaded now, with the browser's own prompt, if it is not), or None."""
        if self.imputed_state["adata"] is None and self.ensure_imputed:
            self.ensure_imputed(self.imputed_state, self.abc_cache)
        return self.imputed_state["adata"]

    def ask_csv_paths(self, what, prefix, token, has_counts, has_imputed):
        """{"merfish": path, "imputed": path} for the .csv files an export of `what` will write: a Save dialog for each of the two datasets that
        has data (a file is only written if its dataset has something to put in it), each proposing `<prefix>_<token>_<dataset>.csv`. None if a
        dialog was cancelled (nothing is exported then)."""
        paths = {}
        for key, present in (("merfish", has_counts), ("imputed", has_imputed)):
            if not present:
                continue
            proposed = self.host["unique_export_paths"](self.target_dir(), f"{prefix}_{token}_{key}", (".csv",))[".csv"].name
            path = self.ask_save_path(f"Save the {what} of the {'MERFISH' if key == 'merfish' else 'imputed'} dataset", proposed, ".csv", "CSV file")
            if path is None:
                return None
            paths[key] = path
        return paths

    def export_degs(self):
        if self.busy:
            return
        chosen = self.group_choice("exporting DE genes")
        if chosen is None:
            return
        level, targets = chosen
        h = self.host
        level_name = LEVEL_LONG[level]
        choice = h["prompt_deg_comparison_mode"](self.root, f"DEGs for {level_name} {', '.join(str(t) for t in targets)}", level_name)
        if choice is None:
            self.var_status.set("DEG export cancelled.")
            return
        mode, reference = choice["mode"], choice["reference"]
        eff = [t for t in targets if t != reference] if mode == "reference" else list(targets)
        if not eff:
            self.var_status.set(f"Every entered {level_name.lower()} ID equals the reference ({reference}); nothing to compare.")
            return
        ids_arr = self.level_ids(level)
        adata, sc, has_counts = self.adata, h["sc"], self.has_counts
        imputed_full = self.imputed_for_export()
        token = safe_token(f"{level}_{'-'.join(str(t) for t in eff)}" + (f"_vsRef{reference}" if mode == "reference" else ""), level)
        paths = self.ask_csv_paths("DE genes", "DEG", token, has_counts, imputed_full is not None)
        if paths is None:
            self.var_status.set("DEG export cancelled.")
            return

        def work():
            written, skipped = [], []
            if has_counts:
                m = adata.copy()
                m.X = m.layers["counts"].copy()
                sc.pp.normalize_total(m)
                sc.pp.log1p(m)
                frames = []
                for t in eff:
                    if mode == "reference":
                        pair = (ids_arr == t) | (ids_arr == reference)
                        comp, gm = m[pair], ids_arr[pair] == t
                    else:
                        comp, gm = m, ids_arr == t
                    if not gm.any() or gm.all():
                        continue
                    df = h["compute_de_genes"](comp, gm, log1p_base=None)
                    df.insert(0, "group_id", t)
                    if mode == "reference":
                        df.insert(1, "reference_id", reference)
                    frames.append(df)
                if frames:
                    out = pd.concat(frames, ignore_index=True)
                    path = paths["merfish"]
                    out.to_csv(path, index=False)
                    written.append((path, len(out), len(frames)))
                else:
                    skipped.append("MERFISH (no specified ID had both its own cells and a group to compare with)")
            else:
                skipped.append("MERFISH (no counts layer on this dataset)")
            if imputed_full is not None:
                pos = imputed_full.obs_names.get_indexer(adata.obs_names)
                found = pos >= 0
                if found.any():
                    sub = h["materialize_subset"](imputed_full, pos[found])
                    iid = ids_arr[found]
                    frames = []
                    for t in eff:
                        if mode == "reference":
                            pair = (iid == t) | (iid == reference)
                            comp, gm = sub[pair], iid[pair] == t
                        else:
                            comp, gm = sub, iid == t
                        if not gm.any() or gm.all():
                            continue
                        df = h["compute_de_genes"](comp, gm, log1p_base=2)
                        df.insert(0, "group_id", t)
                        if mode == "reference":
                            df.insert(1, "reference_id", reference)
                        frames.append(df)
                    if frames:
                        out = pd.concat(frames, ignore_index=True)
                        path = paths["imputed"]
                        out.to_csv(path, index=False)
                        written.append((path, len(out), len(frames)))
                    else:
                        skipped.append("imputed (no specified ID had both its own cells and a group to compare with)")
                else:
                    skipped.append("imputed (none of this run's cells found in the imputed dataset)")
            else:
                skipped.append("imputed (dataset not loaded)")
            return written, skipped

        def done(res):
            if isinstance(res, BaseException):
                self.var_status.set(f"Could not export DE genes: {res}")
                return
            written, skipped = res
            if written:
                summary = "; ".join(f"{p.name} ({n} rows, {k} of {len(eff)} IDs)" for p, n, k in written)
                msg = (f"Exported {summary}" + (f" vs. reference {level_name.lower()} {reference}" if mode == "reference" else "")
                       + f" to {', '.join(sorted({str(p.parent) for p, _n, _k in written}))}.")
                if skipped:
                    msg += f" Skipped: {'; '.join(skipped)}."
                self.var_status.set(msg)
            else:
                self.var_status.set(f"Could not export DE genes: {'; '.join(skipped) or 'nothing to export'}.")

        self.background("Computing DE genes...", work, done)

    def export_expression(self):
        if self.busy:
            return
        chosen = self.group_choice("exporting expression")
        if chosen is None:
            return
        level, targets = chosen
        h = self.host
        ids_arr = self.level_ids(level)
        adata, sc, has_counts = self.adata, h["sc"], self.has_counts
        imputed_full = self.imputed_for_export()
        token = safe_token(f"{level}_{'-'.join(str(t) for t in targets)}", level)
        paths = self.ask_csv_paths("expression", "EXPR", token, has_counts, imputed_full is not None)
        if paths is None:
            self.var_status.set("Expression export cancelled.")
            return

        def work():
            written, skipped = [], []
            if has_counts:
                m = adata.copy()
                m.X = m.layers["counts"].copy()
                sc.pp.normalize_total(m)
                sc.pp.log1p(m)
                df = pd.DataFrame({"gene_symbol": m.var["gene_symbol"].values, "ensembl_id": m.var_names})
                included = []
                for t in targets:
                    gm = ids_arr == t
                    if gm.any():
                        df[f"mean_log2_expr_{t}"] = h["mean_log2_expression"](m, gm, log1p_base=None).values
                        included.append(t)
                if included:
                    df = df.sort_values("gene_symbol", kind="stable").reset_index(drop=True)
                    path = paths["merfish"]
                    df.to_csv(path, index=False)
                    written.append((path, len(df), len(included)))
                else:
                    skipped.append("MERFISH (no specified ID had any matching cells)")
            else:
                skipped.append("MERFISH (no counts layer on this dataset)")
            if imputed_full is not None:
                pos = imputed_full.obs_names.get_indexer(adata.obs_names)
                found = pos >= 0
                if found.any():
                    sub = h["materialize_subset"](imputed_full, pos[found])
                    iid = ids_arr[found]
                    df = pd.DataFrame({"gene_symbol": sub.var["gene_symbol"].values, "ensembl_id": sub.var_names})
                    included = []
                    for t in targets:
                        gm = iid == t
                        if gm.any():
                            df[f"mean_log2_expr_{t}"] = h["mean_log2_expression"](sub, gm, log1p_base=2).values
                            included.append(t)
                    if included:
                        df = df.sort_values("gene_symbol", kind="stable").reset_index(drop=True)
                        path = paths["imputed"]
                        df.to_csv(path, index=False)
                        written.append((path, len(df), len(included)))
                    else:
                        skipped.append("imputed (no specified ID had any matching cells)")
                else:
                    skipped.append("imputed (none of this run's cells found in the imputed dataset)")
            else:
                skipped.append("imputed (dataset not loaded)")
            return written, skipped

        def done(res):
            if isinstance(res, BaseException):
                self.var_status.set(f"Could not export expression: {res}")
                return
            written, skipped = res
            if written:
                msg = ("Exported " + "; ".join(f"{p.name} ({n} rows, {k} of {len(targets)} IDs)" for p, n, k in written)
                       + f" to {', '.join(sorted({str(p.parent) for p, _n, _k in written}))}.")
                if skipped:
                    msg += f" Skipped: {'; '.join(skipped)}."
                self.var_status.set(msg)
            else:
                self.var_status.set(f"Could not export expression: {'; '.join(skipped) or 'nothing to export'}.")

        self.background("Computing expression...", work, done)

    # ---- coordinates and the 3D views

    def selected_group_cells(self, level, targets):
        """(table, scope): one row per cell of the groups: the whole brain when the all-cell table is loaded, else only this run's cells."""
        bg = self.bg
        frames = []
        barr = bg["levels"].get(level) if bg else None
        use_bg = bg is not None and barr is not None and bg.get("sec_arr") is not None
        if use_bg:
            ids = core.extract_leading_numeric_id(pd.Series(barr)).to_numpy()
            for t in targets:
                rows = np.flatnonzero(ids == t)
                if not rows.size:
                    continue
                f = pd.DataFrame({"cell_label": bg["ids"][rows], "group_id": t, "brain_section_label": bg["sec_arr"][rows]})
                for tl in ("class", "subclass", "supertype", "cluster"):
                    v = bg["levels"].get(tl)
                    if v is not None:
                        f[tl] = v[rows]
                f["section_x_mm"] = bg["x"][rows]
                f["section_y_mm"] = bg["y"][rows]
                if bg.get("z") is not None:
                    f["section_z_mm"] = bg["z"][rows]
                frames.append(f)
        else:
            ids = self.level_ids(level)
            for t in targets:
                rows = np.flatnonzero(ids == t)
                if not rows.size:
                    continue
                f = pd.DataFrame({"cell_label": self.adata.obs_names[rows], "group_id": t})
                if self.section_of is not None:
                    f["brain_section_label"] = self.section_of[rows]
                for tl in ("class", "subclass", "supertype", "cluster"):
                    if self.info.get(tl) is not None:
                        f[tl] = self.info[tl][rows]
                if self.cx is not None:
                    f["section_x_mm"] = self.cx[rows]
                    f["section_y_mm"] = self.cy[rows]
                frames.append(f)
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["cell_label", "group_id"])
        umap_rows = self.run_index.get_indexer(table["cell_label"])
        in_run = umap_rows >= 0
        table["in_run"] = in_run
        table["umap1"] = np.where(in_run, self.coords[np.maximum(umap_rows, 0), 0], np.nan)
        table["umap2"] = np.where(in_run, self.coords[np.maximum(umap_rows, 0), 1], np.nan)
        return table, ("whole brain" if use_bg else "this run only")

    def attach_ccf(self, table, shell_points=0):
        h = self.host
        if not h["ensure_ccf_downloaded"](self.abc_cache, parent=self.root):
            return None, False
        aligned, shell = h["run_with_loading_window"](
            "Reading the CCF coordinates (the first read of this 4-million-cell table takes about a minute).",
            lambda: h["load_ccf_for_cells"](self.abc_cache, table["cell_label"].tolist(), shell_points))
        for source, name in (("x_ccf", "ccf_x_mm"), ("y_ccf", "ccf_y_mm"), ("z_ccf", "ccf_z_mm")):
            table[name] = aligned[source].to_numpy()
        names = h["load_parcellation_names"](self.abc_cache)
        if names is not None:
            parcel = aligned["parcellation_index"].to_numpy()
            parcel = np.where(np.isfinite(parcel), parcel, -1).astype(int)
            for column in ("division", "structure", "substructure"):
                table[f"ccf_{column}"] = names[column].reindex(parcel).to_numpy()
        return shell, True

    def export_coordinates(self):
        if self.busy:
            return
        chosen = self.group_choice("exporting coordinates")
        if chosen is None:
            return
        level, targets = chosen
        stem = f"coordinates_{level}_{'-'.join(str(t) for t in targets)}"
        proposed = self.host["unique_export_paths"](self.target_dir(), stem, (".csv",))[".csv"].name
        path = self.ask_save_path("Save the coordinates of the cells of those groups", proposed, ".csv", "CSV file")
        if path is None:
            self.var_status.set("Coordinates export cancelled.")
            return
        self.root.config(cursor="watch")
        self.var_status.set("Gathering cells...")
        self.root.update()
        try:
            table, scope = self.selected_group_cells(level, targets)
            if table.empty:
                self.var_status.set("None of those IDs has any cells to export.")
            else:
                _shell, have_ccf = self.attach_ccf(table)
                front = ["cell_label", "group_id", "brain_section_label", "class", "subclass", "supertype", "cluster", "section_x_mm", "section_y_mm",
                         "section_z_mm", "ccf_x_mm", "ccf_y_mm", "ccf_z_mm", "ccf_division", "ccf_structure", "ccf_substructure", "in_run", "umap1", "umap2"]
                table = table[[c for c in front if c in table.columns]]
                table.to_csv(path, index=False)
                self.var_status.set(f"Saved {len(table):,} cells ({scope})" + ("" if have_ccf else " (no CCF columns: that download was declined)") + f" to {path}.")
        except Exception as e:
            self.var_status.set(f"Could not export coordinates: {e}")
            print(f"Could not export coordinates: {e}", flush=True)
        self.root.config(cursor="")

    def prepare_3d(self, action):
        chosen = self.group_choice(action)
        if chosen is None:
            return None
        level, targets = chosen
        h = self.host
        table, scope = self.selected_group_cells(level, targets)
        if table.empty:
            self.var_status.set("None of those IDs has any cells to show.")
            return None
        shell, have_ccf = self.attach_ccf(table, shell_points=h["THREED_SHELL_POINTS"])
        if have_ccf:
            columns = ("ccf_z_mm", "ccf_x_mm", "ccf_y_mm")
            axis_labels = ("left - right (CCF z, mm)", "anterior - posterior (CCF x, mm)", "dorsal - ventral (CCF y, mm)")
            shell_xyz = shell[:, [2, 0, 1]]
            outline = h["brain_outline_segments"](shell_xyz)
        else:
            columns = ("section_x_mm", "section_z_mm", "section_y_mm")
            axis_labels = ("section x (mm)", "section position (mm)", "section y (mm)")
            shell_xyz, outline = None, None
        level_name = LEVEL_LONG[level]
        groups = []
        for t in targets:
            rows = table[table["group_id"] == t]
            hover = None
            if have_ccf and "ccf_structure" in rows.columns:
                hover = (rows["ccf_structure"].fillna("?").astype(str) + " | section " + rows["brain_section_label"].astype(str).str.rsplit(".", n=1).str[-1]
                         + " | " + rows["cell_label"].astype(str)).to_numpy()
            color = self.last_ids["colors"].get(t)
            groups.append({"label": f"{level_name} {t} ({len(rows):,})", "color": hexcolor(color) if color is not None else "red",
                           "xyz": rows[list(columns)].to_numpy(dtype=float), "hover": hover})
        structures = table["ccf_structure"].to_numpy() if have_ccf and "ccf_structure" in table.columns else None
        return {"groups": groups, "shell_xyz": shell_xyz, "outline": outline, "axis_labels": axis_labels, "structures": structures,
                "title": f"3D view: {level_name.lower()} {', '.join(str(t) for t in targets)} ({scope})", "have_ccf": have_ccf, "level": level,
                "targets": targets}

    def show_3d(self):
        if self.busy:
            return
        self.root.config(cursor="watch")
        self.var_status.set("Preparing the 3D view...")
        self.root.update()
        try:
            p = self.prepare_3d("opening the 3D view")
            if p is not None:
                self.view3d = getattr(self, "view3d", [])
                self.view3d.append(self.host["open_3d_group_view"](p["groups"], p["shell_xyz"], p["outline"], p["axis_labels"], p["title"]))
                self.var_status.set("3D view opened: drag to rotate, click legend entries to hide/show a group."
                                    + ("" if p["have_ccf"] else " (CCF unavailable: section coordinates, so the sections look stacked.)"))
        except Exception as e:
            self.var_status.set(f"Could not open the 3D view: {e}")
            print(f"Could not open the 3D view: {e}", flush=True)
        self.root.config(cursor="")

    def show_3d_browser(self):
        if self.busy:
            return
        try:
            import plotly  # noqa: F401
        except ImportError:
            self.var_status.set("The browser 3D view needs plotly: run  python -m pip install plotly  and restart.")
            return
        self.root.config(cursor="watch")
        self.var_status.set("Preparing the 3D view...")
        self.root.update()
        try:
            p = self.prepare_3d("opening the 3D view")
            if p is not None:
                h = self.host
                target = self.target_dir()
                path = h["unique_export_paths"](target, f"3d_{p['level']}_{'-'.join(str(t) for t in p['targets'])}", (".html",))[".html"]
                edges = h["build_3d_edges"](self.abc_cache, p["structures"], self.root) if p["have_ccf"] else None
                self.var_status.set("Writing the 3D page...")
                self.root.update()
                h["write_3d_group_html"](p["groups"], p["shell_xyz"], p["outline"], p["axis_labels"], p["title"], path, edges=edges)
                h["open_with_default_viewer"](path)
                self.var_status.set(f"Opened {path.name} in your browser (saved in {target}).")
        except Exception as e:
            self.var_status.set(f"Could not open the 3D view in the browser: {e}")
            print(f"Could not open the 3D view in the browser: {e}", flush=True)
        self.root.config(cursor="")


LEVEL_LONG = {"class": "Class", "subclass": "Subclass", "supertype": "Supertype", "cluster": "Cluster", "leiden": "Leiden", "neurotransmitter": "Neurotransmitter"}
