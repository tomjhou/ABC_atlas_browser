"""A Tk version of the atlas browser's launch screen: the output folder, then one of "Open Brain Browser / ROI Chooser", "Load
Existing ROI Dataset" or "Open 3D Viewer". Which cells to analyze (Neurons, Non-neurons or All cells) is asked later, by
choose_run_options(), once the sections and ROIs are confirmed. It is a function of the same shape as the matplotlib one (prompt_startup_panel in
ABC_atlas_browser.py): the browser passes in the few of its own functions it needs (recent folders, the check of a run folder, the 3D
viewer launcher, the settings window), so this file has no start-up side effects of its own.

    outcome = prompt_startup_panel_tk(out_folder, abc_cache, helpers)

`outcome` is (action, cell_type_selection, load_folder, out_folder) with action 'new_run' or 'load_existing', or None if the window was closed
or Exit was clicked. The cell type is None for a new run (not chosen yet) and unused for an existing one (the run says which it is).
"""
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

CELL_TYPES = (("Neurons", "Neurons"), ("NonNeurons", "Non-neurons"), ("All", "All cells"))      # (value, the words on the screen)
ACCENT = "#1f5fa8"
RECOVERABLE = {"ROI coordinates", "Processed AnnData"}      # (cached files of a run that can be made again: not required to load it)


SUBSAMPLE_CAPS = (("200,000 cells (recommended)", 200_000), ("No limit: every cell", None))
LARGE_PROJECT_CELLS = 200_000            # a selection of more cells than this is a "large project": the dialog asks how to handle it


def choose_run_options(current="Neurons", counts=None):
    """The one window between confirming the sections and ROIs and the long analysis that follows, so that the analysis then runs without
    asking anything more: which cells to analyze, how to handle a large project (a selection of more than LARGE_PROJECT_CELLS cells: use a
    random subset, or all), the project name that is added to the folder name, and whether to make the optional subclass / supertype plots.
    `counts`: {"Neurons": n, "NonNeurons": n, "All": n}, the cells in the selection, if known (they are shown, and the large-project question
    is asked only when the chosen kind has more than LARGE_PROJECT_CELLS; if not known it is always asked).
    Returns a dict {"cell_type": "Neurons" | "NonNeurons" | "All", "cap": the most cells to use, or None for no limit, "label": the project
    name (may be empty), "optional_plots": bool}, or None if cancelled."""
    root = tk.Tk()
    root.title("Analysis options")
    root.resizable(False, False)
    cell_type = tk.StringVar(root, current)
    cap_choice = tk.StringVar(root, SUBSAMPLE_CAPS[0][0])
    custom_cap = tk.StringVar(root, "")
    label = tk.StringVar(root, "")
    optional = tk.BooleanVar(root, False)
    answer = {"value": None}
    f = ttk.Frame(root, padding=22)
    f.pack()

    ttk.Label(f, text="Which cells do you want to analyze?", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(f, text="The cells of this kind in the sections and ROIs you chose are made into the UMAP.", foreground="gray30",
              wraplength=400, justify=tk.LEFT).pack(anchor="w", pady=(2, 4))
    for value, text in CELL_TYPES:
        shown = text if not counts or counts.get(value) is None else f"{text}  ({counts[value]:,} cells)"
        ttk.Radiobutton(f, text=shown, value=value, variable=cell_type).pack(anchor="w", pady=(3, 0))

    slot = ttk.Frame(f)                                           # (where the large-project question goes, when it is asked: see show_large)
    slot.pack(anchor="w")
    large = ttk.Frame(slot)
    ttk.Label(large, text="How to handle large projects", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(16, 0))
    large_text = tk.StringVar(large, "")
    ttk.Label(large, textvariable=large_text, foreground="gray30", wraplength=400, justify=tk.LEFT).pack(anchor="w", pady=(2, 4))
    for text, _cap in SUBSAMPLE_CAPS:
        ttk.Radiobutton(large, text=text, value=text, variable=cap_choice).pack(anchor="w", pady=(3, 0))
    custom_row = ttk.Frame(large)
    custom_row.pack(anchor="w", pady=(3, 0))
    ttk.Radiobutton(custom_row, text="At most", value="custom", variable=cap_choice).pack(side=tk.LEFT)
    custom_entry = ttk.Entry(custom_row, textvariable=custom_cap, width=10)
    custom_entry.pack(side=tk.LEFT, padx=(6, 6))
    ttk.Label(custom_row, text="cells").pack(side=tk.LEFT)
    custom_cap.trace_add("write", lambda *args: cap_choice.set("custom") if custom_cap.get() else None)         # (typing a number picks "At most")

    ttk.Label(f, text="Project name (optional)", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(16, 0))
    ttk.Label(f, text="Added to the end of the project's folder name, to remember what it is, e.g. RMTg-VTA.", foreground="gray30",
              wraplength=400, justify=tk.LEFT).pack(anchor="w", pady=(2, 4))
    ttk.Entry(f, textvariable=label, width=44).pack(anchor="w")

    ttk.Checkbutton(f, text="Also make the optional subclass / supertype UMAP plots and spatial maps (slower)", variable=optional).pack(
        anchor="w", pady=(16, 0))

    row = ttk.Frame(f)
    row.pack(fill=tk.X, pady=(18, 0))

    def is_large():
        n = counts.get(cell_type.get()) if counts else None
        return n is None or n > LARGE_PROJECT_CELLS

    def show_large(*args):
        if is_large():
            n = counts.get(cell_type.get()) if counts else None
            large_text.set((f"This selection has {n:,} cells. " if n is not None else "") +
                           "A UMAP of that many cells takes a very long time. Use a random subset of this size, or all of them:")
            large.pack(anchor="w")
        else:
            large.pack_forget()

    def done(ok):
        if not ok:
            answer["value"] = None
            root.destroy()
            return
        cap = LARGE_PROJECT_CELLS            # (a selection that is not large is not limited by this)
        if is_large():
            cap = dict(SUBSAMPLE_CAPS).get(cap_choice.get(), "custom")
            if cap == "custom":
                try:
                    cap = int(custom_cap.get().replace(",", "").strip())
                    if cap < 1:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("How to handle large projects", "Enter a whole number of cells, or pick one of the other choices.", parent=root)
                    return
        answer["value"] = {"cell_type": cell_type.get(), "cap": cap, "label": label.get().strip(), "optional_plots": bool(optional.get())}
        root.destroy()

    ok_button = ttk.Button(row, text="Continue", command=lambda: done(True))
    ok_button.pack(side=tk.RIGHT)
    ttk.Button(row, text="Cancel", command=lambda: done(False)).pack(side=tk.RIGHT, padx=(0, 8))
    root.bind("<Return>", lambda e: done(True))
    root.bind("<Escape>", lambda e: done(False))
    root.protocol("WM_DELETE_WINDOW", lambda: done(False))
    cell_type.trace_add("write", show_large)
    root.update_idletasks()
    show_large()
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - root.winfo_width()) // 2}+{(sh - root.winfo_height()) // 4}")
    root.attributes("-topmost", True)
    ok_button.focus_force()
    root.mainloop()
    return answer["value"]


def prompt_startup_panel_tk(out_folder, abc_cache, helpers):
    """`helpers`: load_recent_output_folders(), remember_recent_output_folder(folder), required_cached_run_files(folder) ->
    {label: Path}, open_3d_viewer(abc_cache), settings_dialog(abc_cache)."""
    root = tk.Tk()
    root.title("AllenABC UMAP Explorer")
    root.resizable(False, False)
    result = {"outcome": None}

    outer = ttk.Frame(root, padding=(28, 20, 28, 22))
    outer.pack()
    top = ttk.Frame(outer)
    top.pack(fill=tk.X)
    ttk.Label(top, text="AllenABC UMAP Explorer", font=("Segoe UI", 20, "bold")).pack(side=tk.LEFT)
    ttk.Button(top, text="Settings…", command=lambda: helpers["settings_dialog"](abc_cache)).pack(side=tk.RIGHT, anchor="n")

    ttk.Label(outer, text="Output folder (where runs are saved and loaded from):", font=("Segoe UI", 11)).pack(anchor="w", pady=(18, 4))
    row = ttk.Frame(outer)
    row.pack(fill=tk.X)
    folder_var = tk.StringVar(root, str(out_folder))

    def recent_items():
        typed = folder_var.get().strip().strip('"')
        items = list(helpers["load_recent_output_folders"]())
        norm = lambda p: os.path.normcase(os.path.abspath(os.path.expanduser(p)))
        if typed and norm(typed) not in {norm(p) for p in items}:
            items = [typed] + items
        return items

    folder_box = ttk.Combobox(row, textvariable=folder_var, width=70, postcommand=lambda: folder_box.config(values=recent_items()))
    folder_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
    folder_box.config(values=recent_items())

    def browse():
        raw = folder_var.get().strip().strip('"')
        initial = raw if raw and Path(os.path.expanduser(raw)).is_dir() else str(out_folder)
        chosen = filedialog.askdirectory(parent=root, title="Choose the output folder", initialdir=initial)
        if chosen:
            folder_var.set(os.path.normpath(chosen))

    ttk.Button(row, text="Browse…", command=browse).pack(side=tk.LEFT, padx=(8, 0))

    def committed_folder():
        """The folder in the box, made if it doesn't exist; None (after saying why) if it can't be."""
        raw = folder_var.get().strip().strip('"')
        if not raw:
            messagebox.showerror("Output folder", "Enter an output folder, or click Browse… to pick one.", parent=root)
            return None
        path = Path(os.path.expanduser(raw))
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Output folder", f"Can't create or access that folder:\n\n{path}\n\n{e}", parent=root)
            return None
        return path

    def new_run():
        folder = committed_folder()
        if folder is None:
            return
        helpers["remember_recent_output_folder"](folder)
        result["outcome"] = ("new_run", None, None, folder)
        root.destroy()

    def load_existing():
        folder = committed_folder()
        if folder is None:
            return
        chosen = filedialog.askdirectory(parent=root, title="Select an existing run folder to load", initialdir=str(folder))
        if not chosen:
            return
        chosen_path = Path(chosen)
        required = helpers["required_cached_run_files"](chosen_path)
        missing = [f"{label}: {path}" for label, path in required.items() if label not in RECOVERABLE and not path.exists()]
        if missing:
            messagebox.showerror("Missing cached files", "This folder is missing required cached file(s):\n\n" + "\n".join(missing), parent=root)
            return                                          # (stay here, so another folder can be picked)
        helpers["remember_recent_output_folder"](folder)
        result["outcome"] = ("load_existing", "All", chosen_path, folder)
        root.destroy()

    def viewer3d():
        helpers["open_3d_viewer"](abc_cache)                # (returns when that window is closed; this one is still here)

    ttk.Separator(outer).pack(fill=tk.X, pady=(20, 14))
    style = ttk.Style(root)
    style.configure("Big.TButton", font=("Segoe UI", 12), padding=(10, 9))
    for text, command in (("Open Brain Browser/ROI Chooser", new_run), ("Load Existing ROI Dataset…", load_existing),
                          ("Open 3D Viewer…", viewer3d), ("Exit", root.destroy)):
        ttk.Button(outer, text=text, style="Big.TButton", command=command).pack(fill=tk.X, pady=4)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - root.winfo_width()) // 2}+{(sh - root.winfo_height()) // 3}")
    root.mainloop()
    return result["outcome"]


if __name__ == "__main__":
    print(prompt_startup_panel_tk(
        Path.cwd(), None, {"load_recent_output_folders": lambda: [], "remember_recent_output_folder": lambda f: None,
                           "required_cached_run_files": lambda p: {}, "open_3d_viewer": lambda c: print("3D viewer"),
                           "settings_dialog": lambda c: print("settings")}))
