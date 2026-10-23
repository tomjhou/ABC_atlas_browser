"""A Tk version of the atlas browser's launch screen: the output folder, then one of "Open Brain Browser / ROI Chooser", "Load
Existing ROI Dataset" or "Open 3D Viewer". Which cells to analyze (Neurons, Non-neurons or All cells) is asked later, by
choose_cells_to_analyze(), once the sections and ROIs are confirmed. It is a function of the same shape as the matplotlib one (prompt_startup_panel in
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


def choose_cells_to_analyze(current="Neurons"):
    """A small window, after the sections and ROIs are confirmed: which cells the analysis (the UMAP and what follows) is made of. Returns
    "Neurons" | "NonNeurons" | "All", or None if cancelled."""
    root = tk.Tk()
    root.title("Cells to analyze")
    root.resizable(False, False)
    choice = tk.StringVar(root, current)
    answer = {"value": None}
    f = ttk.Frame(root, padding=22)
    f.pack()
    ttk.Label(f, text="Which cells do you want to analyze?", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(f, text="The cells in the sections and ROIs you chose, of this kind, are made into the UMAP.", foreground="gray30",
              wraplength=330, justify=tk.LEFT).pack(anchor="w", pady=(2, 8))
    for value, text in CELL_TYPES:
        ttk.Radiobutton(f, text=text, value=value, variable=choice).pack(anchor="w", pady=(4, 0))
    row = ttk.Frame(f)
    row.pack(fill=tk.X, pady=(16, 0))

    def done(ok):
        answer["value"] = choice.get() if ok else None
        root.destroy()

    ok_button = ttk.Button(row, text="Continue", command=lambda: done(True))
    ok_button.pack(side=tk.RIGHT)
    ttk.Button(row, text="Cancel", command=lambda: done(False)).pack(side=tk.RIGHT, padx=(0, 8))
    root.bind("<Return>", lambda e: done(True))
    root.bind("<Escape>", lambda e: done(False))
    root.protocol("WM_DELETE_WINDOW", lambda: done(False))
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - root.winfo_width()) // 2}+{(sh - root.winfo_height()) // 3}")
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
