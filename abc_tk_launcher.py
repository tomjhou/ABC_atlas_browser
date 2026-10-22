"""A Tk version of the atlas browser's launch screen: the cell type and output folder, then one of "Open Brain Browser / ROI Chooser", "Load
Existing ROI Dataset" or "Open 3D Viewer". It is a function of the same shape as the matplotlib one (prompt_startup_panel in
ABC_atlas_browser.py): the browser passes in the few of its own functions it needs (recent folders, the check of a run folder, the 3D
viewer launcher, the settings window), so this file has no start-up side effects of its own.

    outcome = prompt_startup_panel_tk(out_folder, abc_cache, helpers)

`outcome` is (action, cell_type_selection, load_folder, out_folder) with action 'new_run', 'load_existing' or 'regenerate' (a new run made
from the ROIs of the run in load_folder, for the chosen cell type), or None if the window was closed or Exit was clicked.
"""
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

CELL_TYPES = ("Neurons", "NonNeurons", "All")
ACCENT = "#1f5fa8"
RUN_PREFIXES = (("umap_nonneurons_", "NonNeurons"), ("umap_neurons_", "Neurons"), ("umap_all_", "All"))      # (a run folder's name starts with its cell type)
RECOVERABLE = {"ROI coordinates", "Processed AnnData"}      # (cached files of a run that can be made again: not required to load it)


def run_cell_type(folder):
    """The cell type ("Neurons" | "NonNeurons" | "All") a run folder was made for, from its name; None if the name doesn't say."""
    name = Path(folder).name.lower()
    for prefix, cell_type in RUN_PREFIXES:
        if name.startswith(prefix):
            return cell_type
    return None


def ask_cell_type_mismatch(parent, folder, run_type, chosen_type):
    """The run was made for another cell type than the one checked. Returns "load" | "regenerate" | "cancel"."""
    top = tk.Toplevel(parent)
    top.title("Cell type mismatch")
    top.transient(parent)
    top.resizable(False, False)
    answer = {"value": "cancel"}
    f = ttk.Frame(top, padding=20)
    f.pack()
    ttk.Label(f, text=f"This run was made for {run_type}, but {chosen_type} is checked on the launcher.", font=("Segoe UI", 11, "bold"),
              wraplength=470, justify=tk.LEFT).pack(anchor="w")
    ttk.Label(f, text=f"{Path(folder).name}", foreground="gray30", wraplength=470, justify=tk.LEFT).pack(anchor="w", pady=(4, 10))
    ttk.Label(f, text=f"Load anyway shows the run as it is ({run_type}). Regenerate makes a new run for {chosen_type} from the same ROI outlines "
                      "(the old run is kept).", wraplength=470, justify=tk.LEFT).pack(anchor="w")
    row = ttk.Frame(f)
    row.pack(fill=tk.X, pady=(16, 0))

    def choose(value):
        answer["value"] = value
        top.destroy()

    ttk.Button(row, text=f"Regenerate as {chosen_type}", command=lambda: choose("regenerate")).pack(side=tk.LEFT)
    ttk.Button(row, text="Cancel", command=lambda: choose("cancel")).pack(side=tk.RIGHT)
    ttk.Button(row, text="Load anyway", command=lambda: choose("load")).pack(side=tk.RIGHT, padx=(0, 8))
    top.protocol("WM_DELETE_WINDOW", lambda: choose("cancel"))
    top.update_idletasks()
    top.geometry(f"+{parent.winfo_rootx() + 40}+{parent.winfo_rooty() + 60}")
    top.grab_set()
    parent.wait_window(top)
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

    ttk.Label(outer, text="Cell type when choosing new section/ROI set", font=("Segoe UI", 11)).pack(anchor="w", pady=(18, 4))
    cell_type = tk.StringVar(root, CELL_TYPES[0])
    radios = ttk.Frame(outer)
    radios.pack(anchor="w")
    for name in CELL_TYPES:
        ttk.Radiobutton(radios, text=name, value=name, variable=cell_type).pack(side=tk.LEFT, padx=(0, 22))

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
        result["outcome"] = ("new_run", cell_type.get(), None, folder)
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
        action = "load_existing"
        run_type = run_cell_type(chosen_path)
        if run_type is not None and run_type != cell_type.get():
            choice = ask_cell_type_mismatch(root, chosen_path, run_type, cell_type.get())
            if choice == "cancel":
                return
            if choice == "regenerate":
                action = "regenerate"
        helpers["remember_recent_output_folder"](folder)
        result["outcome"] = (action, cell_type.get(), chosen_path, folder)
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
