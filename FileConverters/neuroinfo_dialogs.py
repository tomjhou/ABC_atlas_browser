"""Dialogs shared by the alignment tool and the section loader (both open NeuroInfo JPX files)."""
import os
import time
import tkinter as tk
from tkinter import filedialog, ttk


def missing_files_dialog(parent, jpx_name, missing):
    """A JPX from NeuroInfo has been added but the files NeuroInfo wrote with it are not beside it. `missing`: what is not there, as text.
    Returns "locate" (the user will point at them), "keep" (add it anyway; they are asked for again later) or "remove" (don't add it)."""
    result = {"value": "remove"}
    dlg = tk.Toplevel(parent)
    dlg.title("NeuroInfo files not found")
    dlg.transient(parent)
    dlg.resizable(False, False)
    text = (f"{jpx_name} comes from NeuroInfo, so this tool needs the files NeuroInfo wrote with it: the reconstruction .xml (the outlines "
            f"and where the sections are) and the case file (.h5).\n\nNot found beside the JPX: {missing}.\n\n"
            "Without them this JPX can't be aligned here: its sections can't be traced back to their source images.\n\n"
            "If you still have the original slide scans (.czi) or section images (.jp2), load those instead: the tool then finds the outlines "
            "itself, and you go on to the order and the alignment from there (anything NeuroInfo had already aligned is not carried over).\n\n"
            "If the files exist but are somewhere else, choose Locate files.")
    ttk.Label(dlg, text=text, justify=tk.LEFT, padding=12, wraplength=560).pack(anchor="w")
    buttons = ttk.Frame(dlg, padding=(12, 0, 12, 12))
    buttons.pack(fill=tk.X)

    def choose(value):
        result["value"] = value
        dlg.destroy()

    ttk.Button(buttons, text="Locate files...", command=lambda: choose("locate")).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Keep it anyway", command=lambda: choose("keep")).pack(side=tk.LEFT, padx=8)
    ttk.Button(buttons, text="Don't add it", command=lambda: choose("remove")).pack(side=tk.RIGHT)
    dlg.protocol("WM_DELETE_WINDOW", lambda: choose("remove"))
    dlg.update_idletasks()
    dlg.geometry(f"+{parent.winfo_rootx() + 100}+{parent.winfo_rooty() + 80}")
    dlg.grab_set()
    dlg.wait_window()
    return result["value"]


def locate_file(parent, kind, expected, folder, candidates, reason="missing", default=None, notes=None):
    """Ask which NeuroInfo file to use: a list of the candidates in the folder (the likeliest selected), or Browse for
    another; for a file that is not there at all, only Browse. Returns the path, or None if the user declines.
    `parent` is the window the dialog belongs to."""
    notes = notes or {}
    what = {"h5": "case file (.h5)", "jpx": "JPX stack (.jpx)"}.get(kind, "reconstruction file (.xml)")
    if kind == "jpx":      # (asked when a reconstruction .xml was opened: its JPX is what is being looked for)
        if reason == "ambiguous":
            text = (f"{expected} wasn't found beside the reconstruction file. " if expected else "") + \
                   f"Which {what} goes with it?\n{folder}"
        else:
            text = f"No {what} was found beside the reconstruction file, in\n{folder}\n\nLocate it?"
    elif reason == "ambiguous":
        text = (f"Which {what} goes with this JPX?\n{folder}" if kind == "h5" else
                f"{expected} wasn't found beside the JPX. Which {what} goes with it?\n{folder}")
    else:
        text = f"No {what} was found beside the JPX, in\n{folder}\n\nLocate it?"
    result = {"path": None}
    dlg = tk.Toplevel(parent)
    dlg.title(f"Choose the {what}" if candidates else "File not found")
    dlg.transient(parent)
    dlg.resizable(False, False)
    ttk.Label(dlg, text=text, justify=tk.LEFT, padding=10, wraplength=520).pack(anchor="w")
    box = None
    if candidates:
        box = tk.Listbox(dlg, height=min(max(len(candidates), 3), 10), width=70, exportselection=False, activestyle="none")
        for name in candidates:
            try:
                st = os.stat(os.path.join(folder, name))
                info = f"{st.st_size / 1e6:.1f} MB, {time.strftime('%Y-%m-%d', time.localtime(st.st_mtime))}"
            except OSError:
                info = ""
            note = notes.get(name)
            box.insert(tk.END, f"{name}   ({info}{'; ' + note if note else ''})")
        box.selection_set(candidates.index(default) if default in candidates else 0)
        box.pack(padx=10)

    def use():
        if box is not None and box.curselection():
            result["path"] = os.path.join(folder, candidates[box.curselection()[0]])
        dlg.destroy()

    def browse():
        types = {"h5": [("HDF5 files", "*.h5 *.H5")], "jpx": [("JPX stacks", "*.jpx")]}.get(kind, [("XML files", "*.xml")])
        path = filedialog.askopenfilename(parent=dlg, title=f"Locate the {what}", initialdir=folder,
                                          filetypes=types + [("All files", "*.*")])
        if path:
            result["path"] = path
            dlg.destroy()

    buttons = ttk.Frame(dlg, padding=10)
    buttons.pack(fill=tk.X)
    if box is not None:
        ttk.Button(buttons, text="Use selected", command=use).pack(side=tk.LEFT)
        box.bind("<Double-Button-1>", lambda e: use())
    ttk.Button(buttons, text="Browse...", command=browse).pack(side=tk.LEFT, padx=6)
    ttk.Button(buttons, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT)
    dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
    dlg.update_idletasks()
    dlg.geometry(f"+{parent.winfo_rootx() + 80}+{parent.winfo_rooty() + 80}")
    dlg.grab_set()
    parent.wait_window(dlg)
    return result["path"]
