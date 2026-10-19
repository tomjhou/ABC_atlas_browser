"""The start window: two steps, one after the other.

  1. Load / detect images - choose Zeiss .czi slide scans, find the sections on them, put them in order (section_loader).
  2. Align images         - the alignment tool (jpx_aligner); after step 1 it opens with those sections, in memory.

Run with a file name (a .jpx, or a project saved by step 1) to go straight to the alignment tool."""
import os
import sys
import tkinter as tk
from tkinter import ttk

import neuroinfo_stack as ns

ns.ensure_packages({"PIL": "pillow"})


def choose():
    """The start window. Returns "detect", "align" or None (closed)."""
    root = tk.Tk()
    root.title("Section alignment")
    root.resizable(False, False)
    choice = []
    outer = ttk.Frame(root, padding=28)
    outer.pack()
    ttk.Label(outer, text="Section alignment", font=("Segoe UI", 18, "bold")).pack(anchor="w")
    ttk.Label(outer, text="Two steps, in this order. Step 2 picks up where step 1 left off.", foreground="gray30").pack(anchor="w", pady=(2, 18))

    def card(number, title, text, key, primary):
        f = tk.Frame(outer, bd=1, relief="solid", padx=16, pady=12, bg="white")
        f.pack(fill=tk.X, pady=6)
        tk.Label(f, text=number, font=("Segoe UI", 26, "bold"), fg="#1f5fa8", bg="white", width=2).pack(side=tk.LEFT)
        mid = tk.Frame(f, bg="white")
        mid.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 16))
        tk.Label(mid, text=title, font=("Segoe UI", 13, "bold"), bg="white", anchor="w").pack(fill=tk.X)
        tk.Label(mid, text=text, bg="white", fg="gray30", anchor="w", justify="left", wraplength=360).pack(fill=tk.X)
        b = ttk.Button(f, text="Start", width=10, command=lambda: (choice.append(key), root.destroy()))
        b.pack(side=tk.RIGHT)
        if primary:
            b.focus_set()

    card("1", "Load / detect images",
         "Choose slide scans (.czi), find the sections on them, and put the sections in order.", "detect", True)
    card("2", "Align images",
         "Line up the sections and export the stack. Opens the sections from step 1, or a JPX you already have.", "align", False)
    ttk.Separator(outer).pack(fill=tk.X, pady=(14, 8))
    ttk.Style(root).configure("Exit.TButton", font=("Segoe UI", 13))
    ttk.Button(outer, text="Exit", width=24, style="Exit.TButton", command=root.destroy).pack(side=tk.BOTTOM, ipady=6, pady=(0, 4))   # (centered)
    root.mainloop()
    return choice[0] if choice else None


def run_aligner(stack=None, path=None, open_dialog=False, back_to_start=False):
    """The alignment tool, until its window is closed. With back_to_start, closing it returns here (to show the start window again)."""
    import jpx_aligner as J
    app = J.Aligner(path if stack is None else None)
    app.return_to_launcher = back_to_start
    if stack is not None:
        app.show_stack(stack, path)
    elif open_dialog and path is None:
        app.root.after(200, app.open_dialog)
    app.run()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:                                  # a file named: straight to the alignment tool
        run_aligner(path=argv[0])
        return
    while True:                               # (the start window comes back whenever a tool is closed)
        what = choose()
        if what == "align":
            run_aligner(open_dialog=True, back_to_start=True)
        elif what == "detect":
            import section_loader
            loader = section_loader.SectionLoader()
            stack = loader.run()
            if stack is not None:
                run_aligner(stack=stack, path=loader.project.path, back_to_start=True)
        else:
            break
    sys.stdout.flush()                        # (leave at once: background threads may be mid-task)
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
