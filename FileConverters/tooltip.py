"""A small note that appears when the pointer rests on a widget: shared by the alignment tool and the section loader."""
import tkinter as tk


class Tooltip:
    """A small note that appears when the pointer rests on a widget. `text` may be a function returning the text, called each time the
    note is shown (for a note that depends on the state of the program)."""

    def __init__(self, widget, text, delay=500, wraplength=260):
        self.widget, self.text, self.delay, self.tip, self.job = widget, text, delay, None, None
        self.wraplength = wraplength
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, event=None):
        self.hide()
        self.job = self.widget.after(self.delay, self.show)

    def show(self):
        self.job = None
        text = self.text() if callable(self.text) else self.text
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        tk.Label(self.tip, text=text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                 wraplength=self.wraplength, padx=4, pady=2).pack()
        self.tip.update_idletasks()
        x = min(self.widget.winfo_rootx(), self.widget.winfo_screenwidth() - self.tip.winfo_reqwidth() - 4)   # (kept on screen)
        self.tip.wm_geometry(f"+{max(x, 0)}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 4}")

    def hide(self, event=None):
        if self.job:
            self.widget.after_cancel(self.job)
            self.job = None
        if self.tip:
            self.tip.destroy()
            self.tip = None
