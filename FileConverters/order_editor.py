"""The editing part of putting sections in order, shared by step 3 of the section loader (its ordering tab) and the alignment tool's
"Thumbnails" window: a thumbnail grid (section_order.OrderGrid) with its controls and what they do.

    editor = OrderEditor(parent, host, enabled=..., after_draw=...)      # makes the grid (editor.grid): pack editor.grid.frame
    editor.reverse_button(bar).pack(...)       # the buttons are made here, placed by the caller in its own tool bar:
    editor.reset_button(bar), editor.undo_button(bar), editor.size_buttons(bar)

What the editor does: Reverse, Reset, Undo (also Ctrl+Z through editor.undo), the right-click menu on a thumbnail (Delete section; Insert
placeholder before this section), the small / medium / large thumbnails, and the history of orders that Undo goes back through. The grid's
order is a list of keys; what a key stands for is the host's business. The host (the window that uses the editor) provides:

    order_caption(key, position) -> (left text, right text)      the captions under a thumbnail
    order_double(key, position)                                  a double-click on a thumbnail
    order_status(text)                                           a message for the status line
    order_new_placeholder(position) -> key or None               make a placeholder (a blank section) that goes before `position`
    order_reset_keys() -> list of keys                           the order Reset puts back (the host undoes what it has to, e.g.
                                                                 takes away the placeholders that were inserted)
    order_reset_help() -> str                                    what Reset does, for its tool tip
    order_restore(keys)                                          called before Undo puts an earlier order back: the host makes sure
                                                                 what those keys stand for exists (a placeholder Reset took away)
    order_size_changed(scale)                                    the thumbnails are to be made again at this size (0.5, 1 or 2 times)
"""
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np

from section_order import OrderGrid
from tooltip import Tooltip

SIZES = {"small": 0.5, "medium": 1.0, "large": 2.0}


def placeholder_image(width, height):
    """The picture standing for a placeholder in a thumbnail: a grey box that says what it is (an RGB uint8 array)."""
    img = np.full((height, width, 3), 58, np.uint8)
    scale = max(width / 190, 0.4)
    (text_w, text_h), _ = cv2.getTextSize("Placeholder", cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 1)
    cv2.putText(img, "Placeholder", ((width - text_w) // 2, (height + text_h) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
                (200, 200, 200), 1, cv2.LINE_AA)
    return img


class OrderEditor:
    def __init__(self, parent, host, enabled=None, after_draw=None):
        self.host = host
        self.history = []                            # the orders before each change, newest last (for Undo)
        self.var_size = tk.StringVar(parent, "medium")
        self.btn_undo = None
        self.grid = OrderGrid(parent, host.order_caption, on_double=host.order_double, enabled=enabled, after_draw=after_draw,
                              on_context=self.context, before_change=self.push_history)

    # ---- the grid's list, as the editor's
    @property
    def order(self):
        return self.grid.order

    @order.setter
    def order(self, value):
        self.grid.order = value

    # ---- the controls (made here, placed by the caller)
    def reverse_button(self, bar):
        return ttk.Button(bar, text="Reverse order", command=self.grid.reverse)

    def reset_button(self, bar):
        button = ttk.Button(bar, text="Reset order", command=self.reset)
        Tooltip(button, self.host.order_reset_help, wraplength=340)
        return button

    def undo_button(self, bar):
        self.btn_undo = ttk.Button(bar, text="Undo", command=self.undo, state="disabled")
        Tooltip(self.btn_undo, "Undoes the last change of the order: a move, a deleted section, an inserted placeholder, Reverse order or "
                               "Reset order (Ctrl+Z does the same).", wraplength=300)
        self.update_undo_button()
        return self.btn_undo

    def size_buttons(self, bar):
        """"Thumbnails: Small Medium Large": half, one and two times the usual size."""
        frame = ttk.Frame(bar)
        ttk.Label(frame, text="Thumbnails:").pack(side=tk.LEFT, padx=(0, 4))
        for text, value in (("Small", "small"), ("Medium", "medium"), ("Large", "large")):
            ttk.Radiobutton(frame, text=text, value=value, variable=self.var_size, style="Toolbutton",
                            command=self.on_size).pack(side=tk.LEFT, padx=1)
        return frame

    # ---- the history
    def clear_history(self):
        self.history = []
        self.update_undo_button()

    def push_history(self, snapshot=None):
        """Remember the order as it is (or `snapshot`) before a change, so that Undo can bring it back."""
        self.history.append(list(self.order if snapshot is None else snapshot))
        del self.history[:-200]
        self.update_undo_button()

    def update_undo_button(self):
        if self.btn_undo is not None:
            self.btn_undo.state(["!disabled"] if self.history else ["disabled"])

    def undo(self, event=None):
        """The order as it was before the last change (a move, a deleted section, a placeholder, Reverse, Reset). Not while the keys go to
        a text box (Ctrl+Z there is the box's)."""
        if event is not None and isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return
        if not self.history:
            return
        snapshot = self.history.pop()
        self.host.order_restore(snapshot)
        self.order = snapshot
        self.grid.selected = None
        self.update_undo_button()
        self.grid.draw()
        self.host.order_status("Undone.")

    # ---- what can be done
    def delete(self, position):
        """Take the section at `position` out of the order (Undo, or Reset, brings it back; the section itself is not touched)."""
        if not 0 <= position < len(self.order):
            return
        if len(self.order) <= 1:
            messagebox.showinfo("Delete section", "At least one section has to stay.", parent=self.grid.canvas)
            return
        self.push_history()
        self.order.pop(position)
        self.grid.selected = None
        self.grid.draw()
        self.host.order_status(f"Deleted section {position + 1}; the sections after it moved up. Undo brings it back.")

    def insert_placeholder(self, position):
        """A placeholder (a blank section that keeps the spacing even where one is missing) before the section at `position`."""
        if not 0 <= position <= len(self.order):
            return
        key = self.host.order_new_placeholder(position)
        if key is None:
            return
        self.push_history()
        self.order.insert(position, key)
        self.grid.selected = key
        self.grid.draw()
        self.host.order_status(f"Placeholder inserted before section {position + 2}: its blank frame is written when the stack is exported.")

    def reset(self):
        self.push_history()
        self.order = list(self.host.order_reset_keys())
        self.grid.selected = None
        self.grid.draw()

    def context(self, key, position, event):
        """Right-click on a thumbnail (it has been selected): a menu of what can be done with that section."""
        menu = tk.Menu(self.grid.canvas, tearoff=0)
        menu.add_command(label="Delete section", command=lambda: self.delete(position))
        menu.add_command(label="Insert placeholder before this section", command=lambda: self.insert_placeholder(position))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    # ---- the size of the thumbnails
    def on_size(self):
        self.set_size(self.var_size.get())

    def set_size(self, name):
        scale = SIZES[name]
        if scale == self.grid.scale:
            return
        self.var_size.set(name)
        self.grid.set_scale(scale)
        self.host.order_size_changed(scale)
