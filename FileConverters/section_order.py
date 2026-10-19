"""A grid of thumbnails for putting sections in order by dragging them: used by step 1 (the third tab of the section loader) and by the
alignment tool (its "Thumbnails..." window). The widget only draws and reorders; what the thumbnails show, and what a double-click
does, is up to the program that uses it.

    grid = OrderGrid(parent, caption=lambda key, i: ("left text", "right text"), on_double=lambda key, i: ...)
    grid.frame.pack(fill=tk.BOTH, expand=True)
    grid.order = [key, ...]                      # the keys (anything hashable) in the order shown
    grid.thumbs[key] = ImageTk.PhotoImage(...)   # any that are ready; call grid.draw() after changing them
"""

import tkinter as tk
from tkinter import ttk

THUMB_W, THUMB_H, CAPTION_H, CELL_PAD = 190, 150, 20, 10
ACCENT = "#1f5fa8"


class OrderGrid:
    def __init__(self, parent, caption, on_double=None, enabled=None, after_draw=None, accent=ACCENT):
        """caption(key, position) -> (text at the left, text at the right) under a thumbnail; on_double(key, position) is called on a
        double-click; enabled() -> False stops it drawing (for a page that is not shown); after_draw() is called after each drawing."""
        self.caption, self.on_double, self.enabled, self.after_draw, self.accent = caption, on_double, enabled, after_draw, accent
        self.order = []                    # the keys, in order
        self.thumbs = {}                   # key -> PhotoImage
        self.drag = None
        self.frame = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.frame, bg="white", highlightthickness=0)
        sb = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.move)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<Double-Button-1>", self.double)

    # ------------------------------------------------------------------ geometry

    def cell_geometry(self):
        w = max(self.canvas.winfo_width(), 300)
        cw, chh = THUMB_W + CELL_PAD, THUMB_H + CAPTION_H + CELL_PAD
        cols = max(1, (w - CELL_PAD) // cw)
        return cols, cw, chh

    def cell_origin(self, i):
        cols, cw, chh = self.cell_geometry()
        return CELL_PAD + (i % cols) * cw, CELL_PAD + (i // cols) * chh

    def index_at(self, x, y):
        """Position of the thumbnail at a point of the window (y as the window gives it, not scrolled), or None."""
        y = self.canvas.canvasy(y)
        cols, cw, chh = self.cell_geometry()
        col, row = (x - CELL_PAD) // cw, (y - CELL_PAD) // chh
        i = int(row * cols + col)
        return i if 0 <= col < cols and 0 <= i < len(self.order) else None

    def slot_at(self, x, y):
        """Insertion position (0 .. len) for a point of the window."""
        cols, cw, chh = self.cell_geometry()
        y = self.canvas.canvasy(y)
        col = int(min(max((x - CELL_PAD) // cw, 0), cols - 1))
        row = int(max((y - CELL_PAD) // chh, 0))
        i = row * cols + col
        cx = CELL_PAD + col * cw + THUMB_W / 2
        if x > cx:
            i += 1
        return min(i, len(self.order))

    # ------------------------------------------------------------------ drawing

    def draw(self):
        c = self.canvas
        c.delete("all")
        if self.enabled is not None and not self.enabled():
            return
        cols, cw, chh = self.cell_geometry()
        rows = (len(self.order) + cols - 1) // cols
        c.config(scrollregion=(0, 0, 10, CELL_PAD + rows * chh))
        for i, key in enumerate(self.order):
            if self.drag and self.drag["moved"] and i == self.drag["index"]:
                continue
            x, y = self.cell_origin(i)
            c.create_rectangle(x, y, x + THUMB_W, y + THUMB_H, outline="#c0c0c0", fill="#101010")
            ph = self.thumbs.get(key)
            if ph:
                c.create_image(x + THUMB_W / 2, y + THUMB_H / 2, image=ph)
            left, right = self.caption(key, i)
            c.create_text(x + 2, y + THUMB_H + 2, anchor="nw", fill="black", font=("Segoe UI", 9), text=left)
            c.create_text(x + THUMB_W - 2, y + THUMB_H + 2, anchor="ne", fill=self.accent, font=("Segoe UI", 9, "bold"), text=right)
        if self.drag and self.drag["moved"]:
            g = self.drag["slot"]
            x, y = self.cell_origin(g)
            if g == len(self.order) and g % cols == 0 and g:
                x, y = self.cell_origin(g - 1)
                x += cw
            c.create_rectangle(x - 6, y, x - 3, y + THUMB_H, fill=self.accent, outline="")
            ph = self.thumbs.get(self.order[self.drag["index"]])
            if ph:
                c.create_image(self.drag["x"], self.drag["y"], image=ph)
        if self.after_draw:
            self.after_draw()

    # ------------------------------------------------------------------ mouse

    def press(self, e):
        i = self.index_at(e.x, e.y)
        if i is not None:
            y = self.canvas.canvasy(e.y)
            self.drag = {"index": i, "x0": e.x, "y0": y, "x": e.x, "y": y, "wy": e.y, "moved": False, "slot": i, "job": None}

    def move(self, e):
        d = self.drag
        if not d:
            return
        y = self.canvas.canvasy(e.y)
        if not d["moved"] and abs(e.x - d["x0"]) + abs(y - d["y0"]) < 6:
            return
        d["moved"], d["x"], d["y"], d["wy"] = True, e.x, y, e.y
        d["slot"] = self.slot_at(e.x, e.y)
        self.draw()
        if d["job"] is None and not 0 <= e.y <= self.canvas.winfo_height():
            d["job"] = self.canvas.after(60, self.autoscroll)

    def autoscroll(self):
        """While a thumbnail is held above or below the window, the grid keeps scrolling that way."""
        d = self.drag
        if not d:
            return
        d["job"] = None
        h = self.canvas.winfo_height()
        if 0 <= d["wy"] <= h:
            return
        self.canvas.yview_scroll(-1 if d["wy"] < 0 else 1, "units")
        d["y"] = self.canvas.canvasy(d["wy"])
        d["slot"] = self.slot_at(d["x"], d["wy"])
        self.draw()
        d["job"] = self.canvas.after(60, self.autoscroll)

    def release(self, e):
        d, self.drag = self.drag, None
        if d and d["moved"]:
            item = self.order.pop(d["index"])
            slot = d["slot"] - (1 if d["slot"] > d["index"] else 0)
            self.order.insert(slot, item)
        self.draw()

    def double(self, e):
        i = self.index_at(e.x, e.y)
        if i is None:
            return
        self.drag = None
        if self.on_double:
            self.on_double(self.order[i], i)

    # ------------------------------------------------------------------ whole-order changes

    def reverse(self):
        self.order.reverse()
        self.draw()
