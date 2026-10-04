"""An image widget that zooms under the mouse and pans by drag.

Two things are worth knowing about how it draws.

THE PIXELS GO THROUGH TK'S PPM READER, not through PIL. This repo's dependencies
are numpy and OpenCV, and adding Pillow to a labelling tool that already has an
image library linked into it is not a trade worth making. `cv2.imencode(".ppm")`
produces a byte string Tk reads directly - measured here at well under a
millisecond for a full canvas, against ~20 ms to encode the same thing as PNG,
which is the difference between panning that tracks the mouse and panning that
lags behind it.

THE MARKS AND THE COURT ARE CANVAS ITEMS, not pixels burned into the image. Drawn
into the buffer they would scale with the zoom - a 2 px dot becomes a 40 px blob
at 20x, exactly when the labeller is zoomed in to place it precisely. As canvas
items they keep their size on screen and the image underneath stays the evidence
rather than the drawing.
"""
import tkinter as tk

import cv2
import numpy as np

#: Per wheel notch. A shade over 1.2 gives roughly a decade in a dozen notches,
#: which is the range between "the whole plate" and "individual pixels".
ZOOM_STEP = 1.25
ZOOM_MIN, ZOOM_MAX = 0.05, 40.0

BACKGROUND = "#1b1b1b"


class ZoomCanvas(tk.Canvas):
    """Shows a BGR ndarray. `on_click(x, y)` and `on_overlay()` are the hooks.

    `origin` is the image coordinate at the canvas's top-left corner and `scale`
    is screen pixels per image pixel, so the whole mapping is those two numbers
    and everything else is derived from them - which is what keeps a mark drawn
    at 20x on the pixel it was placed on at 0.4x.
    """

    def __init__(self, master, on_click=None, on_overlay=None, **kw):
        tk.Canvas.__init__(self, master, background=BACKGROUND,
                           highlightthickness=0, **kw)
        self.image = None
        self.scale_ = 1.0
        self.origin = np.zeros(2)
        self.on_click = on_click
        self.on_overlay = on_overlay
        self._photo = None
        self._item = None
        self._pan = None
        self._pending = None
        self._deferred_fit = None

        self.bind("<Configure>", self._on_configure)
        self.bind("<Button-1>", self._on_button1)
        for b in ("2", "3"):                     # middle or right drag pans
            self.bind("<Button-%s>" % b, self._pan_start)
            self.bind("<B%s-Motion>" % b, self._pan_move)
            self.bind("<ButtonRelease-%s>" % b, self._pan_end)
        self.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, ZOOM_STEP))
        self.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 1.0 / ZOOM_STEP))
        self.bind("<MouseWheel>",            # Windows/macOS deliver it this way
                  lambda e: self.zoom_at(e.x, e.y,
                                         ZOOM_STEP if e.delta > 0 else 1.0 / ZOOM_STEP))

    # -- the mapping -------------------------------------------------------
    def to_image(self, sx, sy):
        return self.origin + np.array([sx, sy], float) / self.scale_

    def to_screen(self, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        return (p - self.origin) * self.scale_

    # -- what is shown -----------------------------------------------------
    def set_image(self, bgr, keep_view=False):
        self.image = bgr
        if bgr is None:
            self.delete("all")
            self._item = self._photo = None
            return
        if not keep_view:
            self.fit()
        else:
            self.redraw()

    def fit(self):
        """The whole plate, centred, with a little air around it."""
        self._deferred_fit = None
        if self.image is None or not self.winfo_exists():
            return
        vw, vh = self.winfo_width(), self.winfo_height()
        if vw <= 1 or vh <= 1:
            # Asked before Tk has laid the widget out - which is every time the
            # first plate is opened, and where the answer would be ZOOM_MIN and
            # would stick, because <Configure> redraws but does not re-fit.
            self._deferred_fit = self.after(30, self.fit)
            return
        h, w = self.image.shape[:2]
        s = min(vw / float(w), vh / float(h)) * 0.98
        self.scale_ = float(np.clip(s, ZOOM_MIN, ZOOM_MAX))
        self.origin = np.array([w / 2.0 - vw / (2 * self.scale_),
                                h / 2.0 - vh / (2 * self.scale_)])
        self.redraw()

    def zoom_at(self, sx, sy, factor):
        """Zoom about a screen point, keeping the image pixel under it fixed."""
        if self.image is None:
            return
        s = float(np.clip(self.scale_ * factor, ZOOM_MIN, ZOOM_MAX))
        if s == self.scale_:
            return
        anchor = self.to_image(sx, sy)
        self.scale_ = s
        self.origin = anchor - np.array([sx, sy], float) / s
        self.redraw()

    # -- events ------------------------------------------------------------
    def _on_configure(self, _e):
        # Coalesced: a window drag fires this dozens of times a second and each
        # one would otherwise re-encode the whole canvas.
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(30, self._configured)

    def _configured(self):
        self._pending = None
        if self.winfo_exists():
            self.redraw()

    def cancel_pending(self):
        """Drop scheduled callbacks before the window goes away.

        Without this a plate opened and closed inside the same 30 ms leaves a
        deferred fit pointing at a destroyed widget, and Tk reports it on stderr
        as an invalid command name - noise on every quit, from work that no
        longer has anywhere to happen.
        """
        for attr in ("_pending", "_deferred_fit"):
            i = getattr(self, attr)
            if i is not None:
                try:
                    self.after_cancel(i)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _on_button1(self, e):
        self.focus_set()
        if self.image is not None and self.on_click is not None:
            self.on_click(*self.to_image(e.x, e.y))

    def _pan_start(self, e):
        self.focus_set()
        self._pan = (e.x, e.y, self.origin.copy())

    def _pan_move(self, e):
        if self._pan is None:
            return
        x0, y0, o0 = self._pan
        self.origin = o0 - np.array([e.x - x0, e.y - y0], float) / self.scale_
        self.redraw()

    def _pan_end(self, _e):
        self._pan = None

    # -- drawing -----------------------------------------------------------
    def redraw(self):
        if self.image is None:
            return
        vw, vh = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        buf = self._render(vw, vh)
        ok, ppm = cv2.imencode(".ppm", buf)
        if not ok:
            return
        # The reference is held on the instance because Tk does not own the
        # image and a garbage-collected PhotoImage draws as an empty rectangle.
        self._photo = tk.PhotoImage(data=ppm.tobytes())
        if self._item is None:
            self._item = self.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.itemconfigure(self._item, image=self._photo)
        self.tag_lower(self._item)
        self.delete("ov")
        if self.on_overlay is not None:
            self.on_overlay()

    def _render(self, vw, vh):
        """The visible part of the image, resampled onto a canvas-sized buffer.

        Cropped before it is resized, so the cost is set by the canvas rather
        than by the zoom: at 20x only the few hundred pixels actually on screen
        are touched. INTER_NEAREST going up, because at that magnification the
        labeller is looking for the pixel where the paint stops and interpolation
        invents a soft edge between it and its neighbour; INTER_AREA coming down,
        because that is the one that does not alias a thin white line out of
        existence.
        """
        h, w = self.image.shape[:2]
        s = self.scale_
        out = np.zeros((vh, vw, 3), np.uint8)
        out[:] = (27, 27, 27)
        x0 = int(np.floor(max(0.0, self.origin[0])))
        y0 = int(np.floor(max(0.0, self.origin[1])))
        x1 = int(np.ceil(min(float(w), self.origin[0] + vw / s)))
        y1 = int(np.ceil(min(float(h), self.origin[1] + vh / s)))
        if x1 <= x0 or y1 <= y0:
            return out
        crop = self.image[y0:y1, x0:x1]
        dw, dh = max(1, int(round((x1 - x0) * s))), max(1, int(round((y1 - y0) * s)))
        r = cv2.resize(crop, (dw, dh),
                       interpolation=cv2.INTER_NEAREST if s > 1.0 else cv2.INTER_AREA)
        ox, oy = int(round((x0 - self.origin[0]) * s)), int(round((y0 - self.origin[1]) * s))
        # Clip against the buffer: rounding can put the paste one pixel over an
        # edge, and numpy would silently truncate the wrong side of it.
        sx0, sy0 = max(0, -ox), max(0, -oy)
        dx0, dy0 = max(0, ox), max(0, oy)
        cw = min(dw - sx0, vw - dx0)
        ch = min(dh - sy0, vh - dy0)
        if cw > 0 and ch > 0:
            out[dy0:dy0 + ch, dx0:dx0 + cw] = r[sy0:sy0 + ch, sx0:sx0 + cw]
        return out

    # -- overlay helpers, in IMAGE coordinates ------------------------------
    def draw_polyline(self, pts, colour, width=2, dash=None):
        p = self.to_screen(pts)
        if len(p) < 2:
            return
        flat = [float(v) for xy in p for v in xy]
        self.create_line(*flat, fill=colour, width=width, tags="ov",
                         dash=dash, smooth=False)

    def draw_dot(self, xy, colour, r=4, outline="#000000", width=1):
        x, y = self.to_screen([xy])[0]
        self.create_oval(x - r, y - r, x + r, y + r, fill=colour,
                         outline=outline, width=width, tags="ov")

    def draw_x(self, xy, colour, gap=3.0, gap_min=6.0, arm=10.0, width=2,
               halo="#000000"):
        """Four arms pointing at an image point, with the point itself left bare.

        For the places a detected line STOPS - see `padelcourt.draw.mark_x` for
        why those get an X rather than a plus, and why the centre is open. Here
        the sub-pixel part comes for free: `to_screen` returns floats and Tk's
        canvas takes them, so the arms keep aiming at the position the walk
        measured however far in the labeller is zoomed.

        THE GAP IS IN IMAGE PIXELS AND THE ARMS ARE IN SCREEN PIXELS, which is
        the one thing here worth explaining. Zooming in on an endpoint is done to
        look at the paint immediately AROUND it - two or three pixels either way,
        whether the stripe carries on past the mark or stops at it - and a gap
        fixed on screen closes over exactly that: at 20x, six screen pixels is a
        third of an image pixel, so the arms land on the stop they are pointing
        at. Held at `gap` image pixels it opens as the zoom goes in and the
        evidence stays visible. The arms have the opposite job - being findable -
        so they keep their length on screen, and `gap_min` is the floor that
        stops the whole mark collapsing into itself with the plate zoomed out.
        """
        x, y = self.to_screen([xy])[0]
        g = max(float(gap_min), float(gap) * self.scale_)
        r = g + float(arm)
        for col, w in ((halo, width + 2), (colour, width)) if halo else ((colour, width),):
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                self.create_line(x + sx * g, y + sy * g, x + sx * r, y + sy * r,
                                 fill=col, width=w, tags="ov")

    def draw_text(self, xy, text, colour, dx=8, dy=-8, anchor="w"):
        x, y = self.to_screen([xy])[0]
        self.create_text(x + dx, y + dy, text=text, fill=colour, anchor=anchor,
                         tags="ov", font=("TkDefaultFont", 9, "bold"))
