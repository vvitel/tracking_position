"""The labelling window.

The loop it is built around: pick a folder, let the walks run over it in the
background while you work, open a plate, mark left / right / far, watch the
quadratic and then the whole court appear over the picture, save, and be on the
next plate that still needs one.

Three decisions in here are worth stating.

THE WALKS RUN IN PROCESSES, THE FIT RUNS IN A THREAD. A walk is ~7 s of OpenCV
per plate and there may be a hundred of them, so they go to a process pool and
the tree updates as each lands. The fit is a couple of seconds of
Levenberg-Marquardt and there is only ever one in flight, so it goes to a thread
with a sequence number - marks change faster than the fit converges, and a result
that arrives for marks that no longer exist is discarded rather than drawn.

PARTIAL WORK IS SAVED, AND STILL COUNTS AS UNFINISHED. Leaving a plate half
marked writes `_court-sides.json` with `complete: false`, so nothing a person did
is ever lost to a misclick on the list - and "the next plate that needs marking"
means the next one without a COMPLETE file, so a half-done plate comes back round
instead of being skipped for having a file.

NOTHING IS SAVED THAT CANNOT BE REDONE FROM THE MARKS. `_court-sides.json` holds
the clicks; `_court-truth.json` holds the court fitted from them. Reopening a
plate re-fits the curves from the clicks rather than reading back the ones it
wrote, so there is exactly one definition of what a marked side means.
"""
import multiprocessing
import os
import queue
import threading
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor, as_completed
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from padelcourt import court as CM

from . import fit as F
from . import sides as SD
from . import store, walks
from .canvas import ZoomCanvas

# ------------------------------------------------------------------ colours --
SIDE_COLOUR = {"left": "#ff5b5b", "right": "#4ade80", "far": "#60a5fa"}
ERASE_COLOUR = "#ffffff"
PERIMETER, SERVICE, NET, KEYPOINT = "#22ff88", "#ffd21e", "#ff8a3d", "#ffffff"
WALK_COLOUR = "#8b8b8b"

#: Where the two service lines' paint stopped, as the walk placed it: marked
#: with an X whose arms converge on the point without covering it, in
#: `sides.CORNER_COLOUR`'s white so that the near pair reads as the same thing
#: whether it arrives with the walk or with a courtside detection file - it IS
#: the same point.
#:
#: White with a dark halo rather than a colour of its own because the two things
#: it must not be mistaken for are both coloured: the marking clicks are
#: red/green/blue and the fitted court's own service lines are `SERVICE` yellow,
#: and an end drawn in either would read as part of what it is there to be
#: checked against.
#:
#: THE GAP IS IN IMAGE PIXELS AND THE ARMS ARE IN SCREEN PIXELS - see
#: `ZoomCanvas.draw_x` for why the two are measured in different units.
#:
#: The lengths are off the screenshots rather than guessed: at the 5 screen
#: pixels of arm this started with, thinner than its own halo, a plate at fit
#: zoom shows four grey smudges - and fit zoom is where the mark is used to
#: decide whether an end is worth zooming in on at all. 3 image pixels of gap is
#: about the half-width of the paint at the near corners, so at working zoom the
#: arms sit just off the stripe.
END_COLOUR, END_GAP, END_ARM, END_W = "#ffffff", 3.0, 10.0, 2

#: How near a mark the eraser has to be clicked, in SCREEN pixels - so the
#: tolerance is the same however far in the labeller is zoomed, which is the
#: point of zooming in to erase one of two marks that are close together.
ERASE_RADIUS = 14.0

#: The two things there are to mark. Which SIDELINE a click belongs to is read
#: off where it landed rather than asked for - see `App.divider_x` - because the
#: two are on opposite halves of the picture and no labeller has ever meant the
#: other one. `keys` is which entries of `marks` the group covers.
GROUPS = {
    "side": {"keys": ("left", "right"), "label": "Sidelines (X = 0 / 10)"},
    "far":  {"keys": ("far", ),         "label": "Far side (Y = 0)"},
}
GROUP_ORDER = ("side", "far")
GROUP_OF = {"left": "side", "right": "side", "far": "far"}

#: The line the sideline split is read against, drawn while marking so the
#: deduction is visible rather than implied.
DIVIDER_COLOUR = "#e879f9"

#: The court, as the segments `draw.draw_overlay` draws, in court metres.
COURT_SEGMENTS = (
    ((0, 0), (CM.WIDTH, 0), PERIMETER, 2),
    ((0, CM.LENGTH), (CM.WIDTH, CM.LENGTH), PERIMETER, 2),
    ((0, 0), (0, CM.LENGTH), PERIMETER, 2),
    ((CM.WIDTH, 0), (CM.WIDTH, CM.LENGTH), PERIMETER, 2),
    ((0, CM.FAR_SERVICE_Y), (CM.WIDTH, CM.FAR_SERVICE_Y), SERVICE, 1),
    ((0, CM.NEAR_SERVICE_Y), (CM.WIDTH, CM.NEAR_SERVICE_Y), SERVICE, 1),
    ((CM.CENTRE_X, CM.FAR_SERVICE_Y), (CM.CENTRE_X, CM.NEAR_SERVICE_Y), SERVICE, 1),
    ((0, CM.NET_Y), (CM.WIDTH, CM.NET_Y), NET, 1),
)

MARK = {store.OK: "OK", store.FAILED: "--", store.MISSING: ""}

#: Open a combobox's list ABOVE the box instead of below it.
#:
#: ttk hangs the list downwards and only flips it up when it would run off the
#: bottom of the SCREEN, which is not the same thing as running off the bottom of
#: the DESKTOP: with the window sat low, or a panel along the bottom edge, the
#: list is placed exactly where it cannot be reached and the last entries of it
#: simply are not there - which is how `chord_lines` and `chord` came to look
#: like they were missing from the courtside dropdown when they were only
#: underneath the taskbar.
#:
#: When the list will not fit above either, ttk's own placement is called and
#: nothing changes - a list off the TOP of the screen is no better than one off
#: the bottom. The x and the width are worked out here the way ttk works them
#: out, `-postoffset` included, because the geometry the stock proc REQUESTS
#: cannot be read back off the window: `wm geometry` on the popdown answers with
#: the 1x1 it is still sitting at until it is mapped, so a wrapper that placed
#: the list by re-reading it shrank the list to a pixel.
POPDOWN_UPWARD = r"""
if {[info commands ttk::combobox::PlacePopdownBelow] eq {}} {
    rename ttk::combobox::PlacePopdown ttk::combobox::PlacePopdownBelow
}
proc ttk::combobox::PlacePopdown {cb popdown} {
    set H [winfo reqheight $popdown]
    if {[winfo rooty $cb] - $H < 0} {
        ttk::combobox::PlacePopdownBelow $cb $popdown
        return
    }
    set x [winfo rootx $cb]
    set y [winfo rooty $cb]
    set w [winfo width $cb]
    set h [winfo height $cb]
    set style [$cb cget -style]
    if {$style eq {}} { set style TCombobox }
    set postoffset [ttk::style lookup $style -postoffset {} {0 0 0 0}]
    foreach var {x y w h} delta $postoffset {
        incr $var $delta
    }
    wm geometry $popdown ${w}x${H}+${x}+[expr {$y - $H}]
}
"""


def popdown_upward(widget):
    """Point every combobox list in this interpreter upwards - see above.

    ttk's placement is one procedure shared by all comboboxes, so this is per
    INTERPRETER and not per widget. That is not worth working around while the
    tool has one dropdown, and the rule is a sensible one for any it grows.
    """
    widget.tk.eval(POPDOWN_UPWARD)


def _try_fit(shape, marks, lines, anchors):
    """(fit, error) - one of the two is always None. Runs off the Tk thread."""
    try:
        return F.fit_court(shape, marks, lines, anchors), None
    except F.FitError as e:
        return None, str(e)
    except BaseException as e:
        return None, "%s: %s" % (e.__class__.__name__, str(e)[:160])


class App(ttk.Frame):

    def __init__(self, master, folder=None, jobs=None):
        ttk.Frame.__init__(self, master, padding=0)
        self.master = master
        self.jobs = jobs or max(1, (os.cpu_count() or 4) - 2)
        self.folder = None
        self.plates = []
        self.index = None
        self.image = None
        self.marks = {k: [] for k in F.SIDE_ORDER}
        self.walk_lines, self.walk_anchors, self.walk_note = [], [], None
        self.walk_ends = {}              # {near_l, near_r, far_l, far_r} in px
        self.divider = None              # the walked centre line, as x(y)
        self.sides_doc, self.sides_note = None, None
        self.dirty = False

        self._fit = None                 # (cam, report) for the marks as they are
        self._fit_error = None
        self._fit_seq = 0                # bumped on every change to the marks
        self._fit_running = False
        self._selecting = False          # guard: we are moving the tree ourselves
        self._closing = False

        self._pool = None
        self._futures = []
        self._queue = queue.Queue()
        self._fit_queue = queue.Queue()
        self._gen_total = self._gen_done = 0

        self._build()
        self.pack(fill="both", expand=True)
        if folder:
            self.open_folder(folder)

    # ------------------------------------------------------------- widgets --
    def _build(self):
        self.mode = tk.StringVar(value="side")
        self.show_court = tk.BooleanVar(value=True)
        self.show_walks = tk.BooleanVar(value=False)
        self.sides_choice = tk.StringVar(value=SD.OFF)
        self.status = tk.StringVar(value="Pick a folder of plates to begin.")
        self.folder_text = tk.StringVar(value="(no folder)")
        self.progress_text = tk.StringVar(value="")
        self.filter_text = tk.StringVar(value="")
        self.filter_text.trace_add("write", lambda *_: self.reload_tree())

        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="Folder...", command=self.choose_folder).pack(side="left")
        ttk.Label(bar, textvariable=self.folder_text, width=52,
                  anchor="w").pack(side="left", padx=(8, 16))
        self.gen_btn = ttk.Button(bar, text="Generate walks", command=self.generate_walks,
                                  state="disabled")
        self.gen_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop_walks, state="disabled")
        self.stop_btn.pack(side="left", padx=(4, 8))
        self.progress = ttk.Progressbar(bar, mode="determinate", length=180)
        self.progress.pack(side="left")
        ttk.Label(bar, textvariable=self.progress_text, width=22,
                  anchor="w").pack(side="left", padx=8)

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(side="top", fill="both", expand=True)

        left = ttk.Frame(pane)
        filt = ttk.Frame(left, padding=(4, 4))
        filt.pack(side="top", fill="x")
        ttk.Label(filt, text="Filter").pack(side="left")
        entry = ttk.Entry(filt, textvariable=self.filter_text)
        entry.pack(side="left", fill="x", expand=True, padx=4)
        # Escape clears rather than a button: the entry is where the hand
        # already is, and a cleared filter is the common next thing to want.
        entry.bind("<Escape>", lambda _e: self.filter_text.set(""))
        ttk.Button(filt, text="x", width=2,
                   command=lambda: self.filter_text.set("")).pack(side="left")

        self.tree = ttk.Treeview(left, columns=("walk", "sides"), show="tree headings",
                                 selectmode="browse")
        self.tree.heading("#0", text="plate")
        self.tree.heading("walk", text="walk")
        self.tree.heading("sides", text="sides")
        self.tree.column("#0", width=280, stretch=True)
        self.tree.column("walk", width=52, anchor="center", stretch=False)
        self.tree.column("sides", width=52, anchor="center", stretch=False)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.tag_configure("ok", foreground="#15803d")
        self.tree.tag_configure("failed", foreground="#b91c1c")
        self.tree.tag_configure("missing", foreground="#6b7280")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        pane.add(left, weight=0)

        right = ttk.Frame(pane)
        self.canvas = ZoomCanvas(right, on_click=self.on_click,
                                 on_overlay=self.draw_overlay, width=1000, height=640)
        self.canvas.pack(side="top", fill="both", expand=True)

        ctl = ttk.Frame(right, padding=(8, 6))
        ctl.pack(side="top", fill="x")
        # Navigation is packed FIRST, against the right edge, so it claims its
        # width before the options do. Laid out the other way round the option
        # grid takes what it wants and the save button falls off a narrow window
        # - which is the one button that must never be unreachable.
        nav = ttk.Frame(ctl)
        nav.pack(side="right", padx=(16, 0))
        ttk.Button(nav, text="< Prev", command=self.prev_plate, width=7).pack(side="left")
        ttk.Button(nav, text="Next >", command=self.next_plate, width=7).pack(side="left", padx=4)
        self.save_btn = ttk.Button(nav, text="Save + next (S)", command=self.save_and_next,
                                   state="disabled")
        self.save_btn.pack(side="left", padx=(8, 0))

        opts = ttk.Frame(ctl)
        opts.pack(side="left", fill="x", expand=True)
        self.count_labels = {}
        col = 0
        for i, group in enumerate(GROUP_ORDER):
            ttk.Radiobutton(opts, text="%d  %s" % (i + 1, GROUPS[group]["label"]),
                            value=group, variable=self.mode,
                            command=self.refresh).grid(row=0, column=col, sticky="w",
                                                       padx=(0, 6))
            col += 1
            for key in GROUPS[group]["keys"]:
                lab = ttk.Label(opts, text="0/%d" % F.MIN_MARKS, width=8)
                lab.grid(row=0, column=col, sticky="w", padx=(0, 8))
                self.count_labels[key] = lab
                col += 1
            col += 1                             # a gap before the next group
        ttk.Radiobutton(opts, text="E  Eraser", value="erase", variable=self.mode,
                        command=self.refresh).grid(row=0, column=col, sticky="w")

        self.court_chk = ttk.Checkbutton(opts, text="Show court (C)", variable=self.show_court,
                                         command=self.on_toggle_court, state="disabled")
        self.court_chk.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(opts, text="Show walk (W)", variable=self.show_walks,
                        command=self.refresh).grid(row=1, column=2, columnspan=2,
                                                   sticky="w", pady=(6, 0))
        self.clear_btn = ttk.Button(opts, text="Clear", command=self.clear_group)
        self.clear_btn.grid(row=1, column=4, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(opts, text="Clear all", command=self.clear_all).grid(
            row=1, column=6, sticky="w", padx=6, pady=(6, 0))

        # Read-only: the list is fixed and a typed value would mean nothing.
        # Disabled until a plate with a detection file beside it is open, so the
        # control says whether the file exists rather than silently drawing
        # nothing when it does not.
        ttk.Label(opts, text="Courtsides (D)").grid(row=1, column=7, sticky="e",
                                                    padx=(12, 4), pady=(6, 0))
        self.sides_box = ttk.Combobox(opts, textvariable=self.sides_choice,
                                      values=list(SD.CHOICES), state="disabled",
                                      width=len(max(SD.CHOICES, key=len)))
        self.sides_box.grid(row=1, column=8, sticky="w", pady=(6, 0))
        popdown_upward(self.sides_box)
        self.sides_box.bind("<<ComboboxSelected>>",
                            lambda _e: (self.canvas.focus_set(), self.refresh(refit=False)))

        ttk.Label(right, textvariable=self.status, anchor="w",
                  padding=(10, 4)).pack(side="top", fill="x")
        pane.add(right, weight=1)

        for seq, fn in (("<Key-1>", lambda: self.set_mode("side")),
                        ("<Key-2>", lambda: self.set_mode("far")),
                        ("<Key-e>", lambda: self.set_mode("erase")),
                        ("<Key-s>", self.save_and_next),
                        ("<Key-c>", lambda: self.toggle(self.show_court, self.on_toggle_court)),
                        ("<Key-w>", lambda: self.toggle(self.show_walks, self.refresh)),
                        ("<Key-d>", self.cycle_sides),
                        ("<Key-f>", self.canvas.fit),
                        ("<Key-n>", self.next_plate),
                        ("<Key-p>", self.prev_plate)):
            self.master.bind(seq, self._hotkey(fn))
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def _hotkey(self, fn):
        """Run `fn` unless the keystroke was meant for the filter box.

        The shortcuts are bound on the toplevel so they work wherever the mouse
        is, which means they also fire while typing in an Entry - and one of them
        is S for save. Letters belong to whoever has the keyboard focus.
        """
        def handler(_e):
            w = self.master.focus_get()
            if isinstance(w, (ttk.Entry, tk.Entry, ttk.Combobox)):
                return
            fn()
        return handler

    # -------------------------------------------------------------- folder --
    def choose_folder(self):
        d = filedialog.askdirectory(title="Folder of plates",
                                    initialdir=self.folder or os.getcwd())
        if d:
            self.open_folder(d)

    def open_folder(self, folder):
        self.autosave()
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            messagebox.showerror("Not a folder", folder)
            return
        self.folder = folder
        self.plates = store.find_plates(folder)
        self.index = None
        self.image = None
        self.canvas.set_image(None)
        self.reload_tree()                       # also writes the folder label
        self.gen_btn.configure(state="normal" if self.plates else "disabled")
        n = sum(1 for p in self.plates if store.lines_state(p) == store.MISSING)
        self.status.set("%d plates, %d still need their walk. %s"
                        % (len(self.plates), n,
                           "Press Generate walks." if n else "Pick one from the list."))
        if self.plates:
            first = self.next_unlabelled(-1)
            self.open_index(0 if first is None else first)

    def matches(self, path):
        """Case-insensitive filter over the filename.

        Every whitespace-separated term has to appear, in any order. These names
        are a MAC address, a UUID and a timestamp run together, so the useful
        query is almost always two fragments from different parts of one -
        `5f-c6 04-01` - and a single substring cannot express that.
        """
        terms = self.filter_text.get().lower().split()
        if not terms:
            return True
        name = os.path.basename(path).lower()
        return all(t in name for t in terms)

    def visible(self):
        """Indices of the plates the filter lets through, in list order."""
        return [i for i, p in enumerate(self.plates) if self.matches(p)]

    def reload_tree(self):
        self.tree.delete(*self.tree.get_children())
        shown = self.visible()
        for i in shown:
            self.tree.insert("", "end", iid=str(i),
                             text=os.path.basename(self.plates[i]), values=("", ""))
            self.update_row(i)
        if self.index is not None:
            self._select_row(self.index)
        if self.folder:
            self.folder_text.set("%s   (%d plates%s)"
                                 % (self.folder, len(self.plates),
                                    ", %d shown" % len(shown)
                                    if len(shown) != len(self.plates) else ""))

    def update_row(self, i):
        if not self.tree.exists(str(i)):
            return
        st = store.state_of(self.plates[i])
        sides = st["sides"]
        if i == self.index:                      # what is on screen, not what is on disk
            n = min(len(self.marks[k]) for k in F.SIDE_ORDER)
            sides = store.OK if n >= F.MIN_MARKS else (
                store.FAILED if any(self.marks[k] for k in F.SIDE_ORDER) else store.MISSING)
        self.tree.item(str(i), values=(MARK[st["lines"]], MARK[sides]),
                       tags=(st["lines"],))

    # --------------------------------------------------------------- walks --
    def generate_walks(self):
        if self._pool is not None:
            return
        todo = [(i, p) for i, p in enumerate(self.plates)
                if store.lines_state(p) == store.MISSING]
        if not todo:
            self.status.set("Every plate already has its %s." % store.LINES)
            return
        self._gen_total, self._gen_done = len(todo), 0
        self.progress.configure(maximum=len(todo), value=0)
        self.progress_text.set("0 / %d" % len(todo))
        self.gen_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.set("Walking %d plates on %d workers..." % (len(todo), self.jobs))
        # Spawn rather than fork: the parent holds an X connection and a Tk
        # interpreter, and forking those into a worker that will never use them
        # is a class of hang not worth the startup it saves.
        self._pool = ProcessPoolExecutor(max_workers=self.jobs,
                                         mp_context=multiprocessing.get_context("spawn"))
        self._futures = {self._pool.submit(walks.generate, p): i for i, p in todo}
        threading.Thread(target=self._drain, args=(dict(self._futures),),
                         daemon=True).start()
        self.after(120, self._poll)

    def _drain(self, futures):
        """Worker thread: results onto the queue, never onto a widget."""
        for fut in as_completed(list(futures)):
            try:
                res = fut.result()
            except BaseException as e:
                res = {"plate": None, "ok": False, "error": str(e)[:200]}
            self._queue.put((futures[fut], res))
        self._queue.put(None)

    def _poll(self):
        if self._closing:
            return
        done = False
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                done = True
                continue
            i, res = item
            self._gen_done += 1
            self.update_row(i)
            if i == self.index:                  # the plate on screen just got its walk
                self.load_walk()
                self.invalidate_fit()
            self.progress.configure(value=self._gen_done)
            self.progress_text.set("%d / %d" % (self._gen_done, self._gen_total))
            if not res.get("ok"):
                self.status.set("walk failed on %s: %s"
                                % (os.path.basename(self.plates[i]), res.get("error")))
        if done:
            self._finish_walks()
        else:
            self.after(120, self._poll)

    def _finish_walks(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self._futures = {}
        self.gen_btn.configure(state="normal" if self.plates else "disabled")
        self.stop_btn.configure(state="disabled")
        bad = sum(1 for p in self.plates if store.lines_state(p) == store.FAILED)
        self.status.set("Walks done: %d of %d written%s."
                        % (self._gen_done, self._gen_total,
                           ", %d plates the walk could not read" % bad if bad else ""))

    def stop_walks(self):
        if self._pool is None:
            return
        self.status.set("Stopping - letting the running walks finish...")
        for fut in self._futures:
            fut.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._pool = None
        self.stop_btn.configure(state="disabled")
        self.gen_btn.configure(state="normal")

    # --------------------------------------------------------------- plates --
    def next_unlabelled(self, after):
        """The next plate the filter shows whose sides are not COMPLETE.

        Filtered, because narrowing the list to one venue and then being sent to
        a plate from another would make the filter useless for the only thing it
        is for - working through a subset.
        """
        shown = self.visible() or list(range(len(self.plates)))
        later = [i for i in shown if i > after] + [i for i in shown if i <= after]
        for i in later:
            if store.sides_state(self.plates[i]) != store.OK:
                return i
        return None

    def _step(self, delta):
        """The next or previous plate the filter shows, wrapping."""
        shown = self.visible()
        if not shown:
            return None
        if self.index in shown:
            return shown[(shown.index(self.index) + delta) % len(shown)]
        ahead = [i for i in shown if i > (self.index or -1)]
        return ahead[0] if ahead and delta > 0 else shown[-1 if delta < 0 else 0]

    def open_index(self, i, keep_view=False):
        if not self.plates:
            return
        self.autosave()
        i = max(0, min(i, len(self.plates) - 1))
        path = self.plates[i]
        img = cv2.imread(path)
        if img is None:
            self.status.set("cannot read %s" % os.path.basename(path))
            return
        self.index, self.image = i, img
        self.marks = {k: [] for k in F.SIDE_ORDER}
        doc = store.read_json(store.sidecar(path, store.SIDES))
        if doc:
            for k in F.SIDE_ORDER:
                self.marks[k] = [list(map(float, p))
                                 for p in doc.get("sides", {}).get(k, {}).get("points", [])]
        self.dirty = False
        self.load_walk()
        self.load_sides()
        self.invalidate_fit()
        self.canvas.set_image(img, keep_view=keep_view)
        self._select_row(i)
        self.status.set("%s   %dx%d%s" % (os.path.basename(path), img.shape[1],
                                          img.shape[0],
                                          "   " + self.walk_note if self.walk_note else ""))
        self.refresh()

    def cycle_sides(self):
        """D steps through the dropdown - off, all, then one reader at a time.

        The comparison is done by flicking between them over one bit of paint,
        so it wants a key rather than a trip to the menu.
        """
        if str(self.sides_box["state"]) == "disabled":
            return
        i = SD.CHOICES.index(self.sides_choice.get()) if \
            self.sides_choice.get() in SD.CHOICES else 0
        self.sides_choice.set(SD.CHOICES[(i + 1) % len(SD.CHOICES)])
        self.refresh(refit=False)

    def load_sides(self):
        """The courtside detections beside this plate, if anyone wrote them.

        The CHOICE is deliberately kept across plates: the whole use of this is
        looking at one reader over a run of plates, and a dropdown that reset to
        `off` on every Next would have to be set again every time.
        """
        self.sides_doc, self.sides_note = SD.load(self.plates[self.index])
        self.sides_box.configure(state="disabled" if self.sides_doc is None
                                 else "readonly")

    def load_walk(self):
        (self.walk_lines, self.walk_anchors, self.walk_ends,
         self.walk_note) = walks.load(self.plates[self.index])
        # The court's own midline, for telling the two sidelines apart. The
        # walked centre line IS world X = 5, so a click is on the left sideline
        # exactly when it falls left of it - which beats the middle of the FRAME,
        # because nothing guarantees the camera is centred on the court and a
        # wide-angle lens bows the midline by tens of pixels across the height.
        self.divider = None
        for L in self.walk_lines:
            if L["name"] == "centre_line" and len(L["pts"]) >= F.MIN_CURVE:
                self.divider = F.fit_curve(L["pts"], True)      # x as a function of y
                break

    def divider_x(self, y):
        """Where the court's middle is at this image row.

        Clamped to the span the centre line was actually walked over rather than
        extrapolated past it: the sidelines run well beyond both ends of the
        paint, and a quadratic evaluated 400 px outside its data can bend
        anywhere. Held at the endpoint it stays between the two sidelines, which
        is all this has to be right about.
        """
        if self.divider is None:
            return self.image.shape[1] / 2.0     # no walk: the middle of the frame
        c, lo, hi = self.divider
        return float(np.polyval(c, min(max(float(y), lo), hi)))

    def side_at(self, x, y):
        return "left" if x < self.divider_x(y) else "right"

    def _select_row(self, i):
        # The open plate may be filtered out of the list, which is not a reason
        # to close it - it just has no row to highlight.
        if not self.tree.exists(str(i)):
            return
        self._selecting = True
        try:
            self.tree.selection_set(str(i))
            self.tree.see(str(i))
        finally:
            self._selecting = False

    def _on_select(self, _e):
        if self._selecting:
            return
        sel = self.tree.selection()
        if sel and int(sel[0]) != self.index:
            self.open_index(int(sel[0]))

    def next_plate(self):
        i = self._step(1)
        if i is not None:
            self.open_index(i)

    def prev_plate(self):
        i = self._step(-1)
        if i is not None:
            self.open_index(i)

    # ---------------------------------------------------------------- marks --
    def set_mode(self, m):
        self.mode.set(m)
        self.refresh()

    def toggle(self, var, then):
        var.set(not var.get())
        then()

    def on_click(self, x, y):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        mode = self.mode.get()
        if mode == "erase":
            self.erase_near(x, y)
        else:
            key = self.side_at(x, y) if mode == "side" else "far"
            self.marks[key].append([float(x), float(y)])
            self.dirty = True
            self.invalidate_fit()
            self.refresh()

    def erase_near(self, x, y):
        """Delete the nearest mark on any side, if one is within reach."""
        best, bd = None, ERASE_RADIUS
        sx, sy = self.canvas.to_screen([[x, y]])[0]
        for key in F.SIDE_ORDER:
            for j, p in enumerate(self.marks[key]):
                px, py = self.canvas.to_screen([p])[0]
                d = float(np.hypot(px - sx, py - sy))
                if d < bd:
                    best, bd = (key, j), d
        if best is None:
            self.status.set("nothing within %d px of there to erase" % int(ERASE_RADIUS))
            return
        key, j = best
        self.marks[key].pop(j)
        self.dirty = True
        self.invalidate_fit()
        self.refresh()

    def clear_group(self):
        """Clear whichever group is selected - both sidelines, or the far side."""
        keys = GROUPS.get(self.mode.get(), {}).get("keys", ())
        if any(self.marks[k] for k in keys):
            for k in keys:
                self.marks[k] = []
            self.dirty = True
            self.invalidate_fit()
            self.refresh()

    def clear_all(self):
        if any(self.marks[k] for k in F.SIDE_ORDER):
            self.marks = {k: [] for k in F.SIDE_ORDER}
            self.dirty = True
            self.invalidate_fit()
            self.refresh()

    def complete(self):
        return all(len(self.marks[k]) >= F.MIN_MARKS for k in F.SIDE_ORDER)

    # ------------------------------------------------------------- the fit --
    def invalidate_fit(self):
        self._fit, self._fit_error = None, None
        self._fit_seq += 1

    def on_toggle_court(self):
        self.refresh()

    def want_fit(self):
        return self.show_court.get() and self.complete() and self.image is not None

    def ensure_fit(self, blocking=False):
        """Start (or, on save, run) the fit for the marks as they stand."""
        if self._fit is not None or self._fit_error is not None:
            return
        if not self.complete() or self.image is None:
            return
        if blocking:
            self._fit, self._fit_error = self._run_fit()
            return
        if self._fit_running:
            return
        self._fit_running = True
        seq = self._fit_seq
        marks = {k: list(v) for k, v in self.marks.items()}
        shape, lines, anchors = self.image.shape, self.walk_lines, self.walk_anchors
        # Onto a queue and polled from here, never `after` from the thread: Tcl
        # is threaded, so a widget call from anywhere but the main thread raises
        # rather than being marshalled, and the result would vanish silently.
        threading.Thread(
            target=lambda: self._fit_queue.put((seq, _try_fit(shape, marks, lines, anchors))),
            daemon=True).start()
        self.after(80, self._poll_fit)

    def _poll_fit(self):
        if self._closing:
            return
        try:
            seq, out = self._fit_queue.get_nowait()
        except queue.Empty:
            if self._fit_running:
                self.after(80, self._poll_fit)
            return
        self._fit_running = False
        if seq != self._fit_seq:                 # the marks moved on; this is stale
            self.ensure_fit()
            return
        self._fit, self._fit_error = out
        self.refresh(refit=False)

    def _run_fit(self):
        return _try_fit(self.image.shape, self.marks, self.walk_lines, self.walk_anchors)

    # ---------------------------------------------------------------- draw --
    def refresh(self, refit=True):
        for key in F.SIDE_ORDER:
            n = len(self.marks[key])
            self.count_labels[key].configure(
                text="%s %d/%d" % (key[0].upper(), n, F.MIN_MARKS),
                foreground="#15803d" if n >= F.MIN_MARKS else "#6b7280")
        group = GROUPS.get(self.mode.get())
        self.clear_btn.configure(
            text="Clear sidelines" if self.mode.get() == "side" else "Clear far side",
            state="normal" if group else "disabled")
        done = self.complete()
        self.save_btn.configure(state="normal" if done else "disabled")
        self.court_chk.configure(state="normal" if done else "disabled")
        if self.index is not None:
            self.update_row(self.index)
        if refit and self.want_fit():
            self.ensure_fit()
        self.canvas.redraw()

    def draw_overlay(self):
        c = self.canvas
        if self.image is None:
            return
        if self.show_walks.get():
            for L in self.walk_lines:
                c.draw_polyline(L["pts"], WALK_COLOUR, width=1)
        # UNDER the marks and the court, because it is what they are being
        # compared against - a detector's 90 points drawn on top of five clicks
        # would hide the thing the clicks are for.
        self.draw_sides()
        if self.mode.get() == "side":
            # Where a click stops being a left mark and starts being a right one.
            h = self.image.shape[0]
            ys = np.linspace(0, h - 1, 60)
            c.draw_polyline([[self.divider_x(y), y] for y in ys], DIVIDER_COLOUR,
                            width=1, dash=(2, 6))
        for key in F.SIDE_ORDER:
            col = SIDE_COLOUR[key]
            pts = self.marks[key]
            curve = F.fit_curve(pts, F.SIDES[key]["swap"])
            if curve is not None:
                c.draw_polyline(F.curve_points(curve, F.SIDES[key]["swap"]), col,
                                width=2, dash=(6, 4))
            active = self.mode.get() == GROUP_OF[key]
            for p in pts:
                c.draw_dot(p, col, r=5 if active else 4,
                           outline="#ffffff" if active else "#000000",
                           width=2 if active else 1)
        if self.want_fit() and self._fit is not None:
            self.draw_court(self._fit[0])
        self.draw_ends()
        self.draw_status_line()

    def draw_ends(self):
        """The four places the service lines' paint STOPPED, as X's.

        LAST, over everything including the court, and that is the whole reason
        it is a method of its own rather than two lines up with the walk's
        polylines. The fitted court's near service line passes THROUGH the near
        pair by construction, so drawn in walk order the one pixel worth looking
        at is the one covered by a projected line - and the question here is
        exactly whether the two coincide.

        Shown with the walk, since that is what produced them, and hidden with it
        too: four X's over a plate being marked are four things in the way of the
        clicking when nobody is asking about the endpoints.
        """
        if not self.show_walks.get():
            return
        for p in self.walk_ends.values():
            self.canvas.draw_x(p, END_COLOUR, gap=END_GAP, arm=END_ARM,
                               width=END_W)

    def draw_sides(self):
        """One or four courtside readers, as their points and their own curve.

        Points are drawn small and the fit dashed, so the two are told apart at
        a glance: the question this overlay answers is whether a reader's CURVE
        sits on the boundary, and a reader can have both its points and its
        curve in the wrong place, or clean points and a curve dragged off them.

        THE READER THE CASCADE ACTUALLY CHOSE IS DRAWN THICKER, and the corner
        each of them had to reach is drawn as an X - the same near service line
        end `draw_ends` marks, marked the same way. Together those two say why
        the choice went the way it did without anyone reading the JSON: a curve
        that sails past the X is one `SIDE_CORNER_MAX` refused.
        """
        if self.sides_doc is None or self.sides_choice.get() == SD.OFF:
            return
        c = self.canvas
        for xy in SD.corners(self.sides_doc).values():
            c.draw_x(xy, SD.CORNER_COLOUR, gap=END_GAP, arm=END_ARM, width=END_W)
        choice = self.sides_choice.get()
        for L in SD.layers(self.sides_doc, choice):
            wide = 3 if L["chosen"] else 2
            if len(L["polyline"]) >= 2:
                c.draw_polyline(L["polyline"], L["colour"], width=wide,
                                dash=None if L["qualifies"] else (5, 4))
            r = SD.dot_radius(L["key"], choice)
            for p in L["points"]:
                c.draw_dot(p, L["colour"], r=r, outline=SD.DOT_OUTLINE, width=1)
        # The legend goes on the picture rather than only in the status line, so
        # that four readers at once stay readable where they overlap. Placed by
        # converting a fixed SCREEN corner back into image coordinates, because
        # the canvas draws in image space and a legend pinned to the image would
        # pan off the edge of the viewport.
        if self.sides_choice.get() == SD.ALL:
            for i, key in enumerate(SD.ORDER):
                c.draw_text(c.to_image(12, 14 + 16 * i), key, SD.COLOUR[key],
                            dx=0, dy=0, anchor="nw")

    def draw_court(self, cam):
        c = self.canvas
        for a, b, col, wdt in COURT_SEGMENTS:
            t = np.linspace(0, 1, 96)[:, None]
            p = cam.project(np.asarray(a, float) + t * (np.asarray(b, float) - np.asarray(a, float)))
            if np.isfinite(p).all():
                c.draw_polyline(p, col, width=wdt)
        for name, X, Y in CM.KEYPOINTS:
            p = cam.project([[X, Y]])[0]
            if np.isfinite(p).all():
                c.draw_dot(p, KEYPOINT, r=3, outline="#000000")

    def draw_status_line(self):
        """One line under the picture: the curves, then the court, then the gates."""
        bits = []
        for key in F.SIDE_ORDER:
            pts = self.marks[key]
            curve = F.fit_curve(pts, F.SIDES[key]["swap"])
            if curve is None:
                bits.append("%s %d" % (key, len(pts)))
            else:
                bits.append("%s %d (curve %.2f px)"
                            % (key, len(pts), F.curve_rms(pts, curve, F.SIDES[key]["swap"])))
        line = "   ".join(bits)
        if self.walk_note:
            line += "   |   " + self.walk_note
        if self.sides_note:
            line += "   |   " + self.sides_note
        sd = SD.summary(self.sides_doc, self.sides_choice.get())
        if sd:
            line += "   |   " + sd
        if self.complete():
            if self._fit_error:
                line += "   |   no court: " + self._fit_error
            elif self._fit is not None:
                cam, rep = self._fit
                cov, prms, ok, why = F.verdict(self.image, cam, rep)
                line += ("   |   line rms %.2f px   coverage %.0f%%   probe %.2f px   %s"
                         % (rep["rms_line_px"], 100 * cov, prms,
                            "clean" if ok else "check: " + why))
            elif self.show_court.get():
                line += "   |   fitting..."
        self.status.set(line)

    # ---------------------------------------------------------------- save --
    def autosave(self):
        """Marks in progress, written as incomplete so the plate comes back round."""
        if self.index is None or not self.dirty:
            return
        if not any(self.marks[k] for k in F.SIDE_ORDER):
            return
        if self.complete():
            return                                # a complete set is saved deliberately
        path = self.plates[self.index]
        curves = {k: F.fit_curve(self.marks[k], F.SIDES[k]["swap"]) for k in F.SIDE_ORDER}
        store.write_json(store.sidecar(path, store.SIDES),
                         F.sides_document(path, self.image.shape, self.marks,
                                          curves, complete=False))
        self.dirty = False

    def save_and_next(self):
        if self.index is None or not self.complete():
            return
        path = self.plates[self.index]
        self.ensure_fit(blocking=True)
        curves = {k: F.fit_curve(self.marks[k], F.SIDES[k]["swap"]) for k in F.SIDE_ORDER}
        # The marks are saved whatever happened to the fit. They are what the
        # person did, they are not recoverable, and a court that would not solve
        # is a reason to go and look at the plate - not a reason to lose them.
        store.write_json(store.sidecar(path, store.SIDES),
                         F.sides_document(path, self.image.shape, self.marks,
                                          curves, complete=True))
        if self._fit is None:
            self.dirty = False
            self.update_row(self.index)
            messagebox.showwarning(
                "Sides saved, no court",
                "%s was written, but the court could not be fitted:\n\n%s\n\n"
                "No %s for this plate." % (store.SIDES, self._fit_error or "?", store.TRUTH))
            self.refresh()
            return
        cam, rep = self._fit
        doc = F.truth_document(self.image, os.path.abspath(path), cam, rep, self.marks,
                               source=os.path.abspath(path))
        store.write_json(store.sidecar(path, store.TRUTH), doc)
        self.dirty = False
        self.update_row(self.index)
        nxt = self.next_unlabelled(self.index)
        if nxt is None:
            self.status.set("Saved. Every plate in this folder is labelled.")
            self.refresh()
            return
        self.open_index(nxt)

    def on_close(self):
        self.autosave()
        self._closing = True
        self.canvas.cancel_pending()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        self.master.destroy()


def run(folder=None, jobs=None):
    root = tk.Tk()
    root.title("padel court - ground truth labeller")
    root.geometry("1500x900")
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    App(root, folder=folder, jobs=jobs)
    root.mainloop()
