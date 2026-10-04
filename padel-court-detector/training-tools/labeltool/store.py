"""The three sidecar files, and which plates still need which.

Every file this tool writes sits beside its plate and is named after it, so a
folder of plates carries its own labelling state and nothing has to be tracked
elsewhere:

    <stem>_court-lines.json   what the detector's walk found, generated
    <stem>_court-sides.json   the three boundaries, marked by a person
    <stem>_court-truth.json   the court fitted from the two, in the same schema
                              the detector's own output uses

A FAILED WALK STILL WRITES ITS FILE. Work is skipped on the presence of
`_court-lines.json`, so a plate whose walk raises would otherwise be re-walked -
seven seconds at a time - on every run of the tool for as long as it sits in the
folder. The file is written with `ok: false` and the reason, which both stops the
retry and puts the failure somewhere it can be read.
"""
import json
import os
import tempfile

LINES = "_court-lines.json"
SIDES = "_court-sides.json"
TRUTH = "_court-truth.json"
SUFFIXES = (LINES, SIDES, TRUTH)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: Pictures `batch_courts.py` writes next to the plates it read. They are images
#: in the same folder and are emphatically not plates.
NOT_PLATES = ("_court_debug", "_court_success", "_court_fail")


def sidecar(plate, suffix):
    """`/x/y_plate.png`, `_court-sides.json` -> `/x/y_plate_court-sides.json`."""
    return os.path.splitext(plate)[0] + suffix


def find_plates(folder):
    """Every image in `folder` that is a plate, in a stable order.

    Not recursive: the folder the user picked is the unit of work, and walking
    into `courts/` would offer the detector's own output pictures for labelling.
    """
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(IMAGE_EXT):
            continue
        if any(m in name for m in NOT_PLATES):
            continue
        out.append(path)
    return out


def read_json(path):
    """The document, or None if it is absent or unreadable.

    Unreadable counts as absent on purpose: a half-written file from a killed
    run should send the plate back to the queue, not stop the tool.
    """
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_json(path, doc):
    """Write via a temporary file in the same folder, then rename over.

    The rename is atomic, so a reader never sees a partial document and a crash
    mid-write cannot destroy the labelling that was already there.
    """
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


# ------------------------------------------------------------- the states --
MISSING, OK, FAILED = "missing", "ok", "failed"


def lines_state(plate):
    """`missing`, `ok`, or `failed` - see the note about failed walks above."""
    doc = read_json(sidecar(plate, LINES))
    if doc is None:
        return MISSING
    return OK if doc.get("ok") else FAILED


def sides_state(plate):
    doc = read_json(sidecar(plate, SIDES))
    if doc is None:
        return MISSING
    return OK if doc.get("complete") else FAILED


def truth_state(plate):
    return OK if os.path.exists(sidecar(plate, TRUTH)) else MISSING


def state_of(plate):
    return {"lines": lines_state(plate), "sides": sides_state(plate),
            "truth": truth_state(plate)}
