#!/usr/bin/env python3
"""Fit a padel court as a metric camera pose, against courtside edges.

    python fit_court.py plate.png                 # one plate, printed
    python fit_court.py plate.png --picture out.jpg
    python fit_court.py --compare /folder/of/labelled/plates

The front end for `padelcourt.fitcourt`, which is a separate solver from the one
`detect_court.py` runs - see that module's docstring for what differs. `--compare`
scores both of them against the `_court-truth.json` beside each plate, by
`compare_sides.court_error_m`, so the two are judged by the same measure the rest
of the corpus work uses.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_sides import (COURT_FAR_TOL_M, COURT_TOL_M, court_error_m,  # noqa: E402
                           plates_with_truth)
from padelcourt import calibrate_image_file, fitcourt  # noqa: E402
from padelcourt.court import Camera  # noqa: E402
import padelcourt.court as CM  # noqa: E402


def _passed(e):
    return (e["near_max"] is not None and e["near_max"] <= COURT_TOL_M
            and (e["far_max"] is None or e["far_max"] <= COURT_FAR_TOL_M))


def show(rep):
    p = rep["pose"]
    L, S = rep["lens"], rep["lens_seed"]
    print("\nlens        %s   f = %.1f   k = %s   centre = (%.1f, %.1f)"
          % (L["model"], L["f"], ", ".join("%.6f" % v for v in L["k"]),
             L["cx"], L["cy"]))
    print("            fitted here, from a %s seed (f = %.1f, k = %s)"
          % (S["source"], S["f"], ", ".join("%.6f" % v for v in S["k"])))
    print("pose        height %.2f m   t = (%.2f, %.2f, %.2f) m   rvec = (%.4f, %.4f, %.4f)%s"
          % (p["height_m"], p["t_m"][0], p["t_m"][1], p["t_m"][2],
             p["rvec"][0], p["rvec"][1], p["rvec"][2],
             "   f FITTED" if p["free_f"] else "   f PINNED"))

    print("\nrounds")
    for h in rep["rounds"]:
        print("  %d  band %.0f/%.0f  moved %6s px  height %.2f m"
              % (h["round"], h["band"][0], h["band"][1],
                 "-" if h["moved_px"] == float("inf") else "%.2f" % h["moved_px"],
                 h["height_m"]))
    print("\nlines                n     rms px   max px    axis   model      measured")
    for n, v in sorted(rep["per_line"].items()):
        print("  %-16s %5d   %6.2f   %6.2f      %s   %7.3f    %7.3f m"
              % (n, v["n"], v["rms_px"], v["max_px"], v["axis"],
                 v["model_m"], v["measured_m"]))
    if rep.get("net_line"):
        v = rep["net_line"]
        print("  %-16s %5d   %6.2f   %6.2f      %s   %7.3f    %7.3f m   %s (drop %.0f L)"
              % ("net_line", v["n"], v["rms_px"], v["max_px"], v["axis"],
                 v["model_m"], v["measured_m"],
                 "fitted" if v["fitted"] else "CHECK ONLY", v["drop_L"]))
    if rep.get("far_half_check"):
        print("\nfar half    read and NOT fitted - see fitcourt.SIDE_Y_FAR")
        for n, v in sorted(rep["far_half_check"].items()):
            print("  %-16s %5d   %6.2f px                %7.3f    %7.3f m"
                  % (n, v["n"], v["rms_px"], v["model_m"], v["measured_m"]))
    q = rep["quality"]
    print("\nverdict     %s%s" % ("ACCEPT" if rep["ok"] else "REJECT",
                                  "" if rep["ok"] else "   " + rep["reason"]))
    print("            near-half rms %.2f px (gate %.1f)   coverage %.0f%%   "
          "probe %.2f px   anchor %.1f px"
          % (q["line_rms_px"], fitcourt.V_RMS_MAX, 100 * q["paint_coverage"],
             q["probe_rms_px"], q["max_anchor_px"] or 0.0))
    if q.get("far_rms_px") is not None:
        print("            far segment rms %.2f px over %d samples - REPORTED, not "
              "gated, see fitcourt.verdict" % (q["far_rms_px"], q["n_far"]))
    print("\nanchors     worst %.2f px" % (rep["max_anchor_px"] or 0.0))
    for n, v in sorted(rep["per_point"].items()):
        print("  %-20s %6.2f px   world (%.2f, %.2f)"
              % (n, v["px"], v["world"][0], v["world"][1]))


def picture(img, rep, out):
    """The fitted court drawn over the plate, with the courtside samples."""
    cam = rep["camera"]
    vis = img.copy()
    for name, (wa, wb) in CM.WORLD_LINES.items():
        col = (0, 255, 255) if "sideline" in name else (255, 255, 255)
        if name == "net_line":
            col = (255, 0, 0)
        pts = cam.polyline(wa, wb, 96).astype(np.int32)
        cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)
    for kp, (x, y) in zip(CM.KP_NAMES, cam.project(CM.KP_WORLD)):
        cv2.drawMarker(vis, (int(round(x)), int(round(y))), (0, 165, 255),
                       cv2.MARKER_TILTED_CROSS, 14, 1)
    txt = "pose fit   height %.2f m   sidelines %s" % (
        rep["pose"]["height_m"],
        "  ".join("%s %.3f m" % (n.split("_")[0], v["measured_m"])
                  for n, v in sorted(rep["per_line"].items()) if "sideline" in n))
    cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(out, vis)
    return out


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
#: Suffixes this tool WRITES, so a second run over a folder it has already
#: written into does not try to calibrate its own output pictures.
MADE = ("_court_debug", "_court_success", "_court_fail")


def plates_in(folder):
    """Every image in `folder` that is not something this tool produced."""
    out = []
    for name in sorted(os.listdir(folder)):
        low = name.lower()
        if not low.endswith(IMAGE_EXT):
            continue
        if any(m in name for m in MADE):
            continue
        out.append(os.path.join(folder, name))
    return out


def _batch_one(job):
    """One plate: fit it, write its JSON and its pictures. Never raises."""
    plate, out_dir, kw, ext = job
    stem = os.path.splitext(os.path.basename(plate))[0]
    row = {"plate": os.path.basename(plate), "stem": stem}
    img = cv2.imread(plate)
    if img is None:
        return dict(row, error="unreadable")
    try:
        rep = fitcourt.fit(img, **kw)
    except BaseException as e:                               # noqa: BLE001 - SystemExit too
        return dict(row, error=str(e)[:140])
    q = rep["quality"]
    row.update(ok=rep["ok"], reason=rep["reason"],
               line_rms_px=round(q["line_rms_px"], 3),
               paint_coverage=round(q["paint_coverage"], 3),
               probe_rms_px=round(q["probe_rms_px"], 3),
               max_anchor_px=round(q["max_anchor_px"] or 0.0, 2),
               height_m=round(rep["pose"]["height_m"], 3),
               f=round(rep["pose"]["f"], 1),
               k1=round(rep["pose"]["k"][0], 6), k2=round(rep["pose"]["k"][1], 6),
               net_y_m=round((rep.get("net_line") or {}).get("measured_m", float("nan")), 3))
    doc = {k: v for k, v in rep.items()
           if k not in ("camera", "lines", "net", "anchor_points")}
    doc["plate"] = plate
    doc["image_size"] = {"width": img.shape[1], "height": img.shape[0]}
    doc["camera_model"] = rep["camera"].to_dict()
    with open(os.path.join(out_dir, stem + ".json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    row["json"] = stem + ".json"
    row.update({k: os.path.basename(v) for k, v in
                fitcourt.write_pictures(img, rep, out_dir, stem, ext).items()
                if isinstance(v, str)})
    return row


def batch(folder, out_dir, jobs, kw, ext):
    """Every plate in a folder -> json + three pictures each, and one index."""
    plates = plates_in(folder)
    if not plates:
        print("no images in %s" % folder, file=sys.stderr)
        return 2
    os.makedirs(out_dir, exist_ok=True)
    print("%d plates -> %s   (%d workers)\n" % (len(plates), out_dir, jobs))
    rows = []
    work = [(p, out_dir, kw, ext) for p in plates]
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, r in enumerate(ex.map(_batch_one, work), 1):
            rows.append(r)
            print("  %3d/%d  %-56s %s" % (
                i, len(plates), r["stem"][:56],
                "FAILED: " + r["error"] if r.get("error") else
                ("ACCEPT" if r["ok"] else "REJECT " + r["reason"][:44])), flush=True)
    ok = sum(1 for r in rows if r.get("ok"))
    err = [r for r in rows if r.get("error")]
    print("\n  %d accepted, %d refused, %d could not be fitted, of %d"
          % (ok, len(rows) - ok - len(err), len(err), len(rows)))
    for name, key in (("line rms px", "line_rms_px"), ("probe rms px", "probe_rms_px"),
                      ("coverage", "paint_coverage"), ("camera height m", "height_m")):
        v = sorted(r[key] for r in rows if r.get(key) is not None)
        if v:
            print("  %-16s p50 %6.3f   p90 %6.3f   range %6.3f - %6.3f"
                  % (name, v[len(v) // 2], v[min(len(v) - 1, int(0.9 * (len(v) - 1)))],
                     v[0], v[-1]))
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(out_dir, "index.csv"), "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")).replace(",", ";") for k in keys) + "\n")
    json.dump(rows, open(os.path.join(out_dir, "index.json"), "w"), indent=1)
    print("\n  wrote %s and index.csv / index.json" % out_dir)
    return 0 if not err else 1


def _lens_of(truth):
    """The lens out of a truth file, in the shape `fitcourt.fixed_lens` wants."""
    c = truth["camera_model"]
    return {"f": c["f"], "k": c["k"], "cx": c["cx"], "cy": c["cy"]}


def _one(job):
    """One plate, both solvers, scored against its truth. Never raises."""
    plate, truth, kw, truth_lens, pics, ext = job
    row = {"plate": os.path.basename(plate)}
    try:
        d = json.load(open(truth))
        tcam = Camera.from_dict(d["camera_model"])
        size = (d["image_size"]["width"], d["image_size"]["height"])
    except Exception as e:                                   # noqa: BLE001
        return dict(row, error="truth: %s" % e)
    img = cv2.imread(plate)
    if img is None:
        return dict(row, error="unreadable")
    try:
        rep = fitcourt.fit(img, **(dict(kw, lens=_lens_of(d)) if truth_lens else kw))
        e = court_error_m(rep["camera"], tcam, size)
        if pics:
            fitcourt.write_pictures(img, rep, pics,
                                    os.path.splitext(os.path.basename(plate))[0], ext)
        row.update(pose_near=e["near_max"], pose_far=e["far_max"],
                   pose_med=e["median"], pose_ok=_passed(e),
                   verdict=rep["ok"], reason=rep["reason"],
                   rms=rep["quality"]["line_rms_px"],
                   coverage=rep["quality"]["paint_coverage"],
                   probe=rep["quality"]["probe_rms_px"],
                   height=rep["pose"]["height_m"],
                   pose_left=rep["per_line"].get("left_sideline", {}).get("measured_m"),
                   pose_right=rep["per_line"].get("right_sideline", {}).get("measured_m"),
                   net_m=(rep.get("net_line") or {}).get("measured_m"))
    except BaseException as e:                               # noqa: BLE001 - SystemExit too
        row["pose_error"] = str(e)[:120]
    try:
        res = calibrate_image_file(plate)
        e = court_error_m(res.camera, tcam, size)
        row.update(base_near=e["near_max"], base_far=e["far_max"],
                   base_med=e["median"], base_ok=_passed(e), base_accept=res.ok)
    except BaseException as e:                               # noqa: BLE001
        row["base_error"] = str(e)[:120]
    return row


def compare(root, jobs, kw, truth_lens, csv_path, pics="", ext=".png"):
    pairs = plates_with_truth(root)
    if not pairs:
        print("no labelled plates in %s" % root, file=sys.stderr)
        return 2
    print("%d labelled plates, %d workers\n" % (len(pairs), jobs))
    rows = []
    work = [(p, t, kw, truth_lens, pics, ext) for p, t in pairs]
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, r in enumerate(ex.map(_one, work), 1):
            rows.append(r)
            print("  %3d/%d  %-52s pose %s %-6s  base %s"
                  % (i, len(pairs), r["plate"][:52], _cell(r, "pose"),
                     "" if "verdict" not in r else
                     ("ACCEPT" if r["verdict"] else "REJECT"),
                     _cell(r, "base")), flush=True)

    def stats(pre):
        v = sorted(r["%s_near" % pre] for r in rows
                   if r.get("%s_near" % pre) is not None)
        ok = sum(1 for r in rows if r.get("%s_ok" % pre))
        if not v:
            return "no result"
        return ("%3d/%d inside tolerance   near-max p50 %.3f  p90 %.3f  worst %.3f m"
                % (ok, len(rows), v[len(v) // 2],
                   v[min(len(v) - 1, int(0.9 * (len(v) - 1)))], v[-1]))

    print("\n  pose fit   %s" % stats("pose"))
    print("  existing   %s" % stats("base"))
    both = [(r.get("pose_ok"), r.get("base_ok")) for r in rows]
    print("\n  pose only  %d      existing only  %d      both  %d      neither  %d"
          % (sum(1 for a, b in both if a and not b),
             sum(1 for a, b in both if b and not a),
             sum(1 for a, b in both if a and b),
             sum(1 for a, b in both if not a and not b)))
    acc = [r for r in rows if r.get("verdict") is not None]
    if acc:
        n_ok = sum(1 for r in acc if r["verdict"])
        agree = sum(1 for r in acc if bool(r["verdict"]) == bool(r.get("pose_ok")))
        print("\n  its OWN verdict says ACCEPT on %d of %d, and agrees with the "
              "truth score on %d" % (n_ok, len(acc), agree))
        wrong = [r for r in acc if r["verdict"] and not r.get("pose_ok")]
        if wrong:
            print("  accepted but outside tolerance (the dangerous direction): %d" % len(wrong))
            for r in wrong:
                print("    %-52s near %.3f m" % (r["plate"][:52], r["pose_near"]))
    h = [r["height"] for r in rows if r.get("height") is not None]
    if h:
        h = sorted(h)
        print("  camera height  p50 %.2f m   range %.2f - %.2f m"
              % (h[len(h) // 2], h[0], h[-1]))
    nm = sorted(r["net_m"] for r in rows if r.get("net_m") is not None)
    if nm:
        print("  net floor line reads  p50 %.2f m   range %.2f - %.2f m   "
              "(nominal 10.0, this reader anchored at %.3f - fitcourt.NET_SHORT)"
              % (nm[len(nm) // 2], nm[0], nm[-1], CM.NET_Y + fitcourt.NET_SHORT))
    bad = [r for r in rows if r.get("pose_error")]
    if bad:
        print("\n  %d plate(s) the pose fit could not do:" % len(bad))
        for r in bad:
            print("    %-58s %s" % (r["plate"][:58], r["pose_error"]))
    if csv_path:
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print("\nwrote %s" % csv_path)
    return 0


def _cell(r, pre):
    if r.get("%s_error" % pre):
        return "FAIL"
    v = r.get("%s_near" % pre)
    return "  -  " if v is None else "%5.3f%s" % (v, "" if r.get("%s_ok" % pre) else "*")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?",
                    help="a plate image, or a FOLDER of them (with --out)")
    ap.add_argument("--out", default="",
                    help="with a folder, write <plate>.json and the three pictures "
                         "for every plate in it here (default <folder>/pose)")
    ap.add_argument("--compare", metavar="DIR",
                    help="score this solver and the existing one over labelled plates")
    ap.add_argument("--picture", default="", help="draw the fit to this one file")
    ap.add_argument("--pictures", default="",
                    help="write <plate>_court_debug / _court_success / _court_fail "
                         "into this folder, for one plate or for every plate of "
                         "--compare. Same names as detect_court.py's, so point it "
                         "at a different folder and the two are comparable")
    ap.add_argument("--ext", default=".png", choices=[".png", ".jpg"],
                    help="picture format (default .png; .jpg is ~10x smaller)")
    ap.add_argument("--json", default="", help="write the report here")
    ap.add_argument("--pin-f", action="store_true",
                    help="hold f at w/2 instead of fitting it - court.py's convention. "
                         "Costs 12 plates on the labelled corpus, see fitcourt.Fixed")
    ap.add_argument("--fix-lens", action="store_true",
                    help="hold the lens exactly as given and fit the 6 pose parameters "
                         "only - the pure design; needs a lens that is actually right, "
                         "see fitcourt.fixed_lens")
    ap.add_argument("--lens", choices=["neutral", "straighten"], default="neutral",
                    help="where the lens starts from (default neutral: f=w/2, k=0)")
    ap.add_argument("--truth-lens", action="store_true",
                    help="with --compare, take the lens from each plate's truth file "
                         "instead of straighten - isolates this solver from the lens")
    ap.add_argument("--no-far-half", action="store_true",
                    help="fit only the near segment of each sideline (Y 10.3-17.6) "
                         "instead of also the stretch between the far service line "
                         "and the net, see fitcourt.FAR_HALF")
    ap.add_argument("--net-weight", type=float, default=0.0,
                    help="put the net's floor line into the residual at this weight "
                         "(0 = measured and reported only, the default)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--csv", default="", help="with --compare, write the rows here")
    a = ap.parse_args()
    kw = {"free_f": not a.pin_f and not a.fix_lens, "free_k": not a.fix_lens,
          "net_weight": a.net_weight, "far_half": not a.no_far_half,
          "lens": None if a.lens == "neutral" else a.lens}

    if a.compare:
        return compare(a.compare, a.jobs, kw, a.truth_lens, a.csv,
                       a.pictures, a.ext)
    if not a.input:
        ap.error("give a plate or a folder, or --compare a folder")
    if os.path.isdir(a.input):
        return batch(a.input, a.out or os.path.join(a.input, "pose"),
                     a.jobs, kw, a.ext)
    if a.out:
        ap.error("--out takes a FOLDER as input; for one plate use --pictures/--json")
    img = cv2.imread(a.input)
    if img is None:
        print("cannot read %s" % a.input, file=sys.stderr)
        return 2
    print("plate %s" % a.input)
    rep = fitcourt.fit(img, verbose=True, **kw)
    show(rep)
    if a.picture:
        print("\npicture: %s" % picture(img, rep, a.picture))
    if a.pictures:
        made = fitcourt.write_pictures(
            img, rep, a.pictures, os.path.splitext(os.path.basename(a.input))[0], a.ext)
        for k, v in sorted(made.items()):
            print("%s: %s" % (k, v))
    if a.json:
        out = {k: v for k, v in rep.items()
               if k not in ("camera", "lines", "net", "anchor_points")}
        out["camera_model"] = rep["camera"].to_dict()
        json.dump(out, open(a.json, "w"), indent=1)
        print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
