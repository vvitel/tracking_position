#!/usr/bin/env python3
"""Detect a padel court in one video or one plate.

    python detect_court.py match.mkv
    python detect_court.py court_plate.png --json out.json --pictures out/

Prints the lens coefficients, the homography and the 13 court keypoints in the
ORIGINAL image's pixels, and exits non-zero if the fit was refused.
"""
import argparse
import contextlib
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from padelcourt import (VIDEO_EXT, calibrate_image_file, calibrate_video,  # noqa: E402
                        courtside_detection_path, save_courtside_detection,
                        undistort_image, walk, write_pictures)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="a video, or a plate image (.png/.jpg)")
    ap.add_argument("--json", default="", help="write the full result here")
    ap.add_argument("--pictures", default="", help="write the three pictures into this folder")
    ap.add_argument("--undistort", default="", help="write the lens-corrected plate here")
    ap.add_argument("--frames", type=int, default=45, help="keyframes to median for the plate")
    ap.add_argument("--force-plate", action="store_true", help="rebuild the plate even if cached")
    ap.add_argument("--rounds", type=int, default=2, help="model-seeded retrace rounds")
    ap.add_argument("--model", default="fisheye", choices=["fisheye", "poly"])
    ap.add_argument("--nk", type=int, default=2, help="number of radial coefficients")
    ap.add_argument("--sides-json", nargs="?", const="", default=None,
                    help="write every courtside technique's points and their fit; "
                         "with no value, beside the plate as "
                         "<plate>_courtside-detection.json")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the result JSON on stdout (progress goes to stderr)")
    a = ap.parse_args()

    # Reading all four costs the one `track_side` call the cascade normally
    # skips, so it is asked for rather than always paid.
    walk.SIDE_RECORD_ALL = a.sides_json is not None

    # In --quiet the JSON is the output, so everything the libraries print -
    # they narrate as they draw - has to be kept off stdout for the result to
    # be pipeable. A FRESH context manager per block: `redirect_stdout` is not
    # reusable, and reusing one raises on the second `with`.
    def chatter():
        return contextlib.redirect_stdout(sys.stderr) if a.quiet else _nothing()

    is_video = a.input.lower().endswith(VIDEO_EXT)
    if not a.quiet:
        print("%s %s" % ("video" if is_video else "plate", a.input))
    try:
        with chatter():
            if is_video:
                res = calibrate_video(a.input, frames=a.frames, force=a.force_plate,
                                      verbose=True, rounds=a.rounds,
                                      model=a.model, nk=a.nk)
            else:
                res = calibrate_image_file(a.input, rounds=a.rounds, model=a.model, nk=a.nk)
    except RuntimeError as e:
        print("FAILED: %s" % e, file=sys.stderr)
        return 2

    d = res.to_dict()

    wanted = {"far_corner_l", "far_corner_r", "near_corner_l", "near_corner_r"}
    keypoints = {kp["name"]: kp["image_px"] for kp in d["keypoints"] if kp["name"] in wanted}

    
    # Save results
    calibration = {
            "lens": {
            "model": d["lens"]["model"],
            "k": d["lens"]["k"],
            "f": d["lens"]["f"],
            "cx": d["lens"]["cx"],
            "cy": d["lens"]["cy"]
            },
            "homography": d["homography"]["matrix_undistorted_px"],
            "keypoints": keypoints
        }
    
    with open("./calibration.json", "w") as f:
            json.dump(calibration, f, indent=4)
    
    if not a.quiet:
        q = d["quality"]
        print("\nplate       %s" % d["plate"])
        print("verdict     %s%s" % ("ACCEPT" if res.ok else "REJECT",
                                    "" if res.ok else "  (%s)" % res.reason))
        print("quality     line rms %.2f px   paint coverage %.0f%%   probe %.2f px"
              % (q["line_rms_px"], 100 * q["paint_coverage"], q["probe_rms_px"]))
        lens = d["lens"]
        print("\nlens        %s   k = %s   f = %.1f   centre = (%.1f, %.1f)"
              % (lens["model"], ", ".join("%.6f" % v for v in lens["k"]),
                 lens["f"], lens["cx"], lens["cy"]))
        for line in lens["equation"].split("\n"):
            print("            %s" % line)
        print("\nhomography  court metres -> pixels of the UNDISTORTED image")
        for row in d["homography"]["matrix_undistorted_px"]:
            print("            [%12.4f %12.4f %12.4f]" % tuple(row))
        print("\nkeypoints   (original image pixels)")
        for kp in d["keypoints"]:
            print("            %-15s world (%5.2f, %6.2f) m   image (%8.2f, %8.2f)%s%s"
                  % (kp["name"], kp["world_m"][0], kp["world_m"][1],
                     kp["image_px"][0], kp["image_px"][1],
                     "   derived" if kp["derived"] else "",
                     "" if kp.get("in_frame", True) else "   OFF FRAME"))

    if a.json:
        res.save_json(a.json)
        if not a.quiet:
            print("\nwrote %s" % a.json)
    if a.sides_json is not None:
        p = save_courtside_detection(
            res, a.sides_json or courtside_detection_path(res.plate_path))
        if not a.quiet:
            print("courtsides: %s" % (p or "nothing recorded - the walk never "
                                           "reached the sidelines"))
    if a.pictures:
        with chatter():
            made = write_pictures(res, a.pictures)
        if not a.quiet:
            for k, v in sorted(made.items()):
                print("%s: %s" % (k, v))
    if a.undistort:
        cv2.imwrite(a.undistort, undistort_image(res.plate, res.camera))
        if not a.quiet:
            print("undistorted: %s" % a.undistort)
    if a.quiet:
        json.dump(d, sys.stdout, indent=1)
        print()
    return 0 if res.ok else 1


@contextlib.contextmanager
def _nothing():
    yield


if __name__ == "__main__":
    sys.exit(main())
