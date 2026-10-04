"""Padel court detection from a video or a plate.

    from padelcourt import calibrate_video
    res = calibrate_video("match.mkv")
    res.keypoints()          # court points in the original image's pixels
    res.homography()         # court metres -> pixels of the undistorted image
    res.lens()               # distortion coefficients, and the equation

The modules underneath, in the order the pipeline uses them:

    plate      a still of the empty court, median of the video's keyframes
    image      what the detector reads off the pixels
    court      court dimensions, the six lines, and the Camera
    walk       the detector: walks out of the near service T, then scans
    solve      the camera fit, and the re-trace that refines it
    calibrate  detect -> solve -> re-trace -> judge
    draw       the overlay and the walk picture
    api        the shape of the answer
"""
from .api import (                                           # noqa: F401
    CourtResult,
    build_plate,
    calibrate_image_file,
    calibrate_plate,
    calibrate_video,
    courtside_detection,
    courtside_detection_path,
    homography_of,
    keypoints_of,
    lens_of,
    save_courtside_detection,
    undistort_image,
    write_pictures,
)
from .plate import VIDEO_EXT                                 # noqa: F401

__all__ = ["CourtResult", "calibrate_plate", "calibrate_image_file", "calibrate_video",
           "build_plate", "keypoints_of", "lens_of", "homography_of", "undistort_image",
           "write_pictures", "courtside_detection", "save_courtside_detection",
           "courtside_detection_path", "VIDEO_EXT"]
