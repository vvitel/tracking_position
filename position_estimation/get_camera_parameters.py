import json
import numpy as np

def get_camera_parameters():
    with open("calibration.json", "r") as f:
        cam = json.load(f)

    K = np.array([
        [cam["lens"]["f"], 0, cam["lens"]["cx"]],
        [0, cam["lens"]["f"], cam["lens"]["cy"]],
        [0, 0, 1]
    ], dtype=np.float64)

    H = np.array(
        cam["homography"],
        dtype=np.float64
    )

    dist_coeffs = np.array([
        cam["lens"]["k"][0],
        cam["lens"]["k"][1],
        0,
        0,
        0
    ], dtype=np.float64)

    keypoints = cam["keypoints"]

    return K, H, dist_coeffs, keypoints