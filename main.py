import argparse
import subprocess
from position_estimation.get_camera_parameters import get_camera_parameters
from position_estimation.pose_estimation import pose_estimation
from position_estimation.apply_homography import apply_homography
from position_estimation.temporal_association import temporal_association

#définition des arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video_path", "--video_path", required=True, type=str)
ap.add_argument("-frame_start", "--frame_start", required=True, type=int)
ap.add_argument("-frame_step", "--frame_step", required=True, type=int)
ap.add_argument("-nb_frame", "--nb_frame", required=False, type=int)
args = ap.parse_args()

video_path = args.video_path
start, step = args.frame_start, args.frame_step
nb_frame = args.nb_frame

#détection du court
subprocess.run(f"python ./padel-court-detector/detect_court.py {video_path}", shell=True)

#parmètres caméra
K, H, dist_coeffs = get_camera_parameters()

#estimation de la pose
pose_estimation(video_path, start, nb_frame, step, K, dist_coeffs)

#appliquer l'homographie
apply_homography(H)

#association temporelle
window_size = 15
temporal_association(window_size)