import cv2
import json
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def pose_estimation(video_path, start, nb_frame, step, K, dist_coeffs, keypoints):
    #chargement des modèles
    model_box = YOLO("yolo26m.pt")
    #model_pose = YOLO("yolo26m-pose.pt")

    #lecture de la vidéo
    cap = cv2.VideoCapture(video_path)
    if nb_frame is None: nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #définir limite terrain
    x_far_corner_l, y_far_corner_l = keypoints["far_corner_l"][0], keypoints["far_corner_l"][1]
    x_near_corner_l, y_near_corner_l = keypoints["near_corner_l"][0], keypoints["near_corner_l"][1]
    x_far_corner_r, y_far_corner_r = keypoints["far_corner_r"][0], keypoints["far_corner_r"][1]
    x_near_corner_r, y_near_corner_r = keypoints["near_corner_r"][0], keypoints["near_corner_r"][1]

    a_right = (y_far_corner_r - y_near_corner_r) / (x_far_corner_r - x_near_corner_r)
    a_left = (y_far_corner_l - y_near_corner_l) / (x_far_corner_l - x_near_corner_l)
    b_right = y_near_corner_r - a_right * x_near_corner_r
    b_left = y_near_corner_l - a_left * x_near_corner_l

    #définir la première frame
    frame_index = start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    dico = {}
    #parcourir les images
    with tqdm() as pbar:
        while True:
            ret, frame = cap.read()

            if not ret: break
            if frame_index >= nb_frame: break
            
            #appliquer le modèle box
            if frame_index % step == 0:
                result_box = model_box(frame, verbose=False, iou=0.2, classes=[0])
                boxes = result_box [0].boxes

                #croper image
                dic_pos_frame, cpt = {}, 0
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bottom_center_x = (x1 + x2) / 2
                    bottom_center_y = y2

                    #vérification de la position du point
                    y_right = a_right * bottom_center_x + b_right
                    y_left = a_left * bottom_center_x + b_left

                    if (bottom_center_y < y_right) or (bottom_center_y < y_left):
                        continue

                    #crop_player = frame[int(y1):int(y2), int(x1):int(x2)]
                    #crop_player = cv2.resize(crop_player, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

                    #appliquer le modèle pose
                    #result_pose = model_pose.predict(source=crop_player, imgsz=960, show=False, verbose=False, classes=[0])
                    #keypoints = result_pose[0].keypoints.data

                    #si pas de pose on garde le centre bas de la box
                    #if len(keypoints) == 0:
                    #CAS OU ON PREND QUE LA BOX
                    dic_pos_frame[cpt] = [bottom_center_x, bottom_center_y]
                    cpt += 1
                    continue

                    #remettre dans le repère initial
                    pose = keypoints[0].cpu().numpy()
                    pose[:, 0] = pose[:, 0] / 4 + x1
                    pose[:, 1] = pose[:, 1] / 4 + y1
                    points = pose[:, :2].astype(np.float32)

                    #retirer la distortion
                    points_undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2), K, dist_coeffs, P=K).reshape(-1, 2)

                    #enregistrer les poses
                    dic_pos_frame[cpt] = points_undistorted.tolist()
                    cpt += 1

                dico[frame_index] = dic_pos_frame
                
            frame_index += 1
            pbar.update(1)

    #enregistrer les résultats
    with open("data_pose.json", "w") as f:
        json.dump(dico, f)

    cap.release()