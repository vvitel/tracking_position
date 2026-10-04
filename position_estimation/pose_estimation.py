import cv2
import json
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def pose_estimation(video_path, start, nb_frame, step, K, dist_coeffs):
    #chargement des modèles
    model_box = YOLO("yolo26m.pt")
    model_pose = YOLO("yolo26m-pose.pt")

    #lecture de la vidéo
    cap = cv2.VideoCapture(video_path)
    if nb_frame is None: nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                result_box = model_box(frame, verbose=False, iou=0.5, classes=[0])
                boxes = result_box [0].boxes

                #croper image
                dic_pos_frame, cpt = {}, 0
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1 = x1, y1
                    x2, y2 = x2, y2

                    crop_player = frame[int(y1):int(y2), int(x1):int(x2)]
                    crop_player = cv2.resize(crop_player, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

                    #appliquer le modèle pose
                    result_pose = model_pose.predict(source=crop_player, imgsz=960, show=False, verbose=False, classes=[0])
                    keypoints = result_pose[0].keypoints.data
                    if len(keypoints) == 0: continue

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