import cv2
import json
import numpy as np

def apply_homography(H):
    H = np.linalg.inv(H)
    
    #ouvrir les données
    with open("data_pose.json", "r") as f:
        dico = json.load(f)

    #transformer les points
    dico_match = {}
    for frame in dico.keys():
        dico_frame, cpt = {}, 0
        for skeleton in range(len(dico[frame])):
            #cas ou on a pas d'estimation de pose
            if len(dico[frame][str(skeleton)]) == 2:
                 x_img = dico[frame][str(skeleton)][0]
                 y_img = dico[frame][str(skeleton)][1]

            else:
                left_ankle = dico[frame][str(skeleton)][15]
                right_ankle = dico[frame][str(skeleton)][16]
                left_shoulder = dico[frame][str(skeleton)][5]
                right_shoulder = dico[frame][str(skeleton)][6]

                #cas ou on a pas les chevilles - on projete les épaules
                if (left_ankle == 0) and (right_ankle == 0):
                    if (left_shoulder != 0) and (right_shoulder != 0):
                        x_mid_shoulder = (left_shoulder[0] + right_shoulder[0]) / 2
                        y_mid_shoulder = (left_shoulder[1] + right_shoulder[1]) / 2
                        p_court = H @ np.array([x_mid_shoulder, y_mid_shoulder, 1.0])
                        x_world  = p_court[0] / p_court[2]
                        y_world  = p_court[1] / p_court[2]
                        #si on arrive là on enregistre la position
                        dico_frame[cpt] = (float(x_world), float(y_world))
                        cpt += 1
                        continue
                    else:
                        continue

                #cas ou on a les deux chevilles
                if (left_ankle != 0) and (right_ankle != 0):
                        x_img = (left_ankle[0] + right_ankle[0]) / 2
                        y_img = (left_ankle[1] + right_ankle[1]) / 2

                #cas ou une seule cheville
                if (left_ankle != 0) and (right_ankle == 0):
                        x_img, y_img = left_ankle[0], left_ankle[1]
                if (left_ankle == 0) and (right_ankle != 0):
                        x_img, y_img = right_ankle[0], right_ankle[1]
            
            point = np.array([[[x_img, y_img]]], dtype=np.float32)
            transform_point = cv2.perspectiveTransform(point, H)

            x_world = transform_point[0][0][0]
            y_world = transform_point[0][0][1]

            #point hors du terrain
            if not (0 <= x_world <= 10 and 0 <= y_world <= 20):
                    continue

            #si on arrive là on enregistre la position
            dico_frame[cpt] = (float(x_world), float(y_world))
            cpt += 1

        dico_match[frame] = dico_frame

    #enregistrement 
    with open("data_pose.json", "w") as f:
        json.dump(dico_match, f)