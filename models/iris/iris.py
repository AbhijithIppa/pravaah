import cv2
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time

mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)


LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [33] #right eye right most landmark
L_H_RIGHT = [133] #right eye leftmost landmark
R_H_LEFT = [362]  #left eye rightmost landmark
R_H_RIGHT = [263] #left eye leftmost landmark

cnt=0
pTime=0


def euclidean_distance(point1, point2):
    x1, y1 =point1.ravel()
    x2, y2 =point2.ravel()
    distance = math.sqrt((x2-x1)*2 + (y2-y1)*2)
    return distance


def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position =""
    if ratio >= 2.94:
        iris_position="left"

    elif ratio > 2.76 and ratio <= 2.92:
        iris_position="center"

    else:
        iris_position = "right"

    return iris_position, ratio

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.6) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])


            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])


            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)


            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)


            cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv.LINE_AA)

            iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])

            if (iris_pos == "left" or iris_pos == "right"):
                cnt += 1

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(frame,f'pos:{iris_pos} cnt:{cnt}',(20,60),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),2)
            cv2.putText(frame, f'FPS:{int(fps)}', (20, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

            print(iris_pos,ratio,cnt)
            cv.imshow("img", frame)
            key = cv.waitKey(1)
            if key ==ord("q"):
                break

cap.release()
cv.destroyAllWindows()