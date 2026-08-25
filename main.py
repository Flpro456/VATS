import cv2
import mediapipe as mp
import numpy as np
import threading
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_vats import detect_pose
from pose_vats import landmarker
from object_vats import object_detect
from object_vats import detector
from pose_vats import body_parts

cap = cv2.VideoCapture(0)
frame_update_1 = threading.Event()
frame_update_2 = threading.Event()
frame_lock = threading.Lock()

def record():
    global frame
    global success
    with frame_lock:
        success, frame = cap.read()
    frame_update_1.set() 
    frame_update_2.set()

def model_1():
    global nuevo_frame
    while True:
        if frame_update_1.is_set():
            with frame_lock:
                nuevo_frame = frame.copy()
            object_detect(success, nuevo_frame)
            frame_update_1.clear()

def model_2():
    global pose_frame
    while True:
        if frame_update_2.is_set():
            with frame_lock:
                pose_frame = frame.copy()
            
            detect_pose(success, pose_frame)
            frame_update_2.clear()

def draw(frame, landmarks, objects):
    width, height, _ = frame.shape()
    for landmark in landmarks:            
        if landmark[2] in body_parts:
            cv2.rectangle(
                frame,
                (landmark[0], landmark[1]),
                (landmark[0] + 50, landmark[1] -20),
                (0, 165, 255),
                30
            )
            cv2.putText(
                frame,
                body_parts[landmark[2]],
                (landmark[0], landmark[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
    for object in objects:
        cv2.rectangle(
            frame,
            (object[0], object[1]),
            (object[0] + object[2], object[1] + object[3]),
            (0, 165, 255),
            2
        )
        cv2.putText(
            frame,
            object[4],
            (object[0], max(object[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

if cap.isOpened():
    global frame
    global success
    record()
    model_1_detection = threading.Thread(target=model_1)
    model_2_detection = threading.Thread(target=model_2)
    model_1_detection.start()
    model_2_detection.start()
    while True:
        
        record()
        draw(frame, detect_pose(success, frame), object_detect(success=success, frame=frame))
        cv2.imshow("VATS - Pose Test", frame)

        # ESC para salir
        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
landmarker.close()
detector.close()
cv2.destroyAllWindows()