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
    global nuevo_frame
    while True:
        if frame_update_2.is_set():
            with frame_lock:
                nuevo_frame = frame.copy()
            
            detect_pose(success, nuevo_frame)
            frame_update_2.clear()



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
        cv2.imshow("VATS - Pose Test", nuevo_frame)

        # ESC para salir
        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
landmarker.close()
detector.close()
cv2.destroyAllWindows()