import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

body_parts = {
    1: "Head",
    12: "Left arm 1", 
    14: "Left arm 2", 
    16: "Left arm 3",
    13: "Right arm 1", 
    15: "Right arm 2", 
    17: "Right arm 3"
}

part_positions = {
    "Head": (0, 0),
    "Left arm 1": (0, 0)
}

POSE_PATH = "pose_landmarker_lite.task"

# Configuración del detector
base_options_pose = python.BaseOptions(
    model_asset_path=POSE_PATH
)

options_pose = vision.PoseLandmarkerOptions(
    base_options=base_options_pose,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(options_pose)



frame_timestamp_ms = 0

def detect_laser(frame_bgr, min_area=4, max_area=400):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Rango de verde (ajustar según cámara/iluminación real)
    lower_green = np.array([40, 60, 200])   # H, S, V — V alto = brillo
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Filtrado morfológico: elimina ruido pequeño, rellena huecos
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Candidato: contorno más "compacto" y brillante dentro del rango de área esperado
    best = None
    best_score = -1
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        # circularidad como filtro de forma (1.0 = círculo perfecto)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        score = circularity  # se puede combinar con brillo medio del contorno
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def detect_pose(success, frame):
        
        global frame_timestamp_ms
        if not success:
            print("No se pudo leer la webcam.")
            return

        # OpenCV usa BGR → MediaPipe necesita RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detectar pose
        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        frame_timestamp_ms += 33

        landmark_list = []
        # Dibujar landmarks
        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            height, width, _ = frame.shape
            index = 0
            for landmark in landmarks:

                index += 1
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                
                landmark_list.append((x, y, index))

        return landmark_list
                    
