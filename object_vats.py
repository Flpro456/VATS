import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "efficientdet_lite0.tflite"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    max_results=1,
    score_threshold=0.4
)

detector = vision.ObjectDetector.create_from_options(options)

frame_timestamp_ms = 0

def object_detect(success, frame):
    global frame_timestamp_ms
    if not success:
        print("No se pudo leer la webcam.")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    frame_timestamp_ms += 33

    # Dibujar detecciones
    for detection in result.detections:

        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        w = bbox.width
        h = bbox.height

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 165, 255),
            2
        )

        # Nombre de la clase
        if detection.categories:

            category = detection.categories[0]

            text = f"{category.category_name} {category.score:.0%}"

            cv2.putText(
                frame,
                text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )