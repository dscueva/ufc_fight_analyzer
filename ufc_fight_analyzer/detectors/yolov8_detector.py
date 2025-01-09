# ufc_fight_analyzer/detectors/yolov8_detector.py

import cv2
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path: str, conf: float, nms: float):
        """
        Initialize the YOLOv8 model with given path and thresholds.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.nms = nms

    def detect_people(self, frame):
        """
        Runs YOLOv8 on a single frame. Returns a list of dicts:
        [
            { 'bbox': [x1, y1, x2, y2], 'conf': 0.98, 'class_id': 0 },
            ...
        ]
        Specifically for the 'person' class (class_id=0 in COCO).
        """
        # YOLOv8 'predict' can take a frame (ndarray), conf, iou, etc.
        results = self.model.predict(
            frame, 
            conf=self.conf, 
            iou=self.nms
        )
        detections = []

        # YOLOv8 might return multiple results if we pass multiple frames
        # but here we pass just one, so results[0] is relevant
        boxes = results[0].boxes  # Boxes object
        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]

            # Filter to 'person' class. By default, class 0 is 'person' in COCO
            if class_id == 0:
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': conf,
                    'class_id': class_id
                })

        return detections