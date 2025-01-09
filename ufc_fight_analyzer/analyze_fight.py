# ufc_fight_analyzer/analyze_fight.py

import cv2
import yaml
import os

from ufc_fight_analyzer.detectors.yolov8_detector import YOLOv8Detector

def main():
    # 1) Load config
    config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_path = config["model_path"]
    conf_threshold = config["confidence_threshold"]
    nms_threshold = config["nms_threshold"]

    # 2) Initialize the detector
    detector = YOLOv8Detector(model_path, conf_threshold, nms_threshold)

    # 3) Video path
    video_path = os.path.join(os.path.dirname(__file__), "..", "ufc_fights", "pereira_rountree_1_min.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 4) Detect
        person_detections = detector.detect_people(frame)

        # 5) Draw boxes
        for det in person_detections:
            bbox = det['bbox']
            conf = det['conf']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("UFC Fight Analysis - Detection Only", frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()