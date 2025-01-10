# ufc_fight_analyzer/analyze_fight.py

import cv2
import yaml
import os
from ultralytics import YOLO

def main():
    # 1) Load config
    config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_path = config["model_path"]
    conf_threshold = config["confidence_threshold"]
    iou_threshold = config["iou_threshold"]
    tracker_type = config.get("tracker", "bytetrack.yaml")  # Default to ByteTrack
    
    # 2) Initialize the YOLO model
    model = YOLO(model_path)
    
    # 3) Video path
    video_path = os.path.join(os.path.dirname(__file__), "..", "ufc_fights", "Pereira_Rountree_1_min.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 4) Perform tracking on the current frame
        results = model.track(
            frame, 
            persist=True, 
            tracker=tracker_type, 
            conf=conf_threshold, 
            iou=iou_threshold
        )
        
        # 5) Extract tracking results and draw bounding boxes with IDs
        for r in results:
            for box in r.boxes:
                if box.id is not None:
                    track_id = box.id
                    # YOLO's boxes are in [x_center, y_center, width, height] format
                    x_center, y_center, width, height = box.xywh.cpu().numpy()[0]
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    conf = box.conf.cpu().numpy()[0]
                    cls_id = int(box.cls.cpu().numpy()[0])
                    label = f"ID {track_id} Conf: {conf:.2f}"
                    color = (0, 255, 0)  # Green color for bounding boxes

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Put label above the bounding box
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        color, 
                        2
                    )
                    
                    # Optional: Draw the center point for debugging
                    cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
        
        # 6) Display the frame
        cv2.imshow("UFC Fight Analysis - Detection + Tracking", frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()