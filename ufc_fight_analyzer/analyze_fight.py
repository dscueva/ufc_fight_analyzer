# ufc_fight_analyzer/analyze_fight.py

import cv2
import yaml
import os
from ultralytics import YOLO
import numpy as np
import torch  # Import torch for device checking

def main():
    # 1) Load config
    config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_path = config["model_path"]
    conf_threshold = config["confidence_threshold"]
    iou_threshold = config["iou_threshold"]
    tracker_type = config.get("tracker", "bytetrack.yaml")  # Default to ByteTrack
    pose_estimation = config.get("pose_estimation", False)
    keypoint_conf_threshold = config.get("keypoint_conf_threshold", 0.3)  # Lowered for testing
    
    # 2) Initialize the YOLO model
    try:
        # Attempt to initialize with task='pose' if supported
        model = YOLO(model_path, task='pose')
        print("Model initialized with task='pose'.")
    except TypeError:
        # If task parameter is not supported, initialize normally
        model = YOLO(model_path)
        print("Model initialized without specifying task.")
    
    # Check if CUDA (GPU) is available and set the device accordingly
    if torch.cuda.is_available():
        model.to('cuda')  # Move the model to GPU
        print("Model moved to GPU.")
    else:
        print("Model is using CPU.")
    
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
        
        # Optional: Resize frame for faster processing
        # frame = cv2.resize(frame, (640, 480))
        
        # 4) Perform tracking and pose estimation
        results = model.track(
            frame, 
            persist=True, 
            tracker=tracker_type, 
            conf=conf_threshold, 
            iou=iou_threshold,
            pose=True,             # Enable pose estimation
            verbose=False
        )
        
        # 5) Annotate the frame using plot()
        annotated_frame = results[0].plot()  # Automatically draws boxes, keypoints, skeletons
        
        # 6) Display the annotated frame
        cv2.imshow("UFC Fight Analysis - Detection + Tracking + Pose", annotated_frame)
        
        # 7) Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()