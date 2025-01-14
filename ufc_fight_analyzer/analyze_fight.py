#!/usr/bin/env python3
"""
analyze_fight.py

This script uses a YOLO model with pose estimation to detect and track fighters in a video,
and displays the annotated frames with bounding boxes, keypoints, and skeletons.
"""

import os
import cv2
import yaml
import numpy as np
import torch
from ultralytics import YOLO

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def initialize_model(model_path: str, use_pose: bool) -> YOLO:
    try:
        if use_pose:
            model = YOLO(model_path, task='pose')
            print("Model initialized with task='pose'.")
        else:
            model = YOLO(model_path)
            print("Model initialized without specifying task.")
    except TypeError:
        model = YOLO(model_path)
        print("Model initialized without specifying task (fallback).")
    return model

def setup_device(model: YOLO) -> torch.device:
    if torch.cuda.is_available():
        model.to('cuda')
        print("Model moved to GPU.")
        return torch.device("cuda")
    else:
        print("Model is using CPU.")
        return torch.device("cpu")

def process_video(model: YOLO, video_path: str, tracker_type: str, conf_threshold: float, iou_threshold: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    processing_times = []
    frame_count = 0  # Frame counter

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = cv2.getTickCount() / cv2.getTickFrequency()
        results = model.track(
            frame, 
            persist=True, 
            tracker=tracker_type, 
            conf=conf_threshold, 
            iou=iou_threshold,
            pose=True,
            verbose=False
        )
        end_time = cv2.getTickCount() / cv2.getTickFrequency()
        processing_time = (end_time - start_time) * 1000  # in ms
        processing_times.append(processing_time)
        avg_processing_time = np.mean(processing_times)
        fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0

        # Extract and print keypoints for each frame
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.xy.cpu().numpy() if hasattr(results[0].keypoints, 'cpu') else results[0].keypoints.xy
            normalized_keypoints = keypoints_data.flatten() / frame.shape[1]

            print(f"Frame {frame_count}: Keypoints data:")
            print(normalized_keypoints)
            print("-" * 50)
        else:
            print(f"Frame {frame_count}: No keypoints detected.")

        # Annotate and display the frame
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame,
                    f"Inference: {avg_processing_time:.1f}ms ({fps:.1f} FPS)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)
        cv2.imshow("UFC Fight Analysis - Detection + Tracking + Pose", annotated_frame)

        # Wait 10ms and check if 'q' or 'Esc' is pressed
        key = cv2.waitKey(10) & 0xFF
        if key in [ord('q'), 27]:
            print("Exit key pressed.")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    config_file = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    config = load_config(config_file)

    model_path = config["model_path"]
    conf_threshold = config["confidence_threshold"]
    iou_threshold = config["iou_threshold"]
    tracker_type = config.get("tracker", "bytetrack.yaml")
    use_pose = config.get("pose_estimation", False)
    
    model = initialize_model(model_path, use_pose)
    device = setup_device(model)

    video_path = os.path.join(os.path.dirname(__file__), "..", "ufc_fights", "pereira_1min.mp4")
    process_video(model, video_path, tracker_type, conf_threshold, iou_threshold)

if __name__ == "__main__":
    main()