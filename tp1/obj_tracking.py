"""
File name: obj_tracking.py
Description: Main script for single object tracking using Kalman Filter
Python Version: 3.7+
"""

import cv2
import numpy as np
import os
import sys

from detector import detect
from kalman_filter import KalmanFilter


def main():
    """
    Main function for object tracking with Kalman Filter.
    """
    # Initialize Kalman Filter with specified parameters
    kf = KalmanFilter(
        dt=0.1,           # Time step
        u_x=1,            # Acceleration in x-direction
        u_y=1,            # Acceleration in y-direction
        std_acc=1,        # Process noise standard deviation
        x_std_meas=0.1,   # Measurement noise standard deviation in x
        y_std_meas=0.1   # Measurement noise standard deviation in y
    )
    
    # Path to video file - relative to tp1 directory
    video_path = '../2D_Kalman-Filter_TP1/video/randomball.avi'
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please ensure the video file is in 2D_Kalman-Filter_TP1/video/")
        sys.exit(1)
    
    # Create video capture object
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)
    
    # Store trajectory for visualization
    trajectory = []
    
    frame_count = 0
    
    print("Starting object tracking...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect objects in the frame
        centers = detect(frame)
        
        # Predict next state
        predicted = kf.predict()
        pred_x, pred_y = int(predicted[0][0]), int(predicted[1][0])
        
        # If centroid is detected, update the filter
        if centers:
            # Use the first detected center (assuming single object tracking)
            measured = centers[0]
            mx, my = int(measured[0][0]), int(measured[1][0])
            
            # Draw detected circle (green color)
            cv2.circle(frame, (mx, my), 10, (0, 255, 0), 2)
            
            # Update Kalman filter with measurement
            estimated = kf.update([mx, my])
            est_x, est_y = int(estimated[0][0]), int(estimated[1][0])
            
            # Add to trajectory
            trajectory.append((est_x, est_y))
        else:
            # No detection, use predicted position
            est_x, est_y = pred_x, pred_y
            trajectory.append((est_x, est_y))
        
        # Draw predicted object position (blue rectangle)
        cv2.rectangle(
            frame,
            (pred_x - 15, pred_y - 15),
            (pred_x + 15, pred_y + 15),
            (255, 0, 0),
            2
        )
        
        # Draw estimated object position (red rectangle)
        cv2.rectangle(
            frame,
            (est_x - 15, est_y - 15),
            (est_x + 15, est_y + 15),
            (0, 0, 255),
            2
        )
        
        # Draw trajectory (tracking path) in yellow
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(
                    frame,
                    trajectory[i - 1],
                    trajectory[i],
                    (0, 255, 255),
                    2
                )
        
        # Add text information
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            "Green: Detection | Blue: Prediction | Red: Estimation",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Display the frame
        cv2.imshow('Kalman Tracking', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Tracking completed. Processed {frame_count} frames.")
    print(f"Trajectory length: {len(trajectory)} points")


if __name__ == '__main__':
    main()

