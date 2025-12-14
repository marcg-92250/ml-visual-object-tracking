"""
File name: main.py
Description: Main script for IoU-based Multiple Object Tracking
Python Version: 3.7+
"""

import os
import sys
import cv2
import numpy as np
from collections import defaultdict

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from data_loader import load_detections
from visualization import get_color, draw_track
from iou_tracker import IOUTracker


def save_results(results, output_file):
    """
    Save tracking results in MOT Challenge format.
    
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 
            <conf>, <x>, <y>, <z>
    
    Args:
        results: List of result dictionaries with keys: 'frame', 'id', 'bbox', 'conf'
        output_file: Path to output file
    """
    # Sort results by frame, then by id
    results.sort(key=lambda x: (x['frame'], x['id']))
    
    with open(output_file, 'w') as f:
        for r in results:
            x, y, w_box, h_box = r['bbox']
            # Format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
            line = f"{r['frame']},{r['id']},{x:.2f},{y:.2f},{w_box:.2f},{h_box:.2f},1,-1,-1,-1\n"
            f.write(line)


def main():
    """
    Main function for IoU-based tracking.
    """
    # Paths - relative to tp2 directory
    base_path = '../ADL-Rundle-6'
    
    img_dir = os.path.join(base_path, 'img1')
    det_file = os.path.join(base_path, 'det', 'Yolov5l', 'det.txt')
    
    # Check if paths exist
    if not os.path.exists(base_path):
        print(f"Error: ADL-Rundle-6 directory not found at {base_path}")
        sys.exit(1)
    
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found: {img_dir}")
        sys.exit(1)
    
    if not os.path.exists(det_file):
        print(f"Error: Detection file not found: {det_file}")
        sys.exit(1)
    
    print("Loading detections...")
    detections = load_detections(det_file)
    print(f"Loaded detections for {len(detections)} frames")
    
    # Initialize tracker
    tracker = IOUTracker(iou_threshold=0.3, max_age=5)
    
    # Get sorted list of image files
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    
    if not img_files:
        print(f"Error: No images found in {img_dir}")
        sys.exit(1)
    
    print(f"Found {len(img_files)} images")
    
    # Get frame dimensions from first image
    first_img_path = os.path.join(img_dir, img_files[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"Error: Could not read first image: {first_img_path}")
        sys.exit(1)
    
    h, w = first_img.shape[:2]
    print(f"Frame dimensions: {w}x{h}")
    
    # Setup video writer
    output_video = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (w, h))
    
    if not out.isOpened():
        print("Warning: Could not open video writer. Continuing without video output.")
        out = None
    
    # Store results
    results = []
    
    print("Starting tracking...")
    print("Press 'q' to quit (if display window is open)")
    
    # Process each frame
    for frame_idx, img_file in enumerate(img_files, start=1):
        frame_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_idx}: {frame_path}")
            continue
        
        # Get detections for this frame (filter by confidence)
        frame_dets = [
            d for d in detections.get(frame_idx, [])
            if d['conf'] > 0.7  # Confidence threshold
        ]
        
        # Update tracker
        tracks = tracker.update(frame_dets)
        
        # Draw tracks and store results
        for track in tracks:
            x, y, w_box, h_box = track.bbox
            x, y, w_box, h_box = int(x), int(y), int(w_box), int(h_box)
            
            # Draw track
            color = get_color(track.id)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            
            # Draw label
            label = f'ID:{track.id}'
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x, y - label_h - 10),
                (x + label_w, y),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Store result
            results.append({
                'frame': frame_idx,
                'id': track.id,
                'bbox': track.bbox,
                'conf': track.conf
            })
        
        # Add frame number
        cv2.putText(
            frame,
            f"Frame: {frame_idx}/{len(img_files)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Display frame
        cv2.imshow('IoU Tracking', frame)
        
        # Write to video
        if out is not None:
            out.write(frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Progress update
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{len(img_files)} frames")
    
    # Cleanup
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Save results
    output_file = 'ADL-Rundle-6.txt'
    save_results(results, output_file)
    
    print(f"\nTracking completed!")
    print(f"Results saved to {output_file}")
    if out is not None:
        print(f"Video saved to {output_video}")
    print(f"Total tracks: {len(set(r['id'] for r in results))}")
    print(f"Total detections: {len(results)}")


if __name__ == '__main__':
    main()

