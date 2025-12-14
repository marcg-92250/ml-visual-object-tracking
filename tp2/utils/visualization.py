"""
File name: visualization.py
Description: Utility functions for visualization
Python Version: 3.7+
"""

import cv2
import numpy as np


def get_color(track_id):
    """
    Generate a consistent color for a track ID.
    
    Args:
        track_id: Unique track identifier
    
    Returns:
        BGR color tuple (B, G, R)
    """
    np.random.seed(track_id * 42)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))


def draw_track(frame, track, color=None):
    """
    Draw a track on a frame.
    
    Args:
        frame: Input frame (BGR image)
        track: Track object with attributes: id, bbox, conf
        color: Optional color tuple (B, G, R). If None, generates color from track ID.
    
    Returns:
        Frame with track drawn
    """
    if color is None:
        color = get_color(track.id)
    
    x, y, w, h = track.bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw label with track ID
    label = f'ID:{track.id}'
    (label_w, label_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    
    # Draw label background
    cv2.rectangle(
        frame,
        (x, y - label_h - 10),
        (x + label_w, y),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame,
        label,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    return frame

