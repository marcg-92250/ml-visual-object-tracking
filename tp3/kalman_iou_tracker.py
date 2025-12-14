"""
File name: kalman_iou_tracker.py
Description: Kalman-Guided IoU Tracker combining Kalman Filter prediction with IoU association
Python Version: 3.7+
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from kalman_filter import KalmanFilter


class Track:
    """
    Represents a single tracked object with Kalman Filter.
    """
    _id_counter = 0
    
    def __init__(self, bbox, conf):
        """
        Initialize a new track with Kalman Filter.
        
        Args:
            bbox: Bounding box (x, y, width, height)
            conf: Detection confidence score
        """
        Track._id_counter += 1
        self.id = Track._id_counter
        self.conf = conf
        self.age = 0
        self.hits = 1
        
        # Store bounding box dimensions
        x, y, w, h = bbox
        self.w = w
        self.h = h
        
        # Initialize Kalman Filter with centroid
        cx = x + w / 2
        cy = y + h / 2
        
        self.kf = KalmanFilter(dt=1/30, std_acc=1, x_std_meas=1, y_std_meas=1)
        self.kf.init_state(cx, cy)
    
    def predict(self):
        """
        Predict the next centroid position.
        
        Returns:
            Predicted centroid (cx, cy) as tuple
        """
        return self.kf.predict()
    
    def update(self, bbox, conf):
        """
        Update track with new detection.
        
        Args:
            bbox: New bounding box
            conf: New confidence score
        """
        x, y, w, h = bbox
        self.w = w
        self.h = h
        self.conf = conf
        
        # Update Kalman Filter with new centroid
        cx = x + w / 2
        cy = y + h / 2
        self.kf.update(cx, cy)
        
        self.age = 0
        self.hits += 1
    
    def get_bbox(self):
        """
        Get current bounding box from Kalman Filter state.
        
        Returns:
            Bounding box (x, y, width, height)
        """
        cx, cy = self.kf.x[0, 0], self.kf.x[1, 0]
        x = cx - self.w / 2
        y = cy - self.h / 2
        return (x, y, self.w, self.h)
    
    def get_predicted_bbox(self):
        """
        Get predicted bounding box for next frame.
        
        Returns:
            Predicted bounding box (x, y, width, height)
        """
        cx, cy = self.predict()
        x = cx - self.w / 2
        y = cy - self.h / 2
        return (x, y, self.w, self.h)
    
    def mark_missed(self):
        """Mark that this track was not matched in the current frame."""
        self.age += 1


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box (x, y, width, height)
        box2: Second bounding box (x, y, width, height)
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    
    # Calculate bottom-right corners
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


class IOUKalmanTracker:
    """
    Kalman-Guided IoU Tracker combining Kalman Filter prediction with IoU association.
    """
    
    def __init__(self, iou_threshold=0.3, max_age=10):
        """
        Initialize the Kalman-IoU Tracker.
        
        Args:
            iou_threshold: Minimum IoU for a valid match (default: 0.3)
            max_age: Maximum number of frames a track can be unmatched before deletion (default: 10)
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        Track._id_counter = 0  # Reset ID counter
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        First predicts the state of each track using Kalman Filter,
        then associates detections to tracks using IoU.
        
        Args:
            detections: List of detection dictionaries with keys 'bbox' and 'conf'
        
        Returns:
            List of active Track objects
        """
        # Predict state for all existing tracks
        for track in self.tracks:
            track.predict()
        
        # If no existing tracks, create new tracks for all detections
        if not self.tracks:
            for det in detections:
                self.tracks.append(Track(det['bbox'], det['conf']))
            return self.tracks
        
        # If no detections, mark all tracks as missed
        if not detections:
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return self.tracks
        
        # Compute IoU matrix between predicted tracks and detections
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))
        
        for i, track in enumerate(self.tracks):
            # Use predicted bounding box for association
            pred_bbox = track.get_bbox()
            for j, det in enumerate(detections):
                iou_matrix[i, j] = compute_iou(pred_bbox, det['bbox'])
        
        # Convert IoU to cost (higher IoU = lower cost)
        cost_matrix = 1 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Process matches
        matched_tracks = set()
        matched_dets = set()
        
        for row, col in zip(row_indices, col_indices):
            # Only accept matches above IoU threshold
            if iou_matrix[row, col] >= self.iou_threshold:
                self.tracks[row].update(
                    detections[col]['bbox'],
                    detections[col]['conf']
                )
                matched_tracks.add(row)
                matched_dets.add(col)
        
        # Mark unmatched tracks as missed
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_missed()
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(Track(det['bbox'], det['conf']))
        
        # Remove tracks that exceed max_age
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        
        return self.tracks

