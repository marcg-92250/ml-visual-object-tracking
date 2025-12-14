"""
File name: iou_tracker.py
Description: IoU-based Multiple Object Tracker using Hungarian Algorithm
Python Version: 3.7+
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    """
    Represents a single tracked object.
    """
    _id_counter = 0
    
    def __init__(self, bbox, conf):
        """
        Initialize a new track.
        
        Args:
            bbox: Bounding box (x, y, width, height)
            conf: Detection confidence score
        """
        Track._id_counter += 1
        self.id = Track._id_counter
        self.bbox = bbox
        self.conf = conf
        self.age = 0  # Number of frames since last update
        self.hits = 1  # Number of times this track has been updated
    
    def update(self, bbox, conf):
        """
        Update track with new detection.
        
        Args:
            bbox: New bounding box
            conf: New confidence score
        """
        self.bbox = bbox
        self.conf = conf
        self.age = 0
        self.hits += 1
    
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


def compute_iou_matrix(tracks, detections):
    """
    Compute IoU matrix between tracks and detections.
    
    Args:
        tracks: List of Track objects
        detections: List of detection dictionaries with 'bbox' key
    
    Returns:
        IoU matrix of shape (n_tracks, n_detections)
    """
    n_tracks = len(tracks)
    n_dets = len(detections)
    
    if n_tracks == 0 or n_dets == 0:
        return np.empty((n_tracks, n_dets))
    
    iou_matrix = np.zeros((n_tracks, n_dets))
    
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou_matrix[i, j] = compute_iou(track.bbox, det['bbox'])
    
    return iou_matrix


class IOUTracker:
    """
    IoU-based Multiple Object Tracker using Hungarian Algorithm.
    """
    
    def __init__(self, iou_threshold=0.3, max_age=5):
        """
        Initialize the IoU Tracker.
        
        Args:
            iou_threshold: Minimum IoU for a valid match (default: 0.3)
            max_age: Maximum number of frames a track can be unmatched before deletion (default: 5)
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        Track._id_counter = 0  # Reset ID counter
    
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with keys 'bbox' and 'conf'
        
        Returns:
            List of active Track objects
        """
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
        
        # Compute IoU similarity matrix
        iou_matrix = compute_iou_matrix(self.tracks, detections)
        
        # Convert IoU to cost (higher IoU = lower cost)
        # Use Hungarian algorithm to find optimal assignment
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

