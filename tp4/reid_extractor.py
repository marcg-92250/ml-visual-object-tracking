"""
File name: reid_extractor.py
Description: Re-Identification feature extractor using OSNet model
Python Version: 3.7+
"""

import cv2
import numpy as np
import onnxruntime as ort
import os


class ReIDExtractor:
    """
    Re-Identification feature extractor using pre-trained OSNet model.
    """
    
    def __init__(self, model_path):
        """
        Initialize the ReID extractor.
        
        Args:
            model_path: Path to ONNX model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ReID model not found: {model_path}")
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        
        # ROI dimensions (Market1501 dataset format)
        self.roi_height = 128
        self.roi_width = 64
        
        # Normalization parameters (ImageNet statistics)
        self.roi_means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.roi_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess_patch(self, im_crop):
        """
        Preprocess an image patch for ReID feature extraction.
        
        Args:
            im_crop: Input image patch (BGR format)
        
        Returns:
            Preprocessed patch ready for model input
        """
        # Resize to model input size
        roi_input = cv2.resize(im_crop, (self.roi_width, self.roi_height))
        
        # Convert BGR to RGB
        roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        
        # Normalize: (pixel / 255.0 - mean) / std
        roi_input = (np.asarray(roi_input, dtype=np.float32) / 255.0 - self.roi_means) / self.roi_stds
        
        # Change from HWC to CHW format (channels first)
        roi_input = np.moveaxis(roi_input, -1, 0)
        
        return roi_input.astype(np.float32)
    
    def extract(self, frame, bbox):
        """
        Extract ReID features from a bounding box in a frame.
        
        Args:
            frame: Input frame (BGR image)
            bbox: Bounding box (x, y, width, height)
        
        Returns:
            Feature vector (normalized to unit length)
        """
        x, y, w, h = bbox
        
        # Extract patch with boundary checks
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
        
        if x2 <= x1 or y2 <= y1:
            # Invalid bounding box, return zero vector
            return np.zeros(512, dtype=np.float32)
        
        patch = frame[y1:y2, x1:x2]
        
        if patch.size == 0:
            return np.zeros(512, dtype=np.float32)
        
        # Preprocess patch
        preprocessed = self.preprocess_patch(patch)
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
        
        # Extract features
        features = self.session.run(None, {self.input_name: preprocessed})[0]
        features = features.flatten()
        
        # L2 normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / (norm + 1e-6)
        
        return features
    
    def extract_batch(self, frame, bboxes):
        """
        Extract ReID features for multiple bounding boxes.
        
        Args:
            frame: Input frame (BGR image)
            bboxes: List of bounding boxes
        
        Returns:
            List of feature vectors
        """
        if not bboxes:
            return []
        
        features = []
        for bbox in bboxes:
            feat = self.extract(frame, bbox)
            features.append(feat)
        
        return features


def euclidean_distance(feat1, feat2):
    """
    Compute Euclidean distance between two feature vectors.
    
    Args:
        feat1: First feature vector
        feat2: Second feature vector
    
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(feat1 - feat2)


def cosine_distance(feat1, feat2):
    """
    Compute cosine distance between two feature vectors.
    
    Args:
        feat1: First feature vector
        feat2: Second feature vector
    
    Returns:
        Cosine distance (1 - cosine similarity)
    """
    return 1 - np.dot(feat1, feat2)


def normalized_similarity(distance):
    """
    Convert distance to normalized similarity score.
    
    Normalized Similarity = 1 / (1 + distance)
    
    Args:
        distance: Distance value
    
    Returns:
        Similarity score between 0 and 1
    """
    return 1 / (1 + distance)

