"""
File name: evaluate.py
Description: Evaluation metrics for multi-object tracking (MOTA, IDF1, etc.)
Python Version: 3.7+
"""

import os
import sys
import time
import numpy as np
from collections import defaultdict

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../tp2/utils'))
from data_loader import load_detections, load_ground_truth

from reid_extractor import ReIDExtractor
from reid_tracker import IOUReIDKalmanTracker
import cv2


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
    
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def evaluate_tracking(results, gt, iou_threshold=0.5):
    """
    Evaluate tracking performance using MOT metrics.
    
    Args:
        results: Dictionary mapping frame to list of result dicts with keys: 'id', 'bbox'
        gt: Dictionary mapping frame to list of GT dicts with keys: 'id', 'bbox'
        iou_threshold: IoU threshold for considering a match (default: 0.5)
    
    Returns:
        Dictionary with evaluation metrics
    """
    id_switches = 0
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Track last matched ID for each GT ID to detect ID switches
    last_match = {}
    
    # Get all frames
    all_frames = sorted(set(results.keys()) | set(gt.keys()))
    
    for frame in all_frames:
        frame_results = results.get(frame, [])
        frame_gt = gt.get(frame, [])
        
        total_gt += len(frame_gt)
        
        # Match results to ground truth
        matched_gt = set()
        matched_res = set()
        
        for i, res in enumerate(frame_results):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching GT object
            for j, g in enumerate(frame_gt):
                if j in matched_gt:
                    continue
                
                iou = compute_iou(res['bbox'], g['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_res.add(i)
                total_tp += 1
                
                # Check for ID switch
                gt_id = frame_gt[best_gt_idx]['id']
                pred_id = res['id']
                
                if gt_id in last_match:
                    if last_match[gt_id] != pred_id:
                        id_switches += 1
                
                last_match[gt_id] = pred_id
            else:
                total_fp += 1
        
        # Unmatched GT objects are false negatives
        total_fn += len(frame_gt) - len(matched_gt)
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    # IDF1: F1 score for ID preservation
    idf1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MOTA: Multiple Object Tracking Accuracy
    mota = 1 - (total_fn + total_fp + id_switches) / total_gt if total_gt > 0 else 0
    
    return {
        'MOTA': mota,
        'IDF1': idf1,
        'Precision': precision,
        'Recall': recall,
        'ID_Switches': id_switches,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn
    }


def run_tracker_and_evaluate():
    """
    Run tracker and evaluate performance.
    """
    # Paths - relative to tp4 directory
    base_path = '../ADL-Rundle-6'
    img_dir = os.path.join(base_path, 'img1')
    det_file = os.path.join(base_path, 'det', 'Yolov5l', 'det.txt')
    gt_file = os.path.join(base_path, 'gt', 'gt.txt')
    reid_model = '../reid_osnet_x025_market1501.onnx'
    
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
    
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found: {gt_file}")
        sys.exit(1)
    
    if not os.path.exists(reid_model):
        print(f"Error: ReID model not found at {reid_model}")
        sys.exit(1)
    
    print("Loading data...")
    detections = load_detections(det_file)
    gt = load_ground_truth(gt_file)
    print(f"Loaded detections for {len(detections)} frames")
    print(f"Loaded ground truth for {len(gt)} frames")
    
    print("Initializing ReID extractor...")
    reid_extractor = ReIDExtractor(reid_model)
    
    print("Initializing tracker...")
    tracker = IOUReIDKalmanTracker(
        iou_threshold=0.3,
        max_age=30,
        alpha=0.6,
        beta=0.4
    )
    
    # Get sorted list of image files
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    
    print(f"Found {len(img_files)} images")
    
    # Store results in format for evaluation
    results = defaultdict(list)
    
    print("Running tracker...")
    start_time = time.time()
    
    # Process each frame
    for frame_idx, img_file in enumerate(img_files, start=1):
        frame_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            continue
        
        # Get detections for this frame
        frame_dets = [
            d for d in detections.get(frame_idx, [])
            if d['conf'] > 0.7
        ]
        
        # Extract ReID features
        bboxes = [det['bbox'] for det in frame_dets]
        features = reid_extractor.extract_batch(frame, bboxes)
        
        # Update tracker
        tracks = tracker.update(frame_dets, features)
        
        # Store results
        for track in tracks:
            results[frame_idx].append({
                'id': track.id,
                'bbox': track.get_bbox()
            })
        
        # Progress update
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{len(img_files)} frames")
    
    elapsed = time.time() - start_time
    fps = len(img_files) / elapsed if elapsed > 0 else 0
    
    print("Evaluating tracking performance...")
    metrics = evaluate_tracking(results, gt)
    
    # Print results
    print("\n" + "=" * 50)
    print("TRACKING EVALUATION RESULTS")
    print("=" * 50)
    print(f"MOTA:        {metrics['MOTA']:.4f}")
    print(f"IDF1:        {metrics['IDF1']:.4f}")
    print(f"Precision:   {metrics['Precision']:.4f}")
    print(f"Recall:      {metrics['Recall']:.4f}")
    print(f"ID Switches: {metrics['ID_Switches']}")
    print(f"TP:          {metrics['TP']}")
    print(f"FP:          {metrics['FP']}")
    print(f"FN:          {metrics['FN']}")
    print("-" * 50)
    print(f"FPS:         {fps:.2f}")
    print(f"Total Time:  {elapsed:.2f}s")
    print("=" * 50)
    
    return metrics, fps


if __name__ == '__main__':
    run_tracker_and_evaluate()

