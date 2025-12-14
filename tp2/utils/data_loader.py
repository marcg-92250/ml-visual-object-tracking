"""
File name: data_loader.py
Description: Utility functions for loading MOT Challenge format detections
Python Version: 3.7+
"""

from collections import defaultdict


def load_detections(det_file):
    """
    Load detections from a MOT Challenge format text file.
    
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 
            <conf>, <x>, <y>, <z>
    
    Args:
        det_file: Path to the detection file (det.txt)
    
    Returns:
        Dictionary mapping frame number to list of detections.
        Each detection is a dict with keys: 'bbox', 'conf'
        bbox format: (bb_left, bb_top, bb_width, bb_height)
    """
    detections = defaultdict(list)
    
    try:
        with open(det_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Try space-separated first, then comma-separated
                parts = line.split()
                if len(parts) < 7:
                    parts = line.split(',')
                
                if len(parts) < 7:
                    continue
                
                # Parse fields
                frame = int(float(parts[0]))  # Handle float frame numbers
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                conf = float(parts[6])
                
                detections[frame].append({
                    'bbox': (bb_left, bb_top, bb_width, bb_height),
                    'conf': conf
                })
    except FileNotFoundError:
        print(f"Error: Detection file not found: {det_file}")
        raise
    except Exception as e:
        print(f"Error loading detections: {e}")
        raise
    
    return detections


def load_ground_truth(gt_file):
    """
    Load ground truth annotations from a MOT Challenge format text file.
    
    Args:
        gt_file: Path to the ground truth file (gt.txt)
    
    Returns:
        Dictionary mapping frame number to list of ground truth objects.
        Each object is a dict with keys: 'id', 'bbox'
    """
    gt = defaultdict(list)
    
    try:
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                
                gt[frame].append({
                    'id': track_id,
                    'bbox': (bb_left, bb_top, bb_width, bb_height)
                })
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {gt_file}")
        raise
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        raise
    
    return gt

