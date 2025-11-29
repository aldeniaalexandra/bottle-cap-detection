import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

def get_bbox_color(image: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """
    Extract the average color (HSV) of the bounding box center.
    
    Args:
        image: The image in BGR format.
        bbox: Normalized YOLO bbox (x_center, y_center, width, height).
        
    Returns:
        Average (H, S, V) of the center crop of the bbox.
    """
    h, w, _ = image.shape
    xc, yc, bw, bh = bbox
    
    # Convert normalized to pixel coordinates
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    
    # Clip to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0)
        
    # Crop the object
    crop = image[y1:y2, x1:x2]
    
    # Take a center crop to avoid background noise (e.g., 50% of the box)
    ch, cw, _ = crop.shape
    m = 0.25 # margin to ignore
    cx1 = int(cw * m)
    cy1 = int(ch * m)
    cx2 = int(cw * (1 - m))
    cy2 = int(ch * (1 - m))
    
    if cx2 > cx1 and cy2 > cy1:
        center_crop = crop[cy1:cy2, cx1:cx2]
    else:
        center_crop = crop
        
    # Convert to HSV
    hsv_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    
    # Calculate average color
    avg_hsv = np.mean(hsv_crop, axis=(0, 1))
    return tuple(avg_hsv)

def determine_color_class(hsv: Tuple[float, float, float]) -> int:
    """
    Determine the class ID based on HSV values.
    
    Args:
        hsv: (H, S, V) tuple. H in [0, 179], S, V in [0, 255].
        
    Returns:
        Class ID:
        0: Light Blue
        1: Dark Blue
        2: Others
    """
    h, s, v = hsv
    
    # Heuristic thresholds (To be calibrated in the notebook)
    # Blue is typically around H=100-130 in OpenCV (0-179 scale)
    # Light blue might have lower Saturation or higher Value?
    # Dark blue might have higher Saturation and lower Value?
    
    # Placeholder logic - needs calibration
    if 90 <= h <= 140:
        if v > 150: # Brighter -> Light Blue
            return 0
        else: # Darker -> Dark Blue
            return 1
    else:
        return 2 # Others
