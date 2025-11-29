import os
import glob
import shutil
import cv2
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from bsort.utils import get_bbox_color, determine_color_class
from pathlib import Path

def prepare_dataset(source_dir: str, output_dir: str, val_size: float = 0.2):
    """
    Prepare the dataset for YOLO training.
    
    Args:
        source_dir: Directory containing raw images and txt files.
        output_dir: Directory to save the prepared dataset.
        val_size: Fraction of validation data.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create directories
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    images = list(source_path.glob("*.jpg"))
    train_imgs, val_imgs = train_test_split(images, test_size=val_size, random_state=42)
    
    def process_files(file_list, split):
        for img_file in file_list:
            txt_file = img_file.with_suffix(".txt")
            if not txt_file.exists():
                continue
                
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            # Read labels
            with open(txt_file, "r") as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                # Original class is ignored, we re-classify
                bbox = [float(x) for x in parts[1:]]
                
                # Determine color class
                hsv = get_bbox_color(img, tuple(bbox))
                cls_id = determine_color_class(hsv)
                
                new_lines.append(f"{cls_id} {' '.join(parts[1:])}\n")
                
            # Save image
            shutil.copy(img_file, output_path / "images" / split / img_file.name)
            
            # Save new label
            with open(output_path / "labels" / split / txt_file.name, "w") as f:
                f.writelines(new_lines)
                
    print("Processing training set...")
    process_files(train_imgs, "train")
    print("Processing validation set...")
    process_files(val_imgs, "val")
    
    # Create dataset.yaml
    dataset_yaml = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "light_blue",
            1: "dark_blue",
            2: "others"
        }
    }
    
    with open("dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)
        
    print("Dataset preparation complete.")

if __name__ == "__main__":
    prepare_dataset("datasets", "prepared_dataset")
