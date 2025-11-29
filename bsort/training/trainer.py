import yaml
from ultralytics import YOLO
import wandb
from typing import Dict, Any

def train_model(cfg: Dict[str, Any]):
    """
    Train the YOLO model based on the configuration.
    
    Args:
        cfg: Configuration dictionary.
    """
    # Initialize WandB
    if cfg.get("wandb", {}).get("enabled", False):
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            config=cfg
        )

    # Load model
    model_name = cfg["model"]["name"] # e.g., 'yolov8n.pt'
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=cfg["data"]["yaml_path"],
        epochs=cfg["training"]["epochs"],
        batch=cfg["training"]["batch_size"],
        imgsz=cfg["training"]["imgsz"],
        lr0=cfg["training"]["lr0"],
        project=cfg["training"]["project_dir"],
        name=cfg["training"]["run_name"],
        device=cfg["training"].get("device", "cpu"),
    )
    
    # Validate
    model.val()
    
    # Export
    if cfg["export"]["enabled"]:
        model.export(format=cfg["export"]["format"])

    if wandb.run:
        wandb.finish()
