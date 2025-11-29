import time
import cv2
from ultralytics import YOLO
from typing import Dict, Any

def run_inference(cfg: Dict[str, Any]):
    """
    Run inference using the trained model.
    
    Args:
        cfg: Configuration dictionary.
    """
    model_path = cfg["inference"]["model_path"]
    source = cfg["inference"]["source"]
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    # For benchmarking, we might want to loop
    if cfg["inference"].get("benchmark", False):
        print("Benchmarking...")
        img = cv2.imread(source) # Assume source is an image for benchmark
        if img is None:
            print("Invalid source for benchmark")
            return
            
        # Warmup
        for _ in range(10):
            model(img, verbose=False)
            
        start_time = time.time()
        iters = 100
        for _ in range(iters):
            model(img, verbose=False)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iters * 1000
        print(f"Average Inference Time: {avg_time:.2f} ms")
        
    else:
        results = model(source, save=True, conf=cfg["inference"]["conf"])
        for result in results:
            print(f"Detected {len(result.boxes)} objects.")

    # Log to WandB if enabled
    if cfg.get("wandb", {}).get("enabled", False) and cfg["inference"].get("benchmark", False):
        import wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            config=cfg,
            job_type="inference"
        )
        wandb.log({"inference/avg_time_ms": avg_time})
        print(f"Logged inference time to WandB project: {cfg['wandb']['project']}")
        wandb.finish()
