# Bottle Cap Sorter (bsort)

**Bottle Cap Sorter (bsort)** is a real-time computer vision system designed to detect and classify bottle caps into three categories: **Light Blue**, **Dark Blue**, and **Others**. The project focuses on high-speed inference (target: 5-10ms) suitable for edge devices like the Raspberry Pi 5.

## Problem Description
The goal is to process a dataset of bottle caps with bounding boxes and classify them by color. The challenge lies in achieving high accuracy while maintaining extremely low latency for real-time applications on edge hardware.
- **Input**: Images with bottle caps.
- **Output**: Bounding boxes with color classification (Light Blue, Dark Blue, Others).
- **Constraints**: Inference time ≤ 5-10ms per frame on Raspberry Pi 5.
- **Trade-off**: Balancing model complexity (accuracy) with speed (latency).

## Approach Overview
We utilize **YOLOv8n** (Nano) as the baseline architecture due to its balance of speed and accuracy.
1.  **Preprocessing & Relabeling**: Raw data is processed to extract bounding boxes and re-label them based on HSV color analysis (Light Blue vs Dark Blue vs Others).
2.  **Training**: Fine-tuning YOLOv8n on the re-labeled dataset.
3.  **Optimization**: Exporting the model to **TFLite** (or NCNN) for optimized edge inference.
4.  **Evaluation**: Measuring mAP and inference speed on target hardware.

## Repository Structure
```
.
├── bsort/                  # Source code package
│   ├── main.py             # CLI entry point
│   ├── data_prep.py        # Dataset preparation script
│   ├── training/           # Training logic
│   ├── inference/          # Inference logic
│   └── utils.py            # Utility functions
├── notebooks/
│   └── colab_demo.ipynb    # Colab notebook for demo/experiments
├── config/
│   └── settings.yaml       # Configuration file
├── datasets/               # Raw dataset directory
├── prepared_dataset/       # Processed dataset (generated)
├── tests/                  # Unit tests
├── Dockerfile              # Docker build definition
├── pyproject.toml          # Project dependencies and metadata
└── README.md               # Project documentation
```

## Installation Guide

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for training)

### Steps
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/aldeniaalexandra/bottle-cap-detection.git
    cd bottle-cap-detection
    ```

2.  **Set up environment** (using `pip`):
    ```bash
    # Create virtual environment (optional but recommended)
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install package
    pip install .
    
    # For development (testing, linting):
    pip install .[dev]
    ```

## Dataset Preparation
1.  **Download Dataset**: Place your raw dataset in the `datasets/` folder.
2.  **Prepare Data**: Run the preparation script to split and re-label data based on color.
    ```bash
    python bsort/data_prep.py
    ```
    This will create a `prepared_dataset/` directory with `images/` and `labels/` for train/val splits, and generate `dataset.yaml`.

## Configuration (`config/settings.yaml`)
The system is configured via `config/settings.yaml`. Key fields:

-   **data**: Path to `dataset.yaml`.
-   **model**: Model architecture (e.g., `yolov8n.pt`).
-   **training**:
    -   `epochs`: Number of training epochs.
    -   `batch_size`: Batch size.
    -   `imgsz`: Input image size (e.g., 640).
-   **inference**:
    -   `conf`: Confidence threshold.
    -   `source`: Path to image/video for inference.
-   **export**: Settings for model export (e.g., `tflite`).
-   **wandb**: Weights & Biases logging configuration.

## How to Use – CLI `bsort`

### Training
To train the model:
```bash
bsort train --config config/settings.yaml
```
This command loads the dataset defined in `dataset.yaml`, trains the YOLO model, logs metrics to WandB (if enabled), and saves the best model to `runs/train/`.

### Inference
To run inference on a single image:
```bash
bsort infer --config config/settings.yaml
```
(Ensure `inference.source` in `settings.yaml` points to your target image).

Output:
-   Annotated image saved in `runs/detect/`.
-   Console output showing detected classes and confidence scores.

## Model Results & Evaluation
(See `notebooks/colab_demo.ipynb` for detailed analysis)

**Note**: Training and evaluation were performed on **Google Colab** using a **Tesla T4 GPU**.

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| Light Blue | N/A* | N/A* | N/A* | N/A* |
| Dark Blue | 0.996 | 1.00 | 0.995 | 0.905 |
| Others | 0.961 | 1.00 | 0.995 | 0.931 |
| **All** | **0.978** | **1.00** | **0.995** | **0.918** |

*\*Note: No 'Light Blue' samples were present in the validation split.*

### Inference Speed
-   **Google Colab (Tesla T4 GPU)**: ~6.8 ms (YOLOv8n)

## Limitations & Future Work
-   **Lighting Conditions**: Color classification based on HSV thresholds may be sensitive to extreme lighting changes.
-   **Dataset Imbalance**: "Others" class might be underrepresented.
-   **Future Work**: Implement quantization (INT8) for further speedup on Edge TPU.

## CI/CD & Quality
This project uses **GitHub Actions** for continuous integration:
-   **Linting**: `black`, `isort`, `pylint`
-   **Testing**: `pytest`
-   **Build**: Docker image build

To run checks locally:
```bash
black bsort tests
isort bsort tests
pylint bsort
pytest
```

## How to Run with Docker
Build the image:
```bash
docker build -t bsort:latest .
```

Run inference inside Docker:
```bash
docker run --rm -v $(pwd):/app bsort:latest \
  bsort infer --config config/settings.yaml
```

## Acknowledgements / References
-   **Ultralytics YOLOv8**: For the object detection framework.
-   **OpenCV**: For image processing.
