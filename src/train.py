"""
YOLO Training Script for Architectural Drawings
Fine-tunes YOLO architectures (YOLOv8, YOLOv11) with domain-specific CAD hyperparameters.
"""

import os
import argparse
from pathlib import Path


def train_model(
    data_yaml: str = "configs/dataset.yaml",
    hyp_yaml: str = "configs/hyp_cad.yaml",
    model_name: str = "yolo11m.pt",
    epochs: int = 150,
    img_size: int = 1024,
    batch_size: int = 8,
    project_dir: str = "runs/train",
    experiment_name: str = "cad_symbol_detector"
):
    """
    Trains YOLO on the tiled architectural electrical dataset.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics must be installed to run training: pip install ultralytics")

    print(f"[INFO] Initializing base model: {model_name}")
    model = YOLO(model_name)

    print(f"[INFO] Starting training on {data_yaml} with image size {img_size} for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        cfg=hyp_yaml if Path(hyp_yaml).exists() else None,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_dir,
        name=experiment_name,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True
    )

    best_weight_path = Path(project_dir) / experiment_name / "weights" / "best.pt"
    print(f"[SUCCESS] Training complete. Best weights saved to: {best_weight_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on Architectural Electrical Symbols")
    parser.add_argument("--data", default="configs/dataset.yaml", help="Path to dataset yaml")
    parser.add_argument("--hyp", default="configs/hyp_cad.yaml", help="Path to hyperparameters yaml")
    parser.add_argument("--model", default="yolo11m.pt", help="Pretrained YOLO model variant")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input tile image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        hyp_yaml=args.hyp,
        model_name=args.model,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch_size=args.batch
    )
