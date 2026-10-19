"""
Dataset Preparation & Tile Slicer Utility
Slices full plan drawings and ground-truth bounding boxes into YOLO-format training tiles.
"""

import json
from pathlib import Path
import cv2
import numpy as np

from src.pdf_processor import PlanSheetPreprocessor
from src.tiler import PlanSheetTiler


CLASS_MAP = {
    "ceiling_mounted_light": 0,
    "recessed_down_light": 1,
    "emergency_light": 2,
    "receptacle_duplex": 3,
    "nurse_call": 4,
    "telephone_outlet": 5,
    "tv_outlet": 6,
    "smoke_detector": 7,
    "switch": 8,
    "electrical_panel": 9,
    "junction_box": 10,
    "exit_sign": 11,
    "fire_alarm_strobe": 12
}


def prepare_yolo_tiles(
    image_path: str = "data/raw/sample_plan.jpg",
    gt_json_path: str = "output/sample_detection.json",
    output_dir: str = "data/tiles",
    tile_size: int = 512,
    overlap_ratio: float = 0.25,
    split_ratio: float = 0.8
):
    """Slices a raw plan sheet and its annotations into YOLO training tiles."""
    img_path = Path(image_path)
    json_path = Path(gt_json_path)
    out_base = Path(output_dir)

    if not img_path.exists() or not json_path.exists():
        print(f"[ERROR] Required inputs not found: {img_path} or {json_path}")
        return

    # Load image and GT detections
    preprocessor = PlanSheetPreprocessor()
    image, meta = preprocessor.load_or_rasterize(img_path)
    h, w = image.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert GT detections to bounding box dicts
    global_boxes = []
    for d in data.get("detections", []):
        c_name = d["type"]
        c_id = CLASS_MAP.get(c_name, 0)
        global_boxes.append({
            "class_id": c_id,
            "x": d["x"],
            "y": d["y"],
            "w": d["width"],
            "h": d["height"]
        })

    # Initialize Tiler
    tiler = PlanSheetTiler(slice_size=tile_size, overlap_ratio=overlap_ratio)
    crops = tiler.extract_slice_crops(image)

    # Prepare output directories
    for split in ["train", "val"]:
        (out_base / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_base / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Generating tiles from {img_path.name} ({w}x{h}). Total tiles: {len(crops)}")

    for idx, item in enumerate(crops):
        crop = item["crop"]
        window = item["window"]
        split = "train" if (idx / len(crops)) < split_ratio else "val"

        tile_name = f"tile_{idx:03d}_{window[0]}_{window[1]}"
        img_out = out_base / "images" / split / f"{tile_name}.jpg"
        lbl_out = out_base / "labels" / split / f"{tile_name}.txt"

        # Save cropped tile image
        cv2.imwrite(str(img_out), crop)

        # Slice corresponding annotations
        local_boxes = tiler.slice_annotations(window, global_boxes)

        # Write YOLO format label file (class_id norm_cx norm_cy norm_w norm_h)
        with open(lbl_out, "w", encoding="utf-8") as lf:
            for b in local_boxes:
                yolo = b["yolo_bbox"]
                lf.write(f"{b['class_id']} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}\n")

    print(f"[SUCCESS] Dataset generated under {out_base}/ with train/val splits.")


if __name__ == "__main__":
    prepare_yolo_tiles()
