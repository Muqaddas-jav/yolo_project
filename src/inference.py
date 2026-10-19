"""
End-to-End Sliced Inference Engine
Processes high-resolution architectural drawings (PDF/Image) via sliding-window tiling,
maps bounding boxes to full-sheet coordinates, deduplicates across seams, and exports structured JSON.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

from .pdf_processor import PlanSheetPreprocessor
from .tiler import PlanSheetTiler
from .nms_stitcher import CoordinateStitcher
from .visualizer import PlanSheetVisualizer


class ArchitecturalPlanDetector:
    """Production inference pipeline for architectural electrical symbol detection."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        slice_size: int = 1024,
        overlap_ratio: float = 0.25,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_dpi: int = 300
    ):
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_dpi = target_dpi

        self.preprocessor = PlanSheetPreprocessor(target_dpi=target_dpi)
        self.tiler = PlanSheetTiler(slice_size=slice_size, overlap_ratio=overlap_ratio)
        self.stitcher = CoordinateStitcher(iou_threshold=iou_threshold, conf_threshold=conf_threshold)
        self.visualizer = PlanSheetVisualizer()

        self.model = None
        self.class_names = {}
        if weights_path and Path(weights_path).exists():
            self._load_yolo_model(weights_path)

    def _load_yolo_model(self, weights_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.class_names = self.model.names
        except ImportError:
            print("[WARN] Ultralytics not installed. Pipeline running in mock/demo mode.")

    def run_inference(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path] = "output",
        page_index: int = 0
    ) -> Dict[str, Any]:
        """
        Executes end-to-end detection on a PDF or raster plan sheet.

        Returns:
            Dictionary containing structured detection results and takeoff summary.
        """
        input_p = Path(input_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sheet_id = input_p.stem

        # 1. Rasterize or load drawing
        image, meta = self.preprocessor.load_or_rasterize(input_p, page_index=page_index)
        h, w = image.shape[:2]

        # 2. Extract sliced windows
        slice_items = self.tiler.extract_slice_crops(image)
        raw_detections = []

        # 3. Model inference across slices
        if self.model is not None:
            for item in slice_items:
                crop = item["crop"]
                wx1, wy1, wx2, wy2 = item["window"]

                results = self.model(crop, conf=self.conf_threshold, verbose=False)[0]
                for box in results.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    c_name = self.class_names.get(cls_id, f"class_{cls_id}")

                    # Ignore detections outside active crop boundary (inside padding)
                    if bx1 >= (wx2 - wx1) or by1 >= (wy2 - wy1):
                        continue

                    # Translate to full sheet coordinates
                    gx1 = wx1 + bx1
                    gy1 = wy1 + by1
                    gx2 = min(w, wx1 + bx2)
                    gy2 = min(h, wy1 + by2)

                    raw_detections.append({
                        "x1": gx1,
                        "y1": gy1,
                        "x2": gx2,
                        "y2": gy2,
                        "confidence": score,
                        "class_id": cls_id,
                        "type": c_name
                    })

        # 4. Cross-seam deduplication via Global NMS
        final_detections = self.stitcher.global_nms(raw_detections)

        # 5. Format to exact Project Specification JSON Schema
        spec_json = self.stitcher.format_to_spec_json(
            final_detections,
            sheet_id=sheet_id,
            sheet_width=w,
            sheet_height=h
        )

        # 6. Save structured JSON
        json_file = out_dir / f"{sheet_id}_detections.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(spec_json, f, indent=2)

        # 7. Render verification overlay image
        annotated_img_path = out_dir / f"{sheet_id}_annotated.png"
        self.visualizer.save_annotated(image, final_detections, annotated_img_path)

        return spec_json
