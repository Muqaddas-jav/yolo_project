"""
NMS and Coordinate Stitching Engine
Translates local slice detections into parent sheet coordinates and performs
cross-seam Non-Maximum Suppression (NMS) to eliminate duplicate boundary detections.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class CoordinateStitcher:
    """Reconstructs global coordinates from sliced window detections and deduplicates seams."""

    def __init__(self, iou_threshold: float = 0.45, conf_threshold: float = 0.35):
        """
        Args:
            iou_threshold: Intersection-over-Union threshold for merging overlapping detections.
            conf_threshold: Minimum confidence score to retain detection.
        """
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    @staticmethod
    def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
        """Computes IoU between two bounding boxes [x1, y1, x2, y2]."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def global_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes class-aware Non-Maximum Suppression across all detections on the full sheet.

        Args:
            detections: List of dicts with keys:
                'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id', 'type'

        Returns:
            Deduplicated list of detections sorted by confidence descending.
        """
        if not detections:
            return []

        # Filter by confidence
        filtered = [d for d in detections if d["confidence"] >= self.conf_threshold]
        if not filtered:
            return []

        # Group by class to avoid suppressing adjacent symbols of different categories
        classes = set(d["class_id"] for d in filtered)
        final_detections = []

        for c_id in classes:
            class_dets = [d for d in filtered if d["class_id"] == c_id]
            # Sort descending by confidence
            class_dets.sort(key=lambda d: d["confidence"], reverse=True)

            kept = []
            while class_dets:
                best = class_dets.pop(0)
                kept.append(best)

                best_box = [best["x1"], best["y1"], best["x2"], best["y2"]]
                # Keep remaining boxes that do NOT heavily overlap with best
                class_dets = [
                    d for d in class_dets
                    if self.calculate_iou(best_box, [d["x1"], d["y1"], d["x2"], d["y2"]]) < self.iou_threshold
                ]

            final_detections.extend(kept)

        # Sort overall by position (y, x) for clean reporting
        final_detections.sort(key=lambda d: (d["y1"], d["x1"]))
        return final_detections

    def format_to_spec_json(
        self,
        detections: List[Dict[str, Any]],
        sheet_id: str = "E-101",
        sheet_width: int = 1024,
        sheet_height: int = 683
    ) -> Dict[str, Any]:
        """
        Formats detections into the structured JSON schema specified in project documentation:
        {
          "sheet": "E-101",
          "dimensions": {"width": 1024, "height": 683},
          "summary_counts": {...},
          "detections": [
             {"type": "light_fixture", "confidence": 0.94, "x": 1250, "y": 840, "width": 32, "height": 32}
          ]
        }
        """
        formatted_list = []
        summary_counts = {}

        for d in detections:
            w = round(d["x2"] - d["x1"], 1)
            h = round(d["y2"] - d["y1"], 1)
            sym_type = d.get("type", f"class_{d.get('class_id', 0)}")

            formatted_list.append({
                "type": sym_type,
                "confidence": round(float(d["confidence"]), 3),
                "x": round(float(d["x1"]), 1),
                "y": round(float(d["y1"]), 1),
                "width": w,
                "height": h
            })

            summary_counts[sym_type] = summary_counts.get(sym_type, 0) + 1

        return {
            "sheet": sheet_id,
            "dimensions": {
                "width": sheet_width,
                "height": sheet_height
            },
            "total_count": len(formatted_list),
            "summary_counts": summary_counts,
            "detections": formatted_list
        }
