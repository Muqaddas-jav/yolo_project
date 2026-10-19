"""
Visualizer and Quality Assurance Overlay Module
Renders detected bounding boxes, class labels, and confidence tags on high-res plan sheets.
"""

from typing import List, Dict, Any, Optional, Union
import cv2
import numpy as np
from pathlib import Path


class PlanSheetVisualizer:
    """Renders computer vision detection takeoffs onto architectural drawings."""

    # Color palette tailored for high-contrast viewing on white/light architectural drawings
    PALETTE = {
        "ceiling_mounted_light": (246, 130, 59),   # Blue (BGR: 59, 130, 246)
        "recessed_down_light": (212, 182, 6),     # Cyan
        "receptacle_duplex": (94, 197, 34),       # Green
        "switch": (8, 179, 234),                  # Amber/Yellow
        "nurse_call": (153, 72, 236),             # Pink
        "telephone_outlet": (246, 92, 139),       # Purple
        "tv_outlet": (22, 115, 249),              # Orange
        "emergency_light": (38, 38, 220),         # Dark Red
        "smoke_detector": (68, 68, 239),          # Bright Red
        "electrical_panel": (128, 0, 128),        # Violet
        "junction_box": (128, 128, 0)             # Teal
    }

    @classmethod
    def get_color(cls, class_name: str) -> tuple:
        return cls.PALETTE.get(class_name, (100, 100, 100))

    def render_overlay(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_labels: bool = True,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draws bounding boxes and labels onto a copy of the image.

        Args:
            image: Full plan sheet as BGR numpy array.
            detections: List of dicts with 'x1', 'y1', 'x2', 'y2', 'type', 'confidence'.
            show_labels: Whether to print class name and confidence text.
            thickness: Bounding box line thickness.
        """
        canvas = image.copy()

        for det in detections:
            x1 = int(det["x1"])
            y1 = int(det["y1"])
            x2 = int(det["x2"])
            y2 = int(det["y2"])
            c_name = det.get("type", "symbol")
            conf = det.get("confidence", 1.0)
            color = self.get_color(c_name)

            # Draw bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            if show_labels:
                label = f"{c_name[:12]} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                font_th = 1
                (lw, lh), _ = cv2.getTextSize(label, font, font_scale, font_th)

                label_y1 = max(0, y1 - lh - 4)
                label_y2 = y1
                # Semi-transparent or solid label background
                cv2.rectangle(canvas, (x1, label_y1), (x1 + lw + 4, label_y2), color, -1)
                cv2.putText(
                    canvas, label, (x1 + 2, label_y2 - 2),
                    font, font_scale, (255, 255, 255), font_th, cv2.LINE_AA
                )

        return canvas

    def save_annotated(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ):
        """Renders detections and saves result image to disk."""
        overlay = self.render_overlay(image, detections)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), overlay)
