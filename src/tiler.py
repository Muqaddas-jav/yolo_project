"""
Image & Annotation Tiler Module
Performs high-resolution window slicing with overlap for training data preparation
and inference window generation on large architectural plan sheets.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from pathlib import Path


class PlanSheetTiler:
    """Slices large plan sheets into smaller patches with horizontal/vertical overlap."""

    def __init__(
        self,
        slice_size: int = 1024,
        overlap_ratio: float = 0.25,
        min_box_area_ratio: float = 0.50
    ):
        """
        Args:
            slice_size: Width and height of square tiles (e.g., 1024px).
            overlap_ratio: Overlap percentage between adjacent windows (e.g., 0.25 for 25%).
            min_box_area_ratio: Minimum ratio of bounding box that must be inside the tile to retain it.
        """
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.overlap_px = int(slice_size * overlap_ratio)
        self.step_size = max(1, slice_size - self.overlap_px)
        self.min_box_area_ratio = min_box_area_ratio

    def compute_slice_windows(self, img_height: int, img_width: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculates all slice coordinate bounds (x1, y1, x2, y2) covering the entire image.
        Ensures complete edge-to-edge coverage.
        """
        windows = []
        
        y_starts = list(range(0, max(1, img_height - self.slice_size + 1), self.step_size))
        if not y_starts or y_starts[-1] + self.slice_size < img_height:
            y_starts.append(max(0, img_height - self.slice_size))

        x_starts = list(range(0, max(1, img_width - self.slice_size + 1), self.step_size))
        if not x_starts or x_starts[-1] + self.slice_size < img_width:
            x_starts.append(max(0, img_width - self.slice_size))

        # Remove duplicates while preserving order
        y_starts = sorted(list(set(y_starts)))
        x_starts = sorted(list(set(x_starts)))

        for y in y_starts:
            for x in x_starts:
                x2 = min(img_width, x + self.slice_size)
                y2 = min(img_height, y + self.slice_size)
                windows.append((x, y, x2, y2))

        return windows

    def extract_slice_crops(
        self,
        image: np.ndarray,
        pad_edge: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extracts image crops from the plan sheet.
        
        Returns:
            List of dicts: {"crop": np.ndarray, "window": (x1, y1, x2, y2), "pad": (pad_h, pad_w)}
        """
        h, w = image.shape[:2]
        windows = self.compute_slice_windows(h, w)
        slices = []

        for (x1, y1, x2, y2) in windows:
            crop = image[y1:y2, x1:x2].copy()
            pad_h = self.slice_size - crop.shape[0]
            pad_w = self.slice_size - crop.shape[1]

            if pad_edge and (pad_h > 0 or pad_w > 0):
                # Pad with white canvas background
                crop = cv2.copyMakeBorder(
                    crop, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=[255, 255, 255]
                )

            slices.append({
                "crop": crop,
                "window": (x1, y1, x2, y2),
                "pad": (pad_h, pad_w)
            })

        return slices

    def slice_annotations(
        self,
        window: Tuple[int, int, int, int],
        global_boxes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transforms global full-sheet bounding boxes into local tile-normalized coordinates.

        Args:
            window: (x1, y1, x2, y2) in full sheet pixels.
            global_boxes: List of dicts: {"class_id": int, "x": float, "y": float, "w": float, "h": float}
        """
        wx1, wy1, wx2, wy2 = window
        tile_w = wx2 - wx1
        tile_h = wy2 - wy1
        local_boxes = []

        for b in global_boxes:
            bx1 = b["x"]
            by1 = b["y"]
            bx2 = bx1 + b["w"]
            by2 = by1 + b["h"]

            # Compute intersection
            ix1 = max(wx1, bx1)
            iy1 = max(wy1, by1)
            ix2 = min(wx2, bx2)
            iy2 = min(wy2, by2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            orig_area = max(1e-5, (bx2 - bx1) * (by2 - by1))
            inter_area = (ix2 - ix1) * (iy2 - iy1)

            # Only retain if sufficient portion lies within this tile
            if (inter_area / orig_area) >= self.min_box_area_ratio:
                # Local coordinates relative to window
                local_x1 = ix1 - wx1
                local_y1 = iy1 - wy1
                local_x2 = ix2 - wx1
                local_y2 = iy2 - wy1

                # Normalize to [0, 1] relative to tile size
                norm_cx = ((local_x1 + local_x2) / 2.0) / self.slice_size
                norm_cy = ((local_y1 + local_y2) / 2.0) / self.slice_size
                norm_w = (local_x2 - local_x1) / self.slice_size
                norm_h = (local_y2 - local_y1) / self.slice_size

                local_boxes.append({
                    "class_id": b["class_id"],
                    "yolo_bbox": [norm_cx, norm_cy, norm_w, norm_h],
                    "pixel_bbox": [local_x1, local_y1, local_x2, local_y2]
                })

        return local_boxes
