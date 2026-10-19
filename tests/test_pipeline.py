"""
Unit tests for PlanSheetPreprocessor, PlanSheetTiler, and CoordinateStitcher.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tiler import PlanSheetTiler
from src.nms_stitcher import CoordinateStitcher


class TestArchitecturalPipeline(unittest.TestCase):

    def test_tiler_window_generation(self):
        tiler = PlanSheetTiler(slice_size=512, overlap_ratio=0.25)
        # 1024 x 683 image
        windows = tiler.compute_slice_windows(683, 1024)
        self.assertGreater(len(windows), 0)
        # Check all windows stay within bounds
        for (x1, y1, x2, y2) in windows:
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 1024)
            self.assertLessEqual(y2, 683)

    def test_coordinate_stitcher_iou(self):
        stitcher = CoordinateStitcher()
        # Identical boxes
        box_a = [10.0, 10.0, 50.0, 50.0]
        box_b = [10.0, 10.0, 50.0, 50.0]
        self.assertAlmostEqual(stitcher.calculate_iou(box_a, box_b), 1.0, places=4)

        # Disjoint boxes
        box_c = [100.0, 100.0, 150.0, 150.0]
        self.assertAlmostEqual(stitcher.calculate_iou(box_a, box_c), 0.0, places=4)

    def test_global_nms_deduplication(self):
        stitcher = CoordinateStitcher(iou_threshold=0.45, conf_threshold=0.30)
        # Two duplicate detections on the same switch from overlapping tiles
        dets = [
            {"x1": 100, "y1": 100, "x2": 130, "y2": 130, "confidence": 0.95, "class_id": 8, "type": "switch"},
            {"x1": 102, "y1": 101, "x2": 132, "y2": 131, "confidence": 0.88, "class_id": 8, "type": "switch"},
            # Distinct receptacle nearby
            {"x1": 200, "y1": 100, "x2": 235, "y2": 130, "confidence": 0.92, "class_id": 3, "type": "receptacle_duplex"}
        ]
        merged = stitcher.global_nms(dets)
        self.assertEqual(len(merged), 2)
        # The higher confidence switch should be kept
        switch_det = [d for d in merged if d["type"] == "switch"][0]
        self.assertEqual(switch_det["confidence"], 0.95)

    def test_json_formatting_schema(self):
        stitcher = CoordinateStitcher()
        dets = [
            {"x1": 50.0, "y1": 60.0, "x2": 80.0, "y2": 90.0, "confidence": 0.94, "type": "light_fixture"}
        ]
        result = stitcher.format_to_spec_json(dets, sheet_id="E-101", sheet_width=1000, sheet_height=800)
        self.assertEqual(result["sheet"], "E-101")
        self.assertIn("detections", result)
        self.assertEqual(result["detections"][0]["type"], "light_fixture")
        self.assertEqual(result["detections"][0]["width"], 30.0)
        self.assertEqual(result["detections"][0]["height"], 30.0)


if __name__ == "__main__":
    unittest.main()
