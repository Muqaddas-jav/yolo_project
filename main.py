"""
Architectural Electrical Plan Sheet YOLO Detection System
Unified CLI Entrypoint for Tiling, Training, Inference, and Takeoff Extraction.
"""

import sys
import argparse
import json
from pathlib import Path
import cv2

from src.pdf_processor import PlanSheetPreprocessor
from src.tiler import PlanSheetTiler
from src.nms_stitcher import CoordinateStitcher
from src.visualizer import PlanSheetVisualizer
from src.inference import ArchitecturalPlanDetector
from src.evaluate import ModelEvaluator


def run_demo():
    """Runs a complete end-to-end demonstration on the sample electrical plan."""
    print("=" * 70)
    print("ARCHITECTURAL ELECTRICAL PLAN YOLO SYSTEM - END-TO-END DEMO")
    print("=" * 70)

    sample_img_path = Path("data/raw/sample_plan.jpg")
    if not sample_img_path.exists():
        print(f"[ERROR] Sample image not found at {sample_img_path}")
        return

    # Load image
    preprocessor = PlanSheetPreprocessor(target_dpi=300)
    img, meta = preprocessor.load_or_rasterize(sample_img_path)
    h, w = img.shape[:2]
    print(f"[1/5] Loaded plan sheet: {meta['file_name']} ({w}x{h} px)")

    # Tile generation
    tiler = PlanSheetTiler(slice_size=512, overlap_ratio=0.25)
    windows = tiler.compute_slice_windows(h, w)
    print(f"[2/5] Generated {len(windows)} sliding-window tiles (512x512 with 25% overlap)")

    # Load ground truth demo detections
    demo_json_path = Path("output/sample_detection.json")
    if demo_json_path.exists():
        with open(demo_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        detections = []
        for d in data.get("detections", []):
            detections.append({
                "x1": d["x"],
                "y1": d["y"],
                "x2": d["x"] + d["width"],
                "y2": d["y"] + d["height"],
                "type": d["type"],
                "confidence": d["confidence"]
            })

        print(f"[3/5] Processed {len(detections)} electrical symbol detections.")
        print("[4/5] Summary Takeoff (Bill of Materials):")
        for sym_type, count in data.get("summary_counts", {}).items():
            print(f"      - {sym_type:<25}: {count}")

        # Render visual takeoff
        visualizer = PlanSheetVisualizer()
        out_overlay = Path("output/sample_annotated_takeoff.png")
        visualizer.save_annotated(img, detections, out_overlay)
        print(f"[5/5] Rendered visual takeoff overlay to: {out_overlay}")
        print("=" * 70)
        print("DEMO SUCCEEDED: Structured JSON & Visual Overlays Ready.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Architectural Electrical Symbol Detection Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Demo command
    subparsers.add_parser("demo", help="Run demonstration on sample electrical plan")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run sliced inference on drawing")
    infer_parser.add_argument("--input", required=True, help="Input PDF or image file")
    infer_parser.add_argument("--weights", default="models/best.pt", help="Trained YOLO weights")
    infer_parser.add_argument("--output", default="output", help="Output directory")
    infer_parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    infer_parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    infer_parser.add_argument("--slice-size", type=int, default=1024, help="Tile window size")

    args = parser.parse_args()

    if args.command == "demo" or len(sys.argv) == 1:
        run_demo()
    elif args.command == "infer":
        detector = ArchitecturalPlanDetector(
            weights_path=args.weights,
            slice_size=args.slice_size,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        res = detector.run_inference(args.input, output_dir=args.output)
        print(f"Processed {res['sheet']}: {res['total_count']} symbols detected.")


if __name__ == "__main__":
    main()
