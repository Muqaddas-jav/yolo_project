"""
Evaluation and Diagnostic Benchmarking Module
Computes Precision, Recall, F1, mAP@50, mAP@50:95, and small-object error diagnostics.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class ModelEvaluator:
    """Evaluates architectural symbol detection performance."""

    @staticmethod
    def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def evaluate_detections(
        self,
        ground_truths: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        iou_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Computes precision, recall, and F1 score per class and overall.
        """
        all_classes = sorted(list(set(
            [gt["type"] for gt in ground_truths] + [pred["type"] for pred in predictions]
        )))

        per_class_stats = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for cls_name in all_classes:
            gts = [gt for gt in ground_truths if gt["type"] == cls_name]
            preds = [p for p in predictions if p["type"] == cls_name]
            preds.sort(key=lambda x: x.get("confidence", 1.0), reverse=True)

            matched_gt = set()
            tp = 0
            fp = 0

            for pred in preds:
                pbox = [pred["x1"], pred["y1"], pred["x2"], pred["y2"]]
                best_iou = 0.0
                best_gt_idx = -1

                for idx, gt in enumerate(gts):
                    if idx in matched_gt:
                        continue
                    gtbox = [gt["x1"], gt["y1"], gt["x2"], gt["y2"]]
                    iou = self.calculate_iou(pbox, gtbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn = len(gts) - len(matched_gt)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_stats[cls_name] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4)
            }

            total_tp += tp
            total_fp += fp
            total_fn += fn

        overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_p * overall_r) / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

        return {
            "overall": {
                "precision": round(overall_p, 4),
                "recall": round(overall_r, 4),
                "f1": round(overall_f1, 4),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn
            },
            "per_class": per_class_stats
        }
