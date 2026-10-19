# Architectural Electrical Plan Sheet Symbol Detection (YOLO)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computer vision pipeline developed to analyze architectural electrical plan sheets (2D CAD exports and vector/scanned PDFs), detect electrical symbols, resolve tile-boundary seams, and extract structured Takeoff (Bill of Materials) data into JSON format.

---

## 1. System Overview & Problem Statement

Architectural electrical drawings (Arch D/E format: $24'' \times 36''$ to $36'' \times 48''$) rendered at $300\text{ DPI}$ produce images spanning **$7,200 \times 10,800$ to $10,800 \times 14,400$ pixels**. Within these massive drawings, individual electrical symbols (receptacles, switches, lights, fire alarm devices) measure only **$20 \times 20$ to $50 \times 50$ pixels** (< 0.005% of the total canvas).

Resizing the full page directly to standard YOLO resolution ($640 \times 640$) causes severe downsampling blur, rendering small symbols invisible. 

This repository solves this via a **Slicing-Aided Hyper Inference (SAHI)** pattern:
1. **PyMuPDF Vector Rasterization**: Renders vector drawings into crisp 300 DPI matrices.
2. **Sliding-Window Slicing**: Deconstructs sheets into $1024 \times 1024$ (or $512 \times 512$) patches with a $25\%$ overlap.
3. **YOLO Detection Engine**: Evaluates patches using domain-adapted CAD hyperparameters.
4. **Global Coordinate Translation**: Maps tile-local detections back to parent plan-sheet coordinates $(x, y, w, h)$.
5. **Cross-Seam Global NMS**: Merges duplicate detections across overlapping tile boundaries.
6. **Structured JSON Takeoff**: Generates machine-readable schedules and visual inspection overlays.

---

## 2. Electrical Symbol Taxonomy

The model detects standardized American National Standards Institute (ANSI) and National Electrical Code (NEC) drafting symbols:

| Class ID | Symbol Class | Description | Standard Appearance |
| :--- | :--- | :--- | :--- |
| `0` | `ceiling_mounted_light` | Standard ceiling luminaire | Circular outline |
| `1` | `recessed_down_light` | Recessed ceiling canister | Circle with cross tick-marks |
| `2` | `emergency_light` | Battery-backed egress luminaire | Solid right-triangle wedge |
| `3` | `receptacle_duplex` | 120V wall outlet | Circle bisected by dual parallel lines |
| `4` | `nurse_call` | Hospital / medical call station | Circle with adjacent `+` symbol |
| `5` | `telephone_outlet` | Telecom / RJ45 outlet | Square with diagonal bisector triangle |
| `6` | `tv_outlet` | Coaxial / CATV drop | Square box with `TV` text label |
| `7` | `smoke_detector` | Life-safety smoke sensor | Hexagon enclosing capital letter `S` |
| `8` | `switch` | Single-pole wall switch | Capital letter `S` adjacent to doorway/wall |
| `9` | `electrical_panel` | Distribution panelboard | Filled or hatched rectangular cabinet |
| `10` | `junction_box` | Wire pull/splice box | Square box with diagonal cross |
| `11` | `exit_sign` | Illuminated egress exit sign | Box enclosing directional arrow |
| `12` | `fire_alarm_strobe` | Audible/visual notification | Square/octagon with strobe horn |

---

## 3. Repository Architecture

```
yolo_project/
├── .github/workflows/ci.yml       # Automated CI test suite
├── configs/
│   ├── dataset.yaml               # YOLO dataset and class mapping configuration
│   └── hyp_cad.yaml               # CAD drafting hyperparameters (rotation, lines)
├── data/
│   ├── raw/                       # Original drawings and reference PDFs
│   └── tiles/                     # 1024x1024 sliced training patches & YOLO labels
│       ├── images/ [train, val]
│       └── labels/ [train, val]
├── models/
│   └── best.pt                    # Fine-tuned YOLO weights
├── notebooks/
│   └── architectural_yolo_walkthrough.ipynb # Interactive demo walkthrough
├── output/
│   ├── sample_detection.json      # Structured JSON takeoff matching specification
│   └── sample_annotated_takeoff.png # Verification image with bounding boxes
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py           # High-DPI PDF rasterizer (PyMuPDF)
│   ├── tiler.py                   # Sliding-window slicer & label translator
│   ├── nms_stitcher.py            # Global coordinate remapper & seam NMS
│   ├── prepare_dataset.py         # Training tile and label generator
│   ├── train.py                   # YOLO fine-tuning pipeline
│   ├── inference.py               # End-to-end sliced inference engine
│   ├── evaluate.py                # Precision, recall, and small-object metrics
│   └── visualizer.py              # Visual inspection overlay renderer
├── tests/
│   └── test_pipeline.py           # Unit tests for tiling, IoU, and NMS
├── main.py                        # Unified CLI entrypoint
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

```bash
# Clone the repository
git clone https://github.com/Muqaddas-jav/yolo_project.git
cd yolo_project

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Quickstart & Usage

### 1. Run the End-to-End Demo
Executes high-resolution tiling, detection, takeoff extraction, and overlay rendering:
```bash
python main.py demo
```

Output:
```
[1/5] Loaded plan sheet: sample_plan.jpg (1024x683 px)
[2/5] Generated 6 sliding-window tiles (512x512 with 25% overlap)
[3/5] Processed 40 electrical symbol detections.
[4/5] Summary Takeoff (Bill of Materials):
      - ceiling_mounted_light    : 8
      - recessed_down_light      : 4
      - switch                   : 8
      - receptacle_duplex        : 6
      - nurse_call               : 4
      - telephone_outlet         : 4
      - tv_outlet                : 2
      - smoke_detector           : 2
      - emergency_light          : 2
[5/5] Rendered visual takeoff overlay to: output/sample_annotated_takeoff.png
```

### 2. Run Inference on a New PDF or CAD Drawing
```bash
python main.py infer --input path/to/drawing.pdf --weights models/best.pt --output output --conf 0.35 --iou 0.45
```

### 3. Generate Training Tiles from Annotated Drawings
```bash
python -m src.prepare_dataset
```

### 4. Train the YOLO Model
Fine-tunes YOLOv8 or YOLO11 using CAD-specific hyperparameters:
```bash
python src/train.py --data configs/dataset.yaml --hyp configs/hyp_cad.yaml --model yolo11m.pt --epochs 150 --imgsz 1024 --batch 8
```

### 5. Run Unit Tests
```bash
python -m unittest discover -s tests
```

---

## 6. Output Format (Specification JSON Schema)

The pipeline outputs structured JSON adhering to the project specification:

```json
{
  "sheet": "E-101",
  "dimensions": {
    "width": 1024,
    "height": 683
  },
  "total_count": 36,
  "summary_counts": {
    "ceiling_mounted_light": 8,
    "recessed_down_light": 4,
    "switch": 8,
    "receptacle_duplex": 6,
    "nurse_call": 4,
    "telephone_outlet": 4,
    "tv_outlet": 2,
    "smoke_detector": 2,
    "emergency_light": 2
  },
  "detections": [
    {
      "type": "receptacle_duplex",
      "confidence": 0.97,
      "x": 57.0,
      "y": 208.0,
      "width": 40.0,
      "height": 26.0
    },
    {
      "type": "switch",
      "confidence": 0.92,
      "x": 60.0,
      "y": 191.0,
      "width": 18.0,
      "height": 20.0
    }
  ]
}
```

---

## 7. Known Architectural Drafting Challenges & Mitigations

1. **Curved Wiring Homeruns / Switch Legs**:
   - *Challenge*: Wiring arcs connecting wall switches to ceiling luminaires intersect symbol contours.
   - *Mitigation*: We apply Albumentations `RandomLine` and `ElasticTransform` augmentations to teach the model to ignore overlapping linework.
2. **Text vs. Symbol Confusion**:
   - *Challenge*: Plain wall switch `S` can be confused with `(S)` in hexagon (smoke detector) or general architectural notes.
   - *Mitigation*: Multi-scale receptive fields (P2 feature pyramid level) and spatial context conditioning.
3. **Multi-Angle Wall Orientations**:
   - *Challenge*: Receptacles and switches rotate $0^\circ, 90^\circ, 180^\circ, 270^\circ$ or along angled partition walls.
   - *Mitigation*: Enabled $180^\circ$ continuous rotation and affine flipping in `configs/hyp_cad.yaml`.
4. **Tile Boundary Slicing Artifacts**:
   - *Challenge*: A device resting on a tile boundary is severed in half.
   - *Mitigation*: $25\%$ window overlap ensures every symbol is captured fully in at least one window; global NMS with IoU merge merges partial detections.

---

## 8. License

This project is licensed under the MIT License - see the LICENSE file for details.
