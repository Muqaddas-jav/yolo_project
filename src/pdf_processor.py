"""
PDF and Plan Sheet Preprocessor Module
Handles high-resolution rasterization of architectural vector/raster PDFs at target DPI
and formats plan drawings for downstream computer vision slicing.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Union
import numpy as np
import cv2

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class PlanSheetPreprocessor:
    """Preprocesses architectural drawings (PDFs, TIFFs, PNGs, JPEGs) into high-res image matrices."""

    def __init__(self, target_dpi: int = 300):
        """
        Args:
            target_dpi: Rendering DPI for vector drawings (standard architectural print is 300 DPI).
        """
        self.target_dpi = target_dpi
        self.zoom_factor = target_dpi / 72.0  # 72 points per inch in PDF spec

    def load_or_rasterize(self, file_path: Union[str, Path], page_index: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads an image directly or rasterizes a PDF page at target DPI.

        Args:
            file_path: Path to drawing file (.pdf, .png, .jpg, .tiff, etc.)
            page_index: Page number (0-indexed) if input is a multi-page PDF.

        Returns:
            Tuple of (BGR image as np.ndarray, metadata dictionary)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input drawing not found at: {file_path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._rasterize_pdf(path, page_index)
        else:
            return self._load_image(path)

    def _rasterize_pdf(self, pdf_path: Path, page_index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF ('fitz') is required for PDF rasterization. Install with: pip install pymupdf"
            )

        doc = fitz.open(str(pdf_path))
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"Page index {page_index} out of range. Document has {len(doc)} pages.")

        page = doc[page_index]
        matrix = fitz.Matrix(self.zoom_factor, self.zoom_factor)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert pixmap buffer to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Format to BGR for OpenCV compatibility
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        metadata = {
            "source_path": str(pdf_path),
            "file_name": pdf_path.name,
            "page_index": page_index,
            "total_pages": len(doc),
            "width": pix.width,
            "height": pix.height,
            "dpi": self.target_dpi,
            "type": "vector_pdf"
        }
        doc.close()
        return img, metadata

    def _load_image(self, img_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Unable to read image at: {img_path}")

        h, w = img.shape[:2]
        metadata = {
            "source_path": str(img_path),
            "file_name": img_path.name,
            "page_index": 0,
            "total_pages": 1,
            "width": w,
            "height": h,
            "dpi": self.target_dpi,
            "type": "raster_image"
        }
        return img, metadata
