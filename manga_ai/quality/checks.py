"""Basic quality checks for each stage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class QualityChecker:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def check_detections(self, detections: List[Dict[str, Any]]) -> List[str]:
        warnings = []
        if not detections and self.config.get("check_empty_detections", True):
            warnings.append("No text regions detected")
        for i, d in enumerate(detections):
            bbox = d.get("bbox")
            if bbox and len(bbox) == 4:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < 50:
                    warnings.append(f"Very small box at index {i} (area={area:.0f})")
            conf = d.get("confidence", 1.0)
            if conf < 0.3:
                warnings.append(f"Low detection confidence {conf:.2f} at index {i}")
        return warnings

    def check_ocr(self, regions: List[Dict[str, Any]]) -> List[str]:
        warnings = []
        for r in regions:
            text = r.get("source_text", "")
            if not text.strip() and self.config.get("check_empty_ocr", True):
                warnings.append(f"Empty OCR for region {r.get('region_id')}")
            conf = r.get("ocr_confidence", 1.0)
            if conf < 0.2:
                warnings.append(
                    f"Low OCR confidence {conf:.2f} for {r.get('region_id')}"
                )
        return warnings

    def check_translations(
        self, regions: List[Dict[str, Any]], translations: List[Dict[str, Any]]
    ) -> List[str]:
        warnings = []
        if self.config.get("check_missing_translations", True):
            expected = {r["region_id"] for r in regions}
            got = {t["region_id"] for t in translations}
            missing = expected - got
            if missing:
                warnings.append(f"Missing translations for: {sorted(missing)}")
            extra = got - expected
            if extra:
                warnings.append(f"Unexpected translation IDs: {sorted(extra)}")
        for t in translations:
            if not t.get("text", "").strip():
                warnings.append(f"Empty translation for {t.get('region_id')}")
        return warnings

    def check_inpaint(
        self, original: np.ndarray, cleaned: np.ndarray, mask: np.ndarray
    ) -> List[str]:
        warnings = []
        if self.config.get("check_output_dimensions", True):
            if original.shape != cleaned.shape:
                warnings.append(
                    f"Shape mismatch: original {original.shape} vs cleaned {cleaned.shape}"
                )
        if mask is None or mask.size == 0:
            warnings.append("Mask is empty or missing")
        return warnings
