"""Pydantic schemas for pipeline data structures."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_REVIEW = "needs_review"


class RegionType(str, Enum):
    TEXT = "text"
    SOUND_EFFECT = "sound_effect"
    OTHER = "other"


class Point(BaseModel):
    x: float
    y: float

    def as_list(self) -> List[float]:
        return [self.x, self.y]


class BoundingBox(BaseModel):
    """Axis-aligned bounding box [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


class DetectionResult(BaseModel):
    """Output of a Detector backend."""

    bbox: List[float] = Field(..., min_length=4, max_length=4)
    polygon: List[List[float]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    region_type: str = "text"


class OCRResult(BaseModel):
    """Output of an OCR backend."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    language: str = "unknown"


class TextRegion(BaseModel):
    """A single detected + OCR'd + translated text region."""

    region_id: str
    page: str
    bbox: List[float]
    polygon: List[List[float]] = Field(default_factory=list)
    region_type: str = "text"
    detection_confidence: float = 0.0
    source_text: str = ""
    ocr_confidence: float = 0.0
    source_language: str = "unknown"
    translated_text: str = ""
    target_language: str = "fa"


class StageMetadata(BaseModel):
    """Metadata recorded for every pipeline stage."""

    stage: str
    backend: str = "unknown"
    model: Optional[str] = None
    model_version: Optional[str] = None
    device: str = "cpu"
    precision: Optional[str] = None
    input_resolution: Optional[List[int]] = None
    output_resolution: Optional[List[int]] = None
    execution_time_ms: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    status: StageStatus = StageStatus.PENDING
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    config_hash: Optional[str] = None
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class PageStatus(BaseModel):
    page: str
    status: StageStatus = StageStatus.PENDING
    stage: Optional[str] = None
    error: Optional[str] = None


class ChapterManifest(BaseModel):
    """Central chapter-level manifest."""

    schema_version: str = "1.0"
    chapter_id: str
    source_language: str = "auto"
    target_language: str = "fa"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat() + "Z")
    updated_at: Optional[str] = None
    pipeline_version: str = "0.1.0"
    pages: List[str] = Field(default_factory=list)
    page_status: List[PageStatus] = Field(default_factory=list)
    stages: Dict[str, StageMetadata] = Field(default_factory=dict)
    regions: List[TextRegion] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    cache_keys: Dict[str, str] = Field(default_factory=dict)


class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    chapter_context: List[Dict[str, Any]] = Field(default_factory=list)
    regions: List[Dict[str, Any]]
    glossary: Dict[str, str] = Field(default_factory=dict)


class TranslationItem(BaseModel):
    region_id: str
    text: str


class TranslationResponse(BaseModel):
    translations: List[TranslationItem]


class ImageInfo(BaseModel):
    path: str
    page_id: str
    width: int
    height: int
    channels: int
    mode: str
    file_hash: str
