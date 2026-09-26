"""Phase 0 acceptance tests – mock pipeline must succeed."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_ai.config import Config
from manga_ai.pipeline import Pipeline
from manga_ai.schemas import ChapterManifest, StageStatus
from manga_ai.detection import MockDetector
from manga_ai.ocr import MockOCR
from manga_ai.translation import MockTranslator
from manga_ai.inpainting import MockInpainter
from manga_ai.masking import MaskGenerator
from manga_ai.utils import list_images, page_id_from_path


@pytest.fixture
def sample_chapter(tmp_path: Path) -> Path:
    """Create a tiny chapter with 2 synthetic pages."""
    chapter = tmp_path / "chapter_001"
    original = chapter / "original"
    original.mkdir(parents=True)

    for i, name in enumerate(["001.png", "002.png"]):
        img = Image.new("RGB", (400, 600), color=(240 - i * 20, 230, 220))
        # Draw a fake "text" rectangle
        pixels = img.load()
        for x in range(50, 200):
            for y in range(80, 120):
                pixels[x, y] = (20, 20, 20)
        img.save(original / name)

    return chapter


def test_config_load():
    cfg = Config.load()
    assert cfg.get("detection.backend") == "mock"
    assert cfg.get("pipeline.target_language") == "fa"
    h = cfg.hash()
    assert isinstance(h, str) and len(h) == 16


def test_config_with_backend():
    cfg = Config.load()
    cfg2 = cfg.with_backend("mock")
    assert cfg2.get("detection.backend") == "mock"
    assert cfg2.get("ocr.backend") == "mock"


def test_mock_detector():
    det = MockDetector()
    img = np.zeros((600, 400, 3), dtype=np.uint8)
    regions = det.detect(img)
    assert len(regions) >= 2
    assert "bbox" in regions[0]
    assert "confidence" in regions[0]


def test_mock_ocr():
    ocr = MockOCR()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = ocr.recognize(img, {"bbox": [0, 0, 50, 50]})
    assert "text" in result
    assert result["confidence"] > 0


def test_mock_translator():
    tr = MockTranslator()
    regions = [
        {"region_id": "001_0000", "source_text": "안녕하세요"},
        {"region_id": "001_0001", "source_text": "고마워."},
    ]
    out = tr.translate_chapter(regions, {"glossary": {}})
    assert len(out) == 2
    assert out[0]["text"] == "سلام"


def test_mask_generator():
    gen = MaskGenerator(dilation_px=4, blur_radius=0)
    regions = [
        {
            "bbox": [10, 10, 50, 40],
            "polygon": [[10, 10], [50, 10], [50, 40], [10, 40]],
        }
    ]
    mask = gen.generate((100, 100), regions)
    assert mask.shape == (100, 100)
    assert mask.max() == 255
    assert mask.sum() > 0


def test_mock_inpainter():
    inp = MockInpainter()
    img = np.ones((64, 64, 3), dtype=np.uint8) * 200
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    out = inp.inpaint(img, mask)
    assert out.shape == img.shape


def test_full_pipeline_mock(sample_chapter: Path):
    cfg = Config.load()
    cfg = cfg.with_backend("mock")
    pipe = Pipeline(cfg, sample_chapter)
    manifest = pipe.process()

    assert isinstance(manifest, ChapterManifest)
    assert len(manifest.pages) == 2
    assert len(manifest.regions) >= 4  # at least 2 per page

    # Artifacts exist
    assert (sample_chapter / "manifest.json").exists()
    assert (sample_chapter / "translation" / "chapter.json").exists()
    assert (sample_chapter / "detection").is_dir()
    assert (sample_chapter / "ocr").is_dir()
    assert (sample_chapter / "masks").is_dir()
    assert (sample_chapter / "cleaned").is_dir()

    cleaned = list((sample_chapter / "cleaned").glob("*.png"))
    assert len(cleaned) == 2

    # Stages recorded
    for stage in ("detection", "ocr", "translation", "masking", "inpainting"):
        assert stage in manifest.stages
        assert manifest.stages[stage].status in (
            StageStatus.SUCCESS,
            StageStatus.FAILED,
        )

    # Translation JSON structure
    chapter_json = json.loads(
        (sample_chapter / "translation" / "chapter.json").read_text(encoding="utf-8")
    )
    assert "regions" in chapter_json
    assert all("translated_text" in r for r in chapter_json["regions"])


def test_page_id_from_path():
    assert page_id_from_path("001.png") == "001"
    assert page_id_from_path("00 (12).jpg") == "012"
    assert page_id_from_path("page_7.webp") == "007"
