"""Main pipeline orchestrator (modular, stage-by-stage)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from manga_ai import __version__
from manga_ai.config import Config
from manga_ai.detection import Detector, create_detector
from manga_ai.inpainting import Inpainter, create_inpainter
from manga_ai.logging import get_logger, log_error, log_stage, setup_logging
from manga_ai.masking import MaskGenerator
from manga_ai.ocr import OCRBackend, create_ocr
from manga_ai.preprocessing import Preprocessor
from manga_ai.quality import QualityChecker
from manga_ai.schemas import (
    ChapterManifest,
    PageStatus,
    StageMetadata,
    StageStatus,
    TextRegion,
)
from manga_ai.translation import Translator, create_translator
from manga_ai.utils import ensure_dir, list_images, page_id_from_path
from manga_ai.utils.io import IMAGE_EXTENSIONS
from manga_ai.utils.cache import ArtifactCache
from manga_ai.utils.device import clear_cuda, get_device, log_vram, peak_vram_mb
from manga_ai.utils.hashing import content_hash

logger = get_logger("manga_ai.pipeline")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Pipeline:
    """Orchestrates the full cleaning & translation pipeline."""

    def __init__(self, config: Config, chapter_dir: str | Path):
        self.config = config
        self.chapter_dir = Path(chapter_dir).resolve()
        self.chapter_id = self.chapter_dir.name

        self.original_dir = self.chapter_dir / "original"
        self.detection_dir = self.chapter_dir / "detection"
        self.ocr_dir = self.chapter_dir / "ocr"
        self.translation_dir = self.chapter_dir / "translation"
        self.masks_dir = self.chapter_dir / "masks"
        self.cleaned_dir = self.chapter_dir / "cleaned"
        self.debug_dir = self.chapter_dir / "debug"
        self.manifest_path = self.chapter_dir / "manifest.json"

        self.force = bool(config.get("pipeline.force", False))
        self.pipeline_version = config.get("pipeline.version", __version__)
        self.cache = ArtifactCache(enabled=config.get("cache.enabled", True))

        setup_logging(
            level=config.get("logging.level", "INFO"),
            structured=config.get("logging.structured", True),
        )

        self.preprocessor = Preprocessor(
            convert_to_rgb=config.get("preprocessing.convert_to_rgb", True),
            max_dimension=config.get("preprocessing.max_dimension"),
            validate=config.get("preprocessing.validate", True),
        )
        self.mask_gen = MaskGenerator(
            dilation_px=config.get("masking.dilation_px", 16),
            blur_radius=config.get("masking.blur_radius", 2),
            use_polygon=config.get("masking.use_polygon", True),
            residual_expand=config.get("masking.residual_expand", True),
            residual_thresh=config.get("masking.residual_thresh", 140),
        )
        self.quality = QualityChecker(config.section("quality"))

        preferred = config.get("device.preferred", "cuda")
        device_id = int(config.get("device.device_id", 0))
        self.device = get_device(preferred, device_id)
        self.use_gpu = self.device.startswith("cuda")

        self._detector: Optional[Detector] = None
        self._ocr: Optional[OCRBackend] = None
        self._translator: Optional[Translator] = None
        self._inpainter: Optional[Inpainter] = None
        self.manifest: Optional[ChapterManifest] = None

    # ------------------------------------------------------------------
    # Backend factories
    # ------------------------------------------------------------------
    def _get_detector(self) -> Detector:
        if self._detector is None:
            backend = self.config.get("detection.backend", "mock")
            try:
                self._detector = create_detector(
                    backend,
                    use_gpu=self.use_gpu,
                    confidence_threshold=self.config.get(
                        "detection.confidence_threshold", 0.3
                    ),
                    languages=self.config.get("detection.languages")
                    or self.config.get("ocr.languages")
                    or ["en"],
                    lang=self.config.get("detection.lang")
                    or self.config.get("pipeline.source_language", "en"),
                    min_size=self.config.get("detection.min_size", 10),
                    paragraph=self.config.get("detection.paragraph", False),
                    detect_max_dimension=self.config.get(
                        "detection.detect_max_dimension", 1600
                    ),
                    model=self.config.get("detection.model")
                    or self.config.get("detection.weights"),
                    imgsz=self.config.get("detection.imgsz", 1024),
                    classes=self.config.get("detection.classes"),
                    variant=self.config.get("detection.variant", "textseg"),
                    auto_download=self.config.get("detection.auto_download", True),
                )
            except Exception as e:
                logger.warning(f"Detector '{backend}' failed ({e}), using mock")
                self._detector = create_detector("mock")
        return self._detector

    def _get_ocr(self) -> OCRBackend:
        if self._ocr is None:
            backend = self.config.get("ocr.backend", "mock")
            try:
                self._ocr = create_ocr(
                    backend,
                    use_gpu=self.use_gpu,
                    device=self.device,
                    language_map=self.config.get("ocr.language_map"),
                    default_lang=self.config.get("pipeline.source_language", "ko"),
                    lang=self.config.get("ocr.lang")
                    or self.config.get("pipeline.source_language", "ko"),
                    languages=self.config.get("ocr.languages"),
                    model=self.config.get("ocr.model"),
                    prompt=self.config.get("ocr.prompt"),
                    max_new_tokens=self.config.get("ocr.max_new_tokens", 128),
                    load_in_4bit=self.config.get("ocr.load_in_4bit", False),
                    load_in_8bit=self.config.get("ocr.load_in_8bit", False),
                    pad_px=self.config.get("ocr.pad_px", 4),
                    min_pixels=self.config.get("ocr.min_pixels"),
                    max_pixels=self.config.get("ocr.max_pixels"),
                    local_files_only=self.config.get("ocr.local_files_only"),
                )
            except Exception as e:
                logger.warning(f"OCR '{backend}' failed ({e}), using mock")
                self._ocr = create_ocr("mock")
        return self._ocr

    def _get_translator(self) -> Translator:
        if self._translator is None:
            backend = self.config.get("translation.backend", "mock")
            try:
                self._translator = create_translator(
                    backend,
                    base_url=self.config.get("translation.base_url"),
                    api_key=self.config.get("translation.api_key"),
                    model=self.config.get("translation.model"),
                    temperature=self.config.get("translation.temperature", 0.2),
                    max_retries=self.config.get("translation.max_retries", 3),
                    device=self.device,
                    max_length=self.config.get("translation.max_length", 256),
                    batch_size=self.config.get("translation.batch_size", 8),
                    source_language=self.config.get("pipeline.source_language", "en"),
                    target_language=self.config.get("pipeline.target_language", "fa"),
                    pivot_model=self.config.get("translation.pivot_model"),
                    load_in_4bit=self.config.get("translation.load_in_4bit", False),
                    load_in_8bit=self.config.get("translation.load_in_8bit", False),
                    max_new_tokens=self.config.get("translation.max_new_tokens", 128),
                    local_files_only=self.config.get("translation.local_files_only", False),
                    validate=self.config.get("translation.validate", True),
                    max_retries=self.config.get("translation.max_retries", 1),
                    n_gpu_layers=self.config.get("translation.n_gpu_layers", -1),
                    n_ctx=self.config.get("translation.n_ctx", 2048),
                    max_tokens=self.config.get("translation.max_tokens", 128),
                    n_threads=self.config.get("translation.n_threads"),
                    chat_format=self.config.get("translation.chat_format"),
                    verbose=self.config.get("translation.verbose", False),
                )
            except Exception as e:
                logger.warning(f"Translator '{backend}' failed ({e}), using mock")
                self._translator = create_translator("mock")
        return self._translator

    def _get_inpainter(self) -> Inpainter:
        if self._inpainter is None:
            backend = self.config.get("inpainting.backend", "mock")
            try:
                self._inpainter = create_inpainter(
                    backend,
                    device=self.device,
                    tile_size=self.config.get("inpainting.tile_size", 768),
                    overlap=self.config.get("inpainting.overlap", 64),
                    precision=self.config.get("inpainting.precision", "fp16"),
                )
            except Exception as e:
                logger.warning(f"Inpainter '{backend}' failed ({e}), falling back to opencv")
                try:
                    self._inpainter = create_inpainter("opencv")
                except Exception:
                    self._inpainter = create_inpainter("mock")
        return self._inpainter

    def _unload_all(self) -> None:
        for backend in (self._detector, self._ocr, self._translator, self._inpainter):
            if backend is not None:
                try:
                    backend.unload()
                except Exception:
                    pass
        self._detector = None
        self._ocr = None
        self._translator = None
        self._inpainter = None
        clear_cuda()

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def _load_or_create_manifest(self, pages: List[str]) -> ChapterManifest:
        if self.manifest_path.exists() and not self.force:
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self.manifest = ChapterManifest(**data)
                # Ensure pages list is up to date
                if not self.manifest.pages:
                    self.manifest.pages = pages
                return self.manifest
            except Exception as e:
                logger.warning(f"Could not load existing manifest: {e}")

        self.manifest = ChapterManifest(
            chapter_id=self.chapter_id,
            source_language=self.config.get("pipeline.source_language", "auto"),
            target_language=self.config.get("pipeline.target_language", "fa"),
            pipeline_version=self.pipeline_version,
            pages=pages,
            page_status=[PageStatus(page=p, status=StageStatus.PENDING) for p in pages],
            config={
                "detection": self.config.section("detection"),
                "ocr": self.config.section("ocr"),
                "translation": {
                    k: v
                    for k, v in self.config.section("translation").items()
                    if k != "api_key"
                },
                "masking": self.config.section("masking"),
                "inpainting": self.config.section("inpainting"),
            },
        )
        self._save_manifest()
        return self.manifest

    def _save_manifest(self) -> None:
        if self.manifest is None:
            return
        self.manifest.updated_at = _now_iso()
        ensure_dir(self.chapter_dir)
        data = self.manifest.model_dump_json(indent=2)
        last_err = None
        for attempt in range(5):
            try:
                self.manifest_path.write_text(data, encoding="utf-8")
                return
            except OSError as e:
                last_err = e
                import time, tempfile, shutil
                time.sleep(0.15 * (attempt + 1))
                try:
                    fd, tmp_name = tempfile.mkstemp(suffix=".json")
                    import os
                    os.close(fd)
                    Path(tmp_name).write_text(data, encoding="utf-8")
                    shutil.move(tmp_name, str(self.manifest_path))
                    return
                except Exception:
                    continue
        logger.warning(f"Could not persist manifest after retries: {last_err}")

    def _set_stage_meta(self, stage: str, meta: StageMetadata) -> None:
        assert self.manifest is not None
        self.manifest.stages[stage] = meta
        self._save_manifest()

    def _update_page_status(
        self, page: str, status: StageStatus, stage: Optional[str] = None, error: Optional[str] = None
    ) -> None:
        if self.manifest is None:
            return
        for ps in self.manifest.page_status:
            if ps.page == page:
                ps.status = status
                ps.stage = stage
                ps.error = error
                break
        else:
            self.manifest.page_status.append(
                PageStatus(page=page, status=status, stage=stage, error=error)
            )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def run_preprocess(self) -> List[Path]:
        log_stage(logger, "preprocess", f"Discovering images in {self.chapter_dir}")

        if self.original_dir.is_dir() and list_images(self.original_dir):
            images = list_images(self.original_dir)
        else:
            images = list_images(self.chapter_dir, recursive=True)
            if images:
                ensure_dir(self.original_dir)
                import shutil
                for img in images:
                    dest = self.original_dir / img.name
                    if not dest.exists():
                        shutil.copy2(img, dest)
                images = list_images(self.original_dir)

        if not images:
            # Last resort: reused cleaned pages (translation-only recovery)
            if self.cleaned_dir.is_dir():
                images = list_images(self.cleaned_dir)
            if not images:
                exts = ", ".join(sorted(IMAGE_EXTENSIONS))
                raise FileNotFoundError(
                    f"No images found in {self.chapter_dir}\n"
                    f"Expected page images in:\n"
                    f"  - {self.original_dir}\n"
                    f"  - {self.chapter_dir} (or one subfolder)\n"
                    f"Supported extensions: {exts}"
                )

        pages = [page_id_from_path(p) for p in images]
        seen: Dict[str, int] = {}
        unique_pages: List[str] = []
        unique_images: List[Path] = []
        for p, img in zip(pages, images):
            if p not in seen:
                seen[p] = 0
                unique_pages.append(p)
                unique_images.append(img)
            else:
                seen[p] += 1
                new_id = f"{p}_{seen[p]}"
                unique_pages.append(new_id)
                unique_images.append(img)

        for d in (
            self.detection_dir,
            self.ocr_dir,
            self.translation_dir,
            self.masks_dir,
            self.cleaned_dir,
            self.debug_dir,
        ):
            ensure_dir(d)

        self._load_or_create_manifest(unique_pages)
        log_stage(logger, "preprocess", f"Found {len(unique_images)} pages")
        return unique_images

    def run_detection(self, images: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
        stage = "detection"
        log_stage(logger, stage, "Starting text detection")
        started = time.perf_counter()
        backend_name = self.config.get("detection.backend", "mock")
        meta = StageMetadata(
            stage=stage,
            backend=backend_name,
            model=backend_name,
            device=self.device,
            status=StageStatus.RUNNING,
            started_at=_now_iso(),
            config_hash=self.config.hash(),
        )
        self._set_stage_meta(stage, meta)

        input_hash = self.cache.hash_images(images)
        if self.cache.should_skip(
            stage, input_hash, backend_name, self.config.hash(),
            self.pipeline_version, self.manifest.cache_keys if self.manifest else {},
            force=self.force,
        ):
            log_stage(logger, stage, "SKIPPED (cache hit)")
            meta.status = StageStatus.SKIPPED
            meta.finished_at = _now_iso()
            self._set_stage_meta(stage, meta)
            # Load existing
            all_detections = {}
            for img_path in images:
                page = page_id_from_path(img_path)
                det_path = self.detection_dir / f"{page}.json"
                if det_path.exists():
                    all_detections[page] = json.loads(det_path.read_text(encoding="utf-8"))
                else:
                    all_detections[page] = []
            return all_detections

        detector = self._get_detector()
        all_detections: Dict[str, List[Dict[str, Any]]] = {}
        errors: List[str] = []

        for img_path in images:
            page = page_id_from_path(img_path)
            try:
                arr, info = self.preprocessor.load(img_path)
                dets = detector.detect(arr)
                for i, d in enumerate(dets):
                    d["page"] = page
                    d["region_index"] = i
                out_path = self.detection_dir / f"{page}.json"
                out_path.write_text(
                    json.dumps(dets, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                all_detections[page] = dets
                warnings = self.quality.check_detections(dets)
                if warnings:
                    meta.warnings.extend([f"page={page}: {w}" for w in warnings])
                self._update_page_status(page, StageStatus.SUCCESS, stage)
                log_stage(logger, stage, f"Page {page}: {len(dets)} regions")
            except Exception as e:
                msg = str(e)
                errors.append(f"page={page}: {msg}")
                log_error(logger, stage, msg, page=page)
                all_detections[page] = []
                self._update_page_status(page, StageStatus.FAILED, stage, msg)

        elapsed = (time.perf_counter() - started) * 1000
        meta.status = StageStatus.SUCCESS if not errors else StageStatus.FAILED
        meta.errors = errors
        meta.execution_time_ms = elapsed
        meta.finished_at = _now_iso()
        meta.duration_ms = elapsed
        meta.peak_vram_mb = peak_vram_mb()
        meta.output_hash = content_hash({k: len(v) for k, v in all_detections.items()})
        if self.manifest:
            self.manifest.cache_keys[stage] = self.cache.make_key(
                input_hash, backend_name, self.config.hash(), self.pipeline_version
            )
        self._set_stage_meta(stage, meta)
        log_vram(stage)

        if self.config.get("device.unload_after_stage", True):
            detector.unload()
            self._detector = None
            clear_cuda()

        return all_detections

    def run_ocr(
        self, images: List[Path], detections: Dict[str, List[Dict[str, Any]]]
    ) -> List[TextRegion]:
        stage = "ocr"
        log_stage(logger, stage, "Starting OCR")
        started = time.perf_counter()
        backend_name = self.config.get("ocr.backend", "mock")
        meta = StageMetadata(
            stage=stage,
            backend=backend_name,
            model=backend_name,
            device=self.device,
            status=StageStatus.RUNNING,
            started_at=_now_iso(),
            config_hash=self.config.hash(),
        )
        self._set_stage_meta(stage, meta)

        ocr = self._get_ocr()
        regions: List[TextRegion] = []
        errors: List[str] = []
        source_lang = self.config.get("pipeline.source_language", "ko")
        if source_lang == "auto":
            source_lang = "ko"

        for img_path in images:
            page = page_id_from_path(img_path)
            dets = detections.get(page, [])
            try:
                arr, _ = self.preprocessor.load(img_path)
                page_regions = []
                for idx, det in enumerate(dets):
                    region_id = f"{page}_{idx:04d}"
                    ocr_result = ocr.recognize(arr, det)
                    tr = TextRegion(
                        region_id=region_id,
                        page=page,
                        bbox=det.get("bbox", [0, 0, 0, 0]),
                        polygon=det.get("polygon", []),
                        region_type=det.get("region_type", "text"),
                        detection_confidence=float(det.get("confidence", 0.0)),
                        source_text=ocr_result.get("text", ""),
                        ocr_confidence=float(ocr_result.get("confidence", 0.0)),
                        source_language=ocr_result.get("language", source_lang),
                        target_language=self.config.get("pipeline.target_language", "fa"),
                    )
                    regions.append(tr)
                    page_regions.append(tr.model_dump())

                out_path = self.ocr_dir / f"{page}.json"
                out_path.write_text(
                    json.dumps(page_regions, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                self._update_page_status(page, StageStatus.SUCCESS, stage)
                log_stage(logger, stage, f"Page {page}: {len(page_regions)}/{len(dets)} successful")
            except Exception as e:
                msg = str(e)
                errors.append(f"page={page}: {msg}")
                log_error(logger, stage, msg, page=page)
                self._update_page_status(page, StageStatus.FAILED, stage, msg)

        warnings = self.quality.check_ocr([r.model_dump() for r in regions])
        meta.warnings = warnings
        elapsed = (time.perf_counter() - started) * 1000
        meta.status = StageStatus.SUCCESS if not errors else StageStatus.FAILED
        meta.errors = errors
        meta.execution_time_ms = elapsed
        meta.finished_at = _now_iso()
        meta.duration_ms = elapsed
        meta.peak_vram_mb = peak_vram_mb()
        if self.manifest:
            self.manifest.regions = regions
            self.manifest.cache_keys[stage] = self.cache.make_key(
                content_hash([r.region_id for r in regions]),
                backend_name,
                self.config.hash(),
                self.pipeline_version,
            )
        self._set_stage_meta(stage, meta)
        log_vram(stage)

        if self.config.get("device.unload_after_stage", True):
            ocr.unload()
            self._ocr = None
            clear_cuda()

        return regions


    # Scan-site / watermark patterns to drop from EN passages
    _OCR_JUNK_PATTERNS = (
        r"luacomic", r"lua\s*scans?", r"luascans", r"read\s+on\s+our",
        r"discord", r"dsc\.?\s*gg", r"http", r"www\.", r"\.net", r"\.com",
        r"isbn", r"fastest\s+release", r"please\s+visit", r"without\s+your\s+support",
        r"only\s+official\s+domain", r"any\s+other\s+domain", r"scam",
        r"carrotoon", r"carro+o+n", r"kwbooks", r"twitter", r"join\s+our",
        r"toomics", r"webtoon", r"tapas",
    )

    def _is_gibberish_token(self, token: str) -> bool:
        """Heuristic OCR garbage token (e.g. ANYNMORE, CARROOON)."""
        import re
        t = (token or "").strip()
        if len(t) < 4:
            return False
        # long run of same letter
        if re.search(r"(.)\1{3,}", t, re.I):
            return True
        letters = [c for c in t if c.isalpha()]
        if len(letters) < 4:
            return False
        vowels = sum(c.lower() in "aeiou" for c in letters)
        # very low vowel ratio on long tokens
        if len(letters) >= 6 and vowels / len(letters) < 0.15:
            return True
        # known OCR mangling
        if re.fullmatch(r"(?i)anynmore|reto|carro+o*n+", t):
            return True
        return False

    def _is_junk_ocr(self, text: str) -> bool:
        import re
        t = (text or "").strip().lower()
        if not t:
            return True
        if len(t) <= 2 and not t.isalpha():
            return True
        for pat in self._OCR_JUNK_PATTERNS:
            if re.search(pat, t, re.I):
                return True
        # mostly non-letters
        letters = sum(c.isalpha() for c in t)
        if letters < max(2, len(t) * 0.35):
            return True
        # whole line is a single gibberish token
        if len(t.split()) == 1 and self._is_gibberish_token(t):
            return True
        return False

    def _clean_ocr_line(self, text: str) -> str:
        import re
        t = (text or "").strip()
        t = t.replace("_", " ")
        t = re.sub(r"\s+", " ", t)
        # Drop gibberish tokens inside otherwise good lines
        kept = [w for w in t.split() if not self._is_gibberish_token(w)]
        return " ".join(kept).strip()
    def run_translation(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Merge OCR lines per page → translate page passages → write EN+FA chapter files."""
        stage = "translation"
        log_stage(logger, stage, "Starting page-level chapter translation")
        started = time.perf_counter()
        backend_name = self.config.get("translation.backend", "mock")
        meta = StageMetadata(
            stage=stage,
            backend=backend_name,
            model=self.config.get("translation.model") or backend_name,
            device="cpu",
            status=StageStatus.RUNNING,
            started_at=_now_iso(),
            config_hash=self.config.hash(),
        )
        self._set_stage_meta(stage, meta)

        translator = self._get_translator()
        source_lang = self.config.get("pipeline.source_language", "en")
        if source_lang == "auto":
            source_lang = "en"
        target_lang = self.config.get("pipeline.target_language", "fa")
        glossary = self.config.get("translation.glossary") or {}

        errors: List[str] = []
        try:
            # --- 1) Group regions by page, merge EN in reading order ---
            from collections import defaultdict
            import re

            by_page: Dict[str, List[TextRegion]] = defaultdict(list)
            for r in regions:
                by_page[str(r.page)].append(r)

            def merge_en(lines: List[str]) -> str:
                parts = []
                for t in lines:
                    t = self._clean_ocr_line(t)
                    if not t or self._is_junk_ocr(t):
                        continue
                    if re.fullmatch(r"[\W\d_#]+", t):
                        continue
                    # skip very low-signal fragments
                    if len(t) < 2:
                        continue
                    parts.append(t)
                if not parts:
                    return ""
                text = " ".join(parts)
                text = re.sub(r"\s+", " ", text)
                text = re.sub(r"\s+([,.!?])", r"\1", text)
                # drop leftover site fragments mid-sentence
                for bad in ("LUACOMIC", "LuaComic", "LUA SCANS", "LUASCAN"):
                    text = text.replace(bad, "")
                text = re.sub(r"\s+", " ", text).strip(" -;,.|_")
                return text.strip()

            page_ids = sorted(by_page.keys(), key=lambda p: (len(p), p))
            page_en: Dict[str, str] = {}
            for page in page_ids:
                items = sorted(
                    by_page[page],
                    key=lambda r: (
                        float((r.bbox or [0, 0, 0, 0])[1]),
                        float((r.bbox or [0, 0, 0, 0])[0]),
                    ),
                )
                page_en[page] = merge_en([r.source_text or "" for r in items])

            # --- 2) Translate each page as ONE passage (better quality) ---
            context = {
                "source_language": source_lang,
                "target_language": target_lang,
                "glossary": glossary,
                "chapter_context": [],
            }
            page_fa: Dict[str, str] = {}
            batch = []
            for page in page_ids:
                en = page_en.get(page) or ""
                if not en:
                    page_fa[page] = ""
                    continue
                batch.append({"region_id": f"page_{page}", "source_text": en, "page": page})

            translations = translator.translate_chapter(batch, context) if batch else []
            for t in translations:
                rid = t.get("region_id", "")
                page = rid.replace("page_", "", 1) if rid.startswith("page_") else rid
                page_fa[page] = (t.get("text") or "").strip()

            # Fill region-level FA with the page passage (for JSON completeness)
            for r in regions:
                r.target_language = target_lang
                r.translated_text = page_fa.get(str(r.page), "")

            # --- 3) Write readable chapter files ---
            ensure_dir(self.translation_dir)
            passages = []
            en_blocks, fa_blocks, both_blocks = [], [], []
            for page in page_ids:
                en = page_en.get(page) or ""
                fa = page_fa.get(page) or ""
                if not en and not fa:
                    continue
                passages.append({"page": page, "en": en, "fa": fa, "n_regions": len(by_page[page])})
                en_blocks.append(f"=== Page {page} ===\n{en}\n")
                fa_blocks.append(f"=== صفحه {page} ===\n{fa}\n")
                both_blocks.append(
                    f"=== Page {page} ===\n"
                    f"EN: {en}\n"
                    f"FA: {fa}\n"
                )

            (self.translation_dir / "chapter_en_fa.txt").write_text(
                "\n".join(both_blocks), encoding="utf-8"
            )
            (self.translation_dir / "chapter_en.txt").write_text(
                "\n".join(en_blocks), encoding="utf-8"
            )
            (self.translation_dir / "chapter_fa.txt").write_text(
                "\n".join(fa_blocks), encoding="utf-8"
            )
            (self.translation_dir / "chapter_full_en.txt").write_text(
                "\n\n".join(p["en"] for p in passages if p["en"]), encoding="utf-8"
            )
            (self.translation_dir / "chapter_full_fa.txt").write_text(
                "\n\n".join(p["fa"] for p in passages if p["fa"]), encoding="utf-8"
            )
            (self.translation_dir / "chapter_passages.json").write_text(
                json.dumps(
                    {
                        "chapter_id": self.chapter_id,
                        "source_language": source_lang,
                        "target_language": target_lang,
                        "passages": passages,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            chapter_json = {
                "chapter_id": self.chapter_id,
                "source_language": source_lang,
                "target_language": target_lang,
                "passages": passages,
                "regions": [r.model_dump() for r in regions],
            }
            (self.translation_dir / "chapter.json").write_text(
                json.dumps(chapter_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log_stage(
                logger,
                stage,
                f"{len(passages)} page passages translated "
                f"({sum(1 for p in passages if p['en'])} non-empty EN)",
            )
            logger.info(
                f"Readable output: {self.translation_dir / 'chapter_en_fa.txt'}"
            )
        except Exception as e:
            msg = str(e)
            errors.append(msg)
            log_error(logger, stage, msg)
            elapsed = (time.perf_counter() - started) * 1000
            meta.status = StageStatus.FAILED
            meta.errors = errors
            meta.execution_time_ms = elapsed
            meta.finished_at = _now_iso()
            meta.duration_ms = elapsed
            self._set_stage_meta(stage, meta)
            if self.config.get("device.unload_after_stage", True):
                try:
                    translator.unload()
                except Exception:
                    pass
                self._translator = None
            raise RuntimeError(f"Translation failed: {msg}") from e

        elapsed = (time.perf_counter() - started) * 1000
        meta.status = StageStatus.SUCCESS if not errors else StageStatus.FAILED
        meta.errors = errors
        meta.execution_time_ms = elapsed
        meta.finished_at = _now_iso()
        meta.duration_ms = elapsed
        if self.manifest:
            self.manifest.regions = regions
            self.manifest.cache_keys[stage] = self.cache.make_key(
                content_hash([r.region_id for r in regions]),
                backend_name,
                self.config.hash(),
                self.pipeline_version,
            )
        self._set_stage_meta(stage, meta)

        if self.config.get("device.unload_after_stage", True):
            translator.unload()
            self._translator = None

        return regions

    def run_masking(
        self, images: List[Path], regions: List[TextRegion]
    ) -> Dict[str, np.ndarray]:
        stage = "masking"
        log_stage(logger, stage, "Generating masks")
        started = time.perf_counter()
        meta = StageMetadata(
            stage=stage,
            backend="polygon_mask",
            device="cpu",
            status=StageStatus.RUNNING,
            started_at=_now_iso(),
            config_hash=self.config.hash(),
        )
        self._set_stage_meta(stage, meta)

        by_page: Dict[str, List[Dict]] = {}
        for r in regions:
            by_page.setdefault(r.page, []).append(r.model_dump())

        masks: Dict[str, np.ndarray] = {}
        errors: List[str] = []

        for img_path in images:
            page = page_id_from_path(img_path)
            try:
                arr, info = self.preprocessor.load(img_path)
                page_regions = by_page.get(page, [])
                mask = self.mask_gen.generate((info.height, info.width), page_regions, image=arr)
                mask_path = self.masks_dir / f"{page}.png"
                cv2.imwrite(str(mask_path), mask)
                masks[page] = mask

                preview = arr.copy()
                if mask.max() > 0:
                    overlay = preview.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    preview = cv2.addWeighted(preview, 0.6, overlay, 0.4, 0)
                debug_path = self.debug_dir / f"{page}_mask_preview.png"
                cv2.imwrite(str(debug_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                log_stage(logger, stage, f"Page {page}: mask generated")
            except Exception as e:
                msg = str(e)
                errors.append(f"page={page}: {msg}")
                log_error(logger, stage, msg, page=page)

        elapsed = (time.perf_counter() - started) * 1000
        meta.status = StageStatus.SUCCESS if not errors else StageStatus.FAILED
        meta.errors = errors
        meta.execution_time_ms = elapsed
        meta.finished_at = _now_iso()
        meta.duration_ms = elapsed
        if self.manifest:
            self.manifest.cache_keys[stage] = self.cache.make_key(
                content_hash(list(masks.keys())),
                "mask",
                self.config.hash(),
                self.pipeline_version,
            )
        self._set_stage_meta(stage, meta)
        return masks

    def run_inpainting(
        self, images: List[Path], masks: Dict[str, np.ndarray]
    ) -> None:
        stage = "inpainting"
        log_stage(logger, stage, "Starting inpainting / text removal")
        started = time.perf_counter()
        backend_name = self.config.get("inpainting.backend", "mock")
        meta = StageMetadata(
            stage=stage,
            backend=backend_name,
            model=backend_name,
            device=self.device,
            status=StageStatus.RUNNING,
            started_at=_now_iso(),
            config_hash=self.config.hash(),
            extra={
                "tile_size": self.config.get("inpainting.tile_size"),
                "overlap": self.config.get("inpainting.overlap"),
            },
        )
        self._set_stage_meta(stage, meta)

        inpainter = self._get_inpainter()
        errors: List[str] = []

        for img_path in images:
            page = page_id_from_path(img_path)
            try:
                arr, info = self.preprocessor.load(img_path)
                mask = masks.get(page)
                if mask is None:
                    mask = np.zeros((info.height, info.width), dtype=np.uint8)
                # Resize mask if preprocessing resized the image
                if mask.shape[:2] != arr.shape[:2]:
                    mask = cv2.resize(mask, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)

                try:
                    cleaned = inpainter.inpaint(arr, mask)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM on page {page}, clearing CUDA and falling back to OpenCV")
                        clear_cuda()
                        from manga_ai.inpainting.opencv_backend import OpenCVInpainter
                        cleaned = OpenCVInpainter().inpaint(arr, mask)
                        meta.warnings.append(f"page={page}: OOM fallback to opencv")
                    else:
                        raise

                warnings = self.quality.check_inpaint(arr, cleaned, mask)
                if warnings:
                    meta.warnings.extend([f"page={page}: {w}" for w in warnings])

                out_path = self.cleaned_dir / f"{page}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))
                self._update_page_status(page, StageStatus.SUCCESS, stage)
                log_stage(logger, stage, f"Page {page}: cleaned")
            except Exception as e:
                msg = str(e)
                errors.append(f"page={page}: {msg}")
                log_error(logger, stage, msg, page=page)
                self._update_page_status(page, StageStatus.FAILED, stage, msg)

        elapsed = (time.perf_counter() - started) * 1000
        meta.status = StageStatus.SUCCESS if not errors else StageStatus.FAILED
        meta.errors = errors
        meta.execution_time_ms = elapsed
        meta.finished_at = _now_iso()
        meta.duration_ms = elapsed
        meta.peak_vram_mb = peak_vram_mb()
        if self.manifest:
            self.manifest.cache_keys[stage] = self.cache.make_key(
                content_hash([p.name for p in images]),
                backend_name,
                self.config.hash(),
                self.pipeline_version,
            )
        self._set_stage_meta(stage, meta)
        log_vram(stage)

        if self.config.get("device.unload_after_stage", True):
            inpainter.unload()
            self._inpainter = None
            clear_cuda()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self) -> ChapterManifest:
        log_stage(logger, "pipeline", f"Processing chapter {self.chapter_id}")
        images = self.run_preprocess()
        detections = self.run_detection(images)
        regions = self.run_ocr(images, detections)
        regions = self.run_translation(regions)
        masks = self.run_masking(images, regions)
        self.run_inpainting(images, masks)
        self._unload_all()
        log_stage(logger, "pipeline", "Completed")
        assert self.manifest is not None
        return self.manifest

    def run_stage(self, stage: str) -> None:
        # Translation can run from existing OCR/manifest without page images.
        if stage in ("translate", "translation"):
            regions: List[TextRegion] = []
            if self.manifest_path.exists():
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self.manifest = ChapterManifest(**data)
                regions = list(self.manifest.regions)
            if not regions and self.ocr_dir.is_dir():
                ensure_dir(self.translation_dir)
                for ocr_path in sorted(self.ocr_dir.glob("*.json")):
                    for item in json.loads(ocr_path.read_text(encoding="utf-8")):
                        regions.append(TextRegion(**item))
            if not regions:
                # Last resort: discover images then load matching OCR jsons
                images = self.run_preprocess()
                for p in images:
                    page = page_id_from_path(p)
                    ocr_path = self.ocr_dir / f"{page}.json"
                    if ocr_path.exists():
                        for item in json.loads(ocr_path.read_text(encoding="utf-8")):
                            regions.append(TextRegion(**item))
            if not regions:
                raise FileNotFoundError(
                    f"No OCR regions found for translation in {self.chapter_dir}.\n"
                    f"Need {self.manifest_path.name} or files under {self.ocr_dir}"
                )
            self.run_translation(regions)
            self._unload_all()
            return

        images = self.run_preprocess()
        if stage in ("detect", "detection"):
            self.run_detection(images)
        elif stage == "ocr":
            detections = {}
            for p in images:
                page = page_id_from_path(p)
                det_path = self.detection_dir / f"{page}.json"
                if det_path.exists():
                    detections[page] = json.loads(det_path.read_text(encoding="utf-8"))
                else:
                    detections[page] = []
            self.run_ocr(images, detections)
        elif stage in ("mask", "masking"):
            regions = []
            if self.manifest_path.exists():
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self.manifest = ChapterManifest(**data)
                regions = list(self.manifest.regions)
            self.run_masking(images, regions)
        elif stage in ("clean", "inpainting"):
            masks = {}
            for p in images:
                page = page_id_from_path(p)
                mask_path = self.masks_dir / f"{page}.png"
                if mask_path.exists():
                    masks[page] = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.run_inpainting(images, masks)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        self._unload_all()
