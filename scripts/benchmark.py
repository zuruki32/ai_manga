#!/usr/bin/env python3
"""Simple benchmark runner for detection / OCR / inpainting stages."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np

from manga_ai.config import Config
from manga_ai.logging import setup_logging, get_logger
from manga_ai.preprocessing import Preprocessor
from manga_ai.utils import list_images, page_id_from_path
from manga_ai.utils.device import peak_vram_mb, reset_peak_vram, get_device

logger = get_logger("manga_ai.benchmark")


@click.command()
@click.argument("stage", type=click.Choice(["detection", "ocr", "inpainting"]))
@click.argument("dataset", type=click.Path(exists=True, file_okay=False))
@click.option("--config", "-c", type=click.Path(exists=True), default=None)
@click.option("--backend", default=None)
@click.option("--limit", default=10, help="Max pages to benchmark")
def main(stage: str, dataset: str, config: str | None, backend: str | None, limit: int):
    """Benchmark a single stage on a folder of images."""
    setup_logging("INFO")
    overrides = {}
    if backend:
        overrides[stage if stage != "detection" else "detection"] = {"backend": backend}
        if stage == "ocr":
            overrides["ocr"] = {"backend": backend}
        if stage == "inpainting":
            overrides["inpainting"] = {"backend": backend}
    cfg = Config.load(config_path=config, overrides=overrides or None)

    images = list_images(dataset)[:limit]
    if not images:
        # try original/ subfolder
        images = list_images(Path(dataset) / "original")[:limit]
    if not images:
        raise SystemExit(f"No images in {dataset}")

    pre = Preprocessor(
        convert_to_rgb=True,
        max_dimension=cfg.get("preprocessing.max_dimension"),
    )
    device = get_device(cfg.get("device.preferred", "cuda"))
    results = []

    if stage == "detection":
        from manga_ai.detection import get_detector
        det = get_detector(cfg.get("detection.backend", "mock"))
        for img_path in images:
            arr, info = pre.load(img_path)
            reset_peak_vram(device)
            t0 = time.perf_counter()
            regions = det.detect(arr)
            elapsed = (time.perf_counter() - t0) * 1000
            vram = peak_vram_mb(device)
            results.append({
                "page": page_id_from_path(img_path),
                "regions": len(regions),
                "time_ms": round(elapsed, 1),
                "peak_vram_mb": round(vram, 1) if vram else None,
            })
            logger.info(f"{img_path.name}: {len(regions)} regions in {elapsed:.0f}ms")
        det.unload()

    elif stage == "ocr":
        from manga_ai.ocr import get_ocr_backend
        from manga_ai.detection import get_detector
        det = get_detector(cfg.get("detection.backend", "mock"))
        ocr = get_ocr_backend(cfg.get("ocr.backend", "mock"))
        for img_path in images:
            arr, info = pre.load(img_path)
            regions = det.detect(arr)
            reset_peak_vram(device)
            t0 = time.perf_counter()
            ok = 0
            for r in regions:
                out = ocr.recognize(arr, r)
                if out.get("text"):
                    ok += 1
            elapsed = (time.perf_counter() - t0) * 1000
            vram = peak_vram_mb(device)
            results.append({
                "page": page_id_from_path(img_path),
                "regions": len(regions),
                "ocr_ok": ok,
                "time_ms": round(elapsed, 1),
                "peak_vram_mb": round(vram, 1) if vram else None,
            })
            logger.info(f"{img_path.name}: OCR {ok}/{len(regions)} in {elapsed:.0f}ms")
        det.unload()
        ocr.unload()

    elif stage == "inpainting":
        from manga_ai.inpainting import get_inpainter
        from manga_ai.masking import MaskGenerator
        from manga_ai.detection import get_detector
        det = get_detector(cfg.get("detection.backend", "mock"))
        mask_gen = MaskGenerator(
            dilation_px=cfg.get("masking.dilation_px", 8),
            blur_radius=cfg.get("masking.blur_radius", 1),
        )
        inp = get_inpainter(
            cfg.get("inpainting.backend", "mock"),
            tile_size=cfg.get("inpainting.tile_size", 768),
            overlap=cfg.get("inpainting.overlap", 64),
        )
        for img_path in images:
            arr, info = pre.load(img_path)
            regions = det.detect(arr)
            mask = mask_gen.generate((info.height, info.width), regions)
            reset_peak_vram(device)
            t0 = time.perf_counter()
            cleaned = inp.inpaint(arr, mask)
            elapsed = (time.perf_counter() - t0) * 1000
            vram = peak_vram_mb(device)
            results.append({
                "page": page_id_from_path(img_path),
                "shape": list(cleaned.shape),
                "time_ms": round(elapsed, 1),
                "peak_vram_mb": round(vram, 1) if vram else None,
            })
            logger.info(f"{img_path.name}: inpaint {elapsed:.0f}ms vram={vram}")
        det.unload()
        inp.unload()

    summary = {
        "stage": stage,
        "pages": len(results),
        "avg_time_ms": round(sum(r["time_ms"] for r in results) / max(1, len(results)), 1),
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
