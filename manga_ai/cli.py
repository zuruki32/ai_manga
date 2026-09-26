"""Command-line interface for manga-ai."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import click

from manga_ai import __version__
from manga_ai.config import Config, list_bundled_configs, resolve_config_path
from manga_ai.logging import get_logger, setup_logging
from manga_ai.pipeline import Pipeline
from manga_ai.utils import list_images, page_id_from_path

logger = get_logger("manga_ai.cli")


def _resolve_config(
    config: Optional[str],
    backend: Optional[str],
    force: bool,
) -> Config:
    overrides = {}
    if force:
        overrides["pipeline"] = {"force": True}
    resolved = resolve_config_path(config) if config else None
    if config and resolved is None:
        available = ", ".join(list_bundled_configs()) or "(none found)"
        raise click.ClickException(
            f"Config not found: {config!r}. "
            f"Pull/checkout the branch that adds it, or pass an absolute path.\n"
            f"Available bundled configs: {available}"
        )
    cfg = Config.load(
        config_path=str(resolved) if resolved else None,
        overrides=overrides or None,
    )
    if backend:
        cfg = cfg.with_backend(backend)
    return cfg


@click.group()
@click.version_option(__version__, prog_name="manga-ai")
def main() -> None:
    """Manga / Manhwa AI Cleaning & Translation Pipeline."""
    pass


@main.command()
@click.argument("chapter_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config",
    "-c",
    type=str,
    default=None,
    help="Config path or bare name (e.g. hybrid_qwen_en / configs/hybrid_qwen_en.yaml)",
)
@click.option("--backend", type=str, default=None, help="Force all backends (e.g. mock)")
@click.option("--force", is_flag=True, help="Ignore cache and re-run all stages")
def process(chapter_dir: str, config: Optional[str], backend: Optional[str], force: bool) -> None:
    """Run the full pipeline on a chapter directory."""
    cfg = _resolve_config(config, backend, force)
    setup_logging(level=cfg.get("logging.level", "INFO"))
    # Make misconfigured GGUF obvious before burning OCR time
    logger.info(
        "Config translation: backend=%r model=%r",
        cfg.get("translation.backend"),
        cfg.get("translation.model"),
    )
    pipe = Pipeline(cfg, chapter_dir)
    manifest = pipe.process()
    done = (
        f"\nDone. Manifest: {pipe.manifest_path}\n"
        f"Pages: {len(manifest.pages)}\n"
        f"Regions: {len(manifest.regions)}"
    )
    # Windows often closes the console handle after long GPU runs (error 6)
    try:
        click.echo(done)
    except Exception:
        try:
            print(done, flush=True)
        except Exception:
            pass


@main.command()
@click.argument("chapter_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config",
    "-c",
    type=str,
    default=None,
    help="Config path or bare name (e.g. hybrid_qwen_en)",
)
@click.option("--backend", type=str, default=None)
@click.option("--force", is_flag=True)
def detect(chapter_dir: str, config: Optional[str], backend: Optional[str], force: bool) -> None:
    """Run text detection only."""
    cfg = _resolve_config(config, backend, force)
    setup_logging(level=cfg.get("logging.level", "INFO"))
    Pipeline(cfg, chapter_dir).run_stage("detection")


@main.command()
@click.argument("chapter_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config",
    "-c",
    type=str,
    default=None,
    help="Config path or bare name (e.g. hybrid_qwen_en)",
)
@click.option("--backend", type=str, default=None)
@click.option("--force", is_flag=True)
def ocr(chapter_dir: str, config: Optional[str], backend: Optional[str], force: bool) -> None:
    """Run OCR only."""
    cfg = _resolve_config(config, backend, force)
    setup_logging(level=cfg.get("logging.level", "INFO"))
    Pipeline(cfg, chapter_dir).run_stage("ocr")


@main.command()
@click.argument("chapter_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config",
    "-c",
    type=str,
    default=None,
    help="Config path or bare name (e.g. hybrid_qwen_en)",
)
@click.option("--backend", type=str, default=None)
@click.option("--force", is_flag=True)
def translate(chapter_dir: str, config: Optional[str], backend: Optional[str], force: bool) -> None:
    """Run chapter-level translation only."""
    cfg = _resolve_config(config, backend, force)
    setup_logging(level=cfg.get("logging.level", "INFO"))
    Pipeline(cfg, chapter_dir).run_stage("translation")


@main.command()
@click.argument("chapter_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config",
    "-c",
    type=str,
    default=None,
    help="Config path or bare name (e.g. hybrid_qwen_en)",
)
@click.option("--force", is_flag=True)
def mask(chapter_dir: str, config: Optional[str], force: bool) -> None:
    """Generate text masks only."""
    cfg = _resolve_config(config, None, force)
    setup_logging(level=cfg.get("logging.level", "INFO"))
    Pipeline(cfg, chapter_dir).run_stage("masking")


@main.command()
@click.argument("chapter_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config",
    "-c",
    type=str,
    default=None,
    help="Config path or bare name (e.g. hybrid_qwen_en)",
)
@click.option("--backend", type=str, default=None)
@click.option("--force", is_flag=True)
def clean(chapter_dir: str, config: Optional[str], backend: Optional[str], force: bool) -> None:
    """Run inpainting / text removal only (alias: manga-ai inpaint)."""
    cfg = _resolve_config(config, backend, force)
    setup_logging(level=cfg.get("logging.level", "INFO"))
    Pipeline(cfg, chapter_dir).run_stage("inpainting")


# Alias used in docs / muscle memory
main.add_command(clean, name="inpaint")


@main.command("benchmark")
@click.argument("stage", type=click.Choice(["detection", "ocr", "inpainting", "full"]))
@click.argument("chapter_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--config",
    "-c",
    type=str,
    default=None,
    help="Config path or bare name (e.g. hybrid_qwen_en)",
)
@click.option("--backend", type=str, default="mock")
def benchmark_cmd(stage: str, chapter_dir: str, config: Optional[str], backend: str) -> None:
    """Benchmark a stage (time, VRAM, counts)."""
    cfg = _resolve_config(config, backend, force=True)
    setup_logging(level="INFO")
    pipe = Pipeline(cfg, chapter_dir)
    images = pipe.run_preprocess()
    t0 = time.perf_counter()

    if stage == "detection":
        dets = pipe.run_detection(images)
        n = sum(len(v) for v in dets.values())
        click.echo(f"detections={n} pages={len(images)}")
    elif stage == "ocr":
        dets = {}
        for p in images:
            page = page_id_from_path(p)
            path = pipe.detection_dir / f"{page}.json"
            dets[page] = json.loads(path.read_text()) if path.exists() else []
        regions = pipe.run_ocr(images, dets)
        click.echo(f"ocr_regions={len(regions)}")
    elif stage == "inpainting":
        masks = {}
        for p in images:
            page = page_id_from_path(p)
            mp = pipe.masks_dir / f"{page}.png"
            if mp.exists():
                import cv2
                masks[page] = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        pipe.run_inpainting(images, masks)
    else:
        pipe.process()

    elapsed = time.perf_counter() - t0
    from manga_ai.utils.device import peak_vram_mb
    vram = peak_vram_mb()
    click.echo(f"time_s={elapsed:.2f}")
    if vram is not None:
        click.echo(f"peak_vram_mb={vram:.0f}")
    click.echo(f"pages_per_hour={len(images) / elapsed * 3600:.1f}" if elapsed > 0 else "")


@main.command("telegram")
@click.option("--token", default=None, help="Bot token (else TELEGRAM_BOT_TOKEN / .env)")
def telegram_cmd(token: Optional[str]) -> None:
    """Run Irisekai Telegram bot (projects, names, ZIP chapters)."""
    from manga_ai.telegram.bot import run_bot

    run_bot(token=token)


if __name__ == "__main__":
    main()
