"""Standalone translation smoke test using a real config's settings.

Verifies the translation backend (e.g. local Gemma) loads and produces Persian,
WITHOUT needing detection/OCR output. It builds the translator exactly like the
pipeline does (same model, 4-bit, style/prompt, glossary, validation), runs it
on a few sample lines (or your own), and writes the result into results/latest/
so it can be committed and reviewed.

Usage:
    python scripts/smoke_translate.py -c hybrid_qwen_gemma_it
    python scripts/smoke_translate.py -c hybrid_qwen_gemma_it --file my_lines.txt
    python scripts/smoke_translate.py -c hybrid_qwen_gemma_it --text "Don't move."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow `python scripts/smoke_translate.py` to import the manga_ai package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manga_ai.config import Config, resolve_config_path
from manga_ai.translation import create_translator
from manga_ai.utils.device import get_device

DEFAULT_SAMPLES = [
    "I will make you pay for this!",
    "Young Master, the enemy has breached the outer wall.",
    "Don't move. One step and you're dead.",
    "Why are you always late, Princess?",
    "System notification: Quest completed.",
    "Please, Your Highness... spare them.",
    "Jennette smiled, but her eyes were cold.",
    "What... what did you just say to me?",
]


def build_translator(cfg: Config):
    device = get_device(cfg.get("device.preferred", "cuda"), int(cfg.get("device.device_id", 0)))
    backend = cfg.get("translation.backend", "mock")
    return create_translator(
        backend,
        base_url=cfg.get("translation.base_url"),
        api_key=cfg.get("translation.api_key"),
        model=cfg.get("translation.model"),
        temperature=cfg.get("translation.temperature", 0.2),
        max_retries=cfg.get("translation.max_retries", 1),
        device=device,
        max_length=cfg.get("translation.max_length", 256),
        batch_size=cfg.get("translation.batch_size", 8),
        source_language=cfg.get("pipeline.source_language", "en"),
        target_language=cfg.get("pipeline.target_language", "fa"),
        pivot_model=cfg.get("translation.pivot_model"),
        load_in_4bit=cfg.get("translation.load_in_4bit", False),
        load_in_8bit=cfg.get("translation.load_in_8bit", False),
        max_new_tokens=cfg.get("translation.max_new_tokens", 128),
        local_files_only=cfg.get("translation.local_files_only", False),
        validate=cfg.get("translation.validate", True),
        style=cfg.get("translation.style", "colloquial_fa"),
        extra_instructions=cfg.get("translation.extra_instructions", "") or "",
        context_window=cfg.get("translation.context_window", 1),
        mt_target_label=cfg.get("translation.mt_target_label", "Persian"),
        n_gpu_layers=cfg.get("translation.n_gpu_layers", -1),
        n_ctx=cfg.get("translation.n_ctx", 2048),
        max_tokens=cfg.get("translation.max_tokens", 128),
        n_threads=cfg.get("translation.n_threads"),
        chat_format=cfg.get("translation.chat_format"),
        verbose=cfg.get("translation.verbose", False),
    ), device, backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--file", default=None, help="Text file, one source line per line")
    ap.add_argument("--text", default=None, help="A single line to translate")
    ap.add_argument("--out", default="results/latest")
    args = ap.parse_args()

    resolved = resolve_config_path(args.config)
    cfg = Config.load(config_path=str(resolved) if resolved else None)

    if args.text:
        lines = [args.text]
    elif args.file:
        lines = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        lines = DEFAULT_SAMPLES

    print(f"config={args.config} backend={cfg.get('translation.backend')} model={cfg.get('translation.model')!r}")
    print(f"loading translator...")
    t0 = time.perf_counter()
    translator, device, backend = build_translator(cfg)
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s on {device}")

    batch = [{"region_id": f"t{i}", "source_text": s, "page": "smoke"} for i, s in enumerate(lines)]
    ctx = {
        "source_language": cfg.get("pipeline.source_language", "en"),
        "target_language": cfg.get("pipeline.target_language", "fa"),
        "glossary": cfg.get("translation.glossary") or {},
        "chapter_context": [],
    }
    t1 = time.perf_counter()
    out = translator.translate_chapter(batch, ctx)
    run_s = time.perf_counter() - t1
    by_id = {o.get("region_id"): (o.get("text") or "").strip() for o in out}

    rows = []
    print("\n--- EN -> FA ---")
    n_ok = 0
    for b in batch:
        fa = by_id.get(b["region_id"], "")
        if fa:
            n_ok += 1
        print(f"EN: {b['source_text']}")
        print(f"FA: {fa}\n")
        rows.append({"en": b["source_text"], "fa": fa})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = [
        f"# Translation smoke test",
        f"config: {args.config}",
        f"backend: {backend}   model: {cfg.get('translation.model')!r}",
        f"device: {device}   load_s: {load_s:.1f}   run_s: {run_s:.1f}",
        f"lines: {len(rows)}   non-empty FA: {n_ok}",
        "",
    ]
    for r in rows:
        txt.append(f"EN: {r['en']}")
        txt.append(f"FA: {r['fa']}")
        txt.append("")
    (out_dir / "translation_smoke.txt").write_text("\n".join(txt), encoding="utf-8")
    (out_dir / "translation_smoke.json").write_text(
        json.dumps({"config": args.config, "backend": backend, "rows": rows,
                    "load_s": load_s, "run_s": run_s, "non_empty": n_ok}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{n_ok}/{len(rows)} lines translated in {run_s:.1f}s -> {out_dir/'translation_smoke.txt'}")

    try:
        translator.unload()
    except Exception:
        pass


if __name__ == "__main__":
    main()
