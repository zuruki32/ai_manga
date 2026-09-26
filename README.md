# manga-ai

**Manga / Manhwa AI Cleaning & Translation Pipeline — V1**

Local-first, modular pipeline:

```
Input Chapter → Preprocess → Detect → OCR → Chapter Translate → Mask → Inpaint → Clean images + JSON
```

**V1 does NOT typeset Persian text back onto pages** (that is V2).

## Target hardware

- NVIDIA RTX 2060 **6 GB VRAM** (or better)
- Also runs fully on **CPU** with `--backend mock`

## Install

```bash
cd manga-ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# Optional real backends
pip install paddlepaddle paddleocr          # detection + OCR
pip install manga-ocr                       # Japanese OCR
pip install simple-lama-inpainting          # LaMa inpainting
# torch is pulled by the above as needed
```

Copy `.env.example` → `.env` and set translation API if using OpenAI-compatible backend:

```
TRANSLATION_BASE_URL=https://api.openai.com/v1
TRANSLATION_API_KEY=sk-...
TRANSLATION_MODEL=gpt-4o-mini
```


## Local models (no OpenAI API)

You do **not** need `TRANSLATION_BASE_URL` or any API key for local mode.

### 1) English → Persian (light, ~300MB)

```bash
pip install transformers sentencepiece sacremoses torch
pip install easyocr   # local OCR

manga-ai process ./data/chapters/my_chapter --config configs/local_hf.yaml
```

Uses **Helsinki-NLP/opus-mt-en-fa** (MarianMT) on GPU/CPU. First run downloads the model once; after that fully offline.

### 2) Korean → Persian (pivot: ko→en→fa)

```bash
manga-ai process ./data/chapters/my_chapter --config configs/local_ko_fa.yaml
```

### 3) OCR options (all local)

| Backend       | Install                                      | Good for              |
|---------------|----------------------------------------------|-----------------------|
| `easyocr`     | `pip install easyocr`                        | EN, KO, JA, ZH        |
| `paddle`      | `pip install paddleocr`                      | KO, ZH, EN (per-crop) |
| `manga_ocr`   | `pip install manga-ocr`                      | Japanese manga        |
| `hybrid_qwen` | `pip install -e ".[gpu,hybrid-qwen]"`         | EN comics via Qwen-VL |
| `router`      | paddle + manga-ocr                           | auto by language      |

Hybrid Qwen (detector boxes + Qwen2.5-VL crop OCR):

```bash
pip install -e ".[gpu,ocr,hybrid-qwen]"
manga-ai process ./data/chapters/my_chapter --config configs/hybrid_qwen_en.yaml
```

### OpenAI API (optional only)

Only if you set `translation.backend: openai_compatible` and:

```
TRANSLATION_BASE_URL=https://api.openai.com/v1
TRANSLATION_API_KEY=sk-...
TRANSLATION_MODEL=gpt-4o-mini
```

For local HF, leave those empty / unset.


## Quick start (mock – no GPU / no models)

```bash
mkdir -p data/chapters/my_chapter/original
# put page images in original/

manga-ai process ./data/chapters/my_chapter --backend mock
```

## Production (6 GB GPU)

```bash
manga-ai process ./data/chapters/my_chapter --config configs/gpu_6gb.yaml
# or full real backends after installing deps:
manga-ai process ./data/chapters/my_chapter --config configs/production.yaml
```

## CLI

```bash
manga-ai process   ./chapter_001
manga-ai process   ./chapter_001 --backend mock
manga-ai process   ./chapter_001 --config configs/gpu_6gb.yaml --force

manga-ai detect    ./chapter_001
manga-ai ocr       ./chapter_001
manga-ai translate ./chapter_001
manga-ai mask      ./chapter_001
manga-ai clean     ./chapter_001

manga-ai benchmark detection  ./chapter_001 --backend mock
manga-ai benchmark full       ./chapter_001 --backend mock
```

## Output layout

```
chapter_001/
├── original/              # immutable sources
├── detection/             # per-page detection JSON
├── ocr/                   # per-page OCR JSON
├── translation/
│   └── chapter.json       # chapter-level translations
├── masks/                 # binary masks
├── cleaned/               # text-removed images
├── debug/                 # mask previews
└── manifest.json          # stages, timings, VRAM, cache keys
```

## Backends (all swappable via config)

| Stage        | Options                                                        |
|--------------|----------------------------------------------------------------|
| Detection    | `mock`, `paddle`, `easyocr`, `yolo_comic`                      |
| OCR          | `mock`, `paddle`, `easyocr`, `manga_ocr`, `hybrid_qwen`, `router` |
| Translation  | `mock`, `openai_compatible`, `huggingface`, `llama_cpp`        |
| Inpainting   | `mock`, `opencv`, `lama`                                       |

`router` OCR: Japanese → Manga-OCR, Korean/Chinese/English → PaddleOCR.  
`hybrid_qwen` OCR: any detector’s boxes → Qwen2.5-VL recognition on each crop.

## 6 GB VRAM policy

- `batch_size = 1`
- Prefer FP16
- **Unload each model after its stage**
- Tiled inpainting (`tile_size: 768`, `overlap: 64` default on gpu_6gb)
- OOM → clear CUDA cache → OpenCV Telea fallback
- Peak VRAM logged in manifest

## Caching

Deterministic cache key per stage:

```
hash(input_hash, model_hash, config_hash, pipeline_version)
```

Re-run only changed stages (use `--force` to ignore).

## Evaluation

```bash
python scripts/evaluate_chapter.py ./data/chapters/chapter_001
```

## Architecture

Each ML stage has an abstract interface + factory (`create_detector`, `create_ocr`, …).  
Swap models without rewriting the pipeline.

## Phase status

| Phase | Description                         | Status |
|-------|-------------------------------------|--------|
| 0     | Bootstrap + mock pipeline           | Done   |
| 1     | Image pipeline, hashing, cache      | Done   |
| 2     | Text detection (Paddle + mock)      | Done   |
| 3     | OCR (Paddle, Manga-OCR, router)     | Done   |
| 4     | OpenAI-compatible translation       | Done   |
| 5     | Mask generation                     | Done   |
| 6     | LaMa + OpenCV inpainting + tiles    | Done   |
| 7     | Full V1 integration                 | Done   |
| 8     | 6 GB VRAM optimization              | Done   |
| 9     | Evaluation helpers                  | Done   |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## V2 roadmap (not in this release)

Persian RTL typesetting, bubble-aware layout, font fitting, web UI.

## License

MIT
