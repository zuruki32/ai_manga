# Run summary — full-pipeline quality review

- **chapter**: `_qa_sample`
- **config**: `hybrid_qwen_gemma_it`
- **pages**: 10  |  **regions (text)**: 0
- **translated lines (FA / EN)**: 0 / 0

## Stage status (watch the `backend` column — `mock` = model missing/failed)
| stage | backend | status | time(ms) | warn | err |
|---|---|---|---|---|---|
| detection | yolo_comic | failed | 57 | 0 | 10 |
| ocr | hybrid_qwen | success | 52 | 0 | 0 |
| translation | huggingface | success | 19 | 0 | 0 |
| masking | polygon_mask | success | 318 | 0 | 0 |
| inpainting | iopaint | success | 131 | 0 | 0 |
- ❌ `detection`: page=001: ultralytics required for yolo_comic detector.
  pip install ultralytics
- ❌ `detection`: page=010: ultralytics required for yolo_comic detector.
  pip install ultralytics
- ❌ `detection`: page=100: ultralytics required for yolo_comic detector.
  pip install ultralytics
- ❌ `detection`: page=101: ultralytics required for yolo_comic detector.
  pip install ultralytics
- ❌ `detection`: page=102: ultralytics required for yolo_comic detector.
  pip install ultralytics
- ❌ `detection`: page=103: ultralytics required for yolo_comic detector.
  pip install ultralytics
- ❌ `detection`: page=104: ultralytics required for yolo_comic detector.
  pip install ultralytics
- ❌ `detection`: page=105: ultralytics required for yolo_comic detector.
  pip install ultralytics
- ❌ `detection`: page=106: ultralytics required for yolo_comic detector.
  pip install ultralytics
- ❌ `detection`: page=107: ultralytics required for yolo_comic detector.
  pip install ultralytics

## Sample pages (6) — `samples/*.jpg` = ORIGINAL | DETECT | MASK | CLEANED
- `samples/001_panels.jpg`
- `samples/010_panels.jpg`
- `samples/100_panels.jpg`
- `samples/101_panels.jpg`
- `samples/102_panels.jpg`
- `samples/103_panels.jpg`

## OCR verification (0 crops) — see `ocr_crops/index.md`

## Translation preview (chapter_scanlation_en_fa.txt)
```
#ترجمه _qa_sample
```
