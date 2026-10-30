# Run summary — full-pipeline quality review

- **chapter**: `Ch.133 - Suddenly Became A Princess One Day`
- **config**: `hybrid_qwen_gemma_it`
- **pages**: 10  |  **regions (text)**: 0
- **translated lines (FA / EN)**: 0 / 0

## Stage status (watch the `backend` column — `mock` = model missing/failed)
| stage | backend | status | time(ms) | warn | err |
|---|---|---|---|---|---|
| detection | yolo_comic | success | 5112 | 4 | 0 |
| ocr | hybrid_qwen | failed | 15789 | 0 | 6 |
| translation | huggingface | success | 3 | 0 | 0 |
| masking | polygon_mask | success | 478 | 0 | 0 |
| inpainting | iopaint | success | 267 | 0 | 0 |
- ⚠️ `detection`: page=006: No text regions detected
- ⚠️ `detection`: page=007: No text regions detected
- ⚠️ `detection`: page=008: No text regions detected
- ⚠️ `detection`: page=009: No text regions detected
- ❌ `ocr`: page=001: The checkpoint you are trying to load has model type `qwen3_vl` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`
- ❌ `ocr`: page=002: The checkpoint you are trying to load has model type `qwen3_vl` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`
- ❌ `ocr`: page=003: The checkpoint you are trying to load has model type `qwen3_vl` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`
- ❌ `ocr`: page=004: The checkpoint you are trying to load has model type `qwen3_vl` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`
- ❌ `ocr`: page=005: The checkpoint you are trying to load has model type `qwen3_vl` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`
- ❌ `ocr`: page=010: The checkpoint you are trying to load has model type `qwen3_vl` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date code by installing Transformers from source with the command `pip install git+https://github.com/huggingface/transformers.git`

## Sample pages (6) — `samples/*.jpg` = ORIGINAL | DETECT | MASK | CLEANED
- `samples/001_panels.jpg`
- `samples/002_panels.jpg`
- `samples/003_panels.jpg`
- `samples/004_panels.jpg`
- `samples/005_panels.jpg`
- `samples/006_panels.jpg`

## OCR verification (0 crops) — see `ocr_crops/index.md`

## Translation preview (chapter_scanlation_en_fa.txt)
```
#ترجمه Ch.133 - Suddenly Became A Princess One Day
```
