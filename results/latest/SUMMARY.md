# Run summary — full-pipeline quality review

- **chapter**: `Ch.133 - Suddenly Became A Princess One Day`
- **config**: `hybrid_qwen_gemma_it`
- **pages**: 10  |  **regions (text)**: 12
- **translated lines (FA / EN)**: 26 / 11

## Stage status (watch the `backend` column — `mock` = model missing/failed)
| stage | backend | status | time(ms) | warn | err |
|---|---|---|---|---|---|
| detection | yolo_comic | success | 3333 | 4 | 0 |
| ocr | hybrid_qwen | success | 29901 | 0 | 0 |
| translation | huggingface | success | 27143 | 0 | 0 |
| masking | polygon_mask | success | 693 | 0 | 0 |
| inpainting | opencv | success | 4862 | 1 | 0 |
- ⚠️ `detection`: page=006: No text regions detected
- ⚠️ `detection`: page=007: No text regions detected
- ⚠️ `detection`: page=008: No text regions detected
- ⚠️ `detection`: page=009: No text regions detected
- ⚠️ `inpainting`: LaMa unavailable (IOPaint is required for the 'iopaint' backend. Install it with:
    pip install 'manga-ai[iopaint]'
    # or: pip install iopaint
(original error: cannot import name 'cached_download' from 'huggingface_hub' (D:\projects\Lib\site-packages\huggingface_hub\__init__.py)))

## Sample pages (6) — `samples/*.jpg` = ORIGINAL | DETECT | MASK | CLEANED
- `samples/001_panels.jpg`
- `samples/004_panels.jpg`
- `samples/002_panels.jpg`
- `samples/003_panels.jpg`
- `samples/005_panels.jpg`
- `samples/010_panels.jpg`

## OCR verification (10 crops) — see `ocr_crops/index.md`

## Translation preview (chapter_scanlation_en_fa.txt)
```
#ترجمه Ch.133 - Suddenly Became A Princess One Day

#صفحه 001
@GENTLE
@راحتم ⟦N⟧
آرام
@I'M PERFECTLY FINE
@آرام باش
حالا گوشت می‌زنه ولی
من خوبم
@IT DOES EAT HAM THOUGH
@خیالم ندارم
دستم خورد!
ولی خب، گوشت می‌زنه، در کل.
@It hit my hand!
@خوردش به دستم!
کجا هستیم؟

#صفحه 002
@WHERE ARE WE?
@جنت... خودش اینو ساخت؟
کجا هستیم؟

#صفحه 003
@JENNETTE... MADE THIS?
@کجا هستیم؟
آیا این رو بانوان جامعه خیلی به چشم‌شون می‌خورد؟
... این رو کی ساخت؟

#صفحه 004
@THAT THE LADIES OF SOCIETY WOULD BE VERY INTERESTED IF THEY SAW THIS?
@آخ... این خودش دستِ جنت بوده؟
فکر می‌کنم برم تو مهمونی تابستونی، حتماً یه بحث داغه‌ست که خانم‌های جامعه اگه ببینن خیلی بهش توجه می‌کنن.
@I'M THINKING OF TAKING IT TO THE SUMMER PARTY, SURELY IT'LL BECOME A HOT TOPIC
@آیا بانوان جامعه اگه این رو می‌دیدن خیلی بهشون می‌گفت؟
کاش می‌دونیم این کارگاه‌چی این رو درست کرده؟
فکر می‌کنم برم باهاش به مهمونی تابستونی ببرمش، حتماً یه بحث داغ میشه.
@Wait, what workshop made this?
@به نظرم برم مهمونی تابستونی، حتماً یه موضوع داغ میشه.
بعد من فقط موادشو می‌گیرم بیرون.

#صفحه 005
@THEN, I'LL JUST TAKE OUT THE FILLING
@کجا این کارگاه این‌ها رو درست کرده؟
تو داری اینو نگه داری.
فقط من پرش رو بردارم.

#صفحه 010
@YOU'RE HOLDING IT
@بگیر.
```
