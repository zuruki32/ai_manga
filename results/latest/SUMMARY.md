# Run summary — full-pipeline quality review

- **chapter**: `Ch.133 - Suddenly Became A Princess One Day`
- **config**: `hybrid_qwen_nllb`
- **pages**: 10  |  **regions (text)**: 12
- **translated lines (FA / EN)**: 11 / 11

## Stage status (watch the `backend` column — `mock` = model missing/failed)
| stage | backend | status | time(ms) | warn | err |
|---|---|---|---|---|---|
| detection | yolo_comic | success | 4204 | 4 | 0 |
| ocr | hybrid_qwen | success | 34804 | 0 | 0 |
| translation | huggingface | success | 6639 | 0 | 0 |
| masking | polygon_mask | success | 676 | 0 | 0 |
| inpainting | anime-lama | success | 5462 | 0 | 0 |
- ⚠️ `detection`: page=006: No text regions detected
- ⚠️ `detection`: page=007: No text regions detected
- ⚠️ `detection`: page=008: No text regions detected
- ⚠️ `detection`: page=009: No text regions detected

## Sample pages (6) — `samples/*.jpg` = ORIGINAL | DETECT | MASK | CLEANED
- `samples/001_panels.jpg`
- `samples/004_panels.jpg`
- `samples/002_panels.jpg`
- `samples/003_panels.jpg`
- `samples/005_panels.jpg`
- `samples/010_panels.jpg`

## OCR verification (10 crops) — see `ocr_crops/index.md`

## Cleaning review — `clean_crops/` (before|after, full-res), `cleaned_full/` (10 full pages)

## Translation preview (chapter_scanlation_en_fa.txt)
```
#ترجمه Ch.133 - Suddenly Became A Princess One Day

#صفحه 001
@GENTLE
@آروم
@I'M PERFECTLY FINE
@من کاملا خوبم
@IT DOES EAT HAM THOUGH
@با اين حال اون رو ميخوره
@It hit my hand!
@دستم رو خورد!

#صفحه 002
@WHERE ARE WE?
@ما کجا هستيم؟

#صفحه 003
@JENNETTE... MADE THIS?
@جِنِت... اينکارو کردي؟

#صفحه 004
@THAT THE LADIES OF SOCIETY WOULD BE VERY INTERESTED IF THEY SAW THIS?
@که خانم هاي جامعه خيلي علاقه مند ميشن اگه اينو ببينن؟
@I'M THINKING OF TAKING IT TO THE SUMMER PARTY, SURELY IT'LL BECOME A HOT TOPIC
@من دارم به فکر بردنش به مهموني تابستون هستم، مطمئناً موضوع داغي خواهد شد
@Wait, what workshop made this?
@صبر کن، کدوم کارگاه اينو درست کرده؟

#صفحه 005
@THEN, I'LL JUST TAKE OUT THE FILLING
@بعدش، من فقط پرش رو ميگيرم بيرون

#صفحه 010
@YOU'RE HOLDING IT
@تو داری نگهش داری
```
