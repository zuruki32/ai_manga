# Run summary — full-pipeline quality review

- **chapter**: `Ch.133 - Suddenly Became A Princess One Day`
- **config**: `hybrid_qwen_gemma_it`
- **pages**: 10  |  **regions (text)**: 12
- **translated lines (FA / EN)**: 11 / 11

## Stage status (watch the `backend` column — `mock` = model missing/failed)
| stage | backend | status | time(ms) | warn | err |
|---|---|---|---|---|---|
| inpainting | anime-lama | success | 5365 | 0 | 0 |

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
@آرام.
@I'M PERFECTLY FINE
@خیالم ندارم.
@IT DOES EAT HAM THOUGH
@چیپس هم می‌خوره ولی [این]
@It hit my hand!
@[دستش] به دستم خورد!

#صفحه 002
@WHERE ARE WE?
@کجا هستیم؟

#صفحه 003
@JENNETTE... MADE THIS?
@تو این کار رو کردی؟

#صفحه 004
@THAT THE LADIES OF SOCIETY WOULD BE VERY INTERESTED IF THEY SAW THIS?
@اگه بانوان جامعه اينو مي‌ديدن يه ذره بي‌حوصله‌شون نيابه؟
@I'M THINKING OF TAKING IT TO THE SUMMER PARTY, SURELY IT'LL BECOME A HOT TOPIC
@به این فکر می‌کنم برم مهمونی تابستونی، حتماً یه موضوع داغ میشه.
@Wait, what workshop made this?
@کدوم کارگاه این رو درست کرده؟

#صفحه 005
@THEN, I'LL JUST TAKE OUT THE FILLING
@Okay, I understand. Please provide the English line.

#صفحه 010
@YOU'RE HOLDING IT
@داری نگهداشتی؟
```
