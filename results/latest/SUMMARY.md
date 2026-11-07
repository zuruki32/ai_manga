# Run summary — full-pipeline quality review

- **chapter**: `Ch.133 - Suddenly Became A Princess One Day`
- **config**: `hybrid_qwen_nllb`
- **pages**: 136  |  **regions (text)**: 137
- **translated lines (FA / EN)**: 115 / 115

## Stage status (watch the `backend` column — `mock` = model missing/failed)
| stage | backend | status | time(ms) | warn | err |
|---|---|---|---|---|---|
| detection | yolo_comic | success | 6525 | 55 | 0 |
| ocr | hybrid_qwen | success | 129591 | 6 | 0 |
| translation | huggingface | success | 15040 | 0 | 0 |
| masking | polygon_mask | success | 4796 | 0 | 0 |
| inpainting | anime-lama | success | 15217 | 0 | 0 |
- ⚠️ `detection`: page=006: No text regions detected
- ⚠️ `detection`: page=007: No text regions detected
- ⚠️ `detection`: page=008: No text regions detected
- ⚠️ `detection`: page=009: No text regions detected
- ⚠️ `detection`: page=011: No text regions detected
- ⚠️ `detection`: page=013: No text regions detected
- ⚠️ `detection`: page=015: No text regions detected
- ⚠️ `detection`: page=019: No text regions detected
- ⚠️ `detection`: page=020: No text regions detected
- ⚠️ `detection`: page=022: No text regions detected
- ⚠️ `detection`: page=025: Low detection confidence 0.29 at index 2
- ⚠️ `detection`: page=026: No text regions detected
- ⚠️ `detection`: page=028: No text regions detected
- ⚠️ `detection`: page=036: No text regions detected
- ⚠️ `detection`: page=037: No text regions detected
- ⚠️ `ocr`: Empty OCR for region 025_0002
- ⚠️ `ocr`: Low OCR confidence 0.00 for 025_0002
- ⚠️ `ocr`: Empty OCR for region 050_0001
- ⚠️ `ocr`: Low OCR confidence 0.00 for 050_0001
- ⚠️ `ocr`: Empty OCR for region 073_0000
- ⚠️ `ocr`: Low OCR confidence 0.00 for 073_0000

## Sample pages (6) — `samples/*.jpg` = ORIGINAL | DETECT | MASK | CLEANED
- `samples/094_panels.jpg`
- `samples/001_panels.jpg`
- `samples/004_panels.jpg`
- `samples/063_panels.jpg`
- `samples/016_panels.jpg`
- `samples/025_panels.jpg`

## OCR verification (40 crops) — see `ocr_crops/index.md`

## Cleaning review — `clean_crops/` (before|after, full-res), `cleaned_full/` (136 full pages)

## Translation preview (chapter_scanlation_en_fa.txt)
```
#ترجمه Ch.133 - Suddenly Became A Princess One Day

#صفحه 000
@THOSE WHO WILL ONE DAY RULE THE WORLD WOULD BE SO JEALOUS AND PURE-HEARTED OVER A CAT
@اونايي که يه روزي بر دنيا فرمانروايي ميکنن خيلي نسبت به يه گربه حسود و خالص قلب خواهند بود
@SO INNOCENT
@خیلی بی گناه

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

#صفحه 012
@IF YOU PET IT LIKE THAT YOU'LL BLOCK ITS VIEW, SO IT'S BETTER TO PET IT FROM UNDERNEATH
@اگه اينجوري دوستش داشته باشي که جلوي چشمش رو بگيري بهتره از زيرش دوستش داشته باشي
@L-LIKE THIS?
@اینو دوست داری؟

#صفحه 014
@COST OF DISPOSING BROKEN FURNITURE AND REQUEST FOR NEW FURNITURE
@هزینه دفع مبلمان شکسته و درخواست مبلمان جدید

#صفحه 016
@FAR
@FAR
@WOW, SO CUTE!
@واو، چقدر خوشگله!
@Is it because we fed it?
@بخاطر اينه که بهش غذا داديم؟

... (325 more lines)
```
