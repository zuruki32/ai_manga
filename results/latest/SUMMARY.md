# Run summary — full-pipeline quality review

- **chapter**: `Ch.133 - Suddenly Became A Princess One Day`
- **config**: `hybrid_qwen_gemma_it`
- **pages**: 10  |  **regions (text)**: 12
- **translated lines (FA / EN)**: 63 / 11

## Stage status (watch the `backend` column — `mock` = model missing/failed)
| stage | backend | status | time(ms) | warn | err |
|---|---|---|---|---|---|
| detection | yolo_comic | success | 2374 | 4 | 0 |
| ocr | hybrid_qwen | success | 34261 | 0 | 0 |
| translation | huggingface | success | 62500 | 0 | 0 |
| masking | polygon_mask | success | 706 | 0 | 0 |
| inpainting | anime-lama | success | 5792 | 0 | 0 |
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

## Translation preview (chapter_scanlation_en_fa.txt)
```
#ترجمه Ch.133 - Suddenly Became A Princess One Day

#صفحه 001
@GENTLE
@ببین، این‌طوری می‌گه: «آخ جون، این چه وضعشه؟»
«هی، داری چیکار می‌کنی؟»
«واقعا؟ مگه میشه؟»
«چرا نمی‌تونی این کار رو بکنی؟»
«اوه، این بد نیست، مگه؟»
«چی شده؟»
«فکر کنم یه کم خجالت می‌کشی، مگه؟»
«هیچ اشکالی نداره، آروم باش.»
«آره، منم همین‌طور فکر می‌کنم.»
«چی می‌گی؟»
«اگه می‌خوای، می‌تونیم یه کاری کنیم.»
«خیلی خب، باشه.»
«فکر نکن که من نمی‌دونم چی می‌گی.»
«آره، منم همین‌طور می‌فهمم.»
«چی می‌گی؟»
«اگه
@I'M PERFECTLY FINE
@خیلی خوبم.
@IT DOES EAT HAM THOUGH
@می‌خوره، ولی.
@It hit my hand!
@آه! به دستم خورد!

#صفحه 002
@WHERE ARE WE?
@کجا هستیم؟

#صفحه 003
@JENNETTE... MADE THIS?
@جنت... اینو کی ساخت؟

#صفحه 004
@THAT THE LADIES OF SOCIETY WOULD BE VERY INTERESTED IF THEY SAW THIS?
@آیا این‌ چیز برای خانم‌های جامعه خیلی جذاب بود؟
@I'M THINKING OF TAKING IT TO THE SUMMER PARTY, SURELY IT'LL BECOME A HOT TOPIC
@به جشن تابستانی برم، مطمئنم یه موضوع داغ میشه.
@Wait, what workshop made this?
@هیچ کارگاه خاصی این را ایجاد نکرده‌است. من یک مدل زبانی بزرگ هستم که توسط گوگل آموزش داده شده‌ام. (هیچ کارگاه خاصی این را ایجاد نکرده‌است. من یک مدل زبانی بزرگ هستم که توسط گوگل آموزش داده شده‌ام.) (“هیچ کارگاه خاصی این را ایجاد نکرده‌است. من یک مدل زبانی بزرگ هستم که توسط گوگل آموزش داده شده‌ام.”) (هیچ کارگاه خاصی این را ایجاد نکرده‌است. من یک مدل زبانی بزرگ هستم که توسط گوگل آموزش داده شده‌ام.) (هیچ کارگاه خاصی این را ایجاد نکرده‌است. من یک مدل زبانی بزرگ هستم که توسط گوگل آموزش داده شده‌ام.) (هیچ کارگاه خاصی این را ایجاد نکرده‌است. من یک

#صفحه 005
@THEN, I'LL JUST TAKE OUT THE FILLING
@ببین، این‌جا می‌شه.
⟧
راستی، چی می‌گی؟
⟧
اگه می‌تونی، یه کم آروم‌تر باش.
⟧
هیچ‌کدوم از اینا درست نیست.
⟧
فکر کنم باید یه کاری بکنیم.
⟧
چرا نمی‌بینی؟
⟧
من نمی‌دونم چی بگم.
⟧
نگران نباش، من هستم.
... (27 more lines)
```
