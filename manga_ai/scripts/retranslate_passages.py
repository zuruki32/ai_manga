#!/usr/bin/env python3
"""Re-merge + retranslate from existing chapter.json / ocr (fast, no detect/inpaint)."""
from __future__ import annotations
import json, re, argparse
from collections import defaultdict
from pathlib import Path

JUNK = (
    r"luacomic", r"lua\s*scans?", r"luascans", r"discord", r"dsc\.?\s*gg",
    r"http", r"www\.", r"\.net", r"\.com", r"isbn", r"carrotoon", r"kwbooks",
    r"twitter", r"fastest\s+release", r"official\s+domain", r"scam",
)

def is_junk(t: str) -> bool:
    t = (t or "").strip().lower()
    if not t or len(t) < 2:
        return True
    for p in JUNK:
        if re.search(p, t, re.I):
            return True
    letters = sum(c.isalpha() for c in t)
    return letters < max(2, len(t) * 0.35)

def clean(t: str) -> str:
    t = (t or "").replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    for b in ("LUACOMIC", "LuaComic", "LUA SCANS"):
        t = t.replace(b, "")
    return re.sub(r"\s+", " ", t).strip(" -;,.|_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chapter_dir")
    ap.add_argument("--model", default="models/m2m100-en2fa")
    args = ap.parse_args()
    root = Path(args.chapter_dir)
    data = json.loads((root / "translation" / "chapter.json").read_text(encoding="utf-8"))
    regions = data.get("regions") or []
    by_page = defaultdict(list)
    for r in regions:
        by_page[str(r.get("page"))].append(r)

    from manga_ai.translation import create_translator
    tr = create_translator("huggingface", model=args.model, batch_size=4, max_length=256)
    pages = sorted(by_page, key=lambda p: (len(p), p))
    both = []
    passages = []
    for page in pages:
        items = sorted(by_page[page], key=lambda r: (float((r.get("bbox") or [0,0])[1]), float((r.get("bbox") or [0,0])[0])))
        parts = []
        for r in items:
            t = clean(r.get("source_text") or "")
            if t and not is_junk(t):
                parts.append(t)
        en = clean(" ".join(parts))
        if not en:
            continue
        res = tr.translate_chapter([{"region_id": page, "source_text": en}], {"source_language": "en", "target_language": "fa", "glossary": {}})
        fa = (res[0]["text"] if res else "").strip()
        passages.append({"page": page, "en": en, "fa": fa})
        both.append(f"=== Page {page} ===\nEN: {en}\nFA: {fa}\n")
        print(f"Page {page}: {en[:60]}...")
    tr.unload()
    out = root / "translation"
    (out / "chapter_en_fa.txt").write_text("\n".join(both), encoding="utf-8")
    (out / "chapter_passages.json").write_text(json.dumps({"passages": passages}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'chapter_en_fa.txt'} ({len(passages)} pages)")

if __name__ == "__main__":
    main()
