"""Mock translator (Korean -> Persian fake mappings)."""

from __future__ import annotations

from typing import Any, Dict, List

from manga_ai.translation.base import Translator

_KO_TO_FA = {
    "안녕하세요": "سلام",
    "무슨 일이야?": "چه خبره؟",
    "빨리 와!": "زود بیا!",
    "정말?": "واقعاً؟",
    "알겠어.": "باشه.",
    "고마워.": "مرسی.",
    "왜 그래?": "چرا اینطوری؟",
    "괜찮아.": "اشکالی نداره.",
    "다음에 봐.": "بعداً می‌بینمت.",
    "이거 뭐야?": "این چیه؟",
}


class MockTranslator(Translator):
    name = "mock"

    def translate_chapter(
        self,
        regions: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        results = []
        for r in regions:
            src = r.get("source_text", "")
            translated = _KO_TO_FA.get(src, f"[ترجمة] {src}")
            # Apply glossary if present
            glossary = context.get("glossary") or {}
            for en, fa in glossary.items():
                if en in translated:
                    translated = translated.replace(en, fa)
            results.append({"region_id": r["region_id"], "text": translated})
        return results
