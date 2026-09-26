"""OpenAI-compatible HTTP translation backend."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx

from manga_ai.translation.base import Translator
from manga_ai.logging import get_logger

logger = get_logger("manga_ai.translation.openai")

SYSTEM_PROMPT = """You are a professional manga/manhwa translator.
Translate the given text regions from {source_lang} to {target_lang}.
Rules:
- Return ONLY valid JSON matching the schema.
- Preserve character names and terminology consistently.
- Use glossary when provided.
- Do not add explanations, notes, or extra fields.
- Never drop a region_id.
- Keep the same number of translations as input regions.
Schema:
{{"translations": [{{"region_id": "...", "text": "..."}}]}}
"""


class OpenAICompatibleTranslator(Translator):
    name = "openai_compatible"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or "gpt-4o-mini"
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def translate_chapter(
        self,
        regions: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError(
                "TRANSLATION_API_KEY not set. Configure .env or translation.api_key"
            )
        source_lang = context.get("source_language", "ko")
        target_lang = context.get("target_language", "fa")
        glossary = context.get("glossary") or {}
        chapter_context = context.get("chapter_context") or []

        payload_regions = [
            {
                "region_id": r["region_id"],
                "text": r.get("source_text", ""),
                "page": r.get("page"),
            }
            for r in regions
        ]

        user_content = {
            "source_language": source_lang,
            "target_language": target_lang,
            "glossary": glossary,
            "previous_dialogue": chapter_context[-20:],  # last 20 for context
            "regions": payload_regions,
        }

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    source_lang=source_lang, target_lang=target_lang
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_content, ensure_ascii=False),
            },
        ]

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                data = self._call_api(messages)
                translations = self._parse_response(data, regions)
                return translations
            except Exception as e:
                last_err = e
                logger.warning(f"Translation attempt {attempt}/{self.max_retries} failed: {e}")
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Translation failed after {self.max_retries} retries: {last_err}")

    def _call_api(self, messages: List[Dict]) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            return resp.json()

    def _parse_response(
        self, data: Dict, original_regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        items = parsed.get("translations") or parsed.get("results") or []
        if not isinstance(items, list):
            raise ValueError("Response missing translations list")

        result = []
        seen = set()
        for item in items:
            rid = item.get("region_id")
            text = item.get("text", "")
            if not rid:
                continue
            result.append({"region_id": rid, "text": text})
            seen.add(rid)

        # Ensure no region is lost
        for r in original_regions:
            rid = r["region_id"]
            if rid not in seen:
                result.append({"region_id": rid, "text": r.get("source_text", "")})
                logger.warning(f"Missing translation for {rid}, kept source text")
        return result

    def unload(self) -> None:
        pass
