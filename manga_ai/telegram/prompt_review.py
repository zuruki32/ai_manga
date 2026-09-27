"""Lightweight AI review for user-supplied translation style prompts."""

from __future__ import annotations

import re
from typing import Any, Dict


_BAD = [
    r"ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"system\s*prompt",
    r"jailbreak",
    r"<script",
    r"https?://",
    r"api[_-]?key",
    r"password",
]


def review_translation_prompt(text: str) -> Dict[str, Any]:
    """Rule-based review (safe default). Returns approved=False if risky."""
    t = (text or "").strip()
    reasons = []
    if len(t) < 8:
        reasons.append("too_short")
    if len(t) > 1200:
        reasons.append("too_long")
    low = t.lower()
    for pat in _BAD:
        if re.search(pat, low):
            reasons.append(f"blocked_pattern:{pat}")
    # Few-shot EN/FA pairs poison Gemma (it copies them into every bubble)
    if re.search(r"(?im)^\s*EN\s*:", t) or re.search(r"(?im)^\s*FA\s*:", t):
        reasons.append("few_shot_examples_forbidden")
    if re.search(r"(?i)\bEN\s*:.*\bFA\s*:", t):
        reasons.append("few_shot_examples_forbidden")
    leaked = (
        "شیطونای کوچولو",
        "چطوری می‌تونستم بخوابم",
        "جوول",
        "دردسرساز",
        "اشکات رو بریزم",
    )
    if any(x in t for x in leaked):
        reasons.append("stock_example_phrase")
    # Must look like translation/style guidance
    style_hints = (
        "ترجم", "لحن", "محاوره", "اسم", "نام", "dialogue", "tone", "style",
        "colloquial", "persian", "فارسی", "مانها", "manhwa", "keep", "name",
    )
    if not any(h in low for h in style_hints) and not any("\u0600" <= c <= "\u06FF" for c in t):
        reasons.append("not_about_translation_style")

    ok = not reasons
    return {
        "approved": ok,
        "reasons": reasons,
        "summary": (
            "OK — style instruction looks safe to add after your confirm."
            if ok
            else "Blocked/needs edit: " + ", ".join(reasons)
        ),
        "text": t,
    }
