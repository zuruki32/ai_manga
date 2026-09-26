"""Per-project store for Telegram bot (folders, config, glossary, tone)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from manga_ai.logging import get_logger

logger = get_logger("manga_ai.telegram.projects")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "configs" / "hybrid_qwen_en.yaml"
PROJECTS_DIR = Path(os.environ.get("MANGA_AI_PROJECTS", str(ROOT / "data" / "projects")))


def slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^\w\s\-]+", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s[:80] or "project"


class ProjectStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or PROJECTS_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> List[Dict[str, Any]]:
        out = []
        for p in sorted(self.root.iterdir()):
            meta = p / "project.json"
            if p.is_dir() and meta.exists():
                try:
                    out.append(json.loads(meta.read_text(encoding="utf-8")))
                except Exception:
                    out.append({"slug": p.name, "title": p.name})
        return out

    def get(self, slug: str) -> Optional[Dict[str, Any]]:
        meta = self.path(slug) / "project.json"
        if not meta.exists():
            return None
        return json.loads(meta.read_text(encoding="utf-8"))

    def path(self, slug: str) -> Path:
        return self.root / slug

    def config_path(self, slug: str) -> Path:
        return self.path(slug) / "config.yaml"

    def glossary_path(self, slug: str) -> Path:
        return self.path(slug) / "glossary.json"

    def chapters_dir(self, slug: str) -> Path:
        d = self.path(slug) / "chapters"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create(self, title: str, template: Optional[Path] = None) -> Dict[str, Any]:
        slug = slugify(title)
        base = self.path(slug)
        if base.exists():
            i = 2
            while (self.root / f"{slug}-{i}").exists():
                i += 1
            slug = f"{slug}-{i}"
            base = self.path(slug)
        base.mkdir(parents=True, exist_ok=False)
        (base / "chapters").mkdir()
        (base / "uploads").mkdir()

        tpl = Path(template or DEFAULT_TEMPLATE)
        if not tpl.exists():
            alt = ROOT / "manga_ai" / "configs" / "hybrid_qwen_en.yaml"
            tpl = alt if alt.exists() else tpl
        cfg_text = tpl.read_text(encoding="utf-8") if tpl.exists() else "pipeline: {}\n"
        cfg = yaml.safe_load(cfg_text) or {}
        cfg.setdefault("translation", {})
        cfg["translation"]["glossary"] = {}
        cfg["translation"]["style"] = "colloquial_fa"
        cfg["translation"]["unit"] = "region"
        cfg["translation"]["output_format"] = "scanlation"
        cfg.setdefault("masking", {})
        cfg["masking"]["dilation_px"] = 18
        cfg["masking"]["bubble_inflate"] = True
        cfg["masking"]["residual_expand"] = True
        cfg.setdefault("inpainting", {})
        cfg["inpainting"]["backend"] = "lama"
        cfg["inpainting"]["passes"] = 2
        cfg["inpainting"]["bg_fill"] = True
        cfg["inpainting"]["soft_edge"] = 3
        cfg["inpainting"]["radius"] = 7

        self.config_path(slug).write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.glossary_path(slug).write_text("{}", encoding="utf-8")
        meta = {
            "slug": slug,
            "title": title.strip(),
            "created": True,
            "pending_prompts": [],
            "approved_prompts": [],
        }
        (base / "project.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Created project {slug} at {base}")
        return meta

    def load_glossary(self, slug: str) -> Dict[str, str]:
        path = self.glossary_path(slug)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in (data or {}).items() if k and v}

    def save_glossary(self, slug: str, glossary: Dict[str, str]) -> None:
        self.glossary_path(slug).write_text(
            json.dumps(glossary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        cfg_path = self.config_path(slug)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        cfg.setdefault("translation", {})
        expanded = dict(glossary)
        for en, fa in list(glossary.items()):
            if en:
                expanded[en.upper()] = fa
                expanded[en.lower()] = fa
        cfg["translation"]["glossary"] = expanded
        cfg_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def add_name(self, slug: str, english: str, persian: str) -> Dict[str, str]:
        en = (english or "").strip()
        fa = (persian or "").strip()
        if not en or not fa:
            raise ValueError("Both English and Persian names are required")
        gloss = self.load_glossary(slug)
        gloss[en] = fa
        gloss[en.upper()] = fa
        gloss[en.lower()] = fa
        self.save_glossary(slug, gloss)
        return gloss

    def load_config(self, slug: str) -> Dict[str, Any]:
        return yaml.safe_load(self.config_path(slug).read_text(encoding="utf-8")) or {}

    def save_meta(self, slug: str, meta: Dict[str, Any]) -> None:
        (self.path(slug) / "project.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add_pending_prompt(self, slug: str, prompt: str, review: Dict[str, Any]) -> None:
        meta = self.get(slug) or {"slug": slug}
        pending = list(meta.get("pending_prompts") or [])
        pending.append({"text": prompt.strip(), "review": review})
        meta["pending_prompts"] = pending
        self.save_meta(slug, meta)

    def approve_pending_prompt(self, slug: str, index: int = -1) -> str:
        meta = self.get(slug) or {}
        pending = list(meta.get("pending_prompts") or [])
        if not pending:
            raise ValueError("No pending prompts")
        item = pending.pop(index)
        approved = list(meta.get("approved_prompts") or [])
        approved.append(item["text"])
        meta["pending_prompts"] = pending
        meta["approved_prompts"] = approved
        self.save_meta(slug, meta)

        cfg = self.load_config(slug)
        cfg.setdefault("translation", {})
        existing = cfg["translation"].get("extra_instructions") or ""
        joined = (existing + "\n" + item["text"]).strip() if existing else item["text"]
        cfg["translation"]["extra_instructions"] = joined
        self.config_path(slug).write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return item["text"]


_CAP_NAME = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_STOP = {
    "The", "A", "An", "I", "You", "He", "She", "It", "We", "They", "My", "Your",
    "His", "Her", "Our", "Their", "This", "That", "These", "Those", "What", "Why",
    "How", "When", "Where", "Who", "Whom", "Which", "And", "But", "Or", "If",
    "Then", "So", "Not", "No", "Yes", "Ok", "Oh", "Ah", "Wow", "Hey", "Hi",
    "Page", "Chapter", "Vol", "Princess", "Prince", "King", "Queen", "Lady",
    "Lord", "Sir", "Miss", "Mr", "Mrs", "Dad", "Mom", "Father", "Mother",
}


def find_missing_names(english_texts: List[str], glossary: Dict[str, str]) -> List[str]:
    known = set()
    for k in glossary.keys():
        known.add(k.lower())
        known.update(k.lower().split())
    found = []
    seen = set()
    for text in english_texts:
        for m in _CAP_NAME.finditer(text or ""):
            name = m.group(1).strip()
            if name in _STOP or len(name) < 2:
                continue
            key = name.lower()
            if key in seen:
                continue
            if key in known or any(p.lower() in known for p in name.split()):
                continue
            seen.add(key)
            found.append(name)
    return found
