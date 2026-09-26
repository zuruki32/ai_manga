"""Abstract Translator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Translator(ABC):
    """Abstract chapter-level translator."""

    name: str = "base"

    @abstractmethod
    def translate_chapter(
        self,
        regions: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Translate all regions of a chapter.

        Args:
            regions: list of region dicts with region_id, source_text, etc.
            context: {
                "source_language": str,
                "target_language": str,
                "glossary": dict,
                "chapter_context": list (previous dialogue)
            }

        Returns:
            list of {"region_id": str, "text": str}
        """
        ...

    def unload(self) -> None:
        pass
