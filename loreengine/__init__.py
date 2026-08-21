"""
loreengine — a lightweight "world wiki" for LLM-driven games.

Zero-dependency alternative to heavy RAG stacks: feeds your game world to
the LLM in doses (index + entity dosing + targeted search + wiki-lookup),
not dumps. Pure Python stdlib.

Quick start::

    from loreengine import LoreEngine, UK

    engine = LoreEngine("knowledge_base", lang=UK)
    system_prompt = WORLD_INDEX + engine.index()          # cached prefix
    pages = engine.pages_for_entities(["Dark Knight"])    # current scene
    extra = engine.relevant(player_action, exclude=[d["key"] for d in pages])
"""

from .engine import LoreEngine, LoreDocument, normalize, tokenize, words_match
from .lang import LanguagePack, UK, EN, PL, PACKS
from .lint import lint_kb, Report

__version__ = "0.1.1"

__all__ = [
    "LoreEngine", "LoreDocument", "normalize", "tokenize", "words_match",
    "LanguagePack", "UK", "EN", "PL", "PACKS",
    "lint_kb", "Report",
    "__version__",
]
