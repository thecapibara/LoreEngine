"""
Lore linter — integrity checks for a knowledge base (dev tool).

Checks:

1. CROSS-REFERENCES: proper-name mentions in documents are checked against
   the page registry. A mention with no page (and not in the allowlist or
   ``extra_names``) is a warning.
2. ENGINE SELF-TEST: for every character page, searching by its canonical
   name and by its quoted aliases must return that very page (catches
   transliteration / fuzzy-match regressions before your game does).
3. INDEX GUARD: while ``index()`` is small it is fully cacheable; the
   linter says when it is time to split the index by category.
4. STRUCTURE: duplicate keys, non-latin keys, oversized documents (they
   bloat prompt dosing).

Game-specific name sources (e.g. a campaign's cast list) are passed via
``extra_names`` — the linter itself knows nothing about any game.

Usage::

    from loreengine.lint import lint_kb
    report = lint_kb("knowledge_base", extra_names={"Cornelius Blackwood"})
    report.print()
    exit_code = 1 if report.errors else 0
"""

import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .engine import LoreDocument, LoreEngine, normalize
from .lang import LanguagePack, UK

# Index size guard (characters). ~6k chars ≈ 2k tokens — still cheap to
# cache. Above that, split the index into per-category indexes.
INDEX_WARN_CHARS = 6000
INDEX_FAIL_CHARS = 9000

# One dosed page costs ~1-1.5k tokens; oversized pages bloat the prompt.
DOC_WARN_CHARS = 3500

# Latin snake_case key for knowledge base files
_KEY_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")


class Report:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.infos: List[str] = []

    def error(self, msg: str) -> None: self.errors.append(msg)
    def warn(self, msg: str) -> None: self.warnings.append(msg)
    def info(self, msg: str) -> None: self.infos.append(msg)

    def __repr__(self) -> str:
        return (f"Report(errors={len(self.errors)}, "
                f"warnings={len(self.warnings)}, infos={len(self.infos)})")

    def print(self) -> None:
        for m in self.infos:
            print(f"  [info] {m}")
        for m in self.warnings:
            print(f"  [WARN] {m}")
        for m in self.errors:
            print(f"  [ERROR] {m}")
        print(f"\nSummary: {len(self.errors)} errors, "
              f"{len(self.warnings)} warnings, {len(self.infos)} infos.")


# ------------------------------------------------------------------ matching
def name_words(name: str) -> List[str]:
    """Meaningful words of a name (lowercase, no punctuation)."""
    return [w for w in normalize(name).split() if len(w) >= 3]


def loose_words_match(w1: str, w2: str) -> bool:
    """Fuzzy word match: shared prefix >=4 and >= min_len-2 chars.
    Catches case endings: dragon/dragona, knight/knight's."""
    if w1 == w2:
        return True
    common = 0
    for a, b in zip(w1, w2):
        if a != b:
            break
        common += 1
    return common >= 4 and common >= min(len(w1), len(w2)) - 2


def mention_hits_entity(candidate: str, known: str) -> bool:
    """Does the candidate mention the known entity: at least one meaningful
    word of the candidate fuzzy-matches a word of the known name."""
    cw = name_words(candidate)
    kw = name_words(known)
    if not cw or not kw:
        return False
    return any(loose_words_match(c, k) for c in cw for k in kw)


# ------------------------------------------------------------------ registry
def build_kb_registry(engine: LoreEngine) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Registry of known names from the KB itself:

    - kb_names: canonical name -> document key (from the first line of
      ``characters`` pages: text before the first " - " or ",");
    - kb_aliases: explicit ``Aliases:`` header entries plus quoted
      nicknames/names from any page's first line ("Doc", «Tavern»)
      -> document key.
    """
    kb_names: Dict[str, str] = {}
    kb_aliases: Dict[str, str] = {}
    intro_re = re.compile(engine.lang.alias_intro_regex) if engine.lang.alias_intro_regex else None
    for doc in engine.documents:
        # Explicit aliases (Aliases: header) are authoritative.
        for alias in doc.get("aliases", []):
            kb_aliases.setdefault(alias, doc["key"])
        first_line = doc["content"].split("\n", 1)[0].replace("*", "")
        if doc["category"] == "characters":
            head = re.split(r"\s+-\s+|,", first_line, maxsplit=1)[0].strip()
            if intro_re:
                head = intro_re.split(head)[0].strip(" ,")
            if head:
                kb_names[head] = doc["key"]
        for m in re.finditer(r'[\"“«]([^\"”»]{3,30})[\"”»]', first_line):
            alias = m.group(1).strip()
            if not alias or not alias[0].isupper():
                continue
            # Skip case-forms of already registered aliases
            if any(loose_words_match(normalize(alias), normalize(existing))
                   for existing in kb_aliases):
                continue
            kb_aliases[alias] = doc["key"]
    return kb_names, kb_aliases


def load_allowlist(path: Optional[str]) -> Set[str]:
    """Names the project explicitly allows to have no page (# comments ok)."""
    if not path or not os.path.isfile(path):
        return set()
    names = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                names.add(line)
    return names


# ----------------------------------------------------------------- mentions
def extract_mentions(text: str, lang: LanguagePack) -> List[Tuple[str, int]]:
    """Proper-name candidates in a document text (per language pack)."""
    found = []
    for m in re.finditer(lang.name_regex, text):
        found.append((m.group(0), m.start()))
    for m in re.finditer(lang.quoted_regex, text):
        name = next((g for g in m.groups() if g), "").strip()
        if name:
            found.append((name, m.start()))
    return found


def strip_leading_stop(name: str, lang: LanguagePack) -> str:
    """Drop a leading service word ("For Marcus" -> "Marcus")."""
    words = name.split()
    while len(words) > 1 and words[0].lower() in lang.leading_stop:
        words = words[1:]
    return " ".join(words)


def is_noise(name: str, lang: LanguagePack) -> bool:
    """Filter obvious non-names (sentence starters, service constructs)."""
    words = name_words(name)
    if not words:
        return True
    if all(w in lang.sentence_starters for w in words):
        return True
    return False


def doc_title_words(doc: LoreDocument) -> Set[str]:
    """Words of the document title (first line before ' - ') — for self-refs."""
    first_line = doc["content"].split("\n", 1)[0].replace("*", "").replace('"', ' ')
    title = first_line.split(" - ")[0][:80]
    return set(normalize(title).split())


def is_self_reference(doc: LoreDocument, name: str) -> bool:
    """The mention refers to the document itself ("Marcus's sword" in marcus_sword)."""
    cw = name_words(name)
    if not cw:
        return True
    title_words = doc_title_words(doc)
    return any(loose_words_match(c, t) for c in cw for t in title_words)


# --------------------------------------------------------------------- lint
def lint_kb(kb_dir: str, lang: LanguagePack = UK,
            extra_names: Iterable[str] = (),
            allowlist_path: Optional[str] = None,
            index_warn_chars: int = INDEX_WARN_CHARS,
            index_fail_chars: int = INDEX_FAIL_CHARS,
            doc_warn_chars: int = DOC_WARN_CHARS) -> Report:
    """Run all checks over a knowledge base and return a :class:`Report`.

    :param kb_dir: knowledge base root.
    :param lang: language pack (mention regexes, noise filters).
    :param extra_names: names known from outside the KB (e.g. your game's
        story cast). Mentions hitting them are reported as INFO, not WARN.
    :param allowlist_path: optional ``.lint_allowlist.txt``.
    """
    rep = Report()
    engine = LoreEngine(kb_dir, lang=lang, verbose=False)
    if not engine.documents:
        rep.error(f"Knowledge base '{kb_dir}' is empty — nothing to lint.")
        return rep

    kb_names, kb_aliases = build_kb_registry(engine)
    allowlist = load_allowlist(allowlist_path)
    extra = {n.strip() for n in (extra_names or ()) if n and n.strip()}
    known_all = list(kb_names) + list(kb_aliases) + list(allowlist)

    # 0. Explicit alias collisions ------------------------------------------
    alias_owners: Dict[str, str] = {}
    for doc in engine.documents:
        for alias in doc.get("aliases", []):
            na = normalize(alias)
            if na in alias_owners and alias_owners[na] != doc["key"]:
                rep.error(f"Alias «{alias}» is declared by both "
                          f"{alias_owners[na]} and {doc['key']}.")
            else:
                alias_owners[na] = doc["key"]

    # 1. Cross-references ---------------------------------------------------
    for doc in engine.documents:
        for name, _pos in extract_mentions(doc["content"], lang):
            name = strip_leading_stop(name, lang)
            if is_noise(name, lang) or is_self_reference(doc, name):
                continue
            if any(mention_hits_entity(name, known) for known in known_all):
                continue
            if any(mention_hits_entity(name, known) for known in extra):
                rep.info(f"{doc['key']}: «{name}» — name from outside the KB "
                         f"(extra_names), no page needed.")
                continue
            rep.warn(f"{doc['key']}: «{name}» is mentioned but has no page — "
                     f"add a document or put the name in the allowlist.")

    # 2. Engine self-test ----------------------------------------------------
    for name, key in kb_names.items():
        found = engine.search_one(name)
        if found is None or found["key"] != key:
            rep.error(f"Search for «{name}» returned "
                      f"{found['key'] if found else 'nothing'} instead of {key}.")
    for alias, key in kb_aliases.items():
        found = engine.search_one(alias)
        if found is None or found["key"] != key:
            rep.warn(f"Search for alias «{alias}» returned "
                     f"{found['key'] if found else 'nothing'} instead of {key}.")

    # 3. Index guard ---------------------------------------------------------
    idx = engine.index()
    if len(idx) >= index_fail_chars:
        rep.error(f"Index is {len(idx)} chars — split it into per-category "
                  f"indexes: engine.index('characters'), engine.index('locations'), …")
    elif len(idx) >= index_warn_chars:
        rep.warn(f"Index is {len(idx)} chars — approaching the cacheable "
                 f"limit ({index_fail_chars}); consider engine.index(category).")
    else:
        rep.info(f"Index: {len(idx)} chars (~{len(idx) // 3} tokens), "
                 f"{len(engine.documents)} documents. Within limits.")

    # 4. Structure -----------------------------------------------------------
    seen_keys: Dict[str, str] = {}
    for doc in engine.documents:
        key = doc["key"]
        if key in seen_keys:
            rep.error(f"Duplicate key '{key}' "
                      f"({seen_keys[key]} and {doc['category']}).")
        seen_keys[key] = doc["category"]
        if not _KEY_RE.match(key):
            rep.warn(f"Key '{key}' is not latin snake_case — the "
                     f"transliteration bridge may miss it.")
        if len(doc["content"]) > doc_warn_chars:
            rep.warn(f"{key}: {len(doc['content'])} chars — oversized page "
                     f"(>{doc_warn_chars}); it bloats prompt dosing.")

    return rep
