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

from .engine import LoreDocument, LoreEngine, PathLikeStr, normalize
from .lang import LanguagePack, UK

# Index size guard (characters). ~6k chars ≈ 2k tokens — still cheap to
# cache. Above that, split the index into per-category indexes.
INDEX_WARN_CHARS = 6000
INDEX_FAIL_CHARS = 9000

# One dosed page costs ~1-1.5k tokens; oversized pages bloat the prompt.
DOC_WARN_CHARS = 3500

# Latin snake_case key for knowledge base files
_KEY_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
# Canonical title/header separator on a document's first line.
_TITLE_SPLIT_RE = re.compile(r"\s+[-—–]\s+|,")


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


def _words_hit_target(cw: List[str], kw: Iterable[str]) -> bool:
    """True when candidate words sufficiently overlap target words."""
    kw_list = list(kw)
    if not cw or not kw_list:
        return False
    if len(cw) == 1:
        return any(loose_words_match(cw[0], k) for k in kw_list)
    matched = sum(
        1 for c in set(cw) if any(loose_words_match(c, k) for k in kw_list)
    )
    return matched == len(cw) or (matched >= 2 and matched * 2 >= len(cw))


def mention_hits_entity(candidate: str, known: str) -> bool:
    """Does the candidate mention the known entity: single-word mentions match
    any word of the known name; multi-word mentions require >= 2 and at least
    half of the candidate's words (or all of them) to match."""
    return _words_hit_target(name_words(candidate), name_words(known))


def _same_alias_form(a: str, b: str) -> bool:
    """Alias case-form duplicate: equal normalized forms, or the same word
    count with every word pair a fuzzy case-form («Таверна» ~ «Таверни»).
    Distinct multi-word names («Таверна на Сході» vs «Таверна») differ in
    word count and are NOT duplicates."""
    wa, wb = normalize(a).split(), normalize(b).split()
    if wa == wb:
        return True
    return (len(wa) == len(wb)
            and all(loose_words_match(x, y) for x, y in zip(wa, wb)))


def _extract_quoted_text(match: re.Match) -> str:
    """Return the first non-empty capture group from a quoted-name match."""
    return next((g for g in match.groups() if g is not None), match.group(0)).strip()


# ------------------------------------------------------------------ registry
def build_kb_registry(engine: LoreEngine) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Registry of known names from the KB itself:

    - kb_names: canonical name -> document key (from any page's first line
      before the first dash/comma when short, plus the document's ``name``);
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
        parts = _TITLE_SPLIT_RE.split(first_line, maxsplit=1)
        has_sep = len(parts) > 1
        head = parts[0].strip()
        if intro_re:
            head = intro_re.split(head)[0].strip(" ,")
        if (
            head
            and len(head.split()) <= 6
            and (has_sep or not head.endswith((".", "!", "?", ";", ":")))
        ):
            kb_names.setdefault(head, doc["key"])
        if doc.get("name"):
            kb_names.setdefault(doc["name"], doc["key"])
        for m in re.finditer(engine.lang.quoted_regex, first_line):
            alias = _extract_quoted_text(m)
            if not alias or not alias[0].isupper():
                continue
            # Skip case-forms of already registered aliases (see
            # _same_alias_form), but keep distinct multi-word aliases:
            # «Таверна на Сході» survives next to a registered «Таверна»,
            # while «Таверни» (a case-form of «Таверна») is still deduped.
            if any(_same_alias_form(alias, existing) for existing in kb_aliases):
                continue
            kb_aliases[alias] = doc["key"]
    return kb_names, kb_aliases


def load_allowlist(path: Optional[PathLikeStr]) -> Set[str]:
    """Names the project explicitly allows to have no page (# comments ok)."""
    if not path:
        return set()
    path_str = os.fspath(path)
    if not os.path.isfile(path_str):
        return set()
    names = set()
    with open(path_str, "r", encoding="utf-8") as f:
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
        name = _extract_quoted_text(m)
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
    """Words of the document title (first line before delimiter) — for self-refs."""
    first_line = doc["content"].split("\n", 1)[0]
    for ch in ("*", '"', "“", "”", "«", "»", "„"):
        first_line = first_line.replace(ch, " ")
    title = _TITLE_SPLIT_RE.split(first_line, maxsplit=1)[0][:80]
    return set(normalize(title).split())


def is_self_reference(doc: LoreDocument, name: str) -> bool:
    """The mention refers to the document itself ("Marcus's sword" in marcus_sword)."""
    cw = name_words(name)
    if not cw:
        return True
    title_words = [w for w in doc_title_words(doc) if len(w) >= 3]
    return _words_hit_target(cw, title_words)


# --------------------------------------------------------------------- lint
def lint_kb(kb_dir: PathLikeStr, lang: LanguagePack = UK,
            extra_names: Iterable[str] = (),
            allowlist_path: Optional[PathLikeStr] = None,
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
    kb_dir_str = os.fspath(kb_dir)
    rep = Report()
    engine = LoreEngine(kb_dir_str, lang=lang, verbose=False)
    if not engine.documents:
        rep.error(f"Knowledge base '{kb_dir_str}' is empty — nothing to lint.")
        return rep

    kb_names, kb_aliases = build_kb_registry(engine)
    allowlist = load_allowlist(os.fspath(allowlist_path) if allowlist_path is not None else None)
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
    # search_one drops <3-char query words by design, so a name made only
    # of such words can never resolve — note it as info instead of failing.
    for name, key in kb_names.items():
        if not engine._query_words(name):
            rep.info(f"Search self-test skipped for «{name}» — no query "
                     f"word reaches the 3-char floor.")
            continue
        found = engine.search_one(name)
        if found is None or found["key"] != key:
            rep.error(f"Search for «{name}» returned "
                      f"{found['key'] if found else 'nothing'} instead of {key}.")
    for alias, key in kb_aliases.items():
        if not engine._query_words(alias):
            rep.info(f"Search self-test skipped for alias «{alias}» — no "
                     f"query word reaches the 3-char floor.")
            continue
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
