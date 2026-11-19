"""
LoreEngine — a lightweight "world wiki" for LLM-driven games.

A zero-dependency alternative to heavy RAG stacks (langchain + vector DBs +
torch). Instead of dumping the whole lore into every prompt (~15k tokens) or
running noisy vector search, LoreEngine feeds the world to the model in
DOSES via a 4-layer hybrid:

1. INDEX (``index()``) — a compact table of contents of every page
   (~1-2k tokens). Put it in the STABLE prefix of your system prompt so the
   provider caches it and it costs almost nothing after the first turn.
   Pass ``index("characters")`` to get a single category when the KB grows
   too large for one cached prefix.

2. DOSING (``pages_for_entities()``) — full pages only for the entities of
   the CURRENT scene (fuzzy name matching). Guaranteed relevant.

3. TARGETED SEARCH (``relevant()``) — given the player's action text, pull
   1-2 pages not already dosed (like classic RAG, but deduplicated, capped,
   and deterministic).

4. WIKI-LOOKUP (``get_pages()``) — let the model itself request specific
   pages (e.g. via a ``wiki_lookup`` field in its JSON reply); feed them in
   on the next turn. ``search_one()`` exposes the single-best-page lookup.

Together this is ~3-5k lore tokens per turn instead of ~15k, and more
accurate than vector search because current-scene pages are always present.
No ML dependencies — pure Python stdlib.

Search machinery: TF-IDF weighting of rare words + a transliteration bridge
(file keys are latin, player queries may be Cyrillic — see
:mod:`loreengine.lang`) + fuzzy prefix comparison (handles case endings:
"дракон" == "дракона" == "дракону", and short-word inflections like
"ліс" ~ "лісу") + a co-occurrence semantic index (distributional semantics)
that expands ``relevant()`` queries with context-related words, reaching
pages that share context rather than keywords. Scales to large in-memory
knowledge bases: a 3-char prefix index keeps candidate scoring sub-linear,
and ``category_summary()`` gives a compact map when even per-category
indexes outgrow a cached prefix.

Knowledge base layout::

    knowledge_base/
        characters/
            dark_knight.txt
            ...
        locations/
            old_tavern.txt
        items/
        lore/

One folder per category, one ``.txt`` per page; the file name (without
extension) is the page key. The first line of a page becomes its teaser in
the index.
"""

import logging
import math
import os
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict, Union

from .lang import LanguagePack, UK

logger = logging.getLogger("loreengine")
PathLikeStr = Union[str, "os.PathLike[str]"]

# --- BM25 & exact-vs-fuzzy ranking tuning ---------------------------------
_BM25_K1 = 1.2
_BM25_B = 0.75
_EXACT_WEIGHT = 1.0
_FUZZY_WEIGHT = 0.7
_BODY_PREF_THRESHOLD = 64  # only build per-doc prefix buckets for longer docs

# --- co-occurrence (distributional semantics) tuning ----------------------
# Words that share documents are treated as semantically related. This is
# what lets relevant() reach pages sharing *context* rather than keywords.
_COOC_MAX_WORDS = 20   # most discriminative body words per doc used for pairs
_COOC_NEIGHBORS = 15   # max semantic neighbors stored per word (memory cap)
_COOC_PRUNE_KEEP = 4 * _COOC_NEIGHBORS      # running top kept at prune time
_COOC_PRUNE_TRIGGER = 8 * _COOC_NEIGHBORS  # prune a word's map past this size
_COOC_EXPAND = 6       # max expansion words added to a relevant() query
_COOC_WEIGHT = 0.5     # max score multiplier for expansion (semantic) words
_COOC_EXPAND_MIN_WEIGHT = 0.2  # min weight for normalized top expansion words
_RELEVANT_MAX_CANDIDATES = 1200  # safety cap on candidate set for relevant()
_NO_APOSTROPHE = str.maketrans("", "", "'\u2019\u02bc")


def normalize(s: str) -> str:
    """Lowercase, strip apostrophes, replace punctuation with spaces, collapse whitespace (NFC).

    Punctuation becomes a space (not nothing), so hyphenated names split
    into searchable words: ``Jean-Luc`` -> ``jean luc``, while apostrophes
    are stripped consistently with :func:`tokenize`: ``O'Brien`` -> ``obrien``.
    """
    s = unicodedata.normalize("NFC", s or "").lower().strip()
    # U+02BC MODIFIER LETTER APOSTROPHE is a word character («сімʼю» would
    # stay one token); strip all apostrophe variants consistently with tokenize().
    s = s.translate(_NO_APOSTROPHE)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def words_match(w1: str, w2: str) -> bool:
    """Case-insensitive fuzzy word comparison: exact match, a shared prefix
    (case endings), or a short-word inflection ("ліс" ~ "лісу", "cat" ~ "cats")."""
    w1 = w1.lower()
    w2 = w2.lower()
    if w1 == w2:
        return True
    # Apostrophes never survive tokenize(), so dropping them here only
    # reconciles raw spellings: «сімʼю» ~ «сім'ю» ~ «сімю».
    w1 = w1.translate(_NO_APOSTROPHE)
    w2 = w2.translate(_NO_APOSTROPHE)
    if w1 == w2:
        return True
    # Short-word inflection: one word extends the other by 1-2 chars
    # (case/plural endings on stems too short for the prefix rule below).
    # Requires a >=3-char stem so we don't match "кіт" ~ "кит".
    if len(w1) >= 3 and len(w2) >= 3:
        short, long = (w1, w2) if len(w1) <= len(w2) else (w2, w1)
        if long.startswith(short) and len(long) - len(short) <= 2:
            return True
    if len(w1) >= 5 and len(w2) >= 5 and w1[:5] == w2[:5]:
        return True
    return False


def tokenize(text: str) -> List[str]:
    """Words for indexing and search.

    Strips apostrophes consistently with :func:`normalize` (``O'Brien`` ->
    ``["obrien"]``, ``сім'ю`` -> ``["сімю"]``), splits on punctuation and
    underscores (``dark_knight`` -> ``["dark", "knight"]``), and glues
    two-part hyphenated tokens when one part is shorter than 3 chars
    (``90-ті`` -> ``["90ті"]``) while keeping multi-part phrases and
    non-hyphenated splits separate (``man-at-arms`` -> ``["man", "at",
    "arms"]``, ``1.Dragon`` -> ``["1", "dragon"]``).
    """
    s = unicodedata.normalize("NFC", text or "").lower()
    s = s.translate(_NO_APOSTROPHE)
    words: List[str] = []
    for chunk in s.split():
        stripped = re.sub(r"^[\W_]+|[\W_]+$", "", chunk)
        if not stripped:
            continue
        parts = [p for p in re.split(r"[\W_]+", stripped) if p]
        if not parts:
            continue
        if (len(parts) == 2 and any(len(p) < 3 for p in parts)
                and re.fullmatch(r"[^\W_]+[-–—][^\W_]+", stripped)):
            words.append("".join(parts))
        else:
            words.extend(parts)
    return words


class LoreDocument(TypedDict):
    """Public shape of a page dict in ``LoreEngine.documents``."""

    key: str
    name: str
    category: str
    content: str
    teaser: str
    aliases: List[str]
    tags: List[str]


class LoreEngine:
    """Loads a knowledge base folder and serves dosed lore to the LLM.

    :param kb_dir: path to the knowledge base root (category folders inside).
    :param lang: a :class:`loreengine.lang.LanguagePack` (default: Ukrainian).
    :param render_label: header label used by :meth:`render` (default "WIKI").
    :param verbose: emit load diagnostics via the ``"loreengine"`` logger.
        Set ``False`` in libraries/tests to stay silent; a missing folder
        then just yields an empty engine instead of logging.
    """

    def __init__(self, kb_dir: PathLikeStr = "knowledge_base", lang: LanguagePack = UK,
                 render_label: str = "WIKI", verbose: bool = True):
        self.kb_dir: str = os.fspath(kb_dir)
        self.lang = lang
        self.render_label = render_label
        self.verbose = verbose
        self.documents: List[LoreDocument] = []
        self._words: List[Set[str]] = []      # body words, parallel to documents
        self._tf: List[Counter[str]] = []     # per-doc body term frequencies
        self._doc_len: List[int] = []         # per-doc body token counts
        self._avgdl: float = 0.0              # average document length (tokens)
        self._key_words: List[Set[str]] = []  # key/name/alias words, parallel
        self._key_translit: List[Set[str]] = []  # transliterated key words (when != orig)
        self._title_words: List[Set[str]] = []  # first-line "Name - ..." words, parallel
        # Per-doc 3-char-prefix -> body words, built only when len(body_words) > 64.
        # Shorter docs scan body_words directly to save memory with identical results.
        self._body_pref: List[Optional[Dict[str, Set[str]]]] = []
        self._body_translit: List[Dict[str, Set[str]]] = []  # translit -> {orig body words}
        self._df: Dict[str, int] = {}         # word -> document frequency (IDF)
        self._pref: Dict[str, Set[int]] = {}  # 3-char prefix -> doc indices
        self._short_key: Dict[str, Set[int]] = {}  # <3-char word -> doc indices
        self._cooc: Dict[str, Dict[str, float]] = {}  # word -> {neighbor: weight}
        self._summary_cache: Optional[str] = None
        self._index_cache: Optional[str] = None
        self._cat_index_cache: Dict[str, str] = {}
        self._full_text_cache: Optional[str] = None
        self.load()

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        if self.verbose:
            logger.log(level, "%s", msg)

    def __repr__(self) -> str:
        return f"LoreEngine(kb_dir={self.kb_dir!r}, documents={len(self.documents)})"

    # ------------------------------------------------------------------ load
    @staticmethod
    def _parse_meta(content: str) -> Tuple[List[str], List[str], str]:
        """Parse optional ``Key: value`` header lines at the top of a page.

        Recognized keys: ``Aliases`` (comma-separated alternate names) and
        ``Tags``. Header parsing stops at the first blank line or the first
        non-header line, so pages without headers come back unchanged
        (fully backwards compatible). Returns ``(aliases, tags, body)``.
        """
        aliases: List[str] = []
        tags: List[str] = []
        lines = content.split("\n")
        start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                start = i + 1
                break
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                k = key.strip().lower()
                if k in ("aliases", "alias", "аліаси", "aliasy"):
                    aliases.extend(a.strip() for a in val.split(",") if a.strip())
                    start = i + 1
                    continue
                if k in ("tags", "tag", "теги", "tagi"):
                    tags.extend(t.strip() for t in val.split(",") if t.strip())
                    start = i + 1
                    continue
            break  # first non-header line -> body starts here
        body = "\n".join(lines[start:]).strip()
        return aliases, tags, body

    def load(self) -> None:
        """Scan kb_dir and load every .txt document.

        Resets all state (documents, indexes, caches) up front, so a failed
        reload (e.g. a missing folder) leaves a consistently empty engine
        instead of stale caches over no documents.
        """
        self.documents = []
        self._words = []
        self._tf = []
        self._doc_len = []
        self._avgdl = 0.0
        self._key_words = []
        self._key_translit = []
        self._title_words = []
        self._body_pref = []
        self._body_translit = []
        self._df = {}
        self._pref = {}
        self._short_key = {}
        self._index_cache = None
        self._cat_index_cache = {}
        self._full_text_cache = None
        self._cooc = {}
        self._summary_cache = None
        if not os.path.isdir(self.kb_dir):
            self._log(
                f"[LoreEngine] Folder '{self.kb_dir}' not found — lore is empty.",
                logging.WARNING,
            )
            return
        try:
            categories = sorted(os.listdir(self.kb_dir))
        except OSError as e:
            self._log(
                f"[LoreEngine] Could not list '{self.kb_dir}': {e}",
                logging.WARNING,
            )
            return
        for category in categories:
            cat_path = os.path.join(self.kb_dir, category)
            if not os.path.isdir(cat_path):
                continue
            try:
                filenames = sorted(os.listdir(cat_path))
            except OSError as e:
                self._log(
                    f"[LoreEngine] Could not list '{cat_path}': {e}",
                    logging.WARNING,
                )
                continue
            for filename in filenames:
                if not filename.endswith(".txt"):
                    continue
                filepath = os.path.join(cat_path, filename)
                if not os.path.isfile(filepath):
                    continue
                try:
                    with open(filepath, "r", encoding="utf-8-sig") as f:
                        raw = f.read()
                except (OSError, UnicodeDecodeError) as e:
                    self._log(
                        f"[LoreEngine] Could not read {filename}: {e}",
                        logging.WARNING,
                    )
                    continue
                # Normalize line endings so CRLF files never leak \r into
                # dosed content, teasers, or linter title parsing.
                raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
                if not raw:
                    continue
                aliases, tags, content = self._parse_meta(raw)
                if not content:
                    if aliases or tags:
                        self._log(
                            f"[LoreEngine] {filename}: header only "
                            "(aliases/tags without body) — page skipped.",
                            logging.WARNING,
                        )
                    continue
                key = os.path.splitext(filename)[0]
                doc: LoreDocument = {
                    "key": key,
                    "name": key.replace("_", " "),
                    "category": category,
                    "content": content,
                    "teaser": self._first_sentence(content),
                    "aliases": aliases,
                    "tags": tags,
                }
                self.documents.append(doc)
                key_words = set(tokenize(key + " " + doc["name"]))
                for alias in aliases:
                    key_words |= set(tokenize(alias))
                # Tags are a deterministic synonym bridge the author controls
                # (e.g. Tags: beer, drinks on a tavern page) — index them.
                for tag in tags:
                    key_words |= set(tokenize(tag))
                body_tokens = tokenize(content)
                tf = Counter(body_tokens)
                body_words = set(tf)
                self._tf.append(tf)
                self._doc_len.append(len(body_tokens))
                self._words.append(body_words)
                body_translit: Dict[str, Set[str]] = {}
                if self.lang.translit:
                    for bw in body_words:
                        tbw = self._translit(bw)
                        if tbw and tbw != bw:
                            body_translit.setdefault(tbw, set()).add(bw)
                self._body_translit.append(body_translit)
                if len(body_words) > _BODY_PREF_THRESHOLD:
                    bp: Dict[str, Set[str]] = {}
                    for bw in body_words:
                        if len(bw) >= 3:
                            bp.setdefault(bw[:3], set()).add(bw)
                        if self.lang.translit:
                            tbw = self._translit(bw)
                            if tbw != bw and len(tbw) >= 3:
                                bp.setdefault(tbw[:3], set()).add(bw)
                    self._body_pref.append(bp)
                else:
                    self._body_pref.append(None)
                self._key_words.append(key_words)
                if self.lang.translit:
                    kt = {
                        tkw for kw in key_words
                        for tkw in (self._translit(kw),)
                        if tkw and tkw != kw
                    }
                else:
                    kt = set()
                self._key_translit.append(kt)
                first_line = content.split("\n", 1)[0]
                title_parts = re.split(r"\s+[-—–]\s+", first_line, maxsplit=1)
                self._title_words.append(
                    set(tokenize(title_parts[0])) if len(title_parts) > 1 else set()
                )
        self._avgdl = (
            sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        )
        # Document frequency (content words + latin keys + aliases + tags + translit forms)
        for i in range(len(self.documents)):
            doc_words = self._words[i] | self._key_words[i]
            if self.lang.translit:
                translit_forms = {
                    tw for w in doc_words
                    for tw in (self._translit(w),)
                    if tw and tw != w and tw not in doc_words
                }
                doc_words = doc_words | translit_forms
            for w in doc_words:
                self._df[w] = self._df.get(w, 0) + 1
        self._build_prefix_index()
        self._build_cooc()
        self._log(f"[LoreEngine] Documents loaded: {len(self.documents)}", logging.INFO)

    def _build_prefix_index(self) -> None:
        """Inverted index: 3-char word prefix -> doc indices, plus a
        verbatim index of short (<3-char) words.

        Every :func:`words_match` acceptance (exact match, shared 5-char
        prefix, short-word inflection with a >=3-char stem) implies a shared
        3-char prefix (for both raw words and their transliterated forms), so
        the prefix index is a guaranteed SUPERSET of everything the precise
        ``_score_doc`` ranker may accept — narrowing to it never loses
        results. Words shorter than 3 chars can only ever match exactly, so
        they are indexed verbatim in ``_short_key`` and a degenerate short
        transliteration still narrows instead of forcing a full scan. Built
        eagerly at load: read-heavy workload, no cold-start latency in the
        game loop.
        """
        self._pref = {}
        self._short_key = {}
        for i in range(len(self.documents)):
            for w in self._words[i] | self._key_words[i]:
                if len(w) >= 3:
                    self._pref.setdefault(w[:3], set()).add(i)
                else:
                    self._short_key.setdefault(w, set()).add(i)
                if self.lang.translit:
                    tw = self._translit(w)
                    if tw and tw != w:
                        if len(tw) >= 3:
                            self._pref.setdefault(tw[:3], set()).add(i)
                        else:
                            self._short_key.setdefault(tw, set()).add(i)

    def _cooc_weight(self, a: str, b: str, count: int) -> float:
        """Normalized co-occurrence strength (cosine * min-IDF * repeat bonus)."""
        df_a = max(1, self._df.get(a, 1))
        df_b = max(1, self._df.get(b, 1))
        cosine = count / math.sqrt(df_a * df_b)
        idf_min = min(self._idf(a), self._idf(b))
        repeat_bonus = 1.5 if count >= 2 else 1.0
        return cosine * idf_min * repeat_bonus

    def _build_cooc(self) -> None:
        """Co-occurrence index: word -> {neighbor: weight}.

        Distributional semantics — "you shall know a word by the company it
        keeps" (Firth). Words appearing in the same document are semantically
        related, so a query about "drink ale gossip" can reach a tavern page
        that never uses those words but shares their context. Powers the
        semantic expansion in :meth:`relevant`.

        Pair counts are normalized by ``count / sqrt(df[a] * df[b]) * min(IDF)``
        with a bonus for repeated co-occurrences (``count >= 2``), so common
        generic words do not dominate and repeated edges outrank one-off
        coincidences while still preserving 1-doc semantic bridges.

        Memory is bounded for giant corpora: each word's neighbor map is
        pruned to a deterministic top-``_COOC_PRUNE_KEEP`` whenever it grows
        past ``_COOC_PRUNE_TRIGGER``, then to ``_COOC_NEIGHBORS`` at the end.
        """
        pair_counts: Dict[str, Dict[str, int]] = {}
        for i in range(len(self.documents)):
            # Keep the most discriminative words (highest IDF) — rare words
            # carry the semantic signal; alphabetical truncation would drop
            # them arbitrarily. Deterministic tie-break by word.
            ws = sorted((w for w in self._words[i]
                         if len(w) >= 3 and w not in self.lang.stopwords),
                        key=lambda w: (-self._idf(w), w))[:_COOC_MAX_WORDS]
            for a in range(len(ws)):
                wa = ws[a]
                for b in range(a + 1, len(ws)):
                    wb = ws[b]
                    for x, y in ((wa, wb), (wb, wa)):
                        nbs = pair_counts.setdefault(x, {})
                        nbs[y] = nbs.get(y, 0) + 1
                        if len(nbs) > _COOC_PRUNE_TRIGGER:
                            top = sorted(
                                nbs.items(),
                                key=lambda kv: (
                                    -self._cooc_weight(x, kv[0], kv[1]),
                                    kv[0],
                                ),
                            )[:_COOC_PRUNE_KEEP]
                            # Prune in place (nbs is pair_counts[x]) so no stale
                            # reference to the discarded map survives.
                            nbs.clear()
                            nbs.update(top)
        self._cooc = {}
        for w, nbs in pair_counts.items():
            scored = [
                (nb, self._cooc_weight(w, nb, cnt))
                for nb, cnt in nbs.items()
            ]
            top_scored = sorted(scored, key=lambda kv: (-kv[1], kv[0]))[:_COOC_NEIGHBORS]
            self._cooc[w] = dict(top_scored)

    @staticmethod
    def _first_sentence(text: str, limit: int = 140) -> str:
        """First sentence of a document, as a short teaser for the index."""
        text = text.replace("\n", " ").strip()
        m = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
        s = m[0].strip() if m else text
        if len(s) > limit:
            cut = s[:limit]
            # Prefer cutting at the last space; fall back to a hard cut so a
            # single long word still yields a teaser instead of vanishing.
            s = cut[:cut.rfind(" ")] if " " in cut else cut
            s = s.rstrip() + "…"
        return s

    def _idf(self, word: str) -> float:
        """Word weight: rare words matter more (log TF-IDF)."""
        n = len(self.documents) or 1
        return math.log(1 + n / max(1, self._df.get(word, 0)))

    # ----------------------------------------------------------------- index
    def index(self, category: Optional[str] = None) -> str:
        """Compact index of all pages — a stable prefix, friendly to caching.

        :param category: optional category name to index only that folder
            (use when the full index outgrows a cheap cached prefix).
        """
        if category is None:
            if self._index_cache is not None:
                return self._index_cache
            lines = []
            by_cat: Dict[str, List[LoreDocument]] = {}
            for doc in self.documents:
                by_cat.setdefault(doc["category"], []).append(doc)
            for cat in sorted(by_cat):
                for doc in by_cat[cat]:
                    lines.append(f"- [{cat}] {doc['key']}: {doc['teaser']}")
            self._index_cache = "\n".join(lines)
            return self._index_cache

        if category in self._cat_index_cache:
            return self._cat_index_cache[category]
        lines = [f"- [{doc['category']}] {doc['key']}: {doc['teaser']}"
                 for doc in self.documents if doc["category"] == category]
        out = "\n".join(lines)
        self._cat_index_cache[category] = out
        return out

    def categories(self) -> List[str]:
        """Sorted list of category names present in the knowledge base."""
        return sorted({doc["category"] for doc in self.documents})

    def category_summary(self) -> str:
        """Compact world-map for very large knowledge bases.

        When even a per-category index outgrows a cheap cached prefix, this
        gives the model a one-screen map — one line per category with its page
        count and a few sample keys — so it can request the right
        ``index(category)`` or ``get_pages()`` next. Stable prefix, cacheable.
        """
        if self._summary_cache is not None:
            return self._summary_cache
        by_cat: Dict[str, List[str]] = {}
        for doc in self.documents:
            by_cat.setdefault(doc["category"], []).append(doc["key"])
        lines = []
        for cat in sorted(by_cat):
            keys = by_cat[cat]
            count = len(keys)
            unit = "page" if count == 1 else "pages"
            sample = ", ".join(keys[:5]) + ("…" if count > 5 else "")
            lines.append(f"- {cat} ({count} {unit}): {sample}")
        self._summary_cache = "\n".join(lines)
        return self._summary_cache

    # ---------------------------------------------------------------- search
    def _translit(self, word: str) -> str:
        """Player-alphabet word -> latin (to match file keys)."""
        return "".join(self.lang.translit.get(c, c) for c in word)

    def _query_words(self, query: str) -> List[str]:
        """Meaningful query words (>=3 chars, minus stopwords)."""
        return [w for w in tokenize(query)
                if len(w) >= 3 and w not in self.lang.stopwords]

    def _expand(self, words: List[str]) -> Tuple[List[str], List[float]]:
        """Expand query words with semantic neighbors from the co-occurrence
        space. Returns ``(words, weights)``: originals weight 1.0, and top
        expansion neighbors normalized into ``[_COOC_EXPAND_MIN_WEIGHT, _COOC_WEIGHT]``
        proportional to their co-occurrence strength. Deterministic; no-op when
        the co-occurrence index is empty.

        Common words are already excluded: the co-occurrence index weights
        pairs by normalized cosine * min-IDF and prunes to the top neighbors per
        word, and IDF down-weights common words at scoring time. The candidate
        cap in :meth:`relevant` bounds latency on large corpora.
        """
        if not self._cooc:
            return list(words), [1.0] * len(words)
        extra: Dict[str, float] = {}
        wordset = set(words)
        for w in words:
            for nb, wt in self._cooc.get(w, {}).items():
                if nb not in wordset:
                    extra[nb] = max(extra.get(nb, 0.0), wt)
        top = sorted(extra.items(), key=lambda kv: (-kv[1], kv[0]))[:_COOC_EXPAND]
        if not top:
            return list(words), [1.0] * len(words)
        max_wt = top[0][1]
        min_wt = top[-1][1]
        span = max_wt - min_wt
        exp_weights: List[float] = []
        for _, wt in top:
            if span <= 0.0:
                exp_weights.append(_COOC_WEIGHT)
            else:
                norm = (wt - min_wt) / span
                exp_weights.append(
                    _COOC_EXPAND_MIN_WEIGHT + (_COOC_WEIGHT - _COOC_EXPAND_MIN_WEIGHT) * norm
                )
        out_words = list(words) + [w for w, _ in top]
        out_weights = [1.0] * len(words) + exp_weights
        return out_words, out_weights

    def _word_matches_set(self, w: str, words: Set[str]) -> bool:
        """True if query word ``w`` (or its transliteration) matches ``words`` (raw or transliterated)."""
        tw = self._translit(w)
        if tw in words or w in words:
            return True
        if any(words_match(tw, kw) or words_match(w, kw) for kw in words):
            return True
        if self.lang.translit:
            for kw in words:
                tkw = self._translit(kw)
                if tkw != kw and (
                    tkw == w or tkw == tw
                    or words_match(w, tkw) or words_match(tw, tkw)
                ):
                    return True
        return False

    def _score_doc(self, i: int, words: List[str],
                   weights: Optional[List[float]] = None,
                   *, entity_only: bool = False) -> float:
        """Score document #i against query words (BM25 + exact-vs-fuzzy + translit).

        ``weights`` (parallel to ``words``) discounts semantic expansion words.
        Exact hits score ``1.0x`` and fuzzy ``words_match`` hits score ``0.7x``
        on both ``key_words`` and ``body_words``. Body hits use BM25 TF scaling
        (``k1=1.2, b=0.75``) over document token counts and lengths.
        When ``entity_only=True`` (scene entity dosing), only ``key_words``
        (key, name, aliases, tags) are scored — body text alone never produces
        a match, and multi-word queries require more than half of their words
        to match ``key_words`` (or all words to match ``key_words`` plus the
        page's ``Name - ...`` first-line title with at least one key hit).
        """
        score = 0.0
        key_words = self._key_words[i]
        key_translit = self._key_translit[i]
        body_words = self._words[i]
        body_pref = self._body_pref[i]
        body_translit = self._body_translit[i]
        doc_tf = self._tf[i]
        dl = self._doc_len[i]
        avgdl = self._avgdl or 1.0
        len_norm = 1.0 - _BM25_B + _BM25_B * (dl / avgdl)
        has_translit = bool(self.lang.translit)
        key_hits = 0
        for idx, w in enumerate(words):
            mult = weights[idx] if weights else 1.0
            tw = self._translit(w)
            idf = self._idf(w)
            if tw != w and self._df.get(tw):
                # The translit bridge resolved this query word to a latin
                # word present in the KB — weight it by the latin form's
                # IDF, not the always-maximal IDF of the unseen native
                # spelling (df=0).
                idf = self._idf(tw)
            # Hit on the document key/name/aliases/tags (latin via translit, or native)
            exact_key = (
                tw in key_words or w in key_words
                or w in key_translit or (tw != w and tw in key_translit)
            )
            if exact_key:
                score += mult * _EXACT_WEIGHT * (3.0 * idf + 2.0)
                key_hits += 1
            elif (any(words_match(tw, kw) or words_match(w, kw) for kw in key_words)
                  or (key_translit and any(
                      words_match(tw, tkw) or words_match(w, tkw)
                      for tkw in key_translit))):
                score += mult * _FUZZY_WEIGHT * (3.0 * idf + 2.0)
                key_hits += 1
            elif not entity_only:
                # Hit on body words: exact (incl. translit equivalence) beats fuzzy.
                exact_tf = doc_tf.get(w, 0)
                if tw != w:
                    exact_tf += doc_tf.get(tw, 0)
                if body_translit:
                    for orig in body_translit.get(w, ()):
                        if orig != w and orig != tw:
                            exact_tf += doc_tf.get(orig, 0)
                    if tw != w:
                        for orig in body_translit.get(tw, ()):
                            if orig != w and orig != tw:
                                exact_tf += doc_tf.get(orig, 0)
                if exact_tf > 0:
                    tf = exact_tf
                    exact_mult = _EXACT_WEIGHT
                else:
                    tf = 0
                    if body_pref is not None:
                        cand_bw = set(body_pref.get(w[:3], ()))
                        if tw != w and len(tw) >= 3:
                            cand_bw |= body_pref.get(tw[:3], set())
                    else:
                        cand_bw = body_words
                    for cw in cand_bw:
                        matched = words_match(w, cw) or (tw != w and words_match(tw, cw))
                        if not matched and has_translit:
                            tcw = self._translit(cw)
                            if tcw != cw:
                                matched = words_match(w, tcw) or (tw != w and words_match(tw, tcw))
                        if matched:
                            tf += doc_tf[cw]
                    exact_mult = _FUZZY_WEIGHT
                if tf > 0:
                    tf_norm = (tf * (_BM25_K1 + 1.0)) / (tf + _BM25_K1 * len_norm)
                    title_words = self._title_words[i]
                    in_title = self._word_matches_set(w, title_words)
                    score += mult * exact_mult * (1.5 if in_title else 1.0) * tf_norm * idf
        if entity_only:
            if not key_hits:
                return 0.0
            if len(words) >= 2 and key_hits * 2 <= len(words):
                title_words = self._title_words[i]
                if not title_words:
                    return 0.0
                allowed = key_words | title_words
                if not all(self._word_matches_set(w, allowed) for w in words):
                    return 0.0
        return score

    def _candidates(self, words: List[str]) -> Sequence[int]:
        """Narrow documents (by index) via the prefix index before scoring.

        Every :func:`words_match` hit shares a >=3-char prefix with some
        query word (native or transliterated), so the 3-char-prefix lookup
        is a guaranteed superset of all scorable documents — the ranker
        filters precisely. A transliteration shorter than 3 chars can only
        match a short word exactly, so it is resolved through the verbatim
        ``_short_key`` index instead of forcing a full scan. Falls back to a
        full scan only when both indexes are empty; an index miss returns
        ``()`` instead of scanning the entire corpus.
        """
        if not self._pref and not self._short_key:
            return range(len(self.documents))
        cand: Set[int] = set()
        for w in words:
            tw = self._translit(w)
            cand |= self._pref.get(w[:3], set())
            if len(tw) >= 3:
                cand |= self._pref.get(tw[:3], set())
            else:
                # Degenerate transliteration: only an exact short-word match
                # is possible, and _short_key vouches for those.
                cand |= self._short_key.get(tw, set())
        if not cand:
            return ()
        return sorted(cand)

    def _min_score(self) -> float:
        """Acceptance floor for :meth:`search_one`.

        With BM25 length normalization and ``0.7x`` fuzzy weighting, a valid
        match on a very long document or a common word in a small KB can
        score below ``0.5``. Flooring at ``min(0.01, log(1 + n))`` accepts
        genuine matches across document lengths and corpus sizes while
        still rejecting zero-score documents.
        """
        return min(0.01, math.log(1 + len(self.documents)))

    def search_one(self, query: str, *, entity_only: bool = False) -> Optional[LoreDocument]:
        """Best single document for a query, or ``None`` if nothing scores.

        Public wrapper over the weighted search used by dosing/wiki-lookup.
        When ``entity_only=True``, restricts matches to entity metadata
        (keys, names, aliases, tags) rather than arbitrary body words.
        """
        words = self._query_words(query)
        if not words:
            return None
        best, best_score = -1, 0.0
        best_tie: Optional[Tuple[str, str]] = None
        for i in self._candidates(words):
            s = self._score_doc(i, words, entity_only=entity_only)
            if s <= 0.0:
                continue
            doc = self.documents[i]
            tie = (doc["key"], doc["category"])
            if s > best_score or (s == best_score and (best_tie is None or tie < best_tie)):
                best, best_score, best_tie = i, s, tie
        if best < 0 or best_score < self._min_score():
            return None
        return self.documents[best]

    # Backwards-compatible private alias.
    def _find_best(self, query: str) -> Optional[LoreDocument]:
        return self.search_one(query)

    def _collect_pages(self, queries: Iterable[object], max_pages: int,
                       *, entity_only: bool = False) -> List[LoreDocument]:
        """Resolve an iterable of queries to unique documents, up to max_pages."""
        docs: List[LoreDocument] = []
        seen: Set[Tuple[str, str]] = set()
        if max_pages <= 0:
            return docs
        for q in queries or []:
            if not isinstance(q, str) or not q.strip():
                continue
            # Cap check before the (potentially costly) search: once the
            # budget is full, stop instead of appending one page past it.
            if len(docs) >= max_pages:
                break
            doc = self.search_one(q.strip(), entity_only=entity_only)
            if doc:
                doc_id = (doc["category"], doc["key"])
                if doc_id not in seen:
                    seen.add(doc_id)
                    docs.append(doc)
        return docs

    def get_pages(self, queries: Iterable[object], max_pages: int = 3) -> List[LoreDocument]:
        """DOCUMENTS for a list of queries (wiki-lookup), deduplicated."""
        return self._collect_pages(queries, max_pages, entity_only=False)

    def pages_for_entities(self, names: Iterable[object], max_pages: int = 6) -> List[LoreDocument]:
        """DOCUMENTS for the entity names of the current scene (dosing)."""
        return self._collect_pages(names, max_pages, entity_only=True)

    def relevant(self, query: str, k: int = 2, exclude: Iterable[str] = ()) -> List[LoreDocument]:
        """Targeted search: up to k DOCUMENTS relevant to the query.

        This is the RAG-like layer: besides exact/fuzzy keyword matching it
        expands the query with semantically-related words from the
        co-occurrence index, so it reaches pages that share *context* with the
        query even without keyword overlap. Deterministic.

        Candidates are gathered in two tiers so large corpora stay fast:
        (1) pages matching the ORIGINAL query words — the correctness anchor,
        always fully included (and may itself exceed the cap); (2) pages
        matching only the semantic expansion words — the reach, added only
        while the candidate set is below ``_RELEVANT_MAX_CANDIDATES``. The
        ranker weights original words 1.0 and expansion words
        ``_COOC_WEIGHT``, so keyword matches outrank semantic-only matches.
        ``search_one`` is exact and unaffected by the cap.
        """
        words = self._query_words(query)
        if not words or k <= 0:
            return []
        exp_words, weights = self._expand(words)
        exp_only = exp_words[len(words):]
        exp_weights = weights[len(words):]
        exclude = set(exclude)
        # Tier 1: keyword matches (always included — they anchor correctness).
        primary = set(self._candidates(words))
        # Tier 2: semantic-only matches, bounded so a generic query on a huge
        # corpus cannot blow up the scoring loop. When exceeding budget,
        # prioritize by total weight of matching expansion words (not key order).
        cand = sorted(primary)
        if exp_only and len(cand) < _RELEVANT_MAX_CANDIDATES:
            sec_scores: Dict[int, float] = {}
            for ew, ew_wgt in zip(exp_only, exp_weights):
                for doc_idx in self._candidates([ew]):
                    if doc_idx not in primary:
                        sec_scores[doc_idx] = sec_scores.get(doc_idx, 0.0) + ew_wgt
            if sec_scores:
                budget = _RELEVANT_MAX_CANDIDATES - len(cand)
                ranked_sec = sorted(
                    sec_scores,
                    key=lambda idx: (-sec_scores[idx], self.documents[idx]["key"],
                                     self.documents[idx]["category"], idx),
                )
                cand.extend(ranked_sec[:budget])
        scored = []
        for i in cand:
            doc = self.documents[i]
            if doc["key"] in exclude or f"{doc['category']}/{doc['key']}" in exclude:
                continue
            s = self._score_doc(i, exp_words, weights)
            if s > 0:
                scored.append((s, i))
        # Deterministic tie-break by (key, category), so equal scores never
        # depend on category or filesystem enumeration order.
        scored.sort(key=lambda x: (-x[0], self.documents[x[1]]["key"],
                                  self.documents[x[1]]["category"]))
        return [self.documents[i] for _, i in scored[:k]]

    # --------------------------------------------------------------- helpers
    def render(self, doc: LoreDocument) -> str:
        """Format a document for injection into a prompt."""
        return f"=== [{self.render_label} · {doc['category'].upper()}] {doc['name']} ===\n{doc['content']}"

    def full_text(self) -> str:
        """All lore as one block (only for a 'full lore' mode, not typical)."""
        if self._full_text_cache is None:
            self._full_text_cache = "\n\n".join(self.render(d) for d in self.documents)
        return self._full_text_cache

    def size_stats(self) -> Dict[str, int]:
        chars = sum(len(d["content"]) for d in self.documents)
        return {"documents": len(self.documents), "characters": chars,
                "categories": len(self.categories()),
                "cooc_words": len(self._cooc)}
