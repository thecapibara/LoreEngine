# LoreEngine

A lightweight **world-wiki engine for LLM-driven games** — a zero-dependency,
deterministic alternative to heavy RAG stacks.

**Pure Python stdlib. No langchain, no vector DB, no torch, no dependencies.**

---

## About

LLM-driven games face the same structural problem: the world is too big to
dump into every prompt. A modest setting — a few dozen characters, locations,
items, legends — easily reaches 15k+ tokens of lore, and feeding all of it on
every turn burns context, money, and the model's attention alike.

The usual answer is RAG: embed every page, store vectors, retrieve the
"closest" chunks. But for games, RAG is a poor fit:

- **Non-deterministic.** The same query can return different pages run to
  run. A game master that forgets who the tavern keeper is between two
  identical scenes breaks immersion.
- **Noisy.** Embedding similarity is fuzzy; it retrieves *vibes*, not the
  specific page the scene needs.
- **Heavy.** Vector DBs, embedding models, and orchestration frameworks are
  a lot of machinery for what is, at heart, a lookup problem.
- **Opaque.** When the wrong page comes back, there is no way to reason
  about why.

**LoreEngine takes a different bet: game lore is a wiki, not a haystack.**
The author already knows the entities, their names, their aliases, and how
they relate. Instead of hiding that structure inside embeddings, LoreEngine
exploits it directly — and feeds the world to the model **in doses, not
dumps**: ~3–5k lore tokens per turn, with the pages the current scene
actually needs *guaranteed* to be present.

What you get:

- **Determinism.** Same knowledge base + same query → same pages, every run.
  No randomness anywhere in the pipeline; every tie is broken by a total
  ordering.
- **Guarantees.** Entity pages for the current scene are found by name
  matching, not similarity — if the page exists, it is retrieved.
- **Zero dependencies.** Python standard library only. `pip install` and it
  works; nothing to compile, no model to download, no service to run.
- **Transparency.** Every result is explainable: which word matched, with
  what weight. The built-in linter audits the wiki itself.
- **Speed.** Sub-millisecond entity lookups and single-digit-millisecond
  semantic search on 20k-page corpora, on a single CPU core.

LoreEngine was built for and battle-tested in a full noir detective game
(9-chapter campaign, Ukrainian player queries against latin file keys), where
deterministic lore dosing is load-bearing: the model must never contradict
established facts, and the right page must arrive exactly when the scene
needs it.

### LoreEngine vs RAG

| | LoreEngine | Typical RAG |
|---|---|---|
| Retrieval | Inverted index + BM25 + co-occurrence | Vector similarity |
| Determinism | Exact, reproducible | Approximate, can vary |
| Dependencies | None (stdlib) | Embedding model + vector store |
| Latency (20k pages) | 0.4 ms lookup / ~15 ms semantic | Embedding + ANN search |
| Explainability | Word-level match + weight | Cosine distance |
| Semantic reach | Co-occurrence expansion (context-based) | Embedding similarity (meaning-based) |
| Best for | Structured world wikis, entity-centric games | Unstructured document QA |

RAG wins on unstructured corpora where no entity structure exists. LoreEngine
wins where the world *is* a wiki — named pages, aliases, categories — and the
game needs guarantees, not vibes.

---

## How it works — 4 layers

| Layer | API | What it does |
|---|---|---|
| **Index** | `engine.index()` | Compact table of contents of every page (~1–2k tokens). Put it in the *stable prefix* of your system prompt — providers cache it, so it's nearly free after turn one. For large KBs, `engine.index("characters")` serves one category at a time (two-layer index). |
| **Dosing** | `engine.pages_for_entities(names)` | Full pages only for the entities of the *current scene* (strict entity-field matching on keys, titles, aliases, and tags — missing entities never fall back to random body matches). |
| **Targeted search** | `engine.relevant(action)` | Given the player's action text, pull 1–2 extra pages (like RAG, but deduplicated, capped, deterministic). Expands the query with semantically-related words, so it reaches pages that share *context* with the action even without keyword overlap. |
| **Wiki-lookup** | `engine.get_pages(queries)` | Let the model itself request pages (e.g. a `wiki_lookup` field in its JSON reply); feed them in next turn. `engine.search_one(query, entity_only=False)` exposes the single-best-page lookup. |

## Search machinery

- **BM25 ranking with exact-vs-fuzzy weighting** — rare words matter more than
  common ones, document length and term frequency saturate cleanly, and exact
  matches outrank fuzzy prefix hits;
- **Transliteration & diacritic folding** — file keys are latin (`dark_knight.txt`),
  player queries can be Cyrillic («Дарк Найт»), and diacritic-free queries
  (e.g. Polish `wroclaw` / `krol`) match diacritic forms (`Wrocław` / `król`)
  across both keys and body text;
- **Fuzzy prefix matching** — handles case endings and inflection:
  `dragon` ≈ `dragons` ≈ `dragonkin`, and short stems too: `ліс` ≈ `лісу`,
  `cat` ≈ `cats`. Multi-part hyphenated phrases (`Jean-Luc Picard`,
  `man-at-arms`) and snake_case identifiers split into searchable words, while
  2-part hyphenated tokens with a short fragment (`< 3` chars) glue back
  together so `90-ті` stays searchable as `90ті` (and reaches a latin key like
  `lyhi_90ti` through the transliteration bridge);
- **Inverted prefix index** — queries narrow to candidate pages via a
  3-char prefix index before the precise BM25 ranker runs. Results are
  identical to a full scan (the index is a guaranteed superset, verified by
  a differential test), misses return in O(1) without scanning the corpus, and
  entity-name lookups stay fast as the KB grows;
- **Semantic expansion** — a co-occurrence index ("you shall know a word by
  the company it keeps") links words that share documents, weighted by
  co-occurrence strength and IDF. `relevant()` expands the query with the
  strongest semantic neighbors using proportional weights below exact keywords,
  so a player action about *drinks and rumors* can surface a tavern page that
  never uses those words. Deterministic, no vectors, no model calls. Ties are
  broken deterministically by `(key, category)` across both `search_one()` and
  `relevant()`, so equal scores never depend on filesystem enumeration order.

## Scale

Measured on synthetic corpora (25-word pages, stdlib Python, single core):

| Corpus | Load + index | Peak RSS | `search_one` | `relevant(k=2)` |
|---|---|---|---|---|
| 10k docs | ~2.5–4.5 s | ~400 MB | ~0.2 ms | ~5–25 ms |
| 20k docs | ~6.6 s | ~415 MB | ~0.4 ms | ~11–17 ms |

Query latency stays flat as the corpus grows: candidate sets come from the
inverted index, and `relevant()` caps its semantic tier so a generic query
can never score the whole corpus. The co-occurrence index is pruned during
construction (top neighbors per word), so memory stays bounded. For a game
KB — hundreds to a few thousand pages — load is well under a second.

## Installation

```bash
pip install loreengine
```

Or from source:

```bash
git clone https://github.com/thecapibara/LoreEngine.git
cd LoreEngine
pip install .
```

Requires Python 3.9+. No runtime dependencies.

## Knowledge base layout

One folder per category, one `.txt` per page (UTF-8, with or without BOM); the
file name is the page key. The first line becomes the page's teaser in the
index. Pages may declare an optional header with explicit aliases and tags
(`Aliases:`/`Tags:`, Ukrainian `Аліаси:`/`Теги:`, or Polish `Aliasy:`/`Tagi:`;
header lines are stripped from the dosed content; pages without a header work
exactly as before):

```text
Aliases: Ser Aldric, The Dark Knight
Tags: npc, iron_guard

Ser Aldric Vane - a wandering mercenary with a cursed blade.
```

```
knowledge_base/
    characters/
        dark_knight.txt
        tavern_keeper.txt
    locations/
        old_tavern.txt
    lore/
        dragon_legend.txt
```

## Quick start

```python
from loreengine import LoreEngine, EN

engine = LoreEngine("knowledge_base", lang=EN)  # verbose=False to silence logging

# 1. Stable prompt prefix (cached by the provider)
system_prompt = BASE_PROMPT + "\n\nWORLD INDEX:\n" + engine.index()

# 2. Dose the current scene
pages = engine.pages_for_entities(["Ser Aldric Vane", "Old Tavern"])

# 3. Pull extras from the player's action
extra = engine.relevant(player_action, exclude=[d["key"] for d in pages])

# 4. Inject
context = "\n\n".join(engine.render(d) for d in pages + extra)
```

## API reference

### `LoreEngine(kb_dir, lang=UK, render_label="WIKI", verbose=True)`

Loads the knowledge base (`str` or `os.PathLike[str]`, e.g. `pathlib.Path`) and
builds all indexes eagerly. When `verbose=True`, diagnostic messages are emitted
through standard `logging.getLogger("loreengine")` (never `print()` to stdout).

| Method | Returns | Purpose |
|---|---|---|
| `index(category=None)` | `str` | Table of contents (all pages or one category) |
| `category_summary()` | `str` | One-line-per-category overview |
| `pages_for_entities(names, max_pages=6)` | `List[LoreDocument]` | Dosing: strict entity-field lookup for scene entities |
| `relevant(query, k=2, exclude=())` | `List[LoreDocument]` | Semantic search over the player's action (`exclude` accepts `"key"` or `"category/key"`) |
| `search_one(query, entity_only=False)` | `LoreDocument \| None` | Single best page for a query (full search or entity-only) |
| `get_pages(queries, max_pages=3)` | `List[LoreDocument]` | Wiki-lookup for model-requested pages |
| `render(doc)` | `str` | Format a page for prompt injection |
| `full_text()` | `str` | Entire KB as one block (special cases only) |
| `size_stats()` | `dict` | Document/character/index statistics |
| `load()` | `None` | Reload the KB and rebuild all state |

Each `LoreDocument` is a `TypedDict`: `key`, `name`, `category`, `content`,
`teaser`, `aliases`, `tags`.

## Languages

Built-in packs: `UK` (Ukrainian), `EN` (English), `PL` (Polish). Each pack
carries stopwords, a transliteration map, and linter regexes. Roll your own:

```python
from loreengine import LanguagePack

DE = LanguagePack(
    stopwords=frozenset({"der", "die", "das", "wer", "was", "wo", ...}),
    translit={"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"},
)
engine = LoreEngine("knowledge_base", lang=DE)
```

## Linter

Dev tool that keeps the wiki consistent:

```bash
python -m loreengine knowledge_base --lang en [--allowlist allowlist.txt] [--strict]
# or via the installed console script:
loreengine-lint knowledge_base --lang en [--allowlist allowlist.txt] [--strict]
```

- cross-references: mentions with no page → warning (allowlist supported);
- engine self-test: searching canonical titles across categories must return
  their own pages (catches transliteration/fuzzy regressions);
- index guard: warns when the index outgrows the cacheable size and points
  at `engine.index(category)` as the fix;
- structure: duplicate keys, non-latin keys, oversized pages, duplicate
  `Aliases:` declarations across pages.

Names that live outside the wiki (e.g. your story's per-chapter cast) are
passed via `extra_names` — the linter knows nothing about any game:

```python
from loreengine import lint_kb, EN
report = lint_kb("knowledge_base", lang=EN, extra_names={"Cornelius Blackwood"})
```

## Example

See [`examples/knowledge_base/`](examples/knowledge_base/) — a tiny fantasy
wiki you can lint and query right away.

## Development

```bash
# Tests (105 tests: search, semantics, scale, linter, CLI, regressions)
python -m unittest discover -s tests -v

# Type checking (strict; the package ships py.typed; install mypy via `pip install -e ".[dev]"`)
python -m mypy --strict loreengine

# Lint the example KB
python -m loreengine examples/knowledge_base --lang en
```

## License

MIT
