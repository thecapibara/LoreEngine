"""Tests for loreengine. Run: python -m unittest discover -s tests -v"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loreengine import LoreEngine, UK, EN, lint_kb, normalize, tokenize, words_match
from loreengine.lint import build_kb_registry

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_KB = os.path.join(HERE, "..", "examples", "knowledge_base")


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LoreEngine(EXAMPLE_KB, lang=EN)

    def test_loads_example_docs(self):
        self.assertEqual(len(self.engine.documents), 4)
        keys = {d["key"] for d in self.engine.documents}
        self.assertEqual(keys, {"dark_knight", "tavern_keeper",
                                "old_tavern", "dragon_legend"})

    def test_index_lists_all_docs(self):
        idx = self.engine.index()
        for key in ("dark_knight", "tavern_keeper", "old_tavern", "dragon_legend"):
            self.assertIn(key, idx)

    def test_dosing_by_canonical_name(self):
        docs = self.engine.pages_for_entities(["Ser Aldric Vane"])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["key"], "dark_knight")

    def test_dosing_by_inflection(self):
        # Fuzzy prefix match must survive inflection: "dragons" -> "dragon"
        docs = self.engine.pages_for_entities(["dragons"])
        self.assertTrue(docs and docs[0]["key"] == "dragon_legend",
                        f"got {[d['key'] for d in docs]}")
    def test_dosing_dedupes(self):
        docs = self.engine.pages_for_entities(["Old Tavern", "the Old Tavern"])
        self.assertEqual(len(docs), 1)

    def test_relevant_excludes(self):
        docs = self.engine.relevant("tavern rumors", k=2, exclude=["old_tavern"])
        self.assertNotIn("old_tavern", [d["key"] for d in docs])

    def test_relevant_empty_query(self):
        self.assertEqual(self.engine.relevant(""), [])

    def test_get_pages_wiki_lookup(self):
        docs = self.engine.get_pages(["dragon", "Aldric"])
        keys = {d["key"] for d in docs}
        self.assertIn("dragon_legend", keys)
        self.assertIn("dark_knight", keys)

    def test_render_label_configurable(self):
        engine = LoreEngine(EXAMPLE_KB, lang=EN, render_label="CODEX")
        doc = engine.documents[0]
        self.assertIn("[CODEX ·", engine.render(doc))

    def test_missing_dir_is_empty_not_crash(self):
        engine = LoreEngine("/nonexistent/kb/path", lang=EN)
        self.assertEqual(engine.documents, [])
        self.assertEqual(engine.index(), "")

    def test_size_stats(self):
        stats = self.engine.size_stats()
        self.assertEqual(stats["documents"], 4)
        self.assertGreater(stats["characters"], 0)

    def test_search_one_public(self):
        doc = self.engine.search_one("Ser Aldric Vane")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["key"], "dark_knight")
        self.assertIsNone(self.engine.search_one(""))
        self.assertIsNone(self.engine.search_one("zzz qqq xxx"))

    def test_index_by_category(self):
        idx = self.engine.index("characters")
        self.assertIn("dark_knight", idx)
        self.assertIn("tavern_keeper", idx)
        self.assertNotIn("old_tavern", idx)
        self.assertEqual(self.engine.index("nonexistent"), "")

    def test_categories(self):
        self.assertEqual(self.engine.categories(),
                         ["characters", "locations", "lore"])

    def test_verbose_false_is_silent(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            LoreEngine(EXAMPLE_KB, lang=EN, verbose=False)
            LoreEngine("/nonexistent/kb/path", lang=EN, verbose=False)
        self.assertEqual(buf.getvalue(), "")

    def test_empty_and_whitespace_files_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "lore")
            os.makedirs(cat)
            for name, body in (("empty.txt", ""), ("spaces.txt", "   \n\t\n")):
                with open(os.path.join(cat, name), "w", encoding="utf-8") as f:
                    f.write(body)
            with open(os.path.join(cat, "real.txt"), "w", encoding="utf-8") as f:
                f.write("A real page.\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            self.assertEqual([d["key"] for d in engine.documents], ["real"])

    def test_teaser_hard_cut_long_word(self):
        # A single word longer than the teaser limit must still yield a
        # teaser (hard cut), not vanish via rsplit.
        teaser = LoreEngine._first_sentence("x" * 300, limit=140)
        self.assertTrue(teaser.endswith("…"))
        self.assertLessEqual(len(teaser), 142)


class TestMatching(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("  Hello, World!  "), "hello world")

    def test_words_match_exact(self):
        self.assertTrue(words_match("dragon", "dragon"))

    def test_words_match_prefix(self):
        self.assertTrue(words_match("dragons", "dragonkin"))

    def test_words_match_no_false_short_prefix(self):
        self.assertFalse(words_match("cat", "castle"))

    def test_words_match_short_inflection(self):
        # Short stems too short for the 5-char prefix rule.
        self.assertTrue(words_match("ліс", "лісу"))
        self.assertTrue(words_match("cat", "cats"))
        self.assertTrue(words_match("син", "сина"))  # no vowel alternation
        # Different stems must not match.
        self.assertFalse(words_match("кіт", "кит"))
        self.assertFalse(words_match("cat", "castle"))


class TestTranslit(unittest.TestCase):
    def test_cyrillic_query_finds_latin_key(self):
        # Ukrainian pack transliterates Cyrillic queries to latin keys:
        # «Дарк Найта» -> dark/nayta -> matches key "dark_knight".
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "characters")
            os.makedirs(cat)
            with open(os.path.join(cat, "dark_knight.txt"), "w", encoding="utf-8") as f:
                f.write("Dark Knight - a wandering mercenary.\n")
            engine = LoreEngine(tmp, lang=UK)
            docs = engine.pages_for_entities(["Дарк Найта"])
            self.assertTrue(docs and docs[0]["key"] == "dark_knight",
                            f"got {[d['key'] for d in docs]}")


class TestAliases(unittest.TestCase):
    def _kb(self, tmp, body):
        cat = os.path.join(tmp, "characters")
        os.makedirs(cat, exist_ok=True)
        with open(os.path.join(cat, "zzz.txt"), "w", encoding="utf-8") as f:
            f.write(body)

    def test_explicit_alias_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._kb(tmp, "Aliases: Ser Aldric, The Dark Knight\n"
                          "A wandering mercenary with a cursed blade.\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            doc = engine.documents[0]
            self.assertEqual(doc["aliases"], ["Ser Aldric", "The Dark Knight"])
            # Header lines are stripped from the dosed content.
            self.assertNotIn("Aliases:", doc["content"])
            # Search finds the page via its explicit alias.
            found = engine.search_one("The Dark Knight")
            self.assertTrue(found and found["key"] == "zzz")

    def test_no_header_backwards_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._kb(tmp, "Ser Aldric Vane - a wandering mercenary.\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            doc = engine.documents[0]
            self.assertEqual(doc["aliases"], [])
            self.assertIn("Ser Aldric Vane", doc["content"])

    def test_alias_collision_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "characters")
            os.makedirs(cat)
            for key in ("a", "b"):
                with open(os.path.join(cat, f"{key}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Aliases: The Dark Knight\nPage {key} body.\n")
            report = lint_kb(tmp, lang=EN)
            self.assertTrue(any("The Dark Knight" in e and "both" in e
                                for e in report.errors), report.errors)


class TestIndexEquivalence(unittest.TestCase):
    """The prefix index must never change what the full scan would return."""

    def test_indexed_results_match_full_scan(self):
        engine = LoreEngine(EXAMPLE_KB, lang=EN, verbose=False)
        queries = ["Ser Aldric Vane", "dragons", "tavern rumors",
                   "Old Tavern", "keeper", "zzz qqq xxx", ""]
        for q in queries:
            indexed = engine.search_one(q)
            engine._pref = {}  # force full scan
            engine._short_key = {}
            scanned = engine.search_one(q)
            engine._build_prefix_index()
            self.assertEqual(indexed and indexed["key"],
                             scanned and scanned["key"], f"query={q!r}")
            indexed_rel = [d["key"] for d in engine.relevant(q, k=3)]
            engine._pref = {}
            engine._short_key = {}
            scanned_rel = [d["key"] for d in engine.relevant(q, k=3)]
            engine._build_prefix_index()
            self.assertEqual(indexed_rel, scanned_rel, f"query={q!r}")


class TestLint(unittest.TestCase):
    def test_example_kb_lints_clean(self):
        report = lint_kb(EXAMPLE_KB, lang=EN)
        self.assertEqual(report.errors, [], report.errors)

    def test_selftest_catches_broken_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "characters")
            os.makedirs(cat)
            # Page key unrelated to the canonical name...
            with open(os.path.join(cat, "zzz.txt"), "w", encoding="utf-8") as f:
                f.write("Bartholomew Crane - a scribe.\n")
            # ...while another page's KEY matches the name: search prefers
            # the key match, so the self-test must flag zzz.
            with open(os.path.join(cat, "bartholomew_crane.txt"), "w", encoding="utf-8") as f:
                f.write("A mysterious page.\nThe name Bartholomew Crane appears here.\n")
            report = lint_kb(tmp, lang=EN)
            self.assertTrue(any("Bartholomew Crane" in e for e in report.errors),
                            report.errors)

    def test_unknown_mention_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "lore")
            os.makedirs(cat)
            with open(os.path.join(cat, "rumors.txt"), "w", encoding="utf-8") as f:
                f.write("Rumors of the valley.\nThey whisper about Cornelius Blackwood.\n")
            report = lint_kb(tmp, lang=EN)
            self.assertTrue(any("Cornelius Blackwood" in w for w in report.warnings),
                            report.warnings)

    def test_extra_names_downgrades_to_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "lore")
            os.makedirs(cat)
            with open(os.path.join(cat, "rumors.txt"), "w", encoding="utf-8") as f:
                f.write("Rumors of the valley.\nThey whisper about Cornelius Blackwood.\n")
            report = lint_kb(tmp, lang=EN, extra_names={"Cornelius Blackwood"})
            self.assertEqual(report.warnings, [], report.warnings)
            self.assertTrue(any("Cornelius Blackwood" in i for i in report.infos))

    def test_allowlist_silences(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "lore")
            os.makedirs(cat)
            with open(os.path.join(cat, "rumors.txt"), "w", encoding="utf-8") as f:
                f.write("Rumors of the valley.\nThey whisper about Cornelius Blackwood.\n")
            allow = os.path.join(tmp, ".lint_allowlist.txt")
            with open(allow, "w", encoding="utf-8") as f:
                f.write("Cornelius Blackwood  # planned DLC\n")
            report = lint_kb(tmp, lang=EN, allowlist_path=allow)
            self.assertEqual(report.warnings, [], report.warnings)

    def test_oversized_page_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "lore")
            os.makedirs(cat)
            with open(os.path.join(cat, "big.txt"), "w", encoding="utf-8") as f:
                f.write("A big page.\n" + "filler " * 1000)
            report = lint_kb(tmp, lang=EN, doc_warn_chars=1000)
            self.assertTrue(any("oversized" in w for w in report.warnings))



class TestReviewFixes(unittest.TestCase):
    """Regressions from the external (Gemini) code review."""

    def test_normalize_hyphen_splits(self):
        self.assertEqual(normalize("Jean-Luc"), "jean luc")

    def test_hyphenated_alias_searchable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "characters")
            os.makedirs(cat)
            with open(os.path.join(cat, "captain.txt"), "w", encoding="utf-8") as f:
                f.write("Aliases: Jean-Luc Picard\nA starship captain.\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            found = engine.search_one("Jean Luc")
            self.assertTrue(found and found["key"] == "captain",
                            f"got {found}")

    def test_crlf_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "lore")
            os.makedirs(cat)
            with open(os.path.join(cat, "page.txt"), "w",
                      encoding="utf-8", newline="") as f:
                f.write("Aliases: Foo\r\nLine one.\r\nLine two.\r\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            doc = engine.documents[0]
            self.assertEqual(doc["aliases"], ["Foo"])
            self.assertNotIn("\r", doc["content"])
            self.assertIn("Line one.\nLine two.", doc["content"])

    def test_reload_missing_dir_clears_caches(self):
        engine = LoreEngine(EXAMPLE_KB, lang=EN, verbose=False)
        self.assertTrue(engine.index())
        self.assertTrue(engine.full_text())
        engine.kb_dir = "/nonexistent/kb/path"
        engine.load()
        self.assertEqual(engine.documents, [])
        self.assertEqual(engine.index(), "")
        self.assertEqual(engine.full_text(), "")

    def test_header_only_page_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "characters")
            os.makedirs(cat)
            with open(os.path.join(cat, "stub.txt"), "w", encoding="utf-8") as f:
                f.write("Aliases: Someone\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            self.assertEqual(engine.documents, [])

    def test_public_docs_have_no_private_fields(self):
        engine = LoreEngine(EXAMPLE_KB, lang=EN, verbose=False)
        for doc in engine.documents:
            self.assertNotIn("_words", doc)
            self.assertNotIn("_key_words", doc)
        found = engine.search_one("Ser Aldric Vane")
        self.assertNotIn("_words", found)

    def test_repr(self):
        engine = LoreEngine(EXAMPLE_KB, lang=EN, verbose=False)
        self.assertIn("documents=4", repr(engine))


class TestSemantic(unittest.TestCase):
    """Co-occurrence (distributional) semantic expansion in relevant()."""

    def _kb(self, tmp):
        cat = os.path.join(tmp, "locations")
        os.makedirs(cat, exist_ok=True)
        # Page A defines the co-occurrence context.
        with open(os.path.join(cat, "forge.txt"), "w", encoding="utf-8") as f:
            f.write("The forge burns hot. Smiths hammer steel into blades.\n")
        # Page B shares context but none of the query words.
        with open(os.path.join(cat, "guild_hall.txt"), "w", encoding="utf-8") as f:
            f.write("The guild hall stands near the forge. Apprentices learn the trade.\n")

    def test_expansion_reaches_zero_overlap_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._kb(tmp)
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            keys = [d["key"] for d in engine.relevant("hammer and steel", k=2)]
            self.assertIn("forge", keys)
            self.assertIn("guild_hall", keys)

    def test_expansion_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._kb(tmp)
            a = LoreEngine(tmp, lang=EN, verbose=False)
            b = LoreEngine(tmp, lang=EN, verbose=False)
            self.assertEqual([d["key"] for d in a.relevant("hammer and steel", k=2)],
                             [d["key"] for d in b.relevant("hammer and steel", k=2)])

    def test_expansion_disabled_without_cooc(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._kb(tmp)
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            engine._cooc = {}
            keys = [d["key"] for d in engine.relevant("hammer and steel", k=2)]
            self.assertEqual(keys, ["forge"])

    def test_tags_are_searchable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "locations")
            os.makedirs(cat)
            with open(os.path.join(cat, "old_tavern.txt"), "w", encoding="utf-8") as f:
                f.write("Tags: beer, gossip\nAn inn.\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            found = engine.search_one("beer")
            self.assertTrue(found and found["key"] == "old_tavern")


class TestScale(unittest.TestCase):
    """Navigation and search behavior on larger knowledge bases."""

    def test_category_summary(self):
        engine = LoreEngine(EXAMPLE_KB, lang=EN, verbose=False)
        summary = engine.category_summary()
        self.assertIn("characters (2 pages)", summary)
        self.assertIn("locations (1 pages)", summary)
        self.assertIn("lore (1 pages)", summary)
        self.assertIn("dark_knight", summary)

    def test_category_summary_cached(self):
        engine = LoreEngine(EXAMPLE_KB, lang=EN, verbose=False)
        self.assertIs(engine.category_summary(), engine.category_summary())

    def test_size_stats_reports_cooc(self):
        engine = LoreEngine(EXAMPLE_KB, lang=EN, verbose=False)
        stats = engine.size_stats()
        self.assertEqual(stats["categories"], 3)
        self.assertGreater(stats["cooc_words"], 0)

    def test_search_scales_sublinear(self):
        # Build a 1200-doc KB; entity lookups must stay fast via the index.
        import random, string, time
        random.seed(1)
        def rw():
            return "".join(random.choice(string.ascii_lowercase)
                           for _ in range(random.randint(4, 8)))
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            names = [rw() + "_" + rw() for _ in range(1200)]
            for key in names:
                body = key.replace("_", " ").title() + " " + " ".join(rw() for _ in range(20))
                with open(os.path.join(tmp, "lore", key + ".txt"), "w", encoding="utf-8") as f:
                    f.write(body + "\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            self.assertEqual(len(engine.documents), 1200)
            q = names[600].replace("_", " ")
            t0 = time.perf_counter()
            for _ in range(50):
                engine.search_one(q)
            per_q = (time.perf_counter() - t0) / 50
            self.assertEqual(engine.search_one(q)["key"], names[600])
            self.assertLess(per_q, 0.05, f"indexed lookup too slow: {per_q*1000:.2f} ms")

    def test_body_pref_scoring_matches_full_scan(self):
        # The per-doc 3-char-prefix bucket in _score_doc must be provably
        # identical to scanning every body word: indexed search_one winners
        # must equal brute-force full-scan winners on random corpora.
        import random, string
        random.seed(11)
        def rw(alpha):
            return "".join(random.choice(alpha)
                           for _ in range(random.randint(3, 8)))
        for trial in range(12):
            cyr = trial % 2 == 0
            alpha = "абвгдежзиклмнопрстуфхцчшщюя" if cyr else string.ascii_lowercase
            vocab = [rw(alpha) for _ in range(random.randint(15, 40))]
            with tempfile.TemporaryDirectory() as tmp:
                for c in ("characters", "locations"):
                    os.makedirs(os.path.join(tmp, c))
                for i in range(random.randint(8, 40)):
                    key = rw(alpha) + "_" + rw(alpha)
                    body = key.replace("_", " ").title() + " " + " ".join(
                        random.choice(vocab) for _ in range(random.randint(4, 15)))
                    cat = "characters" if i % 2 else "locations"
                    with open(os.path.join(tmp, cat, key + ".txt"), "w",
                              encoding="utf-8") as f:
                        f.write(body + "\n")
                lang = UK if cyr else EN
                eng = LoreEngine(tmp, lang=lang, verbose=False)
                queries = [random.choice(vocab) for _ in range(10)]
                queries += [" ".join(random.sample(vocab, 2)) for _ in range(8)]
                for q in queries:
                    got = eng.search_one(q)
                    words = eng._query_words(q)
                    best_i, best_s = -1, 0.0
                    for i in range(len(eng.documents)):
                        s = eng._score_doc(i, words)
                        if s > best_s:
                            best_i, best_s = i, s
                    full = eng.documents[best_i] if best_i >= 0 and best_s >= eng._min_score() else None
                    self.assertEqual(got and got["key"], full and full["key"],
                                     f"trial={trial} q={q!r}")

    def test_single_doc_kb_body_match_accepted(self):
        # On a one-page KB the max IDF is log(2) < 1.0, so the nominal
        # acceptance floor would reject a genuine body-word match. The
        # threshold must be floored at log(1+n) so tiny KBs still work.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            with open(os.path.join(tmp, "lore", "solo.txt"), "w",
                      encoding="utf-8") as f:
                f.write("A single lonely page about ravens.\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            doc = engine.search_one("ravens")
            self.assertTrue(doc and doc["key"] == "solo")
            self.assertEqual([d["key"] for d in engine.relevant("ravens", k=2)],
                             ["solo"])

    def test_relevant_negative_k_returns_empty(self):
        # k <= 0 must yield an empty list, not a negative slice (which would
        # silently return all-but-last results).
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            with open(os.path.join(tmp, "lore", "a.txt"), "w",
                      encoding="utf-8") as f:
                f.write("Ravens fly at dusk over the moor.\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            self.assertEqual(engine.relevant("ravens", k=-1), [])
            self.assertEqual(engine.relevant("ravens", k=0), [])

    def test_relevant_cap_keeps_keyword_matches(self):
        # Even when the semantic-tier candidate cap is tiny, pages matching
        # the ORIGINAL query words (tier 1) must never be dropped.
        import loreengine.engine as eng_mod
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            # 30 filler docs sharing a common word, plus the target doc.
            for i in range(30):
                with open(os.path.join(tmp, "lore", f"filler_{i}.txt"), "w",
                          encoding="utf-8") as f:
                    f.write(f"Common meadow grass grows here. Entry {i}.\n")
            with open(os.path.join(tmp, "lore", "target.txt"), "w",
                      encoding="utf-8") as f:
                f.write("The obsidian obelisk hums with ancient power.\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            old_cap = eng_mod._RELEVANT_MAX_CANDIDATES
            try:
                eng_mod._RELEVANT_MAX_CANDIDATES = 2
                docs = engine.relevant("obsidian obelisk", k=2)
                self.assertIn("target", [d["key"] for d in docs])
            finally:
                eng_mod._RELEVANT_MAX_CANDIDATES = old_cap

    def test_short_translit_narrows_not_full_scan(self):
        # A query word whose transliteration collapses below 3 chars (UK soft
        # sign 'ь' -> '') must still narrow via the _short_key index instead
        # of forcing a full-corpus scan, and must return the right page.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            # A page whose key normalizes to a 1-char word "a" (via 'аьь').
            with open(os.path.join(tmp, "lore", "a.txt"), "w",
                      encoding="utf-8") as f:
                f.write("A mysterious single-letter page.\n")
            with open(os.path.join(tmp, "lore", "other.txt"), "w",
                      encoding="utf-8") as f:
                f.write("An unrelated page about ravens and moors.\n")
            engine = LoreEngine(tmp, lang=UK, verbose=False)
            # 'аьь' translits to 'a' (1 char) — must exact-match key word 'a'.
            doc = engine.search_one("аьь")
            self.assertTrue(doc and doc["key"] == "a", doc)
            # And it must NOT have fallen back to a full scan: candidates
            # should be a strict subset, not the whole corpus.
            cand = engine._candidates(["аьь"])
            self.assertLess(len(list(cand)), len(engine.documents))

    def test_reload_between_nonempty_kbs(self):
        # Reloading from one non-empty KB to another must fully replace state
        # (no stale docs/words/indexes from the first KB).
        with tempfile.TemporaryDirectory() as tmp1, \
             tempfile.TemporaryDirectory() as tmp2:
            os.makedirs(os.path.join(tmp1, "lore"))
            os.makedirs(os.path.join(tmp2, "lore"))
            with open(os.path.join(tmp1, "lore", "first.txt"), "w",
                      encoding="utf-8") as f:
                f.write("The first knowledge base page about dragons.\n")
            with open(os.path.join(tmp2, "lore", "second.txt"), "w",
                      encoding="utf-8") as f:
                f.write("The second knowledge base page about ravens.\n")
            engine = LoreEngine(tmp1, lang=EN, verbose=False)
            self.assertEqual({d["key"] for d in engine.documents}, {"first"})
            engine.kb_dir = tmp2
            engine.load()
            self.assertEqual({d["key"] for d in engine.documents}, {"second"})
            self.assertEqual(len(engine._words), 1)
            self.assertEqual(len(engine._key_words), 1)
            self.assertEqual(len(engine._body_pref), 1)
            self.assertIsNone(engine.search_one("dragons"))
            self.assertEqual(engine.search_one("ravens")["key"], "second")

    def test_cooc_prune_bounds_neighbors(self):
        # Force the incremental prune trigger: one hub word co-occurring with
        # >_COOC_PRUNE_TRIGGER distinct words must end with <= _COOC_NEIGHBORS
        # neighbors, deterministically.
        import loreengine.engine as eng_mod
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            # Hub word + many distinct companions across many docs.
            n = eng_mod._COOC_PRUNE_TRIGGER + 40
            for i in range(n):
                with open(os.path.join(tmp, "lore", f"doc_{i}.txt"), "w",
                          encoding="utf-8") as f:
                    f.write(f"hubword companion{i:04d} unique{i:04d} text.\n")
            engine = LoreEngine(tmp, lang=EN, verbose=False)
            nbs = engine._cooc.get("hubword", {})
            self.assertLessEqual(len(nbs), eng_mod._COOC_NEIGHBORS)
            # Determinism: rebuild yields identical neighbor maps.
            engine2 = LoreEngine(tmp, lang=EN, verbose=False)
            self.assertEqual(engine._cooc, engine2._cooc)


class TestTokenize(unittest.TestCase):
    """Punctuation-split chunks with a short fragment glue back together."""

    def test_short_fragment_glues(self):
        # "90-ті" must not vanish as two sub-3-char fragments.
        self.assertEqual(tokenize("90-ті"), ["90ті"])
        self.assertEqual(tokenize("35-ти"), ["35ти"])
        self.assertEqual(tokenize("сім'ю"), ["сімю"])

    def test_long_fragments_stay_split(self):
        self.assertEqual(tokenize("Jean-Luc"), ["jean", "luc"])
        self.assertEqual(tokenize("студентка-практикантка"),
                         ["студентка", "практикантка"])

    def test_glued_token_reaches_latin_key(self):
        # The NoireV2 regression: query "90-ті" must find key lyhi_90ti
        # through the transliteration bridge (90ті -> 90ti).
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            with open(os.path.join(tmp, "lore", "lyhi_90ti.txt"), "w",
                      encoding="utf-8") as f:
                f.write("Легенди 90-х — бандити, бартер і беззаконня.\n")
            engine = LoreEngine(tmp, lang=UK, verbose=False)
            doc = engine.search_one("Слухаю плітки про 90-ті")
            self.assertTrue(doc and doc["key"] == "lyhi_90ti", f"got {doc}")

    def test_glued_body_word_searchable(self):
        # A glued token in the body is indexed and findable by the same
        # glued query form.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            with open(os.path.join(tmp, "lore", "era.txt"), "w",
                      encoding="utf-8") as f:
                f.write("Це була епоха 90-тих років.\n")
            engine = LoreEngine(tmp, lang=UK, verbose=False)
            doc = engine.search_one("90-тих")
            self.assertTrue(doc and doc["key"] == "era", f"got {doc}")


class TestReviewFixesRound2(unittest.TestCase):
    """Regressions from the second review round (robustness fixes)."""

    def test_non_utf8_page_skipped(self):
        # One bad-bytes .txt must not crash load() — it is skipped like
        # any other unreadable file.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            with open(os.path.join(tmp, "lore", "good.txt"), "w",
                      encoding="utf-8") as f:
                f.write("Це хороша сторінка про старий млин.\n")
            with open(os.path.join(tmp, "lore", "bad.txt"), "wb") as f:
                f.write(b"\xff\xfe bad bytes \x80\x81 not utf-8\n")
            engine = LoreEngine(tmp, lang=UK, verbose=False)
            self.assertEqual([d["key"] for d in engine.documents], ["good"])
            doc = engine.search_one("млин")
            self.assertTrue(doc and doc["key"] == "good", f"got {doc}")

    def test_apostrophe_spellings_agree(self):
        # U+02BC MODIFIER LETTER APOSTROPHE is a \w char; all three
        # spellings of «сімʼю» must tokenize identically and match.
        spellings = ["сімʼю", "сім'ю", "сім’ю"]
        self.assertEqual(tokenize(" ".join(spellings)), ["сімю"] * 3)
        self.assertEqual(normalize("сімʼю"), normalize("сім'ю"))
        for a in spellings:
            for b in spellings:
                self.assertTrue(words_match(a, b), f"{a!r} ~ {b!r}")

    def test_apostrophe_spellings_resolve_same_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lore"))
            with open(os.path.join(tmp, "lore", "family.txt"), "w",
                      encoding="utf-8") as f:
                f.write("Він обійняв свою сімʼю і заплакав.\n")
            engine = LoreEngine(tmp, lang=UK, verbose=False)
            for q in ("сімʼю", "сім'ю", "сім’ю"):
                doc = engine.search_one(q)
                self.assertTrue(doc and doc["key"] == "family",
                                f"q={q!r} got {doc}")
            docs = engine.pages_for_entities(["сім'ю"])
            self.assertTrue(docs and docs[0]["key"] == "family")

    def test_max_pages_zero_or_negative_returns_empty(self):
        engine = LoreEngine(EXAMPLE_KB, lang=EN, verbose=False)
        self.assertEqual(engine.get_pages(["dragon"], max_pages=0), [])
        self.assertEqual(engine.get_pages(["dragon"], max_pages=-1), [])
        self.assertEqual(
            engine.pages_for_entities(["dragon"], max_pages=0), [])
        self.assertEqual(
            engine.pages_for_entities(["dragon"], max_pages=-2), [])
        # A positive cap still yields exactly that many pages.
        docs = engine.get_pages(["dragon", "Aldric", "Old Tavern"],
                                max_pages=2)
        self.assertEqual(len(docs), 2)

    def test_alias_dedupe_keeps_distinct_multiword(self):
        # «Таверна на Сході» is NOT a case-form of «Таверна»; «Таверни» is.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "locations"))
            with open(os.path.join(tmp, "locations", "old_tavern.txt"), "w",
                      encoding="utf-8") as f:
                f.write("Old Tavern («Таверна», «Таверни», "
                        "«Таверна на Сході») - шумне місце.\n")
            engine = LoreEngine(tmp, lang=UK, verbose=False)
            _names, kb_aliases = build_kb_registry(engine)
            self.assertIn("Таверна", kb_aliases)
            self.assertIn("Таверна на Сході", kb_aliases)
            self.assertNotIn("Таверни", kb_aliases)

    def test_short_name_self_test_is_info_not_error(self):
        # search_one drops <3-char query words by design, so a canonical
        # name like "Yo" cannot resolve — the linter notes it, not fails.
        with tempfile.TemporaryDirectory() as tmp:
            cat = os.path.join(tmp, "characters")
            os.makedirs(cat)
            with open(os.path.join(cat, "yo.txt"), "w",
                      encoding="utf-8") as f:
                f.write("Yo - a man of few letters.\n")
            report = lint_kb(tmp, lang=EN)
            self.assertFalse(report.errors, report.errors)
            self.assertTrue(any("Yo" in i and "skipped" in i
                                for i in report.infos), report.infos)


if __name__ == "__main__":
    unittest.main()
