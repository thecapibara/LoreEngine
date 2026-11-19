"""
Language packs for LoreEngine.

A LanguagePack carries everything that is player-language-specific:

- ``stopwords`` — query words that carry no search meaning;
- ``translit`` — mapping from the player's alphabet to latin, used as a
  bridge between queries (e.g. Cyrillic «Дарк Найт») and document file keys
  (e.g. ``dark_knight.txt``). Empty mapping = no transliteration
  (fine when keys and queries share an alphabet);
- linter settings — regexes for proper-name candidates and noise filters
  used by :mod:`loreengine.lint`.

Built-in packs: :data:`UK` (Ukrainian), :data:`EN` (English), :data:`PL`
(Polish). Roll your own by constructing ``LanguagePack`` — nothing else
in the engine is language-specific.
"""

from dataclasses import dataclass, field
import types
from typing import FrozenSet, Mapping


def _name_regex(upper: str, word: str) -> str:
    """Build proper-name candidate regex (2-3 capitalized words, same line)."""
    return rf"\b[{upper}][{word}]+(?:[ \t]+[{upper}][{word}]+){{1,2}}\b"


def _quoted_regex(upper: str, word: str) -> str:
    """Build quoted name/alias regex (1-4 words, matched quote pairs, no newlines/sentences)."""
    inner = (
        rf"[{upper}]"
        rf"(?:[{word}]{{0,20}}(?:[ \t]+[{word}]{{1,20}}){{1,3}}|[{word}]{{2,30}})"
    )
    return rf'"({inner})"|“({inner})”|«({inner})»|„({inner})”'


_EN_UPPER = r"A-ZÀ-ÖØ-Þ"
_EN_WORD = r"a-zà-öø-ÿA-ZÀ-ÖØ-Þ'\u2019\-"
_UK_UPPER = r"А-ЯІЇЄҐ"
_UK_WORD = r"а-яіїєґА-ЯІЇЄҐ'\u02bc\u2019\-"
_PL_UPPER = r"A-ZĄĆĘŁŃÓŚŹŻ"
_PL_WORD = r"a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ'\u2019\-"


@dataclass(frozen=True)
class LanguagePack:
    """Player-language settings. All fields optional."""

    #: Query words ignored by search (no search meaning).
    stopwords: FrozenSet[str] = frozenset()

    #: Player-alphabet char -> latin string bridge for matching file keys.
    translit: Mapping[str, str] = field(default_factory=dict)

    #: Linter: regex finding proper-name candidates (2-3 capitalized words).
    name_regex: str = _name_regex(_EN_UPPER, _EN_WORD)

    #: Linter: regex finding quoted names/nicknames ("Doc", «Doc»).
    quoted_regex: str = _quoted_regex(_EN_UPPER, _EN_WORD)

    #: Linter: words that start sentences but are not names (noise filter).
    sentence_starters: FrozenSet[str] = frozenset()

    #: Linter: leading service words stripped from a candidate
    #: ("For Marcus" -> "Marcus").
    leading_stop: FrozenSet[str] = frozenset()

    #: Linter: intro words stripped from a character page header to get the
    #: canonical name ("... better known as X" -> cut before it).
    alias_intro_regex: str = r"\bbetter known as\b|\bknown as\b|\bnamed\b"

    def __post_init__(self) -> None:
        # Freeze translit mapping so frozen LanguagePack instances cannot be mutated.
        object.__setattr__(
            self, "translit", types.MappingProxyType(dict(self.translit))
        )


# ---------------------------------------------------------------------------
# Ukrainian
# ---------------------------------------------------------------------------
UK = LanguagePack(
    stopwords=frozenset({
        "про", "що", "хто", "така", "такий", "таке", "який", "яка", "яке", "які",
        "відомо", "розкажи", "більше", "чому", "де", "коли", "був", "була", "було",
        "це", "цей", "ця", "ці", "вона", "він", "вони", "його", "її", "їх", "мені",
        "тобі", "ним", "ній", "них", "тут", "там", "тоді", "тепер", "все", "всі",
        "вся", "можеш", "скажи", "покажи", "детально", "інформація", "справа",
        "справи", "справі", "місце", "місці", "місця", "щось", "небудь", "дуже",
    }),
    translit={
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'h', 'ґ': 'g', 'д': 'd', 'е': 'e',
        'є': 'ye', 'ж': 'zh', 'з': 'z', 'и': 'y', 'і': 'i', 'ї': 'i', 'й': 'y',
        'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
        'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch',
        'ш': 'sh', 'щ': 'shch', 'ь': '', 'ю': 'yu', 'я': 'ya',
    },
    name_regex=_name_regex(_UK_UPPER, _UK_WORD),
    quoted_regex=_quoted_regex(_UK_UPPER, _UK_WORD),
    sentence_starters=frozenset({
        "він", "вона", "воно", "вони", "це", "цей", "ця", "ці", "той", "та", "те",
        "справа", "справи", "справі", "ніч", "ночі", "ранок", "ранку", "день",
        "дня", "вечір", "вечора", "кімната", "кімнаті", "кабінет", "кабінеті",
        "машина", "машини", "двері", "дверей", "телефон", "телефону", "документ",
        "документи",
        "один", "одна", "одне", "двоє", "троє", "кілька", "багато", "майже",
        "саме", "навіть", "одного", "якось", "колись", "тоді", "тепер", "згодом",
        "за", "зі", "з", "від", "на", "у", "в", "до", "по", "при",
        "але", "проте", "однак", "також", "тому", "бо", "щоб", "як", "якби",
        "раптом", "нарешті", "зрештою", "загалом", "зазвичай", "кажуть",
        "ходять", "ходить", "відомо", "невідомо", "точно", "мабуть", "можливо",
    }),
    leading_stop=frozenset({
        "для", "хоча", "зовні", "сьогодні", "саме", "навіть", "раптом", "нарешті",
        "зрештою", "одного", "якось", "колись", "тоді", "тепер", "згодом", "але",
        "проте", "однак", "також", "тому", "щоб", "кажуть", "ходять", "відомо",
        "мабуть", "можливо", "точно", "нехай", "щоправда", "втім", "отже", "тож",
        "більш", "менш", "переважно", "здебільшого", "часто", "іноді", "завжди",
    }),
    alias_intro_regex=r"\bбільш відомий\b|\bвідомий як\b|\bна ім'я\b",
)

# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------
EN = LanguagePack(
    stopwords=frozenset({
        "about", "who", "what", "which", "where", "when", "why", "how", "the",
        "this", "that", "these", "those", "there", "here", "was", "were", "been",
        "have", "has", "had", "his", "her", "their", "its", "him", "them", "she",
        "he", "they", "you", "your", "mine", "tell", "show", "more", "know",
        "some", "any", "very", "much", "many", "place", "thing", "something",
        "anything", "please", "could", "would", "should", "will", "can", "are",
        "not", "and", "for", "with", "from", "into", "then", "now", "all",
    }),
    translit={},
    name_regex=_name_regex(_EN_UPPER, _EN_WORD),
    quoted_regex=_quoted_regex(_EN_UPPER, _EN_WORD),
    sentence_starters=frozenset({
        "the", "a", "an", "this", "that", "these", "those", "it", "its", "he",
        "she", "they", "there", "here", "city", "town", "village", "house",
        "street", "square", "club", "building", "room", "office", "bar", "shop",
        "market", "station", "park", "bridge", "river", "night", "morning",
        "evening", "day", "door", "phone", "letter", "paper", "archive", "court",
        "school", "hospital", "church", "factory", "hotel", "restaurant", "old",
        "new", "big", "small", "one", "two", "three", "several", "many", "most",
        "some", "almost", "even", "once", "suddenly", "finally", "then", "now",
        "later", "after", "before", "but", "however", "also", "because", "when",
        "while", "if", "although", "perhaps", "maybe",
    }),
    leading_stop=frozenset({
        "in", "on", "at", "the", "a", "an",
        "for", "although", "today", "even", "suddenly", "finally", "once",
        "afterwards", "then", "now", "later", "but", "however", "also",
        "because", "so", "thus", "perhaps", "maybe", "probably", "certainly",
        "mostly", "often", "sometimes", "always", "never",
    }),
    alias_intro_regex=r"\bbetter known as\b|\bknown as\b|\bnamed\b",
)

# ---------------------------------------------------------------------------
# Polish
# ---------------------------------------------------------------------------
PL = LanguagePack(
    stopwords=frozenset({
        "o", "co", "kto", "jaki", "jaka", "jakie", "który", "która", "gdzie",
        "kiedy", "dlaczego", "jak", "był", "była", "było", "to", "ten", "ta",
        "te", "on", "ona", "oni", "jego", "jej", "ich", "mu", "im", "tu", "tam",
        "wtedy", "teraz", "wszystko", "wszyscy", "możesz", "powiedz", "pokaż",
        "więcej", "informacja", "sprawa", "sprawy", "miejsce", "coś", "bardzo",
    }),
    translit={
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o',
        'ś': 's', 'ź': 'z', 'ż': 'z',
    },
    name_regex=_name_regex(_PL_UPPER, _PL_WORD),
    quoted_regex=_quoted_regex(_PL_UPPER, _PL_WORD),
    sentence_starters=frozenset({
        "on", "ona", "ono", "oni", "to", "ten", "ta", "te", "miasto", "miasta",
        "klub", "dom", "domu", "ulica", "ulicy", "plac", "uniwersytet", "komisariat",
        "sprawa", "sprawy", "noc", "nocy", "rano", "dzień", "dnia", "wieczór",
        "pokój", "pokoju", "gabinet", "bar", "rynek", "dworzec", "park", "most",
        "samochód", "drzwi", "telefon", "dokument", "archiwum", "sąd", "szkoła",
        "szpital", "kościół", "fabryka", "hotel", "restauracja", "sklep", "kiosk",
        "stary", "stara", "nowy", "nowa", "duży", "duża", "jeden", "jedna",
        "kilka", "wiele", "prawie", "właśnie", "nawet", "kiedyś", "wtedy",
        "teraz", "potem", "za", "ze", "z", "od", "na", "w", "do", "po", "przy",
        "ale", "jednak", "także", "dlatego", "żeby", "jak", "nagle", "wreszcie",
        "ogólnie", "zwykle", "mówią", "podobno", "może", "chyba",
    }),
    leading_stop=frozenset({
        "dla", "chociaż", "dzisiaj", "właśnie", "nawet", "nagle", "wreszcie",
        "kiedyś", "wtedy", "teraz", "potem", "ale", "jednak",
        "także", "dlatego", "żeby", "mówią", "podobno", "może", "chyba",
        "raczej", "głównie", "często", "czasem", "zawsze",
    }),
    alias_intro_regex=r"\bznany jako\b|\bzwany\b",
)

PACKS = {"uk": UK, "en": EN, "pl": PL}
