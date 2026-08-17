"""CLI: ``python -m loreengine <kb_dir> [--lang uk|en|pl] [--strict]``.

Lints a knowledge base and prints the report. With ``--strict``, warnings
count as errors (non-zero exit).
"""

import argparse
import sys
from typing import List, Optional

from .lang import PACKS, UK
from .lint import lint_kb


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loreengine",
        description="Lint a LoreEngine knowledge base.")
    parser.add_argument("kb_dir", help="path to the knowledge base root")
    parser.add_argument("--lang", default="uk", choices=sorted(PACKS),
                        help="language pack for mention detection (default: uk)")
    parser.add_argument("--allowlist", default=None,
                        help="path to .lint_allowlist.txt")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as errors")
    args = parser.parse_args(argv)

    lang = PACKS.get(args.lang, UK)
    report = lint_kb(args.kb_dir, lang=lang, allowlist_path=args.allowlist)
    report.print()

    errors = len(report.errors) + (len(report.warnings) if args.strict else 0)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
