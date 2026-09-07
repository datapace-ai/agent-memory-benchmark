"""Compare model tracks: python -m membench.compare results/tracks/*/summary.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from membench.config import REPO_ROOT
from membench.report import compare_tracks


def main(argv: list[str] | None = None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        paths = sorted((REPO_ROOT / "results" / "tracks").glob("*/summary.json"))
    named = {p.parent.name: json.loads(p.read_text()) for p in paths}
    text = compare_tracks(named)
    out = REPO_ROOT / "results" / "tracks" / "compare.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
