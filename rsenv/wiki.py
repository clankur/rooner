"""Bounded, offline search over a local rs-sdk wiki and learnings snapshot."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WIKI_PATH = Path(__file__).parent / "bridge" / "rs-sdk"
MAX_RESULTS = 3
PASSAGE_CHARS = 1200
_STOP_WORDS = frozenset("a an and are at be can do for from how i in is it of on or the to what where with".split())


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if term not in _STOP_WORDS]


@dataclass(frozen=True)
class Passage:
    source: str
    title: str
    text: str
    terms: Counter[str]


class WikiIndex:
    def __init__(self, root: Path) -> None:
        self.passages: list[Passage] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for directory in ("wiki", "learnings"):
            for path in sorted((root / directory).rglob("*.md")):
                try:
                    content = path.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeError) as exc:
                    logger.warning("Cannot read wiki source %s: %s", path, exc)
                    continue
                if not content:
                    continue
                source = path.relative_to(root).as_posix()
                title = next(
                    (line.lstrip("# ") for line in content.splitlines() if line.startswith("# ")),
                    path.stem.replace("-", " "),
                )
                # Overlap preserves recipes and table rows near a passage boundary.
                for start in range(0, len(content), PASSAGE_CHARS - 200):
                    excerpt = content[start : start + PASSAGE_CHARS]
                    terms = Counter(_terms(excerpt))
                    terms.update(_terms(f"{title} {title} {path.stem}"))
                    passage_id = len(self.passages)
                    self.passages.append(Passage(source, title, excerpt, terms))
                    for term in terms:
                        self.postings[term].append(passage_id)
                    if start + PASSAGE_CHARS >= len(content):
                        break
        self.average_length = sum(sum(p.terms.values()) for p in self.passages) / max(len(self.passages), 1)
        if not self.passages:
            logger.warning("No wiki/learnings Markdown found under %s", root)

    def search(self, query: str) -> str:
        if not self.passages:
            return json.dumps(
                {
                    "results": [],
                    "error": "Wiki unavailable: populate rsenv/bridge/rs-sdk with bridge/setup.sh, "
                    "or set wiki_data_path to an rs-sdk checkout containing wiki/ and learnings/.",
                }
            )

        scores: dict[int, float] = defaultdict(float)
        for term in sorted(set(_terms(query))):
            matches = self.postings.get(term, [])
            idf = math.log(1 + (len(self.passages) - len(matches) + 0.5) / (len(matches) + 0.5))
            for passage_id in matches:
                passage = self.passages[passage_id]
                frequency = passage.terms[term]
                length = sum(passage.terms.values())
                scores[passage_id] += (
                    idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / self.average_length))
                )

        results: list[dict[str, str]] = []
        seen_sources: set[str] = set()
        for passage_id in sorted(scores, key=lambda idx: (-scores[idx], idx)):
            passage = self.passages[passage_id]
            if passage.source in seen_sources:
                continue
            seen_sources.add(passage.source)
            results.append({"source": passage.source, "title": passage.title, "excerpt": passage.text})
            if len(results) == MAX_RESULTS:
                break
        return json.dumps({"results": results}, ensure_ascii=False)


@lru_cache(maxsize=4)
def get_wiki_index(wiki_data_path: str = "") -> WikiIndex:
    """Keep the corpus fixed within a process; restart after updating the checkout."""
    root = Path(wiki_data_path).expanduser() if wiki_data_path else DEFAULT_WIKI_PATH
    return WikiIndex(root)
