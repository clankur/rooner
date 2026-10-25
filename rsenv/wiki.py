"""Bounded, offline search over a local rs-sdk wiki and learnings snapshot."""

from __future__ import annotations

import json
import logging
import math
import re
from bisect import bisect_right
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


def _name(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


@dataclass(frozen=True)
class Excerpt:
    text: str
    lines: frozenset[int]


class Document:
    def __init__(self, content: str) -> None:
        self.lines = content.splitlines()
        self.offsets: list[int] = []
        self.contexts: list[list[int]] = []
        headings: list[int] = []
        table: list[int] = []
        offset = 0
        for i, line in enumerate(self.lines):
            self.offsets.append(offset)
            offset += len(line) + 1
            heading = re.match(r"^(#{1,6}) ", line)
            if heading:
                level = len(heading[1])
                headings = [h for h in headings if len(self.lines[h]) - len(self.lines[h].lstrip("#")) < level]
                table = []
            elif not line.startswith("|"):
                table = []
            elif i + 1 < len(self.lines) and re.fullmatch(r"\|[\s:|-]+\|", self.lines[i + 1]):
                table = [i, i + 1]
            self.contexts.append(headings + table)
            if heading:
                headings = headings + [i]

    def excerpt(self, start: int, end: int, excluded: frozenset[int] = frozenset()) -> Excerpt:
        first = max(0, bisect_right(self.offsets, start) - 1)
        last = bisect_right(self.offsets, end - 1)
        emitted: set[int] = set()
        body: set[int] = set()
        output: list[str] = []
        size = 0
        for i in range(first, last):
            if i in excluded or i in emitted:
                continue
            additions = list(dict.fromkeys(j for j in self.contexts[i] + [i] if j not in emitted))
            extra = sum(len(self.lines[j]) + 1 for j in additions)
            if size + extra > PASSAGE_CHARS:
                # Never truncate a table row or its headers to fit the budget.
                if body:
                    break
                continue
            output.extend(self.lines[j] for j in additions)
            emitted.update(additions)
            body.add(i)
            size += extra
        return Excerpt("\n".join(output).strip(), frozenset(emitted))


@dataclass(frozen=True)
class Passage:
    source: str
    title: str
    text: str
    terms: Counter[str]
    start: int
    end: int


class WikiIndex:
    def __init__(self, root: Path) -> None:
        self.passages: list[Passage] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.documents: dict[str, Document] = {}
        self.exact_names: dict[str, set[str]] = defaultdict(set)
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
                self.documents[source] = Document(content)
                for name in (title, path.stem):
                    self.exact_names[_name(name)].add(source)
                # Keep ranking windows stable; restore Markdown context only
                # when assembling results so chunking does not change BM25.
                for start in range(0, len(content), PASSAGE_CHARS - 200):
                    excerpt = content[start : start + PASSAGE_CHARS]
                    terms = Counter(_terms(excerpt))
                    terms.update(_terms(f"{title} {title} {path.stem}"))
                    passage_id = len(self.passages)
                    self.passages.append(Passage(source, title, excerpt, terms, start, start + len(excerpt)))
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
        query_terms = set(_terms(query))
        for term in sorted(query_terms):
            matches = self.postings.get(term, [])
            idf = math.log(1 + (len(self.passages) - len(matches) + 0.5) / (len(matches) + 0.5))
            for passage_id in matches:
                passage = self.passages[passage_id]
                frequency = passage.terms[term]
                length = sum(passage.terms.values())
                scores[passage_id] += (
                    idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / self.average_length))
                )

        exact_sources = self.exact_names.get(_name(query), set())
        ranked = sorted(
            scores,
            key=lambda idx: (
                self.passages[idx].source not in exact_sources,
                self.passages[idx].start if self.passages[idx].source in exact_sources else -scores[idx],
                idx,
            ),
        )
        by_source: dict[str, list[int]] = defaultdict(list)
        for idx in ranked:
            by_source[self.passages[idx].source].append(idx)

        results: list[dict[str, str]] = []
        seen_sources: set[str] = set()
        for passage_id in ranked:
            passage = self.passages[passage_id]
            if passage.source in seen_sources:
                continue
            seen_sources.add(passage.source)
            document = self.documents[passage.source]
            excerpt = document.excerpt(passage.start, passage.end)
            if not excerpt.text:
                continue
            results.append({"source": passage.source, "title": passage.title, "excerpt": excerpt.text})
            if len(results) == MAX_RESULTS:
                break

            # Reserve at most one additional slot for complementary evidence
            # from this document, rather than suppressing every later section.
            covered = set(_terms(excerpt.text))
            complement: Excerpt | None = None
            best_coverage = 0
            for idx in by_source[passage.source]:
                candidate = self.passages[idx]
                extra = document.excerpt(candidate.start, candidate.end, excerpt.lines)
                coverage = len((set(_terms(extra.text)) & query_terms) - covered)
                if coverage > best_coverage:
                    complement = extra
                    best_coverage = coverage
            if complement is not None:
                results.append({"source": passage.source, "title": passage.title, "excerpt": complement.text})
                if len(results) == MAX_RESULTS:
                    break
        return json.dumps({"results": results}, ensure_ascii=False)


@lru_cache(maxsize=4)
def get_wiki_index(wiki_data_path: str = "") -> WikiIndex:
    """Keep the corpus fixed within a process; restart after updating the checkout."""
    root = Path(wiki_data_path).expanduser() if wiki_data_path else DEFAULT_WIKI_PATH
    return WikiIndex(root)
