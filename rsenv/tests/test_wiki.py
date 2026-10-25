import json
from pathlib import Path

import pytest

from rsenv import parse_tool_call
from rsenv.tools import TOOL_SCHEMAS
from rsenv.wiki import MAX_RESULTS, PASSAGE_CHARS, Document, WikiIndex


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    (tmp_path / "wiki" / "items").mkdir(parents=True)
    (tmp_path / "learnings").mkdir()
    (tmp_path / "wiki" / "items" / "knife.md").write_text(
        "# Knife\n\nA knife is used with logs for fletching. Spawns in Lumbridge."
    )
    (tmp_path / "learnings" / "fletching.md").write_text(
        "# Fletching\n\nKnife spawn: Lumbridge (3224, 3202). Pick it up before fletching logs."
    )
    (tmp_path / "wiki" / "items" / "fish.md").write_text("# Fish\n\nCatch fish with a net.")
    (tmp_path / "wiki" / "generate.ts").write_text("typescript_only_keyword")
    (tmp_path / "private.md").write_text("outside_corpus_keyword")
    return tmp_path


def test_searches_both_sources_with_relevant_excerpts(wiki_root: Path) -> None:
    index = WikiIndex(wiki_root)
    result = json.loads(index.search("Where can I find a knife in Lumbridge?"))
    assert {hit["source"] for hit in result["results"]} == {"wiki/items/knife.md", "learnings/fletching.md"}
    assert any("3224, 3202" in hit["excerpt"] for hit in result["results"])
    assert index.search("knife Lumbridge") == index.search("knife Lumbridge")


def test_results_are_bounded_deduplicated_and_search_entire_documents(wiki_root: Path) -> None:
    for number in range(5):
        (wiki_root / "wiki" / f"long-{number}.md").write_text(
            "# Long reference\n" + "ordinary text\n" * 500 + "\nRareingredient recipe: mix with water.\n" * 100
        )
    results = json.loads(WikiIndex(wiki_root).search("rareingredient"))["results"]
    assert len(results) == MAX_RESULTS
    assert len({hit["source"] for hit in results}) == MAX_RESULTS
    assert all(len(hit["excerpt"]) <= PASSAGE_CHARS for hit in results)
    assert all("Rareingredient" in hit["excerpt"] for hit in results)


@pytest.mark.parametrize(
    "query", ["nonexistent", "!!!", "the and", "typescript_only_keyword", "outside_corpus_keyword"]
)
def test_no_matches_and_only_markdown_in_allowed_directories(wiki_root: Path, query: str) -> None:
    assert json.loads(WikiIndex(wiki_root).search(query)) == {"results": []}


def test_missing_corpus_is_explicit(tmp_path: Path) -> None:
    result = json.loads(WikiIndex(tmp_path / "missing").search("knife"))
    assert result["results"] == []
    assert "Wiki unavailable" in result["error"]


@pytest.mark.parametrize("query", ["", "   ", "x" * 301])
def test_invalid_query_is_rejected(query: str) -> None:
    text = "<tool_call>" + json.dumps({"name": "wikiSearch", "arguments": {"query": query}}) + "</tool_call>"
    assert parse_tool_call(text) is None


def test_wiki_schema_and_json_xml_parsing() -> None:
    schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == "wikiSearch")
    assert schema["parameters"]["required"] == ["query"]
    for text in (
        '<tool_call>{"name":"wikiSearch","arguments":{"query":" knife "}}</tool_call>',
        "<tool_call><function=wikiSearch><parameter=query> knife </parameter></function></tool_call>",
    ):
        parsed = parse_tool_call(text)
        assert parsed is not None
        assert parsed[0] == "wikiSearch"
        assert parsed[1].query == "knife"


@pytest.mark.parametrize("query", ["knife", "  KNIFE!! ", "fishing", "FISHING?"])
def test_exact_title_beats_short_related_item_pages(wiki_root: Path, query: str) -> None:
    (wiki_root / "wiki" / "skills").mkdir()
    (wiki_root / "wiki" / "skills" / "fishing.md").write_text(
        "# Fishing\n\nCatch fish with suitable equipment.\n" + "Different methods require different levels.\n" * 50
    )
    (wiki_root / "wiki" / "items" / "knife.md").write_text(
        "# Knife\n\nUse this tool for fletching.\n" + "Available from shops and ground spawns.\n" * 50
    )
    for name in ("Adamant knife", "Bronze knife", "Iron knife", "Fishing pass", "Fishing trophy", "Fishing potion"):
        (wiki_root / "wiki" / "items" / f"{name.lower().replace(' ', '-')}.md").write_text(f"# {name}\n\n{name}.")
    results = json.loads(WikiIndex(wiki_root).search(query))["results"]
    expected = "wiki/items/knife.md" if "knife" in query.lower() else "wiki/skills/fishing.md"
    assert results[0]["source"] == expected
    assert (
        "Use this tool" in results[0]["excerpt"] if "knife" in query.lower() else "Catch fish" in results[0]["excerpt"]
    )


def test_normalized_filename_alias_matches(wiki_root: Path) -> None:
    (wiki_root / "wiki" / "angling-guide.md").write_text("# Fishing\n\nCatch fish with a net.")
    results = json.loads(WikiIndex(wiki_root).search("ANGLING guide!"))["results"]
    assert results[0]["source"] == "wiki/angling-guide.md"


def test_table_excerpt_restores_headings_and_headers_without_partial_rows(wiki_root: Path) -> None:
    header = "| Item | Bronze | Iron | Steel | Mithril | Adamant | Rune |"
    platebody = "| Platebody | 18 | 33 | 48 | 68 | 88 | 99 |"
    content = (
        "# Smithing\n\n## Forging\n\n"
        + header
        + "\n|---|---|---|---|---|---|---|\n"
        + "".join(f"| Equipment {i} | 1 | 15 | 30 | 50 | 70 | 85 |\n" for i in range(30))
        + platebody
        + "\n\n### Bars Per Item\n\nPlatebody: 5 bars.\n"
    )
    (wiki_root / "wiki" / "smithing.md").write_text(content)
    results = json.loads(WikiIndex(wiki_root).search("smithing steel platebody level bars"))["results"]
    excerpt = next(r["excerpt"] for r in results if platebody in r["excerpt"])
    assert "# Smithing" in excerpt and "## Forging" in excerpt
    assert header in excerpt
    assert "Platebody: 5 bars" in excerpt
    for result in results:
        assert len(result["excerpt"]) <= PASSAGE_CHARS
        if result["source"] == "wiki/smithing.md":
            assert all(line in content.splitlines() for line in result["excerpt"].splitlines())


def test_table_row_crossing_window_boundary_is_returned_whole() -> None:
    content = "# Smithing\n## Forging\n| Item | Steel |\n|---|---|\n| Platebody | 48 |\n"
    start = content.index("Platebody") + 3
    excerpt = Document(content).excerpt(start, start + 5)
    assert excerpt.text == "# Smithing\n## Forging\n| Item | Steel |\n|---|---|\n| Platebody | 48 |"


def test_complementary_sections_share_source_without_repeating_rows(wiki_root: Path) -> None:
    lobster = "| 40 | Lobster | Lobster pot | Cage/Harpoon |"
    content = (
        "# Fishing\n\nFishing requires suitable equipment.\n\n## Fish Table\n"
        "| Level | Fish | Equipment | Spot Type |\n|---|---|---|---|\n"
        + "".join(f"| {i} | Other fish {i} | Net | Net/Bait |\n" for i in range(18))
        + lobster
        + "\n"
        + "".join(f"| {i} | Other fish {i} | Net | Net/Bait |\n" for i in range(41, 55))
        + "\n## Fishing Locations\n| Location | Spot Type |\n|---|---|\n| Karamja | Cage/Harpoon |\n"
    )
    (wiki_root / "wiki" / "fishing.md").write_text(content)
    results = json.loads(WikiIndex(wiki_root).search("fishing lobster level equipment locations"))["results"]
    excerpts = [r["excerpt"] for r in results if r["source"] == "wiki/fishing.md"]
    assert len(results) <= MAX_RESULTS
    assert len(excerpts) == 2
    combined = "\n".join(excerpts)
    assert lobster in combined and "| Karamja | Cage/Harpoon |" in combined
    assert combined.count(lobster) == 1
    rows = [line for excerpt in excerpts for line in excerpt.splitlines() if "Other fish" in line]
    assert len(rows) == len(set(rows))
    assert all(len(excerpt) <= PASSAGE_CHARS for excerpt in excerpts)


def test_overlong_rows_are_not_truncated() -> None:
    row = "| " + "long cell " * PASSAGE_CHARS + " |"
    content = "# Reference\n| Item |\n|---|\n" + row + "\n"
    excerpt = Document(content).excerpt(0, len(content))
    assert len(excerpt.text) <= PASSAGE_CHARS
    assert "long cell" not in excerpt.text
