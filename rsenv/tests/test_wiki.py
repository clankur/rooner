import json
from pathlib import Path

import pytest

from rsenv import parse_tool_call
from rsenv.tools import TOOL_SCHEMAS
from rsenv.wiki import MAX_RESULTS, PASSAGE_CHARS, WikiIndex


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
