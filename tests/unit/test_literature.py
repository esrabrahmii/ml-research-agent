"""Smoke tests for arXiv literature search. Doesn't hit the network."""
from mlra.agent.literature import Paper, format_for_prompt


def test_paper_short_cite():
    p = Paper(arxiv_id="2102.00001", title="Foo", authors="Smith, John, Doe, Jane", year=2021,
              summary="...", url="https://arxiv.org/abs/2102.00001")
    assert "Smith" in p.short_cite()
    assert "2021" in p.short_cite()


def test_format_for_prompt_empty():
    assert format_for_prompt([]) is None


def test_format_for_prompt_with_papers():
    papers = [
        Paper("2102.00001", "Title A", "A. Smith", 2021, "Abstract A.", "url_a"),
        Paper("2103.00002", "Title B", "B. Jones", 2022, "Abstract B.", "url_b"),
    ]
    out = format_for_prompt(papers)
    assert out is not None
    assert "[1]" in out and "[2]" in out
    assert "Smith" in out and "Jones" in out
    assert "arXiv:2102.00001" in out
