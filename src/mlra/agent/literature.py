"""arXiv search — retrieve relevant papers for a research question.

The agent grounds its plan in real literature instead of pretending to know.
Each result is small (title + abstract + ID + URL) so we can pass several
into the planner prompt as context.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: str
    year: int
    summary: str  # truncated abstract
    url: str

    def short_cite(self) -> str:
        first_author = self.authors.split(",")[0].split(" and ")[0].strip()
        last = first_author.split()[-1] if first_author else "?"
        return f"{last} et al. {self.year}"


_STOPWORDS = {
    "how", "does", "do", "is", "are", "the", "a", "an", "of", "to", "on", "in",
    "for", "and", "or", "but", "with", "without", "vs", "at", "as", "by", "from",
    "into", "than", "what", "which", "when", "where", "why", "this", "that",
    "these", "those", "it", "be", "was", "were", "been", "have", "has", "had",
    "will", "would", "should", "could", "can", "may", "might", "i", "you", "we",
    "any", "some", "all", "each", "more", "less", "between", "across", "over",
}


def _keywords(text: str) -> str:
    """Strip stopwords from a natural-language question to make a keyword search query."""
    import re
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
    keep = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return " ".join(keep)


def _do_search(query: str, max_results: int, max_summary_chars: int) -> list[Paper]:
    try:
        import arxiv
    except ImportError:
        return []

    client = arxiv.Client(page_size=max_results, num_retries=2)
    search_obj = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[Paper] = []
    try:
        for result in client.results(search_obj):
            authors_str = ", ".join(a.name for a in result.authors[:4])
            summary = (result.summary or "").replace("\n", " ").strip()
            if len(summary) > max_summary_chars:
                summary = summary[:max_summary_chars].rsplit(" ", 1)[0] + "…"
            papers.append(
                Paper(
                    arxiv_id=result.entry_id.rsplit("/", 1)[-1],
                    title=result.title.strip(),
                    authors=authors_str,
                    year=result.published.year if result.published else 0,
                    summary=summary,
                    url=result.entry_id,
                )
            )
    except Exception:
        pass
    return papers


def search(query: str, max_results: int = 4, max_summary_chars: int = 400) -> list[Paper]:
    """Search arXiv. Tries the literal query first, then a keyword-only fallback."""
    papers = _do_search(query, max_results, max_summary_chars)
    if papers:
        return papers
    # Fallback: strip stopwords and retry — arxiv keyword AND-matching is brittle on full questions
    kw_query = _keywords(query)
    if kw_query and kw_query != query.lower().strip():
        papers = _do_search(kw_query, max_results, max_summary_chars)
    return papers


def format_for_prompt(papers: list[Paper]) -> str | None:
    """Format a list of papers for inclusion in an LLM system prompt."""
    if not papers:
        return None
    blocks = []
    for i, p in enumerate(papers, 1):
        blocks.append(
            f"[{i}] {p.short_cite()} — \"{p.title}\"\n"
            f"    arXiv:{p.arxiv_id}\n"
            f"    {p.summary}"
        )
    return "\n\n".join(blocks)
