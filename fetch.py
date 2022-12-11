""" Fetching arXiv papers from public API. """

import itertools
import urllib.request as libreq
from typing import List, Tuple
from datetime import date
from functools import reduce

import feedparser

from paper import Paper

# Construct API query.
CATEGORY = "cs.LG"
PAGE_RESULTS = 100
QUERY = "http://export.arxiv.org/api/query?"
QUERY += f"search_query=cat:{CATEGORY}"
QUERY += "&sortBy=submittedDate&sortOrder=descending"
QUERY += "&start={}"
QUERY += f"&max_results={PAGE_RESULTS}"


def get_papers(start: date) -> List[Paper]:
    """Download and parse papers since ``start``."""
    pages = itertools.count(start=0, step=PAGE_RESULTS)
    queries = map(QUERY.format, pages)
    responses = map(lambda q: libreq.urlopen(q).read(), queries)
    batches = map(parse_response, responses)
    papers = []
    for batch in batches:
        batch = list(filter(lambda p: p.published >= start, batch))
        papers = reduce(lambda acc, p: acc + [p], batch, papers)
        if len(batch) == 0:
            break
    return papers


def parse_response(response: bytes) -> List[Paper]:
    """Parse an Atom response from the arXiv API into a list of papers."""
    papers = []
    for entry in feedparser.parse(response.decode()).entries:
        abs_pos = entry.id.find("abs/")
        bare_id = entry.id[abs_pos + 4 :]
        papers.append(
            Paper(
                identifier=bare_id,
                title=entry.title,
                authors=[a["name"] for a in entry.authors],
                published=date(*entry.published_parsed[:3]),
                abstract=entry.summary,
                link=entry.link,
            )
        )
    return papers
