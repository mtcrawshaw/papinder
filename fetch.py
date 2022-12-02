""" Fetching arXiv papers from public API. """

import urllib.request as libreq
from datetime import date
from typing import List

import feedparser

from paper import Paper


CATEGORY = "cs.LG"
PAGE_RESULTS = 100


def get_daily_papers() -> List[Paper]:
    """ Get a list of papers released today on arXiv and return as Paper objects. """

    # Construct API query.
    query_template = "http://export.arxiv.org/api/query?"
    query_template += f"search_query=cat:{CATEGORY}"
    query_template += "&sortBy=submittedDate&sortOrder=descending"
    query_template += "&start={start}"
    query_template += f"&max_results={PAGE_RESULTS}"

    # Query API in pages until we get all papers from today.
    papers = []
    start = 0
    current_date = None
    finished = False
    while not finished:
        query = query_template.format(start=start)
        with libreq.urlopen(query) as url:
            response = url.read()

        # Parse API response.
        batch_papers = parse_response(response)
        if len(batch_papers) == 0:
            finished = True

        # Check if we have gotten all papers from the current date.
        for paper in batch_papers:

            # Set current_date if it hasn't yet been set. Note that this won't be
            # today's date, it will be the date on which the most recent batch of papers
            # were released.
            if current_date is None:
                current_date = paper.published

            if paper.published == current_date:
                papers.append(paper)
            else:
                finished = True
                break

        start += PAGE_RESULTS

    return papers


def parse_response(response: bytes) -> List[Paper]:
    """ Parse an Atom response from the arXiv API into a list of papers. """

    feed = feedparser.parse(response.decode())
    papers = []
    for entry in feed.entries:
        abs_pos = entry.id.find("abs/")
        bare_id = entry.id[abs_pos+4:]
        first_dash = entry.published.find("-")
        second_dash = entry.published.find("-", first_dash + 1)
        T_pos = entry.published.find("T")
        year = int(entry.published[:first_dash])
        month = int(entry.published[first_dash + 1: second_dash])
        day = int(entry.published[second_dash + 1: T_pos])
        papers.append(Paper(
            identifier=bare_id,
            title=entry.title,
            authors=[a["name"] for a in entry.authors],
            published=date(year, month, day),
            abstract=entry.summary,
        ))

    return papers
