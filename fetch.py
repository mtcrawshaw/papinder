""" Fetching arXiv papers from public API. """

import urllib.request as libreq
import time
from datetime import date, timedelta
from typing import List, Tuple

import feedparser

from paper import Paper


CATEGORY = "cs.LG"
PAGE_RESULTS = 1000
WAIT_SECONDS = 5


def get_papers(init_date=None, checkpoint=None) -> List[Paper]:
    """
    Get a list of arXiv papers released on or after the later of ``init_date`` and
    ``checkpoint`` minus 7 days. ``checkpoint`` is the last date that the user "caught
    up" and rated all papers which were currently available at the time of rating. This
    is a bit janky, but it's the cleanest way I've thought of to deal with the fact that
    the arXiv API only dates papers by when they were submitted, but not all papers
    which are submitted on the same day are released on the same day. The 7 day cushion
    accounts for the event where we rate papers which were submitted on a given day, and
    sometime after there are other papers released which were submitted on that same
    day.
    """

    # Construct API query.
    query_template = "http://export.arxiv.org/api/query?"
    query_template += f"search_query=cat:{CATEGORY}"
    query_template += "&sortBy=submittedDate&sortOrder=descending"
    query_template += "&start={start}"
    query_template += f"&max_results={PAGE_RESULTS}"

    # Construct start date for papers.
    if checkpoint is None and init_date is None:
        start_date = None
    elif checkpoint is None and init_date is not None:
        start_date = init_date
    elif checkpoint is not None and init_date is not None:
        start_date = max(init_date, checkpoint - timedelta(days=7))
    else:
        raise NotImplementedError

    # Query API in pages until we get all papers from the desired range.
    results_len = None
    papers = []
    start = 0
    finished = False
    while not finished:
        time.sleep(WAIT_SECONDS)
        query = query_template.format(start=start)
        with libreq.urlopen(query) as url:
            response = url.read()

        # Parse API response and check if we are finished paging results. We include a
        # check for empty entries list: This sometimes happens when the result set is
        # too big (>30,000), which is not officially supported by the arXiv API.
        batch_papers, results_len = parse_response(response)
        if len(batch_papers) == 0:
            if start == results_len:
                finished = True
            else:
                print(f"Received empty results list after {start}/{results_len}. Retrying...")
                continue

        # Check if we have gotten all papers from the desired range.
        for paper in batch_papers:

            # Set start date to date of most recent paper, if necessary.
            if start_date is None:
                start_date = paper.published
                init_date = start_date

            if paper.published >= start_date:
                papers.append(paper)
            else:
                finished = True
                break

        start += PAGE_RESULTS

    return papers, init_date


def parse_response(response: bytes) -> Tuple[List[Paper], int]:
    """ Parse an Atom response from the arXiv API into a list of papers. """

    feed = feedparser.parse(response.decode())

    # Parse papers.
    results_len = feed.feed.opensearch_totalresults
    papers = []
    for entry in feed.entries:
        abs_pos = entry.id.find("abs/")
        bare_id = entry.id[abs_pos+4:]
        first_dash = entry.updated.find("-")
        second_dash = entry.updated.find("-", first_dash + 1)
        T_pos = entry.updated.find("T")
        year = int(entry.updated[:first_dash])
        month = int(entry.updated[first_dash + 1: second_dash])
        day = int(entry.updated[second_dash + 1: T_pos])
        papers.append(Paper(
            identifier=bare_id,
            title=entry.title,
            authors=[a["name"] for a in entry.authors],
            published=date(year, month, day),
            abstract=entry.summary,
            link=entry.link,
        ))

    return papers, results_len
