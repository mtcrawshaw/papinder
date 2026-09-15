""" Fetching arXiv papers from public API. """

import time
import socket
import urllib.request as libreq
import urllib.parse as libparse
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

import feedparser

from paper import Paper


CATEGORY = "cs.LG"
API_URL = "https://export.arxiv.org/api/query"

# The arXiv API returns results in slices of at most 2000, and asks callers to
# incorporate a 3 second delay between successive requests.
PAGE_RESULTS = 2000
WAIT_SECONDS = 3

# The API refuses (HTTP 500) to page deep into a large result set, so we split
# the requested date range into windows of this many days and fetch each window
# separately. A week of cs.LG is only ~1000-1500 papers, so each window is
# typically a single page with no deep paging.
WINDOW_DAYS = 7

# Retry settings for flaky responses (timeouts, 5xx errors, truncated feeds).
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 15

# arXiv fronts the API with a CDN/WAF layer that enforces a stricter, IP-based
# rate limit than the "3 seconds between requests" guidance in their docs, and
# will respond 429 "Rate exceeded" even to well-paced, low-volume traffic (this
# is a widely-reported 2026 issue, not specific to this code). Retrying quickly
# only prolongs it, so a 429 gets its own exponential backoff instead of the
# flat retry above.
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_BASE_WAIT = 30
RATE_LIMIT_MAX_WAIT = 600

# arXiv asks API clients to identify themselves via User-Agent, including contact info.
USER_AGENT = (
    "papinder/1.0 (personal arXiv paper recommender, low-volume script; "
    "contact: michael.crawshaw.4@gmail.com)"
)

# When neither an init date nor a checkpoint is known (first run), only look back
# this many days from the most recent submission.
DEFAULT_LOOKBACK_DAYS = 3

# Cushion applied to the checkpoint, in days. ``checkpoint`` is the last date the
# user "caught up" and rated all papers available at the time. We re-fetch a few
# days before that because a paper submitted on a given day may only be announced
# (and so become visible to us) a day or two later.
CHECKPOINT_CUSHION_DAYS = 7


def get_papers(init_date=None, checkpoint=None) -> Tuple[List[Paper], date]:
    """
    Get a list of arXiv papers in ``CATEGORY`` submitted on or after the later of
    ``init_date`` and ``checkpoint`` minus ``CHECKPOINT_CUSHION_DAYS``.

    Rather than downloading the entire category and filtering client-side (which
    exceeds the arXiv API's supported result-set size and frequently times out),
    this restricts the query with a ``submittedDate`` range. The range is fetched
    one ``WINDOW_DAYS``-day window at a time, since the API returns HTTP 500 when
    asked to page deep into a large result set.
    """

    # Determine the earliest submission date we care about.
    start_date = _resolve_start_date(init_date, checkpoint)
    if init_date is None:
        init_date = start_date

    # End bound a day past "now" (GMT) so we don't miss anything near the boundary.
    end_date = (datetime.now(timezone.utc) + timedelta(days=1)).date()

    # Fetch one date window at a time, de-duplicating across window boundaries.
    papers = []
    seen_ids = set()
    windows = list(_date_windows(start_date, end_date, WINDOW_DAYS))
    for i, (win_start, win_end) in enumerate(windows):
        if i > 0:
            time.sleep(WAIT_SECONDS)
        print(
            f"Fetching papers submitted {win_start} to {win_end} "
            f"(window {i + 1}/{len(windows)})."
        )
        for paper in _query_window(win_start, win_end):
            if paper.identifier not in seen_ids:
                seen_ids.add(paper.identifier)
                papers.append(paper)

    # Guard against any boundary slop in the API's date filtering.
    papers = [p for p in papers if p.published >= start_date]

    return papers, init_date


def _date_windows(start_date: date, end_date: date, window_days: int):
    """ Yield ``(win_start, win_end)`` pairs tiling ``[start_date, end_date)``. """

    win_start = start_date
    while win_start < end_date:
        win_end = min(win_start + timedelta(days=window_days), end_date)
        yield win_start, win_end
        win_start = win_end


def _query_window(win_start: date, win_end: date) -> List[Paper]:
    """ Fetch every paper in ``CATEGORY`` submitted in ``[win_start, win_end)``. """

    # submittedDate bounds are GMT timestamps to the minute. The range is
    # inclusive on both ends, so consecutive windows share their boundary minute;
    # the caller de-duplicates by identifier to absorb that overlap.
    start_stamp = f"{win_start.strftime('%Y%m%d')}0000"
    end_stamp = f"{win_end.strftime('%Y%m%d')}0000"
    search_query = (
        f"cat:{CATEGORY} AND submittedDate:[{start_stamp} TO {end_stamp}]"
    )

    papers = []
    start = 0
    total_results = None
    while total_results is None or start < total_results:
        if start > 0:
            time.sleep(WAIT_SECONDS)
        batch_papers, total_results = _query_page(search_query, start)
        papers.extend(batch_papers)
        start += PAGE_RESULTS

    return papers


def _resolve_start_date(init_date: Optional[date], checkpoint: Optional[date]) -> date:
    """ Work out the earliest submission date to fetch. """

    if checkpoint is not None and init_date is not None:
        return max(init_date, checkpoint - timedelta(days=CHECKPOINT_CUSHION_DAYS))
    if checkpoint is not None:
        return checkpoint - timedelta(days=CHECKPOINT_CUSHION_DAYS)
    if init_date is not None:
        return init_date

    # First run: anchor to the most recent submission date in the category.
    latest = _latest_submitted_date()
    return latest - timedelta(days=DEFAULT_LOOKBACK_DAYS)


def _latest_submitted_date() -> date:
    """ Return the submission date of the most recent paper in ``CATEGORY``. """

    batch_papers, _ = _query_page(
        f"cat:{CATEGORY}", start=0, max_results=1, sort_order="descending"
    )
    if not batch_papers:
        return date.today()
    return batch_papers[0].published


def _query_page(
    search_query: str,
    start: int,
    max_results: int = PAGE_RESULTS,
    sort_order: str = "ascending",
) -> Tuple[List[Paper], int]:
    """ Fetch and parse a single page of API results, retrying on failure. """

    params = libparse.urlencode({
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": sort_order,
    })
    url = f"{API_URL}?{params}"
    request = libreq.Request(url, headers={"User-Agent": USER_AGENT})

    last_error = None
    attempt = 0
    rate_limit_attempt = 0
    while True:
        try:
            with libreq.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                raw = response.read()
            batch_papers, total_results = parse_response(raw)

            # arXiv sometimes returns an empty feed for a page that should have
            # results (a known symptom of overloaded / truncated responses).
            # Retry unless we've genuinely reached the end of the result set.
            if not batch_papers and start < total_results:
                raise RuntimeError(
                    f"empty page at {start}/{total_results}"
                )

            return batch_papers, total_results

        except libreq.HTTPError as e:
            if e.code == 429:
                rate_limit_attempt += 1
                if rate_limit_attempt > RATE_LIMIT_MAX_RETRIES:
                    raise RuntimeError(
                        f"arXiv API kept rate-limiting us after "
                        f"{RATE_LIMIT_MAX_RETRIES} backoffs"
                    ) from e
                wait = min(
                    RATE_LIMIT_BASE_WAIT * (2 ** (rate_limit_attempt - 1)),
                    RATE_LIMIT_MAX_WAIT,
                )
                print(
                    f"Rate limited by arXiv API (429). Backing off {wait}s "
                    f"(backoff {rate_limit_attempt}/{RATE_LIMIT_MAX_RETRIES})."
                )
                time.sleep(wait)
                continue

            last_error = e
            attempt += 1
            print(f"arXiv API request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"arXiv API request failed after {MAX_RETRIES} attempts: {last_error}"
                ) from e
            time.sleep(RETRY_WAIT_SECONDS)

        except (libreq.URLError, socket.timeout, RuntimeError) as e:
            last_error = e
            attempt += 1
            print(f"arXiv API request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"arXiv API request failed after {MAX_RETRIES} attempts: {last_error}"
                ) from e
            time.sleep(RETRY_WAIT_SECONDS)


def parse_response(response: bytes) -> Tuple[List[Paper], int]:
    """ Parse an Atom response from the arXiv API into a list of papers. """

    feed = feedparser.parse(response)
    if not hasattr(feed.feed, "opensearch_totalresults"):
        raise RuntimeError("malformed API response (no opensearch:totalResults)")
    total_results = int(feed.feed.opensearch_totalresults)

    papers = []
    for entry in feed.entries:
        abs_pos = entry.id.find("abs/")
        bare_id = entry.id[abs_pos + 4:]
        published = date(*entry.updated_parsed[:3])
        papers.append(Paper(
            identifier=bare_id,
            title=entry.title,
            authors=[a["name"] for a in entry.authors],
            published=published,
            abstract=entry.summary,
            link=entry.link,
        ))

    return papers, total_results
