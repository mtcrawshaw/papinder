""" Fetching arXiv papers via the OAI-PMH bulk-harvesting interface. """

import re
import time
import socket
import urllib.request as libreq
import urllib.parse as libparse
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from typing import List, Optional, Tuple

from paper import Paper


CATEGORY = "cs.LG"

# OAI-PMH sets are named "<archive>:<archive>:<subcategory>", e.g. "cs:cs:LG".
_ARCHIVE, _SUBCATEGORY = CATEGORY.split(".")
OAI_SET = f"{_ARCHIVE}:{_ARCHIVE}:{_SUBCATEGORY}"

# export.arxiv.org/oai2 (the URL in every older writeup) 301-redirects here as of
# March 2025. This is a separate host/service from the search API used previously
# in this file, and is arXiv's own recommended path for exactly this use case: see
# https://info.arxiv.org/help/bulk_data.html ("OAI-PMH ... the preferred way to
# bulk-download or keep an up-to-date copy of arXiv metadata").
OAI_URL = "https://oaipmh.arxiv.org/oai"

# "arXivRaw" is the only metadata format that exposes real per-version history
# (each <version> has its own date). The plain "arXiv" format's <created> field
# does *not* reliably mean "original v1 submission date" once a paper has been
# revised -- verified against a live record where it tracked the latest revision
# instead of the original 2021 submission date.
METADATA_PREFIX = "arXivRaw"

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}

# arXiv's Terms of Use ask for at most one request every 3 seconds, across all of
# their APIs (search, OAI-PMH, RSS).
WAIT_SECONDS = 3

# Retry settings for flaky responses (timeouts, 5xx errors, malformed XML).
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 15

# The search API (previously used in this file) sits behind a CDN/WAF that
# enforces a much stricter, capacity-based rate limit than documented and will
# 429 even low-volume, well-paced traffic -- a widely-reported, arXiv-acknowledged
# 2026 issue. OAI-PMH is separate infrastructure and hasn't shown this in testing,
# but we keep the same defensive backoff in case it ever does; retrying a rate
# limit quickly only prolongs it.
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_BASE_WAIT = 30
RATE_LIMIT_MAX_WAIT = 600

# arXiv asks API clients to identify themselves via User-Agent, including contact info.
USER_AGENT = (
    "papinder/1.0 (personal arXiv paper recommender, low-volume script; "
    "contact: michael.crawshaw.4@gmail.com)"
)

# When neither an init date nor a checkpoint is known (first run), only look back
# this many days.
DEFAULT_LOOKBACK_DAYS = 3

# Cushion applied to the checkpoint, in days. ``checkpoint`` is the last date the
# user "caught up" and rated all papers available at the time. We re-fetch a few
# days before that as a safety margin (processing lag, timezone edge effects).
CHECKPOINT_CUSHION_DAYS = 7


def get_papers(init_date=None, checkpoint=None) -> Tuple[List[Paper], date]:
    """
    Get a list of arXiv papers in ``CATEGORY`` that were created or revised on or
    after the later of ``init_date`` and ``checkpoint`` minus
    ``CHECKPOINT_CUSHION_DAYS``.

    Each returned ``Paper`` reflects the *latest version* known for its arXiv id
    as of the harvest, with an identifier that includes that version (e.g.
    "2109.02355v2") and a ``published`` date equal to that version's own date --
    not the paper's original v1 submission date. This is deliberate: it's what
    lets a later revision of a paper resurface as a new, unrated item (a new
    version means a new identifier and a new date) instead of being silently
    deduplicated against a version that was already rated.

    OAI-PMH's ``from``/``until`` selective harvesting filters on when a record's
    metadata last changed, not on original submission date, so it naturally
    surfaces both brand-new papers and revisions of old ones -- exactly the two
    cases we want. Pagination uses OAI-PMH's resumptionToken mechanism, which
    (unlike the search API used previously) is designed to handle arbitrarily
    large harvests without degrading.
    """

    start_date = _resolve_start_date(init_date, checkpoint)
    if init_date is None:
        init_date = start_date
    end_date = date.today()

    if start_date > end_date:
        return [], init_date

    print(f"Harvesting {CATEGORY} via OAI-PMH from {start_date} to {end_date}.")

    papers = []
    params = {
        "verb": "ListRecords",
        "metadataPrefix": METADATA_PREFIX,
        "set": OAI_SET,
        "from": start_date.isoformat(),
        "until": end_date.isoformat(),
    }
    page = 0
    while True:
        if page > 0:
            time.sleep(WAIT_SECONDS)
        root = _oai_request(params)

        error = root.find("oai:error", NS)
        if error is not None:
            if error.get("code") == "noRecordsMatch":
                break
            raise RuntimeError(f"OAI-PMH error ({error.get('code')}): {error.text}")

        list_records = root.find("oai:ListRecords", NS)
        if list_records is None:
            raise RuntimeError("malformed OAI-PMH response (no ListRecords or error)")

        for record in list_records.findall("oai:record", NS):
            paper = _parse_record(record)
            if paper is not None:
                papers.append(paper)

        page += 1
        print(f"Fetched page {page} ({len(papers)} papers so far).")

        token = list_records.find("oai:resumptionToken", NS)
        if token is None or not token.text:
            break
        params = {"verb": "ListRecords", "resumptionToken": token.text}

    # Guard against any boundary slop in the date filtering.
    papers = [p for p in papers if p.published >= start_date]

    return papers, init_date


def _resolve_start_date(init_date: Optional[date], checkpoint: Optional[date]) -> date:
    """ Work out the earliest date to fetch. """

    if checkpoint is not None and init_date is not None:
        return max(init_date, checkpoint - timedelta(days=CHECKPOINT_CUSHION_DAYS))
    if checkpoint is not None:
        return checkpoint - timedelta(days=CHECKPOINT_CUSHION_DAYS)
    if init_date is not None:
        return init_date

    return date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)


def _parse_record(record: ET.Element) -> Optional[Paper]:
    """ Turn one OAI-PMH <record> into a Paper reflecting its latest version. """

    header = record.find("oai:header", NS)
    if header.get("status") == "deleted":
        return None

    metadata = record.find("oai:metadata", NS)
    if metadata is None:
        return None
    raw = metadata.find("raw:arXivRaw", NS)
    if raw is None:
        return None

    paper_id = raw.find("raw:id", NS).text
    versions = raw.findall("raw:version", NS)
    if not versions:
        return None
    latest = versions[-1]
    version_label = latest.get("version")
    version_date = parsedate_to_datetime(latest.find("raw:date", NS).text).date()

    abstract_el = raw.find("raw:abstract", NS)
    abstract = abstract_el.text.strip() if abstract_el is not None and abstract_el.text else ""

    return Paper(
        identifier=f"{paper_id}{version_label}",
        title=raw.find("raw:title", NS).text.strip(),
        authors=_parse_authors(raw.find("raw:authors", NS).text),
        published=version_date,
        abstract=abstract,
        link=f"https://arxiv.org/abs/{paper_id}",
    )


def _parse_authors(text: str) -> List[str]:
    """ Split arXivRaw's free-form author string into individual names. """

    text = re.sub(r"\s+and\s+", ", ", text.strip())
    return [name.strip() for name in text.split(",") if name.strip()]


def _oai_request(params: dict) -> ET.Element:
    """ Make one OAI-PMH request and parse the XML response, retrying on failure. """

    url = f"{OAI_URL}?{libparse.urlencode(params)}"
    request = libreq.Request(url, headers={"User-Agent": USER_AGENT})

    last_error = None
    attempt = 0
    rate_limit_attempt = 0
    while True:
        try:
            with libreq.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                raw = response.read()
            return ET.fromstring(raw)

        except libreq.HTTPError as e:
            if e.code == 429:
                rate_limit_attempt += 1
                if rate_limit_attempt > RATE_LIMIT_MAX_RETRIES:
                    raise RuntimeError(
                        f"arXiv OAI-PMH kept rate-limiting us after "
                        f"{RATE_LIMIT_MAX_RETRIES} backoffs"
                    ) from e
                wait = min(
                    RATE_LIMIT_BASE_WAIT * (2 ** (rate_limit_attempt - 1)),
                    RATE_LIMIT_MAX_WAIT,
                )
                print(
                    f"Rate limited by arXiv OAI-PMH (429). Backing off {wait}s "
                    f"(backoff {rate_limit_attempt}/{RATE_LIMIT_MAX_RETRIES})."
                )
                time.sleep(wait)
                continue

            last_error = e
            attempt += 1
            print(f"OAI-PMH request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"OAI-PMH request failed after {MAX_RETRIES} attempts: {last_error}"
                ) from e
            time.sleep(RETRY_WAIT_SECONDS)

        except (libreq.URLError, socket.timeout, ET.ParseError) as e:
            last_error = e
            attempt += 1
            print(f"OAI-PMH request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"OAI-PMH request failed after {MAX_RETRIES} attempts: {last_error}"
                ) from e
            time.sleep(RETRY_WAIT_SECONDS)
