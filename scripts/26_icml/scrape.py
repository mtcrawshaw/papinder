#!/usr/bin/env python3
"""
scrape_icml2026.py  (session-only, fast version)
================================================

Scrape ICML 2026 papers from the eight POSTER SESSION pages only. This makes
just 8 HTTP requests total (one per session) instead of one request per paper,
so it finishes in seconds. Every field you need is present on the session page.

For each paper it stores:
    - title
    - authors            (list of strings)
    - abstract
    - poster_number      (the board position, e.g. "125")
    - url                (the paper's poster page, built from the card link)
    - session            (poster session number, 1-8)
    - session_datetime   (presentation slot for that session, e.g.
                          "Tue 7 Jul 10:30 a.m. KST — 12:15 p.m. KST")
    - session_location   (e.g. "HALL A")

The session pages are login-gated, so you must supply your logged-in cookie.

------------------------------------------------------------------------------
Two ways to run
------------------------------------------------------------------------------
A) LIVE (fetch the 8 pages yourself, using your cookie):

     export ICML_COOKIE='<the full Cookie header from your browser>'
     python scrape_icml2026.py

   or with a Netscape cookies.txt:

     python scrape_icml2026.py --cookies cookies.txt

B) OFFLINE (parse pages you've already saved from the browser):

   If live fetching is a hassle, just save each of the 8 session pages from
   Chrome (right-click the <html> node -> Copy -> Copy outerHTML, paste into a
   file), then point the script at them:

     python scrape_icml2026.py --html session1.html session2.html ...

   Offline mode needs no cookie at all. The session number is read from each
   page's own "Poster Session N" title, so file order doesn't matter.

Output: icml2026_papers.json   (override with -o)

    pip install requests beautifulsoup4
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

BASE = "https://icml.cc"

# Session URLs in order. Index 0 -> "Poster Session 1", etc. Used in live mode;
# in offline mode the session number is read from each page instead.
SESSION_URLS = [
    "https://icml.cc/virtual/2026/session/68685",  # Session 1
    "https://icml.cc/virtual/2026/session/68686",  # Session 2
    "https://icml.cc/virtual/2026/session/68687",  # Session 3
    "https://icml.cc/virtual/2026/session/68688",  # Session 4
    "https://icml.cc/virtual/2026/session/68689",  # Session 5
    "https://icml.cc/virtual/2026/session/68690",  # Session 6
    "https://icml.cc/virtual/2026/session/68691",  # Session 7
    "https://icml.cc/virtual/2026/session/68692",  # Session 8
]

POSTER_HREF_RE = re.compile(r"/virtual/2026/poster/(\d+)")
SESSION_NUM_RE = re.compile(r"Poster\s+Session\s+(\d+)", re.IGNORECASE)
AUTHOR_SPLIT_RE = re.compile(r"\s*[⋅·•‧・]\s*")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Networking (live mode only)
# --------------------------------------------------------------------------- #
def build_session(cookie_header: Optional[str], cookie_file: Optional[str]):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    })
    retry = Retry(total=4, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    if cookie_file:
        jar = http.cookiejar.MozillaCookieJar(cookie_file)
        jar.load(ignore_discard=True, ignore_expires=True)
        s.cookies = jar  # type: ignore[assignment]
    if cookie_header:
        for part in cookie_header.split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                s.cookies.set(k, v, domain="icml.cc")
    return s


def fetch(sess, url: str) -> Optional[str]:
    import requests
    try:
        r = sess.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except requests.RequestException as exc:
        print(f"  ! request failed for {url}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Parsing a session page
# --------------------------------------------------------------------------- #
def parse_session_page(html: str, fallback_session: Optional[int] = None) -> List[Dict]:
    """Return one record per paper found on a single session page."""
    soup = BeautifulSoup(html, "html.parser")

    # --- session-level fields (from the page header) -----------------------
    session_num = fallback_session
    header = soup.select_one("h2.card-title.main-title") or soup.select_one("h2.card-title")
    if header:
        m = SESSION_NUM_RE.search(header.get_text())
        if m:
            session_num = int(m.group(1))
    if session_num is None:  # last resort: <title>Track: Poster Session N</title>
        if soup.title:
            m = SESSION_NUM_RE.search(soup.title.get_text())
            if m:
                session_num = int(m.group(1))

    # Session location, e.g. "HALL A" (an h5 right under the title).
    session_location = None
    loc_el = soup.select_one("h5.text-center.text-muted")
    if loc_el:
        session_location = _clean(loc_el.get_text())

    # Session date/time slot, e.g. "Tue 7 Jul 10:30 a.m. KST — 12:15 p.m. KST".
    session_datetime = _parse_session_datetime(soup)

    # --- per-paper cards ---------------------------------------------------
    records: List[Dict] = []
    for card in soup.find_all("div", class_="track-schedule-card"):
        link = card.find("a", href=POSTER_HREF_RE)
        if not link:
            continue  # not a paper card

        pid_m = POSTER_HREF_RE.search(link.get("href", ""))
        poster_page_id = pid_m.group(1) if pid_m else None

        title = _clean(link.get_text())

        # Poster board number, e.g. "#125" -> "125".
        poster_number = None
        pos = card.find("div", attrs={"title": "Poster Position"})
        if pos:
            poster_number = _clean(pos.get_text()).lstrip("#").strip()

        # Authors: the text-muted paragraph, split on middot variants.
        authors: List[str] = []
        auth_p = card.find("p", class_="text-muted")
        if auth_p:
            authors = [a for a in AUTHOR_SPLIT_RE.split(_clean(auth_p.get_text())) if a]

        # Abstract: the .abstract block (drop its empty leading/trailing <p>s).
        abstract = None
        abs_div = card.find("div", class_="abstract")
        if abs_div:
            abstract = _clean(abs_div.get_text(" ", strip=True)) or None

        records.append({
            "session": session_num,
            "poster_number": poster_number,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": f"{BASE}/virtual/2026/poster/{poster_page_id}" if poster_page_id else None,
            "session_datetime": session_datetime,
            "session_location": session_location,
        })

    return records


def _parse_session_datetime(soup) -> Optional[str]:
    """Find the header line like 'Tue 7 Jul 10:30 a.m. KST — 12:15 p.m. KST'."""
    pat = re.compile(
        r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{1,2}\s+[A-Z][a-z]{2}.*?"
        r"[AaPp]\.?[Mm]\.?.*?—.*?[AaPp]\.?[Mm]\.?(?:\s+[A-Z]{2,4})?",
        re.DOTALL,
    )
    for div in soup.find_all("div"):
        # Only look at small leaf-ish divs to avoid grabbing the whole page.
        if div.find("div"):
            continue
        txt = _clean(div.get_text(" ", strip=True))
        if not txt or len(txt) > 120:
            continue
        m = pat.search(txt)
        if m:
            return _clean(m.group(0))
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape ICML 2026 papers from the 8 session pages.")
    ap.add_argument("-o", "--output", default="icml2026_papers.json")
    ap.add_argument("--html", nargs="+", metavar="FILE",
                    help="parse already-saved session HTML files (offline mode; no cookie needed)")
    ap.add_argument("--cookies", help="Netscape cookies.txt for live authenticated fetching")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between the 8 live requests")
    args = ap.parse_args()

    all_records: List[Dict] = []

    if args.html:
        # ---- OFFLINE MODE --------------------------------------------------
        for path in args.html:
            with open(path, encoding="utf-8", errors="replace") as fh:
                html = fh.read()
            recs = parse_session_page(html)
            snum = recs[0]["session"] if recs else "?"
            print(f"{os.path.basename(path)}: session {snum}, {len(recs)} papers")
            all_records.extend(recs)
    else:
        # ---- LIVE MODE -----------------------------------------------------
        cookie_header = os.environ.get("ICML_COOKIE")
        if not cookie_header and not args.cookies:
            print("No cookie provided. The session pages are login-gated, so live mode\n"
                  "needs one. Either set ICML_COOKIE / pass --cookies, or use offline\n"
                  "mode:  python scrape_icml2026.py --html session1.html ...",
                  file=sys.stderr)
            sys.exit(2)
        sess = build_session(cookie_header, args.cookies)
        for idx, url in enumerate(SESSION_URLS, start=1):
            html = fetch(sess, url)
            if not html:
                continue
            recs = parse_session_page(html, fallback_session=idx)
            if not recs:
                print(f"session {idx}: 0 papers — cookie may be invalid/expired "
                      f"(page likely showed 'Registration Required').", file=sys.stderr)
            else:
                print(f"session {idx}: {len(recs)} papers")
            all_records.extend(recs)
            time.sleep(args.delay)

    if not all_records:
        print("No papers parsed. Aborting.", file=sys.stderr)
        sys.exit(1)

    all_records.sort(key=lambda r: (r.get("session") or 99, r.get("poster_number") or ""))
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(all_records, fh, ensure_ascii=False, indent=2)

    n_abs = sum(1 for r in all_records if r.get("abstract"))
    n_sess = len({r["session"] for r in all_records if r.get("session")})
    print(f"\nDone. Wrote {len(all_records)} papers across {n_sess} sessions "
          f"to {args.output} ({n_abs} have an abstract).")


if __name__ == "__main__":
    main()
