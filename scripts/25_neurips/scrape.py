#!/usr/bin/env python3
"""
Scrape NeurIPS 2025 papers: title, authors, abstract.

Usage:
  python scrape_neurips2025.py --out-json neurips2025.json
"""

import re
import time
import json
import sys
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://neurips.cc/virtual/2025/papers.html"  # You can also use the user-provided URL with filters

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NeurIPS-scraper/1.0; +https://example.com/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

POSTER_HREF_RE = re.compile(r"^/virtual/2025/poster/\d+$")


def get(url, session, retries=3, backoff=1.5, timeout=30):
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            # Retry on transient 5xx/429
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep((backoff ** attempt))
                continue
            resp.raise_for_status()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep((backoff ** attempt))
    raise RuntimeError(f"Failed to GET {url}")


def discover_poster_urls(session):
    """Return a sorted list of absolute poster URLs discovered from the index page."""
    resp = get(INDEX_URL, session)
    soup = BeautifulSoup(resp.text, "html.parser")
    poster_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if POSTER_HREF_RE.match(href):
            poster_urls.add(requests.compat.urljoin(INDEX_URL, href))
    # Also support “current” index (paranoid fallback)
    if not poster_urls:
        alt = "https://neurips.cc/virtual/current/papers.html"
        resp2 = get(alt, session)
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        for a in soup2.find_all("a", href=True):
            href = a["href"]
            if POSTER_HREF_RE.match(href):
                poster_urls.add(requests.compat.urljoin(alt, href))
    return sorted(poster_urls)


def extract_title_authors_abstract(html):
    import re
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # --------------------------
    # Title & Authors (robust)
    # --------------------------
    title = ""
    authors = []

    poster_h3 = None
    for h3 in soup.find_all("h3"):
        if h3.get_text(strip=True).lower() == "poster":
            poster_h3 = h3
            break

    if poster_h3:
        # Title is the next <h2>, authors the next <h3> after that
        title_h2 = poster_h3.find_next("h2")
        if title_h2:
            title = title_h2.get_text(strip=True)
            authors_h3 = title_h2.find_next("h3")
            if authors_h3:
                raw = authors_h3.get_text(" ", strip=True)
                if "·" in raw:
                    authors = [a.strip() for a in raw.split("·") if a.strip()]
                else:
                    authors = [a.strip() for a in re.split(r"\s*,\s*", raw) if a.strip()]

    # Fallbacks
    if not title:
        for h2 in reversed(soup.find_all("h2")):
            t = h2.get_text(strip=True)
            if t and t.lower() != "main navigation":
                title = t
                break
    if not authors:
        # First <h3> after the chosen title that isn't literally "Poster"
        title_el = None
        for h2 in soup.find_all("h2"):
            if h2.get_text(strip=True) == title:
                title_el = h2
                break
        maybe_h3 = title_el.find_next("h3") if title_el else None
        if maybe_h3 and maybe_h3.get_text(strip=True).lower() != "poster":
            raw = maybe_h3.get_text(" ", strip=True)
            if "·" in raw:
                authors = [a.strip() for a in raw.split("·") if a.strip()]
            else:
                authors = [a.strip() for a in re.split(r"\s*,\s*", raw) if a.strip()]

    # --------------------------
    # Abstract (fallback to 1st version behavior)
    # --------------------------
    abstract_text = ""

    # Find a text node containing "abstract:" (case-insensitive)
    abstract_label = None
    for el in soup.find_all(string=True):
        if isinstance(el, str) and "abstract:" in el.lower():
            abstract_label = el
            break

    if abstract_label:
        node = abstract_label.parent

        # Case A: parent already has "Abstract: <text>" — split and use the latter
        parent_text = node.get_text(separator="\n")
        parts = re.split(r"(?i)abstract:\s*", parent_text, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            abstract_text = parts[1].strip()
        else:
            # Case B: walk forward siblings until a section break (headers/nav/hr)
            pieces = []
            for sib in node.next_siblings:
                name = getattr(sib, "name", None)
                if name in {"h2", "h3", "nav", "hr"}:
                    break
                if hasattr(sib, "get_text"):
                    txt = sib.get_text(" ", strip=True)
                else:
                    txt = str(sib).strip()
                if txt:
                    pieces.append(txt)
            abstract_text = " ".join(pieces).strip()

    # Final cleanups
    title = re.sub(r"\s+", " ", title).strip()
    abstract_text = re.sub(r"\s+", " ", abstract_text).strip()

    return title, authors, abstract_text


def scrape_all(out_json, sleep_sec=0.4):
    session = requests.Session()
    posters = discover_poster_urls(session)
    print(f"Discovered {len(posters)} poster pages")

    rows = []
    for i, url in enumerate(posters, 1):
        try:
            resp = get(url, session)
            title, authors, abstract = extract_title_authors_abstract(resp.text)
            rows.append(
                {
                    "url": url,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                }
            )
            if i % 25 == 0:
                print(f"  ...fetched {i}/{len(posters)}")
        except Exception as e:
            sys.stderr.write(f"[warn] failed {url}: {e}\n")
        time.sleep(sleep_sec)  # be polite

    # Write JSON
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    print("Done.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="neurips2025_papers.json")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between requests (seconds)")
    args = parser.parse_args()
    scrape_all(args.out_json, sleep_sec=args.sleep)


if __name__ == "__main__":
    main()
