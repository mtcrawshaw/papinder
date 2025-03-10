import urllib.request as libreq
import feedparser

from fetch import parse_response


query = "http://export.arxiv.org/api/query?"
query += "search_query=cat:cs.LG"
query += "&sortBy=submittedDate&sortOrder=descending"
query += "&start=0"
query += "&max_results=10"

with libreq.urlopen(query) as url:
    response = url.read()

print(response.decode())
feed = feedparser.parse(response.decode())
for entry in feed.entries:
    print(entry.published)
