""" This script should be run from the project root. """

import os
import json

from paper import Paper
from recommender import recommended_sort


LINE_LEN = 88
PAPERS_PATH = "scripts/26_icml/icml2026_papers.json"
REPORT_PATH = "scripts/26_icml/report.txt"

SESSIONS = list(range(1, 9))

# Read paper information for each session.
with open(PAPERS_PATH, "r") as papers_file:
    papers_info = json.load(papers_file)

papers = {}
for session in SESSIONS:
    papers[session] = []

for paper_info in papers_info:
    session = int(paper_info["session"])
    papers[session].append(Paper(
        identifier=int(paper_info["poster_number"]),
        title=str(paper_info["title"]),
        authors=list(paper_info["authors"]),
        published=None,
        abstract=str(paper_info["abstract"]),
        link=str(paper_info["url"]),
    ))

# Generate recommendations for each session.
for session in SESSIONS:

    sorted_papers, pred_ratings, prioritized, num_prioritized = recommended_sort(papers[session])

    # Write report and save to file.
    report_name = os.path.basename(REPORT_PATH)
    dot_pos = report_name.find(".")
    session_report_path = os.path.join(
        os.path.dirname(REPORT_PATH),
        report_name[:dot_pos] + f"_{session}" + report_name[dot_pos:]
    )

    with open(session_report_path, "w") as session_file:

        for i, paper in enumerate(sorted_papers):
            pred_rating = pred_ratings[i]
            prioritize = prioritized[i]

            prefix = f"[{i+1}/{len(sorted_papers)}]"
            session_file.write("\n" + "=" * LINE_LEN + "\n\n")
            session_file.write(prefix + " " + paper.title + "\n")
            session_file.write(paper.link + "\n")
            session_file.write(f"Poster \#{paper.identifier}\n")
            session_file.write("\n\n")
            if len(paper.authors) > 0:
                session_file.write(paper.authors[0])
                for author in paper.authors[1:]:
                    session_file.write(f", {author}")
                if prioritize:
                    session_file.write("\nPrioritized!")
                session_file.write("\n\n")
            elif prioritize:
                session_file.write("Prioritized!\n\n")
            session_file.write(paper.abstract + "\n")
            session_file.write(f"\nPredicted rating: {pred_rating}\n")
