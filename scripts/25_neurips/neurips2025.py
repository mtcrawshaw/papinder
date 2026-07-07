""" This script should be run from the project root. """

import json

from paper import Paper
from recommender import recommended_sort


LINE_LEN = 88
PAPERS_PATH = "scripts/neurips2025/neurips2025_papers.json"


# Read paper information.
papers = []
with open(PAPERS_PATH, "r") as papers_file:
    papers_info = json.load(papers_file)
for paper_info in papers_info:
    papers.append(Paper(
        identifier=None,
        title=str(paper_info["title"]),
        authors=list(paper_info["authors"]),
        published=None,
        abstract=str(paper_info["abstract"]),
        link=str(paper_info["url"]),
    ))

# Generate recommendations.
papers, pred_ratings, prioritized, num_prioritized = recommended_sort(papers)

# Print report.
for i, paper in enumerate(papers):
    pred_rating = pred_ratings[i]
    prioritize = prioritized[i]

    prefix = f"[{i+1}/{len(papers)}]"
    print("\n" + "=" * LINE_LEN + "\n")
    print(prefix + " " + paper.title)
    print(paper.link)
    print("")
    if len(paper.authors) > 0:
        print(paper.authors[0], end="")
        for author in paper.authors[1:]:
            print(f", {author}", end="")
        if prioritize:
            print("\nPrioritized!", end="")
        print("\n")
    elif prioritize:
        print("Prioritized!")
        print("\n")
    print(paper.abstract)
    print(f"\nPredicted rating: {pred_rating}")
