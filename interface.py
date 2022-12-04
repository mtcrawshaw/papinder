""" Text-based interface. """

from datetime import date
from typing import Tuple, List

from paper import Paper


LINE_LEN = 88


def get_ratings(
    papers: List, batch_size: int, init_date: date, checkpoint: date,
) -> dict:
    """ Get ratings for all papers in a list. """

    ratings = {}

    print("\n" + "=" * LINE_LEN + "\n")
    print(f"Initial date: {init_date}")
    print(f"Last checkpoint: {checkpoint}")
    print(f"Today's date: {date.today()}")
    print(f"Collected {batch_size} papers since checkpoint.")

    for i, paper in enumerate(papers):
        try:
            prefix = f"[{i+1}/{batch_size}]"
            rating = get_rating(paper, prefix)
            ratings[paper.identifier] = (paper, rating)
        except KeyboardInterrupt:
            print("\n\nSaving partial results.\n")
            break

    print("\n" + "=" * LINE_LEN + "\n")
    return ratings


def get_rating(paper: Paper, prefix: str) -> bool:
    """ Print information for a paper and collect/return user rating. """

    # Print paper information.
    print("\n" + "=" * LINE_LEN + "\n")
    print(prefix + " " + paper.title)
    print(paper.published)
    print(paper.link)
    print("")
    if len(paper.authors) > 0:
        print(paper.authors[0], end="")
        for author in paper.authors[1:]:
            print(f", {author}", end="")
        print("\n")
    print(paper.abstract)

    # Get user rating.
    rating = None
    while rating is None:
        print("\nRating: ", end="")
        inp = input()
        if inp in ["0", "1"]:
            rating = (inp == "1")
        else:
            print("Rating must be 0 or 1.")

    return rating
