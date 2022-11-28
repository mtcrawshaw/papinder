""" Text-based interface. """

from typing import Tuple, List

from paper import Paper


LINE_LEN = 88


def get_ratings(papers: List) -> dict:
    """ Get ratings for all papers in a list. """
    ratings = {}
    for paper in papers:
        try:
            rating = get_rating(paper)
            ratings[paper.id] = (paper.title, rating)
        except KeyboardInterrupt:
            print("\n\nSaving partial results.")
            break

    return ratings


def get_rating(paper: Paper) -> bool:
    """ Print information for a paper and collect/return user rating. """

    # Print paper information.
    print("\n" + "=" * LINE_LEN + "\n")
    print(paper.title)
    print("")
    if len(paper.authors) > 0:
        print(paper.authors[0], end="")
        for author in paper.authors[:1]:
            print(f", {author}", end="")
        print("\n")
    print(paper.abstract)

    # Get user rating.
    rating = None
    while rating is None:
        print("\nRating: ", end="")
        inp = input()
        if inp in ["0", "1"]:
            rating = bool(inp)
        else:
            print("Rating must be 0 or 1.")

    return rating
