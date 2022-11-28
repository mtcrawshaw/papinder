""" Definition of Paper object to store information for a single paper. """

from typing import List


class Paper:
    def __init__(self, id: str, title: str, authors: List[str], abstract: str) -> None:
        self.id = str(id)
        self.title = str(title)
        self.authors = list(authors)
        self.abstract = str(abstract)
