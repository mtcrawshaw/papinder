""" Definition of Paper object to store information for a single paper. """

from datetime import date
from typing import List


class Paper:
    def __init__(
        self,
        identifier: str,
        title: str,
        authors: List[str],
        published: date,
        abstract: str,
        link: str,
    ) -> None:
        self.identifier = str(identifier)
        self.title = str(title)
        self.authors = list(authors)
        self.published = published
        self.abstract = str(abstract)
        self.link = str(link)

    def __repr__(self):
        return "\n".join(
            [
                self.title,
                str(self.published),
                self.link,
                "",
                ", ".join(self.authors),
                self.abstract,
            ]
        )
