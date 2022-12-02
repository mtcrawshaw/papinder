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
        msg = ""
        msg += f"ID: {self.identifier}\n"
        msg += f"Title: {self.title}\n"
        msg += f"Authors: {self.authors}\n"
        msg += f"Published: {self.published}\n"
        msg += f"Abstract: {self.abstract}\n"
        msg += f"Link: {self.link}\n"
        return msg
