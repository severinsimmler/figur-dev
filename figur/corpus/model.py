from dataclasses import dataclass
from pathlib import Path
import logging

import lxml.etree
import xmltodict


@dataclass
class Document:
    filepath: str

    @property
    def tree(self):
        """Element tree.
        """
        return lxml.etree.parse(self.filepath)

    @property
    def text(self):
        """Plain text.
        """
        with Path(self.filepath).open("r", encoding="utf-8") as document:
            return document.read()

    @property
    def tokens(self):
        """Tokens with their indices.
        """
        identifiers = self.tree.xpath("text/body/p//w/@xml:id")
        tokens = self.tree.xpath("text/body/p//w/text()")
        # There must be as much identifiers as tokens:
        assert len(identifiers) == len(tokens)
        for i, token in zip(identifiers, tokens):
            yield i, token

    @property
    def entities(self):
        """Named entities with their token indices from a document.
        """
        document = xmltodict.parse(self.text)
        for annotation in document["TEI"]["text"]["body"]["p"]["persName"]:
            # Annotation must have a `type` attribute:
            if "@type" in annotation:
                try:
                    if isinstance(annotation["w"], list):
                        for fragment in annotation["w"]:
                            yield fragment["@xml:id"], annotation["@type"]
                    else:
                        yield annotation["w"]["@xml:id"], annotation["@type"]
                except TypeError:
                    logging.debug(f"Unable to process '{annotation}'.")
            else:
                logging.debug(f"No type attribute in '{annotation}'.")

    @property
    def content(self):
        """Index mapped to tokens.
        """
        entities = dict(self.entities)
        document = dict()
        for index, token in self.tokens:
            if index in entities:
                document[index] = (token, entities[index])
            else:
                document[index] = (token, "")
        return document

    @property
    def sentences(self):
        """Tokens of all sentences.
        """
        content = self.content
        joins = self.tree.xpath("text/body/p//join[@results='s']//@target")
        for join in joins:
            identifiers = join.replace("#", "").strip().split(" ")
            sentence = list()
            for identifier in identifiers:
                if identifier in content:
                    sentence.append(content[identifier])
                else:
                    logging.debug(f"No token with identifier '{identifier}' in "
                                   "document, but found in sentence element.")
            yield sentence


@dataclass
class Corpus(NLPTaskDataFetcher):
    directory: str

    def fetch(self, train, dev, test):
        return self.load_column_corpus(self.directory,
                                       {1: "text", 2: "ner"},
                                       train_file=train,
                                       dev_file=dev,
                                       test_file=test)
