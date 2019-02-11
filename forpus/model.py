from dataclasses import dataclass
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
        """Text of the document.
        """
        with open(self.filepath, "r", encoding="utf-8") as document:
            return document.read()

    @property
    def tokens(self):
        """Tokens with their indices from a document.
        """
        identifiers = self.tree.xpath("text/body/p//w/@xml:id")
        tokens = self.tree.xpath("text/body/p//w/text()")
        assert len(identifiers) == len(tokens)
        for i, token in zip(identifiers, tokens):
            yield i, token

    @property
    def entities(self):
        """Named entities with their token indices from a document.
        """
        document = xmltodict.parse(self.text)
        for annotation in document["TEI"]["text"]["body"]["p"]["persName"]:
            if "@type" in annotation:
                try:
                    if isinstance(annotation["w"], list):
                        for fragment in annotation["w"]:
                            yield fragment["@xml:id"], annotation["@type"]
                    else:
                        yield annotation["w"]["@xml:id"], annotation["@type"]
                except TypeError:
                    logging.debug(f"Unable to process annotation '{annotation}'.")
            else:
                logging.debug(f"No '@type' in {annotation}.")

    @property
    def content(self):
        """Content of the document.
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
        """Sentences of the document.
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

