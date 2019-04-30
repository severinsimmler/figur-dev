from spacy.lang.de import German


def segment_sentences(text: str):
    nlp = German()
    document = nlp(text)
    for sentence in document.sents:
        yield sentence.string.strip()
