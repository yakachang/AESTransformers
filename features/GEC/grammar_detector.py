from collections import Counter

import spacy
from more_itertools import split_into

from .inference import correct

# from nltk import sent_tokenize
from .utils import extract_edits, sent_tokenize

# nlp = spacy.load(os.environ.get('SPACY_MODEL', 'en_core_web_md'), disable=['ner'])
nlp = spacy.load("en_core_web_sm", disable=["ner"])


def collect_edits(lines, sent_boundaries_list, corrected_paragraphs):
    offset = 0
    for line, sent_boundaries, corrected_sents in zip(
        lines, sent_boundaries_list, corrected_paragraphs
    ):
        for (start, end), corrected in zip(sent_boundaries, corrected_sents):
            orig = nlp(line[start:end])
            cor = nlp(corrected)
            for edit in extract_edits(orig, cor, offset=offset + start):
                yield edit
        offset += len(line)


def gec(text: str):

    if text.strip():
        # keep ends for later offset compuation
        lines = text.splitlines(True)
        # get sentence boundaries for each line, rstrip because trailing spaces are not needed
        sent_boundaries_list = [sent_tokenize(line.rstrip()) for line in lines]

        # chain sentences for running correction in batch
        inputs = [
            line[start:end]
            for line, sent_boundaries in zip(lines, sent_boundaries_list)
            for start, end in sent_boundaries
        ]
        outputs = correct(inputs)

        corrected_paragraphs = tuple(
            split_into(outputs, map(len, sent_boundaries_list))
        )
        edits = tuple(collect_edits(lines, sent_boundaries_list, corrected_paragraphs))

        grammar_error_list = [item["type"] for item in edits]

        return dict(Counter(grammar_error_list))

    return {}
