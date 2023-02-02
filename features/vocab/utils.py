ERROR_TYPES = [
    "Collocation",
    "Form",
    "Conjunction",
    "Preposition",
    "Punctuation",
    "Spelling",
    "Word Order",
    "Other",
]


def get_vocabs(doc, STOP_WORDS):
    vocabs = set()

    for sent in doc.sents:
        for token in sent:
            if token.lemma_ not in STOP_WORDS and not token.is_punct:
                vocabs.add(token.lemma_)

    return vocabs


def get_word_count(doc):
    words = []

    for sent in doc.sents:
        for token in sent:
            if not token.is_punct:
                words.append(token.lemma_)

    return len(words)


def get_usage_error_num(grammar_error_types):

    usage_error_num = 0

    for type in ERROR_TYPES:
        if type not in ["Form", "Conjunction"]:
            usage_error_num += grammar_error_types.get(type, 0)

    return usage_error_num
