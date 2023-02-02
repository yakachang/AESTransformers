# import math

from . import level_analyzer, utils


# Word Usage #
def get_vocab_analysis(doc, STOP_WORDS):
    vocabs = utils.get_vocabs(doc, STOP_WORDS)
    word_count = utils.get_word_count(doc)
    vocab_level_dist = level_analyzer.get_vocab_level_dist(doc, STOP_WORDS)

    return {
        # "vocabs": vocabs,
        "vocab_num": len(vocabs),
        "word_count": word_count,
        # "avg_vocab": len(vocabs) / word_count,
        "vocab_level_dist": vocab_level_dist,
    }
