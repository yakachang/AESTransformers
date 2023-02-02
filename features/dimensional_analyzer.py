from typing import Dict

import spacy

import pattern_detector
from GEC import grammar_detector
from grammar import grammar_analyzer
from sentence import sent_analyzer
from vocab import vocab_analyzer

nlp = spacy.load("en_core_web_sm")

STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def analyze_text(text: str) -> Dict:
    """
    Report essay's dimensional analysis.
    :param text: str
    :return: Dict
    """
    doc = nlp(text)

    # GEC
    grammar_error_types = grammar_detector.gec(text)

    # Grammar Pattern
    grammar_patterns = pattern_detector.pattern_detector(doc)

    # Vocabularies #
    vocab_analysis = vocab_analyzer.get_vocab_analysis(
        doc, grammar_error_types, grammar_patterns, STOP_WORDS
    )

    # Grammars #
    grammar_analysis = grammar_analyzer.get_grammar_analysis(grammar_error_types, doc)

    # Sentence Structure #
    sent_analysis = sent_analyzer.get_sent_analysis(doc, grammar_error_types)

    return {
        "vocab": {**vocab_analysis},
        # "grammar_pattern_num": sum(grammar_patterns.values()),
        "grammar": {**grammar_analysis},
        "sent": {**sent_analysis},
        "grammar_patterns": grammar_patterns,
        "grammar_error_types": grammar_error_types,
    }
