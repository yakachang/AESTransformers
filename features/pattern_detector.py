from collections import Counter
from typing import Dict

from GrammarPattern.parser import PatternParse


def calculate_pattern_num(pattern_dict: Dict) -> Dict:
    """
    Statisticize grammar patterns with their frequency
    :param pattern_dict: Dict
    :return: Dict
    """

    pattern_list = []

    for pattern_info in pattern_dict.values():
        for item in pattern_info:
            pattern_list.append(item["pattern"])

    pattern_counter = dict(Counter(pattern_list))

    return pattern_counter


def pattern_detector(doc) -> Dict:
    """
    Grammar pattern parse and return the grammar patterns with their frequency
    :param doc: spacy doc object
    :return: Dict
    """

    # return calculate_pattern_num(PatternParse(doc).simplify(print_result=True))
    return calculate_pattern_num(PatternParse(doc).simplify())
