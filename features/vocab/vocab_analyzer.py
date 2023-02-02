import math

from . import level_analyzer, utils

level2score = {"A1": 15, "A2": 35, "B1": 45, "B2": 55, "C1": 65, "C2": 85}

vocabs_num2level = {50: "A1", 70: "A2", 100: "B1", 150: "B2", 200: "C1", 300: "C2"}

word_count2level = {100: "A1", 150: "A2", 200: "B1", 250: "B2", 350: "C1", 500: "C2"}

spell_error_num2level = {10: "A1", 7: "A2", 5: "B1", 3: "B2", 1: "C1", 0: "C2"}

grammar_pattern_num2level = {5: "A1", 7: "A2", 10: "B1", 15: "B2", 20: "C1", 25: "C2"}


def calculate_score(
    vocabs, word_count, vocab_level_dist, spell_error_num, grammar_patterns
):

    scores = {}

    # Num of vocabs
    vocab_num = len(vocabs)
    best_level = "A1"
    for upper_bound, level in vocabs_num2level.items():
        if vocab_num >= upper_bound and level2score[level] > level2score[best_level]:
            best_level = level
    scores["vocab_num"] = level2score[best_level]

    # Num of words
    best_level = "A1"
    for upper_bound, level in word_count2level.items():
        if word_count >= upper_bound and level2score[level] > level2score[best_level]:
            best_level = level
    scores["word_count"] = level2score[best_level]

    # Vocabs' level distribution
    level_score = 0
    for level, freq in vocab_level_dist.items():
        level_score += level2score[level] * freq
    scores["level_dist"] = level_score

    # Spell Correctness
    best_level = "A1"
    for lower_bound, level in spell_error_num2level.items():
        if (
            spell_error_num <= lower_bound
            and level2score[level] > level2score[best_level]
        ):
            best_level = level
    scores["spell_correctness"] = level2score[best_level]

    # Grammar Patterns
    grammar_pattern_num = len(grammar_patterns.keys())
    best_level = "A1"
    for upper_bound, level in grammar_pattern_num2level.items():
        if (
            grammar_pattern_num >= upper_bound
            and level2score[level] > level2score[best_level]
        ):
            best_level = level
    scores["grammar_pattern"] = level2score[best_level]

    return scores


# Word Usage #
def get_vocab_analysis(doc, grammar_error_types, grammar_patterns, STOP_WORDS):
    vocabs = utils.get_vocabs(doc, STOP_WORDS)
    word_count = utils.get_word_count(doc)
    vocab_level_dist = level_analyzer.get_vocab_level_dist(doc, STOP_WORDS)
    # usage_error_num = utils.get_usage_error_num(grammar_error_types)
    spell_error_num = grammar_error_types.get("Spelling", 0)

    scores = calculate_score(
        vocabs, word_count, vocab_level_dist, spell_error_num, grammar_patterns
    )

    return {
        # "vocabs": vocabs,
        "vocab_num": len(vocabs),
        "word_count": word_count,
        # "avg_vocab": len(vocabs) / word_count,
        "vocab_level_dist": vocab_level_dist,
        # "usage_error_num": usage_error_num,
        "spelling": {
            "correct": word_count - spell_error_num,
            "mistake": spell_error_num,
        },
        "scores": scores,
        "score": math.ceil(sum(scores.values()) / len(scores.keys())),
    }
