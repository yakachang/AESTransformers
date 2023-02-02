import math

from . import utils

level2score = {"A1": 15, "A2": 35, "B1": 50, "B2": 60, "C1": 70, "C2": 90}

form_error_num2level = {7: "A1", 5: "A2", 3: "B1", 2: "B2", 1: "C1", 0: "C2"}
prep_error_num2level = {7: "A1", 5: "A2", 3: "B1", 2: "B2", 1: "C1", 0: "C2"}
punc_error_num2level = {7: "A1", 5: "A2", 3: "B1", 2: "B2", 1: "C1", 0: "C2"}


def calculate_score(form_error_num, prep_error_num, punc_error_num):

    scores = {}

    # Form
    best_level = "A1"
    for lower_bound, level in form_error_num2level.items():
        if (
            form_error_num <= lower_bound
            and level2score[level] > level2score[best_level]
        ):
            best_level = level
    scores["form_usage"] = level2score[best_level]

    # Preposition
    best_level = "A1"
    for lower_bound, level in prep_error_num2level.items():
        if (
            prep_error_num <= lower_bound
            and level2score[level] > level2score[best_level]
        ):
            best_level = level
    scores["prep_usage"] = level2score[best_level]

    # Punctuation
    best_level = "A1"
    for lower_bound, level in punc_error_num2level.items():
        if (
            punc_error_num <= lower_bound
            and level2score[level] > level2score[best_level]
        ):
            best_level = level
    scores["punc_usage"] = level2score[best_level]

    return scores


def get_grammar_analysis(grammar_error_types, doc):

    form_error_num = grammar_error_types.get("Form", 0)
    form_num = utils.get_counts(doc)

    prep_error_num = grammar_error_types.get("Preposition", 0)
    prep_num = utils.get_counts(doc, pos_list=["ADV", "PART"])

    punc_error_num = grammar_error_types.get("Punctuation", 0)
    punc_num = utils.get_counts(doc, pos_list=["PUNCT"])

    scores = calculate_score(form_error_num, prep_error_num, punc_error_num)

    return {
        "form_usage": {"correct": form_num - form_error_num, "mistake": form_error_num},
        "prep_usage": {"correct": prep_num - prep_error_num, "mistake": prep_error_num},
        "punc_usage": {"correct": punc_num - punc_error_num, "mistake": punc_error_num},
        "scores": scores,
        "score": math.ceil(sum(scores.values()) / len(scores.keys())),
    }
