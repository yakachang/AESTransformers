import math
import statistics

from . import dependency_parser

level2score = {"A1": 15, "A2": 35, "B1": 45, "B2": 55, "C1": 65, "C2": 85}

sents_num2level = {5: "A1", 7: "A2", 10: "B1", 20: "B2", 30: "C1", 40: "C2"}

sent_len_avg2level = {5: "A1", 7: "A2", 10: "B1", 20: "B2", 30: "C1", 40: "C2"}

tree_avg_height2level = {3: "A1", 5: "A2", 7: "B1", 8: "B2", 10: "C1", 12: "C2"}

conj_error_num2level = {12: "A1", 9: "A2", 7: "B1", 5: "B2", 3: "C1", 0: "C2"}


def calculate_score(sent_num, sent_len_avg, tree_avg_height, conj_error_num):

    scores = {}

    # Num of sents
    best_level = "A1"
    for upper_bound, level in sents_num2level.items():
        if sent_num >= upper_bound and level2score[level] > level2score[best_level]:
            best_level = level
    scores["sent_num"] = level2score[best_level]

    # Sent length average
    best_level = "A1"
    for upper_bound, level in sent_len_avg2level.items():
        if sent_len_avg >= upper_bound and level2score[level] > level2score[best_level]:
            best_level = level
    scores["sent_len_avg"] = level2score[best_level]

    # Average height of Syntax tree
    best_level = "A1"
    for upper_bound, level in tree_avg_height2level.items():
        if (
            tree_avg_height >= upper_bound
            and level2score[level] > level2score[best_level]
        ):
            best_level = level
    scores["tree_avg_height"] = level2score[best_level]

    # Word Usage
    best_level = "A1"
    for lower_bound, level in conj_error_num2level.items():
        if (
            conj_error_num <= lower_bound
            and level2score[level] > level2score[best_level]
        ):
            best_level = level
    scores["conj_usage"] = level2score[best_level]

    return scores


def get_sent_analysis(doc, grammar_error_types):
    sent_num = len(list(doc.sents))
    sent_len_avg = statistics.mean(len([token for token in sent]) for sent in doc.sents)
    tree_avg_height = dependency_parser.get_average_heights(doc)
    conj_error_num = grammar_error_types.get("Conjunction", 0)

    scores = calculate_score(sent_num, sent_len_avg, tree_avg_height, conj_error_num)

    return {
        "sent_num": sent_num,
        "sent_len_avg": sent_len_avg,
        "tree_avg_height": tree_avg_height,
        "scores": scores,
        "score": math.ceil(sum(scores.values()) / len(scores.keys())),
    }
