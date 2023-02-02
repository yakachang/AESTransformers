import statistics

from . import dependency_parser


def get_sent_analysis(doc):
    sent_num = len(list(doc.sents))
    sent_len_avg = statistics.mean(len([token for token in sent]) for sent in doc.sents)
    tree_avg_height = dependency_parser.get_average_heights(doc)

    return {
        "sent_num": sent_num,
        "sent_len_avg": sent_len_avg,
        "tree_avg_height": tree_avg_height,
    }
