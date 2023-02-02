from typing import Dict

from nltk.tokenize import sent_tokenize


def calculate_sent_len(text: str) -> int:

    sents = sent_tokenize(text)

    sent_len_avg = round(sum([len(sent) for sent in sents]) / len(sents), 4)

    return len(sents), sent_len_avg


def report(text: str) -> Dict:

    sent_num, sent_len_avg = calculate_sent_len(text)

    return {
        "sent_num": sent_num,
        "sent_len_avg": sent_len_avg,
    }
